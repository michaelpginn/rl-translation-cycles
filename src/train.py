"""Main training loop: GRPO with cycle-consistency rewards."""

from __future__ import annotations

import copy
import logging
import math
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from src.config.experiment_config import ExperimentConfig
from src.data import FineWebTrainDataset, FloresEvalDataset
from src.distributed import DistributedConfig
from src.evaluate import evaluate
from src.modeling.grpo import (
    compute_grpo_loss,
    compute_logprobs,
    generate_translations_and_rewards,
)
from src.modeling.mem_profile import log_mem

logger = logging.getLogger(__name__)


def train(
    model,
    tokenizer,
    dev_dataset: FloresEvalDataset,
    devtest_dataset: FloresEvalDataset,
    config: ExperimentConfig,
    dist_config: DistributedConfig,
) -> None:
    """Run the full GRPO training loop.

    Args:
        config: experiment configuration
        rank: distributed rank (0 for single-process)
        world_size: number of distributed processes
    """
    # Reference model (frozen copy)
    log_mem("before_ref_model_copy")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    log_mem("after_ref_model_copy")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    log_mem("after_optimizer_init")

    # Number of optimizer steps
    total_optimizer_steps = (
        config.max_epochs
        * config.train_num_sentences
        * config.inner_update_steps
        // (config.batch_size * dist_config.world_size * config.grad_acc_steps)
    )

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_optimizer_steps - config.warmup_steps, 1
        )
        return max(0.1, 0.5 * (1 + math.cos(progress * math.pi)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_dataset = FineWebTrainDataset(config, tokenizer)
    sampler: DistributedSampler[FineWebTrainDataset] | None = None
    if dist_config.distributed:
        sampler = DistributedSampler(
            train_dataset, num_replicas=dist_config.world_size, rank=dist_config.rank
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
    )
    if dist_config.is_main:
        logger.info("Running initial evaluation...")
    model.eval()
    dev_metrics = evaluate(model, tokenizer, dev_dataset, config, dist_config)
    devtest_metrics = evaluate(model, tokenizer, devtest_dataset, config, dist_config)
    if dist_config.is_main:
        wandb.log({"dev": dev_metrics, "devtest": devtest_metrics}, step=0)

    # Training loop
    num_batch_rollouts = 0  # Unlike total_optimizer_steps, this is the real number of batches (# loss.backward() calls)
    pbar = tqdm(
        total=total_optimizer_steps,
        desc="Training",
        disable=not dist_config.is_main,
    )
    with torch.amp.autocast_mode.autocast(
        dist_config.device_type, dtype=torch.bfloat16
    ):
        for epoch in range(config.max_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_metrics = defaultdict(float)
            optimizer.zero_grad()

            accumulated_prompts: list[list[str]] = []
            accumulated_completions: list[list[list[str]]] = []
            accumulated_backtranslations: list[list[list[list[str]]]] = []
            accumulated_rewards: list[torch.Tensor] = []
            old_logprobs: list[torch.Tensor] = []
            logprobs_mask: list[torch.Tensor] = []
            ref_logprobs: list[torch.Tensor] = []

            for batch_idx, english_sentences in enumerate(train_loader):
                # 1. Generate rollouts
                prompts, completions, backtranslations, rewards = (
                    generate_translations_and_rewards(
                        model, tokenizer, english_sentences, config
                    )
                )
                accumulated_prompts.append(prompts)
                accumulated_completions.append(completions)
                accumulated_backtranslations.append(backtranslations)
                accumulated_rewards.append(rewards)

                # 2. Compute old log probs
                old_lps, mask = compute_logprobs(
                    model,
                    tokenizer,
                    prompts,
                    completions,
                    with_grad=False,
                    config=config,
                )
                old_logprobs.append(old_lps)
                logprobs_mask.append(mask)

                # 3. Compute ref log probs
                ref_lps, _ = compute_logprobs(
                    ref_model,
                    tokenizer,
                    prompts,
                    completions,
                    with_grad=False,
                    config=config,
                )
                ref_logprobs.append(ref_lps)

                # 3. Optimizer step every grad_acc steps
                if (num_batch_rollouts + 1) % config.grad_acc_steps == 0:
                    for _ in range(config.inner_update_steps):
                        mean_kl_div = 0.0
                        mean_reward = 0.0
                        step_loss = 0.0

                        for inner_batch_idx in range(config.grad_acc_steps):
                            rewards = accumulated_rewards[inner_batch_idx]
                            policy_lps, _ = compute_logprobs(
                                model,
                                tokenizer,
                                accumulated_prompts[inner_batch_idx],
                                accumulated_completions[inner_batch_idx],
                                with_grad=True,
                                config=config,
                            )
                            loss, kl_div = compute_grpo_loss(
                                policy_logprobs=policy_lps,
                                old_logprobs=old_logprobs[inner_batch_idx],
                                ref_logprobs=ref_logprobs[inner_batch_idx],
                                mask=logprobs_mask[inner_batch_idx],
                                rewards=rewards,
                                config=config,
                            )
                            loss /= config.grad_acc_steps
                            log_mem("after_fwd_loss")
                            loss.backward()

                            # Track metrics
                            mean_kl_div += kl_div.mean().item()
                            mean_reward += rewards.mean().item()
                            step_loss += loss.detach().item()

                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.grad_norm
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        # Logging results
                        epoch_metrics["loss"] += step_loss / config.grad_acc_steps
                        epoch_metrics["kl_div"] += mean_kl_div / config.grad_acc_steps
                        epoch_metrics["reward"] += mean_reward / config.grad_acc_steps
                        if dist_config.is_main:
                            train_log = {
                                "train": {
                                    "lr": scheduler.get_last_lr()[0],
                                    "grad_norm": grad_norm(model),
                                    "epoch": epoch + 1,
                                    "loss": step_loss / config.grad_acc_steps,
                                    "kl_div": mean_kl_div / config.grad_acc_steps,
                                    "reward": mean_reward / config.grad_acc_steps,
                                },
                            }
                            if (
                                config.eval_every_n_steps > 0
                                and total_optimizer_steps
                                % (config.eval_every_n_steps * 2)
                                == 0
                            ):
                                example_outputs = []
                                for prompt_idx in range(
                                    min(5, len(accumulated_prompts))
                                ):
                                    for compl_idx in range(
                                        len(accumulated_completions[0])
                                    ):
                                        example_outputs.append(
                                            [
                                                accumulated_prompts[prompt_idx],
                                                prompt_idx,
                                                accumulated_completions[prompt_idx][
                                                    compl_idx
                                                ],
                                                compl_idx,
                                                accumulated_backtranslations[
                                                    prompt_idx
                                                ][compl_idx][
                                                    0
                                                ],  # doesn't work if multiple backtransl
                                            ]
                                        )
                                table = wandb.Table(
                                    columns=[
                                        "original_english",
                                        "fwd_idx",
                                        "predicted_target",
                                        "bwd_idx",
                                        "final_predicted_english",
                                    ],
                                    data=example_outputs,
                                )
                                train_log["train/examples"] = table  # type:ignore
                            wandb.log(train_log, step=num_batch_rollouts)
                        pbar.update()
                        total_optimizer_steps += 1

                        # Periodic eval
                        if (
                            config.eval_every_n_steps > 0
                            and total_optimizer_steps % config.eval_every_n_steps == 0
                        ):
                            logger.info("Running intermediate evaluation...")
                            dev_metrics = evaluate(
                                model, tokenizer, dev_dataset, config, dist_config
                            )
                            devtest_metrics = evaluate(
                                model, tokenizer, devtest_dataset, config, dist_config
                            )
                            if dist_config.is_main:
                                wandb.log(
                                    {"dev": dev_metrics, "devtest": devtest_metrics},
                                    step=num_batch_rollouts,
                                )
                            model.train()
                num_batch_rollouts += 1

                if dist_config.is_main:
                    # Show continuous metrics for this epoch
                    pbar.set_postfix(
                        loss=f"{epoch_metrics['loss'] / (batch_idx + 1):.4f}",
                        fwd_r=f"{epoch_metrics['reward'] / (batch_idx + 1):.2f}",
                    )

            logger.info("Running end-of-epoch evaluation...")
            dev_metrics = evaluate(model, tokenizer, dev_dataset, config, dist_config)
            devtest_metrics = evaluate(
                model, tokenizer, devtest_dataset, config, dist_config
            )
            if dist_config.is_main:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "dev": dev_metrics,
                        "devtest": devtest_metrics,
                    },
                    step=num_batch_rollouts,
                )
                ckpt_dir = os.path.join(config.models_dir, f"epoch_{epoch + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                logger.info(f"Saved checkpoint to {ckpt_dir}")

    if dist_config.is_main:
        wandb.finish()
        logger.info("Training complete.")


def grad_norm(model):
    # Log grad norm
    grad_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        grad_norm += param_norm.item() ** 2
    grad_norm = grad_norm**0.5
    return grad_norm
