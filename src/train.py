"""Main training loop: GRPO with cycle-consistency rewards."""

from __future__ import annotations

import copy
import logging
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from src.config.config import ExperimentConfig
from src.data import EvalDataset, FineWebTrainDataset, NLLBTrainDataset
from src.distributed import DistributedConfig
from src.evaluate import evaluate
from src.modeling.grpo import (
    compute_fwd_and_bwd_logprobs,
    compute_grpo_loss,
    generate_translations_and_rewards,
)
from src.modeling.mem_profile import log_mem

logger = logging.getLogger(__name__)


def copy_frozen(model):
    log_mem("before_ref_model_copy")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    log_mem("after_ref_model_copy")
    return ref_model


def train(
    model,
    tokenizer,
    dev_dataset: EvalDataset,
    test_dataset: EvalDataset,
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
    ref_model = copy_frozen(model)

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

    # Linear warmup
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if config.train_dataset == "breakend/nllb-multi-domain":
        train_dataset = NLLBTrainDataset(config, tokenizer)
    else:
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
    if not config.skip_initial_eval:
        if dist_config.is_main:
            logger.info("Running initial evaluation...")
        model.eval()
        dev_metrics = evaluate(model, tokenizer, dev_dataset, config, dist_config)
        test_metrics = evaluate(model, tokenizer, test_dataset, config, dist_config)
        if dist_config.is_main:
            wandb.log({"dev": dev_metrics, "test": test_metrics}, step=0)

    # Training loop
    num_optimizer_steps = (
        0  # Number of optimizer steps, ie multiple batches with grad acc
    )
    num_batch_rollouts = 0  # Unlike num_optimizer_steps, this is the real number of batches (# loss.backward() calls)
    pbar = tqdm(
        total=total_optimizer_steps,
        desc="Training",
        disable=not dist_config.is_main,
    )
    acc_fwd_prompts: list[list[str]] = []
    acc_fwd_completions: list[list[list[str]]] = []
    acc_bwd_prompts: list[list[str]] = []
    acc_bwd_completions: list[list[list[str]]] = []  # Flattened
    acc_backtranslations: list[list[list[list[str]]]] = []  # Not flattened
    acc_rewards: list[torch.Tensor] = []
    fwd_old_logprobs: list[torch.Tensor] = []  # Corresponds to acc_completions
    fwd_logprobs_mask: list[torch.Tensor] = []
    fwd_ref_logprobs: list[torch.Tensor] = []
    bwd_old_logprobs: list[torch.Tensor] = []  # Corr. to acc_backtranslatons
    bwd_logprobs_mask: list[torch.Tensor] = []
    bwd_ref_logprobs: list[torch.Tensor] = []

    with torch.amp.autocast_mode.autocast(
        dist_config.device_type, dtype=torch.bfloat16
    ):
        for epoch in range(config.max_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_metrics = defaultdict(float)
            optimizer.zero_grad()

            for batch_idx, english_sentences in enumerate(train_loader):
                # 1. Generate rollouts
                (
                    fwd_prompts,
                    fwd_completions,
                    bwd_prompts,
                    bwd_completions,
                    backtranslations,
                    rewards,
                ) = generate_translations_and_rewards(
                    model, tokenizer, english_sentences, config
                )
                acc_fwd_prompts.append(fwd_prompts)
                acc_fwd_completions.append(fwd_completions)
                acc_bwd_prompts.append(bwd_prompts)
                acc_bwd_completions.append(bwd_completions)
                acc_backtranslations.append(backtranslations)
                acc_rewards.append(rewards)

                # 2. Compute old log probs
                fwd_old_lps, fwd_mask, bwd_old_lps, bwd_mask = (
                    compute_fwd_and_bwd_logprobs(
                        model,
                        tokenizer,
                        fwd_prompts,
                        fwd_completions,
                        bwd_prompts,
                        bwd_completions,
                        with_grad=False,
                        config=config,
                    )
                )
                if fwd_old_lps is not None and fwd_mask is not None:
                    fwd_old_logprobs.append(fwd_old_lps)
                    fwd_logprobs_mask.append(fwd_mask)
                if bwd_old_lps is not None and bwd_mask is not None:
                    bwd_old_logprobs.append(bwd_old_lps)
                    bwd_logprobs_mask.append(bwd_mask)

                # 3. Compute ref log probs
                fwd_ref_lps, _, bwd_ref_lps, _ = compute_fwd_and_bwd_logprobs(
                    ref_model,
                    tokenizer,
                    fwd_prompts,
                    fwd_completions,
                    bwd_prompts,
                    bwd_completions,
                    with_grad=False,
                    config=config,
                )
                if fwd_ref_lps is not None:
                    fwd_ref_logprobs.append(fwd_ref_lps)
                if bwd_ref_lps is not None:
                    bwd_ref_logprobs.append(bwd_ref_lps)

                # 3. Optimizer step every grad_acc steps
                if (num_batch_rollouts + 1) % config.grad_acc_steps == 0:
                    # Inner step - can perform multiple gradient steps without new rollouts
                    for inner_step_idx in range(config.inner_update_steps):
                        fwd_mean_kl_div = 0.0
                        bwd_mean_kl_div = 0.0
                        fwd_step_loss = 0.0
                        bwd_step_loss = 0.0
                        mean_bleu_reward = 0.0
                        mean_chrf_reward = 0.0

                        for inner_batch_idx in range(config.grad_acc_steps):
                            rewards = acc_rewards[inner_batch_idx]
                            fwd_policy_lps, _, bwd_policy_lps, _ = (
                                compute_fwd_and_bwd_logprobs(
                                    model,
                                    tokenizer,
                                    acc_fwd_prompts[inner_batch_idx],
                                    acc_fwd_completions[inner_batch_idx],
                                    acc_bwd_prompts[inner_batch_idx],
                                    acc_bwd_completions[inner_batch_idx],
                                    with_grad=True,
                                    config=config,
                                )
                            )
                            if fwd_policy_lps is not None:
                                loss, kl_div = compute_grpo_loss(
                                    policy_logprobs=fwd_policy_lps,
                                    old_logprobs=fwd_old_logprobs[inner_batch_idx],
                                    ref_logprobs=fwd_ref_logprobs[inner_batch_idx],
                                    mask=fwd_logprobs_mask[inner_batch_idx],
                                    rewards=rewards,
                                    config=config,
                                )
                                loss /= config.grad_acc_steps
                                loss *= config.alpha
                                log_mem("after_fwd_loss")
                                loss.backward()
                                fwd_mean_kl_div += kl_div.mean().item()
                                fwd_step_loss += loss.detach().item()
                            if bwd_policy_lps is not None:
                                loss, kl_div = compute_grpo_loss(
                                    policy_logprobs=bwd_policy_lps,
                                    old_logprobs=bwd_old_logprobs[inner_batch_idx],
                                    ref_logprobs=bwd_ref_logprobs[inner_batch_idx],
                                    mask=bwd_logprobs_mask[inner_batch_idx],
                                    rewards=rewards,
                                    config=config,
                                )
                                loss /= config.grad_acc_steps
                                loss *= 1 - config.alpha
                                log_mem("after_fwd_loss")
                                loss.backward()
                                bwd_mean_kl_div += kl_div.mean().item()
                                bwd_step_loss += loss.detach().item()
                            mean_bleu_reward += rewards[0].mean().item()
                            mean_chrf_reward += rewards[1].mean().item()

                        unclipped_grad_norm = grad_norm(model)
                        if config.grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config.grad_norm
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        # Logging results
                        mean_bleu_reward /= config.grad_acc_steps
                        mean_chrf_reward /= config.grad_acc_steps
                        epoch_metrics["reward_bleu"] += mean_bleu_reward
                        epoch_metrics["reward_chrf"] += mean_chrf_reward
                        if dist_config.is_main:
                            train_log = {
                                "train": {
                                    "lr": scheduler.get_last_lr()[0],
                                    "grad_norm": unclipped_grad_norm,
                                    "epoch": epoch + 1,
                                    "fwd_loss": fwd_step_loss,
                                    "bwd_loss": bwd_step_loss,
                                    "fwd_kl_div": fwd_mean_kl_div
                                    / config.grad_acc_steps,
                                    "bwd_kl_div": bwd_mean_kl_div
                                    / config.grad_acc_steps,
                                    "reward_bleu": mean_bleu_reward,
                                    "reward_chrf": mean_chrf_reward,
                                    "inner_step_idx": inner_step_idx,
                                },
                            }
                            if (
                                config.eval_every_n_steps > 0
                                and num_optimizer_steps
                                % (config.eval_every_n_steps * 2)
                                == 0
                            ):
                                table = build_wandb_table(
                                    acc_fwd_prompts,
                                    acc_fwd_completions,
                                    acc_backtranslations,
                                    acc_rewards,
                                )
                                train_log["train/examples"] = table  # type:ignore
                            wandb.log(train_log, step=num_optimizer_steps)
                        pbar.update()
                        num_optimizer_steps += 1

                        # Periodic eval
                        if (
                            config.eval_every_n_steps > 0
                            and num_optimizer_steps % config.eval_every_n_steps == 0
                        ):
                            logger.info("Running intermediate evaluation...")
                            dev_metrics = evaluate(
                                model, tokenizer, dev_dataset, config, dist_config
                            )
                            test_metrics = evaluate(
                                model, tokenizer, test_dataset, config, dist_config
                            )
                            if dist_config.is_main:
                                wandb.log(
                                    {"dev": dev_metrics, "test": test_metrics},
                                    step=num_optimizer_steps,
                                )
                            model.train()

                    # Reset accumulators
                    acc_fwd_prompts = []
                    acc_fwd_completions = []
                    acc_bwd_prompts = []
                    acc_bwd_completions = []
                    acc_backtranslations = []
                    acc_rewards = []
                    fwd_old_logprobs = []
                    fwd_logprobs_mask = []
                    fwd_ref_logprobs = []
                    bwd_old_logprobs = []
                    bwd_logprobs_mask = []
                    bwd_ref_logprobs = []

                    if (
                        config.reference_update_steps > 0
                        and ((num_batch_rollouts + 1) / config.grad_acc_steps)
                        % config.reference_update_steps
                        == 0
                    ):
                        logger.info("Updating reference model")
                        ref_model = copy_frozen(model)
                num_batch_rollouts += 1

                if dist_config.is_main:
                    # Show continuous metrics for this epoch
                    pbar.set_postfix(
                        loss=f"{epoch_metrics['loss'] / (batch_idx + 1):.4f}",
                        fwd_r=f"{epoch_metrics['reward'] / (batch_idx + 1):.2f}",
                    )

            logger.info("Running end-of-epoch evaluation...")
            dev_metrics = evaluate(model, tokenizer, dev_dataset, config, dist_config)
            test_metrics = evaluate(model, tokenizer, test_dataset, config, dist_config)
            if dist_config.is_main:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "dev": dev_metrics,
                        "test": test_metrics,
                    },
                    step=num_optimizer_steps,
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


def build_wandb_table(
    acc_prompts: list[list[str]],
    acc_completions: list[list[list[str]]],
    acc_backtranslations: list[list[list[list[str]]]],
    acc_rewards: list[torch.Tensor],
):
    """Builds wandb table from first 5 prompts.

    Args:
        acc_prompts:     (num_batches, batch_size)
        acc_completions: (num_batches, batch_size, group_size)
        acc_completions: (num_batches, batch_size, group_size, 1 | group_size)
        acc_rewards:     (num_batches, 2, batch_size, group_size)
    """
    example_outputs: list[list[str | int | float]] = []
    flattened_prompts = [p for batch in acc_prompts for p in batch]
    flattened_completions = [c for batch in acc_completions for c in batch]
    flattened_backtranslations = [b for batch in acc_backtranslations for b in batch]
    flattened_rewards = torch.concat(acc_rewards, dim=1)
    for prompt_idx in range(min(5, len(flattened_prompts))):
        for compl_idx in range(len(flattened_completions[0])):
            bleu: float = flattened_rewards[0][prompt_idx][compl_idx].item()
            chrf: float = flattened_rewards[0][prompt_idx][compl_idx].item()
            row: list[str | int | float] = [
                flattened_prompts[prompt_idx],
                prompt_idx,
                flattened_completions[prompt_idx][compl_idx],
                compl_idx,
                flattened_backtranslations[prompt_idx][compl_idx][
                    0
                ],  # doesn't work if multiple backtransl
                bleu,
                chrf,
            ]
            example_outputs.append(row)
    return wandb.Table(
        columns=[
            "original_english",
            "fwd_idx",
            "predicted_target",
            "bwd_idx",
            "final_predicted_english",
            "bleu",
            "chrf",
        ],
        data=example_outputs,
    )
