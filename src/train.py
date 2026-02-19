"""Main training loop: GRPO with cycle-consistency rewards."""

from __future__ import annotations

import copy
import logging
import math
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb
from src.config.experiment_config import ExperimentConfig
from src.data import FineWebTrainDataset
from src.distributed import DistributedConfig
from src.evaluate import evaluate
from src.modeling.grpo import run_grpo_step
from src.modeling.mem_profile import log_mem

logger = logging.getLogger(__name__)


def train(
    model,
    tokenizer,
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
    total_steps = (
        config.max_epochs
        * config.train_num_sentences
        // (
            config.batch_size
            * dist_config.world_size
            * config.gradient_accumulation_steps
        )
    )

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
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
    eval_metrics = evaluate(model, tokenizer, config, dist_config)
    if dist_config.is_main:
        wandb.log({"eval": eval_metrics}, step=0)

    # Training loop
    global_step = 0
    pbar = tqdm(
        total=total_steps,
        desc="Training",
        disable=not dist_config.is_main,
    )
    for epoch in range(config.max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_metrics = {
            "loss": 0.0,
            "fwd_loss": 0.0,
            "bwd_loss": 0.0,
            "mean_fwd_reward": 0.0,
            "mean_bwd_reward": 0.0,
        }
        optimizer.zero_grad()
        for batch_idx, english_sentences in enumerate(train_loader):
            log_mem(f"batch_{batch_idx}_start")
            # GRPO step (generation is done inside with torch.no_grad)
            with torch.amp.autocast_mode.autocast(
                dist_config.device_type, dtype=torch.bfloat16
            ):
                result = run_grpo_step(
                    model,
                    ref_model,
                    tokenizer,
                    list(english_sentences),
                    config,
                )
            log_mem(f"batch_{batch_idx}_after_grpo")
            loss = result["loss"] / config.gradient_accumulation_steps
            loss.backward()
            log_mem(f"batch_{batch_idx}_after_backward")

            # Optimizer step
            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if dist_config.is_main:
                    wandb.log(
                        {
                            "step": global_step,
                            "train": {
                                "lr": scheduler.get_last_lr()[0],
                                **result["metrics"],
                            },
                        }
                    )

            # Periodic eval
            if (
                config.eval_every_n_steps > 0
                and global_step % config.eval_every_n_steps == 0
            ):
                logger.info("Running intermediate evaluation...")
                eval_metrics = evaluate(model, tokenizer, config, dist_config)
                if dist_config.is_main:
                    wandb.log(
                        {
                            "step": global_step,
                            **{"eval": eval_metrics},
                        }
                    )
                model.train()
            global_step += 1

            # Update epoch metrics
            for k in epoch_metrics:
                if k in result["metrics"]:
                    epoch_metrics[k] += result["metrics"][k]
            if dist_config.is_main:
                # Show continuous metrics for this epoch
                pbar.set_postfix(
                    loss=f"{epoch_metrics['loss'] / (batch_idx + 1):.4f}",
                    fwd_r=f"{epoch_metrics['mean_fwd_reward'] / (batch_idx + 1):.2f}",
                    bwd_r=f"{epoch_metrics['mean_bwd_reward'] / (batch_idx + 1):.2f}",
                )
                pbar.update()

        logger.info("Running end-of-epoch evaluation...")
        eval_metrics = evaluate(model, tokenizer, config, dist_config)
        if dist_config.is_main:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                }
            )
            ckpt_dir = os.path.join(config.models_dir, f"epoch_{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved checkpoint to {ckpt_dir}")

    if dist_config.is_main:
        wandb.finish()
        logger.info("Training complete.")
