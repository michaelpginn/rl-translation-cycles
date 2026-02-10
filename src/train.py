"""Main training loop: GRPO with cycle-consistency rewards."""

from __future__ import annotations

import copy
import logging
import math
import os
import random
from typing import Any

import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.experiment_config import ExperimentConfig
from src.data.loading import FineWebTrainDataset
from src.evaluation.evaluate import run_eval
from src.modeling.grpo import run_grpo_step

logger = logging.getLogger(__name__)


def run_train(config: ExperimentConfig, rank: int = 0, world_size: int = 1) -> None:
    """Run the full GRPO training loop.

    Args:
        config: experiment configuration
        rank: distributed rank (0 for single-process)
        world_size: number of distributed processes
    """
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    # Initialize wandb on rank 0
    if rank == 0:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Load model and tokenizer
    logger.info(f"Loading model: {config.pretrained_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model: Any = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model = model.to(device)

    # Reference model (frozen copy)
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Wrap in DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
        raw_model = model.module
    else:
        raw_model = model

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    # Learning rate scheduler with warmup
    total_steps = (
        config.max_epochs
        * config.train_num_sentences
        // (config.batch_size * world_size * config.gradient_accumulation_steps)
    )

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(
            total_steps - config.warmup_steps, 1
        )
        return max(0.1, 0.5 * (1 + math.cos(progress * math.pi)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Load training data
    train_dataset = FineWebTrainDataset(config)

    sampler: DistributedSampler[FineWebTrainDataset] | None = None
    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
    )

    # Run initial eval
    if rank == 0:
        logger.info("Running initial evaluation...")
    raw_model.eval()
    eval_metrics = run_eval(raw_model, tokenizer, config, rank, world_size)
    if rank == 0:
        wandb.log({"step": 0, **{f"eval/{k}": v for k, v in eval_metrics.items()}})

    # Training loop
    global_step = 0
    for epoch in range(config.max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        # Randomly pick a target language per batch
        lang_order = list(config.target_languages) * (
            len(train_loader) // len(config.target_languages) + 1
        )
        random.shuffle(lang_order)

        epoch_metrics = {
            "loss": 0.0,
            "fwd_loss": 0.0,
            "bwd_loss": 0.0,
            "mean_fwd_reward": 0.0,
            "mean_bwd_reward": 0.0,
        }
        num_batches = 0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.max_epochs}",
            disable=rank != 0,
        )

        optimizer.zero_grad()
        for batch_idx, english_sentences in pbar:
            target_lang = lang_order[batch_idx % len(lang_order)]

            # GRPO step (generation is done inside with torch.no_grad)
            result = run_grpo_step(
                raw_model, ref_model, tokenizer,
                list(english_sentences), target_lang, config,
            )

            loss = result["loss"] / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log to wandb
                if rank == 0:
                    log_dict = {
                        "step": global_step,
                        "train/loss": result["metrics"]["loss"],
                        "train/fwd_loss": result["metrics"]["fwd_loss"],
                        "train/bwd_loss": result["metrics"]["bwd_loss"],
                        "train/mean_fwd_reward": result["metrics"]["mean_fwd_reward"],
                        "train/mean_bwd_reward": result["metrics"]["mean_bwd_reward"],
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/target_lang": target_lang,
                    }
                    wandb.log(log_dict)

                # Periodic eval
                if (
                    config.eval_every_n_steps > 0
                    and global_step % config.eval_every_n_steps == 0
                ):
                    raw_model.eval()
                    eval_metrics = run_eval(
                        raw_model, tokenizer, config, rank, world_size
                    )
                    if rank == 0:
                        wandb.log(
                            {
                                "step": global_step,
                                **{f"eval/{k}": v for k, v in eval_metrics.items()},
                            }
                        )
                    model.train()

            # Update progress bar
            for k in epoch_metrics:
                if k in result["metrics"]:
                    epoch_metrics[k] += result["metrics"][k]
            num_batches += 1

            if rank == 0:
                pbar.set_postfix(
                    loss=f"{epoch_metrics['loss'] / num_batches:.4f}",
                    fwd_r=f"{epoch_metrics['mean_fwd_reward'] / num_batches:.2f}",
                    bwd_r=f"{epoch_metrics['mean_bwd_reward'] / num_batches:.2f}",
                )

        # End-of-epoch eval
        raw_model.eval()
        eval_metrics = run_eval(raw_model, tokenizer, config, rank, world_size)
        if rank == 0:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    **{f"eval/{k}": v for k, v in eval_metrics.items()},
                }
            )

        # Save checkpoint
        if rank == 0:
            ckpt_dir = os.path.join(config.models_dir, f"epoch_{epoch + 1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            raw_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved checkpoint to {ckpt_dir}")

    if rank == 0:
        wandb.finish()
        logger.info("Training complete.")
