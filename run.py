"""Main entry point for rl-translation-cycles experiments."""

from __future__ import annotations

import argparse
import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from src.config import ExperimentConfig, config_to_dataclass
from src.distributed import cleanup_distributed, setup_distributed
from src.evaluate import evaluate
from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RL Translation Cycles: GRPO for low-resource translation"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to .cfg experiment config file",
    )
    parser.add_argument(
        "--overrides",
        "-o",
        nargs="*",
        default=[],
        help="Config overrides as key=value pairs",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    config = config_to_dataclass(args.config, args.overrides, ExperimentConfig)
    dist_config = setup_distributed()

    if dist_config.is_main:
        logger.info(f"Config: {config}")
        logger.info(f"Mode: {config.mode}")
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={**vars(config), "distributed": vars(dist_config)},
        )
        # log_pip_freeze_artifact(f"pip-freeze-{wandb.run.id}")  # type:ignore

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            import flash_attn  # noqa: F401 # type:ignore

            attn_impl = "flash_attention_2"
            logger.info("Using flash_attention")
        except ImportError:
            attn_impl = "eager"
            logger.info("Falling back to eager attention")
        model: Any = AutoModelForCausalLM.from_pretrained(
            config.pretrained_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        model = model.to(dist_config.device)
        model.gradient_checkpointing_enable()
        if dist_config.distributed:
            logger.info("Wrapping model in DDP")
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[dist_config.local_rank]
            )

        if config.mode == "train":
            train(model, tokenizer, config, dist_config)
        elif config.mode == "eval":
            metrics = evaluate(model, tokenizer, config, dist_config)
            if dist_config.is_main:
                wandb.log({"eval": metrics})
        else:
            raise ValueError(f"Unknown mode: {config.mode}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
