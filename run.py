"""Main entry point for rl-translation-cycles experiments."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from src.config import ExperimentConfig, config_to_dataclass
from src.distributed import cleanup_distributed, setup_distributed

logging.basicConfig(
    level=logging.INFO,
    format="\033[90m%(asctime)s \033[36m[%(levelname)s] \033[1;33m%(module)s\033[0m: %(message)s",
)
logger = logging.getLogger(__name__)


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
    args = parser.parse_args()

    config = config_to_dataclass(args.config, args.overrides, ExperimentConfig)
    rank, world_size = setup_distributed()

    if rank == 0:
        logger.info(f"Config: {config}")
        logger.info(f"Mode: {config.mode}")

    try:
        if config.mode == "train":
            from src.train import run_train

            run_train(config, rank, world_size)
        elif config.mode == "eval":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from src.evaluation.evaluate import run_eval

            tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model: Any = AutoModelForCausalLM.from_pretrained(
                config.pretrained_model,
                dtype=torch.bfloat16,
            )
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            metrics = run_eval(model, tokenizer, config, rank, world_size)
            if rank == 0:
                import json

                logger.info(f"Evaluation results:\n{json.dumps(metrics, indent=2)}")
        else:
            raise ValueError(f"Unknown mode: {config.mode}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
