"""Distributed training utilities."""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int]:
    """Initialize distributed training if available.

    Returns:
        (rank, world_size) tuple. (0, 1) if not distributed.
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        logger.info(
            f"Initialized distributed: rank={rank}, "
            f"world_size={world_size}, local_rank={local_rank}"
        )
        return rank, world_size
    else:
        logger.info("Running in single-process mode")
        return 0, 1


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank: int) -> torch.device:
    """Get the device for the current rank."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")
