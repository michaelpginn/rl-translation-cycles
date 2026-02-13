"""Memory profiling utilities for MPS and CUDA."""

import logging

import torch

logger = logging.getLogger(__name__)


def log_mem(label: str) -> None:
    """Log current GPU memory usage (MPS or CUDA)."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        allocated = torch.mps.current_allocated_memory() / 1e9
        driver = torch.mps.driver_allocated_memory() / 1e9
        logger.info(f"[MEM {label}] allocated={allocated:.2f}GB  driver={driver:.2f}GB")
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        logger.debug(
            f"[MEM {label}] allocated={allocated:.2f}GB  reserved={reserved:.2f}GB  peak={peak:.2f}GB"
        )
    else:
        return
