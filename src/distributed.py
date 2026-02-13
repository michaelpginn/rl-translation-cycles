import logging
import os
import pprint
from dataclasses import dataclass
from socket import gethostname

import torch

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    world_size: int
    rank: int  # Will be 0 if not distributed
    local_rank: int
    device: torch.device
    device_type: str
    distributed: bool

    @property
    def is_main(self):
        return self.rank == 0


def setup_distributed() -> DistributedConfig:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed (torchrun)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        logger.info(
            f"Hello from rank {rank} of {world_size} on {gethostname()} "
            f"(local_rank {local_rank}, device {device})",
        )

        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

        config = DistributedConfig(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            device=device,
            device_type="cuda",
            distributed=True,
        )
    else:
        # Single GPU setup
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        config = DistributedConfig(
            world_size=1,
            rank=0,
            local_rank=0,
            device=device,
            device_type=str(device),
            distributed=False,
        )
    logger.info("Distributed training parameters: \n%s", pprint.pformat(config))
    return config


def cleanup_distributed():
    """Clean up distributed process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
