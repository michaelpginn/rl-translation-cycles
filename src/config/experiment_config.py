import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExperimentConfig:
    mode: Literal["train", "eval"]
    language: str
    pretrained_model: str = "Qwen/Qwen3-0.6B"

    max_tokens: int = 256

    train_dataset: str = "HuggingFaceFW/fineweb"
    train_dataset_subset: str = "sample-10BT"
    train_num_sentences: int = 20000
    train_max_sentence_len: int = 128

    eval_dataset: str = "openlanguagedata/flores_plus"
    eval_split: str = "devtest"
    eval_num_sentences: int | None = None

    # GRPO
    grpo_group_size: int = 4
    grpo_beta: float = 0.1
    grpo_temperature: float = 0.5
    grpo_top_p: float = 0.9
    grpo_epsilon: float = (
        0.2  # Not used for now, used if doing multiple updates/rollout
    )
    reward_metric: Literal["bleu", "chrf"] = "chrf"
    alpha: float = 1.0  # weight for forward translation reward

    # Training
    max_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    grad_norm: float = 1.0
    optimizer: str = "adamw"
    warmup_steps: int = 100
    eval_every_n_steps: int = 50

    # Logging
    wandb_project: str = "rl-translation-cycles"
    wandb_run_name: str | None = None

    # Checkpointing
    models_dir: str = "checkpoints"

    # System
    seed: int = 42
    slurm_job_id: str | None = field(
        default_factory=lambda: os.environ.get("SLURM_JOB_ID")
    )
