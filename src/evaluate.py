"""Evaluation: greedy inference on FLORES-200 and metric computation."""

from __future__ import annotations

import logging
import pprint
from typing import Any

import pandas as pd
import sacrebleu
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.data import FloresEvalDataset
from src.distributed import DistributedConfig
from src.modeling.generation import greedy_decode
from src.modeling.prompts import make_backward_prompt, make_forward_prompt

logger = logging.getLogger(__name__)


def generate(
    model: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    dist_config: DistributedConfig,
):
    """Runs greedy generation for the FLORES dataset in the specified language.

    Returns:
        A tuple (forward_df, backward_df) of DataFrames with predictions and references"""
    dataset = FloresEvalDataset(config)
    sampler: DistributedSampler | None = None
    if dist_config.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist_config.world_size,
            rank=dist_config.rank,
            shuffle=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
    )

    fwd_predictions: list[str] = []
    fwd_references: list[str] = []
    bwd_predictions: list[str] = []
    bwd_references: list[str] = []

    model.eval()
    for batch in tqdm(loader, desc="Evaluating", disable=not dist_config.is_main):
        eng_sentences = batch["eng"]
        tgt_sentences = batch["tgt"]

        # Forward: eng -> target
        fwd_prompts = [make_forward_prompt(s, config.language) for s in eng_sentences]
        fwd_preds = greedy_decode(
            model, tokenizer, fwd_prompts, max_new_tokens=config.max_tokens
        )
        fwd_predictions.extend(fwd_preds)
        fwd_references.extend(tgt_sentences)

        # Backward: target -> eng
        bwd_prompts = [make_backward_prompt(s, config.language) for s in tgt_sentences]
        bwd_preds = greedy_decode(
            model, tokenizer, bwd_prompts, max_new_tokens=config.max_tokens
        )
        bwd_predictions.extend(bwd_preds)
        bwd_references.extend(eng_sentences)

    if dist_config.world_size > 1:
        fwd_predictions = _gather_strings(fwd_predictions, dist_config.world_size)
        fwd_references = _gather_strings(fwd_references, dist_config.world_size)
        bwd_predictions = _gather_strings(bwd_predictions, dist_config.world_size)
        bwd_references = _gather_strings(bwd_references, dist_config.world_size)

    forward_df = pd.DataFrame.from_dict(
        {
            "source": bwd_references,
            "target_gold": fwd_references,
            "target_pred": fwd_predictions,
        }
    )
    backward_df = pd.DataFrame.from_dict(
        {
            "source": fwd_references,
            "target_gold": bwd_references,
            "target_pred": bwd_predictions,
        }
    )
    return forward_df, backward_df


def evaluate(
    model: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    dist_config: DistributedConfig,
) -> dict:
    """Evaluate translation quality on FLORES-200 for a single language pair.

    Runs greedy decoding for:
      1. eng -> target (forward)
      2. target -> eng (backward)

    Computes corpus-level BLEU and chrF for both directions.

    Returns:
        dict of metrics
    """
    forward_df, backward_df = generate(model, tokenizer, config, dist_config)

    if dist_config.is_main:
        fwd_preds = forward_df["target_pred"].tolist()
        fwd_gold = [forward_df["target_gold"].tolist()]
        bwd_preds = backward_df["target_pred"].tolist()
        bwd_gold = [backward_df["target_gold"].tolist()]

        logger.info(
            f"First few forward (pred, gold): {pprint.pformat(list(zip(fwd_preds[:5], fwd_gold[:5])))}"
        )
        logger.info(
            f"First few backward (pred, gold): {pprint.pformat(list(zip(bwd_preds[:5], bwd_gold[:5])))}"
        )

        metrics = {
            config.language: {
                "forward": {
                    "bleu": sacrebleu.corpus_bleu(fwd_preds, fwd_gold).score,
                    "chrf": sacrebleu.corpus_chrf(fwd_preds, fwd_gold).score,
                },
                "backward": {
                    "bleu": sacrebleu.corpus_bleu(bwd_preds, bwd_gold).score,
                    "chrf": sacrebleu.corpus_chrf(bwd_preds, bwd_gold).score,
                },
            }
        }
        logger.info(f"Metrics: {pprint.pformat(metrics)}")
        return metrics
    return {}


def _gather_strings(local_strings: list[str], world_size: int) -> list[str]:
    """Gather string lists from all ranks using torch.distributed."""
    import pickle

    import torch.distributed as dist

    local_data = pickle.dumps(local_strings)
    local_tensor = torch.tensor(list(local_data), dtype=torch.uint8, device="cuda")

    # Gather sizes
    local_size = torch.tensor([len(local_data)], dtype=torch.long, device="cuda")
    all_sizes = [
        torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)
    ]
    dist.all_gather(all_sizes, local_size)

    max_size = int(max(s.item() for s in all_sizes))
    padded = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
    padded[: len(local_data)] = local_tensor

    all_padded = [
        torch.zeros(max_size, dtype=torch.uint8, device="cuda")
        for _ in range(world_size)
    ]
    dist.all_gather(all_padded, padded)

    gathered: list[str] = []
    for i, size in enumerate(all_sizes):
        data = bytes(all_padded[i][: int(size.item())].cpu().tolist())
        gathered.extend(pickle.loads(data))

    return gathered
