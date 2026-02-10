"""Evaluation: greedy inference on FLORES-200 and metric computation."""

from __future__ import annotations

import logging
from typing import Any

import sacrebleu
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.data.loading import FloresEvalDataset
from src.modeling.generation import greedy_decode
from src.modeling.prompts import make_backward_prompt, make_forward_prompt

logger = logging.getLogger(__name__)


def evaluate_translation(
    model: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    target_lang: str,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, float]:
    """Evaluate translation quality on FLORES-200 for a single language pair.

    Runs greedy decoding for:
      1. eng -> target (forward)
      2. target -> eng (backward)

    Computes corpus-level BLEU and chrF for both directions.

    Returns:
        dict of metrics
    """
    dataset = FloresEvalDataset(config, target_lang)

    sampler: DistributedSampler[FloresEvalDataset] | None = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=False
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
    for batch in tqdm(loader, desc=f"Eval {target_lang}", disable=rank != 0):
        eng_sentences = batch["source"]
        tgt_sentences = batch["target"]

        # Forward: eng -> target
        fwd_prompts = [make_forward_prompt(s, target_lang) for s in eng_sentences]
        fwd_preds = greedy_decode(
            model, tokenizer, fwd_prompts, max_new_tokens=config.max_tokens
        )
        fwd_predictions.extend(fwd_preds)
        fwd_references.extend(tgt_sentences)

        # Backward: target -> eng
        bwd_prompts = [make_backward_prompt(s, target_lang) for s in tgt_sentences]
        bwd_preds = greedy_decode(
            model, tokenizer, bwd_prompts, max_new_tokens=config.max_tokens
        )
        bwd_predictions.extend(bwd_preds)
        bwd_references.extend(eng_sentences)

    # Gather predictions across ranks if distributed
    if world_size > 1:
        fwd_predictions = _gather_strings(fwd_predictions, world_size)
        fwd_references = _gather_strings(fwd_references, world_size)
        bwd_predictions = _gather_strings(bwd_predictions, world_size)
        bwd_references = _gather_strings(bwd_references, world_size)

    metrics: dict[str, float] = {}
    if rank == 0:
        # Forward metrics
        fwd_bleu = sacrebleu.corpus_bleu(fwd_predictions, [fwd_references])
        fwd_chrf = sacrebleu.corpus_chrf(fwd_predictions, [fwd_references])
        metrics[f"{target_lang}/fwd_bleu"] = fwd_bleu.score
        metrics[f"{target_lang}/fwd_chrf"] = fwd_chrf.score

        # Backward metrics
        bwd_bleu = sacrebleu.corpus_bleu(bwd_predictions, [bwd_references])
        bwd_chrf = sacrebleu.corpus_chrf(bwd_predictions, [bwd_references])
        metrics[f"{target_lang}/bwd_bleu"] = bwd_bleu.score
        metrics[f"{target_lang}/bwd_chrf"] = bwd_chrf.score

        logger.info(
            f"[{target_lang}] fwd_bleu={fwd_bleu.score:.2f} "
            f"fwd_chrf={fwd_chrf.score:.2f} "
            f"bwd_bleu={bwd_bleu.score:.2f} "
            f"bwd_chrf={bwd_chrf.score:.2f}"
        )

    return metrics


def _gather_strings(local_strings: list[str], world_size: int) -> list[str]:
    """Gather string lists from all ranks using torch.distributed."""
    import pickle

    import torch.distributed as dist

    local_data = pickle.dumps(local_strings)
    local_tensor = torch.tensor(
        list(local_data), dtype=torch.uint8, device="cuda"
    )

    # Gather sizes
    local_size = torch.tensor([len(local_data)], dtype=torch.long, device="cuda")
    all_sizes = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]
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


def run_eval(
    model: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, float]:
    """Run evaluation across all target languages."""
    all_metrics: dict[str, float] = {}
    for lang in config.target_languages:
        lang_metrics = evaluate_translation(
            model, tokenizer, config, lang, rank, world_size
        )
        all_metrics.update(lang_metrics)

    # Compute averages
    if rank == 0:
        fwd_bleus = [v for k, v in all_metrics.items() if k.endswith("/fwd_bleu")]
        bwd_bleus = [v for k, v in all_metrics.items() if k.endswith("/bwd_bleu")]
        fwd_chrfs = [v for k, v in all_metrics.items() if k.endswith("/fwd_chrf")]
        bwd_chrfs = [v for k, v in all_metrics.items() if k.endswith("/bwd_chrf")]

        if fwd_bleus:
            all_metrics["avg/fwd_bleu"] = sum(fwd_bleus) / len(fwd_bleus)
            all_metrics["avg/bwd_bleu"] = sum(bwd_bleus) / len(bwd_bleus)
            all_metrics["avg/fwd_chrf"] = sum(fwd_chrfs) / len(fwd_chrfs)
            all_metrics["avg/bwd_chrf"] = sum(bwd_chrfs) / len(bwd_chrfs)

        logger.info(f"Average metrics: {all_metrics}")

    return all_metrics
