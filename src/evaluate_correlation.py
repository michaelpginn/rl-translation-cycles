from __future__ import annotations

import logging
import pathlib
from collections import defaultdict
from typing import Any

import pandas
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import ExperimentConfig
from src.data import EvalDataset
from src.distributed import DistributedConfig
from src.modeling.generation import greedy_decode, sample_completions
from src.modeling.prompts import make_backward_prompt, make_forward_prompt
from src.modeling.rewards import compute_sentence_metric

logger = logging.getLogger(__name__)


def evaluate_correlation(
    model: Any,
    tokenizer: Any,
    dataset: EvalDataset,
    config: ExperimentConfig,
    dist_config: DistributedConfig,
):
    if dist_config.distributed:
        raise NotImplementedError()
    gs = config.grpo_group_size
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size * config.grpo_group_size,
        shuffle=False,
    )
    model.eval()
    metrics: dict[str, list[float]] = defaultdict(list)
    running_idx = 0
    for batch in tqdm(loader, desc="Evaluating", disable=not dist_config.is_main):
        eng_sentences = batch["eng"]
        tgt_sentences = batch["tgt"]
        bs = len(eng_sentences)

        fwd_prompts = [make_forward_prompt(s, config.language) for s in eng_sentences]
        fwd_preds, _ = sample_completions(
            model,
            tokenizer,
            fwd_prompts,
            num_samples=gs,
            max_new_tokens=config.max_tokens,
            temperature=config.grpo_temperature,
            top_p=config.grpo_top_p,
            top_k=config.grpo_top_k,
        )
        bwd_prompts = [
            make_backward_prompt(s, config.language)
            for group in fwd_preds
            for s in group
        ]
        bwd_preds = greedy_decode(
            model, tokenizer, bwd_prompts, max_new_tokens=config.max_tokens
        )
        bwd_preds = [bwd_preds[i * gs : (i + 1) * gs] for i in range(bs)]

        # Compute metrics
        for metric in ["bleu", "chrf"]:
            fwd_scores = torch.tensor(
                [
                    compute_sentence_metric(
                        fwd_preds[idx],
                        [tgt_sentences[idx]] * gs,
                        metric,
                    )
                    for idx in range(bs)
                ]
            )
            fwd_scores_norm = (
                fwd_scores - fwd_scores.mean(dim=-1, keepdim=True)
            ) / fwd_scores.std(dim=-1, keepdim=True)
            fwd_scores_norm = fwd_scores_norm.reshape(bs * gs).tolist()
            metrics[metric].extend(fwd_scores_norm)

            round_trip_scores = torch.tensor(
                [
                    compute_sentence_metric(
                        bwd_preds[idx],
                        [eng_sentences[idx]] * gs,
                        metric,
                    )
                    for idx in range(bs)
                ]
            )
            round_trip_scores_norm = (
                round_trip_scores - round_trip_scores.mean(dim=-1, keepdim=True)
            ) / round_trip_scores.std(dim=-1, keepdim=True)
            round_trip_scores_norm = round_trip_scores_norm.reshape(bs * gs).tolist()
            metrics[f"roundtrip_{metric}"].extend(round_trip_scores_norm)
        sent_indices = []
        for idx in range(running_idx, running_idx + bs):
            sent_indices.extend([idx] * gs)
        running_idx = running_idx + bs
        metrics["sent_idx"].extend(sent_indices)
    df = pandas.DataFrame.from_dict(metrics)
    path = pathlib.Path(f"{config.language}_metrics.csv")
    df.to_csv(path)
    logger.info(f"Wrote metrics to {path.resolve()}")
