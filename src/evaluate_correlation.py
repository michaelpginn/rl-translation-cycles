from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.experiment_config import ExperimentConfig
from src.data import FloresEvalDataset
from src.distributed import DistributedConfig
from src.modeling.generation import greedy_decode, sample_completions
from src.modeling.prompts import make_backward_prompt, make_forward_prompt
from src.modeling.rewards import compute_sentence_metric


def evaluate_correlation(
    model: Any,
    tokenizer: Any,
    dataset: FloresEvalDataset,
    config: ExperimentConfig,
    dist_config: DistributedConfig,
):
    if dist_config.distributed:
        raise NotImplementedError()
    bs = 2 * config.batch_size * (config.grpo_group_size**2)
    gs = config.grpo_group_size
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
    )
    model.eval()
    metrics: dict[str, list[float]] = defaultdict(list)
    for batch in tqdm(loader, desc="Evaluating", disable=not dist_config.is_main):
        eng_sentences = batch["eng"]
        tgt_sentences = batch["tgt"]

        fwd_prompts = [make_forward_prompt(s, config.language) for s in eng_sentences]
        fwd_preds, _ = sample_completions(
            model,
            tokenizer,
            fwd_prompts,
            num_samples=gs,
            max_new_tokens=config.max_tokens,
            temperature=config.grpo_temperature,
            top_p=config.grpo_top_p,
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
    df = pandas.DataFrame.from_dict(metrics)
    df.to_csv(f"{config.language}_metrics.csv")
