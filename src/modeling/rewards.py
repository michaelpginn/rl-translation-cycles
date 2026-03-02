"""Cycle-consistency reward computation."""

import logging
from pprint import pformat

import sacrebleu
import torch

logger = logging.getLogger(__name__)


def compute_sentence_metric(
    predictions: list[str],
    references: list[str],
    metric: str = "chrf",
) -> list[float]:
    """Compute per-sentence BLEU or chrF scores.

    Args:
        predictions: predicted sentences
        references: reference sentences
        metric: "bleu" or "chrf"

    Returns:
        list of per-sentence scores (0-100 scale)
    """
    scores: list[float] = []
    for pred, ref in zip(predictions, references):
        if metric == "bleu":
            scores.append(sacrebleu.sentence_bleu(pred, [ref]).score)
        elif metric == "chrf":
            scores.append(sacrebleu.sentence_chrf(pred, [ref]).score)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return scores


def compute_cycle_rewards(
    original_english: list[str],
    forward_translations: list[list[str]],
    back_translations: list[list[list[str]]],
    metric: str = "chrf",
    log=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cycle-consistency rewards for GRPO.

    For each original sentence i:
      - forward_translations[i] has g candidates (eng -> target)
      - back_translations[i][j] has g candidates for forward_translations[i][j]
        (target -> eng)

    Back-translation reward: metric(back_translation, original_english)
    Forward reward for candidate j: unnormalized sum of back-translation rewards
        for all back-translations of candidate j
    Total reward = alpha * forward_reward + back_reward

    Args:
        original_english: batch of original English sentences
        forward_translations: [batch_size, g] forward translations
        back_translations: [batch_size, g, g] back translations
        metric: "bleu" or "chrf"

    Returns:
        forward_rewards: [batch_size, g] rewards for forward step
        backward_rewards: [batch_size, g, g] rewards for backward step
    """
    batch_size = len(original_english)
    g_fwd = len(forward_translations[0])
    g_bwd = len(back_translations[0][0])

    backward_rewards = torch.zeros(batch_size, g_fwd, g_bwd)
    forward_rewards = torch.zeros(batch_size, g_fwd)

    for i in range(batch_size):
        for j in range(g_fwd):
            back_preds = back_translations[i][j]
            refs = [original_english[i]] * g_bwd
            scores = compute_sentence_metric(back_preds, refs, metric)

            for k in range(g_bwd):
                backward_rewards[i, j, k] = scores[k]

            # Forward reward = unnormalized sum of backward rewards
            forward_rewards[i, j] = sum(scores)

    if log:
        logger.info(f"""First example:
Original eng: {original_english[0]}
Forw pred: {forward_translations[0]}
Forw rwd:  {forward_rewards[0].tolist()}
Back pred: {pformat(back_translations[0])}
Back rwd: {backward_rewards[0]}""")

    return forward_rewards, backward_rewards
