"""Generation utilities for sampling and greedy decoding."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()  # type: ignore[misc]
def sample_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[list[str]], torch.Tensor, torch.Tensor]:
    """Sample multiple completions for each prompt.

    Returns:
        texts: list of list of decoded strings, shape [batch, num_samples]
        all_log_probs: padded log-prob tensor, shape [batch * num_samples, max_len]
        all_token_ids: padded token id tensor, shape [batch * num_samples, max_len]
    """
    # Repeat each prompt num_samples times
    expanded_prompts = []
    for p in prompts:
        expanded_prompts.extend([p] * num_samples)

    inputs = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    prompt_len = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated_ids = outputs.sequences[:, prompt_len:]
    # Compute per-token log probs from scores
    all_log_probs = []
    for step_scores in outputs.scores:
        log_probs = torch.log_softmax(step_scores, dim=-1)
        all_log_probs.append(log_probs)

    # Stack: [batch * num_samples, seq_len, vocab]
    log_probs_stack = torch.stack(all_log_probs, dim=1)

    # Gather the log probs of the actually generated tokens
    token_log_probs = log_probs_stack.gather(
        2, generated_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Decode texts
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    texts = []
    for i in range(len(prompts)):
        batch_texts = decoded[i * num_samples : (i + 1) * num_samples]
        texts.append([t.strip() for t in batch_texts])

    return texts, token_log_probs, generated_ids


@torch.no_grad()  # type: ignore[misc]
def greedy_decode(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Greedy decoding for evaluation."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    prompt_len = inputs.input_ids.shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [t.strip() for t in decoded]
