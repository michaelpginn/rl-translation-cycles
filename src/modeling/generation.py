"""Generation utilities for sampling and greedy decoding."""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.config.config import ExperimentConfig
from src.modeling.mem_profile import log_mem

logger = logging.getLogger(__name__)


@torch.no_grad()  # type: ignore[misc]
def sample_completions(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    target_lang: str,
    num_samples: int,
    config: ExperimentConfig,
) -> tuple[list[list[str]], torch.Tensor]:
    """Sample multiple completions for each prompt.

    Returns:
        texts: list of list of decoded strings, shape [batch, num_samples]
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
        truncation=False,
    ).to(model.device)

    prompt_len = inputs.input_ids.shape[1]

    log_mem("sample_completions_before_generate")
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_tokens,
        do_sample=True,
        temperature=config.grpo_temperature,
        top_p=config.grpo_top_p,
        top_k=config.grpo_top_k,
        return_dict_in_generate=True,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        **({"stop_strings": [config.stop_string]} if config.stop_string else {}),
        **(
            {"forced_bos_token_id": tokenizer.convert_tokens_to_ids(target_lang)}
            if config.is_nllb
            else {}
        ),
    )
    log_mem("sample_completions_after_generate")

    generated_ids = outputs.sequences[:, prompt_len:]

    # Decode texts
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    texts: list[list[str]] = []
    for i in range(len(prompts)):
        batch_texts = decoded[i * num_samples : (i + 1) * num_samples]
        batch_texts = [t.strip() for t in batch_texts]
        texts.append(batch_texts)

    return texts, generated_ids


@torch.no_grad()  # type: ignore[misc]
def greedy_decode(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    target_lang: str,
    config: ExperimentConfig,
) -> list[str]:
    """Greedy decoding for evaluation."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_tokens,
        do_sample=False,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        **({"stop_strings": [config.stop_string]} if config.stop_string else {}),
        **(
            {"forced_bos_token_id": tokenizer.convert_tokens_to_ids(target_lang)}
            if config.is_nllb
            else {}
        ),
    )
    generated_ids = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    decoded = [t.strip() for t in decoded]
    return decoded
