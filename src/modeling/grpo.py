"""GRPO (Group Relative Policy Optimization) for translation."""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.config.config import ExperimentConfig
from src.modeling.generation import greedy_decode, sample_completions
from src.modeling.mem_profile import log_mem
from src.modeling.prompts import make_backward_prompt, make_forward_prompt
from src.modeling.rewards import compute_cycle_rewards

logger = logging.getLogger(__name__)


def generate_translations_and_rewards(
    model: Any,
    tokenizer: Any,
    english_sentences: list[str],
    config: ExperimentConfig,
):
    """Generates translations and computes round-trip rewards.

    Returns:
        - fwd_prompts (list[str]): (bs,) list of forward prompts
        - fwd_texts (list[list[str]]): (bs, gs) list of predicted forward translations
        - bwd_texts (list[list[list[str]]]): (bs, gs, 1 or gs) list of 1 or multiple backtranslations
        - rewards (Tensor): (2, bs, gs) normalized float rewards
    """
    batch_size = len(english_sentences)

    # 1. Forward translation (eng -> target)
    log_mem("grpo_step_start")
    model.eval()
    if config.is_nllb:
        tokenizer.src_lang = "eng_Latn"
        tokenizer.tgt_lang = config.language
    fwd_prompts = [make_forward_prompt(s, config) for s in english_sentences]
    with torch.no_grad():
        fwd_texts, _ = sample_completions(
            model,
            tokenizer,
            prompts=fwd_prompts,
            target_lang=config.language,
            num_samples=config.grpo_group_size,
            config=config,
        )
    log_mem("after_fwd_generation")

    # 2. Back translation (target -> eng) for each forward candidate
    if config.is_nllb:
        tokenizer.src_lang = config.language
        tokenizer.tgt_lang = "eng_Latn"
    all_back_texts: list[list[list[str]]] = []  # [batch, g_fwd, g_bwd]
    all_bwd_prompts = []  # flat list for loss computation
    all_bwd_completions = []
    for i in range(batch_size):
        if not config.grpo_greedy_backward:
            group_back: list[list[str]] = []
            for j in range(config.grpo_group_size):
                bwd_prompt = make_backward_prompt(fwd_texts[i][j], config)
                with torch.no_grad():
                    bwd_texts_ij, _ = sample_completions(
                        model,
                        tokenizer,
                        prompts=[bwd_prompt],
                        target_lang="eng_Latn",
                        num_samples=config.grpo_num_backward,
                        config=config,
                    )
                group_back.append(bwd_texts_ij[0])
                all_bwd_prompts.extend([bwd_prompt] * config.grpo_num_backward)
                all_bwd_completions.extend(bwd_texts_ij[0])
                log_mem(f"after_bwd_generation_i{i}_j{j}")
            all_back_texts.append(group_back)
        else:
            # Greedy, just a single backtrans
            bwd_prompts = [
                make_backward_prompt(fwd_text, config) for fwd_text in fwd_texts[i]
            ]
            bwd_texts_i = greedy_decode(
                model,
                tokenizer,
                prompts=bwd_prompts,
                target_lang="eng_Latn",
                config=config,
                override_max_tokens=config.max_tokens * 2,
            )
            all_bwd_prompts.extend(bwd_prompts)
            all_bwd_completions.extend(bwd_texts_i)
            all_back_texts.append([[t] for t in bwd_texts_i])

    # 3. Compute rewards
    forward_rewards, _ = compute_cycle_rewards(
        english_sentences,
        fwd_texts,
        all_back_texts,
    )
    forward_rewards = forward_rewards.to(model.device)

    # Reset tokenizer since we'll use it for lps next
    if config.is_nllb:
        tokenizer.src_lang = "eng_Latn"
        tokenizer.tgt_lang = config.language
    return fwd_prompts, fwd_texts, all_back_texts, forward_rewards


def compute_logprobs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    completions: list[list[str]],
    with_grad: bool,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes logprobs for a list of prompts and nested list of completions.

    Returns:
        - policy_logprobs: (bs * gs, seq) tensor of log probs
        - mask: (bs * gs, seq) mask for completion
    """
    # TODO: Update for seq2seq
    flat_prompts = [p for p in prompts for _ in range(config.grpo_group_size)]
    flat_completions = [c for group in completions for c in group]
    full_texts = [p + " " + c for p, c in zip(flat_prompts, flat_completions)]
    if config.model_type == "decoder":
        encodings = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=config.max_tokens,
        ).to(model.device)
        prompt_lengths = (
            tokenizer(
                flat_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                max_length=config.max_tokens,
            )
            .attention_mask.sum(dim=-1)
            .to(model.device)
        )
        completion_lengths = encodings.attention_mask.sum(dim=-1) - prompt_lengths
        if (completion_lengths <= 0).any():
            logger.warning(
                f"Zero/negative completion lengths: {completion_lengths.tolist()}"
            )
        labels = encodings.input_ids[:, 1:]
        mask: torch.Tensor = encodings.attention_mask[:, 1:]
        mask *= torch.arange(labels.size(-1), device=labels.device).unsqueeze(
            0
        ) >= labels.size(-1) - completion_lengths.unsqueeze(1)
        if mask.sum() == 0:
            logger.warning("Skipping loss computation because mask is all 0s!")
            return torch.tensor(0.0, device=model.device, requires_grad=True), mask
    elif config.model_type == "seq2seq":
        encodings = tokenizer(
            flat_prompts,
            text_target=flat_completions,
            return_tensors="pt",
            padding=True,
            truncation=False,
            max_length=config.max_tokens,
        ).to(model.device)
        labels = encodings.labels[:, 1:]
        mask = (labels != tokenizer.pad_token_id).int()
    else:
        raise NotImplementedError()

    log_mem("compute_logprobs_start")
    logger.debug(f"Size of inputs: {encodings.input_ids.shape}")
    with torch.set_grad_enabled(with_grad):
        logits = model(**encodings).logits
        lse = logits[:, :-1, :].logsumexp(dim=-1)
        chosen_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        logprobs = chosen_logits - lse
        del logits, lse, chosen_logits
    log_mem("compute_logprobs_end")
    return logprobs, mask


def compute_grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    rewards: torch.Tensor,  # (2,bs,gs)
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = (
        (rewards - rewards.mean(dim=-1, keepdim=True))
        / (rewards.std(dim=-1, keepdim=True).clamp(min=1e-8))
    ).reshape(2, -1)  # (2,bs*gs)
    if config.reward_metric == "bleu":
        advantages = advantages[0]
    elif config.reward_metric == "chrf":
        advantages = advantages[1]
    elif config.reward_metric == "both":
        # TODO: Right now just an unweighted mean, maybe want weighting factor
        advantages = advantages.mean(dim=0)

    policy_ratio = torch.exp(policy_logprobs - old_logprobs)
    pg_loss_term = policy_ratio * advantages.unsqueeze(1)
    clipped_pg = torch.clamp(
        policy_ratio, min=1 - config.grpo_epsilon, max=1 + config.grpo_epsilon
    ) * advantages.unsqueeze(1)
    pg_loss_term = torch.minimum(pg_loss_term, clipped_pg)

    # Compute kl divergence (second term)
    log_ref_ratio = policy_logprobs - ref_logprobs
    kl_divergence = torch.exp(log_ref_ratio) - log_ref_ratio - 1
    if config.grpo_beta > 0:
        token_level_loss = -(clipped_pg - config.grpo_beta * kl_divergence)
    else:
        token_level_loss = -clipped_pg
    # DrGRPO style loss average (so sequences have equal weight regardless of length)
    loss = (token_level_loss * mask).sum(dim=-1) / mask.sum(dim=-1)
    loss = loss.mean()
    return loss, kl_divergence
