"""GRPO (Group Relative Policy Optimization) for translation."""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.config.experiment_config import ExperimentConfig
from src.modeling.generation import sample_completions
from src.modeling.mem_profile import log_mem
from src.modeling.prompts import make_backward_prompt, make_forward_prompt
from src.modeling.rewards import compute_cycle_rewards

logger = logging.getLogger(__name__)


def _compute_grpo_loss(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    advantages: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Compute the DAPO policy gradient loss for a batch.

    Uses sentence-level rewards distributed evenly over completion tokens.

    Args:
        model: the policy model being trained
        ref_model: frozen reference model for KL penalty
        tokenizer: tokenizer
        prompts: list of prompt strings
        completions: list of completion strings
        flat_fwd_advantages: [batch_size] tensor of *normalized* rewards
        beta: KL penalty coefficient
        max_tokens: max sequence length

    Returns:
        scalar loss tensor
    """
    # Tokenize full sequences (prompt + completion)
    full_texts = [p + " " + c for p, c in zip(prompts, completions)]
    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_tokens,
    ).to(model.device)
    prompt_lengths = (
        tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_tokens,
        )
        .attention_mask.sum(dim=-1)
        .to(model.device)
    )
    completion_lengths = encodings.attention_mask.sum(dim=-1) - prompt_lengths
    labels = encodings.input_ids[:, 1:]
    mask = encodings.attention_mask[:, 1:]
    mask *= torch.arange(labels.size(-1), device=labels.device).unsqueeze(
        0
    ) >= labels.size(-1) - completion_lengths.unsqueeze(1)

    # Compute prob ratio (first term)
    log_mem(f"grpo_loss_before_policy_fwd (n_seqs={len(prompts)})")
    logger.debug(f"Size of inputs: {encodings.input_ids.shape}")
    policy_logprobs = (
        torch.log_softmax(model(**encodings).logits, dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1)
    )
    log_mem("grpo_loss_after_policy_fwd")
    policy_probs = torch.exp(
        policy_logprobs - policy_logprobs.detach()
    ) * advantages.unsqueeze(1)

    # Compute kl divergence (second term)
    if config.grpo_beta > 0:
        with torch.no_grad():
            ref_logprobs = (
                torch.log_softmax(ref_model(**encodings).logits, dim=-1)
                .gather(dim=-1, index=labels.unsqueeze(-1))
                .squeeze(-1)
                .detach()
            )
        log_mem("grpo_loss_after_ref_fwd")
        log_ref_ratio = policy_logprobs - ref_logprobs
        kl_divergence = torch.exp(log_ref_ratio) - log_ref_ratio - 1
        token_level_loss = -(policy_probs - config.grpo_beta * kl_divergence)
    else:
        token_level_loss = -policy_probs
    loss = (token_level_loss * mask).sum() / mask.sum()
    return loss


def run_grpo_step(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    english_sentences: list[str],
    config: ExperimentConfig,
) -> dict:
    """Run one GRPO step on a batch of English sentences.

    1. Generate g forward translations (eng -> target) per sentence
    2. For each forward translation, generate g back translations (target -> eng)
    3. Compute cycle-consistency rewards
    4. Compute GRPO loss for both forward and backward steps
    5. Return combined loss and metrics

    Args:
        model: policy model
        ref_model: frozen reference model
        tokenizer: tokenizer
        english_sentences: batch of English sentences
        target_lang: target language code
        config: experiment config

    Returns:
        dict with 'loss' tensor and various metrics
    """
    batch_size = len(english_sentences)

    # Step 1: Forward translation (eng -> target)
    log_mem("grpo_step_start")
    fwd_prompts = [make_forward_prompt(s, config.language) for s in english_sentences]
    with torch.no_grad():
        fwd_texts, _ = sample_completions(
            model,
            tokenizer,
            prompts=fwd_prompts,
            num_samples=config.grpo_group_size,
            max_new_tokens=config.max_tokens,
            temperature=config.grpo_temperature,
            top_p=config.grpo_top_p,
        )
    log_mem("after_fwd_generation")

    # Step 2: Back translation (target -> eng) for each forward candidate
    all_back_texts = []  # [batch, g_fwd, g_bwd]
    all_bwd_prompts = []  # flat list for loss computation
    all_bwd_completions = []

    for i in range(batch_size):
        group_back = []
        for j in range(config.grpo_group_size):
            bwd_prompt = make_backward_prompt(fwd_texts[i][j], config.language)
            with torch.no_grad():
                bwd_texts_ij, _ = sample_completions(
                    model,
                    tokenizer,
                    prompts=[bwd_prompt],
                    num_samples=config.grpo_group_size,
                    max_new_tokens=config.max_tokens,
                    temperature=config.grpo_temperature,
                    top_p=config.grpo_top_p,
                )
            group_back.append(bwd_texts_ij[0])  # g back translations
            all_bwd_prompts.extend([bwd_prompt] * config.grpo_group_size)
            all_bwd_completions.extend(bwd_texts_ij[0])
            log_mem(f"after_bwd_generation_i{i}_j{j}")
        all_back_texts.append(group_back)

    # Step 3: Compute rewards
    forward_rewards, backward_rewards = compute_cycle_rewards(
        english_sentences,
        fwd_texts,
        all_back_texts,
        metric=config.reward_metric,
    )
    forward_rewards = forward_rewards.to(model.device)
    backward_rewards = backward_rewards.to(model.device)

    log_mem("after_all_generation")
    # Step 4: GRPO loss for backward step (target -> eng)
    eps = 1e-4
    bwd_advantages = (
        backward_rewards - backward_rewards.mean(dim=-1, keepdim=True)
    ) / (backward_rewards.std(dim=-1).unsqueeze(-1) + eps)
    flat_bwd_advantages = bwd_advantages.reshape(-1)
    bwd_loss: torch.Tensor = torch.tensor(0.0, device=model.device)
    # Run in batches so we use a constant amount of mem
    for group_idx in range(config.grpo_group_size):
        start_idx = group_idx * config.grpo_group_size
        end_idx = (group_idx + 1) * config.grpo_group_size
        bwd_loss += _compute_grpo_loss(
            model,
            ref_model,
            tokenizer,
            all_bwd_prompts[start_idx:end_idx],
            all_bwd_completions[start_idx:end_idx],
            flat_bwd_advantages[start_idx:end_idx],
            config,
        )
    log_mem("after_bwd_loss")

    # Step 5: GRPO loss for forward step (eng -> target)
    flat_fwd_prompts = []
    flat_fwd_completions = []
    for i in range(batch_size):
        for j in range(config.grpo_group_size):
            flat_fwd_prompts.append(fwd_prompts[i])
            flat_fwd_completions.append(fwd_texts[i][j])

    # Total forward reward = alpha * sum_of_backward_rewards + backward_reward_contribution
    fwd_advantages = (forward_rewards - forward_rewards.mean(dim=-1, keepdim=True)) / (
        forward_rewards.std(dim=-1) + eps
    )
    flat_fwd_advantages = fwd_advantages.reshape(-1)
    fwd_loss = _compute_grpo_loss(
        model,
        ref_model,
        tokenizer,
        flat_fwd_prompts,
        flat_fwd_completions,
        flat_fwd_advantages,
        config,
    )
    log_mem("after_fwd_loss")

    total_loss = config.alpha * fwd_loss + bwd_loss

    metrics = {
        "loss": total_loss.item(),
        "fwd_loss": fwd_loss.item(),
        "bwd_loss": bwd_loss.item(),
        "mean_fwd_reward": forward_rewards.mean().item(),
        "mean_bwd_reward": backward_rewards.mean().item(),
        "mean_total_reward": (forward_rewards.mean() + backward_rewards.mean()).item(),
    }

    return {"loss": total_loss, "metrics": metrics}
