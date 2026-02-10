"""GRPO (Group Relative Policy Optimization) for translation."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from src.config.experiment_config import ExperimentConfig
from src.modeling.generation import sample_completions
from src.modeling.prompts import make_backward_prompt, make_forward_prompt
from src.modeling.rewards import compute_cycle_rewards

logger = logging.getLogger(__name__)


def _compute_grpo_loss(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    prompts: list[str],
    completions: list[str],
    rewards: torch.Tensor,
    beta: float,
    max_tokens: int,
) -> torch.Tensor:
    """Compute the GRPO policy gradient loss for a batch.

    Uses sentence-level rewards distributed evenly over completion tokens.

    Args:
        model: the policy model being trained
        ref_model: frozen reference model for KL penalty
        tokenizer: tokenizer
        prompts: list of prompt strings
        completions: list of completion strings
        rewards: [batch_size] tensor of rewards
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
        max_length=max_tokens,
    ).to(model.device)

    # Get prompt lengths to mask prompt tokens from loss
    prompt_encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    prompt_lengths = prompt_encodings.attention_mask.sum(dim=1).to(model.device)

    # Forward pass through policy and reference
    with torch.no_grad():
        ref_outputs = ref_model(**encodings)
        ref_logits = ref_outputs.logits

    policy_outputs = model(**encodings)
    policy_logits = policy_outputs.logits

    # Shift for next-token prediction
    shift_logits = policy_logits[:, :-1, :]
    shift_ref_logits = ref_logits[:, :-1, :]
    shift_labels = encodings.input_ids[:, 1:]
    shift_mask = encodings.attention_mask[:, 1:]

    # Log probs
    policy_log_probs = F.log_softmax(shift_logits, dim=-1)
    ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)

    # Gather log probs of actual tokens
    policy_token_lp = policy_log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    ref_token_lp = ref_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Create completion mask (1 for completion tokens, 0 for prompt/padding)
    batch_size, seq_len = shift_labels.shape
    positions = torch.arange(seq_len, device=model.device).unsqueeze(0).expand(batch_size, -1)
    completion_mask = (positions >= (prompt_lengths.unsqueeze(1) - 1)) & (shift_mask == 1)
    completion_mask = completion_mask.float()

    # Normalize rewards within group (GRPO: subtract mean, divide by std)
    if rewards.numel() > 1 and rewards.std() > 0:
        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        norm_rewards = rewards - rewards.mean()
    norm_rewards = norm_rewards.to(model.device)

    # Per-token advantage: sentence-level reward distributed evenly
    num_completion_tokens = completion_mask.sum(dim=1, keepdim=True).clamp(min=1)
    per_token_reward = (norm_rewards.unsqueeze(1) * completion_mask) / num_completion_tokens

    # Policy gradient loss: -E[advantage * log_prob]
    pg_loss = -(per_token_reward * policy_token_lp * completion_mask).sum() / completion_mask.sum().clamp(min=1)

    # KL penalty: D_KL(policy || ref) per token
    kl_div = (policy_token_lp - ref_token_lp) * completion_mask
    kl_loss = beta * kl_div.sum() / completion_mask.sum().clamp(min=1)

    return pg_loss + kl_loss


def run_grpo_step(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    english_sentences: list[str],
    target_lang: str,
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
    g = config.grpo_group_size
    batch_size = len(english_sentences)

    # Step 1: Forward translation (eng -> target)
    fwd_prompts = [
        make_forward_prompt(s, target_lang) for s in english_sentences
    ]
    fwd_texts, fwd_log_probs, fwd_token_ids = sample_completions(
        model, tokenizer, fwd_prompts, g,
        max_new_tokens=config.max_tokens,
        temperature=config.grpo_temperature,
        top_p=config.grpo_top_p,
    )

    # Step 2: Back translation (target -> eng) for each forward candidate
    all_back_texts = []  # [batch, g_fwd, g_bwd]
    all_bwd_prompts = []  # flat list for loss computation
    all_bwd_completions = []

    for i in range(batch_size):
        batch_back = []
        for j in range(g):
            bwd_prompt = make_backward_prompt(fwd_texts[i][j], target_lang)
            bwd_texts_ij, _, _ = sample_completions(
                model, tokenizer, [bwd_prompt], g,
                max_new_tokens=config.max_tokens,
                temperature=config.grpo_temperature,
                top_p=config.grpo_top_p,
            )
            batch_back.append(bwd_texts_ij[0])  # g back translations
            all_bwd_prompts.extend([bwd_prompt] * g)
            all_bwd_completions.extend(bwd_texts_ij[0])
        all_back_texts.append(batch_back)

    # Step 3: Compute rewards
    forward_rewards, backward_rewards = compute_cycle_rewards(
        english_sentences, fwd_texts, all_back_texts,
        metric=config.reward_metric, alpha=config.alpha,
    )

    # Step 4: GRPO loss for backward step (target -> eng)
    flat_bwd_rewards = backward_rewards.reshape(-1)
    bwd_loss = _compute_grpo_loss(
        model, ref_model, tokenizer,
        all_bwd_prompts, all_bwd_completions,
        flat_bwd_rewards, config.grpo_beta, config.max_tokens,
    )

    # Step 5: GRPO loss for forward step (eng -> target)
    flat_fwd_prompts = []
    flat_fwd_completions = []
    for i in range(batch_size):
        for j in range(g):
            flat_fwd_prompts.append(fwd_prompts[i])
            flat_fwd_completions.append(fwd_texts[i][j])

    # Total forward reward = alpha * sum_of_backward_rewards + backward_reward_contribution
    flat_fwd_rewards = forward_rewards.reshape(-1)
    fwd_loss = _compute_grpo_loss(
        model, ref_model, tokenizer,
        flat_fwd_prompts, flat_fwd_completions,
        flat_fwd_rewards, config.grpo_beta, config.max_tokens,
    )

    total_loss = fwd_loss + bwd_loss

    metrics = {
        "loss": total_loss.item(),
        "fwd_loss": fwd_loss.item(),
        "bwd_loss": bwd_loss.item(),
        "mean_fwd_reward": forward_rewards.mean().item(),
        "mean_bwd_reward": backward_rewards.mean().item(),
        "mean_total_reward": (forward_rewards.mean() + backward_rewards.mean()).item(),
    }

    return {"loss": total_loss, "metrics": metrics}
