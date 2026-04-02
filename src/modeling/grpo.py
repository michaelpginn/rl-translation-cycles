"""GRPO (Group Relative Policy Optimization) for translation."""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.config.experiment_config import ExperimentConfig
from src.modeling.generation import greedy_decode, sample_completions
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        loss (Tensor), kl_divergence (Tensor)
    """
    # Tokenize full sequences (prompt + completion)
    full_texts = [p + " " + c for p, c in zip(prompts, completions)]
    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=config.max_tokens,
    ).to(model.device)
    prompt_lengths = (
        tokenizer(
            prompts,
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
    mask = encodings.attention_mask[:, 1:]
    mask *= torch.arange(labels.size(-1), device=labels.device).unsqueeze(
        0
    ) >= labels.size(-1) - completion_lengths.unsqueeze(1)
    if mask.sum() == 0:
        logger.warning("Skipping loss computation because mask is all 0s!")
        return torch.tensor(0.0, device=model.device, requires_grad=True), None

    # Compute prob ratio (first term)
    log_mem(f"grpo_loss_before_policy_fwd (n_seqs={len(prompts)})")
    logger.debug(f"Size of inputs: {encodings.input_ids.shape}")
    policy_logits = model(**encodings).logits
    policy_lse = policy_logits[:, :-1, :].logsumexp(dim=-1)
    policy_chosen = policy_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    policy_logprobs = policy_chosen - policy_lse
    del policy_logits, policy_lse, policy_chosen
    log_mem("grpo_loss_after_policy_fwd")
    policy_probs = torch.exp(
        policy_logprobs - policy_logprobs.detach()
    ) * advantages.unsqueeze(1)

    # Compute kl divergence (second term)
    if config.grpo_beta > 0:
        with torch.no_grad():
            ref_logits = ref_model(**encodings).logits.detach()
            ref_lse = ref_logits[:, :-1, :].logsumexp(dim=-1)
            ref_chosen = ref_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            ref_logprobs = ref_chosen - ref_lse
            del ref_logits, ref_lse, ref_chosen
        log_mem("grpo_loss_after_ref_fwd")
        log_ref_ratio = policy_logprobs - ref_logprobs
        kl_divergence = torch.exp(log_ref_ratio) - log_ref_ratio - 1
        token_level_loss = -(policy_probs - config.grpo_beta * kl_divergence)
    else:
        kl_divergence = None
        token_level_loss = -policy_probs
    loss = (token_level_loss * mask).sum() / mask.sum()
    return loss, kl_divergence


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
    5. Run backward on losses
    6. Return combined loss and metrics

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
    model.eval()
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
    all_back_texts: list[list[list[str]]] = []  # [batch, g_fwd, g_bwd]
    all_bwd_prompts = []  # flat list for loss computation
    all_bwd_completions = []
    for i in range(batch_size):
        if not config.greedy_backward:
            group_back: list[list[str]] = []
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
        else:
            # Greedy, just a single
            bwd_prompts = [
                make_backward_prompt(fwd_text, config.language)
                for fwd_text in fwd_texts[i]
            ]
            bwd_texts_i, _ = greedy_decode(
                model,
                tokenizer,
                prompts=bwd_prompts,
                max_new_tokens=config.max_tokens,
            )
            all_bwd_prompts.extend(bwd_prompts)
            all_bwd_completions.extend(bwd_texts_i)
            all_back_texts.append([[t] for t in bwd_texts_i])

    # Step 3: Compute rewards
    forward_rewards, backward_rewards = compute_cycle_rewards(
        english_sentences,
        fwd_texts,
        all_back_texts,
        metric=config.reward_metric,
    )
    forward_rewards = forward_rewards.to(model.device)
    backward_rewards = backward_rewards.to(model.device)

    torch.cuda.empty_cache()
    model.train()
    log_mem("after_all_generation")

    # Step 4: GRPO loss for backward step (target -> eng)
    bwd_loss = 0.0
    bwd_kl_div = 0.0

    bwd_std = backward_rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
    bwd_advantages = (
        backward_rewards - backward_rewards.mean(dim=-1, keepdim=True)
    ) / bwd_std
    flat_bwd_advantages = bwd_advantages.reshape(-1)

    if config.alpha > 0:
        # Run in batches so we use a constant amount of mem
        for group_idx in range(config.grpo_group_size):
            start_idx = group_idx * config.grpo_group_size * batch_size
            end_idx = (group_idx + 1) * config.grpo_group_size * batch_size
            group_bwd_loss, group_bwd_kl_div = _compute_grpo_loss(
                model,
                ref_model,
                tokenizer,
                all_bwd_prompts[start_idx:end_idx],
                all_bwd_completions[start_idx:end_idx],
                flat_bwd_advantages[start_idx:end_idx],
                config,
            )
            group_bwd_loss /= config.gradient_accumulation_steps
            group_bwd_loss *= config.alpha
            log_mem(f"after_bwd_loss_g{group_idx}")
            group_bwd_loss.backward()
            bwd_loss += group_bwd_loss.detach().item()
            log_mem(f"after_bwd_loss_g{group_idx}_backward")
            bwd_kl_div += (
                group_bwd_kl_div.mean().item() if group_bwd_kl_div is not None else 0
            )
        bwd_kl_div /= config.grpo_group_size
        log_mem("after_all_bwd_loss")

    # Step 5: GRPO loss for forward step (eng -> target)
    flat_fwd_prompts = []
    flat_fwd_completions = []
    for i in range(batch_size):
        for j in range(config.grpo_group_size):
            flat_fwd_prompts.append(fwd_prompts[i])
            flat_fwd_completions.append(fwd_texts[i][j])

    fwd_std = forward_rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
    fwd_advantages = (forward_rewards - forward_rewards.mean(dim=-1, keepdim=True)) / (
        fwd_std
    )
    flat_fwd_advantages = fwd_advantages.reshape(-1)

    if config.alpha < 1:
        fwd_loss, fwd_kl_div = _compute_grpo_loss(
            model,
            ref_model,
            tokenizer,
            flat_fwd_prompts,
            flat_fwd_completions,
            flat_fwd_advantages,
            config,
        )
        fwd_loss /= config.gradient_accumulation_steps
        fwd_loss *= 1 - config.alpha
        log_mem("after_fwd_loss")
        fwd_loss.backward()
        fwd_loss = fwd_loss.detach().item()
        log_mem("after_fwd_loss_backward")
        fwd_kl_div = fwd_kl_div.mean().item() if fwd_kl_div is not None else 0.0
    else:
        fwd_loss = 0.0
        fwd_kl_div = 0.0

    # The total loss is detached and just for logging purposes
    total_loss = fwd_loss + bwd_loss

    # Build example table rows for wandb (up to 10 source sentences)
    num_examples = min(10, batch_size)
    g = config.grpo_group_size
    example_rows = [
        [english_sentences[i], j, fwd_texts[i][j], k, all_back_texts[i][j][k]]
        for i in range(num_examples)
        for j in range(g)
        for k in range(g)
    ]

    normalized_fwd_rewards = forward_rewards / config.grpo_group_size
    metrics = {
        "loss": total_loss,
        "fwd_loss": fwd_loss,
        "bwd_loss": bwd_loss,
        "mean_fwd_reward": normalized_fwd_rewards.mean().item(),
        "mean_bwd_reward": backward_rewards.mean().item(),
        "mean_total_reward": (
            normalized_fwd_rewards.mean() + backward_rewards.mean()
        ).item(),
        "fwd_kl_div": fwd_kl_div,
        "bwd_kl_div": bwd_kl_div,
        "mean_fwd_std": fwd_std.mean().item(),
        "mean_bwd_std": bwd_std.mean().item(),
    }

    return {"loss": total_loss, "metrics": metrics, "example_rows": example_rows}
