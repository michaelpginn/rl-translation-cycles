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


# fwd_std = forward_rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
# fwd_advantages = (forward_rewards - forward_rewards.mean(dim=-1, keepdim=True)) / (
#     fwd_std
# )
# flat_fwd_advantages = fwd_advantages.reshape(-1)

# if config.alpha < 1:
#     fwd_loss, fwd_kl_div = _compute_grpo_loss(
#         model,
#         ref_model,
#         tokenizer,
#         flat_fwd_prompts,
#         flat_fwd_completions,
#         flat_fwd_advantages,
#         config,
#     )
#     fwd_loss /= config.gradient_accumulation_steps
#     fwd_loss *= 1 - config.alpha
#     log_mem("after_fwd_loss")
#     fwd_loss.backward()
#     fwd_loss = fwd_loss.detach().item()
#     log_mem("after_fwd_loss_backward")
#     fwd_kl_div = fwd_kl_div.mean().item() if fwd_kl_div is not None else 0.0
# else:
#     fwd_loss = 0.0
#     fwd_kl_div = 0.0

# # The total loss is detached and just for logging purposes
# total_loss = fwd_loss + bwd_loss

# # Build example table rows for wandb (up to 10 source sentences)
# num_examples = min(10, batch_size)
# g = config.grpo_group_size
# example_rows = [
#     [english_sentences[i], j, fwd_texts[i][j], k, all_back_texts[i][j][k]]
#     for i in range(num_examples)
#     for j in range(g)
#     for k in range(g)
# ]

# normalized_fwd_rewards = forward_rewards / config.grpo_group_size
# metrics = {
#     "loss": total_loss,
#     "fwd_loss": fwd_loss,
#     "bwd_loss": bwd_loss,
#     "mean_fwd_reward": normalized_fwd_rewards.mean().item(),
#     "mean_bwd_reward": backward_rewards.mean().item(),
#     "mean_total_reward": (
#         normalized_fwd_rewards.mean() + backward_rewards.mean()
#     ).item(),
#     "fwd_kl_div": fwd_kl_div,
#     "bwd_kl_div": bwd_kl_div,
#     "mean_fwd_std": fwd_std.mean().item(),
#     "mean_bwd_std": bwd_std.mean().item(),
# }

# return {"loss": total_loss, "metrics": metrics, "example_rows": example_rows}


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
        - rewards (Tensor): (bs, gs) unnormalized float rewards
    """
    batch_size = len(english_sentences)

    # 1. Forward translation (eng -> target)
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

    # 2. Back translation (target -> eng) for each forward candidate
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
            # Greedy, just a single backtrans
            bwd_prompts = [
                make_backward_prompt(fwd_text, config.language)
                for fwd_text in fwd_texts[i]
            ]
            bwd_texts_i = greedy_decode(
                model,
                tokenizer,
                prompts=bwd_prompts,
                max_new_tokens=config.max_tokens,
            )
            all_bwd_prompts.extend(bwd_prompts)
            all_bwd_completions.extend(bwd_texts_i)
            all_back_texts.append([[t] for t in bwd_texts_i])

    # 3. Compute rewards
    forward_rewards, _ = compute_cycle_rewards(
        english_sentences,
        fwd_texts,
        all_back_texts,
        metric=config.reward_metric,
    )
    forward_rewards = forward_rewards.to(model.device)
    return fwd_prompts, fwd_texts, all_back_texts, forward_rewards


def compute_logprobs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    completions: list[list[str]],
    with_grad: bool,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes gradient-free logprobs for a list of prompts and nested list of completions.

    Returns:
        - policy_logprobs: (bs * gs, seq) tensor of log probs
        - mask: (bs * gs, seq) mask for completion
    """
    flat_prompts = [p for p in prompts for _ in range(config.grpo_group_size)]
    flat_completions = [c for group in completions for c in group]
    full_texts = [p + " " + c for p, c in zip(flat_prompts, flat_completions)]
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

    log_mem(f"grpo_loss_before_old_policy_fwd (n_seqs={len(prompts)})")
    logger.debug(f"Size of inputs: {encodings.input_ids.shape}")
    with torch.set_grad_enabled(with_grad):
        policy_logits = model(**encodings).logits
        policy_lse = policy_logits[:, :-1, :].logsumexp(dim=-1)
        policy_chosen = policy_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        policy_logprobs = policy_chosen - policy_lse
        del policy_logits, policy_lse, policy_chosen
    log_mem("grpo_loss_after_policy_fwd")
    return policy_logprobs, mask


def compute_grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    rewards: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = (
        (rewards - rewards.mean(dim=-1, keepdim=True))
        / (rewards.std(dim=-1, keepdim=True).clamp(min=1e-8))
    ).reshape(-1)  # (bs*gs,)
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
    loss = (token_level_loss * mask).sum() / mask.sum()
    return loss, kl_divergence
