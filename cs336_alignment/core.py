"""Hand-written, backend-independent policy gradient primitives.

N counts responses, not prompts. Token tensors have shape (N, T), and masks
are aligned with next-token labels rather than unshifted input tokens.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    if not prompt_strs or len(prompt_strs) != len(output_strs):
        raise ValueError("Expected equally sized, nonempty prompt/response lists")
    prompts = [tokenizer.encode(s, add_special_tokens=False) for s in prompt_strs]
    responses = [tokenizer.encode(s, add_special_tokens=False) for s in output_strs]
    if any(not p for p in prompts):
        raise ValueError("Each prompt must contain at least one token")
    sequences = [p + r for p, r in zip(prompts, responses)]
    width = max(map(len, sequences))
    if width < 2:
        raise ValueError("Need at least two tokens to construct next-token labels")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, (p, sequence) in enumerate(zip(prompts, sequences)):
        ids[i, : len(sequence)] = torch.tensor(sequence)
        mask[i, len(p) : len(sequence)] = True
    return {
        "input_ids": ids[:, :-1],
        "labels": ids[:, 1:],
        "response_mask": mask[:, 1:],
    }


def compute_entropy(logits):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(-1)


def get_response_log_probs(
    model, input_ids, labels, return_token_entropy=False, chunk_size=128
):
    logits = model(input_ids).logits
    scores, entropies = [], []

    # Avoid materializing a second full (N,T,V) fp32 tensor. The model's logits
    # still occupy memory; microbatch size controls that allocation.
    def score_block(block, target):
        logp = F.log_softmax(block.float(), dim=-1)
        selected = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            entropy = (
                -(logp.detach().exp() * logp.detach()).sum(-1)
                if return_token_entropy
                else torch.empty(0, device=block.device)
            )
        return selected, entropy

    for start in range(0, logits.shape[1], chunk_size):
        block = logits[:, start : start + chunk_size]
        target = labels[:, start : start + chunk_size]
        # Merely chunking softmax would still retain all fp32 softmax outputs
        # for backward. Recompute them instead, keeping only original logits.
        selected, entropy = (
            checkpoint(score_block, block, target, use_reentrant=False)
            if logits.requires_grad
            else score_block(block, target)
        )
        scores.append(selected)
        if return_token_entropy:
            entropies.append(entropy)
    result = {"log_probs": torch.cat(scores, dim=1)}
    if return_token_entropy:
        result["token_entropy"] = torch.cat(entropies, dim=1)
    return result


def compute_rollout_rewards(
    reward_fn: Callable, rollout_responses, repeated_ground_truths
):
    if not rollout_responses or len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("Rewards require nonempty aligned responses and answers")
    grades = [
        reward_fn(r, g) for r, g in zip(rollout_responses, repeated_ground_truths)
    ]
    rewards = torch.tensor([g["reward"] for g in grades], dtype=torch.float32)
    return rewards, {
        key: sum(float(g[key]) for g in grades) / len(grades)
        for key in ("reward", "format_reward", "answer_reward")
    }


def compute_group_normalized_rewards(
    raw_rewards,
    group_size,
    baseline="mean",
    advantage_eps=1e-6,
    advantage_normalizer="std",
):
    if group_size < 1 or raw_rewards.numel() % group_size:
        raise ValueError("Responses must form complete groups")
    groups = raw_rewards.reshape(-1, group_size)
    means = groups.mean(-1, keepdim=True)
    if baseline == "mean":
        advantages = groups - means
    elif baseline == "none":
        advantages = groups.clone()
    elif baseline == "loo":
        if group_size < 2:
            raise ValueError("Leave-one-out requires at least two responses")
        advantages = groups - (groups.sum(-1, keepdim=True) - groups) / (group_size - 1)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    if advantage_normalizer == "std":
        if group_size < 2:
            raise ValueError("Sample standard deviation requires group_size >= 2")
        advantages = advantages / (groups.std(-1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (means + advantage_eps)
    elif advantage_normalizer != "none":
        raise ValueError(f"Unknown advantage normalizer: {advantage_normalizer}")
    return advantages.reshape(-1), {
        "reward_mean": raw_rewards.mean().item(),
        "advantage_abs_mean": advantages.abs().mean().item(),
    }


def compute_policy_gradient_loss(
    raw_rewards_or_advantages,
    policy_log_probs,
    importance_reweighting_method="none",
    old_log_probs=None,
    cliprange=None,
    response_mask=None,
):
    advantages = raw_rewards_or_advantages.detach().reshape(-1, 1).to(policy_log_probs)
    method = importance_reweighting_method
    if method == "none":
        return -advantages * policy_log_probs, {}
    if old_log_probs is None or old_log_probs.shape != policy_log_probs.shape:
        raise ValueError("Importance sampling requires aligned old_log_probs")
    difference = policy_log_probs - old_log_probs.detach().to(policy_log_probs)
    if method == "gspo":
        mask = (
            torch.ones_like(difference)
            if response_mask is None
            else response_mask.to(difference)
        )
        lengths = mask.sum(-1, keepdim=True)
        if (lengths == 0).any():
            raise ValueError("GSPO requires at least one response token")
        ratio = ((difference * mask).sum(-1, keepdim=True) / lengths).exp()
    else:
        ratio = difference.exp()
    if method == "noclip":
        return -advantages * ratio, {}
    if method not in ("grpo", "gspo") or cliprange is None or cliprange < 0:
        raise ValueError("Expected grpo/gspo with nonnegative cliprange")
    clipped = ratio.clamp(1 - cliprange, 1 + cliprange)
    objective = torch.minimum(advantages * ratio, advantages * clipped)
    # GSPO has one scalar loss per response; expand for the common masked
    # sequence aggregator without taking another length factor in the gradient.
    loss = -objective.expand_as(policy_log_probs)
    active_clip = ((advantages >= 0) & (ratio > 1 + cliprange)) | (
        (advantages < 0) & (ratio < 1 - cliprange)
    )
    return loss, {
        "clip_fraction": active_clip.float().expand_as(policy_log_probs).detach()
    }


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss,
    mask,
    loss_normalization="sequence",
    normalization_constant=None,
):
    masked = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        return (masked.sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    if loss_normalization == "constant":
        if normalization_constant is None or normalization_constant <= 0:
            raise ValueError("A positive fixed normalizer is required")
        return masked.sum() / normalization_constant
    raise ValueError(f"Unknown loss normalization: {loss_normalization}")


def grpo_train_step(
    model,
    tokenizer,
    optimizer,
    gradient_accumulation_steps,
    max_grad_norm,
    reward_fn,
    repeated_prompts,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    baseline="mean",
    advantage_eps=1e-6,
    advantage_normalizer="std",
    importance_reweighting_method="none",
    old_log_probs=None,
    cliprange=None,
    loss_normalization="sequence",
    normalization_constant=None,
    *,
    precomputed_advantages=None,
    microbatch_size=None,
):
    """One update; precomputed advantages preserve full-rollout groups off policy.

    Sequence losses are weighted by ORIGINAL batch size, even after pruning.
    Constant losses are already divided by the full batch's fixed normalizer.
    """
    n = len(rollout_responses)
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    rewards, stats = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    if precomputed_advantages is None:
        advantages, advantage_stats = compute_group_normalized_rewards(
            rewards, group_size, baseline, advantage_eps, advantage_normalizer
        )
        stats.update(advantage_stats)
    else:
        advantages = precomputed_advantages.detach().cpu().reshape(-1)
        if advantages.numel() != n:
            raise ValueError("Wrong precomputed advantage count")
    batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    if old_log_probs is not None and old_log_probs.shape != batch["input_ids"].shape:
        raise ValueError("Old log-probs must align with the padded training batch")
    device = next(model.parameters()).device
    size = microbatch_size or math.ceil(n / gradient_accumulation_steps)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    entropy_sum, token_count, clip_sum = 0.0, 0, 0.0
    active = advantages.nonzero().flatten()
    for start in range(0, len(active), size):
        indices = active[start : start + size]
        inputs = batch["input_ids"][indices].to(device)
        labels = batch["labels"][indices].to(device)
        mask = batch["response_mask"][indices].to(device)
        scored = get_response_log_probs(model, inputs, labels, True)
        old = None if old_log_probs is None else old_log_probs[indices].to(device)
        per_token, info = compute_policy_gradient_loss(
            advantages[indices].to(device),
            scored["log_probs"],
            importance_reweighting_method,
            old,
            cliprange,
            mask,
        )
        loss = aggregate_loss_across_microbatch(
            per_token, mask, loss_normalization, normalization_constant
        )
        if loss_normalization == "sequence":
            loss = loss * (len(indices) / n)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(
                "Nonfinite policy loss; checkpoint retained by runner"
            )
        loss.backward()
        loss_sum += loss.detach()
        count = mask.sum().item()
        token_count += count
        entropy_sum += (scored["token_entropy"].detach() * mask).sum().item()
        if "clip_fraction" in info:
            clip_sum += (info["clip_fraction"] * mask).sum().item()
    if len(active):
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm if max_grad_norm is not None else float("inf"),
        )
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step()
    else:
        norm = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)
    stats.update(
        loss=loss_sum.item(),
        gradient_norm=float(norm),
        token_entropy=entropy_sum / max(token_count, 1),
        clip_fraction=clip_sum / max(token_count, 1),
        active_responses=len(active),
        scored_response_tokens=token_count,
    )
    return loss_sum, stats
