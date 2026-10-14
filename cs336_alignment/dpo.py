"""Exact per-instance DPO plus response-only scoring for offline caches."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .core import get_response_log_probs
from .data import sft_text


def compute_per_instance_dpo_loss(
    lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected
):
    def score(model, response):
        device = next(model.parameters()).device
        ids = torch.tensor(
            tokenizer.encode(sft_text(prompt, response) + tokenizer.eos_token),
            device=device,
        ).unsqueeze(0)
        return get_response_log_probs(model, ids[:, :-1], ids[:, 1:])["log_probs"].sum()

    positive = score(lm, response_chosen)
    negative = score(lm, response_rejected)
    with torch.no_grad():
        ref_positive = score(lm_ref, response_chosen).to(positive.device)
        ref_negative = score(lm_ref, response_rejected).to(positive.device)
    return -F.logsigmoid(beta * ((positive - negative) - (ref_positive - ref_negative)))


def preference_tokens(tokenizer, row, max_length):
    """Both alternatives share identical prompt tokens; retain final EOS.

    Long prompts use their rightmost tokens; responses use their first tokens.
    At least half the context remains available for the response.
    """
    if max_length < 4:
        raise ValueError("DPO context must be >= 4")
    prefix = tokenizer.encode(sft_text(row["prompt"], ""), add_special_tokens=True)
    prefix = prefix[-max_length // 2 :]
    result = {}
    for key in ("chosen", "rejected"):
        output = tokenizer.encode(row[key], add_special_tokens=False)
        output = output[: max_length - len(prefix) - 1] + [tokenizer.eos_token_id]
        ids = prefix + output
        result[key] = {
            "input_ids": torch.tensor(ids[:-1]).unsqueeze(0),
            "labels": torch.tensor(ids[1:]).unsqueeze(0),
            "mask": torch.tensor(
                [False] * (len(prefix) - 1) + [True] * len(output)
            ).unsqueeze(0),
        }
    return result


def score_preference(model, tokens):
    device = next(model.parameters()).device
    result = {}
    for key, item in tokens.items():
        score = get_response_log_probs(
            model, item["input_ids"].to(device), item["labels"].to(device)
        )["log_probs"]
        result[key] = (score * item["mask"].to(device)).sum()
    return result


def cached_dpo_loss(scores, reference, beta):
    device = scores["chosen"].device
    ref_margin = torch.as_tensor(
        reference["chosen"] - reference["rejected"], device=device
    )
    margin = (scores["chosen"] - scores["rejected"]) - ref_margin
    return -F.logsigmoid(beta * margin), margin.detach()
