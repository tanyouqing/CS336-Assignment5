"""Additional semantic checks; run on Linux with the original course suite."""

from __future__ import annotations

import copy

import pytest
import torch

from cs336_alignment.core import (
    aggregate_loss_across_microbatch,
    compute_group_normalized_rewards,
    compute_policy_gradient_loss,
    grpo_train_step,
)
from cs336_alignment.data import split_rows
from cs336_alignment.dpo import cached_dpo_loss, compute_per_instance_dpo_loss
from cs336_alignment.evaluation import parse_safety_judgment
from cs336_alignment.runtime import restore_state, save_checkpoint


@pytest.mark.parametrize("normalization", ["sequence", "constant"])
def test_accumulation_keeps_original_denominator(normalization):
    full = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    whole = aggregate_loss_across_microbatch(full, mask, normalization, 32)
    (expected,) = torch.autograd.grad(whole, full)
    fragmented = full.detach().clone().requires_grad_()
    for start, end in ((0, 2), (2, 3)):
        loss = aggregate_loss_across_microbatch(
            fragmented[start:end], mask[start:end], normalization, 32
        )
        if normalization == "sequence":
            loss *= (end - start) / 3
        loss.backward()
    torch.testing.assert_close(fragmented.grad, expected)


def test_gspo_excludes_prompt_and_padding_from_ratio():
    log_probs = torch.tensor([[99.0, 0.1, 0.2, -99.0]], requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    per_token, _ = compute_policy_gradient_loss(
        torch.ones(1), log_probs, "gspo", torch.zeros_like(log_probs), 10.0, mask
    )
    loss = aggregate_loss_across_microbatch(per_token, mask)
    (gradient,) = torch.autograd.grad(loss, log_probs)
    assert gradient[0, 0] == gradient[0, 3] == 0
    torch.testing.assert_close(loss, -torch.tensor(0.15).exp())


def test_full_rollout_advantages_are_not_minibatch_advantages():
    raw = torch.tensor([1.0, 0.0, 0.0, 0.0])
    global_advantages, _ = compute_group_normalized_rewards(
        raw, 4, advantage_normalizer="none"
    )
    local_advantages, _ = compute_group_normalized_rewards(
        raw[:2], 2, advantage_normalizer="none"
    )
    assert not torch.equal(global_advantages[:2], local_advantages)


def test_loo_is_finite_group_rescaling():
    raw = torch.tensor([1.0, 0.0, 0.0, 1.0])
    mean, _ = compute_group_normalized_rewards(raw, 4, advantage_normalizer="none")
    loo, _ = compute_group_normalized_rewards(
        raw, 4, baseline="loo", advantage_normalizer="none"
    )
    torch.testing.assert_close(loo, mean * 4 / 3)


def test_zero_advantage_pruning_preserves_update(tiny_train_model, tokenizer):
    model1, model2 = copy.deepcopy(tiny_train_model), copy.deepcopy(tiny_train_model)
    prompts, responses = ["Hello"] * 4, ["world", "test", "world", "test"]
    weights = torch.tensor([1.0, 0.0, -1.0, 0.0])

    def reward(y, truth):
        return {"reward": 1.0, "format_reward": 1.0, "answer_reward": 1.0}

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    from cs336_alignment.core import get_response_log_probs, tokenize_prompt_and_output

    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    scores = get_response_log_probs(model1, batch["input_ids"], batch["labels"])[
        "log_probs"
    ]
    loss = aggregate_loss_across_microbatch(
        -weights[:, None] * scores, batch["response_mask"]
    )
    loss.backward()
    optimizer1.step()
    grpo_train_step(
        model2,
        tokenizer,
        optimizer2,
        4,
        None,
        reward,
        prompts,
        responses,
        ["42"] * 4,
        4,
        precomputed_advantages=weights,
        microbatch_size=1,
    )
    for left, right in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(left, right)


def test_dpo_margin_sign_and_gradient():
    scores = {
        "chosen": torch.tensor(-2.0, requires_grad=True),
        "rejected": torch.tensor(-3.0, requires_grad=True),
    }
    ref_scores = {"chosen": -2.5, "rejected": -2.8}
    loss, margin = cached_dpo_loss(scores, ref_scores, 0.1)
    expected = -torch.nn.functional.logsigmoid(torch.tensor(0.07))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert scores["chosen"].grad < 0 and scores["rejected"].grad > 0


def test_reference_cache_matches_online_and_rejects_changed_tokens(
    tmp_path, tiny_train_model, tokenizer
):
    from cs336_alignment.dpo import preference_tokens, score_preference
    from cs336_alignment.supplement import reference_cache
    from cs336_alignment.runtime import RunLog
    from cs336_alignment.data import row_hash

    rows = [{"prompt": "Hello", "chosen": "world", "rejected": "test", "source": "toy"}]
    config = {
        "model": "tiny-test",
        "adapter": None,
        "max_length": 16,
        "qlora": False,
        "attention": "sdpa",
        "output": str(tmp_path),
        "reference_cache": None,
    }
    log = RunLog(tmp_path, config)
    cache = reference_cache(tiny_train_model, tokenizer, rows, config, log, None)
    online = score_preference(
        tiny_train_model, preference_tokens(tokenizer, rows[0], 16)
    )
    for side in ("chosen", "rejected"):
        torch.testing.assert_close(
            online[side].detach(),
            torch.tensor(cache["scores"][row_hash(rows[0])][side]),
        )
    changed = [{**rows[0], "chosen": "another test"}]
    with pytest.raises(ValueError, match="fingerprint"):
        reference_cache(tiny_train_model, tokenizer, changed, config, log, None)
    assert all(p.grad is None for p in tiny_train_model.parameters())
    log.close()


def test_per_instance_dpo_reference_is_frozen(tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_embd=8,
        n_head=2,
        n_layer=1,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    policy, reference = GPT2LMHeadModel(config), GPT2LMHeadModel(config)
    loss = compute_per_instance_dpo_loss(
        policy, reference, tokenizer, 0.1, "Hello", "world", "test"
    )
    loss.backward()
    assert any(p.grad is not None for p in policy.parameters())
    assert all(p.grad is None for p in reference.parameters())


def test_checkpoint_roundtrip_and_optimizer_resume(
    tmp_path, tiny_train_model, tokenizer
):
    model = tiny_train_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ids = torch.tensor([[3, 4, 5]])

    def update():
        optimizer.zero_grad(set_to_none=True)
        model(ids).logits.square().mean().backward()
        optimizer.step()

    update()
    save_checkpoint(model, tokenizer, optimizer, tmp_path / "ckpt", 1, {"test": True})
    update()
    expected = {k: p.detach().clone() for k, p in model.named_parameters()}
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tmp_path / "ckpt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    state = restore_state(tmp_path / "ckpt", optimizer, {"test": True})
    assert state["step"] == 1
    update()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name])


def test_deduplicated_split_and_invalid_judge():
    rows = [{"prompt": str(i), "response": str(i)} for i in range(8)]
    train, val = split_rows(rows + rows, 2, 0)
    assert len(train) + len(val) == 8
    assert {r["prompt"] for r in train}.isdisjoint(r["prompt"] for r in val)
    assert parse_safety_judgment("False") is True
    assert parse_safety_judgment("True") is False
    assert parse_safety_judgment("Probably safe") is None
