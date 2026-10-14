"""Tiny CPU end-to-end tests intended for the Linux validation stage."""

from __future__ import annotations

import gzip
import json

import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from cs336_alignment.cli import COMMON, SUPPLEMENT
from cs336_alignment.data import write_jsonl
from cs336_alignment.supplement import train_supplement


def test_sft_resume_matches_uninterrupted_and_dpo_runs(tmp_path, tokenizer):
    torch.manual_seed(12)
    model_path = tmp_path / "base"
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=128,
            n_embd=8,
            n_head=2,
            n_layer=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    data_path = tmp_path / "sft.jsonl"
    write_jsonl(
        data_path,
        [{"prompt": f"Hello {i}", "response": "world test"} for i in range(20)],
    )
    defaults = (
        COMMON
        | SUPPLEMENT
        | {
            "model": str(model_path),
            "device": "cpu",
            "qlora": False,
            "attention": "eager",
            "gradient_checkpointing": False,
            "max_length": 16,
            "train_data": str(data_path),
            "n_train": 12,
            "n_val": 2,
            "train_batch_size": 2,
            "eval_every": 1,
            "validation_sequences": 2,
            "checkpoint_every": 1,
            "kind": "sft",
        }
    )
    resumed_dir, uninterrupted_dir = tmp_path / "resumed", tmp_path / "uninterrupted"
    train_supplement(defaults | {"output": str(resumed_dir), "steps": 1})
    train_supplement(
        defaults
        | {"output": str(resumed_dir), "steps": 2, "resume": str(resumed_dir / "last")}
    )
    train_supplement(defaults | {"output": str(uninterrupted_dir), "steps": 2})
    resumed = AutoModelForCausalLM.from_pretrained(resumed_dir / "last")
    uninterrupted = AutoModelForCausalLM.from_pretrained(uninterrupted_dir / "last")
    for a, b in zip(resumed.parameters(), uninterrupted.parameters()):
        torch.testing.assert_close(a, b)
    initial = AutoModelForCausalLM.from_pretrained(model_path)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(initial.parameters(), resumed.parameters())
    )

    hh = tmp_path / "hh"
    hh.mkdir()
    for source in (
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled",
    ):
        with gzip.open(hh / f"{source}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for i in range(4):
                prompt = f"\n\nHuman: Hello {source} {i}\n\nAssistant: "
                handle.write(
                    json.dumps(
                        {"chosen": prompt + "world", "rejected": prompt + "test"}
                    )
                    + "\n"
                )
    dpo_config = defaults | {
        "kind": "dpo",
        "model": str(resumed_dir / "last"),
        "train_data": str(hh),
        "per_source": 2,
        "n_val": 4,
        "steps": 1,
        "learning_rate": 1e-3,
        "output": str(tmp_path / "dpo"),
    }
    train_supplement(dpo_config)
    assert (tmp_path / "dpo" / "reference.json").exists()
    assert (tmp_path / "dpo" / "last" / "COMPLETE.json").exists()
    updated = AutoModelForCausalLM.from_pretrained(tmp_path / "dpo" / "last")
    assert any(
        not torch.equal(a, b)
        for a, b in zip(updated.parameters(), resumed.parameters())
    )
