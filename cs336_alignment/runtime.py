"""Single-GPU loading, generation, telemetry and resumable checkpoints."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def software_versions():
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    for package in ("torch", "transformers", "peft", "bitsandbytes", "vllm"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def load_model(
    model_path,
    device="cuda:0",
    qlora=False,
    adapter=None,
    train=True,
    gradient_checkpointing=True,
    attention="sdpa",
    lora_rank=16,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use --device cpu only for tiny-model verification"
        )
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The single-GPU profile requires BF16 support")
    tokenizer_source = (
        adapter
        if adapter and (Path(adapter) / "tokenizer_config.json").exists()
        else model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    options = {
        "torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
        "attn_implementation": attention,
    }
    if qlora:
        if not device.startswith("cuda"):
            raise ValueError("The supported QLoRA profile requires a CUDA GPU")
        from transformers import BitsAndBytesConfig

        options.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": device},
        )
    model = AutoModelForCausalLM.from_pretrained(model_path, **options)
    if not qlora:
        model.to(device)
    if qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing and train
        )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif qlora and train:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.0,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
    if train and gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if train:
        # Deterministic scoring matches the rollout policy; stochastic dropout
        # would make repeated old/new log-prob comparisons meaningless.
        for layer in model.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.0
        model.train()
        model.config.use_cache = False
    else:
        model.eval()
    return model, tokenizer


class StopText(StoppingCriteria):
    def __init__(self, tokenizer, prefix_length, stop):
        self.tokenizer, self.prefix_length, self.stop = tokenizer, prefix_length, stop

    def __call__(self, input_ids, scores, **kwargs):
        return torch.tensor(
            [
                self.stop
                in self.tokenizer.decode(
                    ids[self.prefix_length :], skip_special_tokens=True
                )
                for ids in input_ids
            ],
            device=input_ids.device,
        )


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts,
    *,
    max_tokens=512,
    temperature=1.0,
    stop=None,
    batch_size=1,
    deadline=None,
):
    """Generate serial microbatches on the SAME model used for training."""
    previous_mode, previous_cache = model.training, model.config.use_cache
    device = next(model.parameters()).device
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    outputs = []
    try:
        for start in range(0, len(prompts), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Generation deadline reached")
            inputs = tokenizer(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            prefix_length = inputs.input_ids.shape[1]
            kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "top_p": 1.0,
            }
            if temperature > 0:
                kwargs.update(temperature=temperature, top_k=0)
            criteria = []
            if stop:
                criteria.append(StopText(tokenizer, prefix_length, stop))
            if deadline is not None:

                class Deadline(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return time.monotonic() >= deadline

                criteria.append(Deadline())
            if criteria:
                kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)
            ids = model.generate(**inputs, **kwargs)[:, prefix_length:]
            for tokens in ids:
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                if stop and stop in text:
                    text = text[: text.index(stop) + len(stop)]
                outputs.append(text)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Generation deadline reached; incomplete rollout is discarded"
                )
    finally:
        tokenizer.padding_side = previous_padding
        model.config.use_cache = previous_cache
        model.train(previous_mode)
    return outputs


class RunLog:
    def __init__(self, directory, config, wandb=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.wandb = None
        self.event("start", config=config, environment=software_versions())
        (self.directory / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        if wandb:
            import wandb as wb

            self.wandb = wb.init(
                project="cs336-a5", config=config, dir=str(self.directory)
            )

    def event(self, event, **data):
        row = {
            "event": event,
            "elapsed_seconds": time.monotonic() - self.started,
            **data,
        }
        if torch.cuda.is_available():
            row.update(
                peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
            )
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
        if self.wandb is not None:
            self.wandb.log(
                {k: v for k, v in row.items() if isinstance(v, (int, float))}
            )

    def close(self):
        if self.wandb is not None:
            self.wandb.finish()


def save_checkpoint(model, tokenizer, optimizer, directory, step, config, extra=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "COMPLETE.json"
    marker.unlink(missing_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "config": config,
        "extra": extra or {},
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    temporary = directory / "training_state.tmp"
    torch.save(state, temporary)
    os.replace(temporary, directory / "training_state.pt")
    (directory / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    marker.write_text(json.dumps({"step": step}), encoding="utf-8")


def restore_state(directory, optimizer, expected_config=None):
    if not (Path(directory) / "COMPLETE.json").exists():
        raise ValueError(
            "Checkpoint is incomplete; select a committed periodic checkpoint"
        )
    # Only load locally generated/trusted checkpoints: this state includes RNG
    # objects and therefore uses the unrestricted PyTorch pickle loader.
    state = torch.load(
        Path(directory) / "training_state.pt", map_location="cpu", weights_only=False
    )
    if expected_config is not None:
        ignored = {
            "output",
            "resume",
            "max_seconds",
            "wandb",
            "steps",
            "checkpoint_every",
        }
        changes = [
            k
            for k, v in state["config"].items()
            if k not in ignored and expected_config.get(k) != v
        ]
        if changes:
            raise ValueError(f"Resume would change training semantics: {changes}")
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
