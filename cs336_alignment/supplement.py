"""Handwritten SFT/DPO loops with QLoRA and frozen offline reference scores."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from .core import get_response_log_probs
from .data import (
    PackedSFTDataset,
    load_hh,
    read_jsonl,
    row_hash,
    sample_rows,
    split_rows,
    write_jsonl,
)
from .dpo import cached_dpo_loss, preference_tokens, score_preference
from .runtime import RunLog, load_model, restore_state, save_checkpoint, seed_all


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_identity(config, tokenizer, rows, model):
    checkpoint = Path(config.get("adapter") or config["model"])
    files = {}
    if checkpoint.is_dir():
        for path in sorted(checkpoint.iterdir()):
            if path.suffix in (".json", ".safetensors", ".bin"):
                files[path.name] = file_digest(path)
    base_files = {}
    base = Path(config["model"])
    if base.is_dir() and base.resolve() != checkpoint.resolve():
        for path in sorted(base.iterdir()):
            if path.suffix in (".json", ".safetensors", ".bin"):
                base_files[path.name] = file_digest(path)
    token_hash = hashlib.sha256()
    for row in rows:
        tokens = preference_tokens(tokenizer, row, config["max_length"])
        for side in ("chosen", "rejected"):
            for key in ("input_ids", "labels", "mask"):
                token_hash.update(tokens[side][key].numpy().tobytes())
    metadata = {
        "base_model": config["model"],
        "base_revision": getattr(model.config, "_commit_hash", None),
        "base_config": model.config.to_dict(),
        "reference_checkpoint": str(checkpoint.resolve()),
        "checkpoint_hashes": files,
        "base_files": base_files,
        "tokenizer_vocab_hash": row_hash(tokenizer.get_vocab()),
        "tokenizer_special_tokens": str(tokenizer.special_tokens_map),
        "token_hash": token_hash.hexdigest(),
        "row_hashes": [row_hash(row) for row in rows],
        "max_length": config["max_length"],
        "qlora": config["qlora"],
        "attention": config["attention"],
        "scoring_version": 1,
    }
    return row_hash(metadata), metadata


def reference_cache(model, tokenizer, rows, config, log, deadline):
    fingerprint, metadata = reference_identity(config, tokenizer, rows, model)
    path = Path(
        config["reference_cache"] or (Path(config["output"]) / "reference.json")
    )
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))
        if cache["fingerprint"] != fingerprint:
            raise ValueError(
                "Reference cache fingerprint mismatch; choose a new cache path"
            )
    else:
        cache = {"fingerprint": fingerprint, "metadata": metadata, "scores": {}}

    def flush():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cache), encoding="utf-8")
        temporary.replace(path)

    was_training = model.training
    model.eval()
    started = time.monotonic()
    scored = 0
    try:
        for row in rows:
            key = row_hash(row)
            if key in cache["scores"]:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Reference scoring deadline; partial cache retained")
            with torch.no_grad():
                scores = score_preference(
                    model, preference_tokens(tokenizer, row, config["max_length"])
                )
            values = {side: float(value) for side, value in scores.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise FloatingPointError("Nonfinite reference scores")
            cache["scores"][key] = values
            scored += 1
            if scored % 32 == 0:
                flush()
    finally:
        flush()
        model.train(was_training)
        log.event(
            "reference_scoring",
            newly_scored=scored,
            cached_examples=len(cache["scores"]),
            seconds=time.monotonic() - started,
            fingerprint=fingerprint,
        )
    return cache


def build_optimizer(model, config, total_steps):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
    else:
        raise ValueError("Unknown optimizer")
    warmup = max(1, int(total_steps * 0.03))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def sft_loss(model, batch):
    device = next(model.parameters()).device
    return -get_response_log_probs(
        model, batch["input_ids"].to(device), batch["labels"].to(device)
    )["log_probs"].mean()


@torch.no_grad()
def validate_sft(model, dataset, limit, microbatch_size, deadline=None):
    previous = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for start in range(0, min(len(dataset), limit), microbatch_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("SFT validation deadline")
            rows = [
                dataset[i]
                for i in range(start, min(start + microbatch_size, len(dataset), limit))
            ]
            batch = {
                key: torch.stack([r[key] for r in rows])
                for key in ("input_ids", "labels")
            }
            total += sft_loss(model, batch).item() * len(rows)
            count += len(rows)
    finally:
        model.train(previous)
    if not count:
        raise ValueError("Validation data produced no packed sequences")
    return total / count


@torch.no_grad()
def validate_dpo(model, tokenizer, rows, cache, config, deadline=None):
    previous = model.training
    model.eval()
    total, correct = 0.0, 0
    try:
        for row in rows:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("DPO validation deadline")
            scores = score_preference(
                model, preference_tokens(tokenizer, row, config["max_length"])
            )
            loss, margin = cached_dpo_loss(
                scores, cache["scores"][row_hash(row)], config["beta"]
            )
            total += float(loss)
            correct += float(margin > 0)
    finally:
        model.train(previous)
    return {
        "validation_loss": total / len(rows),
        "implicit_reward_accuracy": correct / len(rows),
    }


def train_supplement(config):
    seed_all(config["seed"])
    log = RunLog(config["output"], config, config["wandb"])
    deadline = (
        time.monotonic() + config["max_seconds"] if config["max_seconds"] else None
    )
    resume = config.get("resume")
    # Reference caching always starts from the ORIGINAL SFT checkpoint, never
    # from the resumed DPO policy. Only one model is resident at any instant.
    adapter = config.get("adapter")
    if (
        config["kind"] == "sft"
        and resume
        and (Path(resume) / "adapter_config.json").exists()
    ):
        adapter = resume
    full_model_path = config["model"]
    if config["kind"] == "sft" and resume and not adapter:
        full_model_path = resume
    model, tokenizer = load_model(
        full_model_path,
        config["device"],
        config["qlora"],
        adapter,
        True,
        config["gradient_checkpointing"],
        config["attention"],
    )
    cache, optimizer, scheduler = None, None, None
    step, cursor, epoch, best = 0, 0, 0, None
    training = validation = None
    try:
        if config["kind"] == "sft":
            rows = read_jsonl(config["train_data"])
            if config["val_data"]:
                validation_rows = sample_rows(
                    read_jsonl(config["val_data"]), config["n_val"], config["data_seed"]
                )
                val_prompts = {r["prompt"] for r in validation_rows}
                rows = [r for r in rows if r["prompt"] not in val_prompts]
                training_rows = sample_rows(
                    rows, config["n_train"], config["data_seed"]
                )
            else:
                remaining, validation_rows = split_rows(
                    rows, config["n_val"], config["data_seed"]
                )
                val_prompts = {r["prompt"] for r in validation_rows}
                training_rows = sample_rows(
                    [r for r in remaining if r["prompt"] not in val_prompts],
                    config["n_train"],
                    config["data_seed"],
                )
            write_jsonl(Path(config["output"]) / "train_selection.jsonl", training_rows)
            write_jsonl(
                Path(config["output"]) / "validation_selection.jsonl", validation_rows
            )
            training = PackedSFTDataset(
                tokenizer,
                None,
                config["max_length"],
                True,
                rows=training_rows,
                seed=config["seed"],
            )
            validation = PackedSFTDataset(
                tokenizer, None, config["max_length"], False, rows=validation_rows
            )
        else:
            if not config.get("adapter") and config["qlora"]:
                raise ValueError(
                    "QLoRA DPO requires --adapter pointing to your SFT checkpoint"
                )
            training, validation = load_hh(
                config["train_data"],
                config["per_source"],
                config["n_val"],
                config["data_seed"],
            )
            write_jsonl(Path(config["output"]) / "train_selection.jsonl", training)
            write_jsonl(
                Path(config["output"]) / "validation_selection.jsonl", validation
            )
            cache = reference_cache(
                model, tokenizer, training + validation, config, log, deadline
            )
            if resume:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model, tokenizer = load_model(
                    config["model"] if config["qlora"] else resume,
                    config["device"],
                    config["qlora"],
                    resume if config["qlora"] else None,
                    True,
                    config["gradient_checkpointing"],
                    config["attention"],
                )
        if not len(training) or not len(validation):
            raise ValueError("Both train and validation datasets must be nonempty")
        total_steps = (
            math.ceil(len(training) / config["train_batch_size"]) * config["epochs"]
        )
        log.event(
            "dataset",
            train_examples=len(training),
            validation_examples=len(validation),
            total_optimizer_steps=total_steps,
            units="packed_sequences" if config["kind"] == "sft" else "preference_pairs",
        )
        optimizer, scheduler = build_optimizer(model, config, total_steps)
        if resume:
            state = restore_state(resume, optimizer, config)
            step = state["step"]
            epoch, cursor, best = (
                state["extra"][k] for k in ("epoch", "cursor", "best")
            )
            scheduler.load_state_dict(state["extra"]["scheduler"])
            if state["extra"]["data_fingerprint"] != row_hash(
                {
                    "train": training_rows if config["kind"] == "sft" else training,
                    "val": validation_rows if config["kind"] == "sft" else validation,
                }
            ):
                raise ValueError("Dataset changed since checkpoint")
        data_fingerprint = row_hash(
            {
                "train": training_rows if config["kind"] == "sft" else training,
                "val": validation_rows if config["kind"] == "sft" else validation,
            }
        )

        def extra():
            return {
                "epoch": epoch,
                "cursor": cursor,
                "best": best,
                "scheduler": scheduler.state_dict(),
                "data_fingerprint": data_fingerprint,
                "reference_fingerprint": cache["fingerprint"] if cache else None,
            }

        while epoch < config["epochs"] and (
            not config["steps"] or step < config["steps"]
        ):
            order = list(range(len(training)))
            random.Random(config["seed"] + epoch).shuffle(order)
            if cursor >= len(order):
                epoch += 1
                cursor = 0
                continue
            if deadline is not None and time.monotonic() >= deadline:
                break
            indices = order[cursor : cursor + config["train_batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            started = time.monotonic()
            if config["kind"] == "sft":
                for start in range(0, len(indices), config["microbatch_size"]):
                    selected = [
                        training[i]
                        for i in indices[start : start + config["microbatch_size"]]
                    ]
                    batch = {
                        key: torch.stack([r[key] for r in selected])
                        for key in ("input_ids", "labels")
                    }
                    loss = sft_loss(model, batch) * (len(selected) / len(indices))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite SFT loss")
                    loss.backward()
                    loss_sum += float(loss.detach())
            else:
                # One preference pair at a time is intentional; accumulate to
                # the effective batch including an incomplete final batch.
                for index in indices:
                    row = training[index]
                    scores = score_preference(
                        model, preference_tokens(tokenizer, row, config["max_length"])
                    )
                    loss, _ = cached_dpo_loss(
                        scores, cache["scores"][row_hash(row)], config["beta"]
                    )
                    loss = loss / len(indices)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite DPO loss")
                    loss.backward()
                    loss_sum += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            cursor += len(indices)
            step += 1
            log.event(
                "train",
                step=step,
                epoch=epoch,
                cursor=cursor,
                loss=loss_sum,
                gradient_norm=float(norm),
                learning_rate=optimizer.param_groups[0]["lr"],
                training_seconds=time.monotonic() - started,
                batch_examples=len(indices),
            )
            if step % config["checkpoint_every"] == 0:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    Path(config["output"]) / f"checkpoint-{step:05d}",
                    step,
                    config,
                    extra(),
                )
            will_finish = (
                cursor >= len(training) and epoch + 1 >= config["epochs"]
            ) or (config["steps"] and step >= config["steps"])
            if (step % config["eval_every"] == 0 or will_finish) and (
                deadline is None or time.monotonic() < deadline
            ):
                if config["kind"] == "sft":
                    metrics = {
                        "validation_loss": validate_sft(
                            model,
                            validation,
                            config["validation_sequences"],
                            config["microbatch_size"],
                            deadline,
                        )
                    }
                    value = -metrics["validation_loss"]
                else:
                    metrics = validate_dpo(
                        model, tokenizer, validation, cache, config, deadline
                    )
                    value = metrics["implicit_reward_accuracy"]
                log.event("validation", step=step, **metrics)
                if best is None or value > best:
                    best = value
                    save_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        Path(config["output"]) / "best",
                        step,
                        config,
                        extra(),
                    )
        completed = (
            epoch >= config["epochs"]
            or cursor >= len(training)
            and epoch + 1 >= config["epochs"]
        )
        log.event("finished" if completed else "stopped", step=step)
        save_checkpoint(
            model,
            tokenizer,
            optimizer,
            Path(config["output"]) / "last",
            step,
            config,
            extra(),
        )
    except TimeoutError as exc:
        log.event("time_limit", step=step, error=str(exc))
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                Path(config["output"]) / "last",
                step,
                config,
                extra(),
            )
    except Exception as exc:
        log.event("failed", step=step, error=repr(exc))
        # Do not overwrite the last valid checkpoint with a partially applied
        # or nonfinite update. Periodic committed checkpoints remain available.
        raise
    finally:
        log.close()
