"""Required reasoning RL: single-model default, optional course vLLM backend."""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from .core import (
    compute_group_normalized_rewards,
    compute_rollout_rewards,
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)
from .data import (
    gsm8k_answer,
    prompt_text,
    read_jsonl,
    row_hash,
    sample_rows,
    write_jsonl,
)
from .evaluation import evaluate_rl, math_reward
from .runtime import (
    RunLog,
    generate,
    load_model,
    restore_state,
    save_checkpoint,
    seed_all,
)

VARIANTS = {
    "grpo": ("mean", "std", "sequence"),
    "grpo_constant": ("mean", "std", "constant"),
    "dr_grpo": ("mean", "none", "constant"),
    "rft": ("none", "none", "constant"),
    "maxrl": ("mean", "mean", "constant"),
    "loo": ("loo", "none", "constant"),
}


@torch.no_grad()
def old_scores(model, batch, microbatch_size):
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, batch["input_ids"].shape[0], microbatch_size):
        outputs.append(
            get_response_log_probs(
                model,
                batch["input_ids"][start : start + microbatch_size].to(device),
                batch["labels"][start : start + microbatch_size].to(device),
            )["log_probs"].cpu()
        )
    return torch.cat(outputs)


def train_rl(config):
    seed_all(config["seed"])
    if config["rollout_batch_size"] % config["group_size"]:
        raise ValueError("Rollout batch must contain complete groups")
    if config["rollout_batch_size"] % config["train_batch_size"]:
        raise ValueError("Train batches must partition a rollout batch")
    if config["train_batch_size"] % config["microbatch_size"]:
        raise ValueError("Train batch must be divisible by microbatch size")
    resume = config.get("resume")
    model, tokenizer = load_model(
        resume or config["model"],
        config["device"],
        train=True,
        gradient_checkpointing=config["gradient_checkpointing"],
        attention=config["attention"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    state = restore_state(resume, optimizer, config) if resume else {"step": 0}
    log = RunLog(config["output"], config, config["wandb"])
    training = sample_rows(
        read_jsonl(config["train_data"]), config["n_train"], config["data_seed"]
    )
    validation = sample_rows(
        read_jsonl(config["val_data"]), config["n_val"], config["data_seed"]
    )
    if not training or not validation:
        raise ValueError("Training and validation data must be nonempty")
    if {r["question"] for r in training} & {r["question"] for r in validation}:
        raise ValueError("Training/validation question overlap")
    data_fingerprint = row_hash({"train": training, "validation": validation})
    if resume and state.get("extra", {}).get("data_fingerprint") != data_fingerprint:
        raise ValueError("Dataset contents changed since checkpoint")
    write_jsonl(
        Path(config["output"]) / "split_manifest.jsonl",
        [
            {"split": split, "hash": row_hash(row)}
            for split, rows in (("train", training), ("val", validation))
            for row in rows
        ],
    )
    baseline, normalizer, aggregation = VARIANTS[config["variant"]]
    template = prompt_text(config["prompt"])
    reward_fn = math_reward(config["prompt"])
    deadline = (
        time.monotonic() + config["max_seconds"] if config["max_seconds"] else None
    )
    server = None
    if config["backend"] == "vllm":
        from .vllm_utils import VLLMServer

        if config["inference_gpu"] == int(config["device"].split(":")[-1]):
            raise ValueError("Course vLLM backend requires a distinct inference GPU")
        server = VLLMServer(
            config["model"], gpu=config["inference_gpu"], seed=config["seed"]
        )
        server.start()
        server.init_weight_sync(config["device"])
    step = state["step"]

    def completions(prompts):
        if server is None:
            return generate(
                model,
                tokenizer,
                prompts,
                max_tokens=config["max_tokens"],
                temperature=1.0,
                stop=None if config["prompt"] == "question_only" else "</answer>",
                batch_size=config["generation_batch_size"],
                deadline=deadline,
            )
        options = {"temperature": 1.0, "top_p": 1.0, "max_tokens": config["max_tokens"]}
        if config["prompt"] != "question_only":
            options.update(stop=["</answer>"], include_stop_str_in_output=True)
        return [r.text for r in server.generate_completions(prompts, options)]

    try:
        while step < config["steps"]:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if server is not None:
                server.sync_policy_weights(model)
            rows = random.choices(
                training, k=config["rollout_batch_size"] // config["group_size"]
            )
            prompts = [
                template.format(question=r["question"])
                for r in rows
                for _ in range(config["group_size"])
            ]
            answers = [
                gsm8k_answer(r) for r in rows for _ in range(config["group_size"])
            ]
            started = time.monotonic()
            responses = completions(prompts)
            sampling_seconds = time.monotonic() - started
            rewards, reward_stats = compute_rollout_rewards(
                reward_fn, responses, answers
            )
            advantages, _ = compute_group_normalized_rewards(
                rewards, config["group_size"], baseline, advantage_normalizer=normalizer
            )
            batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
            old = (
                old_scores(model, batch, config["microbatch_size"])
                if config["importance"] != "none"
                else None
            )
            stats = []
            training_started = time.monotonic()
            # Do not reshuffle groups or recompute advantages inside minibatches.
            for start in range(0, len(responses), config["train_batch_size"]):
                end = start + config["train_batch_size"]
                local = tokenize_prompt_and_output(
                    prompts[start:end], responses[start:end], tokenizer
                )
                old_local = (
                    None
                    if old is None
                    else old[start:end, : local["input_ids"].shape[1]]
                )
                _, info = grpo_train_step(
                    model,
                    tokenizer,
                    optimizer,
                    config["train_batch_size"] // config["microbatch_size"],
                    1.0,
                    reward_fn,
                    prompts[start:end],
                    responses[start:end],
                    answers[start:end],
                    config["group_size"],
                    baseline=baseline,
                    advantage_normalizer=normalizer,
                    importance_reweighting_method=config["importance"],
                    old_log_probs=old_local,
                    cliprange=config["cliprange"],
                    loss_normalization=aggregation,
                    normalization_constant=config["train_batch_size"]
                    * config["max_tokens"],
                    precomputed_advantages=advantages[start:end],
                    microbatch_size=config["microbatch_size"],
                )
                stats.append(info)
            step += 1
            training_seconds = time.monotonic() - training_started
            mean_stats = {k: sum(r[k] for r in stats) / len(stats) for k in stats[0]}
            generated_tokens = sum(
                len(tokenizer.encode(r, add_special_tokens=False)) for r in responses
            )
            log.event(
                "train",
                step=step,
                optimizer_steps=step * len(stats),
                **mean_stats,
                rollout_reward=reward_stats["reward"],
                rollout_format_reward=reward_stats["format_reward"],
                sampling_seconds=sampling_seconds,
                training_seconds=training_seconds,
                generated_tokens=generated_tokens,
                generated_tokens_per_second=generated_tokens
                / max(sampling_seconds, 1e-9),
            )
            if step == 1 or step % config["rollouts_every"] == 0:
                write_jsonl(
                    Path(config["output"]) / f"rollouts-{step:05d}.jsonl",
                    [
                        {
                            "prompt": p,
                            "output": y,
                            "answer": a,
                            "reward": float(r),
                            "advantage": float(v),
                        }
                        for p, y, a, r, v in zip(
                            prompts, responses, answers, rewards, advantages
                        )
                    ],
                )
            # Commit at completed rollout boundaries before expensive evaluation.
            if step % config["checkpoint_every"] == 0:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    Path(config["output"]) / f"checkpoint-{step:05d}",
                    step,
                    config,
                    {"data_fingerprint": data_fingerprint},
                )
            if step % config["eval_every"] == 0 or step == config["steps"]:
                if deadline is None or time.monotonic() < deadline:
                    if server is not None:
                        server.sync_policy_weights(model)
                    metrics, examples = evaluate_rl(
                        model,
                        tokenizer,
                        validation,
                        config["prompt"],
                        config["max_tokens"],
                        config["generation_batch_size"],
                        deadline,
                        completions if server else None,
                    )
                    log.event("validation", step=step, **metrics)
                    write_jsonl(
                        Path(config["output"]) / f"validation-{step:05d}.jsonl",
                        examples,
                    )
        log.event("finished" if step >= config["steps"] else "time_limit", step=step)
    except TimeoutError:
        log.event(
            "time_limit",
            step=step,
            note="Incomplete inference discarded; completed updates retained",
        )
    except Exception as exc:
        log.event("failed", step=step, error=repr(exc))
        # Never commit a policy after a failed, partially applied rollout.
        step = -1
        raise
    finally:
        if step >= 0:
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                Path(config["output"]) / "last",
                step,
                config,
                {"data_fingerprint": data_fingerprint},
            )
        if server is not None:
            server.stop()
        log.close()
