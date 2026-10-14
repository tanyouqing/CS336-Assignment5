"""Bounded subprocess benchmarks, serial experiment manifests and real plots."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .data import read_jsonl, write_jsonl
from .runtime import software_versions


def run_bounded(command, seconds, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (directory / "console.log").open("w", encoding="utf-8") as console:
        process = subprocess.Popen(
            command,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
        )
        try:
            code = process.wait(timeout=seconds)
            status = "completed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
            code, status = process.returncode, "timeout"
    return {
        "status": status,
        "exit_code": code,
        "wall_seconds": time.monotonic() - started,
    }


def invoke_config(kind, settings, directory, seconds):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "launch.json"
    settings = {**settings, "output": str(directory.resolve())}
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return run_bounded(
        [
            sys.executable,
            "-m",
            "cs336_alignment.cli",
            kind,
            "--config",
            str(path.resolve()),
        ],
        seconds,
        directory,
    )


def benchmark(config):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark must run on the rented Linux CUDA machine")
    if config["gpu_hours"] <= 0:
        raise ValueError("Benchmark time budget must be positive")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    # Model downloads happen separately so a slow network does not consume the
    # entire compute pilot. No paid API or rental is created by this command.
    from huggingface_hub import snapshot_download

    models = [config["model"]] + ([config["sft_model"]] if config["sft_model"] else [])
    started = time.monotonic()
    for name in models:
        if not Path(name).exists():
            snapshot_download(name, ignore_patterns=["*.msgpack", "*.h5", "*.ot"])
    downloads = time.monotonic() - started
    deadline = time.monotonic() + min(config["gpu_hours"] * 3600, 1800)
    jobs = [
        (
            "rl",
            "rl",
            {
                "model": config["model"],
                "steps": 3,
                "rollout_batch_size": 8,
                "train_batch_size": 8,
                "group_size": 4,
                "max_tokens": 512,
                "n_train": 64,
                "n_val": 16,
                "eval_every": 100,
                "checkpoint_every": 100,
            },
        )
    ]
    jobs.append(
        (
            "rl-offpolicy",
            "rl",
            {
                "model": config["model"],
                "steps": 3,
                "rollout_batch_size": 8,
                "train_batch_size": 2,
                "group_size": 4,
                "max_tokens": 512,
                "importance": "grpo",
                "n_train": 64,
                "n_val": 16,
                "eval_every": 100,
                "checkpoint_every": 100,
            },
        )
    )
    if config["sft_model"] and config["sft_data"]:
        jobs.append(
            (
                "sft",
                "sft",
                {
                    "model": config["sft_model"],
                    "train_data": config["sft_data"],
                    "val_data": config["sft_val_data"],
                    "n_train": 128,
                    "n_val": 8,
                    "max_length": 512,
                    "train_batch_size": 4,
                    "steps": 3,
                    "eval_every": 100,
                    "checkpoint_every": 100,
                },
            )
        )
        jobs.append(
            (
                "dpo",
                "dpo",
                {
                    "model": config["sft_model"],
                    "adapter": str((output / "sft" / "last").resolve()),
                    "train_data": config["hh_data"],
                    "per_source": 8,
                    "n_val": 8,
                    "max_length": 512,
                    "train_batch_size": 2,
                    "steps": 3,
                    "eval_every": 100,
                    "checkpoint_every": 100,
                },
            )
        )
    report = {
        "environment": software_versions(),
        "download_seconds": downloads,
        "compute_limit_seconds": min(config["gpu_hours"] * 3600, 1800),
        "jobs": [],
        "note": "512-token context/generation limits. Cost extrapolation remains approximate; actual sampled lengths and validation dominate variability.",
    }
    for index, (name, kind, settings) in enumerate(jobs):
        remaining = deadline - time.monotonic()
        if remaining <= 15:
            report["jobs"].append({"name": name, "status": "not_run_budget"})
            continue
        if (
            kind == "dpo"
            and not (output / "sft" / "last" / "adapter_config.json").exists()
        ):
            report["jobs"].append(
                {"name": name, "status": "blocked_missing_sft_checkpoint"}
            )
            continue
        allowance = remaining / (len(jobs) - index)
        settings.update(
            device=config["device"],
            attention=config["attention"],
            max_seconds=max(1, allowance - 15),
        )
        result = invoke_config(kind, settings, output / name, max(1, allowance - 10))
        metrics_path = output / name / "metrics.jsonl"
        records = read_jsonl(metrics_path) if metrics_path.exists() else []
        train = [r for r in records if r["event"] == "train"]
        if result["status"] == "completed" and len(train) < settings["steps"]:
            result["status"] = "partial"
        measured = train[1:]  # First update includes warmup, not steady state.
        timing = {}
        for key in (
            "training_seconds",
            "sampling_seconds",
            "generated_tokens_per_second",
            "peak_allocated_gb",
            "peak_reserved_gb",
        ):
            values = [r[key] for r in measured if key in r]
            timing[key] = sum(values) / len(values) if values else None
        cost = None
        if kind == "rl" and measured and timing["sampling_seconds"] is not None:
            # 64/8 rollout scale, plus two 256-example validations over 25
            # rollout steps. No claim of linear scaling accuracy is made.
            cost = (
                25 * 8 * (timing["sampling_seconds"] + timing["training_seconds"])
                + 2 * 32 * timing["sampling_seconds"]
            ) / 3600
        supplement_cost = None
        datasets = [r for r in records if r["event"] == "dataset"]
        if kind == "sft" and measured and datasets:
            estimated_sequences = datasets[-1]["train_examples"] * 16
            supplement_cost = (
                ((estimated_sequences + 31) // 32)
                * timing["training_seconds"]
                * 8
                / 3600
            )
        elif kind == "dpo" and measured:
            references = [r for r in records if r["event"] == "reference_scoring"]
            reference_seconds = 0.0
            if references and references[-1]["newly_scored"]:
                reference_seconds = (
                    references[-1]["seconds"] * 2248 / references[-1]["newly_scored"]
                )
            supplement_cost = (
                128 * timing["training_seconds"] * 8 + reference_seconds
            ) / 3600
        report["jobs"].append(
            {
                "name": name,
                **result,
                "measured_updates": len(measured),
                "approx_25_batch_rl_gpu_hours": cost,
                "approx_full_reduced_supplement_gpu_hours": supplement_cost,
                **timing,
            }
        )
        (output / "benchmark_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )
    (output / "environment-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    (output / "benchmark_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def matrix(config):
    from .cli import COMMON, RL, SUPPLEMENT

    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    original = config["profile"] == "original"
    if config["profile"] not in ("single_gpu", "original"):
        raise ValueError("Profile must be single_gpu or original")
    defaults = COMMON | RL | {"model": config["model"], "device": config["device"]}
    if original:
        defaults.update(
            n_train=6400,
            n_val=1024,
            rollout_batch_size=256,
            train_batch_size=256,
            group_size=8,
            steps=200,
            backend="vllm",
            microbatch_size=8,
        )
    seeds = range(4) if original else (0,)

    def add(name, kind, settings, dependencies=(), priority=2):
        jobs.append(
            {
                "name": name,
                "kind": kind,
                "config": settings,
                "dependencies": list(dependencies),
                "priority": priority,
                "status": "not_run",
                "estimated_gpu_hours": None,
            }
        )

    for seed in seeds:
        add(f"grpo-seed{seed}", "rl", defaults | {"seed": seed}, priority=0)
        if not original:
            add(f"grpo-short-seed{seed}", "rl", defaults | {"seed": seed, "steps": 25})
        for variant in ("grpo_constant", "dr_grpo", "rft", "maxrl", "loo"):
            add(
                f"{variant}-seed{seed}",
                "rl",
                defaults
                | {"seed": seed, "variant": variant, "steps": 200 if original else 25},
            )
        for importance in ("none", "noclip", "grpo", "gspo"):
            add(
                f"offpolicy-{importance}-seed{seed}",
                "rl",
                defaults
                | {
                    "seed": seed,
                    "importance": importance,
                    "train_batch_size": 8 if original else 2,
                    "microbatch_size": 8 if original else 1,
                    "cliprange": 3e-4 if importance == "gspo" else 0.2,
                    "steps": 200 if original else 25,
                },
            )
        for prompt in ("question_only", "r1_zero_three_shot_gsm8k"):
            add(
                f"prompt-{prompt}-seed{seed}",
                "rl",
                defaults
                | {"seed": seed, "prompt": prompt, "steps": 200 if original else 25},
            )
        for lr in (3e-6, 3e-5):
            add(
                f"lr-{lr}-seed{seed}",
                "rl",
                defaults
                | {"seed": seed, "learning_rate": lr, "steps": 200 if original else 25},
            )
    for prompt in ("question_only", "r1_zero", "r1_zero_three_shot_gsm8k"):
        add(
            f"baseline-{prompt}",
            "eval",
            COMMON
            | {
                "model": config["model"],
                "task": "rl",
                "prompt": prompt,
                "data": "data/gsm8k/test.jsonl",
                "limit": 1024 if original else 256,
            },
            priority=0,
        )
    if not original:
        for variant, importance in (
            ("grpo", "none"),
            ("dr_grpo", "none"),
            ("grpo", "gspo"),
        ):
            add(
                f"repeat-{variant}-{importance}-seed1",
                "rl",
                defaults
                | {
                    "seed": 1,
                    "variant": variant,
                    "importance": importance,
                    "train_batch_size": 2 if importance == "gspo" else 64,
                    "cliprange": 3e-4 if importance == "gspo" else 0.2,
                    "steps": 25,
                },
                priority=3,
            )
    if config["sft_model"] and config["sft_data"]:
        supplement = (
            COMMON
            | SUPPLEMENT
            | {"model": config["sft_model"], "device": config["device"]}
        )
        if original:
            supplement.update(
                qlora=False,
                n_train=0,
                per_source=0,
                n_val=200,
                train_batch_size=32,
                learning_rate=2e-5,
            )
        add(
            "sft",
            "sft",
            supplement
            | {"train_data": config["sft_data"], "val_data": config["sft_val_data"]},
            priority=0,
        )
        sft_checkpoint = str((output / "jobs" / "sft" / "last").resolve())
        dpo_settings = supplement | {
            "train_data": config["hh_data"],
            "train_batch_size": 64 if original else 16,
            "learning_rate": 1e-6 if original else 1e-5,
            "weight_decay": 0.0,
            "optimizer": "rmsprop" if original else "adamw",
        }
        dpo_settings.update(
            {"model": sft_checkpoint} if original else {"adapter": sft_checkpoint}
        )
        add("dpo", "dpo", dpo_settings, ("sft",), priority=0)
        for stage in ("base", "sft", "dpo"):
            for task, path, limit in (
                ("mmlu", "data/mmlu", 256),
                ("gsm8k", "data/gsm8k/test.jsonl", 256),
                ("alpaca", "data/alpaca_eval/alpaca_eval_gpt4_turbo.json", 100),
                ("safety", "data/simple_safety_tests/simple_safety_tests.csv", 0),
            ):
                settings = COMMON | {
                    "model": config["sft_model"],
                    "qlora": not original,
                    "task": task,
                    "stage": stage,
                    "data": path,
                    "limit": 0 if original else limit,
                }
                dependency = []
                if stage != "base":
                    checkpoint = str((output / "jobs" / stage / "last").resolve())
                    settings.update(
                        {"model": checkpoint} if original else {"adapter": checkpoint}
                    )
                    dependency = [stage]
                add(f"{stage}-{task}", "eval", settings, dependency, priority=1)
    manifest_path = Path(config.get("manifest") or output / "manifest.json")
    if manifest_path.exists():
        # Never discard execution status by regenerating a manifest in-place.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "profile": config["profile"],
            "jobs": jobs,
            "automatic_judging": "deferred",
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not config["execute"]:
        print(
            f"Wrote/read {len(manifest['jobs'])} jobs at {manifest_path}; no experiments launched"
        )
        return
    if not config["budget_gpu_hours"] or config["budget_gpu_hours"] <= 0:
        raise ValueError("Execution requires an explicit positive --budget-gpu-hours")
    deadline = time.monotonic() + config["budget_gpu_hours"] * 3600
    for job in sorted(manifest["jobs"], key=lambda r: r["priority"]):
        if job["status"] == "completed":
            continue
        if any(
            next(j for j in manifest["jobs"] if j["name"] == dep)["status"]
            != "completed"
            for dep in job["dependencies"]
        ):
            job["status"] = "blocked_dependency"
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 15:
            break
        settings = job["config"] | {"max_seconds": max(1, remaining - 15)}
        directory = output / "jobs" / job["name"]
        if (directory / "last" / "training_state.pt").exists() and job["kind"] in (
            "rl",
            "sft",
            "dpo",
        ):
            settings["resume"] = str((directory / "last").resolve())
        result = invoke_config(job["kind"], settings, directory, remaining - 10)
        # A time-limited runner may exit successfully before reaching its target.
        metrics = (
            read_jsonl(directory / "metrics.jsonl")
            if (directory / "metrics.jsonl").exists()
            else []
        )
        finished = any(r["event"] in ("finished", "evaluation") for r in metrics[-3:])
        job.update(result)
        if result["status"] == "completed" and not finished:
            job["status"] = "partial"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def summarize(config):
    root = Path(config["runs"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(root.rglob("metrics.jsonl")):
        run = str(path.parent.relative_to(root))
        rows.extend({"run": run, **r} for r in read_jsonl(path))
    if not rows:
        (output / "status.json").write_text(
            json.dumps(
                {"status": "no_results", "note": "No synthetic metrics generated"}
            ),
            encoding="utf-8",
        )
        print("No real metrics found")
        return
    # Resuming into the same run can repeat an evaluation at a saved boundary.
    # Keep the most recent observation for that exact event/step pair.
    indexed = {}
    without_step = []
    for row in rows:
        if "step" in row:
            indexed[(row["run"], row["event"], row["step"])] = row
        else:
            without_step.append(row)
    rows = without_step + list(indexed.values())
    fields = sorted({key for row in rows for key in row})
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        "loss",
        "gradient_norm",
        "token_entropy",
        "reward",
        "format_reward",
        "rollout_reward",
        "rollout_format_reward",
        "average_response_length",
        "clip_fraction",
        "validation_loss",
        "implicit_reward_accuracy",
    )
    for metric in metrics:
        events = sorted({r["event"] for r in rows if metric in r and "step" in r})
        for event in events:
            figure, axis = plt.subplots(figsize=(8, 4))
            for run in sorted({r["run"] for r in rows}):
                selected = [
                    r
                    for r in rows
                    if r["run"] == run
                    and r["event"] == event
                    and metric in r
                    and "step" in r
                ]
                if selected:
                    axis.plot(
                        [r["step"] for r in selected],
                        [r[metric] for r in selected],
                        label=run,
                    )
            axis.set(
                xlabel="step",
                ylabel=metric,
                title=f"{event}: {metric} (individual runs)",
            )
            axis.legend(fontsize=6)
            figure.tight_layout()
            figure.savefig(output / f"{event}-{metric}.png", dpi=160)
            plt.close(figure)
    evaluations = [
        r for r in rows if r["event"] in ("evaluation", "validation", "safety_judgment")
    ]
    write_jsonl(output / "evaluations.jsonl", evaluations)
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "real_metrics_only",
                "runs": len({r["run"] for r in rows}),
                "automatic_judging": "pending unless explicitly run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
