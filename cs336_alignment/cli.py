"""Run with python -m cs336_alignment.cli COMMAND [--config FILE ...]."""

from __future__ import annotations

import argparse
import json

COMMON = {
    "model": "allenai/OLMo-2-0425-1B",
    "device": "cuda:0",
    "seed": 0,
    "data_seed": 0,
    "output": "runs/default",
    "attention": "sdpa",
    "max_seconds": 0,
    "wandb": False,
    "qlora": False,
    "adapter": None,
    "resume": None,
    "gradient_checkpointing": True,
    "generation_batch_size": 1,
}
RL = {
    "train_data": "data/gsm8k/train.jsonl",
    "val_data": "data/gsm8k/test.jsonl",
    "n_train": 1024,
    "n_val": 256,
    "steps": 50,
    "learning_rate": 1e-5,
    "rollout_batch_size": 64,
    "train_batch_size": 64,
    "group_size": 4,
    "microbatch_size": 1,
    "max_tokens": 512,
    "variant": "grpo",
    "importance": "none",
    "cliprange": 0.2,
    "prompt": "r1_zero",
    "eval_every": 10,
    "rollouts_every": 10,
    "checkpoint_every": 25,
    "backend": "hf",
    "inference_gpu": 1,
}
SUPPLEMENT = {
    "model": "meta-llama/Meta-Llama-3.1-8B",
    "qlora": True,
    "train_data": None,
    "val_data": None,
    "n_train": 2048,
    "n_val": 200,
    "per_source": 512,
    "max_length": 512,
    "epochs": 1,
    "steps": 0,
    "microbatch_size": 1,
    "train_batch_size": 32,
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "eval_every": 25,
    "checkpoint_every": 100,
    "validation_sequences": 128,
    "beta": 0.1,
    "reference_cache": None,
}
SUPPLEMENT["optimizer"] = "adamw"
EVAL = {
    "data": "data/gsm8k/test.jsonl",
    "task": "rl",
    "stage": "base",
    "limit": 256,
    "prompt": "r1_zero",
    "max_tokens": 512,
}
TOOLS = {
    "sft_model": None,
    "sft_data": None,
    "sft_val_data": None,
    "hh_data": "data/hh",
    "profile": "single_gpu",
    "gpu_hours": 0.5,
    "runs": "runs",
    "execute": False,
    "budget_gpu_hours": None,
    "manifest": None,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        "command",
        choices=(
            "rl",
            "sft",
            "dpo",
            "eval",
            "judge-safety",
            "benchmark",
            "matrix",
            "summarize",
        ),
    )
    parser.add_argument("--config", help="JSON config; explicit flags override it")
    defaults = COMMON | RL | SUPPLEMENT | EVAL | TOOLS
    # The per-command defaults are resolved AFTER parsing. SUPPRESS prevents
    # argparse defaults from silently replacing a JSON setting.
    for key, value in defaults.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction)
        else:
            datatype = (
                type(value)
                if value is not None
                else (float if key == "budget_gpu_hours" else str)
            )
            parser.add_argument(flag, type=datatype)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    config_path = arguments.pop("config", None)
    specific = (
        RL
        if command == "rl"
        else SUPPLEMENT
        if command in ("sft", "dpo")
        else EVAL
        if command in ("eval", "judge-safety")
        else TOOLS
    )
    config = COMMON | specific
    if command == "dpo":
        config.update(train_batch_size=16, learning_rate=1e-5, weight_decay=0.0)
    if config_path:
        supplied = json.loads(open(config_path, encoding="utf-8").read())
        unknown = set(supplied) - set(defaults) - {"kind"}
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        config.update(supplied)
    unknown = set(arguments) - set(config)
    if unknown:
        parser.error(f"Options do not apply to {command}: {sorted(unknown)}")
    config.update(arguments)
    for key in (
        "microbatch_size",
        "train_batch_size",
        "group_size",
        "generation_batch_size",
        "max_length",
        "eval_every",
        "checkpoint_every",
        "rollouts_every",
        "epochs",
        "max_tokens",
    ):
        if key in config and config[key] <= 0:
            parser.error(f"{key} must be positive")
    for key in ("steps", "max_seconds", "limit", "n_train", "n_val", "per_source"):
        if key in config and config[key] < 0:
            parser.error(f"{key} must be nonnegative")
    if command == "rl":
        from .rl import train_rl, VARIANTS

        if config["variant"] not in VARIANTS or config["importance"] not in (
            "none",
            "noclip",
            "grpo",
            "gspo",
        ):
            parser.error("Invalid variant/importance method")
        if config["prompt"] not in (
            "question_only",
            "r1_zero",
            "r1_zero_three_shot_gsm8k",
        ):
            parser.error("Invalid required-RL prompt")
        train_rl(config)
    elif command in ("sft", "dpo"):
        if not config["train_data"]:
            parser.error("--train-data is required: SFT JSONL(.gz) or HH directory")
        config["kind"] = command
        from .supplement import train_supplement

        train_supplement(config)
    elif command == "eval":
        if config["task"] not in ("rl", "mmlu", "gsm8k", "alpaca", "safety") or config[
            "stage"
        ] not in ("base", "sft", "dpo"):
            parser.error("Invalid evaluation task/stage")
        from .evaluation import evaluate

        evaluate(config)
    elif command == "judge-safety":
        from .evaluation import judge_safety

        judge_safety(config)
    else:
        from .experiments import benchmark, matrix, summarize

        {"benchmark": benchmark, "matrix": matrix, "summarize": summarize}[command](
            config
        )


if __name__ == "__main__":
    main()
