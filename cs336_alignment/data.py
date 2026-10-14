"""Course dataset loaders, deterministic subsets, and packed SFT documents."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rows, limit, seed=0):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in indices[:limit]] if limit else [rows[i] for i in indices]


def row_hash(row):
    return hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def split_rows(rows, n_val, seed=0):
    # Dedup before splitting; identical examples must not occur in both sets.
    unique = {row_hash(row): row for row in rows}
    ordered = sample_rows(list(unique.values()), 0, seed)
    if len(ordered) <= n_val:
        raise ValueError("Not enough unique examples for disjoint train/validation")
    return ordered[n_val:], ordered[:n_val]


def prompt_text(name, safety=False):
    return (
        ROOT
        / "cs336_alignment"
        / ("prompts_safety" if safety else "prompts")
        / f"{name}.prompt"
    ).read_text(encoding="utf-8")


def sft_text(prompt, response):
    return prompt_text("alpaca_sft", True).format(instruction=prompt, response=response)


class PackedSFTDataset(Dataset):
    def __init__(
        self, tokenizer, dataset_path, seq_length, shuffle, *, rows=None, seed=None
    ):
        if seq_length < 1 or tokenizer.eos_token is None:
            raise ValueError("Positive sequence length and EOS are required")
        documents = list(read_jsonl(dataset_path) if rows is None else rows)
        if shuffle:
            (random if seed is None else random.Random(seed)).shuffle(documents)
        tokens = []
        for row in documents:
            # Default tokenizer special tokens (including Llama BOS) are part
            # of the supplement's packed fixture. Main RL deliberately differs.
            tokens.extend(
                tokenizer.encode(
                    sft_text(row["prompt"], row["response"]) + tokenizer.eos_token
                )
            )
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return max(0, (len(self.tokens) - 1) // self.seq_length)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.seq_length
        return {
            "input_ids": self.tokens[start : start + self.seq_length],
            "labels": self.tokens[start + 1 : start + self.seq_length + 1],
        }


def iterate_batches(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def parse_mmlu_response(mmlu_example, model_output):
    del mmlu_example
    match = re.search(
        r"\b(?:the\s+)?correct\s+answer\s+is\s*[:\-]?\s*\(?([ABCD])\b",
        model_output,
        re.I,
    )
    if match:
        return match.group(1).upper()
    match = re.fullmatch(r"\s*\(?([ABCD])\)?[.\s]*", model_output, re.I)
    return match.group(1).upper() if match else None


def parse_gsm8k_response(model_output):
    numbers = re.findall(
        r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+", model_output
    )
    return numbers[-1].replace(",", "") if numbers else None


def gsm8k_answer(row):
    return row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")


def load_mmlu(directory, limit=256, seed=0):
    groups = []
    directory = Path(directory)
    if (directory / "test").exists():
        directory = directory / "test"
    for path in sorted(directory.glob("*.csv")):
        subject = path.stem.removesuffix("_test").replace("_", " ")
        with path.open(encoding="utf-8") as handle:
            rows = [
                {
                    "subject": subject,
                    "question": r[0],
                    "options": r[1:5],
                    "answer": r[5],
                }
                for r in csv.reader(handle)
                if len(r) == 6
            ]
        groups.append(sample_rows(rows, 0, seed))
    # Round-robin subjects prevents a small subset from containing only the
    # first alphabetically named subject. Same subset for every checkpoint.
    result = []
    while any(groups) and (not limit or len(result) < limit):
        for group in groups:
            if group and (not limit or len(result) < limit):
                result.append(group.pop())
    return result


def load_hh(directory, per_source=512, n_val=200, seed=0):
    sources = (
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled",
    )
    pools = {}
    seen = set()
    for source in sources:
        accepted = []
        for row in read_jsonl(Path(directory) / f"{source}.jsonl.gz"):
            chosen, rejected = row["chosen"], row["rejected"]
            if chosen.count("Human:") != 1 or rejected.count("Human:") != 1:
                continue
            if "\n\nAssistant:" not in chosen or "\n\nAssistant:" not in rejected:
                continue
            pc, yc = chosen.split("\n\nAssistant:", 1)
            pr, yr = rejected.split("\n\nAssistant:", 1)
            if pc != pr:
                continue
            item = {
                "prompt": pc.removeprefix("\n\nHuman:").removeprefix("Human:").strip(),
                "chosen": yc.strip(),
                "rejected": yr.strip(),
                "source": source,
            }
            identity = row_hash({k: item[k] for k in ("prompt", "chosen", "rejected")})
            if identity not in seen and item["chosen"] and item["rejected"]:
                seen.add(identity)
                accepted.append(item)
        pools[source] = sample_rows(accepted, 0, seed)
    validation = []
    for index, source in enumerate(sources):
        count = n_val // len(sources) + (index < n_val % len(sources))
        if len(pools[source]) <= count:
            raise ValueError(f"Not enough single-turn HH examples: {source}")
        validation.extend(pools[source][:count])
        pools[source] = pools[source][count:]
    training = [
        row
        for source in sources
        for row in (pools[source][:per_source] if per_source else pools[source])
    ]
    # Keep a prompt and all of its alternatives confined to one split.
    val_prompts = {r["prompt"] for r in validation}
    training = [r for r in training if r["prompt"] not in val_prompts]
    return sample_rows(training, 0, seed), validation
