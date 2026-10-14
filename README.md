# CS336 Spring 2026 Assignment 5: Alignment

## Implemented single-GPU learning workflow

Both the reasoning RL and optional SFT/DPO code paths are implemented with
handwritten losses and training loops. The default learning profile targets a
single 24–32 GB CUDA GPU, with QLoRA for Llama 8B and sequential HF generation
for OLMo 1B. Numerical tests and hardware validation must be run on Linux;
implementation status does not mean the experiments have been completed.

- [Single-GPU runbook](docs/RUNBOOK.md)
- [Chinese implementation guide](docs/IMPLEMENTATION_GUIDE_CN.md)
- [Complete Chinese post-training tutorial](docs/POST_TRAINING_TUTORIAL_CN.md)
- [Offline interactive learning page](docs/post_training_tutorial.html)
- [Code-first PyTorch pipeline walkthrough (Chinese)](docs/CODE_WALKTHROUGH_CN.md)
- [Offline execution walkthrough](docs/code_walkthrough.html)
- [Requirements and verification status](docs/REQUIREMENTS_STATUS.md)
- [Report template](docs/REPORT_TEMPLATE.md)

Entry point: `python -m cs336_alignment.cli --help`. Run all original and
additional semantic tests with `bash scripts/check_linux.sh`. Automatic judging
is deferred; no paid service or GPU rental is launched by these tools.

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

