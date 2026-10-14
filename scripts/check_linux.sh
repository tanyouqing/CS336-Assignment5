#!/usr/bin/env bash
# Run from the assignment5-alignment root. No GPU/model download required.
set -euo pipefail
python -m pytest tests/test_grpo.py tests/test_data.py tests/test_metrics.py tests/test_dpo.py tests/test_invariants.py tests/test_training_smoke.py
