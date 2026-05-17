#!/usr/bin/env bash
# Canonical pytest invocation for any GPU environment.
# Used by notebooks/colab_gpu_tests.ipynb and runnable directly on any
# Turing+ machine with bitsandbytes installed.
set -euo pipefail

pytest -v --tb=short \
  -m "requires_compatible_gpu and requires_checkpoint" \
  --no-cov \
  tests/integration/
