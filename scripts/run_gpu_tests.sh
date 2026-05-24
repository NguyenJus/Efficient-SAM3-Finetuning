#!/usr/bin/env bash
# Canonical pytest invocation for any GPU environment.
# Used by notebooks/colab_gpu_tests.ipynb and runnable directly on any
# compatible machine with bitsandbytes installed.
#
# Usage:
#   scripts/run_gpu_tests.sh [local|t4|xl]
#
# Hardware tiers (see docs/testing/gpu-test-policy.md):
#   local — fits the GTX 1080 (<=~7 GB, CC 6.0+, NF4 + float16). Dev box via
#           `uv sync --extra gpu-pascal`. Marker: gpu_local.
#   t4    — needs >8 GB and <=16 GB, or bf16-representative numerics. Colab T4.
#           Marker: gpu_t4.
#   xl    — beyond a T4 (>16 GB / larger arch). Cloud auto-provision (#124).
#           Marker: gpu_xl. Likely near-empty initially.
#
# (Test counts per tier are documented in gpu-test-policy.md, not hardcoded here.)
#
# Stateful test-skipping convention (--deselect):
#   When iterating on GPU tests, Claude (or any operator) appends
#   `--deselect <nodeid>` flags to the pytest invocation below as
#   individual tests are confirmed passing on real GPU hardware. This lets
#   the GPU runner skip already-green tests on subsequent runs without
#   editing the test files.
#
#   The mandatory FINAL ALL-GREEN PASS strips every `--deselect` flag and
#   re-runs the full suite to prove it is green end-to-end on a real GPU.
#   No PR may merge with `--deselect` flags left in this script; the CI job
#   `gpu-deselect-check` in `.github/workflows/ci.yml` greps for them and
#   fails the PR if any remain.
set -euo pipefail
TIER="${1:-local}"

case "$TIER" in
  local) MARKER_EXPR="gpu_local" ;;
  t4)    MARKER_EXPR="gpu_t4" ;;
  xl)    MARKER_EXPR="gpu_xl" ;;
  *) echo "usage: $0 [local|t4|xl]" >&2; exit 2 ;;
esac

PATHS="tests/gpu/ tests/integration/ tests/predict/"

# Use `python -m pytest` (not bare `pytest`) so the test runner picks the
# same interpreter that `pip install -e .` populated. Bare `pytest` on
# PATH can resolve to a different Python (common in Colab) and trigger
# `ModuleNotFoundError: No module named 'custom_sam_peft'`.
# PATHS is a controlled space-separated list of paths; intentional word split.
# shellcheck disable=SC2086
"${PYTHON:-python}" -m pytest -v --tb=short \
  -m "$MARKER_EXPR" --no-cov $PATHS
