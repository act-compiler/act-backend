#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

./docker/exec.sh \
  "pip install -e . -q && \
  pip install -r tests/requirements.txt -q && \
  pytest tests/test_generators.py && \
  pytest tests/test_qkv.py && \
  pytest tests/test_qkv_new.py"
