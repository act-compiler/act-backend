#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Quote each argument for the container's shell
ARGS=""
if [ $# -gt 0 ]; then
  ARGS=$(printf '%q ' "$@")
fi

# The wheel, built anew: a stale build/ keeps dropped files
./docker/exec.sh \
  "rm -rf build && pip install . -q && \
  pip install -r tests/requirements.txt -q && \
  pytest tests/ ${ARGS}"
