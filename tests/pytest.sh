#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Quote each argument for the container's shell
ARGS=""
if [ $# -gt 0 ]; then
  ARGS=$(printf '%q ' "$@")
fi

./docker/exec.sh \
  "pip install -e . -q && \
  pip install -r tests/requirements.txt -q && \
  pytest tests/ ${ARGS}"
