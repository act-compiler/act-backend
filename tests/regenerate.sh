#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

./docker/exec.sh \
  "pip install -e . -q && \
  pip install -r tests/requirements.txt -q && \
  python tests/regenerate.py examples/QKV && \
  python tests/regenerate.py examples/QKV_new"
