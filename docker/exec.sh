#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
source act.env

# Pick the cuda13 image and pass the GPU on a GPU host
VARIANT=""
GPU_FLAGS=()
if nvidia-smi >/dev/null 2>&1 &&
  docker info -f '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  VARIANT="-cuda13"
  GPU_FLAGS=(--gpus all)
fi

ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  IMAGE_NAME="devanshdvj/act-env:${ACT_ENV_TAG}${VARIANT}-amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  IMAGE_NAME="devanshdvj/act-env:${ACT_ENV_TAG}${VARIANT}-arm64"
else
  echo "Error: Unsupported architecture: $ARCH"
  exit 1
fi

HOST_MOUNT="$(pwd)/.."

# The container writes as root: restore the caller's ownership at exit
restore_ownership() {
  local status=$?
  ./chown.sh >/dev/null 2>&1 || echo "warning: could not restore the ownership of the files this run wrote" >&2
  exit "$status"
}
trap restore_ownership EXIT

docker run --rm --entrypoint bash \
  ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
  -v "${HOST_MOUNT}:/workspace:rw" \
  -w /workspace \
  "${IMAGE_NAME}" \
  -ilc "$*"
