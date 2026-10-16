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
# One name per call: the trap removes this run's container
CONTAINER_NAME="act-backend-exec-$$-${RANDOM}"

# At exit, remove the container and restore the caller's ownership
restore_ownership() {
  local status=$?
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  ./chown.sh >/dev/null 2>&1 || echo "warning: could not restore the ownership of the files this run wrote" >&2
  exit "$status"
}
trap restore_ownership EXIT
# A signal ends the run at once, with its status
trap 'exit 130' INT
trap 'exit 143' TERM

# Created in the foreground: a signal waits for the create
docker create --rm --name "${CONTAINER_NAME}" --entrypoint bash \
  ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
  -v "${HOST_MOUNT}:/workspace:rw" \
  -w /workspace \
  "${IMAGE_NAME}" \
  -ilc "$*" >/dev/null
# Backgrounded, so a signal reaches the trap mid-run
docker start -a "${CONTAINER_NAME}" &
wait $!
