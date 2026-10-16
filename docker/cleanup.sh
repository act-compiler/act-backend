#!/bin/bash
set -euo pipefail

# Change to script directory
cd "$(dirname "$0")"
source act.env

# Pick the cuda13 image on a GPU host
VARIANT=""
if nvidia-smi >/dev/null 2>&1 &&
  docker info -f '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  VARIANT="-cuda13"
fi
# --cpu or --cuda13 overrides the host's default
case "${1:-}" in
  --cpu)
    VARIANT=""
    shift
    ;;
  --cuda13)
    VARIANT="-cuda13"
    shift
    ;;
esac

usage() {
  echo "Usage: $0 [--cpu|--cuda13] [-h|--help]"
}

# Past the image flag, only --help is read
if [ $# -gt 0 ]; then
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
fi

# Detect architecture
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
  IMAGE_NAME="devanshdvj/act-env:${ACT_ENV_TAG}${VARIANT}-amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  IMAGE_NAME="devanshdvj/act-env:${ACT_ENV_TAG}${VARIANT}-arm64"
else
  echo "Error: Unsupported architecture: $ARCH" >&2
  exit 1
fi

# collect candidate containers (all containers using the image)
LISTED=$(docker ps -a --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}}')
# Not piped: a failed docker ps ends the script
mapfile -t CONTAINERS < <(awk -v img="${IMAGE_NAME}" '$3 == img' <<<"${LISTED}")

NUM_CONTAINERS=${#CONTAINERS[@]}
if [ ${NUM_CONTAINERS} -eq 0 ]; then
  echo "No containers using image '$IMAGE_NAME' found."
  exit 0
fi

echo "Found ${NUM_CONTAINERS} container(s) using image '${IMAGE_NAME}':"
for line in "${CONTAINERS[@]}"; do
  echo "$line"
done
echo
for line in "${CONTAINERS[@]}"; do
  CID=$(echo "$line" | cut -d' ' -f1)
  CNAME=$(echo "$line" | cut -d' ' -f2)
  read -rp "Remove container ${CNAME} (${CID})? [y/N] " RESP
  case "${RESP}" in
    y | Y)
      echo "Removing ${CNAME}..."
      docker rm -f "${CID}"
      ;;
    *)
      echo "Keeping ${CNAME}."
      ;;
  esac
done

echo
echo "Done."
