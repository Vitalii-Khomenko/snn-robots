#!/usr/bin/env bash
# Build stackchan_v4 against the cached ESP-IDF toolchain in the
# stackchan-firmware-builder:local Docker image (see ../firmware-builder).
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
repo_root="$(cd .. && pwd)"
if [[ ! -f "$repo_root/.tooling/esp-idf-v5.5.4/export.sh" ]]; then
  echo "no ESP-IDF in $repo_root/.tooling: run ./bootstrap.sh once" >&2
  exit 1
fi

docker run --rm \
  -v "$repo_root/stackchan-v4:/project" \
  -v "$repo_root/.tooling:/cache" \
  -e IDF_TOOLS_PATH=/cache/espressif \
  -e TZ=Europe/Berlin \
  --dns "${FIRMWARE_DNS_SERVER:-192.168.8.1}" \
  -w /project \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  stackchan-firmware-builder:local \
  bash -c '
    source /cache/esp-idf-v5.5.4/export.sh
    idf.py "$@"
  ' bash "${@:-build}"
