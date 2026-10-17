#!/usr/bin/env bash
# Flash only. No monitor is ever started.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
repo_root="$(cd .. && pwd)"
port="${SERIAL_DEVICE:-${1:-/dev/ttyACM0}}"
baud="${FLASH_BAUD:-460800}"
source tools/serial_guard.sh
flash_container="stackchan-life-flash"
cleanup() { docker rm -f "$flash_container" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker run --rm --init --name "$flash_container" \
  -v "$repo_root/stackchan-life:/project" \
  -v "$repo_root/.tooling:/cache" \
  -e IDF_TOOLS_PATH=/cache/espressif -e TZ=Europe/Berlin \
  --user "$(id -u):$(id -g)" --group-add "$(stat -c %g "$port")" \
  --device "$port:$port" \
  --dns "${FIRMWARE_DNS_SERVER:-192.168.8.1}" -w /project \
  stackchan-firmware-builder:local bash -ec '
    source /cache/esp-idf-v5.5.4/export.sh
    exec idf.py -p "$1" -b "$2" flash
  ' bash "$port" "$baud" &
flash_client=$!
wait "$flash_client"
