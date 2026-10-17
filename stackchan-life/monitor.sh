#!/usr/bin/env bash
# Optional foreground UART monitor. build.sh and flash.sh never start it.
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
repo_root="$(cd .. && pwd)"
port="${SERIAL_DEVICE:-${1:-/dev/ttyACM0}}"
monitor_name="stackchan-life-monitor"
source tools/serial_guard.sh

if docker ps -q --filter "name=^/${monitor_name}$" | grep -q .; then
  echo "ERROR: $monitor_name is already running" >&2
  exit 2
fi

cleanup()
{
  docker rm -f "$monitor_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker run --rm \
  --name "$monitor_name" \
  --label stackchan.role=monitor \
  -v "$repo_root/stackchan-life:/project" \
  -v "$repo_root/.tooling:/cache" \
  -e IDF_TOOLS_PATH=/cache/espressif \
  -e TZ=Europe/Berlin \
  --device "$port:$port" \
  -w /project \
  stackchan-firmware-builder:local \
  bash -c '
    source /cache/esp-idf-v5.5.4/export.sh
    exec python /cache/esp-idf-v5.5.4/tools/idf_monitor.py -p "$1" -b 115200 /project/build/stackchan_neuromorphic.elf
  ' bash "$port" &
monitor_client=$!
wait "$monitor_client"
