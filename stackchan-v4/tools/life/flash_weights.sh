#!/usr/bin/env bash
# Writes a weights blob (tools/life/pack_weights.py) into a weights partition over the cable.
#   tools/life/flash_weights.sh [a|b] [blob]        default: slot a, build/weights.bin
# The robot reads the live slot at boot (NVS weights/slot, 'a' when unset). Over Wi-Fi: tools/life/ota.py (later).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
repo_root="$(cd .. && pwd)"
slot="${1:-a}"
blob="${2:-build/weights.bin}"
port="${SERIAL_DEVICE:-/dev/ttyACM0}"
source tools/serial_guard.sh
docker run --rm --init \
  -v "$repo_root/stackchan-v4:/project" -v "$repo_root/.tooling:/cache" \
  -e IDF_TOOLS_PATH=/cache/espressif --user "$(id -u):$(id -g)" --group-add "$(stat -c %g "$port")" \
  --device "$port:$port" -w /project stackchan-firmware-builder:local bash -ec '
    source /cache/esp-idf-v5.5.4/export.sh >/dev/null
    python "$IDF_PATH/components/partition_table/parttool.py" --port "$1" write_partition \
      --partition-name "weights_$2" --input "$3"
  ' bash "$port" "$slot" "$blob"
