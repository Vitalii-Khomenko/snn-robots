#!/usr/bin/env bash
# The morning report and dashboard of yesterday (and of today so far), run by the systemd user timer life-morning.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../hybrid"
yesterday="$(date -d yesterday +%F)"
today="$(date +%F)"
./py ../life/life_report.py "$yesterday" "$today" >/dev/null
for d in "$yesterday" "$today"; do ./py ../life/dashboard.py "$d" || true; done
