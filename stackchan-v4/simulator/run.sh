#!/usr/bin/env bash
# Build (if needed) and run a scenario:  simulator/run.sh simulator/scenarios/X.scn [options]
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/build.sh" >/dev/null
exec "$here/build/stackchan_sim" "$@"
