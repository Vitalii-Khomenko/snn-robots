#!/usr/bin/env bash
# Behaviour regression on the digital twin (tools/hybrid/behavior_test.py): tracking a visitor,
# startle and orienting, sleeping at night, singing in the dawn window, looking around alone.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$here/build.sh" >/dev/null
cd "$here/../tools/hybrid"
exec ./py behavior_test.py
