#!/usr/bin/env bash
# Behaviour regression on the digital twin (tools/hybrid/behavior_test.py): tracking a visitor,
# startle and orienting, sleeping at night, singing in the dawn window, looking around alone.
set -Eeuo pipefail
export SIM_IMU_V3=1  # the v3 brain was trained with the old IMU mounting (world.cpp)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$here/build.sh" >/dev/null
cd "$here/../tools/hybrid"
exec ./py behavior_test.py
