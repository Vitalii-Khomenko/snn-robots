#!/usr/bin/env bash
# Smoke test of the simulator: builds, runs a short scenario twice and checks that the run is
# sane and bit-for-bit reproducible.
set -Eeuo pipefail
export SIM_IMU_V3=1  # the v3 brain was trained with the old IMU mounting (world.cpp)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$here/build.sh" >/dev/null
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
run() { "$here/build/stackchan_sim" "$here/scenarios/play_session.scn" --minutes 3 --out "$1" >/dev/null; }
run "$tmp/a"
run "$tmp/b"
fail() { echo "FAIL: $*"; exit 1; }
grep -q "runtime: RobotRuntime ready" "$tmp/a/log.txt" || fail "runtime did not start"
grep -q "vision NN trained, auditory NN trained" "$tmp/a/log.txt" || fail "a perception network is missing"
[[ $(grep -c "runtime: stats" "$tmp/a/log.txt") -ge 30 ]] || fail "too few stats lines"
! grep -q "^E (" "$tmp/a/log.txt" || fail "firmware logged an error: $(grep -m1 '^E (' "$tmp/a/log.txt")"
grep -q "executor: voice: " "$tmp/a/log.txt" || fail "never babbled"
grep -q "snn_resets=0 policy_faults=0" "$tmp/a/log.txt" || fail "spiking core reset or policy fault"
travel=$(grep -o "travel=[0-9]*deg" "$tmp/a/log.txt" | tail -1 | tr -dc 0-9)
[[ $travel -gt 100 ]] || fail "head barely moved ($travel deg)"
cmp -s "$tmp/a/trace.csv" "$tmp/b/trace.csv" || fail "two identical runs differ (not deterministic)"
cmp -s "$tmp/a/log.txt" "$tmp/b/log.txt" || fail "logs of identical runs differ"
echo "PASS: hybrid brain runs in the twin, head travel ${travel} deg in 3 min, babbling seen, reproducible"
