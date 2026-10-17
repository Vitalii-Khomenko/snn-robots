#!/usr/bin/env bash
# Policy training cycle: teacher rollouts -> train -> learner rollouts (DAgger) x N -> train.
#   tools/hybrid/dagger.sh [iterations]
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
iters="${1:-2}"
rm -rf data/policy
./py rollout.py teacher 150 data/policy/it0 --minutes 8 --first-seed 5000 --quiet 0.25 | tail -1
./py train_policy.py --epochs 40 --qat 8 | grep -E "episodes|epoch 48"
for i in $(seq 1 "$iters"); do
  ../../simulator/build.sh >/dev/null
  ./py rollout.py learner 100 "data/policy/it$i" --minutes 8 --first-seed $((7000 + 1000 * i)) --quiet 0.25 | tail -1
  ./py train_policy.py --epochs 40 --qat 8 | grep -E "episodes|epoch 48"
done
../../simulator/build.sh >/dev/null
