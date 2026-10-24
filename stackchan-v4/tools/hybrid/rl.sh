#!/usr/bin/env bash
# Reinforcement stage of the policy (train_rl.py): explore rollouts in the twin -> PPO update -> export, repeated.
# Every EVAL iterations the deterministic policy is scored on 15 quiet and 15 lively minutes (reward.py) and the best
# one is kept (data/policy_net_rl_best.pt + main/hybrid/policy_model_data.h copy in data/policy_model_data_rl_best.h).
#   tools/hybrid/rl.sh [iterations] [episodes] [first] [eval-every]
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
iters="${1:-40}"
eps="${2:-64}"
first="${3:-0}"
every="${4:-4}"
mkdir -p data/rl
best=-1e9
[[ -f data/rl/best_score ]] && best=$(cat data/rl/best_score)
evaluate() {
  ../../simulator/build.sh >/dev/null
  local tot=0 line=""
  for sc in idle_endurance play_session; do
    ../../simulator/build/stackchan_sim "../../simulator/scenarios/$sc.scn" --minutes 15 --out "data/rl/eval_$sc" \
      --bus-dump "data/rl/eval_$sc.bin" >/dev/null
    local r
    r=$(./py reward.py "data/rl/eval_$sc.bin" | head -1 | grep -oE 'total [-+0-9.]+' | cut -d' ' -f2)
    local mv calls
    mv=$(grep -c "executor: move" "data/rl/eval_$sc/log.txt" || true)
    calls=$(grep -c "executor: voice:" "data/rl/eval_$sc/log.txt" || true)
    tot=$(python3 -c "print($tot + $r)")
    line="$line $sc reward $r moves/15min $mv calls/15min $calls |"
  done
  echo "  EVAL it$1: total $tot |$line"
  if python3 -c "import sys; sys.exit(0 if $tot > $best else 1)"; then
    best=$tot
    echo "$best" > data/rl/best_score
    cp data/policy_net.pt data/policy_net_rl_best.pt
    cp ../../main/hybrid/policy_model_data.h data/policy_model_data_rl_best.h
    echo "  -> best so far"
  fi
}
[[ "$first" == 0 ]] && evaluate start
for i in $(seq "$first" $((first + iters - 1))); do
  rm -rf "data/rl/it$i"
  ./py rollout.py explore "$eps" "data/rl/it$i" --minutes 8 --quiet 0.5 --first-seed $((20000 + 1000 * i)) | tail -1
  ./py train_rl.py "data/rl/it$i" | grep -E "mean reward|epoch 10"
  rm -rf "data/rl/it$i"   # several GB per iteration
  if (( (i + 1) % every == 0 )); then evaluate "$i"; fi
done
