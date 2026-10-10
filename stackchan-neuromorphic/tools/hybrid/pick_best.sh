#!/usr/bin/env bash
# Several DAgger trainings (different rollout seeds) -> the behaviour audit of each -> keep the one with the fewest failed
# rules (training noise: the same teacher, learned a little differently each time).
#   tools/hybrid/pick_best.sh FIRST LAST     (candidates data/candidates/cFIRST..cLAST; existing ones count too)
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
for c in $(seq "$1" "$2"); do
  cp data/policy_net_imitation.pt data/policy_net.pt
  sed -i "s/--first-seed 5000/--first-seed $((5000 + 100000 * c))/; s/--first-seed \$((7000 + 1000 \* i))/--first-seed \$((7000 + 100000 * $c + 1000 * i))/" dagger.sh
  ./dagger.sh 2 2>&1 | grep -E "episodes|epoch 48|Error" || true
  git checkout -q dagger.sh 2>/dev/null || sed -i "s/--first-seed $((5000 + 100000 * c))/--first-seed 5000/; s/--first-seed \$((7000 + 100000 \* $c + 1000 \* i))/--first-seed \$((7000 + 1000 * i))/" dagger.sh
  ./py audit.py > /dev/null 2>&1 || true
  mkdir -p "data/candidates/c$c"
  cp data/policy_net.pt ../../main/hybrid/policy_model_data.h "data/candidates/c$c/"
  cp ../../simulator/out/audit/report.txt "data/candidates/c$c/audit.txt"
  # the behaviour regression too: a candidate that breaks an established behaviour loses against any audit finding
  ../../simulator/build.sh >/dev/null
  ../../simulator/behavior_test.sh > "data/candidates/c$c/behavior.txt" 2>&1 || true
  echo "candidate c$c: $(grep -E '^[0-9]+ failed' data/candidates/c$c/audit.txt | tail -1), behaviour test fails $(grep -c '  FAIL' data/candidates/c$c/behavior.txt)"
done
score() {  # behaviour-test failures x 100 + audit failures
  local a b
  a=$(grep -oE '^[0-9]+ failed' "$1/audit.txt" | tail -1 | cut -d' ' -f1)
  b=$( [[ -f "$1/behavior.txt" ]] && grep -c '  FAIL' "$1/behavior.txt" || echo 100)
  echo $((b * 100 + a))
}
best=$(for d in data/candidates/c*; do echo "$(score "$d") $d"; done | sort -n | head -1 | cut -d' ' -f2)
echo "best: $best ($(tail -1 $best/audit.txt))"
cp "$best/policy_net.pt" data/policy_net.pt
cp "$best/policy_model_data.h" ../../main/hybrid/policy_model_data.h
../../simulator/build.sh >/dev/null
