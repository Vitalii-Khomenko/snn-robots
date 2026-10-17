#!/usr/bin/env bash
# Builds the perception-training data generator (simulator/datagen).
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
mkdir -p "$here/build"
cd "$root"
srcs=(simulator/datagen/datagen.cpp simulator/core/vrtos.cpp simulator/core/log.cpp simulator/hal/platform.cpp
      simulator/world/world.cpp simulator/world/scenario.cpp main/hybrid/vision_frontend.cpp
      main/hybrid/audio_frontend.cpp main/motion/motion_engine.cpp)
objs=()
for src in "${srcs[@]}"; do
  obj="$here/build/dg_$(echo "$src" | tr '/' '_' | sed 's/\.cpp$/.o/')"
  objs+=("$obj")
  if [[ ! -f "$obj" || "$src" -nt "$obj" || -n "$(find main/ simulator/ -name '*.h' -newer "$obj" -print -quit)" ]]; then
    g++ -std=c++17 -O2 -g -Wall -Wextra -pthread -Isimulator/shims -Icomponents/si12t/include \
        -DSIM_DEFAULT_OUT="\"$here/out\"" -c "$src" -o "$obj" &
  fi
done
wait
g++ -pthread -o "$here/build/stackchan_datagen" "${objs[@]}"
echo "built $here/build/stackchan_datagen"
