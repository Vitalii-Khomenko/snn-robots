#!/usr/bin/env bash
# Builds the StackChan simulator: the real firmware sources from ../main
# plus the host platform layer, world model and virtual RTOS in this folder.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
mkdir -p "$here/build"

firmware=(
  main/perception/vision_frontend.cpp main/perception/audio_frontend.cpp main/perception/auditory_net.cpp
  main/core/brain_loop.cpp main/v4/channel_encoder.cpp main/v4/snn_v4.cpp main/v4/nn_v4.cpp main/v4/brain_v4.cpp main/v4/muse_v4.cpp
  main/v4/v4_executor.cpp main/voice/droid_synth.cpp main/voice/babble_net.cpp main/voice/babble_voice.cpp
  main/motion/motion_engine.cpp main/ui/face_style.cpp main/sensors/microphone_engine.cpp main/sensors/camera_engine.cpp
  main/sensors/imu_engine.cpp main/sensors/touch_engine.cpp main/sensors/presence_engine.cpp
  main/power/power_status_engine.cpp
)
simulator=(
  simulator/main.cpp simulator/core/vrtos.cpp simulator/core/log.cpp simulator/hal/platform.cpp
  simulator/world/world.cpp simulator/world/scenario.cpp simulator/runtime/sim_runtime.cpp
)

cd "$root"
# Firmware files are compiled exactly as they are; warnings there belong to
# the firmware build, so only the simulator's own files use -Werror.
objs=()
for src in "${firmware[@]}" "${simulator[@]}"; do
  obj="$here/build/$(echo "$src" | tr '/' '_' | sed 's/\.cpp$/.o/')"
  objs+=("$obj")
  if [[ ! -f "$obj" || "$src" -nt "$obj" || -n "$(find main/ simulator/ -name '*.h' -newer "$obj" -print -quit)" ]]; then
    flags=(-Wall -Wextra)
    [[ "$src" == simulator/* ]] && flags+=(-Werror)
    g++ -std=c++17 -O2 -g "${flags[@]}" -pthread -Isimulator/shims -Icomponents/si12t/include \
        -DSIM_DEFAULT_OUT="\"$here/out\"" -c "$src" -o "$obj" &
  fi
done
wait
g++ -pthread -o "$here/build/stackchan_sim" "${objs[@]}"
echo "built $here/build/stackchan_sim"
