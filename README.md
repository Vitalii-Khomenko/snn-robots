# StackChan neuromorphic brain

![Platform B](stackchan-life/docs/BRAIN_MAP.svg)

Firmware for an **M5Stack CoreS3 StackChan** desktop robot whose behaviour comes from a hybrid brain running
entirely on the ESP32-S3: small vision and hearing networks, a 768-neuron **spiking core** (attention, novelty,
boredom, curiosity, sleep...), a recurrent policy that picks actions and one of 21 faces, and an analogue droid
voice composed by a network. No behaviour tree, no canned animations or sounds.

**Platform B** (`stackchan-life/`, what the robot runs now) turns it into a robot that **learns from its own life**:
- a world model whose surprise is curiosity;
- five needs that give the reward;
- an actor network that decides, plus a plastic layer that keeps learning during the day;
- a night sleep on the home PC that retrains the world model and the actor from the diary of every tick, and the eyes
  and ears from the robot's own frames and sounds (labelled by large teacher networks). The new weights go back over
  Wi-Fi without a reboot.

The project includes a **digital twin** simulator that runs the real firmware code in a virtual room, and the
whole training pipeline (PC, PyTorch) for every network.

## Start here

| | |
|---|---|
| [stackchan-life/README.md](stackchan-life/README.md) | **platform B**: the learning loop, current state, getting started |
| [stackchan-life/docs/ARCHITECTURE.md](stackchan-life/docs/ARCHITECTURE.md#methods-and-technologies-platform-b) | platform B: methods and technologies (world model, homeostatic reward, AWR actor, online actor-critic, teacher distillation, the night sleep) |
| [stackchan-life/docs/DEVELOPMENT.md](stackchan-life/docs/DEVELOPMENT.md) | platform B tools: diary, media, training, the night sleep, OTA |
| [stackchan-neuromorphic/README.md](stackchan-neuromorphic/README.md) | the v3 base: what the robot does, **getting started**, build / flash, configuration, log reference |
| [docs/ARCHITECTURE.md](stackchan-neuromorphic/docs/ARCHITECTURE.md) | the brain and the rules you must not break |
| [docs/HARDWARE.md](stackchan-neuromorphic/docs/HARDWARE.md) | board, wiring, servos, sensors, flashing and recovery |
| [docs/DEVELOPMENT.md](stackchan-neuromorphic/docs/DEVELOPMENT.md) | simulator, tests, training pipeline, recording real data |
| [README.ru.md](README.ru.md) | краткое описание по-русски |

## Status

A personal research project, developed and tested on one robot. The simulator test (`simulator/test.sh`) passes on a fresh clone
with only a C++ toolchain. The person detector is the least mature part; see "What is not done yet" in the project README and
the known limitations in [HARDWARE.md](stackchan-neuromorphic/docs/HARDWARE.md). The StackChan body and CoreS3 are M5Stack products;
this repository is not affiliated with M5Stack.

## Repository layout

- `stackchan-life/` - platform B (current): the same firmware base plus the learning loop, `tools/life/` on the PC, docs
- `stackchan-neuromorphic/` - the v3 base: firmware (`main/`, `components/`), simulator, training tools, docs
- `firmware-builder/` - Dockerfile of the ESP-IDF v5.5.4 build image (`stackchan-firmware-builder:local`)

Not in the repository (too large or private; how to get them is in the project README, "Getting started"):
`secrets/` (Wi-Fi password, owner's device IPs; templates are provided), `datasets/` (COCO, ESC-50, LibriSpeech),
`.tooling/` (toolchain cache), `backups/`, regenerable training data and the owner's real recordings.

## License

[MIT](LICENSE). Third-party components keep their own licenses (ESP-IDF, LVGL, Espressif components, Bosch BMI270 API;
datasets: COCO CC BY 4.0, ESC-50 CC BY-NC 3.0 / CC BY 3.0 by clip, LibriSpeech CC BY 4.0). The teacher networks of
platform B are downloaded by the user and not redistributed here (Ultralytics YOLOv8 AGPL-3.0, YuNet / OpenCV Zoo,
Silero VAD MIT, PANNs MIT).
