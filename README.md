# StackChan neuromorphic brain

![Brain map](stackchan-neuromorphic/docs/BRAIN_MAP.svg)

Firmware for an **M5Stack CoreS3 StackChan** desktop robot whose behaviour comes from a hybrid brain running
entirely on the ESP32-S3: small vision and hearing networks, a 768-neuron **spiking core** (attention, novelty,
boredom, curiosity, sleep...), a recurrent policy that picks actions and one of 21 faces, and an analogue droid
voice composed by a network. No behaviour tree, no canned animations or sounds.

The project includes a **digital twin** simulator that runs the real firmware code in a virtual room, and the
whole training pipeline (PC, PyTorch) for every network.

## Start here

| | |
|---|---|
| [stackchan-neuromorphic/README.md](stackchan-neuromorphic/README.md) | what the robot does, **getting started**, build / flash, configuration, log reference |
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

- `stackchan-neuromorphic/` - the project: firmware (`main/`, `components/`), simulator, training tools, docs
- `firmware-builder/` - Dockerfile of the ESP-IDF v5.5.4 build image (`stackchan-firmware-builder:local`)

Not in the repository (too large or private; how to get them is in the project README, "Getting started"):
`secrets/` (Wi-Fi password, owner's device IPs; templates are provided), `datasets/` (COCO, ESC-50, LibriSpeech),
`.tooling/` (toolchain cache), `backups/`, regenerable training data and the owner's real recordings.

## License

[MIT](LICENSE). Third-party components keep their own licenses (ESP-IDF, LVGL, Espressif components, Bosch BMI270 API;
datasets: COCO CC BY 4.0, ESC-50 CC BY-NC 3.0 / CC BY 3.0 by clip, LibriSpeech CC BY 4.0).
