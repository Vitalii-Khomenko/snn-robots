# StackChan neuromorphic brain

![Brain v4](stackchan-v4/docs/BRAIN_V4_MAP.svg)

Firmware for an **M5Stack CoreS3 StackChan** desktop robot whose behaviour comes from neural networks running
entirely on the ESP32-S3. No behaviour tree, no canned animations or sounds.

**Brain v4** (`stackchan-v4/`, what the robot runs now, `mode v4`) is built like an animal's nervous system:
- every sense becomes one number 0..1 (13 channels: loudness, sharp sounds, voice, head and screen touch, shake,
  tilt, proximity, light, motion, hunger, tiredness, loneliness), each scale divided into innate zones of boredom,
  comfort, interest, anxiety and anger;
- a 768-neuron **spiking analyzer** (SNN) reacts to them at once - receptive fields, zone neurons that habituate,
  a recurrent mixing pool, an attention map - and passes 32 numbers on (valence, arousal, threat, surprise,
  per-channel salience, drives, where to look);
- one **main network** (GRU) picks one of 21 emotions and all the actions that go with it: the droid voice, the LED
  colour and pattern, and the head - gaze, nod "yes", shake "no", flinch, doze;
- a C++ executor keeps the safety rules (quiet nights, call budget, servo limits).

Inside the two networks: [stackchan-v4/docs/NETWORKS_V4.svg](stackchan-v4/docs/NETWORKS_V4.svg).

**Platform B** (`stackchan-life/`, `mode life`, same firmware) is a robot that **learns from its own life**: a world
model whose surprise is curiosity, five needs that give the reward, an actor network, and a night sleep on the home PC
that retrains them from the diary of every tick. The **v3 hybrid brain** (`mode brain`) is still in the firmware; its
original project folder is in git history (tag `v3-neuromorphic`).

The project includes a **digital twin** simulator that runs the real firmware code in a virtual room, and the whole
training pipeline (PC, PyTorch) for every network.

## Start here

| | |
|---|---|
| [stackchan-v4/README.md](stackchan-v4/README.md) | **brain v4**: layout, training chain, modes; then platform B and v3 below it |
| [stackchan-v4/BRAIN_V4_PLAN.txt](stackchan-v4/BRAIN_V4_PLAN.txt) | the v4 design (in Russian) and [the sense scales](stackchan-v4/BRAIN_V4_INPUT_SCALES.txt) |
| [stackchan-life/README.md](stackchan-life/README.md) | **platform B**: the learning loop, getting started |
| [stackchan-life/docs/ARCHITECTURE.md](stackchan-life/docs/ARCHITECTURE.md#methods-and-technologies-platform-b) | methods and technologies (world model, homeostatic reward, AWR actor, online actor-critic, teacher distillation, the night sleep) |
| [stackchan-life/docs/HARDWARE.md](stackchan-life/docs/HARDWARE.md) | board, wiring, servos, sensors, flashing and recovery |
| [stackchan-life/docs/DEVELOPMENT.md](stackchan-life/docs/DEVELOPMENT.md) | simulator, tests, diary, media, training, OTA |
| [README.ru.md](README.ru.md) | краткое описание по-русски |

## Status

A personal research project, developed and tested on one robot. The simulator tests run on a fresh clone with only a
C++ toolchain. Brain v4 runs on the robot since 2026-10-24; learning from its own life (stage 6) is next. The StackChan
body and CoreS3 are M5Stack products; this repository is not affiliated with M5Stack.

## Repository layout

- `stackchan-v4/` - brain v4 (current): firmware (`main/`, `components/`), simulator, `tools/v4/` training, docs
- `stackchan-life/` - platform B: the same firmware base plus the learning loop, `tools/life/` on the PC, docs
- `firmware-builder/` - Dockerfile of the ESP-IDF v5.5.4 build image (`stackchan-firmware-builder:local`)

Not in the repository (too large or private; how to get them is in the project README, "Getting started"):
`secrets/` (Wi-Fi password, owner's device IPs; templates are provided), `datasets/` (COCO, ESC-50, LibriSpeech),
`.tooling/` (toolchain cache), `backups/`, regenerable training data and the owner's real recordings.

## License

[MIT](LICENSE). Third-party components keep their own licenses (ESP-IDF, LVGL, Espressif components, Bosch BMI270 API;
datasets: COCO CC BY 4.0, ESC-50 CC BY-NC 3.0 / CC BY 3.0 by clip, LibriSpeech CC BY 4.0). The teacher networks of
platform B are downloaded by the user and not redistributed here (Ultralytics YOLOv8 AGPL-3.0, YuNet / OpenCV Zoo,
Silero VAD MIT, PANNs MIT).
