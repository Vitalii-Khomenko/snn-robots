# StackChan neuromorphic brain

![Brain v4](stackchan-v4/docs/png/BRAIN_V4_MAP.png)

Firmware for an **M5Stack CoreS3 StackChan** desktop robot whose behaviour comes from neural networks running
entirely on the ESP32-S3. No behaviour tree, no canned animations or sounds.

**Brain v4** (`stackchan-v4/`, what the robot runs now) is built like an animal's nervous system:
- every sense becomes one number 0..1 (13 channels: loudness, sharp sounds, voice, head and screen touch, shake,
  tilt, proximity, light, motion, hunger, tiredness, loneliness), each scale divided into innate zones of boredom,
  comfort, interest, anxiety and anger;
- a 768-neuron **spiking analyzer** (SNN) reacts to them at once - receptive fields, zone neurons that habituate,
  a recurrent mixing pool, an attention map - and passes 32 numbers on (valence, arousal, threat, surprise,
  per-channel salience, drives, where to look);
- one **main network** (GRU) picks one of 21 emotions and all the actions that go with it: the droid voice, the LED
  colour and pattern, and the head - gaze, nod "yes", shake "no", flinch, doze;
- a C++ executor keeps the safety rules (quiet nights, call budget, servo limits).

Inside the two networks: [stackchan-v4/docs/png/NETWORKS_V4.png](stackchan-v4/docs/png/NETWORKS_V4.png).

The robot runs only brain v4; the home PC keeps the diary of every tick (statistics, analysis, debugging) and trains
the networks. The earlier brains are history: **platform B** (`stackchan-life/`, 2026-10-11..24), a robot that
learned from its own life (a world model, five needs as reward, an actor network, a nightly sleep on the PC), and
the **v3 hybrid brain** (git tag `v3-neuromorphic`). How it all developed: [stackchan-v4/docs/HISTORY.md](stackchan-v4/docs/HISTORY.md).

The project includes a **digital twin** simulator that runs the real firmware code in a virtual room, and the whole
training pipeline (PC, PyTorch) for every network.

## Start here

| | |
|---|---|
| [stackchan-v4/README.md](stackchan-v4/README.md) | **brain v4**: getting started, folders, status, rollback |
| [stackchan-v4/docs/ARCHITECTURE.md](stackchan-v4/docs/ARCHITECTURE.md) | channels and zones, the spiking analyzer, the main network, the executor, the diary |
| [stackchan-v4/docs/DEVELOPMENT.md](stackchan-v4/docs/DEVELOPMENT.md) | build, flash, simulator and tests, the training chain, the diary, recipes |
| [stackchan-v4/docs/HARDWARE.md](stackchan-v4/docs/HARDWARE.md) | board, wiring, servos, sensors, memory, flashing and recovery |
| [stackchan-v4/docs/HISTORY.md](stackchan-v4/docs/HISTORY.md) | v1 .. v4, platform B, the move to Omarchy, lessons, where the old code lives |
| [stackchan-life/README.md](stackchan-life/README.md) | platform B (history, the way back) |
| [README.ru.md](README.ru.md) | краткое описание по-русски |

## Status

A personal research project, developed and tested on one robot. The simulator tests run on a fresh clone with only a
C++ toolchain. Brain v4 runs on the robot since 2026-10-24; learning from its own life (stage 6) is next. The StackChan
body and CoreS3 are M5Stack products; this repository is not affiliated with M5Stack.

## Repository layout

- `stackchan-v4/` - brain v4 (current): firmware (`main/`, `components/`), simulator, `tools/v4/` training,
  `tools/robot/` (updates, diary), docs
- `stackchan-life/` - platform B (history and the way back): its firmware, PC tools and the diary data
- `firmware-builder/` - Dockerfile of the ESP-IDF v5.5.4 build image (`stackchan-firmware-builder:local`)

Not in the repository (too large or private; how to get them is in the project README, "Getting started"):
`secrets/` (Wi-Fi password, owner's device IPs; templates are provided), `datasets/` (COCO, ESC-50, LibriSpeech),
`.tooling/` (toolchain cache), `backups/`, regenerable training data and the owner's real recordings.

## License

[MIT](LICENSE). Third-party components keep their own licenses (ESP-IDF, LVGL, Espressif components, Bosch BMI270 API;
datasets: COCO CC BY 4.0, ESC-50 CC BY-NC 3.0 / CC BY 3.0 by clip, LibriSpeech CC BY 4.0). The teacher networks of
platform B are downloaded by the user and not redistributed here (Ultralytics YOLOv8 AGPL-3.0, YuNet / OpenCV Zoo,
Silero VAD MIT, PANNs MIT).
