# stackchan-v4: brain v4

![Brain v4](docs/png/BRAIN_V4_MAP.png)

Firmware and tools of the StackChan desktop robot (M5Stack CoreS3, ESP32-S3) with **brain v4**, built like an
animal's nervous system:

```
senses -> 13 channels, each a number 0..1 with innate zones (boredom, comfort, interest, anxiety, anger)
       -> spiking analyzer, 768 LIF neurons, every 50 ms: the "instant nervous system", 32 numbers out
       -> main network (GRU), every 100 ms: one emotion out of 21 + every action that goes with it
       -> executor: face, droid voice, LED colour and pattern, head (gaze, nod "yes", shake "no", flinch, doze)
          inside fixed safety rules (silent nights, a call budget, servo limits)
```

Everything runs on the robot. The home PC receives the diary of every tick (statistics, analysis, debugging) and
trains the networks; nothing else leaves the robot (no frames, no sound).

Inside the two networks: [docs/png/NETWORKS_V4.png](docs/png/NETWORKS_V4.png). Both pictures are drawn from the code
(`tools/v4_map.py`, SVG in `docs/`).

The 21 faces, pixel art in the style of [airwitech.com](https://airwitech.com) (the robot's head is square, so is its
face):

![The faces](docs/png/FACES_V4.png)

## Documentation

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the brain: channels and zones, the spiking analyzer, the main network, the executor, the diary |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | build, flash, simulator and tests, the training chain, the diary on the PC, recipes |
| [docs/HARDWARE.md](docs/HARDWARE.md) | the board, servos, sensors, memory, flashing and recovery, known limits |
| [docs/HISTORY.md](docs/HISTORY.md) | how the project got here (v1 .. v4, platform B), what was learned, where the old code lives |
| [BRAIN_V4_PLAN.txt](BRAIN_V4_PLAN.txt), [BRAIN_V4_INPUT_SCALES.txt](BRAIN_V4_INPUT_SCALES.txt) | the owner's design of v4 and the sense scales (in Russian) |
| `JOURNAL.txt` | the development journal (in Russian, newest first): what was done, seen, measured, left open |

## Getting started

Needs Linux with Docker, the robot on USB (`/dev/ttyACM0`) for the first flash, and ~10 GB of disk for the toolchain
cache (`../.tooling`). Training needs Python 3 with PyTorch (a CUDA GPU helps) and numba.

```bash
git clone https://github.com/Vitalii-Khomenko/snn-robots.git && cd snn-robots
docker build -t stackchan-firmware-builder:local firmware-builder     # ESP-IDF v5.5.4 image
cd stackchan-v4
cp secrets/wifi_credentials.example.h secrets/wifi_credentials.h      # edit: your Wi-Fi
cp secrets/known_devices.example.h    secrets/known_devices.h         # edit: your phone's IP (or an empty list)
cp secrets/life_log_host.example.h    secrets/life_log_host.h         # edit: the PC that keeps the diary, OTA token
./build.sh build && ./flash.sh                                        # first time over the cable
```

The trained networks are generated headers in the repository (`main/v4/*_data.h`, `main/perception/auditory_model_data.h`,
`main/voice/babble_model_data.h`): the firmware builds and runs without training. After the first flash:

```bash
cp tools/robot/lifelogd.service ~/.config/systemd/user/ && systemctl --user enable --now lifelogd   # the diary
cp tools/robot/sleep-v4.* tools/robot/compress-diary.* ~/.config/systemd/user/
systemctl --user enable --now sleep-v4.timer compress-diary.timer                                  # learning, compression
python3 tools/robot/ota.py status                      # the robot answers over Wi-Fi
./build.sh build && python3 tools/robot/ota.py firmware   # every later update, over Wi-Fi, with rollback
```

## Folders

```
main/                firmware (ESP-IDF component)
  v4/                brain v4: channel_encoder, snn_v4 (analyzer), nn_v4 (main network), brain_v4, v4_executor,
                     v4_spec.h (generated from tools/v4/spec_v4.py), the trained weights (*_data.h)
  core/              brain_loop (senses -> brain v4 -> executor, shared with the simulator), ms_clock, I2C recovery
  perception/        the ear (audio front end + Auditory NN: loudness, speech) and the vision front end (motion)
  sensors/ motion/ ui/ led/ voice/ audio/ power/    hardware engines: camera, microphones, IMU, touch, proximity /
                     light; servos; the face; LEDs; the droid voice; the speaker; power rails and the PMU
  net/               Wi-Fi + NTP, the diary over UDP (life_log), the update server (ota_server)
  robot_runtime.*    bring-up, the main loop, servo watchdog, stats lines
simulator/           the digital twin: the real firmware in a simulated room (scenarios/*.scn)
tools/v4/            spec, channels, synthetic lives, teachers, training, export, parity, behaviour tests, diary readers,
                     stage 6: reward_v4.py, sleep_v4.py, learned.py
tools/robot/         ota.py (status, firmware, learned blob over Wi-Fi), lifelogd.py (diary receiver), compress_diary.py,
                     the systemd units (lifelogd, sleep-v4 at 03:30, compress-diary at 04:30)
tools/               v4_map.py + publish_maps.sh (the pictures), face_preview, capture_serial.py, serial_guard.sh
components/          vendored drivers: ftservo (servos), bmi270 (IMU), py32_io_expander, si12t (head pad)
secrets/             your Wi-Fi, device IPs, diary host and OTA token (templates *.example.h; real files never committed)
data -> ../stackchan-life/data   the diary (data/life/<day>/<hour>.scl[.gz]) shared with platform B's history
```

## Status (2026-10-25)

- Brain v4 drives the robot since 2026-10-24; the firmware holds nothing else (v3 and platform B were removed from it
  the same day: CPU busy 45 % -> 37 %, internal RAM minimum 14 -> 21 KB).
- Spiking analyzer: correlation with its teacher 0.96-0.99 on synthetic and real days; 3-5 ms per tick on the robot.
- Main network: emotion agreement with its teacher 0.90 when acting alone; ~210-260 calls an hour alone by day;
  every touch answered; petting never angers; pokes do; silent at night; frightened by shakes and bangs
  (`tools/v4/behavior_test_v4.py`, all pass).
- Stage 6, learning from its own life, runs since 2026-10-24: every night at 03:30 the PC replays the robot's decisions,
  rewards them by how people answered (petting, talk, coming close vs shaking, rough pokes, leaving), fine-tunes the
  main network's action heads (AWR), shifts the zones of loudness, sounds, voice, proximity, light and motion by
  experience, checks everything in the simulator and sends it to the robot without a reboot
  ([ARCHITECTURE](docs/ARCHITECTURE.md#7-what-the-robot-learns-from-its-life-stage-6)).
- The proximity sensor works (a hand at the face: up to ~1100 counts, 2026-10-24).
- The first night of stage 6 (2026-10-25 03:30): the heads stayed (KL 0.058 over the 0.05 gate), the sharp-sound zone
  moved +0.01 (habituated); the blob went live without a reboot.
- Every mood moves its head in its own way (2026-10-25): turning away when cross, a hanging swaying head when sad,
  bobbing when laughing, wide looks when curious, ... (ARCHITECTURE section 3).
- Twice as tolerant (2026-10-25): every input needs about twice the stimulus to alarm or anger it (claps, bangs,
  pokes, shaking, motion, light), see ARCHITECTURE section 1.
- A speaker button on the screen (2026-10-25): the volume 0-10 is set on the fly, beeps at the new level and is kept
  across reboots.

## Rollback

Firmware updates over Wi-Fi boot on trial: a firmware that does not reach a healthy minute boots back into the previous
one by itself. To go back to platform B on purpose: build and send `../stackchan-life` (`./build.sh build &&
python3 tools/life/ota.py firmware` there, then `python3 tools/life/ota.py mode life`), and re-enable its PC services
(`mediad`, `life-sleep.timer`, `life-morning.timer`).
