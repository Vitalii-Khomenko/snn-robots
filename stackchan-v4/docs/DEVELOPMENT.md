# Development

Everything below runs from `stackchan-v4/` unless a path says otherwise. Python tools in `tools/v4/` run through
`tools/v4/py`, which uses `$STACKCHAN_PY` (default `~/.venvs/stackchan/bin/python`; packages in
`tools/requirements.txt` plus PyTorch).

## 1. The workflow after any change

1. Firmware change -> `simulator/build.sh && tools/v4/py behavior_test_v4.py` (all PASS).
2. Network or channel change -> the parity tests (section 4) first.
3. `./build.sh build` (no new warnings), then `python3 tools/robot/ota.py firmware` (the robot boots it on trial and
   confirms it after a healthy minute, else it goes back by itself).
4. Look at the robot's lines for a few minutes (section 5): no `E (`, sane `stats:`, `v4:` and `power:` lines.
5. Write what was done, seen and measured into `JOURNAL.txt` (Russian, newest first); update the docs.
6. Diagram changed -> `tools/publish_maps.sh` (SVG -> high-quality PNG in `docs/png/` -> GitHub).

Never flash at night (the owner sleeps next to the robot, 21:45-06:00): a reboot makes sounds.

## 2. Build and flash

```bash
./build.sh build                       # ESP-IDF v5.5.4 in the stackchan-firmware-builder:local image, cache ../.tooling
./build.sh reconfigure                 # after adding a source file under main/ (the source list is globbed)
python3 tools/robot/ota.py firmware    # over Wi-Fi (normal way); build/stackchan_v4.bin
python3 tools/robot/ota.py status      # {"firmware": "ota_0", "brain": "v4", "uptime_s": ..., "diary_sent": ...}
python3 tools/robot/ota.py touch 5     # head pad sensitivity 0..7, kept in NVS
python3 tools/robot/ota.py learned F   # a learned blob (stage 6): switched in without a reboot, on trial for a minute
./flash.sh                             # over the cable (first flash, recovery); nothing else may hold /dev/ttyACM0
python3 tools/capture_serial.py --seconds 30   # bounded serial capture, no reset
```

The robot finds the PC from `secrets/life_log_host.h` (the diary host and the OTA token). `ota.py` finds the robot
from `data/life/robot.json`, written by the diary receiver. The PC must keep its address (192.168.8.110 here).

## 3. The simulator (digital twin)

`simulator/` runs the real firmware code (the ear, the channels, both networks, the executor, motion, the voice, the
sensor engines) in a simulated room with scripted people, sounds, touches, light, shakes and the time of day.

```bash
simulator/build.sh                                   # g++, no ESP-IDF needed
simulator/run.sh simulator/scenarios/petting.scn [--minutes N] [--seed N] [--echo] [--voice-wav F] [--learned F] [--v4-dump F]
# -> simulator/out/<scenario>/log.txt (the robot's log), trace.csv (10 Hz: head, face, channels, mood), summary.txt
tools/v4/py behavior_test_v4.py                      # 13 scenarios in parallel, ~40 s
```

| Behaviour test | Scenario | Checks |
|---|---|---|
| touch_response | touches now and then | >= 80 % of touches answered within 2 s (a call or a gesture) |
| pokes | taps every 0.8 s | it gets cross (annoyed / angry face or a head shake) |
| petting | slow strokes | never angry after a touch; answers the hand |
| night_quiet | a night | no calls at all; sleepy most of the time |
| play_session | play | >= 5 different faces, talks |
| day_alone | an empty room by day | babbles 40..600 calls/h, moods change |
| learn_rough | shaken, picked up | frightened (scared / dizzy / surprised / flinch) |
| hearing | claps and bangs | startled |
| hard_day | 14 minutes alone of bangs (every 10 s) and shakes (every minute), 10 quiet minutes, then a person close strokes its head every 30 s for 6 minutes | the body budget falls to <= 0.45 (minutes 12-24) and the hand refills it by >= 0.15 (minute 31 against 24; [BODY](BODY.md) step 2) |
| hard_day_control | the same without the hard stretch | the budget stays >= 0.75 |
| seek_motion | someone walks about the room (a few seconds at ~1 m/s, then stands), often outside the picture | report only: the share of the time the head points within 25° of them (the simulated walker is a low-contrast figure, found as often as by chance: 0.37 over 5 seeds, the same with nobody there; the robot's diary judges the motion sense) |
| seek_light | a lamp switched on beside the desk (-55°), outside the picture | the head finds it within 2 minutes and points at it >= 0.05 more of the time than the same run with the lamp left off (`lamp_over_chance`; until 2026-10-26 >= 50 % absolute, the old busy head's level). 10 seeds, 2026-10-26: 45 % vs 16 %, found after 49-67 s |
| contingent | someone close answers the robot's calls and gestures with a pet, 0.8-2 s later, 85 % of the time (`responder`) | report over 3 seeds: joy after an answer (a joyful face 1.5 s later; 0.35 was a gate until 2026-10-27 evening - one seed's luck, 6 seeds gave 0.09-0.33), the answered act again within 8 s over chance (CONTINGENCY) |
| control | the same person and as many pets, at random moments | calls and gestures a minute over 3 seeds: `contingent` / `control` >= 1.05 (6 seeds: 1.10) |
| still_face | answered for 4 minutes, then ignored by the same person close by, answered again at 11:00 | over 3 seeds: low faces (sad / bored) >= 20 % of 7:00-11:00; never angry or annoyed after 5:00 |

**Time.** Firmware tasks (main loop, microphone, camera, voice) run on a virtual RTOS (`simulator/core/vrtos`): each
in its own thread but strictly one at a time, and time advances only when all wait. The same scenario and seed always
give the same result. A brain tick costs modelled CPU time (`--brain-cost`, default 4 ms). A 10 min scenario takes
~5 s. The simulator's IMU mounting matches the real head (pitch rotates gravity); its servos are quieter than the
real ones (do not tune servo-noise filters in the simulator). The light sensor answers the firmware's 4-byte burst read
(since 2026-10-25; from 2026-10-24 the simulated light channel had read 0, a dark room, in every scenario).

**Scenario format.** A `*.scn` file describes the room and the events. `#` starts a comment. Times are `s`, `m:ss` or
`h:mm:ss` since power-on. Azimuth `+` is to the robot's right (like head yaw), elevation `+` is up.

```text
duration 15:00   clock 07:25   sunrise 08:40   sunset 20:00   seed 1   charging on   battery 100   owner away
room_light off
window 170 20 40 30                       # azimuth, elevation, width, height (deg) [gain G]
lamp desk 100 12 on [brightness B] [radius R]
noise_floor 25      fov 60

at 1:00 touch_head 400                    # hand on the head, ms (also pet, tap_screen, double_/triple_tap_screen)
at 2:00 clap 70 [level]                   # a clap from an azimuth
at 2:05 talk -40 3 [level]                # synthetic speech: azimuth, seconds   (talk_ramp: getting louder)
at 2:10 bang 30 [level]                   # a door slam: low, ringing
at 2:12 hiss -50 0.6 [level]              # a burst of high-passed noise
at 2:15 tone 20 1200 1.5 [level]          # azimuth, Hz, seconds
at 2:20 whistle 0 800 1600 1.2 [level]    # a sweep
at 2:30 clip 60 /path/clip.raw 0.8 speech # a recorded sound (raw int16 24 kHz)
at 4:00 noise 400 0.9                     # background noise level, spectral tilt 0 white .. 1 low-pass (a fan)
at 5:00 lamp desk off          at 5:00 room_light on
at 6:00 person alex enter -40 1.5         # azimuth, distance (m); then walk AZ SECONDS, approach DIST SECONDS,
                                          # come_close, step_back, leave
at 6:30 thing ball enter 30 10 8 0.6      # a non-person object: azimuth, elevation, size (deg), reflectance;
at 6:31 thing ball move 60 10 2           # then move AZ EL SECONDS, leave
at 7:00 shake 2    at 7:10 pick_up 25    at 7:20 put_down    at 7:30 owner home    at 7:40 log any text
every 0:05 from 2:00 until 2:30 touch_head 300
at 0:50 responder call,gesture pet 0.85 0.8 2.0   # a person who answers the robot's acts (call / gesture / face) with
                                          # pet / talk / tap, p, latency s lo..hi (docs/CONTINGENCY.md)
at 0:50 responder random pet 20           # ... or answers at random, every ~20 s (the control);  responder off
```

Besides the test scenarios (`behavior_test_v4.py` CASES; since 2026-10-27 also `contingent`, `control`, `still_face`): `conversation`, `startle_and_orient`, `vision_motion`, `ignored`, `alone_dark_morning`,
`bright_door`, `idle_owner_home`, `idle_endurance`, `full_day` (15 min .. 24 h, for looking at the robot's life).

## 4. Training chain

```
spec_v4.py ──> main/v4/v4_spec.h
calibrate.py                 real days -> channel statistics per zone (out/calibration.txt)
train_snn_v4.py data         1400 synthetic + 400 real 5-minute lives -> out/snn_data.npz (~0.9 GB, rebuildable)
train_snn_v4.py train        SNN, 20 epochs (surrogate gradients, int8 QAT in the last 30 %) -> data/snn_v4.pt
export_snn_v4.py ──> main/v4/snn_v4_data.h        parity_snn_v4.py   (C++ == PyTorch)
train_nn_v4.py data          the trained SNN over the same lives -> out/nn_snn.npz (~0.6 GB, rebuildable)
muse.py make                 the Muse's fixed reservoir (seeded), calibrated on those lives -> main/v4/muse_v4_data.h
                             parity_muse.py (C++ == numpy); a new Muse means a retrained main network
train_nn_v4.py train [2]     imitation of teacher_nn.py + DAgger rounds -> data/nn_v4.pt, out/nn_report.txt
export_nn_v4.py ──> main/v4/nn_v4_data.h          parity_nn_v4.py
behavior_test_v4.py          then build, flash, watch
```

- `synth_v4.py` paints synthetic lives into the **raw** sensor signals (the owner coming and going, talk, TV, the
  vacuum cleaner, claps, petting, pokes, being picked up, shaken, covered for the night, the battery, sensor quirks
  such as phantom head touches and false speech at night); `channels.py` turns them into the 13 channels exactly as
  the firmware does (`parity_channels.py`).
- `teacher_snn.py` is the innate appraisal the SNN learns (zone valence / arousal, salience, habituation, needs,
  attention). `teacher_nn.py` is the behaviour teacher of the main network (memory, the owner's rules: talks often,
  every touch answered, petting never angers, quiet nights, nod / shake used sensibly).
- Real days: `calibrate.raw_hour(day, hour)` reads brain v4's diary (raw readings logged, exact) or platform B's
  older days (`legacy_diary.py`, approximations for shake and motion; no light or proximity).
- A full retrain takes ~1 h on an RTX-class GPU (the SNN data pass is the slowest). Checkpoints: `tools/v4/data/*.pt`
  (in git); reports and logs: `tools/v4/out/*.txt|*.log` (in git); the large `.npz` are not.

## 5. Learning from life (stage 6)

```bash
tools/v4/py reward_v4.py 2026-10-25            # a day's reward events (pet, tap, talk, near, shake, tilt, poke, left_alone)
tools/v4/py contingency.py 2026-10-27          # the contingency detector's twin over a day: answered acts, agency, ignored, the robot's ct_* vs the twin
tools/v4/py contingency.py --measure 2026-10-26  # step 0: P(answer | act) vs chance over whole days (docs/CONTINGENCY.md)
tools/v4/py parity_contingency.py [DAY]       # C++ ContingencyV4 == the twin on real diary days
tools/v4/py body.py --measure 2026-10-26      # the body's channels hour by hour, the sleep pressure's twin (docs/BODY.md)
tools/v4/py body.py 2026-10-28                # the twin's sleep pressure and body budget against the robot's (raw_stamina, raw_budget)
tools/v4/py parity_body.py [DAY]              # C++ BodyV4 == the twin on real diary days
tools/v4/py jerks.py 2026-10-27 [--since HH:MM]  # the head's own jerks (gyro) and hangs per servo rail switch-on, by hour
tools/v4/py contingency.py --self 2026-10-26  # me or not me: sense events that follow the robot's own acts at once
python tools/robot/listen.py [SECONDS]        # record the robot's microphone (24 kHz, ms-stamped) -> data/sound/<day>/<time>.wav
tools/v4/py sound_check.py data/sound/DAY/TIME  # every speech onset of the ear vs the sound and the robot's own acts, + .png
tools/v4/py voice_check.py data/sound/DAY/TIME  # the voice check over a recording;  parity_voice_check.py: C++ == twin
tools/v4/py sleep_v4.py --dry-run              # the night without sending: replay, reward, AWR, zones, gates
tools/v4/py sleep_v4.py --dry-run --min-hours 0.05   # the whole chain on a few minutes of life (a test)
journalctl --user -u sleep-v4                  # last night; one line per night in data/life/sleep_v4.log
python3 tools/robot/ota.py status              # learned_version / learned_slot / learned_trial on the robot
```

To undo what the robot learned: send an older blob from `data/learned/` with `ota.py learned`, or the innate one
(`tools/v4/py learned.py innate out/learned_innate.bin`). Retraining the networks themselves (section 4) keeps working:
the learned heads are a fine-tuning of the main network, so after a new `nn_v4.pt` remove `data/learned/live.bin` and
send the innate blob, the next nights start from the new network.

## 6. The diary and the robot's lines

- `tools/robot/lifelogd.py` (systemd user service `lifelogd`) receives the UDP diary into
  `data/life/<day>/<hour>.scl` and `log.txt`; `compress-diary.timer` gzips finished days at 04:30 (~5x smaller).
- `tools/v4/py diary.py 2026-10-25` - a day's summary (hours, losses, channels, faces, calls, gestures).
- In Python: `ticks, lines = diary.load(day)`; `ticks["v"][:, diary.col("snn_valence")]`; `lines` are all log lines.

| Line | Meaning |
|---|---|
| `runtime: brain v4: spiking analyzer trained, main network trained, auditory NN trained` | boot: weights present |
| `runtime: head touch (level N)` / `... ignored: brief, during own motion` / `head touch storm ...` | the head pad and its filters |
| `runtime: screen touch`, `runtime: loud sound: peak=N`, `runtime: hear: speech 0.73 ...` | other senses |
| `v4: face: Happy (valence ... arousal ...)` | a new face (the SNN's mood at that moment) |
| `v4: call (face=..., reaction)`, `v4: voice: B- C- ... (781 Hz)` | the voice: a phrase started / ended |
| `v4: gesture: nod (yes) amp 9 deg x2` | nod, shake, flinch |
| `runtime: stats: busy=... internal_min=... tick_us=... cam_fps=... servo_repowers=... diary=sent/dropped` | every 5 s: the machine |
| `runtime: v4: als=.. ps=.. ch=13 values snn=on/3246us val= aro= ... drives=1 night=0 face= calls= nods= shakes=` | every 5 s: the brain |
| `runtime: power: vbus_min=... vsys_min=... servo_rail=ons/ms` | every 5 s: the power rails; the servo rail's switch-ons and ms on (power-gated moves) |
| `runtime: loop_us (servos face leds+pad imu+mic sensors brain diary supervise, mean/max): ...` | every 5 s: where the main loop's time goes, per stage, us |
| `runtime: servo rail power cycle (...)` | the servo watchdog acted (HARDWARE.md) |
| `runtime: speaker volume 5/10 (codec 76)` | the owner set the volume on the screen (the status JSON: `volume`) |
| `runtime: boot reason: ...`, `crash of the last run: ...` | why it (re)started |

## 7. Pictures

`tools/v4_map.py` draws `docs/BRAIN_V4_MAP.svg` (the whole robot) and `docs/NETWORKS_V4.svg` (inside the networks)
from `spec_v4.py`. `tools/publish_maps.sh` redraws them, renders `docs/png/*.png` at 2x (4800 px, for the website) with
`rsvg-convert`, commits only `docs/*.svg` and `docs/png/`, and pushes. `tools/face_preview/run.sh` renders all 21 faces
into `docs/faces.png` and the named sheet `docs/png/FACES_V4.png` (2x, for the website) exactly as the robot draws
them (`run.sh vector`: the old smooth faces). New or changed faces: the glyphs and the composition per emotion are in
`main/ui/pixel_face.cpp` (`renderPixelFace`); `publish_maps.sh` redraws and publishes the sheet too.

## 8. Recipes

- **Change a zone border:** `ZONE_TABLE` in `spec_v4.py` -> `./py spec_v4.py` -> retrain both networks (the SNN's zone
  layer is wired from the table) -> parity -> behaviour tests -> flash.
- **Change how a channel is computed:** `main/v4/channel_encoder.cpp` and `tools/v4/channels.py` together ->
  `parity_channels.py` -> retrain (the synthetic data goes through the new encoder).
- **Change behaviour:** the teacher (`teacher_nn.py`), then `train_nn_v4.py data && train_nn_v4.py train`; the SNN stays.
- **Add a raw reading to the diary:** `RAW` (or `BODY`) in `spec_v4.py`, `./py spec_v4.py`, fill it in
  `RobotRuntime::logTick` in the same order (a `static_assert` checks the count).
- **Test the learned-blob chain:** `tools/v4/py learned.py innate out/learned_innate.bin` packs the compiled-in heads
  with no zone shift; in the simulator (`--learned`) the log must be identical to a run without it; on the robot
  `ota.py learned` must switch and confirm without a reboot.
- **Look at a moment of the robot's life:** `diary.load(day, hours=[h])`, find the time in `lines`, plot `ticks["v"]`.

## 9. The PC side

| Service (systemd user) | What |
|---|---|
| `lifelogd` | the diary receiver (`tools/robot/lifelogd.service`) |
| `sleep-v4.timer` | brain v4's night sleep at 03:30 (`tools/v4/sleep_v4.py`, stage 6) |
| `compress-diary.timer` | gzips finished diary days daily at 04:30 |

Platform B's services (`mediad`, `life-sleep`, `life-morning`) were removed on 2026-10-24; their unit files stay in
`../stackchan-life/tools/life/` for the way back.

The project lives on the external drive `/run/media/warmond/KIOXIA/Arhiv/For Lenovo Laptop/Codex/StackChan`, linked as
`~/Codex/StackChan` (the services use that path). Git: `core.fileMode=false` (the drive shows every file as
executable).
