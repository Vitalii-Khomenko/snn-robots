# Development: simulator, training, tests, tools, history

## Workflow

For every change to the brain, behaviour, voice or motion:

1. Read the rules in [ARCHITECTURE.md](ARCHITECTURE.md#rules-you-must-not-break).
2. Try it in the simulator (a targeted scenario, then the regression tests):
   ```bash
   simulator/test.sh             # smoke test + determinism, ~1 min
   simulator/behavior_test.sh    # behaviour regression (scenarios in parallel), ~5 min
   ```
3. Build, flash, and check the boot log on the robot:
   ```bash
   ./build.sh build && ./flash.sh && python3 tools/capture_serial.py --seconds 30
   ```
   After every firmware change the robot is flashed (the owner tests live).
4. Keep `simulator/runtime/sim_runtime.cpp` in sync with `main/robot_runtime.cpp` (loop order, callbacks, stats).
   Everything between the engines and the brain is shared code in `main/core/brain_loop.*`.
5. If an interface changed (a feature, an action), edit `tools/hybrid/spec.py`, run `python3 tools/hybrid/spec.py`,
   retrain what depends on it, and run the parity tests.
6. Update the documentation (these four files) and redraw the map with `python3 tools/hybrid_map.py`.

The shared Python environment (torch with CUDA, numpy, scipy, soundfile, Pillow, faster-whisper) lives in
`tools/vocal_decoder/.venv` (historical name; `tools/hybrid/py` runs it from `tools/hybrid`).

## The simulator (digital twin)

```bash
simulator/run.sh simulator/scenarios/play_session.scn            # build and run
simulator/build/stackchan_sim SCENARIO.scn --minutes 8 --voice-wav /tmp/voice.raw   # what the speaker played
```

**Principle: the real firmware code runs.** The simulator compiles the files from `main/` unchanged and
substitutes virtual hardware:

| Real firmware code (unchanged) | Model in the simulator |
|---|---|
| `hybrid/*` (front ends, Vision / Auditory NN, spiking core, policy, body sense, hybrid brain) | - |
| `sensors/camera_engine` (own task) | camera driver: a rendered room 160x120 RGB565, exposure locked, sensor noise, a frame every 143 ms |
| `sensors/microphone_engine` (own task) | room acoustics: noise, voices, claps, tones, recorded clips with direction, servo whine and clicks, the robot's own speaker |
| `voice/*`, `behavior/body_executor` | `AudioOutput`: I2S pacing at 24 kHz, the sound goes back into the microphones and through the self-sound analyzer |
| `motion/motion_engine` | SCS bus and servo mechanics (speed, inertia, 0.3125 deg per unit) |
| `sensors/imu_engine`, `touch_engine`, `presence_engine`, `power/*` | LTR-553 and AXP2101 registers over I2C, BMI270 (accelerometer **and gyro**) and Si12T drivers |
| `core/brain_loop` (shared) + `RobotRuntime` -> `runtime/sim_runtime.cpp` | Wi-Fi: connect ~9 s, NTP ~40 s, owner ping every 20 min |

**Time.** Firmware tasks (main loop, microphone, camera, voice) run on a virtual RTOS (`core/vrtos`): each in its
own thread but strictly one at a time, and time advances only when all wait. The same scenario and seed always give
the same result (`simulator/test.sh` checks it bit for bit). A brain tick costs modelled CPU time (`--brain-cost`,
default 4 ms). A 10 min scenario takes ~5 s.

**Output** in `simulator/out/<scenario>/`: `log.txt` (the robot's UART log format with the same `brain:` and
`stats` lines and events, plus `world:` lines), `summary.txt` (per minute), `trace.csv` (every 100 ms: pose,
emotion, sound, SNN drives, attention, policy outputs, `person_x` = the nearest visible person's image position).

Options: `--minutes N`, `--seed N`, `--out DIR`, `--echo`, `--brain-cost MS`, `--voice-wav F` (raw int16, 24 kHz,
everything the speaker plays), `--bus-dump F` (every tick: 100 policy inputs, 3 efference values, 51 actions;
float32), `--policy-pipe` (the policy runs outside: the sim writes the 100 inputs to stdout and reads 51 actions
from stdin - used by `tools/hybrid/rollout.py`).

### Scenario format

A `*.scn` file describes the room and the events. `#` starts a comment. Times are `s`, `m:ss` or `h:mm:ss` since
power-on. Azimuth `+` is to the robot's right (like head yaw), elevation `+` is up.

```text
duration 15:00   clock 07:25   sunrise 08:40   sunset 20:00   seed 1   charging on   battery 100   owner away
room_light off
window 170 20 40 30                       # azimuth, elevation, width, height (deg) [gain G]
lamp desk 100 12 on [brightness B] [radius R]
noise_floor 25      fov 60
learned voice 0.6 0                       # experience the robot starts with (AssocMemory: cue, petting, rough handling)

at 1:00 touch_head 400                    # hand on the head, ms (also pet, tap_screen, double_/triple_tap_screen)
at 2:00 clap 70 [level]                   # a clap from an azimuth
at 2:05 talk -40 3 [level]                # synthetic speech: azimuth, seconds   (talk_ramp: getting louder)
at 2:10 bang 30 [level]                   # a door slam: low, ringing
at 2:12 hiss -50 0.6 [level]              # a burst of high-passed noise
at 2:15 tone 20 1200 1.5 [level]          # azimuth, Hz, seconds
at 2:20 whistle 0 800 1600 1.2 [level]    # a sweep
at 2:30 clip 60 /path/clip.raw 0.8 speech+clap+sharp+continuous+low+high+tonal   # a recorded sound (raw int16 24 kHz)
at 4:00 noise 400 0.9                     # background noise level, spectral tilt 0 white .. 1 low-pass (a fan)
at 5:00 lamp desk off          at 5:00 room_light on
at 6:00 person alex enter -40 1.5         # azimuth, distance (m); then walk AZ SECONDS, approach DIST SECONDS,
                                          # come_close, step_back, leave
at 6:30 thing ball enter 30 10 8 0.6      # a non-person object: azimuth, elevation, size (deg), reflectance;
at 6:31 thing ball move 60 10 2           # then move AZ EL SECONDS, leave
at 7:00 shake 2    at 7:10 pick_up 25    at 7:20 put_down    at 7:30 owner home    at 7:40 log any text
every 0:05 from 2:00 until 2:30 touch_head 300
```

### Scenarios

| Scenario | Length | Checks |
|---|---|---|
| `play_session` | 12 min | someone comes home: speech, touch, taps, claps, shaking, habituation to repeated touch, calming down after |
| `touch_response` | 9 min | isolated touches (each must get an answer), a burst of rapid touches (annoyed, grumbles), quiet, two more |
| `startle_and_orient` | 4 min | sharp sounds from each side and a voice: startle, look around for the source, calm down |
| `learn_petting` | 8 min | a fresh robot; a visitor talks to it and pets it 3 s later, eight times, then only talks: the link voice -> petting is learned (log) and petting is expected (`expect_reward` >= 0.4) |
| `learn_rough` | 7 min | a fresh robot; a clap and a shake 2 s later, six times, then claps alone: bang -> rough handling learned, `expect_threat` >= 0.4 at the next clap |
| `conversation` | 3.5 min | a quiet minute, then two minutes of talk (4 s speech, 4 s pause, from the side): calls a minute while people talk (>= 3.5) and the share of them in the pauses (>= 35 %) |
| `hearing` | 4 min | a fan, claps every 0.5 s, a door every 8 s (habituation), a clap from a new place, `talk_ramp`; checks startles and calls, not head turns |
| `vision_motion` | 6 min | a person walks across the view (the head follows), then straight at the robot (looming), then an object moved |
| `night_quiet` | 40 min | evening into night: falls asleep and stays still; a touch at night gets a calm reaction |
| `petting` | 6 min | a visitor strokes the head two or three times in a row, four times: no Angry / Annoyed face, no grumble |
| `day_alone` | 90 min | midday alone: looks around, stamina recovers, chatters and invites |
| `alone_dark_morning`, `bright_door`, `idle_owner_home`, `idle_endurance`, `full_day` | 15-60 min, 24 h | older long-run scenarios (a bright door must not capture the robot, owner presence is not an interaction, an hour idle keeps living, a whole day) |

### Fidelity

Estimated, not measured (constants in `world/world.cpp`): servo click and whine levels in the microphones,
speaker -> microphone coupling, camera field of view (60 deg), servo speed and response, lamp and window
brightness, IMU axes, microphone channel sides, the person (a grey block with a head), the sound spectra.
Real-world quality of the perception networks must be judged on the robot; the simulator's rooms are much
simpler than reality (that is why COCO photos, ESC-50 / LibriSpeech clips and recordings from the robot's own
microphone are in the training data). Not modelled: the servo watchdog, servo modes, inter-microphone time
differences, the face drawing (check the look with `tools/face_preview`).

### Layout

```text
simulator/
  main.cpp               CLI
  run.sh, build.sh       build (g++, no ESP-IDF) and run; sources used by the firmware are listed in build.sh
  build_datagen.sh       the perception data generator (datagen/datagen.cpp) and frames_feed
  core/vrtos.*, log.*    virtual RTOS and time, shared log in UART format
  shims/                 host versions of ESP-IDF headers (esp_timer, FreeRTOS, BSP, I2C, codec, camera, SCSCL, BMI270, LVGL)
  hal/platform.cpp       their implementations, backed by the world model
  world/world.*          room, light, people, things, sounds (incl. clips), mechanics, sensors, ground truth
  world/scenario.*       *.scn parser
  runtime/sim_runtime.*  mirror of RobotRuntime
  datagen/               datagen.cpp (Vision / Auditory training data with ground truth), frames_feed.cpp (real photos)
  scenarios/, out/       scenarios; results (not for version control)
```

## Tests

| Command | What it checks |
|---|---|
| `simulator/test.sh` | builds, runs 3 min twice: the runtime starts, both perception networks are present, no firmware errors, the robot babbles, no spiking-core reset / policy fault, the head moves, **two runs are bit-identical** |
| `simulator/behavior_test.sh` | `tools/hybrid/behavior_test.py`: tracks a visitor (centred >= 50 % of the time visible), answers >= 80 % of isolated touches within 2.5 s, grumbles at a burst of touches, startles at claps and turns, startles and answers with calls in `hearing`, sleeps at night (>= 25 % Sleepy), does not get angry at two or three strokes (`petting`), looks around alone, follows a moving person, learns what comes before petting / being shaken by itself (`learn_petting`, `learn_rough`), no errors |
| `tools/hybrid/parity.py`, `parity_snn.py`, `parity_policy.py` | C++ versus PyTorch on the perception nets (max diff ~1e-6), on the spiking core (mean diff ~0.0075) and on the policy (max diff ~3e-6 over a 3 minute run, on the inputs the policy really saw) |
| `tools/hybrid/evaluate.py KIND SCENARIO [MINUTES]` | metrics of one run (`builtin` = the compiled GRU, `teacher`, or `learner`) |

## Training the networks (`tools/hybrid`)

Order matters: each stage's inputs come from the stages before it. Everything runs on the PC GPU (RTX A3000, torch
2.6 + CUDA 12.4) with `tools/hybrid/py`.

| Stage | Data | Command | Output |
|---|---|---|---|
| 0. interfaces | - | `python3 spec.py` | `main/hybrid/hybrid_spec.h` |
| 1. Vision NN | simulator frames with ground truth (`gen_data.py vision 180 --minutes 5`, plus 130 empty-room scenarios from seed 3000), COCO val2017 photos (`make_coco_vision.py`), **this robot's camera** (`record_vision.py`, `data/vision/real_*.bin`, 8x, NaN-masked labels) | `./py train_vision.py --epochs 40 --qat 8` | `vision_model_data.h`, `data/vision_net.pt` |
| 2. Auditory NN | simulator sounds, ESC-50 / LibriSpeech clips (`prepare_clips.py`, `gen_data.py audio 160`), real recordings (`record_audio.py`) | `./py train_audio.py --epochs 50 --qat 8` | `auditory_model_data.h` |
| 3. Spiking core | synthetic days (`synth.py`), drive teacher (`teacher_drives.py`), optionally rollout buses (`--real`) | `./py train_snn.py --epochs 14 --qat 4 --seqs 320 --batch 64 --rebuild`; then `./py parity_snn.py` exports the header | `snn_model_data.h` |
| 4. Policy (51 outputs, incl. the 21 faces and the LED colour) | closed-loop rollouts, relabelled by `teacher_policy.py` | `./dagger.sh 2` (teacher rollouts, then 2 DAgger iterations, ~15 min) | `policy_model_data.h` |
| 5. Voice | `tools/voice`: prosody bank, droid teacher | see below | `main/voice/babble_model_data.h` |

After stages 1-3 run `./py parity.py` (and rebuild `simulator/build.sh`) before generating rollouts: the policy is
trained on the actual outputs of the stages before it.

- `rollout.py KIND N DIR`: N random scenarios (`rollout_scenario`: visitors, touches, claps, light, time of day)
  with the sim behind a pipe; `teacher` acts with `teacher_policy.label`, `learner` with the trained torch policy.
  Each episode writes `bus.bin`. `train_policy.py` relabels every logged input with the teacher and trains the GRU
  with truncated BPTT; quantization-aware in the last epochs. A DAgger iteration is ~3 min of rollouts + 1 min of
  training.
- `evaluate.py` and `behavior_test.py` measure what the policy does in closed loop; always do it after retraining.
- Randomness: the teacher draws none; the pacemaker noise is an input (see ARCHITECTURE).

### Datasets

Downloaded to `../datasets` (outside the repository), all public: COCO val2017 images and person keypoints (about 1
GB), ESC-50 (environmental sounds, ~600 MB), LibriSpeech dev-clean (speech, ~340 MB). `prepare_clips.py` converts
the sounds to the robot's 24 kHz raw files with class masks (`tools/hybrid/data/clips`); `make_coco_vision.py`
turns photos into the robot's view. The intermediate data (`tools/hybrid/data`, several GB) is regenerable.

### Recording the real microphone and camera

**Microphone.**
1. Set `kAudioRecording = true` in `main/core/debug_flags.h`; for sessions of impulsive or clean sounds (clap, sharp, silence, speech) also
   `kParkHead = true` (head parked, voice muted, so neither the servos nor the robot's own voice end up in the file); build and flash.
2. `tools/hybrid/py record_audio.py LABEL SECONDS` (labels: silence, speech, clap, sharp, continuous, tonal, mixed)
   reads the Auditory NN's input from the serial port in parts of <= 55 s and writes `data/audio/real_<label>_<n>.bin`.
   Everything captured in a session gets the label's class, so keep other sounds out of it.
   `record_audio.py` never overwrites a session (it takes the first free number).
   A convenient way to work with the owner: a small shell script that plays beeps on the PC (`play -n synth ...`: three short beeps and a
   long high one = start, two low = stop) and runs several sessions in a row.
3. `train_audio.py [--mix 0] [--rw 15]` counts real sessions 15x; `real_silence_7`, `real_speech_2`, `real_clap_4` and `real_sharp_3` are
   held out and the `real_mixed_*` sessions (no label) are only looked at; in clap / sharp sessions the frames after a jump of energy over
   the room's background keep the class label, both clap and sharp are set on them (see ARCHITECTURE); the report counts how many
   impulses of each held-out session are heard. The old clap / sharp sessions recorded with a free head (servo noise) were deleted;
   live recordings of normal operation are in `data/audio_live` (for looking).
4. Set both flags back to `false`, build, flash. The firmware's `hear:` lines (a rising edge of clap / sharp / speech above 0.5, with
   `own motion` when the head moves) show what the ear makes of a live sound.

**Camera.**
1. Set `kVisionRec` in `main/core/debug_flags.h`: `Free` (the robot behaves as usual and scans the room: for empty-room
   sessions) `Sweep` (the head sweeps yaw +-90 deg in 70 s and pitch -6..22 deg in 29 s: **the best way to record empty rooms from every direction**, the free head turns rarely), or `Still` (the head is parked facing forward: for a person standing in view), build and flash. The camera
   task prints the Vision NN's input tensor as `vframe:` lines (~1.3 / s).
2. `tools/hybrid/py record_vision.py empty|front|side SECONDS` writes `data/vision/real_<label>_<n>.bin`
   (`empty`: all labels 0; `front`: person and face 1; `side`: person 1; everything unknowable is NaN and masked).
3. `train_vision.py [--holdout real_x.bin,...]` counts real files 8x; by default nothing is held out (every recording trains), pass
   `--holdout` for an honest test and then train again without it. `vision_fp.py` prints person / face rates per real recording. If a
   person steps in and out of a session, `relabel_by_background.py FILE` labels each frame from the median background.
4. Set `kVisionRec` back to `Off`, build, flash. `VISION_PLAN.txt` is the plan for the next recordings and training.

`RECORDING_PLAN.txt` has the sound sessions (all recorded; claps and sharp sounds redone with a parked head on 2026-10-03) and the
video sessions V1-V9.
`kDatasetLogging` does the same for the whole policy input (`serial_dataset.py` converts it to a rollout file).

### The voice (`tools/voice`)

```bash
../vocal_decoder/.venv/bin/python prosody_bank.py     # real pitch contours from LibriSpeech -> data/prosody_bank.npz
../vocal_decoder/.venv/bin/python train_babble.py     # teacher phrases -> BabbleNet -> main/voice/babble_model_data.h
g++ -std=c++17 -O2 -o voice_preview voice_preview.cpp ../../main/voice/droid_synth.cpp
../vocal_decoder/.venv/bin/python render_teacher.py out.wav 2   # render the teacher's phrases for the 21 faces (note per line: base kind shape ending dur f0 slope loud fm harm vib rate gap)
../vocal_decoder/.venv/bin/python babble_sample.py out.wav 2    # sample the trained network for the 21 faces, render with the synthesizer
```

Demo: `tools/voice/droid_demo.wav` (the current droid voice sampled from the network, two phrases for each of the 21 faces). To hear what the robot said in any simulation: `--voice-wav`, then wrap the raw file as a
24 kHz mono WAV. Change the sound without retraining by editing the constants in `droid_synth.*`; change what is
said (sound kinds, glides, durations) in `droid_teacher.py` and retrain (a minute).

## Tools

| Tool | Use |
|---|---|
| `tools/hybrid_map.py` | draws `docs/BRAIN_MAP.svg` from the spec's numbers |
| `tools/networks_map.py` | draws `docs/NETWORKS.svg`: the five networks with their file sizes (read from the model headers and checkpoints) |
| `tools/capture_serial.py` | bounded serial capture (max 60 s, no DTR/RTS toggling) |
| `tools/serial_guard.sh` | lock and open-handle check used by `flash.sh` / `monitor.sh` |
| `tools/face_preview` | renders the 21 faces on the PC (`run.sh` -> `docs/faces.png`) |
| `tools/hybrid/*` | training, data generation, rollouts, evaluation, parity (above) |
| `tools/hybrid/audit.py [--minutes M] [--robot-log FILE]` | **behaviour auditor**: a simulated day in 7 two-hour stretches (night, morning with the owner, alone in the day, the owner at the PC ignoring the robot, evening play, TV evening, alone before dawn), a census of every behaviour (faces, invitations, startles, grumbles, voice styles, LED colours, sleep, learned links, head travel), rules checked on every tick (an invitation needs somebody, anger needs a reason, two or three strokes never anger it, touches answered, no songs, boredom and the need for company really grow, ...), and the spiking core against the drive teacher in the closed loop; `--robot-log` audits a serial capture of the real robot. Report `simulator/out/audit/report.txt`. Every bug the owner finds becomes a rule here |
| `tools/hybrid/pick_best.sh FIRST LAST` | trains DAgger candidates with different rollout seeds, runs the audit and the behaviour regression on each, installs the best (behaviour-test failures count 100x) |
| `tools/hybrid/reward.py`, `train_rl.py`, `rl.sh [iterations] [episodes] [first]` | reinforcement stage of the policy (experimental, see History): the reward, PPO on explore rollouts |
| `tools/hybrid/autonomy_report.py [scenario] [minutes]` | one hour alone in the sim: faces, entropy, calls, glances, invites, drives (is the robot independent?) |
| `tools/hybrid/record_audio.py`, `record_vision.py` | record this robot's microphone / camera in labelled sessions (above) |
| `tools/voice/*` | droid teacher, voice trainer, WAV renderers (above) |
| `tools/vocal_decoder/.venv` | the shared Python environment (torch + CUDA) used by `tools/hybrid/py`; the folder name is historical |

## History

- **v1 (2026-09-16..26)**: an SNN with ~100 semantic populations, reaction arbiter, behaviour orchestrator and a
  library of canned sounds.
- **v2 NeuroBrain (2026-09-26..30)**: one recurrent spiking network (grew from 4860 to 6780 LIF cells) with
  encoders and decoders, hand-tuned; a retina with direction cells; a vocal decoder (trained GRU) + DSP vocal
  tract. Lessons that survived: servos in PWM mode stall; the servo bus has no feedback; stalls scale with the
  packet rate; the servos are loud the whole turn; sound direction from 2 mics 5 cm apart is useless; the robot is
  deaf to quiet sounds while it turns; DC gain of slow cells; endogenous activity.
- **v3 hybrid brain (2026-10-01)**: from the "Hybrid Neuro-AI Architecture" concept: Vision NN, Auditory NN, a
  768-neuron SNN, a GRU policy, the babble voice. Built, trained and flashed in a day (the numbers are in
  ARCHITECTURE). First live test the same evening revealed the problems below.
- **v3 refinements (2026-10-02)**: droid voice (first a Klatt speech synthesizer, then an analogue-style one with 10 kinds of
  sound and a voice for every face), song governor, real recordings of the microphone (13 sessions) and the camera,
  the camera-frame bug (signed age), person hold and screen symbol, the policy chooses the face (49 outputs), shake / tilt
  felt for 2 s / 1.5 s, simulator shake / lift events. The old projects were archived and removed from the workspace.

- **hearing rebuilt (2026-10-03)**: the owner said the robot ignores sounds. Cause: the "clap" sessions were servo noise (head free). Re-recorded
  with `kParkHead` (clap x6, sharp x3, silence), impulses found by an energy jump over the background, clap + sharp merged into one
  "bang", the policy teacher answers every bang with a call (DAgger x2), the executor lets bangs use the call reserve, `hear:` log lines,
  `hearing` test now counts startles and calls. Next: vision (false persons in an empty room).

- **livelier idle (2026-10-03, evening)**: the owner found the robot passive (few sounds, a still head). Measured: 145 calls and 200 head
  moves an hour on the robot, 108 / 80 in the quiet simulator after the retraining. Cause: head exploring needs `explore > 0.25` and the
  calm, not-bored robot sat at 0.02; idle babble was throttled by the same calm state. Teacher: +0.22 restlessness in `explore_p`, lower
  babble threshold; DAgger x2 -> simulator idle 205 calls / 229 glances an hour (and the robot, 8 calls and 5 moves in a minute). Note that
  voices in the house count as stimulation, so a robot in a lived-in house is bored less and needs this push.

Lessons of v3 (kept in ARCHITECTURE as tuning lessons): a network never says zero; real microphones differ from
simulated ones (domain gap: per-band baseline, level gating, real recordings); the vision net hallucinated persons
in dark frames; the robot's own servos look like claps (and once filled the "clap" recordings themselves); annoyance needs evidence; attention that wears out cannot
follow a face; a creeping head never sleeps; voice rendering belongs in its own task; a 63 Hz amplitude modulation
sounds hoarse; people judge a voice by ear, so keep demos and knobs close at hand.
- **Code review and cleanup (2026-10-03)**: the whole firmware read through. Fixed: the rear LEDs flickered more the longer the robot
  ran (breathing phase was `fmod(uptime, period)` with a period that changes every tick; now integrated, and the LEDs are one I2C
  burst at 25 Hz, skipped when unchanged, instead of 14 transfers at 50 Hz); the voice task could clear `BabbleVoice::active()` just as a
  phrase was queued (the robot then heard its own voice as the world); `SelfSoundAnalyzer` was written by the voice task and read by the
  main loop without a lock; the stats line's `active` subtracted the voice task's DMA wait (read ~0 % while talking, and it feeds the
  heart rate); `holdPerson` re-anchored a remembered person between frames, so the memory walked with the head; the owner's arrival
  lasted 3 ticks (+0.9 social need) instead of the one-tick pulse of training; a NaN in the Auditory NN now reads as silence. Old
  archives, logs, rejected weights and redundant dataset archives removed; one backup of the current state is kept (README, Rollback).
- **Second review (2026-10-03)**: dead code removed with no change in behaviour (simulator bit-identical, teacher labels identical):
  the microphone's zero-crossing / direction / interaural-lag code, the servo bus probe and its UART helpers, unused NN kernels,
  physiology and IMU fields, the speaker wait counter. `servo_failed` / `status` left the stats line; `imu` / `gyro_max` came in.
  New rule: sound heard during own motion does not make the executor "follow" (live log: servo whine read as speech 0.5-0.97).
  A tighter servo watchdog was tried and reverted (the gyro misses moves, see HARDWARE).
- **Learning on the robot (2026-10-03)**: the associative memory (`hybrid/assoc_memory.*`, ARCHITECTURE) - 11 cue groups of
  integrate-and-fire cells, two outcome groups (petting, rough handling), plastic synapses with a three-factor rule (onset of
  the outcome x eligibility x prediction error; extinction per cue episode), kept in NVS. The bus lost the sound direction
  (two microphones 5 cm apart) and the sound prediction error (always 0) and gained `expect_reward` / `expect_threat`
  (policy input still 100, spiking core input 258). The Auditory NN has 8 outputs. `led_activity` now drives the rear LEDs
  (brightness, a running wave). Retrained: Auditory NN, spiking core, policy (teacher: looks around for a sound's source,
  anticipation and wariness, LED activity; rollouts with experience and visitors' habits).
  First attempts of the learning rule, kept as lessons: the robot's own squeak during a shake took the credit for the shake
  (now only the onset teaches, 3 s quiet after it, the own call is no cue); extinction in the gap between a voice and the hand
  that followed a second later erased the voice's link (now per episode, after the trace has faded).
- **2026-10-04, owner's live test**: the robot sang in bouts after 21:25 (the dusk song window of the song governor, the
  droid voice in its singing mode) - songs are gone: the governor, the sway and the dancing face were removed, the executor
  dropped the policy's wish to sing (later: the policy has no sing output). Two or three touches made it angry. Tests:
  `petting` (no anger); `dawn_song`, `dusk_hush`, `dusk_song`,
  `night_quiet_song` removed.
- **2026-10-04, decisions moved into the networks**: the drive teacher lets touches annoy only in a run, the policy teacher
  never sings and picks the LED colour (`led_r/g/b`); the spiking core and the policy were retrained, and the executor's
  hand-made rules went: the song veto, the face hysteresis and the 15 s face hold (the policy is trained against flicker of its
  face, gaze and colour). The tighter anger limit of the morning was reverted to the old safety net (`petting` passes with it).
  Tried and not kept: training the head target against jitter (the target then glided and the head crept 13000 deg in 20
  min) and dropping the 8 s pacing of idle glances (400 glances an hour): the head moves through a saccade generator with
  that pacing. Touch: a held finger came out as a touch every 0.9 s (now one press until 0.4 s without contact), and real taps
  of 180-250 ms were dropped as servo phantoms while the head moved (a level-3 contact now counts from 120 ms).
- **2026-10-04, reinforcement stage (experiment, not deployed)**: `tools/hybrid/reward.py` (what moving, talking, being
  petted and annoyed are worth), `rollout.py explore` (block-held Gaussian noise on yaw, pitch, hold, sound_act),
  `train_rl.py` (PPO, float weights, value net, imitation anchor) and `rl.sh`. The reward is right: 15 idle minutes score
  +0.6 with the executor's 8 s glance pacing and -36.5 without it. 18 iterations x 64 episodes raised the reward of the
  noisy rollouts from -2 to about +17, but the deterministic policy learned to resist its own exploration noise, not the
  idle rhythm (524 glances and 392 calls an hour alone, reward -40). So the robot keeps the imitation policy with the 8 s
  pacing and the call budget. Lessons: train in float (fake INT8 made tiny updates jump), one optimizer step per minibatch
  of whole episodes (many small steps inside an epoch clipped a third of the ratios). Next: more quiet episodes in the
  rollouts, exploration on the timing outputs only (hold, sound_act) with longer blocks, many more iterations.
  Checkpoint: `data/policy_net_rl_experiment.pt`.
- **2026-10-04 (night), the rhythm moved into the network**: two longer PPO runs (40 iterations planned, timing-only
  exploration, half quiet episodes, best-checkpoint evaluation) did not beat the hand rule either (scores 12.5 / -13 / 10.9
  / -3.1 against 16.6 at the start) and were stopped. What worked: a **teacher with a memory** (`teacher_policy.Memory`,
  labelling the whole episode): a small glance while nobody is followed waits 8 s after the head stopped. DAgger taught the
  GRU that rhythm; the executor's 8 s rule is gone (idle hour: 267 glances, 222 calls; reward of 15 idle minutes +2.7, the
  hand rule +0.6). The call budget was tried the same way and failed (the call is a sparse pulse; labels that drop some of
  the pulses made the robot silent, 0 calls in a conversation), so it stays in the executor.
- **2026-10-04 (evening), invitations and the auditor**: the owner never got an invitation to play. Two bugs: the teacher
  invited only when NOBODY was in view (it nodded at an empty room), and the slow needs were drained by the spiking core's
  sleep readout, which idles at 0.01-0.04 in the day (`HybridBrain::updateSlow` took 0.9 x sleep per second: the need for
  company stuck at 0.3). Now: sleep counts only above 0.2 (dead zone); somebody merely in view barely satisfies the need for
  company (touch and talk do); the invitation goes to somebody around who does not play with the robot. Test `ignored`.
  Why such bugs slipped through: the tests checked scripted situations written for each behaviour, not whether each behaviour
  happens in the situations of a real day; the sleep bug lived between two modules that are right on their own. Hence
  `audit.py`. Its first full day found: invitations to an empty room still happen now and then (the GRU learned the new gate
  imperfectly), the head stares at a person who sits still for hours (2-3 glances an hour - the teacher never looks away),
  the spiking core drifts from its teacher in the closed loop (too little sleep at night, touch habituation too high alone),
  and six faces never appeared in the day (Crying, Love, Shy, Confused, Cool, Sick).
- **2026-10-04 (late), fixing what the audit found**: the spiking core fine-tuned on closed-loop rollout buses (`train_snn.py
  --real` had never worked: it skipped every 9514-tick episode as shorter than 9600, and its validation crashed); the teacher
  glances away now and then from a person who just sits there (6 s pacemaker, 40 deg aside of where they sit); faces redesigned
  so each has a situation of the day (Love while stroked, Shy for 2 s after a screen tap - the 3-tick tap was shorter than the
  4 ticks a face needs, Cool content after play, Confused when a sound has no source, Crying after a long loneliness, Angry
  when touched again while cross; rare faces weigh more in training); a quarter of the rollouts have somebody sitting there,
  a quarter are quiet, some have an empty battery. Two real bugs the audit led to: the IMU's tilt latched for good after the
  robot was lifted and put back (the baseline was re-taken only after its own head moved, and the startled robot held still:
  37 % of an evening 'tilted', startled, touches unanswered) - now 3 s of rest re-take it; and a touch went unanswered when
  the network's answer came a tick after the touch (the reaction window was one tick; now 0.5 s, and the voice keeps the
  reaction open until it is answered). Loom / recede are not passed on while the head turns (143 startles in a TV evening
  were the robot's own glance back to the person). DAgger is noisy: `pick_best.sh` trains several candidates, audits and
  regression-tests each and keeps the best; the robot runs candidate c3: all behaviour tests pass, all 21 faces in the day,
  4 audit findings left (few glances away at the PC, 9 face changes a minute there, 16700 deg/h with an empty battery,
  no Angry in that day).

## Open ideas

- Claps while the head moves (the servo noise is as loud as the clap): more data of that situation, or a quieter way of moving.
- Vision: more varied real sessions and the plan in `VISION_PLAN.txt`.
- Fine-tune the spiking core on the logged rollouts and real robot buses (`train_snn.py --real`), then redo DAgger.
- Reward-based refinement of the policy (tools exist: `reward.py`, `train_rl.py`, `rl.sh`; two attempts failed, see History):
  a reward that sees the benefit of waiting (fewer moves in a quiet room is a small signal under everything else), or a
  recurrent value net; the call budget (still in the executor) is the next thing to move into the network.
- Associative memory: more outcomes (being talked to, a person coming, being left alone), cues from the vision features of a
  particular person, a save to NVS right after a link is learned (now at most every 10 min), and better hearing while the robot
  itself talks or moves (half of the claps are missed then, so it learns from fewer of them).
- int8 convolutions with ESP-NN for the Vision NN; a smaller spiking-core input projection.
- Rhythm sense (the old hand-built beat tracker) as a new bus feature, nods on the beat.
- Real faces: a COCO-keypoint-trained head at higher resolution, or ESP-DL face detection.
- Voice: a reward for engagement; more kinds of sound; a quiet-hours schedule learned from the owner.
