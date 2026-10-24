# History

How the StackChan brain got to v4, what each step taught, and where the old code and data live. The day-by-day
record is in `JOURNAL.txt` (v4) and `../stackchan-life/JOURNAL.txt` (platform B), both in Russian.

## Timeline

| When | Version | Folder | What it was |
|---|---|---|---|
| 2026-09-16 .. 26 | v1 | (deleted) | an SNN of ~100 semantic populations, a reaction arbiter, a behaviour orchestrator, canned sounds |
| 2026-09-26 .. 30 | v2 "NeuroBrain" | (deleted) | one hand-tuned recurrent spiking network (4860 -> 6780 LIF cells), a retina with direction cells, a vocal decoder GRU + DSP vocal tract |
| 2026-10-01 .. 10 | v3 hybrid brain | `stackchan-neuromorphic/` (git tag `v3-neuromorphic`) | Vision NN + Auditory NN + a 768-LIF spiking core + a GRU policy (51 actions) + the droid voice; associative memory learning on the robot |
| 2026-10-11 .. 24 | platform B | `../stackchan-life/` | "a robot that learns from its own life": the UDP diary, OTA, a world model, five needs as reward, an AWR actor, online plasticity, a nightly sleep on the PC retraining world model, actor and perception |
| 2026-10-24 .. | **brain v4** | `stackchan-v4/` (this folder) | 13 zoned sense channels -> spiking analyzer (instant evaluation) -> one main network deciding the emotion and every action |

## v1 and v2 (September 2026)

Spiking networks wired by hand. They worked as demos but every behaviour was a tuning job, and the cells could not
hold slow states. Lessons that outlived them (all still true on this hardware):

- The servos stall in PWM mode (angle limits 0/0): position mode must be set after every power-on.
- The servo bus has no feedback (the RX line never toggles): only the gyro tells whether the head moved.
- Servo stalls scale with the packet rate: 10 packets/s, never more.
- The servos are loud during the whole turn (13k-32k peaks), not only at start and stop: the robot is nearly deaf
  while it turns.
- Two microphones 5 cm apart give no sound direction.

## v3: the hybrid brain (2026-10-01 .. 10)

Built in a day from the owner's "Hybrid Neuro-AI Architecture" concept, then refined for ten days with live tests:

- **Hearing.** Real microphones differ from simulated ones (per-band baseline, level gating, recordings of the robot's
  own microphone). The first "clap" recordings were the robot's own servo noise (head free): re-recorded with a parked
  head, clap + sharp merged into one "bang".
- **Vision.** The camera net saw persons in dark frames and on a still chair; a frame-age bug (unsigned wrap) dropped
  two thirds of the frames on the robot only.
- **Voice.** A droid (R2-D2-like) babble composed by a small network, rendered in its own task (writes from the main
  loop stuttered); loudness and character judged by the owner's ear.
- **Decisions into networks.** The owner wants the networks to decide, not hand rules. Songs, face holds and vetoes
  moved into the teachers; a reinforcement-learning stage (PPO) was tried and did not beat imitation + DAgger.
- **Learning on the robot.** An associative memory (three-factor plastic synapses) learned "voice -> petting" and
  "clap -> rough handling" during the day.
- **Slow states.** A spiking network cannot count minutes: slow needs became clocks in code (kept in v4).

## Platform B (2026-10-11 .. 24)

A copy of v3 in `../stackchan-life/`, built in stages with gates:

- **Stage 0:** the diary - every brain tick and every log line over UDP to the home PC (`lifelogd`), ~1 GB a day;
  updates over Wi-Fi with bootloader rollback; a weights blob switched without a reboot.
- **Stages 1-5:** a GRU world model on the robot whose surprise is curiosity; five needs giving a reward; a data
  collector (motor babbling) for 3 days; an AWR actor trained on the diary deciding moves and calls ("mode life");
  online actor-critic plasticity by day; a nightly sleep at 03:30 retraining the world model, the critic, the actor
  and, from 2026-10-17, the eyes and ears on the robot's own frames and sounds labelled by large teacher networks
  (YOLOv8, YuNet, Silero VAD, PANNs) on the PC.
- **What it taught:** internal RAM is the scarce resource (the update server's stack once killed a boot); the servo
  hangs and black-screen power-offs are hardware (a hung servo drains the supply: the servo rail is now cut on a
  VBUS sag); a microSD card on the screen's SPI bus crashes the robot; the robot lives in a dark room most of the
  time; the perception nets' nightly updates rarely passed their gates (forgetting the owner's real speech).

## The move to Omarchy (2026-10-18)

The PC moved to Omarchy (Arch Linux). The checklist is `../backups/MIGRATION.txt`. What matters later:

- The project lives on the external drive `/run/media/warmond/KIOXIA/Arhiv/For Lenovo Laptop/Codex/StackChan`
  (fstab entry), linked as `~/Codex/StackChan`; systemd user units use that path.
- The PC must keep the address 192.168.8.110: the diary host is compiled into the firmware (`secrets/`). The robot
  is 192.168.8.129. UDP 47000 is open in the firewall for 192.168.8.0/24.
- The training venv is `~/.venvs/stackchan` (Python 3.13 via mise, torch 2.6.0+cu124), set as `$STACKCHAN_PY` in
  `~/.config/environment.d/stackchan.conf` and `~/.bashrc`. Serial access on Arch: group `uucp`.
- The drive shows every file as executable: git runs with `core.fileMode=false`. Symlinks work (ntfs3).
- `loginctl enable-linger warmond` keeps the user services running without a login.

## Brain v4 (2026-10-24)

The owner described a new structure in the morning and approved the plan
([BRAIN_V4_PLAN.txt](../BRAIN_V4_PLAN.txt)): an SNN as the instant analyzer of stimuli, as in an animal, and one NN
deciding the emotion and all actions (face, voice, LED colour and blinking, head, with mandatory "yes" nods and "no"
shakes). Decisions: the head cannot be turned by hand (no such channel); the inner channels hunger, tiredness and
loneliness stay; the SNN gets only numbers; zones may shift with experience later; training on synthetic lives
first, then real ones.

The same day: the scales and zones, the channel encoder (C++ and Python twins), a calibration on 7 real days, the
synthetic-life generator, two teachers, the SNN (correlation 0.96-0.99 with its teacher) and the main network
(agreement 0.90), the executor, the simulator mode, behaviour tests, OTA flash. The owner's live check (petting,
shaking, the light) found real-world problems, each fixed in the senses, not in the networks:

- weak knocks and words counted as sharp sounds (only loud peaks from ~6000 count now);
- the boot's own click (peak 32768 at 4.4 s) frightened the robot for 20 s (masked for the first 10 s);
- the servo rail's power cycle jolted the head and read as a shake (IMU ignored for 3 s);
- the head's own pitch tilted the IMU and read as a tilt (side lean + roll band);
- the light sensor read 0 (burst read from 0x88 and gain 8x fixed it); the proximity sensor stays weak.

Then the cleanup: the robot runs only brain v4. Removed from the firmware: the v3 spiking core and policy, the vision
net, the world model, the needs, the associative memory, platform B's actor, collector and online learner, the
weights blob, the media endpoints, the SD diary. Main loop busy 45 % -> 37 %, internal RAM minimum 14 -> 21 KB, the
image 2.57 -> 1.96 MB, the diary 0.75 -> ~0.3 KB a tick (float16, raw readings included). Removed from the folder:
platform B's and v3's tools, docs and notes. The old diary days were gzipped (9.95 GB -> 2.23 GB). Platform B's PC
services (`mediad`, `life-sleep`, `life-morning`) were disabled.

## Where the old things are

| What | Where |
|---|---|
| v3 sources, docs, tools (the `stackchan-neuromorphic/` folder) | git tag `v3-neuromorphic` on GitHub; `../backups/stackchan-neuromorphic-v3-2026-10-04.tgz` + `.bin` |
| platform B (firmware with modes brain / collector / life, PC tools, the nightly sleep) | `../stackchan-life/` (untouched apart from reading gzipped days) |
| training of the Auditory NN and of the voice (BabbleNet) | `../stackchan-life/tools/hybrid/train_audio.py`, `tools/voice/` |
| the owner's real recordings (camera frames, microphone sessions, sound clips) | `../stackchan-life/tools/hybrid/data/{vision,audio,clips}` (private, not in git) |
| platform B's nightly models and the teacher networks | `../stackchan-life/data/sleep/`, `data/teachers/` |
| the diary of every day since 2026-10-11 | `../stackchan-life/data/life/` (`stackchan-v4/data` points there) |
| public datasets (COCO val2017, ESC-50, LibriSpeech dev-clean) | `../datasets/` (not in git) |

## Lessons for whoever continues

- Test every sense on the real robot before blaming a network: in v4 every live problem was a sensor or a filter.
- The robot hears and feels itself: servos, voice, power cycles. Every channel needs an "own action" mask.
- Keep the C++ and Python twins identical (parity tests) and log the raw readings: then any day can be re-encoded.
- A spiking network evaluates; it does not count time. Slow clocks stay in code.
- Never flash or reboot at night: the owner sleeps next to the robot (21:45-06:00).
- Watch `internal_min` after every firmware change; internal RAM runs out before PSRAM does.
