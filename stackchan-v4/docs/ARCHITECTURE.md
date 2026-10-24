# Architecture of brain v4

The owner's idea (2026-10-24, [BRAIN_V4_PLAN.txt](../BRAIN_V4_PLAN.txt)): an animal has a fast nervous system that
evaluates every stimulus at once, and a brain that decides what to do about it. Brain v4 copies that split:

| Part | Where | Rate | Job |
|---|---|---|---|
| Channel encoder | `main/v4/channel_encoder.*` | 20 Hz | every sense -> one number 0..1 (13 channels) + "where" (2) |
| Spiking analyzer (SNN) | `main/v4/snn_v4.*` | 20 Hz | instant, innate evaluation: 32 numbers (feeling, salience, needs, attention) |
| Main network (NN) | `main/v4/nn_v4.*` | 10 Hz | the one decider: an emotion and every action |
| Executor | `main/v4/v4_executor.*` | 20 Hz | the "muscles": how to show it, inside safety rules |
| Brain loop | `main/core/brain_loop.*` | 50 Hz pass, 20 Hz tick | wires the engines to all of the above; the same code runs in the simulator |

![Brain v4](png/BRAIN_V4_MAP.png)

One file defines every name and size on both sides: `tools/v4/spec_v4.py` (channels, zones, SNN outputs, the main
network's inputs and outputs, emotions, LED colours and patterns, head modes, the diary columns). `./py spec_v4.py`
writes `main/v4/v4_spec.h`.

## 1. Channels and zones

Each channel is a number 0..1. Its scale is divided into innate **zones**: boredom, comfort, interest, anxiety, anger
(not every channel has all five; zones may overlap, borders are soft over ±0.03). The table is the owner's
([BRAIN_V4_INPUT_SCALES.txt](../BRAIN_V4_INPUT_SCALES.txt)) and may later shift by up to ±0.15 with experience.

| # | Channel | Made from | Zones (low -> high) |
|---|---|---|---|
| 1 | loud | the ear's loudness over the room's floor, 10 s average x 1.6 | boredom, comfort, interest, anxiety, anger |
| 2 | sharp | loud peaks (claps, bangs, knocks), each weighted by strength, 15 s memory, 3 = full | interest, anxiety, anger |
| 3 | voice | share of speech over 60 s (Auditory NN speech > 0.4) | boredom, comfort, interest, anxiety |
| 4 | touch_head | strokes on the head pad (weight by pad level), holding counts, 30 s memory | boredom, comfort, interest, anxiety, anger |
| 5 | touch_screen | taps on the face, 30 s memory | interest, comfort, anxiety, anger |
| 6 | shake | spread of \|acceleration\| over 0.5 s (0.03 .. 0.8 g), decays over 3 s | interest, anxiety, anger |
| 7 | tilt | lean from the learned upright (the head's own pitch excluded) | comfort, interest, anxiety |
| 8 | near | IR proximity over its baseline (600 counts = full) | boredom, comfort, interest, anxiety |
| 9 | light | ambient light, log scale (20000 counts = full) | comfort, interest, anger |
| 10 | motion | share of the picture that moves (head's own turn removed), 5 s average | boredom, interest, anxiety, anger |
| 11 | hunger | 1 - battery | comfort, anxiety |
| 12 | tired | 0.75 x the night ramp (21:00-21:45 up, 06:00-06:45 down) + 0.25 x (1 - servo stamina) | comfort |
| 13 | lonely | 1 - exp(-t / 40 min) since the last contact (touch, speech, near; motion caps it at 5 min) | comfort, boredom |
| - | motion_x, motion_y | where the motion is (-1..1) | for attention |

Slow clocks (minutes, hours) are computed in code, not by the SNN: a spiking network cannot count minutes (v3's lesson,
2026-10-03). `tools/v4/channels.py` is the same encoder for training; a parity test keeps them identical (error 0).

## 2. The spiking analyzer (SNN)

768 leaky integrate-and-fire neurons in four layers, two substeps of 25 ms per tick, event-driven C++ with int8 weights
(per-row scales, 236 KB, copied to PSRAM at boot): 3-5 ms per tick on the ESP32-S3.

| Layer | Neurons | What it does |
|---|---|---|
| ENC | 130 | per channel 8 Gaussian receptive fields over 0..1 + an ON and an OFF cell (a value that jumps fires at once) |
| ZONE | 260 | 13 channels x 5 zones x 4 neurons, fixed innate wiring from the zone table; the 4 neurons of a zone tire with time constants 1 / 8 / 30 / 90 s (habituation: a steady stimulus fades, a new one stands out) |
| MIX | 282 | recurrent pool, 80 % excitatory / 20 % inhibitory (Dale's law), 12 % connectivity: context across channels |
| ATT | 96 | 12 x 8 attention map: a bump where motion is, weighted by salience |

A readout of 32 leaky integrators (0.15 .. 3 s) reads all 768 neurons:

- **core affect (6):** valence, arousal, under-stimulation, over-stimulation, threat, surprise
- **salience (10)** and **hedonic value (10)** of each outer channel (1-10)
- **drives (3):** hunger, sleepiness, loneliness
- **attention (3):** x, y, strength

It is trained to embody an innate appraisal (`tools/v4/teacher_snn.py`, built from the zone table: zone valence and
arousal, salience per zone, jumps as news, habituation) with surrogate gradients and truncated backpropagation through
time; the last 30 % of the epochs run with int8 fake quantisation, as on the robot. The SNN only gets numbers and knows
which channel each comes from; it does not choose emotions.

## 3. The main network (NN)

One network decides everything (the owner's choice: one decider, the voice is only a synthesizer):

```
input 44 = 32 SNN numbers + 12 own state -> GRU 64 -> emotion head: 21 emotions (softmax) + strength
                                                    -> [GRU state, emotion] -> hidden 64 -> action heads (39)
```

- **Own state (12):** last valence and arousal, how long the face has been shown, time since a sound and since a
  gesture, head yaw and pitch, LED level, whether the last touch was answered, and three pacemakers (slow, fast,
  jitter: random processes the network can see, so its "spontaneous" choices are learnable and reproducible).
- **Outputs (61):** 21 emotions + strength; voice 10 (say, 4 styles: calm / ask / play / grumble, arousal, valence,
  register, length, rate); LEDs 15 (8 colours, 5 patterns: steady / breathe / pulse / flicker / wave, brightness,
  rate); head 14 (9 modes: hold, look, scan, nod, shake, flinch, look down, look up, doze; gaze yaw / pitch; gesture
  amplitude, speed, repeats).
- Float, ~31 k parameters, runs every second tick (10 Hz), the first 5 s after a boot only settle the SNN.
- Trained by imitation of a behaviour teacher with memory (`tools/v4/teacher_nn.py`) on the SNN's outputs, then two
  rounds of DAgger (the network acts, its choices feed back into its own inputs, the teacher labels what it really
  visits). Emotion agreement with the teacher when acting alone: 0.90 (real days 0.92).

## 4. The executor

The network decides **what**; `V4Executor` decides only **how**, inside fixed rules:

| Output | How it is shown | Rules |
|---|---|---|
| face | the chosen emotion (21 faces, smooth morph); eyes follow the SNN's attention | a choice must last 2 ticks; a shown face stays >= 1.5 s |
| voice | the droid babble (BabbleNet + synth) with the network's style and mood | call budget 5 tokens, 1 per 10 s (a reaction may spend the last one); **silent 21:45-06:00** and for the first 2 min of a boot without clock |
| LEDs | colour, pattern, brightness, rate | brightness <= 6 % at night |
| head | look / scan to the gaze, look up / down, doze; gestures on the network's wish: nod 5-12°, shake 5-14°, 1-3 repeats; flinch +12° | nod / shake at most every 4 s and never at night; flinch every 1.5 s at most; <= 8 servo commands/s; yaw ±100°, pitch -20..35° |

What really happened (shown face, said, gestured, head pose, LED level) goes back into the network's own state.

## 5. The brain loop and its filters

`BrainLoop` (shared by `RobotRuntime` and the simulator's `SimRuntime`) runs every 20 ms pass and ticks the brain every
50 ms. Hand-written filters that keep the senses honest (each found on the real robot, see HISTORY):

- **Own motion:** the servos click up to 32k into the microphone: loud / onset events are dropped while the head moves
  and 350 ms after; camera frames taken while the head turns are not used; the servos' own share is taken off the
  shake channel; the IMU is ignored for 3 s after a servo power cycle.
- **Own voice:** the microphone is muted while the voice plays; the ear is told when the robot itself speaks.
- **Head pad phantoms:** a touch during a fast move counts only if it is held (level 3 >= 120 ms, level 2 >= 250 ms);
  more than 8 touches in 10 minutes with nobody heard, moving or near in those 10 minutes are pad noise.
- **Sharp sounds:** only loud peaks from ~6000 count (weak taps and words do not); the boot click (first 10 s) is not
  a bang.

## 6. The diary

Every tick goes to the home PC over UDP (`main/net/life_log.*`, kind 3 datagrams, four ticks each, float16), with
every log line. Columns (`spec_v4.DIARY`, 147): the 22 raw readings the channels are made from, the 15 channel values
(after the zone shift: what the analyzer saw), the 32 SNN numbers, the 61 network outputs, what the body did (head
yaw / pitch, shown face, said, gestured) and the main network's 12 own-state inputs of its last decision. With the raw
readings any day can be re-encoded after a change of the channels; with the own state every decision can be replayed
exactly on the PC (stage 6). Flags: bit 0 brain v4 drives the body, bit 1 night rules, bit 2 the main network decided
this tick. `tools/robot/lifelogd.py` stores it, `tools/v4/diary.py` reads it (also the first 135-column layout).

Every 5 s the robot logs three lines: `stats:` (CPU, memory, timing, servos, sensors, voice, diary counters), `v4:`
(raw light / proximity, the 13 channels, the SNN's mood, the network's state, face, calls, gestures) and `power:`
(rails of the power management chip).

## 7. What the robot learns from its life (stage 6)

Learning happens on the PC at night; the robot only receives the result, without a reboot:

- **The learned blob** (`main/v4/learned_blob.h`, made by `tools/v4/learned.py`): new weights of the main network's
  action heads (voice, LEDs, head: `act1`, `act2`, 8039 floats) and one zone shift per channel. The GRU, the emotion
  head and the spiking analyzer stay innate.
- **Zone shift** (`main/v4/zone_shift.h`): per channel a shift s of its alarm border a (the lower border of its first
  anxiety or anger zone), |s| <= 0.15, applied as a monotone warp of the channel value before the analyzer (a + s maps
  onto a; 0 and 1 stay), so the analyzer's innate wiring stays as trained. s > 0 is habituation (an always-humming
  fridge stops being alarming), s < 0 sensitization.
- **In flash** (`main/net/learned_store.*`): the partitions `weights_a` / `weights_b`, NVS namespace `learned`. A new
  blob (`PUT /learned`, `tools/robot/ota.py learned FILE`) goes into the idle slot, the main loop switches to it between
  two ticks, watches it for a minute and confirms it; if the main network fails on it, the robot goes back to the
  previous blob, also without a reboot. No blob: the innate (compiled-in) networks.
- **Clean start** (2026-10-24): at boot the firmware erases what the earlier brains left in flash (NVS namespaces
  `life`, `assoc`, `weights` and platform B's weights blobs); brain v4's own settings live in NVS `v4` (head pad
  sensitivity) and `learned`.

## 8. Tasks and timing

| Task | Core | Priority | Work |
|---|---|---|---|
| main loop | 0 | 1 | engines, brain tick (ear 1.2 ms, SNN 3-5 ms, NN every 2nd tick), executor, diary copy |
| voice | 1 | 5 | renders 20 ms chunks of the droid voice |
| microphone | 1 | 3 | 16-band filter bank, loud / onset detectors |
| camera | 1 | 2 | ~7 fps, motion front end (~3 ms a frame) |
| diary sender | 0 | 1 | UDP to the PC from a PSRAM ring |
| update server | 0 | 2 | HTTP: status, touch sensitivity, learned blob, firmware |

Measured 2026-10-24: main loop busy ~37 %, internal RAM free ~35 KB (minimum 21 KB), PSRAM free ~7.9 MB.
