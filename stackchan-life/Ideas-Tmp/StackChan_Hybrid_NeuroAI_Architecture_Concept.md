# StackChan Hybrid Neuro-AI Architecture
## New Concept for an Autonomous Embodied Droid

**Target platform:** M5Stack StackChan / CoreS3 / ESP32-S3  
**Primary stack:** C/C++, ESP-IDF, FreeRTOS  
**Goal:** Build a fully autonomous robot in which perception, internal state, learned decision-making, movement, facial expression, and synthetic droid vocalization are handled by specialized subsystems connected in one closed sensorimotor loop.

---

## 1. Core Concept

Do not use one monolithic neural network for everything.

Use a hybrid architecture:

1. **Vision NN** for camera perception.
2. **Auditory Perception NN** for microphone perception.
3. **Compact SNN** for attention, novelty, habituation, internal drives, arousal, boredom, curiosity, and spontaneous activity.
4. **Small recurrent Policy NN**, preferably a GRU, for action selection over time.
5. **Existing trained Sound Generation NN** for generating droid vocalizations from the robot's state.
6. **Deterministic C/C++ safety and hardware layers** for servo limits, timing, drivers, watchdogs, and power management.

```text
                    WORLD
                      │
        ┌─────────────┼──────────────┐
        │             │              │
      CAMERA        MICROPHONES    TOUCH / IMU
        │             │              │
        ▼             ▼              ▼
   Vision NN     Auditory NN     Body/Sensor Encoder
        │             │              │
        └─────────────┼──────────────┘
                      ▼
                Feature Vector
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Neuromorphic SNN          Policy GRU
   internal state            action choice
          │                       │
          └───────────┬───────────┘
                      ▼
                 Action Vector
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
       Motion       Face     Sound Generation NN
          │           │           │
          ▼           ▼           ▼
        Servos       LCD       Droid Voice
          │
          └──────────────┬──────────────┐
                         ▼              │
                    Body Feedback       │
                         └───────────────┘
```

The robot operates continuously:

```text
SEE / HEAR / FEEL
        ↓
INTERNAL STATE
        ↓
CHOOSE ACTION
        ↓
ACT
        ↓
OBSERVE RESULT
        ↓
UPDATE STATE
        ↺
```

---

## 2. Why Hybrid Instead of One Large SNN

A pure SNN is useful for:

```text
attention
novelty
habituation
temporal traces
spontaneous activity
arousal
boredom
curiosity
internal drives
```

But a pure SNN becomes difficult to train and debug when it is also responsible for:

```text
camera interpretation
sound recognition
motor policy
long temporal decisions
vocal synthesis
```

A small recurrent network is better suited to learned action selection.

Therefore:

```text
SNN         = What matters? How do I feel? What happened recently?
Policy GRU  = Given this situation, what should I do next?
Vision NN   = What am I seeing?
Auditory NN = What am I hearing?
Sound NN    = How should I sound?
C++ runtime = How do I execute this safely?
```

---

## 3. Main Design Rules

- No neural network directly writes PWM or raw hardware registers.
- Hard servo limits remain deterministic.
- Audio gain limits remain deterministic.
- Watchdog and battery protection remain deterministic.
- Raw video and raw audio are compressed into features before reaching the brain.
- The SNN must remain internally active even in silence.
- The policy network must be small enough to run predictably on ESP32-S3.
- Learning must be bounded.
- Every neural module must have a safe fallback.

---

## 4. Hardware Layer

Inputs:

```text
camera
microphones
touch
IMU
battery state
servo position
servo velocity
ambient light
speaker state
SD card
```

Outputs:

```text
head yaw
head pitch
face / eyes
rear LEDs
speaker
```

---

## 5. Deterministic Runtime Layer

Keep in C/C++:

```text
camera acquisition
microphone sampling
audio DMA
servo update loop
servo limits
display drawing
touch sampling
IMU sampling
battery monitoring
SD filesystem
watchdog
FreeRTOS scheduling
```

This layer is the robot's "spinal cord and physiology."

It should not decide personality.

---

## 6. Vision Perception Network

The camera gets its own compact perception network.

Initial outputs:

```text
motion_left
motion_center
motion_right

person_probability
face_probability

looming
receding

brightness
visual_novelty

target_x
target_y

visual_activity
```

Suggested pipeline:

```text
Camera frame
   ↓
resize / grayscale
   ↓
small Vision NN
   ↓
16-64 features
```

The Vision NN answers:

> What visually important event is happening?

It does not choose the robot's reaction.

---

## 7. Auditory Perception Network

This is a new future module.

It is completely separate from the existing Sound Generation NN.

Its purpose is:

> What did the robot hear?

Initial classes/features:

```text
silence
speech_like
clap
sharp_noise
continuous_noise
low_frequency_event
high_frequency_event
novel_sound
approaching_loudness
left_direction
right_direction
sound_energy
```

Optional later keyword layer:

```text
hello
music
stop
sleep
wake
yes
no
dance
```

Suggested pipeline:

```text
Microphones
   ↓
filter bank / FFT / compact spectral features
   ↓
Auditory NN
   ↓
audio feature vector
```

---

## 8. Existing Sound Generation Network

Keep the current trained sound-generation network if it is stable.

Its task:

> Given the robot's internal state, how should the robot sound?

Possible inputs:

```text
valence
arousal
curiosity
annoyance
boredom
playfulness
alertness
social drive
action context
```

Possible outputs:

```text
pitch
pitch glide
duration
vibrato
intensity
sound family
pause
rhythm
```

Example:

```text
high curiosity
moderate arousal
low annoyance
        ↓
Sound Generation NN
        ↓
rising chirp + short trill
```

This network is the robot's vocal system.

---

## 9. Compact Neuromorphic SNN

The SNN should become smaller and more focused than the previous all-in-one brain.

Recommended first target:

```text
512-1024 LIF neurons
```

Potential later target:

```text
~1500 neurons
```

only if profiling shows clear benefit.

Its job is to represent:

```text
attention
novelty
habituation
curiosity
arousal
boredom
annoyance
social need
playfulness
alertness
fatigue
sleep pressure
music drive
recent sensory traces
prediction error
```

---

## 10. Example SNN Population Map

Example for 768 neurons:

```text
Attention                 96
Novelty                   64
Habituation               64
Curiosity                 64
Arousal                   64
Boredom                   48
Annoyance                 48
Social drive              48
Playfulness               48
Fatigue / sleep           48
Short sensory traces      96
Prediction / mismatch     80
----------------------------
Total                    768
```

The exact distribution may change after profiling.

---

## 11. Endogenous SNN Activity

The SNN must not become silent when no external event occurs.

Suggested update:

```text
20-50 Hz
```

Example:

```text
silence
↓
weak recurrent activity continues
↓
boredom slowly rises
↓
attention drifts
↓
small gaze movement
↓
possible quiet chirp
```

The external world should perturb an already living internal state, not simply wake a dead network.

---

## 12. Sensory Traces

Important events should decay gradually.

Example:

```text
touch
100% → 80% → 60% → 40% → 20% → 0
```

over approximately:

```text
0.5-5 seconds
```

Suggested traces:

```text
touch_trace
sound_trace
person_trace
motion_trace
novelty_trace
reward_trace
```

This gives temporal context.

---

## 13. Habituation

Repeated identical events should become less novel.

Example:

```text
first touch
→ strong novelty

second touch
→ moderate novelty

many repeated touches
→ low novelty
→ annoyance rises
```

Novelty should recover over time.

---

## 14. Internal Drives

Maintain slow-changing internal values:

```text
curiosity
boredom
arousal
annoyance
social_need
playfulness
energy
fatigue
sleep_pressure
music_drive
```

Suggested range:

```text
0.0 ... 1.0
```

These values may be represented partly by population activity and partly by explicit normalized state.

---

## 15. Policy Network

The policy network answers:

> What should I do next?

Recommended initial architecture:

```text
Input: 64-128 features
GRU: 64 hidden units
Dense: 32-64
Output: 16-32 values
```

A smaller `GRU 32` may be tested first if needed.

Inputs:

```text
vision features
auditory features
touch / IMU
body state
SNN state
previous action
prediction error
```

---

## 16. Example Policy Input Vector

```text
Vision:
  target_x
  target_y
  motion_left
  motion_right
  looming
  person_probability
  visual_novelty

Audio:
  sound_energy
  speech_like
  sharp_noise
  audio_novelty
  direction

Body:
  yaw
  pitch
  yaw_velocity
  pitch_velocity
  touched
  IMU tilt

SNN:
  curiosity
  arousal
  boredom
  annoyance
  social
  playfulness
  fatigue
  attention

Previous action:
  last_yaw
  last_pitch
  last_sound_intensity
  last_action_type
```

---

## 17. Policy Outputs

Use normalized continuous outputs:

```text
head_yaw_target       -1.0 .. +1.0
head_pitch_target     -1.0 .. +1.0
head_speed             0.0 .. 1.0
head_vigor             0.0 .. 1.0

gaze_x                -1.0 .. +1.0
gaze_y                -1.0 .. +1.0

sound_activation       0.0 .. 1.0
sound_arousal          0.0 .. 1.0
sound_valence         -1.0 .. +1.0

blink_probability      0.0 .. 1.0
led_activity           0.0 .. 1.0

hold_probability       0.0 .. 1.0
explore_probability    0.0 .. 1.0
```

Sound-related policy outputs feed the existing Sound Generation NN.

---

## 18. Motor Execution

Correct control chain:

```text
Policy NN
↓
normalized motor intent
↓
Motion Engine
↓
easing / interpolation
↓
Servo Safety
↓
servo driver
```

The network never produces raw PWM.

---

## 19. Body Feedback

After every action, measure what actually happened.

Feedback:

```text
actual yaw
actual pitch
actual velocity
IMU motion
camera movement
speaker state
servo response
```

Return this to:

```text
SNN
Policy NN
```

---

## 20. Efference Copy

When the policy asks for movement:

```text
head_yaw = +0.40
```

the internal state receives a copy:

```text
SELF_MOTION_INTENT = +0.40
```

This allows the robot to compare intention with reality.

---

## 21. Prediction Error

Example:

```text
expected yaw = +0.40
actual yaw   = +0.31
```

Then:

```text
prediction_error = 0.09
```

The same idea applies to self-generated sound.

Expected microphone response:

```text
robot generates chirp
+
microphones detect chirp
→ self-generated event
```

Unexpected microphone event:

```text
no self-sound command
+
new sound
→ external novelty
```

This supports primitive self-vs-world distinction.

---

## 22. Policy Training Strategy

Do not start with full reinforcement learning directly on the robot.

### Stage 1: Imitation Learning

Record:

```text
sensor state
SNN state
desired action
```

Train the policy to imitate sensible behavior.

### Stage 2: Controlled Exploration

Allow small variations inside safe ranges.

### Stage 3: Reward Learning

Possible rewards:

```text
target moves toward image center       +1.0
successful attention response          +0.5
novel interaction                      +0.3
stable relaxed idle                    +0.2

repetitive action                      -0.2
rapid unstable oscillation             -0.5
servo saturation                       -1.0
```

### Stage 4: Limited On-Device Adaptation

Only small bounded updates are allowed.

---

## 23. Training Data Logging

Add a dataset logging mode.

Record:

```text
timestamp
vision features
audio features
touch / IMU
SNN state
policy hidden state
selected action
actual body result
prediction error
reward
```

Example:

```text
t=48123
vision_target_x=-0.62
audio_novelty=0.10
curiosity=0.82
arousal=0.44
yaw_action=-0.37
actual_yaw=-0.33
reward=+0.4
```

This dataset can later be used for offline retraining.

---

## 24. Why Not One End-to-End Raw Network

Do not initially do:

```text
raw camera
+
raw audio
+
raw IMU
↓
one large network
↓
servo commands
```

This requires a large dataset, is difficult to debug, and wastes compute.

Prefer:

```text
specialized perception
↓
compact features
↓
internal state
↓
policy
```

---

## 25. Autonomous Behavior

The robot should act even without immediate external stimulation.

Example:

```text
boredom rises
↓
policy sees high boredom + low alertness
↓
head slowly scans room
↓
camera detects motion
↓
curiosity rises
↓
robot turns toward it
```

Another example:

```text
social_need rises
↓
policy chooses small attention-seeking behavior
↓
quiet chirp
↓
wait
```

---

## 26. Sleep Mode

Conditions:

```text
high sleep_pressure
low external activity
low arousal
long inactivity
```

During sleep:

```text
SNN remains minimally active
vision runs slower
audio wake detector remains active
head moves to safe posture
rear LEDs pulse slowly
screen shows low heartbeat state
```

Wake events:

```text
touch
loud sound
keyword
strong movement
```

---

## 27. Virtual Physiology

Maintain a physiology state:

```cpp
struct PhysiologyState {
    float heartRate;
    float breathingRate;
    float arousal;
    float fatigue;
    float thermalStress;
};
```

Example heartbeat ranges:

```text
IDLE      55-65 BPM
CURIOUS   70-85 BPM
ACTIVE    85-105 BPM
EXCITED   110-135 BPM
SLEEP     35-45 BPM
```

Heartbeat should mainly follow the internal state.

CPU load and temperature may slightly influence it but must not dominate it.

---

## 28. Display and Rear LED Behavior

The display can visualize:

```text
heartbeat
eyes
arousal
sleep state
attention
```

Rear LEDs may visualize:

```text
breathing
heartbeat
sleep
alert
music rhythm
```

These animations should reflect internal physiology rather than random decoration.

---

## 29. Music Engine

Keep the existing MIDI-like Song Engine deterministic.

The policy may output:

```text
start_music_probability
stop_music_probability
music_energy
```

Music feeds back into the SNN:

```text
music playing
↓
music_drive decreases
arousal changes
playfulness may rise
```

---

## 30. Audio Architecture

There are two different neural audio systems.

### Auditory Perception NN

```text
microphone → What did I hear?
```

### Sound Generation NN

```text
internal state → How should I sound?
```

Full chain:

```text
MIC
↓
Auditory NN
↓
SNN + Policy
↓
Sound Generation NN
↓
Speaker
```

They must remain independent.

---

## 31. Vision Architecture

```text
Camera
↓
Vision NN
↓
features
↓
SNN + Policy
↓
head / gaze
↓
camera view changes
↓
feedback
```

This creates active vision.

The robot can move to improve what it sees.

---

## 32. Initial Resource Budget

Recommended first hybrid version:

```text
Vision NN:
small task-specific model

Auditory NN:
small classifier

SNN:
512-1024 neurons

Policy:
GRU 64 hidden
~20k-100k parameters

Sound Generation NN:
existing trained model

MIDI Engine:
existing deterministic system
```

Do not enlarge these modules before profiling.

---

## 33. Real-Time Priorities

Conceptual priority:

```text
1. Safety / watchdog
2. Audio DMA
3. Servo update
4. Sensor acquisition
5. Perception
6. Policy
7. SNN
8. Display
9. Logging / SD
```

Final FreeRTOS priorities must be determined experimentally.

---

## 34. Failure Handling

If one neural module fails:

```text
Vision NN failure
→ continue without vision

Auditory NN failure
→ disable hearing perception

Policy invalid output
→ clamp or reject

SNN unstable
→ reset internal state

Sound NN unstable
→ mute or use fallback chirp

SD missing
→ use built-in defaults
```

No neural failure may create unsafe motor behavior.

---

## 35. Debug Telemetry

Expose compact diagnostics:

```text
VISION:
 target_x=-0.42
 motion=0.61
 person=0.73

AUDIO:
 energy=0.14
 novelty=0.02
 speech_like=0.31

SNN:
 curiosity=0.78
 arousal=0.41
 boredom=0.22
 annoyance=0.05

POLICY:
 yaw=-0.36
 pitch=+0.12
 speed=0.44
 sound=0.27

BODY:
 actual_yaw=-0.33
 actual_pitch=+0.10

PRED_ERR=0.04
```

---

## 36. A/B Experiment

Compare:

### Architecture A

```text
Current large SNN:
~6700 neurons
~209k synapses
```

### Architecture B

```text
Hybrid:
SNN 512-1024
+
GRU policy
+
Vision NN
+
Auditory NN
+
existing Sound Generation NN
```

Measure:

```text
CPU
RAM
latency
reactivity
spontaneous behavior
stability
behavior variety
learning speed
time spent inactive
repetition rate
```

Do not assume the hybrid architecture is better. Test it.

---

## 37. Development Plan

### Phase 1
Freeze working hardware and safety code.

Do not change:

```text
drivers
servo safety
working Sound Generation NN
working MIDI system
stable display code
```

### Phase 2
Create a unified Feature Bus.

### Phase 3
Refactor SNN to 512-1024 neurons focused on internal state.

### Phase 4
Add a small GRU policy.

### Phase 5
Connect GRU outputs to head and gaze through the Motion Engine.

### Phase 6
Connect policy + SNN state to the existing Sound Generation NN.

### Phase 7
Add Vision NN.

### Phase 8
Add Auditory Perception NN.

### Phase 9
Add efference copy and prediction error.

### Phase 10
Log full sensor-state-action datasets.

### Phase 11
Train policy offline with imitation learning.

### Phase 12
Add bounded reward-based adaptation.

---

## 38. Acceptance Criteria

The system is successful when:

1. the robot remains fully standalone;
2. vision affects behavior through a dedicated perception network;
3. microphone perception is handled by a separate auditory network;
4. sound generation remains a separate trained network;
5. the SNN maintains continuous internal state;
6. habituation works;
7. the robot does not require an external stimulus for all activity;
8. the GRU produces continuous motor intent;
9. all motor output passes deterministic safety;
10. perception and internal state both influence behavior;
11. repeated events do not always produce identical reactions;
12. calm idle remains possible without total inactivity;
13. body feedback returns into the brain;
14. prediction error is available;
15. training data can be logged;
16. the hybrid design can be directly compared with the current large-SNN design;
17. failure of any neural module remains physically safe.

---

## 39. Final Mental Model

Think of the system as specialized artificial brain regions:

```text
Vision NN       = eyes / visual perception
Auditory NN     = ears / auditory perception
SNN             = drives / attention / internal state
Policy GRU      = action-selection cortex
Sound NN        = vocal system
Motion Engine   = spinal / motor execution
C++ Safety      = hard reflexes and physical limits
```

The final loop:

```text
WORLD
 ↓
PERCEPTION
 ↓
INTERNAL STATE
 ↓
LEARNED POLICY
 ↓
ACTION
 ↓
BODY
 ↓
FEEDBACK
 ↺
```

The goal is not to imitate a human brain.

The goal is to create a compact, trainable, understandable, autonomous embodied intelligence that fits the real limits of the ESP32-S3 and produces richer behavior than either a scripted robot or one monolithic neural network.
