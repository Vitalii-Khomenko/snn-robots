# Contingency: the sense of "I did that" (concept, 2026-10-27)

The owner, 2026-10-27: *"work out the sense of causality."* Then: *"do step 1 and flash, then steps two and three."*
Built and on the robot:
- step 1 since 2026-10-27 06:29;
- step 2 since 06:48;
- step 3 since 07:53 (network 1d448676, blob 26102708; ARCHITECTURE sections 3 and 5).

Steps 4-5 are the plan. The
science behind it is in [FOUNDATIONS](FOUNDATIONS.md) (sections 5-7).

## 1. Why this, and why now

At two to three months a baby discovers that the world answers it. John S. Watson (1972) put infants on a pillow that
turned a mobile above them when they moved their head. After a few days of that game they smiled and cooed at the
mobile. Babies whose mobile turned just as often, but on its own, did not. The joy was not in the toy. It was in
**"I make it move."** From there it goes on:

- **Piaget's circular reactions.** The child repeats an act that made something happen (kick, the mobile shakes,
  kick again). Later it varies the act on purpose to see what changes: the first experiments.
- **The still face (Tronick, 1978).** A mother keeps looking at her baby but stops answering. Within a minute the
  baby tries everything (smiles, calls, reaches), then turns away and gets upset. When she answers again it
  recovers. A broken contingency hurts more than being alone.
- **Turn-taking (proto-conversation).** Long before words, mother and baby take turns: one acts, waits, and the
  other answers in about a second.

The robot has none of this. It acts (calls, nods, faces, lights, ideas of the Muse) and its senses change, but
nothing links the two. Only the night sleep scores a whole day after the fact (`reward_v4.py`). In the moment, an
answer is just one more stimulus. For the owner this is what makes the robot feel *not quite there*: it does not
notice that it was answered. A robot that notices, repeats what worked and gets upset when it is ignored feels like
somebody who understands.

## 2. What the diary says today (step 0, measured 2026-10-27)

Two days of the robot's life (2026-10-25 and 10-26, daytime, with someone around in the last 2 minutes). Each
spontaneous act counts (nothing answered in the 10 s before it, so the robot's own answers to touches are not
counted). The question: did an answer come 0.5-4 s after it, more often than in the same window after any moment
without an act? (`tools/v4/contingency.py`.)

| Act (n) | Pet | Tap | Someone near | Speech | Motion |
|---|---|---|---|---|---|
| call (2209) | 0.000 vs 0.000 | 0.001 vs 0.001 | 0.000 vs 0.001 | masked (own voice) | 0 vs 0 |
| gesture (185) | 0.005 vs 0.000 | 0.011 vs 0.001 | 0.005 vs 0.001 | 0.038 vs 0.216 | 0 vs 0 |
| face change (3108) | 0.001 vs 0.000 | 0.003 vs 0.001 | 0.001 vs 0.001 | 0.243 vs 0.216 | 0 vs 0 |

What this shows:

1. **The robot cannot hear an answer.** The speech sense fires 205-280 times an hour by day. That is mostly not
   the owner: fans, the TV, the robot's own servos. It is masked for 5 s after each of the robot's own calls, so a
   call can never be answered by voice. Speech is the most natural answer, and today it is pure noise.
2. **Touch answers are rare** (0.6-2.7 pets an hour by day), but they do follow gestures more than chance:
   tap 0.011 vs 0.001, pet 0.005 vs 0.000. There is a real signal, only a thin one.
3. **Motion onsets in still frames are almost never seen.** The camera mostly sees motion while the head turns
   (false), and that is gated out.

So the first job is not the feeling but the senses: **telling "the world answered" from "that was me" and from
noise.** Infants solve this first too. Their own body is perfectly contingent (I move, I feel it at once), so it
gets labelled "me". People are highly but imperfectly contingent, so they get labelled "other" (Bahrick & Watson,
1985; Gergely & Watson, 1999).

## 3. The design

```
 executor acts ---------> act log (kind, time; an efference copy)
                                   |
 senses -> answer events ---------+--> contingency detector --> 6 inputs to the NN (PREV):
   (pet, tap, clean speech, near,  |      per act kind: answered / tried,       agency, answered, ignored,
    motion in still frames,        |      against the background rate           c_call, c_gesture, c_face
    someone coming into view)      |
                                   +--> self-contingency: answers that ALWAYS come at once = me -> masked
```

### 3.1 Acts: an efference copy

The executor already knows everything it does. Each act gets a record: kind and time. The kinds are a call (by
style), a nod, a shake, a face change (by category: happy, play, odd, sad), an LED pattern change, a glance, and a
Muse idea act. A ring buffer of the last 8 acts is enough.

### 3.2 Answers: what counts as the world answering

| Answer | From | Weight | Notes |
|---|---|---|---|
| pet | raw_head_ev, gentle | 1.0 | clean |
| tap | raw_screen_ev, gentle | 0.8 | clean |
| speech | a NEW speech onset after >= 2 s of quiet, not own voice | 0.6 | only after step 2 cleans it |
| someone near | raw_near onset | 0.5 | clean, rare |
| motion | motion onset in a still frame (lightOk) | 0.3 | noisy |
| coming into view | att_strength rising while the head is still | 0.3 | |

### 3.3 The detector

It is a small statistics block in C++ (with a Python twin, like `channels.py`). It is not a network: it must be
honest and fast.

- **Background rate.** λ = answers per second at moments with no act in the 5 s before, and only while someone
  is around. Chance of an answer landing in the window by accident: p0 = 1 - exp(-λ · 3.5 s).
- **Per act kind k.** Leaky counts of acts n_k and of answered acts h_k. Answered means an answer in 0.3-4 s.
  When several acts share a window, the latest one gets most of the credit (an eligibility trace).
- **Contingency.** c_k = (h_k + p0·m) / (n_k + m) - p0, with m = 3.

  The prior m keeps a single lucky answer from making a superstition: it takes about three answered acts before
  c_k means anything (Rescorla-Wagner in spirit: learn from the surprise, not from co-occurrence).
- **Two time scales.**
  - "Now" (τ ≈ 5 min): the current game with the person in front of it.
  - "Life" (τ ≈ 3 days, saved in the learned blob): what usually works with this owner.

### 3.4 What the main network sees (6 new PREV inputs)

| Input | Meaning |
|---|---|
| agency | how much the world answers me right now (max over kinds of c_k "now", 0..1) |
| answered | a pulse when an answer just followed my act, decaying over ~3 s: "it worked!" |
| ignored | rises while acts go unanswered with someone around, falls slowly: the still face |
| c_call, c_gesture, c_face | which kind of act works now (and, through the life scale, with this owner) |

There is no online weight learning: the counts are the fast memory. The network learns from its teacher what to do
with them. This is how a child's game works within minutes, long before the night's sleep.

### 3.5 What the teacher teaches (the behaviour)

| Rule | From | What the robot does |
|---|---|---|
| Joy of control | Watson 1972 | answered: happy, very happy or laughing (stronger the higher agency), a bright flash |
| Do it again | Piaget, secondary circular reaction | answered: repeat the same act in 1.5-4 s, up to 3-5 times, then the game tires (habituation) |
| Vary it | Piaget, tertiary circular reaction | after 2-3 repeats: the same act a little different (amplitude, style, pitch) |
| Bids, then the still face | Tronick 1978 | someone is there and nothing is answered: bid harder (ask-style call, look at them), then after 1-2 min sad, then quiet and turned away. No anger. |
| Repair | Tronick | answered after being ignored: extra joy, ignored resets |
| Wait for the answer | proto-conversation | after an act aimed at a person: hold still 2-3 s, as if waiting; do not call over the owner's speech; answer the owner's speech within ~1 s |
| No learned helplessness | Seligman | alone is not ignored (that is loneliness, already a need); ignored fades over ~10 min and is zero after a boot |

### 3.6 Self versus other (steps 2 and 4)

This uses the same detector, but in reverse. For each kind of sense event it also counts how often the event comes
within 0.5 s of each act kind.

- **A sense that answers almost always and at once is me.** Examples: speech right after my own call (a third of
  the speech onsets on 2026-10-24 were the voice's tail), camera motion during my own head turn, sound peaks from
  my servos.
- **Today these are fixed by hand.** The 5 s own-voice mask, the `ownMotion` flag in `BrainLoop::hear`, the
  lightOk gate.
- **Infants learn the same thing on their own.** They find their body this way. The detector would find such
  links by itself and report them in the diary first. Later it would mask them (with a cap, so that a real
  answer is never masked for long).
- **Delay matters** (Blakemore, Wolpert & Frith: you cannot tickle yourself because the brain predicts the touch).
  "Me" is a link with a fixed latency under ~0.3 s. A person needs 0.5-3 s.

## 4. How we would know it works

**Simulator** (new scripted people in `simulator/`):

| Scenario | The person | Expected |
|---|---|---|
| responsive | pets within 1-2 s after a nod, 80 % of the time, never otherwise | nods at least 2x more often than in the control within 10 minutes; happy after the answered nods |
| control | pets just as often, at random times | no change in nods |
| still face | sits in view, never answers | bids rise within 1 minute, sad within 3 minutes, never angry; recovers at the first answer |
| own echo | a fake "speech" event 0.2 s after every call | after step 2: labelled self, no longer counted as an answer |

All existing behaviour tests must still pass.

**On the robot** (a diary report per day, `tools/v4/contingency.py DAY`):

- ΔP per act kind;
- how often an answered act was repeated within 4 s;
- time from being ignored to sad;
- the share of the day with agency > 0.3.

The real test is the owner: does it feel like the robot noticed you?

## 5. Cost

Per tick: a ring of 8 acts, about 20 leaky counters, a few comparisons, so microseconds. The NN gets 6 more inputs
(49 -> 55), +1.2 k parameters, so a retrain (as for the Muse). Diary: +6 columns.

## 6. Steps

| Step | What | Changes behaviour? |
|---|---|---|
| 0 | Measure ΔP from the diary | no (done, section 2) |
| 1 | Detector in C++ and Python (parity), diary columns, daily reports; a few days of real numbers (built 2026-10-27: `contingency_v4.cpp`, `contingency.py`, `parity_contingency.py`; diary `ct_*`) | no |
| 2 | Clean the answers: why speech fires 200+ times an hour (a short recording with the owner's OK), then self-contingency labelling of sense events (built 2026-10-27, below) | no, only cleaner senses |
| 3 | 6 NN inputs, the teacher's rules (3.5), simulator scenarios (4), retrain, flash (built 2026-10-27, 6b) | yes |
| 4 | The life scale in the learned blob; at night the Muse's idea biases and the AWR also see c_k | slowly |
| 5 | Turn-taking with voice: answer within ~1 s, wait, no overlap | yes |

Steps 1-2 are safe and show whether there is enough signal. If after two days an answered act is still rarer
than one an hour, the robot first needs better ears (step 2) before step 3 can be felt.

## 6a. What step 2 found (2026-10-27)

**Recordings.** The robot streams its microphone to the home PC on request (`net/sound_tap.h`, `GET /listen`,
`tools/robot/listen.py`), stamped with the diary's clock. `tools/v4/sound_check.py` lines every speech onset of the
ear up with the sound and the robot's own acts.

**Where the ear's speech came from.** In 5 minutes with the owner talking nearby, the ear started speech 31 times
(373/h):
- 16 were the robot's own voice: during a call, or its tail and echo in the 0.3-2 s after;
- 8 were its servos: the whine at 5-6 kHz and the clicks of each eased step;
- 6 were near-silence;
- none was the owner.

The owner's voice was there: harmonic stacks at 100-1500 Hz. But the ear's onsets fell on the robot's own sounds.

**The voice check** (`perception/voice_check.cpp`, ARCHITECTURE section 5) asks for periodicity at 80-400 Hz above
the room's floor. It lets the owner's speech through and holds the robot's own sounds back, with 1.5 s for the
tail of its own voice:
- a minute with nobody talking: speech 9.9 % -> 0.5 % of the time;
- live: 14 % -> 5 %.

**Me or not me** (`contingency.py --self`) and a mistake made on the way. The first table showed a sharp sound 0-0.5 s
after the robot's own call 14-15x more often than at any moment. That looked like the robot startled by its own
speaker, and sharp sounds were dropped while it spoke (06:55). Wrong.
- Those sharp sounds were in the call's own tick. The ear runs before the executor, so the sound came first and the
  call was the robot's answer to it.
- Counted strictly after the act, the link is gone (x0.3). The suppression was taken out again with step 3.
- The lesson for self-contingency: "after" must mean a later tick. Otherwise every reaction looks like an echo.
- No fast self-link is large enough to act on. Taps and someone near after a gesture are few (9-23 a day); they are
  most likely a double tap or a hand still there after the robot nodded at the first.

## 6b. What step 3 gave (2026-10-27)

**Simulator** (`behavior_test_v4.py`, all pass):

| Scenario | Result |
|---|---|
| contingent | 96 answers, 55 credited. A joyful face 1.5 s after an answer 42 % of the time |
| contingent vs control | calls and gestures 9.3 vs 8.2 a minute (x1.13); nods 36 vs 24 |
| still_face | sad or bored 30 % of 7:00-11:00, never angry, "Confused" bids before |

**Not reached:** the same act again right after its answer (Piaget's repetition) is not above chance:
- calls are at the voice budget in every scenario (70 in 12 min), so there is no room for an extra one;
- the network learns the teacher's timed nod-again only weakly.

The robot does more of what is answered, not the same act at once. Open.

**Training lessons:**
- An answer must be felt as well as counted (`contingency.inject`).
- A simulated person answers at a person's pace.
- Blobs carry the network's fingerprint (`nn.id`).

**Watch on the robot:** the census of real-life slices showed sad 20 % and the head down 23 %. That world has
someone silent 20 % of the time, so it is not the robot's real day. The diary will tell (`contingency.py DAY`:
ignored > 0.5, faces).

## 7. Open questions

- **How should the owner answer?** Petting and voice are natural. A wave at the camera would be easier to
  detect: motion in a still frame while the robot holds its head.
- **Should "answered" be visible?** A tiny reaction at the moment of the answer (the eyes widen, a short chirp)
  would show the owner that the robot noticed. That closes the loop for the owner too: the owner learns the
  robot's contingency just as the robot learns the owner's.
- **Should a child-like robot keep a few acts the owner never answers?** Exploration, as in the Muse's open
  question. Yes, probably: the circular reaction tires, and the Muse brings new acts.
