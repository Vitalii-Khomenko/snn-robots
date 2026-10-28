# Creativity: a "Muse" network (2026-10-26)

The owner, 2026-10-26: *"for creativity and unpredictability we probably need an additional NN that gives new
signals - work it out as a concept."* Then: *"implement it all the way."* Steps 1-4 below are built the same evening
(ARCHITECTURE sections 3 and 7). The curious glances in the executor (section 4) stay as the head's own small
chance.

## 0. Status (2026-10-28, after an audit: what is built against what the concept below promised)

| Part | Concept | Built | Measured | Open |
|---|---|---|---|---|
| Reservoir | 64 tanh neurons at the edge of chaos, driven by mood and attention | as proposed (`muse_v4.cpp`, `muse.py`, parity 9e-7) | 5 kinds of ideas by itself (`day_alone`) | a spectral radius near 1 is not by itself proof of creativity |
| Ideas | 5 ideas the main network acts on | as proposed; what each idea means is the teacher's fixed rule | the network acts on them (simulator) | the ideas do not invent new acts: a *generator of structured initiatives* over a fixed repertoire |
| Habituation | "bored of itself": pushes the reservoir to new regions | damps an idea's own positive output (90 s); **it does not feed back into the reservoir** | - | feed back into the reservoir, or drop the claim |
| Selection | AWR trains the readout at night | **the readout is fixed**; the night moves one bias per idea (`muse.bias`, +-0.05 a night, within +-0.6) | 5 biases moved by < 0.01 so far (the ideas were hardly answered) | learning the readout itself |
| Curiosity | "learning progress, after Oudeyer" | **the error of a 1 s ridge forward model of the channels, fitted on the same day**: unexpectedness, not progress; the robot's own act is not among its inputs, so a noisy room scores high with nothing learned | - | a forward model with the robot's own act and context, and progress measured on held-out moments (Oudeyer, Kaplan & Hafner 2007) |

## 1. Where variety comes from today

| Source | What it is | Limit |
|---|---|---|
| Pacemakers | 3 noise drifts (Ornstein-Uhlenbeck, ~3 s / ~60 s / ~0.5 s) the main network sees as inputs (`PV_PACE_*`). The teacher reads its "random" choices from them (`teacher_nn.pick`), so the network learns to vary the same way. | Pure noise: it knows nothing of the room, the mood or what was done a minute ago. Its only effect is to pick among the teacher's fixed options. |
| The teacher's repertoire | `teacher_nn.py`: 21 moods, each with a head manner, voice style, LEDs, and gestures. | The network imitates it. It cannot do anything the teacher never did. |
| The executor | Curious glances, random hold times. | Hand rules: they move the head only, and they never learn. |
| Night learning | AWR on the action heads, from the owner's reactions (`reward_v4.py`). | It re-weights what exists. It does not invent. |

So the robot is **varied but not inventive**. Every minute is different, yet nothing is new: the same set of moves
comes in a different order. Noise only shuffles that set.

## 2. The idea

A small extra network, the **Muse**, runs next to the analyzer (SNN) and the main network (NN). It gives the main
network new inputs: *ideas*. An idea is a short-lived, structured impulse, such as "look over there", "play with
sound", "try a funny face" or "be still and watch". Three things together make it creative rather than random:

1. **Structured spontaneity: a chaotic reservoir.** About 64 tanh neurons, recurrently connected at the edge of chaos
   (an echo-state network with spectral radius ~1.05). Mood, attention, boredom and time of day drive it weakly.
   Its trajectories never repeat, yet they are coherent: an idea lasts seconds, it develops, and it changes because
   of what happens around the robot. This replaces the blind pacemakers. Same inputs make similar ideas, and
   different days make different ones.
2. **Novelty: it gets bored of itself.** It keeps a memory of its own recent ideas: a habituation trace over the
   readout, like light_spot's place memory. An idea that was tried recently is damped. *(Concept: dropping a
   repeated idea pushes the reservoir towards unexplored regions. Built: the damping acts on the idea's output only
   and does not reach the reservoir - section 0.)*
3. **Selection: it learns what was good.** The night sleep already scores the day: touch, voice, the owner coming
   near, set against being ignored or poked (`reward_v4.py`). Ideas that drew an answer from the owner come back
   more often, and ideas nobody cared about fade. *(Concept: AWR trains the readout. Built: one bias per idea moves a
   little each night - section 0.)* A **curiosity** term comes on top: a small forward model predicts the channels a
   second ahead, and an idea after which the world turned out differently than predicted is worth more, even with
   nobody there. *(Concept: learning progress, after Oudeyer. Built: the prediction error itself, which a noisy room
   also has - section 0.)*

```
 senses -> ChannelEncoder -> SNN (768 LIF) ----------------> NN (GRU policy) -> executor
                               |  mood, attention, boredom    ^
                               v                              | 8 idea inputs (replace the 3 pacemakers)
                        Muse: reservoir 64 tanh -> readout 8 -+
                               ^        |
                               |        +-> habituation of its own ideas (damps repeats)
                         night sleep: one bias per idea, from reward_v4 + curiosity (forward model error)
```

### What the outputs are (as built)

| # | Output | What the main network learns to do with it (teacher) |
|---|---|---|
| 0-2 | pace_fast / pace_slow / pace_jitter | the old pacemakers' job (every "random" pick of the teacher: where to look, how wide, which face of a mood), same spread and time scales, now coloured by the mood |
| 3 | play | a sound game: short lively calls every 2 s, play style, a wave of colour, a lively face (6 s) |
| 4 | show | a face it rarely makes (wink, cool, love, confused, surprised, laughing, shy), the head up, a flicker (5 s) |
| 5 | still | be still and watch: no babble, the head holds, the lights breathe low (10 s) |
| 6 | social | an asking call, the head to where someone was last seen, a pulse (6 s) |
| 7 | surprise | a nod dance (3 reps), a call, a burst of light (2.5 s) |

(The concept had "where" and "tempo" as ideas of their own; the pacemakers already carry them.)

The Muse **proposes** and never acts itself. The main network decides, through what it learned from the teacher.
The teacher gets a rule per idea, such as "with play > 0.6 and no reason to be quiet: a sound game". It also gets
a *creativity budget* that grows with boredom and falls with threat, sleepiness and the night.

## 3. What stays fixed (safety)

- Everything the executor guards stays as it is: silence and no head moves at night, servo packet limits, the nod
  and shake gaps, power-gated moves, eased turns.
- Reactions come first. A touch, a fright or a bang outweigh any idea: the teacher's rules already rank them higher.
- The Muse only feeds inputs into the main network, so it can be switched off by replacing its outputs with the old
  pacemakers. The current behaviour is the fallback.

## 4. Cost on the robot

The reservoir is 64x64 plus 64 inputs and an 8-wide readout: about 5 000 multiply-adds per step at 10 Hz, well
under 0.5 ms. The SNN takes ~3.7 ms and the NN ~5.2 ms. The reservoir weights are fixed (seeded) and live in
flash. The readout is fixed too; the night's one bias per idea goes into the learned blob (`muse.bias`, 8 floats).
The habituation state lives in RAM and starts afresh at a reboot.

## 5. How we would know it works

- **Behaviour entropy** in the simulator and the diary: the share of distinct 10-second "phrases" (head mode +
  face + sound style n-grams) and how often a phrase repeats within an hour. The target is clearly above today's
  level, with all behaviour tests still passing.
- **Coherence:** an idea's effect lasts. Random flicker scores high on entropy too, so the test also checks that a
  phrase holds for seconds.
- **The owner's verdict** stays the real test: does it surprise in a good way?
- **Night numbers:** how often Muse-driven acts drew an answer, and whether that share grows over the days.

## 6. Steps (all four built 2026-10-26)

| Step | What | Retrain | Rough size |
|---|---|---|---|
| 1 | Reservoir Muse in C++ and its Python twin (parity, like `channels.py`). Its 3 slowest readouts replace the pacemakers 1:1 (same range and time scales), so the teacher keeps working unchanged. Retrain the NN on Muse sequences. Measure entropy. | NN | a day |
| 2 | 8 idea outputs plus the habituation of its own ideas; teacher rules per idea, with a few new micro-acts (sound game, double glance, nod dance, "stillness"); the creativity budget | NN | 2-3 days |
| 3 | Night learning from reward_v4 (built as one bias per idea, not the readout; the same gates: behaviour tests) | none by hand | a day |
| 4 | Curiosity: a forward model of the SNN channels; its error at an idea's end counts as reward (an error, not learning progress) | none by hand | 1-2 days |

Step 1 alone already changes something visible: the "random" choices begin to depend on the room and the mood, and
they hold together for longer. Each step is flashed only after the behaviour tests and a simulator day, and can be
reverted by putting the pacemakers back.

## 7. Open questions

- Should ideas be visible? A small mark on the face (the eyes glancing first, a "thought" LED) would make an idea
  readable before the act, and help the owner tell an idea from a glitch.
- Should the robot learn the owner's taste only, or also keep some ideas nobody rewarded? A pure reward loop
  narrows the behaviour over the weeks. A floor of exploration (~10 % of ideas unweighted) keeps it open.
- How much surprise is pleasant? A daily cap on "surprise" acts, starting low, tuned by the owner.
