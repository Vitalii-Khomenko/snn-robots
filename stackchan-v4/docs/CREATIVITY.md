# Creativity: a "Muse" network (concept, 2026-10-26)

The owner, 2026-10-26: *"for creativity and unpredictability we probably need an additional NN that gives new
signals - work it out as a concept."* This is that concept. Nothing here is built yet. The curious glances in the
executor (ARCHITECTURE section 4) are its hand-made forerunner.

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
   readout, like light_spot's place memory. An idea that was tried recently is damped. Dropping a repeated idea
   pushes the reservoir towards unexplored regions. This is the part that makes "no two days alike" true and not
   only "no two minutes".
3. **Selection: it learns what was good.** The night sleep already scores the day: touch, voice, the owner coming
   near, set against being ignored or poked (`reward_v4.py`). The same AWR step trains the Muse's readout. Ideas
   that drew an answer from the owner come back more often, and ideas nobody cared about fade. A **curiosity**
   term comes on top: a small forward model predicts the next SNN channels. An idea after which the world turned
   out differently than predicted (learning progress, after Oudeyer) is worth more, even with nobody there.

```
 senses -> ChannelEncoder -> SNN (768 LIF) ----------------> NN (GRU policy) -> executor
                               |  mood, attention, boredom    ^
                               v                              | 8 idea inputs (replace the 3 pacemakers)
                        Muse: reservoir 64 tanh -> readout 8 -+
                               ^        |
                               |        +-> habituation of its own ideas (damps repeats)
                         night sleep: AWR on the readout from reward_v4 + curiosity (forward model error)
```

### What an idea is (the 8 outputs)

| # | Idea | What the main network learns to do with it (teacher) |
|---|---|---|
| 0-1 | where (yaw, pitch) | look there: a glance, or a slow sweep to it |
| 2 | tempo | quick, busy moves and calls vs slow and dreamy ones |
| 3 | play | a sound game: a motif repeated with variations (babble voice), a little "song" |
| 4 | show | try a face combination it rarely makes, an LED pattern, a gesture |
| 5 | stillness | be still and watch (creativity includes doing nothing on purpose) |
| 6 | social | call to the owner: a question tone, a look at where they were last seen |
| 7 | surprise | the strength of the urge; at its peak a short, unusual act (a double glance, a "dance" of nods) |

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
flash. The readout (8x64) and the habituation state go into the learned blob, which the night sleep already
writes, so the blob gains one more array.

## 5. How we would know it works

- **Behaviour entropy** in the simulator and the diary: the share of distinct 10-second "phrases" (head mode +
  face + sound style n-grams) and how often a phrase repeats within an hour. The target is clearly above today's
  level, with all behaviour tests still passing.
- **Coherence:** an idea's effect lasts. Random flicker scores high on entropy too, so the test also checks that a
  phrase holds for seconds.
- **The owner's verdict** stays the real test: does it surprise in a good way?
- **Night numbers:** how often Muse-driven acts drew an answer, and whether that share grows over the days.

## 6. Steps

| Step | What | Retrain | Rough size |
|---|---|---|---|
| 1 | Reservoir Muse in C++ and its Python twin (parity, like `channels.py`). Its 3 slowest readouts replace the pacemakers 1:1 (same range and time scales), so the teacher keeps working unchanged. Retrain the NN on Muse sequences. Measure entropy. | NN | a day |
| 2 | 8 idea outputs plus the habituation of its own ideas; teacher rules per idea, with a few new micro-acts (sound game, double glance, nod dance, "stillness"); the creativity budget | NN | 2-3 days |
| 3 | Night learning of the readout from reward_v4 (AWR, with the same gates: KL, behaviour tests) | none by hand | a day |
| 4 | Curiosity: a forward model of the SNN channels; its error at an idea's end counts as reward | none by hand | 1-2 days |

Step 1 alone already changes something visible: the "random" choices begin to depend on the room and the mood, and
they hold together for longer. Each step is flashed only after the behaviour tests and a simulator day, and can be
reverted by putting the pacemakers back.

## 7. Open questions

- Should ideas be visible? A small mark on the face (the eyes glancing first, a "thought" LED) would make an idea
  readable before the act, and help the owner tell an idea from a glitch.
- Should the robot learn the owner's taste only, or also keep some ideas nobody rewarded? A pure reward loop
  narrows the behaviour over the weeks. A floor of exploration (~10 % of ideas unweighted) keeps it open.
- How much surprise is pleasant? A daily cap on "surprise" acts, starting low, tuned by the owner.
