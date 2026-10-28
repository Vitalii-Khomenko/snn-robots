# Episodic memory: a story of its own (concept, 2026-10-28)

The third stage of the owner's road (2026-10-27: contingency, then the body, then episodic memory) and the audit's
A15: *"after a reboot the robot keeps nothing of its recent history; there is no episodic memory and no model of a
particular person"* ([AUDIT](AUDIT_2026-10-28_RU.md)). The rule of [FOUNDATIONS](FOUNDATIONS.md): measure first, one
question at a time, and before any code say how the robot's behaviour will change.

## 1. What the diary says (step 0, measured 2026-10-28)

`tools/v4/episodes.py 2026-10-25 .. 28`. A **moment** is an outcome the night already scores (`reward_v4`: a pet, a
tap, talk, someone near, a shake, a tilt, a poke, being left alone) or the analyzer startled (a surprise or threat
onset). Its **context**: the hour, the room lit or dark, someone around (near, a voice or motion in the last 2
minutes), the robot's own act in the 4 s before.

| Day | Moments by day | An hour | Kinds |
|---|---|---|---|
| 10-25 | 1045 in 15.2 h | 69 | startle 573, tap / talk 353, two at once 72, pet 41, shake 6 |
| 10-26 | 546 in 15.4 h | 36 | tap / talk 233, startle 202, two at once 100, pet 9, shake 2 |
| 10-27 | 706 in 15.3 h | 46 | startle 263, two at once 211, tap / talk 183, pet 47, shake 2 |
| 10-28 | 520 in 11.5 h | 45 | startle 313, two at once 162, tap / talk 37, pet 8 |

What this shows:

1. **A memory must choose.** Forty to seventy moments an hour, most of them small startles and short talk, are not a
   story. The vivid ones are rare: pets 8-47 a day, shakes 2-6. A person remembers a few moments of a day, not every
   sound.
2. **Good moments have a context.** Someone was around for 71-98 % of them, the room lit for 80-88 %, and 24-73 % came
   right after the robot's own act (more since the contingency, 10-27).
3. **Yesterday predicts today, for the good moments.** The rate of good moments per context (hour x lit x someone
   around) on one day predicts the next day's better than the plain daily rate. The error is 12.8 against 18.8 an
   hour: when people come, they come at similar hours and conditions. For the bad moments it does not help (7.5
   against 6.6): shakes and pokes are rare and scattered.

So a memory worth having would know *when and in what situation good things usually happen*, and keep a few vivid
moments with their context.

## 2. From people

- **Episodic memory** (Tulving 1972): what happened, where and when, as one's own experience. It is distinct from
  knowing facts.
- **Emotion picks what is kept** (McGaugh 2004): arousal at the moment strengthens the memory of it. That is why we
  remember the shock and not the drive home.
- **Sleep replays the day** (Wilson & McNaughton 1994; replay in the rat hippocampus). Salient and rewarded
  experiences are replayed more (prioritized replay), and the night consolidates them into lasting knowledge.
- **Anticipation** (Pavlov; Schultz's dopamine neurons, FOUNDATIONS 6): a context that reliably came before something
  good is itself felt as a promise. An infant who is fed at the same hour fusses before it.

## 3. The design (a proposal; the steps change nothing until the owner agrees)

1. **Choose a few moments.** A day keeps the 20-40 most vivid moments: strong outcomes and big startles, ranked by
   the outcome's size and the analyzer's arousal at the moment (McGaugh). Each keeps:
   - its context: hour, lit, someone around, the place the head pointed;
   - the robot's act before it;
   - the outcome over 30 s;
   - the mood before and after.

   The PC keeps them from the diary; the robot keeps a small ring of the last days' (NVS, a few KB, written at night
   by the sleep, never per tick).
2. **Expectation, a new input of the main network.** "How good this situation usually is": the rate of good moments
   in the present context over the last days (the measure of section 1, which works for good moments).
   - What it changes: at the hour and in the light when someone usually comes and pets it, the robot is more
     expectant (looks to where people were, bids sooner); where nothing good usually happens, calmer.
   - Cost: one NN input and a retrain. The teacher learns a rule: high expectation and nobody there yet means look
     out and bid.
3. **The night replays the vivid moments first.** The decisions near a vivid moment weigh more in `train_heads`
   (prioritized replay), and the report lists the day's few moments, so the owner can read the robot's day.
4. **A face for a memory** (later, the owner's choice): when a context comes back that once went very well or very
   badly, a short sign - a glance to where it happened, a little "remembering" face.

## 4. How we would know it works

- **The diary:** the expectation input rises before good moments, more than at random times (the check of section 1,
  live).
- **The simulator:** a scenario of three "days" (a person who comes and pets at the same hour each day, between
  reboots, with the memory kept). On the third day the robot looks to the door and bids before the person comes,
  more than in a control where the person comes at random hours.
- **The owner:** does the robot seem to know you, and your habits?

## 5. Open questions for the owner

- Should the robot show that it remembers (a sign, step 4), or only behave a little differently?
- What should be remembered beyond touch and voice: places in the room (where the lamp is, where people sit)?
- Should bad moments be remembered too (a wary look after being shaken there)? The diary says they do not repeat by
  context, so a memory of them would mostly be wrong.

## 6. Steps

| Step | What | Changes behaviour? |
|---|---|---|
| 0 | Measure the moments and whether yesterday predicts today (`episodes.py`) | no (done, section 1) |
| 1 | The day's vivid moments in the night report; the expectation computed from the diary, written in the report | no |
| 2 | The expectation as a network input; the teacher's rule; retrain; the three-day simulator scenario | yes |
| 3 | Prioritized replay at night | slowly |
| 4 | A sign of remembering (the owner decides) | yes |
