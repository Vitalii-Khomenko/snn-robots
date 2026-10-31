# The body and its needs (concept, 2026-10-27)

The second stage of the road in [FOUNDATIONS](FOUNDATIONS.md) section 11: after the sense of "I did that"
([CONTINGENCY](CONTINGENCY.md)), the robot's own body. Barrett's line is the reason: *a robot whose mood follows its
real body will feel more honest than one whose mood is only set by events.*

Steps 0-2 are built (2026-10-27). Code: `main/v4/body_v4.cpp` (the robot), `tools/v4/body.py` (its twin and the measurements), `tools/v4/parity_body.py`.

## 1. Why this, and why now

The analyzer has had three need channels since the start of v4: hunger, tired, lonely. They feed its drives
(`hunger_n`, `sleepiness`, `loneliness`), and the teacher turns the drives into behaviour (sleepy above 0.65, sick
or sad when hungry, sad or bored when lonely and under-stimulated). The question was whether a real body ever
fills them.

## 2. What the diary says (step 0, measured 2026-10-27)

Three whole days (2026-10-25 .. 27), `body.py --measure`:

| Channel | Made from | What it did |
|---|---|---|
| hunger | 1 - battery | **0.00 every tick.** The robot is always on USB; the battery reads 99-100 %. |
| tired | 0.75 x the night ramp + 0.25 x (1 - servo stamina) | **a clock.** Stamina was 1.00 every tick: a full reserve was 120 000 degrees of travel, the robot moves ~9 000 an hour and rests back in 5 minutes. |
| lonely | minutes without contact | the only real need. Until 2026-10-27 morning it hardly moved (mean 0.006 by day): **the robot's own voice reset it**, 3.4 calls a minute. The voice check of CONTINGENCY step 2 ended that; now 0.14 by day with the owner silent at the desk. |

The rest of the body is flat too:
- the PMU die is 53-56 °C day and night;
- the CPU is busy 31-37 %;
- the servo rail is on 40-55 % of the time by day.

Its day is flat as well. From 06:00 to 21:00 every hour looks the same:
- ~150° of head travel a minute;
- 3.4 calls a minute (5.8 since contingency step 3);
- arousal 0.14-0.19.

At 21:45 it falls asleep by the clock, whatever the day was.

**So the body is a clock and a timer.** A child gets tired from a busy day, naps, wakes fresh, and is cranky when
overtired; this robot does not, whatever happened.

## 3. Step 1: sleep pressure (built 2026-10-27)

**From people: Borbely's two-process model** (1982; Borbely, Daan, Wirz-Justice & Deboer, *J. Sleep Res.* 2016):

- **Process S, sleep pressure,** builds up while awake (adenosine) and drains in sleep. Adults: ~18 h up and
  ~4 h down. Children build and drain faster, hence their naps (Jenni & LeBourgeois 2006).
- It builds faster with use. Learning a motor task deepens the next sleep in exactly the cortex that learned it
  (Huber, Ghilardi, Massimini & Tononi, *Nature* 2004).
- **Process C, the circadian clock,** sends an alerting signal that holds the day's pressure off (the opponent
  process, Edgar, Dement & Fuller 1993). So a rested adult feels alert all day, and sleepiness arrives in the evening.

**In the robot** (`BodyV4`, after the executor every tick):

- **The load:** a 10-minute average of what the body did, against a usual day (2026-10-29 07-16 h, with the looser
  voice budget: 3.0°/s, 7.6 calls a minute, arousal 0.14; 2.3, 5.5 and 0.16 from 10-27 before - with them a usual day
  of 10-29 was a load of 1.38 and the robot felt tired from mid-morning):
  - effort = 0.4 x head travel / 3.0° per s + 0.3 x calls / 7.6 a minute + 0.3 x arousal / 0.14;
  - load = 1 + 2 x (effort - 1), within 0.25 .. 3. A usual day is 1. Play counts double; an idle hour builds
    pressure at a quarter of the rate.
- **Awake:** S -> 1 with 12 h / load.
- **Asleep:** S drains with 3 h. Asleep means the owner's night, or a nap: the sleepy face shown **and the body
  quiet** - no call and the head still for 20 s (2026-10-28, below).
- **The clock** (process C) is the usual day's curve, `usual(hour)`: S of a day at load 1, 0.05 at 06:00, 0.42 at
  noon, 0.74 at 21:45.
- **Tired channel** = 0.75 x the night ramp + 2 x (S - usual(hour)), the part above zero. Only pressure above a
  usual day's is felt.
- **The diary:** `raw_stamina` = 1 - S since 2026-10-27; it was 1.0 before.
- **A reboot** keeps S and the budget (2026-10-28, below); with nothing saved S starts on the usual curve once the
  clock is known. The load starts at 1.

**What this changes:**

- **A usual day:** nothing. The same tired channel as before, 0 by day.
- **A busy day:**
  - 4 hours of play at effort 1.5 (load 2) gives tired ~0.3 by noon;
  - the analyzer is calmer then (its arousal drops by 0.3 x sleepiness);
  - in the dark, sleepiness passes 0.65 (tired + 0.25), so it naps;
  - the nap drains S below the usual curve, and it wakes.
- **A quiet day:** pressure below usual is not felt, and the evening is the clock's.
- **The night still belongs to the clock and the owner.** Sleep starts 21:00-21:45 and the night rules are
  unchanged, so it is never up later.
- **The morning waits for the owner (2026-10-31).** On Saturday 10-31 the owner slept on next to the robot in the
  dark; at 06:00 the night ended by the clock, and from 06:26 it called 9 times a minute until the owner woke at 07:13
  and turned it down (Sunday 10-25: calls from 06:34, the owner's first touch 06:52). Now after 06:00 it sleeps on -
  night rules, and the tired channel held at the night's 0.75 (`ChannelEncoder::Raw::morning`) - until someone comes:
  a touch, the room lit for 20 s, someone near (> 0.3), a voice for a while (the voice channel > 0.3), someone seen
  moving; at 09:00 at the latest. Then it wakes over 3 minutes (a reboot by day starts awake).
  - The diary, 7 mornings 10-25..31: on the weekdays the first of these came at 06:00-06:21, when the robot called
    first anyway (no change); on the weekend 18 and 51 minutes after its first call. None came in the night hours
    (00:30-06:00), the light only when it was really on.
  - The log: `morning: 06:00, dark and quiet - sleeps on until someone comes`, `morning: up at 07:13 (a touch)`.
  - The simulator: `sleep_in` (06:30, dark, nobody up; the owner's hand at 07:00: 0 calls before it, tired 0.75, 63
    calls in the 10 minutes after, awake 5 minutes after the hand) and `early_riser` (06:05, the light on at 06:08:
    silent before, talks after).
  - Not a diary column: `diary.morning_of` rebuilds it from the night flag (the channels' twin, `channels.encode`'s
    `morning`).

**First try, rejected (offline, the analyzer's C++ on 2026-10-26 with a new tired channel):** tired = 0.75 x night
+ 0.6 x S, with no alerting signal.
- From mid-morning the analyzer's arousal fell by a third: 0.165 -> 0.11 at 10:00, with S only 0.32.
- In the afternoon dark (the owner out, lights off) sleepiness went above 0.65 for 30-100 % of every hour.

A robot dull every day is the opposite of what the owner wants. That is why the felt part became the pressure
*above* the usual curve.

**Tests:**
- `parity_body.py`: C++ == twin on a whole day, 6e-8.
- `parity_channels.py`: the tired channel over its whole range.
- `behavior_test_v4.py`: all pass. A scenario at 14:00 starts on the usual curve and stays there, so tired = 0.
- Live after the flash: `sleep_pressure=0.59` at 16:05 = `usual(16.05)` (the `stats:` line, with `load=`).

## 4. Step 2: the body budget (built 2026-10-27 evening)

**From people:**
- **Allostasis** (Sterling 2012; Barrett): the brain runs a budget of the body's resources. Effort and arousal
  spend it, rest and sleep refill it. A low budget is felt as unpleasant, and the world looks worse.
- **Social baseline theory** (Coan & Sbarra 2015): a trusted person near is the brain's default. Alone, everything
  costs more. Holding a partner's hand damped the brain's response to a threat of shock (Coan, Schaefer & Davidson
  2006).

**In the robot** (`BodyV4`, the same tick as the sleep pressure):

| | |
|---|---|
| spent by the load | empty in 50 h at load 1 (-0.3 over a usual 15 h day); **half as fast with someone near or talking** |
| spent by fear and too much | the analyzer's threat or over above 0.3: empty in 10 min at full |
| spent by an ache | each servo rail power cycle (a hung servo, a sagging supply) -0.0007. There are ~20 an hour by day, 50 on a bad morning: kept small on purpose, or the robot would be low all day for its hardware's sake |
| refilled by sleep | the night or a nap: 2 h to the full |
| refilled by a gentle hand | each pet or tap (as the contingency detector counts one, not a poke) fills 6 % of what is missing |
| felt | hunger = 1 - budget / 0.7 (or 1 - battery off USB, whichever is more): a budget above 0.7 is not felt |
| diary | `raw_budget` (a new column: 174 values); the `stats:` line shows `budget=`; a reboot keeps it (2026-10-28; with nothing saved it starts at 0.85) |

**Measured before the flash** (the twin on three whole days): the budget stayed between 0.61 and 0.99, so hunger
was at most 0.13. A usual day is unchanged.
- The lowest point was the morning of 2026-10-25: 50 power cycles an hour and the owner's claps.
- Petting at 13:00 that day brought it from 0.66 back to 0.86.

**Found on the way:** the innate tolerance of 2026-10-25 squeezed every channel with an alarm zone, hunger too. A
budget of 0.25 reached the analyzer as 0.41 instead of 0.64. Hunger is the body's own need, not something the world
does to the robot, so it is no longer squeezed (`zone_shift.h`, `learned.tolerant`).

**Simulator** (`hard_day` against `hard_day_control`, in `behavior_test_v4.py`):
- **The hard stretch:** 14 minutes alone, a bang every 10 s and a shake every minute. The budget falls 0.85 -> 0.26
  and hunger reaches 0.62.
- **The quiet stretch that follows:** the analyzer's valence is 0.45 against the control's 0.49. The faces barely
  differ: alone, both are bored and sleepy.
- **Comfort:** 6 minutes of strokes bring the budget 0.26 -> 0.59. The control stays at 0.85-0.91.
- **2026-10-28:** with the network's head input as it was trained (ARCHITECTURE section 4) the robot is frightened
  less in the hard stretch: the low is 0.36-0.53 over 12 seeds (0.23-0.42 before); the test asks <= 0.6.
- **Not yet visible:** the "unwell" faces need hunger above 0.75 (the teacher's rule), a budget below 0.17. A hard
  stretch colours the mood but does not change the face.

Live after the flash (16:55): the robot equals the twin (budget 0.0002), load 0.95.

## 4a. A rest is a state, not a face; the body outlives a reboot (2026-10-28, after an audit)

**The rest.** Asleep was "the night, or the sleepy face shown". The face was enough, whatever the body did.
- In the diary, by day under a sleepy face: 1.48 h on 10-27 and 1.10 h on 10-28.
- The robot made 187 and 289 calls an hour then, and moved its head 1.1 and 1.9 deg/s (a usual day: 2.3).
- All that time the budget refilled and the pressure drained as in sleep.

Now a nap is the sleepy face **and** a quiet body: no call and the head still (under 0.1 deg a tick) for 20 s
(`BodyV4::resting()`, the twin's `Body.resting`). A call, a head move or another face ends it at once. Only 44 % (10-27)
and 19 % (10-28) of the sleepy-face time by day was such a rest.

**The body across a reboot.**
- **Before:** every boot started the budget at 0.85 and the pressure on the usual curve, whatever the day had been.
- **Saving:** the robot saves both every 10 minutes into NVS (`RobotRuntime::keepBody`, ~150 short writes a day).
- **Restoring:** once the clock is known it takes them back, and counts the time it was off as rest.
  - The log line: `body: restored after N s off: saved pressure P budget B -> ...`.
  - The twin (`body.py`) reads it and does the same.
- **On the robot** (16:53, a reboot 197 s after the save): `saved pressure 0.613279 budget 0.843280 -> pressure
  0.602194 budget 0.847510`, as the twin computes; on the new firmware the twin equals the robot within 3e-4 (the
  diary's float16).
- **Not taken:** a state older than two days, or from the future.
- **Lost:** the last minutes before a power loss.

**Measured on the twin (whole days, one body across the reboots, as with saving).**
- On 10-28 the budget's low by day is 0.64 with the real rest, against 0.73 with the face alone.
- Hunger is felt in 21 % of the day, but at most 0.09 and 0.009 on average.
- 10-25 .. 10-27: no difference to speak of.
- **No retuning.** Without the saving, each boot that day started at 0.85 instead of the night's ~1.0, and the
  afternoon fell to 0.55. With it, a usual day stays almost unfelt, as step 2 wanted.

**Simulator:** all behaviour tests pass. `hard_day`: the low is 0.48 (≤ 0.6), the refill +0.22. `hard_day_control`:
the low is 0.84.

**Still open (the audit):** hunger mixes two causes in one channel, the battery (only off USB) and the budget. The
network cannot tell them apart. While the robot lives on USB the battery part is always 0. Splitting them needs a
new channel and a retrain.

## 5. Next steps (the order may change with what the diary shows)

1. **Watch steps 1-2 for a few days** (`body.py DAY`: the twin against `raw_stamina` and `raw_budget`, hour by
   hour): the load of real days, the felt pressure, naps, the budget's lows and what refilled it.
2. **A low budget in behaviour.** Now it only lowers the analyzer's valence. **Tried 2026-10-27 evening, not
   deployed.** The attempt is saved as `tools/v4/out/body_step3_attempt_2026-10-27.patch`, its training log as
   `out/retrain_nn_2026-10-27_body.log`.
   - **Teacher rules:**
     - hunger_n > 0.4 with someone around: asking calls every 12 s, a look to them, a sad / shy face;
     - the same alone: babbles half as often, bored or sleepy when nothing happens;
     - sleepiness 0.35-0.65 by day: babbles up to half as often, gestures slower.
   - **Training lives:** 20 % of the synthetic lives got a tired day and 20 % a low budget.
   - **The network learned it worse than the old one:**
     - agreement with the teacher 0.83 -> 0.74 (real lives 0.78 -> 0.65);
     - the still face failed (low faces 0.17, wanted >= 0.2), and "very happy" crowded the faces;
     - in `hard_day` the robot did not get quieter (calls 4.9 against 4.4 a minute).
   - **The analyzer itself is fine:** a budget of 0.26 reads hunger_n 0.70. Above 0.8 of the channel it falls
     back (1.0 -> 0.46): it has never seen that. The next try needs fewer, clearer rules and a retrain of the
     analyzer with real lows.
3. **Real interoception.** The hung servos are already an ache in the budget; the USB sag too. The heat is flat: no
   signal there. The hangs themselves are a hardware problem worth its own look: ~20 an hour.
4. **Needs drive behaviour.** Lonely already makes the robot seek and bid. Rest-seeking (dim, quieter, slower head
   when the pressure is high) is next.
5. **Hunger, honestly.** The battery is real only off USB. If the owner ever carries the robot about, it is ready.

## 6. How we would know it works

- The diary: after an afternoon of play the evening's tired is above 0 and the head and voice are calmer.
- In the dark after a busy stretch it naps, and the nap drains the pressure (`raw_stamina` rises).
- A usual day is unchanged: the tired channel and the share of sleepy faces by day stay as they were.
- The owner's eye: does it look tired after play, and fresh in the morning?
- After a hard stretch the budget is low and the robot's valence lower; a hand refills it (`raw_budget`).

## 7. Open questions

- **What is "effort" for this body?** Head travel and calls are mostly its own and the same every hour; arousal is
  what the world brings. Should being petted (pleasant arousal) tire it as much as being shaken?
- **Should a nap be allowed in a lit room?** Now only with the felt pressure very high (tired > 0.65 means
  S - usual > 0.33).
- **Is crankiness wanted?** A tired child is irritable. The owner decides whether the robot may be. (Step 2 chose
  no: a low budget lowers valence, not the threshold of anger.)
- **Should talking to it refill the budget too,** or only touch? Now a voice nearby only halves the costs.
