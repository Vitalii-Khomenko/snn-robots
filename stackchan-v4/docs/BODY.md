# The body and its needs (concept, 2026-10-27)

The second stage of the road in [FOUNDATIONS](FOUNDATIONS.md) section 11: after the sense of "I did that"
([CONTINGENCY](CONTINGENCY.md)), the robot's own body. Barrett's line is the reason: *a robot whose mood follows its
real body will feel more honest than one whose mood is only set by events.*

Code: `main/v4/body_v4.cpp` (the robot), `tools/v4/body.py` (its twin and the measurements), `tools/v4/parity_body.py`.

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

- **The load:** a 10-minute average of what the body did, against a usual day (measured on 2026-10-26/27):
  - effort = 0.4 x head travel / 2.5° per s + 0.3 x calls / 4 a minute + 0.3 x arousal / 0.15;
  - load = 1 + 2 x (effort - 1), within 0.25 .. 3. A usual day is 1. Play counts double; an idle hour builds
    pressure at a quarter of the rate.
- **Awake:** S -> 1 with 12 h / load.
- **Asleep:** S drains with 3 h. Asleep means the owner's night, or a nap: the sleepy face is shown.
- **The clock** (process C) is the usual day's curve, `usual(hour)`: S of a day at load 1, 0.05 at 06:00, 0.42 at
  noon, 0.74 at 21:45.
- **Tired channel** = 0.75 x the night ramp + 2 x (S - usual(hour)), the part above zero. Only pressure above a
  usual day's is felt.
- **The diary:** `raw_stamina` = 1 - S since 2026-10-27; it was 1.0 before.
- **A reboot** keeps nothing: S starts on the usual curve once the clock is known, and the load at 1.

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

## 4. Next steps (the order may change with what the diary shows)

1. **Watch step 1 for a few days** (`body.py DAY`: the twin against `raw_stamina`, hour by hour): the load of real
   days, how often the felt pressure is above 0, naps.
2. **The body budget (allostasis, Barrett; Sterling 2012).**
   - Energy spent by effort and arousal, restored by rest and, crucially, by people. Coan's social baseline theory:
     a trusted person near lowers the cost of everything.
   - A low budget makes the world look worse (Barrett): the analyzer's threat a little higher, valence a little lower.
     That is crankiness, measured.
   - The hunger channel would carry it: it is dead now and already wired to "sick / sleepy / sad".
3. **Real interoception.**
   - A hung servo is the one real body event: the gyro sees no rotation, the rail is cycled, ~20 times an hour by day (210 by 16:00 on 2026-10-27).
     For a creature that is pain or discomfort in the neck. Now it is only a log line.
   - The USB input sagging is the same story.
   - The heat is flat: no signal there.
4. **Needs drive behaviour.** Lonely already makes the robot seek and bid. Rest-seeking (dim, quieter, slower head
   when the pressure is high) is next.
5. **Hunger, honestly.** The battery is real only off USB. If the owner ever carries the robot about, it is ready;
   otherwise hunger is the budget of step 2.

## 5. How we would know it works

- The diary: after an afternoon of play the evening's tired is above 0 and the head and voice are calmer.
- In the dark after a busy stretch it naps, and the nap drains the pressure (`raw_stamina` rises).
- A usual day is unchanged: the tired channel and the share of sleepy faces by day stay as they were.
- The owner's eye: does it look tired after play, and fresh in the morning?

## 6. Open questions

- **What is "effort" for this body?** Head travel and calls are mostly its own and the same every hour; arousal is
  what the world brings. Should being petted (pleasant arousal) tire it as much as being shaken?
- **Should a nap be allowed in a lit room?** Now only with the felt pressure very high (tired > 0.65 means
  S - usual > 0.33).
- **Is crankiness wanted?** A tired child is irritable. The owner decides whether the robot may be.
