# Foundations: what we take from the study of people (2026-10-27)

The owner, 2026-10-27: *"I keep trying to study the human myself, to understand how we are made, to build a better
robot... write these books down as the foundation for future development, with their main ideas. We will keep
working in this direction."* And: *"I tried this 8 months ago but got completely buried."*

This file is that foundation. For each source it gives:

- what it says, in plain words;
- the thoughts worth keeping;
- what it means for StackChan: what is already in the robot, and what could come next.

It is not a literature review. Only what we can use is here. The last section says how not to get buried again.

---

## 1. Jaak Panksepp: the emotions are old, and there are seven of them

**Read:**
- *Affective Neuroscience: The Foundations of Human and Animal Emotions* (1998);
- the easier *The Archaeology of Mind* (2012, with Lucy Biven).

**Main ideas**

- Deep in the brain, under the cortex, every mammal has the same few **emotional systems**. Each is a circuit with
  its own chemistry. Stimulating it electrically gives the same feeling and the same behaviour in a rat, a cat or a
  person, and animals will work to switch the pleasant ones on and the unpleasant ones off.
- **SEEKING:**
  - the eager wanting and exploring, driven by dopamine;
  - it is not pleasure itself but anticipation, curiosity, "let's see what's there";
  - it underlies all the others.
- **RAGE:** anger at being restrained or frustrated.
- **FEAR:** freeze or flee from danger.
- **LUST:** sexual desire.
- **CARE:** tenderness, looking after the young.
- **PANIC/GRIEF:** separation distress, the crying of a lost pup. Soothed by touch, opioids and oxytocin.
- **PLAY:**
  - rough-and-tumble play;
  - it needs safety, and fear or hunger switch it off at once;
  - rats tickled by a hand "laugh" (ultrasonic 50 kHz chirps) and come back for more;
  - play is how young mammals learn social rules.
- **Three levels:**
  1. primary processes: these raw feelings;
  2. secondary: learning (conditioning, what predicts what);
  3. tertiary: thinking about it, in the cortex.

  The lower levels work without the upper ones. Children born without most of their cortex still laugh, play and
  get upset.

**Thoughts worth keeping**

- Feelings are not an add-on to intelligence. They come first: the brain's oldest way to value what is happening.
- A creature needs few emotions to look alive, but they must be *real drives* that make it do things, not labels.
- SEEKING is the engine. A robot without it only reacts.

**For StackChan**

| Panksepp | In the robot today | Next |
|---|---|---|
| SEEKING | interest zone, the Muse and its curiosity, seeking light and motion (CREATIVITY, ARCHITECTURE 3) | contingency: seeking what *answers* (CONTINGENCY) |
| RAGE | anger zone: pokes, rough handling | |
| FEAR | anxiety zone: shakes, bangs, threat | |
| PANIC/GRIEF | the lonely need, "left alone" in the reward | the still face: grief when ignored by someone present (CONTINGENCY 3.5) |
| CARE | comfort zone: being petted | the robot *giving* care: soothing a sad-sounding voice? |
| PLAY | the Muse's play idea | play as a game WITH the owner, through contingency, and only when safe |
| LUST | not used | |

---

## 2. Lisa Feldman Barrett: the brain predicts, and emotions are built

**Read:** *How Emotions Are Made: The Secret Life of the Brain* (2017).

**Main ideas**

- **The brain is a prediction machine.** It does not wait for the senses. It constantly guesses what comes next,
  and the senses only correct the guess (prediction error).
- **The body budget (allostasis).** The brain's first job is to run the body: energy, water, temperature, heart,
  ahead of need. It predicts what the body will need and spends or saves in advance.
- **Interoception.** The brain senses the body's inside. That sense feels like **core affect**, two dimensions
  that are always there:
  - pleasant .. unpleasant (valence);
  - calm .. activated (arousal).

  This is Russell's circumplex (1980).
- **Emotions are constructed.** "Anger" or "fear" are not fixed circuits with fingerprints in the face or body. The
  brain makes an emotion when it explains its core affect with a concept learned from life and culture: "this
  feeling, here, now, is anger". The same body state can become anger, fear or excitement depending on the
  situation.
- **Emotional granularity.** People with more, finer emotion concepts (irritated vs resentful vs disappointed)
  cope better. Concepts are tools.
- **Feelings affect perception.** A tired body budget makes the world look more threatening.

**Thoughts worth keeping**

- Panksepp and Barrett disagree, loudly. He says there are innate emotion circuits; she says emotions are built
  from core affect and concepts. For a robot **both are useful at different levels**:
  - Panksepp for the drives that must exist from birth;
  - Barrett for how the moment is read and named.
- Mood is the body budget speaking. A robot whose mood follows its real "body" (battery, heat, effort, how long
  without contact) will feel more honest than one whose mood is only set by events.

**For StackChan**

- **Already:** the analyzer's core affect (valence, arousal) comes from channels, the main network then *picks one
  of 21 emotions* to explain it. That is very close to Barrett's construction: affect below, category above. The
  body needs (hunger, tired, lonely) are interoceptive channels.
- **Measured (BODY step 0, 2026-10-27):** the body's channels were a clock and a timer. Hunger was 0 on USB, the
  servo stamina never moved, tired came from the hour alone.
- **Built (BODY step 1):** tiredness from what the robot did (sleep pressure, section 9a).
- **Next:**
  - a real body budget: effort and arousal spent, rest and people restoring it (BODY step 2), the hung servos as
    discomfort, all as interoception;
  - prediction in the analyzer: the Muse's curiosity already has a small forward model (`sleep_v4.surprise_of`).
    The same idea in the day loop would let surprise be "what I did not predict" rather than "what is new".

---

## 3. Antonio Damasio: no reason without feeling

**Read:**
- *Descartes' Error: Emotion, Reason, and the Human Brain* (1994);
- then *The Feeling of What Happens* (1999) and *The Strange Order of Things* (2018).

**Main ideas**

- **Patients with damage to the ventromedial prefrontal cortex** keep their intelligence, memory and logic, yet
  their lives fall apart. They cannot decide; they make ruinous choices again and again. Examples: Phineas Gage,
  and Damasio's patient "Elliot". What they lost is the *feeling* attached to outcomes.
- **Somatic markers.**
  - Past outcomes leave a bodily trace: a gut feeling linked to a situation.
  - When a similar choice comes, the marker fires before any thinking. It narrows the options, steering away from
    what hurt and towards what worked.
  - In the Iowa gambling task (Bechara and Damasio), healthy people's skin starts to react to the bad decks before
    they can say why. The patients' does not.
- **Emotion vs feeling.** An emotion is the body's response (heart, face, posture). A feeling is the brain
  *perceiving* that body state. The body comes first.
- **Layers of self.**
  - A proto-self: the brain's moment-to-moment map of the body.
  - A core self: "something is happening to me now".
  - An autobiographical self: memory of one's own story.
- **Homeostasis** (2018) is the root of everything: life must keep itself within bounds, and feelings are how
  that effort becomes experience.

**Thoughts worth keeping**

- Learning from experience works through marks left on situations, not through rules.
- A self starts from a body map plus "this is happening to me". Contingency is exactly the second part.

**For StackChan**

- **Already:** the night sleep is a somatic-marker system in slow motion.
  - The day's outcomes (petting vs poking) re-weight the action heads.
  - The zones shift by experience.
  - The Muse's ideas get biases.
- **Next:**
  - markers available in the day: the contingency detector's "life" scale (CONTINGENCY 3.3) is a marker per act
    kind;
  - an autobiographical layer: a few vivid moments of each day remembered with their context and replayed at night
    (an episodic memory, see section 9).

---

## 4. Pierre-Yves Oudeyer: curiosity as learning progress

**Read:**
- Oudeyer, Kaplan & Hafner, *Intrinsic Motivation Systems for Autonomous Mental Development* (IEEE Transactions on
  Evolutionary Computation, 2007);
- Gottlieb & Oudeyer, *Towards a neuroscience of active sampling and curiosity* (Nature Reviews Neuroscience,
  2018);
- talks and the Flowers lab pages (Inria).

**Main ideas**

- **Curiosity is not a liking for the new or for the unpredictable.** What is too easy is boring. What is
  impossible to predict (noise, a TV) is a trap. The interesting is where **my predictions are getting better**:
  learning progress.
- **The Playground experiment.**
  - A Sony AIBO on a baby play mat, with an object to bite, an object to bash, and an adult robot that imitates
    its sounds.
  - With learning progress as its only motive, it went through stages by itself: first its own body, then bashing,
    then biting, then "talking" to the other robot.
  - The order of stages resembles infants'. Nobody programmed it.
- **Development organises itself** when a creature chooses its activities by learning progress: it moves on when
  something is mastered.
- **Children explore the same way.** They play longest with toys of intermediate difficulty.

**Thoughts worth keeping**

- The noise trap is real for our robot: the speech sense fires 200+ times an hour on noise (CONTINGENCY 2). A
  naive curiosity would stare at the TV forever.
- Development is a sequence. First "my body and what I cause", then "objects", then "people".

**For StackChan**

- **Already:** the Muse's curiosity term rewards ideas after which the world changed more than predicted.
- **Next:**
  - turn it from surprise into *progress*: compare the forward model's error early and late in a day or week, and
    reward the act kinds whose effects became more predictable;
  - contingency is the first stage of the sequence.

---

## 5. Contingency and the first social life (the basis of CONTINGENCY.md)

**Read:**
- John S. Watson, *Smiling, cooing, and "the game"* (Merrill-Palmer Quarterly, 1972);
- Bahrick & Watson, *Detection of intermodal proprioceptive-visual contingency as a potential basis of
  self-perception in infancy* (Developmental Psychology, 1985);
- Gergely & Watson, the *social biofeedback* model (1996, 1999);
- Tronick et al., the **still-face** studies (1978 on);
- Jean Piaget, *The Origins of Intelligence in Children* (1936; English 1952);
- Colwyn Trevarthen on primary intersubjectivity and Mary Catherine Bateson on proto-conversation (1970s).

**Main ideas**

- **Watson's mobile.** Infants who controlled a mobile by moving their head learned to do it, then smiled and
  cooed at it. Infants with an equally moving but uncontrolled mobile did not. Contingency itself is rewarding,
  and it is social: "the game" is what babies later play with people.
- **Perfect vs imperfect contingency.**
  - Until about 3 months infants prefer perfectly contingent feedback, like their own leg on a video: this is how
    they find their own body (Bahrick & Watson).
  - Then they switch to preferring high but imperfect contingency, which is what people give.
  - Self = perfect; other = high but imperfect.
- **Social biofeedback.** Parents mirror the baby's feelings, a little exaggerated (marked). That is how the baby
  learns what its own inner states are.
- **The still face.** When a responsive parent goes unresponsive, infants first try to restart the game (smiles,
  calls), then withdraw and show distress. The repair afterwards matters as much as the break.
- **Circular reactions (Piaget).**
  - Primary: repeating actions on one's own body, 1-4 months.
  - Secondary: repeating actions that made something happen in the world, 4-8 months.
  - Tertiary: varying the action on purpose to see what changes, 12-18 months, the first experiments.
- **Proto-conversation.** Baby and mother take turns with sounds and faces long before words, with pauses and
  timing like real conversation.

**Thoughts worth keeping**

- The feeling of being understood comes from timing: an answer in about a second, to *my* act.
- Ignoring hurts more than absence. A robot should feel the difference too.

**For StackChan:** all of [CONTINGENCY](CONTINGENCY.md).

---

## 6. Learning from surprise

**Read:**
- Rescorla & Wagner, *A theory of Pavlovian conditioning* (1972);
- Schultz, Dayan & Montague, *A neural substrate of prediction and reward* (Science, 1997);
- Seligman on learned helplessness (1967, 1975);
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2018, free online), for the engineering side.

**Main ideas**

- **Rescorla-Wagner.**
  - An animal learns that A predicts B not from A and B occurring together, but from **surprise**: the
    difference between what happened and what was predicted. Learning stops when B is fully predicted.
  - **Blocking:** a new cue that comes with an already-predicted outcome is not learned at all.
- **Dopamine is a prediction error.** Schultz's monkeys: dopamine neurons fire at an unexpected reward. Once a cue
  predicts the reward they fire at the cue instead, and they dip when a predicted reward fails to come. This is
  the temporal-difference error of reinforcement learning, found in the brain. It links to Panksepp's SEEKING.
- **Learned helplessness.** Animals whose actions had no effect on shocks stopped trying, even later when they
  could escape. A lack of contingency teaches passivity.

**Thoughts worth keeping**

- Count surprise, not co-occurrence. Contingency ΔP and the Rescorla-Wagner rule are the same idea.
- Guard against helplessness: being ignored must fade, and every day starts fresh.

**For StackChan**

- **Already:** the night AWR learns from reward advantage, a cousin of these rules.
- **Next:** the contingency detector's prior and its background rate (CONTINGENCY 3.3) are Rescorla-Wagner in
  practice; "ignored" fades on purpose (3.5).

---

## 7. Me and not me: the efference copy

**Read:**
- von Holst & Mittelstaedt, *The reafference principle* (1950);
- Blakemore, Wolpert & Frith, *Central cancellation of self-produced tickle sensation* (Nature Neuroscience, 1998).

**Main ideas**

- When the brain sends a motor command, it keeps a copy (the **efference copy**) and predicts the sensory
  consequences. Senses caused by one's own action (reafference) are subtracted, so the world does not seem to
  jump when the eyes move.
- **You cannot tickle yourself**: the prediction cancels the sensation. With a robot arm adding a delay of 0.2 s
  between your hand and the touch, it tickles again. "Me" is a precise, fast prediction.
- Some symptoms of schizophrenia (feeling one's own actions as caused by others) may come from a broken
  prediction of this kind.

**Thoughts worth keeping**

- A creature must know which changes it caused itself before it can notice anyone else.

**For StackChan**

- **Already, by hand:**
  - the own-voice and own-motion flags of the ear (`BrainLoop::hear`, `self_sound.h`);
  - the camera's lightOk gate during turns;
  - the 5 s own-voice mask in the reward.
- **Next:** learn it (CONTINGENCY 3.6): a sense event that follows my act always and within ~0.3 s is me.

---

## 8. Interest has a sweet spot

**Read:** D. E. Berlyne, *Conflict, Arousal, and Curiosity* (1960). The inverted U goes back to Wundt.

**Main ideas**

- Pleasure is an **inverted U** of arousal and novelty. Too little is boring; too much is unpleasant and
  frightening. The best is in between.
- Collative properties drive interest: novelty, surprise, complexity, uncertainty.

**For StackChan**

- **Already:** the analyzer's under / over numbers and the boredom vs anxiety zones (ARCHITECTURE).
- **Next:** none. This is the oldest foundation in the robot.

---

## 9. Robots that came before: Braitenberg and Breazeal

**Read:**
- Valentino Braitenberg, *Vehicles: Experiments in Synthetic Psychology* (1984). Short, playful, a must.
- Cynthia Breazeal, *Designing Sociable Robots* (2002), about Kismet (MIT).
- Rodney Brooks, *Intelligence without representation* (1991), for the layered, body-first approach.

**Main ideas**

- **Braitenberg.**
  - Tiny vehicles with two sensors wired straight to two wheels look afraid, aggressive, in love or curious.
  - *Law of uphill analysis and downhill invention:* it is much easier to build a mechanism that looks
    psychological than to analyse one from outside.
  - Simple mechanisms, combined, read as a mind.
- **Breazeal's Kismet** (late 1990s) is the closest ancestor of StackChan's design:
  - a face robot with drives (social, stimulation, fatigue) kept in a comfortable range;
  - an emotion system from appraisals (arousal, valence, stance);
  - expressive face and voice;
  - proto-dialogue turn-taking with people;
  - it regulated the interaction itself: too much stimulation made it look away, too little made it seek.
  - People treated it like an infant and naturally taught it.
- **Brooks.** Build from the body up, in layers that each work alone. Do not wait for a model of the world.

**Thoughts worth keeping**

- Our robot is in good company. Kismet's lesson is that **regulating the interaction** (seek, look away, take
  turns) makes people engage much more than any single expression does.
- Downhill invention is our method: build a small mechanism, watch the owner's reaction, keep what works.

**For StackChan**

- Kismet's drives and arousal regulation exist in the robot: needs, zones, the inverted U.
- Its turn-taking does not exist yet: CONTINGENCY step 5.

---

## 9a. Alexander Borbely: sleep pressure and the clock (the basis of BODY.md step 1)

**Read:**
- Borbely, *A two process model of sleep regulation* (Human Neurobiology, 1982);
- Borbely, Daan, Wirz-Justice & Deboer, *The two-process model of sleep regulation: a reappraisal* (J. Sleep
  Research, 2016);
- Huber, Ghilardi, Massimini & Tononi, *Local sleep and learning* (Nature, 2004).

**Main ideas**

- **Two processes decide when we are sleepy:**
  - **S, sleep pressure,** builds up while awake (adenosine) and drains in sleep, both exponentially;
  - **C, the circadian clock,** sends an alerting signal by day that holds the pressure off. This is why a rested
    adult is alert all day and sleepy in the evening (Edgar, Dement & Fuller 1993).
- **Use builds pressure.** The cortex that learned a motor task sleeps deeper in exactly that place the next night.
- **Children** build and drain the pressure faster, hence naps (Jenni & LeBourgeois 2006).

**For StackChan**

- **Built (BODY step 1, 2026-10-27):** `BodyV4` keeps S, built faster by head travel, calls and arousal. It drains
  in the night or in a nap. The tired channel feels only S above a usual day's curve: the clock holds a usual day off.
- **Rejected:** S without the clock's alerting signal. It made the robot dull from mid-morning every day.

---

## 10. The common thread: predictive processing

**Read:**
- Andy Clark, *Surfing Uncertainty* (2016), readable;
- Karl Friston, *The free-energy principle: a unified brain theory?* (Nature Reviews Neuroscience, 2010), hard.

**Main ideas**

Almost every section above is the same idea seen from different sides. The brain is a hierarchy of predictions
about the world and the body, and it is busy reducing its prediction error in two ways:

- by updating the predictions (perception, learning);
- by acting so that the world matches them (action).

Each section is one face of this:

- emotion is the prediction of the body (Barrett);
- dopamine is the prediction error of reward (Schultz);
- the self is what is perfectly predicted from my own commands (von Holst, Blakemore);
- curiosity is the reward for getting better at predicting (Oudeyer).

**For StackChan:** a direction, not a step. If one day the analyzer predicts its next channels, and the robot's
surprise, curiosity, contingency and self-other split all come from that one forward model, the parts would grow
into one brain instead of a collection of mechanisms.

---

## 11. How not to get buried again

The owner tried this road 8 months ago and got lost in it. The literature is endless and every paper opens three
more. Rules for this project:

1. **One question at a time**, from the robot's behaviour, not from the books. "It does not notice it was
   answered" comes first; then the book that answers it. Not the other way round.
2. **Every idea gets a test before it gets code**:
   - a number in the diary (like the ΔP table in CONTINGENCY 2);
   - a simulator scenario;
   - the owner's ear and eye.

   If it cannot be measured, it waits.
3. **Steps that change nothing come first** (measure, log, report). Only then does behaviour change, and it is
   reversible (the learned blob, the fallback).
4. **This file grows only with what we used.** A new source enters when a step of the robot is built on it, with
   one line in its "For StackChan".
5. **The order of development follows infants'**, as in Oudeyer's experiment:
   1. its own body and what it causes (contingency, self vs other; the body's needs: BODY.md);
   2. people's answers (still face, turn-taking);
   3. play with the owner;
   4. memory of its own story (episodes);
   5. a predicting brain.

   Nothing from a later stage before the earlier one works.

## Reading order for the owner

| | Source | Why |
|---|---|---|
| 1 | Braitenberg, *Vehicles* | an evening, and the whole approach becomes clear |
| 2 | Barrett, *How Emotions Are Made* | prediction, the body budget, how emotions are built; a Russian translation exists |
| 3 | Panksepp & Biven, *The Archaeology of Mind* | the seven drives, SEEKING and PLAY |
| 4 | Damasio, *Descartes' Error* | why a decider needs feelings |
| 5 | Breazeal, *Designing Sociable Robots* | someone already built a creature like ours: what worked |
| 6 | Oudeyer's papers and talks | curiosity and development, for when the robot should grow by itself |
| 7 | Clark, *Surfing Uncertainty* | the thread that ties it together |
