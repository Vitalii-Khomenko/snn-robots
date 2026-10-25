"""The teacher of brain v4's main network: from the spiking analyzer's 32 numbers to one emotion and everything that
goes with it - voice, LED colour and pattern, head (gaze, nod "yes", shake "no", flinch, ...). BRAIN_V4_PLAN.txt
sections 6-7 and the owner's rules: talks often (child-like babble), every touch is answered, petting never angers,
quiet at night.

The teacher has a memory (the shown emotion and how long, when it last spoke / gestured, the open touch) and its
random choices are read from the three pacemaker inputs (pace_fast / pace_slow / pace_jitter) that the network sees
too - a random choice the network cannot see could not be learned (v3 lesson).

Training metadata only: the robot runs the trained network, never this code.
"""
import numpy as np

from spec_v4 import (EMOTIONS, HEAD_MODES, LED_COLORS, LED_PATTERNS, NN_OUT, PREV, SNN_OUT, SOUND_STYLES)

DT = 0.1  # the main network runs at 10 Hz
S = {n: i for i, n in enumerate(SNN_OUT)}
O = {n: i for i, n in enumerate(NN_OUT)}
P = {n: i for i, n in enumerate(PREV)}

# per emotion: valence, arousal (for the feedback inputs), voice style, register, LED colour, LED pattern
EMO = {
    "normal":     (0.55, 0.30, "calm", 0.5, "white", "steady"),
    "happy":      (0.80, 0.55, "play", 0.65, "yellow", "breathe"),
    "very_happy": (0.90, 0.80, "play", 0.75, "yellow", "wave"),
    "laughing":   (0.90, 0.85, "play", 0.8, "orange", "pulse"),
    "sad":        (0.20, 0.25, "calm", 0.3, "blue", "breathe"),
    "crying":     (0.10, 0.50, "calm", 0.4, "blue", "pulse"),
    "sleepy":     (0.45, 0.05, "calm", 0.2, "blue", "breathe"),
    "surprised":  (0.55, 0.85, "ask", 0.85, "white", "flicker"),
    "angry":      (0.10, 0.90, "grumble", 0.15, "red", "flicker"),
    "annoyed":    (0.25, 0.65, "grumble", 0.3, "orange", "pulse"),
    "love":       (0.95, 0.45, "calm", 0.55, "pink", "pulse"),
    "shy":        (0.70, 0.35, "calm", 0.6, "pink", "breathe"),
    "wink":       (0.80, 0.55, "play", 0.7, "yellow", "pulse"),
    "curious":    (0.65, 0.55, "ask", 0.65, "cyan", "wave"),
    "confused":   (0.45, 0.50, "ask", 0.5, "violet", "wave"),
    "thinking":   (0.55, 0.35, "ask", 0.45, "cyan", "breathe"),
    "bored":      (0.35, 0.15, "calm", 0.3, "blue", "breathe"),
    "scared":     (0.10, 0.90, "ask", 0.9, "violet", "flicker"),
    "dizzy":      (0.35, 0.70, "ask", 0.6, "violet", "wave"),
    "cool":       (0.70, 0.35, "calm", 0.35, "cyan", "steady"),
    "sick":       (0.20, 0.15, "grumble", 0.2, "violet", "breathe"),
}
assert list(EMO) == EMOTIONS
MIN_SHOW = {"surprised": 1.2}
DEFAULT_MIN_SHOW = 1.5


# How the head moves in each mood by day (2026-10-25, the owner: in an angry or sad mood the head did not move at all -
# every emotion moves, each in its own way): the head mode, the yaw amplitude, the pitch centre and amplitude
# (gaze pitch > 0 is down), how much of the wandering follows the fast pacemaker (~3 s; the rest the slow one, ~60 s)
# and how much the jitter (~0.5 s), and where to: "wander" around, "away" from whoever it attends to (cross, shy).
# The gaze is a function of the pacemakers the network sees, so the network can learn it exactly; the executor turns
# it into a move at most every 2.5 s and only for a visible change.
MANNER = {
    #              mode    yaw   pitch  p.amp  fast  jitter  kind
    "normal":     ("scan", 0.45, 0.00, 0.15, 0.6, 0.0, "wander"),   # calm looking around
    "happy":      ("scan", 0.45, -0.15, 0.20, 0.8, 0.1, "wander"),  # lively, a little up
    "very_happy": ("look", 0.55, -0.25, 0.25, 0.9, 0.2, "wander"),
    "laughing":   ("look", 0.35, -0.30, 0.45, 0.7, 0.4, "wander"),  # bobbing with the laugh
    "sad":        ("scan", 0.30, 0.65, 0.10, 0.4, 0.0, "wander"),   # the head hangs, sways slowly
    "crying":     ("scan", 0.35, 0.75, 0.15, 0.6, 0.1, "wander"),
    "sleepy":     ("scan", 0.20, 0.55, 0.20, 0.3, 0.0, "wander"),   # nodding off by day
    "surprised":  ("look", 0.60, -0.35, 0.15, 0.8, 0.3, "wander"),  # quick looks, up
    "angry":      ("look", 0.70, 0.15, 0.10, 0.8, 0.2, "away"),     # turns away, abrupt
    "annoyed":    ("scan", 0.55, 0.10, 0.10, 0.7, 0.1, "away"),
    "love":       ("scan", 0.25, -0.20, 0.15, 0.6, 0.0, "wander"),  # a gentle sway
    "shy":        ("scan", 0.45, 0.50, 0.15, 0.6, 0.0, "away"),     # looks away and down
    "wink":       ("look", 0.40, -0.10, 0.15, 0.8, 0.1, "wander"),
    "curious":    ("look", 0.75, -0.10, 0.30, 0.9, 0.1, "wander"),  # wide, inquisitive
    "confused":   ("look", 0.50, 0.00, 0.25, 0.6, 0.4, "wander"),   # this way, that way
    "thinking":   ("scan", 0.40, -0.45, 0.15, 0.5, 0.0, "wander"),  # up and aside
    "bored":      ("scan", 0.85, 0.20, 0.15, 0.3, 0.0, "wander"),   # a slow wide wander
    "scared":     ("look", 0.60, -0.10, 0.15, 0.6, 0.4, "wander"),  # nervous glances
    "dizzy":      ("scan", 0.50, 0.00, 0.35, 0.5, 0.5, "wander"),   # wobbling
    "cool":       ("scan", 0.50, 0.00, 0.08, 0.4, 0.0, "wander"),   # a slow cool sweep
    "sick":       ("scan", 0.25, 0.60, 0.10, 0.3, 0.0, "wander"),
}
assert list(MANNER) == EMOTIONS
# 2026-10-25 (the owner): the head stayed within ~15 deg of the centre (the network's gaze p99 0.37). The yaw amplitudes
# are 1.4x wider and the wandering is shaped so it stays out at the sides (sign * sqrt): a look around the room, not a
# wobble round the centre.
GAZE_WIDE = 1.4
# Something moving in view: the head turns to it. att_x is where in the picture (-1..1, the camera's ~62 deg wide), so
# the direction is the head's own yaw plus half of it (gaze units: 1 = 60 deg).
ATT_LOOK = 0.12
CAM_HALF = 0.5
# ... and keeps watching that place a while after it stopped (2026-10-25 evening: someone who walked and then stood still
# vanished from the motion sense in 1-4 s and the head wandered off; the simulator's walker was found less than by
# chance). The network has the memory for it (its own head yaw + the GRU).
WATCH_S = 10.0


def seen_yaw(head_yaw, ax):
    """The gaze yaw of what is at picture position ax while the head is at head_yaw."""
    return float(np.clip(head_yaw + ax * CAM_HALF, -1, 1))


def manner(emo, s, pace, head_yaw=0.0, watch=None):
    """-> (head mode, gaze) of the mood's own head motion (MANNER). Someone in view: the head stays on them and moves
    a little around them - unless the mood turns away from them. watch: (yaw, pitch) of where something was seen
    lately, the head stays round there."""
    mode, amp, pc, pamp, fast, jit, kind = MANNER[emo]
    pf, ps, pj = pace
    amp = min(1.0, amp * GAZE_WIDE) * (0.7 + 0.3 * (ps + 1) / 2)  # the slow pacemaker varies how wide
    wy = float(np.clip(fast * pf + (1 - fast) * ps + jit * pj, -1, 1))
    wy = float(np.sign(wy) * np.sqrt(abs(wy)))
    wp = float(np.clip(0.6 * ps - 0.4 * pf + jit * pj, -1, 1))
    pitch = pc + pamp * wp
    att, ax, ay = s[S["att_strength"]], s[S["att_x"]], s[S["att_y"]]
    if kind == "away":
        if att > ATT_LOOK and abs(ax) > 0.05:
            side = -1.0 if seen_yaw(head_yaw, ax) > 0 else 1.0  # the other way from them
        else:
            side = 1.0 if wy >= 0 else -1.0
        yaw = side * amp * (0.6 + 0.4 * abs(wy))
    elif att > ATT_LOOK:
        yaw = seen_yaw(head_yaw, ax) + 0.2 * amp * wy
        pitch = ay * 0.4 + 0.35 * (pitch - pc) + 0.5 * pc
    elif watch is not None:
        yaw = watch[0] + 0.2 * amp * wy
        pitch = watch[1] + 0.35 * (pitch - pc) + 0.5 * pc
    else:
        yaw = amp * wy
    return mode, (float(np.clip(yaw, -1, 1)), float(np.clip(pitch, -1, 1)))


def pick(options, u):
    """Deterministic choice from a pacemaker value u in [-1, 1] (the network sees u)."""
    k = int((np.clip(u, -0.999, 0.999) + 1) / 2 * len(options))
    return options[k]


class Teacher:
    def __init__(self):
        self.reset()

    def reset(self):
        self.emo = "normal"
        self.emo_age = 10.0
        self.since_sound = 10.0
        self.since_gesture = 30.0
        self.touch_open = False      # a touch is waiting for its answer
        self.touch_age = 0.0
        self.since_touch = 10.0  # s since the last touch onset
        self.prev_touch = 0.0
        self.prev_screen = 0.0
        self.prev_sur = 0.0
        self.prev_over = 0.0
        self.prev_threat = 0.0
        self.head = (0.0, 0.0)
        self.watch, self.watch_t = (0.0, 0.0), 0.0  # where something was seen in view, s left to keep watching it
        self.led_level = 0.3
        self.gesture = None          # (mode, ticks left)
        self.speak_ticks = 0

    # ---- what the network sees of the past (PREV), from the executed state
    def prev_inputs(self, pace):
        v, a = EMO[self.emo][0], EMO[self.emo][1]
        x = np.zeros(len(PREV), np.float32)
        x[P["prev_valence"]], x[P["prev_arousal"]] = v, a
        x[P["emo_age"]] = min(self.emo_age / 20.0, 1.0)
        x[P["since_sound"]] = min(self.since_sound / 30.0, 1.0)
        x[P["since_gesture"]] = min(self.since_gesture / 30.0, 1.0)
        x[P["head_yaw"]], x[P["head_pitch"]] = self.head
        x[P["led_level"]] = self.led_level
        x[P["touch_answered"]] = 0.0 if self.touch_open else 1.0
        x[P["pace_fast"]], x[P["pace_slow"]], x[P["pace_jitter"]] = pace
        return x

    def choose_emotion(self, s, pace, events):
        """-> (emotion, urgent): what this moment calls for."""
        pf, ps, pj = pace
        val, aro, under, over, threat, sur = (s[S[n]] for n in ("valence", "arousal", "under", "over", "threat",
                                                                 "surprise"))
        sleepy, lonely, hunger = s[S["sleepiness"]], s[S["loneliness"]], s[S["hunger_n"]]
        touch = max(s[S["sal_touch_head"]], s[S["sal_touch_screen"]])
        touch_hed = s[S["hed_touch_head"]] if s[S["sal_touch_head"]] >= s[S["sal_touch_screen"]] else \
            s[S["hed_touch_screen"]]
        # urgent states first
        if threat > 0.45:
            shaken = s[S["sal_shake"]] > s[S["sal_tilt"]]
            return ("dizzy" if shaken and pj > -0.3 else "scared"), True
        if events["touch_onset"] or touch > 0.12:
            if touch_hed < 0.4:  # poked, drummed on, touched while frightened
                if over > 0.62:
                    return "angry", events["touch_onset"]
                return pick(["annoyed", "annoyed", "confused"], pj), events["touch_onset"]
            if s[S["sal_touch_screen"]] > s[S["sal_touch_head"]]:
                return pick(["happy", "wink", "shy", "laughing", "surprised"], pj), events["touch_onset"]
            return pick(["love", "happy", "shy", "laughing", "love", "very_happy"], pj), events["touch_onset"]
        if events["surprise_onset"]:
            return "surprised", True
        if over > 0.62:
            return "angry", True
        if over > 0.4:
            return "annoyed", events["over_onset"]
        if sleepy > 0.65:
            return "sleepy", False
        if hunger > 0.75:
            return pick(["sick", "sleepy", "sad"], ps), False
        if under > 0.42 and lonely > 0.5:
            return pick(["sad", "bored", "sad", "crying"] if under > 0.6 else ["bored", "sad", "bored"], ps), False
        if under > 0.3:
            return pick(["bored", "thinking", "bored", "sleepy"], ps), False
        voice = s[S["sal_voice"]]
        motion = s[S["sal_motion"]]
        if voice > 0.15 and s[S["hed_voice"]] > 0.45:
            return pick(["happy", "curious", "thinking", "cool", "very_happy"], ps), False
        if motion > 0.1 and s[S["att_strength"]] > ATT_LOOK:
            return pick(["curious", "curious", "thinking", "surprised"], ps), False
        if s[S["att_strength"]] > 0.25:  # something new in view to look at (a new bright spot: the camera, 2026-10-25)
            return pick(["curious", "curious", "surprised", "thinking"], ps), False
        if s[S["sal_light"]] > 0.12 and s[S["hed_light"]] > 0.5:  # real days: p99.9 0.07 (habituated)
            # a light came on or the sun came out: no direction to it (one light sensor), so a wide curious look round
            return pick(["curious", "surprised", "curious", "happy"], ps), False
        if val > 0.62 and aro > 0.55:
            return pick(["very_happy", "laughing", "happy"], ps), False
        if val > 0.58:
            return pick(["happy", "cool", "normal", "wink"], ps), False
        if val < 0.38:
            return pick(["sad", "confused", "annoyed"], ps), False
        return pick(["normal", "normal", "thinking", "curious", "cool", "normal"], ps), False

    def step(self, s, pace, executed=None):
        """s: the SNN's 32 numbers, pace: 3 pacemaker values. -> target NN_OUT vector. `executed`: what the robot
        really did this tick (the network's decision during DAgger) - updates the memory instead of the target."""
        s = np.asarray(s)
        y = np.zeros(len(NN_OUT), np.float32)
        touch_now = max(s[S["sal_touch_head"]], s[S["sal_touch_screen"]])
        touch_onset = touch_now > 0.12 and touch_now - self.prev_touch > 0.04
        self.since_touch = 0.0 if touch_onset else self.since_touch + DT
        events = {"touch_onset": touch_onset,
                  # news that is not a touch (a touch is its own news: answered as a touch, not as a fright). The
                  # analyzer's surprise peaks a tick or two after the touch's onset: 1.5 s after a touch onset a rise
                  # of surprise is still the touch (2026-10-24: every stroke on the head made the robot flinch)
                  "surprise_onset": (not touch_onset and touch_now < 0.3 and self.since_touch > 1.5 and
                                     s[S["surprise"]] > 0.35 and s[S["surprise"]] - self.prev_sur > 0.1),
                  "over_onset": s[S["over"]] > 0.4 and self.prev_over <= 0.4,
                  "threat_onset": s[S["threat"]] > 0.45 and self.prev_threat <= 0.45}
        self.prev_touch, self.prev_sur = touch_now, s[S["surprise"]]
        self.prev_over, self.prev_threat = s[S["over"]], s[S["threat"]]
        if events["touch_onset"]:
            self.touch_open, self.touch_age = True, 0.0
        night = s[S["sleepiness"]] > 0.65

        want, urgent = self.choose_emotion(s, pace, events)
        min_show = MIN_SHOW.get(self.emo, DEFAULT_MIN_SHOW)
        emo = self.emo
        if want != self.emo and (urgent or self.emo_age >= min_show):
            # a calm change of mood needs a while; reactions are at once
            if urgent or self.emo_age > 4.0 + 3.0 * (pace[1] + 1):
                emo = want
            elif want in ("sleepy", "bored", "sad") and self.emo_age > 2.0:
                emo = want
        v, a, style, register, color, pattern = EMO[emo]
        aro = s[S["arousal"]]
        y[O["emo_" + emo]] = 1.0
        y[O["emo_strength"]] = float(np.clip(0.4 + 0.6 * max(aro, abs(s[S["valence"]] - 0.5) * 2), 0, 1))

        # ---- voice: answer every touch and every fright; chatter often by day, almost never at night
        say = False
        if night:
            pass  # silent at night: the owner sleeps next to the robot (the executor enforces it too)
        elif self.touch_open and self.since_sound > 0.7:
            say = True
        elif (events["surprise_onset"] or events["threat_onset"] or events["over_onset"]) and self.since_sound > 0.7:
            say = True
        elif not night and self.since_sound > 6.0 + 10.0 * (1 - aro) and pace[0] > 0.55 - 0.3 * aro:
            say = True  # idle babble: ~ every 8-25 s by day
        if say:
            self.speak_ticks = 3
        if self.speak_ticks > 0:
            y[O["say"]] = 1.0
            self.speak_ticks -= 1
        y[O["style_" + style]] = 1.0
        y[O["snd_arousal"]] = float(np.clip(0.3 * a + 0.7 * aro, 0, 1)) * (0.5 if night else 1.0)
        y[O["snd_valence"]] = float(np.clip(v * 2 - 1, -1, 1))
        y[O["snd_register"]] = register
        y[O["snd_length"]] = float(np.clip(0.25 if emo in ("surprised", "scared") else 0.4 + 0.4 * (pace[2] + 1) / 2,
                                           0, 1))
        y[O["snd_rate"]] = float(np.clip(0.3 + 0.6 * aro, 0, 1))

        # ---- LEDs
        y[O["led_" + color]] = 1.0
        y[O["pat_" + pattern]] = 1.0
        bright = 0.06 if night else float(np.clip(0.25 + 0.65 * aro, 0, 1))
        y[O["led_bright"]] = bright
        y[O["led_rate"]] = float(np.clip(0.15 + 0.8 * aro, 0, 1))

        # ---- head
        mode = "hold"
        gaze = self.head
        att = s[S["att_strength"]]
        if att > ATT_LOOK and not night:
            self.watch = (seen_yaw(self.head[0], s[S["att_x"]]), float(np.clip(s[S["att_y"]] * 0.4, -1, 1)))
            self.watch_t = WATCH_S
        else:
            self.watch_t = max(0.0, self.watch_t - DT)
        if self.gesture is not None:
            mode, left = self.gesture
            self.gesture = (mode, left - 1) if left > 1 else None
        elif self.since_gesture > 1.5:
            nod_ok = self.since_gesture > 4.0 and not night  # the executor's gesture spacing (power peaks)
            # a startle is for a sudden thing out there (a bang, a shake, something rushing close, a light), never for a
            # hand: at a touch's start the analyzer's surprise can rise before the touch's own salience does
            outer = max(s[S["sal_sharp"]], s[S["sal_loud"]], s[S["sal_shake"]], s[S["sal_tilt"]], s[S["sal_near"]],
                        s[S["sal_light"]], s[S["sal_motion"]])
            touchy = max(s[S["sal_touch_head"]], s[S["sal_touch_screen"]])
            if (events["threat_onset"] or events["surprise_onset"]) and outer > touchy:
                mode = "flinch"
            elif nod_ok and emo in ("angry", "annoyed") and (events["over_onset"] or events["touch_onset"] or
                                                             emo != self.emo) and pace[2] > -0.6:
                mode = "shake"
            elif nod_ok and emo in ("love", "happy", "very_happy", "laughing", "wink", "shy") and events["touch_onset"]:
                mode = "nod"  # a stroke answered with "yes, more" (2026-10-24: the owner saw the head hardly move)
            elif nod_ok and self.touch_open and self.touch_age > 0.3 and emo in ("love", "happy", "shy", "laughing", "very_happy",
                                                                      "wink"):
                mode = "nod"
            elif nod_ok and emo == "sad" and self.emo_age < 0.3 and pace[2] > 0.3:
                mode = "shake"
            if mode != "hold":
                self.gesture = (mode, 12)  # ~1.2 s
        if mode == "hold":
            head_touch = s[S["sal_touch_head"]] >= s[S["sal_touch_screen"]]
            if night and emo == "sleepy":
                mode = "doze"
            elif self.touch_open and self.touch_age < 2.0 and emo not in ("angry", "annoyed", "scared", "sad"):
                # a hand: the head lifts towards it (on the head pad) or looks at whoever tapped the face
                mode, gaze = ("look_up", gaze) if head_touch else ("look", (0.0, -0.2))
            elif emo in ("sad", "shy", "crying") and self.emo_age < 3:
                mode = "look_down"
            elif emo == "thinking" and self.emo_age < 3:
                mode = "look_up"
            elif not night:
                mode, gaze = manner(emo, s, pace, self.head[0], self.watch if self.watch_t > 0 else None)
            elif att > 0.2:
                mode, gaze = "look", (seen_yaw(self.head[0], s[S["att_x"]]), float(np.clip(s[S["att_y"]] * 0.4, -1, 1)))
        y[O["head_" + mode]] = 1.0
        y[O["gaze_yaw"]], y[O["gaze_pitch"]] = gaze
        amp = {"nod": 0.4 + 0.5 * aro, "shake": 0.35 + 0.6 * aro if emo != "sad" else 0.3}.get(mode, 0.5)
        y[O["gest_amp"]] = float(np.clip(amp, 0, 1))
        y[O["gest_speed"]] = float(np.clip(0.3 + 0.6 * aro, 0, 1)) * (0.5 if night else 1.0)
        y[O["gest_reps"]] = float(np.clip(0.3 + 0.5 * aro, 0, 1))

        # ---- the memory follows what was really done
        self.apply(executed if executed is not None else y)
        return y

    def apply(self, y):
        emo = EMOTIONS[int(np.argmax([y[O["emo_" + e]] for e in EMOTIONS]))]
        if emo != self.emo:
            self.emo, self.emo_age = emo, 0.0
        else:
            self.emo_age += DT
        if y[O["say"]] > 0.5:
            if self.since_sound > 0.7:
                self.touch_open = False
            self.since_sound = 0.0
        else:
            self.since_sound += DT
        mode = HEAD_MODES[int(np.argmax([y[O["head_" + m]] for m in HEAD_MODES]))]
        if mode in ("nod", "shake", "flinch"):
            if self.since_gesture > 0.5:
                self.touch_open = False
            self.since_gesture = 0.0
        else:
            self.since_gesture += DT
        if mode in ("look", "scan"):
            self.head = (0.7 * self.head[0] + 0.3 * float(y[O["gaze_yaw"]]),
                         0.7 * self.head[1] + 0.3 * float(y[O["gaze_pitch"]]))
        self.touch_age += DT
        if self.touch_age > 3.0:
            self.touch_open = False
        self.led_level = float(y[O["led_bright"]])


def pacemakers(rng, T):
    """Three slow random drifts in [-1, 1] at 10 Hz (fast ~3 s, slow ~60 s, jitter ~0.5 s)."""
    out = np.zeros((T, 3), np.float32)
    for k, tau in enumerate((3.0, 60.0, 0.5)):
        a = np.exp(-DT / tau)
        v = rng.normal()
        n = rng.normal(0, np.sqrt(1 - a * a), T)
        for t in range(T):
            v = a * v + n[t]
            out[t, k] = np.tanh(v * 0.9)
    return out


def rollout(snn_seq, rng):
    """Teacher on a 10 Hz sequence of SNN outputs -> (inputs [T, 44], targets [T, 61])."""
    T = len(snn_seq)
    pace = pacemakers(rng, T)
    tch = Teacher()
    X = np.zeros((T, len(SNN_OUT) + len(PREV)), np.float32)
    Y = np.zeros((T, len(NN_OUT)), np.float32)
    for t in range(T):
        X[t, :len(SNN_OUT)] = snn_seq[t]
        X[t, len(SNN_OUT):] = tch.prev_inputs(pace[t])
        Y[t] = tch.step(snn_seq[t], pace[t])
    return X, Y


if __name__ == "__main__":
    import collections
    from synth_v4 import make_channels
    from teacher_snn import teach
    rng = np.random.default_rng(1)
    cnt = collections.Counter()
    heads = collections.Counter()
    says = 0
    ticks = 0
    for _ in range(20):
        Sq = teach(make_channels(rng, 12000))[::2]
        X, Y = rollout(Sq, rng)
        for y in Y:
            cnt[EMOTIONS[int(np.argmax(y[:21]))]] += 1
            heads[HEAD_MODES[int(np.argmax([y[O["head_" + m]] for m in HEAD_MODES]))]] += 1
        says += int(((Y[1:, O["say"]] > 0.5) & (Y[:-1, O["say"]] < 0.5)).sum())
        ticks += len(Y)
    h = ticks * DT / 3600
    print(f"{h:.1f} h: {says / h:.0f} calls/h")
    print("emotions:", {k: f"{v / ticks * 100:.1f}%" for k, v in cnt.most_common()})
    print("head:", {k: f"{v / ticks * 100:.1f}%" for k, v in heads.most_common()})
