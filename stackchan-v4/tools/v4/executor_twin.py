"""What the robot's executor makes of the main network's wishes, for training (2026-10-28, an audit A10): the teacher's
memory - and so the network's own-state inputs (since_sound, since_gesture, emo_age) - must follow what the body did,
as BrainV4::executed does on the robot, not what was wished.

Until then the teacher reset since_sound at every decision that wanted a call. On the robot it resets only when a call
sounds; the call budget holds back a third of the wanted calls, so after a decision wanting a call the robot saw
since_sound near 0 in 9 % of the cases (training: always) and above 3 s in 84 % (diary 10-27/28). Gestures: 15 %
against always.

The gates mirror main/v4/v4_executor.cpp and main/voice/babble_voice.cpp, at the network's 10 Hz (the executor runs at
20 Hz; one decision = two ticks):
  voice   a call starts when say rises above SAY_ON with the voice armed (say fell below SAY_REARM since the last call),
          the phrase over and REFRACTORY_S since the last; or at once in a reaction (a touch, a surprise, a threat:
          REACTION_S). The budget: TOKENS_MAX tokens, one per TOKEN_S; a call costs one and needs IDLE_TOKENS (a
          reaction, or within TURN_S after an answer rose above TURN_ON: REACTION_TOKENS). Silent at night.
  head    a mode counts once wished at two decisions in a row; a nod or shake starts on the mode's rising edge at most
          every GESTURE_GAP_S, a flinch at most every FLINCH_GAP_S (and 1.5 s after any gesture); none at night.
  face    a new face is shown once FACE_HOLD_S has passed since the last change.
"""
import numpy as np

from spec_v4 import EMOTIONS, HEAD_MODES, NN_OUT, SNN_OUT

DT = 0.1
O = {n: i for i, n in enumerate(NN_OUT)}
S = {n: i for i, n in enumerate(SNN_OUT)}
EMO_IDX = [O["emo_" + e] for e in EMOTIONS]
HEAD_IDX = [O["head_" + m] for m in HEAD_MODES]
NOD, SHAKE, FLINCH = (HEAD_MODES.index(m) for m in ("nod", "shake", "flinch"))

SAY_ON, SAY_REARM = 0.55, 0.3
# V4Executor::kTokensMax, kTokenMs, kIdleTokens, kReactionTokens (5, 10 s, 2.5, 1 until 2026-10-28: ~6 calls a minute)
TOKENS_START, TOKENS_MAX, TOKEN_S, IDLE_TOKENS, REACTION_TOKENS = 3.0, 6.0, 5.0, 1.0, 0.5
REFRACTORY_S, REACTION_MIN_S, REACTION_S = 0.7, 0.15, 0.5
GESTURE_GAP_S, FLINCH_GAP_S, FLINCH_AFTER_S = 4.0, 15.0, 1.5
FACE_HOLD_S = 1.5
TURN_ON, TURN_S = 0.3, 4.0  # V4Executor::kTurnOn, kTurnMs


def phrase_s(length):
    """How long a phrase sounds (the simulator's BabbleVoice: 2.1 s on average at the usual length 0.4-0.8)."""
    return 1.2 + 1.8 * float(np.clip(length, 0.0, 1.0))


class ExecTwin:
    def __init__(self):
        self.tokens = TOKENS_START
        self.armed, self.reaction_open, self.prev_reaction = True, False, False
        self.since_call, self.busy = 99.0, 0.0
        self.reaction_t, self.prev_touch = 0.0, 0.0
        self.wish_prev, self.mode, self.prev_mode = 0, 0, 0
        self.gesture_left, self.since_gesture, self.since_flinch = 0.0, 99.0, 99.0
        self.shown, self.face_age = 0, 99.0
        self.prev_answered, self.turn_t = 0.0, 0.0

    def step(self, y, s, night, answered=0.0):
        """One decision: the wishes y (NN_OUT), the analyzer's numbers s, the night, the detector's answered pulse -> what
        was done:
        {"said": a call started, "gesture": a nod / shake / flinch started, "emotion": the face shown (index)}."""
        # voice (the reaction window and the budget as V4Executor::applyVoice; arming as BabbleVoice::step)
        touch = max(s[S["sal_touch_head"]], s[S["sal_touch_screen"]])
        event = (touch > 0.12 and touch - self.prev_touch > 0.02) or s[S["surprise"]] > 0.35 or s[S["threat"]] > 0.45
        self.prev_touch = touch
        self.reaction_t = REACTION_S if event else max(0.0, self.reaction_t - DT)
        reaction = self.reaction_t > 0
        self.tokens = min(TOKENS_MAX, self.tokens + DT / TOKEN_S)
        self.since_call += DT
        self.busy = max(0.0, self.busy - DT)
        say = float(y[O["say"]])
        if say < SAY_REARM:
            self.armed = True
        if reaction and not self.prev_reaction:
            self.reaction_open = True
        if not reaction:
            self.reaction_open = False
        self.prev_reaction = reaction
        self.turn_t = TURN_S if (answered > TURN_ON and self.prev_answered <= TURN_ON) else max(0.0, self.turn_t - DT)
        self.prev_answered = answered
        allow = not night and self.tokens >= (REACTION_TOKENS if reaction or self.turn_t > 0 else IDLE_TOKENS)
        if self.reaction_open:
            may = say > SAY_ON and allow and self.since_call >= REACTION_MIN_S
        else:
            may = self.armed and say > SAY_ON and allow and self.busy <= 0 and self.since_call >= REFRACTORY_S
        said = bool(may)
        if said:
            self.reaction_open = self.armed = False
            self.since_call = 0.0
            self.tokens -= 1.0
            self.busy = phrase_s(y[O["snd_length"]])

        # head: debounced mode, gestures on its rising edge (V4Executor::applyHead)
        wish = int(np.argmax(y[HEAD_IDX]))
        if wish == self.wish_prev:
            self.mode = wish
        self.wish_prev = wish
        edge = self.mode != self.prev_mode
        self.prev_mode = self.mode
        self.gesture_left = max(0.0, self.gesture_left - DT)
        self.since_gesture += DT
        self.since_flinch += DT
        gesture = False
        if self.gesture_left <= 0 and edge and not night:
            if self.mode in (NOD, SHAKE) and self.since_gesture >= GESTURE_GAP_S:
                gesture = True
                reps = 1 + int(round(float(np.clip(y[O["gest_reps"]], 0, 1)) * 2))
                self.gesture_left = 2 * reps * (0.45 - 0.15 * float(y[O["gest_speed"]]))
            elif self.mode == FLINCH and self.since_flinch >= FLINCH_GAP_S and self.since_gesture >= FLINCH_AFTER_S:
                gesture = True
                self.gesture_left = 1.2
                self.since_flinch = 0.0
        if gesture:
            self.since_gesture = 0.0

        # face (V4Executor::applyFace: a shown face stays FACE_HOLD_S)
        want = int(np.argmax(y[EMO_IDX]))
        self.face_age += DT
        if want != self.shown and self.face_age >= FACE_HOLD_S:
            self.shown, self.face_age = want, 0.0
        return {"said": said, "gesture": gesture, "emotion": self.shown}
