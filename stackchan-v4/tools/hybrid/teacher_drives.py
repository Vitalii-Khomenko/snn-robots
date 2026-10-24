"""The designed internal-state dynamics the spiking core is trained to embody.

Given the sensory/body bus at 20 Hz, DriveTeacher produces what the SNN's
readouts should say: drives (curiosity, boredom, arousal, annoyance, social
need, playfulness, fatigue, sleep pressure, music drive, alertness), the
attention target, novelty per channel with habituation, decaying sensory
traces and surprise. Time constants follow the concept document (traces
0.5-5 s, habituation tens of seconds, boredom / social need minutes).

This is the *imitation* stage of the SNN: it is training metadata only; the
robot runs the trained spiking network, never this code.
"""
import numpy as np

from spec import SNN, SNN_INPUT, sin

DT = 0.05
IDX = {n: sin(n) for n in SNN_INPUT}
OUT = {n: i for i, n in enumerate(SNN)}


def ema(x, target, tau):
    return x + (target - x) * min(1.0, DT / tau)


class DriveTeacher:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hab_t = self.hab_s = self.hab_v = 0.0
        self.tr = np.zeros(6)  # touch, sound, person, motion, novelty, reward
        self.nov_v = self.nov_a = self.nov_t = 0.0
        self.arousal = 0.25
        self.alert = 0.1
        self.curiosity = 0.25
        self.boredom = 0.0
        self.annoy = 0.0
        self.social = 0.2
        self.playful = 0.15
        self.fatigue = 0.0
        self.sleep = 0.0
        self.music = 0.0
        self.satiety = 0.0
        self.surprise = 0.0
        self.stim_s = 0.3
        self.att = np.zeros(3)
        self.att_hold = 0.0
        self.att_prev = np.zeros(2)
        self.att_hab = 0.0
        self.alone_s = 0.0

    def step(self, x):
        g = lambda n: float(x[IDX[n]])
        touch = max(g("touch_head"), g("touch_screen"))
        shake, tilt = g("shake"), g("tilt")
        loud = max(g("loud"), 0.9 * g("clap"), 0.7 * g("sharp") if g("sharp") > 0.5 else 0.0)
        speech = g("speech")
        person = max(g("person"), g("face"))
        loom = g("loom")
        vis_ev = max(g("activity"), 0.7 * abs(g("light_change")), 0.8 * loom)
        night, dawn, dusk = g("night"), g("dawn"), g("dusk")
        stamina = g("stamina")
        owner_near, owner_arr = g("owner_near"), g("owner_arrived")

        # ---- habituation: repeats of the same kind of event matter less
        self.hab_t += 0.12 * touch * (1 - self.hab_t) - DT * self.hab_t / 40.0
        self.hab_s += 0.10 * loud * (1 - self.hab_s) - DT * self.hab_s / 30.0
        self.hab_v += 0.04 * vis_ev * (1 - self.hab_v) - DT * self.hab_v / 20.0
        raw_t = touch * (1 - self.hab_t)
        raw_a = max(g("novelty"), loud) * (1 - self.hab_s)
        raw_v = max(vis_ev, 0.6 * g("target_conf") * float(g("target_conf") > 0.5) * 0.0) * (1 - self.hab_v)
        self.nov_t = ema(self.nov_t, raw_t, 0.4)
        self.nov_a = ema(self.nov_a, raw_a, 0.5)
        self.nov_v = ema(self.nov_v, raw_v, 0.5)
        nov = max(self.nov_t, self.nov_a, self.nov_v)

        # ---- sensory traces
        reward_raw = 0.5 * touch * (1 - self.annoy) + 0.4 * g("face") + 0.3 * speech * float(person > 0.3) + 0.6 * owner_arr
        raw = np.array([touch, max(loud, speech * 0.8), person, g("activity"), nov, min(1.0, reward_raw)])
        taus = np.array([2.5, 1.5, 3.0, 1.0, 2.0, 3.0])
        self.tr = np.clip(self.tr * (1 - DT / taus) + raw * (5 * DT / taus), 0, 1)

        # ---- arousal / alertness
        base = 0.28 * (1 - self.sleep)
        tgt = base + loud * (1 - 0.7 * self.hab_s) + 0.7 * touch * (1 - 0.6 * self.hab_t) + 0.9 * loom + 0.9 * shake
        tgt += 0.5 * self.nov_a + 0.5 * self.nov_v + 0.35 * person + 0.4 * self.music
        tgt = float(np.clip(tgt, 0, 1)) * (1 - 0.6 * self.sleep)
        self.arousal = ema(self.arousal, tgt, 2.5 if tgt > self.arousal else 3.5)
        self.alert = ema(self.alert, float(np.clip(max(self.arousal, self.nov_a, self.nov_v, 0.8 * loom), 0, 1)) * (1 - self.sleep), 1.5)

        # ---- curiosity
        tgt = 0.22 + 0.9 * max(self.nov_v, self.nov_a) + 0.35 * person + 0.25 * g("light_strength") - 0.6 * self.annoy - 0.7 * self.sleep
        tgt = float(np.clip(tgt, 0, 1))
        self.curiosity = ema(self.curiosity, tgt, 3.0 if tgt > self.curiosity else 12.0)

        # ---- boredom: slow rise when nothing happens, quick fall with stimulation
        # weak sensor noise (a vision "activity" of 0.1, a faint person) is not stimulation: with a dead zone the slow drives can
        # really build up in a quiet room (they used to saturate at ~0.4 because of such noise); what the robot does itself
        # (moving, calling) keeps it a little busy and so bounds them
        dz = lambda v, th=0.2: max(0.0, (v - th) / (1.0 - th))
        stim = max(dz(vis_ev), 0.7 * dz(max(loud, speech)), dz(touch), 0.6 * dz(person, 0.3), dz(nov),
                   0.15 * g("self_voice"))
        self.stim_s = ema(self.stim_s, stim, 5.0)
        quiet = max(0.0, 1 - self.stim_s * 2.5)
        self.boredom += DT * (0.003 * (1 - self.boredom) * quiet - 0.1 * self.boredom * float(np.clip((self.stim_s - 0.05) / 0.3, 0.0, 1.0)))
        self.boredom = float(np.clip(self.boredom * (1 - 0.5 * self.sleep * DT), 0, 1))

        # ---- annoyance: the same thing again and again
        # touches annoy only in a run: two or three strokes are petting (habituation below 0.55), a burst of quick
        # touches pushes it up (owner, 2026-10-04: two or three touches made the robot angry)
        rise = touch * 3.0 * max(0.0, self.hab_t - 0.55) / 0.45 + loud * self.hab_s * 1.5 + shake * 1.5 + 0.3 * tilt
        self.annoy = float(np.clip(self.annoy + DT * (rise * (1 - self.annoy) * 1.5) - DT * self.annoy / 25.0, 0, 1))

        # ---- social need: grows when alone, satisfied by company
        presence = max(dz(self.tr[2], 0.5), dz(self.tr[0]), dz(self.tr[1]) * 0.5 * float(speech > 0.3))
        self.social += DT * (0.0035 * (1 - self.social) * (1 - presence) - 0.15 * self.social * presence
                             - 0.002 * self.social * g("self_voice"))
        self.social = float(np.clip(self.social + 0.3 * owner_arr, 0, 1)) * (1 - 0.9 * DT * self.sleep)

        # ---- playfulness, fatigue, sleep
        tgt = (0.15 + 0.7 * self.tr[5] + 0.3 * person * (1 - self.annoy)) * (1 - 0.6 * self.fatigue) * (1 - self.sleep)
        self.playful = ema(self.playful, float(np.clip(tgt, 0, 1)), 20.0)
        self.fatigue = ema(self.fatigue, float(np.clip((0.75 - stamina) / 0.75 + 0.15 * night, 0, 1)), 20.0)
        tgt = night * 0.95 + self.fatigue * 0.4 - 1.3 * max(self.tr[0], self.tr[1], 0.9 * self.tr[2]) - 0.8 * self.arousal
        tgt = float(np.clip(tgt, 0, 1))
        self.sleep = ema(self.sleep, tgt, 45.0 if tgt > self.sleep else 4.0)

        # ---- music drive (dawn / dusk windows, satiety after singing)
        window = max(dawn, dusk)
        self.satiety += DT * g("self_voice") * 0.04 * (1 - self.satiety) - DT * self.satiety / 300.0
        tgt = window * (1 - self.satiety) * (1 - self.annoy) * (1 - 0.7 * self.sleep) * (1 - 0.5 * self.arousal)
        self.music = ema(self.music, float(tgt), 6.0)

        # ---- surprise
        tgt = max(g("pe_motor"), 0.7 * g("pe_vision"), 0.6 * max(self.nov_a, self.nov_v))
        self.surprise = ema(self.surprise, float(np.clip(tgt, 0, 1)), 1.0)

        # ---- attention: the most salient place, held, wearing out
        cands = [
            (g("target_conf") * (0.55 + 0.45 * float(person > 0.3)), g("target_x"), g("target_y")),
            (0.6 * g("light_strength"), g("light_x"), g("light_y")),
            (0.22 + 0.1 * self.boredom, g("wander_x"), g("wander_y") * 0.6),
        ]
        s, ax, ay = max(cands, key=lambda c: c[0])
        # wearing out: staring at the same place lowers its salience
        if np.hypot(ax - self.att_prev[0], ay - self.att_prev[1]) < 0.2:
            self.att_hab = min(1.0, self.att_hab + DT / 12.0)
        else:
            self.att_hab = max(0.0, self.att_hab - DT / 1.5)
        s *= (1 - 0.6 * self.att_hab) * (1 - self.sleep)
        self.att_prev = 0.9 * self.att_prev + 0.1 * np.array([ax, ay])
        self.att[0] = ema(self.att[0], ax, 0.4 if s > 0.15 else 3.0)
        self.att[1] = ema(self.att[1], ay, 0.4 if s > 0.15 else 3.0)
        self.att[2] = ema(self.att[2], float(np.clip(s, 0, 1)), 0.5)

        out = np.zeros(len(SNN), np.float32)
        for name, val in (("curiosity", self.curiosity), ("boredom", self.boredom), ("arousal", self.arousal),
                          ("annoyance", self.annoy), ("social", self.social), ("playful", self.playful),
                          ("fatigue", self.fatigue), ("sleep", self.sleep), ("music", self.music),
                          ("alert", self.alert), ("att_x", self.att[0]), ("att_y", self.att[1]),
                          ("att_strength", self.att[2]), ("nov_vis", self.nov_v), ("nov_aud", self.nov_a),
                          ("nov_touch", self.nov_t), ("tr_touch", self.tr[0]), ("tr_sound", self.tr[1]),
                          ("tr_person", self.tr[2]), ("tr_motion", self.tr[3]), ("tr_novelty", self.tr[4]),
                          ("tr_reward", self.tr[5]), ("hab_touch", self.hab_t), ("hab_sound", self.hab_s),
                          ("surprise", self.surprise)):
            out[OUT[name]] = val
        return out


def run(bus):
    t = DriveTeacher()
    return np.stack([t.step(x) for x in bus])
