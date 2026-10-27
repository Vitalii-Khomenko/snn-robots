"""Is it a voice? (docs/CONTINGENCY.md step 2; main/perception/voice_check.cpp is the robot's, parity_voice_check.py.)

The ear's speech class fires ~250 times an hour on the robot's own sounds: the tail of its voice and the servos'
whine and clicks (recordings of 2026-10-27, sound_check.py). A human voice is periodic at its pitch (80-400 Hz) and
stands above the room; servo whine (~6 kHz) and clicks are not. This check:

  mono 24 kHz -> high-pass 70 Hz, low-pass 900 Hz (twice) -> every 6th sample (4 kHz) -> every 25 ms over the last
  50 ms: periodicity = the best normalized autocorrelation at lags 10..50 (400..80 Hz), level = mean square in dB;
  the room's floor follows the level down at once and up slowly (~12 s);
  voiced frame = periodicity > 0.6 and level > floor + 6 dB; voice = a ~250 ms average of voiced frames (0..1)

  ./py voice_check.py data/sound/2026-10-27/063836     voice over a recording, against the ear's speech (diary)
"""
import math
import os
import sys

import numpy as np

RATE, DECIM = 24000, 6
HOP, WIN, LAG_LO, LAG_HI = 100, 200, 10, 50
PERIODIC, ABOVE_DB = 0.6, 6.0
FLOOR_DOWN, FLOOR_UP, VOICE_RATE = 0.3, 0.002, 0.1


def biquad(kind, fc, q=1 / math.sqrt(2), fs=RATE):
    """RBJ cookbook coefficients (b0, b1, b2, a1, a2), normalized, as float32 like the robot."""
    w = 2 * math.pi * fc / fs
    al = math.sin(w) / (2 * q)
    c = math.cos(w)
    if kind == "lp":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
    else:
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    a0 = 1 + al
    return np.array([b[0] / a0, b[1] / a0, b[2] / a0, -2 * c / a0, (1 - al) / a0], np.float32)


STAGES = [biquad("hp", 70.0), biquad("lp", 900.0), biquad("lp", 900.0)]


class VoiceCheck:
    """Sample by sample like the robot (float32): feed(mono int16 array) -> list of (sample index, voice) per hop."""

    def __init__(self):
        self.z = np.zeros((len(STAGES), 2), np.float32)
        self.ring = np.zeros(256, np.float32)
        self.n = 0          # decimated samples so far
        self.phase = 0
        self.floor = None
        self.voice = np.float32(0)
        self.last = (0.0, -120.0)

    def feed(self, mono):
        out = []
        x = (np.asarray(mono, np.float32) / np.float32(32768.0)).astype(np.float32)
        # the biquads in transposed direct form II, float32, sample by sample (as the robot)
        for k, (b0, b1, b2, a1, a2) in enumerate(STAGES):
            z1, z2 = self.z[k]
            y = np.empty_like(x)
            for i in range(len(x)):
                xi = x[i]
                yi = np.float32(b0 * xi + z1)
                z1 = np.float32(b1 * xi - a1 * yi + z2)
                z2 = np.float32(b2 * xi - a2 * yi)
                y[i] = yi
            self.z[k] = (z1, z2)
            x = y
        for i in range(len(x)):
            self.phase += 1
            if self.phase < DECIM:
                continue
            self.phase = 0
            self.ring[self.n % 256] = x[i]
            self.n += 1
            if self.n % HOP == 0 and self.n >= WIN + LAG_HI:
                out.append((i, self._frame()))
        return out

    def _frame(self):
        idx = (self.n - 1 - np.arange(WIN + LAG_HI)[::-1]) % 256
        w = self.ring[idx].astype(np.float32)        # oldest .. newest
        cur = w[LAG_HI:]
        e0 = np.float32(np.dot(cur, cur))
        best = 0.0
        for lag in range(LAG_LO, LAG_HI + 1):
            prev = w[LAG_HI - lag:LAG_HI - lag + WIN]
            el = np.float32(np.dot(prev, prev))
            r = np.float32(np.dot(cur, prev)) / np.float32(math.sqrt(float(e0) * float(el)) + 1e-12)
            best = max(best, float(r))
        level = 10.0 * math.log10(float(e0) / WIN + 1e-12)
        if self.floor is None:
            self.floor = level
        self.floor += (level - self.floor) * (FLOOR_DOWN if level < self.floor else FLOOR_UP)
        voiced = best > PERIODIC and level > self.floor + ABOVE_DB
        self.voice = np.float32(self.voice + ((1.0 if voiced else 0.0) - self.voice) * VOICE_RATE)
        self.last = (best, level)
        return float(self.voice)


def over_recording(stem):
    import sound_check as SC
    stem, s, meta = SC.load_sound(stem)
    mono = np.round(s * 32768).astype(np.int16)
    vc = VoiceCheck()
    frames = vc.feed(mono)
    sms = SC.sample_ms(meta, len(s))
    return stem, np.array([sms[i] for i, _ in frames]), np.array([v for _, v in frames], np.float32), meta


if __name__ == "__main__":
    import diary as D
    import sound_check as SC
    stem, fms, voice, meta = over_recording(sys.argv[1])
    t = SC.diary_for(meta)
    sp = np.nan_to_num(t["v"][:, D.col("raw_speech")])
    v_at = np.interp(t["ms"].astype(np.float64), fms, voice)
    on = sp > 0.5
    print(f"{os.path.basename(stem)}: voice > 0.3 in {np.mean(voice > 0.3):.3f} of the frames; ear speech > 0.5 in "
          f"{on.mean():.3f} of the ticks, of which with voice > 0.3 within the tick: {np.mean(v_at[on] > 0.3) if on.any() else 0:.3f}")
