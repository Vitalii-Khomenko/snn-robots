"""What the ear calls speech (docs/CONTINGENCY.md step 2): a recording of the robot's microphone
(tools/robot/listen.py) lined up with the diary of the same moments, tick by tick.

  ./py sound_check.py data/sound/2026-10-27/063836      (the .wav / .npz stem; relative to the project or absolute)

For every speech onset of the ear (raw_speech > 0.5 after 2 s below) it prints what the sound was and what the robot
itself was doing: its own call (said within 5 s, the speaker mute), its head moving (the own-motion flag), how loud
and how voice-like (pitch periodicity 80-400 Hz) the sound was, its spectral centre. A picture goes next to the
recording (<stem>.png): the spectrogram with the ear's speech, the robot's calls and head motion under it.
"""
import os
import sys
import wave

import numpy as np

import diary as D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TICK_MS = 50


def load_sound(stem):
    if not os.path.isabs(stem):
        stem = os.path.join(ROOT, stem)
    stem = stem[:-4] if stem.endswith((".wav", ".npz")) else stem
    meta = np.load(stem + ".npz")
    with wave.open(stem + ".wav") as w:
        s = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    return stem, s, meta


def diary_for(meta):
    """The diary ticks of the recording's boot inside its time span."""
    import time
    t0 = float(meta["pc_start"])
    ms = meta["ms"].astype(np.int64)
    day = time.strftime("%Y-%m-%d", time.localtime(t0))
    hour = time.localtime(t0).tm_hour
    t, _ = D.load(day, hours=[hour, (hour + 1) % 24])
    span = (t["recv"] > t0 - 30) & (t["recv"] < t0 + (ms[-1] - ms[0]) / 1000 + 60)
    inside = span & (t["ms"] >= ms[0]) & (t["ms"] <= ms[-1])
    boots, counts = np.unique(t["boot"][inside], return_counts=True)
    keep = inside & (t["boot"] == boots[np.argmax(counts)])
    return {k: a[keep] for k, a in t.items()}


def sample_ms(meta, n):
    """ms since boot of every sample: block times smoothed (the DMA delivers blocks in bursts)."""
    ms = meta["ms"].astype(np.float64)
    k = np.arange(len(ms))
    a, b = np.polyfit(k, ms, 1)  # the clock: ms per block (~5.33) and the start
    return a * (np.arange(n) / 128.0) + b


def features(x, rate):
    """Loudness (dBFS), voice-likeness (best normalized autocorrelation at 80-400 Hz), spectral centre (Hz)."""
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    lo, hi = rate // 400, rate // 80
    seg = x[: rate // 10] if len(x) > rate // 10 else x
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    voice = float(np.max(ac[lo:hi]) / (ac[0] + 1e-12)) if len(ac) > hi else 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    centre = float(np.sum(f * spec) / (np.sum(spec) + 1e-12))
    return 20 * np.log10(rms), voice, centre


def main(stem):
    stem, s, meta = load_sound(stem)
    rate = int(meta["rate"])
    t = diary_for(meta)
    if len(t["ms"]) == 0:
        sys.exit("no diary ticks for this recording (is lifelogd running?)")
    sms = sample_ms(meta, len(s))
    c = lambda n: np.nan_to_num(t["v"][:, D.col(n)])
    speech, said = c("raw_speech"), c("said") > 0
    yaw, pitch = c("head_yaw_deg"), c("head_pitch_deg")
    moving = np.abs(np.diff(yaw, prepend=yaw[0])) + np.abs(np.diff(pitch, prepend=pitch[0])) > 0.05
    flags = np.repeat(meta["flags"], 128)[: len(s)]
    tick_ms = t["ms"].astype(np.float64)

    def sound_at(m0, m1):
        i0, i1 = np.searchsorted(sms, [m0, m1])
        return s[i0:i1], flags[i0:i1]

    floor_db = np.percentile([features(sound_at(m - TICK_MS, m)[0], rate)[0] for m in tick_ms
                              if len(sound_at(m - TICK_MS, m)[0]) > 64], 10)
    # ear onsets
    on = speech > 0.5
    quiet = int(2000 / TICK_MS)
    onsets = [i for i in np.flatnonzero(on) if not on[max(i - quiet, 0):i].any()]
    print(f"{os.path.basename(stem)}: {len(s) / rate:.0f} s, {len(t['ms'])} diary ticks, room floor {floor_db:.0f} dBFS; "
          f"speech > 0.5 in {on.mean():.2f} of the ticks, {len(onsets)} onsets ({len(onsets) / (len(s) / rate) * 3600:.0f}/h)")
    causes = {}
    print("   t s   speech  own call  muted  head  own-motion  loud dB  voice  centre Hz   cause")
    for i in onsets:
        m = tick_ms[i]
        x, fl = sound_at(m - 1000, m)  # the ear looks back ~1 s
        if len(x) < rate // 4:
            continue
        db, voice, centre = features(x, rate)
        own_call = said[max(i - 100, 0):i + 1].any()
        muted = float(np.mean(fl & 1))
        motion = float(np.mean((fl & 2) > 0))
        head = moving[max(i - 20, 0):i + 1].any()
        if own_call or muted > 0:
            cause = "own voice"
        elif head:
            cause = "own servos"
        elif db - floor_db < 6:
            cause = "near-silence"
        elif voice > 0.5:
            cause = "voice-like"
        else:
            cause = "noise"
        causes[cause] = causes.get(cause, 0) + 1
        print(f"  {(m - tick_ms[0]) / 1000:5.1f}  {speech[i]:.2f}    {'yes' if own_call else ' - '}     {muted:.2f}   "
              f"{'yes' if head else ' - '}     {motion:.2f}     {db:6.1f}  {voice:.2f}   {centre:6.0f}     {cause}")
    print("causes:", ", ".join(f"{k} {v}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(18, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    secs = (sms - tick_ms[0]) / 1000.0
    ax[0].specgram(s, NFFT=512, Fs=rate, noverlap=256, cmap="magma", xextent=(secs[0], secs[-1]), vmin=-130)
    ax[0].set_ylim(0, 8000)
    ax[0].set_ylabel("Hz")
    ts = (tick_ms - tick_ms[0]) / 1000.0
    ax[1].plot(ts, speech, label="ear: speech", color="tab:blue")
    ax[1].fill_between(ts, 0, moving * 0.3, color="tab:green", alpha=0.4, label="head moving")
    ax[1].vlines(ts[said], 0, 1, color="tab:red", label="own call")
    for i in onsets:
        ax[1].axvline(ts[i], color="k", ls=":", lw=0.8)
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=80)
    print("picture:", stem + ".png")


if __name__ == "__main__":
    main(sys.argv[1])
