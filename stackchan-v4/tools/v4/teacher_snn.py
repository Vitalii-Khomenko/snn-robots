"""The innate evaluation brain v4's spiking analyzer learns to embody (BRAIN_V4_PLAN.txt, sections 4-5).

From the 13 channels + "where" (20 Hz) to the 32 numbers of spec_v4.SNN_OUT: global feeling (valence, arousal,
under- / over-stimulation, threat, surprise), salience and hedonic value per external channel (with habituation:
the same stimulus again matters less), the body's needs and where to look. Built from the zone table of
spec_v4 (soft zones), so the network inherits the owner's scales as its instinct.

Training metadata only: the robot runs the trained spiking network, never this code.
"""
import numpy as np
from numba import njit

from spec_v4 import CHANNELS, EXTERNAL, SNN_OUT, ZONE_AROUSAL, ZONE_TABLE, ZONE_VALENCE, ZONES, zones_of

DT = 0.05
NC, NE = len(CHANNELS), len(EXTERNAL)
# how much each zone draws attention to its channel
ZONE_SALIENCE = {"boredom": 0.0, "comfort": 0.08, "interest": 0.45, "anxiety": 0.75, "anger": 0.95}
# a jump (onset) of a channel is news: gain per external channel
JUMP_GAIN = np.array([2.5, 2.0, 1.5, 4.0, 4.0, 3.0, 2.5, 2.0, 2.5, 2.5], np.float64)
# channels whose anxiety / anger zones mean "too much" (over) - the rest (shake, tilt) mean danger (threat)
OVER_W = np.array([1.0, 0.8, 0.6, 1.0, 1.0, 0.0, 0.0, 0.6, 0.7, 0.8], np.float64)


def zone_matrix(X):
    """[T, NC, 5] soft memberships per channel and zone (0 where a channel has no such zone)."""
    T = len(X)
    M = np.zeros((T, NC, len(ZONES)), np.float64)
    for c, ch in enumerate(CHANNELS):
        for z, m in zones_of(ch, X[:, c]).items():
            M[:, c, ZONES.index(z)] = m
    return M


ZV = np.array([ZONE_VALENCE[z] for z in ZONES])
ZA = np.array([ZONE_AROUSAL[z] for z in ZONES])
ZS = np.array([ZONE_SALIENCE[z] for z in ZONES])
IB, IC, II, IX, IA = (ZONES.index(z) for z in ZONES)  # boredom, comfort, interest, anxiety, anger


@njit(cache=True)
def _clip(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


@njit(cache=True)
def _teach(X, M, ZV, ZA, ZS, JG, OW, out, st):
    k_slow = np.exp(-DT / 20.0)     # what a channel usually reads (novelty is the difference)
    k_hab = np.exp(-DT / 60.0)      # habituation: salience spent recently
    k_sur = np.exp(-DT / 0.5)
    a_val = 1.0 - np.exp(-DT / 2.0)
    a_aro = 1.0 - np.exp(-DT / 1.5)
    a_att = 1.0 - np.exp(-DT / 0.3)
    T = X.shape[0]
    slow = st[0:10]
    hab = st[10:20]
    prev = st[20:30]
    val, aro, under, over, threat, surprise = st[30], st[31], st[32], st[33], st[34], st[35]
    ax, ay, astr = st[36], st[37], st[38]
    hunger_s, sleep_s = st[39], st[40]
    sal = np.zeros(10)
    hed = np.zeros(10)
    for t in range(T):
        x = X[t]
        jump_max = 0.0
        for c in range(10):
            v = x[c]
            zv = 0.0
            zs = 0.0
            for z in range(5):
                zv += M[t, c, z] * ZV[z]
                zs += M[t, c, z] * ZS[z]
            jump = v - prev[c]
            prev[c] = v
            nov = abs(v - slow[c]) * 2.0
            slow[c] = slow[c] * k_slow + v * (1.0 - k_slow)
            on = max(0.0, jump) * JG[c] * 10.0      # per-tick jump x10: a 0.1 step = 1
            jump_max = max(jump_max, on * (1.0 - 0.5 * hab[c]))
            raw = _clip(0.7 * zs + nov + 0.5 * on)
            s = raw * (1.0 - 0.6 * hab[c])
            hab[c] = _clip(hab[c] * k_hab + s * 0.02)
            sal[c] = s
            hed[c] = _clip(0.5 + 0.5 * zv)
        surprise = max(surprise * k_sur, _clip(jump_max))
        # threat: shaken, tilted, a bang
        t_shake = M[t, 5, 3] + M[t, 5, 4] * 0.5 + M[t, 5, 2] * 0.15
        t_tilt = M[t, 6, 3] * (0.4 + 0.6 * x[6]) + M[t, 6, 2] * 0.15
        t_bang = M[t, 1, 4] * 0.6
        th_now = 1.0 - (1.0 - _clip(t_shake)) * (1.0 - _clip(t_tilt)) * (1.0 - _clip(t_bang))
        threat = th_now if th_now > threat else threat + (th_now - threat) * (1.0 - np.exp(-DT / 3.0))
        # touched while in danger is not pleasant
        for c in (3, 4):
            hed[c] = _clip(hed[c] - 0.35 * threat * (1.0 if x[c] > 0.03 else 0.0))
        # over-stimulation: anxiety / anger zones of the "too much" channels, combined like independent sources
        keep = 1.0
        for c in range(10):
            keep *= 1.0 - _clip(OW[c] * (M[t, c, 4] * 0.9 + M[t, c, 3] * 0.4))
        ov_now = 1.0 - keep
        over = ov_now if ov_now > over else over + (ov_now - over) * (1.0 - np.exp(-DT / 5.0))
        hunger, tired, lonely = x[10], x[11], x[12]
        dark = 1.0 if x[8] < 0.1 else 0.0
        hunger_s += (hunger - hunger_s) * (1.0 - np.exp(-DT / 5.0))
        sl_now = _clip(tired + 0.25 * dark - 0.4 * max(0.0, aro - 0.4))
        sleep_s += (sl_now - sleep_s) * (1.0 - np.exp(-DT / 10.0))
        # valence: salience-weighted hedonic of the world, then the body's needs
        num = 0.0
        den = 0.0
        smax = 0.0
        ssum = 0.0
        for c in range(10):
            w = 0.15 + sal[c]
            num += w * (hed[c] - 0.5)
            den += w
            smax = max(smax, sal[c])
            ssum += sal[c]
        v_now = 0.5 + 1.4 * num / den - 0.3 * max(0.0, hunger - 0.5) - 0.2 * max(0.0, lonely - 0.3) - 0.25 * threat
        val += (_clip(v_now) - val) * a_val
        # arousal
        zmax = 0.0
        for c in range(10):
            za = 0.0
            for z in range(5):
                za += M[t, c, z] * ZA[z]
            zmax = max(zmax, za * (0.3 + sal[c]))
        a_now = 0.12 + 0.45 * zmax + 0.25 * smax + 0.1 * ssum / 10.0 + 0.35 * surprise + 0.3 * threat - 0.3 * sleep_s
        aro += (_clip(a_now) - aro) * a_aro
        # under-stimulation: alone a long time and nothing going on (not at night: then it is sleep)
        bored = (M[t, 0, 0] + M[t, 2, 0] + M[t, 3, 0] + M[t, 7, 0] + M[t, 9, 0]) / 5.0
        u_now = _clip((0.8 * lonely + 0.3 * bored * np.sqrt(lonely)) * (1.0 - 0.7 * sleep_s) - 0.6 * ssum / 10.0)
        under += (u_now - under) * (1.0 - np.exp(-DT / 5.0))
        # attention: the most salient source with a place (movement in view, a touch, someone at the face)
        cand_s = (sal[9] * (1.0 if x[9] > 0.02 else 0.3), sal[3], sal[4], sal[7] * 0.8)
        cand_x = (x[13], 0.0, 0.0, 0.0)
        cand_y = (x[14], -0.7, 0.0, -0.1)
        best = 0
        for k in range(4):
            if cand_s[k] > cand_s[best]:
                best = k
        tx, ty, ts = cand_x[best], cand_y[best], cand_s[best]
        if ts < 0.05:
            tx, ty = 0.0, 0.0
        ax += (tx - ax) * a_att
        ay += (ty - ay) * a_att
        astr += (_clip(ts) - astr) * a_att
        out[t, 0], out[t, 1], out[t, 2], out[t, 3], out[t, 4], out[t, 5] = val, aro, under, over, threat, surprise
        for c in range(10):
            out[t, 6 + c] = sal[c]
            out[t, 16 + c] = hed[c]
        out[t, 26], out[t, 27], out[t, 28] = hunger_s, sleep_s, lonely
        out[t, 29], out[t, 30], out[t, 31] = ax, ay, astr
    st[30], st[31], st[32], st[33], st[34], st[35] = val, aro, under, over, threat, surprise
    st[36], st[37], st[38], st[39], st[40] = ax, ay, astr, hunger_s, sleep_s


def new_state(X0=None):
    st = np.zeros(41)
    if X0 is not None:
        st[0:10] = X0[:10]
        st[20:30] = X0[:10]
    st[30], st[31] = 0.5, 0.2
    return st


def teach(X, st=None):
    """X [T, 15] channels -> Y [T, 32] targets in spec_v4.SNN_OUT order."""
    X = np.asarray(X, np.float64)
    if st is None:
        st = new_state(X[0])
    Y = np.zeros((len(X), len(SNN_OUT)))
    _teach(X, zone_matrix(X), ZV, ZA, ZS, JUMP_GAIN, OVER_W, Y, st)
    return Y.astype(np.float32)


assert [c for c in CHANNELS[:10]] == EXTERNAL and all(c in ZONE_TABLE for c in CHANNELS)

if __name__ == "__main__":
    from synth_v4 import make_channels
    rng = np.random.default_rng(3)
    Xs = [make_channels(rng, 12000) for _ in range(30)]
    Ys = [teach(X) for X in Xs]
    Y = np.concatenate(Ys)
    for i, n in enumerate(SNN_OUT):
        print(f"{n:16s} mean {Y[:, i].mean():.3f} sd {Y[:, i].std():.3f} p01 {np.percentile(Y[:, i], 1):+.2f} "
              f"p99 {np.percentile(Y[:, i], 99):+.2f}")
