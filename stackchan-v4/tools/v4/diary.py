"""The robot's diary (main/net/life_log.h, received by tools/robot/lifelogd.py into data/life/<day>/<hour>.scl).

  ticks, lines = load("2026-10-25")              # one day; load(day, hours=[20, 21]) for some hours
  ticks["v"][:, col("ch_loud")]                  # [T] float32, columns named as in spec_v4.DIARY
  raw, hours = raw_for_channels(ticks)           # the ChannelEncoder inputs (channels.encode) of these ticks
  ./py diary.py 2026-10-25 [...]                 # summary of a day: hours, losses, channels, faces, calls

ticks: dict of arrays sorted by (receive time, tick seq): boot, seq, ms (since boot), unix (s, NaN before the clock
synced), recv (PC receive time, s), flags (bit 0 brain v4 drives, bit 1 night rules, bit 2 the main network decided
this tick), v [T, len(DIARY)].
lines: list of (recv time, boot, ms, text) - every ESP_LOG line of the robot.

Days before brain v4's own diary (until 2026-10-24) are platform B's layout: legacy_diary.py. Old hours may be
gzip-compressed (<hour>.scl.gz, tools/robot/compress_diary.py); both readers take either.
"""
import glob
import gzip
import os
import struct
import sys

import numpy as np

from spec_v4 import CHANNELS, DIARY, EMOTIONS, RAW

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIFE = os.path.join(ROOT, "data", "life")
HEADER = struct.Struct("<4sBBHII")  # magic, kind, n, n_values, boot id, datagram seq
KIND_LINE, KIND_V4 = 2, 3
# The v4 layout grew; older layouts are read by column name, missing columns as NaN:
#   135 values (2026-10-24, until 14:30): no pv_* columns
#   147 values (until 2026-10-25 evening): no camera bright spot (raw_light_*, ch_light_*, raw_head_yaw)
#   154 values (until 2026-10-26 evening): no Muse ideas (pv_idea_*; the pv_pace_* were blind noise then)
#   159 values (until 2026-10-27 06:29): no contingency (ct_*)
#   165 values (until 2026-10-27 06:48): no ear before the voice check (ear_*)
#   167 values (until 2026-10-27 morning): the network did not see the contingency yet (no pv_agency .. pv_works_face)
#   173 values (until 2026-10-27 evening): no body budget (raw_budget)
#   174 values (until 2026-10-28 evening): no `held` (what the executor held back of the network's wishes)
_NEW_LIGHT = {"raw_light_x", "raw_light_y", "raw_light_str", "raw_head_yaw", "ch_light_x", "ch_light_y", "ch_light_spot"}
_CT_PV = {"pv_agency", "pv_answered", "pv_ignored", "pv_works_call", "pv_works_gesture", "pv_works_face"}
_L174 = [n for n in DIARY if n != "held"]
_L173 = [n for n in _L174 if n != "raw_budget"]
_L167 = [n for n in _L173 if n not in _CT_PV]
_L165 = [n for n in _L167 if not n.startswith("ear_")]
_L159 = [n for n in _L165 if not n.startswith("ct_")]
_L154 = [n for n in _L159 if not n.startswith("pv_idea_")]
_L147 = [n for n in _L154 if n not in _NEW_LIGHT]
_L135 = [n for n in _L147 if not n.startswith("pv_")]
LAYOUTS = {len(DIARY): DIARY, len(_L174): _L174, len(_L173): _L173, len(_L167): _L167, len(_L165): _L165, len(_L159): _L159, len(_L154): _L154, len(_L147): _L147, len(_L135): _L135}
KNOWN_LAYOUTS = tuple(LAYOUTS)
_INDEX = {n: i for i, n in enumerate(DIARY)}
FLAG_DRIVES, FLAG_NIGHT, FLAG_DECIDED = 1, 2, 4


def col(name):
    return DIARY.index(name)


def day_files(day, hours=None, root=None):
    """The day's hour files, .scl or .scl.gz, in hour order."""
    folder = os.path.join(root or LIFE, day)
    files = sorted(glob.glob(os.path.join(folder, "*.scl")) + glob.glob(os.path.join(folder, "*.scl.gz")))
    if hours is not None:
        files = [f for f in files if int(os.path.basename(f)[:2]) in hours]
    return files


def read_file(path):
    """-> list of (recv time, datagram bytes). A file cut short while being written ends at its last whole datagram."""
    with (gzip.open if path.endswith(".gz") else open)(path, "rb") as f:
        data = f.read()
    out, i = [], 0
    while i + 10 <= len(data):
        n, t = struct.unpack_from("<Hd", data, i)
        i += 10
        if i + n > len(data):
            break
        out.append((t, data[i:i + n]))
        i += n
    return out


def datagrams(day, hours=None, root=None):
    """(recv time, kind, n, n_values, boot, payload) of every diary datagram of the day."""
    for path in day_files(day, hours, root):
        for t, d in read_file(path):
            if len(d) < HEADER.size or d[:4] != b"SCL1":
                continue
            _, kind, n, nv, boot, _ = HEADER.unpack_from(d)
            # a datagram whose size does not fit its kind is skipped, not trusted (lifelogd.checked)
            if kind == KIND_V4 and len(d) != HEADER.size + 8 + n * (10 + 2 * nv):
                continue
            if kind == KIND_LINE and len(d) < HEADER.size + 4:
                continue
            yield t, kind, n, nv, boot, d


def load(day, hours=None, root=None):
    boot, seq, ms, unix, recv, flags, values, lines = [], [], [], [], [], [], [], []
    for t, kind, n, nv, b, d in datagrams(day, hours, root):
        if kind == KIND_V4:
            if nv not in KNOWN_LAYOUTS:
                raise ValueError(f"{day}: {nv} values per tick, this reader knows {KNOWN_LAYOUTS}")
            (u0,) = struct.unpack_from("<q", d, HEADER.size)
            rec = np.dtype([("seq", "<u4"), ("ms", "<u4"), ("flags", "<u2"), ("v", "<f2", (nv,))])
            r = np.frombuffer(d, rec, count=n, offset=HEADER.size + 8)
            boot.append(np.full(n, b, np.uint32))
            seq.append(r["seq"])
            ms.append(r["ms"])
            unix.append(np.where(u0 > 0, u0 / 1000.0 + (r["ms"] - r["ms"][0]) / 1000.0, np.nan))
            recv.append(np.full(n, t))
            flags.append(r["flags"])
            vals = r["v"].astype(np.float32)
            if nv != len(DIARY):
                full = np.full((n, len(DIARY)), np.nan, np.float32)
                full[:, [_INDEX[c] for c in LAYOUTS[nv]]] = vals
                vals = full
            values.append(vals)
        elif kind == KIND_LINE:
            (m,) = struct.unpack_from("<I", d, HEADER.size)
            lines.append((t, b, m, d[HEADER.size + 4:].decode("utf-8", "replace")))
    lines.sort()
    if not seq:
        return None, lines
    ticks = dict(boot=np.concatenate(boot), seq=np.concatenate(seq), ms=np.concatenate(ms), unix=np.concatenate(unix),
                 recv=np.concatenate(recv), flags=np.concatenate(flags), v=np.concatenate(values))
    order = np.lexsort((ticks["seq"], ticks["recv"]))
    return {k: a[order] for k, a in ticks.items()}, lines


def raw_for_channels(ticks):
    """The raw sense readings the robot logged -> channels.encode's input dict, and the hour of day of every tick."""
    v = ticks["v"]
    raw = {r: v[:, col("raw_" + r)].astype(np.float32) for r in RAW}
    return raw, raw["hour"].copy()


def summary(day):
    ticks, lines = load(day)
    if ticks is None:
        print(f"{day}: no brain v4 ticks ({len(lines)} log lines; older days: legacy_diary.py)")
        return
    T = len(ticks["seq"])
    missing = 0
    for b in np.unique(ticks["boot"]):
        s = np.sort(ticks["seq"][ticks["boot"] == b])
        missing += int(np.sum(np.diff(s.astype(np.int64)) - 1))
    v = ticks["v"]
    print(f"{day}: {T} ticks ({T / 20 / 3600:.2f} h), {len(np.unique(ticks['boot']))} boot(s), {missing} ticks missing "
          f"({100 * missing / max(1, T + missing):.2f} %), {len(lines)} log lines, "
          f"{100 * np.mean(ticks['flags'] & 2 > 0):.0f} % under night rules")
    for c in CHANNELS:
        x = v[:, col("ch_" + c)]
        print(f"  {c:13s} mean {x.mean():.3f}  p90 {np.percentile(x, 90):.3f}  max {x.max():.3f}")
    faces = np.bincount(v[:, col("face")].astype(int).clip(0, len(EMOTIONS) - 1), minlength=len(EMOTIONS)) / T
    top = np.argsort(faces)[::-1][:6]
    print("  faces: " + ", ".join(f"{EMOTIONS[i]} {100 * faces[i]:.0f} %" for i in top if faces[i] > 0))
    print(f"  calls {int(v[:, col('said')].sum())}, gestures {int(v[:, col('gesture')].sum())}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summary(d)
