"""Reads the robot's diary (written by lifelogd.py) into numpy, for reports and training.

  ticks, lines = load("2026-10-11")             # one day; load("2026-10-11", hours=[20, 21]) for some hours
  ticks["values"][:, col("touch_head")]          # [T] float, the tick columns are named by col()
  ./py ../life/lifedata.py 2026-10-11            # summary of a day (run from tools/hybrid)

ticks: dict of arrays sorted by (boot, tick seq): boot, seq, ms (since boot), unix (s, NaN before the clock synced),
recv (PC receive time, s), flags, values [T, n_values]. lines: list of (recv time, boot, ms, text).
"""
import glob
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "hybrid"))
from spec import ACTION, EFF, POLICY_INPUT  # noqa: E402

HEADER = struct.Struct("<4sBBHII")
SCALE = 4096.0
# the tick columns of diary format SCL1 (main/net/life_log.h): policy input, action, efference copy, then (from
# 2026-10-11, stage 2) the motivation (main/hybrid/motivation.h). Older days have NaN in those columns.
MOTIVATION = ["need_energy", "need_social", "need_stimulation", "need_rest", "need_comfort", "reward", "novelty",
              "move_deg100"]
# ... then (stage 1) the world model on the robot (main/hybrid/world_model.h)
WORLD = ["wm_rms_vision", "wm_rms_audio", "wm_rms_body", "wm_lp10"]
# ... then (stage 4, 2026-10-14) the robot's online learning (main/hybrid/online_learner.h)
ONLINE = ["online_delta", "online_value"]
# ... then (bus v4, 2026-10-17) the perception networks' hidden layers, stored x 1/4
EMB = [f"vemb{i}" for i in range(64)] + [f"aemb{i}" for i in range(32)]
# ... then (2026-10-17) the third learned expectation (AssocMemory: being talked to)
COLUMNS = list(POLICY_INPUT) + ["act_" + a for a in ACTION] + list(EFF) + MOTIVATION + WORLD + ONLINE + EMB + \
    ["expect_talk"]
_n = len(COLUMNS)
KNOWN_LAYOUTS = (_n - 1 - len(EMB) - len(ONLINE) - len(WORLD) - len(MOTIVATION),
                 _n - 1 - len(EMB) - len(ONLINE) - len(WORLD), _n - 1 - len(EMB) - len(ONLINE), _n - 1 - len(EMB),
                 _n - 1, _n)


def col(name):
    return COLUMNS.index(name)


def read_file(path):
    """-> list of (recv time, datagram bytes)."""
    out = []
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    while i + 10 <= len(data):
        n, t = struct.unpack_from("<Hd", data, i)
        i += 10
        if i + n > len(data):
            break  # cut short while being written
        out.append((t, data[i:i + n]))
        i += n
    return out


def load(day, hours=None, root=None):
    folder = os.path.join(root or os.path.join(ROOT, "data", "life"), day)
    files = sorted(glob.glob(os.path.join(folder, "*.scl")))
    if hours is not None:
        files = [f for f in files if int(os.path.basename(f)[:2]) in hours]
    boot, seq, ms, unix, recv, flags, values, lines = [], [], [], [], [], [], [], []
    for path in files:
        for t, d in read_file(path):
            if len(d) < HEADER.size or d[:4] != b"SCL1":
                continue
            _, kind, n, nv, b, _ = HEADER.unpack_from(d)
            if kind == 1:
                if nv not in KNOWN_LAYOUTS:
                    raise ValueError(f"{path}: {nv} values per tick, this reader knows {KNOWN_LAYOUTS}")
                (u0,) = struct.unpack_from("<q", d, HEADER.size)
                rec = np.dtype([("seq", "<u4"), ("ms", "<u4"), ("flags", "<u2"), ("v", "<i2", (nv,))])
                r = np.frombuffer(d, rec, count=n, offset=HEADER.size + 8)
                boot.append(np.full(n, b, np.uint32))
                seq.append(r["seq"])
                ms.append(r["ms"])
                unix.append(np.where(u0 > 0, u0 / 1000.0 + (r["ms"] - r["ms"][0]) / 1000.0, np.nan))
                recv.append(np.full(n, t))
                flags.append(r["flags"])
                vals = r["v"].astype(np.float32) / SCALE
                if nv < len(COLUMNS):
                    vals = np.concatenate([vals, np.full((n, len(COLUMNS) - nv), np.nan, np.float32)], axis=1)
                values.append(vals)
            elif kind == 2:
                (m,) = struct.unpack_from("<I", d, HEADER.size)
                lines.append((t, b, m, d[HEADER.size + 4:].decode("utf-8", "replace")))
    if not seq:
        return None, lines
    ticks = dict(boot=np.concatenate(boot), seq=np.concatenate(seq), ms=np.concatenate(ms), unix=np.concatenate(unix),
                 recv=np.concatenate(recv), flags=np.concatenate(flags),
                 values=np.concatenate(values))
    order = np.lexsort((ticks["seq"], ticks["recv"]))
    ticks = {k: v[order] for k, v in ticks.items()}
    lines.sort()
    return ticks, lines


def summary(day):
    ticks, lines = load(day)
    if ticks is None:
        print(f"{day}: no ticks, {len(lines)} log lines")
        return
    T = len(ticks["seq"])
    boots = np.unique(ticks["boot"])
    missing = 0
    for b in boots:
        s = np.sort(ticks["seq"][ticks["boot"] == b])
        missing += int(np.sum(np.diff(s.astype(np.int64)) - 1))
    span = ticks["recv"][-1] - ticks["recv"][0]
    print(f"{day}: {T} ticks ({T / 20 / 3600:.2f} h of life over {span / 3600:.2f} h), {len(boots)} boot(s), "
          f"{missing} ticks missing ({100 * missing / max(1, T + missing):.2f} %), {len(lines)} log lines")
    v = ticks["values"]
    for name in ("person", "face", "speech", "touch_head", "touch_screen", "battery", "night", "owner_near"):
        print(f"  {name:13s} mean {v[:, col(name)].mean():.3f}  >0.5 {100 * np.mean(v[:, col(name)] > 0.5):5.1f} %")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summary(d)
