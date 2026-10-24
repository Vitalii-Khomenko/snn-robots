"""The morning report: what the robot's day looked like, from its diary (lifedata.py). Stage 0 of platform B: we
only look, nothing here judges or changes the robot.

  ./py ../life/life_report.py [YYYY-MM-DD ...]     (from tools/hybrid; default: yesterday and today)

Per hour: how long it lived, who was around (person / face / speech / owner), what reached it (touches, shakes),
what it did (head moves and travel, calls, faces), how varied that was (entropy of its "posture": face x sound x
holding still, in bits), and events (boots, new firmware or weights, learned links, warnings). The report is also
written to data/life/<day>/report.txt.
"""
import datetime as dt
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lifedata as L  # noqa: E402

EMO = [c for c in L.COLUMNS if c.startswith("act_emo_")]
STAT = re.compile(r"\b(moves|calls|travel)=(\d+)")


def onsets(x):
    on = x > 0.5
    return int(np.sum(on[1:] & ~on[:-1]) + (on[0] if len(on) else 0))


def entropy(labels):
    if not labels:
        return 0.0
    p = np.array(list(Counter(labels).values()), float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def hour_of(t):
    return dt.datetime.fromtimestamp(t).hour


def report(day):
    ticks, lines = L.load(day)
    out = [f"# {day}"]
    if ticks is None:
        out.append("no diary for this day")
        return "\n".join(out)
    v = ticks["values"]
    c = lambda name: v[:, L.col(name)]  # noqa: E731
    hours = np.array([hour_of(t) for t in ticks["recv"]])
    emo = np.argmax(v[:, [L.col(e) for e in EMO]], axis=1)
    posture = list(zip(emo, c("act_sound_act") > 0.5, c("act_hold") > 0.5))

    # cumulative counters of the 5 s stats line, per boot -> per hour increments
    inc = defaultdict(lambda: Counter())
    last = {}
    events = []
    for t, boot, ms, text in lines:
        h = hour_of(t)
        if "stats:" in text:
            vals = {k: int(x) for k, x in STAT.findall(text)}
            prev = last.get(boot)
            if prev:
                for k, x in vals.items():
                    inc[h][k] += max(0, x - prev.get(k, x))
            last[boot] = vals
        elif ("firmware in" in text or "weights" in text or "learn:" in text or " E (" in text
              or text.startswith("E (") or "RobotRuntime ready" in text or "rolled back" in text
              or "boot reason" in text or "crash of the last run" in text):
            stamp = dt.datetime.fromtimestamp(t).strftime("%H:%M:%S")
            events.append(f"{stamp} {boot:08x} {re.sub(r'^[IWE] \\(\\d+\\) ', '', text)[:110]}")

    # the PC's own notes in log.txt (lifelogd's silence watch)
    txt = os.path.join(L.ROOT, "data", "life", day, "log.txt")
    if os.path.exists(txt):
        for ln in open(txt, encoding="utf-8", errors="replace"):
            if "-------- PC:" in ln and "PC: power" not in ln:
                events.append(ln.strip().replace(" -------- PC:", " PC:"))
        events.sort()
    missing = 0
    for b in np.unique(ticks["boot"]):
        s = np.sort(ticks["seq"][ticks["boot"] == b]).astype(np.int64)
        missing += int(np.sum(np.diff(s) - 1))
    boots = len(np.unique(ticks["boot"]))
    out.append(f"{len(v) / 72000:.2f} h of life in the diary, {boots} boot(s), {missing} ticks missing "
               f"({100 * missing / max(1, len(v) + missing):.2f} %), {len(lines)} log lines")
    out.append("")
    out.append("hour  life  person  face  speech  owner  touch  shake  moves  travel  calls  faces  posture bits")
    for h in sorted(set(hours)):
        m = hours == h
        sel = lambda name: c(name)[m]  # noqa: E731
        faces = len(set(emo[m]))
        out.append(
            f"{h:02d}   {m.sum() / 1200:4.0f}m  {100 * np.mean(sel('person') > 0.5):5.0f}%  "
            f"{100 * np.mean(sel('face') > 0.5):3.0f}%  {100 * np.mean(sel('speech') > 0.5):5.0f}%  "
            f"{100 * np.mean(sel('owner_near') > 0.5):4.0f}%  {onsets(sel('touch_head')) + onsets(sel('touch_screen')):5d}  "
            f"{onsets(np.maximum(sel('shake'), sel('tilt'))):5d}  {inc[h]['moves']:5d}  {inc[h]['travel']:6d}  "
            f"{inc[h]['calls']:5d}  {faces:5d}  {entropy([p for p, k in zip(posture, m) if k]):11.2f}")
    out.append("")
    out.append(f"posture entropy over the whole day: {entropy(posture):.2f} bits; faces shown: "
               + ", ".join(f"{EMO[i][8:]} {n}" for i, n in Counter(emo).most_common(8)))
    out.append("")
    out.append("events:")
    out.extend("  " + e for e in events[-40:]) if events else out.append("  none")
    return "\n".join(out)


def main():
    days = sys.argv[1:] or [(dt.date.today() - dt.timedelta(days=1)).isoformat(), dt.date.today().isoformat()]
    for day in days:
        text = report(day)
        print(text + "\n")
        folder = os.path.join(L.ROOT, "data", "life", day)
        if os.path.isdir(folder):
            with open(os.path.join(folder, "report.txt"), "w") as f:
                f.write(text + "\n")


if __name__ == "__main__":
    main()
