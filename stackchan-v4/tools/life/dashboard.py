"""A day of the robot's life as one HTML page with charts (no packages beyond numpy; open it in any browser).

  ./py ../life/dashboard.py [YYYY-MM-DD]     (from tools/hybrid; default today) -> data/life/<day>/dashboard.html

Per minute of the day: who was around (person, face, voice, owner at home), what reached the robot (touches, shakes),
what it did (head travel, its own voice), how varied it was (posture entropy per 10 minutes), its needs and reward
(stage 2) and the battery. Thin
vertical lines are boots. Minutes without diary are gaps.
"""
import datetime as dt
import html
import os
import sys
import warnings
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lifedata as L  # noqa: E402

warnings.filterwarnings("ignore", "Mean of empty slice")  # minutes without the motivation columns
W, PAD_L, PAD_R = 960, 56, 16
EMO = [c for c in L.COLUMNS if c.startswith("act_emo_")]


def minute_index(t, day0):
    return ((t - day0) // 60).astype(int)


def per_minute(values, idx, n, how):
    out = np.full(n, np.nan)
    for m in np.unique(idx):
        if 0 <= m < n:
            sel = values[idx == m]
            out[m] = how(sel)
    return out


def entropy(labels):
    if len(labels) == 0:
        return np.nan
    p = np.array(list(Counter(labels).values()), float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def panel(title, note, series, height, ymax, t0, t1, boots, unit=""):
    """series: list of (label, css colour var, minute values, 'line'|'bar')."""
    plot_w = W - PAD_L - PAD_R
    top, bottom = 34, height - 22
    X = lambda m: PAD_L + (m - t0) / max(1, t1 - t0) * plot_w  # noqa: E731
    Y = lambda v: bottom - min(v, ymax) / ymax * (bottom - top)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{PAD_L}" y="18" class="t">{html.escape(title)}</text>',
             f'<text x="{W - PAD_R}" y="18" class="n" text-anchor="end">{html.escape(note)}</text>']
    for frac in (0, 0.5, 1):
        y = Y(ymax * frac)
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="g"/>')
        parts.append(f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" class="n" text-anchor="end">{ymax * frac:g}{unit}</text>')
    h0 = int(np.ceil(t0 / 60))
    for h in range(h0, int(t1 // 60) + 1):
        x = X(h * 60)
        parts.append(f'<text x="{x:.1f}" y="{height - 6}" class="n" text-anchor="middle">{h % 24:02d}:00</text>')
    for b in boots:
        if t0 <= b <= t1:
            parts.append(f'<line x1="{X(b):.1f}" x2="{X(b):.1f}" y1="{top}" y2="{bottom}" class="boot"/>')
    legend_x = PAD_L + 260
    for k, (label, colour, vals, kind) in enumerate(series):
        parts.append(f'<rect x="{legend_x + k * 130}" y="9" width="10" height="10" rx="2" fill="var({colour})"/>'
                     f'<text x="{legend_x + k * 130 + 14}" y="18" class="n">{html.escape(label)}</text>')
        if kind == "bar":
            bw = max(1.0, plot_w / max(1, t1 - t0) * 0.9)
            for m in range(int(t0), int(t1) + 1):
                v = vals[m] if m < len(vals) else np.nan
                if np.isfinite(v) and v > 0:
                    parts.append(f'<rect x="{X(m):.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{bottom - Y(v):.1f}" '
                                 f'fill="var({colour})"/>')
        else:
            path, pen = [], False
            for m in range(int(t0), int(t1) + 1):
                v = vals[m] if m < len(vals) else np.nan
                if np.isfinite(v):
                    path.append(f'{"L" if pen else "M"}{X(m):.1f} {Y(v):.1f}')
                    pen = True
                else:
                    pen = False
            parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="var({colour})" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def build(day):
    ticks, lines = L.load(day)
    if ticks is None:
        return None
    day0 = dt.datetime.fromisoformat(day).timestamp()
    t = ticks["recv"]
    idx = minute_index(t, day0)
    n = 24 * 60
    t0, t1 = max(0, idx.min() - 5), min(n - 1, idx.max() + 5)
    v = ticks["values"]
    c = lambda name: v[:, L.col(name)]  # noqa: E731
    pct = lambda name: per_minute(c(name) > 0.5, idx, n, lambda s: 100 * s.mean())  # noqa: E731

    def onsets_per_min(x):
        on = x > 0.5
        start = on & ~np.concatenate([[False], on[:-1]])
        return per_minute(start, idx, n, lambda s: float(s.sum()))

    # head travel: body yaw is degrees / 100, pitch degrees / 35 (BodySense)
    deg = np.concatenate([[0.0], np.hypot(np.diff(c("yaw") * 100), np.diff(c("pitch") * 35))])
    deg[np.concatenate([[True], np.diff(ticks["boot"].astype(np.int64)) != 0])] = 0
    emo = np.argmax(v[:, [L.col(e) for e in EMO]], axis=1)
    posture = emo * 4 + (c("act_sound_act") > 0.5) * 2 + (c("act_hold") > 0.5)
    ent = np.full(n, np.nan)
    for m0 in range(0, n, 10):
        sel = (idx >= m0) & (idx < m0 + 10)
        if sel.sum() > 600:
            ent[m0:m0 + 10] = entropy(posture[sel])
    boots = []
    seen = set()
    for tt, b, ms, _ in lines:
        if b not in seen:
            seen.add(b)
            boots.append((tt - ms / 1000 - day0) / 60)

    panels = [
        panel("Who was around", "% of each minute", [
            ("person", "--c1", pct("person"), "line"), ("face", "--c2", pct("face"), "line"),
            ("voice", "--c3", pct("speech"), "line"), ("owner home", "--c4", pct("owner_near"), "line")],
            200, 100, t0, t1, boots, "%"),
        panel("What reached it", "touches and shakes per minute", [
            ("head touch", "--c1", onsets_per_min(c("touch_head")), "bar"),
            ("screen touch", "--c2", onsets_per_min(c("touch_screen")), "bar"),
            ("shake / lift", "--c5", onsets_per_min(np.maximum(c("shake"), c("tilt"))), "bar")],
            170, 10, t0, t1, boots),
        panel("What it did", "head travel, degrees per minute", [
            ("head travel", "--c1", per_minute(deg, idx, n, lambda s: float(s.sum())), "bar")],
            170, 600, t0, t1, boots),
        panel("Its own voice", "% of each minute it was speaking", [
            ("speaking", "--c3", pct("speaker"), "bar")], 150, 50, t0, t1, boots, "%"),
        panel("How varied", "posture entropy per 10 min (face x voice x stillness), bits", [
            ("entropy", "--c2", ent, "line")], 150, 6, t0, t1, boots),
        panel("Needs", "0 = satisfied, 1 = pressing (from 2026-10-11, main/hybrid/motivation.h)", [
            ("social", "--c1", per_minute(c("need_social"), idx, n, np.nanmean), "line"),
            ("stimulation", "--c2", per_minute(c("need_stimulation"), idx, n, np.nanmean), "line"),
            ("rest", "--c3", per_minute(c("need_rest"), idx, n, np.nanmean), "line"),
            ("comfort", "--c5", per_minute(c("need_comfort"), idx, n, np.nanmean), "line"),
            ("energy", "--c4", per_minute(c("need_energy"), idx, n, np.nanmean), "line")],
            190, 1, t0, t1, boots),
        panel("Reward", "sum per minute; positive = a need was eased", [
            ("reward", "--c3", per_minute(np.maximum(c("reward"), 0), idx, n, np.nansum), "bar"),
            ("cost", "--c5", per_minute(np.maximum(-c("reward"), 0), idx, n, np.nansum), "bar")],
            160, 0.5, t0, t1, boots),
        panel("World model", "prediction error of the next tick (RMS), from 2026-10-11 (main/hybrid/world_model.h)", [
            ("vision", "--c1", per_minute(c("wm_rms_vision"), idx, n, np.nanmean), "line"),
            ("audio", "--c3", per_minute(c("wm_rms_audio"), idx, n, np.nanmean), "line"),
            ("body", "--c2", per_minute(c("wm_rms_body"), idx, n, np.nanmean), "line")],
            170, 0.2, t0, t1, boots),
        panel("Curiosity", "learning progress: the error falling (slow minus fast average) x 10", [
            ("learning progress", "--c4", per_minute(c("wm_lp10"), idx, n, np.nanmean), "line")],
            150, 0.1, t0, t1, boots),
        panel("Battery", "%", [("battery", "--c4", per_minute(c("battery") * 100, idx, n, np.mean), "line")],
              140, 100, t0, t1, boots, "%"),
    ]
    missing = 0
    for b in np.unique(ticks["boot"]):
        s = np.sort(ticks["seq"][ticks["boot"] == b]).astype(np.int64)
        missing += int(np.sum(np.diff(s) - 1))
    summary = (f"{len(v) / 72000:.1f} h of life in the diary, {len(seen)} boot(s), {missing} ticks missing "
               f"({100 * missing / max(1, len(v) + missing):.2f} %), {len(lines)} log lines")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>StackChan day {day}</title>
<style>
:root {{ --bg:#fbfbfa; --ink:#1f2328; --quiet:#6b7280; --grid:#e5e7eb; --c1:#2563eb; --c2:#d97706; --c3:#059669;
  --c4:#7c3aed; --c5:#dc2626; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181d; --ink:#e6e6e6; --quiet:#9aa0a6; --grid:#2b2f36;
  --c1:#60a5fa; --c2:#fbbf24; --c3:#34d399; --c4:#a78bfa; --c5:#f87171; }} }}
body {{ background:var(--bg); color:var(--ink); font:14px/1.4 system-ui, sans-serif; margin:0 auto; max-width:{W}px;
  padding:16px; }}
svg {{ width:100%; height:auto; display:block; margin:8px 0 18px; }}
.t {{ fill:var(--ink); font-weight:600; font-size:14px; }} .n {{ fill:var(--quiet); font-size:11px; }}
.g {{ stroke:var(--grid); }} .boot {{ stroke:var(--quiet); stroke-dasharray:3 3; }}
h1 {{ font-size:18px; margin:0 0 4px; }} p {{ color:var(--quiet); margin:0 0 8px; }}
</style></head><body>
<h1>StackChan, {day}</h1><p>{html.escape(summary)}. Dashed lines: boots.</p>
{"".join(panels)}
</body></html>"""


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    page = build(day)
    if page is None:
        sys.exit(f"{day}: no diary")
    out = os.path.join(L.ROOT, "data", "life", day, "dashboard.html")
    with open(out, "w") as f:
        f.write(page)
    print(out)


if __name__ == "__main__":
    main()
