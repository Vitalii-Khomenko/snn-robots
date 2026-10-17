#!/usr/bin/env python3
"""Turns a simulator run (trace.csv, log.txt, summary.txt) into one
self-contained report.html with timeline charts. Standard library only.

    python3 simulator/report.py simulator/out/<scenario>
"""
import csv
import html
import re
import sys
from pathlib import Path

W, PLOT_H, LEFT, RIGHT = 1100, 120, 70, 20


def load(run):
    rows = list(csv.DictReader(open(run / "trace.csv")))
    step = max(1, len(rows) // 1000)  # ~1 point per pixel column
    rows = rows[::step]
    events = []
    for line in open(run / "log.txt", errors="replace"):
        m = re.match(r"(\w) \((\d+)\) ([\w_]+): (.*)", line.strip())
        if not m:
            continue
        t = int(m.group(2)) / 1000.0
        tag, msg = m.group(3), m.group(4)
        if tag == "world" and "exposure" not in msg:
            events.append((t, "world", msg))
        elif tag == "neural_drive":
            events.append((t, "drive", msg))
        elif tag == "runtime" and any(k in msg for k in ("touch", "loud sound", "tilt", "shake", "presence")):
            events.append((t, "sense", msg))
    return rows, events


def xmap(t, t0, t1):
    return LEFT + (t - t0) / max(t1 - t0, 1e-9) * (W - LEFT - RIGHT)


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02d}"


def chart(title, rows, series, lo, hi, t0, t1, unit=""):
    out = [f'<svg viewBox="0 0 {W} {PLOT_H + 30}" class="chart"><text x="4" y="14" class="title">{html.escape(title)}</text>']
    top, bottom = 22, 22 + PLOT_H
    for frac in (0, 0.5, 1):
        y = bottom - frac * (bottom - top)
        val = lo + frac * (hi - lo)
        out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{LEFT - 6}" y="{y + 4:.1f}" class="axis" text-anchor="end">{val:g}{unit}</text>')
    for key, color, label in series:
        pts = []
        for r in rows:
            try:
                v = float(r[key])
            except (KeyError, ValueError):
                continue
            v = min(max(v, lo), hi)
            y = bottom - (v - lo) / (hi - lo) * (bottom - top)
            pts.append(f"{xmap(float(r['t_s']), t0, t1):.1f},{y:.1f}")
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.4"/>')
    lx = 24 + 7.5 * len(title)  # legend starts after the title
    for key, color, label in series:
        out.append(f'<rect x="{lx}" y="5" width="12" height="4" fill="{color}"/><text x="{lx + 16}" y="12" class="axis">{html.escape(label)}</text>')
        lx += 20 + 7 * len(label)
    out.append("</svg>")
    return "\n".join(out)


def event_strip(events, t0, t1):
    colors = {"world": "#6b7280", "drive": "#db2777", "sense": "#2563eb"}
    rows = {"world": 0, "sense": 1, "drive": 2}
    names = {"world": "world (scenario)", "sense": "sensors", "drive": "neural events"}
    out = [f'<svg viewBox="0 0 {W} 90" class="chart"><text x="4" y="14" class="title">Events</text>']
    for kind, row in rows.items():
        y = 30 + row * 20
        out.append(f'<text x="{LEFT - 6}" y="{y + 4}" class="axis" text-anchor="end">{names[kind]}</text>')
        out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y}" y2="{y}" class="grid"/>')
    for t, kind, msg in events:
        x = xmap(t, t0, t1)
        y = 30 + rows[kind] * 20
        out.append(f'<circle cx="{x:.1f}" cy="{y}" r="3.2" fill="{colors[kind]}"><title>{fmt_time(t)} {html.escape(msg)}</title></circle>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows, events = load(run)
    if not rows:
        sys.exit("empty trace.csv")
    t0, t1 = float(rows[0]["t_s"]), float(rows[-1]["t_s"])
    summary = (run / "summary.txt").read_text() if (run / "summary.txt").exists() else ""
    ticks = "".join(
        f'<span style="left:{xmap(t, t0, t1) / W * 100:.2f}%">{fmt_time(t)}</span>'
        for t in [t0 + (t1 - t0) * i / 8 for i in range(9)]
    )
    parts = [
        chart("Head: yaw (actual servo angle)", rows, [("head_yaw", "#2563eb", "yaw")], -130, 130, t0, t1, "°"),
        chart("Head: pitch", rows, [("head_pitch", "#16a34a", "pitch")], -30, 60, t0, t1, "°"),
        chart("Endogenous drives (spikes per cell per tick)", rows,
              [("bored", "#2563eb", "boredom"), ("explore", "#0891b2", "exploration"), ("lonely", "#f59e0b", "loneliness"),
               ("arouse", "#dc2626", "arousal"), ("sleep", "#7c3aed", "sleep")], 0, 0.2, t0, t1),
        chart("Face (network outputs)", rows, [("valence", "#16a34a", "valence"), ("arousal", "#dc2626", "arousal"),
                                           ("lids", "#7c3aed", "lids")], -1, 1, t0, t1),
        chart("Core", rows, [("core", "#b45309", "active core fraction"), ("assoc", "#ca8a04", "reservoir"),
                             ("map", "#2563eb", "attention map")], 0, 0.3, t0, t1),
        chart("Body and light", rows, [("stamina", "#16a34a", "stamina"), ("light01", "#ea580c", "light spot strength"),
                                    ("lamp_in_view", "#6b7280", "lamp in view")], 0, 1, t0, t1),
        chart("Network tick on ESP32 (model)", rows, [("snn_tick_ms", "#6b7280", "ms")], 0, 60, t0, t1),
        event_strip(events, t0, t1),
    ]
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StackChan sim — {html.escape(run.name)}</title>
<style>
:root {{ --bg:#fff; --fg:#111827; --muted:#6b7280; --grid:#e5e7eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111827; --fg:#f3f4f6; --muted:#9ca3af; --grid:#374151; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px system-ui, sans-serif; margin:16px; max-width:1140px; }}
.chart {{ width:100%; height:auto; display:block; margin:6px 0; }}
.title {{ font-weight:600; font-size:13px; fill:var(--fg); }}
.axis {{ font-size:11px; fill:var(--muted); }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.ticks {{ position:relative; height:16px; font-size:11px; color:var(--muted); }}
.ticks span {{ position:absolute; transform:translateX(-50%); }}
pre {{ font-size:11px; overflow-x:auto; background:rgba(127,127,127,.08); padding:8px; }}
</style></head><body>
<h1>Simulation: {html.escape(run.name)}</h1>
<p>Time since power-on. Hover a point in "Events" for details.</p>
<div class="ticks">{ticks}</div>
{"".join(parts)}
<h2>Per-minute summary</h2><pre>{html.escape(summary)}</pre>
</body></html>"""
    (run / "report.html").write_text(page, encoding="utf-8")
    print(f"wrote {run / 'report.html'}")


if __name__ == "__main__":
    main()
