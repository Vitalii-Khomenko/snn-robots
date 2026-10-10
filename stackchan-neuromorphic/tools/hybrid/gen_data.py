"""Random worlds for training the perception networks.

  ./py gen_data.py vision 120 --minutes 6   # 120 random scenarios, vision data
  ./py gen_data.py audio 80 --minutes 6

Writes scenarios to data/scn_<mode>/ and records to data/<mode>/ with
simulator/build/stackchan_datagen (build with simulator/build_datagen.sh).
"""
import argparse
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATAGEN = os.path.join(ROOT, "simulator", "build", "stackchan_datagen")


def t(sec):
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def empty_scenario(seed, minutes, r):
    """Hard negatives: rooms with textures, lamps and windows but nobody in them."""
    L = [f"duration {minutes}:00", f"clock {r.choice([3, 8, 12, 17, 21])}:00", f"seed {seed}",
         f"room_light {r.choice(['on', 'off'])}", "sunrise 07:30", "sunset 19:30"]
    if r.random() < 0.8:
        L.append(f"window {r.uniform(-180, 180):.0f} {r.uniform(5, 30):.0f} {r.uniform(20, 60):.0f} {r.uniform(15, 40):.0f} gain {r.uniform(2, 8):.1f}")
    for i in range(r.choice([0, 1, 2, 3, 4])):
        L.append(f"lamp l{i} {r.uniform(-180, 180):.0f} {r.uniform(-5, 35):.0f} {r.choice(['on', 'on', 'off'])} brightness {r.uniform(2, 12):.1f} radius {r.uniform(2, 8):.1f}")
    return "\n".join(L) + "\n"


def vision_scenario(seed, minutes):
    r = random.Random(seed)
    dur = minutes * 60
    if seed >= 3000:
        return empty_scenario(seed, minutes, r)
    style = seed % 3  # 0 general, 1 approach/retreat heavy, 2 moving-objects heavy
    if style == 1:
        return loom_scenario(seed, minutes, r)
    L = [f"duration {minutes}:00", f"clock {r.choice([2, 7, 10, 13, 16, 19, 21, 23])}:{r.randrange(60):02d}", f"seed {seed}",
         f"room_light {r.choice(['on', 'on', 'off'])}", "sunrise 07:30", "sunset 19:30"]
    if r.random() < 0.6:
        L.append(f"window {r.uniform(-180, 180):.0f} {r.uniform(5, 30):.0f} {r.uniform(20, 50):.0f} {r.uniform(15, 35):.0f}")
    lamps = []
    for i in range(r.choice([0, 0, 1, 1, 2, 3])):
        name = f"lamp{i}"
        L.append(f"lamp {name} {r.uniform(-180, 180):.0f} {r.uniform(-5, 30):.0f} {r.choice(['on', 'on', 'off'])} "
                 f"brightness {r.uniform(2, 9):.1f} radius {r.uniform(2, 5):.1f}")
        lamps.append(name)
    # lamps and room light toggle now and then
    for _ in range(r.randrange(2, 10)):
        if lamps and r.random() < 0.7:
            L.append(f"at {t(r.uniform(5, dur))} lamp {r.choice(lamps)} {r.choice(['on', 'off'])}")
        else:
            L.append(f"at {t(r.uniform(5, dur))} room_light {r.choice(['on', 'off'])}")
    # people
    for p in range(r.choice([0, 1, 1, 2, 3, 3])):
        name = f"p{p}"
        now = r.uniform(3, dur * 0.3)
        az, dist = r.uniform(-90, 90), r.uniform(0.8, 4.0)
        L.append(f"at {t(now)} person {name} enter {az:.0f} {dist:.1f}")
        for _ in range(r.randrange(5, 16)):
            now += r.uniform(2, 12)
            if now >= dur - 2:
                break
            c = r.random()
            if c < 0.4:
                L.append(f"at {t(now)} person {name} walk {r.uniform(-120, 120):.0f} {r.uniform(1.5, 8):.1f}")
            elif c < 0.7:
                L.append(f"at {t(now)} person {name} approach {r.uniform(0.4, 4.0):.1f} {r.uniform(1.0, 4.0):.1f}")
            else:
                L.append(f"at {t(now)} person {name} step_back {r.uniform(0.8, 4):.1f}")
        if r.random() < 0.6 and now < dur - 3:
            L.append(f"at {t(min(dur - 1, now + r.uniform(3, 20)))} person {name} leave")
    # generic moving objects
    for k in range(r.choice([2, 4, 6, 8]) if style == 2 else r.choice([0, 1, 2])):
        name = f"o{k}"
        now = r.uniform(3, dur - 10)
        L.append(f"at {t(now)} thing {name} enter {r.uniform(-120, 120):.0f} {r.uniform(-10, 30):.0f} {r.uniform(4, 16):.0f} {r.uniform(0.15, 0.95):.2f}")
        for _ in range(r.randrange(3, 10)):
            now += r.uniform(1, 8)
            if now >= dur - 1:
                break
            L.append(f"at {t(now)} thing {name} move {r.uniform(-120, 120):.0f} {r.uniform(-10, 30):.0f} {r.uniform(0.8, 6):.1f}")
        L.append(f"at {t(min(dur - 0.5, now + r.uniform(2, 15)))} thing {name} leave")
    return "\n".join(L) + "\n"


def loom_scenario(seed, minutes, r):
    dur = minutes * 60
    L = [f"duration {minutes}:00", f"clock {r.choice([9, 12, 15, 18])}:00", f"seed {seed}", "room_light on",
         "sunrise 07:30", "sunset 19:30"]
    if r.random() < 0.5:
        L.append(f"lamp l0 {r.uniform(-150, 150):.0f} {r.uniform(0, 25):.0f} on brightness {r.uniform(3, 8):.1f}")
    az = r.uniform(-25, 25)
    now = r.uniform(1, 4)
    dist = r.uniform(1.5, 3.5)
    L.append(f"at {t(now)} person p0 enter {az:.0f} {dist:.1f}")
    while now < dur - 6:
        now += r.uniform(0.8, 5.0)
        c = r.random()
        if c < 0.38:
            nd = max(0.4, dist - r.uniform(0.8, 2.5))
            L.append(f"at {t(now)} person p0 approach {nd:.1f} {r.uniform(0.8, 3.0):.1f}")
            dist = nd
        elif c < 0.76:
            nd = min(4.5, dist + r.uniform(0.8, 2.5))
            L.append(f"at {t(now)} person p0 approach {nd:.1f} {r.uniform(0.8, 3.0):.1f}")
            dist = nd
        else:
            az = max(-60.0, min(60.0, az + r.uniform(-40, 40)))
            L.append(f"at {t(now)} person p0 walk {az:.0f} {r.uniform(1, 4):.1f}")
    return "\n".join(L) + "\n"


CLIPS = None


def clip_pool(val):
    global CLIPS
    if CLIPS is None:
        import json
        CLIPS = json.load(open(os.path.join(HERE, "data", "clips", "index.json")))
    return [c for c in CLIPS if (c["fold"] == 5) == val]


SOUND_KINDS = ["clap", "talk", "bang", "hiss", "tone", "whistle"]


def audio_scenario(seed, minutes):
    r = random.Random(seed)
    dur = minutes * 60
    L = [f"duration {minutes}:00", f"seed {seed}", "clock 14:00", f"noise_floor {r.choice([15, 25, 25, 40, 70])}"]
    now = r.uniform(1, 4)
    noise_on = False
    pool = clip_pool(seed % 10 == 0) if os.path.exists(os.path.join(HERE, "data", "clips", "index.json")) else []
    while now < dur - 3:
        if pool and r.random() < 0.6:
            c = r.choice(pool)
            cls = c["classes"] or "none"
            L.append(f"at {t(now)} clip {r.uniform(-120, 120):.0f} {os.path.join(HERE, 'data', 'clips', c['file'])} "
                     f"{r.choice([0.25, 0.4, 0.6, 0.8, 1.0, 1.0])} {cls}")
            now += r.uniform(1.0, 4.5)
            continue
        kind = r.choice(SOUND_KINDS + ["noise", "noise"])
        az = r.uniform(-120, 120)
        lvl = r.choice([0.25, 0.4, 0.6, 0.8, 1.0, 1.0])
        if kind == "clap":
            L.append(f"at {t(now)} clap {az:.0f} {lvl:.2f}")
            if r.random() < 0.3:  # a clap train
                for k in range(1, r.randrange(2, 6)):
                    L.append(f"at {t(now + k * r.uniform(0.3, 0.8))} clap {az:.0f} {lvl:.2f}")
        elif kind == "talk":
            L.append(f"at {t(now)} talk {az:.0f} {r.uniform(0.8, 5):.1f} {lvl:.2f}")
        elif kind == "bang":
            L.append(f"at {t(now)} bang {az:.0f} {lvl:.2f}")
        elif kind == "hiss":
            L.append(f"at {t(now)} hiss {az:.0f} {r.uniform(0.2, 1.5):.1f} {lvl:.2f}")
        elif kind == "tone":
            hz = r.choice([180, 250, 440, 800, 1200, 2000, 3000, 4200])
            L.append(f"at {t(now)} tone {az:.0f} {hz} {r.uniform(0.3, 2.5):.1f} {lvl:.2f}")
        elif kind == "whistle":
            a = r.choice([600, 1000, 1800, 2800])
            L.append(f"at {t(now)} whistle {az:.0f} {a} {a * r.choice([0.5, 1.5, 2.0]):.0f} {r.uniform(0.4, 2.0):.1f} {lvl:.2f}")
        else:
            if noise_on:
                L.append(f"at {t(now)} noise 0")
                noise_on = False
            else:
                L.append(f"at {t(now)} noise {r.choice([150, 250, 400, 700])} {r.choice([0, 0.5, 0.9, 1.0])}")
                noise_on = True
        now += r.uniform(0.6, 6.0)
    return "\n".join(L) + "\n"


def run(job):
    mode, scn, out, minutes, seed = job
    subprocess.run([DATAGEN, mode, scn, "--out", out, "--minutes", str(minutes), "--seed", str(seed)],
                   check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["vision", "audio"])
    ap.add_argument("count", type=int)
    ap.add_argument("--minutes", type=float, default=6)
    ap.add_argument("--first-seed", type=int, default=1000)
    a = ap.parse_args()
    scn_dir = os.path.join(HERE, "data", f"scn_{a.mode}")
    out_dir = os.path.join(HERE, "data", a.mode)
    os.makedirs(scn_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for i in range(a.count):
        seed = a.first_seed + i
        scn = os.path.join(scn_dir, f"{a.mode}_{seed}.scn")
        with open(scn, "w") as f:
            f.write((vision_scenario if a.mode == "vision" else audio_scenario)(seed, int(a.minutes)))
        jobs.append((a.mode, scn, os.path.join(out_dir, f"{a.mode}_{seed}.bin"), a.minutes, seed))
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        list(ex.map(run, jobs))
    print(f"{len(jobs)} {a.mode} files in {out_dir}")
