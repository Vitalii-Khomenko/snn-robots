#!/usr/bin/env python3
"""The PC end of the robot's diary (main/net/life_log.h; read with tools/v4/diary.py): receives the UDP datagrams and keeps them forever.

  data/life/YYYY-MM-DD/HH.scl   every datagram as received: uint16 length, float64 receive time (unix s), payload
  data/life/YYYY-MM-DD/log.txt  the robot's log lines, readable (receive time, boot id, ms since boot, text)
  data/life/robot.json          where the robot is (its address, for tools/robot/ota.py), its current boot id and the
                                MAC it was first seen with: a datagram from another MAC (ARP table) moves the address
                                no more (2026-10-28, an audit: any UDP sender could take the robot's address and be
                                sent the OTA token). A new robot: delete robot.json.

Plain python, no packages: it runs as a systemd user service (tools/robot/lifelogd.service). Every minute it prints
one line: datagrams, ticks and lines received, and how many datagrams were lost on the way (gaps in their numbers).

  python3 tools/robot/lifelogd.py [--port 47000] [--dir data/life]
"""
import argparse
import json
import os
import re
import signal
import socket
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HEADER = struct.Struct("<4sBBHII")  # magic, kind, n, n_values, boot id, datagram seq
ANSI = re.compile(r"\x1b\[[0-9;]*m")


TICK_BYTES = 4 + 4 + 2  # a kind 3 tick: seq, ms, flags, then n_values float16


def checked(data):
    """(kind, n, boot, seq) of a well-formed datagram, else None: the size must fit the kind (2026-10-28, an audit: a
    16-byte kind 2 datagram stopped the service with a struct.error, after it was already stored)."""
    if len(data) < HEADER.size or data[:4] != b"SCL1":
        return None
    _, kind, n, nv, boot, seq = HEADER.unpack_from(data)
    if kind == 3:
        if len(data) != HEADER.size + 8 + n * (TICK_BYTES + 2 * nv) or n == 0:
            return None
    elif kind == 2:
        if len(data) < HEADER.size + 4:
            return None
    elif kind != 1:  # platform B's ticks (history): kept as they come
        return None
    return kind, n, boot, seq


def mac_of(address):
    """The MAC the PC's ARP table has for this IPv4 address, or None."""
    try:
        for line in open("/proc/net/arp").read().splitlines()[1:]:
            w = line.split()
            if len(w) >= 4 and w[0] == address and w[3] != "00:00:00:00:00:00":
                return w[3].lower()
    except OSError:
        pass
    return None


def write_robot(path, info):
    """robot.json, atomically (ota.py may read it at any moment)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(info, f)
    os.replace(tmp, path)


class Day:
    """The open files of the current hour; switches at the hour and the day."""

    def __init__(self, root):
        self.root = root
        self.key = None
        self.bin = self.txt = None

    def files(self, t):
        lt = time.localtime(t)
        key = (time.strftime("%Y-%m-%d", lt), lt.tm_hour)
        if key != self.key:
            self.close()
            folder = os.path.join(self.root, key[0])
            os.makedirs(folder, exist_ok=True)
            self.bin = open(os.path.join(folder, f"{key[1]:02d}.scl"), "ab")
            self.txt = open(os.path.join(folder, "log.txt"), "a", encoding="utf-8")
            self.key = key
        return self.bin, self.txt

    def flush(self):
        for f in (self.bin, self.txt):
            if f:
                f.flush()

    def close(self):
        for f in (self.bin, self.txt):
            if f:
                f.close()
        self.bin = self.txt = None


def pc_state():
    """One line about the PC the robot hangs on by USB: mains, battery, CPU load, GPU load. False touches and servo
    hangs came in hours, 2026-10-11; is it the laptop's power (an ungrounded charger), or its load?"""
    def read(path):
        try:
            return open(path).read().strip()
        except OSError:
            return "?"
    ac = read("/sys/class/power_supply/AC/online")
    bat = read("/sys/class/power_supply/BAT0/capacity")
    st = read("/sys/class/power_supply/BAT0/status")
    load = read("/proc/loadavg").split(" ")[0]
    gpu = "?"
    try:
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().replace(", ", "%/") + "W"
    except Exception:  # noqa: BLE001 - no GPU tool: just not reported
        pass
    return f"ac={ac} battery={bat}% {st.replace(' ', '_')} load={load} gpu={gpu}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47000)
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "life"))
    a = ap.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
    sock.bind(("0.0.0.0", a.port))
    sock.settimeout(1.0)
    day = Day(a.dir)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    print(f"lifelogd: listening on udp {a.port}, writing {a.dir}", flush=True)
    last_seq = {}  # boot id -> last datagram seq
    robot = None  # (address, boot id) last written to robot.json
    robot_path = os.path.join(a.dir, "robot.json")
    try:
        pinned_mac = json.load(open(robot_path)).get("mac")
    except (OSError, ValueError):
        pinned_mac = None
    stats = dict(dgrams=0, ticks=0, lines=0, lost=0, bad=0)
    # the silence watch: nothing from the robot for 2 minutes is written into the day's log.txt (it froze on
    # 2026-10-11 18:10 and nobody knew when), and so is its return
    last_heard, silent_since, last_boot = None, None, None
    next_report = time.time() + 60
    next_flush = time.time() + 5
    try:
        while not stop:
            try:
                data, sender = sock.recvfrom(2048)
            except socket.timeout:
                data = None
            except KeyboardInterrupt:
                break
            now = time.time()
            if data:
                head = checked(data)
                mac = None
                if head is not None and robot is not None and robot[0] != sender[0]:
                    mac = mac_of(sender[0])
                    if pinned_mac and mac != pinned_mac:  # another device, or one the ARP table does not know yet
                        head = None
                        stats["foreign"] = stats.get("foreign", 0) + 1
                if head is None:
                    stats["bad"] += 1
                else:
                    kind, n, boot, seq = head
                    if robot != (sender[0], boot):
                        robot = (sender[0], boot)
                        os.makedirs(a.dir, exist_ok=True)
                        pinned_mac = pinned_mac or mac or mac_of(sender[0])
                        write_robot(robot_path, dict(address=sender[0], boot=f"{boot:08x}", since=now, mac=pinned_mac))
                    prev = last_seq.get(boot)
                    if prev is not None and seq > prev + 1:
                        stats["lost"] += seq - prev - 1
                    if prev is None or seq > prev:
                        last_seq[boot] = seq
                    binf, txtf = day.files(now)
                    if silent_since is not None:
                        gap = (now - silent_since) / 60
                        txtf.write(f"{time.strftime('%H:%M:%S', time.localtime(now))} -------- PC: the robot is back "
                                   f"after {gap:.1f} min of silence ({'new boot' if boot != last_boot else 'same boot, no reboot'})\n")
                        print(f"lifelogd: robot back after {gap:.1f} min", flush=True)
                        silent_since = None
                    last_heard, last_boot = now, boot
                    binf.write(struct.pack("<Hd", len(data), now) + data)
                    stats["dgrams"] += 1
                    if kind in (1, 3):  # 1 platform B's ticks, 3 brain v4's
                        stats["ticks"] += n
                    elif kind == 2:
                        stats["lines"] += 1
                        ms = struct.unpack_from("<I", data, HEADER.size)[0]
                        text = ANSI.sub("", data[HEADER.size + 4:].decode("utf-8", "replace")).strip()
                        stamp = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"
                        txtf.write(f"{stamp} {boot:08x} {ms:>10d} {text}\n")
            if last_heard is not None and silent_since is None and now - last_heard > 120:
                silent_since = last_heard
                _, txtf = day.files(now)
                stamp = time.strftime("%H:%M:%S", time.localtime(last_heard))
                txtf.write(f"{time.strftime('%H:%M:%S', time.localtime(now))} -------- PC: the robot is silent "
                           f"since {stamp} (no diary for 2 min)\n")
                print(f"lifelogd: robot silent since {stamp}", flush=True)
            if now >= next_flush:
                day.flush()
                next_flush = now + 5
            if now >= next_report:
                _, txtf = day.files(now)
                txtf.write(f"{time.strftime('%H:%M:%S', time.localtime(now))} -------- PC: power {pc_state()}\n")
                print("lifelogd: last minute " + " ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
                stats = dict.fromkeys(stats, 0)
                next_report = now + 60
    finally:
        day.close()
        print("lifelogd: stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
