#!/usr/bin/env python3
"""Updates the robot over Wi-Fi (main/net/ota_server.h). The robot's address comes from data/life/robot.json (written
by lifelogd.py from the diary), the token from secrets/life_log_host.h.

  python3 tools/life/ota.py status
  python3 tools/life/ota.py weights [build/weights.bin]          new weights: switched in at once, no reboot
  python3 tools/life/ota.py firmware [build/stackchan_life.bin]  new firmware, on trial after the reboot
  python3 tools/life/ota.py touch 0..7                            the head pad's sensitivity (no reboot)
  python3 tools/life/ota.py mode brain|collector|life            who acts: v3 policy, babbling, the platform B actor (reboots)

After an upload it waits until the robot confirms the new part after a healthy minute; if the robot falls back to
the old one instead, it says so and exits with 1.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def token():
    text = open(os.path.join(ROOT, "secrets", "life_log_host.h")).read()
    m = re.search(r'#define\s+LIFE_OTA_TOKEN\s+"([^"]+)"', text)
    if not m:
        sys.exit("no LIFE_OTA_TOKEN in secrets/life_log_host.h")
    return m.group(1)


def address():
    path = os.path.join(ROOT, "data", "life", "robot.json")
    if not os.path.exists(path):
        sys.exit("robot address unknown: is lifelogd running and the robot on Wi-Fi?")
    return json.load(open(path))["address"]


def request(method, path, body=None, timeout=10):
    req = urllib.request.Request(f"http://{address()}{path}", data=body, method=method,
                                 headers={"X-Token": token(), "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def status(timeout=5):
    return json.loads(request("GET", "/status", timeout=timeout))


def wait_status(cond, seconds, what):
    end = time.time() + seconds
    while time.time() < end:
        try:
            s = status(timeout=3)
            if cond(s):
                return s
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(3)
    sys.exit(f"timed out waiting for {what}")


def upload_weights(path):
    body = open(path, "rb").read()
    version = int.from_bytes(body[4:8], "little")
    print(f"weights: version {version}, {len(body)} bytes -> {address()} ...", flush=True)
    print("  " + request("PUT", "/weights", body, timeout=180).strip())
    s = wait_status(lambda s: s["weights_version"] == version, 30, "the switch")
    print(f"  switched without a reboot (slot {s['weights_slot']}); on trial for a minute ...", flush=True)
    s = wait_status(lambda s: not s["weights_trial"] or s["weights_version"] != version, 150, "the end of the trial")
    if s["weights_version"] != version or s["weights_trial"]:
        sys.exit(f"  the robot went back to the old weights: {s}")
    print(f"  confirmed: {s}")


def upload(kind, path):
    if kind == "weights":
        return upload_weights(path)
    before = status()
    body = open(path, "rb").read()
    print(f"{kind}: {len(body)} bytes -> {address()} ...", flush=True)
    print("  " + request("PUT", f"/{kind}", body, timeout=180).strip())
    trial_key = "weights_trial" if kind == "weights" else "firmware_trial"
    # the reboot: uptime goes back (the robot may have been up for less than a minute before)
    last = [before["uptime_s"] + 1]

    def rebooted(s):
        back = s["uptime_s"] < last[0]
        last[0] = s["uptime_s"]
        return back

    s = wait_status(rebooted, 120, "the reboot")
    if not s[trial_key]:
        sys.exit(f"  the robot is back but the new {kind} is not on trial: {s}")
    print(f"  rebooted, new {kind} on trial; waiting for a healthy minute ...", flush=True)
    # a rollback shows below as "back to the old one" (the robot reboots into it and runs on)
    s = wait_status(lambda s: s["uptime_s"] >= 65, 200, "a healthy minute")
    if s[trial_key]:
        sys.exit(f"  still on trial after a minute: {s}")
    changed = (s["weights_version"] != before["weights_version"]) if kind == "weights" else \
        (s["firmware"] != before["firmware"])
    if not changed:
        sys.exit(f"  the robot went back to the old {kind}: {s}")
    print(f"  confirmed: {s}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(json.dumps(status(), indent=1))
    elif cmd == "weights":
        upload("weights", sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "build", "weights.bin"))
    elif cmd == "mode" and len(sys.argv) > 2 and sys.argv[2] in ("collector", "brain", "life"):
        print("  " + request("PUT", "/mode", sys.argv[2].encode()).strip())
        s = wait_status(lambda s: s.get("mode") == sys.argv[2] and s["uptime_s"] < 60, 90, "the reboot")
        print(f"  mode now: {s['mode']}")
    elif cmd == "touch" and len(sys.argv) > 2 and sys.argv[2] in "01234567" and len(sys.argv[2]) == 1:
        print("  " + request("PUT", "/touch", sys.argv[2].encode()).strip(), "- level now", status()["touch_level"])
    elif cmd == "firmware":
        upload("firmware", sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "build", "stackchan_life.bin"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
