#!/usr/bin/env python3
"""Explicit bounded capture, without pyserial DTR/RTS reset sequences."""
import argparse
import fcntl
import os
import select
import sys
import termios
import time
import tty

parser = argparse.ArgumentParser()
parser.add_argument("port", nargs="?", default="/dev/ttyACM0")
parser.add_argument("--seconds", type=float, default=20)
args = parser.parse_args()
if not 0 < args.seconds <= 60:
    parser.error("--seconds must be in (0, 60]")
port = os.path.realpath(args.port)
with open("/tmp/stackchan-" + os.path.basename(port) + ".lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        fcntl.ioctl(fd, termios.TIOCEXCL)
        tty.setraw(fd, termios.TCSANOW)
        attributes = termios.tcgetattr(fd)
        attributes[2] &= ~(termios.HUPCL | termios.CRTSCTS)
        attributes[4] = attributes[5] = termios.B115200
        termios.tcsetattr(fd, termios.TCSANOW, attributes)
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            if select.select([fd], [], [], min(0.2, max(0.0, until - time.monotonic())))[0]:
                data = os.read(fd, 4096)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
    finally:
        fcntl.ioctl(fd, termios.TIOCNXCL)
        os.close(fd)
