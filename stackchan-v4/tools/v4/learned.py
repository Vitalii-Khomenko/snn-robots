"""What the robot learned from its own life (stage 6) as one blob (main/v4/learned_blob.h) and the zone shift
(main/v4/zone_shift.h), on the PC side.

  blob = pack(version, heads, shift, muse_bias, nn)   heads: {"act1_w": [64, 85], "act1_b": [64], "act2_w": [39, 64],
                                                     "act2_b": [39]}; nn: nn_id() of the network the heads belong to
  version, arrays = unpack(blob)
  x2 = warp(x, channel, shift)              what the robot's analyzer sees of channel value x
  ./py learned.py innate out/learned_innate.bin [VERSION]   the compiled-in heads, no shift (tests the chain; after a
                                                            retrained network: what the robot should start from)
"""
import os
import struct
import sys
import zlib

import numpy as np

from spec_v4 import CHANNELS, SHIFT_MAX, ZONE_TABLE

MAGIC = b"V4L1"
HEADER = struct.Struct("<4sIIII12x")
ENTRY = struct.Struct("<32sIIII")
HEADS = ("act1_w", "act1_b", "act2_w", "act2_b")
MUSE_BIAS_MAX = 0.6  # muse.py BIAS_MAX


def alarm(c):
    """The alarm border of channel c (lower border of its first anxiety / anger zone), or None."""
    z = ZONE_TABLE[CHANNELS[c]]
    los = [z[k][0] for k in ("anxiety", "anger") if k in z]
    return min(los) if los else None


TOLERANCE = 2.0  # main/v4/zone_shift.h kTolerance (2026-10-25): every input twice as tolerable


def tolerant(c, a):
    """main/v4/zone_shift.h ZoneShift::tolerant: the raw level that reads as the alarm border a of channel c (sharp:
    not warped, its tolerance is the encoder's wider scale, channels.py SHARP_FULL; hunger: the body's own need, not
    warped since 2026-10-27, docs/BODY.md step 2)."""
    return a if CHANNELS[c] in ("sharp", "hunger") else min(TOLERANCE * a, a + 0.6 * (1 - a))


def warp(x, c, s):
    """main/v4/zone_shift.h ZoneShift::apply for channel c: tolerant(c, a) + s maps onto a, 0 and 1 stay."""
    a = alarm(c)
    if a is None or a <= 0:
        return np.asarray(x, np.float32)
    m = min(max(tolerant(c, a) + s, 0.02), 0.98)
    x = np.asarray(x, np.float32)
    if m == a:
        return x
    return np.where(x <= m, x * (a / m), a + (x - m) * ((1 - a) / (1 - m))).astype(np.float32)


TRUNK = ("gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0", "emo.weight", "emo.bias")


def nn_id(state):
    """The fingerprint of a main network's trunk (the GRU and the emotion head; the action heads are what the night
    changes): CRC-32 of its float32 weights. export_nn_v4.py compiles it in (nnv4::kNnId); a blob's heads are used only
    with the trunk they were made for (2026-10-27: a retrained network must not take the old network's heads)."""
    crc = 0
    for k in TRUNK:
        v = state[k]
        v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        crc = zlib.crc32(np.ascontiguousarray(v, np.float32).tobytes(), crc)
    return crc & 0xFFFFFFFF


def id_of(arrays):
    """The nn_id stored in an unpacked blob, or None (blobs before 2026-10-27 have none)."""
    a = arrays.get("nn.id")
    return None if a is None or len(a) != 2 else (int(a[0]) << 16) | int(a[1])


def pack(version, heads=None, shift=None, muse_bias=None, nn=None):
    arrays = []
    if heads is not None:
        assert nn is not None, "the heads need the fingerprint of their network (nn_id)"
        arrays += [("nn." + k, np.asarray(heads[k], np.float32).ravel()) for k in HEADS]
        arrays.append(("nn.id", np.array([nn >> 16, nn & 0xFFFF], np.float32)))  # two exact 16-bit halves
    if shift is not None:
        s = np.clip(np.asarray(shift, np.float32), -SHIFT_MAX, SHIFT_MAX)
        assert s.shape == (len(CHANNELS),)
        arrays.append(("zones.shift", s))
    if muse_bias is not None:  # the Muse's ideas (muse.py, sleep_v4.learn_muse)
        b = np.clip(np.asarray(muse_bias, np.float32), -MUSE_BIAS_MAX, MUSE_BIAS_MAX)
        assert b.shape == (8,)
        arrays.append(("muse.bias", b))
    offset = HEADER.size + ENTRY.size * len(arrays)
    table, data = b"", b""
    for name, a in arrays:
        data += b"\0" * ((-(offset + len(data))) % 16)
        table += ENTRY.pack(name.encode(), 2, a.size, offset + len(data), 0)
        data += a.tobytes()
    body = table + data
    return HEADER.pack(MAGIC, version, len(arrays), HEADER.size + len(body), zlib.crc32(body) & 0xFFFFFFFF) + body


def unpack(blob):
    magic, version, count, total, crc = HEADER.unpack_from(blob)
    if magic != MAGIC or total > len(blob) or zlib.crc32(blob[HEADER.size:total]) & 0xFFFFFFFF != crc:
        raise ValueError("not a valid learned blob")
    out = {}
    for i in range(count):
        name, typ, n, off, _ = ENTRY.unpack_from(blob, HEADER.size + i * ENTRY.size)
        out[name.rstrip(b"\0").decode()] = np.frombuffer(blob, np.float32, n, off).copy()
    return version, out


def heads_of(model):
    """The action heads of an nn_v4.NnV4 as numpy arrays."""
    return {"act1_w": model.act1.weight.detach().cpu().numpy(), "act1_b": model.act1.bias.detach().cpu().numpy(),
            "act2_w": model.act2.weight.detach().cpu().numpy(), "act2_b": model.act2.bias.detach().cpu().numpy()}


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "innate":
        version = int(sys.argv[3]) if len(sys.argv) == 4 else 1
        import torch
        from nn_v4 import NnV4
        m = NnV4()
        m.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "data", "nn_v4.pt"), weights_only=False,
                                     map_location="cpu")["state"])
        blob = pack(version, heads_of(m), np.zeros(len(CHANNELS)), nn=nn_id(m.state_dict()))
        open(sys.argv[2], "wb").write(blob)
        print(f"wrote {sys.argv[2]}: {len(blob)} bytes, version {version} (innate heads, no shift)")
    else:
        sys.exit(__doc__)
