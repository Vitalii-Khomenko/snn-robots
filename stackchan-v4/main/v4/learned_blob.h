#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

// What brain v4 learned from its own life (stage 6), as one blob made by tools/v4/sleep_v4.py (pack_learned):
//
//   header (32 B): magic "V4L1", uint32 version, entry count, total bytes, CRC-32 (zlib) of everything after the
//                  header, 3 x uint32 reserved
//   entries (48 B each): char name[32], uint32 type (2 = float32), count, offset, reserved
//   data, every array 16-byte aligned
//
// Arrays: "nn.act1_w", "nn.act1_b", "nn.act2_w", "nn.act2_b" (the main network's action heads, retrained each night)
// "zones.shift" (13: how far each channel's alarm border moved with experience) and "muse.bias" (8: how much more
// or less often each of the Muse's ideas comes, from what the owner answered; muse_v4.h). Plain C++: the robot reads it
// from flash (net/learned_store.h), the simulator from a file (--learned).
namespace learned {

struct Header {
    char magic[4];
    uint32_t version, count, total, crc;
    uint32_t reserved[3];
};
struct Entry {
    char name[32];
    uint32_t type, count, offset, reserved;
};
static_assert(sizeof(Header) == 32 && sizeof(Entry) == 48, "blob layout (tools/v4/sleep_v4.py)");
inline constexpr char kMagic[4] = {'V', '4', 'L', '1'};

// A blob in memory (checked by the caller: magic, size, CRC). find() -> the float array called name if it has exactly
// `count` values, else nullptr.
struct View {
    const uint8_t* base = nullptr;

    uint32_t version() const { return base ? reinterpret_cast<const Header*>(base)->version : 0; }
    const float* find(const char* name, int count) const
    {
        if (!base) return nullptr;
        const auto* hd = reinterpret_cast<const Header*>(base);
        const auto* e = reinterpret_cast<const Entry*>(base + sizeof(Header));
        for (uint32_t i = 0; i < hd->count; ++i, ++e) {
            if (std::strncmp(e->name, name, sizeof(e->name)) != 0) continue;
            if (e->type != 2 || e->count != static_cast<uint32_t>(count) || e->offset % 4 != 0 ||
                e->offset + e->count * 4u > hd->total)
                return nullptr;
            return reinterpret_cast<const float*>(base + e->offset);
        }
        return nullptr;
    }
};

// CRC-32 as zlib computes it (the PC packs with zlib.crc32).
inline uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0)
{
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

// A whole blob checked: magic, size within `size`, CRC.
inline bool valid(const uint8_t* p, size_t size)
{
    if (!p || size < sizeof(Header)) return false;
    const auto* hd = reinterpret_cast<const Header*>(p);
    return std::memcmp(hd->magic, kMagic, 4) == 0 && hd->total >= sizeof(Header) && hd->total <= size &&
           crc32(p + sizeof(Header), hd->total - sizeof(Header)) == hd->crc;
}

}  // namespace learned
