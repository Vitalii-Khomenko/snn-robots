#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>

// The diary on the microSD card: every datagram LifeLog makes is also kept here, so nothing is lost while the PC is
// off or Wi-Fi is down. Same record format as the PC's files (tools/life/lifelogd.py): uint16 length, float64 time
// (the robot's unix time, 0 before the clock synced), datagram. Files /sdcard/LIFE/NNNNNNNN.SCL, a new one every
// 10 minutes (8.3 names: long file names are off in this build); the oldest are deleted when less than 15 % of the
// card is free. The card shares the SPI wires with the screen; it is mounted (and formatted if it is not FAT) the
// first time write() runs, from LifeLog's task. Not thread-safe: one task only.
class SdDiary {
public:
    // false while there is no usable card (it retries the mount every 30 s)
    bool write(const void* data, size_t size, double unixTime, uint32_t nowMs);
    bool mounted() const { return mounted_; }
    uint32_t written() const { return written_; }
    uint32_t errors() const { return errors_; }

private:
    bool mount(uint32_t nowMs);
    bool openNext(uint32_t nowMs);
    void prune();
    void fail();

    bool mounted_ = false;
    uint32_t next_mount_ms_ = 0;
    FILE* f_ = nullptr;
    char* buf_ = nullptr;  // stdio buffer in PSRAM: the card gets few large writes
    uint32_t file_no_ = 0, file_opened_ms_ = 0, last_flush_ms_ = 0;
    uint32_t written_ = 0, errors_ = 0;
};
