#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>

// The networks' weights in flash, outside the firmware (platform B: the nightly training replaces them without a
// rebuild). Two data partitions, weights_a and weights_b (partitions.csv). A partition holds one blob, written by
// tools/life/pack_weights.py:
//
//   header (32 B): magic "SCW1", uint32 version, entry count, total bytes, CRC-32 of everything after the header,
//                  3 x uint32 reserved
//   entries (48 B each): char name[32] ("policy.inp_w"), uint32 type (1 int8, 2 float32), count, offset, reserved
//   data, every array 16-byte aligned
//
// NVS "weights": "slot" (0 a, 1 b) is the confirmed one; a new blob is written into the other slot and put on trial
// ("next", "tries"). The robot switches to it at once, without a reboot (requestSwitch(): the main loop calls open()
// again, hands the nets the new arrays, then releaseOld()), and it becomes "slot" after a healthy minute (confirm()).
// A reboot during the trial boots it again; two boots without that minute and the trial ends: back to "slot".
//
// The partition is memory-mapped (read through the flash cache like the compiled-in arrays). Anything missing, of
// the wrong size, or a blob that fails its CRC: the caller keeps the compiled-in weights.
class WeightStore {
public:
    // Picks the slot (trial or confirmed), maps it and checks the blob. false = no usable blob (and nothing changed:
    // a blob mapped before stays). Called again for a switch: the old mapping stays until releaseOld().
    bool open();
    void releaseOld();
    // A new blob is on trial: switch to it now (set by the OTA server's task, taken by the main loop).
    void requestSwitch() { switch_.store(true); }
    bool takeSwitchRequest() { return switch_.exchange(false); }
    // The array called name (e.g. "policy.inp_w") if it has exactly this type and count, else nullptr.
    const void* find(const char* name, bool int8, int count) const;
    uint32_t version() const { return version_; }
    char slot() const { return slot_; }
    bool onTrial() const { return trial_; }
    // After a healthy minute: a blob on trial becomes the confirmed one.
    void confirm();

    // Writing a new blob into the slot that is not live (main/net/ota_server.h). Not while one is on trial.
    bool beginWrite(size_t total);
    bool write(const void* data, size_t size);
    // Checks the CRC of what was written and puts it on trial (the caller then asks for the switch).
    bool finishWrite();
    char idleSlot() const { return slot_ == 'a' ? 'b' : 'a'; }

private:
    const uint8_t* base_ = nullptr;
    uint32_t handle_ = 0, old_handle_ = 0;  // esp_partition_mmap_handle_t
    bool mapped_ = false, old_mapped_ = false;
    std::atomic<bool> switch_{false};
    uint32_t count_ = 0, version_ = 0;
    char slot_ = 'a';
    bool trial_ = false;
    // write state
    const void* wpart_ = nullptr;  // esp_partition_t*
    size_t wtotal_ = 0, wdone_ = 0;
    uint32_t wcrc_ = 0;
    uint8_t whead_[32] = {};
};
