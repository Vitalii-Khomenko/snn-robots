#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "learned_blob.h"

// What brain v4 learned from its own life (stage 6, v4/learned_blob.h) in flash, outside the firmware: the PC's night
// sleep sends a new blob over Wi-Fi (PUT /learned), the robot switches to it at once without a reboot (a reboot makes
// sounds, and the owner sleeps next to the robot), watches it for a minute and keeps it - or goes back to the blob it
// had, also without a reboot.
//
// Two data partitions, weights_a and weights_b (partitions.csv; platform B's weights lived there - their blobs have
// another magic and read as empty). NVS namespace "learned": "slot" (0 a, 1 b) is the confirmed blob, "next" the one
// on trial. A reboot during the trial boots it again; two boots without the healthy minute end the trial.
// No valid blob: the compiled-in (innate) networks and no zone shift.
class LearnedStore {
public:
    // Picks the slot (trial or confirmed), maps it and checks the blob. false = no usable blob. Called again for a
    // switch: the old mapping stays until releaseOld().
    bool open();
    void releaseOld();
    learned::View view() const { return learned::View{base_}; }
    uint32_t version() const { return base_ ? version_ : 0; }
    char slot() const { return slot_; }
    bool onTrial() const { return trial_.load(); }
    // A new blob is on trial: switch to it now (set by the update server's task, taken by the main loop).
    void requestSwitch() { switch_.store(true); }
    bool takeSwitchRequest() { return switch_.exchange(false); }
    // After a healthy minute: the blob on trial becomes the confirmed one.
    void confirm();
    // The blob on trial failed: forget it (the caller then open()s the confirmed one again).
    void revert();

    // Writing a new blob into the slot that is not live (net/ota_server.h). Not while one is on trial, nor while a
    // written one waits for the main loop to switch to it (pending_: 2026-10-28, an audit - a second upload in that
    // moment erased the slot about to be opened).
    bool beginWrite(size_t total);
    bool write(const void* data, size_t size);
    // Checks the blob and puts it on trial (the caller then asks for the switch).
    bool finishWrite();
    char idleSlot() const { return slot_ == 'a' ? 'b' : 'a'; }

private:
    const uint8_t* base_ = nullptr;
    uint32_t handle_ = 0, old_handle_ = 0;  // esp_partition_mmap_handle_t
    bool mapped_ = false, old_mapped_ = false;
    std::atomic<bool> switch_{false}, pending_{false}, trial_{false};
    uint32_t version_ = 0;
    std::atomic<char> slot_{'a'};
    // write state
    const void* wpart_ = nullptr;  // esp_partition_t*
    size_t wtotal_ = 0, wdone_ = 0;
    uint32_t wcrc_ = 0;
    uint8_t whead_[sizeof(learned::Header)] = {};
};
