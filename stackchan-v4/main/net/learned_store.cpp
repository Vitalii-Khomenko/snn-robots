#include "learned_store.h"

#include <algorithm>
#include <cstring>
#include <esp_log.h>
#include <esp_partition.h>
#include <nvs.h>

namespace {
constexpr const char* kTag = "learned";
constexpr const char* kNs = "learned";
constexpr esp_partition_subtype_t kSubtype = static_cast<esp_partition_subtype_t>(0x41);

const esp_partition_t* partition(char slot)
{
    return esp_partition_find_first(ESP_PARTITION_TYPE_DATA, kSubtype, slot == 'b' ? "weights_b" : "weights_a");
}
}  // namespace

bool LearnedStore::open()
{
    // The mapping in use moves aside: the network reads it until the caller hands it the new arrays (same thread,
    // right after this call); releaseOld() unmaps it a moment later.
    if (mapped_) {
        if (old_mapped_) esp_partition_munmap(old_handle_);
        old_handle_ = handle_;
        old_mapped_ = true;
        mapped_ = false;
    }
    base_ = nullptr;
    uint8_t slot = 0, next = 0xff, tries = 0;
    trial_ = false;
    nvs_handle_t h;
    if (nvs_open(kNs, NVS_READWRITE, &h) == ESP_OK) {
        nvs_get_u8(h, "slot", &slot);
        nvs_get_u8(h, "next", &next);
        nvs_get_u8(h, "tries", &tries);
        if (next <= 1) {
            if (tries >= 2) {  // it never reached a healthy minute twice: back to the confirmed blob
                ESP_LOGW(kTag, "the blob in slot %c failed two boots: back to slot %c", next ? 'b' : 'a', slot ? 'b' : 'a');
                nvs_erase_key(h, "next");
                nvs_erase_key(h, "tries");
            } else {
                nvs_set_u8(h, "tries", static_cast<uint8_t>(tries + 1));
                slot = next;
                trial_ = true;
            }
            nvs_commit(h);
        }
        nvs_close(h);
    }
    const char new_slot = slot ? 'b' : 'a';
    const esp_partition_t* part = partition(new_slot);
    if (!part) {
        ESP_LOGW(kTag, "no weights_%c partition: innate networks", new_slot);
        return false;
    }
    const void* map = nullptr;
    esp_partition_mmap_handle_t handle;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &map, &handle) != ESP_OK) {
        ESP_LOGW(kTag, "weights_%c: mmap failed: innate networks", new_slot);
        return false;
    }
    const auto* p = static_cast<const uint8_t*>(map);
    if (!learned::valid(p, part->size)) {
        esp_partition_munmap(handle);
        if (trial_) {
            // the blob on trial does not read (its flash went bad): forget it now and open the confirmed one - left on
            // trial it would have been confirmed after a minute, and the good blob's slot taken for the next upload
            ESP_LOGW(kTag, "weights_%c: the blob on trial does not read: back to the confirmed slot", new_slot);
            revert();
            return open();
        }
        ESP_LOGI(kTag, "weights_%c holds no learned blob: innate networks", new_slot);
        slot_ = new_slot;
        pending_ = false;
        return false;
    }
    handle_ = handle;
    mapped_ = true;
    base_ = p;
    version_ = learned::View{p}.version();
    slot_ = new_slot;
    pending_ = false;
    ESP_LOGI(kTag, "weights_%c: learned blob version %u%s", new_slot, static_cast<unsigned>(version_),
             trial_ ? " (new, on trial)" : "");
    return true;
}

void LearnedStore::releaseOld()
{
    if (!old_mapped_) return;
    esp_partition_munmap(old_handle_);
    old_mapped_ = false;
}

void LearnedStore::confirm()
{
    if (!trial_) return;
    nvs_handle_t h;
    if (nvs_open(kNs, NVS_READWRITE, &h) != ESP_OK) {
        ESP_LOGE(kTag, "confirm: NVS does not open: the blob stays on trial");
        return;
    }
    // the slot first (an interrupted confirm leaves "next" too: the next boot puts the same blob on trial again)
    esp_err_t err = nvs_set_u8(h, "slot", slot_ == 'b' ? 1 : 0);
    if (err == ESP_OK) err = nvs_commit(h);
    if (err == ESP_OK) {
        nvs_erase_key(h, "next");
        nvs_erase_key(h, "tries");
        err = nvs_commit(h);
    }
    nvs_close(h);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "confirm: NVS write failed (%s): the blob stays on trial", esp_err_to_name(err));
        return;
    }
    trial_ = false;
    ESP_LOGI(kTag, "learned blob version %u (slot %c) confirmed after a healthy minute", static_cast<unsigned>(version_),
             slot_.load());
}

void LearnedStore::revert()
{
    trial_ = false;
    nvs_handle_t h;
    if (nvs_open(kNs, NVS_READWRITE, &h) != ESP_OK) return;
    nvs_erase_key(h, "next");
    nvs_erase_key(h, "tries");
    nvs_commit(h);
    nvs_close(h);
}

bool LearnedStore::beginWrite(size_t total)
{
    if (trial_ || pending_) return false;  // the other slot is the fallback of the blob on trial / about to be opened
    const esp_partition_t* part = partition(idleSlot());
    if (!part || total < sizeof(learned::Header) || total > part->size) return false;
    // erase what the blob needs, rounded up to whole 4 KB sectors
    const size_t erase = (total + part->erase_size - 1) / part->erase_size * part->erase_size;
    if (esp_partition_erase_range(part, 0, erase) != ESP_OK) return false;
    wpart_ = part;
    wtotal_ = total;
    wdone_ = 0;
    wcrc_ = 0;
    return true;
}

bool LearnedStore::write(const void* data, size_t size)
{
    const auto* part = static_cast<const esp_partition_t*>(wpart_);
    if (!part || wdone_ + size > wtotal_) return false;
    const auto* p = static_cast<const uint8_t*>(data);
    // the header goes in last, so a blob cut short is never taken for a valid one
    size_t skip = 0;
    if (wdone_ < sizeof(whead_)) {
        skip = std::min(size, sizeof(whead_) - wdone_);
        std::memcpy(whead_ + wdone_, p, skip);
    }
    if (size > skip) {
        wcrc_ = learned::crc32(p + skip, size - skip, wcrc_);
        if (esp_partition_write(part, wdone_ + skip, p + skip, size - skip) != ESP_OK) return false;
    }
    wdone_ += size;
    return true;
}

bool LearnedStore::finishWrite()
{
    const auto* part = static_cast<const esp_partition_t*>(wpart_);
    wpart_ = nullptr;
    learned::Header hd;
    std::memcpy(&hd, whead_, sizeof(hd));
    if (!part || wdone_ != wtotal_ || std::memcmp(hd.magic, learned::kMagic, 4) != 0 || hd.total != wtotal_ ||
        hd.crc != wcrc_) {
        ESP_LOGW(kTag, "new blob rejected (size, magic or CRC)");
        return false;
    }
    if (esp_partition_write(part, 0, whead_, sizeof(whead_)) != ESP_OK) return false;
    nvs_handle_t h;
    if (nvs_open(kNs, NVS_READWRITE, &h) != ESP_OK) return false;
    esp_err_t err = nvs_set_u8(h, "next", idleSlot() == 'b' ? 1 : 0);
    if (err == ESP_OK) err = nvs_set_u8(h, "tries", 0);
    if (err == ESP_OK) err = nvs_commit(h);
    nvs_close(h);
    const bool ok = err == ESP_OK;
    if (ok) pending_ = true;  // no new upload until the main loop has opened this one
    ESP_LOGI(kTag, "new learned blob version %u in slot %c: on trial", static_cast<unsigned>(hd.version), idleSlot());
    return ok;
}
