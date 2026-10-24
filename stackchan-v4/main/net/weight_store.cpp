#include "weight_store.h"

#include <algorithm>
#include <cstring>
#include <esp_log.h>
#include <esp_partition.h>
#include <esp_rom_crc.h>
#include <nvs.h>

namespace {
constexpr const char* kTag = "weights";
constexpr esp_partition_subtype_t kSubtype = static_cast<esp_partition_subtype_t>(0x41);

struct Header {
    char magic[4];
    uint32_t version, count, total, crc;
    uint32_t reserved[3];
};
struct Entry {
    char name[32];
    uint32_t type, count, offset, reserved;
};
static_assert(sizeof(Header) == 32 && sizeof(Entry) == 48, "blob layout (tools/life/pack_weights.py)");
}  // namespace

namespace {
const esp_partition_t* partition(char slot)
{
    return esp_partition_find_first(ESP_PARTITION_TYPE_DATA, kSubtype, slot == 'b' ? "weights_b" : "weights_a");
}
}  // namespace

bool WeightStore::open()
{
    uint8_t slot = 0, next = 0xff, tries = 0;
    nvs_handle_t h;
    if (nvs_open("weights", NVS_READWRITE, &h) == ESP_OK) {
        nvs_get_u8(h, "slot", &slot);
        nvs_get_u8(h, "next", &next);
        nvs_get_u8(h, "tries", &tries);
        if (next <= 1) {
            if (tries >= 2) {  // it never reached a healthy minute twice: back to the confirmed slot
                ESP_LOGW(kTag, "new weights in slot %c failed two boots: back to slot %c", next ? 'b' : 'a',
                         slot ? 'b' : 'a');
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
    const char* label = slot ? "weights_b" : "weights_a";
    const esp_partition_t* part = partition(new_slot);
    if (!part) {
        ESP_LOGW(kTag, "no %s partition: compiled-in weights", label);
        return false;
    }
    const void* map = nullptr;
    esp_partition_mmap_handle_t handle;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &map, &handle) != ESP_OK) {
        ESP_LOGW(kTag, "%s: mmap failed: compiled-in weights", label);
        return false;
    }
    const auto* hd = static_cast<const Header*>(map);
    if (std::memcmp(hd->magic, "SCW1", 4) != 0) {
        ESP_LOGI(kTag, "%s is empty: compiled-in weights", label);
        esp_partition_munmap(handle);
        return false;
    }
    if (hd->total < sizeof(Header) || hd->total > part->size ||
        esp_rom_crc32_le(0, static_cast<const uint8_t*>(map) + sizeof(Header), hd->total - sizeof(Header)) != hd->crc) {
        ESP_LOGW(kTag, "%s: blob damaged (size or CRC): compiled-in weights", label);
        esp_partition_munmap(handle);
        return false;
    }
    if (mapped_) {  // a switch: the nets still read the old arrays until releaseOld()
        old_handle_ = handle_;
        old_mapped_ = true;
    }
    handle_ = handle;
    mapped_ = true;
    base_ = static_cast<const uint8_t*>(map);
    count_ = hd->count;
    version_ = hd->version;
    slot_ = new_slot;
    ESP_LOGI(kTag, "%s: blob version %u, %u arrays, %u bytes%s", label, static_cast<unsigned>(version_),
             static_cast<unsigned>(count_), static_cast<unsigned>(hd->total), trial_ ? " (new, on trial)" : "");
    return true;
}

const void* WeightStore::find(const char* name, bool int8, int count) const
{
    if (!base_) return nullptr;
    const auto* hd = reinterpret_cast<const Header*>(base_);
    const auto* e = reinterpret_cast<const Entry*>(base_ + sizeof(Header));
    for (uint32_t i = 0; i < count_; ++i, ++e) {
        if (std::strncmp(e->name, name, sizeof(e->name)) != 0) continue;
        const uint32_t bytes = e->count * (int8 ? 1u : 4u);
        if (e->type != (int8 ? 1u : 2u) || e->count != static_cast<uint32_t>(count) || e->offset % 4 != 0 ||
            e->offset + bytes > hd->total) {
            ESP_LOGW(kTag, "%s: type or size does not match this firmware", name);
            return nullptr;
        }
        return base_ + e->offset;
    }
    return nullptr;
}

void WeightStore::releaseOld()
{
    if (!old_mapped_) return;
    esp_partition_munmap(old_handle_);
    old_mapped_ = false;
}

void WeightStore::confirm()
{
    if (!trial_) return;
    trial_ = false;
    nvs_handle_t h;
    if (nvs_open("weights", NVS_READWRITE, &h) != ESP_OK) return;
    nvs_set_u8(h, "slot", slot_ == 'b' ? 1 : 0);
    nvs_erase_key(h, "next");
    nvs_erase_key(h, "tries");
    nvs_commit(h);
    nvs_close(h);
    ESP_LOGI(kTag, "weights in slot %c (version %u) confirmed after a healthy minute", slot_,
             static_cast<unsigned>(version_));
}

bool WeightStore::beginWrite(size_t total)
{
    if (trial_) return false;  // the other slot is the fallback of the blob on trial
    const esp_partition_t* part = partition(idleSlot());
    if (!part || total < sizeof(Header) || total > part->size) return false;
    // erase what the blob needs, rounded up to whole 4 KB sectors
    const size_t erase = (total + part->erase_size - 1) / part->erase_size * part->erase_size;
    if (esp_partition_erase_range(part, 0, erase) != ESP_OK) return false;
    wpart_ = part;
    wtotal_ = total;
    wdone_ = 0;
    wcrc_ = 0;
    return true;
}

bool WeightStore::write(const void* data, size_t size)
{
    const auto* part = static_cast<const esp_partition_t*>(wpart_);
    if (!part || wdone_ + size > wtotal_) return false;
    const auto* p = static_cast<const uint8_t*>(data);
    // the header goes in last, so a blob cut short is never taken for a valid one
    size_t skip = 0;
    if (wdone_ < sizeof(Header)) {
        skip = std::min(size, sizeof(Header) - wdone_);
        std::memcpy(whead_ + wdone_, p, skip);
    }
    if (size > skip) {
        wcrc_ = esp_rom_crc32_le(wcrc_, p + skip, size - skip);
        if (esp_partition_write(part, wdone_ + skip, p + skip, size - skip) != ESP_OK) return false;
    }
    wdone_ += size;
    return true;
}

bool WeightStore::finishWrite()
{
    const auto* part = static_cast<const esp_partition_t*>(wpart_);
    wpart_ = nullptr;
    Header hd;
    std::memcpy(&hd, whead_, sizeof(hd));
    if (!part || wdone_ != wtotal_ || std::memcmp(hd.magic, "SCW1", 4) != 0 || hd.total != wtotal_ || hd.crc != wcrc_) {
        ESP_LOGW(kTag, "new blob rejected (size, magic or CRC)");
        return false;
    }
    if (esp_partition_write(part, 0, whead_, sizeof(whead_)) != ESP_OK) return false;
    nvs_handle_t h;
    if (nvs_open("weights", NVS_READWRITE, &h) != ESP_OK) return false;
    nvs_set_u8(h, "next", idleSlot() == 'b' ? 1 : 0);
    nvs_set_u8(h, "tries", 0);
    const bool ok = nvs_commit(h) == ESP_OK;
    nvs_close(h);
    ESP_LOGI(kTag, "new blob version %u in slot %c: on trial", static_cast<unsigned>(hd.version), idleSlot());
    return ok;
}
