#include "sd_diary.h"

#include <bsp/esp-bsp.h>
#include <dirent.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_vfs_fat.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

namespace {
constexpr const char* kTag = "sddiary";
constexpr const char* kDir = BSP_SD_MOUNT_POINT "/LIFE";
constexpr uint32_t kFileMs = 10 * 60 * 1000;  // a new file every 10 minutes (~4 MB)
constexpr uint32_t kFlushMs = 10 * 1000;      // at most 10 s of diary lost when the power goes
constexpr size_t kBufBytes = 32 * 1024;
constexpr uint64_t kKeepFreePercent = 15;

// NNNNNNNN.SCL -> NNNNNNNN, or 0 for anything else
uint32_t fileNumber(const char* name)
{
    if (std::strlen(name) != 12 || strcasecmp(name + 8, ".SCL") != 0) return 0;
    char* end = nullptr;
    const unsigned long n = std::strtoul(name, &end, 10);
    return end == name + 8 ? static_cast<uint32_t>(n) : 0;
}
}  // namespace

bool SdDiary::mount(uint32_t nowMs)
{
    if (static_cast<int32_t>(nowMs - next_mount_ms_) < 0) return false;
    next_mount_ms_ = nowMs + 30000;
    esp_vfs_fat_mount_config_t mount = {};
    mount.format_if_mount_failed = true;  // the owner allowed it: a card that is not FAT becomes one
    mount.max_files = 3;
    mount.allocation_unit_size = 32 * 1024;
    bsp_sdcard_cfg_t cfg = {};
    cfg.mount = &mount;
    const esp_err_t err = bsp_sdcard_sdspi_mount(&cfg);
    if (err != ESP_OK) {
        ESP_LOGW(kTag, "no card (%s), next try in 30 s", esp_err_to_name(err));
        return false;
    }
    mkdir(kDir, 0775);
    // continue the numbering after the newest file already on the card
    DIR* d = opendir(kDir);
    if (d) {
        while (dirent* e = readdir(d)) {
            const uint32_t n = fileNumber(e->d_name);
            if (n > file_no_) file_no_ = n;
        }
        closedir(d);
    }
    uint64_t total = 0, free_bytes = 0;
    esp_vfs_fat_info(BSP_SD_MOUNT_POINT, &total, &free_bytes);
    ESP_LOGI(kTag, "card mounted: %llu MB, %llu MB free, last file %08u", total >> 20, free_bytes >> 20,
             static_cast<unsigned>(file_no_));
    if (!buf_) buf_ = static_cast<char*>(heap_caps_malloc(kBufBytes, MALLOC_CAP_SPIRAM));
    mounted_ = true;
    return true;
}

void SdDiary::prune()
{
    uint64_t total = 0, free_bytes = 0;
    if (esp_vfs_fat_info(BSP_SD_MOUNT_POINT, &total, &free_bytes) != ESP_OK || total == 0) return;
    while (free_bytes * 100 < total * kKeepFreePercent) {
        uint32_t oldest = 0;
        DIR* d = opendir(kDir);
        if (!d) return;
        while (dirent* e = readdir(d)) {
            const uint32_t n = fileNumber(e->d_name);
            if (n && n != file_no_ && (!oldest || n < oldest)) oldest = n;
        }
        closedir(d);
        if (!oldest) return;
        char path[48];
        std::snprintf(path, sizeof(path), "%s/%08u.SCL", kDir, static_cast<unsigned>(oldest));
        unlink(path);
        ESP_LOGI(kTag, "card full: deleted %08u.SCL", static_cast<unsigned>(oldest));
        if (esp_vfs_fat_info(BSP_SD_MOUNT_POINT, &total, &free_bytes) != ESP_OK) return;
    }
}

bool SdDiary::openNext(uint32_t nowMs)
{
    if (f_) fclose(f_);
    f_ = nullptr;
    ++file_no_;
    prune();
    char path[48];
    std::snprintf(path, sizeof(path), "%s/%08u.SCL", kDir, static_cast<unsigned>(file_no_));
    f_ = fopen(path, "ab");
    if (!f_) return false;
    if (buf_) setvbuf(f_, buf_, _IOFBF, kBufBytes);
    file_opened_ms_ = last_flush_ms_ = nowMs;
    return true;
}

void SdDiary::fail()
{
    ++errors_;
    if (f_) fclose(f_);
    f_ = nullptr;
    // not bsp_sdcard_unmount(): it would free the SPI bus, and the screen lives on it
    esp_vfs_fat_sdcard_unmount(BSP_SD_MOUNT_POINT, bsp_sdcard_get_handle());
    mounted_ = false;
    ESP_LOGW(kTag, "card write failed: unmounted, next try in 30 s");
}

bool SdDiary::write(const void* data, size_t size, double unixTime, uint32_t nowMs)
{
    if (!mounted_ && !mount(nowMs)) return false;
    if (!f_ || nowMs - file_opened_ms_ >= kFileMs) {
        if (!openNext(nowMs)) {
            fail();
            return false;
        }
    }
    const uint16_t len = static_cast<uint16_t>(size);
    const bool ok = fwrite(&len, sizeof(len), 1, f_) == 1 && fwrite(&unixTime, sizeof(unixTime), 1, f_) == 1 &&
                    fwrite(data, 1, size, f_) == size;
    if (!ok) {
        fail();
        return false;
    }
    if (nowMs - last_flush_ms_ >= kFlushMs) {
        last_flush_ms_ = nowMs;
        if (fflush(f_) != 0 || fsync(fileno(f_)) != 0) {
            fail();
            return false;
        }
    }
    ++written_;
    return true;
}
