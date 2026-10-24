#include "life_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <esp_attr.h>
#include <esp_heap_caps.h>
#include <esp_log.h>
#include <esp_random.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/ringbuf.h>
#include <freertos/task.h>
#include <lwip/sockets.h>
#include <sys/time.h>

#include "wifi_engine.h"

#include <freertos/semphr.h>
#if __has_include("life_log_host.h")
#include "life_log_host.h"  // secrets/: the PC that collects the diary
#endif

namespace {
constexpr const char* kTag = "lifelog";
constexpr size_t kRingBytes = 96 * 1024;  // ~15 s of diary while Wi-Fi hiccups
constexpr size_t kMaxLine = 1300;  // the 5 s stats line is ~1 KB; still one unfragmented datagram
LifeLog* s_self = nullptr;
vprintf_like_t s_serial = nullptr;
SemaphoreHandle_t s_line_lock = nullptr;  // guards s_line: no 260 B buffer on small task stacks (sys_evt has 2.3 KB)

struct __attribute__((packed)) LineDatagram {
    char magic[4];
    uint8_t kind, n;
    uint16_t nValues;
    uint32_t bootId, seq;
    uint32_t ms;
    char text[kMaxLine];
};
EXT_RAM_BSS_ATTR LineDatagram s_line;  // 1.3 KB: internal RAM is scarce (Wi-Fi, camera, the OTA server)

// float -> IEEE 754 half (round to nearest; overflow -> infinity, NaN stays NaN). numpy reads it as '<f2'.
uint16_t toHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t e8 = (x >> 23) & 0xffu;
    uint32_t mant = x & 0x7fffffu;
    if (e8 == 0xffu) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    const int32_t e = static_cast<int32_t>(e8) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (e <= 0) {  // subnormal half (or zero)
        if (e < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - e);
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) ++h;
        return static_cast<uint16_t>(sign | h);
    }
    uint32_t h = sign | (static_cast<uint32_t>(e) << 10) | (mant >> 13);
    if (mant & 0x1000u) ++h;  // a carry into the exponent is still the right rounding
    return static_cast<uint16_t>(h);
}
}  // namespace

bool LifeLog::begin(const WifiEngine& wifi)
{
#ifndef LIFE_LOG_HOST
    ESP_LOGW(kTag, "no secrets/life_log_host.h: the diary is off");
    return false;
#else
    wifi_ = &wifi;
    ring_ = xRingbufferCreateWithCaps(kRingBytes, RINGBUF_TYPE_NOSPLIT, MALLOC_CAP_SPIRAM);
    if (!ring_) {
        ESP_LOGE(kTag, "no memory for the ring buffer: the diary is off");
        return false;
    }
    s_line_lock = xSemaphoreCreateMutex();
    if (!s_line_lock) return false;
    boot_id_ = esp_random();
    std::memcpy(pending_.h.magic, "SCL1", 4);
    pending_.h.kind = 3;
    pending_.h.nValues = kValues;
    pending_.h.bootId = boot_id_;
    s_self = this;
    if (xTaskCreatePinnedToCoreWithCaps(&LifeLog::taskEntry, "lifelog", 6144, this, 1, nullptr, 0,
                                        MALLOC_CAP_SPIRAM) != pdPASS) {
        s_self = nullptr;
        ESP_LOGE(kTag, "sender task not started: the diary is off");
        return false;
    }
    s_serial = esp_log_set_vprintf(&LifeLog::teeLog);
    ESP_LOGI(kTag, "diary -> %s:%d (boot %08x)", LIFE_LOG_HOST, LIFE_LOG_PORT, static_cast<unsigned>(boot_id_));
    return true;
#endif
}

void LifeLog::push(const void* data, size_t size)
{
    if (xRingbufferSend(static_cast<RingbufHandle_t>(ring_), data, size, 0) != pdTRUE) dropped_.fetch_add(1);
}

void LifeLog::tick(uint32_t tickSeq, uint32_t nowMs, uint16_t flags, const float* values)
{
    if (!ring_) return;
    if (pending_n_ == 0) {
        timeval tv{};
        gettimeofday(&tv, nullptr);
        // before NTP the clock starts at 1970: 0 tells the PC "not synced"
        pending_.unixMs = tv.tv_sec > 1600000000 ? static_cast<int64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000 : 0;
    }
    TickRec& r = pending_.t[pending_n_];
    r.seq = tickSeq;
    r.ms = nowMs;
    r.flags = flags;
    for (int i = 0; i < kValues; ++i) r.v[i] = toHalf(values[i]);
    if (++pending_n_ < kTicksPerDatagram) return;
    pending_.h.n = static_cast<uint8_t>(pending_n_);
    pending_.h.seq = seq_.fetch_add(1);
    push(&pending_, sizeof(pending_));
    pending_n_ = 0;
}

// Every ESP_LOG line: to the serial console as before, and into the diary. Runs on whatever task logs, so it only
// formats on its own stack and hands the line to the ring buffer (thread-safe, never blocks).
int LifeLog::teeLog(const char* fmt, va_list args)
{
    va_list copy;
    va_copy(copy, args);
    const int ret = s_serial ? s_serial(fmt, args) : vprintf(fmt, args);
    LifeLog* self = s_self;
    // a line waits up to 5 ms for another task's line; longer is not worth it: counted as dropped
    if (self && xSemaphoreTake(s_line_lock, pdMS_TO_TICKS(5)) == pdTRUE) {
        LineDatagram& d = s_line;
        std::memcpy(d.magic, "SCL1", 4);
        d.kind = 2;
        d.n = 1;
        d.nValues = 0;
        d.bootId = self->boot_id_;
        d.ms = static_cast<uint32_t>(esp_timer_get_time() / 1000);
        int len = vsnprintf(d.text, sizeof(d.text), fmt, copy);
        len = std::clamp(len, 0, static_cast<int>(sizeof(d.text)) - 1);
        while (len > 0 && (d.text[len - 1] == '\n' || d.text[len - 1] == '\r')) --len;
        if (len > 0) {
            d.seq = self->seq_.fetch_add(1);
            self->push(&d, offsetof(LineDatagram, text) + static_cast<size_t>(len));
        }
        xSemaphoreGive(s_line_lock);
    } else if (self) {
        self->dropped_.fetch_add(1);
    }
    va_end(copy);
    return ret;
}

void LifeLog::taskEntry(void* arg)
{
    static_cast<LifeLog*>(arg)->taskLoop();
}

void LifeLog::taskLoop()
{
#ifdef LIFE_LOG_HOST
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(LIFE_LOG_PORT);
    inet_pton(AF_INET, LIFE_LOG_HOST, &to.sin_addr);
    auto ring = static_cast<RingbufHandle_t>(ring_);
    while (true) {
        // Per datagram nothing is logged in here: every line would come back through teeLog into this ring.
        // the ring is the only copy: it waits there until Wi-Fi is up (boot lines, short drop-outs)
        if (!wifi_->hasIp()) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }
        size_t size = 0;
        void* item = xRingbufferReceive(ring, &size, pdMS_TO_TICKS(1000));
        if (!item) continue;
        if (wifi_->hasIp() && sock_ < 0) sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        bool ok = false;
        for (int attempt = 0; attempt < 5 && !ok && wifi_->hasIp() && sock_ >= 0; ++attempt) {
            ok = sendto(sock_, item, size, 0, reinterpret_cast<const sockaddr*>(&to), sizeof(to)) == static_cast<int>(size);
            if (!ok) vTaskDelay(pdMS_TO_TICKS(20 << attempt));  // the Wi-Fi tx queue is full for a moment
        }
        // A backlog (the boot lines, after a drop-out) goes out paced, ~100 datagrams/s: sent all at once it overran
        // the Wi-Fi tx buffers (4 datagrams lost at boot, 2026-10-11). The live diary is ~5 datagrams/s + lines.
        UBaseType_t waiting = 0;
        vRingbufferGetInfo(ring, nullptr, nullptr, nullptr, nullptr, &waiting);
        if (waiting > 0) vTaskDelay(pdMS_TO_TICKS(10));
        vRingbufferReturnItem(ring, item);
        if (ok) sent_.fetch_add(1);
        else if (wifi_->hasIp()) dropped_.fetch_add(1);
    }
#endif
}
