#include "wifi_engine.h"

#include <cstdio>
#include <ctime>
#include <new>

#include <bsp/esp-bsp.h>
#include <esp_event.h>
#include <esp_log.h>
#include <esp_netif.h>
#include <esp_netif_sntp.h>
#include <esp_sntp.h>
#include <esp_wifi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#include <lwip/ip_addr.h>
#include <sys/time.h>
#include <ping/ping_sock.h>

// ../secrets is on the include path (see main/CMakeLists.txt) specifically
// so these can be included by plain name, same as any other project header.
#include "known_devices.h"
#include "wifi_credentials.h"

namespace {
constexpr const char* kTag = "wifi";

// This project's own build.sh/flash.sh already run the toolchain container
// with TZ=Europe/Berlin - the most concrete evidence available of where
// this specific robot lives, short of asking. POSIX TZ rule for Central
// European (Summer) Time, with the EU's standard last-Sunday-of-March/
// October DST switchover. If this robot is ever used somewhere else,
// change this one string.
constexpr const char* kTimezone = "CET-1CEST,M3.5.0,M10.5.0/3";

void wifiEventHandler(void* arg, esp_event_base_t base, int32_t id, void* data)
{
    auto* self = static_cast<WifiEngine*>(arg);
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        // Unconditional reconnect - there is no exponential backoff here on
        // purpose: this robot has no UI to show a persistent connection
        // error, and a plain immediate retry is what every consumer IoT
        // device does for "router rebooted"/"briefly out of range". esp_wifi
        // already rate-limits the underlying radio retries internally.
        // ... but away from the known network (another office, 2026-11-07) a failed connect came back at once, for as
        // long as the robot was there: after kFastRetries the retries go to taskLoop(), one every kSlowRetryMs.
        self->has_ip_.store(false);
        const int failures = self->connect_failures_.fetch_add(1) + 1;
        const auto* d = static_cast<const wifi_event_sta_disconnected_t*>(data);
        if (failures <= WifiEngine::kFastRetries) {
            ESP_LOGW(kTag, "disconnected (reason %d), reconnecting", d ? d->reason : 0);
            esp_wifi_connect();
        } else if (failures == WifiEngine::kFastRetries + 1) {
            ESP_LOGW(kTag, "no network after %d tries (reason %d): trying every 30 s, the robot lives on by itself",
                     WifiEngine::kFastRetries, d ? d->reason : 0);
        }
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(kTag, "got IP");
        self->connect_failures_.store(0);
        // Found on first real hardware test: esp_wifi_sta_get_ap_info()
        // (taskLoop()'s own connected_ check) succeeds as soon as the radio
        // associates, which is BEFORE DHCP has actually handed out an IP -
        // a presence-check ping fired right at that gap failed outright
        // ("ping_sock: send error=0"), since there was no usable network
        // stack yet to send it from. This flag tracks the real thing the
        // ping needs.
        self->has_ip_.store(true);
    }
}

struct PingContext {
    SemaphoreHandle_t ended = nullptr;
    volatile bool success   = false;
};

void onPingSuccess(esp_ping_handle_t /*hdl*/, void* args)
{
    static_cast<PingContext*>(args)->success = true;
}

void onPingTimeout(esp_ping_handle_t /*hdl*/, void* /*args*/)
{
    // success stays false; completion is signalled from onPingEnd().
}

void onPingEnd(esp_ping_handle_t /*hdl*/, void* args)
{
    // esp_ping always calls this last, after success/timeout, from its own
    // thread. It is the only safe point to release the waiter: the
    // success/timeout callbacks may run long after esp_ping_stop(), which
    // (like esp_ping_delete_session()) only sets a flag and never waits.
    // Waiting on those instead - with the context on pingOnce()'s stack -
    // let a late timeout callback give a freed semaphore: Guru Meditation
    // LoadProhibited in onPingTimeout seen on hardware ~9 s after boot.
    xSemaphoreGive(static_cast<PingContext*>(args)->ended);
}

int bcd(uint8_t v) { return (v >> 4) * 10 + (v & 0x0F); }
uint8_t toBcd(int v) { return static_cast<uint8_t>((v / 10) << 4 | (v % 10)); }

// days since 1970-01-01 of a civil date (H. Hinnant's days_from_civil): newlib has no timegm
int64_t daysFromCivil(int y, int m, int d)
{
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const int64_t yoe = y - era * 400;
    const int64_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const int64_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}

// One-shot ICMP echo, true if a reply came back within timeout_ms. Runs
// entirely on the caller's own task/stack - safe to call from WifiEngine's
// background task, not safe to call from the ~20ms main loop (a real
// network round-trip, worst case the full timeout).
bool pingOnce(const char* ip_str, uint32_t timeout_ms)
{
    ip_addr_t target{};
    if (!ipaddr_aton(ip_str, &target)) {
        ESP_LOGW(kTag, "known_devices.h: '%s' is not a valid IP, skipping", ip_str);
        return false;
    }

    // Heap, not stack: if the ping thread never reports back, the context
    // is leaked (a few bytes) rather than freed under a live callback.
    auto* ctx = new (std::nothrow) PingContext();
    if (ctx == nullptr) {
        return false;
    }
    ctx->ended = xSemaphoreCreateBinary();
    if (ctx->ended == nullptr) {
        delete ctx;
        return false;
    }

    esp_ping_config_t config = ESP_PING_DEFAULT_CONFIG();
    config.target_addr         = target;
    config.count                 = 1;
    config.timeout_ms          = timeout_ms;

    esp_ping_callbacks_t callbacks{};
    callbacks.on_ping_success = onPingSuccess;
    callbacks.on_ping_timeout = onPingTimeout;
    callbacks.on_ping_end       = onPingEnd;
    callbacks.cb_args             = ctx;

    esp_ping_handle_t ping = nullptr;
    if (esp_ping_new_session(&config, &callbacks, &ping) != ESP_OK) {
        vSemaphoreDelete(ctx->ended);
        delete ctx;
        return false;
    }
    esp_ping_start(ping);
    // count=1, so the thread always reaches onPingEnd() after one
    // send/receive; the generous bound only guards a starved thread.
    const bool ended = xSemaphoreTake(ctx->ended, pdMS_TO_TICKS(timeout_ms + 10000)) == pdTRUE;
    esp_ping_stop(ping);
    esp_ping_delete_session(ping);
    if (!ended) {
        ESP_LOGW(kTag, "ping thread did not finish; leaking its context");
        return false;
    }
    const bool success = ctx->success;
    vSemaphoreDelete(ctx->ended);
    delete ctx;
    return success;
}

}  // namespace

// The BM8563 keeps UTC: registers 0x02..0x08 are seconds (bit 7: the clock lost its power, the time is not to be
// trusted), minutes, hours, day, weekday, month (bit 7: century), year, in BCD.
bool WifiEngine::readRtc(time_t& utc)
{
    uint8_t reg = 0x02, r[7] = {};
    if (!rtc_ || i2c_master_transmit_receive(rtc_, &reg, 1, r, sizeof(r), 50) != ESP_OK) return false;
    const int sec = bcd(r[0] & 0x7F), min = bcd(r[1] & 0x7F), hour = bcd(r[2] & 0x3F), day = bcd(r[3] & 0x3F);
    const int month = bcd(r[5] & 0x1F), year = 2000 + bcd(r[6]);
    if ((r[0] & 0x80) || year < 2026 || month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || min > 59 ||
        sec > 59)
        return false;
    utc = static_cast<time_t>(daysFromCivil(year, month, day) * 86400 + hour * 3600 + min * 60 + sec);
    return true;
}

void WifiEngine::writeRtc(time_t utc)
{
    if (!rtc_) return;
    struct tm t{};
    gmtime_r(&utc, &t);
    const uint8_t start[2] = {0x00, 0x00};  // control 1: the clock runs (STOP cleared)
    const uint8_t w[8] = {0x02, toBcd(t.tm_sec), toBcd(t.tm_min), toBcd(t.tm_hour), toBcd(t.tm_mday),
                          static_cast<uint8_t>(t.tm_wday), toBcd(t.tm_mon + 1), toBcd(t.tm_year % 100)};
    if (i2c_master_transmit(rtc_, start, sizeof(start), 50) != ESP_OK || i2c_master_transmit(rtc_, w, sizeof(w), 50) != ESP_OK)
        ESP_LOGW(kTag, "clock: could not set the RTC");
}

bool WifiEngine::begin()
{
    setenv("TZ", kTimezone, 1);
    tzset();

    // the clock before any network: the board's RTC (the I2C bus is up: power_rails / power_status came first)
    i2c_master_bus_handle_t bus = nullptr;
    if (i2c_master_get_bus_handle(BSP_I2C_NUM, &bus) == ESP_OK) {
        i2c_device_config_t cfg = {};
        cfg.dev_addr_length = I2C_ADDR_BIT_LEN_7;
        cfg.device_address = kRtcAddr;
        cfg.scl_speed_hz = 400000;
        if (i2c_master_bus_add_device(bus, &cfg, &rtc_) != ESP_OK) rtc_ = nullptr;
    }
    time_t utc = 0;
    if (readRtc(utc)) {
        const timeval tv{utc, 0};
        settimeofday(&tv, nullptr);
        time_synced_.store(true);
        struct tm local{};
        localtime_r(&utc, &local);
        ESP_LOGI(kTag, "clock: from the RTC, %04d-%02d-%02d %02d:%02d:%02d (before any network)", local.tm_year + 1900,
                 local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec);
    } else {
        ESP_LOGW(kTag, "clock: the RTC %s - no clock until NTP", rtc_ ? "is not set or lost its power" : "does not answer");
    }

    esp_err_t err = esp_netif_init();
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "esp_netif_init failed: %d", err);
        return false;
    }
    // RobotRuntime::begin() already calls esp_event_loop_create_default()
    // - no, it does not (only NVS is shared infrastructure there); Wi-Fi is
    // the first and only user of the default event loop in this firmware,
    // so this owns creating it. ESP_ERR_INVALID_STATE just means it already
    // exists (harmless, not a real failure) - a future subsystem is free to
    // also depend on the default loop existing.
    err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(kTag, "esp_event_loop_create_default failed: %d", err);
        return false;
    }
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    err                             = esp_wifi_init(&init_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "esp_wifi_init failed: %d", err);
        return false;
    }

    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifiEventHandler, this, nullptr);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifiEventHandler, this, nullptr);

    wifi_config_t wifi_config{};
    std::snprintf(reinterpret_cast<char*>(wifi_config.sta.ssid), sizeof(wifi_config.sta.ssid), "%s", kWifiSsid);
    std::snprintf(reinterpret_cast<char*>(wifi_config.sta.password), sizeof(wifi_config.sta.password), "%s",
                  kWifiPassword);
    // WPA2-minimum, not WPA3-only or open - matches an ordinary home router
    // rather than assuming a specific security mode.
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    err = esp_wifi_start();
    // No modem power save. Added while hunting the 2026-09-27 head freeze;
    // the real cause turned out to be the servos' PWM mode (see
    // MotionEngine::writeServo), but the robot runs on USB power and a
    // steady radio keeps current spikes off the servo rail, so it stays.
    esp_wifi_set_ps(WIFI_PS_NONE);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "esp_wifi_start failed: %d", err);
        return false;
    }

    // Started here rather than waited-on: esp_netif_sntp syncs
    // asynchronously in the background once the network is up, regardless
    // of whether that happens before or after this call.
    esp_sntp_config_t sntp_config = ESP_NETIF_SNTP_DEFAULT_CONFIG("pool.ntp.org");
    esp_netif_sntp_init(&sntp_config);

    if (xTaskCreate(&WifiEngine::taskEntry, "wifi_engine", 4096, this, 3, nullptr) != pdPASS) {
        ESP_LOGE(kTag, "background task creation failed");
        return false;
    }
    return true;
}

void WifiEngine::taskEntry(void* arg)
{
    static_cast<WifiEngine*>(arg)->taskLoop();
}

void WifiEngine::taskLoop()
{
    // Connection/sync status is worth reflecting promptly (RobotRuntime
    // logs it in its periodic stats line), but the LAN presence scan itself
    // stays on its own much coarser kPresenceCheckIntervalMs clock (see
    // that constant's comment) - hence two different periods sharing one
    // task instead of one loop paced by the slower interval.
    constexpr uint32_t kStatusPollMs = 5000;
    uint32_t ms_since_presence_check = kPresenceCheckIntervalMs;  // check once on first pass
    while (true) {
        // esp_wifi_sta_get_ap_info() only succeeds while actually
        // associated - simpler and more precise than tracking a flag from
        // the event handler's WIFI_EVENT/IP_EVENT callbacks, which would
        // need its own synchronization to hand off to this task safely.
        wifi_ap_record_t ap_info{};
        connected_.store(esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK);

        // COMPLETED once per sync (the status goes back to RESET when read): every hour while online
        if (sntp_get_sync_status() == SNTP_SYNC_STATUS_COMPLETED) {
            const time_t now = time(nullptr);
            time_t rtc = 0;
            const bool had = readRtc(rtc);
            if (!ntp_synced_.load())
                ESP_LOGI(kTag, "clock: NTP synced; the RTC %s%lld s", had ? "was off by " : "was not set, ",
                         had ? static_cast<long long>(rtc - now) : 0LL);
            writeRtc(now);
            ntp_synced_.store(true);
            time_synced_.store(true);
        }

        if (connect_failures_.load() > kFastRetries && !connected_.load()) {
            ms_since_retry_ += kStatusPollMs;
            if (ms_since_retry_ >= kSlowRetryMs) {
                ms_since_retry_ = 0;
                esp_wifi_connect();
            }
        }

        if (ms_since_presence_check >= kPresenceCheckIntervalMs) {
            // has_ip_, not connected_ - see that member's own comment on why
            // radio association alone isn't enough for a ping to work.
            if (has_ip_.load()) {
                checkKnownDevices();
                ms_since_presence_check = 0;
            }
            // Without an IP the timer stays at-or-past the threshold, so the
            // check retries every kStatusPollMs until Wi-Fi is back (resetting
            // it once meant the first real check came 20 minutes late). The
            // owner state is left as it was: a Wi-Fi drop says nothing about
            // whether they are home.
        }

        vTaskDelay(pdMS_TO_TICKS(kStatusPollMs));
        ms_since_presence_check += kStatusPollMs;
    }
}

void WifiEngine::checkKnownDevices()
{
    if (kKnownDeviceIps.empty()) {
        known_device_nearby_.store(false);
        return;
    }
    for (int attempt = 0; attempt < kPingAttempts; ++attempt) {
        for (const char* ip : kKnownDeviceIps) {
            if (pingOnce(ip, 1000)) {
                missed_checks_ = 0;
                known_device_nearby_.store(true);
                return;
            }
        }
    }
    if (++missed_checks_ >= kAwayAfterMisses) known_device_nearby_.store(false);
}

float WifiEngine::hourOfDay() const
{
    if (!time_synced_.load()) {
        return 12.0f;
    }
    time_t now = time(nullptr);
    struct tm local_tm{};
    localtime_r(&now, &local_tm);
    return static_cast<float>(local_tm.tm_hour) + static_cast<float>(local_tm.tm_min) / 60.0f;
}
