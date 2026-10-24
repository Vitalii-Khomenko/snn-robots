#include "wifi_engine.h"

#include <cstdio>
#include <ctime>
#include <new>

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
        ESP_LOGW(kTag, "disconnected, reconnecting");
        self->has_ip_.store(false);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(kTag, "got IP");
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

bool WifiEngine::begin()
{
    setenv("TZ", kTimezone, 1);
    tzset();

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

        if (sntp_get_sync_status() == SNTP_SYNC_STATUS_COMPLETED) {
            time_synced_.store(true);
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
