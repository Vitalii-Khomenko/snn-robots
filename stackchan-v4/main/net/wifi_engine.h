#pragma once
#include <atomic>
#include <cstdint>
#include <ctime>
#include <driver/i2c_master.h>

// Wi-Fi station connection + NTP time sync (the clock of brain v4's night rules and tired channel) + a periodic
// "is a known device on this LAN" presence check (owner=near/away in the stats line).
//
// The clock without a network (2026-11-07, the owner: shown in another office, away from the home Wi-Fi, the robot
// had no time). The CoreS3's BM8563 RTC (I2C 0x51, kept running by the battery while the robot is off) is set from
// every NTP sync and read in begin(), before any network: the clock is known from the first tick wherever the robot
// is switched on. Without a known network the reconnects slow down after kFastRetries to one every kSlowRetryMs
// (until then a failed connect was retried at once, a radio scan after another for as long as it was away).
//
// Runs its own FreeRTOS task for reconnect handling and the periodic
// presence scan: the Wi-Fi/lwIP stack already does its own event-driven
// work off the main loop's thread, and a single ping's round-trip can take
// up to a second - far too slow for the ~20ms main loop budget, same
// reasoning as CameraEngine/MicrophoneEngine's background tasks. begin()
// itself does not block waiting for association or NTP sync; isConnected()/
// isTimeSynced() report readiness as it happens.
//
// Credentials come from secrets/wifi_credentials.h - a real file with this
// robot's actual SSID/password, deliberately kept out of main/ (see
// README.md, Configuration) so it's never mistaken for example/placeholder text or
// swept into a project archive. Known-device identifiers for presence
// detection come from secrets/known_devices.h, which ships with an empty
// IP list by default: the presence check is then a structural no-op
// (knownDeviceNearby() always false) until real reserved/static IPs are
// added there - see that file's own comment for why IPs rather than MACs.
class WifiEngine {
public:
    // Starts the Wi-Fi driver in station mode and the NTP client. Never
    // blocks for association or sync - both happen asynchronously and are
    // reflected in isConnected()/isTimeSynced() once they occur. Returns
    // false only if the underlying esp_wifi/esp_netif init calls fail
    // (bad heap state, not "no AP in range" - that just keeps retrying).
    bool begin();

    bool isConnected() const { return connected_.load(); }
    bool hasIp() const { return has_ip_.load(); }
    // The clock is known: from NTP, or from the RTC until the first NTP sync.
    bool isTimeSynced() const { return time_synced_.load(); }
    // "synced" (NTP), "rtc" (the board's clock, no NTP yet) or "unsynced"
    const char* timeSource() const { return !time_synced_.load() ? "unsynced" : ntp_synced_.load() ? "synced" : "rtc"; }
    // Local hour of day as 0.0..24.0 (e.g. 13.5 = 13:30) - only meaningful
    // once isTimeSynced() is true; returns 12.0 (neutral midday) before
    // then so a caller that forgets to check isTimeSynced() degrades to
    // "always daytime" rather than misreading an unset clock as midnight.
    float hourOfDay() const;
    // True if the last periodic scan (every kPresenceCheckIntervalMs) found
    // any of secrets/known_devices.h's IPs responding on the LAN. Always
    // false while disconnected, or if that list is empty.
    bool knownDeviceNearby() const { return known_device_nearby_.load(); }

    // True once IP_EVENT_STA_GOT_IP has actually fired - distinct from
    // isConnected() (radio association only). Found the hard way: gating
    // the presence-check ping on association alone let it fire in the
    // window after associating but before DHCP handed out an IP, where it
    // has no usable network stack to send from at all ("ping_sock: send
    // error=0" on real hardware). Public (unlike the private atomics below
    // with their own getters) because wifiEventHandler() - a plain
    // function bridging ESP-IDF's C event-callback API, not a member - sets
    // it directly from the IP_EVENT/WIFI_EVENT_STA_DISCONNECTED handlers in
    // wifi_engine.cpp; nothing outside that file has a reason to touch it.
    std::atomic<bool> has_ip_{false};
    // Failed connects in a row (reset on an IP): the handler retries at once only for the first kFastRetries, then
    // taskLoop() every kSlowRetryMs. Public for the same reason as has_ip_.
    std::atomic<int> connect_failures_{0};
    static constexpr int kFastRetries = 10;

private:
    static constexpr uint32_t kSlowRetryMs = 30000;
    static constexpr uint8_t kRtcAddr = 0x51;
    bool readRtc(time_t& utc);
    void writeRtc(time_t utc);
    static void taskEntry(void* arg);
    void taskLoop();
    void checkKnownDevices();

    std::atomic<bool> connected_{false};
    std::atomic<bool> time_synced_{false};
    std::atomic<bool> ntp_synced_{false};
    i2c_master_dev_handle_t rtc_ = nullptr;
    uint32_t ms_since_retry_ = 0;
    std::atomic<bool> known_device_nearby_{false};
    // Consecutive checks in which no known device answered. A phone in
    // power save often misses a ping: "away" only after kAwayAfterMisses
    // checks in a row, so one missed ping does not send the owner away and
    // bring them "home" again 20 minutes later (a false OwnerArrived).
    int missed_checks_ = 0;
    static constexpr int kAwayAfterMisses = 3;
    static constexpr int kPingAttempts = 3;

    // 20 minutes, per explicit request - frequent enough to notice "the
    // owner came home" within a reasonable window, infrequent enough that
    // pinging a phone that's mostly asleep on the radio doesn't matter for
    // battery/airtime.
    static constexpr uint32_t kPresenceCheckIntervalMs = 20 * 60 * 1000;
};
