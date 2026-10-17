#pragma once
#include <atomic>
#include <cstdint>

// Wi-Fi station connection + NTP time sync + periodic "is a known device on
// this LAN" presence check, feeding SnnCore's NIGHT_MODE/DAY_MODE/
// OWNER_NEARBY populations (see neuromorphic/snn_core.h's Population enum
// and RobotRuntime::updateWifiEnvironment()).
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
    bool isTimeSynced() const { return time_synced_.load(); }
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

private:
    static void taskEntry(void* arg);
    void taskLoop();
    void checkKnownDevices();

    std::atomic<bool> connected_{false};
    std::atomic<bool> time_synced_{false};
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
