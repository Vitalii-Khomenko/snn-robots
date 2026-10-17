#pragma once

// Template. Copy to wifi_credentials.h (same folder, git-ignored) and fill in
// your own network. Included only by main/net/wifi_engine.cpp.
// Leave the strings empty to build without Wi-Fi (also set kWifiEnabled = false
// in main/robot_runtime.cpp).

inline constexpr const char* kWifiSsid     = "your-network-name";
inline constexpr const char* kWifiPassword = "your-password";
