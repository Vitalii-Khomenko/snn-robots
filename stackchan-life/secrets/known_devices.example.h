#pragma once
#include <array>

// Template. Copy to known_devices.h (same folder, git-ignored).
//
// IP addresses of devices whose presence means "the owner is home"; they are
// pinged every 20 minutes. Use IPs, not MACs (phones randomize their MAC) and
// give each device a DHCP reservation on your router.
// The array size in the type must match the number of entries.
//
// No devices at all: use std::array<const char*, 0> kKnownDeviceIps{};
inline constexpr std::array<const char*, 1> kKnownDeviceIps{"192.168.1.50"};
