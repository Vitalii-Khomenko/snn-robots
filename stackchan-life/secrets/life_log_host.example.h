#pragma once
// Where the robot sends its diary (main/net/life_log.h): the PC that runs
// tools/life/lifelogd.py. Give that PC a fixed address in the router (DHCP
// reservation), or the diary goes nowhere after it gets a new one.
// Copy to life_log_host.h and edit. Without life_log_host.h the diary is off.
#define LIFE_LOG_HOST "192.168.1.10"
#define LIFE_LOG_PORT 47000
// Over-the-air updates (main/net/ota_server.h, tools/life/ota.py) need this token; make your own random one:
//   python3 -c "import secrets; print(secrets.token_hex(16))"
#define LIFE_OTA_TOKEN "change-me"
