# ftservo/py32_io_expander/si12t/bmi270 are the vendor driver components
# in components/ - motion/,
# sensors/ and led/ call into them directly, same as
# main/CMakeLists.txt. esp32-camera is a transitive dependency of the
# m5stack_core_s3 BSP already (main/idf_component.yml) - listed explicitly
# here too since camera_engine.cpp includes esp_camera.h directly.
#
# net/ (Wi-Fi STA + NTP + LAN device-presence ping) brings radio behavior
# back after the note below was true: esp_wifi/esp_netif/esp_event for the
# connection itself, lwip for the SNTP client and ICMP ping session used by
# WifiEngine::checkKnownDevices(). Bluetooth stays off (CONFIG_BT_ENABLED=n
# in sdkconfig.defaults) - nothing here needs it.
#
# ../secrets holds real, non-example credentials (SSID/password, known-
# device IPs) - kept as a sibling of main/ rather than inside it so it reads
# as obviously separate from ordinary source, see README.md (Configuration).
idf_component_register(
    SRC_DIRS "." "core" "hybrid" "voice" "behavior" "motion" "audio" "sensors" "ui" "led" "power" "net"
    INCLUDE_DIRS "." "core" "hybrid" "voice" "behavior" "motion" "audio" "sensors" "ui" "led" "power" "net" "../secrets"
    PRIV_REQUIRES ftservo py32_io_expander si12t bmi270 esp32-camera nvs_flash esp_partition esp_wifi esp_netif esp_event lwip)

# Number crunching: the networks' kernels and the vocal chain are built for
# speed (-O3, unrolled loops). Plain floating point, no -ffast-math, so the
# results stay bit-identical to the simulator's.
set_source_files_properties(
    hybrid/vision_net.cpp hybrid/vision_frontend.cpp hybrid/auditory_net.cpp hybrid/audio_frontend.cpp
    hybrid/spiking_core.cpp hybrid/policy_gru.cpp voice/droid_synth.cpp voice/babble_net.cpp
    PROPERTIES COMPILE_OPTIONS "-O3;-funroll-loops")
