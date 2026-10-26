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
    SRC_DIRS "." "core" "perception" "voice" "motion" "audio" "sensors" "ui" "led" "power" "net" "v4"
    INCLUDE_DIRS "." "core" "perception" "voice" "motion" "audio" "sensors" "ui" "led" "power" "net" "v4" "../secrets"
    PRIV_REQUIRES ftservo py32_io_expander si12t bmi270 esp32-camera nvs_flash esp_partition esp_wifi esp_netif esp_event lwip esp_ringbuf app_update esp_http_server espcoredump espressif__esp_lcd_touch)

# Number crunching: the networks' kernels and the vocal chain are built for
# speed (-O3, unrolled loops). Plain floating point, no -ffast-math, so the
# results stay bit-identical to the simulator's.
set_source_files_properties(
    perception/vision_frontend.cpp perception/auditory_net.cpp perception/audio_frontend.cpp voice/droid_synth.cpp
    voice/babble_net.cpp v4/snn_v4.cpp v4/nn_v4.cpp v4/muse_v4.cpp
    PROPERTIES COMPILE_OPTIONS "-O3;-funroll-loops")
