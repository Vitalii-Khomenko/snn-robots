idf_component_register(
    SRCS "src/SCS.cpp" "src/SCSerial.cpp" "src/SCSCL.cpp"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_timer
)
