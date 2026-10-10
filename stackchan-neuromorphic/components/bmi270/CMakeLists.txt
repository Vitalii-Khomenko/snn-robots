idf_component_register(
    SRCS "src/bmi270.cpp"
         "include/BMI270_SensorAPI/bmi2.c"
         "include/BMI270_SensorAPI/bmi270.c"
         "include/BMI270_SensorAPI/bmi270_context.c"
         "include/BMI270_SensorAPI/bmi270_dsd.c"
         "include/BMI270_SensorAPI/bmi270_legacy.c"
         "include/BMI270_SensorAPI/bmi270_maximum_fifo.c"
         "include/BMI270_SensorAPI/bmi2_ois.c"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_timer
)
