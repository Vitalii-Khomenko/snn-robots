#pragma once
// Host shim: I2C transactions are routed to register models of the robot's
// chips (LTR-553 presence/ALS, AXP2101 PMIC) in simulator/hal.
#include <cstddef>
#include <cstdint>
#include "../esp_err.h"
typedef struct sim_i2c_bus* i2c_master_bus_handle_t;
typedef struct sim_i2c_dev* i2c_master_dev_handle_t;
typedef enum { I2C_ADDR_BIT_LEN_7 = 0, I2C_ADDR_BIT_LEN_10 = 1 } i2c_addr_bit_len_t;
typedef struct {
    i2c_addr_bit_len_t dev_addr_length;
    uint16_t device_address;
    uint32_t scl_speed_hz;
    uint32_t scl_wait_us;
    struct { uint32_t disable_ack_check : 1; } flags;
} i2c_device_config_t;
esp_err_t i2c_master_get_bus_handle(int port, i2c_master_bus_handle_t* out);
esp_err_t i2c_master_bus_add_device(i2c_master_bus_handle_t bus, const i2c_device_config_t* cfg,
                                    i2c_master_dev_handle_t* out);
esp_err_t i2c_master_transmit(i2c_master_dev_handle_t dev, const uint8_t* data, size_t len, int timeout_ms);
esp_err_t i2c_master_transmit_receive(i2c_master_dev_handle_t dev, const uint8_t* tx, size_t tx_len,
                                      uint8_t* rx, size_t rx_len, int timeout_ms);
