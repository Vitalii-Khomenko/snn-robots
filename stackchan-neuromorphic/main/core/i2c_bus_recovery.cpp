#include "i2c_bus_recovery.h"

#include <bsp/esp-bsp.h>
#include <driver/gpio.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <rom/ets_sys.h>

namespace {
constexpr const char* kTag = "i2c_recovery";

void delayHalfPeriod()
{
    // ~100kHz-ish bit period, well within every slave on this bus's specs -
    // this only ever runs a handful of times at boot, speed doesn't matter.
    ets_delay_us(5);
}
}  // namespace

void recoverStuckI2cBus()
{
    gpio_config_t scl_cfg = {};
    scl_cfg.pin_bit_mask  = 1ULL << BSP_I2C_SCL;
    scl_cfg.mode           = GPIO_MODE_OUTPUT_OD;
    scl_cfg.pull_up_en    = GPIO_PULLUP_ENABLE;
    gpio_config(&scl_cfg);

    gpio_config_t sda_cfg = {};
    sda_cfg.pin_bit_mask  = 1ULL << BSP_I2C_SDA;
    sda_cfg.mode           = GPIO_MODE_INPUT;
    sda_cfg.pull_up_en    = GPIO_PULLUP_ENABLE;
    gpio_config(&sda_cfg);

    gpio_set_level(BSP_I2C_SCL, 1);
    delayHalfPeriod();

    if (gpio_get_level(BSP_I2C_SDA) == 1) {
        // Bus already idle (both lines high) - nothing to recover, and
        // toggling SCL for no reason risks disturbing a slave that's fine.
        return;
    }

    ESP_LOGW(kTag, "SDA held low at boot - clocking stuck slave free");

    // Standard I2C bus-recovery procedure: a slave stuck mid-transaction
    // is always waiting for more SCL clock edges before it'll release
    // SDA. Up to 9 clocks covers the worst case (a full byte + NACK).
    for (int i = 0; i < 9 && gpio_get_level(BSP_I2C_SDA) == 0; ++i) {
        gpio_set_level(BSP_I2C_SCL, 0);
        delayHalfPeriod();
        gpio_set_level(BSP_I2C_SCL, 1);
        delayHalfPeriod();
    }

    // Manually drive a STOP condition (SDA rising while SCL is high) so
    // the bus is left in a clean idle state for the real I2C driver,
    // rather than just abandoning mid-sequence.
    sda_cfg.mode = GPIO_MODE_OUTPUT_OD;
    gpio_config(&sda_cfg);
    gpio_set_level(BSP_I2C_SDA, 0);
    delayHalfPeriod();
    gpio_set_level(BSP_I2C_SCL, 1);
    delayHalfPeriod();
    gpio_set_level(BSP_I2C_SDA, 1);
    delayHalfPeriod();

    if (gpio_get_level(BSP_I2C_SDA) == 0) {
        ESP_LOGE(kTag, "SDA still low after recovery attempt - bus may still be wedged");
    } else {
        ESP_LOGI(kTag, "I2C bus recovered");
    }

    // Hand both pins back to plain floating inputs - bsp_i2c_init()
    // (the real I2C driver init, called right after this from
    // RobotRuntime::begin()) reconfigures them itself.
    gpio_reset_pin(BSP_I2C_SCL);
    gpio_reset_pin(BSP_I2C_SDA);
}
