#include "power_rails.h"

#include <bsp/esp-bsp.h>
#include <esp_log.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

namespace power {
namespace {

constexpr uint16_t kAxp2101Addr = 0x34;
constexpr const char* kTag      = "power";

esp_err_t writeReg(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t value)
{
    uint8_t buf[2] = {reg, value};
    return i2c_master_transmit(dev, buf, sizeof(buf), 1000);
}

esp_err_t readReg(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t* value)
{
    return i2c_master_transmit_receive(dev, &reg, 1, value, 1, 1000);
}

}  // namespace

i2c_master_dev_handle_t addI2cDevice(uint16_t addr)
{
    i2c_master_bus_handle_t bus = nullptr;
    esp_err_t err                = i2c_master_get_bus_handle(BSP_I2C_NUM, &bus);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_get_bus_handle failed: %s", esp_err_to_name(err));
        return nullptr;
    }

    i2c_device_config_t dev_cfg = {};
    dev_cfg.dev_addr_length     = I2C_ADDR_BIT_LEN_7;
    dev_cfg.device_address      = addr;
    dev_cfg.scl_speed_hz        = 400000;

    i2c_master_dev_handle_t dev = nullptr;
    err                         = i2c_master_bus_add_device(bus, &dev_cfg, &dev);
    if (err != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_bus_add_device(0x%02x) failed: %s", addr, esp_err_to_name(err));
        return nullptr;
    }
    return dev;
}

void enableAxp2101PowerRails()
{
    i2c_master_dev_handle_t axp = addI2cDevice(kAxp2101Addr);
    if (!axp) {
        return;
    }

    uint8_t reg90 = 0;
    readReg(axp, 0x90, &reg90);
    writeReg(axp, 0x90, reg90 | 0b10110100);
    writeReg(axp, 0x97, 0b11110 - 2);
    writeReg(axp, 0x69, 0b00110101);
    writeReg(axp, 0x30, 0b111111);
    writeReg(axp, 0x90, 0xBF);
    writeReg(axp, 0x94, 33 - 5);
    writeReg(axp, 0x95, 33 - 5);
    writeReg(axp, 0x27, 0x00);

    i2c_master_bus_rm_device(axp);
}

void boostSpeakerPaVoltage()
{
    i2c_master_dev_handle_t axp = addI2cDevice(kAxp2101Addr);
    if (!axp) {
        return;
    }
    // ALDO1 voltage select, reg 0x92: value = (V - 0.5) / 0.1
    writeReg(axp, 0x92, 28 /* 3.3V */);
    i2c_master_bus_rm_device(axp);
}

m5::PY32IOExpander_Class* getBodyIoExpander()
{
    static m5::PY32IOExpander_Class* instance = nullptr;
    if (instance) {
        return instance;
    }

    i2c_master_bus_handle_t bus = nullptr;
    if (i2c_master_get_bus_handle(BSP_I2C_NUM, &bus) != ESP_OK) {
        ESP_LOGE(kTag, "i2c_master_get_bus_handle failed for PY32 expander");
        return nullptr;
    }

    static m5::PY32IOExpander_Class io_expander(bus);
    uint32_t start = esp_timer_get_time() / 1000;
    while (!io_expander.begin()) {
        if (esp_timer_get_time() / 1000 - start > 1200) {
            ESP_LOGE(kTag, "PY32IOExpander.begin() timed out");
            return nullptr;
        }
        vTaskDelay(pdMS_TO_TICKS(200));
    }

    instance = &io_expander;
    return instance;
}

bool enableServoPower()
{
    m5::PY32IOExpander_Class* io_expander = getBodyIoExpander();
    if (!io_expander) {
        return false;
    }

    io_expander->setDirection(0, true);  // output
    io_expander->setPullMode(0, true);   // pull-up
    // A firmware flash resets the ESP32 but normally leaves the body and its
    // servos powered. Force the same short VM reset that a real power cycle
    // provides, otherwise an SCS servo can retain a half-finished UART/
    // torque state and ignore the first commands after flashing.
    io_expander->digitalWrite(0, false);  // VM EN = servo motor power off
    vTaskDelay(pdMS_TO_TICKS(500));
    io_expander->digitalWrite(0, true);   // VM EN = servo motor power on
    vTaskDelay(pdMS_TO_TICKS(300));
    return true;
}

void readServoPowerState(int& version, int& vmOut, int& vmPin)
{
    version = vmOut = vmPin = -1;
    m5::PY32IOExpander_Class* io = getBodyIoExpander();
    if (!io) return;
    version = io->readVersion();
    vmOut = io->getWriteValue(0) ? 1 : 0;
    vmPin = io->digitalRead(0) ? 1 : 0;
}

}  // namespace power
