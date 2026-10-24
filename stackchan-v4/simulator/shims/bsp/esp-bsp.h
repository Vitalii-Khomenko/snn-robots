#pragma once
// Host shim for the M5Stack CoreS3 BSP - only what the reused sources call.
#include "../esp_camera.h"
#include "../esp_codec_dev.h"
#include "../esp_err.h"
#define BSP_I2C_NUM 1
#define BSP_CAMERA_DEFAULT_CONFIG {PIXFORMAT_RGB565, FRAMESIZE_QVGA, 12, 2, CAMERA_GRAB_WHEN_EMPTY}
esp_err_t bsp_i2c_init();
esp_codec_dev_handle_t bsp_audio_codec_microphone_init();
