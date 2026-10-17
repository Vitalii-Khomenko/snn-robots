#pragma once
// Host shim for esp32-camera: frames are rendered from the simulated room.
#include <cstddef>
#include <cstdint>
#include "esp_err.h"
typedef enum { FRAMESIZE_QQVGA = 0, FRAMESIZE_QVGA = 1 } framesize_t;
typedef enum { PIXFORMAT_RGB565 = 0 } pixformat_t;
typedef enum { CAMERA_GRAB_WHEN_EMPTY = 0, CAMERA_GRAB_LATEST = 1 } camera_grab_mode_t;
typedef struct {
    pixformat_t pixel_format;
    framesize_t frame_size;
    int jpeg_quality;
    size_t fb_count;
    camera_grab_mode_t grab_mode;
} camera_config_t;
typedef struct {
    uint8_t* buf;
    size_t len;
    size_t width;
    size_t height;
    pixformat_t format;
} camera_fb_t;
typedef struct _sensor sensor_t;
struct _sensor {
    int (*set_exposure_ctrl)(sensor_t* sensor, int enable);
    int (*set_gain_ctrl)(sensor_t* sensor, int enable);
    int (*set_whitebal)(sensor_t* sensor, int enable);
};
esp_err_t esp_camera_init(const camera_config_t* config);
esp_err_t esp_camera_deinit();
sensor_t* esp_camera_sensor_get();
camera_fb_t* esp_camera_fb_get();
void esp_camera_fb_return(camera_fb_t* fb);
