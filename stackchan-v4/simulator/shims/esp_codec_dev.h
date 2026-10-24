#pragma once
// Host shim: microphone reads come from the simulated room acoustics and
// block for the audio they return (virtual time), like the I2S DMA does.
#include <cstddef>
#include <cstdint>
typedef struct sim_codec* esp_codec_dev_handle_t;
typedef struct {
    uint8_t bits_per_sample;
    uint8_t channel;
    uint16_t channel_mask;
    uint32_t sample_rate;
    int mclk_multiple;
} esp_codec_dev_sample_info_t;
int esp_codec_dev_open(esp_codec_dev_handle_t dev, esp_codec_dev_sample_info_t* fs);
int esp_codec_dev_read(esp_codec_dev_handle_t dev, void* data, int len);
int esp_codec_dev_write(esp_codec_dev_handle_t dev, void* data, int len);
int esp_codec_dev_set_in_gain(esp_codec_dev_handle_t dev, float db);
int esp_codec_dev_set_out_vol(esp_codec_dev_handle_t dev, int volume);
