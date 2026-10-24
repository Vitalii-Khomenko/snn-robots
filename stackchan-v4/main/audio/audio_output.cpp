#include "audio_output.h"

#include <bsp/esp-bsp.h>
#include <esp_codec_dev_defaults.h>
#include <aw88298_dac.h>
#include <esp_log.h>

#include "../power/power_rails.h"

namespace {
constexpr const char* kTag      = "audio_out";
constexpr uint16_t kAw88298Addr = 0x36 << 1;
}  // namespace

bool AudioOutput::begin()
{
    // bsp_audio_codec_speaker_init() does more than build a codec handle: it
    // also runs bsp_enable_feature(BSP_FEATURE_SPEAKER), which sets the
    // AXP2101 ALDO2/ALDO3 codec rails *and* flips the AW9523 GPIO bit that
    // separately gates the AW88298 ("Enable Codec AW88298"). Call it for
    // that side effect, then discard its pa_gain=15 handle and build our
    // own on top of the now-enabled hardware.
    esp_codec_dev_handle_t bsp_handle = bsp_audio_codec_speaker_init();
    if (bsp_handle) {
        esp_codec_dev_delete(bsp_handle);
    }
    power::boostSpeakerPaVoltage();

    const audio_codec_data_if_t* i2s_data_if = bsp_audio_get_codec_itf();
    if (!i2s_data_if) {
        ESP_LOGE(kTag, "bsp_audio_init() must run before AudioOutput::begin()");
        return false;
    }

    i2c_master_bus_handle_t bus = nullptr;
    if (i2c_master_get_bus_handle(BSP_I2C_NUM, &bus) != ESP_OK) {
        return false;
    }

    audio_codec_i2c_cfg_t i2c_cfg = {};
    i2c_cfg.port                  = BSP_I2C_NUM;
    i2c_cfg.addr                  = kAw88298Addr;
    i2c_cfg.bus_handle            = bus;
    const audio_codec_ctrl_if_t* ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    if (!ctrl_if) {
        return false;
    }

    const audio_codec_gpio_if_t* gpio_if = audio_codec_new_gpio();

    aw88298_codec_cfg_t aw_cfg       = {};
    aw_cfg.ctrl_if                    = ctrl_if;
    aw_cfg.gpio_if                    = gpio_if;
    aw_cfg.reset_pin                  = -1;
    aw_cfg.hw_gain.pa_voltage         = 5.0f;
    aw_cfg.hw_gain.codec_dac_voltage  = 3.3f;
    aw_cfg.hw_gain.pa_gain            = 1;
    const audio_codec_if_t* codec_if = aw88298_codec_new(&aw_cfg);
    if (!codec_if) {
        return false;
    }

    esp_codec_dev_cfg_t dev_cfg = {};
    dev_cfg.dev_type             = ESP_CODEC_DEV_TYPE_OUT;
    dev_cfg.codec_if             = codec_if;
    dev_cfg.data_if              = i2s_data_if;
    speaker_                     = esp_codec_dev_new(&dev_cfg);
    if (!speaker_) {
        return false;
    }

    esp_codec_dev_sample_info_t fs = {};
    fs.bits_per_sample              = 16;
    fs.channel                      = 1;
    fs.channel_mask                 = 0;
    fs.sample_rate                  = kSampleRate;
    fs.mclk_multiple                = 0;

    // Opened once here and left open (see the header comment for why) -
    // esp_codec_dev_set_out_vol() must come after open(), or it can return
    // ESP_CODEC_DEV_WRONG_STATE ("Driver not open yet").
    if (esp_codec_dev_open(speaker_, &fs) != ESP_OK) {
        ESP_LOGE(kTag, "esp_codec_dev_open failed");
        return false;
    }
    if (esp_codec_dev_set_out_vol(speaker_, 90) != ESP_OK) {
        ESP_LOGE(kTag, "esp_codec_dev_set_out_vol failed");
        return false;
    }
    volume_ = 90;
    return true;
}

void AudioOutput::playPcm(const int16_t* samples, size_t count, int volume)
{
    if (!speaker_) {
        return;
    }
    // An I2C write to the amplifier: only when the volume actually changes,
    // not on every 20 ms chunk.
    if (volume != volume_) {
        esp_codec_dev_set_out_vol(speaker_, volume);
        volume_ = volume;
    }
    self_sound_.process(samples, count);
    // blocks until the DMA takes it: this paces the voice task
    esp_codec_dev_write(speaker_, const_cast<int16_t*>(samples), count * sizeof(int16_t));
}
