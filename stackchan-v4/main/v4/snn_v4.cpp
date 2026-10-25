#include "snn_v4.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "snn_v4_data.h"

#if __has_include(<esp_heap_caps.h>)
#include <esp_heap_caps.h>
#define SNNV4_HAVE_PSRAM 1
#endif

namespace {
constexpr int kMapW = 12, kMapH = 8;
static_assert(kMapW * kMapH == SnnV4::kAtt, "attention map size");
}  // namespace

SnnV4::SnnV4()
    : w_in_t_(snnv4::kWInT), w_rec_t_(snnv4::kWRecT), w_att_t_(snnv4::kWAttT), w_ro_t_(snnv4::kWRoT)
{
    reset();
}

bool SnnV4::trained() const { return snnv4::kTrained; }

void SnnV4::reset()
{
    v_enc_.fill(0.0f);
    v_zone_.fill(0.0f);
    a_zone_.fill(0.0f);
    v_mix_.fill(0.0f);
    s_mix_.fill(0);
    v_att_.fill(0.0f);
    ro_.fill(0.0f);
    has_prev_ = false;
}

void SnnV4::useFastWeights()
{
#ifdef SNNV4_HAVE_PSRAM
    auto copy = [](const int8_t*& p, size_t n) {
        if (void* m = heap_caps_malloc(n, MALLOC_CAP_SPIRAM)) {
            std::memcpy(m, p, n);
            p = static_cast<const int8_t*>(m);
        }
    };
    copy(w_in_t_, sizeof(snnv4::kWInT));
    copy(w_rec_t_, sizeof(snnv4::kWRecT));
    copy(w_att_t_, sizeof(snnv4::kWAttT));
    copy(w_ro_t_, sizeof(snnv4::kWRoT));
#endif
}

bool SnnV4::tick(const In& in, Out& out)
{
    if (!snnv4::kTrained) return false;
    const int64_t t0 = clock_us_ ? clock_us_() : 0;
    using namespace snnv4;

    // ---- input currents of this tick: level neurons (receptive fields), ON / OFF, the attention bump
    float i_enc[kEnc];
    for (int c = 0; c < kChannels; ++c) {
        const float x = in[c];
        for (int k = 0; k < kGrf; ++k) {
            const float d = x - (static_cast<float>(k) + 0.5f) / kGrf;
            i_enc[c * kGrf + k] = kGrfGain * std::exp(-d * d / (2.0f * kGrfSigma * kGrfSigma));
        }
        const float dx = has_prev_ ? x - prev_[c] : 0.0f;
        i_enc[kChannels * kGrf + 2 * c] = std::max(dx, 0.0f) * kOnOffGain;
        i_enc[kChannels * kGrf + 2 * c + 1] = std::max(-dx, 0.0f) * kOnOffGain;
        prev_[c] = x;
    }
    has_prev_ = true;
    float bump[kAtt];
    {
        // a bump where the movement is, and one where a new bright spot is (2026-10-25)
        constexpr int w = v4::CH_COUNT;
        const float amp = std::clamp(in[v4::CH_MOTION] * 4.0f, 0.0f, 1.0f) * kBumpGain;
        const float mx = in[w + v4::WH_MOTION_X], my = in[w + v4::WH_MOTION_Y];
        const float lamp = std::clamp(in[w + v4::WH_LIGHT_SPOT], 0.0f, 1.0f) * kBumpGain;
        const float lx = in[w + v4::WH_LIGHT_X], ly = in[w + v4::WH_LIGHT_Y];
        auto g = [](float gx, float gy, float cx, float cy) {
            return std::exp(-((gx - cx) * (gx - cx) / (2.0f * kBumpSx * kBumpSx) +
                              (gy - cy) * (gy - cy) / (2.0f * kBumpSy * kBumpSy)));
        };
        for (int y = 0; y < kMapH; ++y)
            for (int x = 0; x < kMapW; ++x) {
                const float gx = (static_cast<float>(x) + 0.5f) / kMapW * 2.0f - 1.0f;
                const float gy = (static_cast<float>(y) + 0.5f) / kMapH * 2.0f - 1.0f;
                bump[y * kMapW + x] = amp * g(gx, gy, mx, my) + lamp * g(gx, gy, lx, ly);
            }
    }

    int32_t ro_acc[kOut] = {};
    int mix_spikes = 0;
    uint16_t* spiked = spiked_.data();  // scratch kept off the main task's stack
    for (int sub = 0; sub < kSubsteps; ++sub) {
        int ns = 0;
        // ENC
        uint8_t s_enc[kEnc];
        for (int i = 0; i < kEnc; ++i) {
            const float v = kBetaEnc * v_enc_[i] + i_enc[i];
            s_enc[i] = v > 1.0f;
            v_enc_[i] = v - s_enc[i];
            if (s_enc[i]) spiked[ns++] = static_cast<uint16_t>(i);
        }
        // ZONE: each listens to its own channel's level neurons, threshold rises with its own spikes
        for (int c = 0; c < kChannels; ++c)
            for (int z = 0; z < v4::ZONE_COUNT; ++z) {
                float drive = 0.0f;
                for (int k = 0; k < kGrf; ++k)
                    if (s_enc[c * kGrf + k]) drive += kZoneW[c][z][k];
                for (int j = 0; j < 4; ++j) {
                    const int i = (c * v4::ZONE_COUNT + z) * 4 + j;
                    const float v = kBetaZone * v_zone_[i] + drive;
                    const bool s = v > 1.0f + a_zone_[i];
                    v_zone_[i] = s ? v - (1.0f + a_zone_[i]) : v;
                    a_zone_[i] = a_zone_[i] * kAdaptKeep[j] + (s ? kAdaptStep[j] : 0.0f);
                    if (s) spiked[ns++] = static_cast<uint16_t>(kEnc + i);
                }
            }
        // MIX: input from the ENC / ZONE spikes, recurrence from its own spikes of the last substep
        int32_t* acc_in = acc_in_.data();
        int32_t* acc_rec = acc_rec_.data();
        std::fill(acc_in_.begin(), acc_in_.end(), 0);
        std::fill(acc_rec_.begin(), acc_rec_.end(), 0);
        for (int n = 0; n < ns; ++n) {
            const int8_t* col = w_in_t_ + static_cast<size_t>(spiked[n]) * kMix;
            for (int j = 0; j < kMix; ++j) acc_in[j] += col[j];
        }
        for (int i = 0; i < kMix; ++i)
            if (s_mix_[i]) {
                const int8_t* col = w_rec_t_ + static_cast<size_t>(i) * kMix;
                for (int j = 0; j < kMix; ++j) acc_rec[j] += col[j];
            }
        for (int j = 0; j < kMix; ++j) {
            const float v = kBetaMix * v_mix_[j] + static_cast<float>(acc_in[j]) * kInScale[j] +
                            static_cast<float>(acc_rec[j]) * kRecScale[j] + kBMix[j];
            s_mix_[j] = v > 1.0f;
            v_mix_[j] = v - s_mix_[j];
            if (s_mix_[j]) {
                spiked[ns++] = static_cast<uint16_t>(kEnc + kZone + j);
                ++mix_spikes;
            }
        }
        // ATT
        int32_t acc_att[kAtt] = {};
        for (int i = 0; i < kMix; ++i)
            if (s_mix_[i]) {
                const int8_t* col = w_att_t_ + static_cast<size_t>(i) * kAtt;
                for (int j = 0; j < kAtt; ++j) acc_att[j] += col[j];
            }
        for (int j = 0; j < kAtt; ++j) {
            const float v = kBetaAtt * v_att_[j] + bump[j] + static_cast<float>(acc_att[j]) * kAttScale[j] + kBAtt[j];
            const bool s = v > 1.0f;
            v_att_[j] = s ? v - 1.0f : v;
            if (s) spiked[ns++] = static_cast<uint16_t>(kEnc + kZone + kMix + j);
        }
        // readout sums over every spike of this substep
        for (int n = 0; n < ns; ++n) {
            const int8_t* col = w_ro_t_ + static_cast<size_t>(spiked[n]) * kOut;
            for (int k = 0; k < kOut; ++k) ro_acc[k] += col[k];
        }
    }
    for (int k = 0; k < kOut; ++k) {
        const float drive = static_cast<float>(ro_acc[k]) * kRoScale[k] / kSubsteps + kBRo[k];
        ro_[k] = ro_[k] * kRoKeep[k] + (1.0f - kRoKeep[k]) * drive;
        out[k] = kSigned[k] ? std::tanh(ro_[k]) : 1.0f / (1.0f + std::exp(-ro_[k]));
    }
    mix_rate_ += (static_cast<float>(mix_spikes) / (kMix * kSubsteps) - mix_rate_) * 0.05f;
    if (clock_us_) last_us_ = static_cast<uint32_t>(clock_us_() - t0);
    return true;
}
