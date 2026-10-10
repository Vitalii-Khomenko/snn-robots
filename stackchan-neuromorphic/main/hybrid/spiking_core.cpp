#include "spiking_core.h"

#include <algorithm>
#include <cmath>

#include "nn_ops.h"
#include "snn_model_data.h"

using namespace hybrid;

SpikingCore::SpikingCore() { reset(); }

bool SpikingCore::trained() const { return snn_model::kTrained; }

const int8_t* SpikingCore::inputWeights() { return snn_model::win_w; }

int SpikingCore::synapseCount() const { return snn_model::kSynapses; }

void SpikingCore::reset()
{
    v_.fill(0.0f);
    acc_.fill(0.0f);
    s_.fill(0);
    z_.fill(0.0f);
    att_.fill(0.0f);
    n_spiked_ = 0;
    mean_rate_ = 0.0f;
    hot_ticks_ = 0;
    unstable_ = false;
}

uint32_t SpikingCore::rnd()
{
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return rng_;
}

// bus + two fixed place codes: where the vision target is and where the
// bright spot is (tools/hybrid/snn_model.py encode()).
void SpikingCore::encode(const float* bus)
{
    std::copy(bus, bus + kSnnInputs, in_.begin());
    auto at = [&](int g, int i) { return bus[g + i]; };
    const float tx = at(kSnnVisionAt, VIS_TARGET_X), ty = at(kSnnVisionAt, VIS_TARGET_Y);
    const float tc = at(kSnnVisionAt, VIS_TARGET_CONF);
    const float lx = at(kSnnVisionAt, VIS_LIGHT_X), ly = at(kSnnVisionAt, VIS_LIGHT_Y);
    const float ls = at(kSnnVisionAt, VIS_LIGHT_STRENGTH) * 0.7f;
    float* tg = in_.data() + kSnnInputs;
    float* lg = tg + kMapCells;
    // Gaussian bumps; cells further than ~3 sigma stay at zero.
    auto bump = [](float* out, float amp, float cx, float cy) {
        std::fill(out, out + kMapCells, 0.0f);
        if (amp < 0.01f) return;
        for (int row = 0; row < kMapH; ++row) {
            const float gy = (static_cast<float>(row) + 0.5f) / kMapH * 2.0f - 1.0f;
            const float dy = (gy - cy) * (gy - cy) / (2 * 0.25f * 0.25f);
            if (dy > 4.5f) continue;
            for (int col = 0; col < kMapW; ++col) {
                const float gx = (static_cast<float>(col) + 0.5f) / kMapW * 2.0f - 1.0f;
                const float dx = (gx - cx) * (gx - cx) / (2 * 0.17f * 0.17f);
                if (dx + dy < 6.0f) out[row * kMapW + col] = amp * std::exp(-(dx + dy));
            }
        }
    };
    bump(tg, tc, tx, ty);
    bump(lg, ls, lx, ly);
}

bool SpikingCore::tick(const float* bus, Values& out)
{
    using namespace snn_model;
    if (!kTrained) return false;
    const int64_t t0 = clock_us_ ? clock_us_() : 0;
    encode(bus);
    // input drive: dense, column by column (INT8 with a scale per input column)
    for (int i = 0; i < kNeurons; ++i) drive_[i] = bias[i];
    for (int c = 0; c < kInputs; ++c) {
        // inputs below 2 % of full scale add nothing the noise does not drown (and
        // every skipped column is 768 bytes less to read from flash)
        if (std::fabs(in_[c]) < kMinInput) continue;
        const float x = in_[c] * win_s[c];
        const int8_t* col = (win_ptr_ ? win_ptr_ : win_w) + c * kNeurons;
        float* d = drive_.data();
        for (int i = 0; i < kNeurons; ++i) d[i] += static_cast<float>(col[i]) * x;
    }
    const int64_t t1 = clock_us_ ? clock_us_() : 0;
    std::array<uint8_t, kNeurons> count{};
    last_spikes_ = 0;
    // Uniform noise with the Gaussian's variance (one random number per
    // neuron instead of four): sigma * sqrt(3) * (2u - 1).
    const float amp = noise_ * 1.7320508f * 2.0f;
    uint32_t rng = rng_;
    for (int sub = 0; sub < kSubsteps; ++sub) {
        // spikes of the previous substep were already scattered into acc_
        n_spiked_ = 0;
        for (int i = 0; i < kNeurons; ++i) {
            float cur = drive_[i] + acc_[i] * rec_s[i];
            if (amp > 0.0f) {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                cur += amp * (static_cast<float>(rng >> 8) * (1.0f / 16777216.0f) - 0.5f);
            }
            const float v = beta[i] * v_[i] * (s_[i] ? 0.0f : 1.0f) + cur;
            v_[i] = v;
            const bool fire = v > theta[i];
            s_[i] = fire ? 1 : 0;
            if (fire) {
                spiked_[n_spiked_++] = static_cast<uint16_t>(i);
                ++count[i];
            }
        }
        acc_.fill(0.0f);
        for (int k = 0; k < n_spiked_; ++k) {
            const int j = spiked_[k];
            for (uint32_t e = rec_off[j]; e < rec_off[j + 1]; ++e) {
                const uint32_t syn = rec_syn[e];  // post neuron << 8 | int8 weight: one cache line, one load
                acc_[syn >> 8] += static_cast<float>(static_cast<int8_t>(syn & 0xffu));
            }
        }
        last_spikes_ += n_spiked_;
    }
    rng_ = rng;
    const int64_t t2 = clock_us_ ? clock_us_() : 0;
    mean_rate_ += (static_cast<float>(last_spikes_) / (kSubsteps * kNeurons) - mean_rate_) * 0.05f;

    // readouts: leaky integrators over the (masked, weighted) population rates
    float ro[kReadouts];
    for (int k = 0; k < kReadouts; ++k) {
        const int o = ro_start[k], n = ro_size[k];
        const int8_t* w = ro_w + ro_woff[k];
        float acc = 0.0f;
        for (int i = 0; i < n; ++i) acc += static_cast<float>(w[i]) * static_cast<float>(count[o + i]);
        acc *= ro_s[k] * (1.0f / kSubsteps);
        z_[k] = ro_alpha[k] * z_[k] + (1.0f - ro_alpha[k]) * acc;
        ro[k] = nn::sigmoid(z_[k] + ro_bias[k]);
    }
    float tot = 0.0f, ax = 0.0f, ay = 0.0f;
    for (int i = 0; i < kMapCells; ++i) {
        att_[i] = 0.8f * att_[i] + 0.2f * static_cast<float>(count[att_start + i]) * (1.0f / kSubsteps);
        const float gx = (static_cast<float>(i % kMapW) + 0.5f) / kMapW * 2.0f - 1.0f;
        const float gy = (static_cast<float>(i / kMapW) + 0.5f) / kMapH * 2.0f - 1.0f;
        tot += att_[i];
        ax += att_[i] * gx;
        ay += att_[i] * gy;
    }
    for (int k = 0; k < kReadouts; ++k) out[ro_out[k]] = ro[k];
    out[SNN_ATT_X] = ax / (tot + 0.05f);
    out[SNN_ATT_Y] = ay / (tot + 0.05f);

    if (clock_us_) {
        const int64_t t3 = clock_us_();
        const float us[3] = {static_cast<float>(t1 - t0), static_cast<float>(t2 - t1), static_cast<float>(t3 - t2)};
        for (int i = 0; i < 3; ++i) stage_us_[i] += (us[i] - stage_us_[i]) * 0.1f;
    }

    // safety: a core that saturates or goes non-finite is restarted by the caller
    const bool finite = nn::finite(out.data(), SNN_COUNT);
    hot_ticks_ = mean_rate_ > 0.4f ? hot_ticks_ + 1 : 0;
    unstable_ = !finite || hot_ticks_ > 40;
    return true;
}
