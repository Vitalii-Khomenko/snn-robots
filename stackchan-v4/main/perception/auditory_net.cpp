#include "auditory_net.h"

#include <algorithm>
#include <cmath>
#include <iterator>

#include "auditory_model_data.h"
#include "nn_ops.h"

bool AuditoryNet::trained() const { return auditory_model::kTrained; }

AuditoryNet::Params AuditoryNet::compiledParams()
{
    using namespace auditory_model;
    return {{{fc1_w, fc1_s, fc1_b}, {fc2_w, fc2_s, fc2_b}, {fc3_w, fc3_s, fc3_b}}};
}

bool AuditoryNet::infer(const float* input)
{
    if (!auditory_model::kTrained) return false;
    const Params& p = p_;
    nn::dense(p[0].w, p[0].s, p[0].b, input, 64, AudioFrontEnd::kInputs, h1_.data(), true);
    nn::dense(p[1].w, p[1].s, p[1].b, h1_.data(), 32, 64, h2_.data(), true);
    nn::dense(p[2].w, p[2].s, p[2].b, h2_.data(), kOutputs, 32, out_.data(), false);
    if (!nn::finite(out_.data(), kOutputs)) {
        ++nan_count_;
        return false;
    }
    for (int i = 0; i < kOutputs; ++i) out_[i] = nn::sigmoid(out_[i]);
    return true;
}

bool AuditoryNet::run(const AudioFrontEnd& front, const Context& ctx, Features& out)
{
    out.fill(0.0f);
    out[AUD_SELF_VOICE] = ctx.selfLevel01;
    out[AUD_OWN_MOTION] = ctx.ownMotion ? 1.0f : 0.0f;
    out[AUD_ONSET] = ctx.onset01;
    out[AUD_LOUD] = ctx.loud01;
    std::array<AudioFrontEnd::Frame, AudioFrontEnd::kSpan> frames;
    if (!front.latest(frames)) {
        out[AUD_SILENCE] = 1.0f;
        return false;
    }
    AudioFrontEnd::buildInput(frames, ctx.selfSpeaking, ctx.ownMotion, in_);
    // Every band is measured against its own quiet baseline (min tracker: drops at
    // once, rises ~0.04 per minute): the network sees "how far above this room's
    // hum", so a microphone or a room with a different noise colour looks the same.
    for (int b = 0; b < AudioFrontEnd::kBands; ++b) {
        const float now_b = in_[b];  // newest frame
        band_floor_[b] = band_floor_valid_ ? std::min(now_b, band_floor_[b] + 0.00003f) : now_b;
    }
    band_floor_valid_ = true;
    for (int h = 0; h < AudioFrontEnd::kHistory; ++h)
        for (int b = 0; b < AudioFrontEnd::kBands; ++b)
            in_[h * AudioFrontEnd::kBands + b] = std::max(0.0f, in_[h * AudioFrontEnd::kBands + b] - band_floor_[b]);

    // Exact measurements.
    const float level = in_[AudioFrontEnd::kBands * AudioFrontEnd::kHistory + 1];
    const bool masked = ctx.selfSpeaking;  // our own voice is not "the world"
    // The quiet baseline of this room / microphone (drops at once, rises over
    // minutes) is subtracted: silence reads 0 wherever the robot stands.
    floor_ = std::min(level, std::min(floor_ + 0.00003f, 0.5f));
    in_[AudioFrontEnd::kBands * AudioFrontEnd::kHistory + 1] = std::max(0.0f, level - floor_);
    out[AUD_ENERGY] = masked ? 0.0f : std::clamp((level - floor_ - 0.03f) / 0.8f, 0.0f, 1.0f);
    {
        // novelty: how far the current spectrum is from the slow background.
        // Frozen while the servos run or we speak (our own noise is not new).
        const AudioFrontEnd::Frame& now = frames[AudioFrontEnd::kSpan - 1];
        if (!background_ready_) {
            background_ = now.band;
            background_ready_ = true;
        }
        float dist = 0.0f;
        for (int b = 0; b < AudioFrontEnd::kBands; ++b) dist += std::fabs(now.band[b] - background_[b]);
        dist /= AudioFrontEnd::kBands;
        if (!ctx.selfSpeaking && !ctx.ownMotion) {
            for (int b = 0; b < AudioFrontEnd::kBands; ++b) background_[b] += (now.band[b] - background_[b]) * 0.01f;
            out[AUD_NOVELTY] = std::clamp(dist * 6.0f, 0.0f, 1.0f);
        }
        // approach: the level has been rising for >= ~1 s
        for (size_t i = 0; i + 1 < envelope_.size(); ++i) envelope_[i] = envelope_[i + 1];
        envelope_.back() = masked || ctx.ownMotion ? 0.0f : level;
        envelope_n_ = std::min<int>(envelope_n_ + 1, static_cast<int>(envelope_.size()));
        if (envelope_n_ == static_cast<int>(envelope_.size())) {
            float early = 0.0f, late = 0.0f;
            for (size_t i = 0; i < envelope_.size() / 2; ++i) {
                early += envelope_[i];
                late += envelope_[envelope_.size() / 2 + i];
            }
            const float half = static_cast<float>(envelope_.size() / 2);
            early /= half;
            late /= half;
            out[AUD_APPROACH] = early > 0.12f ? std::clamp((late - early) / (early + 0.1f) * 1.5f, 0.0f, 1.0f) : 0.0f;
        }
    }
    if (!infer(in_.data())) {  // missing network or NaN: hear nothing rather than an undefined class
        out[AUD_SILENCE] = 1.0f;
        return false;
    }
    // A network trained on simulated and recorded sounds can mistake a real
    // room's hiss for speech or noise. When nothing rises above the quiet
    // baseline of this room there is nothing to classify: the classes fade out
    // with the level (full strength from +0.15 above the baseline).
    const float gate = masked ? 0.0f : std::clamp((level - floor_ - 0.03f) / 0.15f, 0.0f, 1.0f);
    for (int i : {1, 2, 3, 4, 5, 6, 7}) out_[i] *= gate;
    out_[0] = 1.0f - gate * (1.0f - out_[0]);
    out[AUD_SILENCE] = out_[0];
    out[AUD_SPEECH] = out_[1];
    out[AUD_CLAP] = out_[2];
    out[AUD_SHARP] = out_[3];
    out[AUD_CONTINUOUS] = out_[4];
    out[AUD_LOW] = out_[5];
    out[AUD_HIGH] = out_[6];
    out[AUD_TONAL] = out_[7];
    return true;
}
