#include "assoc_memory.h"

#include <algorithm>
#include <cmath>

using namespace hybrid;

namespace {
constexpr float kDt = 0.05f;
constexpr float kEligTau = 3.5f;     // s: how far back a cue still counts as "just before"
constexpr float kEtaOnset = 0.3f;    // learning rate at the onset of an outcome
constexpr float kEtaExtinct = 0.15f;   // extinction per episode of a cue that no outcome followed
constexpr float kEtaPresent = 0.001f;  // ... and per tick while a cue is simply there (much slower: base rate)
// Salience (Rescorla-Wagner's alpha) per cue: events that come and go (a voice, a bang, motion) are what predicts;
// the context that is there for minutes (a person, a face, the owner at home, night) learns at a third of the rate -
// otherwise it soaked up the whole prediction after two pairings and blocked the voice (Kamin blocking).
constexpr float kSalience[AssocMemory::kCues] = {1.0f, 1.0f, 1.0f, 1.0f, 0.35f, 0.35f, 1.0f, 1.0f, 1.0f, 0.35f, 0.35f};
constexpr float kEpisodeElig = 0.25f;  // a cue's episode closes when its trace falls below this
constexpr int kQuietTicks = 60;      // nothing is learned for 3 s after an outcome (it is the robot's reaction)
constexpr float kForgetTau = 4.0f * 86400.0f;  // s: unconfirmed links fade over days
constexpr float kOutLeak = 0.8f;     // outcome cells: membrane decay per tick
constexpr float kRateKeep = 0.75f;   // output: firing-rate smoothing per tick (~0.2 s: a clap is short)
constexpr float kOutGain = 1.5f;     // expectation -> outcome cells' input current (they start firing from ~0.11)

float live(float x) { return std::clamp((x - 0.15f) / 0.85f, 0.0f, 1.0f); }  // below 0.15 is perception noise
}  // namespace

const char* AssocMemory::cueName(int c)
{
    static const char* const names[kCues] = {"voice", "bang", "tone", "steady noise", "person", "face",
                                             "motion", "approach", "light change", "owner home", "night"};
    return c >= 0 && c < kCues ? names[c] : "?";
}

void AssocMemory::cues(const Bus& bus, std::array<float, kCues>& c) const
{
    const auto& a = bus.audio;
    const auto& v = bus.vision;
    // what the ear hears while our own servos run is mostly the servos
    const float ext = a[AUD_OWN_MOTION] > 0.5f ? 0.0f : 1.0f;
    c[0] = live(a[AUD_SPEECH]) * ext;
    c[1] = live(std::max(a[AUD_CLAP], a[AUD_SHARP])) * ext;
    c[2] = live(a[AUD_TONAL]) * ext;
    c[3] = live(a[AUD_CONTINUOUS]) * ext;
    c[4] = live(v[VIS_PERSON]);
    c[5] = live(v[VIS_FACE]);
    c[6] = live(std::max({v[VIS_MOTION_L], v[VIS_MOTION_C], v[VIS_MOTION_R]}));
    c[7] = live(std::max(v[VIS_LOOM], a[AUD_APPROACH]));
    c[8] = live(std::fabs(v[VIS_LIGHT_CHANGE]) * 1.5f);
    c[9] = bus.ctx[CTX_OWNER_NEAR];
    c[10] = bus.ctx[CTX_NIGHT];
}

void AssocMemory::tick(Bus& bus)
{
    // ---- cue cells: integrate-and-fire, rate = cue strength (staggered phases); eligibility traces of their spikes
    std::array<float, kCues> cue{};
    cues(bus, cue);
    const float elig_decay = std::exp(-kDt / kEligTau);
    for (int c = 0; c < kCues; ++c) {
        int spikes = 0;
        for (int k = 0; k < kCueCells; ++k) {
            float& m = cue_v_[c][k];
            m += cue[c] * (1.05f + 0.1f * static_cast<float>(k));  // a full-strength cue fires every cell at once
            if (m >= 1.0f) {
                m -= 1.0f;
                ++spikes;
            }
            if (cue[c] <= 0.0f) m *= 0.5f;  // a silent cue does not keep a half-charged membrane for long
        }
        elig_[c] = std::max(elig_[c] * elig_decay, static_cast<float>(spikes) / kCueCells);
        if (spikes > 0 && !in_episode_[c]) {
            in_episode_[c] = true;
            reinforced_[c].fill(false);
        }
    }

    // ---- the teaching signals: the outcomes themselves
    const float touch = std::max(bus.body[BODY_TOUCH_HEAD], bus.body[BODY_TOUCH_SCREEN]);
    const bool us[kOutcomes] = {touch > 0.5f && bus.snn[SNN_ANNOYANCE] < 0.45f,
                                std::max(bus.body[BODY_SHAKE], bus.body[BODY_TILT]) > 0.5f};
    const float forget = 1.0f - kDt / kForgetTau;
    for (int o = 0; o < kOutcomes; ++o) {
        // expectation through the plastic synapses
        float expect = 0.0f;
        for (int c = 0; c < kCues; ++c) expect += w_[c][o] * elig_[c];
        expect = std::min(expect, 1.0f);
        const bool onset = us[o] && !was_on_[o];
        was_on_[o] = us[o];
        since_onset_[o] = onset ? 0 : std::min(since_onset_[o] + 1, 1000);
        // three-factor rule: presynaptic eligibility x outcome prediction error
        for (int c = 0; c < kCues; ++c) {
            const float before = w_[c][o];
            float w = before * forget;
            if (onset) {
                w += kEtaOnset * kSalience[c] * elig_[c] * (1.0f - expect);
                if (elig_[c] >= kEpisodeElig) reinforced_[c][o] = true;
            } else if (!us[o] && since_onset_[o] > kQuietTicks) {
                if (in_episode_[c] && cue[c] <= 0.0f && elig_[c] < kEpisodeElig && !reinforced_[c][o])
                    w -= kEtaExtinct * w;  // it came and nothing followed: the link loses 15 %
                else if (cue[c] > 0.0f)
                    w -= kEtaPresent * elig_[c] * expect;
            }
            w = std::clamp(w, 0.0f, kMaxWeight);
            w_[c][o] = w;
            change_ = std::max(change_, std::fabs(w - before));
        }
        // outcome cells: leaky integrate-and-fire driven only by the learned expectation (thresholds 0.8 .. 1.2)
        int spikes = 0;
        for (int k = 0; k < kOutCells; ++k) {
            float& m = out_v_[o][k];
            m = kOutLeak * m + kOutGain * expect;
            if (m >= 0.8f + 0.4f * static_cast<float>(k) / (kOutCells - 1)) {
                m = 0.0f;
                ++spikes;
            }
        }
        rate_[o] = kRateKeep * rate_[o] + (1.0f - kRateKeep) * static_cast<float>(spikes) / kOutCells;
    }
    for (int c = 0; c < kCues; ++c)  // episodes whose trace has faded are closed
        if (in_episode_[c] && cue[c] <= 0.0f && elig_[c] < kEpisodeElig) in_episode_[c] = false;
    bus.learn[LEARN_EXPECT_REWARD] = std::clamp(rate_[Reward] * 1.5f, 0.0f, 1.0f);
    bus.learn[LEARN_EXPECT_THREAT] = std::clamp(rate_[Threat] * 1.5f, 0.0f, 1.0f);
}

void AssocMemory::setWeights(const Weights& w)
{
    for (int c = 0; c < kCues; ++c)
        for (int o = 0; o < kOutcomes; ++o) {
            const float x = w[c][o];
            w_[c][o] = std::isfinite(x) ? std::clamp(x, 0.0f, kMaxWeight) : 0.0f;
            logged_[c][o] = std::floor(std::min(w_[c][o], 0.5f) / 0.25f) * 0.25f;
        }
}

float AssocMemory::takeChange()
{
    const float c = change_;
    change_ = 0.0f;
    return c;
}

bool AssocMemory::takeEvent(int& cue, int& outcome, float& weight, bool& rising)
{
    for (int c = 0; c < kCues; ++c)
        for (int o = 0; o < kOutcomes; ++o) {
            const float level = std::floor(std::min(w_[c][o], 0.5f) / 0.25f) * 0.25f;
            // hysteresis: a level is left downwards only 0.1 below it
            const bool up = level > logged_[c][o];
            const bool down = w_[c][o] < logged_[c][o] - 0.1f;
            if (!up && !down) continue;
            logged_[c][o] = up ? level : std::floor(std::max(w_[c][o], 0.0f) / 0.25f) * 0.25f;
            cue = c;
            outcome = o;
            weight = w_[c][o];
            rising = up;
            return true;
        }
    return false;
}
