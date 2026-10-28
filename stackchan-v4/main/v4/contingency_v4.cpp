#include "contingency_v4.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr float kWeight[ContingencyV4::kAnswers] = {1.0f, 0.8f, 0.6f, 0.5f, 0.3f};  // pet, tap, speech, near, motion
constexpr uint32_t kQuietMs[ContingencyV4::kAnswers] = {2000, 2000, 2000, 3000, 3000};  // an onset after this quiet
constexpr uint32_t kWinLoMs = 500, kWinHiMs = 4000;  // an answer this long after an act is the act's
constexpr uint32_t kReactMs = 3000;      // an act this soon after an answer is a reaction to it
constexpr uint32_t kNoActMs = 5000;      // background: no act in the last 5 s
constexpr uint32_t kAroundMs = 120000;   // somebody is around: a touch, someone near or moving in view, last 2 min
constexpr uint32_t kActQuietMs = 1000;   // a call / a gesture after this quiet is a new act
constexpr float kTauNowS = 300.0f, kTauBgS = 1800.0f, kTauPulseS = 3.0f, kTauIgnoredS = 300.0f;
constexpr float kPrior = 3.0f;           // acts' worth of "chance" before a kind of act means anything
// the background rate's prior, seconds with no answer (contingency.py BG_PRIOR_S: 10 tells pets at random from answers
// better, 2026-10-28; 60 until a network trained with it goes with it)
constexpr float kBgPriorS = 60.0f;
constexpr float kIgnoreStep = 0.1f;  // per unanswered call or gesture (a face is not a bid: nobody answers every face)
constexpr float kWindowS = (kWinHiMs - kWinLoMs) / 1000.0f;

bool within(bool seen, uint32_t now, uint32_t then, uint32_t ms) { return seen && now - then <= ms; }
}  // namespace

const char* ContingencyV4::actName(int k)
{
    static const char* const names[kActs] = {"call", "gesture", "face"};
    return k >= 0 && k < kActs ? names[k] : "?";
}

const char* ContingencyV4::answerName(int a)
{
    static const char* const names[kAnswers] = {"pet", "tap", "speech", "near", "motion"};
    return a >= 0 && a < kAnswers ? names[a] : "?";
}

void ContingencyV4::finish(Pending& p)
{
    p.live = false;
    if (!p.around) return;
    n_[p.act] += 1.0f;
    bool any = false;
    for (int a = 0; a < kAnswers; ++a) {
        h_[p.act][a] += p.answered[a] ? 1.0f : 0.0f;
        any = any || p.answered[a];
    }
    if (!any && p.act != kFace) ignored_ += (1.0f - ignored_) * kIgnoreStep;
}

void ContingencyV4::step(const In& in, Out& out, Credit& credit)
{
    const uint32_t now = in.ms;
    const float dt = started_ ? std::clamp(static_cast<float>(static_cast<int32_t>(now - last_ms_)) / 1000.0f, 0.0f, 1.0f)
                              : 0.0f;
    started_ = true;
    last_ms_ = now;
    const float kn = std::exp(-dt / kTauNowS), kb = std::exp(-dt / kTauBgS);
    for (int k = 0; k < kActs; ++k) {
        n_[k] *= kn;
        for (auto& v : h_[k]) v *= kn;
    }
    for (auto& v : bg_n_) v *= kb;
    bg_s_ *= kb;
    pulse_ *= std::exp(-dt / kTauPulseS);
    ignored_ *= std::exp(-dt / kTauIgnoredS);
    credit = Credit{};

    // what the robot did this tick
    bool act[kActs] = {false, false, false};
    if (in.said) {
        act[kCall] = !within(said_seen_, now, last_said_ms_, kActQuietMs);
        last_said_ms_ = now;
        said_seen_ = true;
    }
    if (in.gesture) {
        act[kGesture] = !within(gesture_seen_, now, last_gesture_ms_, kActQuietMs);
        last_gesture_ms_ = now;
        gesture_seen_ = true;
    }
    act[kFace] = face_ >= 0 && in.face != face_;
    face_ = in.face;

    // what came from outside
    bool cond[kAnswers];
    cond[kPet] = in.headEv > 0.0f && in.touchHead < 0.7f;
    cond[kTap] = in.screenEv > 0.0f && in.touchScreen < 0.4f;
    cond[kSpeech] = in.speech > 0.5f;  // the ear already holds back our own voice and its tail (BrainLoop::hear)
    cond[kNear] = in.nearRaw > 150.0f;
    cond[kMotion] = in.motion > 0.15f && std::fabs(in.headYaw - yaw_) < 0.3f;
    yaw_ = in.headYaw;
    bool onset[kAnswers];
    for (int a = 0; a < kAnswers; ++a) {
        onset[a] = cond[a] && !within(seen_[a], now, last_true_[a], kQuietMs[a]);
        if (cond[a]) {
            last_true_[a] = now;
            seen_[a] = true;
        }
    }
    if (onset[kPet] || onset[kTap] || onset[kNear] || in.nearRaw > 50.0f || in.motion > 0.1f) {
        last_around_ms_ = now;
        around_seen_ = true;
    }
    const bool around = within(around_seen_, now, last_around_ms_, kAroundMs);

    if (!in.active) {
        for (auto& p : pending_) p.live = false;
    } else {
        const bool no_act = !within(act_seen_, now, last_act_ms_, kNoActMs);
        if (around && no_act) bg_s_ += dt;
        for (int a = 0; a < kAnswers; ++a) {
            if (!onset[a]) continue;
            // the latest bid (a call, a gesture) in the window, else the latest face (2026-10-28, an audit: the latest act
            // of any kind took it, and on 10-27 a face change after a call took 53 of the 325 answers - the calls had
            // 84: a face is not a bid). The same tick: call < gesture.
            Pending* best = nullptr;
            for (auto& p : pending_) {
                if (!p.live || now - p.ms < kWinLoMs || now - p.ms > kWinHiMs) continue;
                const bool bid = p.act != kFace, best_bid = best && best->act != kFace;
                if (!best || bid > best_bid || (bid == best_bid && (p.ms > best->ms || (p.ms == best->ms && p.act > best->act))))
                    best = &p;
            }
            if (a == kPet || a == kTap || a == kSpeech) ignored_ = 0.0f;  // petted, tapped, talked to: not ignored
            if (best) {
                best->answered[a] = true;
                pulse_ = std::max(pulse_, kWeight[a]);
                if (!credit.any) credit = Credit{true, best->act, static_cast<Answer>(a), now - best->ms};
            } else if (around && no_act) {
                bg_n_[a] += 1.0f;
            }
        }
        for (auto& p : pending_)
            if (p.live && now - p.ms > kWinHiMs) finish(p);
        const bool reaction = within(answer_seen_, now, last_answer_ms_, kReactMs);
        for (int k = 0; k < kActs; ++k) {
            if (!act[k] || reaction) continue;
            Pending* slot = nullptr;
            for (auto& p : pending_)
                if (!p.live) { slot = &p; break; }
            if (!slot) {  // eight acts within 4 s: the oldest is closed early
                slot = &pending_[0];
                for (auto& p : pending_)
                    if (p.ms < slot->ms) slot = &p;
                finish(*slot);
            }
            *slot = Pending{};
            slot->ms = now;
            slot->act = static_cast<Act>(k);
            slot->around = around;
            slot->live = true;
        }
    }
    for (int a = 0; a < kAnswers; ++a)
        if (onset[a]) {
            last_answer_ms_ = now;
            answer_seen_ = true;
        }
    if (act[kCall] || act[kGesture] || act[kFace]) {
        last_act_ms_ = now;
        act_seen_ = true;
    }

    // how much more often than chance each kind of act is answered
    float p0[kAnswers];
    for (int a = 0; a < kAnswers; ++a) p0[a] = 1.0f - std::exp(-bg_n_[a] / (bg_s_ + kBgPriorS) * kWindowS);
    float best = 0.0f;
    for (int k = 0; k < kActs; ++k) {
        float c = 0.0f;
        for (int a = 0; a < kAnswers; ++a) c += kWeight[a] * ((h_[k][a] + p0[a] * kPrior) / (n_[k] + kPrior) - p0[a]);
        c = std::clamp(c, -1.0f, 1.0f);
        out[v4::CT_CALL + k] = c;
        best = std::max(best, c);
    }
    out[v4::CT_AGENCY] = std::min(best, 1.0f);
    out[v4::CT_ANSWERED] = pulse_;
    out[v4::CT_IGNORED] = ignored_;
}
