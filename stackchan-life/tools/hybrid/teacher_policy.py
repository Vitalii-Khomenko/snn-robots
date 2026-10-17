"""The designed behaviour the policy GRU imitates (stage 1 of the concept's
training strategy), written as a pure function of the policy input vector.

Because it only looks at the inputs - perception, body, context, noise
pacemakers and the spiking core's state - it can label ANY logged rollout
(DAgger): run the learner in the simulator, relabel what it saw with this
teacher, retrain. Randomness is never drawn here: the noise pacemakers are
part of the input, so "choose a random place to look" is a deterministic
function the network can learn from what it sees.

Training metadata only: the robot runs the trained GRU, never this code.
"""
import numpy as np

from spec import ACTION, ACTION_SIGNED, pin

A = {n: i for i, n in enumerate(ACTION)}
# rear LED colour per face (what BodyExecutor::showEmotion used to set in code)
LED_COLOUR = {"normal": (0, 25, 30), "bored": (0, 25, 30), "sleepy": (0, 0, 14),
              **{n: (60, 30, 0) for n in ("happy", "very_happy", "laughing", "love", "shy", "wink", "cool")},
              "sad": (0, 10, 45), "crying": (0, 10, 45),
              **{n: (50, 0, 50) for n in ("surprised", "scared", "dizzy", "confused")},
              "angry": (60, 0, 0), "annoyed": (60, 0, 0)}
EMOS = [n[4:] for n in ACTION if n.startswith('emo_')]
YAW_RANGE, PITCH_MIN, PITCH_MAX = 100.0, -20.0, 35.0
HALF_VIEW_YAW, HALF_VIEW_PITCH = 30.0, 22.0


def sig(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(k * x, -30, 30)))


def deg_to_yaw(deg):
    return np.clip(deg / YAW_RANGE, -1, 1)


def deg_to_pitch(deg):
    return np.clip((deg - PITCH_MIN) / (PITCH_MAX - PITCH_MIN) * 2 - 1, -1, 1)


class Memory:
    """What the teacher remembers along an episode (the rhythm the executor used to impose by hand): when the head last
    stopped. label() runs it over a whole sequence; a rollout keeps one per episode and labels tick by
    tick. The policy GRU learns it from its own memory - it sees its head speed and its voice on the bus."""

    def __init__(self):
        self.since_move = 10 ** 6   # ticks since the head last stopped
        self.moving = False
        self.shy = 0                 # ticks left of the shy face after a tap on the screen


GLANCE_GAP = 160                     # ticks (8 s) between small glances while nobody is followed


def label(X, memory=None):
    """policy inputs [T, 100] -> target actions [T, 51] (post-activation space)."""
    c = lambda n: X[:, pin(n)]
    T = len(X)
    Y = np.zeros((T, len(ACTION)), np.float32)
    a = lambda n: Y[:, A[n]]

    yaw_deg, pitch_deg = c("yaw") * 100.0, c("pitch") * 35.0
    sleep, fatigue, arousal = c("sleep"), c("fatigue"), c("arousal")
    # boredom and social need: the exact slow clocks (HybridBrain::updateSlow), not the spiking core's weak versions
    curiosity, boredom, annoy, social = c("curiosity"), c("slow_boredom"), c("annoyance"), c("slow_social")
    playful, surprise = c("playful"), c("surprise")
    att_s, att_x, att_y = c("att_strength"), c("att_x"), c("att_y")
    tr_touch, tr_person, tr_sound, tr_reward = c("tr_touch"), c("tr_person"), c("tr_sound"), c("tr_reward")
    hab_s = c("hab_sound")
    wx, wy, ws, jit = c("wander_x"), c("wander_y"), c("wander_slow"), c("jitter")
    night = c("night")
    asleep = sig(sleep - 0.55, 14)  # 0..1: the robot is (going) to sleep
    # what it learned by itself (AssocMemory): this situation usually comes before being petted / handled roughly
    anticip = sig(c("expect_reward") - 0.35, 14) * (1 - asleep)
    wary = sig(c("expect_threat") - 0.35, 14) * (1 - asleep)

    # ------------------------------------------------------------ events
    loud = np.maximum.reduce([c("loud"), 0.9 * c("clap"), 0.7 * (c("sharp") > 0.5) * c("sharp")])
    hab_eff = np.clip(hab_s - 0.3, 0, 1) / 0.7  # the readout idles near 0.3
    startle_ev = np.clip(np.maximum.reduce([loud * (1 - 0.7 * hab_eff), 1.1 * c("loom"), 1.2 * c("shake"), 0.8 * c("tilt")]) - 0.3, 0, 1) / 0.7
    startle = (startle_ev > 0.55).astype(np.float32) * (1 - asleep * 0.3)
    touch = np.maximum(c("touch_head"), c("touch_screen"))
    petted = np.clip(tr_touch - 0.25, 0, 1) / 0.75 * (annoy < 0.45)
    irritated = sig(annoy - 0.45, 12)
    anticip = anticip * (1 - irritated)
    greet = np.clip(c("owner_arrived") + 0.6 * (c("face") > 0.6) * (social > 0.3) * (tr_person < 0.5), 0, 1)

    # ------------------------------------------------------------ head
    # attention: where the spiking core says the interesting thing is (image
    # coordinates of the camera: +x right, +y down; the view is +-30 / +-22 deg)
    # a person in view is followed directly from what the eyes report (the
    # spiking attention would wear out on a face that stays in the same place)
    person_track = ((c("person") > 0.4) & (c("target_conf") > 0.4)).astype(np.float32)
    att_x = np.where(person_track > 0.5, c("target_x"), att_x)
    att_y = np.where(person_track > 0.5, c("target_y"), att_y)
    att_s = np.where(person_track > 0.5, np.maximum(att_s, 0.7), att_s)
    attend = np.maximum(sig(att_s - 0.32, 18), person_track) * (1 - asleep)
    # a sound just heard: its direction cannot be told (two microphones 5 cm apart), so the robot looks around for the
    # source - a place in the room the slow pacemaker picks (absolute, so the turn ends there); the eyes take over if a
    # person shows up
    look_yaw = np.clip(wx * 80.0, -90.0, 90.0)
    sound_orient = ((tr_sound > 0.3) & (person_track < 0.5) & (np.abs(look_yaw - yaw_deg) > 15.0)).astype(np.float32) * (1 - asleep)
    attend = np.maximum(attend, sound_orient)
    track_yaw = np.where(sound_orient > 0.5, look_yaw, yaw_deg + att_x * HALF_VIEW_YAW * 0.9)
    track_pitch = np.where(sound_orient > 0.5, 10.0, pitch_deg - att_y * HALF_VIEW_PITCH * 0.9)
    # exploring: a slow scan of the room, then glances (pacemaker driven)
    explore_p = (np.clip(1.2 * boredom + 0.3 * curiosity - 0.3 * arousal - 0.6 * att_s + 0.22, 0, 1) * (1 - asleep)  # +0.22: a restless head even when calm
                 * (1 - 0.85 * night) * (1 - np.clip(1.5 * sleep, 0, 1)))
    # glances, not a creep: the scan target moves in 25 degree steps, so the
    # head is still most of the time (a constantly moving head hears and feels
    # nothing but itself)
    scan_yaw = np.round((0.75 * ws + 0.25 * wx) * 85.0 / 25.0) * 25.0  # a glance every minute or so, not a creep
    scan_pitch = 8.0 + np.round(wy * 12.0 / 6.0) * 6.0
    rest_yaw, rest_pitch = yaw_deg, pitch_deg
    yaw_t = np.where(attend > 0.5, track_yaw, np.where(explore_p > 0.25, scan_yaw, rest_yaw))
    pitch_t = np.where(attend > 0.5, track_pitch, np.where(explore_p > 0.25, scan_pitch, rest_pitch))
    # being petted: lean into the hand; irritated: turn away from it
    pitch_t = pitch_t - 12.0 * petted * (1 - irritated)
    yaw_t = yaw_t + 25.0 * irritated * np.sign(wx + 1e-3) * (tr_touch > 0.3)
    # sleep: head down, still
    pitch_t = np.where(asleep > 0.5, -17.0, pitch_t)
    yaw_t = np.where(asleep > 0.5, yaw_deg, yaw_t)
    # expecting rough handling: duck a little and freeze
    pitch_t = pitch_t - 8.0 * wary
    a("yaw")[:] = deg_to_yaw(np.clip(yaw_t, -YAW_RANGE, YAW_RANGE))
    a("pitch")[:] = deg_to_pitch(np.clip(pitch_t, PITCH_MIN, PITCH_MAX))
    aligned = (np.abs(att_x) < 0.12) & (np.abs(att_y) < 0.15) & (attend > 0.5)
    a("hold")[:] = np.maximum(np.where((asleep > 0.5) | aligned, 1.0, np.where(explore_p > 0.25, 0.0, 0.55)), wary)
    a("explore")[:] = explore_p
    speed = 0.28 + 0.22 * attend * (0.5 + att_s) + 0.55 * startle + 0.1 * explore_p
    a("speed")[:] = np.clip(speed * (1 - 0.35 * fatigue), 0.1, 1.0)
    a("vigor")[:] = np.clip(0.2 + 0.7 * arousal + 0.4 * startle, 0, 1)
    a("startle")[:] = startle

    # ------------------------------------------------------------ face
    mood = (0.12 + 0.55 * playful + 0.5 * tr_reward + 0.3 * (c("person") > 0.5) * (1 - annoy) - 0.95 * annoy
            - 0.3 * boredom * (1 - playful) - 0.25 * fatigue + 0.25 * greet + 0.35 * anticip - 0.4 * wary)
    face_v = np.clip(mood - 0.5 * startle * (1 - playful), -1, 1)
    face_a = np.clip(arousal + 0.45 * startle, 0, 1)
    a("face_valence")[:] = face_v
    a("face_arousal")[:] = face_a
    a("lids")[:] = np.clip(1.15 * sleep + 0.25 * fatigue * night, 0, 1)
    a("gaze_x")[:] = np.clip(att_x * 0.9 * attend + (1 - attend) * wx * 0.5, -1, 1)
    a("gaze_y")[:] = np.clip(att_y * 0.7 * attend + (1 - attend) * wy * 0.3, -1, 1)
    a("blink")[:] = sig(-jit - 0.45, 14) * (1 - 0.7 * asleep) + 0.8 * (sleep > 0.3) * sig(jit - 0.2, 12)
    # how lively the rear glow is: arousal and play, a flare on a startle or a touch, eagerness when it expects petting
    a("led_activity")[:] = np.clip(0.2 + 0.5 * arousal + 0.2 * playful + 0.45 * startle + 0.35 * touch + 0.4 * anticip, 0, 1) * (1 - 0.7 * asleep)

    # ------------------------------------------------------------ voice
    quiet = (1 - asleep)
    # when to call (pulses from the pacemakers: the pulse itself is deterministic given the inputs)
    pulse_a = sig(jit - 0.5, 18)                       # ~ one call a minute at social 0.6
    pulse_b = sig(wx - 0.85, 25)
    seek = (social > 0.35) * (tr_person < 0.25) * (arousal < 0.6) * (1 - night) * sig(jit - 0.75, 18) * sig(3 * ws + 0.2, 4)  # lonely: calls for company in bursts
    reply = (c("speech") > 0.5) * (tr_person > 0.3) * pulse_a * (social > 0.25)
    pet_purr = petted * sig(jit - 0.1, 12) * (1 - irritated)
    grumble = irritated * sig(jit - 0.0, 10) * (tr_touch > 0.3)
    frighten = startle * (1 - 0.5 * asleep)
    greet_call = greet * (tr_reward > 0.2)
    curious_call = (curiosity > 0.55) * (att_s > 0.4) * pulse_b * (1 - night)
    # every touch gets an answer, at once: a short happy murmur, or a grumble if it is too much
    touch_reply = (touch > 0.5).astype(np.float32)
    # ... and every clear bang too (a clap, a knock): the startle fades with habituation, the answer does not
    bang_reply = (loud > 0.6).astype(np.float32) * (1 - 0.5 * asleep)
    # a conversation nearby (people talking, also on TV): the robot tries to take part. While voices are heard it murmurs now and then;
    # in the 0.5-3 s after a phrase ended (the sound trace has faded to 0.12-0.7) it answers in the pause, like a turn in a dialogue
    talk = (c("speech") > 0.45) * (loud < 0.5) * (1 - asleep) * (1 - night)
    in_pause = sig((tr_sound - 0.12) * 15, 1) * sig((0.7 - tr_sound) * 15, 1) * (c("speech") < 0.3) * (loud < 0.5) * (1 - asleep) * (1 - night)
    converse_ok = (1 - irritated) * (1 - 0.6 * annoy)
    conv_reply = np.maximum(0.9 * in_pause * sig(jit + 0.6, 8), 0.75 * talk * sig(jit - 0.6, 14)) * converse_ok
    # idle chatter: babbling to itself while awake, curious and not alone in the dark (~5 calls a minute)
    # it never falls silent when alone: the urge to babble follows curiosity AND boredom smoothly (no cliff at one value)
    chat_drive = np.clip(0.40 + 1.0 * curiosity, 0, 1)
    chatter = (arousal > 0.12) * (1 - night) * (1 - asleep) * sig(jit - (0.55 - 0.4 * chat_drive), 18) * sig(3 * ws + 0.5, 4) * (1 - 0.3 * boredom)
    # it expects to be petted (it learned what comes before): an eager chirp now and then, asking for it
    eager_call = anticip * (tr_touch < 0.2) * sig(jit - 0.35, 14)
    act = np.maximum.reduce([seek, reply, pet_purr, grumble, 0.9 * frighten, greet_call, curious_call,
                             0.95 * touch_reply, 0.85 * bang_reply, 0.9 * chatter, conv_reply, 0.9 * eager_call])
    act = act * np.where(touch_reply > 0.5, 1.0, 1.0 - 0.7 * wary)  # wary: quiet (but a touch is still answered)
    a("sound_act")[:] = np.clip(act * np.where(touch_reply > 0.5, 1.0 - 0.5 * asleep, quiet), 0, 1)
    a("sound_arousal")[:] = np.clip(arousal + 0.3 * startle, 0, 1)
    a("sound_valence")[:] = np.clip(face_v, -1, 1)
    # what the call is like
    pitch = 0.38 + 0.3 * arousal + 0.12 * np.maximum(face_v, 0) + 0.1 * curiosity - 0.18 * irritated - 0.15 * sleep + 0.12 * startle
    a("voice_register")[:] = np.clip(pitch, 0.05, 0.95)
    glide = 0.5 * curiosity * (att_s > 0.3) + 0.3 * playful + 0.5 * seek - 0.55 * irritated - 0.35 * sleep - 0.2 * boredom
    a("voice_slope")[:] = np.clip(glide, -1, 1)
    a("voice_expr")[:] = np.clip(0.15 + 0.45 * playful + 0.2 * (1 - np.abs(face_v)), 0, 1)
    a("voice_length")[:] = np.clip(0.22 + 0.4 * sleep + 0.1 * boredom - 0.1 * startle - 0.12 * touch_reply, 0, 1)
    a("voice_rate")[:] = np.clip(0.25 + 0.45 * playful + 0.25 * arousal, 0, 1)
    fam = np.stack([
        0.2 + 0.5 * sleep + 0.4 * boredom,                       # whistle: calm, long, smooth
        0.3 + 0.6 * curiosity * (att_s > 0.3) + 0.5 * startle + 0.3 * seek + 0.6 * eager_call,  # pip / ask: attentive, quick
        0.1 + 0.8 * playful + 0.4 * greet,              # trill: lively
        0.05 + 0.9 * irritated + 0.3 * startle * (face_v < 0)], 1)             # grumble: low, rough
    fam[:, 1] += 0.9 * conv_reply            # pip / ask: attentive, a turn in the dialogue
    fam[:, 0] += 0.9 * touch_reply * (1 - irritated)
    fam[:, 2] += 0.8 * touch_reply * (1 - irritated)
    fam[:, 3] += 1.2 * touch_reply * irritated
    fam = fam / np.maximum(fam.max(1, keepdims=True), 1e-3)                    # winner takes the call
    fam = np.where(fam > 0.85, 1.0, 0.25 * fam)
    for i, n in enumerate(("style_calm", "style_ask", "style_play", "style_grumble")):
        a(n)[:] = fam[:, i]

    # ------------------------------------------------------------ gestures
    # an invitation to play goes to SOMEBODY: a person around (in view now or a moment ago, or the owner's phone at home) who is
    # not playing with it (no touch, nobody talking to it) while it feels the need for company; the slow mood decides the
    # moment (it used to need nobody in view - it nodded at an empty room and never at the owner sitting next to it)
    someone = np.maximum((tr_person > 0.2).astype(np.float32), c("owner_near"))
    invite_gate = (sig(social - 0.38, 25) * someone * (tr_touch < 0.2) * (c("speech") < 0.3) * (playful > 0.1)
                   * (arousal < 0.75) * (1 - asleep) * (1 - night))
    a("invite")[:] = invite_gate * sig(ws - 0.6, 14)  # one nod series per rising edge; the executor allows one every 4 min

    # ------------------------------------------------------------ which face
    # one-hot over the 21 faces; the first matching rule wins (what BodyExecutor::applyFace used to decide in code)
    # a shaken / lifted / turned-over body is felt for ~2 s: the face and the state answer it
    shake_l, tilt_l = c("shake"), c("tilt")
    upside = (np.abs(c("imu_roll")) > 0.6) | (np.abs(c("imu_pitch")) > 0.7)
    ah = np.maximum(a("face_arousal"), 0.85 * np.maximum(shake_l, 0.8 * tilt_l))
    a("face_arousal")[:] = ah
    a("face_valence")[:] = np.minimum(face_v, face_v - 0.5 * np.maximum(shake_l, tilt_l))
    v = a("face_valence")
    surprise = (c("surprise") > 0.45) | (startle > 0.5)
    shaken = c("shake") > 0.5
    inv = a("invite") > 0.55
    emo = np.full(T, EMOS.index("normal"))
    rules = [
        (c("battery") < 0.08, "sick"),
        (a("lids") > 0.5, "sleepy"),
        ((tilt_l > 0.5) & upside, "scared"),
        (shaken, "dizzy"),
        (tilt_l > 0.5, "surprised"),
        (inv & (jit > 0), "wink"),
        (inv, "very_happy"),
        (surprise & (v < -0.2), "scared"),
        (surprise, "surprised"),
        # being stroked (before the expectations: while it is stroked, the expectation of petting is high too, and it
        # took the face)
        # being stroked while not cross: love (petting raises arousal, so it can not require a calm moment); a tap on the
        # screen, its face: shy
        ((petted > 0.3) & (touch > 0.5) & (irritated < 0.5) & (v > 0.0), "love"),
        ((petted > 0.3) & (tr_touch > 0.4) & (irritated < 0.5) & (v > 0.2), "love"),
        ((c("touch_screen") > 0.5) & (v > -0.1) & (irritated < 0.5), "shy"),
        # content after somebody played with it and left it be: cool (before the expectations: right after play the
        # expectation of petting is still high and took the face)
        ((tr_reward > 0.3) & (tr_touch < 0.15) & (ah < 0.6) & (v > 0.2) & (irritated < 0.3), "cool"),
        # what it learned: rough handling usually comes now - afraid; petting usually comes now - delighted
        ((wary > 0.5) & (anticip < 0.5), "scared"),
        ((anticip > 0.5) & (ah > 0.45), "very_happy"),
        (anticip > 0.5, "happy"),
        ((v > 0.55) & (ah < 0.35), "love"),
        ((v > 0.3) & (ah > 0.65), "laughing"),
        ((v > 0.3) & (ah > 0.35), "very_happy"),
        (v > 0.3, "happy"),
        ((v < -0.35) & (ah > 0.65), "scared"),
        ((v < -0.6) & (ah > 0.35), "angry"),
        ((irritated > 0.7) & (touch > 0.5), "angry"),   # touched again while already cross
        ((v < -0.35) & (ah > 0.35), "annoyed"),
        ((v < -0.6), "crying"),
        ((v < -0.35), "sad"),
        ((talk > 0.5) & (annoy < 0.35), "curious"),         # listening
        ((in_pause > 0.5) & (annoy < 0.35), "happy"),         # its turn
        (c("pe_motor") > 0.3, "confused"),
        # it heard something, looks around for it and finds nobody: confused
        ((sound_orient > 0.5) & (person_track < 0.5) & (annoy < 0.35), "confused"),
        (ah > 0.55, "curious"),
        ((c("nov_aud") > 0.08) & (ah < 0.4), "thinking"),
        # an inner life while nobody is there: moods that follow the slow drives and the pacemakers
        ((boredom > 0.55) & (ah < 0.4) & (night < 0.5), "bored"),
        ((curiosity > 0.36) & (ah < 0.55), "curious"),
        # a long loneliness: crying; a shorter one: sad
        ((social > 0.85) & (boredom > 0.6) & (tr_person < 0.1) & (night < 0.5) & (annoy < 0.3), "crying"),
        ((social > 0.42) & (tr_person < 0.1) & (night < 0.5) & (annoy < 0.3) & (ah < 0.45), "sad"),
        ((playful > 0.30) & (annoy < 0.3) & (night < 0.5), "happy"),
        ((wx > 0.7) & (ah < 0.45) & (night < 0.5), "thinking"),   # a thought passes (the 6 s pacemaker, not the 0.6 s one: no flicker)
    ]
    decided = np.zeros(T, bool)
    for cond, name in rules:
        pick_ = cond & ~decided
        emo[pick_] = EMOS.index(name)
        decided |= pick_
    for i, n in enumerate(EMOS):
        a("emo_" + n)[:] = (emo == i).astype(np.float32)
    # the rear glow's colour goes with the face (0..1 of the LEDs' working range, 64 of 255)
    col = np.array([LED_COLOUR.get(n, (35, 35, 35)) for n in EMOS], np.float32)[emo] / 64.0
    a("led_r")[:], a("led_g")[:], a("led_b")[:] = col[:, 0], col[:, 1], col[:, 2]
    # ---- the rhythm (it needs the past): the teacher's memory along the sequence
    mem = memory if memory is not None else Memory()
    speed = np.hypot(c("yaw_vel"), c("pitch_vel")) * 180.0
    follow = (c("person") > 0.4) | ((c("own_motion") < 0.5) &
                                    (np.maximum.reduce([c("speech"), c("clap"), c("sharp"), c("loud")]) > 0.35))
    big = np.hypot(yaw_t - yaw_deg, pitch_t - pitch_deg) >= 40.0
    hold = a("hold")
    # watching somebody who just sits there (no touch, no talking): now and then a glance away and back (the audit of
    # 2026-10-04: the head stared at the owner at the PC for hours). Triggered by the 6 s pacemaker (no long counter: the
    # GRU did not learn a 25-55 s count), 40 deg aside of where the person sits in the room (absolute: the turn ends there).
    idle_watch = (person_track > 0.5) & (tr_touch < 0.2) & (c("speech") < 0.3) & (asleep < 0.5) & (sound_orient < 0.5)
    person_world = yaw_deg + c("target_x") * HALF_VIEW_YAW
    away = idle_watch & (wx > 0.8)
    away_yaw = np.clip(person_world + np.where(ws > 0, 40.0, -40.0), -YAW_RANGE, YAW_RANGE)
    a("yaw")[:] = np.where(away, deg_to_yaw(away_yaw), a("yaw"))
    a("gaze_x")[:] = np.where(away, 0.0, a("gaze_x"))
    hold[:] = np.where(away, 0.0, hold)
    # a tap on the screen, its face: shy for 2 s (the tap itself is a 3-tick pulse, shorter than the 4 ticks a face needs)
    tap = c("touch_screen") > 0.5
    shy_ok = (face_v > -0.1) & (irritated < 0.5)
    emo_cols = [A["emo_" + n] for n in EMOS]
    for t in range(T):
        mem.shy = 40 if tap[t] else max(0, mem.shy - 1)
        if mem.shy > 0 and shy_ok[t]:
            Y[t, emo_cols] = 0.0
            Y[t, A["emo_shy"]] = 1.0
        moving = speed[t] > 8.0
        if mem.moving and not moving:
            mem.since_move = 0
        mem.moving = moving
        mem.since_move += 1
        # a small glance while nobody is followed waits 8 s after the last movement (a restless head is deaf: its servos are loud)
        if not follow[t] and not big[t] and mem.since_move < GLANCE_GAP:
            hold[t] = 1.0
    return np.clip(Y, [-1 if n in ACTION_SIGNED else 0 for n in ACTION], 1).astype(np.float32)
