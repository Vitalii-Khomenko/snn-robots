#pragma once
// Build-time switches for recording real data from the robot (all off in a normal build).
// Vision recording (tools/hybrid/record_vision.py): the camera task prints the Vision NN's input tensor
// ("vframe:" lines, ~3.5 per second). Free = the robot behaves as usual (scans the room: empty-room
// negatives from every direction, but the head turns rarely); Still = the head is parked facing forward (a person stands in view);
// Sweep = the head sweeps slowly over the whole range, yaw +-90 deg in 70 s and pitch -6..22 deg in 29 s (empty rooms from every direction).
enum class VisionRec { Off, Free, Still, Sweep };
inline constexpr VisionRec kVisionRec = VisionRec::Off;

// Microphone recording (tools/hybrid/record_audio.py): the Auditory NN's input, 20 lines a second ("audio:" lines).
inline constexpr bool kAudioRecording = false;
// Clean sound recordings: the head is parked facing forward after the first command and the robot stays silent, so that
// neither the servos nor its own voice end up in a clap / speech session.
inline constexpr bool kParkHead = false;
