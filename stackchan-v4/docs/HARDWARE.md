# Hardware

M5Stack **CoreS3** (ESP32-S3, 16 MB flash, 8 MB PSRAM, 320×240 touch LCD,
GC0308 camera, two microphones, AW88298 speaker amplifier, BMI270 IMU,
LTR-553 proximity / light, AXP2101 power management) on the **StackChan
body**: two Feetech SCSCL servos (yaw, pitch), a Si12T head touch pad, a
PY32 IO expander that switches servo power, LEDs (12 RGB565 LEDs, written as one 24-byte burst plus a refresh, at most 25 times a second).

## Wiring and addresses

| Part | Connection |
|---|---|
| SCS servo bus | UART1, 1 Mbaud, TX GPIO6, RX GPIO7; yaw ID 1 (zero 460), pitch ID 2 (zero 620); 0.3125° per unit |
| Servo power (VM EN) | PY32 IO expander at I²C `0x6F`, pin 0 |
| Bus / boost enables | AW9523 BUS_OUT_EN, BOOST_EN (same as the official M5 firmware) |
| Speaker | AW88298 via the BSP codec, always opened at 24000 Hz with `pa_gain = 1` (the BSP default gain is ~14 dB too quiet and its 22050 Hz rate never wakes the amp); the codec is opened once and left open (reopening per sound glitched the shared I²S bus and the microphones); the volume is written to the amplifier only when it changes (it was an I²C write on every 20 ms chunk); the playback rate was measured on 2026-09-30 (1.5 s of silence: 23,995 samples/s consumed of 24,000 - the mono stream on the stereo bus shared with the microphones plays at the right speed); loudness is set by `droid::Synth::level` (0.17 of full scale since 2026-10-02: peaks ~5k, RMS ~2.5k, at the owner's request after the voice had been made louder twice; the amplifier stayed at 90 until 2026-10-25; since then the owner sets the level 0-10 with the speaker button on the screen, ARCHITECTURE.md); the voice is rendered and written by its own audio task (core 1, priority 5), because writes from the main loop stuttered whenever a brain tick ran long |
| Microphones | stereo I²S (shared bus with the speaker), own task (core 1, priority 3) running the 16-band filter bank |
| IMU | BMI270 (in the head: head pitch rotates gravity) |
| Touch | Si12T head pad; LCD touch via LVGL `LV_EVENT_PRESSED` (an atomic flag, handled in the main loop) |
| Proximity / light | LTR-553 |
| Power | AXP2101 (battery level, charging direction; "USB present but not charging" cannot be told apart) |
| Camera | GC0308, QQVGA RGB565, ~7 fps (143 ms per frame), one frame buffer (a one-off `cam_hal: FB-SIZE: 30720 != 38400` warning at start-up is harmless); the task runs the vision front end (~3 ms per frame: brain v4's motion sense and the brightest spot) on core 1 at priority 2, below the microphone |

## Servos

**Position mode.** SCSCL servos with angle limits 0/0 are in motor (PWM)
mode, where the "goal time" field is read as motor power (a goal time of 50
gave 5 % power: packets accepted, head barely moving). `MotionEngine`
writes limits 0…1000 once after every power-on, then enables torque
(refreshed every 3 s, `kTorqueRefreshMs`).

**No feedback exists on this board.** The RX line (GPIO7) never toggles, not
even an echo; receiving on the open-drain TX line (GPIO6) shows only our own
packet (`ff ff 01 02 01 fb`). The body board apparently has a one-way buffer
ESP32 → bus. Position, load and error registers cannot be read; `bus_.Level
= 0` (no ACK wait). Only the gyro tells whether the head really moved.
(The bring-up probe that found this was removed from the firmware on 2026-10-03.)

> **Careful:** `gpio_set_direction()` in ESP-IDF 5 reconnects the pad to
> plain GPIO and cuts UART TX off: a bus probe once made the servos deaf
> until it was removed. `SCSerial::setWiring()` only toggles open drain
> (`gpio_od_enable/disable`).

**Stalls come from the packet rate.** The servos occasionally hang (hold
torque, ignore positions) until their power is cycled. Measured watchdog
repowers:

| Position packets | Stall |
|---|---|
| 50/s broadcast (SYNC_WRITE) | every ~24 s |
| 50/s addressed + 1 ms gap | every ~36 s |
| 20/s | every ~75 s |
| **10/s, goal time 120 ms** (current) | **1 in 5 min, then 0 in 8 min** |

So positions go out every 100 ms (`kUpdateMs`), addressed with a 1 ms gap
(`kSyncWrite = false`), each packet written to the UART in one piece; the
servo glides between points and the speed profile shapes the motion. Do not
raise the packet rate. The spring is integrated in ≤20 ms substeps (a 50 ms
step made the stiff spring blow up and slammed the head to its limit).

Since 2026-10-20 positions go out sparsely (`kSparseWrites`: one packet per move with its destination and time, one
more when it settles): ~15 packets/min instead of 72, but the power cycles fell only 20 % (1036 -> 832 a day) and
the camera still showed the head standing in ~58 % of 25-50° turns. Both servos hang together (336 windows vs 7 with
one), the frozen head is stiff to the hand (torque on), VM EN stays on, pose and speed do not matter. Since
2026-10-22 positions went as one broadcast SYNC_WRITE (`kSyncWrite`: no servo answers it, so no reply can collide):
no fewer stalls, and the same evening the whole board froze 3 times, each 30-130 s after a servo power cycle -
turned off again the same day.

**2026-10-24, brain v4:** the livelier main network sent every small gaze correction: 2600-4900 packets an hour
(45-80 a minute), 60-100 servo power cycles an hour, and the gyro saw only 26-36 % of the commanded rotation - the owner
saw a head that hardly moved. The stall rate per move stayed the same (1.1-1.9 per 100 moves); what grew was the number
of moves. The executor now sends a gaze move at most every 2.5 s and only for a visible change (8° / 5°).

**Servo watchdog** (`RobotRuntime::superviseServos`): if two consecutive
2 s windows had ≥10° commanded and <1.5° real rotation (gyro), it switches
VM EN off for 500 ms and on again and resyncs the motion model; at most
once per 10 s. One window was not enough (slow drift below the gyro's
~3°/s floor gave a false repower). A soft recovery (mode + torque +
position again) helped in only 2 of ~14 stalls and was removed. Log:
`servo rail power cycle (...) py32=0x41 vm_en out=1`. Every boot also
power-cycles the servos once.

**2026-10-25, sparse moves:** with one gaze move per >= 2.5 s a hung head rarely got 10° in two windows in a row: on
2026-10-24 the gyro saw nothing in 61 % of the 5 s windows with commanded motion, in runs of up to 3 minutes, and the
watchdog did not act. Now the commanded degrees the gyro has not confirmed also add up (windows of >= 6° commanded and
< 1.5° real; any window with >= 3° real resets the sum): 25° unconfirmed count as a stall. Watch `servo_repowers`
against `real_rotation`: a power cycle after which the head turns at once was a real stall; many cycles with the head
turning fine in between would be the false ones of 2026-10-03 (below).

**2026-10-25, every mood moving:** with a head motion for every emotion and gaze moves every 2.5 s the robot sent
50-120 packets a minute; right after the boot the servos hung again 12 s after each power cycle, the series rule
then waited 1-2 minutes and the head stood still. The executor now sends a gaze move at most every 4 s and only for
>= 12° yaw / 8° pitch (simulator: 7-11 moves a minute instead of 12-18).

**2026-10-25, power-gated moves** (the owner's idea, `MotionEngine::kGatedPower`): the servo rail (VM EN) is on
only while the head moves. A move switches it on and waits 200 ms for the servos to start (`kRailBootMs`; the
position-mode limits, torque and both positions go out again); the rail goes off 600 ms after the servo should have
arrived (`kRailHoldMs`) and stays off at least 300 ms (`kRailMinOffMs`). Gesture keyframes come closer together than
that, so a nod or shake keeps the rail on. A hung servo now costs one move instead of a dead head until the watchdog
notices (and, in a series, minutes of waiting), and a still head draws no current. The watchdog in this mode just
switches the rail off (the next move switches it on); the screen is not re-initialised after it. The rail switching
(`railSwitching`, 400 ms) raises the microphone threshold and counts as own motion for the shake channel and the head
pad, not for the camera (as own motion it cost ~1 s of sight after every move). The simulator runs the same timing
without a switch (`setRailSwitch`). First 11
minutes (12:53-13:04): 8 switch-ons a minute, the rail on 30 % of the time, the gyro saw 85 % of the commanded
rotation (30-38 % between 08:00 and 11:00 the same day), dead windows 12 % (57-66 %), no watchdog power cycle (~55 an
hour in the morning), VBUS never under 4.74 V. Open: whether the head sinks under its own weight with the rail off
(`servo_rail=ons/ms` in the `power:` line).

**2026-10-27, the head's own jerks** (the owner: "sometimes it jerks its head hard to one side and back"). The diary's
head angles are the commanded model's, which never moves faster than ~40°/s. Only the gyro sees the servo running off
by itself, in the 5 s `stats:` windows: a peak over 300°/s with 40° more real rotation than commanded (`jerks.py`).
- **How often:** 7-13 such windows a day until 2026-10-25, then 103 on 10-26 and 59 by 17:00 on 10-27. About 2 % of
  the rail switch-ons, none with the rail off. Each comes with a clipped "loud sound" (`peak=32768 own-motion`): the
  head hitting its stop.
- **Two causes, both at a switch-on:**
  - **Motor mode.** The limits went out 200 ms after power-on with the torque right behind them. A servo still
    starting missed them and stayed in motor mode (its EEPROM holds 0/0 from the official firmware). It took the
    torque and read the goal time as motor power. Since the eased waypoints (10-25 evening) that time is ~500, half
    power instead of the old creep - hence the jump on 10-26. 10-25 had 0.1 % jerks but 12.8 % "hangs": the creep.
  - **A goal far from the real head.** The board cannot hear the servos (the RX line is dead). After a hang the model
    moved on, a limp head sags, a hand turns it; the first goal at the waypoint's time was then a dash. Example: 17:30,
    170° at 426°/s right after two hangs.
- **The fixes:**
  - the limits go out twice;
  - then the goal where the model is, and only then the torque;
  - the first packet after a switch-on takes at least 0.6 s (`kFirstMoveMs`), 1.5 s after the watchdog cut the
    power (`kFirstAfterCutMs`).
- **Diagnostics:** every switch-on now logs the gyro's peak of the next 1.5 s when it is over 300°/s
  (`servos: a jerk of N deg/s ... (after a hang | no hang)`, `servo_jerks=` in `stats:`).
- **Hangs** (the watchdog's "commanded motion but the gyro saw none") went from 0-3 an hour on 10-26 by day to ~20
  an hour from that evening on, when the glances slowed to a 12°/s peak (19:42). A third of them came within 8 s of a
  glance's start, 6 times the chance. Glances are back to a 20°/s peak (`kGlanceSpeedDegS` 40), 40-70°, as a test.
- **After the first fix** (17:15-17:36): 1 jerk window in 56 switch-ons. **After the second** (from 17:36): the only
  jerk came right after a hang (340°/s logged).
- **The hour that followed** (18:01-19:00, glances at 20°/s): 5 jerk windows in 245 switch-ons (2.0 %), 11 logged at
  a switch-on, 8 of them with no hang; 21 hangs (8.6 %), the same as with the 12°/s glances. So the glance speed
  does not cause the hangs: a glance's start is simply where the rail comes on. The 20°/s glances are kept.
- **Position mode in the EEPROM** (19:06): two copies of the limits 4 ms apart are lost together when a servo is not
  up yet. Position mode now also goes into the servos' EEPROM (unlock, limits 0/1000, lock: `eeprom_writes_`, at
  the first `kEepromWrites` = 3 switch-ons of each boot, ~10 EEPROM writes a day). A servo that misses every
  packet after a power-on then still wakes in position mode. The hour after: 2 jerk windows in 271 switch-ons
  (0.7 %), 4 logged at a switch-on (2 after a hang); 17 hangs (6.3 %). Fewer, not none; a whole day will tell.
- **Still open: the hangs themselves**, ~20 an hour since 10-26 evening (0-3 that day). Nothing in the firmware
  timing explains them so far; the supply (`vbus_min` dips to ~4.8 V at a switch-on) and the cable are next.

**2026-10-25 evening, eased moves** (the owner, sitting next to it: "a short whizz of the motors every few seconds -
it should move slowly and smoothly"): the diary showed 11 moves a minute, one every ~4.4 s, each ~14° in ~1.3 s, and an
SCS servo given one destination and a time runs there at one constant speed - every move started and stopped with a
jerk. A gaze turn that takes 0.8 s or more (`MotionEngine::kEasedMoves`) now follows half a cosine - from rest,
fastest in the middle, to rest - with the asked speed at its peak, and goes to the servo as waypoints every ~0.45 s
(2-8 per move, each with the time to reach it plus 60 ms so the servo glides on instead of stopping). The motion
model follows the same curve. Gesture keyframes and other short moves stay single packets. Fewer moves come from the
executor (ARCHITECTURE section 4: the wandering gaze keeps a place 5-11 s); the flinch is softer (8° at 90°/s, was
12° at 160°/s).

**Diagnosing a frozen head:** if the face lives but the head does not move,
compare `cmd_travel` with `real_rotation` in `runtime: stats`. Commanded
50° and real 0° means the servos ignore positions: suspect mode, packet rate
or power (cable, supply), not the brain. If `servo_repowers` keeps growing,
check that `kUpdateMs` was not lowered. `imu=reads/fails` and `gyro_max`
(deg/s) show that the gyro is alive. The gyro does not see every move: on
2026-10-03 it read ~0 deg/s while the owner watched the head turn, so a
watchdog tighter than two 2 s windows of >= 10 deg power-cycled healthy
servos again and again (tried and reverted).

## IMU: shake and tilt

`ImuEngine` still reports shake / tilt events (for the log: `shake detected`, `tilt detected`).

Brain v4 reads the accelerometer itself: the shake channel is the spread of |acceleration| over 0.5 s (our own
servos' share taken off while they move), the tilt channel the lean from a learned upright (`ChannelEncoder`). The
IMU channels are not read for 3 s after a servo power cycle (the head's jolt as the servos take hold is not a shake).
The gyro serves the servo watchdog (real rotation vs commanded travel).

## Microphones and speaker

- The microphone task (core 1, priority 3) runs the 16-band filter bank (~0.27 ms per 5.3 ms block), the loud-sound
  and onset detectors (millisecond timestamps) and a stereo balance. Loud and onset events are not passed to the
  brain while the head moves and 350 ms after (see below).
- While the speaker plays, the main loop keeps extending the mute window
  (`suppressFor`), and the continuous level reads 0 too (the robot once
  heard its own chirps at `peak=32768` and fed itself in a loop).
- **The servos are loud during the whole movement, not only at start and
  stop.** Measured on the robot (`mic_peak=still/edge/mid-turn` in the
  stats, max block peak per 5 s): still 200–5000 (the room), within 400 ms
  of a start or stop 5k–28k, in the steady middle of a turn 13k–32767 (the
  10 Hz position steps click; they can clip the ADC). So while the head
  moves the loud-sound threshold is raised to 24000 with a 350 ms tail (the
  servo stops ~120 ms after the last packet, which can leave up to 100 ms
  after the model stopped), and a loud sound heard during own motion is
  not passed on at all (the v2 firmware attenuated them x0.3). The robot
  is therefore nearly deaf to sounds below ~24000 while it turns: a
  hardware limit, not a tuning choice. (The simulator's servos are quieter
  mid-turn; do not tune this in the simulator.)
- Inter-microphone time difference was measured on claps at 40 cm left,
  right and front: lag below one sample (≲40 µs) with a random sign, levels
  almost equal. The microphones are too close for a direction estimate;
  only a coarse balance is used (the onset lag measurement was removed from
  the firmware on 2026-10-03).
- "Listening by turning" (turning the head to compare levels) was measured
  and gave no usable effect.

## Camera

GC0308 at QQVGA, one frame buffer, capture task on core 1 at priority 2, ~7 fps.
Exposure, white balance and gain are locked after a short warm-up so the
vision front end does not see global brightness drift. The task is paced from the start of the previous frame
(7.0 fps). Brain v4 uses the motion sense - the share of the picture that moves after the head's own turn is
searched out, and its centre (the "where" values for attention) - and, since 2026-10-25, the brightest spot (8x8-sample
blocks over the frame mean, from 60 levels brighter, full at 130: a lamp or a window glares to 255 over a mean of ~118,
a white sheet reads ~+90..110; `VisionFrontEnd::findLight`): where it is and how strong. A frame taken while the head
itself moves is not used. Dark frames (luminance spread < 3.5 levels) are "blind". No frame leaves the robot.

## Display

LVGL on core 1. The pixel face is rendered as a grid in the main loop, compared with the one on the screen, and only
the rectangle of the pixels that changed is redrawn: a blink ~15 ms, nothing at all when an animation step lands on the
same pixels (the full-screen vector face took 35-70 ms a frame). The main loop takes the display lock with a 2 ms timeout
and skips the frame otherwise, so it never blocks on the display.
`face_fps` and `face_frame_max_us` are in the stats.

## Memory, tasks, configuration

- `RobotRuntime` (with the whole brain: the analyzer's state, front ends, network scratch) is a static object placed
  in PSRAM with `EXT_RAM_BSS_ATTR`
  (`CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y`). Before that, `.bss` in
  internal DRAM competed with the Wi-Fi and camera DMA buffers
  (`internal_free` fell to ~23–33 KB and the display froze). Every driver
  allocates its own DMA buffers in internal RAM in `begin()`.
- Internal RAM: ~35 KB free at runtime (minimum ~21 KB, brain v4 firmware 2026-10-24). Brain v4's int8 weight tables
  (236 KB) are copied from flash to PSRAM at boot (flash here is DIO, about half the speed of PSRAM).
- Wi-Fi buffers are trimmed (static RX 4, dynamic RX/TX 12, no A-MPDU):
  traffic: the diary (~5 datagrams/s + log lines), a ping every 20 min, NTP. Modem power save is off
  (`WIFI_PS_NONE`), the robot runs on USB.
- Tasks: main loop (core 0, 8 KB stack, registered with the task watchdog, 20 ms loop, the brain ticks every 50 ms),
  **voice** (core 1, priority 5, 6 KB: renders 20 ms chunks and blocks on the codec), microphone (core 1, priority
  3), **camera** (core 1, priority 2, 5 KB), diary sender (core 0, priority 1), update server (core 0, priority 2),
  Wi-Fi (priority 3), LVGL. The camera is below the microphone so a frame never keeps the microphone DMA waiting,
  and off core 0 so it never preempts the main loop.
- Safety config: `CONFIG_ESP_TASK_WDT_PANIC=y` (a hung loop reboots),
  `CONFIG_COMPILER_STACK_CHECK_MODE_NORM=y`, brownout detector at level 7,
  panic prints and reboots. `stack_headroom` in the stats is the main task's least free stack in bytes (~4.3 KB of
  8 KB; until 2026-10-27 it was printed 4x too high, "~15 KB"), `mic_stack` the microphone task's (4 KB stack; the
  voice check keeps its window in its object, not on that stack).
- CPU 240 MHz, 64-byte data cache lines. The numeric files (`v4/snn_v4`, `v4/nn_v4`, `perception/*`,
  `voice/babble_net`, `voice/droid_synth`) are compiled with `-O3 -funroll-loops` (no fast-math).
- `sdkconfig.defaults` is complete (16 MB flash, PSRAM pins / mode / 80 MHz,
  custom partition table, the options above): `rm sdkconfig && ./build.sh
  build` is safe. It once was not, and a regenerated config silently lost
  PSRAM (camera failed: `frame buffer malloc failed`) and the partition table.

## Flash partitions

Platform B's layout (2026-10-11, `partitions.csv`), kept: changing it needs a cable flash. Brain v4 keeps what it
learned from its life (stage 6) in the two weights slots.

| Name | Offset | Size | Use |
|---|---|---|---|
| nvs | 0x9000 | 24 KB | Wi-Fi calibration, `v4` (head pad sensitivity), `learned` (which learned blob is live / on trial); the earlier brains' namespaces are erased at boot |
| otadata | 0xF000 | 8 KB | which app slot boots, and its trial state |
| phy_init | 0x11000 | 4 KB | RF |
| ota_0, ota_1 | 0x20000, 0x420000 | 4 MB each | the application (~2 MB); an update goes to the idle one and boots on trial: the bootloader returns to the other if it does not reach a healthy minute (`CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE`) |
| weights_a, weights_b | 0x820000, 0x9A0000 | 1.5 MB each | the learned blob (`main/net/learned_store.h`, ~32 KB): the live one and the previous / next one |

`flash.sh` writes the bootloader, partition table, `otadata` (reset: `ota_0` boots) and the application into `ota_0`.
`tools/robot/ota.py firmware` updates the application over Wi-Fi (with a reboot), `tools/robot/ota.py learned FILE`
the learned blob (without one). Without a learned blob the networks run on their compiled-in weights.

## microSD

The slot shares the screen's SPI bus (SPI3: SCK 36, MOSI 37, MISO 35, card CS GPIO4). With the card's CS floating an
inserted card answers the screen's traffic and the picture flickers or vanishes: the firmware drives GPIO4 high at
boot. Writing the diary to the card (platform B, 2026-10-11) crashed the robot in a boot loop (assert in
`spi_hal_setup_trans`: an SD transaction started while the screen's DMA transfer was running); brain v4 has no card
code. If it is tried again: rare large writes while the screen is idle.

## Flashing and recovery

**Main rule: nothing else may have `/dev/ttyACM0` open while flashing**
(no `idf_monitor`, `screen`, `minicom`). A leftover monitor that took part
of the USB stream caused 1–5 % write errors (`chip stopped responding`,
`Invalid head of packet`). `flash.sh` checks a lock, open handles (`fuser`)
and the devices of all running Docker containers, and refuses; it never
kills other processes.

```bash
docker ps --format '{{.ID}} {{.Image}} {{.Command}} {{.Status}}'   # find a stale monitor
docker rm -f <the stale monitor container>                         # remove only it
fuser -v /dev/ttyACM0                                              # port must be free
./build.sh build && ./flash.sh                                     # wait for "Hash of data verified"
python3 tools/capture_serial.py --seconds 30                       # then a bounded capture
```

`capture_serial.py` uses plain termios (pyserial's open caused a USB chip
reset even with DTR/RTS off), disables HUPCL, never toggles DTR/RTS and
stops by itself (max 60 s). Do not wrap an unnamed `docker run` of
`idf_monitor` in `timeout`: it can kill only the client and leave the
monitor running; `monitor.sh` uses a named container, a lock and cleanup.

**After an interrupted write:**

1. Stop all serial monitors (remove only their containers).
2. Put the CoreS3 in download mode: hold BOOT, press Reset, release BOOT
   when the USB port appears (`ls -l /dev/ttyACM*`, `lsusb | grep 303a`).
3. Flash again: `./flash.sh`.
4. Press Reset or replug. If the log shows `boot: DOWNLOAD(USB/UART0)` /
   `waiting for download`, BOOT is still held.

If USB keeps dropping: another data-capable cable, a direct port without a
dock. If the chip answers with its MAC in download mode, it is alive; an
incomplete write just keeps the app from booting.

**Decoding a crash:** check that the `ELF file SHA256` in the log matches
`sha256sum build/stackchan_v4.elf`, then

```bash
../.tooling/espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin/xtensa-esp32s3-elf-addr2line -pfiaC \
    -e build/stackchan_v4.elf <addresses from Backtrace>
```

A known, fixed case: a late `esp_ping` callback (`onPingTimeout`) after
`pingOnce()` had returned.

## Drivers (`components/`)

Vendored drivers: `ftservo` (Feetech SCS / SCSCL), `bmi270` (Bosch SDK +
wrapper), `py32_io_expander`, `si12t`. A review fixed:

- `SCS::syncReadEnd()` used `delete` on an array (`delete[]` now);
  `SCSCL::ReadToqueEnable` read a word from a one-byte register.
- `si12t`: `esp_err_t` codes were OR-ed together; now the first real error is
  kept; no log of uninitialized memory after a failed read.
- `BMI270`, `PY32IOExpander_Class`: copying is deleted (double free of the
  I²C handle), constructors no longer abort on I²C registration errors,
  `malloc` per I²C write replaced by a stack buffer, sensor ranges come from
  the same enums as the configuration.
- `PY32IOExpander_Class`: a recursive mutex around register
  read-modify-write (two tasks could lose a bit), write errors are logged.
- `SCSerial::readSCS`: one blocking `uart_read_bytes` instead of a 1 ms
  polling loop; packets go out in one UART write.
- Open: `PY32IOExpander_Class::setPwmDuty()` scales 0–255 to 12 bit by an
  assumption not confirmed by a datasheet or measurement (`TODO(unverified)`).

## Known limitations

- No servo feedback (hardware); only the gyro sees real motion.
- No recognition of people or faces in brain v4: the camera gives only motion (share and place).
- The proximity sensor reads 0-16 of 2047 even with a hand at the face (2026-10-24, LED at 40 kHz x 8 pulses); the
  near channel stays almost empty.
- No sound direction (two microphones 5 cm apart).
- The robot has no clock without Wi-Fi / NTP: quiet for the first 2 minutes after a boot, then daytime rules.
- Nearly deaf to quiet sounds while turning (servo noise, see above).
- Millisecond timestamps wrap after 49.7 days of uptime; all comparisons go
  through `main/core/ms_clock.h` (`msReached` for periodic deadlines,
  `msWithin` for short windows) so nothing freezes at the wrap.
- Owner presence: up to 3 pings per 20 min check; "away" only after 3
  failed checks in a row (phones in power save miss single pings; a Wi-Fi
  drop leaves the state unchanged).
- The proximity / ambient-light sensor is polled at 10 Hz (it was read on
  every 20 ms loop pass: 200 I²C transactions per second on the shared bus).
