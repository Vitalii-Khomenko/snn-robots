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
| Speaker | AW88298 via the BSP codec, always opened at 24000 Hz with `pa_gain = 1` (the BSP default gain is ~14 dB too quiet and its 22050 Hz rate never wakes the amp); the codec is opened once and left open (reopening per sound glitched the shared I²S bus and the microphones); the volume is written to the amplifier only when it changes (it was an I²C write on every 20 ms chunk); the playback rate was measured on 2026-09-30 (1.5 s of silence: 23,995 samples/s consumed of 24,000 - the mono stream on the stereo bus shared with the microphones plays at the right speed); loudness is set by `droid::Synth::level` (0.17 of full scale since 2026-10-02: peaks ~5k, RMS ~2.5k, at the owner's request after the voice had been made louder twice; the amplifier stays at 90); the voice is rendered and written by its own audio task (core 1, priority 5), because writes from the main loop stuttered whenever a brain tick ran long |
| Microphones | stereo I²S (shared bus with the speaker), own task (core 1, priority 3) running the 16-band filter bank |
| IMU | BMI270 (in the head: head pitch rotates gravity) |
| Touch | Si12T head pad; LCD touch via LVGL `LV_EVENT_PRESSED` (an atomic flag, handled in the main loop) |
| Proximity / light | LTR-553 |
| Power | AXP2101 (battery level, charging direction; "USB present but not charging" cannot be told apart) |
| Camera | GC0308, QQVGA RGB565, ~7 fps (143 ms per frame), one frame buffer (a one-off `cam_hal: FB-SIZE: 30720 != 38400` warning at start-up is harmless); the task runs the vision front end and the CNN (~26 ms per frame) on core 1 at priority 2, below the microphone |

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

**Servo watchdog** (`RobotRuntime::superviseServos`): if two consecutive
2 s windows had ≥10° commanded and <1.5° real rotation (gyro), it switches
VM EN off for 500 ms and on again and resyncs the motion model; at most
once per 10 s. One window was not enough (slow drift below the gyro's
~3°/s floor gave a false repower). A soft recovery (mode + torque +
position again) helped in only 2 of ~14 stalls and was removed. Log:
`servo rail power cycle (...) py32=0x41 vm_en out=1`. Every boot also
power-cycles the servos once.

**Diagnosing a frozen head:** if the face lives but the head does not move,
compare `cmd_travel` with `real_rotation` in `runtime: stats`. Commanded
50° and real 0° means the servos ignore positions: suspect mode, packet rate
or power (cable, supply), not the brain. If `servo_repowers` keeps growing,
check that `kUpdateMs` was not lowered. `imu=reads/fails` and `gyro_max`
(deg/s) show that the gyro is alive. The gyro does not see every move: on
2026-10-03 it read ~0 deg/s while the owner watched the head turn, so a
watchdog tighter than two 2 s windows of >= 10 deg power-cycled healthy
servos again and again (tried and reverted).

## Gyro: the vestibular sense

Shake detection stays on while the head moves, with the jolt threshold
doubled (a smooth turn changes gravity in this head-mounted IMU by well
under 1 m/s² per sample; a shaking hand gives tens). A detected shake is felt by the brain for 2 s; tilt (the body held up
or turned over: sustained deviation from the at-rest gravity above 3 m/s² for 15 samples, released below 1.5) for 1.5 s and for as
long as it stays tilted (`ImuEngine::tiltActive`). The simulator injects both events from its world model, because its own IMU
path is held back while the head "moves". The at-rest gravity is re-taken after 3 s of lying still wherever the body lies (2026-10-04: lifted and put back, the
robot otherwise stayed 'tilted' for good, startled, and touches went unanswered).

The gyro finds by itself which axis is head yaw and which is pitch: it
correlates each axis with the velocity the spring model expects (EMA 0.995,
only while the model moves faster than 5°/s) and locks when |correlation| >
50 (yaw) and > 20 (pitch): `gyro=learning` → `gyro=calibrated`. Then the
bus gets the real head velocity and a model / gyro mismatch feeds the
motor prediction error (compared with the model velocity up to 4 ticks back, dead
zone 0.08, because the servos lag the model by 50–150 ms). If someone holds
the head or a servo hangs, the brain feels it as a surprise. The efference
copy still comes from the command, which must arrive before the motion.

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
(a 26 ms network run does not lower the frame rate: 7.0 fps). Vision features older than 350 ms fade out before
they reach the brain, and a stalled camera goes blind after 2 s. Dark frames (luminance spread < 3.5 levels) are
"blind": no person, face, target or approach is reported.

## Display

LVGL on core 1; a face frame takes ~36 ms and is drawn only when something
changes (5–10 fps). The main loop takes the display lock with a 2 ms timeout
and skips the frame otherwise, so it never blocks on the display.
`face_fps` and `face_frame_max_us` are in the stats.

## Memory, tasks, configuration

- `RobotRuntime` (with the whole brain: spiking core state, front ends, network scratch) is a static object placed
  in PSRAM with `EXT_RAM_BSS_ATTR`
  (`CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY=y`). Before that, `.bss` in
  internal DRAM competed with the Wi-Fi and camera DMA buffers
  (`internal_free` fell to ~23–33 KB and the display froze). Every driver
  allocates its own DMA buffers in internal RAM in `begin()`.
- Internal RAM: ~47 KB free at runtime (minimum ~40 KB). The Vision NN's layer activations (9 KB) live in
  internal SRAM when more than 36 KB would remain after Wi-Fi and the camera took theirs; the spiking core's input
  weights (231 KB) are copied from flash to PSRAM at boot (flash here is DIO, about half the speed).
- Wi-Fi buffers are trimmed (static RX 4, dynamic RX/TX 12, no A-MPDU):
  traffic is one ping every 20 min and NTP. Modem power save is off
  (`WIFI_PS_NONE`), the robot runs on USB.
- Tasks: main loop (core 0, 8 KB stack, registered with the task watchdog, 20 ms loop, the brain ticks every 50 ms),
  **voice** (core 1, priority 5, 6 KB: renders 20 ms chunks and blocks on the codec), microphone (core 1, priority
  3), **camera** (core 1, priority 2, 5 KB), Wi-Fi (priority 3), LVGL. The camera is below the microphone so a
  26 ms network run never keeps the microphone DMA waiting, and off core 0 so it never preempts the main loop.
- Safety config: `CONFIG_ESP_TASK_WDT_PANIC=y` (a hung loop reboots),
  `CONFIG_COMPILER_STACK_CHECK_MODE_NORM=y`, brownout detector at level 7,
  panic prints and reboots. `stack_headroom` in the stats (~15 KB margin now).
- CPU 240 MHz, 64-byte data cache lines. The numeric files (`hybrid/*net*`, `spiking_core`, `audio_frontend`,
  `vision_frontend`, `policy_gru`, `voice/babble_*`) are compiled with `-O3 -funroll-loops` (no fast-math).
- `sdkconfig.defaults` is complete (16 MB flash, PSRAM pins / mode / 80 MHz,
  custom partition table, the options above): `rm sdkconfig && ./build.sh
  build` is safe. It once was not, and a regenerated config silently lost
  PSRAM (camera failed: `frame buffer malloc failed`) and the partition table.

## Flash partitions

Platform B layout (2026-10-11, `partitions.csv`). `nvs` kept the v3 offset and size, so the learned memory survived
the change.

| Name | Offset | Size | Use |
|---|---|---|---|
| nvs | 0x9000 | 24 KB | Wi-Fi calibration, the learned associative memory (`assoc`), the live weights slot (`weights`: `slot`, `next`, `tries`) |
| otadata | 0xF000 | 8 KB | which app slot boots, and its trial state |
| phy_init | 0x11000 | 4 KB | RF |
| ota_0, ota_1 | 0x20000, 0x420000 | 4 MB each | the application (~2 MB); an update goes to the idle one and boots on trial: the bootloader returns to the other if it does not reach a healthy minute (`CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE`) |
| weights_a, weights_b | 0x820000, 0x9A0000 | 1.5 MB each | the networks' weights outside the firmware (`main/net/weight_store.h`, blob ~0.5 MB); a new blob goes to the idle slot and is on trial for two boots |

`flash.sh` writes the bootloader, partition table, `otadata` (reset: `ota_0` boots) and the application into `ota_0`.
`tools/life/flash_weights.sh [a|b]` writes a weights blob over the cable; `tools/life/ota.py` updates both over Wi-Fi.
Without a blob the networks run on their compiled-in weights.

## microSD

The slot shares the screen's SPI bus (SPI3: SCK 36, MOSI 37, MISO 35, card CS GPIO4). With the card's CS floating an
inserted card answers the screen's traffic and the picture flickers or vanishes: the firmware drives GPIO4 high at
boot. Writing the diary to the card (`main/net/sd_diary.*`) crashed the robot in a boot loop (assert in
`spi_hal_setup_trans`: an SD transaction started while the screen's DMA transfer was running) and is switched off
(`kSdEnabled` in `main/net/life_log.cpp`). If it is tried again: rare large writes while the screen is idle.

## Flashing and recovery

**Main rule: nothing else may have `/dev/ttyACM0` open while flashing**
(no `idf_monitor`, `screen`, `minicom`). A leftover monitor that took part
of the USB stream caused 1–5 % write errors (`chip stopped responding`,
`Invalid head of packet`). `flash.sh` checks a lock, open handles (`fuser`)
and the devices of all running Docker containers, and refuses; it never
kills other processes.

```bash
docker ps --format '{{.ID}} {{.Image}} {{.Command}} {{.Status}}'   # find a stale monitor
docker rm -f stackchan-neuromorphic-monitor                        # remove only it
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
`sha256sum build/stackchan_neuromorphic.elf`, then

```bash
../.tooling/espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin/xtensa-esp32s3-elf-addr2line -pfiaC \
    -e build/stackchan_neuromorphic.elf <addresses from Backtrace>
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
- Person and face detection come from a 24 k-parameter CNN on a 32x24 image: it finds people-sized shapes, not
  identities, and was trained on simulated people and COCO photos; expect false alarms and misses.
- No sound direction (two microphones 5 cm apart).
- The robot has no clock without Wi-Fi / NTP (no night).
- Nearly deaf to quiet sounds while turning (servo noise, see above).
- Millisecond timestamps wrap after 49.7 days of uptime; all comparisons go
  through `main/core/ms_clock.h` (`msReached` for periodic deadlines,
  `msWithin` for short windows) so nothing freezes at the wrap.
- Owner presence: up to 3 pings per 20 min check; "away" only after 3
  failed checks in a row (phones in power save miss single pings; a Wi-Fi
  drop leaves the state unchanged).
- The proximity / ambient-light sensor is polled at 10 Hz (it was read on
  every 20 ms loop pass: 200 I²C transactions per second on the shared bus).
