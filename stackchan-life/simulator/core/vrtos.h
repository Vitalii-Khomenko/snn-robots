#pragma once
// Virtual RTOS: runs the firmware's FreeRTOS tasks (main loop, microphone,
// camera) as host threads on a single virtual clock.
//
// Exactly one task runs at a time. A task gives up the CPU only when it
// blocks (vTaskDelay, codec read, I2S write, or an explicit busy period
// that models CPU time). The scheduler then resumes the task with the
// earliest wake-up time (ties: the one that blocked first) and moves the
// virtual clock forward to it. Nothing depends on host timing, so the same
// scenario and seed always produce the same run.

#include <cstdint>
#include <functional>
#include <string>

namespace vrtos {

// Virtual time since boot.
uint64_t nowUs();

// Creates a task; it starts at the current virtual time, after the caller
// blocks next. Returns false only if called before start().
bool spawn(const std::string& name, std::function<void()> body);

// Blocks the calling task for `us` of virtual time (0 = yield).
void sleepFor(uint64_t us);

// Blocks the calling task until an absolute virtual time.
void sleepUntil(uint64_t us);

// Models CPU work: the calling task keeps the CPU "busy" for `us`, other
// tasks (other core / preemption) run meanwhile, time advances.
inline void busyFor(uint64_t us) { sleepFor(us); }

// Runs `main_task` as the first task and returns when it returns. Other
// tasks are abandoned (the process is expected to exit afterwards).
void start(std::function<void()> main_task);

// Name of the task currently holding the CPU (for logs).
const char* currentTaskName();

}  // namespace vrtos
