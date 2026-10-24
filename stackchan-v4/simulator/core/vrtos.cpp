#include "vrtos.h"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace vrtos {
namespace {

struct Task {
    std::string name;
    uint64_t wake_us = 0;
    uint64_t order = 0;  // FIFO among equal wake times
    bool running = false;
    std::condition_variable cv;
};

std::mutex g_mutex;
std::vector<std::unique_ptr<Task>> g_tasks;
uint64_t g_now_us = 0;
uint64_t g_order = 0;
bool g_started = false;
thread_local Task* t_self = nullptr;

// Picks the next task to run and hands it the CPU. Called with the lock
// held by the task that is giving the CPU away.
void scheduleNextLocked()
{
    Task* next = nullptr;
    for (auto& task : g_tasks) {
        if (!next || task->wake_us < next->wake_us ||
            (task->wake_us == next->wake_us && task->order < next->order)) {
            next = task.get();
        }
    }
    if (next->wake_us > g_now_us) g_now_us = next->wake_us;
    next->running = true;
    next->cv.notify_one();
}

void blockLocked(std::unique_lock<std::mutex>& lock, uint64_t wake_us)
{
    Task* self = t_self;
    self->wake_us = wake_us < g_now_us ? g_now_us : wake_us;
    self->order = ++g_order;
    self->running = false;
    scheduleNextLocked();
    self->cv.wait(lock, [self] { return self->running; });
}

}  // namespace

uint64_t nowUs()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_now_us;
}

const char* currentTaskName()
{
    return t_self ? t_self->name.c_str() : "?";
}

bool spawn(const std::string& name, std::function<void()> body)
{
    std::unique_lock<std::mutex> lock(g_mutex);
    if (!g_started) return false;
    g_tasks.push_back(std::make_unique<Task>());
    Task* task = g_tasks.back().get();
    task->name = name;
    task->wake_us = g_now_us;
    task->order = ++g_order;
    std::thread([task, body = std::move(body)] {
        {
            std::unique_lock<std::mutex> inner(g_mutex);
            t_self = task;
            task->cv.wait(inner, [task] { return task->running; });
        }
        body();
        // Firmware tasks never return; if one does, park it forever.
        std::unique_lock<std::mutex> inner(g_mutex);
        task->wake_us = UINT64_MAX;
        task->running = false;
        scheduleNextLocked();
        task->cv.wait(inner, [] { return false; });
    }).detach();
    return true;
}

void sleepUntil(uint64_t us)
{
    std::unique_lock<std::mutex> lock(g_mutex);
    blockLocked(lock, us);
}

void sleepFor(uint64_t us)
{
    std::unique_lock<std::mutex> lock(g_mutex);
    blockLocked(lock, g_now_us + us);
}

void start(std::function<void()> main_task)
{
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_started = true;
        g_tasks.push_back(std::make_unique<Task>());
        g_tasks.back()->name = "main";
        g_tasks.back()->running = true;
        t_self = g_tasks.back().get();
    }
    main_task();
}

}  // namespace vrtos
