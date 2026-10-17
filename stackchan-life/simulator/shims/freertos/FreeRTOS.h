#pragma once
// Host shim for FreeRTOS types/macros; tasks run on simulator/core/vrtos.
#include <cstdint>
typedef int BaseType_t;
typedef unsigned int UBaseType_t;
typedef uint32_t TickType_t;
typedef void* TaskHandle_t;
typedef void (*TaskFunction_t)(void*);
#define configTICK_RATE_HZ 100  /* CONFIG_FREERTOS_HZ=100 on the robot */
#define portTICK_PERIOD_MS (1000 / configTICK_RATE_HZ)
#define pdMS_TO_TICKS(ms) ((TickType_t)(((uint64_t)(ms) * configTICK_RATE_HZ) / 1000))
#define pdPASS 1
#define pdFAIL 0
#define pdTRUE 1
#define pdFALSE 0
#define portMAX_DELAY 0xffffffffu
typedef struct { int unused; } portMUX_TYPE;
#define portMUX_INITIALIZER_UNLOCKED {0}
// One task runs at a time in the simulator, so critical sections are no-ops.
#define portENTER_CRITICAL(mux) do { (void)(mux); } while (0)
#define portEXIT_CRITICAL(mux) do { (void)(mux); } while (0)
