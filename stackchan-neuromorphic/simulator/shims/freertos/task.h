#pragma once
#include "FreeRTOS.h"
void vTaskDelay(TickType_t ticks);
BaseType_t xTaskCreate(TaskFunction_t fn, const char* name, uint32_t stack, void* arg, UBaseType_t prio,
                       TaskHandle_t* out);
BaseType_t xTaskCreatePinnedToCore(TaskFunction_t fn, const char* name, uint32_t stack, void* arg,
                                   UBaseType_t prio, TaskHandle_t* out, BaseType_t core);
UBaseType_t uxTaskGetStackHighWaterMark(TaskHandle_t task);
