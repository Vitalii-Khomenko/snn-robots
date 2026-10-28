#include "ota_server.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <esp_timer.h>
#include <esp_attr.h>
#include <esp_http_server.h>
#include <esp_log.h>
#include <esp_ota_ops.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "learned_store.h"
#include "sound_tap.h"

#if __has_include("life_log_host.h")
#include "life_log_host.h"  // secrets/: LIFE_OTA_TOKEN
#endif

namespace {
constexpr const char* kTag = "ota";
constexpr size_t kChunk = 4096;
constexpr int kMaxTimeouts = 6;  // in a row, of the server's receive timeout (recv_wait_timeout 10 s): a minute of silence

bool authorized(httpd_req_t* req)
{
#ifdef LIFE_OTA_TOKEN
    char token[80] = {};
    if (httpd_req_get_hdr_value_str(req, "X-Token", token, sizeof(token)) == ESP_OK &&
        std::strcmp(token, LIFE_OTA_TOKEN) == 0)
        return true;
#endif
    httpd_resp_send_err(req, HTTPD_403_FORBIDDEN, "token");
    return false;
}

void rebootSoon()
{
    // after the answer has left: the uploader learns that it worked
    xTaskCreate(
        [](void*) {
            vTaskDelay(pdMS_TO_TICKS(1500));
            esp_restart();
        },
        "reboot", 2048, nullptr, 1, nullptr);
}

// Streams the request body to sink(data, size) in chunks; false on a broken upload or a sink error.
template <typename Sink>
bool receive(httpd_req_t* req, Sink sink)
{
    EXT_RAM_BSS_ATTR static char buf[kChunk];  // one upload at a time (the server has one task); PSRAM: internal RAM is scarce
    size_t left = req->content_len;
    int timeouts = 0;
    while (left > 0) {
        const int n = httpd_req_recv(req, buf, left < kChunk ? left : kChunk);
        // a sender that went quiet ends the upload (2026-10-28, an audit: it waited forever, and the server has one
        // task - status and every update blocked until a reboot)
        if (n == HTTPD_SOCK_ERR_TIMEOUT) {
            if (++timeouts >= kMaxTimeouts) return false;
            continue;
        }
        timeouts = 0;
        if (n <= 0) return false;
        if (!sink(buf, static_cast<size_t>(n))) return false;
        left -= static_cast<size_t>(n);
    }
    return true;
}

esp_err_t statusHandler(httpd_req_t* req)
{
    if (!authorized(req)) return ESP_OK;
    auto* self = static_cast<OtaServer::StatusFn*>(req->user_ctx);
    char line[512];
    const int n = (*self)(line, sizeof(line));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, line, n);
    return ESP_OK;
}

esp_err_t learnedHandler(httpd_req_t* req)
{
    if (!authorized(req)) return ESP_OK;
    auto* store = static_cast<LearnedStore*>(req->user_ctx);
    ESP_LOGI(kTag, "new learned blob: %u bytes into slot %c", static_cast<unsigned>(req->content_len), store->idleSlot());
    if (!store->beginWrite(req->content_len)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "cannot write now (size, or a blob is on trial)");
        return ESP_OK;
    }
    const bool ok = receive(req, [&](const char* d, size_t n) { return store->write(d, n); }) && store->finishWrite();
    if (!ok) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "upload broken or blob rejected");
        return ESP_OK;
    }
    store->requestSwitch();  // the main loop switches between two ticks, no reboot
    httpd_resp_sendstr(req, "ok: switching now, on trial for a minute\n");
    return ESP_OK;
}

esp_err_t touchHandler(httpd_req_t* req)
{
    if (!authorized(req)) return ESP_OK;
    char body[8] = {};
    const int n = httpd_req_recv(req, body, sizeof(body) - 1);
    const int level = n > 0 ? body[0] - '0' : -1;
    auto* fn = static_cast<OtaServer::SettingFn*>(req->user_ctx);
    if (!(*fn)(level)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "body: 0..7");
        return ESP_OK;
    }
    httpd_resp_sendstr(req, "ok\n");
    return ESP_OK;
}

// GET /listen?s=N: the microphone for N seconds (1..600, default 30), as it is read - a header ("SND1", uint32 rate,
// uint32 frames per block, uint32 ms since boot now), then sound_tap::Block after Block. The server answers nothing
// else meanwhile.
esp_err_t listenHandler(httpd_req_t* req)
{
    if (!authorized(req)) return ESP_OK;
    char query[32] = {}, value[8] = {};
    int seconds = 30;
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK &&
        httpd_query_key_value(query, "s", value, sizeof(value)) == ESP_OK)
        seconds = std::atoi(value);
    seconds = std::clamp(seconds, 1, 600);
    if (!sound_tap::start()) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "busy or no memory");
        return ESP_OK;
    }
    ESP_LOGI(kTag, "listen: the microphone to the PC for %d s", seconds);
    httpd_resp_set_type(req, "application/octet-stream");
    const uint32_t head[4] = {0x31444e53u /* "SND1" */, sound_tap::kRate, sound_tap::kFrames,
                              static_cast<uint32_t>(esp_timer_get_time() / 1000)};
    bool ok = httpd_resp_send_chunk(req, reinterpret_cast<const char*>(head), sizeof(head)) == ESP_OK;
    constexpr size_t kBatch = 16;  // blocks per chunk (~4 KB)
    EXT_RAM_BSS_ATTR static sound_tap::Block batch[kBatch];
    const int64_t end_us = esp_timer_get_time() + static_cast<int64_t>(seconds) * 1000000;
    uint32_t blocks = 0;
    while (ok && esp_timer_get_time() < end_us) {
        size_t n = 0;
        while (n < kBatch && sound_tap::take(&batch[n], sizeof(batch[n]), n ? 20 : 200) == sizeof(batch[n])) ++n;
        if (n) ok = httpd_resp_send_chunk(req, reinterpret_cast<const char*>(batch), n * sizeof(batch[0])) == ESP_OK;
        blocks += n;
    }
    sound_tap::stop();
    ESP_LOGI(kTag, "listen: done, %u blocks sent, %u dropped%s", static_cast<unsigned>(blocks),
             static_cast<unsigned>(sound_tap::dropped()), ok ? "" : ", the PC hung up");
    if (ok) httpd_resp_send_chunk(req, nullptr, 0);
    return ESP_OK;
}

esp_err_t firmwareHandler(httpd_req_t* req)
{
    if (!authorized(req)) return ESP_OK;
    const esp_partition_t* part = esp_ota_get_next_update_partition(nullptr);
    esp_ota_handle_t ota = 0;
    ESP_LOGI(kTag, "new firmware: %u bytes into %s", static_cast<unsigned>(req->content_len), part ? part->label : "?");
    if (!part || esp_ota_begin(part, req->content_len, &ota) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "ota begin");
        return ESP_OK;
    }
    const bool got = receive(req, [&](const char* d, size_t n) { return esp_ota_write(ota, d, n) == ESP_OK; });
    // esp_ota_end checks the image (header, checksum); a broken one never becomes bootable
    if (!got || esp_ota_end(ota) != ESP_OK || esp_ota_set_boot_partition(part) != ESP_OK) {
        if (!got) esp_ota_abort(ota);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "upload broken or image rejected");
        return ESP_OK;
    }
    ESP_LOGI(kTag, "firmware in %s: on trial after the reboot", part->label);
    httpd_resp_sendstr(req, "ok: on trial after the reboot\n");
    rebootSoon();
    return ESP_OK;
}
}  // namespace

bool OtaServer::begin(LearnedStore& learned, StatusFn status, SettingFn touchSensitivity)
{
#ifndef LIFE_OTA_TOKEN
    ESP_LOGW(kTag, "no LIFE_OTA_TOKEN in secrets/life_log_host.h: updates over Wi-Fi are off");
    return false;
#else
    learned_ = &learned;
    status_ = std::move(status);
    touch_ = std::move(touchSensitivity);
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 4608;  // internal RAM (it writes flash); 6 KB pushed the free internal RAM to 5 KB
    cfg.core_id = 0;
    cfg.task_priority = 2;
    cfg.max_uri_handlers = 5;
    cfg.recv_wait_timeout = 10;
    httpd_handle_t server = nullptr;
    if (httpd_start(&server, &cfg) != ESP_OK) {
        ESP_LOGE(kTag, "http server not started");
        return false;
    }
    const httpd_uri_t uris[] = {
        {"/status", HTTP_GET, statusHandler, &status_},
        {"/firmware", HTTP_PUT, firmwareHandler, nullptr},
        {"/touch", HTTP_PUT, touchHandler, &touch_},
        {"/learned", HTTP_PUT, learnedHandler, learned_},
        {"/listen", HTTP_GET, listenHandler, nullptr},
    };
    for (const auto& u : uris) httpd_register_uri_handler(server, &u);
    ESP_LOGI(kTag, "updates over Wi-Fi on port %d", cfg.server_port);
    return true;
#endif
}
