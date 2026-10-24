#include "ota_server.h"

#include <cstring>
#include <esp_attr.h>
#include <esp_http_server.h>
#include <esp_log.h>
#include <esp_ota_ops.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "weight_store.h"

#if __has_include("life_log_host.h")
#include "life_log_host.h"  // secrets/: LIFE_OTA_TOKEN
#endif

namespace {
constexpr const char* kTag = "ota";
constexpr size_t kChunk = 4096;

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
    while (left > 0) {
        const int n = httpd_req_recv(req, buf, left < kChunk ? left : kChunk);
        if (n == HTTPD_SOCK_ERR_TIMEOUT) continue;
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

esp_err_t weightsHandler(httpd_req_t* req)
{
    if (!authorized(req)) return ESP_OK;
    auto* store = static_cast<WeightStore*>(req->user_ctx);
    ESP_LOGI(kTag, "new weights: %u bytes into slot %c", static_cast<unsigned>(req->content_len), store->idleSlot());
    if (!store->beginWrite(req->content_len)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "cannot write now (size, or a blob is on trial)");
        return ESP_OK;
    }
    const bool ok = receive(req, [&](const char* d, size_t n) { return store->write(d, n); }) && store->finishWrite();
    if (!ok) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "upload broken or blob rejected");
        return ESP_OK;
    }
    store->requestSwitch();  // the main loop switches the nets over, no reboot (a reboot sounds, also at night)
    httpd_resp_sendstr(req, "ok: switching now, on trial for a minute\n");
    return ESP_OK;
}

esp_err_t modeHandler(httpd_req_t* req)
{
    if (!authorized(req)) return ESP_OK;
    char body[16] = {};
    const int n = httpd_req_recv(req, body, sizeof(body) - 1);
    int mode = -1;
    if (n > 0 && std::strncmp(body, "brain", 5) == 0) mode = 0;
    if (n > 0 && std::strncmp(body, "collector", 9) == 0) mode = 1;
    if (n > 0 && std::strncmp(body, "life", 4) == 0) mode = 2;
    if (n > 0 && std::strncmp(body, "v4", 2) == 0) mode = 3;
    auto* fn = static_cast<OtaServer::ModeFn*>(req->user_ctx);
    if (mode < 0 || !(*fn)(mode)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "body: brain, collector or life");
        return ESP_OK;
    }
    httpd_resp_sendstr(req, "ok: the new mode after the reboot\n");
    rebootSoon();
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

EXT_RAM_BSS_ATTR uint8_t media_buf[128 * 1024];

esp_err_t mediaHandler(httpd_req_t* req, int kind)
{
    if (!authorized(req)) return ESP_OK;
    auto* fn = static_cast<OtaServer::MediaFn*>(req->user_ctx);
    const int n = (*fn)(kind, media_buf, sizeof(media_buf));
    if (n <= 0) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "no data");
        return ESP_OK;
    }
    httpd_resp_set_type(req, "application/octet-stream");
    httpd_resp_send(req, reinterpret_cast<const char*>(media_buf), n);
    return ESP_OK;
}
esp_err_t snapshotHandler(httpd_req_t* req) { return mediaHandler(req, 0); }
esp_err_t audioHandler(httpd_req_t* req) { return mediaHandler(req, 1); }

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

bool OtaServer::begin(WeightStore& weights, StatusFn status, ModeFn mode, SettingFn touchSensitivity, MediaFn media)
{
#ifndef LIFE_OTA_TOKEN
    ESP_LOGW(kTag, "no LIFE_OTA_TOKEN in secrets/life_log_host.h: updates over Wi-Fi are off");
    return false;
#else
    weights_ = &weights;
    status_ = std::move(status);
    mode_ = std::move(mode);
    touch_ = std::move(touchSensitivity);
    media_ = std::move(media);
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.stack_size = 4608;  // internal RAM (it writes flash); 6 KB pushed the free internal RAM to 5 KB
    cfg.core_id = 0;
    cfg.task_priority = 2;
    cfg.max_uri_handlers = 8;
    cfg.recv_wait_timeout = 10;
    httpd_handle_t server = nullptr;
    if (httpd_start(&server, &cfg) != ESP_OK) {
        ESP_LOGE(kTag, "http server not started");
        return false;
    }
    const httpd_uri_t uris[] = {
        {"/status", HTTP_GET, statusHandler, &status_},
        {"/weights", HTTP_PUT, weightsHandler, weights_},
        {"/firmware", HTTP_PUT, firmwareHandler, nullptr},
        {"/mode", HTTP_PUT, modeHandler, &mode_},
        {"/touch", HTTP_PUT, touchHandler, &touch_},
        {"/snapshot", HTTP_GET, snapshotHandler, &media_},
        {"/audio", HTTP_GET, audioHandler, &media_},
    };
    for (const auto& u : uris) httpd_register_uri_handler(server, &u);
    ESP_LOGI(kTag, "updates over Wi-Fi on port %d", cfg.server_port);
    return true;
#endif
}
