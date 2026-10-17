/*
 * SCSerial.cpp
 * Feetech serial servo hardware interface layer implementation
 * Date: 2024.4.2
 * Author:
 */

#include "SCSerial.h"

#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "FT_SCSerial";

SCSerial::SCSerial()
{
	IOTimeOut = 10;
	uart_num = UART_NUM_MAX;
}

SCSerial::SCSerial(u8 End):SCS(End)
{
	IOTimeOut = 10;
	uart_num = UART_NUM_MAX;
}

SCSerial::SCSerial(u8 End, u8 Level):SCS(End, Level)
{
	IOTimeOut = 10;
	uart_num = UART_NUM_MAX;
}

bool SCSerial::begin(uart_port_t uart_num, int baud_rate, int tx_pin, int rx_pin, int buf_size)
{
	this->uart_num = uart_num;
	this->tx_pin = tx_pin;
	this->rx_pin = rx_pin;

	uart_config_t uart_config = {};
	uart_config.baud_rate = baud_rate;
	uart_config.data_bits = UART_DATA_8_BITS;
	uart_config.parity = UART_PARITY_DISABLE;
	uart_config.stop_bits = UART_STOP_BITS_1;
	uart_config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
	uart_config.rx_flow_ctrl_thresh = 0;
	uart_config.source_clk = UART_SCLK_DEFAULT;

	esp_err_t ret = uart_driver_install(uart_num, buf_size, buf_size, 0, NULL, 0);
	if (ret != ESP_OK) {
		ESP_LOGE(TAG, "Failed to install UART driver: %s", esp_err_to_name(ret));
		return false;
	}

	ret = uart_param_config(uart_num, &uart_config);
	if (ret != ESP_OK) {
		ESP_LOGE(TAG, "Failed to configure UART: %s", esp_err_to_name(ret));
		uart_driver_delete(uart_num);
		this->uart_num = UART_NUM_MAX;
		return false;
	}

	ret = uart_set_pin(uart_num, tx_pin, rx_pin, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
	if (ret != ESP_OK) {
		ESP_LOGE(TAG, "Failed to set UART pins: %s", esp_err_to_name(ret));
		uart_driver_delete(uart_num);
		this->uart_num = UART_NUM_MAX;
		return false;
	}

	return true;
}

void SCSerial::end()
{
	if (uart_num < UART_NUM_MAX) {
		uart_driver_delete(uart_num);
		uart_num = UART_NUM_MAX;
	}
}

// Both overloads now share one implementation: a single blocking
// uart_read_bytes() call with an overall timeout for the whole nLen-byte
// packet, instead of a hand-rolled single-byte polling loop (uart_read_bytes
// already blocks on the driver's internal queue/ISR, so this is both lower
// latency and cheaper than waking up every 1 ms to check
// uart_get_buffered_data_len()). This also removes the previous
// inconsistency where the IOTimeOut-based overload measured an inter-byte
// ("idle") timeout while the explicit-TimeOut overload measured a total
// timeout for the same nLen bytes - every servo response here is a small,
// fixed-size packet, so a single overall timeout is the simpler and more
// predictable choice for both.
int SCSerial::readSCS(unsigned char *nDat, int nLen, unsigned long TimeOut)
{
	if (uart_num >= UART_NUM_MAX || nLen <= 0) {
		return 0;
	}

	int size = uart_read_bytes(uart_num, nDat, static_cast<uint32_t>(nLen), pdMS_TO_TICKS(TimeOut));
	return size > 0 ? size : 0;
}

int SCSerial::readSCS(unsigned char *nDat, int nLen)
{
	return readSCS(nDat, nLen, IOTimeOut);
}

int SCSerial::writeSCS(unsigned char *nDat, int nLen)
{
	if (uart_num >= UART_NUM_MAX || nDat == NULL) {
		return 0;
	}

	return uart_write_bytes(uart_num, reinterpret_cast<const char *>(nDat), nLen);
}

int SCSerial::writeSCS(unsigned char bDat)
{
	if (uart_num >= UART_NUM_MAX) {
		return 0;
	}

	return uart_write_bytes(uart_num, reinterpret_cast<const char *>(&bDat), 1);
}

void SCSerial::rFlushSCS()
{
	if (uart_num >= UART_NUM_MAX) {
		return;
	}

	uart_flush_input(uart_num);
}

void SCSerial::wFlushSCS()
{
	if (uart_num >= UART_NUM_MAX) {
		return;
	}

	uart_wait_tx_done(uart_num, pdMS_TO_TICKS(100));
}
