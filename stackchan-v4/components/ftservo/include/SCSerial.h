/*
 * SCSerial.h
 * Feetech serial servo hardware interface layer implementation
 * Date: 2024.11.22
 * Author: txl
 */

#ifndef _SCSERIAL_H
#define _SCSERIAL_H

#include "driver/gpio.h"
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "SCS.h"

class SCSerial : public SCS
{
public:
	SCSerial();
	SCSerial(u8 End);
	SCSerial(u8 End, u8 Level);
	bool begin(uart_port_t uart_num, int baud_rate, int tx_pin, int rx_pin, int buf_size = 1024);
	void end();

protected:
	int writeSCS(unsigned char *nDat, int nLen);//Output nLen bytes.
	int readSCS(unsigned char *nDat, int nLen);//Input nLen bytes; uses IOTimeOut as the overall timeout.
	int readSCS(unsigned char *nDat, int nLen, unsigned long TimeOut);//Input nLen bytes; TimeOut ms is the overall timeout for all nLen bytes.
	int writeSCS(unsigned char bDat);//Output one byte.
	void rFlushSCS();//
	void wFlushSCS();//
public:
	unsigned long IOTimeOut;//Input/output timeout.
	uart_port_t uart_num;//Serial port number.
	int tx_pin = -1, rx_pin = -1;
};

#endif
