/*
 * SCS.h
 * Feetech serial servo communication layer protocol implementation
 * Date: 2024.11.21
 * Author: txl
 */

#ifndef _SCS_H
#define _SCS_H

#include "INST.h"

class SCS{
public:
	SCS();
	SCS(u8 End);
	SCS(u8 End, u8 Level);
	int genWrite(u8 ID, u8 MemAddr, u8 *nDat, u8 nLen);//Regular write instruction.
	int regWrite(u8 ID, u8 MemAddr, u8 *nDat, u8 nLen);//Asynchronous write instruction.
	int RegWriteAction(u8 ID = 0xfe);//Execute asynchronous write instruction.
	void syncWrite(u8 ID[], u8 IDN, u8 MemAddr, u8 *nDat, u8 nLen);//Synchronous write instruction.
	int writeByte(u8 ID, u8 MemAddr, u8 bDat);//Write one byte.
	int writeWord(u8 ID, u8 MemAddr, u16 wDat);//Write two bytes.
	int Read(u8 ID, u8 MemAddr, u8 *nData, u8 nLen);//Read instruction.
	int readByte(u8 ID, u8 MemAddr);//Read one byte.
	int readWord(u8 ID, u8 MemAddr);//Read two bytes.
	int Ping(u8 ID);//Ping instruction.
	int syncReadPacketTx(u8 ID[], u8 IDN, u8 MemAddr, u8 nLen);//Send a synchronous read instruction packet.
	int syncReadPacketRx(u8 ID, u8 *nDat);//Decode a synchronous read response; return the number of memory bytes on success, or 0 on failure.
	int syncReadRxPacketToByte();//Decode one byte.
	int syncReadRxPacketToWrod(u8 negBit=0);//Decode two bytes; negBit=0 means no sign bit.
	void syncReadBegin(u8 IDN, u8 rxLen, u32 TimeOut);//Begin synchronous read.
	void syncReadEnd();//End synchronous read.
	int Reset(u8 ID);//Reset the servo state.
	int Recal(u8 ID);//Reset the servo midpoint.
	u8 getState() { return u8Status; }
	u8 getLastError() { return u8Error; }
public:
	u8 Level;//Servo response level.
	u8 End;//Processor endianness.
	u8 u8Status;//Servo status.
	u8 u8Error;//Communication status.
	u8 syncReadRxPacketIndex;
	u8 syncReadRxPacketLen;
	u8 *syncReadRxPacket;
	u8 *syncReadRxBuff;
	u16 syncReadRxBuffLen;
	u16 syncReadRxBuffMax;
	u32 syncTimeOut;
protected:
	virtual int writeSCS(unsigned char *nDat, int nLen) = 0;
	virtual int readSCS(unsigned char *nDat, int nLen) = 0;
	virtual int readSCS(unsigned char *nDat, int nLen, unsigned long TimeOut) = 0;
	virtual int writeSCS(unsigned char bDat) = 0;
	virtual void rFlushSCS() = 0;
	virtual void wFlushSCS() = 0;
protected:
	void writeBuf(u8 ID, u8 MemAddr, u8 *nDat, u8 nLen, u8 Fun);
	void Host2SCS(u8 *DataL, u8* DataH, u16 Data);//Split one 16-bit number into two 8-bit numbers.
	u16	SCS2Host(u8 DataL, u8 DataH);//Combine two 8-bit numbers into one 16-bit number.
	int	Ack(u8 ID);//Return acknowledgement.
	int checkHead();//Check the frame header.
};
#endif
