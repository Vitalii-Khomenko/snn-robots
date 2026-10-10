/*
 * SCS.cpp
 * Feetech serial servo communication layer protocol implementation
 * Date: 2024.12.21
 * Author: txl
 */

#include <stddef.h>
#include "SCS.h"

SCS::SCS()
{
	Level = 1;//All instructions except broadcast instructions return an acknowledgement
	u8Status = 0;
}

SCS::SCS(u8 End)
{
	Level = 1;
	this->End = End;
	u8Status = 0;
}

SCS::SCS(u8 End, u8 Level)
{
	this->Level = Level;
	this->End = End;
	u8Status = 0;
}

//Split one 16-bit number into two 8-bit numbers.
//DataL is the low byte and DataH is the high byte.
void SCS::Host2SCS(u8 *DataL, u8* DataH, u16 Data)
{
	if(End){
		*DataL = (Data>>8);
		*DataH = (Data&0xff);
	}else{
		*DataH = (Data>>8);
		*DataL = (Data&0xff);
	}
}

//Combine two 8-bit numbers into one 16-bit number.
//DataL is the low byte and DataH is the high byte.
u16 SCS::SCS2Host(u8 DataL, u8 DataH)
{
	u16 Data;
	if(End){
		Data = DataL;
		Data<<=8;
		Data |= DataH;
	}else{
		Data = DataH;
		Data<<=8;
		Data |= DataL;
	}
	return Data;
}

// The whole packet is assembled first and handed to the UART in ONE call.
// It used to go out as three writes (header, data, checksum): a task switch
// between them left a gap in the middle of a packet on the wire, and the
// servo's parser can then resynchronise inside the packet.
void SCS::writeBuf(u8 ID, u8 MemAddr, u8 *nDat, u8 nLen, u8 Fun)
{
	u8 pkt[64];
	if(nDat && nLen > sizeof(pkt) - 7){
		nLen = sizeof(pkt) - 7;
	}
	u8 msgLen = 2;
	int n = 0;
	pkt[n++] = 0xff;
	pkt[n++] = 0xff;
	pkt[n++] = ID;
	if(nDat){
		msgLen += nLen + 1;
	}
	pkt[n++] = msgLen;
	pkt[n++] = Fun;
	u8 CheckSum = ID + msgLen + Fun;
	if(nDat){
		pkt[n++] = MemAddr;
		CheckSum += MemAddr;
		for(u8 i=0; i<nLen; i++){
			pkt[n++] = nDat[i];
			CheckSum += nDat[i];
		}
	}
	pkt[n++] = ~CheckSum;
	writeSCS(pkt, n);
}

//Regular write instruction.
//Servo ID, memory table address, data to write, and data length.
int SCS::genWrite(u8 ID, u8 MemAddr, u8 *nDat, u8 nLen)
{
	rFlushSCS();
	writeBuf(ID, MemAddr, nDat, nLen, INST_WRITE);
	wFlushSCS();
	return Ack(ID);
}

//Asynchronous write instruction.
//Servo ID, memory table address, data to write, and data length.
int SCS::regWrite(u8 ID, u8 MemAddr, u8 *nDat, u8 nLen)
{
	rFlushSCS();
	writeBuf(ID, MemAddr, nDat, nLen, INST_REG_WRITE);
	wFlushSCS();
	return Ack(ID);
}

//Execute asynchronous write instruction.
//Servo ID.
int SCS::RegWriteAction(u8 ID)
{
	rFlushSCS();
	writeBuf(ID, 0, NULL, 0, INST_REG_ACTION);
	wFlushSCS();
	return Ack(ID);
}

//Synchronous write instruction.
//Servo ID array, ID array length, memory table address, data to write, and data length.
void SCS::syncWrite(u8 ID[], u8 IDN, u8 MemAddr, u8 *nDat, u8 nLen)
{
	rFlushSCS();
	// One UART write for the whole packet (see writeBuf()).
	u8 pkt[200];
	if((nLen+1)*IDN + 8 > (int)sizeof(pkt)){
		IDN = (sizeof(pkt) - 8) / (nLen+1);
	}
	u8 mesLen = ((nLen+1)*IDN+4);
	int n = 0;
	pkt[n++] = 0xff;
	pkt[n++] = 0xff;
	pkt[n++] = 0xfe;
	pkt[n++] = mesLen;
	pkt[n++] = INST_SYNC_WRITE;
	pkt[n++] = MemAddr;
	pkt[n++] = nLen;
	u8 Sum = 0xfe + mesLen + INST_SYNC_WRITE + MemAddr + nLen;
	for(u8 i=0; i<IDN; i++){
		pkt[n++] = ID[i];
		Sum += ID[i];
		for(u8 j=0; j<nLen; j++){
			pkt[n++] = nDat[i*nLen+j];
			Sum += nDat[i*nLen+j];
		}
	}
	pkt[n++] = ~Sum;
	writeSCS(pkt, n);
	wFlushSCS();
}

int SCS::writeByte(u8 ID, u8 MemAddr, u8 bDat)
{
	rFlushSCS();
	writeBuf(ID, MemAddr, &bDat, 1, INST_WRITE);
	wFlushSCS();
	return Ack(ID);
}

int SCS::writeWord(u8 ID, u8 MemAddr, u16 wDat)
{
	u8 bBuf[2];
	Host2SCS(bBuf+0, bBuf+1, wDat);
	rFlushSCS();
	writeBuf(ID, MemAddr, bBuf, 2, INST_WRITE);
	wFlushSCS();
	return Ack(ID);
}

//Read instruction.
//Servo ID, memory table address, output data buffer, and data length.
int SCS::Read(u8 ID, u8 MemAddr, u8 *nData, u8 nLen)
{
	rFlushSCS();
	writeBuf(ID, MemAddr, &nLen, 1, INST_READ);
	wFlushSCS();
	u8Error = 0;
	if(!checkHead()){
		u8Error = ERR_NO_REPLY;
		return 0;
	}
	u8 bBuf[4];
	u8Status = 0;
	if(readSCS(bBuf, 3)!=3){
		u8Error = ERR_NO_REPLY;
		return 0;
	}
	if(bBuf[0]!=ID && ID!=0xfe){
		u8Error = ERR_SLAVE_ID;
		return 0;
	}
	if(bBuf[1]!=(nLen+2)){
		u8Error = ERR_BUFF_LEN;
		return 0;
	}
	int Size = readSCS(nData, nLen);
	if(Size!=nLen){
		u8Error = ERR_NO_REPLY;
		return 0;
	}
	if(readSCS(bBuf+3, 1)!=1){
		u8Error = ERR_NO_REPLY;
		return 0;
	}
	u8 calSum = bBuf[0]+bBuf[1]+bBuf[2];
	u8 i;
	for(i=0; i<Size; i++){
		calSum += nData[i];
	}
	calSum = ~calSum;
	if(calSum!=bBuf[3]){
		u8Error = ERR_CRC_CMP;
		return 0;
	}
	u8Status = bBuf[2];
	return Size;
}

//Read one byte; return -1 on timeout.
int SCS::readByte(u8 ID, u8 MemAddr)
{
	u8 bDat;
	int Size = Read(ID, MemAddr, &bDat, 1);
	if(Size!=1){
		return -1;
	}else{
		return bDat;
	}
}

//Read two bytes; return -1 on timeout.
int SCS::readWord(u8 ID, u8 MemAddr)
{	
	u8 nDat[2];
	int Size;
	u16 wDat;
	Size = Read(ID, MemAddr, nDat, 2);
	if(Size!=2)
		return -1;
	wDat = SCS2Host(nDat[0], nDat[1]);
	return wDat;
}

//Ping instruction; return the servo ID or -1 on timeout.
int	SCS::Ping(u8 ID)
{
	rFlushSCS();
	writeBuf(ID, 0, NULL, 0, INST_PING);
	wFlushSCS();
	u8Status = 0;
	if(!checkHead()){
		u8Error = ERR_NO_REPLY;
		return -1;
	}
	u8 bBuf[4];
	u8Error = 0;
	if(readSCS(bBuf, 4)!=4){
		u8Error = ERR_NO_REPLY;
		return -1;
	}
	if(bBuf[0]!=ID && ID!=0xfe){
		u8Error = ERR_SLAVE_ID;
		return -1;
	}
	if(bBuf[1]!=2){
		u8Error = ERR_BUFF_LEN;
		return -1;
	}
	u8 calSum = ~(bBuf[0]+bBuf[1]+bBuf[2]);
	if(calSum!=bBuf[3]){
		u8Error = ERR_CRC_CMP;
		return -1;			
	}
	u8Status = bBuf[2];
	return bBuf[0];
}

int SCS::checkHead()
{
	u8 bDat;
	u8 bBuf[] = {0, 0};
	u8 Cnt = 0;
	while(1){
		if(!readSCS(&bDat, 1)){
			return 0;
		}
		bBuf[1] = bBuf[0];
		bBuf[0] = bDat;
		if(bBuf[0]==0xff && bBuf[1]==0xff){
			break;
		}
		Cnt++;
		if(Cnt>10){
			return 0;
		}
	}
	return 1;
}

int	SCS::Ack(u8 ID)
{
	u8Error = 0;
	if(ID!=0xfe && Level){
		if(!checkHead()){
			u8Error = ERR_NO_REPLY;
			return 0;
		}
		u8Status = 0;
		u8 bBuf[4];
		if(readSCS(bBuf, 4)!=4){
			u8Error = ERR_NO_REPLY;
			return 0;
		}
		if(bBuf[0]!=ID){
			u8Error = ERR_SLAVE_ID;
			return 0;
		}
		if(bBuf[1]!=2){
			u8Error = ERR_BUFF_LEN;
			return 0;
		}
		u8 calSum = ~(bBuf[0]+bBuf[1]+bBuf[2]);
		if(calSum!=bBuf[3]){
			u8Error = ERR_CRC_CMP;
			return 0;
		}
		u8Status = bBuf[2];
	}
	return 1;
}

int	SCS::syncReadPacketTx(u8 ID[], u8 IDN, u8 MemAddr, u8 nLen)
{
	rFlushSCS();
	syncReadRxPacketLen = nLen;
	u8 checkSum = (4+0xfe) + IDN + MemAddr + nLen + INST_SYNC_READ;
	u8 i;
	writeSCS(0xff);
	writeSCS(0xff);
	writeSCS(0xfe);
	writeSCS(IDN+4);
	writeSCS(INST_SYNC_READ);
	writeSCS(MemAddr);
	writeSCS(nLen);
	for(i=0; i<IDN; i++){
		writeSCS(ID[i]);
		checkSum += ID[i];
	}
	checkSum = ~checkSum;
	writeSCS(checkSum);
	wFlushSCS();
	
	syncReadRxBuffLen = readSCS(syncReadRxBuff, syncReadRxBuffMax, syncTimeOut);
	return syncReadRxBuffLen;
}

void SCS::syncReadBegin(u8 IDN, u8 rxLen, u32 TimeOut)
{
	syncReadRxBuffMax = IDN*(rxLen+6);
	syncReadRxBuff = new u8[syncReadRxBuffMax];
	syncTimeOut = TimeOut;
}

void SCS::syncReadEnd()
{
	if(syncReadRxBuff){
		delete[] syncReadRxBuff;
		syncReadRxBuff = NULL;
	}
}

int SCS::syncReadPacketRx(u8 ID, u8 *nDat)
{
	u16 syncReadRxBuffIndex = 0;
	syncReadRxPacket = nDat;
	syncReadRxPacketIndex = 0;
	u8Error = 0;
	while((syncReadRxBuffIndex+6+syncReadRxPacketLen)<=syncReadRxBuffLen){
		u8 bBuf[] = {0, 0, 0};
		u8 calSum = 0;
		while(syncReadRxBuffIndex<syncReadRxBuffLen){
			bBuf[0] = bBuf[1];
			bBuf[1] = bBuf[2];
			bBuf[2] = syncReadRxBuff[syncReadRxBuffIndex++];
			if(bBuf[0]==0xff && bBuf[1]==0xff && bBuf[2]!=0xff){
				break;
			}
		}
		if(bBuf[2]!=ID){
			continue;
		}
		if(syncReadRxBuff[syncReadRxBuffIndex++]!=(syncReadRxPacketLen+2)){
			continue;
		}
		u8Status = syncReadRxBuff[syncReadRxBuffIndex++];
		calSum = ID + (syncReadRxPacketLen+2) + u8Status;
		for(u8 i=0; i<syncReadRxPacketLen; i++){
			syncReadRxPacket[i] = syncReadRxBuff[syncReadRxBuffIndex++];
			calSum += syncReadRxPacket[i];
		}
		calSum = ~calSum;
		if(calSum!=syncReadRxBuff[syncReadRxBuffIndex++]){
			u8Error = ERR_CRC_CMP;
			return 0;
		}
		return syncReadRxPacketLen;
	}
	return 0;
}

int SCS::syncReadRxPacketToByte()
{
	if(syncReadRxPacketIndex>=syncReadRxPacketLen){
		u8Error = ERR_BUFF_LEN;
		return -1;
	}
	return syncReadRxPacket[syncReadRxPacketIndex++];
}

int SCS::syncReadRxPacketToWrod(u8 negBit)
{
	if((syncReadRxPacketIndex+1)>=syncReadRxPacketLen){
		u8Error = ERR_BUFF_LEN;
		return -1;
	}
	int Word = SCS2Host(syncReadRxPacket[syncReadRxPacketIndex], syncReadRxPacket[syncReadRxPacketIndex+1]);
	syncReadRxPacketIndex += 2;
	if(negBit){
		if(Word&(1<<negBit)){
			Word = -(Word & ~(1<<negBit));
		}
	}
	return Word;
}

int SCS::Reset(u8 ID)
{
	rFlushSCS();
	writeBuf(ID, 0, NULL, 0, INST_RESET);
	wFlushSCS();
	return Ack(ID);
}

int SCS::Recal(u8 ID)
{
	rFlushSCS();
	writeBuf(ID, 0, NULL, 0, INST_CAL);
	wFlushSCS();
	return Ack(ID);
}
