/*
 * SCSCL.h
 * Feetech SCSCL series serial servo application layer implementation
 * Date: 2024.4.2
 * Author: 
 */

#ifndef _SCSCL_H
#define _SCSCL_H

//Memory table definitions.
//-------EPROM (read-only)--------
#define SCSCL_VERSION_L 3
#define SCSCL_VERSION_H 4

//-------EPROM (read/write)--------
#define SCSCL_ID 5
#define SCSCL_BAUD_RATE 6
#define SCSCL_MIN_ANGLE_LIMIT_L 9
#define SCSCL_MIN_ANGLE_LIMIT_H 10
#define SCSCL_MAX_ANGLE_LIMIT_L 11
#define SCSCL_MAX_ANGLE_LIMIT_H 12
#define SCSCL_CW_DEAD 26
#define SCSCL_CCW_DEAD 27

//-------SRAM (read/write)--------
#define SCSCL_TORQUE_ENABLE 40
#define SCSCL_GOAL_POSITION_L 42
#define SCSCL_GOAL_POSITION_H 43
#define SCSCL_GOAL_TIME_L 44
#define SCSCL_GOAL_TIME_H 45
#define SCSCL_GOAL_SPEED_L 46
#define SCSCL_GOAL_SPEED_H 47
#define SCSCL_LOCK 48

//-------SRAM (read-only)--------
#define SCSCL_PRESENT_POSITION_L 56
#define SCSCL_PRESENT_POSITION_H 57
#define SCSCL_PRESENT_SPEED_L 58
#define SCSCL_PRESENT_SPEED_H 59
#define SCSCL_PRESENT_LOAD_L 60
#define SCSCL_PRESENT_LOAD_H 61
#define SCSCL_PRESENT_VOLTAGE 62
#define SCSCL_PRESENT_TEMPERATURE 63
#define SCSCL_MOVING 66
#define SCSCL_PRESENT_CURRENT_L 69
#define SCSCL_PRESENT_CURRENT_H 70

#include "SCSerial.h"

class SCSCL : public SCSerial
{
public:
	SCSCL();
	SCSCL(u8 End);
	SCSCL(u8 End, u8 Level);
	int WritePos(u8 ID, u16 Position, u16 Time, u16 Speed = 0);//Regular write position instruction for one servo.
	int RegWritePos(u8 ID, u16 Position, u16 Time, u16 Speed = 0);//Asynchronous write position instruction for one servo (applied by RegWriteAction).
	void SyncWritePos(u8 ID[], u8 IDN, u16 Position[], u16 Time[], u16 Speed[]);//Synchronous write position instruction for multiple servos.
	int PWMMode(u8 ID);//PWM mode.
	int WritePWM(u8 ID, s16 pwmOut);//PWM output mode instruction.
	int SwitchMode(int ID, u8 mode);//Switch between position mode and PWM mode.
	int EnableTorque(u8 ID, u8 Enable);//Torque control instruction.
	int unLockEprom(u8 ID);//Unlock EEPROM.
	int LockEprom(u8 ID);//Lock EEPROM.
	int FeedBack(int ID);//Read servo feedback information.
	int ReadPos(int ID);//Read position.
	int ReadSpeed(int ID);//Read speed.
	int ReadLoad(int ID);//Read the voltage percentage output to the motor (0-1000).
	int ReadVoltage(int ID);//Read voltage.
	int ReadTemper(int ID);//Read temperature.
	int ReadMove(int ID);//Read movement status.
	int ReadCurrent(int ID);//Read current.
	int ReadToqueEnable(int ID);//Read torque enable status.
private:
	u8 Mem[SCSCL_PRESENT_CURRENT_H-SCSCL_PRESENT_POSITION_L+1];
	int min_angle[256] = {0};
	int max_angle[256] = {0};
	bool angle_limit_cached[256] = {false};
};

#endif
