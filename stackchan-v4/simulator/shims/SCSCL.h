#pragma once
// Host shim for the Feetech SCSCL servo bus: position writes go to the
// simulated servo mechanics (simulator/world).
#include <cstdint>
#define UART_NUM_1 1
#define SCSCL_MIN_ANGLE_LIMIT_L 9
#define SCSCL_MAX_ANGLE_LIMIT_L 11
class SCSCL {
public:
    int Level = 1;
    uint8_t getState() { return 0; }
    int writeWord(uint8_t, uint8_t, uint16_t) { return 1; }
    int readWord(uint8_t, uint8_t) { return -1; }
    bool begin(int uart, int baud, int tx, int rx);
    int WritePos(uint8_t id, int16_t position, uint16_t time_ms, uint16_t speed);
    int EnableTorque(uint8_t id, uint8_t enable);
    void SyncWritePos(uint8_t ID[], uint8_t IDN, uint16_t Position[], uint16_t Time[], uint16_t Speed[])
    {
        for (int i = 0; i < IDN; ++i) WritePos(ID[i], static_cast<int16_t>(Position[i]), Time ? Time[i] : 0, Speed ? Speed[i] : 0);
    }
    // Read-back is not modelled (the robot's diagnostics use it).
    int ReadPos(int) { return -1; }
    int ReadLoad(int) { return -1; }
    int ReadTemper(int) { return -1; }
    int ReadVoltage(int) { return -1; }
    int ReadToqueEnable(int) { return -1; }
};
