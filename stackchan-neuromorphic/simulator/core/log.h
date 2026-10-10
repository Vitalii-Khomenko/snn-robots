#pragma once
// One log for firmware ESP_LOGx output and simulator notes, in the robot's
// UART format: "I (12345) tag: message" with the virtual time in ms.

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>

namespace sim {

void setLogFile(FILE* file);            // nullptr = discard
void setLogEcho(bool echo);             // also print to stdout
void logLine(char level, const char* tag, const std::string& message);

// Counts of "tag: message-prefix" occurrences (message up to the first
// digit or '='), used for the run summary. Reset per summary window.
const std::map<std::string, int>& logCounts();
void resetLogCounts();

}  // namespace sim
