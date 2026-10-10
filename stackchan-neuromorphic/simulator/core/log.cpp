#include "log.h"

#include <cctype>

#include "vrtos.h"

namespace sim {
namespace {
FILE* g_file = nullptr;
bool g_echo = false;
std::map<std::string, int> g_counts;
}  // namespace

void setLogFile(FILE* file) { g_file = file; }
void setLogEcho(bool echo) { g_echo = echo; }
const std::map<std::string, int>& logCounts() { return g_counts; }
void resetLogCounts() { g_counts.clear(); }

void logLine(char level, const char* tag, const std::string& message)
{
    const unsigned long long ms = vrtos::nowUs() / 1000;
    if (g_file) std::fprintf(g_file, "%c (%llu) %s: %s\n", level, ms, tag, message.c_str());
    if (g_echo) std::printf("%c (%llu) %s: %s\n", level, ms, tag, message.c_str());
    // Key: tag + message up to the first number, e.g. "neural_output: emotion=".
    std::string key = std::string(tag) + ": ";
    for (char c : message) {
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '(') break;
        key += c;
        if (c == '=') break;
    }
    ++g_counts[key];
}

}  // namespace sim
