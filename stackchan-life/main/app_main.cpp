#include <esp_attr.h>

#include "robot_runtime.h"

extern "C" void app_main()
{
    // Wi-Fi (net/wifi_engine.h) and the camera each need real internal DMA-
    // capable DRAM for their own driver buffers, and together they left
    // this board's internal heap genuinely tight (~33KB free, see
    // README.md's Wi-Fi RAM caveat) - tight enough to be a real suspect for
    // reported freezes. RobotRuntime itself (all its engines' own state,
    // plus SnnCore's neuron/synapse arrays - now much larger, see
    // snn_core.h) needs none of that: nothing in it is a DMA source/
    // destination, each driver's own DMA buffer is allocated separately
    // inside its begin() with the right capability flags regardless of
    // where this object lives. EXT_RAM_BSS_ATTR (see
    // CONFIG_SPIRAM_ALLOW_BSS_SEG_EXTERNAL_MEMORY in sdkconfig.defaults)
    // moves its ~static storage into PSRAM's 8MB instead, freeing that
    // internal RAM for Wi-Fi/camera to actually use. PSRAM access is a bit
    // slower than internal DRAM, but the main loop runs at 40-55% CPU busy
    // with real headroom - not the tight resource here, internal RAM is.
    static EXT_RAM_BSS_ATTR RobotRuntime runtime;
    if (runtime.begin()) {
        runtime.loop();
    }
}
