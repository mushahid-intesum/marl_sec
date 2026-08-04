#include "timing.h"
#include "esp_cpu.h"

uint32_t timing_get_cycles(void)
{
    return esp_cpu_get_cycle_count();
}

uint32_t timing_cycles_to_us(uint32_t cycles)
{
    return cycles / CPU_FREQ_MHZ;
}

uint32_t timing_elapsed(uint32_t start, uint32_t end)
{
    return end - start;
}
