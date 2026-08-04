#ifndef TIMING_H
#define TIMING_H

#include <stdint.h>

#define CPU_FREQ_MHZ 240

uint32_t timing_get_cycles(void);
uint32_t timing_cycles_to_us(uint32_t cycles);
uint32_t timing_elapsed(uint32_t start, uint32_t end);

#endif
