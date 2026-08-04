#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>

#define INFERENCE_MAX_OPS 16
#define TENSOR_ARENA_SIZE (32 * 1024)

typedef struct {
    uint32_t total_cycles;
    uint8_t n_ops;
    uint32_t per_op_cycles[INFERENCE_MAX_OPS];
} timing_trace_t;

#ifdef __cplusplus
extern "C" {
#endif

int inference_init(void);
int inference_run(const float *obs, int obs_len, int *action_out,
                  timing_trace_t *trace_out);
int inference_get_input_size(void);

#ifdef __cplusplus
}
#endif

#endif
