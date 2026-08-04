#include "inference.h"
#include "timing.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static const tflite::Model *s_model = nullptr;
static tflite::MicroInterpreter *s_interpreter = nullptr;
static tflite::MicroProfiler s_profiler;

static tflite::MicroMutableOpResolver<3> s_resolver;

extern "C" int inference_init(void)
{
    s_model = tflite::GetModel(model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        return -1;
    }

    s_resolver.AddFullyConnected();
    s_resolver.AddRelu();
    s_resolver.AddSoftmax();

    static tflite::MicroInterpreter interpreter(
        s_model, s_resolver, tensor_arena, TENSOR_ARENA_SIZE,
        nullptr, &s_profiler);
    s_interpreter = &interpreter;

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        return -2;
    }

    return 0;
}

extern "C" int inference_run(const float *obs, int obs_len, int *action_out,
                              timing_trace_t *trace_out)
{
    TfLiteTensor *input = s_interpreter->input(0);
    if (input == nullptr) return -1;

    int input_size = input->dims->data[input->dims->size - 1];
    if (obs_len > input_size) obs_len = input_size;

    for (int i = 0; i < obs_len; i++) {
        input->data.f[i] = obs[i];
    }

    s_profiler.ClearEvents();

    uint32_t start = timing_get_cycles();
    TfLiteStatus status = s_interpreter->Invoke();
    uint32_t end = timing_get_cycles();

    if (status != kTfLiteOk) return -2;

    trace_out->total_cycles = timing_elapsed(start, end);

    s_profiler.UpdateTotalTicksPerTag();

    uint32_t num_events = s_profiler.GetNumEvents();
    trace_out->n_ops = (num_events > INFERENCE_MAX_OPS)
                           ? INFERENCE_MAX_OPS
                           : (uint8_t)num_events;

    for (uint32_t i = 0; i < trace_out->n_ops; i++) {
        uint32_t ticks = s_profiler.GetEventDuration(i);
        trace_out->per_op_cycles[i] = ticks;
    }
    for (int i = trace_out->n_ops; i < INFERENCE_MAX_OPS; i++) {
        trace_out->per_op_cycles[i] = 0;
    }

    s_profiler.ClearEvents();

    TfLiteTensor *output = s_interpreter->output(0);
    if (output == nullptr) return -3;

    int out_size = output->dims->data[output->dims->size - 1];
    int best_action = 0;
    float best_val = output->data.f[0];
    for (int i = 1; i < out_size; i++) {
        if (output->data.f[i] > best_val) {
            best_val = output->data.f[i];
            best_action = i;
        }
    }

    *action_out = best_action;
    return 0;
}

extern "C" int inference_get_input_size(void)
{
    if (s_interpreter == nullptr) return -1;
    TfLiteTensor *input = s_interpreter->input(0);
    if (input == nullptr) return -1;
    return input->dims->data[input->dims->size - 1];
}
