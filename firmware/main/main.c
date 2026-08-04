#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "inference.h"
#include "protocol.h"
#include "timing.h"

void app_main(void)
{
    protocol_init();

    printf("timing_sca: initializing inference engine\n");

    int rc = inference_init();
    if (rc != 0) {
        printf("timing_sca: inference init failed (%d)\n", rc);
        return;
    }

    int input_size = inference_get_input_size();
    printf("timing_sca: ready, input_size=%d\n", input_size);

    obs_packet_t obs_pkt;
    response_packet_t resp_pkt;
    timing_trace_t trace;

    while (1) {
        if (protocol_receive_obs(&obs_pkt) != 0) {
            continue;
        }

        int action = 0;
        int result = inference_run(obs_pkt.obs, obs_pkt.obs_len, &action, &trace);

        if (result != 0) {
            resp_pkt.action = 0xFF;
            resp_pkt.total_cycles = 0;
            resp_pkt.n_ops = 0;
        } else {
            resp_pkt.action = (uint8_t)action;
            resp_pkt.total_cycles = trace.total_cycles;
            resp_pkt.n_ops = trace.n_ops;
            for (int i = 0; i < trace.n_ops; i++) {
                resp_pkt.per_op_cycles[i] = trace.per_op_cycles[i];
            }
        }

        protocol_send_response(&resp_pkt);
    }
}
