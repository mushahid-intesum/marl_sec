#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

#define SYNC_BYTE_0 0xAA
#define SYNC_BYTE_1 0x55
#define UART_NUM UART_NUM_0
#define UART_BAUD 921600
#define UART_BUF_SIZE 2048
#define MAX_OBS_LEN 64

typedef struct {
    uint16_t obs_len;
    float obs[MAX_OBS_LEN];
} obs_packet_t;

typedef struct {
    uint8_t action;
    uint32_t total_cycles;
    uint8_t n_ops;
    uint32_t per_op_cycles[16];
} response_packet_t;

void protocol_init(void);
int protocol_receive_obs(obs_packet_t *pkt);
void protocol_send_response(const response_packet_t *pkt);

#endif
