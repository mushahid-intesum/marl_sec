#include "protocol.h"
#include "driver/uart.h"
#include <string.h>

void protocol_init(void)
{
    uart_config_t cfg = {
        .baud_rate = UART_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    uart_param_config(UART_NUM, &cfg);
    uart_driver_install(UART_NUM, UART_BUF_SIZE, UART_BUF_SIZE, 0, NULL, 0);
}

static int read_exact(uint8_t *buf, int len)
{
    int received = 0;
    while (received < len) {
        int n = uart_read_bytes(UART_NUM, buf + received, len - received, pdMS_TO_TICKS(5000));
        if (n <= 0) {
            return -1;
        }
        received += n;
    }
    return 0;
}

int protocol_receive_obs(obs_packet_t *pkt)
{
    uint8_t sync[2];

    while (1) {
        if (read_exact(&sync[0], 1) != 0) return -1;
        if (sync[0] != SYNC_BYTE_0) continue;
        if (read_exact(&sync[1], 1) != 0) return -1;
        if (sync[1] == SYNC_BYTE_1) break;
    }

    uint8_t len_buf[2];
    if (read_exact(len_buf, 2) != 0) return -1;
    pkt->obs_len = (uint16_t)(len_buf[0] | (len_buf[1] << 8));

    if (pkt->obs_len > MAX_OBS_LEN) return -1;

    int data_bytes = pkt->obs_len * sizeof(float);
    if (read_exact((uint8_t *)pkt->obs, data_bytes) != 0) return -1;

    return 0;
}

void protocol_send_response(const response_packet_t *pkt)
{
    uint8_t sync[2] = {SYNC_BYTE_0, SYNC_BYTE_1};
    uart_write_bytes(UART_NUM, sync, 2);
    uart_write_bytes(UART_NUM, &pkt->action, 1);
    uart_write_bytes(UART_NUM, (const uint8_t *)&pkt->total_cycles, 4);
    uart_write_bytes(UART_NUM, &pkt->n_ops, 1);
    int op_bytes = pkt->n_ops * sizeof(uint32_t);
    uart_write_bytes(UART_NUM, (const uint8_t *)pkt->per_op_cycles, op_bytes);
}
