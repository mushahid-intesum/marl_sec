import struct
import time
import numpy as np
from typing import Optional, Tuple
from collect.dataset import TimingDataset

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 921600
TIMEOUT = 5.0
SYNC_BYTES = bytes([0xAA, 0x55])
MAX_RETRIES = 3
RETRY_DELAY = 0.1


class ProtocolEncoder:

    @staticmethod
    def encode_obs(obs: np.ndarray) -> bytes:
        obs = obs.astype(np.float32)
        obs_len = len(obs)
        pkt = SYNC_BYTES + struct.pack("<H", obs_len)
        pkt += obs.tobytes()
        return pkt

    @staticmethod
    def decode_response(data: bytes) -> Optional[dict]:
        if len(data) < 6:
            return None
        action = struct.unpack_from("<B", data, 0)[0]
        total_cycles = struct.unpack_from("<I", data, 1)[0]
        n_ops = struct.unpack_from("<B", data, 5)[0]
        expected_len = 6 + n_ops * 4
        if len(data) < expected_len:
            return None
        per_op = []
        for i in range(n_ops):
            c = struct.unpack_from("<I", data, 6 + i * 4)[0]
            per_op.append(c)
        return {
            "action": action,
            "total_cycles": total_cycles,
            "n_ops": n_ops,
            "per_op_cycles": np.array(per_op, dtype=np.uint32),
        }


class SerialBridge:

    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE,
                 timeout: float = TIMEOUT):
        import serial
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.encoder = ProtocolEncoder()
        time.sleep(0.5)
        self.ser.reset_input_buffer()

    def _wait_for_sync(self) -> bool:
        deadline = time.time() + TIMEOUT
        state = 0
        while time.time() < deadline:
            b = self.ser.read(1)
            if len(b) == 0:
                return False
            if state == 0 and b[0] == 0xAA:
                state = 1
            elif state == 1 and b[0] == 0x55:
                return True
            elif state == 1:
                state = 0
        return False

    def _read_exact(self, n: int) -> Optional[bytes]:
        data = self.ser.read(n)
        if len(data) != n:
            return None
        return data

    def send_observation(self, obs: np.ndarray) -> Optional[dict]:
        pkt = self.encoder.encode_obs(obs)
        self.ser.write(pkt)
        self.ser.flush()

        if not self._wait_for_sync():
            return None

        header = self._read_exact(6)
        if header is None:
            return None

        n_ops = header[5]
        op_data = self._read_exact(n_ops * 4) if n_ops > 0 else b""
        if op_data is None:
            return None

        return self.encoder.decode_response(header + op_data)

    def collect_dataset(self, observations: np.ndarray,
                        obs_dim: int, act_dim: int) -> TimingDataset:
        ds = TimingDataset(obs_dim=obs_dim, act_dim=act_dim)
        n = len(observations)
        for i in range(n):
            result = None
            for attempt in range(MAX_RETRIES):
                result = self.send_observation(observations[i])
                if result is not None:
                    break
                time.sleep(RETRY_DELAY)
            if result is not None:
                ds.add_sample(
                    observations[i],
                    result["action"],
                    result["total_cycles"],
                    result["per_op_cycles"],
                )
        return ds

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
