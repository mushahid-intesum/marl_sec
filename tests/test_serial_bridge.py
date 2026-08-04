import struct
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from collect.serial_bridge import ProtocolEncoder, SerialBridge, SYNC_BYTES


class TestProtocolEncoder:

    def test_encode_obs_sync_bytes(self):
        obs = np.array([0.5, 0.3], dtype=np.float32)
        pkt = ProtocolEncoder.encode_obs(obs)
        assert pkt[:2] == SYNC_BYTES

    def test_encode_obs_length(self):
        obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        pkt = ProtocolEncoder.encode_obs(obs)
        obs_len = struct.unpack_from("<H", pkt, 2)[0]
        assert obs_len == 4

    def test_encode_obs_values(self):
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pkt = ProtocolEncoder.encode_obs(obs)
        for i in range(3):
            val = struct.unpack_from("<f", pkt, 4 + i * 4)[0]
            assert val == pytest.approx(obs[i])

    def test_encode_obs_total_size(self):
        obs = np.array([0.5] * 9, dtype=np.float32)
        pkt = ProtocolEncoder.encode_obs(obs)
        assert len(pkt) == 2 + 2 + 9 * 4

    def test_encode_single_obs(self):
        obs = np.array([0.42], dtype=np.float32)
        pkt = ProtocolEncoder.encode_obs(obs)
        assert len(pkt) == 2 + 2 + 4

    def test_decode_response_basic(self):
        action = 2
        total_cycles = 15000
        n_ops = 3
        per_op = [5000, 6000, 4000]
        data = struct.pack("<B", action)
        data += struct.pack("<I", total_cycles)
        data += struct.pack("<B", n_ops)
        for c in per_op:
            data += struct.pack("<I", c)
        result = ProtocolEncoder.decode_response(data)
        assert result["action"] == 2
        assert result["total_cycles"] == 15000
        assert result["n_ops"] == 3
        np.testing.assert_array_equal(result["per_op_cycles"],
                                       np.array(per_op, dtype=np.uint32))

    def test_decode_response_zero_ops(self):
        data = struct.pack("<B", 1)
        data += struct.pack("<I", 8000)
        data += struct.pack("<B", 0)
        result = ProtocolEncoder.decode_response(data)
        assert result["action"] == 1
        assert result["n_ops"] == 0
        assert len(result["per_op_cycles"]) == 0

    def test_decode_response_too_short(self):
        data = b"\x00\x01\x02"
        result = ProtocolEncoder.decode_response(data)
        assert result is None

    def test_decode_response_truncated_ops(self):
        data = struct.pack("<B", 0)
        data += struct.pack("<I", 1000)
        data += struct.pack("<B", 5)
        data += struct.pack("<I", 100)
        result = ProtocolEncoder.decode_response(data)
        assert result is None

    def test_encode_decode_roundtrip(self):
        obs = np.random.rand(9).astype(np.float32)
        pkt = ProtocolEncoder.encode_obs(obs)
        obs_len = struct.unpack_from("<H", pkt, 2)[0]
        assert obs_len == 9
        decoded = []
        for i in range(obs_len):
            decoded.append(struct.unpack_from("<f", pkt, 4 + i * 4)[0])
        np.testing.assert_array_almost_equal(obs, np.array(decoded, dtype=np.float32))


class TestProtocolCompatibility:

    def test_matches_c_firmware_format(self):
        obs = np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float32)
        pkt = ProtocolEncoder.encode_obs(obs)
        assert pkt[0] == 0xAA
        assert pkt[1] == 0x55
        length_le = struct.unpack("<H", pkt[2:4])[0]
        assert length_le == 4
        for i in range(4):
            val = struct.unpack("<f", pkt[4 + i * 4:4 + (i + 1) * 4])[0]
            assert val == pytest.approx(obs[i])

    def test_response_matches_c_firmware_format(self):
        resp_bytes = struct.pack("<B", 3)
        resp_bytes += struct.pack("<I", 24000)
        resp_bytes += struct.pack("<B", 2)
        resp_bytes += struct.pack("<I", 10000)
        resp_bytes += struct.pack("<I", 14000)
        result = ProtocolEncoder.decode_response(resp_bytes)
        assert result["action"] == 3
        assert result["total_cycles"] == 24000
        assert result["n_ops"] == 2
        assert result["per_op_cycles"][0] == 10000
        assert result["per_op_cycles"][1] == 14000


class TestSerialBridgeMocked:

    def _make_response_bytes(self, action, total_cycles, per_op):
        n_ops = len(per_op)
        data = SYNC_BYTES
        data += struct.pack("<B", action)
        data += struct.pack("<I", total_cycles)
        data += struct.pack("<B", n_ops)
        for c in per_op:
            data += struct.pack("<I", c)
        return data

    def _make_bridge(self, response_bytes):
        bridge = SerialBridge.__new__(SerialBridge)
        mock_ser = MagicMock()
        bridge.ser = mock_ser
        bridge.encoder = ProtocolEncoder()

        buf = bytearray(response_bytes)
        pos = [0]

        def mock_read(n):
            chunk = bytes(buf[pos[0]:pos[0] + n])
            pos[0] += n
            return chunk

        mock_ser.read = mock_read
        mock_ser.write = MagicMock()
        mock_ser.flush = MagicMock()
        return bridge

    def test_send_observation_success(self):
        resp = self._make_response_bytes(1, 10000, [3000, 4000, 3000])
        bridge = self._make_bridge(resp)

        obs = np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float32)
        result = bridge.send_observation(obs)

        assert result is not None
        assert result["action"] == 1
        assert result["total_cycles"] == 10000
        assert result["n_ops"] == 3

    def test_send_observation_timeout(self):
        bridge = SerialBridge.__new__(SerialBridge)
        mock_ser = MagicMock()
        bridge.ser = mock_ser
        bridge.encoder = ProtocolEncoder()
        mock_ser.read.return_value = b""
        mock_ser.write = MagicMock()
        mock_ser.flush = MagicMock()

        obs = np.array([0.5], dtype=np.float32)
        result = bridge.send_observation(obs)

        assert result is None

    def test_send_observation_zero_ops(self):
        resp = self._make_response_bytes(2, 5000, [])
        bridge = self._make_bridge(resp)

        obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        result = bridge.send_observation(obs)

        assert result is not None
        assert result["action"] == 2
        assert result["n_ops"] == 0


class TestMultipleObservations:

    def test_batch_encoding(self):
        obs_batch = np.random.rand(10, 9).astype(np.float32)
        packets = [ProtocolEncoder.encode_obs(obs) for obs in obs_batch]
        assert len(packets) == 10
        for pkt in packets:
            assert pkt[:2] == SYNC_BYTES
            assert len(pkt) == 2 + 2 + 9 * 4

    def test_various_obs_dims(self):
        for dim in [4, 9, 18]:
            obs = np.random.rand(dim).astype(np.float32)
            pkt = ProtocolEncoder.encode_obs(obs)
            decoded_len = struct.unpack_from("<H", pkt, 2)[0]
            assert decoded_len == dim
