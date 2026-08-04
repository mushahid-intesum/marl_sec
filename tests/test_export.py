import os
import tempfile
import numpy as np
import pytest
import torch
from export.to_tflite import (
    build_mlp, export_model, verify_export, tflite_to_c_header,
    _extract_weights, _build_keras_model, _convert_fp32,
)
from envs.configs import GRID_NAV_CONFIG, CARTPOLE_CONFIG, SIMPLE_SPREAD_CONFIG


class TestBuildMLP:

    def test_output_shape_grid_nav(self):
        model = build_mlp(9, 5, [32, 32])
        x = torch.randn(1, 9)
        y = model(x)
        assert y.shape == (1, 5)

    def test_output_shape_cartpole(self):
        model = build_mlp(4, 2, [32, 32])
        x = torch.randn(1, 4)
        y = model(x)
        assert y.shape == (1, 2)

    def test_output_shape_spread(self):
        model = build_mlp(18, 5, [64, 64])
        x = torch.randn(1, 18)
        y = model(x)
        assert y.shape == (1, 5)

    def test_deterministic(self):
        model = build_mlp(9, 5, [32, 32])
        model.eval()
        x = torch.randn(1, 9)
        with torch.no_grad():
            y1 = model(x).numpy()
            y2 = model(x).numpy()
        np.testing.assert_array_equal(y1, y2)


class TestExtractWeights:

    def test_correct_count(self):
        model = build_mlp(9, 5, [32, 32])
        weights = _extract_weights(model)
        assert len(weights) == 3

    def test_shapes(self):
        model = build_mlp(9, 5, [32, 32])
        weights = _extract_weights(model)
        assert weights[0][0].shape == (32, 9)
        assert weights[0][1].shape == (32,)
        assert weights[1][0].shape == (32, 32)
        assert weights[1][1].shape == (32,)
        assert weights[2][0].shape == (5, 32)
        assert weights[2][1].shape == (5,)


class TestKerasReconstruction:

    def test_fp32_output_matches_pytorch(self):
        torch.manual_seed(42)
        model = build_mlp(9, 5, [32, 32])
        model.eval()

        weights = _extract_weights(model)
        keras_model = _build_keras_model(9, 5, [32, 32], weights)

        x_np = np.random.randn(10, 9).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        with torch.no_grad():
            pt_out = model(x_pt).numpy()
        keras_out = keras_model.predict(x_np, verbose=0)

        np.testing.assert_allclose(pt_out, keras_out, atol=1e-5)


class TestExportModel:

    def test_produces_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = build_mlp(9, 5, [32, 32])
            result = export_model(model, GRID_NAV_CONFIG, tmpdir, "test_model")
            assert os.path.exists(result["fp32_tflite"])
            assert os.path.exists(result["fp32_header"])
            assert os.path.exists(result["int8_tflite"])
            assert os.path.exists(result["int8_header"])

    def test_tflite_files_not_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = build_mlp(9, 5, [32, 32])
            result = export_model(model, GRID_NAV_CONFIG, tmpdir, "test_model")
            assert result["fp32_size"] > 0
            assert result["int8_size"] > 0

    def test_int8_smaller_than_fp32(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = build_mlp(9, 5, [32, 32])
            result = export_model(model, GRID_NAV_CONFIG, tmpdir, "test_model")
            assert result["int8_size"] <= result["fp32_size"]

    def test_cartpole_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = build_mlp(4, 2, [32, 32])
            result = export_model(model, CARTPOLE_CONFIG, tmpdir, "cartpole")
            assert os.path.exists(result["fp32_tflite"])

    def test_spread_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = build_mlp(18, 5, [64, 64])
            result = export_model(model, SIMPLE_SPREAD_CONFIG, tmpdir, "spread")
            assert os.path.exists(result["fp32_tflite"])


class TestVerifyExport:

    def test_fp32_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.manual_seed(42)
            model = build_mlp(9, 5, [32, 32])
            result = export_model(model, GRID_NAV_CONFIG, tmpdir, "test")
            verify = verify_export(model, result["fp32_tflite"],
                                   GRID_NAV_CONFIG, n_samples=50, atol=1e-4)
            assert verify["passed"]
            assert verify["max_diff"] < 1e-4

    def test_int8_reasonable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            torch.manual_seed(42)
            model = build_mlp(9, 5, [32, 32])
            result = export_model(model, GRID_NAV_CONFIG, tmpdir, "test")
            verify = verify_export(model, result["int8_tflite"],
                                   GRID_NAV_CONFIG, n_samples=50, atol=1.0)
            assert verify["action_mismatch_rate"] < 0.5


class TestCHeader:

    def test_header_format(self):
        data = b"\x00\x01\x02\x03"
        header = tflite_to_c_header(data, "test_model")
        assert "test_model" in header
        assert "0x00" in header
        assert "#ifndef" in header
        assert "#endif" in header

    def test_header_length_constant(self):
        data = b"\xaa\xbb\xcc"
        header = tflite_to_c_header(data, "my_model")
        assert "my_model_len = 3" in header
