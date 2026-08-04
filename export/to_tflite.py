import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from envs.configs import EnvConfig

NUM_CALIBRATION_SAMPLES = 200


def build_mlp(obs_dim: int, act_dim: int, hidden_sizes: List[int]) -> nn.Sequential:
    layers = []
    prev = obs_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, act_dim))
    return nn.Sequential(*layers)


def _extract_weights(model: nn.Module) -> List[tuple]:
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            weights.append((w, b))
    return weights


def _build_keras_model(obs_dim: int, act_dim: int, hidden_sizes: List[int],
                       weights: List[tuple]):
    import tensorflow as tf

    layers = []
    for i, h in enumerate(hidden_sizes):
        if i == 0:
            layers.append(tf.keras.layers.Dense(h, activation="relu",
                                                input_shape=(obs_dim,)))
        else:
            layers.append(tf.keras.layers.Dense(h, activation="relu"))
    layers.append(tf.keras.layers.Dense(act_dim))

    model = tf.keras.Sequential(layers)
    model.build((None, obs_dim))

    weight_idx = 0
    for layer in model.layers:
        if hasattr(layer, "kernel"):
            w, b = weights[weight_idx]
            layer.set_weights([w.T, b])
            weight_idx += 1

    return model


def _convert_fp32(keras_model) -> bytes:
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    return converter.convert()


def _convert_int8(keras_model, representative_data: np.ndarray) -> bytes:
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_dataset():
        for i in range(len(representative_data)):
            yield [representative_data[i:i + 1].astype(np.float32)]

    converter.representative_dataset = rep_dataset
    return converter.convert()


def _generate_calibration_data(config: EnvConfig,
                               n_samples: int = NUM_CALIBRATION_SAMPLES) -> np.ndarray:
    samples = np.random.uniform(
        config.obs_low, config.obs_high, size=(n_samples, config.obs_dim)
    ).astype(np.float32)
    return samples


def tflite_to_c_header(tflite_bytes: bytes, var_name: str = "model_data") -> str:
    lines = []
    lines.append(f"#ifndef {var_name.upper()}_H")
    lines.append(f"#define {var_name.upper()}_H")
    lines.append("")
    lines.append(f"const unsigned char {var_name}[] = {{")
    hex_vals = [f"0x{b:02x}" for b in tflite_bytes]
    for i in range(0, len(hex_vals), 12):
        chunk = ", ".join(hex_vals[i:i + 12])
        lines.append(f"  {chunk},")
    lines.append("};")
    lines.append(f"const unsigned int {var_name}_len = {len(tflite_bytes)};")
    lines.append("")
    lines.append(f"#endif")
    return "\n".join(lines)


def export_model(pytorch_model: nn.Module, config: EnvConfig,
                 output_dir: str, model_name: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    pytorch_model.eval()
    weights = _extract_weights(pytorch_model)
    keras_model = _build_keras_model(config.obs_dim, config.act_dim,
                                     config.hidden_sizes, weights)

    fp32_bytes = _convert_fp32(keras_model)
    fp32_path = os.path.join(output_dir, f"{model_name}_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(fp32_bytes)

    fp32_header = tflite_to_c_header(fp32_bytes, f"{model_name}_fp32")
    fp32_header_path = os.path.join(output_dir, f"{model_name}_fp32.h")
    with open(fp32_header_path, "w") as f:
        f.write(fp32_header)

    cal_data = _generate_calibration_data(config)
    int8_bytes = _convert_int8(keras_model, cal_data)
    int8_path = os.path.join(output_dir, f"{model_name}_int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(int8_bytes)

    int8_header = tflite_to_c_header(int8_bytes, f"{model_name}_int8")
    int8_header_path = os.path.join(output_dir, f"{model_name}_int8.h")
    with open(int8_header_path, "w") as f:
        f.write(int8_header)

    return {
        "fp32_tflite": fp32_path,
        "fp32_header": fp32_header_path,
        "int8_tflite": int8_path,
        "int8_header": int8_header_path,
        "fp32_size": len(fp32_bytes),
        "int8_size": len(int8_bytes),
    }


def verify_export(pytorch_model: nn.Module, tflite_path: str,
                  config: EnvConfig, n_samples: int = 100,
                  atol: float = 1e-5) -> dict:
    import tensorflow as tf

    pytorch_model.eval()
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_inputs = _generate_calibration_data(config, n_samples)
    max_diff = 0.0
    mismatched_actions = 0

    for i in range(n_samples):
        inp = test_inputs[i:i + 1]

        with torch.no_grad():
            pt_out = pytorch_model(torch.from_numpy(inp)).numpy()

        interpreter.set_tensor(input_details[0]["index"], inp.astype(np.float32))
        interpreter.invoke()
        tf_out = interpreter.get_tensor(output_details[0]["index"])

        diff = np.abs(pt_out - tf_out).max()
        max_diff = max(max_diff, diff)

        pt_action = np.argmax(pt_out, axis=1)
        tf_action = np.argmax(tf_out, axis=1)
        if pt_action[0] != tf_action[0]:
            mismatched_actions += 1

    return {
        "max_diff": float(max_diff),
        "action_mismatch_rate": mismatched_actions / n_samples,
        "passed": max_diff < atol,
    }
