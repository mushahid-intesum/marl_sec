import numpy as np
import time
from typing import Optional
from collect.dataset import TimingDataset
from envs.configs import EnvConfig

NOISE_SCALE = 50
BASE_CYCLES = 12000
ACTION_BIAS = 200


class SimulatedCollector:

    def __init__(self, tflite_path: str, config: EnvConfig):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.config = config

    def _run_inference(self, obs: np.ndarray) -> dict:
        inp = obs.reshape(1, -1).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]["index"], inp)

        start = time.perf_counter_ns()
        self.interpreter.invoke()
        end = time.perf_counter_ns()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        action = int(np.argmax(output, axis=1)[0])

        real_ns = end - start

        simulated_cycles = BASE_CYCLES + action * ACTION_BIAS
        simulated_cycles += int(np.sum(obs * 100))
        simulated_cycles += np.random.randint(0, NOISE_SCALE)

        n_ops = 5
        per_op = np.zeros(n_ops, dtype=np.uint32)
        per_op[0] = int(simulated_cycles * 0.30) + action * 20
        per_op[1] = int(simulated_cycles * 0.05)
        per_op[2] = int(simulated_cycles * 0.35) + action * 30
        per_op[3] = int(simulated_cycles * 0.05)
        per_op[4] = int(simulated_cycles * 0.25) + action * 15

        return {
            "action": action,
            "total_cycles": simulated_cycles,
            "per_op_cycles": per_op,
            "real_ns": real_ns,
        }

    def collect(self, observations: np.ndarray) -> TimingDataset:
        ds = TimingDataset(obs_dim=self.config.obs_dim, act_dim=self.config.act_dim)
        for i in range(len(observations)):
            result = self._run_inference(observations[i])
            ds.add_sample(
                observations[i],
                result["action"],
                result["total_cycles"],
                result["per_op_cycles"],
            )
        return ds
