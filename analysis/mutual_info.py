import numpy as np
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, Optional
from collect.dataset import TimingDataset

N_BOOTSTRAP = 100
MI_N_NEIGHBORS = 5


def compute_mi_total(dataset: TimingDataset,
                     n_neighbors: int = MI_N_NEIGHBORS) -> Dict[str, float]:
    arrays = dataset.to_arrays()
    timing = arrays["total_cycles"].astype(np.float64).reshape(-1, 1)
    actions = arrays["actions"]

    mi = mutual_info_classif(timing, actions, discrete_features=False,
                             n_neighbors=n_neighbors, random_state=42)
    return {
        "mi_bits": float(mi[0]) / np.log(2),
        "mi_nats": float(mi[0]),
        "max_possible_bits": float(np.log2(dataset.act_dim)),
    }


def compute_mi_per_op(dataset: TimingDataset,
                      n_neighbors: int = MI_N_NEIGHBORS) -> Dict[int, float]:
    arrays = dataset.to_arrays()
    actions = arrays["actions"]
    per_op = arrays["per_op_cycles"].astype(np.float64)

    n_ops = per_op.shape[1]
    result = {}
    for op_idx in range(n_ops):
        col = per_op[:, op_idx]
        if np.std(col) < 1e-10:
            result[op_idx] = 0.0
            continue
        mi = mutual_info_classif(col.reshape(-1, 1), actions,
                                 discrete_features=False,
                                 n_neighbors=n_neighbors, random_state=42)
        result[op_idx] = float(mi[0]) / np.log(2)
    return result


def compute_mi_bootstrap(dataset: TimingDataset,
                         n_bootstrap: int = N_BOOTSTRAP,
                         n_neighbors: int = MI_N_NEIGHBORS,
                         seed: int = 42) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    arrays = dataset.to_arrays()
    timing = arrays["total_cycles"].astype(np.float64).reshape(-1, 1)
    actions = arrays["actions"]
    n = len(actions)

    mi_samples = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        t_boot = timing[idx]
        a_boot = actions[idx]
        if len(np.unique(a_boot)) < 2:
            continue
        mi = mutual_info_classif(t_boot, a_boot, discrete_features=False,
                                 n_neighbors=n_neighbors, random_state=42)
        mi_samples.append(float(mi[0]) / np.log(2))

    mi_arr = np.array(mi_samples)
    return {
        "mi_mean": float(np.mean(mi_arr)),
        "mi_std": float(np.std(mi_arr)),
        "ci_low": float(np.percentile(mi_arr, 2.5)),
        "ci_high": float(np.percentile(mi_arr, 97.5)),
        "n_bootstrap": len(mi_samples),
    }
