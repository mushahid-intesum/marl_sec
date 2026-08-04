import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from collect.dataset import TimingDataset


def timing_kruskal_wallis(dataset: TimingDataset) -> Dict[str, float]:
    groups = dataset.get_timing_by_action()
    if len(groups) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "n_groups": len(groups)}
    group_arrays = [groups[a] for a in sorted(groups.keys())]
    stat, p = stats.kruskal(*group_arrays)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_groups": len(groups),
        "significant_005": p < 0.05,
        "significant_001": p < 0.01,
    }


def timing_anova(dataset: TimingDataset) -> Dict[str, float]:
    groups = dataset.get_timing_by_action()
    if len(groups) < 2:
        return {"f_statistic": 0.0, "p_value": 1.0, "n_groups": len(groups)}
    group_arrays = [groups[a] for a in sorted(groups.keys())]
    f_stat, p = stats.f_oneway(*group_arrays)
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p),
        "n_groups": len(groups),
    }


def timing_spearman(dataset: TimingDataset) -> Dict[str, Dict[str, float]]:
    arrays = dataset.to_arrays()
    timing = arrays["total_cycles"].astype(np.float64)
    obs = arrays["observations"]
    result = {}
    for dim in range(obs.shape[1]):
        corr, p = stats.spearmanr(obs[:, dim], timing)
        result[f"obs_dim_{dim}"] = {
            "correlation": float(corr),
            "p_value": float(p),
            "significant": p < 0.05,
        }
    return result


def timing_cohens_d(dataset: TimingDataset) -> Dict[str, float]:
    groups = dataset.get_timing_by_action()
    if len(groups) < 2:
        return {}
    actions = sorted(groups.keys())
    result = {}
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            a_i, a_j = actions[i], actions[j]
            g1, g2 = groups[a_i], groups[a_j]
            n1, n2 = len(g1), len(g2)
            if n1 < 2 or n2 < 2:
                continue
            pooled_std = np.sqrt(((n1 - 1) * np.var(g1, ddof=1) +
                                  (n2 - 1) * np.var(g2, ddof=1)) /
                                 (n1 + n2 - 2))
            if pooled_std < 1e-10:
                d = 0.0
            else:
                d = (np.mean(g1) - np.mean(g2)) / pooled_std
            result[f"action_{a_i}_vs_{a_j}"] = float(d)
    return result


def timing_summary_stats(dataset: TimingDataset) -> Dict[str, Dict[str, float]]:
    groups = dataset.get_timing_by_action()
    result = {}
    for action, cycles in groups.items():
        result[f"action_{action}"] = {
            "mean": float(np.mean(cycles)),
            "std": float(np.std(cycles)),
            "median": float(np.median(cycles)),
            "min": float(np.min(cycles)),
            "max": float(np.max(cycles)),
            "count": int(len(cycles)),
            "cv": float(np.std(cycles) / np.mean(cycles)) if np.mean(cycles) > 0 else 0.0,
        }
    return result
