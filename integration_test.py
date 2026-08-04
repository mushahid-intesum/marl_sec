import os
import numpy as np
import torch

from envs.configs import CARTPOLE_CONFIG, GRID_NAV_CONFIG
from export.to_tflite import build_mlp, export_model
from collect.sampler import ObservationSampler
from collect.simulated_collector import SimulatedCollector
from analysis.leakage_report import generate_report, format_report_table
from analysis.plots import (plot_timing_distributions, plot_confusion_matrix,
                              plot_per_layer_breakdown)
from analysis.mutual_info import compute_mi_per_op

OUTPUT_DIR = "models"
FIGURE_DIR = "figures"
DATA_DIR = "data"
N_SAMPLES = 2000
SEED = 42


def run_smoke_test():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    configs = [
        ("cartpole", CARTPOLE_CONFIG),
        ("grid_nav", GRID_NAV_CONFIG),
    ]

    reports = []

    for env_name, config in configs:
        print(f"\n{'='*60}")
        print(f"  {env_name.upper()}: obs_dim={config.obs_dim}, act_dim={config.act_dim}")
        print(f"{'='*60}")

        torch.manual_seed(SEED)
        model = build_mlp(config.obs_dim, config.act_dim, config.hidden_sizes)
        print(f"  [1/5] Built MLP: {config.obs_dim} -> {config.hidden_sizes} -> {config.act_dim}")

        result = export_model(model, config, OUTPUT_DIR, env_name)
        print(f"  [2/5] Exported TFLite: FP32={result['fp32_size']}B, INT8={result['int8_size']}B")

        sampler = ObservationSampler(config)
        obs = sampler.uniform(N_SAMPLES, seed=SEED)
        print(f"  [3/5] Generated {N_SAMPLES} observations")

        for quant in ["fp32", "int8"]:
            tflite_path = result[f"{quant}_tflite"]
            collector = SimulatedCollector(tflite_path, config)
            dataset = collector.collect(obs)
            label = f"{env_name}_{quant}"

            ds_path = os.path.join(DATA_DIR, f"{label}.npz")
            dataset.save(ds_path)
            print(f"  [4/5] Collected {len(dataset)} timing traces ({quant})")

            report = generate_report(dataset, label=label)
            reports.append(report)

            mi_bits = report["mutual_information"]["mi_bits"]
            rf_acc = report["rf_classifier"]["accuracy"]
            kw_p = report["kruskal_wallis"]["p_value"]
            print(f"  [5/5] Analysis: MI={mi_bits:.3f} bits, RF_acc={rf_acc:.3f}, KW_p={kw_p:.2e}")

            fig = plot_timing_distributions(
                dataset, title=f"Timing Distribution: {label}",
                save_path=os.path.join(FIGURE_DIR, f"{label}_violin.png"))
            fig.clear()

            cm = np.array(report["rf_confusion_matrix"])
            fig = plot_confusion_matrix(
                cm, title=f"RF Confusion: {label}",
                save_path=os.path.join(FIGURE_DIR, f"{label}_confusion.png"))
            fig.clear()

            fig = plot_per_layer_breakdown(
                dataset, n_ops=5, title=f"Per-Layer: {label}",
                save_path=os.path.join(FIGURE_DIR, f"{label}_layers.png"))
            fig.clear()

    print(f"\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"{'='*60}\n")
    print(format_report_table(reports))

    print(f"\nFigures saved to: {os.path.abspath(FIGURE_DIR)}/")
    print(f"Datasets saved to: {os.path.abspath(DATA_DIR)}/")
    print(f"Models saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nSmoke test PASSED")

    return reports


if __name__ == "__main__":
    run_smoke_test()
