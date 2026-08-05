import os
import numpy as np

from collect.serial_bridge import SerialBridge
from collect.sampler import ObservationSampler
from collect.simulated_collector import SimulatedCollector
from collect.dataset import TimingDataset
from analysis.leakage_report import generate_report, format_report_table
from analysis.plots import (plot_timing_distributions, plot_confusion_matrix,
                              plot_per_layer_breakdown, plot_mi_heatmap)
from export.to_tflite import export_model
from envs.configs import CARTPOLE_CONFIG, GRID_NAV_CONFIG

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 921600
N_SAMPLES = 2000
SEED = 42

DATA_DIR = "data"
FIGURE_DIR = "figures"
MODEL_DIR = "models"

CONFIGS = {
    "cartpole": {
        "config": CARTPOLE_CONFIG,
        "tflite": "models/cartpole_fp32.tflite",
        "enabled": True,
    },
    "grid_nav_agent_0": {
        "config": GRID_NAV_CONFIG,
        "tflite": "models/grid_nav_agent_0_fp32.tflite",
        "enabled": True,
    },
}

COLLECT_HARDWARE = True
COLLECT_SIMULATED = True


def _collect_simulated(tflite_path, config, n_samples, obs):
    collector = SimulatedCollector(tflite_path, config)
    return collector.collect(obs)


def _collect_hardware(config, obs):
    with SerialBridge(port=SERIAL_PORT, baud=BAUD_RATE) as bridge:
        return bridge.collect_dataset(obs, config.obs_dim, config.act_dim)


def _analyze_and_plot(dataset, label):
    report = generate_report(dataset, label=label)

    plot_timing_distributions(
        dataset, title=f"Timing: {label}",
        save_path=os.path.join(FIGURE_DIR, f"{label}_violin.png")).clear()
    cm = np.array(report["rf_confusion_matrix"])
    plot_confusion_matrix(
        cm, title=f"RF: {label}",
        save_path=os.path.join(FIGURE_DIR, f"{label}_confusion.png")).clear()
    plot_per_layer_breakdown(
        dataset, n_ops=5, title=f"Layers: {label}",
        save_path=os.path.join(FIGURE_DIR, f"{label}_layers.png")).clear()

    mi = report["mutual_information"]["mi_bits"]
    rf = report["rf_classifier"]["accuracy"]
    print(f"    MI={mi:.4f} bits, RF_acc={rf:.3f}")

    return report


def run_sim2real():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    all_reports = []

    for env_name, cfg in CONFIGS.items():
        if not cfg["enabled"]:
            continue

        config = cfg["config"]
        tflite_path = cfg["tflite"]

        if not os.path.exists(tflite_path):
            print(f"  SKIPPED {env_name}: {tflite_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {env_name.upper()}")
        print(f"{'='*60}")

        sampler = ObservationSampler(config)
        obs = sampler.uniform(N_SAMPLES, seed=SEED)

        if COLLECT_SIMULATED:
            label_sim = f"{env_name}_sim"
            print(f"\n  [SIM] Collecting {N_SAMPLES} simulated traces...")
            ds_sim = _collect_simulated(tflite_path, config, N_SAMPLES, obs)
            ds_sim.save(os.path.join(DATA_DIR, f"{label_sim}.npz"))
            print(f"  [SIM] Analyzing...")
            report_sim = _analyze_and_plot(ds_sim, label_sim)
            all_reports.append(report_sim)

        if COLLECT_HARDWARE:
            label_hw = f"{env_name}_hw"
            print(f"\n  [HW] Collecting {N_SAMPLES} hardware traces...")
            try:
                ds_hw = _collect_hardware(config, obs)
                ds_hw.save(os.path.join(DATA_DIR, f"{label_hw}.npz"))
                print(f"  [HW] Got {len(ds_hw)} traces. Analyzing...")
                report_hw = _analyze_and_plot(ds_hw, label_hw)
                all_reports.append(report_hw)
            except Exception as e:
                print(f"  [HW] FAILED: {e}")
                print(f"  [HW] Is the ESP32 connected on {SERIAL_PORT}?")

    print(f"\n{'='*60}")
    print("  SIM vs REAL COMPARISON")
    print(f"{'='*60}\n")

    if all_reports:
        print(format_report_table(all_reports))

        mi_data = {}
        for r in all_reports:
            mi_data[r["label"]] = {
                "MI (bits)": r["mutual_information"]["mi_bits"],
                "RF Acc": r["rf_classifier"]["accuracy"],
                "MLP Acc": r["mlp_classifier"]["accuracy"],
            }
        plot_mi_heatmap(
            mi_data, title="Sim vs Real Leakage",
            save_path=os.path.join(FIGURE_DIR, "sim2real_heatmap.png")).clear()

    print(f"\nFigures: {os.path.abspath(FIGURE_DIR)}/")
    print(f"Datasets: {os.path.abspath(DATA_DIR)}/")

    return all_reports


if __name__ == "__main__":
    run_sim2real()
