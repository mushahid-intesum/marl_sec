import os
import numpy as np
import torch

from train.config import PPOConfig, TrainConfig
from train.runner import train_cartpole, train_grid_nav, train_simple_spread
from export.to_tflite import export_model
from collect.sampler import ObservationSampler
from collect.simulated_collector import SimulatedCollector
from collect.dataset import TimingDataset
from analysis.leakage_report import generate_report, format_report_table
from analysis.plots import (plot_timing_distributions, plot_confusion_matrix,
                              plot_per_layer_breakdown)
from envs.configs import CARTPOLE_CONFIG, GRID_NAV_CONFIG, SIMPLE_SPREAD_CONFIG

OUTPUT_DIR = "models"
FIGURE_DIR = "figures"
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
N_COLLECT_SAMPLES = 2000
SEED = 42

TRAIN_CARTPOLE = True
TRAIN_GRID_NAV = True
TRAIN_SIMPLE_SPREAD = False

CARTPOLE_TIMESTEPS = 50000
GRID_NAV_TIMESTEPS = 100000
SPREAD_TIMESTEPS = 100000

SKIP_TRAINING = False
USE_HARDWARE = False
SERIAL_PORT = "/dev/ttyUSB0"


def _ensure_dirs():
    for d in [OUTPUT_DIR, FIGURE_DIR, DATA_DIR, CHECKPOINT_DIR]:
        os.makedirs(d, exist_ok=True)


def _export_and_analyze(actor, config, env_name, label_prefix=""):
    results = export_model(actor, config, OUTPUT_DIR, env_name)
    print(f"    Exported: FP32={results['fp32_size']}B, INT8={results['int8_size']}B")

    sampler = ObservationSampler(config)
    obs = sampler.uniform(N_COLLECT_SAMPLES, seed=SEED)

    reports = []
    for quant in ["fp32", "int8"]:
        label = f"{label_prefix}{env_name}_{quant}"
        collector = SimulatedCollector(results[f"{quant}_tflite"], config)
        dataset = collector.collect(obs)
        dataset.save(os.path.join(DATA_DIR, f"{label}.npz"))

        report = generate_report(dataset, label=label)
        reports.append(report)

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
        print(f"    {quant}: MI={mi:.3f} bits, RF_acc={rf:.3f}")

    return reports


def _collect_hardware(config, env_name, model_path):
    from collect.serial_bridge import SerialBridge
    sampler = ObservationSampler(config)
    obs = sampler.uniform(N_COLLECT_SAMPLES, seed=SEED)

    label = f"{env_name}_hw"
    with SerialBridge(port=SERIAL_PORT) as bridge:
        dataset = bridge.collect_dataset(obs, config.obs_dim, config.act_dim)
    dataset.save(os.path.join(DATA_DIR, f"{label}.npz"))

    report = generate_report(dataset, label=label)
    plot_timing_distributions(
        dataset, title=f"HW Timing: {label}",
        save_path=os.path.join(FIGURE_DIR, f"{label}_violin.png")).clear()

    print(f"    hardware: MI={report['mutual_information']['mi_bits']:.3f}, "
          f"RF_acc={report['rf_classifier']['accuracy']:.3f}")
    return [report]


def run_pipeline():
    _ensure_dirs()
    all_reports = []

    if TRAIN_CARTPOLE:
        print("\n" + "=" * 60)
        print("  CARTPOLE (PPO)")
        print("=" * 60)

        if not SKIP_TRAINING:
            tc = TrainConfig(total_timesteps=CARTPOLE_TIMESTEPS, save_dir=CHECKPOINT_DIR)
            result = train_cartpole(train_config=tc)
            actor = result["actor"]
            print(f"  Training done. Mean reward: {result['mean_reward']:.1f}")
        else:
            from train.ppo import PPOAgent
            agent = PPOAgent(CARTPOLE_CONFIG.obs_dim, CARTPOLE_CONFIG.act_dim, PPOConfig())
            agent.load(os.path.join(CHECKPOINT_DIR, "cartpole_ppo.pt"))
            actor = agent.actor

        reports = _export_and_analyze(actor, CARTPOLE_CONFIG, "cartpole")
        all_reports.extend(reports)

        if USE_HARDWARE:
            hw = _collect_hardware(CARTPOLE_CONFIG, "cartpole",
                                   os.path.join(OUTPUT_DIR, "cartpole_fp32.h"))
            all_reports.extend(hw)

    if TRAIN_GRID_NAV:
        print("\n" + "=" * 60)
        print("  GRID NAV (IPPO)")
        print("=" * 60)

        if not SKIP_TRAINING:
            tc = TrainConfig(total_timesteps=GRID_NAV_TIMESTEPS, save_dir=CHECKPOINT_DIR)
            result = train_grid_nav(train_config=tc)
            actors = result["actors"]
            print(f"  Training done for {len(actors)} agents")
        else:
            from train.ppo import PPOAgent
            actors = {}
            for a in ["agent_0", "agent_1"]:
                agent = PPOAgent(GRID_NAV_CONFIG.obs_dim, GRID_NAV_CONFIG.act_dim, PPOConfig())
                agent.load(os.path.join(CHECKPOINT_DIR, f"grid_nav_{a}.pt"))
                actors[a] = agent.actor

        for agent_name, actor in actors.items():
            name = f"grid_nav_{agent_name}"
            print(f"\n  Analyzing {name}...")
            reports = _export_and_analyze(actor, GRID_NAV_CONFIG, name)
            all_reports.extend(reports)

    if TRAIN_SIMPLE_SPREAD:
        print("\n" + "=" * 60)
        print("  SIMPLE SPREAD (IPPO)")
        print("=" * 60)

        if not SKIP_TRAINING:
            pc = PPOConfig(hidden_sizes=[64, 64])
            tc = TrainConfig(total_timesteps=SPREAD_TIMESTEPS, save_dir=CHECKPOINT_DIR)
            result = train_simple_spread(ppo_config=pc, train_config=tc)
            actors = result["actors"]
            print(f"  Training done for {len(actors)} agents")
        else:
            from train.ppo import PPOAgent
            actors = {}
            for i in range(3):
                a = f"adversary_{i}" if i < 1 else f"agent_{i - 1}"
                agent = PPOAgent(SIMPLE_SPREAD_CONFIG.obs_dim,
                                SIMPLE_SPREAD_CONFIG.act_dim,
                                PPOConfig(hidden_sizes=[64, 64]))
                agent.load(os.path.join(CHECKPOINT_DIR, f"simple_spread_{a}.pt"))
                actors[a] = agent.actor

        for agent_name, actor in actors.items():
            name = f"spread_{agent_name}"
            print(f"\n  Analyzing {name}...")
            reports = _export_and_analyze(actor, SIMPLE_SPREAD_CONFIG, name)
            all_reports.extend(reports)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60 + "\n")
    print(format_report_table(all_reports))
    print(f"\nFigures: {os.path.abspath(FIGURE_DIR)}/")
    print(f"Datasets: {os.path.abspath(DATA_DIR)}/")
    print(f"Models: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"Checkpoints: {os.path.abspath(CHECKPOINT_DIR)}/")

    return all_reports


if __name__ == "__main__":
    run_pipeline()
