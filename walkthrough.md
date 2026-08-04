# Walkthrough — Timing Side-Channel Analysis of MARL on ESP32-S3

## What Was Built

A complete pipeline for analyzing whether inference timing on ESP32-S3 leaks information about MARL agent actions.

```mermaid
graph LR
    A[Train Policy] --> B[Export TFLite]
    B --> C[Flash ESP32-S3]
    C --> D[Collect Timing]
    D --> E[Analyze Leakage]
    E --> F[Generate Report]
```

## Project Structure

```
marl_sec/
├── envs/                    ← 3 environments (grid_nav, cartpole, simple_spread)
├── export/                  ← PyTorch → TFLite (FP32 + INT8) + C header
├── firmware/                ← ESP32-S3 ESP-IDF project with TFLite Micro
├── collect/                 ← Serial bridge, sampler, dataset, simulated collector
├── analysis/                ← MI, correlation, classifiers, reports, plots
├── tests/                   ← 133 tests across all modules
├── integration_test.py      ← End-to-end smoke test
├── data/                    ← Collected timing datasets (.npz)
├── figures/                 ← Generated analysis plots (.png)
└── models/                  ← Exported TFLite models + C headers
```

## Changes Made

### Phase 1: Environments + Export (60 tests)

| File | Purpose |
|---|---|
| [configs.py](file:///home/mushahidintesum/Music/marl_sec/envs/configs.py) | Dataclass configs: obs_dim, act_dim, bounds for all 3 envs |
| [grid_nav.py](file:///home/mushahidintesum/Music/marl_sec/envs/grid_nav.py) | 2-agent cooperative grid navigation with comm channel |
| [wrappers.py](file:///home/mushahidintesum/Music/marl_sec/envs/wrappers.py) | CartPole + MPE Simple Spread with normalization |
| [to_tflite.py](file:///home/mushahidintesum/Music/marl_sec/export/to_tflite.py) | PyTorch MLP → Keras → TFLite (FP32/INT8) + C header |

### Phase 2: Analysis Pipeline (56 tests)

| File | Purpose |
|---|---|
| [dataset.py](file:///home/mushahidintesum/Music/marl_sec/collect/dataset.py) | TimingDataset: store/load/group timing traces |
| [sampler.py](file:///home/mushahidintesum/Music/marl_sec/collect/sampler.py) | Uniform, on-policy, adversarial observation generation |
| [mutual_info.py](file:///home/mushahidintesum/Music/marl_sec/analysis/mutual_info.py) | MI(timing; action) with bootstrap CIs |
| [correlation.py](file:///home/mushahidintesum/Music/marl_sec/analysis/correlation.py) | Kruskal-Wallis, ANOVA, Spearman, Cohen's d |
| [classifier.py](file:///home/mushahidintesum/Music/marl_sec/analysis/classifier.py) | Random Forest + MLP timing→action classifiers |
| [leakage_report.py](file:///home/mushahidintesum/Music/marl_sec/analysis/leakage_report.py) | Aggregate all metrics into summary table |
| [plots.py](file:///home/mushahidintesum/Music/marl_sec/analysis/plots.py) | Violin plots, MI heatmaps, confusion matrices |

### Phase 3: ESP32-S3 Firmware (12 files)

| File | Purpose |
|---|---|
| [timing.c](file:///home/mushahidintesum/Music/marl_sec/firmware/main/timing.c) | CPU cycle counter at 240MHz |
| [protocol.c](file:///home/mushahidintesum/Music/marl_sec/firmware/main/protocol.c) | UART binary protocol @ 921600 baud |
| [inference.cc](file:///home/mushahidintesum/Music/marl_sec/firmware/main/inference.cc) | TFLite Micro + MicroProfiler per-op timing |
| [main.c](file:///home/mushahidintesum/Music/marl_sec/firmware/main/main.c) | Entry point: receive obs → infer → send timing |

### Phase 4: Serial Bridge (17 tests)

| File | Purpose |
|---|---|
| [serial_bridge.py](file:///home/mushahidintesum/Music/marl_sec/collect/serial_bridge.py) | UART communication + protocol encode/decode |

### Phase 5: Integration

| File | Purpose |
|---|---|
| [simulated_collector.py](file:///home/mushahidintesum/Music/marl_sec/collect/simulated_collector.py) | Software-simulated ESP32 timing for pipeline validation |
| [integration_test.py](file:///home/mushahidintesum/Music/marl_sec/integration_test.py) | End-to-end smoke test: build → export → collect → analyze |

---

## Test Results

```
133 passed in 15.93s
```

## Smoke Test Results (Simulated)

| Config | MI (bits) | RF Accuracy | KW p-value | Max Cohen's d |
|---|---|---|---|---|
| cartpole_fp32 | 0.011 | 99.5% | 1.98e-9 | 0.274 |
| cartpole_int8 | 0.022 | 99.8% | 1.71e-9 | 0.276 |
| grid_nav_fp32 | 0.976 | 99.8% | 1.46e-281 | 8.879 |
| grid_nav_int8 | 1.005 | 100.0% | 1.15e-286 | 8.838 |

> [!NOTE]
> These results use simulated timing (action-dependent cycle counts + noise) to validate the pipeline. Real ESP32 results will show whether actual hardware timing has exploitable leakage. The simulated results confirm the analysis pipeline correctly detects leakage when present.

### Generated Figures

![Timing Distribution by Action](grid_nav_fp32_violin.png)

![RF Classifier Confusion Matrix](grid_nav_fp32_confusion.png)

---

## Next Steps (Your Tasks)

### 1. Train Your Policies
Train IPPO/PPO policies for each environment using your preferred framework.

### 2. Export to TFLite
```python
from export.to_tflite import export_model
from envs.configs import GRID_NAV_CONFIG

result = export_model(your_trained_model, GRID_NAV_CONFIG, "models", "grid_nav")
```

### 3. Build & Flash Firmware
```bash
cp models/grid_nav_fp32.h firmware/main/model_data.h
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash
```

### 4. Collect Real Timing Data
```python
from collect.serial_bridge import SerialBridge
from collect.sampler import ObservationSampler
from envs.configs import GRID_NAV_CONFIG

sampler = ObservationSampler(GRID_NAV_CONFIG)
obs = sampler.uniform(10000)

with SerialBridge(port="/dev/ttyUSB0") as bridge:
    dataset = bridge.collect_dataset(obs, obs_dim=9, act_dim=5)
    dataset.save("data/grid_nav_real.npz")
```

### 5. Run Analysis
```python
from analysis.leakage_report import generate_report, format_report_table

report = generate_report(dataset, label="grid_nav_real_fp32")
print(format_report_table([report]))
```
