# Timing Side-Channel Analysis of MARL Inference on ESP32-S3

Analyze whether an adversary can infer the intended action or observation features of a MARL agent by measuring inference timing on ESP32-S3 hardware.

> [!NOTE]
> **Resolved:** ESP32-S3 (with SIMD/vector acceleration), ESP-IDF installed, PyTorch → ONNX → TFLite export pipeline.

## Threat Model

```
┌─────────────────────────────────────────────────┐
│                   ATTACKER                      │
│  Can observe:                                   │
│    • Inference duration (total + per-layer)      │
│    • Number of inferences per timestep           │
│    • Network packet timing (if WiFi used)        │
│  Cannot observe:                                │
│    • Raw observations fed to the policy          │
│    • Internal activations or weights             │
│    • Source code (black-box timing only)          │
│                                                 │
│  Goal: Predict the agent's ACTION from timing   │
│        without seeing the observation input      │
└─────────────────────────────────────────────────┘
```

## Environments

### Env 1: Cooperative Grid Navigation (Custom, Primary)

```
A . . . .     2 agents, 5×5 grid, swapped goals
. . . . .     Comm: position + intended direction
# # . # #     Shared reward, collision penalty
. . . . .     
. . . . B     Policy: MLP(9 → 32 → 32 → 5)
```

### Env 2: MPE Simple Spread (PettingZoo Benchmark)

3 agents cooperatively covering 3 landmarks. Standard MARL benchmark.
Policy: MLP(18 → 64 → 64 → 5)

### Env 3: CartPole (Gymnasium Baseline)

Single-agent baseline. Establishes whether timing leakage is a general phenomenon.
Policy: MLP(4 → 32 → 32 → 2)

## Experiment Matrix

| # | Experiment | Question |
|---|---|---|
| E1 | 10K traces per env/model, group by action | Do actions have different timing distributions? |
| E2 | MI(total_time; action) across configs | How many bits of action info leak? |
| E3 | Timing→action classifier accuracy | Can an attacker operationally exploit leakage? |
| E4 | INT8 vs FP32 leakage comparison | Does quantization amplify or suppress leakage? |
| E5 | Small/medium/large policy comparison | Does model size affect leakage? |
| E6 | Per-layer timing analysis | Which operator is the primary leakage source? |
| E7 | Timing→observation correlation | Does timing leak the input, not just the output? |

---

## Phase 1 — Environments + Export Pipeline

> Pure Python. No hardware needed. Can start immediately.

### Deliverables

| File | Purpose |
|---|---|
| `envs/__init__.py` | Package init |
| `envs/grid_nav.py` | 2-agent cooperative grid navigation environment |
| `envs/wrappers.py` | Standardized wrappers for MPE Simple Spread + CartPole |
| `envs/configs.py` | Dataclass configs: obs_dim, act_dim, normalization bounds |
| `export/__init__.py` | Package init |
| `export/to_tflite.py` | PyTorch MLP → ONNX → TFLite (FP32 + INT8) + C header |
| `tests/test_grid_nav.py` | Environment mechanics tests |
| `tests/test_wrappers.py` | Wrapper shape/normalization tests |
| `tests/test_export.py` | Round-trip export correctness tests |
| `requirements.txt` | All Python dependencies |

### Exit Criteria

- [ ] `pytest tests/test_grid_nav.py` — all pass
- [ ] `pytest tests/test_wrappers.py` — all pass
- [ ] `pytest tests/test_export.py` — all pass (dummy MLP → TFLite → load → outputs match)
- [ ] Grid nav env runs 1000 steps without error
- [ ] Wrappers produce correct shapes for all 3 environments
- [ ] Export produces valid `.tflite` files (FP32 + INT8) and a `model_data.h` C header

---

## Phase 2 — Analysis Pipeline

> Pure Python. Validated entirely with synthetic data (no ESP32 needed).
> Can run in parallel with Phase 1.

### Deliverables

| File | Purpose |
|---|---|
| `collect/__init__.py` | Package init |
| `collect/dataset.py` | Save/load timing datasets as `.npz` |
| `collect/sampler.py` | Generate observation batches (uniform, on-policy, adversarial) |
| `analysis/__init__.py` | Package init |
| `analysis/mutual_info.py` | MI(timing; action) with bootstrap CIs |
| `analysis/correlation.py` | Kruskal-Wallis, Spearman, ANOVA, Cohen's d |
| `analysis/classifier.py` | Random Forest + MLP timing→action predictors |
| `analysis/leakage_report.py` | Aggregate all metrics into summary table |
| `analysis/plots.py` | Violin plots, MI heatmaps, confusion matrices |
| `tests/test_dataset.py` | Save/load round-trip, schema validation |
| `tests/test_sampler.py` | Output shapes, bounds correctness |
| `tests/test_mutual_info.py` | Known-leakage → high MI, independent → MI ≈ 0 |
| `tests/test_correlation.py` | Synthetic correlated/uncorrelated verification |
| `tests/test_classifier.py` | Perfect-leakage → ~100%, random → ~1/N accuracy |
| `tests/test_plots.py` | Smoke tests: figures render without error |

### Exit Criteria

- [ ] All `pytest tests/test_*.py` for Phase 2 files pass
- [ ] Synthetic "perfect leakage" data: MI ≈ log2(5) ≈ 2.32 bits, classifier ≈ 100%
- [ ] Synthetic "zero leakage" data: MI ≈ 0, classifier ≈ 20% (random for 5 actions)
- [ ] Plots generate valid PNG files

---

## Phase 3 — ESP32-S3 Firmware

> C / ESP-IDF. Requires ESP-IDF toolchain but not a connected board for compilation.

### Deliverables

| File | Purpose |
|---|---|
| `firmware/CMakeLists.txt` | Top-level ESP-IDF project config |
| `firmware/sdkconfig.defaults` | ESP32-S3 target, UART baud rate, PSRAM, flash size |
| `firmware/main/CMakeLists.txt` | Component registration with esp-tflite-micro |
| `firmware/main/main.c` | Entry point: init UART, load model, inference loop |
| `firmware/main/inference.h` | Inference API: `run_inference()` with timing trace output |
| `firmware/main/inference.c` | TFLite Micro interpreter + MicroProfiler integration |
| `firmware/main/timing.h` | Cycle counter API: `timing_start/stop/to_us()` |
| `firmware/main/timing.c` | `esp_cpu_get_cycle_count()` wrapper at 240MHz |
| `firmware/main/protocol.h` | UART binary protocol structs and constants |
| `firmware/main/protocol.c` | Send/receive observation and timing data over UART |
| `firmware/main/model_data.h` | Placeholder (you replace with exported model) |

### UART Protocol

```
Laptop → ESP32:  [0xAA 0x55] [obs_len: u16] [obs: obs_len × f32]
ESP32 → Laptop:  [0xAA 0x55] [action: u8] [total_cycles: u32] [n_ops: u8] [per_op: n_ops × u32]
```

### ESP32-S3 Specifics

- Target: `esp32s3` (set via `idf.py set-target esp32s3`)
- CPU: Xtensa LX7 dual-core @ 240MHz
- Cycle counter: `esp_cpu_get_cycle_count()` (4.17ns resolution)
- TFLite Micro: `espressif/esp-tflite-micro` component with ESP-NN SIMD acceleration
- The S3's vector ISA accelerates quantized matmul, which may **reduce** timing variation vs. non-SIMD ESP32 — this is itself an interesting finding

### Exit Criteria

- [ ] `idf.py build` succeeds with placeholder model
- [ ] Firmware boots, prints "ready" on serial monitor
- [ ] Sends valid response packet when observation packet is received
- [ ] Timing measurements have < 1% coefficient of variation on identical inputs (measurement stability)

---

## Phase 4 — Data Collection Bridge

> Python. Requires ESP32-S3 connected via USB/UART.

### Deliverables

| File | Purpose |
|---|---|
| `collect/serial_bridge.py` | pyserial communication with ESP32, sync, retry, validation |
| `tests/test_serial_bridge.py` | Protocol encoding/decoding tests (mock serial, no hardware needed for tests) |

### How Collection Works

```
┌──────────┐    UART     ┌──────────┐
│  Laptop  │ ──────────→ │ ESP32-S3 │
│          │  obs vector  │          │
│ sampler  │ ←────────── │ TFLite   │
│ dataset  │  timing+act  │ inference│
└──────────┘             └──────────┘
```

1. `sampler.py` generates N observations (from env config)
2. `serial_bridge.py` sends each observation to ESP32 via UART
3. ESP32 runs inference, measures timing, sends back `(action, timing_trace)`
4. `dataset.py` stores all `(obs, action, timing)` tuples as `.npz`

### Exit Criteria

- [ ] `pytest tests/test_serial_bridge.py` — passes (protocol tests, no hardware)
- [ ] With ESP32 connected: send 100 observations, receive 100 valid responses
- [ ] Collected dataset saves and loads correctly
- [ ] Action predictions from ESP32 match TFLite interpreter on laptop (correctness check)

---

## Phase 5 — Integration & Smoke Test

> End-to-end validation. Requires trained model + ESP32-S3.

### What You Do

1. Train CartPole PPO (simplest env, fastest to verify)
2. Run `export/to_tflite.py` → produces `model_data.h`
3. Copy `model_data.h` into `firmware/main/`
4. `idf.py build && idf.py -p /dev/ttyUSB0 flash`
5. Run collection: generates `data/cartpole_fp32.npz`
6. Run analysis: generates `figures/` + summary table

### What I Verify (in code)

- Export pipeline produces valid TFLite model
- ESP32 actions match laptop TFLite interpreter
- Timing variance is low on identical inputs (measurement noise floor)
- Analysis pipeline produces all expected outputs

### Exit Criteria

- [ ] Full pipeline runs end-to-end on CartPole (simplest case)
- [ ] 1K timing traces collected and stored
- [ ] Analysis report generated with MI, classifier accuracy, and plots
- [ ] Results are plausible (non-zero timing variance across different observations)

---

## Project Structure

```
marl_sec/
├── envs/
│   ├── __init__.py
│   ├── grid_nav.py
│   ├── wrappers.py
│   └── configs.py
├── export/
│   ├── __init__.py
│   └── to_tflite.py
├── firmware/
│   ├── main/
│   │   ├── main.c
│   │   ├── inference.c
│   │   ├── inference.h
│   │   ├── timing.c
│   │   ├── timing.h
│   │   ├── protocol.c
│   │   ├── protocol.h
│   │   ├── model_data.h          ← generated by export
│   │   └── CMakeLists.txt
│   ├── CMakeLists.txt
│   └── sdkconfig.defaults
├── collect/
│   ├── __init__.py
│   ├── serial_bridge.py
│   ├── sampler.py
│   └── dataset.py
├── analysis/
│   ├── __init__.py
│   ├── mutual_info.py
│   ├── correlation.py
│   ├── classifier.py
│   ├── leakage_report.py
│   └── plots.py
├── tests/
│   ├── test_grid_nav.py
│   ├── test_wrappers.py
│   ├── test_export.py
│   ├── test_sampler.py
│   ├── test_dataset.py
│   ├── test_mutual_info.py
│   ├── test_correlation.py
│   ├── test_classifier.py
│   ├── test_serial_bridge.py
│   └── test_plots.py
├── data/                         ← timing datasets (gitignored)
├── figures/                      ← generated plots (gitignored)
├── models/                       ← trained models + TFLite (gitignored)
├── requirements.txt
└── .gitignore
```

## Verification Plan

### Automated (every phase)

```bash
python -m pytest tests/ -v
```

### Export Correctness

Dummy MLP → TFLite → load back → compare on 100 random inputs. FP32: bit-exact. INT8: within ±1 quant level.

### Firmware Correctness

100 known observations → ESP32 actions must match laptop TFLite interpreter.

### Analysis Correctness

Synthetic "perfect leakage" → MI ≈ log2(num_actions), classifier ≈ 100%.
Synthetic "zero leakage" → MI ≈ 0, classifier ≈ random.

### End-to-End

CartPole: train → export → flash → collect 1K traces → analyze → verify plots.
