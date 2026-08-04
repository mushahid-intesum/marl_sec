# Task Tracker — Timing Side-Channel Analysis of MARL on ESP32-S3

## Phase 1: Environments + Export Pipeline
- [x] Create directory structure
- [x] `requirements.txt`
- [x] `.gitignore`
- [x] `envs/configs.py` — dataclass configs for all 3 environments
- [x] `envs/__init__.py`
- [x] `envs/grid_nav.py` — cooperative grid navigation environment
- [x] `envs/wrappers.py` — wrappers for MPE Simple Spread + CartPole
- [x] `export/__init__.py`
- [x] `export/to_tflite.py` — PyTorch → TFLite (FP32 + INT8) + C header
- [x] `tests/test_grid_nav.py` — 28 tests passing
- [x] `tests/test_wrappers.py` — 16 tests passing
- [x] `tests/test_export.py` — 16 tests passing
- [x] All Phase 1 tests pass (60/60)

## Phase 2: Analysis Pipeline
- [x] `collect/dataset.py` — 11 tests passing
- [x] `collect/sampler.py` — 8 tests passing
- [x] `analysis/mutual_info.py` — 8 tests passing
- [x] `analysis/correlation.py` — 11 tests passing
- [x] `analysis/classifier.py` — 8 tests passing
- [x] `analysis/leakage_report.py`
- [x] `analysis/plots.py` — 8 tests passing
- [x] All Phase 2 tests pass (56/56, 116 total)

## Phase 3: ESP32-S3 Firmware
- [x] `firmware/main/timing.{h,c}` — cycle counter at 240MHz
- [x] `firmware/main/protocol.{h,c}` — UART binary protocol (921600 baud)
- [x] `firmware/main/inference.{h,cc}` — TFLite Micro + MicroProfiler
- [x] `firmware/main/main.c` — entry point, inference loop
- [x] `firmware/CMakeLists.txt` + sdkconfig.defaults + idf_component.yml
- [x] `firmware/main/model_data.h` — placeholder model (2492 bytes)
- [x] Protocol encode/decode verified (Python ↔ C byte-compatible)
- [ ] `idf.py build` succeeds — requires ESP-IDF on your local machine

## Phase 4: Data Collection Bridge
- [x] `collect/serial_bridge.py` — ProtocolEncoder + SerialBridge with retry logic
- [x] `tests/test_serial_bridge.py` — 17 tests passing (protocol, mocked serial, compatibility)
- [x] All Phase 4 tests pass (17/17, 133 total)

## Phase 5: Integration
- [x] `collect/simulated_collector.py` — software-simulated ESP32 timing
- [x] `integration_test.py` — end-to-end smoke test
- [x] Smoke test PASSED: 4 configs × full analysis pipeline
- [x] Generated 12 figures, 4 datasets, 8 model files
- [x] Walkthrough created
