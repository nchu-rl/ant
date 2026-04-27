## Why

Greenfield 專案需要可重現的模擬基底。沒有穩定的 Ant-v4 + MuJoCo + Gymnasium 執行環境，後續 reward / training / evaluation 等所有 capability 都無法落地。本 change 是整個研究專題的 foundation。

## What Changes

- 鎖定核心相依套件版本：MuJoCo、`gymnasium[mujoco]`、Stable-Baselines3、PyTorch、NumPy。
- 提供環境工廠函式 `make_ant_env(seed, render_mode)` 統一建立 Ant-v4 環境，集中設定 seed、render mode、frame skip 等。
- 提供 smoke test：以隨機動作跑完一個完整 episode，驗證觀測 / 動作維度正確、reset / step 行為符合 Gymnasium API。
- 文件化安裝步驟，含 OS、Python 版本、MuJoCo binary、GPU 驅動需求。

## Capabilities

### New Capabilities
- `env-setup`: Ant-v4 模擬環境的安裝、版本鎖定、環境工廠、與 smoke test 驗證機制。

### Modified Capabilities
（無）

## Impact

- 新增程式碼：`src/env/factory.py`、`tests/smoke_test.py`
- 新增依賴宣告：`pyproject.toml` 或 `requirements.txt`（含 lock 檔）
- 新增文件：`README.md` 的「Environment setup」區塊
- 後續所有 change（terrain-generation、reward-design、ppo-training、sac-training、domain-randomization、evaluation、experiment-tracking）皆相依本 change

## Non-Goals

- 不包含複雜地形生成（屬 `add-terrain-generation`）
- 不修改 Ant-v4 機器人幾何（DoF、關節數）
- 不包含獎勵函數調整（屬 `add-reward-design`）
- 不包含任何訓練 / 評估 pipeline

## Assumptions & Open Questions

**Assumptions（沿用 config.yaml 預設）**
- Python 3.10+，Stable-Baselines3 ≥ 2.0，`gymnasium[mujoco]` ≥ 0.29，PyTorch ≥ 2.0
- 使用 venv 作為環境管理工具（conda 列為替代）
- 開發 OS：Linux / macOS（Windows 透過 WSL2）

**Open Questions**
- 訓練機具體規格未提；假設至少單張 NVIDIA GPU（≥ 8GB VRAM）+ CUDA 11.8/12.x，或 CPU-only fallback。
- 是否需要支援 headless render（offscreen GL context for cluster training）？預設支援 `render_mode="rgb_array"`。
- MuJoCo binary 安裝方式（系統套件 vs `pip install mujoco`）：預設用 `pip install mujoco`，因 Gymnasium 已將其作為相依。
