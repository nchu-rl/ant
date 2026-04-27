## Why

簡報第一頁與第五頁將 PPO 與 SAC 並列為本研究選用之 RL 演算法。SAC（Haarnoja 2018）是 off-policy、最大熵 actor-critic 框架，在連續控制任務上樣本效率通常優於 PPO。本研究將 SAC 作為對照組，回答「在本 reward / terrain 設定下，PPO vs SAC 哪一個收斂更快、最終分數更高、能耗更低？」這需要一個與 PPO 平起平坐、超參數同等被嚴謹設定的 SAC pipeline。

## What Changes

- 新增 `src/training/sac.py`：以 Stable-Baselines3 SAC 訓練 Ant，與 PPO pipeline 共用 env / reward / terrain / config / checkpoint / 續訓機制。
- 新增 `configs/sac_default.yaml`：SAC 專屬超參數。
- 新增 CLI：`python -m training.sac --config configs/sac_default.yaml`。
- 抽取 PPO / SAC 共用邏輯至 `src/training/common.py`（若 `add-ppo-training` 尚未抽，本 change 補上）。
- 確保 SAC run 的目錄結構與 PPO 對等，以利 evaluation 比較。

## Capabilities

### New Capabilities
- `sac-training`: Ant 在 MuJoCo 環境的 SAC 訓練 pipeline、超參數設定、checkpoint、續訓、CLI 介面，作為 PPO 的對照組。

### Modified Capabilities
（無）

## Impact

- 程式碼：`src/training/sac.py`、`configs/sac_default.yaml`、可能修改 `src/training/common.py`
- 訓練產出：`runs/sac/<ts>/` 與 `runs/ppo/<ts>/` 結構鏡像
- 相依：`add-env-setup`、`add-reward-design`、`add-terrain-generation`、`add-experiment-tracking`
- 被相依：`add-evaluation`（同時評估 PPO / SAC）

## Non-Goals

- 不負責 PPO vs SAC 的最終效能比較（屬 `add-evaluation`）。
- 不負責 HPO。
- 不導入 SAC 變體（CrossQ、REDQ 等）— 嚴守簡報引用之原版 SAC。

## Assumptions & Open Questions

**Assumptions**
- 沿用 SB3 `SAC` + `MlpPolicy`，自動 entropy tuning（`ent_coef="auto"`，Haarnoja 2019 推薦）。
- 神經網路架構與 PPO 對齊：[256, 256] tanh，便於公平比較。
- 訓練 timesteps 對齊 PPO 的 5e6（SAC 樣本效率較高，但用同樣預算便於比較）。

**Open Questions**
- SAC 預設超參數（lr、batch、buffer size、tau、γ、train_freq、gradient_steps）：兩份文件未提。預設見 design.md，使用 SB3 MuJoCo benchmark 數值。
- SAC 的 replay buffer 較大（≥ 1e6），對記憶體需求較高；具體 RAM 預算未提。預設 32GB。
- SAC 不需 vectorized env 達到吞吐（off-policy），但 SB3 仍支援 `n_envs > 1`；預設 `n_envs=1` 以對齊主流 SAC 實作。
