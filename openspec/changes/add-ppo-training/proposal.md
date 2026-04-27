## Why

簡報第七頁將 PPO 列為主演算法：「處理高維度連續性動作空間時，具有極佳的收斂穩定性，能防止訓練過程中步幅過大導致模型不穩定或崩潰」。沒有可重複執行、可儲存 / 載入 checkpoint 的訓練 pipeline，後續評估、ablation、與 SAC 對照都無從進行。

## What Changes

- 新增 `src/training/ppo.py`：以 Stable-Baselines3 PPO 訓練 Ant，封裝環境工廠、reward、terrain（透過上游 capability）。
- 新增 `configs/ppo_default.yaml`：所有超參數集中、可重現。
- 新增 CLI：`python -m training.ppo --config configs/ppo_default.yaml --total-timesteps 5e6`。
- 新增 checkpoint 機制：定期存檔（每 N steps）、最佳模型、final 模型；可從 checkpoint 續訓。
- 新增 vectorized env（`SubprocVecEnv` / `DummyVecEnv`）以平行化採樣。
- 新增訓練摘要：訓練結束印出 metrics、儲存路徑、執行時長。

## Capabilities

### New Capabilities
- `ppo-training`: Ant 在 MuJoCo 環境的 PPO 訓練 pipeline、超參數設定、checkpoint、續訓、CLI 介面。

### Modified Capabilities
（無）

## Impact

- 程式碼：`src/training/ppo.py`、`configs/ppo_default.yaml`、`src/training/common.py`（共用 utility）
- 訓練產出：`runs/ppo/<timestamp>/{checkpoints,final.zip,config.yaml,tb/}`
- 相依：`add-env-setup`（環境工廠）、`add-reward-design`（自訂 reward）、`add-terrain-generation`（可選地形）、`add-experiment-tracking`（log 介面）
- 被相依：`add-evaluation`（載入訓練結果評估）、`add-domain-randomization`（透過本訓練 pipeline 啟用）

## Non-Goals

- 不負責超參數搜尋（HPO）— 提供 config 介面，搜尋工具屬未來擴充。
- 不負責真實機器人部署。
- 不實作 SAC（屬 `add-sac-training`，獨立 capability 以方便對照）。
- 不負責評估 / 比較 metrics（屬 `add-evaluation`）。

## Assumptions & Open Questions

**Assumptions**
- 使用 SB3 內建 `PPO`，`MlpPolicy`，搭配 `VecNormalize` 標準化 obs / reward。
- 神經網路採對稱 MLP，預設 [256, 256] tanh。
- 訓練長度預設 5e6 timesteps（2026 年 RL benchmark 上 Ant-v4 可在此規模收斂）。
- 平行環境數預設 8 個（與一般 server CPU 核心數一致）。

**Open Questions**
- 具體超參數（learning rate、n_steps、batch_size、clip_range、γ、λ）：兩份文件都沒寫。預設值見 design.md，使用 SB3 經 MuJoCo benchmark 調整過的數值（lr 3e-4、n_steps 2048、batch_size 64、γ 0.99、λ 0.95、clip 0.2）。
- 訓練硬體：兩份文件未提具體規格。預設於 GPU + 8 核 CPU 上估時 4–8 小時 / 5e6 steps；CPU-only 估 24+ 小時。
- 是否包含 curriculum（先 plain 再 rough）？預設不包含，由 domain-randomization change 處理多地形混訓。
