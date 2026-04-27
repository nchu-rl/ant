## ADDED Requirements

### Requirement: SAC 訓練可由 YAML config 啟動
The system SHALL launch a complete SAC training run from a single YAML config file via CLI, in a way that mirrors the PPO pipeline.

#### Scenario: 預設 config 跑 smoke 訓練
- **GIVEN** `configs/sac_default.yaml`
- **WHEN** 執行 `python -m training.sac --config configs/sac_default.yaml --total-timesteps 20000`
- **THEN** 訓練 MUST 完成而不報錯
- **AND** 在 `runs/sac/<ts>/` 產出 `final.zip`、`config.yaml`、`tb/`、`train.log`

#### Scenario: 缺欄位的 config 即時拒絕
- **GIVEN** YAML config 缺 `buffer_size` 必填欄位
- **WHEN** CLI 啟動
- **THEN** 系統 MUST 在訓練前拋出 `ValueError` 並列出缺失欄位

### Requirement: 與 PPO 對等的 run 目錄結構
SAC run directories SHALL mirror PPO run directories so that downstream evaluation can treat them uniformly.

#### Scenario: PPO / SAC 兩個 run 同類型 artifact 都存在
- **GIVEN** 一個 PPO run 與一個 SAC run 各完成
- **WHEN** 比對兩個 run 目錄
- **THEN** 必要 artifact (`config.yaml`、`final.zip`、`tb/`、`checkpoints/`、`train.log`) MUST 在兩個 run 中都存在
- **AND** SAC run 額外包含 `replay_buffer.pkl`（若 `--save-replay-buffer`）

### Requirement: Entropy 自動調整
By default the system SHALL use SAC with automatic entropy coefficient tuning.

#### Scenario: 預設 config `ent_coef` 為 "auto"
- **GIVEN** `configs/sac_default.yaml` 未修改
- **WHEN** 啟動訓練並檢查 SB3 內部 `model.ent_coef`
- **THEN** 該值 MUST 為字串 `"auto"`，且 `model.target_entropy` MUST 等於 `-action_dim`（即 -8）

#### Scenario: 訓練中 ent_coef 數值變化
- **GIVEN** 一個訓練至少 100k steps 的 run
- **WHEN** 檢視 TensorBoard
- **THEN** `train/ent_coef` 曲線 MUST 隨 timesteps 變化（非常數）

### Requirement: Off-policy buffer 與 checkpoint
The system SHALL persist policy weights at every checkpoint, and optionally persist the replay buffer when requested.

#### Scenario: 預設不存 replay buffer
- **GIVEN** CLI 不帶 `--save-replay-buffer`
- **WHEN** 訓練到 200k steps
- **THEN** `runs/sac/<ts>/checkpoints/` 內 MUST 不含 `replay_buffer_*.pkl`

#### Scenario: 啟用後存 replay buffer
- **GIVEN** CLI 帶 `--save-replay-buffer`
- **WHEN** 訓練到 200k steps
- **THEN** `runs/sac/<ts>/replay_buffer.pkl` MUST 存在
- **AND** 其檔案大小 MUST > 0

### Requirement: 續訓
The system SHALL support resuming SAC training, with a clear behavior whether the replay buffer is restored.

#### Scenario: 續訓時無 buffer 檔案 → 重新填充
- **GIVEN** 一個無 `replay_buffer.pkl` 的 run（曾以預設續訓設定中止）
- **WHEN** `python -m training.sac --resume-from runs/sac/<ts>`
- **THEN** 訓練 MUST 從 checkpoint 接續
- **AND** stdout SHALL 印出警告：buffer 從空開始重新填充

#### Scenario: 續訓時有 buffer 檔案 → 載入
- **GIVEN** 一個含 `replay_buffer.pkl` 的 run
- **WHEN** 續訓
- **THEN** SB3 內部 `model.replay_buffer.size()` MUST > 0 即刻

### Requirement: 與 PPO 對齊的公平基準
SAC training SHALL use the same env, reward configuration, terrain configuration, total timesteps, and policy net architecture as the corresponding PPO run unless explicitly diverged in config.

#### Scenario: 預設 config 與 PPO 對齊
- **GIVEN** `configs/ppo_default.yaml` 與 `configs/sac_default.yaml`
- **WHEN** 比對 `env`、`reward`、`terrain`、`total_timesteps`、`policy_kwargs.net_arch` 欄位
- **THEN** 五項 MUST 完全相同

### Requirement: 失敗情境的可觀測性
The system SHALL surface NaN and plateau conditions identically to the PPO pipeline.

#### Scenario: NaN 立即中止
- **GIVEN** 訓練中 critic loss 出現 NaN
- **WHEN** callback 偵測到
- **THEN** 訓練 MUST 立即停止
- **AND** 最後 checkpoint 路徑記錄於 `train.log`

#### Scenario: 長時間無改善 → 警告
- **GIVEN** `ep_rew_mean` 連續 20% 預算未超過歷史最佳 + 5%
- **WHEN** 訓練到該檢查點
- **THEN** stdout / `train.log` SHALL 印出明顯警告與後續操作建議
