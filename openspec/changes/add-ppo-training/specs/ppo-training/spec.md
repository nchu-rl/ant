## ADDED Requirements

### Requirement: PPO 訓練可由 YAML config 啟動
The system SHALL launch a complete PPO training run from a single YAML config file via CLI.

#### Scenario: 使用預設 config 跑短訓練不報錯
- **GIVEN** `configs/ppo_default.yaml` 與已安裝環境
- **WHEN** 執行 `python -m training.ppo --config configs/ppo_default.yaml --total-timesteps 10000`
- **THEN** 訓練 MUST 在規定 timesteps 內結束
- **AND** 在 `runs/ppo/<ts>/` 產出 `final.zip`、`final_vec_normalize.pkl`、`config.yaml`、`tb/` 目錄

#### Scenario: 缺欄位的 config 即時拒絕
- **GIVEN** YAML config 缺 `total_timesteps` 必填欄位
- **WHEN** CLI 啟動
- **THEN** 系統 MUST 在訓練開始前拋出 `ValueError`，列出缺失欄位

### Requirement: Run 完整快照
Every training run SHALL persist a complete config snapshot under the run directory so the run can be reproduced.

#### Scenario: run config 與啟動 config 等價
- **GIVEN** 一次完整訓練 run
- **WHEN** 比較啟動的 `configs/ppo_default.yaml` 與 `runs/ppo/<ts>/config.yaml`
- **THEN** 訓練相關欄位 MUST 完全等價（含 reward weights、terrain config、PPO hyperparams、seed、git commit hash）

### Requirement: VecNormalize 狀態與 checkpoint 同步
At every checkpoint the system SHALL persist both the policy weights and the VecNormalize statistics.

#### Scenario: 每個 checkpoint 配對存在
- **GIVEN** 一次訓練到 200k steps
- **WHEN** 檢視 `checkpoints/`
- **THEN** 每個 `step_<N>.zip` MUST 對應一個 `vec_normalize_<N>.pkl`

#### Scenario: 評估時載入錯誤的 VecNormalize 會被偵測
- **GIVEN** 一個訓練好的 model 與一個明顯不匹配的 `vec_normalize_*.pkl`（不同訓練 run）
- **WHEN** evaluation pipeline 嘗試載入
- **THEN** 系統 MUST 在比對 metadata（如 obs shape、checkpoint step）後拋出 RuntimeError

### Requirement: 續訓
The system SHALL support resuming training from a previous run directory and continue toward the original `total_timesteps` budget.

#### Scenario: 續訓從最新 checkpoint 接上
- **GIVEN** 一個訓練到 100k steps 的 run（總預算 200k）
- **WHEN** 執行 `python -m training.ppo --resume-from runs/ppo/<ts>`
- **THEN** 訓練 MUST 從 100k steps 繼續，至 200k 結束
- **AND** 訓練曲線在 100k 連續無重置

#### Scenario: 續訓完成後 final.zip 為新版本
- **GIVEN** 上述續訓完成
- **WHEN** 載入 `final.zip` 與 `final_vec_normalize.pkl`
- **THEN** 模型可以正確 inference，不拋例外

### Requirement: 平行採樣
The system SHALL use vectorized environments (`SubprocVecEnv` by default) for sample collection.

#### Scenario: n_envs 設定生效
- **GIVEN** config `n_envs: 4`
- **WHEN** 啟動訓練並檢查 SB3 內部 vec env
- **THEN** vec env 環境數量 MUST 等於 4

#### Scenario: n_envs=1 時 fallback DummyVecEnv
- **GIVEN** config `n_envs: 1`
- **WHEN** 啟動訓練
- **THEN** 系統 SHOULD 使用 `DummyVecEnv` 以利除錯

### Requirement: TensorBoard logs
The system SHALL write SB3-style TensorBoard logs to `<run_dir>/tb/`.

#### Scenario: TensorBoard 可開啟並顯示 reward
- **GIVEN** 一個訓練至少 10k steps 的 run
- **WHEN** 執行 `tensorboard --logdir runs/ppo/<ts>/tb`
- **THEN** 可看到 `rollout/ep_rew_mean` 等標準 SB3 metrics

### Requirement: 失敗 / 不收斂的可觀測性
When training rewards stop improving for an extended period, the system SHALL surface this in the run logs so the operator can act.

#### Scenario: 長時間無改善時印警告
- **GIVEN** 訓練連續 20% 的 timesteps `ep_rew_mean` 未超過歷史最佳 + 5%
- **WHEN** 訓練到該檢查點
- **THEN** 系統 SHALL 在 stdout / `train.log` 印出明顯警告
- **AND** 警告 MUST 提供可選續訓 / 中止指引

#### Scenario: NaN reward / loss 立即中止
- **GIVEN** 訓練過程中出現 NaN 在 reward 或 policy loss
- **WHEN** SB3 callback 偵測到
- **THEN** 訓練 MUST 立即停止
- **AND** stack trace + 最後 checkpoint 路徑必須記錄於 `train.log`

### Requirement: Seed 完整可控
Training SHALL be reproducible across runs given the same config (env seed, torch seed, vec env seeds).

#### Scenario: 兩次相同 config 的訓練前 100 step return 一致
- **GIVEN** 相同 YAML config 與 seed
- **WHEN** 在相同硬體連跑兩次
- **THEN** 前 100 step 的 `info["reward/total"]` 序列 MUST 逐元素相等（容差 1e-6）
- **AND** 此測試只在 `n_envs=1` + 確定性後端設定下保證
