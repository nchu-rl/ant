## ADDED Requirements

### Requirement: 統一 logger 介面
The system SHALL expose a single `ExperimentLogger` abstraction that PPO training, SAC training, and evaluation all use without knowing the backend.

#### Scenario: 訓練端不需知道 backend 數量
- **GIVEN** PPO 訓練腳本透過 `make_logger(run_dir, config)` 取得 logger
- **WHEN** 跑 100 step 並各自 `log_scalar`
- **THEN** 訓練腳本程式碼 MUST 完全不引用 `wandb` 或 `torch.utils.tensorboard` 任一模組

### Requirement: TensorBoard 永遠啟用
TensorBoard logging SHALL always be active, regardless of `tracking.yaml` or environment.

#### Scenario: 不設 WANDB_API_KEY 時 TB 仍寫入
- **GIVEN** `WANDB_API_KEY` 未設定、`configs/tracking.yaml.wandb.enabled=false`
- **WHEN** 執行 PPO smoke 訓練
- **THEN** `runs/ppo/<ts>/tb/` 目錄 MUST 存在且包含 event 檔
- **AND** `tensorboard --logdir <ts>/tb` 可顯示 metrics

### Requirement: W&B 雙條件啟用
W&B logging SHALL only activate when BOTH the env var `WANDB_API_KEY` is set AND `configs/tracking.yaml.wandb.enabled` is true.

#### Scenario: 缺 API key 時 fallback TB only
- **GIVEN** `tracking.yaml.wandb.enabled=true`，`WANDB_API_KEY` 未設定
- **WHEN** `make_logger(...)` 呼叫
- **THEN** 回傳的 `MultiLogger` MUST 不包含 `WandbLogger`
- **AND** stdout SHALL 印出 fallback 訊息

#### Scenario: 兩條件齊備時 W&B 啟用
- **GIVEN** `tracking.yaml.wandb.enabled=true`，`WANDB_API_KEY` 已設定
- **WHEN** `make_logger(...)` 呼叫
- **THEN** 回傳的 `MultiLogger` MUST 同時包含 `TensorBoardLogger` 與 `WandbLogger`

### Requirement: W&B 失敗不可中斷訓練
If `WandbLogger.init` or any `log_*` call raises a network or quota error, the system SHALL catch it, log a warning, and continue with TB only.

#### Scenario: W&B init 失敗 → 降級
- **GIVEN** W&B init 模擬拋出 `ConnectionError`
- **WHEN** `make_logger(...)` 執行
- **THEN** 不可向呼叫端 propagate 例外
- **AND** 回傳的 logger MUST 仍含 TB
- **AND** `train.log` 必有 warning 記錄此降級

#### Scenario: 訓練中 W&B log 失敗 → 訓練繼續
- **GIVEN** 訓練進行中 `WandbLogger.log_scalar` 拋網路錯誤
- **WHEN** 下一 step 呼叫 logger
- **THEN** 訓練 MUST 繼續未中斷
- **AND** 該錯誤不能重複污染 stdout（rate limit warning）

### Requirement: Config 快照
Logger SHALL persist a complete config snapshot at training start, with runtime metadata.

#### Scenario: log_config 寫入完整欄位
- **GIVEN** 一次訓練啟動
- **WHEN** logger.log_config 完成
- **THEN** TB 的 hparams 與 W&B（若啟用）的 config 都 MUST 含 `algo`、`total_timesteps`、`seed`、`env`、`reward`、`terrain`、`randomization`、`hyperparams`、`runtime.git_hash`

#### Scenario: 敏感欄位不上傳
- **GIVEN** config 內含意外的 `secrets.api_key` 欄位（防呆）
- **WHEN** 上傳至 W&B
- **THEN** 該欄位 MUST 被 strip 掉，不出現在 W&B run config

### Requirement: Scalar 命名空間
Custom scalar names SHALL follow the namespace defined in design.md D5.

#### Scenario: 訓練中 reward 分量曲線可見
- **GIVEN** 一次 PPO 訓練至少 10k steps
- **WHEN** 開啟 TensorBoard
- **THEN** 必有 scalar `reward/forward`、`reward/healthy`、`reward/ctrl_cost`、`reward/contact_cost`、`reward/total`

#### Scenario: 評估指標寫入 eval 命名空間
- **GIVEN** 一次 evaluation 完成
- **WHEN** 檢視 logger 寫入
- **THEN** 必有 scalar `eval/plain/episode_return`、`eval/rough/forward_distance_m`、`eval/slope/fall_rate`、`eval/stairs/mean_ctrl_cost` 等

### Requirement: Artifact 上傳
Logger SHALL accept artifact upload requests and route them appropriately to each backend.

#### Scenario: final.zip 上傳成功
- **GIVEN** 訓練完成
- **WHEN** 呼叫 `logger.log_artifact(path="runs/ppo/<ts>/final.zip", name="final", type="model")`
- **THEN** TB backend SHALL 在 `runs/ppo/<ts>/artifacts.json` 增加一筆紀錄（path、name、type、timestamp、sha256）
- **AND** W&B backend（若啟用）SHALL 上傳 artifact

#### Scenario: 缺檔案時報錯
- **GIVEN** 不存在的 path
- **WHEN** `logger.log_artifact(path=...)`
- **THEN** 系統 MUST 拋 `FileNotFoundError`，並 **不可** 寫入 `artifacts.json`

### Requirement: Tag 與 group
W&B runs SHALL include tags `[algo, terrain_set, dr_preset]` and group `run_<run_id>` for dashboard filtering.

#### Scenario: tag 與 group 設定正確
- **GIVEN** 一個 PPO + conservative DR + held_out_v1 set 訓練
- **WHEN** 該 run 出現在 W&B dashboard
- **THEN** tags MUST 含 `ppo`、`held_out_v1`、`conservative`
- **AND** group MUST 為對應 `run_<run_id>`

### Requirement: 訓練 / 評估 finish 時清理
Logger SHALL flush and close backends at the end of training or evaluation.

#### Scenario: training 結束後 W&B run 標記為 finished
- **GIVEN** 一次完整訓練 + `logger.finish()` 呼叫
- **WHEN** 在 W&B dashboard 檢視該 run
- **THEN** 該 run 狀態 MUST 為 `finished`，不停留在 `running`
