## Why

研究專題會跑大量訓練 run（PPO、SAC、不同 reward 權重、不同 DR 強度、不同 terrain 配置）。沒有統一的實驗追蹤介面：(1) 每 run 的 config 與結果易遺失、(2) 比較不同 run 必須手動翻 log、(3) 簡報結尾的「貢獻表格」難以重現。

config.yaml 預設 TensorBoard 為內建 logger、Weights & Biases 為選用。本 change 把這兩者抽象為一致的 logging 介面，與訓練 / 評估 pipeline 接合。

## What Changes

- 新增 `src/tracking/logger.py`：統一介面 `ExperimentLogger`，方法包括 `log_config(dict)`、`log_scalar(name, value, step)`、`log_artifact(path)`、`finish()`。
- 提供兩個實作：`TensorBoardLogger`（永遠啟用）、`WandbLogger`（選用，依環境變數 `WANDB_API_KEY` 啟用）。
- 訓練 pipeline 在啟動時 instantiate 兩個 logger（合成 `MultiLogger`），把超參數、reward 分量、DR 取樣摘要、checkpoint 路徑等送進 logger。
- 評估 pipeline 同樣支援：把 eval 報告 metrics 推送至 logger。
- 新增 `configs/tracking.yaml`：W&B project / entity / tags 等設定（含本機開發開關）。

## Capabilities

### New Capabilities
- `experiment-tracking`: 統一的訓練 / 評估實驗追蹤介面，支援 TensorBoard 與 Weights & Biases 雙後端，含 config 快照、scalar 指標、artifact 上傳、tag 與 group 機制。

### Modified Capabilities
（無）

## Impact

- 程式碼：`src/tracking/`、修改 `src/training/{ppo,sac}.py` 與 `src/eval/evaluator.py` 增加 logging hook
- 新增依賴：`wandb`（可選）；TensorBoard 已由 SB3 引入
- 相依：`add-env-setup`、`add-ppo-training`、`add-sac-training`、`add-evaluation`、`add-reward-design`、`add-domain-randomization`
- 被相依：未來成果報告會直接從 W&B / TensorBoard 取圖

## Non-Goals

- 不導入 MLflow / Sacred 等其他追蹤系統。
- 不負責長期 storage backup（TensorBoard logs 留在本機，W&B 由其雲端負責）。
- 不取代 `runs/<algo>/<ts>/` 本地存檔機制。

## Assumptions & Open Questions

**Assumptions**
- TensorBoard logger 永遠開啟（SB3 內建 + 自訂 scalar）。
- W&B 為選用，由環境變數 `WANDB_API_KEY` + `configs/tracking.yaml.enabled=true` 雙條件啟用；任一缺失則 fallback 至 TB only。
- W&B project 名稱預設 `ant-rl-terrain`（本研究專題代號）。

**Open Questions**
- 團隊 W&B entity（個人或組織）兩份文件未提；預設留白，由團隊填入。
- 是否啟用 W&B Sweeps 做 HPO？本 change 不啟用，留 hook 給未來。
- 是否 log 影片 rollout（W&B `Video`）？預設關閉（render 成本高），保留 CLI flag。
