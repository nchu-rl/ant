## 1. 介面與 backend 實作

- [ ] 1.1 [S] `src/tracking/__init__.py` + `src/tracking/logger.py:ExperimentLogger`（abstract）
- [ ] 1.2 [M] `TensorBoardLogger`：log_config（寫 hparams + 寫 `config.yaml`）、log_scalar、log_artifact（寫 artifacts.json + sha256）、finish（flush）
- [ ] 1.3 [M] `WandbLogger`：log_config（過濾 secrets）、log_scalar、log_artifact（`wandb.log_artifact`）、finish（`wandb.finish`）
- [ ] 1.4 [S] `MultiLogger`：fan-out 至每個 sub-logger
- [ ] 1.5 [M] `make_logger(run_dir, config)`：根據環境變數 + `tracking.yaml` 組合，含 W&B 失敗 fallback

## 2. 配置

- [ ] 2.1 [S] `configs/tracking.yaml`：`wandb.enabled`、`wandb.project`、`wandb.entity`（預設 `<TBD>`）、`wandb.tags_extra`、`video.enabled` (預設 false)
- [ ] 2.2 [S] secrets whitelist filter：明列允許 key，其餘 strip

## 3. 整合至訓練 / 評估

- [ ] 3.1 [M] PPO `train_ppo`：在 setup 階段 `make_logger(run_dir, full_config)` → `log_config`；訓練 callback 內 log reward 分量、DR 摘要；結束時 `log_artifact(final.zip, best.zip)` + `finish()`
- [ ] 3.2 [M] SAC `train_sac`：同上模式
- [ ] 3.3 [M] `evaluate(run_dir, ...)`：在每個 terrain bucket 完成後 log scalar，eval 報告完成後 `log_artifact(eval.json, eval.md)`

## 4. Runtime metadata 收集

- [ ] 4.1 [S] `src/tracking/runtime.py:collect_runtime() -> dict`（git_hash、git_dirty、python、torch、cuda、hostname、start_time）
- [ ] 4.2 [S] 整合至 `log_config` 自動加上

## 5. 測試

- [ ] 5.1 [S] `tests/test_logger_factory.py`：缺 API key → MultiLogger 不含 W&B
- [ ] 5.2 [S] `tests/test_logger_factory.py`：兩條件齊備 → 含 W&B（mock）
- [ ] 5.3 [S] `tests/test_logger_factory.py`：W&B init 拋例外 → fallback TB only，不向上傳遞
- [ ] 5.4 [S] `tests/test_logger_tb.py`：TB log_scalar 寫出 event 檔
- [ ] 5.5 [S] `tests/test_logger_tb.py`：log_artifact 缺檔案 → FileNotFoundError
- [ ] 5.6 [S] `tests/test_logger_tb.py`：`artifacts.json` 含 sha256
- [ ] 5.7 [S] `tests/test_secrets_filter.py`：意外的 secret key 被 strip

## 6. 文件

- [ ] 6.1 [S] `docs/experiment_tracking.md`：W&B setup 步驟（`wandb login`、`tracking.yaml`）、tag 規範、artifact 命名
- [ ] 6.2 [S] README 鏈結 + 一張 W&B / TB 截圖示意（截圖留 TODO，先放空白）
