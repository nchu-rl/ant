## Context

SB3 內建 TensorBoard logger 在訓練時會自動記錄 `rollout/ep_rew_mean`、`train/loss` 等。但本研究還需 log：
- 自訂 reward 分量（r_forward、r_healthy、c_ctrl、c_contact）
- DR 每 episode 取樣摘要
- 評估 metrics（forward_distance、fall_rate 等）
- 整個訓練 / 評估 config snapshot
- checkpoint / final.zip / eval 報告當作 artifact

W&B 提供更好的多 run 比較與 sharable dashboard，但對網路與 API key 有依賴；故必須是「可選擇 fallback 到 TB only」。

## Goals / Non-Goals

**Goals:**
- 一個統一的 `ExperimentLogger` 介面，PPO / SAC / Eval 共用。
- TensorBoard 永遠開、W&B 條件開啟。
- Config 快照、scalar、artifact 三類 logging 都覆蓋。
- 加入 W&B 不會增加訓練 critical path 的失敗風險（網路斷線 → degrade 而不 crash）。

**Non-Goals:**
- 不做 HPO / Sweeps。
- 不做 video logging（預設）。
- 不取代 SB3 內建 TB callback（本 logger 與其並存）。

## Decisions

### D1. 統一介面 `ExperimentLogger`
**選擇**：抽象基底類別，方法 `log_config`、`log_scalar`、`log_artifact`、`finish`。實作 `TensorBoardLogger`、`WandbLogger`、`MultiLogger`（合成）。
**理由**：呼叫端不需知道 backend 數量；增刪 backend 只在 instantiation 處改。

### D2. W&B 啟用條件
**選擇**：必須同時滿足 (a) 環境變數 `WANDB_API_KEY` 已設定 (b) `configs/tracking.yaml` 中 `wandb.enabled=true`。否則只啟用 TB。
**理由**：避免 CI / 本機開發誤上傳；雙條件比單條件更明確。

### D3. 失敗降級
**選擇**：W&B `init` 或 `log` 失敗（網路、API quota）→ catch、印警告、降級為 TB only，**不可** crash 訓練。
**理由**：訓練 run 動輒數小時，不能因網路一閃即死。

### D4. Config 快照欄位
從訓練 config 拷貝，加上 runtime 資訊：
```yaml
algo: ppo|sac
total_timesteps: int
seed: int
env: { ... }
reward: { ... }
terrain: { ... }
randomization: { ... }
hyperparams: { ... }
runtime:
  git_hash: str
  git_dirty: bool
  python_version: str
  torch_version: str
  cuda_available: bool
  hostname: str
  start_time_iso: str
```

### D5. Scalar 命名空間
- SB3 自動：`rollout/*`、`train/*`、`time/*`
- 自訂 reward：`reward/forward`、`reward/healthy`、`reward/ctrl_cost`、`reward/contact_cost`、`reward/total`
- DR：`dr/body_mass_multiplier_mean`、`dr/friction_multiplier_mean`、`dr/terrain_type_*_freq`（每 N episode 統計）
- Eval：`eval/<terrain>/<metric>` (e.g., `eval/rough/forward_distance_m`)

### D6. Artifact 上傳
- TensorBoard：不支援 artifact，只記錄路徑與 metadata 至 `runs/<algo>/<ts>/artifacts.json`。
- W&B：透過 `wandb.log_artifact`，類別為 `model` / `eval-report`。
- final.zip 與 best.zip 各上傳一次，eval 報告每次 evaluate 上傳一次。

### D7. Tags 與 grouping
W&B 預設 tags：`[algo, terrain_set, dr_preset]`。group：`run_<run_id>`。便於 dashboard 多 run 比較。

## Risks / Trade-offs

- [風險] W&B 上傳 artifact（model 約 1–10 MB）增加訓練尾段時間 → Mitigation：只上傳 final + best；中間 checkpoint 留本地。
- [風險] 機敏 config 進 W&B（API key、私網 hostname）→ Mitigation：上傳前過濾敏感 key（whitelist）。
- [Trade-off] 雙 backend 增加實作複雜度 → 介面已抽象，呼叫端零負擔。

## Migration Plan

Greenfield。

## Open Questions

- W&B entity 預設留白還是放置佔位（如 `<your-entity>`）？暫填 `<TBD>`，setup 文件提示填入。
- 是否在 PR 中強制要求 W&B run 連結？預設不要求，由團隊規範。

## 關鍵介面

```python
# src/tracking/logger.py
class ExperimentLogger:
    def log_config(self, config: dict) -> None: ...
    def log_scalar(self, name: str, value: float, step: int) -> None: ...
    def log_artifact(self, path: str, name: str, type: str) -> None: ...
    def finish(self) -> None: ...

class TensorBoardLogger(ExperimentLogger): ...
class WandbLogger(ExperimentLogger): ...
class MultiLogger(ExperimentLogger):
    def __init__(self, loggers: list[ExperimentLogger]): ...

def make_logger(run_dir: str, config: dict) -> ExperimentLogger:
    """根據 configs/tracking.yaml 與環境變數決定組合。"""
```

## 訓練超參數
不適用（本 change 不訓練模型）。
