## Context

本 capability 是研究主軸：簡報第七頁明確採 PPO，第四頁強調神經網路大規模並行訓練。PPO 的核心優勢（Schulman 2017）為 trust region 近似 + clipped surrogate objective，避免單步策略更新過大導致崩潰。在 MuJoCo Ant-v4 上，PPO 為公開 benchmark 中收斂最穩定的 on-policy 方法之一。

Stable-Baselines3 的 `PPO` 已封裝完整、且與 Gymnasium / `VecNormalize` 整合，故本 change 直接套用。

## Goals / Non-Goals

**Goals:**
- 一條指令即可重現整個訓練：`python -m training.ppo --config <yaml>`。
- 訓練配置（env、reward、terrain、algo hyperparams、seed）全部寫入 YAML，run 完整保留 config snapshot。
- 支援 vectorized env、checkpoint、續訓、TensorBoard log。
- 訓練結果可被 `add-evaluation` 直接載入。

**Non-Goals:**
- 不做 HPO。
- 不負責 W&B 整合（屬 `add-experiment-tracking`，本 change 留 hook）。
- 不負責 multi-terrain 混訓策略（屬 `add-domain-randomization`）。

## Decisions

### D1. 使用 SB3 PPO
**選擇**：`stable_baselines3.PPO` + `MlpPolicy`。
**理由**：成熟、文件完備、社群有大量 Ant-v4 baseline，加速調試。
**替代方案**：CleanRL（教學風格）、自寫 PyTorch — 駁回，研究專題時間有限。

### D2. VecNormalize obs + reward
**選擇**：`VecNormalize(norm_obs=True, norm_reward=True)`。
**理由**：MuJoCo 連續控制 RL 的標準做法；穩定訓練。
**替代方案**：不 normalize — 駁回，已知會降低 PPO 收斂性。

### D3. SubprocVecEnv（多進程）
**選擇**：訓練時 `SubprocVecEnv(n=8)`；smoke / debug 時 `DummyVecEnv(n=1)`。
**理由**：MuJoCo 在單進程下 GIL 限制 throughput；多進程顯著加速。
**替代方案**：DummyVecEnv 串行 — 慢，僅 debug 用。

### D4. 預設超參數（SB3 MuJoCo baseline）
| 參數 | 預設 | 範圍 |
|------|------|------|
| `learning_rate` | 3e-4 | [1e-4, 1e-3] |
| `n_steps` | 2048 | {1024, 2048, 4096} |
| `batch_size` | 64 | {32, 64, 128} |
| `n_epochs` | 10 | [5, 20] |
| `gamma` | 0.99 | [0.95, 0.999] |
| `gae_lambda` | 0.95 | [0.9, 1.0] |
| `clip_range` | 0.2 | {0.1, 0.2, 0.3} |
| `ent_coef` | 0.0 | [0.0, 0.01] |
| `vf_coef` | 0.5 | — |
| `max_grad_norm` | 0.5 | — |
| `policy_kwargs.net_arch` | [256, 256] | — |
| `total_timesteps` | 5e6 | 1e6–1e7 |
| `n_envs` | 8 | {4, 8, 16} |

**所有值均為合理初始範圍，最終值由初步訓練曲線判讀後微調。**

### D5. Checkpoint 策略
- 每 100k env steps 存一次 `runs/ppo/<ts>/checkpoints/step_<N>.zip` + `vec_normalize_<N>.pkl`。
- 訓練結束存 `final.zip`。
- 「最佳」模型：以 episode return 平均為準（每 50k steps 評估 5 個 episode），存於 `best.zip`。

### D6. 續訓介面
**選擇**：`--resume-from <run_dir>` 自動載入最新 checkpoint + VecNormalize 狀態 + 已用 timesteps。
**理由**：研究會多次中斷訓練，續訓必要。

### D7. Run 目錄結構
```
runs/ppo/<YYYYMMDD-HHMMSS>/
├── config.yaml         # 訓練啟動時的完整快照
├── checkpoints/
│   ├── step_100000.zip
│   ├── vec_normalize_100000.pkl
│   └── ...
├── best.zip
├── final.zip
├── final_vec_normalize.pkl
├── tb/                  # TensorBoard logs
└── train.log            # stdout 鏡像
```

## Risks / Trade-offs

- [風險] PPO 在複雜地形下可能不收斂 → Mitigation：先在 plain 上跑 baseline 收斂，再開 terrain randomization；保留 baseline run 供對照。
- [風險] VecNormalize 狀態未存 → 評估時 obs 分佈錯位 → Mitigation：每次 checkpoint 同步存 `.pkl`。
- [Trade-off] 平行環境多 → CPU 競用、記憶體增加；少 → 慢。預設 8，視機器調整。
- [風險] 訓練不收斂時無早停機制 → Mitigation：log 監控 + 人工早停；訓練 20% 仍無上升即中止。

## Migration Plan

Greenfield。

## Open Questions

- 觀察 / 動作的標準化是否要與 evaluation 對齊？是 — eval 必須載入相同 VecNormalize 狀態，spec 已要求。
- 是否要支援多 GPU？預設不支援（SB3 PPO 單 GPU 即可，Ant-v4 規模不需要）。

## 關鍵介面

```python
# src/training/ppo.py
def train_ppo(config_path: str, resume_from: str | None = None) -> str: ...  # returns run_dir

# CLI
# python -m training.ppo --config configs/ppo_default.yaml [--resume-from runs/ppo/20260427-120000]
```

## 訓練超參數
完整見 D4。所有值寫入 `configs/ppo_default.yaml` 並於每次 run 拷貝至 `runs/ppo/<ts>/config.yaml`。
