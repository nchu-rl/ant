## Context

SAC 為 off-policy 演算法，與 PPO 在實作上有結構差異：
- SAC 需要 replay buffer（GB 等級）。
- SAC 通常以 `n_envs=1` 採樣，每 env step 後做 `gradient_steps` 次梯度更新。
- SAC 的 entropy 自動調整需要 target entropy 與 log_alpha optimizer。

本 change 在保留與 PPO 共用的 env / reward / terrain / checkpoint 邏輯下，建立平行的 SAC pipeline。

## Goals / Non-Goals

**Goals:**
- 與 `add-ppo-training` 結構鏡像：CLI、目錄、checkpoint、續訓、log。
- 公平對照：相同 env、reward、terrain、總 timesteps、policy net arch。
- 共用 utility：`src/training/common.py` 為 PPO / SAC 共用基底。

**Non-Goals:**
- 不導入 SAC 變體。
- 不做 PPO vs SAC 比較（屬 evaluation）。
- 不做 HPO。

## Decisions

### D1. 使用 SB3 SAC
**選擇**：`stable_baselines3.SAC` + `MlpPolicy`。
**理由**：成熟、與 PPO 共用 SB3 架構便於統一 callback / log。
**替代方案**：CleanRL / 自寫 — 駁回，研究專題時間有限。

### D2. 自動 entropy tuning
**選擇**：`ent_coef="auto"`，target entropy = `-action_dim` = -8（SB3 預設）。
**理由**：Haarnoja 2019 推薦；避免手調 α，提升不同 reward scale 下的穩定度。
**替代方案**：固定 α — 駁回，需另外 tune。

### D3. n_envs 與訓練節奏
**選擇**：`n_envs=1`、`train_freq=1` step、`gradient_steps=1`。
**理由**：對齊 SAC 主流實作；off-policy 樣本效率不靠平行採樣。
**替代方案**：`n_envs>1` — SB3 支援但會偏離主流 baseline、難對照文獻。

### D4. 預設超參數（SB3 MuJoCo baseline）
| 參數 | 預設 | 範圍 |
|------|------|------|
| `learning_rate` | 3e-4 | [1e-4, 1e-3] |
| `buffer_size` | 1_000_000 | [5e5, 2e6] |
| `learning_starts` | 10_000 | [1e3, 1e5] |
| `batch_size` | 256 | {128, 256, 512} |
| `tau` | 0.005 | [0.001, 0.02] |
| `gamma` | 0.99 | [0.95, 0.999] |
| `train_freq` | 1 | {1, 4} (steps) |
| `gradient_steps` | 1 | {1, 4} |
| `ent_coef` | "auto" | {"auto", float} |
| `policy_kwargs.net_arch` | [256, 256] | — |
| `total_timesteps` | 5e6 | 1e6–1e7 |

**所有值為合理初始範圍，最終由初步訓練曲線判讀調整。**

### D5. 與 PPO 對等的 run 目錄
```
runs/sac/<YYYYMMDD-HHMMSS>/
├── config.yaml
├── checkpoints/step_<N>.zip
├── replay_buffer.pkl   # 可選，續訓時必要
├── best.zip
├── final.zip
├── tb/
└── train.log
```

### D6. Replay buffer 持久化
**選擇**：續訓時可選 `--save-replay-buffer`，預設關閉（檔案大）。
**理由**：完整續訓需要 buffer，但 buffer 通常 1–4 GB；多數情況可接受 fresh-start buffer。
**替代方案**：強制存 buffer — 駁回，磁碟成本過高。

### D7. 與 PPO 共用 utility
抽取至 `src/training/common.py`：
- run_dir 建立、config 快照、git hash、log redirect
- env factory wrapper（套 reward / terrain / VecNormalize）
- checkpoint metadata 寫入
- NaN / plateau callback

## Risks / Trade-offs

- [風險] SAC 在不平整地形 reward shaping 下可能被熵項主導，產生隨機亂動 → Mitigation：監控 `ent_coef_loss`、必要時固定 α。
- [風險] Buffer 1e6 在 27 維 obs + 8 維 action 下約 ~0.5 GB，在多 env 下倍增 → Mitigation：保持 `n_envs=1`。
- [Trade-off] 不存 buffer → 中斷續訓後 buffer 是空的，會有效能 dip → 視專題時間取捨。

## Migration Plan

Greenfield。

## Open Questions

- 是否要在 SAC 採用 `VecNormalize`？SB3 預設不建議 reward normalize on off-policy（會破壞 buffer 樣本分佈）。預設僅 `norm_obs=True, norm_reward=False`。
- 是否與 PPO 共用 `EvalCallback` 環境？是 — eval env 需獨立但配置相同，避免 reward / VecNormalize 污染。

## 關鍵介面

```python
# src/training/sac.py
def train_sac(config_path: str, resume_from: str | None = None) -> str: ...

# CLI
# python -m training.sac --config configs/sac_default.yaml [--resume-from runs/sac/<ts>] [--save-replay-buffer]
```

## 訓練超參數
完整見 D4。所有值寫入 `configs/sac_default.yaml`。
