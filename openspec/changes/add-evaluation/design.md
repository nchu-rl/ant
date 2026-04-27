## Context

簡報第四頁定義性能評估流程：「在隨機生成的複雜地形中測試機器人的強健性」。第十頁將「突破 30% 的城市地形障礙」列為目標。本 capability 的職責是把訓練成果轉成可比較的數字。

評估必須與訓練嚴格分離：
1. Held-out terrain seed 不可與訓練重疊。
2. VecNormalize 統計來自訓練 run，不可在評估期間更新。
3. Domain randomization 預設關閉（僅切換 terrain），讓比較變因可控。

## Goals / Non-Goals

**Goals:**
- 一條指令評估 PPO / SAC run。
- 統一的指標集，PPO / SAC 結果可直接並列。
- Held-out terrain set 可版本化（`held_out_v1.yaml`、未來 v2…）。
- 報告同時產出機器可讀（JSON / CSV）與人讀（Markdown 摘要表）。

**Non-Goals:**
- 不做訓練。
- 不做統計推論（t-test）— 留給未來擴充。

## Decisions

### D1. Deterministic policy 評估
**選擇**：呼叫 `model.predict(obs, deterministic=True)`。
**理由**：對齊 SB3 慣例、避免取樣方差污染比較。
**替代方案**：stochastic — 列為可選 flag，預設關閉。

### D2. VecNormalize 凍結
**選擇**：載入 `<run>/final_vec_normalize.pkl`，設 `training=False`、`norm_reward=False`。
**理由**：保留 obs 標準化的訓練分佈；reward 不再 normalize 才能跨 run 比較絕對分數。
**替代方案**：使用 evaluation 環境統計重新 normalize — 駁回，破壞跨 run 比較性。

### D3. Held-out set 設計
**選擇**：4 種 terrain × 25 個固定 seed = 100 episode。Seed 集合與訓練範圍 disjoint：
- 訓練 episode seeds：`[0, 100_000)`（pseudo-range；實際由訓練 RNG 派生，不顯式設定）。
- Held-out seeds：`[1_000_000, 1_000_025)` 對應每種 terrain 25 個。
**理由**：seed 上界足夠遠以保證不重疊。
**替代方案**：用 hash 派生 seed — 駁回，不易追蹤。

### D4. 指標清單
| 指標 | 定義 | 單位 |
|------|------|------|
| `episode_return` | Σ reward over episode | dimensionless |
| `forward_distance_m` | x_final − x_initial | meters |
| `survival_steps` | episode length | steps |
| `fall_rate` | 1 if `terminated` else 0 | bool → 平均 = 比例 |
| `mean_ctrl_cost` | mean of `info["reward/ctrl_cost"]` per step（取絕對值）| dimensionless |
| `mean_contact_cost` | mean of `info["reward/contact_cost"]` per step | dimensionless |
| `reward_decomposition` | 四項分量在 episode 內各自加總 | dimensionless |

按 terrain 分組 aggregate（mean / std），跨 terrain 再給 overall。

### D5. 報告格式
- `eval.json`：完整原始資料（每 episode 一行）。
- `episode_logs.csv`：同上，CSV。
- `eval.md`：人類可讀摘要表（per-terrain × per-metric 的 mean ± std）。

### D6. Held-out config 版本化
`configs/held_out_v1.yaml`：明列每個 episode 的 (terrain_type, terrain_params, seed, max_steps)。寫死後不可改；若調整 → 新版本檔。

## Risks / Trade-offs

- [風險] VecNormalize 不匹配 → 評估分數失真 → Mitigation：載入時驗證 obs shape、checkpoint metadata。
- [風險] Held-out 不夠多樣 → 結論不可信 → Mitigation：第一版用 100 episode；不足可擴至 v2。
- [Trade-off] 100 episode 評估在 GPU 上約需 10–30 分鐘，可接受。

## Migration Plan

Greenfield。

## Open Questions

- Held-out v1 在 stairs 上是否包含「機器人物理上不可能跨越」的 corner case？故意包，作為失敗壓力測試。
- 是否計算 success rate（forward_distance 達某門檻）？預設不算（門檻難定），先看 forward_distance 分佈。

## 關鍵介面

```python
# src/eval/evaluator.py
@dataclass
class EvalResult:
    run_id: str
    held_out_id: str
    per_episode: list[dict]      # one dict per episode
    per_terrain: dict[str, dict] # mean/std per metric per terrain
    overall: dict                # mean/std across all episodes

def evaluate(run_dir: str, held_out_path: str, deterministic: bool = True) -> EvalResult: ...

# CLI
# python -m eval.evaluator --run runs/ppo/<ts> --held-out configs/held_out_v1.yaml
```

## 訓練超參數
不適用（評估 only）。
