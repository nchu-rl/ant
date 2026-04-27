## Why

簡報第四頁「性能評估」與第十頁「貢獻」明確要求：在隨機生成的複雜地形中測試 robustness、確認模型「不是背下步態」、量化「能突破 30% 城市地形障礙」。沒有獨立、可重現的評估 pipeline，訓練成果無法被科學地比較（PPO vs SAC、不同 reward 權重、不同 DR 強度）。

## What Changes

- 新增 `src/eval/evaluator.py`：載入訓練好的 policy + VecNormalize，在指定地形 set 上 rollout 收集 metrics。
- 新增 held-out terrain set：固定 seed 的 N 個 episode 配置（plain × M、rough × M、slope × M、stairs × M），確保訓練未見。
- 新增評估指標：episode return（總分）、平均 X 軸位移（forward distance）、生存時長（survival steps）、翻覆率（fall rate）、控制成本（mean ctrl cost / energy proxy）、reward 分量分解。
- 新增報告產出：`reports/<run_id>/eval.json` 與人類可讀的 `eval.md` 摘要表。
- 新增 CLI：`python -m eval.evaluator --run runs/ppo/<ts> --held-out configs/held_out_v1.yaml`。

## Capabilities

### New Capabilities
- `evaluation`: 訓練後策略在 held-out terrain set 上的 rollout、量化評估指標、報告產出機制。

### Modified Capabilities
（無）

## Impact

- 程式碼：`src/eval/evaluator.py`、`configs/held_out_v1.yaml`、`src/eval/metrics.py`
- 產出：`runs/{ppo,sac}/<ts>/eval/<held_out_id>/{eval.json, eval.md, episode_logs.csv}`
- 相依：`add-env-setup`、`add-terrain-generation`、`add-reward-design`、`add-ppo-training`（載入模型）、`add-sac-training`（載入模型）
- 被相依：成果報告（人工撰寫，不在 OpenSpec 範圍）

## Non-Goals

- 不負責訓練本身。
- 不做學術級統計檢定（t-test、bootstrap CI）— 報告中只提供 mean ± std；後續若需可擴充。
- 不導入 sim-to-real 評估。

## Assumptions & Open Questions

**Assumptions**
- Held-out set 預設 4 種地形 × 25 個 seed = 100 episode，episode_max_steps = 1000。
- 評估時 randomization 設為 `None`，僅切換 terrain config，避免引入 DR 雜訊干擾比較。
- `held_out_v1.yaml` 的 seed 在訓練 config 的隨機種子之外（避免訓練看過）。

**Open Questions**
- 評估時 policy 採 deterministic action（mean）或 stochastic（取樣）？預設 deterministic（與一般 SB3 evaluation 慣例一致）。
- 翻覆率定義：episode 因 unhealthy 提前終止 → 算翻覆。是否將 truncation（達 max_steps）視為「未翻覆」？是。
- 是否評估能耗的物理意義（焦耳）而非 dimensionless ctrl cost？預設 dimensionless（公式 Σ action²），物理單位列為未來擴充。
