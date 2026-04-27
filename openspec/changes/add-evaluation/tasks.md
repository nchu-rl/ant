## 1. Held-out terrain config

- [ ] 1.1 [S] `configs/held_out_v1.yaml`：4 種 terrain × 25 個固定 seed = 100 episode；每 episode 寫死 (terrain_type, params, seed, max_steps=1000)
- [ ] 1.2 [S] `configs/README.md`：說明 held-out 不可變規則 + 版本化政策
- [ ] 1.3 [S] `src/eval/heldout.py`：YAML schema + 載入函式

## 2. Evaluator 主流程

- [ ] 2.1 [M] `src/eval/evaluator.py:evaluate(run_dir, held_out_path, deterministic=True)`
- [ ] 2.2 [M] 載入 model + VecNormalize：驗證 metadata（obs shape、algo type）；不匹配 → RuntimeError
- [ ] 2.3 [S] VecNormalize：set training=False、norm_reward=False
- [ ] 2.4 [M] 對 held-out 每個 episode：建立對應 terrain env、reset(seed)、rollout 至 terminate / truncate、收集 metrics

## 3. Metrics

- [ ] 3.1 [S] `src/eval/metrics.py`：定義七項 metric 計算
- [ ] 3.2 [S] reward_decomposition 從 `info["reward/*"]` 累加
- [ ] 3.3 [S] aggregate：per-terrain (4 桶) + overall

## 4. Reporter

- [ ] 4.1 [S] `eval.json` 寫入（per_episode + per_terrain + overall + metadata：run_id, held_out_id, git_hash, eval_timestamp）
- [ ] 4.2 [S] `episode_logs.csv` 寫入（每 episode 一列，欄位包含 metric + terrain_type + seed）
- [ ] 4.3 [M] `eval.md` 產出：人讀摘要表（per-terrain × per-metric mean ± std）；overall 列在最後

## 5. CLI

- [ ] 5.1 [S] CLI：`--run`、`--held-out`、`--deterministic/--stochastic`、`--output-dir`（預設 `<run>/eval/<held_out_id>`）

## 6. 測試

- [ ] 6.1 [S] `tests/test_eval_loader.py`：缺檔案 / 不匹配 → 適當例外
- [ ] 6.2 [M] `tests/test_eval_smoke.py`：以 PPO smoke run + 5 episode held-out 跑完不報錯
- [ ] 6.3 [S] `tests/test_eval_metrics.py`：fall_rate 對 truncated / terminated 行為正確
- [ ] 6.4 [S] `tests/test_eval_metrics.py`：reward_decomposition 加總 ≈ episode_return（容差 1e-6）
- [ ] 6.5 [S] `tests/test_eval_determinism.py`：兩次 deterministic 評估結果一致
- [ ] 6.6 [S] `tests/test_eval_seed_disjoint.py`：held_out_v1 所有 seed ≥ 1_000_000

## 7. 實際評估

- [ ] 7.1 [M] [GPU 30 min] PPO baseline run × held_out_v1
- [ ] 7.2 [M] [GPU 30 min] SAC baseline run × held_out_v1
- [ ] 7.3 [S] 比較產出，挑出明顯差異 terrain（給後續 reward / DR 微調作參考）

## 8. 文件

- [ ] 8.1 [S] `docs/evaluation.md`：metric 定義、CLI 用法、報告解讀範例
