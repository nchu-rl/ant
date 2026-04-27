## ADDED Requirements

### Requirement: 載入訓練 run
The system SHALL load a trained run by directory, including the policy weights, VecNormalize statistics, and the original training config.

#### Scenario: 正常 PPO run 可載入
- **GIVEN** 一個包含 `final.zip`、`final_vec_normalize.pkl`、`config.yaml` 的 run 目錄
- **WHEN** 呼叫 `evaluate(run_dir, held_out_path)`
- **THEN** 載入 MUST 成功
- **AND** policy `model.predict(obs)` 可呼叫不報錯

#### Scenario: 缺 VecNormalize 的 run 拒絕
- **GIVEN** 一個缺 `final_vec_normalize.pkl` 的 run 目錄
- **WHEN** 呼叫 `evaluate(...)`
- **THEN** 系統 MUST 拋出 `FileNotFoundError` 並提示要從 checkpoints 恢復

#### Scenario: 不匹配的 VecNormalize 被偵測
- **GIVEN** 一個 run，其 `final.zip` 與 `final_vec_normalize.pkl` 並非來自同一訓練（obs shape 不一致）
- **WHEN** 嘗試載入
- **THEN** 系統 MUST 拋出 `RuntimeError` 並列出不匹配欄位

### Requirement: VecNormalize 凍結
During evaluation, the system SHALL set `VecNormalize.training=False` and `norm_reward=False`.

#### Scenario: 評估中 VecNormalize 統計不變
- **GIVEN** 載入後的 eval env
- **WHEN** 跑完 100 episode
- **THEN** VecNormalize 內部 `obs_rms.mean` 與 `obs_rms.var` MUST 與載入時相同（容差 0）

### Requirement: Deterministic 評估（預設）
The evaluator SHALL by default use `deterministic=True` when calling the policy.

#### Scenario: 預設 deterministic 一致性
- **GIVEN** 兩次呼叫 `evaluate(run_dir, held_out_path)`（相同檔案，預設參數）
- **WHEN** 比對輸出 `per_episode[i]["episode_return"]`
- **THEN** 兩次結果 MUST 逐元素相等

#### Scenario: 顯式 stochastic 模式可切換
- **GIVEN** `evaluate(..., deterministic=False)`
- **WHEN** 連續呼叫兩次
- **THEN** 至少有一個 episode 的 return 會不同（取樣引入方差）

### Requirement: Held-out 與訓練 seed 不重疊
The held-out terrain set seeds SHALL be disjoint from the training seed range, as documented in the held-out config.

#### Scenario: held_out_v1 seed 在指定範圍
- **GIVEN** `configs/held_out_v1.yaml`
- **WHEN** 解析所有 episode 的 seed
- **THEN** 所有 seed MUST ≥ 1_000_000

### Requirement: 指標完整性
For every evaluation episode, the system SHALL record episode_return、forward_distance_m、survival_steps、fall_rate、mean_ctrl_cost、mean_contact_cost、reward_decomposition (4 項).

#### Scenario: 單 episode 結果含全部欄位
- **GIVEN** 一次 evaluate 完成
- **WHEN** 檢查任一 `per_episode` entry
- **THEN** 所有上述 keys MUST 存在
- **AND** 數值型 metric MUST 為 finite（非 NaN / Inf）

#### Scenario: per_terrain 摘要正確分組
- **GIVEN** held-out v1（4 種 terrain × 25 episode）
- **WHEN** 檢查 `per_terrain`
- **THEN** keys MUST 為 `{"plain", "rough", "slope", "stairs"}`
- **AND** 每個 terrain bucket MUST 含 25 episode 之 mean / std

### Requirement: 翻覆率定義
`fall_rate` SHALL be 1 if the episode terminated due to unhealthy condition (`terminated=True`), and 0 if it ended via `truncated=True` (max steps reached).

#### Scenario: 達 max_steps 不算翻覆
- **GIVEN** 一個 episode 跑滿 1000 步而 `terminated=False, truncated=True`
- **WHEN** 計算 metric
- **THEN** `fall_rate` MUST 為 0

#### Scenario: unhealthy 提前結束算翻覆
- **GIVEN** 一個 episode 在 200 步因 unhealthy 結束（`terminated=True`）
- **WHEN** 計算 metric
- **THEN** `fall_rate` MUST 為 1

### Requirement: 報告產出
The system SHALL write `eval.json`、`episode_logs.csv`、`eval.md` to `<run_dir>/eval/<held_out_id>/`.

#### Scenario: 三個檔案皆產出
- **GIVEN** evaluate 完成
- **WHEN** 檢查輸出目錄
- **THEN** 上述三檔 MUST 存在
- **AND** `eval.md` MUST 包含 per-terrain × per-metric 的表格

#### Scenario: eval.md 含關鍵 metrics 列
- **GIVEN** `eval.md`
- **WHEN** 解析 markdown
- **THEN** 表格 column MUST 含 `episode_return`、`forward_distance_m`、`fall_rate`

### Requirement: PPO / SAC 結果可並列
PPO 與 SAC 的 evaluation 輸出格式 SHALL identical so that downstream comparison tools can read them uniformly.

#### Scenario: 兩種演算法的 eval.json schema 等同
- **GIVEN** 一個 PPO run 的 `eval.json` 與一個 SAC run 的 `eval.json`
- **WHEN** 解析兩者的 top-level keys
- **THEN** key set MUST 完全相同

### Requirement: Held-out config 不可變
A held-out config file with a fixed identifier (e.g., `held_out_v1.yaml`) MUST NOT be modified after first use; revisions SHALL produce a new versioned file (`held_out_v2.yaml`).

#### Scenario: 變更需新版本（流程性要求，由 PR review 強制）
- **GIVEN** PR 試圖修改 `configs/held_out_v1.yaml` 內任何 episode 設定
- **WHEN** PR review
- **THEN** 應拒絕該 PR 並要求改為新增 `held_out_v2.yaml`
- **AND** 此規則寫於 `configs/README.md`
