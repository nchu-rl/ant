## Why

簡報第九頁將「獎勵函數」稱為「導航研究的靈魂」，明確列出四個分量：r_forward、r_healthy、c_ctrl、c_contact。Gymnasium 內建 Ant-v4 的預設 reward 雖含類似項目，但：(1) 預設權重未必符合本研究目標、(2) 預設未懲罰軀幹接觸（c_contact）、(3) 後續多目標調校（前進速度 vs 能量消耗）需要單一可調介面。本 change 將獎勵抽出成可配置元件。

## What Changes

- 新增 `src/env/reward.py`：四項分量分別為純函式，`RewardConfig` 集中權重。
- 新增 reward wrapper：覆蓋 Ant-v4 預設 reward，使用本研究自訂版本，並將四項分量寫入 `info` dict 供 logging。
- 提供權重 sweep 工具（在 experiment-tracking change 中使用）的最小介面：`RewardConfig` 可序列化為 YAML / JSON。
- 文件化各分量數學定義與權重的物理意義。

## Capabilities

### New Capabilities
- `reward-design`: 自訂四項加權獎勵函數（r_forward、r_healthy、c_ctrl、c_contact）的計算、wrapper、設定、與分量觀測機制。

### Modified Capabilities
（無）

## Impact

- 程式碼：`src/env/reward.py`、修改 `src/env/factory.py` 加入可選 reward wrapper
- 後續 `add-ppo-training`、`add-sac-training` 將以本 capability 提供的 reward 訓練
- 後續 `add-experiment-tracking` 將 log 四項分量
- 後續 `add-evaluation` 評估時可分別比較 reward 分量

## Non-Goals

- 不引入額外 reward 項目（不加 forward velocity smoothness、joint limit penalty 等）— 嚴格遵守簡報四項。
- 不調整 observation space。
- 不負責超參數搜尋本身（屬於 ppo/sac training 的 hyperparameter sweep）。

## Assumptions & Open Questions

**Assumptions**
- 四項分量定義依簡報第九頁逐字稿與 Gymnasium Ant-v4 慣例：
  - `r_forward = forward_reward_weight * (x_t - x_{t-1}) / dt`
  - `r_healthy = healthy_reward` 當 z ∈ healthy_z_range 且未翻覆，否則 0
  - `c_ctrl = ctrl_cost_weight * Σ action²`
  - `c_contact = contact_cost_weight * 軀幹與地面接觸力²`（簡報明確指「軀幹」與地面接觸）
- 總 reward = r_forward + r_healthy − c_ctrl − c_contact（c_* 為正值，符號在加總時為負）。

**Open Questions**
- 各權重初值：兩份文件未提。預設沿用 Gymnasium Ant-v4 慣例（forward 1.0、healthy 1.0、ctrl 0.5、contact 5e-4），實際數值在 design.md 給合理範圍，待初步訓練調整。
- `healthy_z_range` 預設 [0.2, 1.0]m（簡報未指定）。
- `c_contact` 是否只算軀幹？還是含腿部？簡報明確只懲罰軀幹接觸，故僅監測 torso geom。
