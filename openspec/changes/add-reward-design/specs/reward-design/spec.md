## ADDED Requirements

### Requirement: 四項分量定義
The system SHALL compute reward as the weighted sum of exactly four components: `r_forward`, `r_healthy`, `c_ctrl`, `c_contact`, matching simulator step semantics defined in 簡報第九頁.

#### Scenario: 平面 + 靜止動作下的分量值符合預期
- **GIVEN** Ant 在平面、所有 action 為 0、torso z 在 healthy_z_range 內
- **WHEN** 執行一個 step
- **THEN** `info["reward/forward"]` MUST 約為 0（容差 1e-3）
- **AND** `info["reward/healthy"]` MUST 等於 `healthy_reward`
- **AND** `info["reward/ctrl_cost"]` MUST 為 0
- **AND** `info["reward/contact_cost"]` MUST 為 0（torso 未觸地）

#### Scenario: 大動作會增加 ctrl cost
- **GIVEN** 兩次 step：一次 action=零向量，一次 action=全 1
- **WHEN** 比較兩次 `info["reward/ctrl_cost"]`
- **THEN** action=全 1 的版本 MUST 顯著更大（差值 > ctrl_cost_weight × 4，因 8 維 action 平方和 = 8）

### Requirement: c_contact 僅針對軀幹
The contact cost SHALL only penalize contact forces between the `torso` body and the ground (`floor` or `hfield`), not leg contacts.

#### Scenario: 腿部接觸地面不被懲罰
- **GIVEN** Ant 站立於平面，所有腳掌接觸地面（torso 未觸地）
- **WHEN** 執行 step
- **THEN** `info["reward/contact_cost"]` MUST 為 0

#### Scenario: 軀幹倒地接觸時被懲罰
- **GIVEN** Ant 軀幹被強制翻倒至接觸地面
- **WHEN** 執行 step
- **THEN** `info["reward/contact_cost"]` MUST > 0
- **AND** 懲罰值 = `contact_cost_weight × Σ(torso 接觸力²)`

### Requirement: r_healthy 與翻覆判定
The system SHALL set `r_healthy` to `healthy_reward` only when torso z is inside `healthy_z_range`, otherwise 0.

#### Scenario: torso z 在範圍內 → 得到 healthy reward
- **GIVEN** Ant torso z = 0.5（落在 [0.2, 1.0] 內）
- **WHEN** 執行 step
- **THEN** `info["reward/healthy"]` == `healthy_reward`

#### Scenario: torso z 超出上界 → r_healthy = 0
- **GIVEN** Ant 被外力舉至 torso z = 2.0
- **WHEN** 執行 step
- **THEN** `info["reward/healthy"]` MUST 為 0

#### Scenario: 翻覆時 episode terminate（預設行為）
- **GIVEN** `RewardConfig(terminate_when_unhealthy=True)`
- **WHEN** torso z 跌破下界
- **THEN** `step()` 回傳的 `terminated` MUST 為 True

### Requirement: RewardConfig 可序列化
`RewardConfig` SHALL provide round-trip serialization via `to_dict` / `from_dict`.

#### Scenario: 序列化往返保留所有欄位
- **GIVEN** 任意合法 `RewardConfig`
- **WHEN** 執行 `RewardConfig.from_dict(cfg.to_dict())`
- **THEN** 還原後的 config 與原物件 `==` 相等（含 `healthy_z_range` tuple）

#### Scenario: 不合法權重被拒絕
- **GIVEN** `RewardConfig(forward_reward_weight=-1.0)`（負值不合理）
- **WHEN** 建立物件
- **THEN** 系統 MUST 拋出 `ValueError`

### Requirement: 分量寫入 info dict
Each `step()` SHALL include reward components and total in `info` dict.

#### Scenario: info 含完整 reward keys
- **GIVEN** 啟用 `CustomAntReward` 的環境
- **WHEN** 執行任一 step
- **THEN** `info` MUST 包含 `reward/forward`、`reward/healthy`、`reward/ctrl_cost`、`reward/contact_cost`、`reward/total`
- **AND** `step()` 回傳的 reward MUST 等於 `info["reward/total"]`

### Requirement: 軀幹 body 不存在時 fail-fast
When the underlying MuJoCo model lacks a body named `torso`, the wrapper SHALL fail at construction time, not silently produce wrong rewards.

#### Scenario: 缺 torso body 時拋例外
- **GIVEN** 一個未含 `torso` body 的 base env（極端情況）
- **WHEN** `CustomAntReward(env, config)` 初始化
- **THEN** 系統 MUST 拋出帶清楚訊息的 `RuntimeError`，提示 model 缺 torso

### Requirement: 介面可選啟用
When the project chooses to use Gymnasium's default Ant reward, the system SHALL allow constructing `make_ant_env` without `reward_config`, preserving prior behavior.

#### Scenario: 不傳 reward_config 時行為與 env-setup 一致
- **GIVEN** 兩個環境：一個 `make_ant_env()`、一個 `make_ant_env(reward_config=None)`
- **WHEN** 跑相同 seed 與動作序列
- **THEN** observation 軌跡 MUST 逐 step 相等
