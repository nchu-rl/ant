## ADDED Requirements

### Requirement: Per-episode 隨機化
On every `reset()`, the system SHALL sample a fresh set of physics and terrain parameters from the configured distributions and apply them to the underlying MuJoCo model.

#### Scenario: 連續兩個 episode 取得不同參數
- **GIVEN** 一個啟用 conservative DR 的環境，episode RNG 推進
- **WHEN** 連續 `reset()` 兩次
- **THEN** 兩次的 `info["dr/body_mass_torso"]`、`info["dr/friction"]`、`info["dr/terrain_type"]` 至少有一項不同

#### Scenario: 參數在 episode 內保持不變
- **GIVEN** 一個 reset 完成的環境
- **WHEN** 連續 100 個 step
- **THEN** 物理參數（body_mass、friction、actuator_gain）逐 step 不變

### Requirement: 配置驅動的分佈
The system SHALL sample each randomized dimension from the distribution declared in `RandomizationConfig`.

#### Scenario: 取樣值落在宣告範圍內
- **GIVEN** `RandomizationConfig(body_mass_range=(0.8, 1.2))`
- **WHEN** 連續 reset 1000 次並收集 `body_mass` multipliers
- **THEN** 所有取樣值 MUST 落在 [0.8, 1.2]
- **AND** 樣本均值與宣告 mean 偏差 < 5%（uniform 分佈 mean = 1.0）

#### Scenario: 不合法 range 拒絕
- **GIVEN** `RandomizationConfig(body_mass_range=(1.2, 0.8))`（顛倒）
- **WHEN** 建立 config
- **THEN** 系統 MUST 拋出 `ValueError`

### Requirement: 預設組（conservative / aggressive）
The system SHALL provide two named presets: `conservative` and `aggressive`, with values matching design.md D3.

#### Scenario: 預設名稱對應正確範圍
- **GIVEN** `RandomizationConfig(preset="conservative")` 與 `preset="aggressive"`
- **WHEN** 比對各欄位範圍
- **THEN** conservative `body_mass_range` MUST 為 (0.8, 1.2)
- **AND** aggressive `body_mass_range` MUST 為 (0.5, 1.5)
- **AND** conservative `slope_deg_range[1]` MUST < aggressive `slope_deg_range[1]`

### Requirement: Seed 可重現
Given the same `RandomizationConfig.seed` and the same episode sequence, sampled parameters MUST be reproducible.

#### Scenario: 相同 seed → 相同取樣序列
- **GIVEN** 兩個以相同 `seed=42` 建立的環境
- **WHEN** 各自 reset 5 次
- **THEN** 兩個環境取樣到的參數序列 MUST 逐 episode 完全相等

### Requirement: Observation noise
When `obs_noise_std_frac > 0`, the system SHALL inject Gaussian noise into the observation passed to the policy, but SHALL NOT affect reward computation or healthy-z determination.

#### Scenario: 觀測有雜訊但 reward 不受影響
- **GIVEN** `RandomizationConfig(obs_noise_std_frac=0.01)`
- **WHEN** 連續 step 並比對 `obs` 與 wrapper 內部記錄的 raw obs
- **THEN** 兩者 MUST 不相同（差值非 0）
- **AND** 同一個 step 在啟用 / 禁用 noise 兩種模式下，reward 與 healthy 判定 MUST 一致

### Requirement: 評估時可關閉
The system SHALL support disabling randomization to obtain deterministic evaluation rollouts.

#### Scenario: randomization=None 時行為與 env-setup 一致
- **GIVEN** `make_ant_env(randomization=None)` 與 `make_ant_env()`
- **WHEN** 相同 seed 與動作序列各跑一個 episode
- **THEN** observation 軌跡 MUST 逐 step 相等

### Requirement: 取樣值寫入 info
Each `reset()` SHALL record sampled parameters in `info` so they can be logged.

#### Scenario: info 含 DR keys
- **GIVEN** 啟用 DR 的環境
- **WHEN** `reset()` 完成
- **THEN** 第一個 `info` dict（或 wrapper 提供的同等介面）MUST 包含至少：
  - `dr/body_mass_multiplier`
  - `dr/friction_multiplier`
  - `dr/actuator_gain_multiplier`
  - `dr/terrain_type`

### Requirement: 機器人不穿透地面
After randomization is applied (including terrain switch), the robot's initial pose SHALL be re-placed so that torso z exceeds the ground height at (x, y) by at least 0.05m.

#### Scenario: 換到 stairs 地形時機器人不卡牆
- **GIVEN** episode 0 為 plain，episode 1 取樣到 stairs（高度 0.20m）
- **WHEN** episode 1 reset 完成
- **THEN** torso z initial value MUST > 0.20 + 0.05 = 0.25m

#### Scenario: 物理參數寫入後仍可前進 forward
- **GIVEN** 任一隨機化下 reset 完的環境
- **WHEN** 跑 50 step random action
- **THEN** 不可拋例外
- **AND** observation 維度仍為 27，且全部 finite
