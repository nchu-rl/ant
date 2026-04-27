## ADDED Requirements

### Requirement: 地形種類列舉
The system SHALL support exactly four terrain types: `plain`, `rough`, `slope`, `stairs`.

#### Scenario: 未支援的地形種類報錯
- **GIVEN** 任意非預期的 `terrain_type` 字串（例如 `"sand"`、`"water"`）
- **WHEN** 建立 `TerrainConfig`
- **THEN** 系統 MUST 在 config 構造階段拋出 `ValueError`，不可延遲到 runtime

#### Scenario: 四種地形皆可建立可執行的環境
- **GIVEN** 各以一種 `terrain_type` 建立 `TerrainConfig`
- **WHEN** 透過 `make_ant_env(terrain=config)` 建立環境並跑一個 episode
- **THEN** 不可拋例外，且初始 observation 維度仍為 27

### Requirement: 地形可重現
The system SHALL produce identical heightfields when given the same `TerrainConfig.seed`.

#### Scenario: 相同 seed 產生相同 heightmap
- **GIVEN** 兩個 `TerrainConfig` 除其他參數相同外，`seed` 皆為 42
- **WHEN** 分別呼叫 `generate_heightfield`
- **THEN** 兩個 heightmap 必須逐元素相等

#### Scenario: 不同 seed 產生不同 heightmap（rough / stairs / slope）
- **GIVEN** 兩個 `TerrainConfig` 僅 `seed` 不同（42 vs 43），`terrain_type` 為 `rough`
- **WHEN** 分別呼叫 `generate_heightfield`
- **THEN** 兩個 heightmap MUST 至少有一格不同

### Requirement: Per-episode 地形更新
The system SHALL regenerate the heightfield on each `env.reset()` when terrain randomization is enabled, without recompiling the MuJoCo model.

#### Scenario: 連續兩個 episode 的地形不同
- **GIVEN** 一個啟用 per-episode terrain 的環境，episode-level RNG 不同 seed
- **WHEN** 連續呼叫 `reset()` 兩次
- **THEN** 環境內部 hfield_data 必須有變化
- **AND** 不可呼叫 `mj_compile` 或重新建立 `MjModel`

#### Scenario: 機器人初始位置不會穿透地面
- **GIVEN** 任意一種非 `plain` 地形已被寫入
- **WHEN** `reset()` 完成
- **THEN** Ant 軀幹的 z 軸初始值必須 ≥ 該 (x, y) 點地面高度 + 0.05m

### Requirement: 18cm 階梯規格
When `terrain_type == "stairs"` with default config, the system SHALL produce stairs with step height 0.18m ± 0.005m to match the 城市標準階梯 痛點 in the proposal.

#### Scenario: 預設階梯高度符合 18cm
- **GIVEN** `TerrainConfig(terrain_type="stairs")`（其他參數預設）
- **WHEN** 產生 heightmap 並量測相鄰階梯之高度差
- **THEN** 階梯高度差 MUST 落在 [0.175, 0.185] 公尺

### Requirement: 邊界條件處理
The system SHALL handle terrain boundary conditions safely when the agent reaches the edge of the heightfield grid.

#### Scenario: 機器人接近 grid 邊界不崩潰
- **GIVEN** 機器人位於 hfield grid 邊緣 0.5m 內
- **WHEN** 執行 `step()`
- **THEN** MuJoCo 不可拋出 `mjFATAL` 或 segfault
- **AND** observation 必須仍為合法 27 維 finite vector

#### Scenario: 過高地形請求被拒絕
- **GIVEN** `TerrainConfig(max_height_m=10.0)`（明顯超過合理範圍）
- **WHEN** 建立 config
- **THEN** 系統 MUST 拋出 `ValueError`，訊息中說明合理上限（例如 1.0m）

### Requirement: 地形視覺化
The system SHALL provide a helper to render a generated heightfield as a grayscale image for debugging.

#### Scenario: 視覺化 helper 輸出可開啟的圖檔
- **GIVEN** 任一已產生的 heightmap
- **WHEN** 呼叫 `save_heightmap_image(hfield, path)`
- **THEN** 在 `path` 產生一個 PNG 檔，且像素值 ∈ [0, 255]
- **AND** 高處對應較亮像素

### Requirement: 介面向後相容
When `terrain` argument is omitted or `None`, `make_ant_env` SHALL behave identically to the implementation defined by `env-setup`（純平面 Ant-v4）.

#### Scenario: 不指定 terrain 時行為與 env-setup 一致
- **GIVEN** 兩個環境：一個由 `make_ant_env()`、一個由 `make_ant_env(terrain=None)`
- **WHEN** 以相同 seed 與隨機動作序列各跑一個 episode
- **THEN** observation 軌跡必須逐 step 相等
