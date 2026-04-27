## Why

簡報第四頁明確提到「在隨機生成的複雜地形中測試機器人的強健性」，並強調避免「背下步態」、要真的學會「適應環境」。沒有可重現、可參數化的複雜地形生成模組，就無法區分「過擬合特定 terrain」與「具備泛化能力」。

## What Changes

- 新增 `src/env/terrain.py`：以 MuJoCo `hfield`（heightfield）動態產生不平整地面。
- 提供 `TerrainConfig` 資料類，集中管理高度範圍、坡度範圍、粗糙度、grid 解析度、地形種類（plain / slope / rough / stairs）。
- 擴充 `make_ant_env`：接受 `terrain: TerrainConfig | None` 參數；`None` 時保留平面地面（預設 Ant-v4 行為）。
- 提供地形產生器 `generate_heightfield(config, rng)` 回傳 numpy heightmap，可被 MuJoCo 載入。
- 提供視覺化 helper（將 heightmap 渲染為灰階圖供 debug）。

## Capabilities

### New Capabilities
- `terrain-generation`: MuJoCo heightfield 複雜地形的可重現隨機生成、參數化、視覺化機制。

### Modified Capabilities
（無 — `env-setup` 的 `make_ant_env` 介面為新增 optional parameter，向後相容，不算 spec-level modification）

## Impact

- 程式碼：`src/env/terrain.py`、修改 `src/env/factory.py`
- 文件：README 新增地形類型示例
- 下游：`add-domain-randomization` 將透過本 capability 隨機切換地形；`add-evaluation` 將透過本 capability 建立 held-out terrain set
- 上游：相依 `add-env-setup`

## Non-Goals

- 不訓練模型、不調整 reward
- 不包含真實感渲染（光照、貼圖等）
- 不模擬軟性地面（泥地、雪地 — 僅幾何高度，摩擦由 domain randomization 處理）
- 不包含動態障礙物（移動物體）

## Assumptions & Open Questions

**Assumptions**
- 採 MuJoCo 內建 `hfield`（不自訂 mesh），grid 解析度預設 64×64，地形面積預設 20m × 20m。
- 地形種類涵蓋簡報「不規則障礙、坡度、樓梯」三類痛點：`rough`、`slope`、`stairs`，再加 `plain`（baseline）。
- 地形隨 episode reset 重新產生（per-episode randomization）。

**Open Questions**
- 地形高度範圍（max bump / max slope）未在文件中明確；預設值見 design.md，待初步訓練後微調。
- 地形 mesh 重新編譯成本是否可接受？預設使用 `mjModel.hfield_data` 直接寫入記憶體，避免重編譯；待 benchmark 驗證。
- 是否需要 18cm 標準階梯（簡報第三頁提到）？預設 `stairs` 地形含此規格，作為固定基準。
