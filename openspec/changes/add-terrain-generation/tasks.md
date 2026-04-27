## 1. 模型 XML 與 hfield asset

- [ ] 1.1 [M] 修改（或衍生）Ant-v4 XML：將 ground 替換為 `hfield` geom，宣告 hfield asset 與 size
- [ ] 1.2 [S] 確認 hfield 預設 grid 64×64、面積 20×20m 在 MuJoCo 中可載入
- [ ] 1.3 [S] 將修改後的 XML 放在 `src/env/assets/ant_terrain.xml`

## 2. TerrainConfig 與 generator

- [ ] 2.1 [S] `src/env/terrain.py:TerrainConfig` dataclass + 參數驗證（拒絕未知 type、超出範圍 max_height）
- [ ] 2.2 [M] 實作 `generate_heightfield`：四種 terrain type
- [ ] 2.3 [S] `plain` 實作（全 0）
- [ ] 2.4 [M] `slope` 實作（線性傾斜，`slope_deg` 控制）
- [ ] 2.5 [M] `rough` 實作（Gaussian noise + σ=1 cell 平滑）
- [ ] 2.6 [M] `stairs` 實作（沿 X 軸階梯，預設高度 0.18m）

## 3. 與 make_ant_env 整合

- [ ] 3.1 [M] 擴充 `make_ant_env`：新增 `terrain` 參數，`None` 時使用原本平面 XML
- [ ] 3.2 [M] 實作 `apply_heightfield(env, hfield)`：寫入 `mjModel.hfield_data` 並呼叫 `mj_forward`
- [ ] 3.3 [M] 在 wrapper 的 `reset()` 接 episode RNG → 呼叫 `generate_heightfield` → `apply_heightfield`
- [ ] 3.4 [S] reset 後將機器人 z 軸放置至地面高度 + 0.05m 安全餘量

## 4. 視覺化與除錯

- [ ] 4.1 [S] `save_heightmap_image(hfield, path)` helper（`Pillow` 寫 PNG）
- [ ] 4.2 [S] CLI / notebook 範例：產生四種地形並輸出 PNG

## 5. 測試

- [ ] 5.1 [S] `tests/test_terrain.py`：seed reproducibility（相同 seed → 相同 hfield）
- [ ] 5.2 [S] `tests/test_terrain.py`：不同 seed → 不同 hfield（針對 rough/stairs/slope）
- [ ] 5.3 [S] `tests/test_terrain.py`：18cm 階梯驗證
- [ ] 5.4 [S] `tests/test_terrain.py`：未知 terrain type 拋 ValueError
- [ ] 5.5 [M] `tests/test_terrain.py`：四種地形各跑 100 step random action 不崩潰
- [ ] 5.6 [S] `tests/test_terrain.py`：邊界 step 不 segfault、observation finite
- [ ] 5.7 [S] `tests/test_terrain.py`：`terrain=None` 行為等同 env-setup

## 6. 文件

- [ ] 6.1 [S] README 新增「Terrain types」區塊，附四種地形示例 PNG
- [ ] 6.2 [S] 文件化 TerrainConfig 各欄位的合理範圍與預設
