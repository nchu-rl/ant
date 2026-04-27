## Context

MuJoCo 提供原生 heightfield (`hfield`) 機制：在 XML 模型中宣告 hfield asset，在執行期將 numpy 陣列寫入 `mjModel.hfield_data` 即可動態變更地面幾何。Gymnasium 的 `Ant-v4` 預設地面為平整 plane，需透過修改 XML 或在 wrapper 中替換 ground geom 才能換成 hfield。

簡報第三頁列出三類典型地形痛點：18cm 階梯（樓梯）、≥15° 坡度（slope）、不規則破損地面（rough）。本 change 將這三類做成顯式 `terrain_type`，再加 `plain` 作 baseline 對照。

## Goals / Non-Goals

**Goals:**
- 可重現：相同 seed + config 產生相同地形。
- 可參數化：高度、坡度、粗糙度、長度可獨立調整。
- 與 `make_ant_env` 整合，零改動於 train/eval pipeline。
- 支援 per-episode 重新生成（搭配 domain randomization）。

**Non-Goals:**
- 真實視覺渲染、軟性地面、動態障礙。
- 訓練流程（屬 ppo-/sac-training）。
- 隨機化策略（屬 domain-randomization；本 change 只暴露介面）。

## Decisions

### D1. heightfield 而非自訂 mesh
**選擇**：MuJoCo `hfield`。
**理由**：原生支援、單一 grid 即可表達多種地形、執行期可直接覆寫 `hfield_data`，不需重編譯模型。
**替代方案**：每次生成 mesh 並 `mj_compile` — 駁回，太慢。

### D2. Per-episode 動態地形
**選擇**：在 wrapper 的 `reset()` 中根據 `terrain_config` 與 episode RNG 重寫 `hfield_data`。
**理由**：訓練泛化、評估 held-out 都需要在 episode 邊界切換地形。
**替代方案**：固定地形、訓練多個 model — 駁回，無法支撐 robustness 測試。

### D3. 地形種類
**選擇**：明確列出 `plain | rough | slope | stairs`，不開放任意組合。
**理由**：簡報痛點明確、便於 evaluation 比較、降低 hyperparameter 維度。
**替代方案**：以連續參數混合 — 列為未來擴充，初期不採。

### D4. 地形範圍與解析度（預設值，TBD）
- Grid 解析度：64×64（每格 ≈ 0.31m，足以呈現 ≥ 18cm 階梯）。
- 地形面積：20m × 20m（足夠 episode 內前進）。
- 高度上限：0.3m（覆蓋 18cm 階梯與小型障礙）。
- 坡度上限：30°（涵蓋 15° 痛點 + 安全餘量）。
- 粗糙度（rough 高度標準差）：0.05m。
- **以上值皆為合理初始範圍，最終由初步訓練調整。**

### D5. 與 `make_ant_env` 介面
**選擇**：`make_ant_env(..., terrain: TerrainConfig | None)`。`None` 時行為與 add-env-setup 完全一致。
**理由**：向後相容；下游 change 可漸進啟用。

## Risks / Trade-offs

- [風險] hfield 邊界處理（episode 走出 grid）→ Mitigation：超出 grid 自動延伸最後一行 / 列；或 reset 時將機器人放在 grid 中央。
- [風險] 過於極端的地形導致機器人初始即穿透地面 → Mitigation：寫入 hfield 後將機器人 z 軸重設至「該位置高度 + 安全餘量 0.05m」。
- [Trade-off] 64×64 解析度對 18cm 階梯邊緣不夠平整 → 可選擇加大解析度但會增加記憶體；先用 64×64 baseline。

## Migration Plan

Greenfield，無遷移。下游 capability 在啟用時於 `make_ant_env` 傳入 `TerrainConfig`。

## Open Questions

- `stairs` 階梯走向（X 軸 / Y 軸 / 旋轉）是否要隨機？預設沿 X 軸（前進方向）。
- Rough 高度的 PSD 是否要限制（避免高頻雜訊讓物理求解器抖動）？預設套用 Gaussian filter (σ=1 cell) 後再寫入。
- 地形大小是否會影響 reward 上限（前進獎勵）？需與 reward-design 對齊；預設設大於可達距離以避免 truncation。

## 關鍵介面

```python
# src/env/terrain.py
@dataclass
class TerrainConfig:
    terrain_type: Literal["plain", "rough", "slope", "stairs"]
    grid_size: tuple[int, int] = (64, 64)
    field_size_m: tuple[float, float] = (20.0, 20.0)
    max_height_m: float = 0.3
    slope_deg: float = 0.0       # for "slope"
    roughness_std_m: float = 0.0 # for "rough"
    stair_height_m: float = 0.18 # for "stairs"
    stair_width_m: float = 0.3   # for "stairs"
    seed: int | None = None

def generate_heightfield(config: TerrainConfig, rng: np.random.Generator) -> np.ndarray: ...
def apply_heightfield(env: gym.Env, hfield: np.ndarray) -> None: ...
```

## 訓練超參數
本 change 不涉及 RL 訓練。地形相關預設參數見 D4。
