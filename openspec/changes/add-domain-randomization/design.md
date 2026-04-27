## Context

Hwangbo et al. (2019) 與 Tobin et al. (2017) 將 DR 確立為連接 sim 與 real 的標準工具。在純模擬研究中，DR 仍重要：強迫 policy 對未見參數泛化，避免 reward hacking 在固定參數下找出脆弱解。

簡報第四頁「在隨機生成的複雜地形中測試機器人的強健性，確保它不是『背下步態』」直接呼應此動機。

## Goals / Non-Goals

**Goals:**
- 一份 `RandomizationConfig` 即可宣告所有要隨機化的維度與分佈。
- 與 terrain 整合：地形類型 / 高度 / 坡度可作為 DR 維度。
- 訓練時開啟、評估時可關閉（評估走 deterministic 或固定 held-out）。
- 兩組預設（保守 / 激進），方便快速啟動。

**Non-Goals:**
- 不做 ADR（Automatic DR）。
- 不修改 base XML 或重新編譯模型。
- 不負責 sim-to-real。

## Decisions

### D1. Per-episode 隨機化
**選擇**：在 `reset()` 取樣一組參數並套用，episode 內固定。
**理由**：物理上合理；step-level randomization 會破壞 dynamics 一致性。
**替代方案**：step-level — 駁回。

### D2. 透過 `mjModel` 修改參數而非 XML 重編
**選擇**：直接寫入 `model.body_mass`、`model.geom_friction`、`model.actuator_gainprm` 等欄位。
**理由**：avoid `mj_compile`，per-episode reset 成本可接受。
**替代方案**：修改 XML + recompile — 駁回。

### D3. 兩組預設範圍

**保守 (conservative)：±20%**
| 參數 | 範圍（相對基準） |
|------|------------------|
| `body_mass`（torso & legs） | × U[0.8, 1.2] |
| `geom_friction[:, 0]`（slide） | × U[0.8, 1.2] |
| `actuator_gainprm` | × U[0.8, 1.2] |
| `obs_noise_std`（fraction of obs std）| 0.005 |
| 地形類型 | discrete uniform over {plain, rough, slope, stairs} |
| 地形高度（rough） | U[0.0, 0.05] m |
| 地形坡度（slope） | U[0°, 10°] |
| 階梯高度（stairs） | U[0.10, 0.20] m |

**激進 (aggressive)：±50%**
| 參數 | 範圍 |
|------|------|
| `body_mass` | × U[0.5, 1.5] |
| `geom_friction[:, 0]` | × U[0.5, 1.5] |
| `actuator_gainprm` | × U[0.5, 1.5] |
| `obs_noise_std` | 0.01 |
| 地形類型 | discrete uniform |
| 地形高度（rough） | U[0.0, 0.10] m |
| 地形坡度（slope） | U[0°, 25°] |
| 階梯高度（stairs） | U[0.10, 0.25] m |

**所有值為合理初始範圍，最終由訓練曲線判讀調整。**

### D4. 訓練 / 評估介面
- 訓練：`make_ant_env(..., randomization=RandomizationConfig(preset="conservative"))`。
- 評估：`make_ant_env(..., randomization=None)` → 確定性；或傳入指定 fixed terrain 用於 held-out。

### D5. 不同維度的 RNG 獨立
**選擇**：`RandomizationWrapper` 持有一個 `np.random.Generator`，由 episode seed 派生；每個維度從中取樣，順序固定。
**理由**：給定 seed 可重現整個 episode 的物理參數。

### D6. 觀測雜訊
**選擇**：在 wrapper 的 `step()` 前加 Gaussian noise（fraction × per-dim obs std）。obs std 由訓練前以隨機策略短跑統計得出，cache 在 config 內。
**理由**：簡單、與 SB3 / VecNormalize 兼容（雜訊在 normalize 之前）。

## Risks / Trade-offs

- [風險] 過於激進的 randomization 讓策略無法收斂 → Mitigation：先用 conservative 訓練收斂後再切 aggressive；或 curriculum（本研究不做，列為未來）。
- [風險] 物理參數寫入後 `mj_forward` 不一致 → Mitigation：每次 reset 後呼叫 `mj_forward(model, data)`。
- [Trade-off] 紀錄每 episode 取樣值會增加 log 體積；預設只 log 摘要統計（mean / std），詳細值可選。

## Migration Plan

Greenfield。

## Open Questions

- Observation noise 對 reward 的計算（healthy 判定用真實 z 或 noisy z）？預設 reward 用真實 z，policy input 用 noisy obs。
- 物理參數重設後是否要重新放置機器人 z？是 — 沿用 terrain 的「地面高度 + 0.05m」邏輯。

## 關鍵介面

```python
# src/env/randomization.py
@dataclass
class RandomizationConfig:
    preset: Literal["conservative", "aggressive", "custom"] = "conservative"
    body_mass_range: tuple[float, float] = (0.8, 1.2)
    friction_range: tuple[float, float] = (0.8, 1.2)
    actuator_gain_range: tuple[float, float] = (0.8, 1.2)
    obs_noise_std_frac: float = 0.005
    terrain_types: tuple[str, ...] = ("plain", "rough", "slope", "stairs")
    rough_height_range: tuple[float, float] = (0.0, 0.05)
    slope_deg_range: tuple[float, float] = (0.0, 10.0)
    stair_height_range: tuple[float, float] = (0.10, 0.20)
    seed: int | None = None

class RandomizationWrapper(gym.Wrapper):
    def __init__(self, env, config: RandomizationConfig): ...
```

## 訓練超參數
不直接訓練；DR 範圍視為訓練 config 一部分。預設範圍見 D3。
