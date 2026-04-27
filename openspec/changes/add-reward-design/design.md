## Context

逐字稿第四頁：「我們建構了多目標獎勵函數，在『前進速度』與『能量消耗』之間取得平衡」。
逐字稿第九頁：四項分量明確列出，且 c_contact 特別針對「軀幹」。

Gymnasium Ant-v4 預設 reward 已含 forward / healthy / ctrl / contact，但：
1. 預設 contact 對「整體外部接觸」計算（含腿），與簡報「軀幹」描述不符。
2. 預設權重不一定吻合本研究目標（簡報強調前進速度 vs 能量消耗的平衡）。
3. 預設 reward 不會把分量寫入 `info`，做 multi-objective 分析時不便。

## Goals / Non-Goals

**Goals:**
- 嚴格實作簡報四項定義。
- 權重集中於 `RewardConfig`，可序列化、可由實驗追蹤工具讀寫。
- 把四項分量寫入 `info` dict 以利 logging。
- 與 `make_ant_env` 整合，可選擇沿用 Gymnasium 預設 reward 或本研究自訂 reward。

**Non-Goals:**
- 不擴充新分量（如 yaw stability、smoothness）— 嚴守簡報四項。
- 不做超參數搜尋。
- 不修改 Ant-v4 物理模型。

## Decisions

### D1. 自訂 reward wrapper（不 monkey-patch Gymnasium）
**選擇**：`RewardWrapper` 包住 base env，覆寫 `step()` 計算自訂 reward，丟棄 base env 給的 reward。
**理由**：與 Gymnasium 約定一致、便於測試、便於關閉（pass-through）。
**替代方案**：Fork Ant-v4 改 reward — 駁回，維護成本高。

### D2. 軀幹接觸 vs 全身接觸
**選擇**：`c_contact` 只懲罰名為 `torso` 的 body 與地面 (`floor`/`hfield`) 之接觸力。
**理由**：簡報第九頁明確指「軀幹」與地面接觸；腿部接觸地面是必要的支撐動作，不應懲罰。
**替代方案**：沿用 Gymnasium 全身接觸 — 駁回，與簡報衝突。

### D3. 預設權重（合理初始範圍，TBD）
| 權重 | 預設 | 範圍 | 說明 |
|------|------|------|------|
| `forward_reward_weight` | 1.0 | [0.5, 2.0] | 越大越鼓勵前進 |
| `healthy_reward` | 1.0 | [0.5, 2.0] | 每步存活回饋 |
| `ctrl_cost_weight` | 0.5 | [0.1, 1.0] | 能量懲罰 |
| `contact_cost_weight` | 5e-4 | [1e-4, 1e-2] | 軀幹碰撞懲罰 |
| `healthy_z_range` | (0.2, 1.0) | — | torso z 合理範圍 (m) |

最終值由初步 PPO 訓練的曲線判讀後調整。

### D4. 分量曝露於 info dict
每 step 在 `info` 中放：
```python
{
  "reward/forward": float,
  "reward/healthy": float,
  "reward/ctrl_cost": float,    # 已取負為實際扣分
  "reward/contact_cost": float, # 已取負為實際扣分
  "reward/total": float,
}
```
方便 TensorBoard / W&B 自動 log。

### D5. RewardConfig 序列化
`asdict()` 即可轉 dict，由實驗追蹤工具直接寫入 YAML / W&B config。提供 `from_dict()` 反序列化以重現實驗。

## Risks / Trade-offs

- [風險] 權重不平衡 → reward hacking（例如機器人原地高頻抖動以最大化 healthy）→ Mitigation：`c_ctrl` 提供能量懲罰；訓練時觀察分量曲線。
- [風險] 軀幹定義在 XML 名稱變動時 wrapper 失效 → Mitigation：在初始化時驗證 `torso` body 存在，否則拋例外。
- [Trade-off] 把預設 Gymnasium reward 完全丟棄會失去「健康度檢查」的細節（如 `terminated_when_unhealthy`）→ Mitigation：在 wrapper 中重新實作 unhealthy termination 邏輯。

## Migration Plan

Greenfield。下游 training change 啟用本 capability 時於 `make_ant_env` 傳入 `reward_config`。

## Open Questions

- 是否要 normalize reward（例如除以 episode length 或加 `RunningMeanStd`）？預設不 normalize，由 SB3 的 `VecNormalize` 處理。
- 翻覆 termination 是否視為 healthy=0 之外的額外大懲罰？預設沿用 Gymnasium 慣例：unhealthy 即 episode terminate（無額外懲罰），由 `r_forward` 累積差異呈現好壞。

## 關鍵介面

```python
# src/env/reward.py
@dataclass
class RewardConfig:
    forward_reward_weight: float = 1.0
    healthy_reward: float = 1.0
    ctrl_cost_weight: float = 0.5
    contact_cost_weight: float = 5e-4
    healthy_z_range: tuple[float, float] = (0.2, 1.0)
    terminate_when_unhealthy: bool = True

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig": ...

class CustomAntReward(gym.Wrapper):
    def __init__(self, env: gym.Env, config: RewardConfig): ...
    # step() 回傳 (obs, total_reward, terminated, truncated, info_with_components)
```

## 訓練超參數
本 change 不直接做訓練；reward 權重視為訓練 config 的一部分，預設與範圍見 D3。
