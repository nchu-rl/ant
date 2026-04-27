## 1. RewardConfig 與分量函式

- [ ] 1.1 [S] `src/env/reward.py:RewardConfig` dataclass + `__post_init__` 驗證（負權重、healthy_z_range 順序）
- [ ] 1.2 [S] `to_dict` / `from_dict` 序列化
- [ ] 1.3 [S] `compute_forward(prev_x, curr_x, dt, weight)` 純函式
- [ ] 1.4 [S] `compute_healthy(z, healthy_z_range, healthy_reward)` 純函式
- [ ] 1.5 [S] `compute_ctrl_cost(action, weight)` 純函式
- [ ] 1.6 [M] `compute_contact_cost(model, data, torso_id, weight)`：只取 torso 與 floor/hfield 的 contact

## 2. CustomAntReward wrapper

- [ ] 2.1 [M] `CustomAntReward(env, config)` 初始化：找出 torso body id，若不存在則 RuntimeError
- [ ] 2.2 [M] `step()`：呼叫 base env、覆蓋 reward、組裝 info dict
- [ ] 2.3 [S] 翻覆 termination 邏輯（`terminate_when_unhealthy`）
- [ ] 2.4 [S] `reset()` 同步 prev_x（用於 r_forward 計算）

## 3. 與 make_ant_env 整合

- [ ] 3.1 [S] `make_ant_env(..., reward_config: RewardConfig | None = None)`，None → 使用 Gymnasium 預設 reward
- [ ] 3.2 [S] 整合順序：base Ant-v4 → terrain wrapper → reward wrapper（reward 在最外層）

## 4. 測試

- [ ] 4.1 [S] `tests/test_reward.py`：分量值在已知姿態下符合公式
- [ ] 4.2 [S] `tests/test_reward.py`：腿部接觸不計入 c_contact
- [ ] 4.3 [S] `tests/test_reward.py`：軀幹接觸地面 → c_contact > 0
- [ ] 4.4 [S] `tests/test_reward.py`：torso z 超出 healthy_z_range 時 r_healthy = 0
- [ ] 4.5 [S] `tests/test_reward.py`：序列化往返
- [ ] 4.6 [S] `tests/test_reward.py`：負權重拋 ValueError
- [ ] 4.7 [S] `tests/test_reward.py`：缺 torso body 時拋 RuntimeError
- [ ] 4.8 [S] `tests/test_reward.py`：`reward_config=None` 行為與 env-setup 一致

## 5. 文件

- [ ] 5.1 [S] `docs/reward.md`：四項數學定義、權重物理意義、預設值與調校建議
- [ ] 5.2 [S] README 鏈結至 `docs/reward.md`
