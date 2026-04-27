## 1. RandomizationConfig

- [ ] 1.1 [S] `src/env/randomization.py:RandomizationConfig` dataclass + `__post_init__` 驗證
- [ ] 1.2 [S] preset 解析：`conservative` / `aggressive` 對應 D3 範圍
- [ ] 1.3 [S] `to_dict` / `from_dict` 序列化

## 2. RandomizationWrapper

- [ ] 2.1 [M] `RandomizationWrapper.__init__(env, config)`：cache obs std（短跑隨機策略 1k step 統計）
- [ ] 2.2 [M] `reset()`：取樣 → 寫入 `mjModel` 欄位 → `mj_forward` → 機器人 z 重置 → 寫 info
- [ ] 2.3 [M] 物理參數寫入：`body_mass`、`geom_friction[:, 0]`、`actuator_gainprm`
- [ ] 2.4 [M] 地形維度整合：呼叫 `add-terrain-generation` 的 `generate_heightfield` + `apply_heightfield`
- [ ] 2.5 [S] `step()`：注入 obs noise（在傳給 policy 之前）

## 3. 與 make_ant_env 整合

- [ ] 3.1 [S] `make_ant_env(..., randomization: RandomizationConfig | None = None)`，None → 跳過 wrapper
- [ ] 3.2 [S] wrapper 順序：base → terrain（若指定固定 terrain）→ randomization → reward

## 4. 訓練 config 整合

- [ ] 4.1 [S] PPO / SAC config schema 新增 `randomization:` 區塊（preset + override 範圍）
- [ ] 4.2 [S] training run 啟動時 log 隨機化配置摘要至 `train.log`

## 5. 測試

- [ ] 5.1 [S] `tests/test_randomization.py`：取樣值落在宣告範圍（1000 次 reset 統計）
- [ ] 5.2 [S] `tests/test_randomization.py`：參數在 episode 內不變
- [ ] 5.3 [S] `tests/test_randomization.py`：相同 seed → 相同取樣序列
- [ ] 5.4 [S] `tests/test_randomization.py`：preset 對應正確範圍
- [ ] 5.5 [S] `tests/test_randomization.py`：obs noise 不影響 reward 與 healthy 判定
- [ ] 5.6 [S] `tests/test_randomization.py`：randomization=None → 行為與 env-setup 一致
- [ ] 5.7 [M] `tests/test_randomization.py`：四種地形隨機切換下 50 step random action 不崩潰

## 6. 文件

- [ ] 6.1 [S] `docs/domain_randomization.md`：兩組 preset 說明、調校建議、與評估流程的關係
