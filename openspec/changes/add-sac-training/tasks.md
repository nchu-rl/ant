## 1. 配置與骨架

- [ ] 1.1 [S] `configs/sac_default.yaml`：填入 D4 預設值，env/reward/terrain/timesteps/net_arch 與 PPO 對齊
- [ ] 1.2 [S] `src/training/config.py`：擴充支援 SAC schema（buffer_size、tau、train_freq 等必填）
- [ ] 1.3 [S] 抽取 PPO/SAC 共用邏輯至 `src/training/common.py`（若上一個 change 未抽則本 change 抽）

## 2. SAC 訓練 pipeline

- [ ] 2.1 [M] `src/training/sac.py:train_sac(config_path, resume_from)`
- [ ] 2.2 [M] 環境建構：env factory + reward + terrain → `DummyVecEnv`（n_envs=1）+ `VecNormalize(norm_obs=True, norm_reward=False)`
- [ ] 2.3 [S] 模型實例化：`SAC("MlpPolicy", env, ent_coef="auto", **hyperparams, tensorboard_log=...)`
- [ ] 2.4 [S] 確認 `target_entropy = -action_dim` 自動設定

## 3. Callback 與 checkpoint

- [ ] 3.1 [M] `CheckpointCallback` 每 100k steps 存 model + VecNormalize；視 `--save-replay-buffer` 決定是否存 buffer
- [ ] 3.2 [M] `EvalCallback` 每 50k steps 跑 5 個 evaluation episode，更新 `best.zip`
- [ ] 3.3 [S] NaN callback：critic / actor loss NaN → 中止
- [ ] 3.4 [S] Plateau warning callback（與 PPO 共用）

## 4. CLI 與續訓

- [ ] 4.1 [S] CLI：支援 `--config`、`--resume-from`、`--total-timesteps`、`--save-replay-buffer`
- [ ] 4.2 [M] 續訓邏輯：載入 model、VecNormalize、選擇性載入 buffer，計算剩餘 timesteps

## 5. 訓練

- [ ] 5.1 [L] [GPU 6 hr] **Baseline plain 地形 5e6 steps SAC 訓練**（與 PPO baseline 相同 env）
- [ ] 5.2 [S] 檢查 `train/ent_coef` 隨訓練變化、`ep_rew_mean` 上升

## 6. 測試

- [ ] 6.1 [S] `tests/test_sac_config.py`：缺欄位拒絕
- [ ] 6.2 [M] `tests/test_sac_smoke.py`：`total_timesteps=20000` smoke 訓練（< 5 分鐘）
- [ ] 6.3 [S] `tests/test_sac_smoke.py`：smoke 後 `final.zip` 可載入並 inference
- [ ] 6.4 [S] `tests/test_sac_resume.py`：訓練 5k → 中止 → 續訓至 10k 不報錯（無 buffer）
- [ ] 6.5 [S] `tests/test_sac_resume.py`：續訓帶 buffer 時 buffer.size() > 0
- [ ] 6.6 [S] `tests/test_sac_parity.py`：sac_default.yaml 與 ppo_default.yaml 在 env/reward/terrain/timesteps/net_arch 五欄位等同

## 7. 文件

- [ ] 7.1 [S] `docs/sac_training.md`：超參數、CLI、續訓、buffer 取捨、與 PPO 對照表
