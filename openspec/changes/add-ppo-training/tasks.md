## 1. 配置與骨架

- [ ] 1.1 [S] `configs/ppo_default.yaml`：填入 D4 預設值
- [ ] 1.2 [S] `src/training/config.py`：YAML schema 與驗證（必填欄位、型別、合理範圍）
- [ ] 1.3 [S] `src/training/__init__.py` + `src/training/common.py`（共用 utility：run_dir 建立、log redirect、git hash）

## 2. 訓練 pipeline

- [ ] 2.1 [M] `src/training/ppo.py:train_ppo(config_path, resume_from)`
- [ ] 2.2 [M] 環境建構：env factory + reward + terrain → `SubprocVecEnv` / `DummyVecEnv`
- [ ] 2.3 [M] `VecNormalize` 套用（norm_obs=True、norm_reward=True、clip_obs=10）
- [ ] 2.4 [S] 模型實例化：`PPO("MlpPolicy", env, **hyperparams, tensorboard_log=...)`
- [ ] 2.5 [S] Seed 設定：torch、numpy、env、CUDA deterministic（在 `n_envs=1` 模式時）

## 3. Callback 與 checkpoint

- [ ] 3.1 [M] `CheckpointCallback`（每 100k steps）：同時存 model + VecNormalize
- [ ] 3.2 [M] `EvalCallback`：每 50k steps 跑 5 個 evaluation episode，更新 `best.zip`
- [ ] 3.3 [S] NaN 偵測 callback：reward / loss NaN → 立即停止
- [ ] 3.4 [S] Plateau 警告 callback：歷史最佳停滯 20% 預算 → log warning

## 4. CLI 與續訓

- [ ] 4.1 [S] CLI 介面：`argparse` / `tyro`，支援 `--config`、`--resume-from`、`--total-timesteps` 覆寫
- [ ] 4.2 [M] 續訓邏輯：偵測 `runs/ppo/<ts>` 下最新 checkpoint，計算剩餘 timesteps
- [ ] 4.3 [S] VecNormalize metadata 驗證（與 checkpoint step 配對）

## 5. 訓練

- [ ] 5.1 [L] [GPU 8 hr] **Baseline plain 地形 5e6 steps 訓練**（無 terrain randomization、無 domain randomization）
- [ ] 5.2 [S] 檢查訓練曲線：`ep_rew_mean` 應穩定上升並收斂

## 6. 測試

- [ ] 6.1 [S] `tests/test_ppo_config.py`：缺欄位 / 型別錯誤的 YAML 拒絕
- [ ] 6.2 [M] `tests/test_ppo_smoke.py`：`total_timesteps=2048` smoke 訓練（< 2 分鐘）
- [ ] 6.3 [S] `tests/test_ppo_smoke.py`：smoke 訓練後 `final.zip` 可載入並 inference
- [ ] 6.4 [S] `tests/test_ppo_resume.py`：訓練 1k → 中止 → 續訓至 2k 不報錯
- [ ] 6.5 [S] `tests/test_ppo_seed.py`：`n_envs=1` 確定性下兩次訓練前 100 step reward 一致

## 7. 文件

- [ ] 7.1 [S] `docs/ppo_training.md`：超參數說明、CLI 用法、續訓步驟、常見問題
- [ ] 7.2 [S] README 鏈結至 `docs/ppo_training.md`
