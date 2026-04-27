## 1. 套件相依與專案骨架

- [ ] 1.1 [S] 建立 `pyproject.toml`，宣告 Python 3.10+ 與核心相依（mujoco, gymnasium[mujoco], stable-baselines3, torch, numpy）
- [ ] 1.2 [S] 選定 lock 工具（`uv` / `pip-tools` 擇一）並產出 lock 檔
- [ ] 1.3 [S] 建立 `src/env/__init__.py`、`tests/` 目錄骨架
- [ ] 1.4 [S] 加入 `.gitignore`（venv、`__pycache__`、TensorBoard logs、`.cache/`）

## 2. 環境工廠函式

- [ ] 2.1 [M] 實作 `src/env/factory.py:make_ant_env(seed, render_mode, frame_skip)`
- [ ] 2.2 [S] 實作 `src/utils/seeding.py:set_global_seed(seed)`（同步 Python random / NumPy / PyTorch / env.reset seed）
- [ ] 2.3 [S] 不支援的 `render_mode` 參數 fail-fast 驗證

## 3. Smoke test

- [ ] 3.1 [S] `tests/test_env_factory.py`：驗證 obs 維度 = 27、action 維度 = 8 且範圍 [-1, 1]
- [ ] 3.2 [S] `tests/test_env_factory.py`：相同 seed → 相同初始 observation
- [ ] 3.3 [M] `tests/smoke_test.py`：跑一個完整 episode 隨機動作、印出 metrics、< 60 秒
- [ ] 3.4 [S] `tests/smoke_test.py`：CUDA 不可用時 fallback CPU 與警告訊息

## 4. 文件

- [ ] 4.1 [S] `README.md` 新增「Environment setup」區塊：OS、Python、MuJoCo binary、CUDA 建議
- [ ] 4.2 [S] 列出已驗證的 GPU 驅動 / CUDA 版本對照表（先填一欄，後續補）
- [ ] 4.3 [S] 文件化常見安裝錯誤與排查步驟

## 5. 驗證

- [ ] 5.1 [S] 在乾淨 venv 中跑完整安裝 → smoke test 全綠
- [ ] 5.2 [S] 在無 GPU 的機器上跑 smoke test，確認 fallback 行為
