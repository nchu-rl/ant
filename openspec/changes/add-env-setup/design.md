## Context

本 change 是研究專題 foundation，後續 7 個 capability 均相依此基底。MuJoCo 自 2.x 起改為公開 binary，並由 DeepMind 維護官方 Python binding；Gymnasium 從 0.29 起整合 `gymnasium[mujoco]`，提供 `Ant-v4` 等標準環境。Stable-Baselines3 ≥ 2.0 對 Gymnasium API 有原生支援。

簡報與逐字稿明確指定使用 MuJoCo Ant-v4（非 Ant-v5），故本 change 鎖定該版本。

## Goals / Non-Goals

**Goals:**
- 建立可重現的 Python + MuJoCo + Gymnasium + SB3 環境。
- 集中環境建立邏輯於工廠函式，避免後續每個 change 各自呼叫 `gym.make`。
- 提供 smoke test 作為 CI 門檻與 onboarding 驗證。
- 文件化所有作業系統 / 驅動相依。

**Non-Goals:**
- 不訓練模型，不修改 reward / observation。
- 不導入 vectorized env（`SubprocVecEnv` 等）— 留給 `add-ppo-training` / `add-sac-training`。
- 不包含 heightfield / terrain（`add-terrain-generation`）。

## Decisions

### D1. Ant-v4（非 Ant-v5）
**選擇**：固定使用 `Ant-v4`。
**理由**：簡報第一頁與第六頁明確使用 v4；v5 的觀測空間與 reward 計算有差異，會影響後續 spec 的 27 維狀態與獎勵公式。
**替代方案**：v5 — 駁回，會破壞既有設計。

### D2. 環境工廠函式
**選擇**：提供 `make_ant_env(seed=None, render_mode=None, frame_skip=5)` 包裝 `gym.make("Ant-v4", ...)`。
**理由**：集中環境設定，後續 capability（terrain、domain randomization）只需擴充工廠，不需散落 `gym.make`。
**替代方案**：每個訓練腳本自行 `gym.make` — 駁回，難以維持一致性與 reproducibility。

### D3. 套件相依管理
**選擇**：`pyproject.toml` + `uv.lock`（或 `pip-tools` 產生 `requirements.lock`）。
**理由**：lock 檔保證重現；`pyproject.toml` 為現代 Python 標準。
**替代方案**：純 `requirements.txt` — 駁回，缺乏 lock。Conda env.yaml — 列為替代。

### D4. GPU/CPU 自動偵測
**選擇**：smoke test 與後續訓練腳本以 `torch.cuda.is_available()` 自動選裝置；不可用時印警告 fallback CPU。
**理由**：研究專題容易在多台機器之間移動；強制 GPU 會降低開發者體驗。

### D5. 隨機性控制
**選擇**：`make_ant_env(seed=...)` 同時設定 `env.reset(seed=)`、Python `random`、NumPy、PyTorch 的 seed；未來訓練腳本沿用同一函式。
**理由**：reproducibility 是研究專題核心要求。

## Risks / Trade-offs

- [風險] MuJoCo binary 對特定 GPU 驅動有相容性問題 → 文件中列出已驗證的驅動版本；CI 跑在受控映像上。
- [風險] Stable-Baselines3 與 Gymnasium 版本耦合緊，升級可能 breaking → 鎖死 minor 版本，升級時走 PR review。
- [Trade-off] 在工廠中包裝 `gym.make` 增加一層間接，但換來集中設定與後續可擴充，淨值正向。

## Migration Plan

Greenfield，無遷移。

## Open Questions

- GPU 規格與 CUDA 版本未提；先在 README 列「建議」配置，由實際機具決定 lock。
- 是否使用 `uv` / `pip-tools` / `poetry` 任一作 lock？預設 `uv`，待團隊確認後固定。
- Headless render 是否要在 smoke test 內驗證？預設僅驗證 `render_mode=None`，rgb_array 留給 evaluation change。

## 關鍵介面

```python
# src/env/factory.py
def make_ant_env(
    seed: int | None = None,
    render_mode: str | None = None,
    frame_skip: int = 5,
) -> gymnasium.Env: ...
```

## 訓練超參數
本 change 不涉及訓練，無超參數。
