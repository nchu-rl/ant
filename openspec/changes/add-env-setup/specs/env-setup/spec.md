## ADDED Requirements

### Requirement: 套件版本鎖定
The system SHALL pin core dependency versions in a lock file so that any contributor can reproduce the same Python + MuJoCo + Gymnasium + SB3 environment.

#### Scenario: 全新機器照 lock 檔安裝後可 import 所有核心套件
- **GIVEN** 一台未安裝過本專案的開發機，已備妥 Python 3.10+
- **WHEN** 開發者依 README 指示執行安裝指令（lock 檔基礎）
- **THEN** `mujoco`、`gymnasium`、`stable_baselines3`、`torch`、`numpy` 皆能 `import` 而不報錯
- **AND** 各套件版本與 lock 檔記錄一致

#### Scenario: 缺少 MuJoCo binary 時報錯訊息可操作
- **GIVEN** Python 環境已裝套件但 MuJoCo binary 未就緒
- **WHEN** 嘗試建立 Ant-v4 環境
- **THEN** 系統 SHALL 拋出帶有指引的錯誤訊息（指向官方安裝步驟或 README 區塊）

### Requirement: 環境工廠函式
The system SHALL provide a factory `make_ant_env(seed, render_mode, frame_skip)` that returns a Gymnasium-compliant Ant-v4 environment with project default configuration.

#### Scenario: 相同 seed 產生相同初始觀測
- **GIVEN** 兩個以相同 `seed` 建立的 Ant 環境
- **WHEN** 分別呼叫 `reset(seed=seed)`
- **THEN** 兩者回傳的初始 observation 必須逐元素相等（容差 0）

#### Scenario: 觀測與動作空間維度符合 Ant-v4 規格
- **GIVEN** 由工廠函式建立的環境
- **WHEN** 檢查 `observation_space` 與 `action_space`
- **THEN** observation 維度 MUST 為 27
- **AND** action 維度 MUST 為 8 且範圍為 `[-1, 1]`

#### Scenario: 不支援的 render_mode 參數要 fail fast
- **GIVEN** 工廠函式收到未支援的 `render_mode`（例如 `"opengl_v1"`）
- **WHEN** 呼叫 `make_ant_env(render_mode="opengl_v1")`
- **THEN** 系統 MUST 在環境建立階段即拋出 `ValueError`，不可延遲到 `step()` 才失敗

### Requirement: Smoke test
The system SHALL ship a runnable smoke test that exercises one full episode with random actions to validate the install.

#### Scenario: smoke test 在合理時間內完成且不拋例外
- **GIVEN** 已依 lock 檔完成安裝
- **WHEN** 執行 smoke test 指令
- **THEN** 測試 MUST 在 60 秒內完成至少一個 episode
- **AND** 印出 episode return、length、是否 terminate / truncate

#### Scenario: GPU 不存在時自動 fallback CPU
- **GIVEN** 系統無可用 CUDA 裝置
- **WHEN** 執行 smoke test
- **THEN** 系統 SHALL 自動 fallback 至 CPU 並印出警告
- **AND** 不可拋出 CUDA 相關例外

#### Scenario: smoke test 失敗時退出碼非零
- **GIVEN** 任何 import 失敗或 step 拋例外
- **WHEN** smoke test 執行完畢
- **THEN** 進程 MUST 以非零 exit code 結束，使 CI 可偵測

### Requirement: 隨機性可控
The system SHALL provide a single entry point that simultaneously seeds the environment, Python `random`, NumPy, and PyTorch.

#### Scenario: 全域 seed 設定後重複實驗結果一致
- **GIVEN** 在腳本起始呼叫 seed 設定函式（如 `set_global_seed(42)`）
- **WHEN** 連續執行兩次相同的隨機動作序列
- **THEN** 兩次的 observation 軌跡必須逐元素相等

### Requirement: 環境設定文件化
The system SHALL document setup steps including supported OS, Python version, MuJoCo binary install, optional GPU driver and CUDA version.

#### Scenario: 新成員依文件能複現環境
- **GIVEN** 新成員拿到 README 與 lock 檔
- **WHEN** 依步驟執行
- **THEN** 能在不額外詢問的情況下完成環境安裝並通過 smoke test
