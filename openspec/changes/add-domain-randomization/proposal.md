## Why

簡報第五頁將領域隨機化（Domain Randomization, DR）列為虛實遷移的關鍵工具，並引述 Hwangbo 2019 指出 DR 是讓策略具動態適應力的核心方法。逐字稿補充：「透過在模擬中引入多樣化的環境參數變化，能顯著降低機器人在現實部署時遇到的落差」。

雖然本研究範圍限定模擬，DR 的價值在於避免訓練出「背下單一地形步態」的 policy，並對應簡報第四頁「真的學會適應環境」之研究目標。本 change 提供 DR 機制供 PPO / SAC pipeline 啟用。

## What Changes

- 新增 `src/env/randomization.py`：定義可隨機化的物理參數（質量、摩擦、地面剛度、致動器 gain、observation noise、地形類型與參數）。
- 新增 `RandomizationConfig`：每個參數包含分佈 spec（uniform / log-uniform / discrete choice）。
- 提供 `RandomizationWrapper`：在每次 `reset()` 依 config 取樣新參數並套用至模型。
- 與 `add-terrain-generation` 整合：地形類型可作為其中一個隨機維度。
- 文件化推薦的隨機化「保守 / 激進」兩組預設。

## Capabilities

### New Capabilities
- `domain-randomization`: 對 Ant 物理參數、地形、observation noise 等進行 per-episode 隨機取樣的機制與設定介面。

### Modified Capabilities
（無）

## Impact

- 程式碼：`src/env/randomization.py`、修改 `src/env/factory.py` 加入可選 wrapper
- 訓練 config 新增 `randomization:` 區塊
- 相依：`add-env-setup`、`add-terrain-generation`
- 被相依：`add-ppo-training` 與 `add-sac-training` 訓練配置中可啟用；`add-evaluation` held-out terrain 可視為極端 DR 設定

## Non-Goals

- 不做自動 curriculum（從窄分佈擴大至寬分佈）— 採固定範圍。
- 不修改 Ant 幾何（DoF 數、骨架）— 僅修改可調物理參數。
- 不負責 sim-to-real 部署本身。

## Assumptions & Open Questions

**Assumptions**
- 隨機化粒度為 per-episode（每次 reset 取樣一次），不在 step 內動態變化。
- 隨機化來源為環境包裝器；不修改 base XML，避免 cache 失效。

**Open Questions**
- 各參數的分佈範圍兩份文件未提；採文獻常見 ±20% 為「保守」、±50% 為「激進」兩組預設（design.md D3）。
- 是否啟用 observation noise（高斯）？預設啟用，σ 為 obs std 的 1%。
- 是否做 actuator delay？預設不做（增加複雜度，本研究範圍未提）。
