# 基於強化學習之四足機器人複雜地形導航研究

> 在 MuJoCo Ant-v4 模擬環境中，使用 PPO / SAC 訓練四足機器人在不平整地形上學會自適應步態，解決傳統輪式機器人在最後一哩物流與災難搜救任務中無法跨越非結構化地形的限制。

本專案為深度強化學習課程專題（第七組）。研究範圍限定於模擬環境中的演算法驗證；實機部署與虛實遷移列為背景理論與未來潛力，非本研究實作目標。

---

## 目錄

1. [研究背景](#研究背景)
2. [技術概覽](#技術概覽)
3. [專案架構（OpenSpec capabilities）](#專案架構openspec-capabilities)
4. [目錄結構](#目錄結構)
5. [docs/ 文件導覽](#docs-文件導覽)
6. [開發狀態](#開發狀態)
7. [Tech stack](#tech-stack)
8. [如何閱讀 OpenSpec changes](#如何閱讀-openspec-changes)
9. [參考文獻](#參考文獻)

---

## 研究背景

主流輪式機器人受限於平整地面：無法越過高於輪徑 50% 的障礙、無法穩定處理 ≥15° 斜坡，全球約 30% 城市人行道存在破損或高度差。在地震、水災等災區則受困率更高，浪費黃金救援時間。

足式機器人能透過步態規劃跨越不規則障礙、在高坡度地形維持平衡，是克服上述痛點的解方。傳統靜態控制（heuristic control）難以應對隨機地形，本研究改採深度強化學習，讓機器人在 MuJoCo 模擬環境中透過自我試錯學會適應性步態。

詳細動機與痛點數據請見 [`docs/proposal_transcript.txt`](docs/proposal_transcript.txt) 第二、三頁，或觀看 [`docs/proposal.mp4`](docs/proposal.mp4)。

## 技術概覽

| 項目 | 內容 |
|---|---|
| 模擬環境 | MuJoCo + Gymnasium，使用 `Ant-v4` 模型 |
| 機器人 | 8 自由度四足結構，每腿含 hip 與 ankle 兩關節 |
| 觀測空間 | 27 維（軀幹高度 / 角度 / 8 關節角度與速度 / 質心速度） |
| 動作空間 | 8 維連續向量 ∈ [-1, 1]，對應 8 個馬達相對扭力 |
| 主演算法 | PPO（Proximal Policy Optimization） |
| 對照演算法 | SAC（Soft Actor-Critic） |
| 策略網路 | Actor-Critic + MLP（預設 [256, 256]） |
| 獎勵函數 | 四項加權：r_forward + r_healthy − c_ctrl − c_contact |
| 強健性測試 | 在 held-out 隨機生成地形上評估 |

獎勵分量定義詳見 `docs/proposal_transcript.txt` 第九頁。

## 專案架構（OpenSpec capabilities）

本研究以 [OpenSpec](https://github.com/.../openspec) 進行 spec-driven 開發，將工作切成 8 個正交 capability，每個對應一個 change proposal：

```
                ┌──────────────┐
                │  env-setup   │  ← Foundation：MuJoCo + Gymnasium + Ant-v4
                └──────┬───────┘
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌──────────────────┐
│  terrain-  │ │  reward-   │ │ (其他下游)        │
│ generation │ │  design    │ │                  │
└─────┬──────┘ └─────┬──────┘ │                  │
      │              │        │                  │
      └──────┬───────┘        │                  │
             ▼                │                  │
   ┌─────────────────────┐    │                  │
   │ domain-             │    │                  │
   │ randomization       │    │                  │
   └────────┬────────────┘    │                  │
            │                 │                  │
       ┌────┴─────┬───────────┘                  │
       ▼          ▼                              │
  ┌─────────┐ ┌─────────┐                        │
  │  ppo-   │ │  sac-   │                        │
  │training │ │training │                        │
  └────┬────┘ └────┬────┘                        │
       └──────┬────┘                             │
              ▼                                  │
        ┌──────────┐                             │
        │evaluation│                             │
        └──────────┘                             │
                                                 │
              ┌──────────────────────────────────┘
              ▼
   ┌─────────────────────┐
   │ experiment-tracking │  ← Cross-cutting：訓練 / 評估皆使用
   └─────────────────────┘
```

| Capability | 範疇 | OpenSpec change |
|---|---|---|
| `env-setup` | Ant-v4 安裝、版本鎖、環境工廠、smoke test | [`openspec/changes/add-env-setup/`](openspec/changes/add-env-setup/) |
| `terrain-generation` | MuJoCo heightfield 隨機生成（plain / rough / slope / stairs，含 18cm 標準階梯） | [`add-terrain-generation/`](openspec/changes/add-terrain-generation/) |
| `reward-design` | 四項加權獎勵函數、軀幹專用 c_contact、序列化介面 | [`add-reward-design/`](openspec/changes/add-reward-design/) |
| `ppo-training` | SB3 PPO pipeline、checkpoint、續訓、NaN/plateau 偵測 | [`add-ppo-training/`](openspec/changes/add-ppo-training/) |
| `sac-training` | SB3 SAC 對照組、buffer 取捨、與 PPO 結構鏡像 | [`add-sac-training/`](openspec/changes/add-sac-training/) |
| `domain-randomization` | conservative / aggressive 兩組預設、per-episode 物理與地形取樣 | [`add-domain-randomization/`](openspec/changes/add-domain-randomization/) |
| `evaluation` | held-out terrain set、7 項評估指標、PPO/SAC 結果可並列 | [`add-evaluation/`](openspec/changes/add-evaluation/) |
| `experiment-tracking` | TensorBoard 永遠開、W&B 雙條件啟用、失敗降級 | [`add-experiment-tracking/`](openspec/changes/add-experiment-tracking/) |

## 目錄結構

```
ant/
├── README.md                          ← 本檔案
├── docs/                              ← 提案文件與簡報媒體（見下節）
│   ├── proposal.pdf
│   ├── proposal.pptx
│   ├── proposal.mp4
│   └── proposal_transcript.txt
└── openspec/                          ← Spec-driven 開發產出
    ├── config.yaml                    ← 專案 context（domain terms / constraints / tech stack）
    ├── changes/                       ← 8 個 capability 的提案
    │   ├── add-env-setup/
    │   │   ├── proposal.md            ← Why / What / Impact / Non-goals / Open Questions
    │   │   ├── design.md              ← 架構決策、技術選型、超參數範圍
    │   │   ├── tasks.md               ← 實作任務清單（S/M/L 估時）
    │   │   └── specs/env-setup/spec.md ← Requirements 與 GIVEN/WHEN/THEN scenarios
    │   ├── add-terrain-generation/
    │   ├── add-reward-design/
    │   ├── add-ppo-training/
    │   ├── add-sac-training/
    │   ├── add-domain-randomization/
    │   ├── add-evaluation/
    │   ├── add-experiment-tracking/
    │   └── archive/                   ← 完成後封存的 change（目前空）
    └── specs/                         ← Capability 完成歸檔後落腳處（目前空）
```

實作開始後會新增：

```
ant/
├── src/                  ← 主程式（env / training / eval / tracking）
├── tests/                ← pytest 測試
├── configs/              ← 訓練 / 評估 / 追蹤 YAML 配置
├── runs/                 ← 訓練輸出（checkpoint、TensorBoard logs、final.zip）
└── reports/              ← 評估報告（eval.json、eval.md、CSV）
```

## docs/ 文件導覽

`docs/` 目錄收錄本專題的提案材料，四個檔案彼此互補。建議閱讀順序：影片 → 逐字稿 → 簡報 PDF → PowerPoint 原始檔（如需編輯）。

### `proposal.mp4` — 提案簡報影片
- **內容**：第七組對 11 頁簡報內容的口頭報告錄影。
- **適合誰**：第一次接觸本專案、想快速建立整體印象的讀者。
- **何時看**：理解研究動機、技術主軸、預期貢獻最直觀的入口。
- **限制**：影片不便檢索；如需引用具體段落請改看逐字稿。

### `proposal_transcript.txt` — 提案簡報逐字稿
- **內容**：影片的中文逐字稿，明確標註「第一頁」～「第十一頁」對應簡報結構。
- **適合誰**：開發者、撰寫程式或文件時需要查證原始陳述的人。
- **何時看**：本 repo 的 OpenSpec 內容（`config.yaml` 與所有 changes）以此為**主要事實來源**；簡報未明寫、但口頭補充的細節以逐字稿為準。
- **特點**：含具體數據（18cm 標準階梯、15° 坡度、30% 城市地形障礙等）、四項獎勵分量定義、相關文獻引用。

### `proposal.pdf` — 提案簡報投影片
- **內容**：簡報投影片的 PDF 匯出版（共 16 頁，對應逐字稿 11 個結構段落 + 封面 / 致謝 / 參考文獻頁）。
- **適合誰**：偏好視覺化、想看圖表 / 流程圖 / 對照表的讀者。
- **何時看**：與逐字稿並讀效果最佳；逐字稿提供口語細節，PDF 提供結構與視覺資訊（架構圖、四階段流程、規格對照表等）。
- **與逐字稿的關係**：當兩者內容衝突或補充時，**口語細節以逐字稿為準**，**結構化決策（流程、規格）以 PDF 為準**。

### `proposal.pptx` — 提案簡報原始檔（PowerPoint）
- **內容**：`proposal.pdf` 的編輯來源檔。
- **適合誰**：團隊成員需要修改、重製、或衍生新簡報時。
- **何時用**：日常閱讀請優先用 PDF（跨平台一致）；只有要再次匯出或修改才開啟 .pptx。

### 速查表

| 想知道… | 看這個 |
|---|---|
| 整個專題在做什麼（10 分鐘版） | `proposal.mp4` |
| 一句話可引用的研究動機、痛點數據、技術選型理由 | `proposal_transcript.txt` |
| 簡報架構、流程圖、規格表 | `proposal.pdf` |
| 修改 / 衍生簡報 | `proposal.pptx` |
| 工程實作該怎麼做、有哪些 requirements / decisions | `openspec/changes/<capability>/` |

## 開發狀態

- [x] 專題提案（簡報、逐字稿、影片）已交付
- [x] OpenSpec context 寫入（`openspec/config.yaml`）
- [x] 8 個 capability 之 proposal / design / specs / tasks 全數完成（`openspec validate --all` 通過）
- [ ] 實作（依 `openspec/changes/add-env-setup/` 起跑）
- [ ] PPO baseline 訓練 + 評估
- [ ] SAC 對照訓練 + 評估
- [ ] 啟用 domain randomization、再次評估
- [ ] 撰寫成果報告

## Tech stack

以下為 RL on MuJoCo 研究專題的常見組合（已寫入 `openspec/config.yaml`），實際可調整：

- **程式語言**：Python 3.10+
- **RL 框架**：Stable-Baselines3（提供 PPO、SAC 實作）+ Gymnasium
- **物理引擎**：MuJoCo（透過 `mujoco` Python binding 與 `gymnasium[mujoco]`）
- **深度學習後端**：PyTorch
- **數值運算**：NumPy（必要時加 Pandas）
- **實驗追蹤**：TensorBoard（內建）；視需要加 Weights & Biases
- **環境管理**：Python venv 或 conda
- **訓練硬體**：本機 / 實驗室 GPU（CUDA），CPU 亦可但訓練時間顯著增加
- **版本控制**：Git；commit 採 conventional commits

## 如何閱讀 OpenSpec changes

每個 change 由四份檔案構成：

1. **`proposal.md`** — 動機、變更範圍、影響、非目標、待釐清項目。先讀。
2. **`design.md`** — 架構決策（含被駁回的替代方案）、技術選型、預設超參數與合理範圍、風險。
3. **`specs/<capability>/spec.md`** — Requirements（SHALL/MUST/SHOULD 標強制等級），每個 requirement 附 GIVEN/WHEN/THEN scenario。
4. **`tasks.md`** — 拆解後的實作任務清單（S/M/L 估時）；訓練類任務註明預估時數與硬體需求。

常用指令：

```bash
# 列出所有 change 與當前狀態
openspec list

# 查看單一 change 的進度
openspec status --change add-env-setup

# 驗證所有 spec / change 格式
openspec validate --all

# 查看單一 spec
openspec show env-setup
```

## 參考文獻

簡報第十一頁列出的關鍵文獻：

- Hwangbo, J. et al. (2019). *Learning agile and dynamic motor skills for legged robots*. Science Robotics.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
- International Federation of Robotics (IFR). *World Robotics Report* — 服務型機器人於非結構化環境之應用。
