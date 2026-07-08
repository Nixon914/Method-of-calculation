# SAAFGWO: Simulated Annealing Artificial Fish Grey Wolf Optimizer

結合模擬退火（Simulated Annealing）與萊維飛行（Lévy Flight）的混合灰狼–人工魚群優化演算法，用於工程問題求解。

> Engineering Problem Solving Using a Hybrid Grey Wolf–Artificial Fish Swarm Optimization Algorithm Enhanced with Simulated Annealing and Lévy Flight
> 2025 Information Systems and Artificial Intelligence Conference, National Pingtung University

**Authors**: Han-Sheng Hsu, Yun-Sheng Chen, Po-Yuan Yang
Department of Intelligent Robotics, National Pingtung University, Pingtung, Taiwan

---

## 📖 簡介 (Abstract)

灰狼優化演算法（Grey Wolf Optimizer, GWO）雖然收斂速度快、結構簡單，但常有族群多樣性不足、探索與開發（exploration-exploitation）失衡、容易陷入局部最優等問題。既有的混合演算法 AFGWO（Artificial Fish Grey Wolf Optimizer）雖有所改善，但仍不足以維持族群多樣性與全域搜尋能力。

本研究提出 **SAAFGWO**，在 AFGWO 架構中導入：

- **模擬退火（SA）**：取代原本的菁英保留機制，以溫度控制機率接受較差解，提升跳脫局部最優的能力
- **萊維飛行（Lévy Flight）**：取代原本以人工魚群更新最差個體的方式，透過短程與長程隨機跳躍加強全域搜尋

實驗以 IEEE CEC2017 標準測試函數集驗證，並進一步應用於彈簧設計（tension/compression spring design）工程優化問題，證實其實用有效性。

---

## 🧠 演算法架構

SAAFGWO 在標準 GWO 的迭代流程中加入三個關鍵機制：

### 1. Lévy Flight 擾動（取代 AFSA 最差個體更新）
每一代針對族群中適應度最差的 5 隻狼，以下列公式產生隨機跳躍步長進行位置擾動：

```
X(t+1) = X(t) + Lévy(β) · (Xα - X(t))
Step size = u / |v|^(1/β),  u ~ N(0, σu²), v ~ N(0, 1), β ∈ (0, 2]
```

透過短距離與長距離交錯的跳躍步長，提升演算法跳出局部最優的機率。

### 2. 模擬退火（SA）機制
每 10 代對目前最優解（alpha 狼）執行一次局部退火搜尋，以下列機率接受較差解：

```
P = exp(-Δf / T)
T(k+1) = α · T(k),  0 < α < 1
```

在搜尋早期允許接受較差解以維持多樣性，隨溫度下降逐漸偏向開發（exploitation），且不改動原始 GWO 結構。

### 3. 二次插值（Quadratic Interpolation）
利用 alpha 狼與兩隻隨機挑選的狼的位置及適應度，計算拋物線插值點作為候選新解，若優於族群中最差個體則取代之，用以加速收斂並補強搜尋精度。

### 演算法流程

```
初始化狼群 X
迭代 t = 1 ... max_iter:
    對最差的 5 隻狼執行 Lévy Flight 擾動
    依標準 GWO 規則更新 alpha / beta / delta 及所有狼的位置
    以二次插值產生新解，嘗試取代族群最差個體
    每 10 代對 alpha 狼執行一次模擬退火局部搜尋
回傳最優解 alpha_score
```

---

## 📁 專案結構

```
.
├── saafgwo.py          # 主演算法程式（simulated_annealing / levy_flight /
│                        #   quadratic_interpolation / gwo_afsa_with_worst_update）
└──README.md
```

---

## ⚙️ 環境需求

```
python >= 3.8
numpy
matplotlib
cec2017
```

安裝：

```bash
pip install numpy matplotlib cec2017
```

> `cec2017` 套件提供 IEEE CEC2017 標準測試函數集（`functions.all_functions`），用於演算法效能評估。

---

## 🚀 使用方式

直接執行主程式，會依序對 CEC2017 各測試函數執行 30 次獨立實驗，並輸出每次結果與統計數據（Best / Mean / Std）：

```bash
python saafgwo.py
```

### 主要參數（於 `test_gwo_afsa_on_cec2017()` 中設定）

| 參數 | 說明 | 預設值 |
|---|---|---|
| `dim` | 問題維度 | 10 |
| `n_wolves` | 狼群規模 | 50 |
| `max_iter` | 最大迭代次數 | 1000 |
| `runs` | 獨立實驗次數 | 30 |
| `lb`, `ub` | 搜尋空間上下界 | -100, 100 |

模擬退火相關參數（於 `gwo_afsa_with_worst_update()` 中設定）：

| 參數 | 說明 | 預設值 |
|---|---|---|
| `T0` | 初始溫度 | 100 |
| `T_min` | 最低溫度（退火終止條件） | 1e-3 |
| `cooling_rate` | 降溫係數 α | 0.9 |

### 自訂使用範例

```python
from saafgwo import gwo_afsa_with_worst_update
import numpy as np

def sphere(x):
    return np.sum(x**2, axis=1)

result = gwo_afsa_with_worst_update(
    f=sphere, dim=10, lb=-100, ub=100,
    n_wolves=50, max_iter=1000
)
best_score = result[0]
print("Best fitness:", best_score)
```

---

## 📊 實驗結果

### CEC2017 測試結果（10 維，節錄）

| Function | SAAFGWO Best | SAAFGWO Mean | AFGWO Best | AFGWO Mean |
|---|---|---|---|---|
| f3  | 300.00  | 300.01  | 300.00 | 300.02 |
| f4  | 400.70  | 405.44  | 400.00 | 845.50 |
| f5  | 504.98  | 532.83  | 503.05 | 566.40 |
| f13 | 1338.57 | 1763.06 | 1570.00 | 2599.00 |
| f26 | 2800.46 | 3850.56 | 2706.40 | 3930.60 |

在多數測試函數上（特別是 hybrid 與 composition 類型），SAAFGWO 相較 AFGWO 展現更佳的全域搜尋能力與收斂穩定性。完整數據請見論文 Table I。

### 工程應用：彈簧設計問題（Tension/Compression Spring Design）

以彈簧重量最小化為目標（含 4 個限制條件），SAAFGWO 求得最佳解：

| 變數 | x1 (線徑) | x2 (平均線圈直徑) | x3 (有效圈數) | 最佳成本 |
|---|---|---|---|---|
| SAAFGWO | 0.05248 | 0.37597 | 10.24781 | **0.012681** |

此結果優於 GWO、WOA、AGWO、CSGWO 等多種對照演算法（詳見論文 Table II）。

---

## ⚠️ 已知限制

- 目前程式碼內含實驗用的計數變數（如 `kqc`, `worstc`, `sac` 等），為研究過程中追蹤各機制觸發次數所用，非演算法必要部分
- SA 局部搜尋固定每 10 代觸發一次，尚未做自適應調整
- 尚未在所有 CEC2017 函數類型與更高維度上做完整驗證，未來將擴展測試範圍

---

## 📄 引用 (Citation)

若使用本專案，請引用：

```bibtex
@inproceedings{hsu2025saafgwo,
  author    = {Han-Sheng Hsu and Yun-Sheng Chen and Po-Yuan Yang},
  title     = {Engineering Problem Solving Using a Hybrid Grey Wolf--Artificial Fish Swarm Optimization Algorithm Enhanced with Simulated Annealing and L{\'e}vy Flight},
  booktitle = {2025 Information Systems and Artificial Intelligence Conference},
  year      = {2025},
  address   = {National Pingtung University, Pingtung, Taiwan}
}
```

## 📚 主要參考文獻

1. S. Mirjalili, S. M. Mirjalili, and A. Lewis, "Grey Wolf Optimizer," *Advances in Engineering Software*, vol. 69, pp. 46–61, 2014.
2. H. Zhang, Y. Zhang, Y. Niu, and Y. Xue, "A grey wolf optimizer combined with artificial fish swarm algorithm for engineering design problems," *Ain Shams Eng. J.*, vol. 15, no. 7, 2024.
3. A. A. Heidari and P. Pahlavani, "An efficient modified grey wolf optimizer with Lévy flight for global optimization tasks," *Applied Soft Computing*, vol. 60, pp. 115–134, 2017.
4. A. Tzanetos and M. Blondin, "A qualitative systematic review of metaheuristics applied to tension/compression spring design problem," *Eng. Appl. Artif. Intell.*, vol. 118, 2023.

完整參考文獻請見論文原文。
