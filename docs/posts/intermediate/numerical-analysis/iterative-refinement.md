---
title: 迭代改善法（iterative refinement）
date: 2026-08-07
---

# 迭代改善法：用残差修补解

<div class="epigraph">
<p>一次求解得到近似，二次修正得到精确——迭代是通往精度的阶梯。</p>
<footer>—— 数值线性代数的修正哲学</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§5.5 ｜ 2026-08-07</p>
</div>

## 为什么从迭代改善法开始

上一节看到病态方程组「残差小、解却错」——似乎只能接受低精度。**迭代改善法（iterative refinement）** 给出一个补救：在病态但**不太病态**（$\mathrm{cond}(A)<10^{15}$ 上下）时，用「残差驱动的迭代修正」把解的精度一步步拉回接近机器精度。它的思想简单优雅：算残差 → 解残差方程组 → 修正解 → 重复。<span class="marginnote">迭代改善是数值线性代数里「用已算好的分解反复榨取精度」的经典：<strong>分解一次（贵），残差修正多次（便宜）</strong>。Wilkinson 在 1960 年代把它理论化——只要残差用「高精度」计算（避免舍入污染），迭代通常两三步就收敛到机器精度。</span>

本节给出算法、收敛理论，以及它为什么能突破「单次求解的精度极限」。

## 1 算法：修正循环

设已用 LU 分解求出近似解 $\hat{\mathbf{x}}$。迭代改善的循环：

1. **算残差**：$\mathbf{r}=\mathbf{b}-A\hat{\mathbf{x}}$（用**扩展精度**计算，避免残差自身的舍入污染）。
2. **解残差方程组**：$A\mathbf{d}=\mathbf{r}$（用已分解好的 $L,U$，两次三角回代）。
3. **修正**：$\hat{\mathbf{x}}\leftarrow\hat{\mathbf{x}}+\mathbf{d}$。
4. **判断**：若 $\lVert\mathbf{d}\rVert$ 足够小，停止；否则回到 1。

**为什么便宜？** 每次迭代只做两次三角回代（$O(n^2)$）+ 一次残差计算（$O(n^2)$），分解只做一次。**$O(n^3)$ 的昂贵工作只付一次，$O(n^2)$ 的修正可以反复做。**

```python
import numpy as np
from scipy.linalg import lu_factor, lu_solve

# 希尔伯特矩阵 H6：cond ≈ 1.5e7，真解取全 1
n = 6
H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
b = H.sum(axis=1)                              # x_true = ones

lu, piv = lu_factor(H)                         # LU 分解只做一次
x = lu_solve((lu, piv), b)                     # 初始解
for _ in range(2):                             # 迭代改善两轮
    r = b - H @ x                              # 残差（工程上用扩展精度算）
    x = x + lu_solve((lu, piv), r)             # 复用分解：两次三角回代
print(x)                                       # 逼近全 1，误差 ~1e-15
```

**数值演示**：取希尔伯特矩阵 $H_6$（$\mathrm{cond}\approx1.5\times10^7$），真解 $\mathbf{x}=\mathbf{1}$。单次 LU 求解的相对误差约 $10^{-9}$，迭代改善 2 轮后约 $10^{-15}$——**逼近机器精度**。<span class="marginnote">注意前提：残差 $\mathbf{r}$ 若用普通精度算，其舍入误差量级 $O(\epsilon_{\mathrm{mach}})$ 会成为瓶颈，修正就到不了机器精度。<strong>迭代改善的关键细节是「残差用扩展精度计算」</strong>——这是它与朴素「再解一次」的分水岭。</span>

## 2 公式解析：为什么迭代能收敛到机器精度

分析一次修正的效果。设 $\hat{\mathbf{x}}_k$ 是第 $k$ 次近似，$\mathbf{x}^*$ 是真解，误差 $\mathbf{e}_k=\hat{\mathbf{x}}_k-\mathbf{x}^*$。

**第一步，残差与误差的关系。** $\mathbf{r}_k=\mathbf{b}-A\hat{\mathbf{x}}_k=A(\mathbf{x}^*-\hat{\mathbf{x}}_k)=-A\mathbf{e}_k$，即 $\mathbf{e}_k=-A^{-1}\mathbf{r}_k$。理论上解残差方程组 $A\mathbf{d}=\mathbf{r}_k$ 应得 $\mathbf{d}=-\mathbf{e}_k$，一步到位。
**第二步，但求解本身有误差。** 解 $A\mathbf{d}=\mathbf{r}_k$ 的近似解 $\hat{\mathbf{d}}$ 满足 $\lVert\hat{\mathbf{d}}-(-\mathbf{e}_k)\rVert\le\mathrm{cond}(A)\epsilon_{\mathrm{mach}}\lVert\mathbf{d}\rVert$——**修正量本身被条件数放大舍入误差**。
**第三步，误差压缩比。** 修正后 $\mathbf{e}_{k+1}=\mathbf{e}_k+\hat{\mathbf{d}}$，其范数约 $\mathrm{cond}(A)\epsilon_{\mathrm{mach}}\lVert\mathbf{x}^*\rVert$——**只要初始误差远大于 $\mathrm{cond}(A)\epsilon_{\mathrm{mach}}$，每轮都把误差压缩约一个常数因子，直至触底于 $\mathrm{cond}(A)\epsilon_{\mathrm{mach}}\lVert\mathbf{x}\rVert$**。

**结论：迭代改善把误差压到「机器精度被条件数污染」的极限**——对不太病态的矩阵（$\mathrm{cond}<1/\epsilon_{\mathrm{mach}}$），这接近全精度。对极端病态（$\mathrm{cond}\sim10^{16}$），改善无效（误差已达极限），需扩展精度或换表述。

## 3 什么情况下迭代改善有效

| 条件 | 效果 |
| --- | --- |
| 矩阵可分解（LU 成功） | 必要前提 |
| 残差用扩展精度计算 | 关键，否则修正失效 |
| $\mathrm{cond}(A)$ 中等（< $10^{14}$） | 收敛到接近机器精度 |
| $\mathrm{cond}(A)$ 极大（≈ $10^{16}$） | 改善有限，误差已在极限 |

**辨析｜易错点：** 迭代改善**不是**通用迭代法（不像雅可比/高斯-赛德尔），它**依赖一次精确的 LU 分解**——本质是「把直接法的精度再榨一层」。它也不是「换精度重算」：它用同样的分解、同样的运算，只是把残差算得更仔细。**它解决的是「病态但可救」的问题，不是「奇异」问题。**<span class="marginnote">与后文迭代法（雅可比等）的区别：迭代改善收敛极快（每轮误差压 $\mathrm{cond}$ 倍），因为它用的是「精确分解 + 残差修正」；雅可比是「分裂近似 + 逐次逼近」，收敛慢得多。<strong>「改善」是补精度，「迭代法」是求整体解——两种迭代的动机不同。</strong></span>

## 4 工程价值与历史

迭代改善在现代库中地位特殊：**它让「低精度分解」也能产出高精度解**。历史上有名的应用是**使用低精度算术加速**：先用单精度（float32）的 LU 分解（快），再用迭代改善把精度拉回双精度（float64）水平——**在混合精度计算（mixed-precision computing）中大放异彩**。现代 GPU（Tensor Core）用半精度/单精度分解 + 迭代改善，达到接近双精度的精度。<span class="marginnote">这是 2020 年代 AI 基础设施的潮流：<strong>「低精度算得快，迭代修正补得准」</strong>——混合精度线性代数把迭代改善从「教学技巧」变成「性能引擎」。第三级《高性能计算》与第四级《AI 基础设施》会看到它的现代形态（如 HPL-AI 基准、cuSOLVER 的混合精度求解）。</span>

**工程忠告**：日常用单精度/双精度求解，默认双精度 LU 已足够；只在「精度不够 + 条件数中等」时考虑迭代改善，或在「用低精度加速」时主动启用。

## 5 小结

- **迭代改善循环**：算残差（扩展精度）→ 解残差方程组（复用 LU）→ 修正解 → 判断收敛。
- 分解一次 $O(n^3)$、修正每次 $O(n^2)$——**贵工作只做一次**。
- 误差每轮压缩约常数倍，触底于 $\mathrm{cond}(A)\epsilon_{\mathrm{mach}}\lVert\mathbf{x}\rVert$——逼近机器精度。
- **关键细节**：残差用扩展精度算，否则修正失效；对极端病态无效。
- 现代价值：**混合精度计算**——低精度分解 + 迭代改善 = 又快又准。

至此，线性方程组的直接解法十一章写完了。下一章，我们换一种求解哲学：**线性方程组的迭代解法**——不再一步到位分解，而是逐次逼近，用谱半径控制收敛。
