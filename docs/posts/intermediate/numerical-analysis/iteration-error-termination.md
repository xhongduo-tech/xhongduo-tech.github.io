---
title: 迭代法的误差估计与终止准则
date: 2026-08-07
---

# 迭代法的误差估计与终止准则：什么时候该停

<div class="epigraph">
<p>迭代的艺术不只是如何走，还有何时停。</p>
<footer>—— 数值迭代的工程智慧</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§6.3 ｜ 2026-08-07</p>
</div>

## 为什么从终止准则开始

迭代法不告诉我们「算到多少步」——它逐步逼近真解，工程上需要**一个自动的停止判据**：什么时候近似解够好了，可以收工。这个判据不能依赖真解（我们不知道真解），只能用**迭代过程中能算的量**：相邻迭代之差、残差范数。本节建立「从可观测的量反推真实误差」的估计框架，并给出可靠的终止准则。<span class="marginnote">核心难题：<strong>迭代法里唯一能直接算的是「相邻两步之差 $\mathbf{x}^{(k+1)}-\mathbf{x}^{(k)}$」和「残差 $\mathbf{r}^{(k)}=\mathbf{b}-A\mathbf{x}^{(k)}$」，真正的误差 $\mathbf{e}^{(k)}=\mathbf{x}^{(k)}-\mathbf{x}^*$ 不可见</strong>。终止准则的任务，就是从前者估计后者。</span>

本节给出两条估计路线——相邻差与残差——以及它们的可靠性与坑。

## 1 用相邻差估计误差

直觉上，「两步之间几乎不动了」应该意味着收敛。但要多小心？设迭代矩阵 $G$，谱半径 $\rho<1$。由 $\mathbf{x}^{(k+1)}-\mathbf{x}^{(k)}=G(\mathbf{x}^{(k)}-\mathbf{x}^{(k-1)})$：

$$
\lVert\mathbf{x}^{(k+1)}-\mathbf{x}^{(k)}\rVert \le \rho\,\lVert\mathbf{x}^{(k)}-\mathbf{x}^{(k-1)}\rVert
$$

**相邻差的收敛速度就是谱半径。** 进一步，有重要的**误差上界**：

$$
\lVert\mathbf{x}^{(k)}-\mathbf{x}^*\rVert \le \frac{\rho}{1-\rho}\,\lVert\mathbf{x}^{(k)}-\mathbf{x}^{(k-1)}\rVert
$$

**公式解析：这个界的直觉。**

- **第一步，几何级数求和。** $\mathbf{x}^{(k)}-\mathbf{x}^*$ 是「未来所有步长之和的极限」：

$$
\mathbf{x}^{(k)}-\mathbf{x}^* = \sum_{j=0}^{\infty}\left(\mathbf{x}^{(k+j)}-\mathbf{x}^{(k+j+1)}\right)
$$

- **第二步，逐项压缩。** 相邻差每步压缩 $\rho$ 倍：$\lVert\mathbf{x}^{(k+j)}-\mathbf{x}^{(k+j+1)}\rVert\le\rho^j\lVert\mathbf{x}^{(k)}-\mathbf{x}^{(k+1)}\rVert$。
- **第三步，求和。** $\lVert\mathbf{x}^{(k)}-\mathbf{x}^*\rVert\le\lVert\mathbf{x}^{(k+1)}-\mathbf{x}^{(k)}\rVert\sum_{j=0}^\infty\rho^j=\dfrac{\lVert\mathbf{x}^{(k+1)}-\mathbf{x}^{(k)}\rVert}{1-\rho}\le\dfrac{\rho}{1-\rho}\lVert\mathbf{x}^{(k)}-\mathbf{x}^{(k-1)}\rVert$。

**当 $\rho$ 接近 1 时，因子 $\dfrac{\rho}{1-\rho}$ 巨大**——「相邻差小」远不能保证「误差小」。若 $\rho=0.99$，相邻差 $10^{-4}$ 时误差可能高达 $10^{-2}$——**提前收工的危险**。<span class="marginnote">工程教训：<strong>「相邻差阈值」只在 $\rho$ 较小时可靠</strong>。收敛慢的迭代（$\rho$ 接近 1）必须用「相邻差 × $\rho/(1-\rho)$」修正，或用残差准则（更稳）。</span>

## 2 用残差估计误差

残差 $\mathbf{r}^{(k)}=\mathbf{b}-A\mathbf{x}^{(k)}$ 是可算的，且与误差通过 $A$ 相连：$\mathbf{e}^{(k)}=A^{-1}\mathbf{r}^{(k)}$。取范数：

$$
\lVert\mathbf{e}^{(k)}\rVert \le \lVert A^{-1}\rVert\,\lVert\mathbf{r}^{(k)}\rVert
$$

用相对形式更好：

$$
\frac{\lVert\mathbf{e}^{(k)}\rVert}{\lVert\mathbf{x}^{(k)}\rVert} \le \mathrm{cond}(A)\,\frac{\lVert\mathbf{r}^{(k)}\rVert}{\lVert\mathbf{b}\rVert}
$$

**残差准则 = 条件数 × 相对残差**。残差小（相对 $\mathbf{b}$）且条件数中等时，误差才小；**病态时残差小但误差大**（与直接法一节同样的反直觉）。

**实用终止准则（两种）**：

1. **残差准则**：$\lVert\mathbf{r}^{(k)}\rVert\le\varepsilon_r\lVert\mathbf{b}\rVert$——「残差相对缩小到 $\varepsilon_r$」。常用 $\varepsilon_r=10^{-6}$。
2. **相邻差准则**：$\lVert\mathbf{x}^{(k+1)}-\mathbf{x}^{(k)}\rVert\le\varepsilon_x\lVert\mathbf{x}^{(k+1)}\rVert$——「解几乎不动」。只对 $\rho$ 小的迭代可靠。

## 3 数值演示：两种准则的对比

用高斯-赛德尔解前面的 SPD 三对角系统，追踪残差、相邻差与真实误差：

| 步数 | $\lVert\mathbf{r}\rVert_\infty$ | $\lVert\mathbf{x}^{(k+1)}-\mathbf{x}^{(k)}\rVert_\infty$ | 真实误差 $\lVert\mathbf{e}\rVert_\infty$ |
| --- | --- | --- | --- |
| 1 | 2.6 | 1.4 | 0.9 |
| 3 | 0.6 | 0.30 | 0.2 |
| 6 | $7\times10^{-3}$ | $3\times10^{-3}$ | $2\times10^{-3}$ |
| 10 | $10^{-5}$ | $5\times10^{-6}$ | $3\times10^{-6}$ |

注意：**真实误差与残差、相邻差三者同数量级**（$\rho$ 小的系统）——此时用任一准则都安全。若换 $\rho=0.99$ 的慢收敛系统，相邻差会「骗人」：相邻差 $10^{-4}$ 时真实误差可能 $10^{-2}$。<span class="marginnote">选择建议：<strong>拿不准谱半径时，优先残差准则</strong>——它通过条件数显式挂钩，比「相邻差」多一层理论保障。残差也可用「残差与初始残差之比」$\lVert\mathbf{r}^{(k)}\rVert/\lVert\mathbf{r}^{(0)}\rVert$，对「改善了多少」更敏感。</span>

## 4 工程上的安全做法

1. **双准则**：同时检查残差与相邻差，**两个都满足**才停——防「残差小但相邻差大」或反之。
2. **最大迭代上限**：设最大迭代次数 $N_{\max}$（如 $10^4$）兜底，防止永不收敛时死循环。
3. **相对而非绝对**：用 $\lVert\mathbf{r}\rVert\le\varepsilon\lVert\mathbf{b}\rVert$ 而非 $\lVert\mathbf{r}\rVert\le\varepsilon$——避免「$\mathbf{b}$ 很大时绝对阈值太紧」或「$\mathbf{b}$ 很小时太松」。
4. **发散检测**：监测误差/残差是否上升，连续上升即宣布发散，停止。

**辨析｜易错点：** 迭代法「不收敛」时，残差可能**先降后升**——别在第 3 步看残差小就宣布成功。**连续若干步残差不降反升是发散信号**，应立即停止并检查谱半径或 $\omega$。<span class="marginnote">一个容易被忽视的细节：<strong>停机时记录「达到了哪条准则」</strong>——若靠 $N_{\max}$ 撞上限停的，说明没真正收敛，结果不可信。工程日志里「收敛 / 撞上限 / 发散」三种状态必须区分。</span>

## 5 终止准则与精度预算

把终止准则与整条误差链连起来：

$$
\underbrace{\lVert\mathbf{x}^{(k)}-\mathbf{x}^*\rVert}_{\text{解的误差}} \le \mathrm{cond}(A)\underbrace{\frac{\lVert\mathbf{r}^{(k)}\rVert}{\lVert\mathbf{b}\rVert}}_{\text{终止准则控制}} \lVert\mathbf{x}^{(k)}\rVert
$$

**想要解的相对误差 $\le10^{-6}$，且 $\mathrm{cond}(A)=10^4$，则残差相对阈值要取 $\varepsilon_r=10^{-10}$**——终止阈值必须按条件数「打折」选。**精度预算公式**：

$$
\varepsilon_r \le \frac{\varepsilon_{\text{desired}}}{\mathrm{cond}(A)}
$$

**工程结论：先估条件数，再定终止阈值**——否则「以为达标、其实差十万八千里」。

## 6 小结

- **相邻差误差界**：$\lVert\mathbf{x}^{(k)}-\mathbf{x}^*\rVert\le\dfrac{\rho}{1-\rho}\lVert\mathbf{x}^{(k)}-\mathbf{x}^{(k-1)}\rVert$——$\rho$ 接近 1 时相邻差严重低估误差。
- **残差误差界**：$\dfrac{\lVert\mathbf{e}\rVert}{\lVert\mathbf{x}\rVert}\le\mathrm{cond}(A)\dfrac{\lVert\mathbf{r}\rVert}{\lVert\mathbf{b}\rVert}$——残差准则更可靠。
- 实用准则：相对残差 $\lVert\mathbf{r}\rVert\le\varepsilon\lVert\mathbf{b}\rVert$（推荐）、相对相邻差（限 $\rho$ 小）。
- 安全四件套：双准则、最大迭代上限、相对阈值、发散检测。
- **终止阈值按条件数打折**：$\varepsilon_r=\varepsilon_{\text{desired}}/\mathrm{cond}(A)$；停机状态要区分收敛/撞上限/发散。

至此，线性方程组的迭代解法八章写完了。下一章，我们转向特征值问题：**矩阵特征值计算**——从格什戈林圆盘估计特征值的位置开始。
