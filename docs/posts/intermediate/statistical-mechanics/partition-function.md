---
title: 配分函数与热力学量
date: 2026-08-07
---

# 配分函数与热力学量

<div class="epigraph">
<p>配分函数是整个统计力学的枢纽：微观的能谱从这里进，宏观的热力学量从这里出。</p>
<footer>—— 乔赛亚 · 威拉德 · 吉布斯（Josiah Willard Gibbs）</footer>
</div>

<div class="article-byline">
<p>第二级 · 统计力学与热力学 ｜ 汪志诚《热力学·统计物理》第7章 §7.1 · Pathria《统计力学》Ch. 3.5 ｜ 2026-08-07</p>
</div>

## 为什么从配分函数开始

上一节引入玻尔兹曼分布时，配分函数 $Z$ 只是归一化分母。但它的地位远不止此：**知道配分函数，就"知道一切"**——内能、自由能、熵、压强、化学势全部能从 $Z$ 求出来。本节把这条「配分函数 → 热力学量」的流水线完整建起来。这是统计物理方法的第一次总装：从此解决任何系统，都只需三步——写能谱、算配分函数、求热力学量。<span class="marginnote">「配分函数」（partition function）之名源于德语 Zustandssumme（状态求和）。对由独立可分辨粒子组成的系统，$Z$ 可分解为单粒子配分函数 $z_1$ 的乘积——这种「因子分解」让许多问题从多体变成单体，是配分函数最强大的工程便利。</span>

## 1 配分函数的定义与性质

对能级 $\varepsilon_l$（简并度 $\omega_l$）的系统，配分函数定义为对微观态的求和：

$$Z = \sum_l \omega_l\, e^{-\beta\varepsilon_l} = \sum_{\text{微观态 } i} e^{-\beta\varepsilon_i}$$

它有三个基本性质：

- **归一化**：$P_i = e^{-\beta\varepsilon_i}/Z$，保证 $\sum_i P_i = 1$。
- **单调性**：$Z$ 是 $\beta$（或说 $1/T$）的单调递减函数，$T \to \infty$ 时 $Z \to$（微观态总数），$T\to 0$ 时 $Z \to$（基态简并度）。
- **因子分解**：对由 $N$ 个独立可分辨粒子组成的系统，$Z = z_1^N$，其中 $z_1$ 是单粒子配分函数；对全同粒子须修正为 $Z = z_1^N/N!$（吉布斯修正）。

<span class="marginnote">吉布斯修正 $1/N!$ 我们已在《统计物理基本概念》见过：全同粒子不可分辨，把 $N$ 个粒子"分别放在哪些单粒子态上"的多计数除掉。不修正会导致吉布斯佯谬——等温等压下混合同种气体计算出的熵增为负。</span>

## 2 从配分函数求热力学量

这是本节的核心工作表。设 $Z(\beta, V, N)$ 为配分函数，则：

| 热力学量 | 公式 | 名字 |
| --- | --- | --- |
| 内能 $U$ | $U = -\dfrac{\partial \ln Z}{\partial \beta}$ | 能量平均 |
| 亥姆霍兹自由能 $F$ | $F = -\dfrac{1}{\beta}\ln Z$ | 最重要的桥 |
| 熵 $S$ | $S = k_B\left(\ln Z + \beta U\right)$ | 由 $F = U - TS$ 反推 |
| 压强 $p$ | $p = -\left(\dfrac{\partial F}{\partial V}\right)_{T,N}$ | 力学量 |
| 化学势 $\mu$ | $\mu = \left(\dfrac{\partial F}{\partial N}\right)_{T,V}$ | 物质流动 |

所有公式都从一个对象 $Z$ 出发。物理图像是：**$Z$ 把微观能谱"压缩编码"成一个复数，热力学量是它的各种导数**。<span class="marginnote">表格里的每条公式都值得自己推一遍：$U$ 从 $\langle E\rangle = \sum_i P_i\varepsilon_i$ 直接算；$F$ 用 $S = k_B\ln\Omega$ 或 $S = -\partial F/\partial T$ 反推验证。公式能记住是下限，能推导才是真掌握——这也是考试与科研的分水岭。</span>

## 3 公式解析：$F = -k_BT\ln Z$ 为什么是"总账本"

这条公式值得逐项拆解：

$$

F = -k_BT\ln Z

$$

- **第一步，自由能的定义**：$F = U - TS$。热力学里它是恒温恒容下的平衡判据（$F$ 取极小）；统计力学里我们"反过来"用 $Z$ 定义它。
- **第二步，验证 $U$ 的兼容**：$-(\partial F/\partial T)_V = S$、$F + TS = U$，代入 $F = -k_BT\ln Z$ 应能回到上一行的 $U = -\partial\ln Z/\partial\beta$。可自行验证这条链自洽。
- **第三步，为什么取对数**：$\ln Z$ 把微观态数目从指数级压缩到"计数级别的可加量"。因为 $Z \sim \Omega \cdot e^{-\beta E}$ 的量级是 $e^{\text{（宏观）}}$，取对数后才得到与 $N$ 成正比的广延量 $F$。
- **第四步，物理意义**：$F$ 是系统在温度 $T$ 下"还能拿出多少可做功的自由能量"。$Z$ 越大（微观态越多、温度越高），$\ln Z$ 越大，$F$ 越负——系统越"自由"，越不稳定地想要释放。

## 4 例子：单粒子配分函数与理想气体

用一条完整流水线示范。自由粒子的能谱为 $\varepsilon = p^2/2m$，单粒子配分函数：

$$z_1 = \frac{1}{h^3}\int e^{-\beta p^2/2m}\,\mathrm{d}^3p\,\mathrm{d}^3q = \frac{V}{h^3}\left(\frac{2\pi m k_BT}{\beta}\right)^{3/2}$$

对 $N$ 个全同粒子，$Z = z_1^N/N!$，取对数（用斯特林公式）：

$$\ln Z = N\left[\ln\frac{V}{N} + \frac{3}{2}\ln\frac{2\pi m k_BT}{h^2} + 1\right]$$

由此立即得到：

- 内能 $U = -\partial\ln Z/\partial\beta = \frac{3}{2}Nk_BT$；
- 状态方程 $p = -\partial F/\partial V = \frac{Nk_BT}{V}$，即 $pV = Nk_BT$——**理想气体状态方程被统计力学"算"出来了**。

这是统计物理方法的完整示范：**从"自由粒子的能谱"这一条微观信息出发，推出一条宏观状态方程**。<span class="marginnote">「$pV = Nk_BT$ 是算出来的而非假设的」是统计物理最好的自我介绍。热力学把它当作实验定律，统计力学把它当作自由粒子配分函数的必然结果。下一节《理想气体》会系统地走完这条流水线，并处理双原子分子等更丰富的能谱。</span>

## 5 公式解析：$U = -\partial\ln Z/\partial\beta$ 的推导演示

这条公式是一切热力学量求法的起点，值得亲手推一遍：

$$
U = -\frac{\partial \ln Z}{\partial\beta} = \frac{1}{Z}\sum_l \omega_l\,\varepsilon_l\, e^{-\beta\varepsilon_l}
$$

- **第一步，对 $\beta$ 求导**：$-\partial Z/\partial\beta = \sum_l \omega_l\varepsilon_l e^{-\beta\varepsilon_l}$，导数的分子恰好是"能量加权的玻尔兹曼因子求和"。
- **第二步，除以 $Z$**：$\frac{1}{Z}\sum_l \omega_l\varepsilon_l e^{-\beta\varepsilon_l}$ 正是"以玻尔兹曼因子为权重的能量平均" $\langle\varepsilon\rangle$。
- **第三步，物理意义**：内能 = 能级的概率加权平均。温度升高，高能级权重大，平均能量上升——$U$ 随 $T$ 增加，热容为正。
- **第四步，推广**：$\langle\varepsilon^2\rangle$ 可同理从 $Z$ 的二阶导求出，涨落 $\langle(\Delta E)^2\rangle = \partial^2\ln Z/\partial\beta^2 = k_BT^2C_V$ 将引出《涨落理论》一节的能量涨落-热容关系。

## 6 例题：两能级系统的配分函数全流程

把「能谱 → $Z$ → 热力学量」的流水线在最小系统上完整走一遍。

**能谱与配分函数**。两能级系统（能级 $\varepsilon_1 < \varepsilon_2$，简并度 $\omega_1, \omega_2$）：

$$Z = \omega_1 e^{-\beta\varepsilon_1} + \omega_2 e^{-\beta\varepsilon_2}$$

**内能**（$U = -\partial\ln Z/\partial\beta$）：

$$U = \frac{\omega_1\varepsilon_1 e^{-\beta\varepsilon_1} + \omega_2\varepsilon_2 e^{-\beta\varepsilon_2}}{\omega_1 e^{-\beta\varepsilon_1} + \omega_2 e^{-\beta\varepsilon_2}}$$

**自由能**（$F = -k_BT\ln Z$）与**熵**（$S = k_B(\ln Z + \beta U)$）同步得出。

**读出温度依赖**：

- $T \to 0$（$\beta \to \infty$）：$Z \to \omega_1 e^{-\beta\varepsilon_1}$，$U \to \varepsilon_1$——系统冻结在最低能级；
- $T \to \infty$（$\beta \to 0$）：$Z \to \omega_1 + \omega_2$，$U \to (\omega_1\varepsilon_1 + \omega_2\varepsilon_2)/(\omega_1+\omega_2)$——所有能级等概率占据；
- 热容 $C = \partial U/\partial T$ 在 $k_BT \sim \Delta\varepsilon$ 处有峰（薛定谔峰），高温低温都趋于零。<span class="marginnote">这个例题的价值在于「全流程可算」：$\varepsilon_1, \varepsilon_2, \omega_1, \omega_2$ 一旦给定，所有热力学量都能显式写出，且每个极限（低温冻结、高温均分）都与直觉吻合。它是理解配分函数方法的「最小完整单元」——把这条流水线走熟，任何系统都只是「换能谱」。</span>

**辨析｜易错点：** 两能级系统的热容峰值在「能级间距 = $k_BT$」处，这不是「相变」——热容有限、光滑、无奇点。相变要求热力学极限下的非解析行为，单个两能级系统永远不会相变。把「热容峰」与「相变」区分开，是理解《Ising 模型与相变》的前提：相变是大量自由度协作的结果，单自由度只有「峰」。

## 7 小结

- **配分函数** $Z = \sum_l \omega_l e^{-\beta\varepsilon_l}$：微观能谱的"压缩编码"，全部热力学量的源头。
- **三大桥梁**：$F = -k_BT\ln Z$、$U = -\partial\ln Z/\partial\beta$、$S = k_B(\ln Z + \beta U)$。
- **因子分解**：独立可分辨粒子 $Z = z_1^N$；全同粒子 $Z = z_1^N/N!$。
- 理想气体状态方程 $pV = Nk_BT$ 可由自由粒子配分函数完整推导——统计力学的示范性胜利。
- 掌握方法骨架（能谱 → $Z$ → 热力学量），就能处理从气体到固体的所有近独立系统。

在下一节，我们把这条流水线第一次完整跑在真实系统上——**理想气体**。
