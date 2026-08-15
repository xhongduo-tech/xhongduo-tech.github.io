---
title: Kohn-Sham 方程
date: 2026-08-07
---

# Kohn-Sham 方程

<div class="epigraph">
<p>我们不是真的相信无相互作用参考体系存在，我们只是需要一个好用的梯子。</p>
<footer>—— 沃尔特 · 科恩（Walter Kohn）与 沈吕九（Lu Jeu Sham）</footer>
</div>

<div class="article-byline">
<p>第四级 · 第一性原理计算与电子结构理论 ｜ R. M. Martin《Electronic Structure》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从 Kohn-Sham 方程开始

上一节的 Hohenberg-Kohn 定理告诉我们「存在一个以密度为变量的普适泛函」，却留下一个致命的悬念：**动能泛函 $T[n]$ 没有已知的精确形式**。若不能算动能，DFT 就只是一句漂亮的口号。<span class="marginnote">直接写出 $T[n]$ 的近似（如 Thomas-Fermi 的 $T \propto \int n^{5/3}$）精度太差，连化学键都描述不了——这正是 Thomas-Fermi 模型在 1927 年诞生、却在化学上长期失败的原因。</span>

1965 年，Kohn 与 Sham 用一个「偷天换日」解决了它：**不要直接逼近 $T[n]$，而是引入一个无相互作用的参考体系，让它的轨道来承担动能**。真实体系的动能写成一个无相互作用动能加上小修正。这一招把 DFT 从「原则上正确」变成了「工程上可运行」，现代几乎全部电子结构软件（VASP、Quantum ESPRESSO、ABACUS、Gaussian 的 DFT 模块）都以 Kohn-Sham 方程为核心。

## 1 无相互作用参考体系

考虑一个虚构体系：$N$ 个**无相互作用**的电子，在某个有效势 $V_{\mathrm{eff}}(\mathbf{r})$ 中运动。它的基态是单粒子波函数（轨道）$\psi_i(\mathbf{r})$，密度为

$$
n(\mathbf{r}) = \sum_{i=1}^{N} |\psi_i(\mathbf{r})|^2
$$

这些轨道满足单粒子的薛定谔方程——这就是 **Kohn-Sham 方程**：

$$
\left[ -\frac{1}{2}\nabla^2 + V_{\mathrm{eff}}(\mathbf{r}) \right]\psi_i(\mathbf{r}) = \varepsilon_i\,\psi_i(\mathbf{r})
$$

**关键的一步**：Kohn-Sham 假设，真实相互作用体系的基态密度，可以**精确地**由某个这样的无相互作用体系表示——即存在一个 $V_{\mathrm{eff}}$，让上述方程的密度恰好等于真实密度。这就是 **Kohn-Sham 构造（Kohn-Sham ansatz）**。<span class="marginnote">Kohn-Sham 构造是一个「存在性假设」：它假定任何可 v-表示的真实密度都能在无相互作用体系里复现。这个假设在大多数实际情形下成立，其严格性是 DFT 数学基础研究的主题之一。</span>

**辨析｜易错点：** 无相互作用体系的轨道 $\psi_i$ **不是**真实的单电子波函数，$\varepsilon_i$ 也不是真实电子的能量（严格说只有最高占据轨道能 $-I$ 近似成立，即 Koopmans 定理的粗略版本）。它们只是**辅助工具**——一群用来拼出正确密度的数学脚手架。<span class="marginnote">Koopmans 定理：在 HF 中轨道能近似等于电离能。在 Kohn-Sham DFT 中这仅对最高占据轨道近似成立，对价带底等深能级系统性不准——这是 DFT 带隙问题的根源之一，后面「能带结构」一节会再遇。</span>

## 2 有效势的构造：把所有近似塞进一个口袋

真实体系的动能泛函 $T[n]$ 未知，Kohn-Sham 的思路是：无相互作用动能 $T_s[n] = \sum_i \langle \psi_i|-\frac12\nabla^2|\psi_i\rangle$ 用轨道精确算，把差值扔进交换关联：

$$
T[n] = T_s[n] + T_c[n], \qquad E_{\mathrm{xc}}[n] = T_c[n] + V_{ee}[n] - E_{\mathrm{Hartree}}[n]
$$

于是总能量泛函写为

$$
E[n] = T_s[n] + E_{\mathrm{Hartree}}[n] + \int n(\mathbf{r})V_{\mathrm{ext}}(\mathbf{r})\,\mathrm{d}\mathbf{r} + E_{\mathrm{xc}}[n]
$$

对 $n(\mathbf{r})$ 变分，得到有效势为三部分之和：

$$
V_{\mathrm{eff}}(\mathbf{r}) = V_{\mathrm{ext}}(\mathbf{r}) + V_{\mathrm{Hartree}}(\mathbf{r}) + V_{\mathrm{xc}}(\mathbf{r})
$$

其中 $V_{\mathrm{Hartree}}(\mathbf{r}) = \int \frac{n(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}\,\mathrm{d}\mathbf{r}'$，而**交换关联势是交换关联能对密度的泛函导数**：

$$
V_{\mathrm{xc}}(\mathbf{r}) = \frac{\delta E_{\mathrm{xc}}[n]}{\delta n(\mathbf{r})}
$$

**Kohn-Sham 方法的全部聪明与全部局限都在这一步**：只要给出 $E_{\mathrm{xc}}[n]$ 的近似，就能从 $V_{\mathrm{eff}}$ 解出轨道、由轨道得到密度、由密度更新 $V_{\mathrm{eff}}$——一个自洽循环就此闭合。DFT 所有后续发展，本质上就是「往 $E_{\mathrm{xc}}$ 这个口袋里塞更好的近似」。

## 3 公式解析：从能量泛函到 Kohn-Sham 方程

先展示「能量极小如何变成方程」，这是 DFT 里最优雅的一段推导。对总能量泛函取变分：

$$
\frac{\delta}{\delta n(\mathbf{r})}\Big[E[n] - \mu\int n(\mathbf{r}')\,\mathrm{d}\mathbf{r}'\Big] = 0
$$

其中 $\mu$ 是拉格朗日乘子（化学势），约束总电子数 $\int n = N$ 守恒。把 $E[n] = T_s[n] + E_{\mathrm{Hartree}}[n] + \int nV_{\mathrm{ext}} + E_{\mathrm{xc}}[n]$ 代入：

- **第一步，逐项求泛函导数**。轨道动能项对密度的导数正是无相互作用体系动能对单粒子密度的变分，它引出 $\psi_i$ 满足的方程；Hartree 项导数得 $V_{\mathrm{Hartree}}(\mathbf{r})$；外势项导数就是 $V_{\mathrm{ext}}(\mathbf{r})$；交换关联项导数得 $V_{\mathrm{xc}}(\mathbf{r})$。
- **第二步，拼成有效势**。三项势加起来得到 $V_{\mathrm{eff}}(\mathbf{r})$。于是「对密度变分」这一句数学操作，就翻译成了「电子在有效势中运动」的单粒子方程——这就是 Kohn-Sham 方程的由来。<span class="marginnote">这里的精妙之处：我们从未假设真实电子是无相互作用的，只是构造了一个「密度相同」的辅助体系，再把全部多体效应压缩进 $V_{\mathrm{xc}}$。变分原理保证了自洽解给出的密度就是真实基态密度。</span>
- **第三步，读化学势**。拉格朗日乘子 $\mu = \partial E/\partial N$ 在自洽时等于体系的化学势——这正是连接 DFT 与热力学、以及后面电化学、掺杂计算中费米能级移动的桥梁。

**辨析｜易错点：** 变分是在**密度**上做的，但最终得到的是**轨道**方程。初学者常困惑「到底在优化什么」。答案：轨道只是密度的参数化，$\psi_i$ 的选择不唯一（酉变换下密度不变），但密度唯一。理解「轨道是脚手架、密度是本体」，就抓住了 Kohn-Sham 的精髓。

## 4 公式解析：Kohn-Sham 方程的自洽环

把整套流程压缩成一张图，Kohn-Sham 方程是循环的心脏：

$$
n^{\mathrm{in}}(\mathbf{r}) \;\xrightarrow{\;V_{\mathrm{eff}} = V_{\mathrm{ext}} + V_H + V_{\mathrm{xc}}\;}\; \hat{H}_{\mathrm{KS}}\,\psi_i = \varepsilon_i\psi_i \;\xrightarrow{\;n^{\mathrm{out}} = \sum|\psi_i|^2\;}\; \text{检查收敛}
$$

分三步拆解这条链：

- **第一步，初始密度**：猜一个 $n^{\mathrm{in}}(\mathbf{r})$，通常由原子的叠加或前一迭代的结果给出。用它构造 Hartree 势与交换关联势，拼出 $V_{\mathrm{eff}}$。
- **第二步，解 Kohn-Sham 方程**：对角化 $\hat{H}_{\mathrm{KS}} = -\frac12\nabla^2 + V_{\mathrm{eff}}$，得到 $N$ 个最低占据轨道 $\psi_i$ 与轨道能 $\varepsilon_i$。这一步在平面波基组里就是一次大矩阵对角化，在局域轨道基组里是广义本征值问题。
- **第三步，混合与迭代**：用新轨道重新拼出 $n^{\mathrm{out}}$。但直接代入常常发散——实际程序都用**密度混合**（如 Pulay 混合、Simple mixing）把新旧密度按权重组合，然后重复循环，直到 $\int |n^{\mathrm{out}}-n^{\mathrm{in}}|$ 小于阈值。<span class="marginnote">密度混合（density mixing）是 SCF 收敛的工程核心：对金属体系，电荷在费米面附近的振动会让裸迭代震荡甚至永不收敛。Pulay 用过去几次迭代的密度做外推，是现代 DFT 代码的默认收敛加速器。</span>

**辨析｜易错点：** Kohn-Sham 方程是**非线性的**——$V_{\mathrm{eff}}$ 依赖 $n$，而 $n$ 来自方程的解。这与普通薛定谔方程「给定势、解波函数」有本质区别。因此「解 Kohn-Sham 方程」永远指「迭代到自洽」，不存在一次性求解。初学者最容易犯的错，就是把某一步迭代的轨道直接当最终答案。

## 5 Kohn-Sham 方程的物理：轨道能、带结构与局限

Kohn-Sham 轨道虽然只是辅助工具，却出奇地有用。它们是构建**能带结构**、**态密度**、**费米面**的语言——固体物理中几乎所有可视化的「电子结构」，画的都是 Kohn-Sham 轨道能 $\varepsilon_{n\mathbf{k}}$。它们也构成后续一切的起点：声子计算要用 Kohn-Sham 势的响应，GW 修正要从 Kohn-Sham 能级出发，分子动力学每一步都要解 Kohn-Sham 方程。

但 Kohn-Sham 图像有一个著名的系统误差——**带隙问题**。对半导体与绝缘体，Kohn-Sham 轨道能给出的带隙系统性偏小（典型的低估 30%–50%）。原因有两层：其一，Kohn-Sham 本征值对应的是无相互作用参考体系的激发，而非真实激发；其二，交换关联泛函近似存在导数不连续性（derivative discontinuity）缺失。<span class="marginnote">导数不连续性：精确的 $E_{\mathrm{xc}}$ 在整数电子数处有尖角，导致交换关联势对密度产生一个「跷跷板」式的跳变；近似泛函平滑了它，于是带隙被低估。这是 LDA/GGA 的老毛病，也是杂化泛函与 GW 修正的动机。</span>

## 6 一张图看清 Kohn-Sham 在整套方法里的位置

把 DFT 家族的谱系整理成表格，Kohn-Sham 方程就是那个承上启下的关节：

| 层次 | 内容 | 回答的问题 |
| --- | --- | --- |
| 定理层 | Hohenberg-Kohn 定理 | 密度能否决定一切？（能） |
| 构造层 | Kohn-Sham 方程 | 如何用轨道算动能？（引入参考体系） |
| 近似层 | 交换关联泛函 LDA/GGA/杂化 | 剩下的 $E_{\mathrm{xc}}$ 怎么逼近？ |
| 计算层 | 平面波/赝势/实空间离散 | 方程如何在计算机上解？ |
| 应用层 | 能带、声子、分子动力学 | 算出来怎么用？ |

这张表也预告了本专题接下来的路线：构造层我们已经走完，下一节进入近似层（交换关联泛函），再往后是计算层（赝势与平面波基组）、应用层（能带、声子、分子动力学），最后是超越 Kohn-Sham 的 GW 与强关联方法。<span class="marginnote">「Kohn-Sham DFT 是基座而非终点」是理解本专题的关键：声子、GW、AIMD 全部把 Kohn-Sham 解作为起点。Kohn-Sham 方程的质量，决定了整套方法的可信度上限。</span>

## 7 小结

- **Kohn-Sham 构造**：用一个无相互作用参考体系的轨道来算动能，把未知的 $T[n]$ 换成「轨道动能 + $E_{\mathrm{xc}}$ 修正」，DFT 由此可运行。
- **有效势** $V_{\mathrm{eff}} = V_{\mathrm{ext}} + V_{\mathrm{Hartree}} + V_{\mathrm{xc}}$，其中 $V_{\mathrm{xc}} = \delta E_{\mathrm{xc}}/\delta n$。
- **求解方式**：非线性自洽循环，密度混合保证收敛；轨道 $\psi_i$、能级 $\varepsilon_i$