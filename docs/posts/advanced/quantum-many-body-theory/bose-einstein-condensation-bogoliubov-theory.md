---
title: 玻色-爱因斯坦凝聚与Bogoliubov理论
date: 2026-08-07
---

# 玻色-爱因斯坦凝聚与Bogoliubov理论

<div class="epigraph">
<p>当玻色子冷却到临界温度以下，它们不再各自为政，而是全体涌入同一个量子态——宏观数量的粒子共享一个波函数。这种「集体的独奏」就是玻色-爱因斯坦凝聚。</p>
<footer>—— 文小刚（Xiao-Gang Wen）《量子多体理论》</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ 文小刚《量子多体理论》 第3章 ｜ 2026-08-07</p>
</div>

## 为什么需要 BEC 与 Bogoliubov 理论

上一节的对称破缺给了我们一个普遍框架：连续对称性破缺 → 序参量 → Goldstone 模。现在把它用到**玻色子系统**——其中最深刻的现象是 **玻色-爱因斯坦凝聚（Bose-Einstein condensation, BEC）**：大量玻色子在低温下「涌进」同一单粒子态，形成宏观量子占据。

BEC 不仅是超流氦、冷原子实验、玻色子气体的基石，更是理解**对称破缺如何在凝聚态里真正发生**的干净范例：凝聚体的波函数 $\langle\hat{\psi}(\mathbf{r})\rangle \neq 0$ 就是 $U(1)$ 对称破缺的序参量，而它的相位涨落就是那个 Goldstone 模（声子型）。**Bogoliubov 理论（Bogoliubov theory）**则给出凝聚体的激发谱：长波极限下是线性声子——这正是超流性的微观基础。<span class="marginnote">BEC 的历史：Bose 1924 年把光子统计推广到物质粒子，Einstein 随即预言凝聚；1938 年 London 指出液氦超流是 BEC 的表现；1995 年 Cornell、Ketterle、Wieman 在铷/钠原子气中首次实现稀薄气体 BEC（2001 年诺贝尔奖）。从理论预言到实验实现，跨了整整 70 年。</span>

## 1 理想玻色气体的凝聚

对**理想玻色气体**（无相互作用），玻色-爱因斯坦分布 $n(\varepsilon) = 1/(e^{\beta(\varepsilon-\mu)}-1)$ 在化学势 $\mu\to0^-$ 时，基态（$\varepsilon=0$）占据数发散。当温度降到临界温度以下，基态占据变成**宏观量级**（与 $N$ 同阶），其余粒子仍分布在激发态上。临界温度：

$$T_c = \frac{2\pi\hbar^2}{mk_B}\Big(\frac{n}{\zeta(3/2)}\Big)^{2/3}$$

其中 $\zeta(3/2) \approx 2.612$ 是黎曼 ζ 函数。凝聚体占比（$T<T_c$）：

$$\frac{N_0}{N} = 1 - \Big(\frac{T}{T_c}\Big)^{3/2}$$

**重点：理想玻色气体在 $T_c$ 处发生凝聚，凝聚体占比 $N_0/N$ 是连续变化的（二级相变），但作为序参量它从零增长到宏观值。** 注意理想气体模型预测的 $T_c$ 与密度、质量的关系，是理解真实 BEC 实验（$n\sim10^{13}\text{cm}^{-3}$，$T_c\sim\mu\text{K}$）的第一把尺子。<span class="marginnote">理想气体的凝聚「几何」：在 $d$ 维中，凝聚只在 $d>2$ 时发生（三维、准二维的有限温度凝聚需要 trap；严格二维只有 $T=0$ 的 BKT 型凝聚）。维度是玻色凝聚的第一道门槛——这与重正化群里「维度决定临界行为」一致。</span>

**辨析｜易错点：** 初学者常以为「BEC = 粒子都落到动能最低态」。更准确地说：**BEC = 单粒子密度矩阵 $\rho_1(\mathbf{r},\mathbf{r}') = \langle\hat{\psi}^\dagger(\mathbf{r}')\hat{\psi}(\mathbf{r})\rangle$ 有宏观本征值**——即有一个单粒子态被宏观占据（$N_0 \sim N$），且伴随**离相位序（off-diagonal long-range order, ODLRO）**：$\rho_1(\mathbf{r},\mathbf{r}') \to N_0\phi^*(\mathbf{r}')\phi(\mathbf{r}) \neq 0$ 当 $|\mathbf{r}-\mathbf{r}'|\to\infty$。无相互作用时是「基态占据」，有相互作用时是「宏观本征值」——后者才是普适定义。

## 2 U(1) 对称破缺与凝聚体波函数

把对称破缺框架套到 BEC 上。凝聚意味着场算符有**非零期望值**：

$$\langle\hat{\psi}(\mathbf{r})\rangle = \phi(\mathbf{r}) \neq 0$$

$\phi(\mathbf{r})$ 就是**凝聚体波函数（凝聚体序参量）**。由于 $\hat{\psi} \to e^{i\alpha}\hat{\psi}$ 是 $U(1)$ 对称变换（粒子数守恒），$\langle\hat{\psi}\rangle \neq 0$ 意味着**粒子数对称性自发破缺**——凝聚体有了确定的相位。凝聚体的相位 $\theta$ 就是破缺方向，全局相位旋转不改变能量。

**重点：BEC = $U(1)$ 自发破缺，Goldstone 模就是凝聚体的相位涨落。** 相位只以导数进入能量（梯度产生超流速度 $\mathbf{v}_s = \frac{\hbar}{m}\nabla\theta$），所以相位模是零质量声子——这正是超流性的根源（下一节 Landau 判据）。凝聚体波函数的演化由**Gross-Pitaevskii 方程（GP 方程）**描述：

$$i\hbar\frac{\partial\phi}{\partial t} = \Big(-\frac{\hbar^2\nabla^2}{2m} + V_{\text{ext}} + g|\phi|^2\Big)\phi$$

其中 $g = 4\pi\hbar^2 a_s/m$ 由 s 波散射长度 $a_s$ 决定。<span class="marginnote">GP 方程是凝聚体的「非线性薛定谔方程」：$g|\phi|^2$ 项是平均场相互作用。它描述涡旋、孤子、暗孤子等宏观量子现象，也是冷原子实验理论分析的标准工具。对稀薄玻色气体，GP 方程是系统性的首阶近似（Gross-Pitaevskii 层级的平均场）。</span>

## 3 Bogoliubov 理论：激发谱

**Bogoliubov 理论**处理**弱相互作用的稀薄玻色气体**的激发。核心步骤：把场算符在凝聚体附近展开：

$$\hat{\psi}(\mathbf{r}) = \phi_0 + \delta\hat{\psi}(\mathbf{r})$$

保留到 $\delta\hat{\psi}$ 的二次项（忽略三次以上的涨落相互作用），哈密顿量变成 $\delta\hat{\psi}$ 的二次型。用**Bogoliubov 变换**对角化：

$$\hat{\alpha}_{\mathbf{k}} = u_{\mathbf{k}}\,\hat{b}_{\mathbf{k}} - v_{\mathbf{k}}\,\hat{b}_{-\mathbf{k}}^\dagger$$

其中 $\hat{b}_{\mathbf{k}}$ 是动量 $\mathbf{k}$ 的平面波涨落算符，$u_{\mathbf{k}}, v_{\mathbf{k}}$ 满足 $u_{\mathbf{k}}^2 - v_{\mathbf{k}}^2 = 1$（保证 $\hat{\alpha}$ 保持玻色对易关系）。对角化后的激发谱：

$$\varepsilon_{\mathbf{k}} = \sqrt{\varepsilon_{\mathbf{k}}^{(0)2} + 2\,g n_0\,\varepsilon_{\mathbf{k}}^{(0)}}, \qquad \varepsilon_{\mathbf{k}}^{(0)} = \frac{\hbar^2 k^2}{2m}$$

**重点：Bogoliubov 谱的长波极限是关键。** 当 $\varepsilon_{\mathbf{k}}^{(0)} \ll gn_0$（长波），$\varepsilon_{\mathbf{k}} \approx \sqrt{\frac{gn_0}{m}}\,\hbar k \equiv c\hbar k$——**线性声子谱**！声速 $c = \sqrt{gn_0/m}$。这正是 Goldstone 模（相位模）：$U(1)$ 破缺给出的零质量声子。<span class="marginnote">Bogoliubov 变换的物理：凝聚体的存在允许「粒子数不守恒」的混合——$\hat{b}_{\mathbf{k}}^\dagger\hat{b}_{-\mathbf{k}}^\dagger$ 项（同时产生两个粒子）在 $U(1)$ 破缺后不再守恒禁止。这就是为什么「对易子+破缺」会自动生成声子：声子 = 相位模式的量子，而相位模式的存在正是粒子数破缺的后果。</span>

**短波极限**：$\varepsilon_{\mathbf{k}}^{(0)} \gg gn_0$ 时，$\varepsilon_{\mathbf{k}} \approx \varepsilon_{\mathbf{k}}^{(0)} + gn_0$——回到自由粒子的抛物线色散，只是加了一个平均场化学势。**整个 Bogoliubov 谱从长波声子平滑过渡到短波自由粒子**，交叉点由愈合长度 $\xi = \hbar/\sqrt{2mgn_0}$ 标定。

## 4 公式解析：Bogoliubov 变换的三步推导

把 Bogoliubov 对角化过程走一遍：

**第一步，写哈密顿量二次型**：$H = \sum_{\mathbf{k}}\varepsilon_{\mathbf{k}}^{(0)}\hat{b}_{\mathbf{k}}^\dagger\hat{b}_{\mathbf{k}} + \frac{gn_0}{2}\sum_{\mathbf{k}\neq0}\big[\hat{b}_{\mathbf{k}}^\dagger\hat{b}_{-\mathbf{k}}^\dagger + \hat{b}_{\mathbf{k}}\hat{b}_{-\mathbf{k}} + 2\hat{b}_{\mathbf{k}}^\dagger\hat{b}_{\mathbf{k}}\big]$。常数项来自凝聚体密度 $n_0$ 的能量。
**第二步，Bogoliubov 变换**：设 $\hat{b}_{\mathbf{k}} = u_{\mathbf{k}}\hat{\alpha}_{\mathbf{k}} + v_{\mathbf{k}}\hat{\alpha}_{-\mathbf{k}}^\dagger$。要求 $\hat{\alpha}$ 仍满足玻色对易 $[\hat{\alpha}_{\mathbf{k}},\hat{\alpha}_{\mathbf{k}'}^\dagger]=\delta_{\mathbf{k}\mathbf{k}'}$，得 $u^2-v^2=1$。代入消去「非对角项」（$\hat{\alpha}\hat{\alpha}$ 与 $\hat{\alpha}^\dagger\hat{\alpha}^\dagger$），确定 $u_{\mathbf{k}},v_{\mathbf{k}}$。
- **第三步，读对角化能量**：对角项系数给出 $\varepsilon_{\mathbf{k}} = \sqrt{\varepsilon_{\mathbf{k}}^{(0)2} + 2gn_0\varepsilon_{\mathbf{k}}^{(0)}}$。$v_{\mathbf{k}}^2$ 给出基态里「粒子对」的占据——凝聚体基态不是真空，而是含无限多「涨落对」的挤压态（squeezed state）。
- **第四步，物理**：长波 $\varepsilon\approx c\hbar k$（声子），短波 $\varepsilon\approx\varepsilon^{(0)}+gn_0$（自由粒子）；$c = \sqrt{gn_0/m}$ 就是超流声速。

**重点：Bogoliubov 变换的本质是「在破缺背景下重对角化」——** 把「粒子数不守恒」的二次型通过 u-v 混合变成「准粒子数守恒」的形式。准粒子（声子）才是真正的基本激发，而裸粒子是「准粒子的叠加」。这个「准粒子不是裸粒子」的思想，是理解超流、超导与所有凝聚的钥匙。

## 5 BEC 与「从极限到大模型」

BEC 是「从极限到大模型」里「**量变到质变**」的最纯物理版本：温度连续下降，粒子一个个占据激发态；到临界点，**一个态的占据突然变成宏观量级**——单个量子态「涌现」出与整个系统同阶的重要性。这与大模型里「涌现能力」的逻辑高度同构：参数/数据连续增长，在某个规模点出现质的跃迁（推理、上下文学习等能力的突然出现）。<span class="marginnote">更精确的类比在<strong>表征坍缩与模式坍缩</strong>：生成模型在训练中出现「模式坍缩」——输出集中到少数几个模式——在结构上与 BEC 的「宏观占据」类似：分布的质量「凝聚」到少数点上。理解「什么条件导致宏观占据/模式坍缩」是两边的共同难题。可参考第四级《生成模型》。</span>

对多体理论自身，BEC 与 Bogoliubov 理论是超流与超导的起点——下一节我们直接回答「为什么超流不粘」：**超流与 Landau 判据**。

## 6 小结

- **BEC**：单粒子密度矩阵出现宏观本征值，序参量 $\langle\hat{\psi}\rangle\neq0$，伴随离相位序（ODLRO）。
- 理想玻色气体的临界温度 $T_c \propto n^{2/3}/m$，凝聚占比 $N_0/N = 1-(T/T_c)^{3/2}$。
- **BEC = $U(1)$ 自发破缺**：凝聚体波函数是序参量，相位涨落是 Goldstone 模。
- **Bogoliubov 谱** $\varepsilon_{\mathbf{k}} = \sqrt{\varepsilon^{(0)2}+2gn_0\varepsilon^{(0)}}$：长波线性声子（$c=\sqrt{gn_0/m}$），短波自由粒子。
- Bogoliubov 变换把粒子数不守恒的二次型重对角化成准粒子，凝聚体基态是「挤压态」。
- 愈合长度 $\xi$ 标定声子-粒子过渡；GP 方程描述凝聚体的平均场动力学。

在下一节，我们研究凝聚体最重要的宏观后果：**超流与 Landau 判据**——为什么液氦流过毛细管毫无摩擦，以及「激发谱的斜率决定超流稳定性」这条深刻的判据。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| 临界温度 | $T_c = \frac{2\pi\hbar^2}{mk_B}\big(\frac{n}{\zeta(3/2)}\big)^{2/3}$ | 理想玻色气体凝聚温度 |
| 凝聚占比 | $N_0/N = 1-(T/T_c)^{3/2}$ | 二级相变，连续增长 |
| 序参量 | $\langle\hat{\psi}(\mathbf{r})\rangle = \phi(\mathbf{r})\neq0$ | $U(1)$ 对称破缺 |
| Bogoliubov 谱 | $\varepsilon_{\mathbf{k}} = \sqrt{\varepsilon^{(0)2} + 2gn_0\varepsilon^{(0)}}$ | 长波声子，短波自由粒子 |
| 声速 | $c = \sqrt{gn_0/m}$ | 超流声速，Goldstone 模 |
| GP 方程 | $i\hbar\partial_t\phi = (-\frac{\hbar^2\nabla^2}{2m}+V+g|\phi|^2)\phi$ | 凝聚体的非线性薛定谔方程 |

**易错复盘**：两点要分清。其一，BEC 不等于超流——理想玻色气体凝聚但 $v_c=0$ 不超流，必须靠相互作用给出线性声子谱（Bogoliubov）；其二，BEC 的普适定义是「单粒子密度矩阵出现宏观本征值」，不是简单的「基态被占据」——有相互作用时两者不同。

**知识连线**：BEC 是第 3 篇对称破缺在玻色子系统的实现（$U(1)$ 破缺 + Goldstone 相位模）；Bogoliubov 变换与第 3 篇 BCS 的配对凝聚、第 4 篇自旋波理论同构——「对称破缺背景上的玻色化 + 对角化」是贯穿全书的方法论模板。GP 方程与超流（§第 3 篇）相连。

**延伸思考**：为什么 $d\le2$ 的理想玻色气体没有有限温凝聚？提示：临界温度公式里 $n^{2/3}$ 的低维对应发散。Bogoliubov 变换 $u^2-v^2=1$ 与玻色对易关系如何自洽？提示：$[\alpha,\alpha^\dagger]=[b,b^\dagger]=1$ 的保持要求。


**实践与辨析**：一道综合题：从 Bogoliubov 谱推导超流声速 $c=\sqrt{gn_0/m}$，并说明为何 $g\to0$ 时声速消失、理想玻色气体不超流。提示：长波极限 $\varepsilon_{\mathbf{k}}\approx\hbar k\sqrt{gn_0/m}$，$c=\lim\varepsilon_k/\hbar k$；$g=0$ 时谱是纯抛物线 $\hbar^2k^2/2m$，$\varepsilon_k/k\to0$