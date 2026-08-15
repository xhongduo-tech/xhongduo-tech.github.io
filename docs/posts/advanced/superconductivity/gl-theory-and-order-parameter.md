---
title: 金兹堡-朗道理论与超导序参量
date: 2026-08-07
---

# 金兹堡-朗道理论与超导序参量

<div class="epigraph">
<p>在临界温度附近，超导体的行为可以用一个复数的序参量来描述——它像波函数，又不完全是一个波函数。</p>
<footer>—— 维塔利 · 金兹堡（Vitaly Ginzburg，1950）</footer>
</div>

<div class="article-byline">
<p>第四级 · 超导物理 ｜ Tinkham《Introduction to Superconductivity》第4章；Ketterson & Song 第9章；张裕恒《超导物理》第4章 ｜ 2026-08-07</p>
</div>

## 为什么从序参量开始

1937 年朗道提出二阶相变的**序参量理论**：每个有序相可以用一个在无序相中为零、有序相中非零的物理量来刻画。1950 年金兹堡与朗道把这一思想用到超导——**超导序参量 $\psi(\boldsymbol{r})$** 是一个复场，其模方 $|\psi|^2$ 正比于超导电子密度 $n_s$，其相位则是宏观量子相位的来源。GL 理论是超导物理的分水岭：它不再满足于「解释现象」，而是写出一个**自由能泛函**，从中一次性导出穿透深度、相干长度、界面能、I/II 类判据与涡旋结构。本篇是理解后半个超导大厦的地基。

## 1 序参量：超导的「波函数」

在 $T_c$ 附近，定义复序参量：

$$
\psi(\boldsymbol{r}) = |\psi(\boldsymbol{r})|\, e^{i\varphi(\boldsymbol{r})}
$$

物理意义：$|\psi(\boldsymbol{r})|^2 = n_s(\boldsymbol{r})$ 是局域超导电子密度；$\varphi$ 是超导相位。$\psi$ 在 $T \ge T_c$ 时处处为零（正常态），在 $T<T_c$ 时非零。<span class="marginnote">GL 序参量长得像量子力学波函数，但金兹堡-朗道强调它是<strong>唯象</strong>的：只有到 1957 年 BCS 理论，人们才明白 $\psi$ 其实是库珀对的质心波函数（对的质心、对内的相对运动被积掉），模方正比于对密度而非单电子密度——这个迟到的对应是 GL 理论「半理论」地位的来源。</span>

序参量的两个自由度（模与相位）携带了两条物理线：**模方**连回磁学（穿透深度由 $|\psi|^2$ 决定），**相位梯度**连回电流（超流速度 $\boldsymbol{v}_s \propto \nabla\varphi$）。这是「相位相干」这一超导本质的第一次严格登场。

## 2 GL 自由能泛函

GL 理论的核心假设：超导体的自由能密度可以写成序参量及其梯度的泛函：

$$
g_s = g_n + \alpha|\psi|^2 + \frac{\beta}{2}|\psi|^4 + \frac{1}{2m^*}\left|\left(-i\hbar\nabla - \frac{e^*}{c}\boldsymbol{A}\right)\psi\right|^2 + \frac{B^2}{8\pi}
$$

其中 $\alpha = \alpha'(T-T_c)$（$T>T_c$ 时为正、$T<T_c$ 为负），$\beta > 0$ 常数，$m^* = 2m$、$e^* = 2e$ 是库珀对的有效质量与电荷。对 $\psi$ 与 $\boldsymbol{A}$ 变分，得到两个**GL 方程**。<span class="marginnote">对 $\psi^*$ 变分给出第一 GL 方程（序参量方程）；对 $\boldsymbol{A}$ 变分给出第二 GL 方程（超流方程，即广义伦敦方程）。两个方程联立才能自洽地解出「序参量空间变化」与「磁场空间分布」的耦合问题。</span>

**第一 GL 方程（序参量方程）**：

$$
\frac{1}{2m^*}\left(-i\hbar\nabla - \frac{e^*}{c}\boldsymbol{A}\right)^2\psi + \alpha\psi + \beta|\psi|^2\psi = 0
$$

**第二 GL 方程（超流方程）**：

$$
\boldsymbol{j}_s = \frac{e^*\hbar}{2m^* i}\left(\psi^*\nabla\psi - \psi\nabla\psi^*\right) - \frac{e^{*2}}{m^*c}|\psi|^2\boldsymbol{A}
$$

**辨析｜易错点：** $\alpha$ 的符号。$T>T_c$ 时 $\alpha>0$，自由能最小在 $\psi=0$（正常态稳定）；$T<T_c$ 时 $\alpha\lt 0$，最小在 $|\psi|^2 = -\alpha/\beta \neq 0$（超导态稳定）。很多人把 $\alpha$ 当成常数，导致无法理解为什么 GL 方程只在 $T<T_c$ 有非零解。

## 3 两个 GL 特征长度

把第一 GL 方程在均匀无场情形下求解，可以定义两个长度：

**GL 相干长度**（序参量恢复的长度）：

$$
\xi_{GL}(T) = \sqrt{\frac{\hbar^2}{2m^*|\alpha|}} = \frac{\xi_{GL}(0)}{\sqrt{1 - T/T_c}}
$$

**GL 穿透深度**（磁场衰减长度，用平衡态 $|\psi_\infty|^2 = -\alpha/\beta$ 代入）：

$$
\lambda_{GL}(T) = \sqrt{\frac{m^*c^2}{4\pi e^{*2}|\psi_\infty|^2}} = \frac{\lambda_{GL}(0)}{\sqrt{1 - T/T_c}}
$$

两个长度都在 $T_c$ 处发散，比值 $\kappa = \lambda_{GL}/\xi_{GL}$ 与温度无关——这就是 **GL 参数**。<span class="marginnote">$\xi_{GL}(T)$ 在 $T_c$ 发散意味着：越接近 $T_c$，序参量的空间变化越平缓，局域近似越成立——这正是 Pippard 非局域效应在 $T_c$ 附近自动失效、GL 局域理论有效的深层原因。</span>

## 4 公式解析：从变分看序参量方程

GL 理论的全部力量来自「变分」二字。把自由能对 $\psi^*$ 变分，逐项看：

$$
\delta F = \int \left[ \frac{1}{2m^*}(-i\hbar\nabla-\frac{e^*}{c}\boldsymbol{A})\psi \cdot (i\hbar\nabla\delta\psi^*) + (\alpha\psi + \beta|\psi|^2\psi)\delta\psi^* \right] d^3r = 0
$$

- **第一步，动能项**：对含 $\nabla\delta\psi^*$ 的项做分部积分，把梯度从 $\delta\psi^*$ 上「搬走」，得到一个对 $\psi$ 作用的动能算子 $(-i\hbar\nabla-e^*\boldsymbol{A}/c)^2/2m^*$，边界项设为零（序参量在远处取平衡值）。
- **第二步，势能项**：$\alpha|\psi|^2 + \beta|\psi|^4/2$ 对 $\psi^*$ 的变分直接给出 $\alpha\psi + \beta|\psi|^2\psi$。
- **第三步，并项**：两项之和为零，即得第一 GL 方程——这是一个**非线性薛定谔方程**，非线性项 $\beta|\psi|^2\psi$ 保证序参量振幅被自洽锁定。
- **第四步，物理解读**：$\alpha\lt 0$ 时，非线性项与动能项竞争：动能项倾向让 $\psi$ 光滑变化，非线性项倾向把 $|\psi|^2$ 钉在 $-\alpha/\beta$。竞争的「空间标尺」就是 $\xi_{GL}$。

## 5 GL 理论的成就与局限

GL 理论的成功清单：给出穿透深度与相干长度的温度依赖、严格定义界面能与 $\kappa$、预演 I/II 类判据、导出磁通量子化与涡旋解（下一篇）、统一了伦敦与 Pippard 的局部图像。

它的局限同样清晰：严格成立的范围限于** $T_c$ 附近的弱场**（序参量缓变、$|\psi|$ 小，展开截断到 $|\psi|^4$ 合理）。远离 $T_c$，或者强耦合体系，GL 泛函需要修正或换成基于微观理论的推广形式（如时间依赖 GL、含涨落的 GL）。<span class="marginnote">尽管「仅近 $T_c$ 严格」，GL 语言（序参量、对称破缺、自由能展开）已成为整个凝聚态物理的通用语法——冷原子超流、超流氦、液晶、甚至粒子物理的希格斯机制都沿用同一套形式。</span>

## 6 GL 理论的现代延伸

GL 理论虽然是 1950 年的「旧理论」，它的思想却在不断延伸，至今仍是活跃工具：

**含时 GL（TDGL）**：加入序参量的弛豫动力学（$\partial_t\psi$），描述非平衡过程——涡旋运动、磁通流动、相滑移都在 TDGL 框架内（见《非平衡超导电性》一篇）。TDGL 是超导「动力学」的标准工作马。

**涨落修正的 GL**：GL 泛函 + 高斯涨落，得到 $T_c$ 之上的涨落电导（Aslamazov-Larkin）与涨落磁化——这是《超导涨落效应》一篇的起点。GL 的「自由能泛函 + 涨落」结构，与统计力学里一切连续相变理论（Landau-Ginzburg-Wilson）完全同构。

**与 BCS 的严格对应（Gor'kov 理论）**：1959 年 Gor'kov 从 BCS 微观理论出发，在 $T_c$ 附近严格推导出 GL 泛函——证明了 GL 是 BCS 的**长波、近 $T_c$ 极限**，并给出了微观参数到 GL 系数的映射：

$$
\alpha = -N(0)\frac{7\zeta(3)}{8\pi^2}\frac{(k_BT_c)^2}{T_c}\left(1 - \frac{T}{T_c}\right), \quad \beta = N(0)\frac{7\zeta(3)}{8\pi^2}\frac{(k_BT_c)^2}{T_c}
$$

**GL 思想的「出圈」**：GL 语言已超越超导本身——超流氦的 Gross-Pitaevskii 方程（冷原子凝聚的序参量方程）、粒子物理的希格斯机制（自发对称破缺 + 序参量）、宇宙学暴胀（标量场驱动）都与 GL 泛函同构。**「一个复数序参量、一个自由能泛函、一组自洽方程」**成为现代物理描述「对称性破缺相」的通用语法。<span class="marginnote">把 GL 的「出圈」落到实处：冷原子实验里，BEC 的凝聚体用 GP 方程描述，涡旋结构与超导涡旋一模一样；宇宙学里，希格斯场绕真空的相位变化（缠绕数）与超导涡旋的相位绕数同属一个数学结构。GL 理论是「从实验室到宇宙」的通用序参量语言——这也是为什么它值得反复咀嚼。</span>

**GL 的适用性警示（再强调）**：GL 严格只适用于 $T_c$ 附近、弱场、缓变序参量。对强耦合、低维、量子临界系统，GL 的朗道展开（截断到 $|\psi|^4$）不再可靠——这正是高温超导研究里「GL 够用吗」的经典争论点。**用 GL 之前先问：序参量缓变吗？涨落小吗？**——这是一切平均场理论的通用纪律。

## 7 小结

- **序参量 $\psi = |\psi|e^{i\varphi}$**：模方 $= n_s$，相位是宏观量子相位；$T \ge T_c$ 时为零。
- **GL 自由能泛函**含 $\alpha|\psi|^2+\beta|\psi|^4/2$ 与动能项；对 $\psi^*$、$\boldsymbol{A}$ 变分得两个 GL 方程。
- $\alpha \propto (T-T_c)$ 控制对称破缺：$T<T_c$ 时 $\alpha\lt 0$，非零解才稳定。
- 两个特征长度 $\xi_{GL}(T)$、$\lambda_{GL}(T)$ 都在 $T_c$ 发散，比值 $\kappa$ 与温度无关。
- GL 方程是非线性薛定谔方程；严格限于 $T_c$ 附近弱场，但语言已普适化。
- **GL 与 BCS 的对应**：Gor'kov 从 BCS 在 $T_c$ 附近严格导出 GL 泛函——GL 是 BCS 的长波近 $T_c$ 极限，微观参数到 GL 系数有明确映射。
- **GL 思想的「出圈」**：冷原子 BEC 的 GP 方程、粒子物理的希格斯机制都与 GL 泛函同构——「复数序参量 + 自由能泛函」是现代物理的通用语法。

在下一节，我们用 GL 理论回答那个决定性问题：$\kappa = \lambda_{GL}/\xi_{GL}$ 的大小如何划分 I 类与 II 类超导体——这就是 **GL 参数与 I 类/II 类超导体划分**。