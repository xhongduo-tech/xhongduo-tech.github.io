---
title: 量子Boltzmann方程与输运
date: 2026-08-07
---

# 量子Boltzmann方程与输运

<div class="epigraph">
<p>当外场把体系推出平衡，粒子的分布函数开始流动：玻尔兹曼方程就是这条「分布函数的河流」的运动学方程——它同时受外场的驱动与散射的制衡。</p>
<footer>—— L. Boltzmann（1872 经典方程）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 8 ｜ 2026-08-07</p>
</div>

## 为什么需要输运理论

前几篇都在**平衡态**或**线性响应**的框架内：加一个「试探场」，算一个「响应系数」。但真实的器件里，体系常常处于**真正的非平衡态**——电流流过导线、热量沿温度梯度传导、载流子在半导体里被电场驱动。非平衡状态下，「分布函数」$f(\mathbf{k},t)$（动量为 $\mathbf{k}$ 的粒子占据率）不再等于平衡的费米函数，而是随时间与空间演化。

**玻尔兹曼方程（Boltzmann equation）**是描述分布函数演化的运动方程：它说分布函数的时空变化，等于外场驱动的「漂移」与散射事件的「碰撞项」之和。它是输运理论的经典主干——电导、热导、Hall 系数、Seebeck 系数都能从它导出。**量子玻尔兹曼方程**则进一步把碰撞项用 Green 函数语言写出，把「散射率」从第一性原理（自能）计算出来，从而把 Boltzmann 与 Kubo 两套框架统一起来。<span class="marginnote">Boltzmann 1872 年提出的 H 定理（碰撞项使熵单调增加）第一次给了「不可逆」一个微观基础。今天它在半导体器件、等离子体、稀薄气体、凝聚态输运里无处不在，是「从微观到宏观」最成功的单一方程。</span>

## 1 经典玻尔兹曼方程

设分布函数 $f(\mathbf{k},\mathbf{r},t)$，外场 $\mathbf{E}$ 与 $\mathbf{B}$，玻尔兹曼方程：

$$\frac{\partial f}{\partial t} + \mathbf{v}_{\mathbf{k}}\cdot\nabla_{\mathbf{r}} f + \Big(\frac{e\mathbf{E}}{\hbar} + \frac{e}{\hbar}\mathbf{v}_{\mathbf{k}}\times\mathbf{B}\Big)\cdot\nabla_{\mathbf{k}} f = \Big(\frac{\partial f}{\partial t}\Big)_{\text{coll}}$$

**重点：方程左边是「漂移项」——分布函数沿相空间流动；右边是「碰撞项」——散射改变分布。** 漂移项是确定性的（来自牛顿/薛定谔演化），碰撞项是随机性的（来自散射事件）。玻尔兹曼方程的实质是：**相空间里的粒子数守恒 + 碰撞作为源与汇**。这个方程本身不依赖任何量子细节，它只要求分布函数是一个「准经典」的缓慢变化的量。<span class="marginnote">玻尔兹曼方程的适用条件：外场变化缓慢（$q\ll k_F$）、碰撞时间尺度远短于外场周期。当体系进入量子相干区（$\hbar/\tau \sim k_BT$，如介观体系），玻尔兹曼方程失效，需要量子动力学方程——这正是本主题后半部分要讨论的。</span>

## 2 碰撞项：散射如何改变分布

碰撞项的具体形式由散射过程决定。对电子-声子散射，碰撞项为「离去率 − 到达率」：

$$\Big(\frac{\partial f_{\mathbf{k}}}{\partial t}\Big)_{\text{coll}} = \sum_{\mathbf{k}'\lambda}\Big[W_{\mathbf{k}',\mathbf{k}} f_{\mathbf{k}'}(1-f_{\mathbf{k}}) - W_{\mathbf{k},\mathbf{k}'} f_{\mathbf{k}}(1-f_{\mathbf{k}'})\Big]$$

其中跃迁率 $W_{\mathbf{k},\mathbf{k}'}$ 由费米黄金规则给出，含能量守恒的 δ 函数与矩阵元：

$$W_{\mathbf{k},\mathbf{k}'} = \frac{2\pi}{\hbar}|g_{\mathbf{q}}|^2\big[N_\lambda\delta(\varepsilon_{\mathbf{k}}-\varepsilon_{\mathbf{k}'}-\hbar\omega_\lambda) + (N_\lambda+1)\delta(\varepsilon_{\mathbf{k}}-\varepsilon_{\mathbf{k}'}+\hbar\omega_\lambda)\big]$$

**重点：碰撞项自动包含泡利阻塞（$1-f$）与声子占据（$N_\lambda$）。** 第一项是「吸收声子」，第二项是「发射声子」；两者之差保证了碰撞项在平衡分布处精确为零——这是细致平衡（detailed balance）的体现：平衡时离去率 = 到达率。任何「碰撞项使 $f$ 趋于平衡费米函数」的性质，都是细致平衡的推论。<span class="marginnote">碰撞项的「$(1-f)$ 因子」是泡利不相容在输运里的体现：一个粒子只能散射进<strong>空的</strong>末态。这个因子在经典 Boltzmann 方程里不存在（经典粒子没有排他性），是「量子玻尔兹曼方程」里第一个量子修正。它保证了分布函数永远 $\le 1$。</span>

## 3 弛豫时间近似与 Drude 公式

碰撞项一般是非线性积分方程，求解繁琐。**弛豫时间近似（relaxation time approximation）**把碰撞项线性化为：

$$\Big(\frac{\partial f}{\partial t}\Big)_{\text{coll}} \approx -\frac{f - f_0}{\tau}$$

其中 $f_0$ 是局域平衡分布，$\tau$ 是弛豫时间。稳态下（$\partial_t f=0$、均匀体系），线性化玻尔兹曼方程给出：

$$-\frac{e\mathbf{E}}{\hbar}\cdot\nabla_{\mathbf{k}} f_0 = -\frac{f-f_0}{\tau} \quad\Rightarrow\quad f = f_0 + e\tau\,\mathbf{v}_{\mathbf{k}}\cdot\mathbf{E}\,\Big(-\frac{\partial f_0}{\partial\varepsilon}\Big)$$

电流密度 $\mathbf{j} = -e\sum_{\mathbf{k}}\mathbf{v}_{\mathbf{k}}f$ 代入后得到 **Drude 电导率**：

$$\sigma = \frac{ne^2\tau}{m^*}$$

**重点：弛豫时间近似把「碰撞项」压缩成一个数 $\tau$，把电导率压缩成 Drude 公式。** 这个公式简洁得惊人：电导 = 密度 × 电荷² × 弛豫时间 / 有效质量。它的物理是「漂移与散射的平衡」：外场加速电子，碰撞又使它回到平衡，稳态电流由两者之比决定。这里 $\tau$ 不再是唯象参数——它可以通过黄金规则从微观散射率算出来（$\tau^{-1} = \sum W_{\mathbf{k},\mathbf{k}'}$）。<span class="marginnote">Drude 公式与 Kubo 公式的等价：Kubo 公式 $\sigma=(i/\omega)\chi_{jj}^R$ 在 $\omega\to0$ 且谱函数有 Lorentzian 展宽 $\Gamma=\hbar/\tau$ 时，精确回到 $\sigma=ne^2\tau/m$。两套框架殊途同归——Boltzmann 方程从「分布函数」出发，Kubo 公式从「关联函数」出发，在弛豫时间近似下会合。</span>

## 4 量子玻尔兹曼方程与 Kadanoff-Baym

当体系进入量子相干区，或需要从第一性原理计算碰撞项时，需要**量子玻尔兹曼方程（quantum Boltzmann equation）**。它的出发点不再是经典的 $f(\mathbf{k})$，而是**Wigner 函数**——Green 函数在动量与时间差上的变换：

$$f(\mathbf{k},\mathbf{r},t) = -i\int d^3r'\, e^{-i\mathbf{k}\cdot\mathbf{r}'}\,\langle c^\dagger(\mathbf{r}-\tfrac{\mathbf{r}'}{2},t)\, c(\mathbf{r}+\tfrac{\mathbf{r}'}{2},t)\rangle$$

**重点：Wigner 函数把 Green 函数「翻译」回准经典分布函数——它是量子力学的相空间分布。** 对它的动力学方程（Kadanoff-Baym 方程，含记忆效应的积分微分方程），在「梯度展开 + 准粒子近似」下退化为量子玻尔兹曼方程，其碰撞项中的散射率用 Green 函数自能表达：$\tau^{-1} = 2\,\text{Im}\,\Sigma(\mathbf{k},\varepsilon_{\mathbf{k}})$。<span class="marginnote">Kadanoff-Baym 方程（1962）是非平衡 Green 函数理论（NEGF）的核心：两个耦合方程（用于 Green 函数 $G^\lt $ 与 $G^>$）含自能积分的记忆效应。它在介观器件、量子输运与超快动力学里是标准工具，也是「从平衡 Green 函数到非平衡动力学」的桥梁。</span>

**辨析｜易错点：** 初学者常以为「玻尔兹曼方程 = 经典，Kubo 公式 = 量子」，两者择一。实际上它们是同一物理的两个视角：**玻尔兹曼方程是分布函数的运动方程（拉格朗日视角），Kubo 公式是响应函数的关联函数（欧拉/整体视角）**。在「弛豫时间近似 + 弱散射」下两者严格等价；超出这个范围（强散射、量子相干）才分道扬镳。输运理论的功力在于知道何时用哪个。

## 5 公式解析：从散射率到电导率

把整套链条走通：微观散射 → 弛豫时间 → 电导率。

**第一步，算散射率**：$\tau^{-1}(\mathbf{k}) = \frac{2\pi}{\hbar}\sum_{\mathbf{k}'\lambda}|g_{\mathbf{q}}|^2 (N_\lambda+1-n_F)\delta(\varepsilon_{\mathbf{k}}-\varepsilon_{\mathbf{k}'}-\hbar\omega_\lambda)$。对弹性散射（声子能量可忽略），角积分给出 $\tau^{-1} \propto k_BT$（高温极限：声子数 $N_\lambda\approx k_BT/\hbar\omega$）。
**第二步，代进 Drude 公式**：$\sigma = ne^2\tau/m^*$。得到 $\sigma \propto 1/T$——**这正是金属电阻率随温度线性上升的实验事实**（$\rho \propto T$）。
- **第三步，读出温度依赖的物理**：声子越多（温度越高），散射越强，弛豫时间越短，电阻越大。低温极限下声子冻结（$N_\lambda\to0$），只剩剩余电阻（杂质散射），电阻趋于常数——Bloch-Grüneisen 曲线的低温平台。
- **第四步，推广**：加上磁场得到 Hall 电导（Hall 系数 $R_H = -1/ne$），加上温度梯度得到 Seebeck 系数——一套框架输出全部输运系数。

**重点：一条「散射率 → τ → σ」的流水线，把微观相互作用翻译成宏观输运。** 玻尔兹曼方程的价值在于它提供了这条流水线的骨架，而 Green 函数（自能、Wigner 函数）提供了第一性原理的输入——两者结合，是凝聚态输运理论的标准工作流。

## 6 输运与「从极限到大模型」

玻尔兹曼方程是「从极限到大模型」里「多尺度耦合」的经典范例：**微观散射（单次碰撞）→ 介观分布（玻尔兹曼方程）→ 宏观输运（Drude/Hall 系数）**。每一层都被「有效化」：单次散射的细节被吸收进弛豫时间 $\tau$，而 $\tau$ 又决定宏观电导。<span class="marginnote">这套「尺度之间的有效压缩」正是迁移学习与预训练-微调的逻辑：预训练把通用知识压缩进参数（相当于算好了「有效散射率」），微调只需在目标任务上重新求解「分布函数」。物理里的「粗粒化」与机器学习里的「表征压缩」分享同一套「先在大量样本上求平均，再在小尺度上做修正」的范式。</span>

对多体理论自身，输运理论是通往量子霍尔效应、拓扑绝缘体、超导输运的必经之路——下一节，我们将回到「平衡态但强关联」的领域：**费米液体理论**，看看相互作用如何把自由电子气变成一群「带外套的准粒子」。

## 7 小结

- **玻尔兹曼方程**：分布函数的漂移（外场驱动）与碰撞（散射）的平衡，是输运理论的经典主干。
- **碰撞项**由离去率 − 到达率构成，自动含泡利阻塞（$1-f$）与细致平衡；平衡分布是其不动点。
- **弛豫时间近似**把碰撞项压缩成 $\tau$，稳态解给出 **Drude 公式** $\sigma = ne^2\tau/m^*$。
- **量子玻尔兹曼方程**从 Wigner 函数出发，散射率由自能计算（$\tau^{-1}=2\text{Im}\,\Sigma$），统一了 Boltzmann 与 Kubo 两套框架。
- Boltzmann（分布函数视角）与 Kubo（关联函数视角）在弛豫时间近似下等价，超出则分道扬镳。
- 散射率 → 弛豫时间 → 电导率的流水线，解释了金属电阻率 $\rho\propto T$ 的实验规律。

在下一节，我们把相互作用对单粒子的影响提升到「理论」的高度：**费米液体理论**——Landau 如何用一组「准粒子参数」描述相互作用电子气，让自由电子气的一整套物理在强相互作用下依然存活。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| 玻尔兹曼方程 | $\partial_t f + \mathbf{v}\cdot\nabla_r f + \frac{e\mathbf{E}}{\hbar}\cdot\nabla_k f = (\partial_t f)_{\text{coll}}$ | 漂移 + 碰撞的分布函数守恒 |
| 碰撞项 | $\sum_{k'}[W_{k'k}f_{k'}(1-f_k) - W_{kk'}f_k(1-f_{k'})]$ | 离去率 − 到达率，泡利阻塞内建 |
| 弛豫时间近似 | $(\partial_t f)_{\text{coll}} = -(f-f_0)/\tau$ | 碰撞项线性化，$\tau$ 唯象 |
| Drude 电导 | $\sigma = ne^2\tau/m^*$ | 漂移与散射的稳态平衡 |
| 散射率 | $\tau^{-1} = 2\,\text{Im}\,\Sigma(\mathbf{k},\varepsilon_{\mathbf{k}})$ | 量子玻尔兹曼：从自能算散射 |

**易错复盘**：两点要分清。其一，Boltzmann 方程与 Kubo 公式不是「经典 vs 量子」的取舍——它们是分布函数视角与关联函数视角，在弛豫时间近似下严格等价；其二，碰撞项的 $1-f$ 因子（泡利阻塞）是量子修正的核心——经典玻尔兹曼没有它，分布函数可能超过 1。

**知识连线**：本篇把第 1 篇的 Green 函数（自能给出散射率）与第 2 篇的 Kubo 公式统一到输运框架；Drude 公式与第 2 篇费米液体的准粒子输运直接相连。「微观散射 → 弛豫时间 → 宏观电导」的层级压缩，是「从极限到大模型」里「尺度间有效化」的经典范例。

## 8 数值算例：铜的输运参数

把公式放进真实金属。室温铜：载流子密度 $n=8.5\times10^{28}\ \mathrm{m^{-3}}$、有效质量 $m^*\approx m_e$、电导率 $\sigma\approx6\times10^7\ \Omega^{-1}\mathrm{m^{-1}}$。由 $\sigma=ne^2\tau/m^*$ 反解弛豫时间：

$$
\tau = \frac{\sigma m^*}{ne^2} \approx \frac{(6\times10^7)(9.1\times10^{-31})}{(8.5\times10^{28})(1.6\times10^{-19})^2}\ \mathrm{s} \approx 2.5\times10^{-14}\ \mathrm{s}
$$

对应平均自由程 $\ell=v_F\tau$：取费米速度 $v_F\approx1.6\times10^6\ \mathrm{m/s}$，得 $\ell\approx40\ \mathrm{nm}$——约 150 个晶格常数。室温下电子在两次散射之间只「自由」走这么远，却足以支撑巨大的宏观电流。

**重点：Drude 公式的威力在于把不可见的微观量（$\tau$、$\ell$）从可测的宏观量（$\sigma$）反推出来。** 电阻率 $\rho=\sigma^{-1}\propto T$ 的线性温度依赖对应 $\tau\propto1/T$；温度降到液氦区后声子冻结，$\tau$ 由杂质散射决定，电阻趋于有限的剩余值——这正是 Bloch-Grüneisen 曲线。

| 金属 | 室温 $\sigma$（$\Omega^{-1}\mathrm{m}^{-1}$） | $\tau$（$\times10^{-14}$ s） | $\ell$（nm） |
| --- | --- | --- | --- |
| 银 | $6.3\times10^7$ | $4.0$ | 57 |
| 铜 | $6.0\times10^7$ | $2.5$ | 40 |
| 铝 | $3.8\times10^7$ | $0.8$ | 15 |

**读数要领**：银的 $\tau$ 最长、$\ell$ 最大，所以电导率最高；铝最差。这张表演示了「弛豫时间近似」的实用价值——不求解积分方程，只用一个 $\tau$ 就抓住了输运的主要差别。低温下 $\ell$ 可增至微米量级，这就是为什么纯度高的金属在低温有极低的电阻。

**实践与辨析**：金属电阻率为什么 $\propto T$？提示：高温下声子数 $N_\lambda\approx k_BT/\hbar\omega$，散射率 $\propto N_\lambda\propto T$，电导 $\propto1/T$；低温只剩杂质散射，电阻趋于常数。