---
title: 张量微积分：协变导数与张量的散度、旋度
date: 2026-08-11
---

# 张量微积分：协变导数与张量的散度、旋度

<div class="epigraph">
<p>黎曼的曲率，是我一生最痛苦也最甜美的领悟。</p>
<footer>—— 埃尔温 · 爱因斯坦（Albert Einstein）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础科学 · 向量与张量初步 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么普通求导在曲线坐标下会「叛变」

本专题至今把张量当作「一个对象的全体分量」。现在要问：**张量随位置变化时，怎么求导才算合法？** 在直角坐标下答案平淡无奇——分量对坐标求偏导就行。但一旦进入曲线坐标或弯曲空间，普通偏导 $\partial V^i/\partial x^j$ **不再是张量**：它的变换律里混进额外的项，因为它只比较了「数值的变化」，没有计入**基向量本身在空间里怎么扭动**。<span class="marginnote">本讲对应 Arfken 第 2 章末节关于协变导数的内容。直觉：一辆车沿直线行驶，惯性导航仪测出「速度向量不变」——但这个向量在球面地图上的分量却在变，因为基向量（经纬方向）自身在转。普通偏导把「分量变」误判为「向量变」。</span>

本讲的补救方案叫**协变导数（covariant derivative）**：在偏导基础上加上「基向量扭转」的修正项，让导数的结果重新成为张量。这是广义相对论、微分几何的入口，也是本专题「从极限到大模型」的曲线坐标段真正的顶峰。

## 1 基向量的变化率：Christoffel 记号

曲线坐标里基向量 $\mathbf e_i$ 在空间各点方向不同。基向量的导数展开到基上，系数就是 **Christoffel 记号（Christoffel symbol）** $\Gamma^k_{\ ij}$：

$$
\frac{\partial \mathbf e_i}{\partial x^j} = \Gamma^k_{\ ij}\,\mathbf e_k
$$

$\Gamma^k_{\ ij}$ 度量「沿 $x^j$ 方向走，基向量 $\mathbf e_i$ 朝 $\mathbf e_k$ 方向偏了多少」。它有两条著名性质：

- **下指标对称**：$\Gamma^k_{\ ij} = \Gamma^k_{\ ji}$（可微坐标函数的混合偏导可交换）。
- **正交曲线坐标下可直接从度量系数算**：$\Gamma^k_{\ ij}$ 由 $g_{ij}$ 及其一阶导决定，不用靠「猜」。<span class="marginnote">在直角坐标（$g=\delta$）下 Christoffel 记号全为零——所以初学微积分从没见过它。球坐标里它处处非零，正对应「经纬方向向量在球面上转」的事实。</span>

## 2 协变导数：给偏导戴上「纠偏项」

**协变导数（covariant derivative）** 是把「分量变 + 基变」合并成「向量变」的正确工具：

$$
\nabla_j V^i = \frac{\partial V^i}{\partial x^j} + \Gamma^i_{\ jk} V^k \quad(\text{逆变向量})
$$

$$
\nabla_j V_i = \frac{\partial V_i}{\partial x^j} - \Gamma^k_{\ ji} V_k \quad(\text{协变向量})
$$

符号规则对称而优美：**上标加 $\Gamma$，下标减 $\Gamma$，求和指标与 $V$ 的指标配对**。为什么一加一减？逆变分量与基「反着走」——基扭转的修正项方向相反；协变分量与基「同向走」——修正项方向相同。这就是上一讲跷跷板原理的微分版。<span class="marginnote">推广规则：一般张量 $T^{i_1\cdots}_{j_1\cdots}$ 的协变导数，对每个上标加一项 $\Gamma$、每个下标减一项 $\Gamma$。这是「指标机械学」最光辉的一步——加号减号完全由指标位置决定。</span>

**核对两个特例**：

- 标量 $\phi$ 的协变导数 $\nabla_i\phi = \partial_i\phi$——普通梯度，本来就正确。
- 直角坐标下 $\Gamma = 0$，协变导数退化为普通偏导——本专题《向量分析》的全部公式自动复原。

## 3 张量的散度与旋度：协变导数的缩并

有了协变导数，张量的散度、旋度就是**协变导数 + 缩并**：

**向量场的散度**（逆变向量缩并）：

$$
\nabla\cdot\mathbf V = \nabla_i V^i = \frac{\partial V^i}{\partial x^i} + \Gamma^i_{\ ik} V^k = \frac{1}{\sqrt{g}}\frac{\partial}{\partial x^i}\left(\sqrt{g}\,V^i\right)
$$

右边最后一步是一个**万金油公式**：$\sqrt{g}$ 是体积元比例（$\sqrt{g}=h_1h_2h_3$），它把「基扭转」全部吸收进一个因子。代入柱、球坐标的 $\sqrt{g}$，立刻还原出《正交曲线坐标》一讲的散度公式——两套语言在此对账成功。

**张量场的散度**（对第二个指标缩并）：

$$
(\nabla\cdot T)_j = \nabla_i T^{\,i}_{\ j}
$$

应力张量的散度 $\nabla\cdot\boldsymbol{\sigma}$ 出现在连续介质力学的运动方程里——下一讲的直接入口。<span class="marginnote">张量的散度 = 应力平衡方程：弹性体每一点的应力张量散度等于该点的惯性力密度。协变导数把「基扭转」修正后，平衡方程才能在任意坐标系下成立。</span>

**旋度的张量形式**：旋度本质是反对称导数，用 $\varepsilon$ 写为

$$
(\nabla\times\mathbf V)^i = \frac{\varepsilon^{ijk}}{\sqrt{g}}\,\frac{\partial V_k}{\partial x^j}
$$

分母里的 $\sqrt{g}$ 正是赝张量血统的体现——旋度是赝向量，需要 $\varepsilon$ 与 $\sqrt{g}$ 配合才能给出正确的向量。

## 4 公式解析：为什么协变导数要「一加一减」

$$

\nabla_j V^i = \frac{\partial V^i}{\partial x^j} + \Gamma^i_{\ jk} V^k

$$

四步拆解这条最重要的张量微积分公式：

- **第一步，分解向量的变化**：沿 $x^j$ 移动，$V^i\mathbf e_i$ 的变化有两部分——分量 $V^i$ 自身的增减（$\partial V^i/\partial x^j$），以及基向量 $\mathbf e_i$ 方向的扭转（来自 $\Gamma$）。
- **第二步，把扭转换算成分量语言**：$\Gamma^i_{\ jk}V^k$ 展开为「基向量每转一度，$V$ 里有多少分量随之偏转」。$\Gamma$ 的下标 $jk$ 说「沿 $x^j$ 走、动的是第 $k$ 个分量」，上标 $i$ 说「偏转落到第 $i$ 个基方向」。
- **第三步，为什么是加号**：$V^i$ 是逆变分量。坐标偏导与基扭转**同向**时，逆变分量的读数要「抵掉」基的转动——两个效应相加才能还原真实的向量变化率。协变分量的公式则因为跷跷板反号而用减号。
- **第四步，检验**：直角坐标 $\Gamma=0$，公式变回普通偏导，与前几讲所有结果自洽；曲线坐标下则新增修正项，保证 $\nabla_jV^i$ 是张量。

一句话：**协变导数 = 普通偏导 + 基向量扭转的补偿**，补偿的方向由指标上下位置自动决定。

## 5 实例：极坐标下牛顿定律如何自动出现

用平面极坐标 $(r,\varphi)$ 把协变导数跑一遍，看看它是否真的「纠偏」。度量 $g_{rr}=1,\ g_{\varphi\varphi}=r^2$，由 $g$ 算出的非零 Christoffel 记号只有三个：

$$
\Gamma^r_{\ \varphi\varphi} = -r, \qquad \Gamma^{\varphi}_{\ r\varphi} = \Gamma^{\varphi}_{\ \varphi r} = \frac{1}{r}
$$

代入逆变向量的协变导数公式 $\nabla_j V^i = \partial_j V^i + \Gamma^i_{\ jk}V^k$，对速度 $\mathbf v = (v^r, v^\varphi)$ 沿自身求导（即加速度）：

$$
a^r = \dot v^r - r\,(v^\varphi)^2, \qquad
a^{\varphi} = \dot v^{\varphi} + \frac{1}{r}(v^r v^{\varphi} + v^{\varphi} v^r) = \dot v^{\varphi} + \frac{2}{r}v^r v^{\varphi}
$$

认出那些修正项了吗？$r(v^\varphi)^2 = \dot r\,\dot\varphi^2 r$ 就是**向心/离心加速度项**，$\dfrac{2}{r}v^r v^\varphi$ 是**科里奥利项**。<span class="marginnote">牛顿定律 $\mathbf F = m\mathbf a$ 里的 $\mathbf a$ 在极坐标里从来不是 $(\ddot r, \ddot\varphi)$ 那么干净——惯常的「离心力」「科里奥利力」全部来自 $\Gamma$ 修正项。协变导数让 $\mathbf F = m\nabla_{\mathbf v}\mathbf v$ 在任意坐标系下写出来都一样。</span>

**这就是协变导数的意义**：它不是给「懂的人」加的繁琐装饰，而是让牛顿定律在极坐标、球坐标、乃至弯曲时空中**保持同一种形式**的必要工具。广义相对论的测地线方程

$$
\frac{d^2 x^i}{d\tau^2} + \Gamma^i_{\ jk}\frac{dx^j}{d\tau}\frac{dx^k}{d\tau} = 0
$$

就是「自由粒子沿协变导数意义下的直线运动」——引力被编码进 $\Gamma$，而不是被当成「力」。四维时空里的自由落体，其实是在弯曲时空中走「直」线。

## 6 小结

- **问题**：曲线坐标/弯曲空间里，基向量自身在变，普通偏导不再是张量。
- **Christoffel 记号** $\Gamma^k_{\ ij}$：基向量的导数系数，下指标对称，直角坐标下为零。
- **协变导数**：$\nabla_jV^i = \partial_jV^i + \Gamma^i_{\ jk}V^k$，$\nabla_jV_i = \partial_jV_i - \Gamma^k_{\ ji}V_k$；上标加、下标减。
- **散度万金油**：$\nabla\cdot\mathbf V = \frac{1}{\sqrt{g}}\partial_i(\sqrt{g}V^i)$，张量散度 $=\nabla_i T^{i}_{\ j}$；旋度带 $\varepsilon$ 与 $\sqrt{g}$，赝向量血统昭然。
- **体系对账**：直角坐标下协变导数退化回普通偏导，正交曲线坐标公式全部自动还原。

在下一节，我们把全部张量语言投向一个真实世界的问题——固体在外力下如何变形、如何应力平衡。**张量在连续介质力学中的应用**，将是本专题的收官之战。
