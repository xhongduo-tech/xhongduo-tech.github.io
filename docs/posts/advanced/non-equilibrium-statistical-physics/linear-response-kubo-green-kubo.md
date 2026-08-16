---
title: 线性响应理论（久保公式、Green-Kubo 关系）
date: 2026-08-07
---

# 线性响应理论（久保公式、Green-Kubo 关系）

<div class="epigraph">
<p>科学无法解开自然的终极奥秘，因为归根结底，我们自己也是这自然的一部分，因而也是我们试图求解的奥秘的一部分。</p>
<footer>—— 马克斯 · 普朗克（Max Planck），1930 年</footer>
</div>

<div class="article-byline">
<p>第四级 · 非平衡统计物理 ｜ Zwanzig《Nonequilibrium Statistical Mechanics》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从线性响应理论继续

前面各讲建立了随机过程的微观模型，但还有一个根本问题没回答：**一个系统对外加微扰（电场、温度梯度、力）的宏观响应，能否只从平衡态的信息算出来？** 线性响应理论给出肯定的答案——系统在平衡态附近的时间关联函数，决定了它对弱外场的全部线性响应。

这套思想由 1957 年日本物理学家久保亮五（Ryogo Kubo）系统化，其核心是**久保公式**；而把久保公式与输运系数相连、把输运系数写成关联函数时间积分的形式，则是 **Green-Kubo 关系**（格林 1952、久保 1957）。它统一了此前各讲的散落结果：爱因斯坦关系、Kramers 公式、昂萨格关系，全都能被线性响应理论的框架收纳。<span class="marginnote">线性响应理论是这个专题的分水岭：此前的工具（朗之万、福克-普朗克）适合描述「系统自身的涨落」，线性响应理论则回答「系统如何被外场驱动」。两者的桥正是涨落-耗散定理——涨落与响应由同一批关联函数决定。</span>

## 1 线性响应的基本设定

考虑一个在平衡态附近被外场扰动的系统。外场耦合到可观测量 $B$，哈密顿量写为 $H = H_0 - F(t)B$，其中 $F(t)$ 是外场强度。我们要问：另一可观测量 $A$ 的期望值偏离平衡态多少？

**线性响应理论的核心结果（久保公式）**：

$$
\langle \Delta A(t) \rangle = \int_{-\infty}^{t} \phi_{AB}(t - t')\,F(t')\,dt'
$$

其中 $\phi_{AB}(\tau)$ 是**响应函数**（也叫线性响应核），它完全由平衡态的时间关联函数决定。对量子系统，响应函数写为：

$$
\phi_{AB}(\tau) = \frac{i}{\hbar}\big\langle [A(\tau), B(0)]\big\rangle_{eq}
$$

**重点：只要知道平衡态下对易子 $\langle [A(\tau),B(0)]\rangle$，就能预测任意弱外场下的响应。** 这是「平衡态信息决定非平衡响应」的最强表述。<span class="marginnote">久保 1957 年的三篇连载论文《Statistical-Mechanical Theory of Irreversible Processes》奠定了这一整套框架。它对标教材 Zwanzig 第7章给出了相对论式的简洁推导：用线性化刘维尔方程 + 对易子代数，两页纸就能得到久保公式。</span>

## 2 输运系数 = 关联函数的积分

线性响应理论在输运问题上的应用产生 Green-Kubo 关系：**输运系数等于相应流的时间关联函数从零到无穷的积分。**

以电导率为例。对电流算符 $J$，线性响应给出 $\langle J(t)\rangle = \sigma F$，其中：

$$
\sigma = \frac{1}{k_B T}\int_0^\infty \big\langle J(0) J(t)\big\rangle_{eq}\,dt
$$

类比地，扩散系数、粘滞系数、热导率都写成同样结构：

$$
D = \frac{1}{n}\int_0^\infty \langle J_x(0)J_x(t)\rangle\,dt,\qquad
\eta = \frac{1}{k_B T V}\int_0^\infty \langle P_{xy}(0)P_{xy}(t)\rangle\,dt
$$

**Green-Kubo 关系把「平衡态的关联函数」与「非平衡的输运系数」划上了等号。**<span class="marginnote">Green-Kubo 关系的实用价值在于分子模拟：平衡分子动力学轨迹里统计流的时间关联函数，积分即得输运系数——这是当今材料模拟中计算扩散、粘滞、热导的标准做法，完全不需要人为施加梯度。</span>

## 3 频率依赖与色散

线性响应理论不仅给出输运系数的零频极限，还给出完整频率依赖。对频率为 $\omega$ 的周期外场，响应函数给出**复导纳（admittance）**：

$$
\chi(\omega) = \int_0^\infty \phi_{AB}(t)\,e^{i\omega t}\,dt = \chi'(\omega) + i\chi''(\omega)
$$

- **实部 $\chi'(\omega)$**：色散——外场与响应同相的分量，存储能量。
- **虚部 $\chi''(\omega)$**：耗散——外场与响应正交的分量，吸收能量。

**重点：实部与虚部由同一个函数 $\phi_{AB}(t)$ 决定，因此不是独立的。** 这一约束在频域表现为 **Kramers-Kronig 关系**：$\chi'(\omega)$ 是 $\chi''(\omega')$ 的整个频域积分变换。任何满足因果性的线性响应系统都自动满足它。<span class="marginnote">Kramers-Kronig 关系是「因果性 ⇔ 解析性」在物理中最漂亮的实例：因为响应只能发生在微扰之后（因果性），$\chi(\omega)$ 作为 $\omega$ 的解析函数在上半平面正则，解析性自动给出实部与虚部的互推。光学折射率与吸收系数的关系就是它的经典应用。</span>

## 4 公式解析：从线性化方程到久保公式

久保公式的推导值得拆成四步，它展示了「把可逆的微观动力学变成不可逆的宏观响应」的关键机制：

$$
\rho(t) = \rho_{eq} + \delta\rho(t)
$$

- **第一步**：写出密度矩阵的刘维尔方程 $\partial\rho/\partial t = -(i/\hbar)[H_0 - F(t)B, \rho]$，把 $\rho$ 拆成平衡部分 $\rho_{eq}\propto e^{-\beta H_0}$ 与小扰动 $\delta\rho$。
- **第二步**：线性化——忽略 $\delta\rho$ 与外场的二阶乘积，得到 $\partial\delta\rho/\partial t = -(i/\hbar)[H_0,\delta\rho] + (i/\hbar)F(t)[B,\rho_{eq}]$。这是非齐次线性方程，源项是外场。
- **第三步**：用相互作用图像形式解，$\delta\rho(t) = \frac{i}{\hbar}\int_{-\infty}^t F(t')\,e^{-iH_0(t-t')/\hbar}[B,\rho_{eq}]e^{iH_0(t-t')/\hbar}\,dt'$。
- **第四步**：对任意可观测量 $A$ 取期望值 $\langle A\rangle = \mathrm{Tr}(A\,\delta\rho)$，对 $\rho_{eq}$ 用迹运算化简 $\mathrm{Tr}(A[B,\rho_{eq}]) = -\mathrm{Tr}([A,B]\rho_{eq})$，最终得到 $\langle\Delta A(t)\rangle = \int \frac{i}{\hbar}\langle[A(t-t'),B(0)]\rangle F(t')dt'$——正是久保公式。

**重点：推导只用了线性化 + 刘维尔方程的可逆性，没有引入任何不可逆假设。** 不可逆性（响应函数的因果结构）来自对时间积分从 $-\infty$ 到 $t$ 的截断——这是「把初始时刻推向无穷远、让记忆被抹平」的物理选择。

## 5 线性响应与之前各讲的统一

线性响应理论是第 4 篇乃至整个专题的会合点：

**爱因斯坦关系**（第 31 讲）：扩散系数 $D$ 的 Green-Kubo 表达，是朗之万理论中 $\langle \xi\xi\rangle = 2\gamma m k_BT$ 的翻版——两者是同一涨落耗散结构的两面。

**昂萨格互易关系**（第 24 讲）：久保公式里的响应函数满足 $\phi_{AB} = \phi_{BA}$（对时反演不变的平衡态），这直接推导出昂萨格互易关系 $L_{ij} = L_{ji}$——线性响应理论给了昂萨格关系一个第一性原理的证明。

**涨落-耗散定理**（第 32 讲）：涨落谱 $S_{AA}(\omega)$ 与耗散 $\chi''(\omega)$ 由 $S_{AA}(\omega) = \frac{2k_B T}{\omega}\chi''(\omega)$ 相连——这就是下一讲的主题。

## 6 例：Drude 模型是线性响应的特例

久保公式听起来抽象，用最古老的输运模型——德鲁德模型（Drude, 1900）——可以把它落到地面上。

德鲁德模型假设电子以弛豫时间 $\tau$ 与晶格碰撞，动力学方程为 $m\dot{v} = -eE - mv/\tau$。稳态 $v = -e\tau E/m$，电流密度 $J = -nev = ne^2\tau E/m$，得电导率：

$$
\sigma_{Drude} = \frac{ne^2\tau}{m}
$$

现在从线性响应的角度看：这正是 Green-Kubo 公式在「电流-电流关联函数以 $\tau$ 指数衰减」假设下的精确结果。若取 $\langle J(0)J(t)\rangle = \langle J^2\rangle e^{-t/\tau}$，则：

$$
\sigma = \frac{1}{k_BT}\int_0^\infty \langle J(0)J(t)\rangle\,dt = \frac{ne^2\tau}{m}
$$

**德鲁德模型只是线性响应理论的一个特例：它的「碰撞弛豫」假设对应关联函数的指数衰减。** 久保公式的贡献在于——不需要任何碰撞图像，只要关联函数已知，输运系数就被确定。<span class="marginnote">这个例子也暴露了德鲁德模型的边界：它不能解释为何真实金属的电导率随温度升高而下降（$\sigma\propto 1/T$，电子-声子散射）。原因正是关联函数并非简单指数——声子的频谱结构使 $\langle J(0)J(t)\rangle$ 带有多时间尺度的尾巴。线性响应理论把「对模型的猜测」换成了「对关联函数的测量」。</span>

**向量子场论延伸**：当外场频率高、或系统处于强关联区，平衡态对易子的线性响应推广为**非平衡格林函数（Keldysh 形式）**——把响应函数换成含时格林函数，可处理时间依赖驱动与初始非平衡态。这是当代凝聚态物理研究量子输运（第 40 讲的量子输运方向）的标准框架，但其根基正是本讲的久保公式。

## 7 小结

- **久保公式**：$\langle\Delta A(t)\rangle = \int\phi_{AB}(t-t')F(t')dt'$，响应函数由平衡态对易子 $\langle[A(\tau),B(0)]\rangle$ 决定。
- **Green-Kubo 关系**：输运系数 = 流关联函数的零频积分，如 $\sigma = \frac{1}{k_BT}\int_0^\infty\langle J(0)J(t)\rangle dt$。
- **频率依赖**：复导纳的实部（色散）与虚部（耗散）由同一函数决定，满足 Kramers-Kronig 关系。
- **推导只用**线性化 + 刘维尔方程，不可逆性来自初始时刻推向无穷远。
- **统一作用**：爱因斯坦关系、昂萨格互易关系都是线性响应理论的推论。

在下一节，我们聚焦线性响应里最核心的一个不等式：涨落与耗散如何由温度精确相连——涨落-耗散定理与 Nyquist 定理。
