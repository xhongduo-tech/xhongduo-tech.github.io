---
title: 线性响应理论（Kubo 公式）
date: 2026-08-07
---

# 线性响应理论（Kubo 公式）

<div class="epigraph">
<p>方程对我来说比政治更重要，因为政治只关乎当下，而一个方程关乎永恒。</p>
<footer>—— 阿尔伯特 · 爱因斯坦（Albert Einstein）</footer>
</div>

<div class="article-byline">
<p>第四级 · 非平衡统计物理 ｜ Zwanzig《Nonequilibrium Statistical Mechanics》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从线性响应理论开始

动理学方程（玻尔兹曼、BGK）需要具体的碰撞模型。但有一个问题更基础也更普适：**给定一个外部扰动（电场、温度梯度、机械力），系统的响应如何只用平衡性质表达？** 答案就是**线性响应理论**——它由久保亮五（Ryogo Kubo）在 1957 年系统建立，被称为「非平衡统计力学最伟大的贡献之一」。

线性响应理论的美在于普适性：不依赖任何模型、任何近似（在弱扰动下），响应函数**精确地**等于某个平衡关联函数。这既是 Green-Kubo 公式的理论基础，也是凝聚态物理中电导、磁化率、介电函数等一切「响应系数」的统一框架。

## 1 外场作为哈密顿量的微扰

设无扰动哈密顿量为 $H_0$。施加与可观测量 $A$ 共轭的外场 $F(t)$，哈密顿量变为：

$$
H(t) = H_0 - F(t)\,A
$$

例如：电场 $E(t)$ 与极化算符 $\mathbf{P}$ 的耦合 $H = H_0 - E(t)\cdot\mathbf{P}$；磁场与磁化；机械力与应变。外场让平衡系综被「推开」，系统的可观测量 $B$ 产生偏离平衡的平均值 $\langle B(t)\rangle_{ne}$。

**线性响应**假设偏离很小，$\langle \Delta B(t)\rangle = \langle B(t)\rangle_{ne} - \langle B\rangle_{eq}$ 与场强 $F$ 成线性关系。目标：把这个线性关系用**无扰动系统**的平衡性质写出来。<span class="marginnote">「线性」的含义：忽略 $F$ 的二阶以上效应。这对大多数实验都足够（场强很小），但强场（激光场、强电场器件）需要非线性响应理论——那是另一个世界（非线性光学、强场物理）。线性响应是「弱场、平衡邻域」的普适起点。</span>

## 2 Kubo 公式

久保的推导把量子力学的含时微扰论推进到一阶。对可观测量 $B$，对 $F(t')A$ 的响应：

$$
\langle \Delta B(t)\rangle = \int_{-\infty}^{t} \chi_{BA}(t - t')\,F(t')\,dt'
$$

其中**响应函数（response function）**由平衡关联函数给出：

$$
\chi_{BA}(t) = \frac{1}{i\hbar}\langle [B(t), A(0)]\rangle_{eq}, \qquad t > 0
$$

这是量子形式。对经典系统取 $\hbar\to 0$ 极限，响应函数化为：

$$
\chi_{BA}(t) = -\beta\,\frac{d}{dt}\langle A(0)\,B(t)\rangle_{eq}, \qquad t > 0
$$

其中 $\beta = 1/k_BT$。这是**Kubo 公式**的经典形态：**线性响应 = 平衡关联函数的时间导数**。<span class="marginnote">Kubo 公式常写成另一种等价形式：$\chi_{BA}(t) = \beta\langle \dot A(0)B(t)\rangle$（利用平稳性把时间导数换到 $A$ 上）。实际计算时，哪种形式积分方便就用哪种。这个「关联函数的导数」结构，正是涨落耗散定理里耗散与关联相连的微观来源。</span>

## 3 公式解析：Kubo 公式

把经典 Kubo 公式拆开，确认每个符号：

$$
\chi_{BA}(t) = -\beta\,\frac{d}{dt}\langle A(0)B(t)\rangle_{eq}, \qquad t>0
$$

- **$\beta = 1/k_BT$**：温度因子。温度越高，同样的外场引起的偏转越小（热涨落「冲淡」响应）——所以低温下响应普遍增强。
- **$\langle A(0)B(t)\rangle_{eq}$**：平衡关联函数（上一讲）。响应函数不是新信息，而是关联函数的**导数**——这正是「涨落决定响应」的定量体现。
- **$d/dt$**：时间导数。它把「关联的静态强度」变成「关联的变化率」——只有随时间变化的关联才产生响应；恒定的关联（不衰减的模式）不贡献耗散性响应。
- **$t > 0$ 的约定（因果性）**：$\chi_{BA}(t) = 0$ 当 $t \lt  0$——**响应不能先于原因**。这个因果性约定让响应函数的傅里叶变换 $\chi(\omega)$ 成为上半复平面的解析函数，进而推出 Kramers-Kronig 关系（下一节）。
- **物理含义**：Kubo 公式说「弱扰动下系统的响应，完全由平衡时的涨落动力学决定」。这是一条不需要模型、不需要近似（弱场内）的**精确关系**——动理学理论的一切近似，都是为了算同一个对象：平衡关联函数。

## 4 Kramers-Kronig 关系

因果性（$\chi(t\lt 0) = 0$）对频率域的响应函数 $\chi(\omega)$ 施加了严格的约束。对 $\chi(\omega)$ 的实部与虚部：

$$
\mathrm{Re}\,\chi(\omega) = \frac{1}{\pi}\,\mathcal{P}\int_{-\infty}^{\infty}\frac{\mathrm{Im}\,\chi(\omega')}{\omega' - \omega}\,d\omega'
$$

$$
\mathrm{Im}\,\chi(\omega) = -\frac{1}{\pi}\,\mathcal{P}\int_{-\infty}^{\infty}\frac{\mathrm{Re}\,\chi(\omega')}{\omega' - \omega}\,d\omega'
$$

这就是**Kramers-Kronig 关系**（$\mathcal{P}$ 是主值积分）。它的意义：**实部与虚部互相决定，知道一个就得到另一个。**<span class="marginnote">Kramers-Kronig 关系让实验家可以「补全数据」：光谱实验通常只容易测吸收（虚部，与耗散相关），用它就能推出折射率（实部，与色散相关）——无需再测。反过来，若一个理论给出的实部与虚部不满足 K-K 关系，说明它违背了因果性，一定错了。这是检验响应理论自洽性的金标准。</span>

**辨析｜易错点：** 实部是色散（储能），虚部是耗散（吸能），但**不是**「实部对应可逆、虚部对应不可逆」那么简单。第14讲的涨落耗散定理给出更精确的对位：虚部 $\chi''(\omega)$ 直接与涨落谱 $S(\omega)$ 成正比（耗散 = 涨落），实部则通过 K-K 关系由虚部决定。因果性、K-K 关系、FDT 三者环环相扣。

## 5 例：电导率

把 Kubo 公式用到最简单的例子——电导。外场是电场 $\mathbf{E}$，电流密度算符 $\mathbf{J}$，哈密顿量微扰 $H' = -\mathbf{E}\cdot\mathbf{P}$（$\mathbf{P}$ 是极化）。Kubo 公式给出电流对电场的响应：

$$
\langle J_\mu(t)\rangle = \int_{-\infty}^t \sigma_{\mu\nu}(t-t')\,E_\nu(t')\,dt'
$$

电导率张量：

$$
\sigma_{\mu\nu}(t) = \beta\int_0^\infty dt\, e^{i\omega t}\,\langle J_\nu(0)J_\mu(t)\rangle_{eq}
$$

（取频率表示后）。**电导率 = 电流自关联函数的拉普拉斯/傅里叶变换**。零频直流电导：

$$
\sigma_{DC} = \beta\int_0^\infty \langle J(0)J(t)\rangle_{eq}\,dt
$$

这正是 Green-Kubo 公式（下一讲的主角）。对比德鲁德模型：$\langle J(0)J(t)\rangle \propto e^{-t/\tau}$ 时积分得 $\sigma_{DC} = ne^2\tau/m$——Kubo 公式**自动重现**德鲁德结果，且不需要「自由电子 + 弛豫」的模型假设，只需微观关联函数。<span class="marginnote">Kubo 公式的普适性在于它同时适用于导体、半导体、离子液体、聚合物——只要算得出电流关联函数。这正是「模型无关」的力量：德鲁德模型假设了弛豫时间 $\tau$，Kubo 公式则把 $\tau$ 还原为关联函数的实际衰减时间，由微观动力学自洽给出。</span>

## 6 线性响应与不可逆过程热力学的统一

线性响应理论把第1篇的唯象框架与微观动力学**焊接到了一起**：

- 唯象方程 $J_i = \sum_j L_{ij}X_j$ 的系数 $L_{ij}$，现在有了微观表达式（Green-Kubo 积分）；
- 昂萨格互易关系 $L_{ij}=L_{ji}$，现在可以从关联函数的对称性直接证明（$C_{AB}(t)=C_{BA}(-t)$ 的性质）；
- 熵产生、定态、最小熵产生——全部在线性响应的框架内获得微观基础。

**第1篇的热力学是「骨架」，第4篇的关联函数是「血肉」。** 线性响应理论证明：宏观不可逆过程的全部系数，都编码在平衡涨落的时间结构里。这是从玻尔兹曼到久保一个世纪的思想收敛。

## 7 小结

- **线性响应理论**处理「弱外场驱动的平衡邻域」，响应与场强线性相关。
- **Kubo 公式** $\chi_{BA}(t) = -\beta\,d\langle A(0)B(t)\rangle/dt$ 把响应函数还原为平衡关联函数的导数，模型无关且精确。
- **因果性**（$\chi(t\lt 0)=0$）推出 **Kramers-Kronig 关系**：响应函数实部与虚部互相决定。
- 电导率是 Kubo 公式的直接应用：$\sigma_{DC} = \beta\int_0^\infty\langle J(0)J(t)\rangle dt$