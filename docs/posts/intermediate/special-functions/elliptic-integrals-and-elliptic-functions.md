---
title: 椭圆积分与椭圆函数
date: 2026-08-07
---

# 椭圆积分与椭圆函数

<div class="epigraph">
<p>椭圆函数是分析学里的一首赋格曲，一切主题在其中交织、变奏、回归。</p>
<footer>—— 尼尔斯 · 亨里克 · 阿贝尔（Niels Henrik Abel）与卡尔 · 古斯塔夫 · 雅可比（C. G. J. Jacobi）精神之写照</footer>
</div>

<div class="article-byline">
<p>第二级 · 特殊函数 ｜ 王竹溪、郭敦仁《特殊函数概论》 第 7 章 ｜ 2026-08-07</p>
</div>

## 为什么从椭圆积分开始

如果你问一个受过数学训练的人「$\int_0^x \frac{dt}{\sqrt{1-t^2}}$ 是什么」，他会毫不犹豫地回答 $\arcsin x$。可若把根号里的式子换成 $1 - k^2\sin^2\theta$，积分 $\int \frac{d\theta}{\sqrt{1 - k^2\sin^2\theta}}$ 就再也无法用初等函数表达——它定义了一类全新的函数，名叫**椭圆积分（elliptic integral）**。<span class="marginnote">「椭圆」这个名字来自真实的历史场景：求椭圆弧长时出现 $\int\sqrt{1-k^2\sin^2\theta}\,d\theta$ 型积分，故而得名。欧拉、勒让德在 18 世纪把它系统化，但真正的突破要等到 19 世纪阿贝尔与雅可比——他们问了一个颠覆性的问题：<strong>把椭圆积分反过来</strong>，会得到什么函数？</span>椭圆积分与椭圆函数把「反三角函数的对偶结构」推广到「椭圆」情形：正如 $\sin$ 与 $\arcsin$ 互逆，**椭圆函数就是椭圆积分的反函数**。这门学科串起了单摆、天体轨道、代数几何与密码学，是 19 世纪数学最辉煌的篇章之一。

## 1 椭圆积分的三种标准形式

勒让德（A.-M. Legendre）把椭圆积分规整成三种**标准形式**。令 $0 \lt  k \lt  1$，$k$ 称为**模（modulus）**，记 $k' = \sqrt{1-k^2}$ 为**补模（complementary modulus）**。

**第一类椭圆积分**：

$$
F(\varphi, k) = \int_0^{\varphi} \frac{d\theta}{\sqrt{1 - k^2\sin^2\theta}}
$$

**第二类椭圆积分**：

$$
E(\varphi, k) = \int_0^{\varphi} \sqrt{1 - k^2\sin^2\theta}\, d\theta
$$

**第三类椭圆积分**：

$$
\Pi(\varphi, n, k) = \int_0^{\varphi} \frac{d\theta}{(1 - n\sin^2\theta)\sqrt{1 - k^2\sin^2\theta}}
$$

当 $\varphi = \pi/2$ 时得到**完全椭圆积分** $K(k) = F(\pi/2, k)$、$E(k) = E(\pi/2, k)$。<span class="marginnote">记号里 $k$ 是模、$k'=\sqrt{1-k^2}$ 是补模，经常遇到「模数换参数」的文献差异（有的书用 $m=k^2$ 作参数）。阅读公式前先确认记号约定，是椭圆函数学习者的一条生存法则。</span>这三大类覆盖了所有「有理函数与 $\sqrt{(1-t^2)(1-k^2t^2)}$ 的积分」——勒让德证明了任意这样的积分都能化为 $F,E,\Pi$ 的组合，这就是他的**归约理论**。

## 2 单摆与椭圆弧长：两个物理起源

**单摆的大角度摆动**是椭圆积分最直观的舞台。长度为 $l$ 的摆从角 $\theta_0$ 释放，能量守恒给出

$$
\dot\theta^2 = \frac{2g}{l}\left(\cos\theta - \cos\theta_0\right)
$$

令 $\sin(\theta/2) = \sin(\theta_0/2)\sin\phi$，代换后摆动周期化为

$$
T = 4\sqrt{\frac{l}{g}}\, K\left(\sin\frac{\theta_0}{2}\right)
$$

其中 $K$ 是第一类完全椭圆积分。<span class="marginnote">当 $\theta_0 \to 0$ 时 $k\to0$，$K(0)=\pi/2$，于是 $T \to 2\pi\sqrt{l/g}$——熟悉的简谐近似回归。角度越大，$K$ 越大，周期越长，这是「小角度近似」失效时椭圆积分给出的精确答案。物理里「精确解 = 初等近似 + 特殊函数修正」的模式在此再清晰不过。</span>**椭圆弧长**则是第二类椭圆积分的直接来源：半轴为 $a,b$（$a\ge b$）的椭圆的周长

$$
L = 4a\, E(e), \qquad e = \sqrt{1 - b^2/a^2} \ \text{（离心率）}
$$

任何「算椭圆周长」的场合——行星轨道周长、加速器束流轨道——都落到 $E(k)$。

## 3 从积分到反函数：椭圆函数的诞生

这是本章的**思想转折点**。对 $\sin$ 而言，$x = \int_0^{\arcsin x} \frac{dt}{\sqrt{1-t^2}}$，即 $x = F(\arcsin x)$ 的逆。仿此，雅可比定义

$$
\operatorname{sn}(u, k) = \sin\varphi \quad\Longleftrightarrow\quad u = F(\varphi, k)
$$

**椭圆正弦 $\operatorname{sn}$（elliptic sine）** 是 $F$ 的反函数。再配以

$$
\operatorname{cn}(u,k) = \cos\varphi, \qquad \operatorname{dn}(u,k) = \sqrt{1 - k^2\operatorname{sn}^2(u,k)}
$$

三个函数满足与三角恒等式神似的恒等式：

$$
\operatorname{sn}^2 u + \operatorname{cn}^2 u = 1, \qquad \operatorname{dn}^2 u + k^2\operatorname{sn}^2 u = 1
$$

以及导数关系 $\frac{d}{du}\operatorname{sn}u = \operatorname{cn}u\,\operatorname{dn}u$、$\frac{d}{du}\operatorname{cn}u = -\operatorname{sn}u\,\operatorname{dn}u$、$\frac{d}{du}\operatorname{dn}u = -k^2\operatorname{sn}u\,\operatorname{cn}u$。<span class="marginnote">对比 $\sin' = \cos$、$\cos' = -\sin$，椭圆版本的三个函数像「三角函数的三个投影在曲线上滚动」——它们不是沿圆周 $(x^2+y^2=1)$ 参数化，而是沿<strong>椭圆曲线</strong> $y^2 = (1-t^2)(1-k^2t^2)$ 滚动。这就是「椭圆函数」这一名字的现代几何含义。</span>

## 4 公式解析：双周期性——椭圆函数的心脏

**椭圆函数最深刻的性质是双周期性（doubly periodic）**。$\operatorname{sn},\operatorname{cn},\operatorname{dn}$ 都有两个独立周期，这是它们与单周期函数 $\sin$、$e^z$ 的本质区别。以 $\operatorname{sn}$ 为例，其周期为 $4K(k)$ 与 $2iK'(k)$，其中 $K' = K(k')$ 是补模的完全椭圆积分：

$$
\operatorname{sn}(u + 4K) = \operatorname{sn}u, \qquad \operatorname{sn}(u + 2iK') = \operatorname{sn}u
$$

逐步拆解这条性质：

- **第一步，看到反函数的几何**：$u = F(\varphi,k)$ 把 $\varphi$ 的实数区间映射到实轴；但 $F$ 作为 $\theta$ 的复函数，在复平面上取值时其值域是周期格。反函数 $\operatorname{sn}$ 于是继承了「在周期格上取值不变」的性质。
- **第二步，写出基本周期格**：实周期 $4K$ 来自 $F$ 的「绕一个分支一圈回到原值」；虚周期 $2iK'$ 来自 $F$ 在虚方向的分支结构。两者张成复平面上的**平行四边形网格**——函数在一个基本平行四边形内的行为，决定了它在整个复平面的行为。
- **第三步，对比单周期**：$\sin(z + 2\pi) = \sin z$ 只有一个周期 $2\pi$，值域在一条带内重复；椭圆函数有两个独立周期 $2\omega_1, 2\omega_2$（这里 $\omega_1 = 2K, \omega_2 = iK'$），值域在**一个平行四边形**内重复。**两周期之比 $\omega_2/\omega_1$ 必须是虚部非零的复数**，这正是「椭圆」名称的代数几何含义。
- **第四步，读出数学结构**：双周期亚纯函数构成一个域（椭圆函数域），其上的极点结构由**留数定理**严格约束——「任何椭圆函数的极点个数与留数和满足特定限制」是 Liouville 定理的内容，也是代数几何里椭圆曲线 $\mathbb{C}/\Lambda$ 的出发点。<span class="marginnote">复环面 $\mathbb{C}/\Lambda$（商掉周期格）是亏格 1 的紧黎曼面，也叫椭圆曲线。<strong>现代密码学（ECC）用的就是这条曲线上的群结构</strong>——「椭圆函数/椭圆积分 ⇄ 椭圆曲线密码」的谱系，从这里到《代数几何》《密码学数学基础》一路延伸。</span>

## 5 加法定理与模函数

椭圆函数满足**加法定理**，是三角加法定理 $\sin(u+v)=\sin u\cos v+\cos u\sin v$ 的推广：

$$
\operatorname{sn}(u+v) = \frac{\operatorname{sn}u\,\operatorname{cn}v\,\operatorname{dn}v + \operatorname{sn}v\,\operatorname{cn}u\,\operatorname{dn}u}{1 - k^2\operatorname{sn}^2u\,\operatorname{sn}^2v}
$$

加法定理意味着椭圆函数关于参数 $u$ 的「平移」构成一个代数群——这是「椭圆函数是椭圆曲线上的函数」的解析证明。

**模函数（modular functions）** 则换了一个问题：$K(k)$ 作为模 $k$ 的函数，在 $k$ 被某些变换（如 $k\to 1/k$、$k\to k'$）替代时满足**模变换公式**。Jacobi 的**虚数变换** $K(k') = K'(k)$ 与**变换公式** $K(\frac{1-k'}{1+k'}) = \frac{1+k'}{2}K(k)$ 使 $K$ 能以极快的收敛速度被计算——这就是**算术-几何平均（AGM）算法**的由来，也是当今计算 $K(k)$ 与 $\pi$ 的超快速算法的核心。<span class="marginnote">高斯早在 1799 年就发现 AGM 与 $K$ 的联系：$K(k) = \frac{\pi}{2\,\mathrm{AGM}(1, k')}$。AGM 迭代二次收敛，几步就能把 $K$ 算到任意精度——1985 年 Brent–Salamin 公式用 AGM 把 $\pi$ 算到百万位，用的正是这条谱系。</span>

## 6 椭圆函数的应用地图与易错点

**精确物理模型**：单摆、双摆、非线性振子（Duffing 方程的精确解用 $\operatorname{cn}$）、旋转体在重力下的运动（Kovalevskaya 陀螺）。
**轨道力学**：二体问题在近抛物线轨道、摄动理论中，位置随时间的关系用椭圆函数；Kepler 方程在椭圆轨道上的解也与其相关。
**代数几何与数论**：椭圆曲线、椭圆函数域、模形式，是费马大定理证明中模性定理的舞台。
- **密码学**：ECC（椭圆曲线密码）的群运算。
- **数值方法**：AGM、椭圆积分的 Carlson 对称形式（$R_F,R_J$），是现代科学计算库（`scipy.special.ellipk` 等）的标准实现。<span class="marginnote">Carlson 对称形式把三类勒让德椭圆积分统一成少数几个对称积分，数值稳定性远优于传统定义，如今被 IEEE 的许多数值库采纳。这也是「定义形式优美、计算形式对称」这一工程智慧的案例。</span>

**辨析｜易错点：** 第一，$K(k)$ 在 $k\to 1$ 时对数发散：$K(k) \sim \frac{4}{\pi}\ln\frac{4}{k'}$（或 $\frac12\ln\frac{16}{1-k^2}$），但 $E(1) = 1$ 有限——「第一类发散、第二类有限」是常见的混淆点。第二，$\operatorname{sn}, \operatorname{cn}, \operatorname{dn}$ 的自变量是「弧长参数 $u$」而非角度，直接代角度会错。第三，$k$ 与 $m=k^2$ 两种参数化极易在查表时混淆。

## 7 小结

- **椭圆积分**：第一类 $F(\varphi,k)$、第二类 $E(\varphi,k)$、第三类 $\Pi(\varphi,n,k)$ 覆盖全部「$\sqrt{(1-t^2)(1-k^2t^2)}$ 型」积分；完全椭圆积分 $K(k),E(k)$ 是端点值。
- **物理起源**：单摆周期 $T = 4\sqrt{l/g}\,K(\sin\theta_0/2)$，椭圆周长 $L = 4aE(e)$。
- **椭圆函数**：$\operatorname{sn},\operatorname{cn},\operatorname{dn}$ 是 $F$ 的反函数，满足与三角恒等式神似的恒等式与加法定理。
- **双周期性**：两个独立周期 $4K$ 与 $2iK'$ 使函数在平行四边形网格上重复，这是它与单周期函数的分水岭，通向椭圆曲线。
- **模函数与 AGM**：模变换公式支撑 $K$ 的超快速计算；$K = \pi/(2\,\mathrm{AGM})$ 通向现代 $\pi$