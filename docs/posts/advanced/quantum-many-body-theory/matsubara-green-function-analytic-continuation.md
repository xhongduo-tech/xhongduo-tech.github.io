---
title: 松原格林函数（虚时技术、频率求和、解析延拓）
date: 2026-08-07
---

# 松原格林函数（虚时技术、频率求和、解析延拓）

<div class="epigraph">
<p>虚时 Green 函数就像一个把温度编码成离散频率的加密器——解密它的钥匙叫解析延拓。</p>
<footer>—— 武谷三男（Mitio Matsubara）1955 年原始论文的当代注脚（转述）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics*, Ch. 3 ｜ 2026-08-07</p>
</div>

## 为什么从松原 Green 函数开始

第 2 篇建立了虚时框架，第 3 篇解剖了零温谱表示。把两者缝合起来，就得到有限温度场论真正能干活的工作台：**松原 Green 函数（Matsubara Green function）**。<span class="marginnote"><strong>命名由来</strong>：1955 年日本物理学家武谷三男提出在虚时轴上定义 Green 函数，用离散的虚频率自动带入玻色/费米统计，一举统一了有限温度微扰论。</span>它在虚时间 $0\le\tau\le\beta$ 上定义，Fourier 变换到离散的 **Matsubara 频率**，计算的关键是**频率求和（Matsubara summation）**，最后用**解析延拓（analytic continuation）**把结果搬回实频率。三步合起来，就是本篇的三根支柱。

## 1 松原 Green 函数的定义

在巨正则系综里定义虚时（松原）Green 函数：

$$\mathcal{G}(\mathbf{k},\tau) = -\langle T_\tau\, c_{\mathbf{k}}(\tau)c_{\mathbf{k}}^\dagger \rangle$$

其中 $T_\tau$ 是虚时序，$\tau\in(0,\beta)$ 是虚时坐标，$c_{\mathbf{k}}(\tau)=e^{\tau(\hat{H}-\mu\hat{N})}c_{\mathbf{k}}e^{-\tau(\hat{H}-\mu\hat{N})}$。<span class="marginnote"><strong>关键约束</strong>：虚时 $\tau$ 永远只在 $[0,\beta]$ 区间内取值。因为 $e^{-\beta(\hat{H}-\mu\hat{N})}$ 的周期结构，松原函数在 $\tau$ 方向有确定的（反）周期边界条件，这是它区别于零温函数的全部秘密。</span>

## 2 Matsubara 频率：周期性带来的离散化

把 $\mathcal{G}$ 沿虚时做 Fourier 变换。虚时区间的周期性约束选出离散频率：

- **费米子**（反周期，$\mathcal{G}(\tau+\beta)=-\mathcal{G}(\tau)$）：奇 Matsubara 频率
$$\omega_n = \frac{(2n+1)\pi}{\beta}, \qquad n\in\mathbb{Z}$$

- **玻色子**（周期，$\mathcal{G}(\tau+\beta)=\mathcal{G}(\tau)$）：偶 Matsubara 频率
$$\omega_n = \frac{2n\pi}{\beta}, \qquad n\in\mathbb{Z}$$

**重点：玻色/费米统计被编码进频率的奇偶性。** 费米频率取奇倍、玻色频率取偶倍，这一选择让松原函数的 $n$ 求和自动重现费米-狄拉克或玻色-爱因斯坦分布。Fourier 变换对是

$$\mathcal{G}(\mathbf{k},\tau) = \frac{1}{\beta}\sum_n e^{-i\omega_n\tau}\mathcal{G}(\mathbf{k},i\omega_n), \qquad \mathcal{G}(\mathbf{k},i\omega_n) = \int_0^\beta d\tau\, e^{i\omega_n\tau}\mathcal{G}(\mathbf{k},\tau)$$

## 3 频率求和：配分函数与微扰展开的引擎

有限温度微扰论里，每个圈积分都伴随一个对 Matsubara 频率的求和 $\frac{1}{\beta}\sum_n$。例如自由费米子松原函数

$$\mathcal{G}(\mathbf{k},i\omega_n) = \frac{1}{i\omega_n - (\varepsilon_{\mathbf{k}}-\mu)}$$

计算物理量时出现形如 $\frac{1}{\beta}\sum_n f(i\omega_n)$ 的和。标准技巧是把求和改写成复平面上的围道积分：让一个辅助函数 $F(z)$ 在 $z=i\omega_n$ 处留数为 $f(i\omega_n)$，再对一条包围全部 Matsubara 极点的围道积分。$F$ 取玻色分布 $1/(e^{\beta z}-1)$（对偶频率）或费米分布（对奇频率）。<span class="marginnote"><strong>为什么可行</strong>：玻色函数 $n_B(z)=1/(e^{\beta z}-1)$ 在 $z=i\omega_n$（偶频）处留数恰好是 $-1/\beta$；用它乘以 $f(z)$ 并对包围所有频率极点的围道积分，围道积分等于 $f(z)$ 物理极点处留数之和——积分化为留数求和，这正是 Matsubara 求和的核心招式。</span>

## 4 解析延拓：从虚频率回到实频率

松原函数只在离散虚频率 $i\omega_n$ 上有定义，实验量（谱函数、电导率）却依赖实频率。桥梁是**解析延拓**：把 $\mathcal{G}(i\omega_n)$ 延拓到复平面，在物理频率轴上取极限

$$G^R(\omega) = \mathcal{G}(i\omega_n\to \omega + i0^+), \qquad G^A(\omega) = \mathcal{G}(i\omega_n\to \omega - i0^+)$$

从虚频率上实轴取极限，分别得到推迟（$+i0^+$）与超前（$-i0^+$）Green 函数。<span class="marginnote"><strong>易错点</strong>：松原函数只定义在虚频率这个离散点集上，延拓到实轴必须「从正确的方向」取极限。不能把 $i\omega_n$ 直接换成实 $\omega$——两者之间隔着谱函数积分。数值上这被称为「病态反演」，是量子蒙特卡洛里最头疼的问题之一。</span>

## 5 公式解析：一次典型的松原求和

计算电子气极化率的虚时版本是检验求和技巧的经典练习。极化率（不插入动量依赖，仅示意结构）

$$\Pi(i\nu_m) = \frac{1}{\beta}\sum_n \frac{1}{i\omega_n - \xi_\mathbf{k}} \cdot \frac{1}{i\omega_n + i\nu_m - \xi_{\mathbf{k}+\mathbf{q}}}$$

其中 $\xi_\mathbf{k}=\varepsilon_\mathbf{k}-\mu$，$\nu_m$ 是玻色频率（因为极化率是玻色型量，动量-动量响应）。求和拆成三步：

- **第一步，识别奇偶**：$i\omega_n$ 是费米频率（粒子传播子），$i\nu_m$ 是玻色频率（外腿动量转移）。外腿玻色、内腿费米是极化率的标志。
- **第二步，围道积分**：对 $i\omega_n$ 求和，把 $n_B(z)f(z)$ 围道积分，等价于 $f(z)$ 在两个物理极点 $z=\xi_\mathbf{k}$ 与 $z=\xi_{\mathbf{k}+\mathbf{q}}-i\nu_m$ 处留数之差。
- **第三步，取留数**：得 $\Pi = \frac{n_F(\xi_\mathbf{k}) - n_F(\xi_{\mathbf{k}+\mathbf{q}})}{i\nu_m - \xi_{\mathbf{k}+\mathbf{q}} + \xi_\mathbf{k}}$。这正是有限温度 Lindhard 响应的雏形——零温极限下它退化为第 5 篇、RPA 计算里用到的标准形式。

**重点：松原求和的结果自动带有正确的统计因子。** 围道积分里出现的费米分布函数 $n_F$ 不是事后贴上去的，而是 Matsubara 频率奇偶性 + 围道技巧的直接产物——这是松原技术最优雅的地方。

## 6 与零温 Green 函数的统一视角

松原 Green 函数是零温 Green 函数的有限温度推广，两者由解析延拓连接成同一个解析对象。物理上：

- 零温极限 $\beta\to\infty$ 时，Matsubara 频率间距 $2\pi/\beta\to 0$，离散频率变成连续频率，松原求和退化为零温频率积分。
- 玻色/费米统计因子在零温下退化为占据态的简单投影。

一句话总结：**零温与有限温度不是两套理论，而是同一 Green 函数在不同频率栅格上的两个视图。**

在数值层面，这一统一还有实际意义：量子蒙特卡洛（QMC）在虚时间轴上采样，得到的正是松原函数在虚时格点上的值；要把它们与实验（实频率）对接，就必须做解析延拓。这是第 12 篇《强关联与数值方法》里 QMC 与 DMFT 都会反复面对的同一条技术路线。<span class="marginnote"><strong>知识连线</strong>：RPA、超导自能、电导率计算全部在松原框架里做频率求和；本篇的围道求和技巧是第 5 篇《相互作用电子气》、第 6 篇《费米液体理论》里无数圈积分的前置技能。</span>

## 7 具体例子：从松原和里长出的费米分布

用一个最简单却极具说服力的例子收束技术细节：从自由费米子松原函数「重新长回」费米分布。占据数等于 $\tau\to0^-$ 极限下 Green 函数的求迹：

$$\langle n_{\mathbf{k}}\rangle = \frac{1}{\beta}\sum_n e^{i\omega_n 0^+}\mathcal{G}(\mathbf{k},i\omega_n) = \frac{1}{\beta}\sum_n \frac{e^{i\omega_n 0^+}}{i\omega_n - \xi_{\mathbf{k}}}$$

把求和换成围道积分：取 $n_F(z)$ 的极点（奇频率处留数恰为 $-1/\beta$），围道变形到包围物理极点 $z=\xi_{\mathbf{k}}$，立刻得到

$$\langle n_{\mathbf{k}}\rangle = n_F(\xi_{\mathbf{k}}) = \frac{1}{e^{\beta\xi_{\mathbf{k}}}+1}$$

这个练习的启发是双重的。其一，**统计分布不是被「放进」理论，而是从松原频率结构里自动「长」出来的**——费米分布出现在结果里，不需要任何额外假设。其二，$e^{i\omega_n 0^+}$ 这个「收敛因子」至关重要：没有它，级数不绝对收敛，求和顺序会出错。这是松原技术里最容易被忽略、却决定成败的一处细节。

## 8 小结

- **松原 Green 函数**定义在虚时 $\tau\in(0,\beta)$，Fourier 变换到离散的 Matsubara 频率。
- 费米子取**奇频率** $\omega_n=(2n+1)\pi/\beta$，玻色子取**偶频率** $\omega_n=2n\pi/\beta$，统计性质由此自动进入。
- **频率求和**用围道积分 + 玻色/费米分布函数的留数技巧完成，结果自动携带正确的统计因子。
- **解析延拓** $i\omega_n\to\omega\pm i0^+$ 把虚频率结果搬回实频率，得到推迟/超前 Green 函数。
- 零温极限 $\beta\to\infty$ 下松原求和退化为零温积分，两套理论是同一个解析对象。
- 病态反演问题（从虚频率数值重建谱函数）是量子蒙特卡洛与实验谱对接的核心难点。

## 9 公式速查：一页纸复习

| 对象 | 表达式 | 一句话要点 |
| --- | --- | --- |
| 费米频率 | $\omega_n=(2n+1)\pi/\beta$ | 奇倍，自动费米统计 |
| 玻色频率 | $\omega_n=2n\pi/\beta$ | 偶倍，自动玻色统计 |
| 自由松原函数 | $1/(i\omega_n-\xi_\mathbf{k})$ | 单极点 |
| 围道求和 | $\frac1\beta\sum_n f(i\omega_n)=\oint n_{B/F}(z)f(z)$ | 留数定理 |
| 推迟延拓 | $\mathcal{G}(i\omega_n\to\omega+i0^+)$ | 因果性在上半平面 |
| 零温极限 | $\beta\to\infty$，频率栅格连续化 | 退化为零温积分 |

**易错复盘**：虚时 $\tau$ 只在 $[0,\beta]$；松原频率奇偶不能搞反；解析延拓必须取对极限方向；围道积分里 $n_B(z)$ 的留数 $-1/\beta$ 别漏系数。这四处是松原技术里最常见的失分点。

**知识连线**：本篇把第 2 篇（虚时形式）与第 3 篇（谱表示）缝合，是第 5 篇《相互作用电子气》与第 8 篇《线性响应与输运》的频率求和引擎；「离散频率编码温度」的松原思想，与「离散 token 编码连续语义」的大模型主线形成有趣的同构。

在下一节，我们将系统展开有限温度微扰论的核心工具：Wick 定理、Feynman 图、Dyson 方程与自能。
