---
title: 平稳过程的谱分解初步
date: 2026-08-07
---

# 平稳过程的谱分解初步

<div class="epigraph">
<p>每一个平稳过程都是一支乐团——无限多的正弦声部，各自奏着独立的随机音量。</p>
<footer>—— 哈拉尔德 · 克拉默（Harald Cramér）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§9.6 ｜ 2026-08-07</p>
</div>

## 把平稳过程「拆成正弦」

Wiener-Khinchin 告诉我们：平稳过程的 ACF 有谱密度 $S(\omega)$。但一个更深的问题悬而未决：**过程本身（不只是它的相关函数）能不能拆成不同频率的正弦波？** 答案是**能**——这就是**谱分解定理（spectral representation theorem）**：
$$
X(t) = \int_{-\infty}^{\infty} e^{i\omega t}\, dZ(\omega),
$$
其中 $Z(\omega)$ 是**正交增量过程**（orthogonal increment process）：不同频率上的增量不相关，且 $E[|dZ(\omega)|^2] = dF(\omega)$（$F$ 是**谱分布函数**，$dF = S\,d\omega$）。<span class="marginnote">谱分解的语言：<strong>$X(t)$ 是「无穷多个频率为 $\omega$ 的正弦 $e^{i\omega t}$，各自带着随机振幅 $dZ(\omega)$」的叠加</strong>。$dZ(\omega)$ 的「能量」$E[|dZ|^2] = dF$ 就是谱密度——过程在频域的「音量分配」。</span>

本节目标：陈述谱分解定理、理解随机振幅结构、并用它解释周期成分与 ARMA。

## 1 谱表示定理

**谱表示定理（Cramér 谱分解）**：任何宽平稳过程 $X(t)$（连续时间）都可以唯一表示为
$$
X(t) = \int_{-\infty}^{\infty} e^{i\omega t}\, dZ(\omega),
$$
其中 $Z(\omega)$ 满足：

1. **正交增量**：$\mathrm{Cov}\big(dZ(\omega), dZ(\omega')\big) = 0$ 对 $\omega \ne \omega'$——**不同频率互不相关**；
2. **谱测度**：$E\big[|dZ(\omega)|^2\big] = dF(\omega)$，其中 $F(\omega)$ 是**谱分布函数**（非降、有界）；
3. **ACF 的谱表示**：$\gamma(\tau) = \int e^{i\omega\tau} dF(\omega)$。

**直觉**：过程 = 「所有频率的正弦 × 随机振幅」的连续叠加。**$dZ(\omega)$ 是第 $\omega$ 频道的随机振幅，谱密度 $S(\omega)$ 是它的功率分配。**<span class="marginnote">这个定理把「平稳」翻译成「频率正交」：<strong>平稳 ⟺ 不同频率成分互不相关</strong>。这正是为什么平稳过程在频域如此干净——时间域的复杂相关，在频率域变成「各自独立的分频轨道」。</span>

## 2 谱密度与谱测度

**绝对连续情形**：$dF(\omega) = S(\omega)\, d\omega$——有谱密度，ACF 绝对可积时成立（白噪声、AR、MA）。
**原子（离散）情形**：$F$ 有跳变——过程含**纯周期成分**（如 $\sin(\omega_0 t + \Phi)$ 在 $\omega_0$ 处谱有尖峰）。
**混合**：一般过程 = 绝对连续（随机部分）+ 离散（周期部分）。

**谱分布函数的角色**：它是「功率随频率的累积分配」——$F(\omega_2) - F(\omega_1)$ 是区间 $[\omega_1, \omega_2]$ 内的功率。<span class="marginnote">「谱尖峰 = 周期成分」是谱分析的核心判读：<strong>谱在某个频率出现 δ 型尖峰，说明过程含确定的正弦周期</strong>；谱平坦则全随机。看谱找尖峰，就是「找隐藏的周期」——时间序列里检测季节性、心率里的周期、声波的音高，全靠这一条。</span>

## 3 公式解析：从谱表示回到 ACF

**目标：用谱表示定理推出 $\gamma(\tau) = \int e^{i\omega\tau} dF(\omega)$——谱表示与 Wiener-Khinchin 的合一。**

第一步，写 ACF。$\gamma(\tau) = E[X(t)\overline{X(t+\tau)}]$，代入谱表示：
$$
\gamma(\tau) = E\Big[ \int e^{i\omega t} dZ(\omega)\, \overline{\int e^{i\omega' (t+\tau)} dZ(\omega')} \Big].
$$
第二步，交换期望与积分。共轭把 $e^{i\omega' (t+\tau)}$ 变 $e^{-i\omega' (t+\tau)}$：
$$
\gamma(\tau) = \int\int e^{i(\omega - \omega')t} e^{-i\omega'\tau}\, E\big[ dZ(\omega)\, \overline{dZ(\omega')} \big].
$$
第三步，用正交增量。$E[dZ(\omega)\overline{dZ(\omega')}] = \delta(\omega - \omega') dF(\omega)$（不同频率不相关）：
$$
\gamma(\tau) = \int e^{-i\omega\tau}\, dF(\omega).
$$
第四步，对称性。$\gamma(-\tau) = \gamma(\tau)$ 给出 $\gamma(\tau) = \int e^{i\omega\tau} dF(\omega)$——**ACF 是谱测度的傅里叶变换，与 Wiener-Khinchin 完全一致**。

**这个推导为什么重要**：它证明谱表示定理与 Wiener-Khinchin 是同一枚硬币的两面——**「过程拆成随机正弦」与「ACF 拆成频谱」互相印证**。谱分解是更根本的：它不仅描述相关函数，还描述过程本身。

## 4 例子：周期成分与 AR(1)

**例（纯周期 + 噪声）**：$X(t) = A\sin(\omega_0 t + \Phi) + \epsilon_t$。谱 = $\omega_0$ 处的尖峰（周期部分）+ 平坦底（白噪声）。**尖峰位置 $\omega_0$ 直接读出周期 $2\pi/\omega_0$。**

**例（AR(1)）**：$X_t = \phi X_{t-1} + \epsilon_t$，谱 $S(\omega) = \frac{\sigma^2}{1 - 2\phi\cos\omega + \phi^2}$。无尖峰（纯随机），但低频功率集中——**平滑过程的谱是「低频鼓包」，不是尖峰**。<span class="marginnote">尖峰 vs 鼓包的判别：<strong>尖峰（δ 型）= 确定性周期；鼓包（连续谱）= 随机但低频偏好</strong>。这个判别把「确定周期」与「随机平滑」分得清清楚楚——是谱分析的第一课。</span>

## 5 谱分解的意义

**滤波再解释**：谱分解下，LTI 滤波只是「每个频率振幅 $dZ(\omega)$ 乘 $H(\omega)$」——频域操作的几何意义一清二楚。
**模拟**：给定谱 $S(\omega)$，可用「谱合成」模拟平稳过程——生成随机振幅的正弦叠加。
**统计**：谱密度估计（周期图平滑）是「非参数地看过程结构」——不需要指定 ARMA 阶数，直接看谱形状。<span class="marginnote">谱分解的思想溢出到机器学习：<strong>「谱方法」（谱聚类、图傅里叶变换）把结构性问题搬到频率域</strong>——平稳过程的谱分解是这一切的数学源头。本站《图神经网络》一章的谱卷积，追根溯源就是这里的 $e^{i\omega t}$ 分解。</span>

## 6 小结

- **谱表示定理**：$X(t) = \int e^{i\omega t} dZ(\omega)$——过程 = 随机振幅的正弦叠加。
- **正交增量**：不同频率不相关；$E[|dZ|^2] = dF$（谱测度）。
- **ACF 谱表示**：$\gamma(\tau) = \int e^{i\omega\tau} dF(\omega)$——与 Wiener-Khinchin 合一。
- **尖峰 vs 鼓包**：尖峰 = 确定周期，鼓包 = 随机低频偏好。
- 应用：滤波的频域几何、谱合成模拟、谱方法。

**谱合成模拟**：给定谱 $S(\omega)$，可按其形状生成随机振幅的正弦叠加来模拟平稳过程——这是「给定 ACF/谱造过程」的工程捷径，也是谱方法在信号生成、蒙特卡洛模拟里的直接用途。

到这里，第九篇《平稳过程》全部结束。从下一篇起，我们把随机过程放回现实世界的舞台：**应用**——金融、保险、排队、强化学习与 MCMC。
