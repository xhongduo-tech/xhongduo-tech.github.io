---
title: 功率谱密度与 Wiener-Khinchin 定理
date: 2026-08-07
---

# 功率谱密度与 Wiener-Khinchin 定理

<div class="epigraph">
<p>时间域里纠缠的相关，在频率域里各安其位——傅里叶是随机过程的分光镜。</p>
<footer>—— 诺伯特 · 维纳（Norbert Wiener）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§9.4 ｜ 2026-08-07</p>
</div>

## 把相关性「分光」

宽平稳过程的自相关函数 $\gamma(\tau)$ 描述「时间域」的相关结构。但很多问题在**频率域**更清楚：信号主要在哪几个频率上有能量？低频波动 vs 高频噪声各占多少？**功率谱密度（power spectral density, PSD）**回答这些问题——它是 $\gamma(\tau)$ 的傅里叶变换。

**Wiener-Khinchin 定理**——随机过程谱理论的核心——断言：**宽平稳过程的自协方差函数与功率谱密度构成一对傅里叶变换**：
$$
S(\omega) = \int_{-\infty}^{\infty} \gamma(\tau)\, e^{-i\omega\tau}\, d\tau, \qquad \gamma(\tau) = \frac{1}{2\pi}\int_{-\infty}^{\infty} S(\omega)\, e^{i\omega\tau}\, d\omega.
$$
**时域与频域完全等价**——知道 ACF 就知道频谱，反之亦然。这个「时频对偶」是整个信号处理、滤波理论的地基。<span class="marginnote">Wiener-Khinchin 的直觉：<strong>ACF 测「时间上隔多远还相关」，PSD 测「哪个频率带多少功率」——两者是同一信息的两副眼镜</strong>。周期分量在 ACF 里是振荡、在谱里是尖峰；白噪声在 ACF 里是脉冲、在谱里是平坦。看谱比看 ACF 更能「一眼识别成分」。</span>

本节目标：定义 PSD、陈述 Wiener-Khinchin、掌握 PSD 的性质与两个经典例子。

## 1 定义与含义

**功率谱密度（PSD）**：宽平稳过程 $X(t)$ 的 PSD 定义为
$$
S(\omega) = \int_{-\infty}^{\infty} \gamma(\tau)\, e^{-i\omega\tau}\, d\tau.
$$

**含义**：$S(\omega)$ 描述「单位频率带宽内的功率」。过程的总方差等于谱的面积：
$$
\gamma(0) = \mathrm{Var}(X(t)) = \frac{1}{2\pi}\int_{-\infty}^{\infty} S(\omega)\, d\omega.
$$
**方差 = 谱下总面积**——PSD 把总波动「分摊」到各个频率，哪里谱高，哪里的频率就贡献更多波动。<span class="marginnote">PSD 的「功率」解释：<strong>对现实信号，PSD 的估计 ≈「各频率成分的平方振幅」</strong>。高频谱尖峰 = 该频率有强周期成分；平坦谱 = 白噪声（所有频率等功率）。工程师看频谱图找尖峰，就像医生看心电图找异常节律。</span>

## 2 PSD 的三个基本性质

1. **非负**：$S(\omega) \ge 0$——功率不能为负（来自 $\gamma$ 的半正定性）；
2. **偶函数**：$S(-\omega) = S(\omega)$——实值过程谱对称；
3. **面积 = 方差**：$\frac{1}{2\pi}\int S(\omega)d\omega = \gamma(0)$——谱下总面积是总波动。

**辨析｜易错点：** 单边谱 vs 双边谱：很多工程书用「单边谱」（只画 $\omega \ge 0$，值翻倍）。**教材里的 $S(\omega)$ 默认双边**，对照工程文献时注意换算。

## 3 经典例子：白噪声与 AR(1)

**例一（白噪声）**：$\gamma(0) = \sigma^2$，$\gamma(\tau) = 0$（$\tau \ne 0$）。傅里叶变换：
$$
S(\omega) = \int \sigma^2 \delta(\tau) e^{-i\omega\tau}d\tau = \sigma^2.
$$
**白噪声的谱是常数 $\sigma^2$——所有频率等功率**。「白」即「光谱均匀」之谓。

**例二（AR(1)）**：$\gamma(\tau) = \sigma^2 \frac{\phi^{|\tau|}}{1 - \phi^2}$。傅里叶变换（几何级数）：
$$
S(\omega) = \frac{\sigma^2}{|1 - \phi e^{-i\omega}|^2} = \frac{\sigma^2}{1 - 2\phi\cos\omega + \phi^2}.
$$
**低频（$\omega \approx 0$）谱高、高频谱低——AR(1) 是「低频为主」的平滑过程**；$\phi$ 越接近 1，低频占比越大（更平滑）。<span class="marginnote">AR(1) 的谱形状与「平滑」直觉一致：<strong>正相关的过程波动缓慢（低频多），负相关则高频多</strong>。看谱能直接读出「这个过程是平滑还是毛糙」——这是谱分析最直观的用途。</span>

## 4 公式解析：Wiener-Khinchin 的证明骨架

**目标：说明「ACF 的傅里叶变换 = 谱」不是天外飞仙，而是「周期图平均」的自然极限。**

第一步，定义周期图。对有限样本 $X_1, \dots, X_n$，定义**周期图（periodogram）**：
$$
I_n(\omega) = \frac{1}{n} \Big| \sum_{t=1}^n X_t e^{-i\omega t} \Big|^2.
$$
这是「样本谱」的朴素估计——直接把数据做傅里叶再取模方。

第二步，展开周期图。$|a|^2 = a\bar a$，把平方展开成双重和：
$$
I_n(\omega) = \frac{1}{n}\sum_{s,t} X_s X_t e^{-i\omega(s-t)}.
$$
第三步，重组为 ACF。把 $\sum_{s,t}$ 按 $s - t = \tau$ 分组：
$$
I_n(\omega) = \sum_{\tau} \hat\gamma_n(\tau)\, e^{-i\omega\tau},
$$
其中 $\hat\gamma_n(\tau)$ 是样本自协方差。**周期图正是「样本 ACF 的傅里叶变换」。**

第四步，取极限。遍历性 + 适当条件：$\hat\gamma_n(\tau) \to \gamma(\tau)$，于是 $I_n(\omega) \to S(\omega)$。**PSD = ACF 的傅里叶变换，而周期图是它的有限样本估计——Wiener-Khinchin 成立。**

**这个推导为什么重要**：它揭示 PSD 的「出身」——**不是凭空定义，而是周期图的极限**。这也解释了谱估计的现实做法：直接算周期图，再平滑（因为周期图本身是 $S(\omega)$ 的有偏且不稳定的估计）。

## 5 应用：从谱读结构

- **周期检测**：谱里的尖峰 = 周期成分（季节、循环）。找尖峰 = 找隐藏周期。
- **滤波设计**（下节）：滤波器在频域是「乘一个权重函数」——设计滤波 = 设计 $H(\omega)$ 形状。
- **信号与噪声分离**：真实信号往往低频为主，噪声高频平坦——低通滤波截掉高频 = 去噪。
- **模型识别**：谱形状提示 AR/MA 阶数（AR 谱是「有理函数」，MA 谱是「多项式」）。<span class="marginnote">谱分析与机器学习主线的连接：<strong>时间序列的特征工程、季节性分解、白噪声检验都依赖谱</strong>；甚至深度学习里的「频谱域卷积」「傅里叶特征」也是同一套思想的延伸。</span>

## 6 小结

- **PSD** $S(\omega) = \int \gamma(\tau)e^{-i\omega\tau}d\tau$——ACF 的傅里叶变换。
- **Wiener-Khinchin**：ACF 与 PSD 互为傅里叶变换对——时频完全等价。
- **性质**：非负、偶、面积 $= \gamma(0)$（总方差）。
- **例子**：白噪声谱平坦 $\sigma^2$；AR(1) 谱 $\frac{\sigma^2}{1-2\phi\cos\omega+\phi^2}$ 低频为主。
- **周期图**：样本谱估计 = 样本 ACF 的傅里叶变换；谱估计 = 周期图 + 平滑。

在下一节，我们把 PSD 变成「加工」工具：**平稳过程的线性变换与滤波**——LTI 系统如何改变输入谱。
