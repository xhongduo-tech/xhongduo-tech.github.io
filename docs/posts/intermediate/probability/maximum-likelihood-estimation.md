---
title: 最大似然估计法
date: 2026-08-07
---

# 最大似然估计法

<div class="epigraph">
<p>把已经发生的事看作最可能发生的事，选一个让观测最「顺理成章」的参数——这就是最大似然。</p>
<footer>—— 罗纳德 · 费希尔（Ronald Fisher），最大似然估计提出者</footer>
</div>

<div class="article-byline">
<p>第二级 · 概率论与数理统计 ｜ 盛骤《概率论与数理统计》§7.1 ｜ 2026-08-07</p>
</div>

## 为什么从最大似然估计开始

矩估计是「用样本矩硬配总体矩」，朴素但常常不是最优。**最大似然估计（maximum likelihood estimation, MLE）**是点估计的正统主力：它选「让已观测数据出现概率最大」的参数。这个原则直指统计推断的哲学核心——**既然这些数据已经发生了，那它们理应在所选参数下最「顺理成章」**。

MLE 是现代统计的发动机：它是费希尔 1912–1922 年间建立的方法论基石，拥有无偏性（渐近）、有效性（达到克拉美—罗下界）、相合性等一系列最优性质；第七章的多数点估计、机器学习里的极大似然训练（交叉熵就是负对数似然）、以及第十章的参数 bootstrap，全都以它为底。**矩估计能给的，MLE 几乎总能给，而且通常更好**。<span class="marginnote">MLE 的核心哲学用一个例子即可穿透：你从箱子里摸出 10 个球有 7 个红球，问红球比例 $\theta$。矩估计也许答 0.7，但 MLE 的逻辑是「在哪个 $\theta$ 下，摸出 7 红 3 白这件事最可能发生」——答案也是 0.7，但它的推导告诉我们「为什么 0.7 是对的」。这条「后验推理」正是贝叶斯思想的亲兄弟。</span>

## 1 似然函数与最大似然估计

**似然函数（likelihood function）**：设总体 $X$ 的密度（或分布律）为 $f(x; \theta)$，$\theta$ 为未知参数，样本 $X_1, \ldots, X_n$ 的观测值为 $x_1, \ldots, x_n$，则

$$L(\theta) = L(x_1, \ldots, x_n; \theta) = \prod_{i=1}^n f(x_i; \theta)$$

是 $\theta$ 的函数（数据已固定）。**最大似然估计**：若 $\hat\theta$ 使 $L(\theta)$ 达到最大，即

$$L(\hat\theta) = \max_{\theta} L(\theta)$$

则称 $\hat\theta$ 为 $\theta$ 的**最大似然估计**。<span class="marginnote">似然函数与密度的区别：密度 $f(x;\theta)$ 是「固定 $\theta$、看 $x$ 怎么变」；似然 $L(\theta)$ 是「固定数据 $x$、看 $\theta$ 怎么变」。函数形式相同，视角相反。正因如此，似然不是「数据出现的概率密度」（它不对 $\theta$ 归一），而是一个「相对评分」——比大小用，不求和。</span>

## 2 对数似然与求导流程

由于连乘求导麻烦，通常取对数把连乘变连加（对数似然 $\ln L(\theta)$ 与 $L(\theta)$ 同极值点，因为 $\ln$ 单调增）：

$$\ln L(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)$$

**求 MLE 的标准流程**：

1. **写似然**：$L(\theta) = \prod_i f(x_i; \theta)$；
2. **取对数**：$\ln L(\theta) = \sum_i \ln f(x_i; \theta)$；
3. **求导令零**：$\frac{d}{d\theta}\ln L(\theta) = 0$，解出 $\hat\theta$；
4. **验证极大**：二阶导 $< 0$，或由实际问题判断（密度通常保证唯一峰）。

<span class="marginnote">「对数似然方程」是 MLE 的标准入口。多参数时对每个参数求偏导、列方程组。若方程无闭式解（如某些复杂模型），就用数值优化——机器学习里梯度下降优化的正是负对数似然。这一章练的「求导解方程」是未来「训练模型」的最小原型。</span>

**例（指数分布）**：$X \sim E(\lambda)$，$f(x;\lambda) = \lambda e^{-\lambda x}$。则

$$\ln L = \sum_{i=1}^n (\ln\lambda - \lambda x_i) = n\ln\lambda - \lambda\sum_i x_i, \qquad \frac{d}{d\lambda}\ln L = \frac{n}{\lambda} - \sum_i x_i = 0 \implies \hat\lambda = \frac{1}{\bar x}$$

**指数分布的 MLE = 样本均值的倒数**——与矩估计结果一致。

## 3 例：正态总体的 MLE

**例（正态）**：$X \sim N(\mu, \sigma^2)$。对数似然：

$$\ln L = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

对 $\mu$ 求导令零：$\sum (x_i - \mu) = 0 \implies \hat\mu = \bar x$；对 $\sigma^2$ 求导令零：$-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i-\mu)^2 = 0 \implies \hat\sigma^2 = \frac1n\sum(x_i - \bar x)^2$。<span class="marginnote">MLE 的正态答案值得注意：<strong>$\hat\mu = \bar x$ 无偏，$\hat\sigma^2 = \frac1n\sum(x_i-\bar x)^2$ 有偏（用 $n$ 而非 $n-1$）</strong>。也就是说，最大似然并不自动给出无偏估计——「无偏」是额外想要的性质，需自行修正为 $S^2$。这正是下一节「估计量评选标准」要讨论的问题。</span>

## 4 公式解析：对数把连乘变连加

MLE 技术上的关键一步是对数变换，拆开看它为何必然有效：

$$

\ln L(\theta) = \ln\left(\prod_{i=1}^n f(x_i; \theta)\right) = \sum_{i=1}^n \ln f(x_i; \theta)

$$

- **第一步，对数法则**：$\ln(ab) = \ln a + \ln b$，把 $n$ 个因子的连乘变成 $n$ 项之和——求导从「积法则」降为「逐项求和」。
- **第二步，单调性**：$\ln$ 严格单调递增，故 $L(\theta)$ 的极大值点与 $\ln L(\theta)$ 的完全相同——最大化目标等价。
- **第三步，数值稳定**：连乘会下溢到 0（$n$ 大时概率积极小而失去精度），对数后是数量级之和，数值稳定——这也是机器学习的损失函数爱用负对数似然的原因。

「对数把乘法变加法、单调保持极值、数值更稳」三合一，让对数似然成为 MLE 计算的标准形式。工程里它几乎总被采用。

## 5 最大似然估计的深入应用与实例

MLE 是点估计的正统主力，它的流程、性质与现代意义值得完整展开。

### 例：指数分布的 MLE

**例**：$X \sim E(\lambda)$。$L = \lambda^n e^{-\lambda\sum x_i}$，$\ln L = n\ln\lambda - \lambda\sum x_i$，求导令零得 $\hat\lambda = 1/\bar x$——与矩估计一致。

### MLE 的流程

| 步骤 | 动作 |
| --- | --- |
| 写似然 | $L(\theta) = \prod f(x_i;\theta)$ |
| 取对数 | $\ln L$（连乘变连加） |
| 求导令零 | 对数似然方程 |
| 解方程 | 得 $\hat\theta$ |
| 验证 | 二阶导 < 0 |

**「写、取、导、解、验」**五步是 MLE 的标准流程。

### MLE 的性质

| 性质 | 内容 |
| --- | --- |
| 相合性 | $\hat\theta \xrightarrow{P} \theta$ |
| 渐近无偏 | $E[\hat\theta] \to \theta$ |
| 渐近正态 | $\sqrt n(\hat\theta-\theta) \to N(0, I^{-1})$ |
| 渐近有效 | 方差达克拉美—劳下界 |

**「MLE 大样本下集相合、无偏、正态、有效于一身」**——这是它的统治地位来源。

### 例：正态的 MLE

**例**：$X \sim N(\mu, \sigma^2)$。$\hat\mu = \bar x$（无偏）、$\hat\sigma^2 = \frac1n\sum(x_i-\bar x)^2$（有偏）——**「MLE 不保证无偏」**，方差估计需改用 $S^2$。

### 与机器学习的关系

交叉熵损失 = 负对数似然——**「训练模型 = 最大化似然」**是大模型与深度学习损失函数的设计根源。

**易错点｜辨析：** ① 似然是 $\theta$ 的函数（数据固定），不是 $\theta$ 的概率密度；② 对数似然与似然同极值点（$\ln$ 单调）；③ 均匀分布的 MLE 在边界取到（参数约束支撑），不能求导——「先判断参数是否进入支撑」。

## 6 小结

- **似然函数** $L(\theta) = \prod_i f(x_i; \theta)$：固定数据、把 $\theta$ 当自变量，衡量「数据在此参数下的合理性」。
- **MLE**：选使 $L(\theta)$ 最大的 $\hat\theta$——「已经发生的，理应最可能」。
- 流程：写似然 → 取对数 → 求导令零 → 解方程；多参数时列方程组。
- 指数分布 MLE = $1/\bar x$；正态 MLE = $(\bar x, \frac1n\sum(x_i-\bar x)^2)$——$\hat\sigma^2$ 有偏。
- MLE 是点估计的正统主力，也是机器学习「训练模型」的数学源头。

在下一节，我们把 MLE 用到具体分布上——**常见分布参数的最大似然估计**。
