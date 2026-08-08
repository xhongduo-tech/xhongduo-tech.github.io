---
title: 最大熵模型的推导：矩约束下的分布
date: 2026-08-07
---

# 最大熵模型的推导：矩约束下的分布

<div class="epigraph">
<p>约束是已知的信息，拉格朗日乘子是它的价格，指数族是市场出清后的均衡分布。</p>
<footer>—— 埃德温 · 杰恩斯（Edwin T. Jaynes）</footer>
</div>

<div class="article-byline">
<p>第二级 · 信息论 ｜ Cover &amp; Thomas《Elements of Information Theory》 §12.1 ｜ 2026-08-07</p>
</div>

## 为什么从「把推导走到底」开始

上一篇用「拉格朗日一击」得到了「MaxEnt 解是指数族」，但那一击有多处值得展开：约束怎么写的、乘子怎么解的、配分函数从哪来、解的存在性由什么保证。

这一篇我们把**最大熵模型（maximum entropy model）**的推导完整走一遍：从矩约束出发，经拉格朗日对偶，得到指数族分布，并看清「配分函数」与「矩匹配」这两个贯穿统计物理与机器学习的核心结构。

推导的结果将是下一篇的钥匙——**最大熵模型与逻辑回归是同一件事**：给分类问题选特征，最大熵模型的解就是一个逻辑回归。<span class="marginnote">最大熵模型的完整推导在 Cover &amp; Thomas §12.1。它是「指数族统计」与「信息论」的交汇：信息论提供「熵最大」的目标，统计提供「配分函数与矩匹配」的工具——两者在拉格朗日对偶处合流。</span>

## 1 设定：矩约束下的优化

**设定**：随机变量 $X \in \mathcal{X}$（有限集），已知 $k$ 个特征函数 $f_1, \dots, f_k$ 的期望：

$$
\mathbb{E}_p[f_j(X)] = \mu_j, \qquad j = 1, \dots, k
$$

例如：$f_1(x) = x$（一阶矩）、$f_2(x) = x^2$（二阶矩）、或「$x$ 满足某条件」的指示函数。

**优化问题**：

$$
\max_p \ H(p) = -\sum_x p(x)\log p(x), \qquad \text{s.t. } \sum_x p(x) f_j(x) = \mu_j,\ \sum_x p(x) = 1
$$

可行集 = 满足矩约束的概率分布（凸集）；目标熵是凹函数——**凹函数在凸集上最大化，最优解存在且唯一（严格凹时）**。<span class="marginnote">「可行集凸 + 目标凹」是 MaxEnt 好解的数学保证：不存在「局部最优陷阱」，KKT 条件就是全局最优。这解释了为什么 MaxEnt 模型训练稳定——它落在「凸优化最舒服的地形」上。</span>

## 2 拉格朗日与对偶

**拉格朗日**：

$$
L(p, \lambda, \lambda_0) = -\sum_x p(x)\log p(x) + \sum_{j=1}^k \lambda_j \Big(\sum_x p(x) f_j(x) - \mu_j\Big) + \lambda_0\Big(\sum_x p(x) - 1\Big)
$$

**对 $p(x)$ 求导置零**（对每个 $x$）：

$$
-\log p(x) - 1 + \sum_j \lambda_j f_j(x) + \lambda_0 = 0
$$

**解出**：

$$
p(x) = \frac{1}{Z(\lambda)} \exp\Big(\sum_{j=1}^k \lambda_j f_j(x)\Big)
$$

其中 $Z(\lambda) = \sum_x e^{\sum_j \lambda_j f_j(x)}$ 是**配分函数（partition function）**，由归一化约束 $\sum p = 1$ 确定。$\lambda_0 = \log Z(\lambda) - 1$ 是「归一化乘子」。<span class="marginnote">「配分函数」这个名字来自统计物理：$Z(\lambda)$ 是所有「状态的权重和」，它把未归一化的指数权重转成概率。物理学家叫它 partition function，统计学家叫它归一化常数，机器学习里叫它「log-sum-exp」——同一个东西，三个名字。</span>

## 3 公式解析：乘子由「矩匹配」确定

乘子 $\lambda$ 由约束 $\mathbb{E}[f_j] = \mu_j$ 确定。代入指数族形式，约束变成：

$$
\mathbb{E}_{p_\lambda}[f_j(X)] = \sum_x \frac{e^{\sum \lambda_j f_j(x)}}{Z(\lambda)} f_j(x) = \mu_j
$$

**关键性质：梯度 = 矩**。

$$
\frac{\partial \log Z(\lambda)}{\partial \lambda_j} = \mathbb{E}_{p_\lambda}[f_j(X)]
$$

于是「找满足约束的 $\lambda$」等价于「调整 $\lambda$ 直到模型矩匹配经验矩」——**矩匹配（moment matching）**。

**对偶问题**：最大化熵 $H(p_\lambda)$ 等价于最小化对数配分函数：

$$
\min_\lambda \ \log Z(\lambda) - \sum_j \lambda_j \mu_j
$$

这是一个**凸优化**（$\log Z$ 关于 $\lambda$ 凸），梯度 = $\mathbb{E}[f_j] - \mu_j$（矩误差），可以用梯度下降求解。<span class="marginnote">「$\partial_\lambda \log Z = \mathbb{E}[f_j]$」是统计里最漂亮的一个等式：配分函数的对数梯度恰好是矩。这让训练变成「把模型矩推向数据矩」的梯度下降——每个特征一个梯度，直观且可扩展。它也是「指数族 = 矩匹配」这一统计规律的核心。</span>

## 4 完整推导：五步走

把整个推导串成五步，形成可复用的模板：

1. **写目标与约束**：$\max H(p)$，矩约束 $\mathbb{E}[f_j] = \mu_j$。
2. **写拉格朗日**：目标 + 乘子 × 约束。
3. **对 $p$ 求导置零**：解出 $p(x) \propto e^{\sum \lambda_j f_j(x)}$。
4. **归一化**：引入配分函数 $Z(\lambda)$。
5. **解乘子**：用矩匹配 $\mathbb{E}_{p_\lambda}[f_j] = \mu_j$（或对偶凸优化）。

**辨析｜易错点：** 三个容易错的地方：

**特征与矩是一一对应的**：每个约束 $\mathbb{E}[f_j] = \mu_j$ 产生一个指数项 $\lambda_j f_j(x)$。特征越多，模型越灵活，但也越容易过拟合。
**矩匹配是「经验矩」**：实际数据只给「样本矩」$\hat \mu_j = \frac1n \sum f_j(x_i)$，MaxEnt 让模型矩匹配样本矩——这是「拟合」的实质。
**解的支撑集**：指数族在所有 $\mathcal{X}$ 上正概率——它不会给出「概率 0」的预测。这在某些任务里是优点（光滑）、在某些里是缺点（不能硬排除）。<span class="marginnote">「指数族永不给零概率」是把双刃剑：语言建模里它保证了「所有词都有非零概率」（平滑），但也会给「不合理的词」留尾巴。现代语言模型用「温度、截断」等手段在后处理阶段补这个「零概率」能力——模型本身做不到硬排除。</span>

**一个具体例子**：$\mathcal{X} = \{0, 1\}$，约束 $\mathbb{E}[X] = \mu$。

- 指数族：$p(1) = \frac{e^{\lambda}}{1 + e^{\lambda}}$，$p(0) = \frac{1}{1+e^{\lambda}}$。
- 矩匹配：$p(1) = \mu$ ⇒ $\lambda = \log\frac{\mu}{1-\mu}$。
- 结果：$p(1) = \mu$——**最大熵解就是「给定均值最诚实的伯努利分布」**，与直觉完全一致。

**与全课程体系的连接：** 最大熵模型的推导是「指数族统计」的入口（第二级《概率论与数理统计》、第四级《机器学习》的广义线性模型）；它的「矩匹配」思想与 EM 算法的「矩 vs 似然」视角呼应；下一篇把它接到逻辑回归上。

## 5 小结

- **最大熵模型**：矩约束下最大化熵，解为指数族 $p(x) \propto e^{\sum \lambda_j f_j(x)}$。
- 推导五步：目标+约束 → 拉格朗日 → 求导 → 归一化（配分函数）→ 矩匹配解乘子。
- **配分函数** $Z(\lambda)$：归一化常数，对数梯度 = 矩。
- **矩匹配**：$\mathbb{E}_{p_\lambda}[f_j] = \mu_j$；对偶问题 $\min \log Z - \sum \lambda_j \mu_j$ 是凸优化。
- **辨析**：特征与矩一一对应；匹配的是经验矩；指数族永不给零概率。
- 二元例子：约束均值 ⇒ 伯努利，$p(1) = \mu$，直觉吻合。

在下一篇，我们揭开最大熵模型最著名的身份：**最大熵模型与逻辑回归的关系**——为什么「选特征做分类」的最诚实答案就是逻辑回归。
