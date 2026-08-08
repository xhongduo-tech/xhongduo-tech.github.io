---
title: 带偏置项的矩阵分解（BiasSVD）
date: 2026-08-07
---

# 带偏置项的矩阵分解（BiasSVD）

<div class="epigraph">
<p>所有模型都是错的，但有些是有用的。</p>
<footer>—— 乔治 · 博克斯（George E. P. Box）</footer>
</div>

<div class="article-byline">
<p>第四级 · 推荐系统 ｜ 项亮《推荐系统实践》第2章 §2.5 隐语义模型 ｜ 2026-08-07</p>
</div>

## 为什么从偏置项开始

上一篇的 Funk-SVD 预测 $\hat r_{ui} = p_u^{\mathsf{T}} q_i$，把「用户 $u$ 对物品 $i$ 的评分」完全归因于两者隐因子的交互。可现实里的评分在交互发生之前，就带着浓重的「底色」：有人天生慷慨，给什么都偏高；有些电影口碑一致、评分却整体偏低。这些底色与「用户到底喜不喜欢这个物品」无关，却占走了评分方差的一大块。

BiasSVD 的做法朴素而有效：**先把可解释的底色（偏置）显式拆出来，剩下的「残差」再交给隐因子交互去消化。** 这个改动在 Netflix 数据上能显著压低 RMSE，也是后一篇 SVD++ 的底座。理解它，是从「能跑的矩阵分解」走向「工业级矩阵分解」的第一步。<span class="marginnote">统计里有个常用套路：观测 = 全局基线 + 结构化偏差 + 噪声。BiasSVD 正是把「全局基线 + 可解释偏差」显式建模，让矩阵分解专心拟合残余的交互信号——你在第二级《线性代数》与第三级《机器学习》里反复见到的「先去掉均值、再建模波动」也同出一辙。</span>

## 1 评分里的系统性偏差

把任意一条评分 $r_{ui}$ 拆开，至少有三个「与交互无关」的成分：

**全局均值 $\mu$**：整个数据集评分的平均水平。Netflix 上约为 3.6 分（满分 5）。这是最粗糙、也最可靠的预测——不知道任何信息时，猜全局均值就是最优解。
**用户偏置 $b_u$**：用户 $u$ 的「打分尺度」。有人平均给 4.2，有人平均给 2.8，$b_u$ 刻画「用户 $u$ 相对全局均值的系统性偏移」。
**物品偏置 $b_i$**：物品 $i$ 的「被评价水平」。神作整体偏高、烂片整体偏低，$b_i$ 刻画「物品 $i$ 相对全局均值的系统性偏移」。

**辨析｜易错点：** $b_u$ 不等于「用户 $u$ 的评分均值减去 $\mu$」那么直接。因为爱打高分的人往往也恰好选了好片，用户均分与物品质量纠缠在一起。正确的做法是**联合估计**：让 $b_u$ 与 $b_i$ 在全部数据上交替迭代拟合，或用带正则的收缩估计。只做「均值差」，会把物品的功劳错记到用户头上。<span class="marginnote">收缩估计（shrinkage）的形式是 $b_i = \frac{\sum_{u \in R(i)} (r_{ui} - \mu)}{\lambda + |R(i)|}$。物品被评的次数越少，$b_i$ 越被拉回 0——少数几条评分不足以支撑一个可靠的偏置。这是「正则 = 向先验收缩」思想的又一次现身。</span>

## 2 基线预测：把非交互成分固定住

把三个偏置相加，得到**基线预测（baseline estimate）**：

$$
\hat b_{ui} = \mu + b_u + b_i
$$

这条式子本身就是一个能用的推荐器——它没用到任何「用户—物品交互」，却往往远强于随机猜。它的作用是**定基准**：真正值得预测的，是基线之外的那部分。

BiasSVD 的完整预测式，就是在基线上叠加隐因子交互：

$$
\hat r_{ui} = \mu + b_u + b_i + q_i^{\mathsf{T}} p_u
$$

**重点：** 一旦偏置被显式建模，隐因子 $p_u$、$q_i$ 就不再需要「背着评分尺度」干活——它们只负责刻画「$u$ 对 $i$ 的偏爱超出平均水准的部分」。职责分离让隐向量更干净、更可解释，也直接压低 RMSE。

## 3 损失函数：把偏置也纳入正则

和 Funk-SVD 一样，只在已知评分集合 $\mathcal K$ 上优化，但**偏置项同样被正则**：

$$
L = \sum_{(u,i) \in \mathcal K} \left( r_{ui} - \mu - b_u - b_i - q_i^{\mathsf{T}} p_u \right)^2
+ \lambda \Big( b_u^2 + b_i^2 + \|p_u\|^2 + \|q_i\|^2 \Big)
$$

**辨析｜易错点：** 一个只有 3 条评分的用户，其 $b_u$ 若不约束，会为了把这三条拟合完美而疯狂偏离。偏置项与隐因子一样会过拟合，必须一起正则。<span class="marginnote">$\lambda$ 在这里是「所有参数统一的正则系数」。更精细的做法是给偏置和隐因子分别设 $\lambda_b$ 与 $\lambda_f$——工程上常让偏置的 $\lambda$ 更小，因为偏置的可信先验更弱。</span>

## 4 公式解析：BiasSVD 的梯度更新

误差定义为

$$
e_{ui} = r_{ui} - \hat r_{ui} = r_{ui} - \mu - b_u - b_i - q_i^{\mathsf{T}} p_u
$$

对单个样本的损失 $\ell = e_{ui}^2 + \lambda(b_u^2 + b_i^2 + \|p_u\|^2 + \|q_i\|^2)$ 求偏导：

- 对 $b_u$：$\dfrac{\partial \ell}{\partial b_u} = -2e_{ui} + 2\lambda b_u$
- 对 $b_i$：$\dfrac{\partial \ell}{\partial b_i} = -2e_{ui} + 2\lambda b_i$
- 对 $p_u$：$\dfrac{\partial \ell}{\partial p_u} = -2e_{ui}\, q_i + 2\lambda\, p_u$
- 对 $q_i$：$\dfrac{\partial \ell}{\partial q_i} = -2e_{ui}\, p_u + 2\lambda\, q_i$

沿负梯度走步长 $\eta$，得到四组更新规则：

$$
b_u \leftarrow b_u + \eta(e_{ui} - \lambda b_u), \qquad b_i \leftarrow b_i + \eta(e_{ui} - \lambda b_i)
$$

$$
p_u \leftarrow p_u + \eta(e_{ui}\, q_i - \lambda p_u), \qquad q_i \leftarrow q_i + \eta(e_{ui}\, p_u - \lambda q_i)
$$

逐步拆解：

- **第一步，算误差 $e_{ui}$**：预测减真实。注意这里的预测已经含了三个偏置。
- **第二步，更新偏置**：$b_u$ 沿着误差方向走。误差为正说明预测偏低，$b_u$ 增大；正则项 $\lambda b_u$ 把它往 0 拽，防止极端用户极端化。
- **第三步，更新 $p_u$、$q_i$**：与 Funk-SVD 完全同形——**误差乘上对方的向量**，信号沿对方方向传播。
- **第四步，观察结构**：$p_u$、$q_i$ 的更新式里没有 $b_u$、$b_i$，偏置的更新式里也没有 $p_u$、$q_i$。**四类参数互不耦合**，这让我们可以分开初始化、分步训练，也让代码只需在 Funk-SVD 上增补几行。

**辨析｜易错点：** 更新 $b_u$ 用的 $e_{ui}$，已经扣掉了 $\mu + b_i + q_i^{\mathsf{T}} p_u$。若有人在实现里把「用户均分」当 $b_u$ 写死、不再训练，就丢掉了联合估计的纠偏能力。$b_u$、$b_i$ 必须是**可训练参数**，与其他参数同步 SGD 迭代。

## 5 从公式到代码

Funk-SVD 的训练循环只加三行就能升级成 BiasSVD：

```python
def bias_svd(ratings, K=20, lr=0.01, reg=0.1, epochs=20):
    """ratings: [(u, i, r), ...]。在 Funk-SVD 上加全局均值 μ 与偏置 b_u、b_i。"""
    users, items = {u for u, _, _ in ratings}, {i for _, i, _ in ratings}
    mu = np.mean([r for _, _, r in ratings])          # 全局均值
    b_u = {u: 0.0 for u in users}                      # 偏置从 0 起步
    b_i = {i: 0.0 for i in items}
    P = {u: np.random.normal(0, 0.1, K) for u in users}   # 隐因子仍要随机化
    Q = {i: np.random.normal(0, 0.1, K) for i in items}

    for _ in range(epochs):
        for u, i, r in ratings:
            e = r - (mu + b_u[u] + b_i[i] + P[u] @ Q[i])  # 误差含偏置
            b_u[u] += lr * (e - reg * b_u[u])             # 四组更新共用同一个 e
            b_i[i] += lr * (e - reg * b_i[i])
            P[u]   += lr * (e * Q[i] - reg * P[u])
            Q[i]   += lr * (e * P[u] - reg * Q[i])
    return mu, b_u, b_i, P, Q
```

预测时一条指令：`mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])`。<span class="marginnote">注意初始化哲学：偏置从 <strong>0</strong> 起步（没有先验偏移，0 是合理起点），而隐因子必须随机化来打破对称——这是第三级《机器学习》里「对称破坏」的老朋友了。</span>

## 6 一个直觉例子

设全局均值 $\mu = 3.6$。用户 A 打分慷慨，$b_A = +0.4$；物品 M 是部晦涩的文艺片，普遍偏低，$b_M = -0.5$。于是 A 给 M 的基线预测是 $3.6 + 0.4 - 0.5 = 3.5$。若 A 实际给了 4 分，残差 $+0.5$ 就交给 $q_M^{\mathsf{T}} p_A$ 去解释——「A 在平均之上，确实偏爱这种调性」。

**重点：** 没有偏置项时，$p_A$ 得同时扛下「A 打分偏高」和「A 偏爱文艺片」两件事；有了偏置，前者归 $b_u$，后者归隐因子。**每个参数各司其职，模型才谈得上泛化。**

## 7 小结

- 评分可拆为 **全局均值 $\mu$ + 用户偏置 $b_u$ + 物品偏置 $b_i$ + 交互残差**。
- **基线预测** $\hat b_{ui} = \mu + b_u + b_i$ 不涉及交互，是天然的基准线。
- **BiasSVD** 预测式 $\hat r_{ui} = \mu + b_u + b_i + q_i^{\mathsf{T}} p_u$ 把隐因子从评分尺度中解放出来，通常显著压低 RMSE。
- 偏置项**同样要正则**；$b_u$、$b_i$ 是**可训练参数**，与隐因子同步 SGD、彼此解耦。
- 工程上偏置初始化为 0、隐因子随机初始化；四类参数互不耦合，便于分步训练。

在下一节，我们要回答一个 Funk-SVD 与 BiasSVD 都回避的问题：**如果数据里根本没有评分，只有「看过/没看过、看了多少次」，矩阵分解还能用吗？** 加权交替最小二乘（ALS-WR）给出了答案——它也是 Spark MLlib 里默认的推荐实现。
