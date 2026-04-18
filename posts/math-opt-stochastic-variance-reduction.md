## 核心结论

随机优化的方差缩减，是用历史梯度或快照梯度构造控制变量，降低随机梯度噪声。控制变量是一个修正项，用来抵消随机采样带来的波动。

核心结论很直接：方差缩减不是换一个优化目标，而是给随机梯度加一个校准项，让更新方向更接近全梯度方向。

普通 SGD 每次只看一小撮样本就决定往哪走，所以单步便宜，但方向容易偏。方差缩减方法像是先算一个可靠的参照方向，再用当前样本做微调。它保留了随机梯度单步计算便宜的优点，同时显著改善后期收敛，特别适合固定训练集、大样本量、普通 SGD 后期抖动明显的场景。

常见方法包括 SVRG、SAG、SAGA。它们都服务于同一个目标：在不每步计算全量梯度的情况下，让随机梯度估计更稳定。理想状态下，构造出的随机方向 $v_t$ 满足：

\[
\mathbb E[v_t\mid w_t]=\nabla F(w_t)
\]

这表示 $v_t$ 在条件期望上等于真实全梯度。条件期望是指在当前参数 $w_t$ 已知的情况下，对随机抽样结果取平均。

| 方法 | 单步成本 | 噪声大小 | 典型特点 |
|---|---:|---:|---|
| SGD | 低 | 大 | 实现简单，后期容易抖动 |
| SVRG / SAGA | 低到中 | 较小 | 用控制变量降低随机波动 |
| 全梯度法 | 高 | 最小 | 每步稳定，但每次要扫全数据 |

---

## 问题定义与边界

方差缩减方法主要研究有限和目标函数。有限和目标函数是指整体目标可以拆成有限个样本损失的平均：

\[
F(w)=\frac1n\sum_{i=1}^n f_i(w)
\]

其中 $w$ 是模型参数，$f_i(w)$ 是第 $i$ 个样本对应的损失函数。很多监督学习问题都符合这个形式，例如线性回归、逻辑回归、线性分类器训练。

更一般地，问题可以写成：

\[
\min_w F(w)+h(w)
\]

其中 $h(w)$ 可以表示非光滑正则项。非光滑是指函数在某些点没有普通导数，例如 $L_1$ 正则 $h(w)=\lambda\|w\|_1$ 在 $0$ 附近不可导。

这里的关键边界是：SVRG、SAGA、SAG 的优势依赖 finite-sum 结构。finite-sum 就是上面这种“固定数量样本损失求平均”的结构。如果数据是持续流入的在线数据，样本总数不固定，或者不能反复访问旧样本，那么很多方差缩减公式的前提就不完整。

| 场景 | 是否适合 | 原因 |
|---|---|---|
| 固定训练集的 finite-sum 目标 | 适合 | 可以反复访问样本并计算历史或快照梯度 |
| 在线学习 | 理论优势弱 | 样本不断变化，难以维护稳定基准 |
| 强稀疏高维数据 | 需要评估 | 梯度表和缓存更新可能很贵 |
| 非光滑正则 | 适合但需扩展 | 通常用近端算子处理 |

快照点 $\tilde w$ 是周期性计算全局梯度时保存下来的参数。它像一个固定参照点，用来构造修正项。控制变量则是用来抵消随机噪声的修正项，它不改变优化目标，只改变梯度估计方式。

新手可以这样理解：如果目标是最小化全体样本的平均损失，SGD 每次只看一个样本，所以方向波动很正常；如果你的数据是一个没有固定结尾的在线日志流，那么“全体样本平均损失”本身就在变化，SVRG/SAGA 的经典假设就不再完全成立。

---

## 核心机制与推导

从 SGD 开始。普通 SGD 在第 $t$ 步随机抽一个样本 $i_t$，用

\[
\nabla f_{i_t}(w_t)
\]

近似全梯度 $\nabla F(w_t)$。它通常是无偏的，但方差可能很大。无偏是指平均方向正确；方差大是指每一次具体抽样的方向可能偏得很远。

SVRG 的核心构造是：

\[
v_t=\nabla f_{i_t}(w_t)-\nabla f_{i_t}(\tilde w)+\mu,\quad
\mu=\frac1n\sum_{i=1}^n \nabla f_i(\tilde w)
\]

然后更新：

\[
w_{t+1}=w_t-\eta v_t
\]

其中 $\eta$ 是学习率。这个公式可以拆成三部分：

| 项 | 含义 |
|---|---|
| $\nabla f_{i_t}(w_t)$ | 当前样本在当前位置的梯度 |
| $-\nabla f_{i_t}(\tilde w)$ | 减掉该样本在快照点的旧梯度 |
| $+\mu$ | 加回快照点处的全局平均梯度 |

玩具例子：令

\[
f_1(w)=\frac12(w-1)^2,\quad f_2(w)=\frac12(w+1)^2
\]

则

\[
F(w)=\frac12(w^2+1),\quad \nabla F(w)=w
\]

设快照点 $\tilde w=1$，则 $\nabla f_1(1)=0$，$\nabla f_2(1)=2$，所以 $\mu=1$。当前点 $w_t=2$ 时：

\[
\nabla f_1(2)=1,\quad \nabla f_2(2)=3
\]

若抽到样本 1：

\[
v_t=1-0+1=2
\]

若抽到样本 2：

\[
v_t=3-2+1=2
\]

两种抽样都得到真实全梯度 $2$。这只是一个简单例子，不代表所有问题都能完全消除方差，但它展示了控制变量如何抵消波动。

无偏性证明也直接：

\[
\begin{aligned}
\mathbb E[v_t\mid w_t]
&=\mathbb E[\nabla f_{i_t}(w_t)-\nabla f_{i_t}(\tilde w)+\mu] \\
&=\frac1n\sum_{i=1}^n\nabla f_i(w_t)-\frac1n\sum_{i=1}^n\nabla f_i(\tilde w)+\mu \\
&=\nabla F(w_t)-\mu+\mu \\
&=\nabla F(w_t)
\end{aligned}
\]

SAGA 不使用周期性快照，而是为每个样本维护一个旧梯度 $g_i$：

\[
v_t=\nabla f_{i_t}(w_t)-g_{i_t}+\frac1n\sum_{j=1}^n g_j
\]

并更新：

\[
g_{i_t}\leftarrow \nabla f_{i_t}(w_t)
\]

SAGA 的基准不是一个快照点，而是一张持续更新的梯度表。它每次只替换一个样本的旧梯度，同时维护所有旧梯度的平均值。

| 方法 | 机制 | 是否无偏 | 主要状态 |
|---|---|---|---|
| SVRG | 快照梯度 + 外循环 | 是 | 快照点、快照全梯度 |
| SAG | 历史梯度平均 | 通常不是严格无偏 | 每个样本的旧梯度 |
| SAGA | 历史梯度平均 + 无偏修正 | 是 | 每个样本的旧梯度及均值 |

如果目标里有非光滑正则项 $h(w)$，更新通常改为近端步：

\[
w_{t+1}=\operatorname{prox}_{\eta h}(w_t-\eta v_t)
\]

近端算子是处理非光滑正则的一种更新规则，可以理解为“先按梯度走一步，再按正则项做一次修正”。对于 $L_1$ 正则，它会产生稀疏参数，也就是把一些参数压到 $0$。

---

## 代码实现

实现时先分清两类状态。SVRG 需要外循环和快照点；SAGA 需要梯度表和梯度平均值。

| 元素 | 含义 |
|---|---|
| 输入 | 数据集、损失函数、学习率、外循环次数或总迭代次数 |
| 状态 | 参数 $w$、快照点 $\tilde w$、梯度表 $g_i$ |
| 输出 | 收敛后的参数 |

SVRG 伪代码：

```text
initialize w
repeat:
    snapshot = w
    mu = full_gradient(snapshot)
    for t in 1..m:
        sample i
        v = grad_f_i(w) - grad_f_i(snapshot) + mu
        w = w - eta * v
```

SAGA 伪代码：

```text
initialize w and gradient table g_i
initialize avg_g
for t in 1..T:
    sample i
    new_g = grad_f_i(w)
    v = new_g - g_i + avg_g
    update avg_g
    g_i = new_g
    w = w - eta * v
```

下面是一个最小 NumPy 版 SVRG，用一维平方损失演示结构。它不是高性能实现，重点是展示数据流：采样、控制变量、更新参数。

```python
import numpy as np

def grad_i(w, a_i):
    # f_i(w) = 0.5 * (w - a_i)^2, grad = w - a_i
    return w - a_i

def full_grad(w, data):
    return np.mean([grad_i(w, a) for a in data])

def svrg_1d(data, w0=5.0, eta=0.1, outer_loops=8, inner_steps=20, seed=0):
    rng = np.random.default_rng(seed)
    w = float(w0)
    n = len(data)

    for _ in range(outer_loops):
        snapshot = w
        mu = full_grad(snapshot, data)

        for _ in range(inner_steps):
            i = rng.integers(n)
            v = grad_i(w, data[i]) - grad_i(snapshot, data[i]) + mu
            w = w - eta * v

    return w

data = np.array([1.0, -1.0])
w = svrg_1d(data)
assert abs(w) < 1e-3

# 对这个玩具问题，F(w)=0.5*(w^2+1)，最优解是 w=0
assert abs(full_grad(0.0, data)) == 0.0
```

SVRG 和 SAGA 的工程差异主要来自状态维护：

| 方法 | 是否存全量梯度表 | 内存开销 | 更新方式 | 适合数据规模 |
|---|---|---:|---|---|
| SVRG | 否 | 较低 | 周期性全量快照 + 内循环随机更新 | 大数据、内存紧张 |
| SAGA | 是 | 较高 | 每步更新一个样本的历史梯度 | 固定数据集、可承受缓存 |
| SAG | 是 | 较高 | 用历史梯度平均更新 | 固定数据集、强调增量更新 |

真实工程例子是大规模稀疏逻辑回归，例如文本分类、CTR 预估、推荐排序。每个样本只激活少量特征，训练集固定且很大，普通 SGD 后期可能明显抖动。此时 SAGA 可以利用历史梯度信息降低噪声，并配合 $L_1$ 或 elasticnet 正则得到稀疏模型。

---

## 工程权衡与常见坑

方差缩减不是无脑更快。它降低了梯度噪声，但引入了额外状态：SVRG 要周期性计算快照全梯度，SAGA 要维护梯度表。工程上需要在速度、内存、实现复杂度之间取平衡。

| 常见坑 | 结果 | 处理方式 |
|---|---|---|
| 目标不是 finite-sum | 理论优势下降 | 优先考虑 SGD 或在线方法 |
| 特征尺度差异大 | 收敛慢或不稳定 | 先做标准化 |
| SAGA 存梯度表 | 内存可能达到 $O(n)$ 或更高 | 评估缓存成本，必要时选 SVRG |
| 快照太久不更新 | 控制变量过期 | 缩短外循环或调小内循环步数 |
| 学习率过大 | 仍然发散 | 方差缩减不等于无限加大学习率 |

特征标准化尤其重要。新手常见错误是直接把不同尺度的特征丢进 `sag` 或 `saga`。如果一列特征范围是 $1$ 到 $10$，另一列是 $0$ 到 $10000$，大尺度特征会主导梯度，算法看起来就会“不稳定”。这不是方差缩减公式失效，而是优化问题的数值条件很差。

为什么快照过旧会慢？因为控制变量依赖基准梯度。当前参数 $w_t$ 离快照点 $\tilde w$ 越远，$\nabla f_i(w_t)-\nabla f_i(\tilde w)$ 的修正越可能变大，基准全梯度 $\mu$ 对当前方向的校准能力会下降。

为什么稀疏高维时要小心？稀疏数据本身有利于快速单步更新，但如果梯度表按密集向量存储，内存会迅速膨胀。一个有千万样本、百万特征的任务，不可能朴素存储每个样本的完整梯度向量。实际实现通常需要稀疏结构、延迟更新或专门优化。

调参顺序建议：

| 步骤 | 目的 |
|---|---|
| 先标准化特征 | 改善数值条件 |
| 再选学习率 | 避免发散或过慢 |
| 再定快照频率 | 平衡全量扫描成本和控制变量质量 |
| 最后检查内存 | 确认梯度表或缓存可承受 |

---

## 替代方案与适用边界

方差缩减要放在随机优化方法谱系里理解。它不是 SGD、Adam、全梯度法的绝对替代品，而是适合一类明确问题：固定数据集、目标可拆成样本损失平均、希望单步便宜但后期更稳。

| 方法 | 优点 | 局限 | 适用场景 |
|---|---|---|---|
| SGD | 实现最简单，适合在线数据 | 后期噪声大 | 在线学习、超大规模流式训练 |
| Momentum / Adam | 深度学习中常用，调参经验丰富 | 不属于 finite-sum 方差缩减 | 神经网络训练 |
| SVRG | 内存较省，后期更稳 | 需要周期性全量扫描 | 大规模固定数据集 |
| SAGA | 支持无偏修正，适合正则化问题 | 需要维护梯度表 | 稀疏逻辑回归、$L_1$/elasticnet |
| 全梯度法 | 每步方向稳定 | 每步成本高 | 中小数据集、全量计算可接受 |

选择建议：

| 条件 | 推荐 |
|---|---|
| 固定数据集，目标函数可分解 | 优先考虑 SVRG / SAGA |
| 内存紧张 | 优先 SVRG 类 |
| 需要稀疏解 | 优先 SAGA |
| 在线或非定长数据 | 优先 SGD 或其他在线方法 |
| 数据规模不大 | 全梯度法或拟牛顿法也可考虑 |

新手版判断规则：如果你的任务是每天不断涌入新数据的在线系统，没法频繁做全量快照，也很难维护稳定的历史梯度表，那么 SVRG/SAGA 的优势会被打折。如果是固定训练集上的大规模分类任务，例如文本分类或广告点击率预估，它们通常更有竞争力。

---

## 参考资料

原始论文：

- [Johnson & Zhang, 2013, SVRG 原始论文](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction)
- [Le Roux, Schmidt & Bach, 2012, SAG 原始论文](https://papers.nips.cc/paper/4633-a-stochastic-gradient-method-with-an-exponential-convergence-_rate-for-finite-training-sets)
- [Defazio, Bach & Lacoste-Julien, 2014, SAGA 原始论文](https://papers.nips.cc/paper/5258-saga-a-fast-incremental-gradient-method-with-support-for-non-strongly-convex-composite-objectives)

工程文档：

- [scikit-learn `LogisticRegression` 文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn：SAGA 处理 L1 多项逻辑回归示例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html)

scikit-learn 文档把 `sag` 和 `saga` 放在大数据场景下更快的位置，并明确 `saga` 支持 `l1`、`elasticnet` 和 multinomial。阅读顺序建议是：先看定义和公式，再看 SVRG、SAG、SAGA 的算法差异，最后看工程实现和适用边界。
