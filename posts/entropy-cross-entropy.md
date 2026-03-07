## 核心结论

信息熵、交叉熵、KL 散度是同一件事的三个视角：真实分布本身有多难预测、你用错分布时要多付出多少代价、以及这份额外代价到底有多大。

香农熵定义为

$$
H(p) = -\sum_x p(x)\log p(x)
$$

它衡量真实分布 $p$ 的平均不确定性。这里的“不确定性”可以直接理解成：你在看到结果前，平均还差多少信息才能确定答案。分布越均匀，熵越高；分布越集中，熵越低。

交叉熵定义为

$$
H(p,q) = -\sum_x p(x)\log q(x)
$$

它衡量当真实世界服从 $p$，但你用模型分布 $q$ 去描述或编码它时的平均代价。白话说，真实答案来自 $p$，你却拿 $q$ 当“信念”去下注，下注越偏，损失越大。

KL 散度定义为

$$
KL(p\parallel q)=\sum_x p(x)\log \frac{p(x)}{q(x)} = H(p,q)-H(p)
$$

它表示“用错模型”带来的额外成本。因为 $H(p)$ 对训练来说是常数，所以最小化交叉熵，等价于最小化 $KL(p\parallel q)$。

玩具例子：真实硬币正反概率是 $p=(0.8,0.2)$，你却认为是 $q=(0.6,0.4)$。那么

- 真实熵 $H(p)\approx 0.50$ nats
- 交叉熵 $H(p,q)\approx 0.59$ nats
- KL 散度 $\approx 0.09$ nats

意思是：你每观察一次这个硬币，平均会多付出约 $0.09$ 个自然对数单位的信息代价。这个“多付出的部分”就是模型和真实分布之间的偏差。

---

## 问题定义与边界

讨论损失函数时，必须先固定两个对象：

| 概念 | 输入 | 输出含义 | 是否可训练 | 边界条件 |
|---|---|---|---|---|
| 香农熵 $H(p)$ | 真实分布 $p$ | 真实源本身的不确定性 | 否 | 当 $p(x)=0$ 时，按极限定义 $0\log 0=0$ |
| 交叉熵 $H(p,q)$ | 真实分布 $p$、模型分布 $q$ | 用 $q$ 描述 $p$ 的平均代价 | 是，对 $q$ 优化 | 若 $q(x)\to 0$ 且 $p(x)>0$，损失趋向无穷大 |
| KL 散度 $KL(p\parallel q)$ | 真实分布 $p$、模型分布 $q$ | 模型相对真实分布的额外代价 | 是，对 $q$ 优化 | 非负，且仅当 $p=q$ 时为 0 |

这里的关键边界是“非对称性”。$KL(p\parallel q)$ 和 $KL(q\parallel p)$ 一般不相等。白话说，“拿模型去逼近真实世界”和“拿真实世界去逼近模型”不是一回事。训练分类器时，标签或经验分布代表“真实分布”，模型输出代表“待优化分布”，因此方向必须固定成 $KL(p\parallel q)$，而不是反过来。

在监督学习里，真实标签常常写成 one-hot 向量。one-hot 的意思是：只有正确类别概率为 1，其他类别为 0。此时交叉熵会退化成非常简单的形式：

$$
H(p,q) = -\log q(y)
$$

其中 $y$ 是真实类别。这说明分类任务本质上在做一件事：提高模型对正确类别的概率估计。

还有一个工程边界是数值稳定性。因为要算 $\log q(x)$，如果 $q(x)=0$，损失会直接发散。因此实现里通常不会先算 `softmax` 再单独 `log`，而是直接用 `log-softmax`，或者至少把概率夹到 $[\varepsilon,1-\varepsilon]$ 区间。

---

## 核心机制与推导

先看多分类模型。模型输出一组 logits，logit 可以理解成“未归一化分数”，记作 $z_1,\dots,z_K$。经过 softmax 后得到预测概率：

$$
\hat y_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

真实标签 $y_i$ 是 one-hot 分布。交叉熵损失为

$$
L = -\sum_i y_i \log \hat y_i
$$

把 softmax 代入：

$$
L = -\sum_i y_i \log \frac{e^{z_i}}{\sum_j e^{z_j}}
= -\sum_i y_i z_i + \log \sum_j e^{z_j}
$$

因为 one-hot 标签满足 $\sum_i y_i=1$，对任意 $z_k$ 求导：

$$
\frac{\partial L}{\partial z_k}
= -y_k + \frac{e^{z_k}}{\sum_j e^{z_j}}
= \hat y_k - y_k
$$

这一步非常重要。它说明 softmax + 交叉熵 的梯度就是“预测概率减真实概率”。梯度信号直接等于残差，没有多余的饱和导数因子，所以优化非常稳定。

流程可以压缩成下面这张表：

| 阶段 | 数学对象 | 作用 |
|---|---|---|
| 输入 | logits $z$ | 模型原始分数 |
| 概率化 | $\hat y = softmax(z)$ | 转成合法概率分布 |
| 计算损失 | $L=-\sum y_i\log \hat y_i$ | 惩罚错误且高置信的预测 |
| 反向传播 | $\partial L/\partial z_i=\hat y_i-y_i$ | 梯度直接反映预测残差 |

玩具例子：二分类里，假设真实标签 $y=1$，模型预测正类概率 $q=0.1$。交叉熵损失是

$$
L=-\log 0.1 \approx 2.3026
$$

如果它来自 sigmoid 输出，则对 logit 的梯度正比于

$$
q-y=0.1-1=-0.9
$$

这是一个很强的负梯度，表示参数需要明显朝“提高正类概率”的方向更新。也就是说，模型越自信地预测错，交叉熵给出的修正信号越强。

对比 MSE。若损失写成

$$
L_{mse} = \frac{1}{2}(q-y)^2
$$

而 $q=\sigma(z)$，则

$$
\frac{\partial L_{mse}}{\partial z}
=(q-y)\cdot \sigma'(z)
=(q-y)\cdot q(1-q)
$$

问题出在 $q(1-q)$。当 $q$ 接近 0 或 1 时，它会很小。于是即便模型错得离谱，只要 sigmoid 已经饱和，梯度也可能被压得很弱。这就是“梯度消失”，白话说就是模型知道自己错了，但改不动。

真实工程例子：一个新闻分类模型要在“财经、体育、科技、娱乐”四类中选一类。模型最后一层输出 4 个 logits，用 softmax 变成 4 个概率，再用交叉熵训练。这样做的意义不是“形式上像概率”，而是它正好对应多项分布的最大似然估计。也就是：训练目标和数据生成假设是一致的。

---

## 代码实现

下面用 NumPy 写一个最小可运行版本，展示 softmax、交叉熵、以及梯度 $\hat y-y$ 的关系。代码里包含数值稳定处理和断言。

```python
import numpy as np

def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_from_logits(logits, y_true):
    probs = softmax(logits)
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    loss = -np.sum(y_true * np.log(probs), axis=1).mean()
    grad = (probs - y_true) / logits.shape[0]
    return loss, probs, grad

# 2 个样本，3 分类
logits = np.array([
    [2.0, 0.5, -1.0],
    [0.1, 1.2, 0.3]
], dtype=np.float64)

# one-hot 标签：第一个样本类别 0，第二个样本类别 2
y_true = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

loss, probs, grad = cross_entropy_from_logits(logits, y_true)

print("probs=\n", probs)
print("loss=", loss)
print("grad=\n", grad)

# 基本正确性检查
assert probs.shape == logits.shape
assert grad.shape == logits.shape
assert np.allclose(probs.sum(axis=1), 1.0)
assert loss > 0

# 梯度和应为 0，因为 softmax 概率和为 1
assert np.allclose(grad.sum(axis=1), 0.0)

# 第一个样本真实类别是 0，若预测不足，类别 0 的梯度应为负，推动该 logit 变大
assert grad[0, 0] < 0
```

这段代码体现了三个工程原则。

第一，`logits - max(logits)` 是数值稳定技巧。指数函数增长极快，不先平移就容易溢出。

第二，`np.clip(..., eps, 1.0)` 是防止 $\log 0$。严格说，稳定实现更常见的是直接计算 `log_softmax`，但这个版本更容易看清公式对应关系。

第三，梯度直接写成 `probs - y_true`。这不是经验写法，而是前面推导出的解析结果。

如果换成真实训练循环，通常不会手写这些细节，而是直接用框架提供的“logits 版本”损失函数，例如 PyTorch 的 `CrossEntropyLoss`。它内部已经把 `log-softmax + NLLLoss` 合并，精度和效率都更好。

---

## 工程权衡与常见坑

交叉熵在分类中几乎是默认选择，不是因为“社区习惯”，而是因为它同时满足概率解释、梯度质量和数值实现三个条件。

| 损失函数 | 常见输出层 | 梯度形式 | 数值风险 |
|---|---|---|---|
| MSE | sigmoid / softmax 后概率 | 误差还要乘激活函数导数 | 饱和区梯度很小 |
| 交叉熵 | logits + softmax / sigmoid | 对 logit 常化简为 $\hat y-y$ | 主要风险是 $\log 0$ |
| BCEWithLogits / CrossEntropyLoss | 直接接 logits | 内部做稳定变换 | 风险最低，推荐 |

常见坑有四类。

第一，把 MSE 用在分类概率输出上。比如预测已经是 $0.999$ 对 $1$，MSE 为 $(0.999-1)^2\approx 10^{-6}$，梯度极小；但如果这是错类概率饱和导致的结果，模型会学得很慢。交叉熵对“高置信错误”更敏感，修正更及时。

第二，先 `softmax` 再 `log` 再自己求平均。这样容易在极小概率处下溢。更稳的做法是直接对 logits 计算 `log-softmax`。

第三，标签格式不一致。多分类交叉熵通常要的是“类别索引”或 one-hot 中的一种固定格式，混用时容易出现维度错误，或者看起来能跑但实际上损失在错位。

第四，把交叉熵值本身当成“准确率”。损失低不等于分类边界一定最优，它只是说明概率校准更接近标签分布。工程上仍要同时看准确率、召回率、AUC 或 F1 等任务指标。

真实工程例子：做广告点击率预测时，正样本很少，模型常会输出接近 0 的概率。如果你手动写 BCE 且没有做数值稳定，`log(1-p)` 或 `log(p)` 很容易出现 `-inf`，训练日志直接变成 `nan`。正确做法是使用框架的 logits 版本二分类交叉熵，必要时再配合类别权重。

---

## 替代方案与适用边界

交叉熵不是万能损失，它只是在“输出要解释成概率分布”的分类问题里最自然。

| 方法 | 简化公式 | 适用场景 | 主要参数影响 |
|---|---|---|---|
| 交叉熵 | $-\sum y_i\log \hat y_i$ | 标准二分类/多分类 | 无额外参数，基线方案 |
| Label Smoothing 交叉熵 | $-\sum \tilde y_i\log \hat y_i$ | 标签可能过硬、模型过度自信 | 平滑系数越大，概率更保守 |
| Focal Loss | $-(1-\hat y_t)^\gamma \log \hat y_t$ | 类别不平衡、易例过多 | $\gamma$ 越大，越关注难例 |
| MSE | $\frac{1}{2}\|y-\hat y\|^2$ | 回归任务、连续值预测 | 假设更接近高斯噪声 |

Label smoothing 的意思是：不把正确类别设成 1，而是设成例如 0.9，其余类别分掉 0.1。白话说，它故意阻止模型把自己训练成“绝对自信”。在大词表分类、蒸馏训练、多标签近邻类别混淆时常见。

Focal loss 则是在交叉熵前乘一个动态权重。预测已经很容易的样本，权重会变小；难样本权重更大。它适合目标检测、欺诈识别这类极度不平衡场景。

但基础逻辑不变：这些方法仍然建立在交叉熵之上。它们不是否定交叉熵，而是对“样本权重”或“目标分布”做修正。

适用边界也要说清楚。如果任务是房价预测、温度预测、库存需求预测，这些输出不是类别概率，而是连续数值，此时 MSE、MAE、Huber loss 往往更合理。因为你要拟合的是数值误差，不是概率分布之间的编码代价。

---

## 参考资料

1. Wikipedia, “Cross-entropy”
   用途：给出香农熵、交叉熵的标准定义，以及编码视角下的解释。  
   链接：https://en.wikipedia.org/wiki/Cross-entropy

2. Wikipedia, “Kullback-Leibler divergence”
   用途：给出 $KL(p\parallel q)$ 的定义、非负性和非对称性。  
   链接：https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

3. TensorTonic, “Cross-Entropy”
   用途：解释 softmax 与交叉熵结合时梯度为何能化简为 $\hat y-y$，适合说明训练稳定性的来源。  
   链接：https://www.tensortonic.com/ml-math/information-theory/cross-entropy

4. Rohan Paul, 关于 logistic regression 与 cross-entropy 的技术笔记
   用途：补充从最大似然角度理解二分类交叉熵，以及常见实现经验。  
   链接：可检索 `Rohan Paul cross entropy logistic regression`

5. Aman AI Journal, 关于 cross-entropy vs MSE 的经验总结
   用途：对比两类损失在概率输出场景下的优化行为与工程选择。  
   链接：可检索 `Aman AI cross entropy vs mse`
