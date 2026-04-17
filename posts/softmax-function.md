## 核心结论

Softmax 的作用是把一组任意实数分数映射成概率分布。这里的“概率分布”指一组非负且总和为 1 的数，可以直接解释为模型对每个类别的相对置信度。它的标准形式是：

$$
\operatorname{Softmax}(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

如果引入温度参数 $\tau>0$，公式写成：

$$
\operatorname{Softmax}(z_i/\tau)=\frac{e^{z_i/\tau}}{\sum_{j=1}^{K}e^{z_j/\tau}}
$$

它有三个核心性质。

| 性质 | 含义 | 工程价值 |
|---|---|---|
| 非负 | 每个输出都 $\ge 0$ | 可以解释为概率 |
| 归一化 | 所有输出之和为 1 | 适合多分类输出 |
| 保序 | 原始分数越大，输出概率通常越大 | 保留类别排序信息 |

玩具例子最容易说明它的效果。给定 logits（未归一化分数，白话解释就是模型内部算出来的原始打分）：

$$
z=[1,2,8]
$$

则：

$$
e^1\approx2.72,\quad e^2\approx7.39,\quad e^8\approx2980.96
$$

归一化后约为：

$$
[0.0009,\ 0.0025,\ 0.9966]
$$

这说明 Softmax 不只是“缩放”，而是先用指数函数放大差距，再归一化成概率。分数最高的 8 最终几乎拿走全部概率质量。

结论可以直接记成一句话：Softmax 适合“多分类训练”，因为它既能输出概率，又能保持可微分；但真正可用的前提是必须处理数值稳定性，尤其是 `exp` 溢出问题。

---

## 问题定义与边界

多分类任务的目标不是只找“谁最大”，而是要得到一组可以训练的概率。这里“可训练”指输出对输入有连续梯度，参数更新时可以通过反向传播计算方向。如果直接做 `argmax`，只能得到一个离散类别编号，无法提供稳定梯度。

因此我们需要一个函数，满足以下要求：

| 要求 | 说明 |
|---|---|
| 输入任意实数 | 模型最后一层通常输出任意正负分数 |
| 输出可解释为概率 | 便于和真实标签比较 |
| 可微分 | 便于梯度下降训练 |
| 对相对差异敏感 | 高分应该更明显地胜出 |

Softmax 正好满足这些条件，但它也有明确边界。

第一，Softmax 只解决“概率化”，不解决“类别是否独立”。如果一个任务允许多个类别同时为真，比如一张图片同时有“猫”和“沙发”，那是多标签问题，通常用 Sigmoid，不用 Softmax。

第二，Softmax 的数值计算有风险。原因是指数函数增长极快。比如：

$$
z=[1000,999,998]
$$

直接计算 $e^{1000}$ 在常见浮点格式里会溢出。标准处理方法是先减去最大值：

$$
\operatorname{Softmax}(z_i)=\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}
$$

因为分子分母同时乘上了 $e^{-\max(z)}$，结果完全不变，但数值规模会稳定得多。上面的例子变成：

$$
[0,-1,-2]
$$

此时再算指数：

$$
[e^0,e^{-1},e^{-2}]\approx[1,0.3679,0.1353]
$$

不会溢出。

第三，Softmax 输出的是“相对概率”，不是“真实世界频率”。如果 logits 全部同时加上一个常数，输出不会变：

$$
\operatorname{Softmax}(z)=\operatorname{Softmax}(z+c)
$$

这说明 Softmax 只关心类别间差值，不关心绝对基线。

真实工程例子是垃圾邮件分类。模型最后输出 `[2.1, -0.3]`，可以解释为“正常邮件”和“垃圾邮件”的原始打分。经过 Softmax 后，得到比如 `[0.917, 0.083]`。这里 0.083 的意思不是“这封邮件客观上有 8.3% 概率是垃圾”，而是“在当前模型参数和当前类别集合下，它相对更像正常邮件”。

---

## 核心机制与推导

Softmax 的机制可以拆成两步：

1. 指数放大差距。
2. 归一化变成分布。

先看为什么要指数。若直接做线性归一化，负数、零值、尺度变化都会带来问题。指数函数有两个关键作用：一是把所有值变成正数，二是让大分数的优势更明显。

例如 logits 为 `[1,2,3]`，分差只是 1；但指数后变成 `[2.72,7.39,20.09]`，相对差距明显扩大。于是 Softmax 天然偏向“把胜者推得更高”。

它的导数也很重要，因为训练依赖梯度。记 Softmax 输出为 $s_i$，则对输入 $z_k$ 的偏导为：

$$
\frac{\partial s_i}{\partial z_k}=s_i(\delta_{ik}-s_k)
$$

其中 $\delta_{ik}$ 是 Kronecker delta，白话解释就是：当 $i=k$ 时取 1，否则取 0。

这条式子说明两件事：

1. 每个输出不仅依赖自己的输入，也依赖其他类别输入。
2. Softmax 的各个类别是耦合的，一个类别变大，其他类别概率会被挤压。

这正是多分类问题想要的行为，因为多个类别通常互斥。

再看它和交叉熵的配合。交叉熵是“预测分布和真实分布差多远”的度量。若真实标签是 one-hot（只有正确类为 1），损失可写为：

$$
L=-\log s_y
$$

其中 $y$ 是正确类别索引。把 Softmax 和交叉熵合起来推导，最终会得到非常简洁的梯度：

$$
\frac{\partial L}{\partial z_i}=s_i-\mathbb{1}(i=y)
$$

这里 $\mathbb{1}(i=y)$ 的白话解释是“如果当前类别就是正确类则为 1，否则为 0”。这也是为什么框架里几乎总把 `softmax + cross entropy` 当成一个整体实现：数学上简洁，数值上也更稳定。

温度参数 $\tau$ 可以理解为“分布尖锐度控制器”。它控制 logits 的放大或压缩：

$$
s_i(\tau)=\frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}
$$

当 $\tau \to 0$ 时，最大 logit 对应的项会主导分母，输出趋近 one-hot，也就是近似 `argmax`：

$$
\lim_{\tau\to 0}s_i(\tau)=
\begin{cases}
1, & i=\arg\max z\\
0, & \text{otherwise}
\end{cases}
$$

当 $\tau \to \infty$ 时，所有 $z_i/\tau \to 0$，于是每一项都接近 $e^0=1$，输出趋近均匀分布：

$$
\lim_{\tau\to \infty}s_i(\tau)=\frac{1}{K}
$$

用同一个玩具例子 $z=[1,2,8]$ 看温度影响：

| 温度 $\tau$ | 输出特征 |
|---|---|
| 0.5 | 分布极尖锐，最大类几乎独占 |
| 1.0 | 标准 Softmax |
| 2.0 | 分布更平滑，次大类占比上升 |
| 10.0 | 接近均匀，区分度明显下降 |

这也是知识蒸馏、采样解码、概率校准里经常调温度的原因。本质上不是改类别顺序，而是改“置信度形状”。

---

## 代码实现

下面给出一个可运行的 Python 实现。重点有两个：

1. 先减去每行最大值，防止 `exp` 溢出。
2. 支持批量输入和温度参数。

```python
import math

def softmax(logits, temperature=1.0):
    assert temperature > 0, "temperature must be positive"
    assert len(logits) > 0, "logits must not be empty"

    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    probs = [x / s for x in exps]

    # 基本性质检查
    assert all(p >= 0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-12
    return probs

# 玩具例子
p = softmax([1, 2, 8])
assert p[2] > 0.99

# 数值稳定性例子：不会因为 1000 溢出
q = softmax([1000, 999, 998])
assert abs(sum(q) - 1.0) < 1e-12
assert q[0] > q[1] > q[2]

# 温度越低，分布越尖锐
low_t = softmax([1, 2, 8], temperature=0.5)
high_t = softmax([1, 2, 8], temperature=5.0)
assert low_t[2] > high_t[2]
```

如果要支持批次，可以按二维数组处理，每一行代表一个样本。核心逻辑不变：对每一行减最大值，再指数归一化。

下面是更接近工程代码的批量版本：

```python
import math

def softmax_batch(batch_logits, temperature=1.0):
    assert temperature > 0
    result = []

    for logits in batch_logits:
        scaled = [x / temperature for x in logits]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        s = sum(exps)
        probs = [x / s for x in exps]
        assert abs(sum(probs) - 1.0) < 1e-12
        result.append(probs)

    return result

batch = [
    [1.0, 2.0, 8.0],
    [3.0, 3.0, 3.0],
]

out = softmax_batch(batch)
assert len(out) == 2
assert out[1] == [1/3, 1/3, 1/3]
```

真实工程里，训练阶段通常不手写这段逻辑，而是直接使用框架提供的稳定实现，比如 PyTorch 的 `cross_entropy`。原因不是“懒”，而是融合实现通常会直接在 log-space 中计算，减少中间结果溢出和精度损失。

一个真实工程例子是三分类新闻主题识别：`体育 / 财经 / 科技`。模型最后一层输出 logits `[4.2, 1.1, -0.7]`。经过 Softmax 后，可能得到 `[0.945, 0.047, 0.008]`。训练时拿这组概率去和真实标签做交叉熵；推理时再取最大概率类别作为最终预测。

---

## 工程权衡与常见坑

Softmax 在数学上简单，但在工程上经常出问题。最常见的不是公式写错，而是数值和训练链路处理不当。

| 常见问题 | 现象 | 原因 | 典型缓解 |
|---|---|---|---|
| `exp` 上溢 | 输出 `inf`、`NaN` | logits 太大 | 先减 `max` |
| `exp` 下溢 | 小项全变 0 | logits 差距过大 | 使用稳定实现，必要时看 log-prob |
| 手动 Softmax 再取 log | 损失出现 `-inf` | 概率太小后再取对数 | 用 `log_softmax` 或 fused cross entropy |
| 梯度饱和 | 学习变慢甚至不动 | 分布过尖锐 | 缩放 logits，检查初始化和学习率 |
| 温度设置错误 | 预测过度自信或过于平滑 | $\tau$ 不合适 | 按任务调参或做校准 |

最值得单独讲的是 Transformer 里的注意力。缩放点积注意力写作：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里 $d_k$ 是 key 向量维度。为什么要除以 $\sqrt{d_k}$？因为当向量维度增大时，点积的方差会变大，导致 logits 的绝对值变大。logits 一大，Softmax 就会变得过尖锐，接近 one-hot，梯度落到饱和区。

可以把它理解为：不缩放时，模型太快“押注一个位置”，其他位置概率被压到接近 0，反向传播信号变弱。除以 $\sqrt{d_k}$ 相当于把 logits 拉回更可训练的范围。

这就是一个真实工程例子。假设 $d_k=512$，如果不缩放，注意力分数的波动范围会明显变大。训练初期参数还没稳定时，Softmax 很容易过饱和，表现为注意力图特别尖、loss 抖动大、收敛慢。加上 $\sqrt{d_k}$ 缩放后，训练通常稳定很多。

另一个常见坑是把 Softmax 当作“置信度真值”。例如模型输出某类概率 0.99，并不等于这个类别真的有 99% 的客观概率。深度模型经常过度自信，因此在高风险场景里还需要做 calibration（概率校准，白话解释就是让模型输出概率更接近真实命中率）。

工程建议可以压缩成三条：

1. 训练时优先用框架提供的 `log_softmax` 或 `cross_entropy`。
2. 自己实现时必须减 `max`。
3. 注意力机制里不要去掉 $\sqrt{d_k}$ 缩放，除非你明确知道替代稳定化手段是什么。

---

## 替代方案与适用边界

Softmax 不是唯一选择。它适合“互斥类别 + 稠密概率分布”的任务，但不适合所有场景。

先和 `argmax` 比较。`argmax` 只是选最大值位置，不输出连续概率，因此不可微，不能直接用于训练。它适合作为推理阶段的最终决策，不适合作为训练层。

再看稀疏替代方案。Sparsemax 和 Entmax 会让部分输出精确变成 0。这里“稀疏”指很多类别直接不分配概率质量，而不是每个类别都有一个极小但非零的值。对于需要可解释注意力或希望概率更集中的场景，这类方法有意义。

| 方法 | 是否可微 | 输出是否稀疏 | 适合场景 | 局限 |
|---|---|---|---|---|
| Softmax | 是 | 否，通常全非零 | 标准多分类、注意力 | 概率分布偏稠密 |
| Sparsemax | 是，分段可微 | 是 | 稀疏选择、可解释注意力 | 优化性质更复杂 |
| Entmax | 是 | 可控稀疏 | 介于 Softmax 和 Sparsemax 之间 | 实现和调参更复杂 |
| Argmax | 否 | 极端稀疏 | 推理决策 | 不能直接反向传播 |

新手最容易混淆的点是：为什么训练不用 `argmax`，推理却常常用 `argmax`？答案是两阶段目标不同。

- 训练阶段需要梯度，所以要用 Softmax 这类连续函数。
- 推理阶段只需要决策，所以可以直接取最大类。

举一个简单对比例子。假设 logits 为 `[2.2, 2.1, -1.0]`。

- `argmax` 的结果是类别 0。
- `softmax` 的结果大约是 `[0.519, 0.469, 0.012]`。

这两个结果都说明类别 0 更可能，但 Softmax 额外告诉你：类别 0 和类别 1 其实差得不远。这对训练很重要，因为梯度会告诉模型“应该继续把 0 和 1 拉开”，而不是只给出一个硬决策。

因此，Softmax 的适用边界可以概括为：

| 场景 | 是否适合 Softmax |
|---|---|
| 单标签多分类 | 适合 |
| Transformer 注意力权重 | 适合 |
| 多标签分类 | 通常不适合，常用 Sigmoid |
| 只做最终选类、不训练 | 可直接用 `argmax` |
| 需要严格稀疏概率 | 可考虑 Sparsemax / Entmax |

---

## 参考资料

| 资料 | 作用 |
|---|---|
| Wikipedia: Softmax function | 定义、基本公式、性质说明 |
| Attention Is All You Need | 缩放点积注意力与 $\sqrt{d_k}$ 的来源 |
| log-sum-exp 数值稳定相关资料 | 解释为什么“减最大值”不改变结果但能稳定计算 |
| 深度学习框架文档中的 `cross_entropy` / `log_softmax` | 工程实现与稳定训练最佳实践 |

1. Softmax function, Wikipedia: https://en.wikipedia.org/wiki/Softmax_function
2. Attention Is All You Need, arXiv: https://arxiv.org/abs/1706.03762
3. Blanchard, Higham, Higham. Accurately computing the log-sum-exp and softmax functions. IMA Journal of Numerical Analysis.
4. PyTorch Documentation, `torch.nn.functional.cross_entropy` and `log_softmax`
