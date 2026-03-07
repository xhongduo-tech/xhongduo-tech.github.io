## 核心结论

LSTM（Long Short-Term Memory，长短期记忆网络）是在普通 RNN（Recurrent Neural Network，循环神经网络）上加了一套“门控”机制。门控的白话解释是：网络不再把所有旧信息和新信息一股脑混在一起，而是先决定“忘多少、写多少、放出多少”。

它的关键不是“多了几个公式”，而是多了一个专门传递长期信息的细胞状态 $C_t$。这个状态可以理解为序列中的长期记忆通道。LSTM 在每个时间步 $t$ 做六件事：

| 符号 | 方程 | 作用 |
| --- | --- | --- |
| $f_t$ | $f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$ | 遗忘门，决定保留多少旧记忆 |
| $i_t$ | $i_t=\sigma(W_i[h_{t-1},x_t]+b_i)$ | 输入门，决定写入多少新信息 |
| $\tilde{C}_t$ | $\tilde{C}_t=\tanh(W_c[h_{t-1},x_t]+b_c)$ | 候选细胞状态，表示“准备写入”的内容 |
| $o_t$ | $o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$ | 输出门，决定从记忆里放出多少 |
| $C_t$ | $C_t=f_t\odot C_{t-1}+i_t\odot \tilde{C}_t$ | 更新后的长期记忆 |
| $h_t$ | $h_t=o_t\odot \tanh(C_t)$ | 当前时刻输出 |

这里的 $\sigma$ 是 sigmoid 函数，输出范围在 $(0,1)$，所以适合做“开关比例”；$\tanh$ 输出范围在 $(-1,1)$，所以适合表示有正负方向的内容。

LSTM 能处理长期依赖，关键在于细胞状态的更新是“加法路径”：

$$
C_t=f_t\odot C_{t-1}+i_t\odot \tilde{C}_t
$$

加法路径的意义是：梯度在反向传播时可以沿着 $C_t \rightarrow C_{t-1}$ 比较直接地传回去，而不是像普通 RNN 那样反复乘一个容易变小的矩阵。这就是常说的“恒定误差传送带”思想。

结论可以压缩成一句话：LSTM 用四个门控制信息流，用细胞状态承载长期记忆，因此比普通 RNN 更能学到长距离依赖；代价是参数量和计算量大约增加到普通单层 RNN 的 4 倍。

---

## 问题定义与边界

问题先定义清楚：什么叫“长期依赖”？就是当前输出需要依赖很久以前的输入，而中间隔了很多时间步。白话讲，就是模型不能只记住“刚刚看到的东西”，还得保留“很早之前但仍然重要的东西”。

玩具例子：序列是“我 在 北京 上班 ， 明天 去 上海 ， 我 的 户口 在 北京”。如果模型要预测最后一句里“北京”相关的语义，它需要记住前面出现过的地点关系。普通 RNN 容易把早期信息冲淡，LSTM 会更稳定。

真实工程例子：机器翻译解码器中，英文句子前面的主语和后面的动词形式往往要一致。例如前文出现复数主语，后文生成动词时要保留这种语法信息。LSTM 的 $C_t$ 就承担了“把这个语法约束保留下来”的角色。

普通 RNN 的更新通常是：

$$
h_t=\phi(W_h h_{t-1}+W_x x_t+b)
$$

其中 $\phi$ 常用 $\tanh$ 或 ReLU。问题在于，当序列很长时，反向传播梯度近似要经过很多次链式乘法。如果每一步导数平均小于 1，就会快速衰减；大于 1 又可能爆炸。于是模型学不会长依赖。

LSTM 的边界也要讲清楚：

| 项目 | 普通 RNN | LSTM |
| --- | --- | --- |
| 长期依赖能力 | 弱 | 强得多 |
| 参数量 | 基准 1 倍 | 约 4 倍 |
| 并行性 | 低 | 低 |
| 单步计算复杂度 | 较低 | 较高 |
| 长序列稳定性 | 差 | 更稳定，但仍可能出问题 |

这里“更稳定”不等于“完全没有梯度问题”。如果序列极长、输入尺度混乱、初始化不合理，LSTM 仍然会训练困难。它解决的是“普通 RNN 明显不够用”的问题，不是把时序建模变成零成本。

因此本文讨论边界是：

1. 只讨论标准 LSTM 单元，不展开 peephole LSTM、双向 LSTM、堆叠 LSTM。
2. 只讨论前向计算和参数方程，不推导完整 BPTT（Backpropagation Through Time，时间反向传播）细节。
3. 默认输入已经数值化为向量 $x_t$，不讨论 tokenizer、embedding 训练策略等前置问题。

---

## 核心机制与推导

LSTM 在每个时间步同时看两类输入：上一时刻输出 $h_{t-1}$ 和当前输入 $x_t$。常见写法是先拼接：

$$
z_t=[h_{t-1},x_t]
$$

然后分别送入四组参数。

### 1. 遗忘门

$$
f_t=\sigma(W_f z_t+b_f)
$$

遗忘门的作用是控制旧记忆 $C_{t-1}$ 留下多少。若某个维度的 $f_t$ 接近 1，说明该维度旧信息基本保留；接近 0，说明该维度被清除。

### 2. 输入门

$$
i_t=\sigma(W_i z_t+b_i)
$$

输入门控制“候选信息”写入多少。它不是直接产生内容，而是产生写入比例。

### 3. 候选细胞状态

$$
\tilde{C}_t=\tanh(W_c z_t+b_c)
$$

候选细胞状态表示当前时刻“准备写进去的新内容”。它经过 $\tanh$，所以每个维度在 $[-1,1]$，既可以表示正向强化，也可以表示负向修正。

### 4. 细胞状态更新

$$
C_t=f_t\odot C_{t-1}+i_t\odot \tilde{C}_t
$$

这里的 $\odot$ 是逐元素乘法，白话解释是“每个维度各算各的比例”。

这一步是 LSTM 的核心。第一项保留旧记忆，第二项写入新记忆。不是覆盖，而是合成。正因为是“旧状态的一部分 + 新内容的一部分”，信息更新更平滑。

### 5. 输出门

$$
o_t=\sigma(W_o z_t+b_o)
$$

输出门决定当前时刻从细胞状态里暴露多少给外部。

### 6. 隐状态输出

$$
h_t=o_t\odot \tanh(C_t)
$$

细胞状态 $C_t$ 偏向“内部长期记忆”，而隐状态 $h_t$ 偏向“当前时刻真正对外可见的表示”。

### 数值玩具例子

假设一维情况下：

- $C_{t-1}=0.5$
- $f_t=0.8$
- $i_t=0.6$
- $\tilde{C}_t=0.2$
- $o_t=0.9$

则：

$$
C_t=0.8\times 0.5+0.6\times 0.2=0.4+0.12=0.52
$$

$$
h_t=0.9\times \tanh(0.52)\approx 0.9\times 0.478\approx 0.430
$$

这个例子说明三件事：

1. 旧记忆没有被直接覆盖，而是保留了 $80\%$。
2. 新信息只写入了候选值的 $60\%$。
3. 即使内部记忆是 $0.52$，最终输出也还要再过一道输出门过滤。

### 为什么加法路径更稳

普通 RNN 的梯度容易消失，本质是重复矩阵乘法导致：

$$
\frac{\partial h_t}{\partial h_{t-k}}
$$

会连乘很多 Jacobian（雅可比矩阵，白话解释是“局部变化率矩阵”）。

LSTM 中，细胞状态的局部导数有一个更直接的形式：

$$
\frac{\partial C_t}{\partial C_{t-1}}=f_t
$$

如果 $f_t$ 在很多步都接近 1，那么梯度就能较稳定地沿着细胞状态传播。它不是绝对恒定，但相比普通 RNN，路径短、干扰少、数值更可控。这也是“恒定误差传送带”这个说法的来源。

---

## 代码实现

下面先写一个最小可运行的 NumPy 版本。它只实现单步前向，重点是让公式和代码逐行对齐。

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lstm_step(x_t, h_prev, c_prev, params):
    W_f, b_f = params["W_f"], params["b_f"]
    W_i, b_i = params["W_i"], params["b_i"]
    W_c, b_c = params["W_c"], params["b_c"]
    W_o, b_o = params["W_o"], params["b_o"]

    z = np.concatenate([h_prev, x_t], axis=0)

    f_t = sigmoid(W_f @ z + b_f)
    i_t = sigmoid(W_i @ z + b_i)
    c_tilde = np.tanh(W_c @ z + b_c)
    o_t = sigmoid(W_o @ z + b_o)

    c_t = f_t * c_prev + i_t * c_tilde
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t, (f_t, i_t, c_tilde, o_t)

# 一维玩具参数，故意构造成接近文中的示例
x_t = np.array([0.1])
h_prev = np.array([0.2])
c_prev = np.array([0.5])

params = {
    "W_f": np.array([[2.0, 1.0]]),
    "b_f": np.array([0.786]),
    "W_i": np.array([[1.5, -1.0]]),
    "b_i": np.array([0.785]),
    "W_c": np.array([[1.0, 1.0]]),
    "b_c": np.array([-0.006]),
    "W_o": np.array([[1.2, 0.8]]),
    "b_o": np.array([0.997]),
}

h_t, c_t, gates = lstm_step(x_t, h_prev, c_prev, params)
f_t, i_t, c_tilde, o_t = gates

assert h_t.shape == (1,)
assert c_t.shape == (1,)
assert np.all((f_t > 0) & (f_t < 1))
assert np.all((i_t > 0) & (i_t < 1))
assert np.all((o_t > 0) & (o_t < 1))
assert np.all((c_tilde >= -1) & (c_tilde <= 1))

# 手工验证状态更新公式
expected_c = f_t * c_prev + i_t * c_tilde
expected_h = o_t * np.tanh(expected_c)
assert np.allclose(c_t, expected_c)
assert np.allclose(h_t, expected_h)

print("f_t =", f_t.round(3))
print("i_t =", i_t.round(3))
print("c_tilde =", c_tilde.round(3))
print("o_t =", o_t.round(3))
print("c_t =", c_t.round(3))
print("h_t =", h_t.round(3))
```

这段代码里最容易看懂的一点是：四个门虽然功能不同，但形式几乎一样，都是“线性变换 + 激活函数”。差别只在于参数不同、激活不同、参与更新的位置不同。

在实际框架里，LSTM 通常不会显式分成四组 `W_f/W_i/W_c/W_o` 暴露给你，而是把它们拼成大矩阵。以 Keras 为例，常见参数映射如下：

| 框架参数 | 含义 | 典型形状 |
| --- | --- | --- |
| `kernel` | 输入到四个门的权重 | $(input\_dim, 4u)$ |
| `recurrent_kernel` | 隐状态到四个门的权重 | $(u, 4u)$ |
| `bias` | 四个门的偏置 | $(4u,)$ |

其中 $u$ 是隐藏单元数。也就是说，工程实现通常把四个门合并为一次大矩阵乘法，再按列切片拆回去，这样更高效。

一个真实工程中的 Keras 例子如下：

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100, 64)),
    tf.keras.layers.LSTM(
        units=128,
        return_sequences=False,
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=0.1,
        recurrent_dropout=0.0,
        unit_forget_bias=True
    ),
    tf.keras.layers.Dense(10, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
```

这里有两个工程上很重要的点：

1. `unit_forget_bias=True` 会给遗忘门一个偏正的初值，目的是让训练初期先“少忘一点”。
2. `clipnorm=1.0` 是梯度裁剪，避免长序列训练时梯度突然爆炸。

如果是文本分类、日志序列分析、传感器时间序列预测，这样的配置是很常见的起点。

---

## 工程权衡与常见坑

LSTM 的优势来自门控，代价也来自门控。最直接的成本就是参数量上升。

设输入维度为 $d$，隐藏维度为 $h$。

普通 RNN 参数量大致为：

$$
dh + h^2 + h
$$

LSTM 参数量大致为：

$$
4(dh + h^2 + h)
$$

因为四个门每个都要一套输入权重、循环权重和偏置。这就是“LSTM 大约是 RNN 4 倍参数量”的来源。

常见工程坑可以直接列出来：

| 坑 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 输入未标准化 | 门输出接近 0 或 1 | sigmoid 饱和 | 对输入做标准化、归一化，必要时加 LayerNorm |
| 遗忘门初值太小 | 模型一开始什么都记不住 | 旧状态被过度清空 | 使用正的遗忘门偏置，如 `unit_forget_bias=True` |
| 序列太长无裁剪 | loss 波动很大甚至 NaN | 梯度爆炸 | 用 `clipnorm` 或 `clipvalue` |
| 隐藏维度过大 | 训练慢、显存占用高 | 参数量按 $h^2$ 增长 | 从较小 `units` 起调，例如 64/128 |
| 过度依赖 `recurrent_dropout` | 训练显著变慢 | 循环路径 dropout 代价高 | 优先调输入 dropout，谨慎开循环 dropout |
| 推理逐步生成太慢 | 延迟高 | RNN 本身难并行 | 缓存状态，按流式场景设计接口 |

真实工程例子：在在线日志异常检测里，系统每秒输入一批事件 embedding，LSTM 负责根据过去几十到几百步行为预测下一个事件类别。如果原始数值特征量纲差异很大，例如某些字段在 $[0,1]$，某些字段在 $[10^4,10^6]$，那拼接后进入门控线性层，sigmoid 很容易饱和，结果就是门几乎一直开满或关死，模型学不到细粒度控制。这个问题不是 LSTM 公式错了，而是输入分布把门控推到了坏区间。

另一个常见误区是：以为 LSTM 就能无限记忆。实际上，若很多步上 $f_t<1$，则旧记忆仍会逐步衰减。LSTM 只是让“是否保留”变成了可学习的，而不是保证永不遗忘。

---

## 替代方案与适用边界

LSTM 并不是时序任务的默认终点。它只是“普通 RNN 不够用时”的经典升级方案。现在更常见的对比对象是 GRU 和 Transformer。

GRU（Gated Recurrent Unit，门控循环单元）可以理解为门控更少、结构更简化的 LSTM。Transformer 则彻底放弃循环，改用自注意力直接建模任意位置之间的关系。

| 模型 | 核心结构 | 参数效率 | 并行性 | 长距离建模 | 典型场景 |
| --- | --- | --- | --- | --- | --- |
| RNN | 单状态递推 | 高 | 低 | 弱 | 很短序列、教学示例 |
| LSTM | 四门 + 细胞状态 | 中 | 低 | 较强 | 中等长度时序、流式生成 |
| GRU | 两门 + 合并状态 | 较高 | 低 | 中等到较强 | 参数预算紧张、移动端 |
| Transformer | 自注意力 | 视规模而定 | 高 | 很强 | 大规模 NLP、多模态、并行训练 |

什么时候优先选 LSTM：

1. 数据量中等，不足以支撑大型 Transformer。
2. 任务天然按时间流到来，需要一步一步处理。
3. 设备侧或服务侧需要保留递归状态，低批量流式推理更合适。
4. 序列长度不是特别夸张，几百步到上千步内仍可接受。

什么时候考虑替代：

1. 如果参数预算更紧，先试 GRU。
2. 如果训练时需要高并行吞吐，优先 Transformer。
3. 如果任务主要依赖局部模式，1D CNN 或 TCN（时序卷积网络）也可能更高效。
4. 如果序列极长，LSTM 即使能训，也往往不如注意力模型灵活。

一个很实际的判断标准是：如果你的业务是传感器流、交易流、在线日志流，且模型要持续接收状态，LSTM 仍然有工程价值；如果你的业务是大规模文本理解、长文档建模、批量离线训练，Transformer 通常更合适。

---

## 参考资料

1. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
作用：LSTM 原始论文，定义了细胞状态与门控思想，是参数方程的理论源头。

2. Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*, Chapter 10: Sequence Modeling.
作用：系统解释 RNN、梯度消失与 LSTM/GRU 的设计动机，适合建立整体认知。

3. TensorFlow Keras Documentation: `tf.keras.layers.LSTM`
作用：查看工程实现中的 API、默认参数、`unit_forget_bias`、`return_sequences` 等设置方法。

4. TensorFlow / Keras 源码中 LSTM layer 与 LSTMCell 的实现
作用：理解框架如何把四组门参数拼成大矩阵，以及训练时的实际计算路径。

5. Colah, “Understanding LSTM Networks”
作用：用较直观的图示解释门控信息流，适合第一次建立结构感。
