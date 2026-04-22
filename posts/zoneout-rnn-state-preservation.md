## 核心结论

Zoneout 是一种用于 RNN 的正则化方法：在每个时间步，它以概率 $p$ 不更新某些隐状态维度，而是直接保留上一时间步的状态值。

它和普通 Dropout 的核心差异不在“有没有随机性”，而在“随机之后怎么处理状态”。普通 Dropout 是把部分激活值置零；Zoneout 是让部分隐状态维度沿用旧值。换句话说，Zoneout 不是随机删掉记忆，而是随机暂停更新。

新手版理解可以这样记：

普通 Dropout 像“把某些信息临时清空”；Zoneout 像“这一步先不改，继续沿用上一拍的记忆”。所以它不会把序列记忆一下子打断。

这对 RNN 很重要。RNN 的隐状态本来就是跨时间传递的信息载体。若在循环连接里直接清零，模型可能更难保留长程依赖；而 Zoneout 通过“保留旧状态”引入随机扰动，同时尽量不破坏记忆链路。

| 场景 | Zoneout 价值 |
|---|---|
| 长序列建模 | 保住长程记忆 |
| 数据较少 | 降低过拟合 |
| RNN / LSTM / GRU | 直接作用于隐状态 |

Zoneout 的典型使用场景包括字符级语言模型、语音识别解码器、长文本序列建模等。它尤其适合仍然使用循环结构、且任务依赖较长上下文的模型。

---

## 问题定义与边界

RNN，中文常叫循环神经网络，是一种按时间顺序处理序列数据的神经网络。它会在每个时间步接收当前输入 $x_t$，同时读取上一时间步的隐状态 $h_{t-1}$，再生成当前隐状态 $h_t$。

隐状态可以理解为模型对“前面已经看过的信息”的压缩表示。比如字符级语言模型读到一句话前半段时，隐状态里会保存关于语法、词形、上下文主题的信息。

Zoneout 解决的是 RNN 的“状态不稳定”问题，而不是所有形式的过拟合问题。

更准确地说，它关注的是 recurrent state，也就是循环结构中跨时间传递的状态。若主要问题发生在输入层、embedding 层或全连接输出层，普通 Dropout、Embedding Dropout 或权重衰减可能更直接。

对比版理解：

若一个字符级语言模型在长句子中频繁丢失前文信息，普通 Dropout 可能让隐状态中的部分信息被直接清零，导致上下文链路断裂。Zoneout 则让部分维度继续保持上一时刻值，从而维持上下文连续性。

| 方法 | 作用位置 | 行为 | 是否保留记忆 |
|---|---|---|---|
| Dropout | 激活/输入 | 随机清零 | 否 |
| Recurrent dropout | 隐状态或循环连接 | 随机丢弃部分连接或状态贡献 | 部分 |
| Zoneout | 隐状态 | 随机沿用旧状态 | 是 |

这里需要区分三个边界。

第一，Zoneout 不是输入噪声。它不直接改输入 $x_t$，而是改当前候选状态和上一状态之间的选择方式。

第二，Zoneout 不是把状态置零。它的随机分支是 $h_{t-1}$，不是 0。这个差异决定了它对长序列更友好。

第三，Zoneout 不是 Transformer 的替代品。它是循环模型内部的正则化技巧。如果模型架构已经换成 Transformer，状态递推机制不存在，Zoneout 就没有直接作用对象。

---

## 核心机制与推导

Zoneout 的计算分两步。

先用 RNN 单元正常计算候选新状态。候选状态是“如果没有 Zoneout，本来应该更新出来的状态”，记作 $\tilde{h}_t$。

再采样一个二值 mask。mask 是一个只包含 0 和 1 的张量，用来决定每个维度是采用新状态，还是保留旧状态。

常用公式是：

$$
m_t \sim \mathrm{Bernoulli}(1-p), \quad
h_t = m_t \odot \tilde{h}_t + (1-m_t)\odot h_{t-1}
$$

其中：

| 符号 | 含义 |
|---|---|
| $p$ | zoneout 概率，也就是保留旧状态的概率 |
| $m_t$ | 当前时间步的二值 mask |
| $\tilde{h}_t$ | 当前输入和旧状态计算出的候选新状态 |
| $h_{t-1}$ | 上一时间步隐状态 |
| $h_t$ | 应用 Zoneout 后的当前隐状态 |
| $\odot$ | 按元素相乘 |

因为 $m_t \sim \mathrm{Bernoulli}(1-p)$，所以 $m_t=1$ 的概率是 $1-p$，表示采用新状态；$m_t=0$ 的概率是 $p$，表示沿用旧状态。

推理时通常不继续随机采样，而是使用期望形式：

$$
\mathbb{E}[h_t] = (1-p)\tilde{h}_t + p h_{t-1}
$$

这样可以避免同一个输入多次推理得到不同输出。也有实现选择推理时直接关闭 Zoneout，只使用正常的新状态。具体采用哪种方式，要看框架实现和训练时的定义。

玩具例子：

设上一时间步状态为：

```text
h_{t-1} = [0.8, -0.4]
```

候选新状态为：

```text
\tilde{h}_t = [0.2, 0.6]
```

zoneout 概率为：

```text
p = 0.5
```

若采样得到：

```text
m_t = [1, 0]
```

则：

$$
h_t = [1,0]\odot[0.2,0.6] + [0,1]\odot[0.8,-0.4] = [0.2,-0.4]
$$

第一维更新为候选新状态 $0.2$，第二维沿用旧状态 $-0.4$。这就是 Zoneout 的核心行为。

对 LSTM，情况稍微多一步。LSTM，中文常叫长短期记忆网络，是一种带有细胞状态 $c_t$ 和输出隐状态 $h_t$ 的 RNN 变体。$c_t$ 更像长期记忆通道，$h_t$ 更像当前输出表示。

因此在 LSTM 中，常见做法是对 $c_t$ 和 $h_t$ 分别使用独立 mask：

$$
c_t = m^c_t \odot \tilde{c}_t + (1-m^c_t)\odot c_{t-1}
$$

$$
h_t = m^h_t \odot \tilde{h}_t + (1-m^h_t)\odot h_{t-1}
$$

这样做的原因是 $c_t$ 和 $h_t$ 的功能不同。只对 $h_t$ 加 Zoneout，可能无法稳定长期记忆通道；只对 $c_t$ 加 Zoneout，又可能没有覆盖输出状态的随机扰动。

真实工程例子：

在语音识别解码器中，模型需要根据前面已经生成的音素、字符或子词预测下一个输出。若一句语音较长，解码器状态需要持续保存发音上下文、语言结构和历史输出信息。直接对循环状态做普通 Dropout 可能破坏这条时间链；Zoneout 则让一部分状态维度保持旧值，使模型在训练时面对“部分状态延迟更新”的扰动，从而减少对某一次精确状态更新的过度依赖。

---

## 代码实现

实现 Zoneout 的关键点是：先计算候选新状态，再用 mask 在“新状态”和“旧状态”之间逐元素切换。

伪代码如下：

```python
# h_prev: 上一时刻隐状态
# h_new: RNN 当前算出的候选状态
# p: zoneout 概率，表示保留旧状态的概率
mask = bernoulli(1 - p, shape=h_prev.shape)
h = mask * h_new + (1 - mask) * h_prev
```

实现要点：

| 步骤 | 说明 |
|---|---|
| 1 | 计算候选新状态 `h_new` |
| 2 | 采样二值 mask |
| 3 | 按维度混合 `h_new` 和 `h_prev` |
| 4 | 推理时改用期望或关闭采样 |

下面是一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示 Zoneout 的状态混合逻辑。

```python
import random

def zoneout(h_prev, h_new, p, mask=None):
    """
    h_prev: 上一时间步状态
    h_new: 候选新状态
    p: 保留旧状态的概率
    mask: 可选，1 表示采用新状态，0 表示保留旧状态
    """
    assert len(h_prev) == len(h_new)
    assert 0.0 <= p <= 1.0

    if mask is None:
        mask = [1 if random.random() < (1 - p) else 0 for _ in h_prev]

    assert len(mask) == len(h_prev)
    return [
        m * new + (1 - m) * old
        for old, new, m in zip(h_prev, h_new, mask)
    ]

h_prev = [0.8, -0.4]
h_new = [0.2, 0.6]
p = 0.5
mask = [1, 0]

h = zoneout(h_prev, h_new, p, mask)
assert h == [0.2, -0.4]

expected = [
    (1 - p) * new + p * old
    for old, new in zip(h_prev, h_new)
]
assert expected == [0.5, 0.09999999999999998]
```

这里的 `mask=[1, 0]` 表示第一维采用新状态，第二维保留旧状态。最后一个 `assert` 演示的是推理阶段的期望形式。

在真实框架中，mask 通常是和 batch、hidden size 同形状的张量。例如隐藏状态形状是 `[batch_size, hidden_dim]`，mask 也应是 `[batch_size, hidden_dim]`。如果是多层 RNN，还要明确每一层是否单独采样 mask。

还要注意 `p` 的语义。有的代码把 `p` 定义为“zoneout 概率”，也就是保留旧状态的概率；有的代码把类似参数写成 keep probability 或 update probability。调参前必须先看实现注释，否则 `p=0.1` 和 `p=0.9` 的含义可能完全相反。

---

## 工程权衡与常见坑

Zoneout 的主要收益是稳定状态传递，但它不是越大越好。

若 $p$ 太小，大多数维度都会正常更新，正则化效果弱。若 $p$ 太大，大多数维度都停留在旧状态，模型会变得过于保守，学习速度下降，甚至学不动。

调参版理解：

如果训练集较小、序列较长，可以先从 `0.1` 或 `0.15` 开始。如果发现验证集过拟合明显，再逐步增加。如果发现训练损失下降很慢，甚至长时间不收敛，通常说明保留旧状态太多，更新被抑制了。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| `p` 语义弄反 | 调参失效 | 先看实现注释 |
| `p` 过大 | 模型学不动 | 从小值起试 |
| 只处理 `h` 不处理 `c` | LSTM 效果不完整 | 分别考虑 |
| 推理时继续采样 | 输出抖动 | 用期望或关闭随机 mask |
| 与其他 recurrent dropout 叠加 | 行为冲突 | 先确认框架限制 |

第一个常见坑是把 Zoneout 当成普通 Dropout 的替代品。它们都能正则化，但作用点不同。输入层过拟合严重时，Zoneout 不一定解决问题；循环状态不稳定时，普通 Dropout 也不一定合适。

第二个常见坑是 LSTM 只处理 `h`。LSTM 的 `c` 是长期记忆通道，若任务依赖长跨度信息，只给输出隐状态 `h` 加 Zoneout 可能不够。工程上更稳的做法是分别实验 `zoneout_h` 和 `zoneout_c`。

第三个常见坑是训练和推理行为不一致但没有记录。训练时采样 mask，推理时用期望，二者并不完全等价。若线上部署要求结果完全确定，推理阶段不能继续随机采样。

第四个常见坑是和其他循环正则化方法叠加过多。例如同时使用 recurrent dropout、weight dropout、Zoneout、LayerNorm，可能导致训练信号变弱。更好的方式是一次只改一个主要变量，先确认单独收益，再考虑组合。

---

## 替代方案与适用边界

Zoneout 适合循环结构中的状态稳定性问题。若问题不在这里，就应考虑其他方法。

| 方案 | 适合什么问题 | 优势 | 局限 |
|---|---|---|---|
| Zoneout | RNN 状态稳定性 | 保留记忆 | 仅适合循环结构 |
| Dropout | 一般过拟合 | 简单通用 | 可能破坏记忆 |
| Recurrent dropout | 循环正则化 | 常见实现多 | 可能不如 Zoneout 稳 |
| Transformer | 长依赖建模 | 并行能力强 | 结构不同 |

如果主要问题是输入层过拟合，普通 Dropout 或 Embedding Dropout 可能更合适。Embedding Dropout 是对词向量或字符向量做随机屏蔽，用来减少模型对少数 token 表示的依赖。

如果主要问题是模型很深、梯度传播不稳定，残差连接、LayerNorm、梯度裁剪可能更直接。LayerNorm，中文常叫层归一化，是对单个样本内部的特征维度做归一化，常用于稳定训练。

如果模型结构允许，Transformer 也可能比 RNN 正则化更直接。Transformer 使用自注意力机制，可以让任意两个位置直接建立关系，不必完全依赖逐步传递的隐状态。不过 Transformer 的计算成本、数据需求和部署方式也不同，不能简单说一定更优。

选择场景版理解：

做字符级语言模型、语音解码器、长文本序列建模时，Zoneout 常比普通 recurrent dropout 更稳。因为这些任务依赖跨时间的隐状态连续性。但如果任务本身不依赖长时序记忆，例如短序列分类、固定长度特征分类、非循环结构的文本分类，Zoneout 的收益可能有限。

一个实际判断标准是：模型错误是否和“忘记前文”有关。

如果错误常表现为长句后半段语法漂移、语音解码后段重复、字符模型无法维持括号或引号结构，Zoneout 值得尝试。如果错误主要来自类别不平衡、输入噪声、标注质量差或输出层过拟合，Zoneout 不是第一优先级。

---

## 参考资料

1. [Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations](https://openreview.net/forum?id=rJqBEPcxe)
2. [teganmaharaj/zoneout](https://github.com/teganmaharaj/zoneout)
3. [NVIDIA OpenSeq2Seq RNN API Docs](https://nvidia.github.io/OpenSeq2Seq/html/api-docs/parts.rnns.html)
4. [PaddleSpeech Tacotron2 Decoder ZoneOutCell](https://www.aidoczh.com/paddlespeech/_modules/paddlespeech/t2s/modules/tacotron2/decoder.html)
