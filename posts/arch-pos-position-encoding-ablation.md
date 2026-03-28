## 核心结论

位置编码是 Transformer 注入顺序信息的机制。白话说，自注意力本身只看“谁和谁相关”，默认并不知道“谁在前、谁在后”，位置编码就是把顺序信息补进去。

做消融实验时，结论通常不是“哪种位置编码绝对最好”，而是“哪种方案在训练长度之外退化最慢”。如果控制架构、参数量、数据集、训练步数都不变，只替换位置编码，常见结果可以概括为下表。

| 方案 | 训练长度内 PPL | 1.5× 长度 | 2× 长度 | 4× 长度 | Passkey Retrieval 趋势 | 典型结论 |
|---|---:|---:|---:|---:|---|---|
| RoPE | 强 | 中等退化 | 直接外推常明显退化 | 需缩放/插值 | 中长距离较强 | 默认强方案，但长外推要配补丁 |
| RoPE + NTK 缩放 | 强 | 稳定 | 明显优于原始 RoPE | 可继续工作 | 明显更稳 | 工程上常见长上下文扩展手段 |
| ALiBi | 强 | 很稳 | 很稳 | 仍较稳 | 远距离检索稳定 | 长度外推最稳的一类 |
| Sinusoidal | 强 | 有退化 | 继续上升 | 常上升 10% 到 30% | 检索下降 | 数学上可扩展，统计上不一定稳 |
| Learnable | 强 | 轻微或无退化 | 超出训练长度后失效 | 完全不适合 | 通常直接不可用 | 训练内好用，外推最差 |
| NoPE | 略差于 RoPE | 可工作 | 依赖隐式模式 | 易不稳 | 检索能力看训练技巧 | 短序列未必差，但远距一般弱 |

一个常见观察是：NoPE（无位置编码）在短序列上并不会立刻崩溃，PPL 往往只比 RoPE 高约 0.5 到 1。原因不是模型“理解了顺序”，而是模型会从注意力模式、局部共现和因果掩码中隐式推断一些位置信息。白话说，它在“猜距离”，不是在“显式表示距离”。

玩具例子可以这样理解：把每个 token 看成一张图片。位置编码就是贴在图片上的“位置标签”。Sinusoidal 用公式直接生成标签，Learnable 用训练得到标签，RoPE 和 ALiBi 则不直接贴标签，而是在“比较两张图片是否相关”时，把距离因素直接算进去。NoPE 则完全不贴标签，让模型自己从排列模式里猜。

---

## 问题定义与边界

这类实验的目标很明确：固定 Transformer 主体，只改变位置编码，然后观察模型在不同上下文长度下的性能变化。

边界要先收紧，否则实验没有解释力：

| 控制项 | 是否固定 | 原因 |
|---|---|---|
| 模型层数、隐藏维度、头数 | 是 | 否则容量变化会污染结论 |
| 数据集与采样方式 | 是 | 位置编码效果容易被数据分布掩盖 |
| 参数总量 | 尽量是 | Learnable 会额外引入参数，需单独标记 |
| 训练步数与优化器 | 是 | 避免把训练充分度误认为编码差异 |
| 最大训练长度 | 是 | 外推能力必须以同一训练长度为起点 |
| 评估任务 | 是 | 统一看 PPL 和 passkey retrieval |

PPL（Perplexity，困惑度）是语言建模里衡量“下一个词猜得准不准”的指标，越低越好。Passkey retrieval 是长上下文检索任务，白话说，就是在很长的文本里埋一个关键信息，最后问模型能不能准确找回来。

建议至少评估四个长度：

| 设置 | 示例 | 观察目标 |
|---|---|---|
| 训练长度 | 1024 | 看训练内拟合能力 |
| 1.5× | 1536 | 看轻度外推是否平滑 |
| 2× | 2048 | 看中度外推是否开始崩 |
| 4× | 4096 | 看真正长上下文泛化能力 |

新手版理解：假设模型只学过 512 token，现在让它处理 768、1024、2048。你要记录两件事。第一，它猜词是不是越来越差。第二，它还能不能在长段落里找回正确的 passkey。

实验结论通常集中在两个维度：

1. 训练长度内谁更容易收敛。
2. 训练长度外谁退化最慢。

这也是为什么“训练内打平”不代表“部署时等价”。Learnable 位置编码经常在训练长度内表现正常，但一旦推理长度超过训练上限，位置向量没有定义，系统层面就已经失效。

---

## 核心机制与推导

Sinusoidal 位置编码是固定函数编码。固定函数的意思是：位置向量不是学出来的，而是直接用公式算出来的。常见定义是

$$
PE_{pos,2i}=\sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{pos,2i+1}=\cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

其中 $pos$ 是位置，$i$ 是维度索引，$d$ 是隐藏维度。它的优点是任意位置都能算出向量，所以“数学上可扩展”。但“能算出来”和“模型能稳定泛化”不是一回事。训练时只见过 1024，推理时到 4096，频率组合虽然仍存在，但模型未必学过这种尺度上的统计规律，因此 PPL 常上升约 10% 到 30%。

RoPE（Rotary Position Embedding，旋转位置编码）把位置信息乘进 query 和 key。白话说，它不是给 token 加标签，而是把向量按照位置旋转一个角度，再做点积。这样两个位置 $i,j$ 的相关性会天然依赖相对距离 $i-j$，而不是各自绝对编号。它的核心好处是：注意力分数对相对位置敏感，这比“把绝对位置向量直接加到输入上”更适合长度泛化。

可以把它想成一张示意图：第 10 个 token 的 query 向量旋转 $\theta_{10}$，第 15 个 token 的 key 向量旋转 $\theta_{15}$，两者点积里真正留下来的，是角度差 $\theta_{10}-\theta_{15}$。所以它天然编码“相距 5”。

ALiBi（Attention with Linear Biases，线性偏置注意力）更直接。它不改 token 表示，而是在注意力分数上减去和距离成正比的偏置：

$$
score_{ij}=\frac{q_i k_j^\top}{\sqrt{d}} - m_h |i-j|
$$

其中 $m_h$ 是第 $h$ 个头的斜率。白话说，距离越远，默认分数越吃亏，但不是彻底看不到。不同头用不同斜率，相当于有的头关注近邻，有的头允许远距。

这也是 ALiBi 外推稳定的原因。它没有“训练到某个位置编号就停”的问题，偏置规则对更长序列仍然定义良好，所以 4× 长度时 PPL 上升往往仍小于 1。

Learnable 位置编码本质是一个查表矩阵：

$$
E_{pos} \in \mathbb{R}^{L_{\max} \times d}
$$

如果训练最大长度是 $L_{\max}=1024$，那么第 1025 个位置根本没有向量。它不是“效果差一点”，而是“定义不存在”。这就是它超长失效的根本原因。

NoPE 没有显式位置编码。为什么短序列还可以工作？因为因果掩码已经告诉模型“只能看左边”，局部统计规律也会让注意力头形成固定模式。例如某些头总爱看前一个 token、前两个 token，这就形成了弱位置感。但这类隐式位置感在长上下文下不稳定，特别是在需要精确定位远处信息时更容易失真。

一个最小玩具例子：

- 位置 10 和 15 的 token 内容完全一样。
- Sinusoidal 会给它们两个不同位置向量。
- RoPE 会让它们在注意力时以不同角度参与点积。
- ALiBi 会在两两比较时显式惩罚距离。
- NoPE 则只能靠上下文结构去猜“这个一样的 token 是不是前面那个”。

---

## 代码实现

实现消融实验时，原则只有一条：模型主体不要动，位置编码做成可插拔模块。

下面给一个可运行的 Python 玩具实现，只演示位置编码接口，不依赖深度学习框架。目的是让实验设计先“结构正确”。

```python
import math

def sinusoidal_position(pos: int, d_model: int):
    assert d_model % 2 == 0
    out = []
    for i in range(d_model // 2):
        denom = 10000 ** (2 * i / d_model)
        out.append(math.sin(pos / denom))
        out.append(math.cos(pos / denom))
    return out

def alibi_bias(i: int, j: int, slope: float):
    return -slope * abs(i - j)

def rope_scale_position(pos: int, scale: float = 1.0):
    # NTK/interpolation 类方法的最简抽象：把原始位置先缩放再送入频率函数
    return pos / scale

# 基础性质检查
v10 = sinusoidal_position(10, 8)
v15 = sinusoidal_position(15, 8)
assert len(v10) == 8
assert v10 != v15

assert alibi_bias(10, 15, 0.1) == -0.5
assert alibi_bias(5, 5, 0.1) == 0.0

assert rope_scale_position(2048, scale=2.0) == 1024.0
print("ok")
```

如果写成工程里的伪代码，接口通常像这样：

```python
def apply_positional_encoding(x, q, k, pos_encoding, positions, config):
    if pos_encoding == "sinusoidal":
        x = x + sinusoidal_table(positions, config.d_model)
    elif pos_encoding == "learnable":
        x = x + learned_table(positions)
    elif pos_encoding == "rope":
        q, k = apply_rope(q, k, positions, theta=config.theta)
    elif pos_encoding == "rope_ntk":
        scaled_pos = positions / config.ntk_scale
        q, k = apply_rope(q, k, scaled_pos, theta=config.theta)
    elif pos_encoding == "alibi":
        bias = build_alibi_bias(positions, config.head_slopes)
        return x, q, k, bias
    elif pos_encoding == "nope":
        if config.uniform_scale:
            q = q * config.q_scale
            k = k * config.k_scale
    else:
        raise ValueError("unknown positional encoding")
    return x, q, k, None
```

推荐实验配置如下：

| 项 | 建议值 |
|---|---|
| 训练长度 | 1024 |
| 评估长度 | 1024 / 1536 / 2048 / 4096 |
| 位置编码 | RoPE / RoPE+NTK / ALiBi / Sinusoidal / Learnable / NoPE |
| 指标 | 验证集 PPL、Passkey Accuracy |
| 保持不变 | 层数、参数量、batch、训练 token 数 |

真实工程例子是大模型长上下文扩展。很多基于 RoPE 的模型在原始训练长度外推时会退化明显，例如直接从 4K 推到 8K，PPL 可能上升约 20%；加入 NTK 缩放或位置插值后，升幅可降到约 3%。这说明工程上不是“换个编码就够了”，而是“编码和伸缩策略要一起设计”。

---

## 工程权衡与常见坑

位置编码的工程问题，核心不是“哪个论文公式更漂亮”，而是“部署长度、显存预算、兼容性、外推稳定性怎么平衡”。

| 坑 | 原因 | 现象 | 规避措施 |
|---|---|---|---|
| Learnable 超长失效 | 只为训练长度内的位置建表 | 直接越界或无定义 | 不用于需要外推的系统 |
| Sinusoidal 可算但不稳 | 公式可扩展，不代表统计可泛化 | 长序列 PPL 持续上升 | 只在长度需求稳定时使用 |
| 原始 RoPE 2× 后退化 | 高频旋转在远距离变得难对齐 | PPL 飙升，检索下降 | 用 NTK 缩放或位置插值 |
| NoPE 长距不稳 | 只有隐式位置模式 | 短序列还行，远检索差 | 配合 UniformScale，并补充长距评估 |
| 只看 PPL 不看检索 | PPL 不能完整覆盖远依赖 | 以为模型能长上下文，其实不会找关键字 | 增加 passkey retrieval |

UniformScale 可以理解为对 NoPE 的注意力分布做尺度修正。白话说，没有显式位置信号时，模型容易把注意力过度集中在少数局部模式上，因此需要人为调节 query/key 的尺度，让分布不要过尖。一个抽象写法是：

$$
\tilde{q} = \alpha q,\quad \tilde{k} = \alpha k
$$

其中 $\alpha$ 是按长度或层数设置的缩放因子。它不是在“创造位置”，而是在“避免注意力因为缺少位置先验而失控”。

新手最容易犯的误判有两个。

第一，只在训练长度内比较。这样经常会得出“Learnable、RoPE、ALiBi 都差不多”的结论，但这个结论只对训练内成立，对部署没有指导意义。

第二，只看语言建模，不看 passkey retrieval。模型可能在长文本里依然能保持还行的平均预测，但对“第 3000 个 token 里藏着什么值”这种精确检索已经明显失效。

一个常见真实故障场景是：模型训练长度 1024，推理时硬开到 2048。Learnable 编码直接没法用；Sinusoidal 可以运行但 PPL 明显上升；原始 RoPE 可运行但远检索变差；ALiBi 往往最平稳。工程团队如果只做了短集验证，很容易在线上才发现长文问答性能掉得很厉害。

---

## 替代方案与适用边界

如果目标是通用 LLM，RoPE 仍然是当前最常见的默认选项，因为它兼顾训练效果、相对位置信息和工程可扩展性。很多主流模型采用它，不是因为它在所有条件下都最优，而是因为它在“可训练性 + 工程可补救性”上比较均衡。

如果目标是低开销长外推，ALiBi 常常更直接。它零参数，逻辑简单，推理到更长长度时不需要额外位置表，也通常比原始 RoPE 更稳。

如果目标是教学、公式推导或固定长度模型，Sinusoidal 很合适。它定义清晰，推导容易，也没有额外参数。

如果目标是明确不会超过训练长度，例如某些固定窗口分类器，Learnable 位置编码仍然可用，因为它训练内表现不差。

如果目标是研究“没有显式位置时模型能学到什么”，可以尝试 NoPE，但它更像研究基线，不是大多数生产系统的首选。

| 方案 | 参数开销 | 外推能力 | 工程复杂度 | 适用场景 |
|---|---:|---|---|---|
| Sinusoidal | 0 | 数学可扩展，统计中等 | 低 | 教学、固定长度、基础模型 |
| Learnable | 高于 0 | 差 | 低 | 明确不外推的任务 |
| RoPE | 0 | 中等到强 | 中 | 通用 LLM 默认方案 |
| RoPE + NTK/插值 | 0 | 强 | 中高 | 需要 32K/128K 长上下文 |
| ALiBi | 0 | 很强 | 低 | 重视外推、低开销部署 |
| NoPE | 0 | 不稳定 | 中 | 研究基线、机制分析 |

一个实用选型口诀：

- 想做 128K 长上下文，优先考虑 `RoPE + NTK 缩放/位置插值`。
- 想要简单、稳、低开销，优先考虑 `ALiBi`。
- 只在固定长度内工作，可以考虑 `Learnable`。
- 需要清晰公式和教学可解释性，用 `Sinusoidal`。
- 想研究模型是否会“自己学位置”，用 `NoPE`。

RoPE 插值的抽象代码通常就是先改位置索引，再走原有旋转逻辑：

```python
def rope_interpolated_positions(positions, train_len, infer_len):
    scale = infer_len / train_len
    return [p / scale for p in positions]
```

本质上，它把“更长的实际位置”压回“训练时见过的有效频率范围”，从而减轻高频旋转失真。

---

## 参考资料

| 名称 | 重点 | 访问提示 |
|---|---|---|
| Positional Encoding Techniques | 系统梳理 Sinusoidal、Learnable、RoPE、ALiBi、NoPE 的定义与适用性 | 想查公式和总览先看这篇 |
| What Is Positional Encoding? A Complete Guide (2026) | 面向工程落地的编码类型综述 | 想快速建立全局概念可先读 |
| The Impact of Positional Encoding on Length Generalization in Transformers | 长度外推实验与评估设计，尤其是训练长度外泛化 | 想看消融和长度泛化证据重点看 |
| Bayesian Attention Mechanism 相关评述 | 涉及长上下文评估、passkey retrieval 等任务视角 | 想理解为什么不能只看 PPL 时可参考 |
| All About the Modern Positional Encodings | 对 ALiBi、RoPE 等现代方案给出更贴近实践的解释 | 想看工程直觉和例子可读这篇 |
| Positional Encoding 工程综述文章 | 汇总常见坑，如 Learnable 失效、RoPE 外推退化、NoPE 稳定化技巧 | 想看部署坑点和规避方法可参考 |

可访问链接：

- https://artificial-intelligence-wiki.com/ai-research/foundation-models-and-architectures/positional-encoding-techniques/
- https://openreview.net/forum?id=Drrl2gcjzl
- https://newsletter.theaiedge.io/p/all-about-the-modern-positional-encodings
- https://www.articsledge.com/post/positional-encoding
