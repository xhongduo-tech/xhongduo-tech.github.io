## 核心结论

Relative Position Encoding，中文常译为“相对位置编码”，指模型不直接记住“第 17 个词”“第 83 个词”这种绝对编号，而是在注意力分数里显式加入“两个词相距多远、方向是什么”的信息。

标准自注意力只计算内容相似度：

$$
e_{ij}=\frac{Q_iK_j^T}{\sqrt{d}}
$$

加入相对位置偏置后，分数变为：

$$
e_{ij}=\frac{Q_iK_j^T}{\sqrt{d}}+b_{i-j}
$$

这里的 $b_{i-j}$ 是“距离偏置”。白话说，模型在决定“我该看谁”时，不只看内容像不像，还会看“这个位置离我近不近、在我前面还是后面”。

它的工程价值很直接：

| 结论 | 含义 |
|---|---|
| 对距离建模，而不是对绝对编号建模 | 同一种“距离模式”可以复用到不同长度序列 |
| 直接加在 attention logits 上 | 位置信息不会像输入嵌入那样更容易被后续层中和 |
| 近距离可细分，远距离可合并 | 既保留局部顺序，又控制参数量与显存 |
| 更适合长文本泛化 | 训练时 512 长度，推理到 1024 或更长时通常更稳 |

一个最小数值例子：若 $Q_iK_j^T=2$，$\sqrt{d}=2$，且某个相对距离对应偏置 $b_{i-j}=0.5$，则

$$
e_{ij}=\frac{2}{2}+0.5=1.5
$$

softmax，中文叫“归一化指数函数”，会把更大的 logits 变成更大的注意力权重。所以这个偏置虽然只是加了 0.5，却能明显改变“更愿意看谁”。

玩具例子：句子是“我 喜欢 吃 苹果”。如果当前 query 是“吃”，模型通常应更关注前面的宾语或动词补足结构。只靠内容相似度，远处一个语义相关词可能分数更高；加入相对偏置后，模型可以额外学到“距离 1 到 3 的位置常常更重要”。

真实工程例子：T5 使用按 bucket 分桶的相对位置 bias；Transformer-XL 则把相对位置直接写进注意力计算。它们的共同点都是把“相对距离”放进 attention score，而不是只在输入层加一个位置向量。

---

## 问题定义与边界

Transformer 原始注意力机制的核心能力是“按内容匹配”。这很强，但天然有一个缺口：内容相似不等于顺序合理。

“位置编码”解决的是顺序问题。但这里要分清两个层面：

| 问题 | Relative Position Encoding 能否解决 |
|---|---|
| 谁在谁前面、相隔多远 | 能 |
| 序列第 1 个位置和第 500 个位置的绝对身份 | 不是主要目标 |
| 长度从训练 512 扩展到推理 1024 | 通常更容易泛化 |
| 无限长上下文、完全无损外推 | 不能保证，仍要看设计 |

绝对位置嵌入，白话说是“给每个位置一个固定 ID 向量”；相对位置偏置则是“给任意两个位置之间的距离一个偏置”。

两者对比如下：

| 维度 | 绝对位置嵌入 | 相对位置偏置 |
|---|---|---|
| 建模对象 | 单个位置编号 | 两个位置的距离与方向 |
| 进入模型的位置 | 通常加到输入 embedding | 直接加到 attention logits |
| 长度泛化 | 较弱，常受最大长度限制 | 较强，可复用距离模式 |
| 是否易跨层共享 | 通常不共享 | 常可共享或结构一致 |
| 对顺序归纳偏置 | 有，但偏间接 | 更直接 |

看两个场景。

场景 A：训练 512，推理 512。  
无论绝对还是相对方法，都能工作，因为训练和推理长度一致，分布没有明显外移。

场景 B：训练 512，推理 1024。  
如果模型只见过绝对位置 0 到 511，那么超过训练范围的绝对位置要么没有参数，要么需要插值、裁剪或外推。相对偏置则不同：距离模式例如“看前一个 token”“看前 8 个 token 内的信息”在 1024 长度里仍然成立。

可以把它想成一个简单的距离映射图：

| 相对距离 $\Delta=i-j$ | bucket |
|---|---|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3,4 | 3 |
| 5~8 | 4 |
| 9~16 | 5 |
| 17~32 | 6 |
| 33 以上 | 7 或更大桶 |

这个设计的边界也要说清楚。Relative Position Encoding 不是“自动理解语法”的魔法。它只是提供一个更合理的顺序先验。内容仍然是主导项，偏置只是帮助模型在内容相近时更偏向某些距离模式。

---

## 核心机制与推导

先看最常见、最容易理解的一版：

$$
e_{ij}=\frac{Q_iK_j^T}{\sqrt{d}}+b_{i-j}
$$

各项含义如下：

| 项 | 作用 | 白话解释 |
|---|---|---|
| $Q_i$ | query 向量 | “当前位置想找什么” |
| $K_j$ | key 向量 | “候选位置能提供什么” |
| $\frac{Q_iK_j^T}{\sqrt{d}}$ | 内容匹配分数 | 内容越相关，分数越高 |
| $b_{i-j}$ | 相对位置偏置 | 按距离修正分数 |

更完整的形式常写为：

$$
e_{ij}=\frac{(x_iW^Q + p_{i,j}^Q)(x_jW^K + p_{i,j}^K)^T}{\sqrt{d}} + b_{i-j}
$$

这里：

- $x_iW^Q$ 和 $x_jW^K$ 是内容投影。
- $p_{i,j}^Q, p_{i,j}^K$ 是相对位置向量，表示“位置 $i$ 看位置 $j$ 时的相对关系”。
- $b_{i-j}$ 是最直接的距离偏置。

把括号展开后，会得到几类项：

$$
(x_iW^Q)(x_jW^K)^T + (x_iW^Q)(p_{i,j}^K)^T + (p_{i,j}^Q)(x_jW^K)^T + (p_{i,j}^Q)(p_{i,j}^K)^T
$$

含义是：不仅内容和内容能相乘，内容和相对位置、相对位置和相对位置也能相互作用。这样模型不只是知道“这两个词内容相关”，还知道“这种相关是否发生在合适的距离上”。

再看一个具体数值。若

- $Q_iK_j^T=2$
- $\sqrt{d}=2$
- $b_{i-j}=0.5$

则：

$$
e_{ij}=2/2+0.5=1.5
$$

如果另一个位置内容分数一样，但偏置是 $-0.3$，则 logits 变成 $0.7$。softmax 后，前者权重会大很多。也就是说，相对位置偏置不是装饰，而是真实参与竞争。

T5 的关键实现是 bucket bias。bucket，白话说是“把很多距离压缩到少数几档”。原因很实际：近距离差异很重要，远距离没必要每个距离都单独学习。

一个典型映射可以写成：

| 距离 $|\Delta|$ | bucket 示例 |
|---|---|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4~7 | 4 |
| 8~15 | 5 |
| 16~31 | 6 |
| 32~63 | 7 |
| 64 以上 | 8 或最后一个桶 |

这相当于“近距离精细，远距离粗略”。它对长文本尤其重要，因为模型真正高频使用的通常是局部依赖，而不是精确区分“相距 173”和“相距 174”。

可以把 attention map 的变化理解为一张热力图：

| 情况 | 热力图特征 |
|---|---|
| 只有内容分数 | 高亮区域由语义相似决定，可能分散 |
| 加相对偏置 | 某些固定距离带更亮，例如主对角线附近更亮 |
| 加因果 mask 后 | 只保留下三角，再叠加距离偏好 |

玩具例子：序列 `[A, B, C, D]`。当前看 `C`。  
若 `A` 和 `B` 的内容分数都差不多，只靠内容时模型可能随机偏向；若系统学到“距离 1 的 token 更常提供局部依赖”，则 `B` 的权重会更高，因为它与 `C` 的相对距离是 1。

真实工程例子：机器翻译中，局部短语对齐很常见，例如英文形容词与名词、助动词与谓词通常在有限距离内相关。相对位置偏置能帮助模型稳定形成“某类信息常出现在附近”的归纳偏置，即使句子总长度从训练时的 128 扩到推理时的 512，这种局部模式仍可复用。

---

## 代码实现

实现上最核心的一步只有一句话：先算内容 logits，再把相对位置 bias 加上去，同时保留 mask。

下面是一个可运行的 Python 玩具实现，演示“距离分桶 + logits 加偏置 + softmax 权重变化”。

```python
import math

def relative_bucket(delta: int) -> int:
    # 只做一个简化版：保留方向，近距离细分，远距离合并
    sign = 0 if delta >= 0 else 4
    d = abs(delta)
    if d == 0:
        base = 0
    elif d == 1:
        base = 1
    elif d <= 3:
        base = 2
    else:
        base = 3
    return sign + base

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

# 假设 query 位置 i=3，分别看 j=0,1,2,3
content_logits = [0.8, 1.0, 1.0, 0.2]

# 8 个 bucket 的共享 bias
bias_table = [
    0.1,   # delta=0
    0.4,   # |delta|=1, forward/backward one side
    0.2,   # |delta| in 2~3
    -0.1,  # far
    0.1,
    0.3,
    0.1,
    -0.2,
]

i = 3
final_logits = []
for j, base_logit in enumerate(content_logits):
    delta = i - j
    bucket = relative_bucket(delta)
    final_logits.append(base_logit + bias_table[bucket])

weights = softmax(final_logits)

# 距离更近的位置 j=2 往往得到更大权重
assert len(weights) == 4
assert abs(sum(weights) - 1.0) < 1e-9
assert weights[2] > weights[0]
print(final_logits)
print(weights)
```

这段代码没有依赖深度学习框架，但逻辑和真实模型一致。

如果写成伪代码，主体通常是：

```python
# q, k: [batch, heads, seq, dim]
logits = matmul(q, k.transpose(-1, -2)) / sqrt(dim)

# relative_positions[i, j] = i - j
bucket_idx = bucketize(relative_positions)   # [seq, seq]
bias = bias_table[bucket_idx]                # [seq, seq] or [heads, seq, seq]

logits = logits + bias
logits = logits + attention_mask
weights = softmax(logits)
```

关键数据结构如下：

| 结构 | 典型形状 | 说明 |
|---|---|---|
| `relative_positions` | `[L, L]` | 每对位置的距离 |
| `bucket_idx` | `[L, L]` | 距离映射后的桶编号 |
| `bias_table` | `[num_buckets]` 或 `[heads, num_buckets]` | 可共享或分头学习 |
| `bias` | `[L, L]` 或 `[heads, L, L]` | 查表后的偏置矩阵 |

真实工程里常见两种实现策略：

| 策略 | 特点 |
|---|---|
| 预计算 `[L, L]` bucket 矩阵 | 速度快，适合固定长度 |
| 动态按当前长度生成 | 更灵活，适合变长输入与缓存推理 |

如果是 sliding window attention，还可以缓存一部分 bias，因为窗口结构重复，没必要每次重新算完整矩阵。

---

## 工程权衡与常见坑

Relative Position Encoding 很实用，但它不是“加上就一定更强”。

最常见的问题如下：

| 问题/现象 | 原因 | 规避方案 |
|---|---|---|
| 长文本推理退化 | bucket 设计只覆盖训练区间，超长距离全被压到同一桶 | 扩展对数区间，单独评估超长长度 |
| 局部依赖过强 | 偏置过大，模型过度偏爱近邻 | 控制初始化尺度，做消融实验 |
| 多头表达冗余 | 每个 head 都学类似 bias | 共享部分 bias，减少参数 |
| 与 mask 交互错误 | 先后顺序错，mask 被偏置污染 | 先加 bias，再加 `-inf` mask |
| 训练稳定但外推差 | 桶太少或分桶不合理 | 近距离更细、远距离按对数扩展 |

一个典型坑是“训练 512，推理 2048，但 bucket 设计没有考虑 2048”。结果是 512 以外的大量距离都落入最后一个桶，模型只能知道“很远”，却分不清“远”和“特别远”。此时 attention map 在长区间上会退化得像“纯内容匹配”。

这也是为什么很多论文会把它和 ALiBi 对比。ALiBi 直接使用线性斜率，不需要离散分桶，超长外推往往更平滑；但它的表达能力不一定总优于可学习 bucket bias，尤其在需要复杂局部模式时。

真实工程例子：做长文档摘要时，训练窗口是 512，线上推理到 2048。如果发现模型在前几百 token 后注意力开始明显发散，首先不要只怀疑数据质量，先检查：

1. bucket 数是否太少。
2. 是否把超长距离全部 clip 到最后一个桶。
3. 不同层是否共享过头，导致表达不足。
4. 长度外推是否做过专门 ablation。

ablation，中文常译“消融实验”，意思是一次只改一个变量，看效果变化。这里最基本的 ablation 是：

| 实验 | 目的 |
|---|---|
| 去掉 relative bias | 看纯内容基线 |
| 保留 bias，但增加 bucket 数 | 看容量是否不足 |
| 换成 ALiBi | 看是否需要更强外推 |
| 保持长度不变，只换位置方案 | 排除数据与训练波动 |

---

## 替代方案与适用边界

Relative Position Encoding 不是唯一方案。实际常拿来对比的至少有三类：relative bias、ALiBi、NoPE。

| 方案 | 参数量 | 可扩展性 | 训练复杂度 | 适用场景 |
|---|---|---|---|---|
| Relative bias | 低到中 | 较强 | 中 | 通用 encoder/decoder、需要局部模式 |
| ALiBi | 几乎无参数 | 很强 | 低 | 追求长度外推、实现简单 |
| NoPE | 无 | 依赖模型自发学习 | 最低 | 研究对照、特定大模型设定 |
| 相对位置向量嵌入 | 中到高 | 取决于设计 | 中到高 | 需要更强位置交互表达 |

ALiBi 的核心是给不同 head 一个固定斜率，让距离越远的 key 受到越大的线性惩罚。白话说，它不学“桶”，而是直接规定“越远越吃亏”。优点是简单、稳定、超长外推常常不错；缺点是表达形式比较硬。

NoPE 指“不显式加入位置编码”。它不是说模型完全不处理顺序，而是把顺序信息交给网络结构和训练数据间接形成。但对多数中小模型或训练资源有限的场景，这通常不是最稳妥的默认选项。

以 T5 风格场景为例，可以这样理解三者：

| 方案 | 可能结果 |
|---|---|
| T5 relative bias | 局部模式和长度共享都较平衡 |
| ALiBi | 超长外推更稳，但某些任务精细位置模式较弱 |
| NoPE | 在部分大规模训练下可行，但对普通工程项目风险更高 |

适用边界可以概括为：

- 如果你要兼顾表达力和工程成熟度，relative bias 是稳妥选择。
- 如果你的核心目标是长度外推，且实现要尽量简单，ALiBi 值得优先试。
- 如果你在做研究对照，或者模型规模极大、训练数据极多，可以把 NoPE 作为实验组，而不是默认生产方案。

结论不是“relative bias 永远最好”，而是“它在多数需要顺序建模、又希望长度泛化的 Transformer 场景里，是一个非常强的默认基线”。

---

## 参考资料

- Shaw et al., *Self-Attention with Relative Position Representations*  
  早期经典工作，核心贡献是把相对位置表示直接注入自注意力，而不是只做绝对位置嵌入。  
  URL: https://arxiv.org/abs/1803.02155

- Raffel et al., *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*  
  T5 论文，采用 relative position bias，并展示了大规模文本到文本框架中的有效性。  
  URL: https://arxiv.org/abs/1910.10683

- Dufter, Schmitt, Schütze, *Position Information in Transformers: An Overview*  
  对绝对、相对、旋转等位置方法做系统梳理，适合建立整体知识地图。  
  URL: https://direct.mit.edu/coli/article/48/3/733/111478/Position-Information-in-Transformers-An-Overview

- *Improving Position Encoding of Transformers for Longer Sequences* 相关综述与整理  
  总结了相对位置向量 $p_{i,j}^{Q/K/V}$ 等更完整的数学形式，适合从公式角度理解。  
  URL: https://link.springer.com/article/10.1007/s10618-023-00948-2

- Press et al., *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*  
  ALiBi 论文，核心思想是用线性 bias 代替离散位置编码，强调长度外推能力。  
  URL: https://arxiv.org/abs/2108.12409

- Transformer-XL: *Attentive Language Models Beyond a Fixed-Length Context*  
  通过相对位置机制与段级记忆突破固定上下文限制，是长上下文建模的重要里程碑。  
  URL: https://arxiv.org/abs/1901.02860

- OpenReview 上关于位置编码与长度泛化的分析工作  
  讨论显式位置编码、相对偏置、NoPE 等方法在长度外推上的差异，提醒工程上必须做实测。  
  URL: https://openreview.net/forum?id=Drrl2gcjzl
