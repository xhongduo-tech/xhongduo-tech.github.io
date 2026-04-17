## 核心结论

相对位置编码的核心，不是给每个 token 一个固定“门牌号”，而是让模型直接感知“另一个 token 离我多远”。T5 采用的是**相对位置偏置**（relative position bias），意思是在注意力分数里额外加一个只由相对距离决定的偏置项。

如果把第 $i$ 个 query 和第 $j$ 个 key 之间的注意力看成一座桥，那么普通绝对位置编码更像先给每个位置修一条编号公路，再让模型自己学“编号 17 和编号 23 的关系”；T5 则先计算桥的跨度 $i-j$，再按跨度查一个偏置值 $u$，直接把“近一点还是远一点”写进注意力 logits。跨度越远，桥可能越难过，但不需要预先准备一张和序列长度绑定的大位置表。

T5 这套方法有两个关键优点：

1. 注意力依赖的是距离，而不是固定索引，所以对“前一个词”“后两个词”“远处的标题”这类关系更自然。
2. 偏置表大小由 `num_buckets` 和 `num_heads` 决定，不随序列长度线性增长；序列能跑多长，更多取决于显存和注意力矩阵本身，而不是位置 embedding 表是否越界。

下表先把“绝对位置”和“相对位置”的依赖差异摆清楚：

| 方案 | 关注对象 | 参数是否绑定最大长度 | 对长序列外推 | 直观含义 |
|---|---|---:|---:|---|
| 绝对位置编码 | 位置编号 $i$ | 是 | 弱 | “我在第 37 个位置” |
| 相对位置偏置 | 距离 $i-j$ | 否，主要由 bucket 决定 | 更自然 | “它离我 3 格远” |

---

## 问题定义与边界

先定义问题。Transformer 自注意力本身只看向量相似度，不知道 token 顺序。也就是说，如果没有额外位置信息，“猫咬狗”和“狗咬猫”在结构上会被看成一组无序元素。**位置编码**就是把顺序信息补回去的机制。

传统绝对位置编码的边界很明确：如果模型预训练时只准备了 512 个位置 embedding，那么推理时输入 2048 个 token，就必须想办法扩展位置表，或者直接越界。这里的“位置表”可以理解成一个查表矩阵，第 0 个位置查第 0 行，第 1 个位置查第 1 行，直到最大长度。

T5 处理的不是“第几个位置”，而是“两个位置相差多少”。因此它不需要一张长度为 2048 的绝对位置表。运行时只要能构造出距离矩阵 $(i-j)$，再把距离映射到 bucket，就能继续给 attention 加偏置。

玩具例子：

- 序列长度是 8。
- query 在位置 6，key 在位置 2。
- 绝对位置方法关心“6 号位置向量”和“2 号位置向量”。
- T5 相对偏置关心距离 $6-2=4$，再查“距离 4 对应哪个 bucket、该加多少偏置”。

真实工程例子：

- 你用 T5 处理一篇 2048 token 的技术文档摘要。
- 绝对位置编码通常要求模型事先拥有至少 2048 个位置槽位。
- T5 不需要新增 2048 行位置 embedding，只需在运行时生成 $2048 \times 2048$ 的距离关系，再把距离压到固定数量的 bucket 中。

可以把边界关系概括成下面这张表：

| 项目 | 绝对位置编码 | T5 相对位置偏置 |
|---|---|---|
| 位置参数规模 | 随最大位置数增长 | 随 bucket 数增长 |
| 序列变长时是否要扩表 | 常常需要 | 通常不需要 |
| 真正限制长序列的因素 | 位置表 + 注意力显存 | 主要是注意力显存 |
| 是否天然表达“相距 3 个 token” | 不直接 | 直接 |

边界也要说清楚。T5 并不是“无限长序列免费支持”。它只是**不再被绝对位置表卡死**。真正的上限仍然来自注意力矩阵的内存开销，因为自注意力本身是 $O(n^2)$。所以正确表述是：相对位置偏置解除的是“位置参数表长度”这层约束，不是把长序列计算成本消掉。

---

## 核心机制与推导

T5 的注意力公式可以写成：

$$
\alpha(i,j)=\mathrm{Softmax}\left(\frac{q_i^\top k_j}{\sqrt{d}} + u_{b(i-j)} + \mathrm{Mask}(i,j)\right)
$$

这里各项含义如下：

- $q_i$：第 $i$ 个位置的 query 向量，白话说，就是“当前位置拿去发起匹配的表示”。
- $k_j$：第 $j$ 个位置的 key 向量，白话说，就是“当前位置暴露给别人匹配的表示”。
- $\frac{q_i^\top k_j}{\sqrt{d}}$：原始注意力相似度。
- $b(i-j)$：bucket 函数，把距离映射到离散桶编号。
- $u_{b(i-j)}$：该 bucket 对应的可学习偏置。
- `Mask`：掩码，用来禁止看未来或忽略 padding。

关键是中间这一项 $u_{b(i-j)}$。它说明位置关系不是混在输入 embedding 里，而是直接加到 logits 上。这样做的含义很直接：模型在做 softmax 前，就已经把“距离近还是远”作为一个显式偏好写进分数。

一个最小数值例子：

- 假设某个 head 上，$q_i^\top k_j=20$
- 维度 $d=128$
- 距离 5 被映射到 bucket 3
- 该 bucket 学到的偏置是 $u_3=-2$

那么该 pair 的打分可写成：

$$
\frac{20 + (-2)}{\sqrt{128}} \approx 1.59
$$

如果再加上 mask，就送入 softmax。这里的负偏置表示：在这个 head 看来，距离为 5 的连接要比原始相似度略难成立。

为什么要用 bucket，而不是每个距离一个单独参数？因为距离空间太大。若序列长度很长，距离可能从 $-(n-1)$ 到 $n-1$。T5 的做法是：

- 近距离：细分，尽量一对一保留。
- 远距离：压缩，把很多大距离合并到少数几个桶。

常见思想可以写成：

$$
b(\Delta)=
\begin{cases}
\Delta, & |\Delta| < k \\
k + \lfloor \log(|\Delta|/k) \rfloor, & |\Delta| \ge k
\end{cases}
$$

这不是唯一实现，但体现了核心思想：**近处精细，远处粗粒度**。

这也解释了 T5 为什么适合语言建模。语言里最敏感的常常是近邻关系，比如主谓、修饰、标点、局部短语；而非常远的关系，很多时候只需要知道“很远”这一事实，不必精确区分 200 和 213。

---

## 代码实现

实现时，核心不是改动 $QK^\top$ 的计算，而是在 logits 上额外加一个形状可广播的 bias 张量。通常需要维护一个大小为 `num_heads x num_buckets` 的参数表。

下面给出一个可运行的 Python 玩具实现，演示“距离矩阵 -> bucket -> 偏置 -> logits”的完整路径：

```python
import math

def relative_position_bucket(relative_position, num_buckets=8, max_distance=16):
    bidirectional = True
    n = relative_position
    result = 0

    if bidirectional:
        half = num_buckets // 2
        result += 0 if n < 0 else half
        n = abs(n)
        num_buckets = half
    else:
        n = max(0, -n)

    max_exact = num_buckets // 2
    if n < max_exact:
        return result + n

    if n >= max_distance:
        return result + num_buckets - 1

    ratio = math.log(n / max_exact) / math.log(max_distance / max_exact)
    bucket = max_exact + int(ratio * (num_buckets - max_exact))
    bucket = min(bucket, num_buckets - 1)
    return result + bucket

def build_bias(seq_len, num_heads=2, num_buckets=8, max_distance=16):
    table = [
        [0.0, -0.1, -0.2, -0.3, -0.5, -0.7, -1.0, -1.2],
        [0.0,  0.1,  0.2,  0.3,  0.2,  0.0, -0.2, -0.4],
    ]
    bias = [[[0.0 for _ in range(seq_len)] for _ in range(seq_len)] for _ in range(num_heads)]

    for i in range(seq_len):
        for j in range(seq_len):
            rel = i - j
            b = relative_position_bucket(rel, num_buckets=num_buckets, max_distance=max_distance)
            for h in range(num_heads):
                bias[h][i][j] = table[h][b]
    return bias

def attention_logit(qk_score, d_model, bias_value, mask_value=0.0):
    return qk_score / math.sqrt(d_model) + bias_value + mask_value

bias = build_bias(seq_len=4, num_heads=2)
logit = attention_logit(qk_score=20.0, d_model=128, bias_value=-0.2)
assert len(bias) == 2
assert len(bias[0]) == 4 and len(bias[0][0]) == 4
assert round(logit, 3) == round(20.0 / math.sqrt(128) - 0.2, 3)
assert relative_position_bucket(1) != relative_position_bucket(12)
```

如果写成更贴近框架的伪码，逻辑一般是：

```python
# positions: [seq]
q_pos = arange(seq_len)[:, None]      # [seq, 1]
k_pos = arange(seq_len)[None, :]      # [1, seq]
rel = q_pos - k_pos                   # [seq, seq]

bucket = relative_position_bucket(rel)    # [seq, seq]
bias = bias_table[:, bucket]              # [heads, seq, seq]
bias = bias[None, :, :, :]                # [batch=1, heads, seq, seq]

logits = (Q @ K.transpose(-1, -2)) / sqrt(d)
logits = logits + bias + mask
attn = softmax(logits, dim=-1)
```

实现细节上有两个要点：

1. bucket 函数通常会做对数压缩和裁剪（clip），避免远距离桶无限增长。
2. bias 张量不是对每个 token 单独存一份，而是由距离矩阵动态索引生成，所以参数规模稳定。

---

## 工程权衡与常见坑

T5 相对位置偏置很实用，但不是“开箱即无脑正确”。下面是工程上最常见的几个坑：

| 问题 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| `max_distance` 太小 | 远距离依赖分不清 | 很多远距离被压进同一桶 | 增大 `max_distance` 或增加 bucket 数 |
| bucket 数太少 | 近距离信息过早被合并 | 分辨率不足 | 近距离任务提高桶密度 |
| 训练/推理 bucket 逻辑不一致 | 推理效果明显崩 | 同一距离映射到不同偏置 | 复用同一实现与配置 |
| 误以为支持超长序列就没有代价 | 显存仍爆 | 注意力矩阵还是 $O(n^2)$ | 结合稀疏注意力或分块方案 |
| 单向/双向设置错 | 编码器或解码器行为异常 | 距离符号处理不同 | 明确 encoder 与 decoder 的 mask 规则 |

最常见的真实工程坑，是 `relative_attention_max_distance` 默认值不适配任务。假设默认是 128，而你的任务需要区分距离 200 和 400 的依赖，比如长代码文件中的变量定义和使用位置、长文档中的章节引用。那么这些距离可能都被映射到同一个“很远” bucket。结果就是：

- 模型知道“它们都很远”
- 但不知道“200”和“400”是否应该有不同偏好”

这在低层尤其明显，因为底层常常承担局部结构建模任务，若远距离全被压平，细粒度依赖会丢失。

另一个容易忽略的问题是实现一致性。训练时如果 bucket 逻辑是“近距离精确、远距离对数压缩”，推理时却换成了别的 clip 规则，那么距离和偏置的对应关系就错位了。这个错误不会像 shape mismatch 那样立刻报错，但模型会出现“能跑、结果差”的隐蔽故障，排查成本很高。

---

## 替代方案与适用边界

T5 的相对位置偏置不是唯一方案。它的优势是结构简单、参数少、工程上容易落地，但不同任务可能更适合别的方法。

先看几种常见方案的对比：

| 方案 | 核心思想 | 额外参数 | 长序列兼容性 | 优点 | 局限 |
|---|---|---:|---:|---|---|
| T5 Relative Bias | 距离映射到可学习偏置 | 低 | 较好 | 简洁、稳定、容易插入 logits | 远距离会被 bucket 压缩 |
| Shaw Relative Embedding | 为相对距离学习向量表示 | 中到高 | 一般 | 粒度更细，可直接参与表示计算 | 参数和实现更重 |
| ALiBi | 直接加线性距离惩罚 | 极低 | 很好 | 外推简单，几乎不加参数 | 偏置形式固定，表达力更硬 |
| Rotary Embedding | 用旋转变换把位置信息写进向量 | 低 | 较好 | 与点积结构耦合自然，现代表现强 | 不像 T5 那样显式可控 |

T5 和 ALiBi 的区别尤其适合初学者把握：

- T5：学习一个“距离到偏置”的查表函数，表达力更强。
- ALiBi：直接规定偏置随距离线性变化，几乎没有额外学习负担。

如果是极端长序列场景，比如几万 token 的上下文，ALiBi 往往更简单，因为它不需要离散 bucket，也没有“超远距离都挤在最后一桶”的问题。但它的斜率设计需要更谨慎，偏置形式也更刚性。

如果任务特别依赖精细相对距离，例如代码补全、结构化文本对齐、局部模式识别，T5 的 learnable bias 往往比纯手工线性偏置更灵活。若进一步追求高性能和现代大模型兼容性，Rotary 也常是更强候选，但它和 T5 的设计哲学不同：Rotary 是把位置信息混入向量几何结构，而 T5 是把位置信息作为独立偏置加到 logits。

因此，适用边界可以简化为：

- 想要简单、稳定、参数小，且显式建模“相对远近”：T5 relative bias 合适。
- 想要极长序列外推，接受更强先验：ALiBi 更直接。
- 想要现代主流大模型兼容和较强表达：Rotary 常更常见。
- 想要更细粒度相对表示，愿意承担更重实现：Shaw 类方法可考虑。

---

## 参考资料

1. T5 论文《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》及相关实现说明。用途：原始相对位置偏置设计来源，定义了把距离映射为 attention bias 的核心思路。  
2. AIML.com 对 T5 架构的解析。用途：适合理解 T5 编码器-解码器结构，以及相对位置偏置在整体架构中的位置。  
3. Hugging Face 社区关于 T5 长序列输入与 `relative_attention_max_distance` 的讨论。用途：帮助理解 bucket、max distance、长序列推理时的实际工程边界。
