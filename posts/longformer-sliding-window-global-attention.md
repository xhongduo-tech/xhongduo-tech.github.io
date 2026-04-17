## 核心结论

Longformer 解决的是“序列很长时，标准 Transformer 的注意力算不动”这个问题。标准自注意力的计算和显存开销是 $O(N^2)$，也就是序列长度 $N$ 翻倍，代价接近变成 4 倍。Longformer 把这件事拆成两部分：

1. 滑动窗口注意力。滑动窗口注意力的意思是：每个 token 只看自己左右一小段邻居，而不是看全体 token。
2. 全局注意力。全局 token 的意思是：少数被显式标记的重要位置，可以和整段文本双向通信。

因此，总复杂度从全连接的 $O(N^2)$ 下降到近似线性的

$$
C_{\text{total}} = O(Nw + N_gN)
$$

其中 $w$ 是窗口大小，$N_g$ 是全局 token 数量。只要 $w$ 和 $N_g$ 相对固定，整体就随 $N$ 线性增长。

玩具例子可以这样理解：一篇 4096 token 的长文里，普通词只看左右各 256 个词；但 `[CLS]`、问题词、段标题这些关键位置被设为全局 token，它们既能读取全文，也能被全文读取。局部窗口负责处理邻近上下文，全局 token 负责跨段汇总信息。

这套设计的核心价值不是“近似全注意力”，而是“明确牺牲稠密连接，换取长文本可用性”。在 $N=4096, w=512, N_g=2$ 的设置下，注意力连接密度大约只有全注意力的 12.6%，显存需求大约降到原来的 1/8 左右，但仍保留了跨段信息通路。

再强调一遍结论：Longformer 不是把全注意力“压缩一下”，而是把注意力图改成“局部为主、全局为辅”的稀疏结构。你要接受一部分连接被直接删掉，换来更长上下文长度和更可控的显存成本。

---

## 问题定义与边界

先定义问题。Longformer 面向的是长文档建模，例如长文分类、抽取式问答、文档级检索、合同分析。这里的“长”通常指超出 BERT 常见 512 token 上限，进入 2k、4k 甚至更长范围。

边界也要说清楚。Longformer 不是让“所有 token 仍然自由互看，只是更省”，而是把 token 分成两类：

| 类型 | 能看见谁 | 能被谁看见 | 典型用途 |
| --- | --- | --- | --- |
| 局部 token | 窗口内邻居 + 所有全局 token | 窗口内邻居 + 所有全局 token | 普通正文、局部语义建模 |
| 全局 token | 全文所有 token | 全文所有 token | `[CLS]`、问题词、标题、特殊锚点 |

它的注意力掩码可以写成：

$$
A_{ij} =
\begin{cases}
Q_iK_j^\top, & |i-j|\leq \frac{w}{2}\ \text{or}\ i\in\mathcal{G}\ \text{or}\ j\in\mathcal{G} \\
-\infty, & \text{otherwise}
\end{cases}
$$

其中 $\mathcal{G}$ 是全局 token 集合。白话说法是：除了窗口邻居和少数特权位置，其他远处 token 一律不允许直接通信。

这里要区分两个概念：

| 概念 | 含义 | 容易混淆的点 |
| --- | --- | --- |
| “能看见” | 当前 token 在计算注意力时，可以访问哪些 key/value | 这是行方向约束 |
| “能被看见” | 其他 token 在计算注意力时，是否可以访问当前 token | 这是列方向约束 |
| 全局 token | 同时打通这两个方向 | 不是只“主动看全文”，而是“主动看全文 + 被全文看” |

这里最容易误解的点有两个。

第一，滑窗不是“自动具备全局理解”。如果没有全局 token，远距离依赖只能靠多层逐步传递，路径会很长。比如问题在文档开头、答案在文档末尾，只靠局部窗口，信息必须跨多层接力。

第二，全局 token 不是模型自己必然会选出来。在工程实现里，通常需要你显式传入 `global_attention_mask`。不标记，模型就按普通局部 token 处理，它不会凭空获得全文可见性。

以问答任务为例，问题 tokens 通常应设为全局 token。原因很直接：问题中的关键词要和全文各段落直接匹配，而不是等待多层传播后才“碰到”答案区域。如果这一步没做，对长距离证据的召回会明显变差。

再给一个新手更容易理解的例子。假设有一份 30 页合同，问题是“违约金比例是多少”。这时：

- 普通正文 token 只需要理解附近语句，比如“甲方”“乙方”“违约金”“百分之十”之间的局部关系。
- 问题 token “违约金比例”应该和全文所有页面直接建立联系，因为答案可能出现在任意一段。
- 章节标题如“违约责任”“费用结算”也适合设为全局 token，因为它们天然起索引作用。

所以 Longformer 的边界很明确：它适合“局部上下文占大头，少数位置需要全局访问”的任务；不适合“几乎所有位置都需要频繁远距离交互”的任务。

---

## 核心机制与推导

Longformer 的推导并不复杂，关键在于把连接数数清楚。

标准全注意力里，每个 token 都和其余 $N$ 个 token 交互，所以连接规模约为：

$$
N \times N = N^2
$$

Longformer 中，普通 token 只连接一个固定窗口。若窗口大小为 $w$，那么 $N$ 个 token 的局部连接数约为：

$$
Nw
$$

再加上 $N_g$ 个全局 token。每个全局 token 需要和全文双向交互，代价约为：

$$
N_gN
$$

所以总成本是：

$$
C_{\text{total}} = Nw + N_gN = N(w+N_g)
$$

因为 $w$、$N_g$ 通常远小于 $N$，而且经常被当作常数，所以它是线性增长。

这里要补一个容易被忽略的细节。严格实现里，局部窗口和全局连接之间可能有重复边，例如某个全局 token 同时也落在局部窗口内。因此上式是工程上常用的数量级估算，不是逐边去重后的精确公式。分析复杂度时，这样写已经足够。

拿一个最小数值例子：

- 序列长度 $N=4096$
- 窗口大小 $w=512$
- 全局 token 数 $N_g=2$

若按近似连接数算，Longformer 需要处理的连接规模约为：

$$
4096 \times (512 + 2) = 2{,}105{,}344
$$

全注意力则是：

$$
4096^2 = 16{,}777{,}216
$$

比例约为：

$$
\frac{2{,}105{,}344}{16{,}777{,}216} \approx 12.55\%
$$

这就是“4k 长文显存约降到 1/8”的来源。注意这是连接规模的近似比例，不等于实际训练显存一定严格降成 12.55%。真实显存还会受 batch size、hidden size、层数、激活保存、实现方式影响。

再看一个更直观的“玩具例子”。假设只有 16 个 token，窗口大小 $w=4$，第 0 个 token 是全局 token。  
那么：

- 第 7 个 token 只能直接看见 5、6、8、9，以及全局 token 0
- 第 15 个 token 无法直接看见第 3 个 token
- 但第 15 个 token 可以把信息发给 0，0 再与第 3 个 token 通信

可以把这个信息流写成一条路径：

$$
15 \rightarrow 0 \rightarrow 3
$$

如果没有全局 token，这条路径就可能变成：

$$
15 \rightarrow 13 \rightarrow 11 \rightarrow 9 \rightarrow 7 \rightarrow 5 \rightarrow 3
$$

路径更长，意味着模型需要更多层才能把远距离信息稳定传过去。

这意味着 Longformer 的全局信息流依赖“窗口传播 + 全局锚点”共同完成，而不是依赖一张稠密矩阵一次性解决。

Longformer 还引入过空洞滑窗。空洞滑窗的意思是：窗口内不是看连续邻居，而是按步长跳着看。若 dilation 为 $d$，单层第 $i$ 个 token 关注的位置可以粗略写成：

$$
\{\, i-kd,\ i-(k-1)d,\ \dots,\ i,\ \dots,\ i+(k-1)d,\ i+kd \,\}
$$

其中 $k$ 控制窗口中保留多少个点。粗略理解是，连续滑窗更擅长抓局部细节，空洞滑窗更擅长在不增加连接数的情况下拉大覆盖范围。论文预训练阶段用过 dilation，但很多现成下游实现更常见的是普通滑窗。

把三种模式放在一起更容易理解：

| 模式 | 单个 token 主要连接谁 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 全注意力 | 所有 token | 表达能力最强 | $O(N^2)$，长序列成本高 |
| 连续滑窗 | 附近连续邻居 | 局部建模稳定，实现直接 | 远距离路径长 |
| 滑窗 + 全局 | 邻居 + 少数锚点 | 在成本和跨段建模之间折中 | 依赖全局 token 设计 |

---

## 代码实现

下面先用一个可运行的 Python 玩具实现，计算不同序列长度下的稀疏注意力密度。这里的“密度”就是：实际保留的注意力连接数，占全注意力 $N^2$ 的比例。

```python
from __future__ import annotations


def compute_attention_density(seq_len: int, window: int, num_global: int) -> float:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window < 0:
        raise ValueError("window must be non-negative")
    if num_global < 0 or num_global > seq_len:
        raise ValueError("num_global must be in [0, seq_len]")

    # 近似公式：局部窗口连接 + 全局连接
    # 这里不对重复边做去重，目的是复现复杂度层面的粗略估算。
    sparse_edges = seq_len * (window + num_global)
    dense_edges = seq_len * seq_len
    density = sparse_edges / dense_edges
    return min(density, 1.0)


def format_percent(x: float) -> str:
    return f"{x * 100:.3f}%"


if __name__ == "__main__":
    assert round(compute_attention_density(4096, 512, 2), 4) == 0.1255
    assert round(compute_attention_density(16384, 512, 2), 4) == 0.0314
    assert compute_attention_density(512, 512, 1) == 1.0

    for n in [512, 4096, 16384]:
        density = compute_attention_density(n, 512, 2)
        print(f"seq_len={n:5d}, density={format_percent(density)}")
```

这段脚本的输出应接近：

```text
seq_len=  512, density=100.000%
seq_len= 4096, density=12.549%
seq_len=16384, density=3.137%
```

按这个公式，密度变化如下：

| 序列长度 N | 窗口 w | 全局 token 数 Ng | 稀疏密度 |
| --- | --- | --- | --- |
| 512 | 512 | 2 | 100.0% |
| 4096 | 512 | 2 | 12.55% |
| 16384 | 512 | 2 | 3.14% |

这张表说明一个事实：窗口固定时，序列越长，Longformer 相对全注意力越省。因为全注意力是平方增长，滑窗是线性增长。

如果你想更进一步，可以直接构造一个注意力掩码，看看“哪些边被保留、哪些边被删掉”。下面这个函数返回一个布尔矩阵，`True` 表示该位置允许注意力连接：

```python
from __future__ import annotations

import numpy as np


def build_longformer_mask(
    seq_len: int,
    window: int,
    global_indices: list[int],
) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if window < 0:
        raise ValueError("window must be non-negative")

    mask = np.zeros((seq_len, seq_len), dtype=bool)
    half = window // 2
    global_set = set(global_indices)

    for idx in global_set:
        if idx < 0 or idx >= seq_len:
            raise ValueError(f"global index out of range: {idx}")

    for i in range(seq_len):
        if i in global_set:
            mask[i, :] = True
            continue

        left = max(0, i - half)
        right = min(seq_len, i + half + 1)
        mask[i, left:right] = True
        for g in global_set:
            mask[i, g] = True

    for g in global_set:
        mask[:, g] = True

    return mask


if __name__ == "__main__":
    mask = build_longformer_mask(seq_len=16, window=4, global_indices=[0])
    print(mask.astype(int))
```

这个实现对应的规则是：

- 普通 token 能看局部窗口。
- 普通 token 额外能看所有全局 token。
- 全局 token 自己能看全文。
- 全文所有 token 也都能看全局 token。

再看 Hugging Face 中最关键的工程接口。真正决定“谁是全局 token”的，不是模型配置名，而是 `global_attention_mask`。

```python
import torch
from transformers import LongformerModel, LongformerTokenizer


def demo_longformer_forward() -> None:
    model_name = "allenai/longformer-base-4096"
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)

    text = (
        "Question: What is the penalty ratio? "
        "Document: Section 1. Payment terms. "
        "Section 2. Breach liability states the penalty ratio is ten percent."
    )

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1      # 通常给首 token 全局注意力
    global_attention_mask[:, 1:6] = 1    # 例如把问题区间也设为全局

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
    )

    last_hidden_state = outputs.last_hidden_state
    print(last_hidden_state.shape)


if __name__ == "__main__":
    demo_longformer_forward()
```

如果环境里还没有安装依赖，最少需要：

```bash
pip install torch transformers sentencepiece
```

如果把实现思路抽成伪代码，大致是这样：

```python
for i in range(seq_len):
    if token_i_is_global:
        attend_indices = all_positions
    else:
        local_positions = positions_in_sliding_window(i, window)
        attend_indices = union(local_positions, global_positions)

    scores = q[i] @ k[attend_indices].T / sqrt(d_k)
    scores += attention_mask_for_invalid_positions
    probs = softmax(scores)
    out[i] = probs @ v[attend_indices]
```

这段逻辑体现了 Longformer 的本质：普通 token 的候选集合很小，全局 token 的候选集合很大。两者共享同一层表示空间，但连接模式不同。

真实工程例子通常不是“把第一个 token 设全局”这么简单。比如文档验证、长合同审查、长报告问答里，更合理的做法是把这些位置设为全局 token：

- 文档标题
- 章节标题
- 问题 tokens
- 特殊分隔符
- 结构化字段起始位置
- 表格标题或字段名

原因是这些位置天然承担“跨段索引”的角色，适合作为信息汇聚点。

新手在这里最容易犯的错误，是把“重要句子”整段都设为全局。这样会让 $N_g$ 增长过快，复杂度中的 $N_gN$ 项抬起来，Longformer 的优势就开始消失。经验上更合理的做法是只选少量锚点，而不是大面积开放全局连接。

---

## 工程权衡与常见坑

Longformer 的优点明确，但工程上有几个坑非常常见。

第一，全局 mask 选错位置。  
如果全局 token 太少，跨段信息流会断；如果全局 token 太多，复杂度中的 $N_gN$ 会迅速增大，线性优势被吃掉。实践里通常只给少数结构锚点开全局，而不是给整段都开。

第二，窗口大小不是越大越好。  
窗口大，局部语境更完整，但显存和吞吐会下降；窗口小，速度和显存更好，但跨句、跨段局部依赖可能被截断。`w=512` 常见，是因为它在精度和成本之间比较平衡，不是理论上的唯一最优。

第三，小窗口不一定线性提升速度。  
这是新手常踩的坑。理论复杂度降低，不等于 wall-clock time 按比例下降。原因在于具体实现依赖底层 kernel、padding、张量布局、`sliding_chunks` 等细节。你把窗口从 512 改到 128，训练时间可能几乎不变，但显存会变。工程上必须实际 benchmark，而不是只看公式。

第四，边界 token 处理容易想错。  
在序列两端，窗口天然不完整。实现时必须正确裁剪索引，否则会越界，或者错误地把 padding 当真实 token 参与注意力。

第五，padding 和全局 mask 叠加时容易出错。  
很多新手会先把一批样本 pad 到同一长度，再按固定位置打 `global_attention_mask`。如果某些位置实际已经是 padding，那么你等于在空白 token 上开了全局注意力，结果既浪费算力，也可能引入异常行为。正确做法是让全局 mask 和真实有效 token 对齐。

第六，任务目标和全局设计必须一致。  
比如分类任务里，`[CLS]` 或首 token 常适合设全局；抽取式问答里，问题 token 更关键；文档结构理解任务里，章节标题和段分隔符更关键。全局 token 选取不是固定模板，而是任务建模的一部分。

下面给一个真实工程对比表，用来理解它为什么常被用于 4k 文档任务：

| 模型 | 4K 文档 VRAM | 吞吐量 | 适合场景 |
| --- | --- | --- | --- |
| Longformer | 约 11.9GB | 约 28 docs/min | 长文分类、QA、编码器任务 |
| LED | 约 18.3GB | 约 19 docs/min | 长文本摘要、seq2seq |
| 全注意力 Transformer | 更高 | 更低 | 短文本或高算力场景 |

这个量级的意义很实际：很多单卡环境可以承受 12GB 左右，但扛不住更高的全注意力开销。于是 Longformer 成为“能上线”的方案，而不只是“理论上更优”的方案。

为了避免把这张表理解错，这里再补三个说明：

| 现象 | 正确理解 |
| --- | --- |
| 显存更低 | 说明稀疏注意力确实减少了连接与中间激活，但不代表所有实现都一样省 |
| 吞吐更高 | 取决于硬件、batch size、序列长度、库版本，不能直接照搬 |
| 4K 能跑 | 不代表 16K 也能无痛扩展，位置编码、训练配置、吞吐都会重新成为瓶颈 |

还有一个常见误区：以为 Longformer 适合所有长序列任务。其实不是。它尤其适合“局部结构强，且可定义少量全局锚点”的任务。如果文本没有明显结构，或者远距离交互非常随机、非常频繁，那么固定滑窗模式未必是最合适的。

可以把工程选型压缩成一句判断题：

- 问题主要靠局部上下文解决吗？
- 是否存在少量天然的全局锚点？
- 你是否真的需要 2k 以上上下文？

这三个问题里，如果前两个答案是“是”，第三个答案也是“是”，Longformer 往往值得试。

---

## 替代方案与适用边界

Longformer 不是唯一的长序列注意力方案。最常被拿来比较的是 BigBird 和 LED。

| 方案 | Local | Global | Random | 典型用途 | 边界 |
| --- | --- | --- | --- | --- | --- |
| Longformer | ✔ | ✔ | ✘ | 长文分类、抽取式 QA | 固定模式，依赖锚点设计 |
| BigBird | ✔ | ✔ | ✔ | 超长序列、理论保证更强 | 实现更复杂，模式更混合 |
| LED | ✔ | ✔ | ✘ | 长文本摘要、生成任务 | 主要用于 encoder-decoder |

BigBird 比 Longformer 多了一类 random attention。random attention 的意思是：除了局部邻居和全局 token，再给每个 token 随机连一些远处位置。白话说法是，它人为加了几条“远程捷径”。这样做的好处是图直径更小，远距离信息传播路径更短，对极长序列更有利。

LED 可以看作“把 Longformer 放到 encoder 端，再配一个标准 decoder”。因此如果任务是摘要、长文本生成、报告重写，LED 往往比纯 Longformer 更自然，因为 decoder 需要生成式建模。

再把三者差异说得更直白一些：

| 你最关心的问题 | 更合适的方案 | 原因 |
| --- | --- | --- |
| 文档很长，但输出是分类标签或抽取结果 | Longformer | 编码器结构直接，局部 + 全局足够 |
| 文档很长，输出是一段新文本 | LED | 需要 decoder 逐步生成 |
| 文档极长，而且担心固定滑窗的信息覆盖不够 | BigBird | 加入随机远程边，图连通性更强 |

怎么选，原则很直接：

- 如果任务是文档分类、抽取式问答、长文编码，优先 Longformer。
- 如果任务是长文本生成、摘要，优先 LED。
- 如果序列极长，且希望稀疏连接更有理论覆盖能力，可以考虑 BigBird。
- 如果序列其实不长，或者任务高度依赖任意位置间的自由交互，标准全注意力可能更简单。

所以 Longformer 的适用边界可以概括成一句话：它最适合“长文本、局部相关性强、少量关键位置需要全局汇总”的任务；一旦任务的核心依赖变成“大量随机的远距离交互”，它就不一定是最佳选择。

最后把选型边界总结成一个表：

| 场景 | 是否推荐 Longformer | 原因 |
| --- | --- | --- |
| 4K 合同分类 | 推荐 | 局部条款理解 + 标题/问题全局汇总 |
| 长文抽取式 QA | 推荐 | 问题 token 可作为全局锚点 |
| 长文本摘要生成 | 一般不优先 | 更适合 LED 这类 encoder-decoder |
| 任意位置频繁两两交互的任务 | 不优先 | 固定滑窗会删掉太多关键远程边 |
| 512 token 内的普通文本任务 | 通常没必要 | 全注意力更直接，实现更简单 |

---

## 参考资料

- Iz Beltagy, Matthew E. Peters, Arman Cohan. *Longformer: The Long-Document Transformer*. arXiv:2004.05150, 2020.
- Hugging Face Transformers 文档：Longformer 模型说明、`attention_window` 与 `global_attention_mask` 的接口定义。
- Hugging Face Transformers 源码：`LongformerModel` 与 Longformer self-attention 的实现细节。
- Zaheer et al. *Big Bird: Transformers for Longer Sequences*. NeurIPS, 2020.
- Longformer 论文附录中关于 sliding window attention、dilated sliding window attention 的说明。
- 若需要工程对比，可进一步查阅 LED（Longformer Encoder-Decoder）论文与 Hugging Face `LEDModel` 文档，对比编码任务和生成任务的差异。
