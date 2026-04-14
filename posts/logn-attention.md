## 核心结论

LogN-Attention 的核心做法很简单：在**推理阶段**，把注意力 logits 乘上一个随上下文长度增长的系数
$$
\beta_N=s\log N,
$$
再做 softmax：
$$
A_{ij}=\frac{\exp(\beta_N a_{ij})}{\sum_{k=1}^n\exp(\beta_N a_{ik})}.
$$

这里的 **logits** 就是 softmax 之前的原始分数，白话说就是“模型还没归一化前认为谁更重要”。当序列从 4k 拉长到 32k、64k 时，不缩放的注意力分数更容易“摊平”，高分 token 的优势被长序列稀释；LogN-Attention 用 $\log N$ 级别的温度放大，把这种摊平拉回来。

结论有两个。

第一，它是**零训练改动**的长度外推技巧。也就是模型参数不变，只改推理时的 attention 计算，就能让长上下文下的注意力分布保持更稀疏、更非均匀。

第二，它不是“越大越好”。理论上可把它看成
$$
\beta_N=\gamma\log N
$$
的临界缩放问题：$\gamma$ 太小，attention 仍然趋于均匀；$\gamma$ 太大，attention 又会过尖，接近只看自己，信息交互反而消失。LogN 的价值就在于它落在一个合理的中间地带。

一个新手版直觉是：原来高分 token 只领先一点点，到了很长序列里几乎看不出来；乘上 $0.4\log N$ 后，这点领先会被持续放大，所以长序列里它仍然能被选中。

---

## 问题定义与边界

**注意力熵**是衡量分布有多分散的量，白话说就是“注意力到底是集中看少数 token，还是平均看很多 token”。长上下文推理的一个典型问题是：训练时上下文长度固定，比如 4k；推理时拉到 32k 或 64k 后，softmax 温度没有随长度变化，导致分布越来越平，出现 **rank collapse**。这个词的白话解释是：token 之间原本有区分度，后来越来越像，模型难以稳定保留关键信息。

LogN-Attention 解决的边界非常明确。

1. 它主要处理的是**推理时上下文变长导致的注意力扩散**。
2. 它默认训练阶段仍然使用普通 attention，不要求重新训练。
3. 它不保证模型自动学会新的长程能力。若模型训练时根本没学到跨几万 token 的依赖，只靠缩放也不能凭空创造能力。

下面用示意表看趋势。数值不是统一基准测试值，而是帮助理解“未缩放会扩散，LogN 会压回去”的方向。

| 上下文长度 $N$ | 未缩放 attention 熵 | LogN 缩放后 attention 熵 | 直观含义 |
|---|---:|---:|---|
| 4k | 中等 | 中等偏低 | 短上下文差距不大 |
| 16k | 明显升高 | 轻微升高 | 未缩放开始变散 |
| 64k | 很高 | 仍可控 | LogN 维持稀疏性 |

新手版理解可以直接记成一句话：4k 训练时注意力还“够劲”，到 32k 推理时像音量被拉小了；LogN 缩放就是按长度把音量再拧上去。

---

## 核心机制与推导

softmax 只关心相对分差。若 logits 是 $1$ 和 $0$，未缩放时：
$$
\text{softmax}(1,0)\approx(0.731,0.269).
$$
如果采用 $s=0.4$，且 $N=4096$，那么
$$
\beta_N=0.4\log 4096\approx 3.33.
$$
此时 logits 从 $(1,0)$ 变成 $(3.33,0)$，于是
$$
\text{softmax}(3.33,0)\approx(0.965,0.035).
$$
这说明“领先 1 分”的优势被明显放大了。

如果为了教学直观，取一个更温和的玩具例子，例如原始分差只有 $0.3$，则未缩放约为 $57\%$ 对 $43\%$；乘上 $\beta_N=3.33$ 后，相当于分差变成 $1.0$ 左右，注意力会接近 $73\%$ 对 $27\%$。这更符合“从略有偏好变成明显偏好”的直觉。

从理论角度看，Critical Attention Scaling 把问题写成 $\beta_N=\gamma\log N$。原因是当 $N$ 增长时，候选 token 数量也在增长。如果温度不随 $N$ 增长，高分项在归一化分母中的优势会被越来越多的普通项冲淡；而乘上 $\log N$ 后，高分项的指数优势会和候选数量增长处在同一量级上。简化理解就是：

- $\gamma$ 太小：分母里“人太多”，高分项压不住，分布趋于均匀。
- $\gamma$ 合适：高分项仍突出，但不会尖到只剩自己。
- $\gamma$ 太大：对角项或极少数 token 权重过大，信息流接近断掉。

**真实工程例子**是长上下文检索。Scale-invariant Attention 的实验里，模型在较短上下文训练后，直接评估到 16k、64k 时，未缩放方法容易失效，而 `LogN+p-RoPE` 在 needle-in-a-haystack 检索和验证损失上都显著强于普通 RoPE/NTK/YaRN 组合，说明简单的 logN 缩放确实能作为强基线使用。

---

## 代码实现

实现位置通常就在 `QK^T / sqrt(d)` 之后、softmax 之前。也就是先算原始 attention logits，再乘以 $\beta_N$。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def logn_attention_probs(logits, context_length, s=0.4):
    assert context_length > 1
    beta = s * math.log(context_length)
    return softmax([x * beta for x in logits])

def plain_attention_probs(logits):
    return softmax(logits)

plain = plain_attention_probs([1.0, 0.0])
scaled = logn_attention_probs([1.0, 0.0], context_length=4096, s=0.4)

assert plain[0] > 0.73 and plain[0] < 0.74
assert scaled[0] > 0.96 and scaled[0] < 0.97
assert scaled[0] > plain[0]
```

如果写成 PyTorch 伪代码，核心只多一行：

```python
beta = s * math.log(context_length)
attn = torch.softmax(logits * beta, dim=-1)
```

工程里常见做法不是所有 head 一刀切，而是按 head 控制：

| Head 类型 | 序列长度 | 是否启用 LogN | 目的 |
|---|---|---|---|
| 全局检索 head | 16k+ | 是 | 保持远距 token 可分辨 |
| 局部模式 head | 任意 | 否或弱启用 | 保留近邻敏感性 |
| 混合 head | 32k+ | 部分启用 | 在稀疏与局部之间折中 |

---

## 工程权衡与常见坑

LogN-Attention 的优点是简单，但代价也很直接。它会把全局熵压低，可与此同时，**最近 token 的总注意力可能下降**。白话说，模型为了不在长序列里“平均看所有内容”，可能会更偏向少数高分远端 token，导致对刚出现的新信息不够敏感。

常见问题和规避方式如下：

| 问题 | 现象 | 常见规避方法 |
|---|---|---|
| local attention collapse | 最近 100 个 token 权重掉太快 | 减小 $s$；只对部分 head 开启 |
| 过度尖峰 | 注意力接近 one-hot，信息交互变差 | 给 $\beta_N$ 设上限；分层启用 |
| 与位置编码冲突 | 远距优势过强或过弱 | 联合 RoPE/p-RoPE/位置 bias 调参 |
| 训练期直接照搬 | 梯度不稳、损失恶化 | 先只用于推理；训练时单独验证 |

这里要特别区分训练与推理。推理阶段只改前向计算，通常比较安全；训练阶段若也使用很强的 logN 缩放，可能引入梯度爆炸或梯度消失。因为 softmax 太尖时，少数位置主导梯度，优化会更脆弱。

---

## 替代方案与适用边界

如果目标是“不重训，立刻把上下文拉长”，LogN-Attention 的性价比很高。但它不是唯一方案。

| 方法 | 是否需要训练改动 | 推理灵活度 | 主要代价 |
|---|---|---|---|
| LogN-Attention | 否 | 高 | 可能损失局部上下文 |
| Learned temperature | 通常需要 | 中 | 需要额外训练或校准 |
| Sparse attention | 通常需要 | 中 | 改模型结构，复杂度高 |
| Scale-invariant attention | 通常需要方法级改造 | 中 | 实现更复杂，但局部保持更好 |

适用边界可以概括成两句。

第一，若模型已经具备一定长程检索能力，只是因为上下文变长导致注意力扩散，那么 LogN 往往有效。

第二，若模型根本没有学到长距离依赖，只会在近邻范围内工作，那么只靠调温度不够，还需要训练策略、位置编码改造，甚至显式稀疏结构。

所以新手可以把它理解成：LogN 解决的是“模型会看，但看不清”的问题；它不解决“模型根本不会看”的问题。

---

## 参考资料

- Chen, Lin, Polyanskiy, Rigollet, *Critical Attention Scaling in Long-Context Transformers*，2025。重点是给出 $\beta_n \asymp \log n$ 的理论临界视角，解释为什么太小会均匀化、太大会接近恒等映射。链接：https://people.lids.mit.edu/yp/homepage/data/2025_attn_scaling.pdf
- Anson et al., *Scale-invariant Attention*，NeurIPS 2025 poster。重点是实验上比较 `No scale`、`LogN` 和 scale-invariant 方法，并指出 LogN 能显著降低熵，但会牺牲一部分对最近局部 token 的关注。OpenReview PDF 的 Figure 1 很关键。链接：https://openreview.net/pdf/335c20d88f091f664d66631abdf286d50f5fe3c2.pdf
- nor, *A short note on some aspects of long context attention*，2025。重点是从长上下文 softmax 缩放与熵控制角度总结这类方法，适合快速建立整体直觉。链接：https://nor-blog.pages.dev/posts/2025-11-27-attention-and-long-context/
