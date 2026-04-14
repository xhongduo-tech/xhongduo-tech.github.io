## 核心结论

Logit Lens 的核心作用，是把 Transformer 每一层的中间表示提前翻译回“词表概率”，从而观察模型在这一层已经倾向输出什么。这里的“中间表示”通常指 residual stream，也就是每层不断累加信息的主干向量；更直白地说，它像模型内部不断修订的“当前草稿”。

它有两个直接价值。

第一，它把“最终为什么输出这个词”拆成逐层轨迹。你不再只看到最后答案，而是能看到答案如何从模糊候选逐步收敛出来。很多任务里，早期层先给出高频、通用、语法上合理的词，后续层再把概率质量集中到具体实体、数字或关系上。

第二，它适合做调试可视化。比如模型最后答错了，你可以看错误词是在早层就被推高，还是后面几层才被放大。前者更像上下文编码、检索注入或知识激活阶段已经偏了；后者更像候选重排、格式约束或对齐偏置在后期把错误答案保留了下来。

但它不是严格的因果解释工具。Logit Lens 只能说明“这一层的表示在最终输出头看来像什么”，不能证明“这一层导致了最终输出”。这点必须分清。它更像监控仪表，不是因果定责工具。

一个最直观的玩具例子，是把句子写成：

`The Eiffel Tower is located in ___`

如果对最后一个位置做逐层观察，早期层可能偏向 `the`、`and`、`a` 这类高频词；中间层开始出现 `France`、`Paris`；最终层才把 `Paris` 的概率明显抬高。这个过程说明，模型不是从第一层就“想好了答案”，而是在逐层累积约束、修正候选，并最终收敛。

---

## 问题定义与边界

我们先把问题说精确。

给定一个 Transformer 模型、一个输入序列、以及某个待预测位置的中间状态 $h_\ell$，Logit Lens 要做的是：不等模型跑完整个网络，而是拿第 $\ell$ 层的 residual stream 直接接到最终输出头上，看看如果“现在就停止”，模型会倾向输出哪些 token。

对 decoder-only 语言模型，一个常见写法是：

$$
\mathrm{logits}_\ell = W_U \cdot \mathrm{LN}_f(h_\ell)
$$

再经过 softmax 得到概率分布：

$$
p_\ell(t)=\frac{e^{\mathrm{logits}_\ell(t)}}{\sum_j e^{\mathrm{logits}_\ell(j)}}
$$

这里：

- $h_\ell$ 是第 $\ell$ 层目标位置的 residual stream
- $\mathrm{LN}_f$ 是模型在最终输出前使用的那层归一化
- $W_U$ 是 unembedding 矩阵，也就是“把隐藏向量翻译成词表分数”的线性映射
- $\mathrm{logits}_\ell(t)$ 是 token $t$ 的未归一化分数
- $p_\ell(t)$ 是第 $\ell$ 层视角下 token $t$ 的预测概率

几个术语要给新手说清楚。

`residual stream` 不是某个神秘模块，它就是层与层之间反复传递、不断叠加更新的主向量通道。注意力层和 MLP 层做出的修改，最后都会加回这条主干里。  
`unembedding` 可以理解成“把隐藏状态翻译回词表”的读出矩阵。模型最终之所以能输出具体单词，不是因为隐藏状态天然就是单词，而是因为最后有一步把向量投影回整个词表。  
`softmax` 的作用是把任意实数分数变成概率分布，而且更大的分数会被指数放大，因此头部候选会更容易拉开差距。

问题边界也要说清楚。

| 边界 | 能做什么 | 不能做什么 |
| --- | --- | --- |
| 输出观察 | 看每层“已经像哪些词” | 不能证明哪层因果地产生该词 |
| 轨迹分析 | 看候选词如何被抬高或压低 | 不能单独解释某个注意力头或某个 MLP 神经元的职责 |
| 调试辅助 | 判断错误倾向出现得早还是晚 | 不能代替严格干预实验 |
| 模型比较 | 粗看不同层的收敛趋势 | 不能默认不同模型、不同层的数值完全可比 |
| 教学展示 | 直观看到“从模糊到清晰”的过程 | 不能把中间层输出当成模型真实训练目标 |

可以把它理解成“逐层拍快照”。你每一层都用最终那套输出格式拍一张照，于是能看到图像逐渐变清楚。但这不表示你知道是谁先动了笔，也不表示这张中途快照与最后成品严格同分布。

---

## 核心机制与推导

为什么这种方法有意义？关键在于 Transformer 的最后一步，本质上就是把最终隐藏状态投影到词表空间。Logit Lens 只是把这一步提前复用到中间层。

如果模型最终层输出是：

$$
\mathrm{logits}_{\text{final}} = W_U \cdot \mathrm{LN}_f(h_L)
$$

那么对任意中间层 $\ell$，我们构造：

$$
\mathrm{logits}_\ell = W_U \cdot \mathrm{LN}_f(h_\ell)
$$

这相当于提出一个受控假设：

“如果模型在第 $\ell$ 层就被迫停止，并直接复用最终输出头，它会更像输出什么？”

这个假设不是模型真实训练目标，但它有很强的观察价值，因为所有层都被映射到同一个词表空间，结果因此具有可读性。你观察的不是“这一层的原始向量长什么样”，而是“这一层的向量如果按最终读出规则解释，会像什么答案”。

softmax 的作用，是把原始分数变成概率。设候选词只有三个：`the`、`France`、`Paris`，logit 分别为 $[1.0, 2.0, 3.5]$。则：

$$
\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

代入后：

$$
p(\text{Paris})=\frac{e^{3.5}}{e^{1.0}+e^{2.0}+e^{3.5}}
\approx
\frac{33.12}{2.72+7.39+33.12}
\approx 0.767
$$

也就是说，虽然 `Paris` 只比 `France` 高 $1.5$ 个 logit，但转成概率后已经接近 $76.7\%$。这解释了一个常见现象：逐层看时，某个候选只要开始持续领先，后续层的概率就会显得收敛得很快。

再看同一个句子：

`The Eiffel Tower is located in ___`

假设某些层的 top token 如下：

| 层号 | 主候选词 | logit | softmax 概率 |
| --- | --- | ---: | ---: |
| 0 | `the` | 1.2 | 3% |
| 2 | `and` | 1.5 | 4% |
| 4 | `France` | 2.6 | 9% |
| 6 | `Paris` | 3.4 | 12% |
| 9 | `Paris` | 6.1 | 41% |
| 最终层 | `Paris` | 12.7 | 93% |

这张表至少说明三件事。

第一，早期层通常先编码通用语言模式，而不是精确答案。`the`、`and` 这类词高频、语法兼容性强，所以容易在早层占优势。这里不能简单理解成“模型什么都不知道”，更准确的说法是：模型此时只知道一部分约束，例如“这里大概率要接一个常见词”或“这里在句法上像一个地点短语位置”。

第二，正确答案不一定一出现就压倒一切。`Paris` 在中间层可能只是第一次进入前列，说明相关知识已经开始可读，但还没完全胜出。对工程调试而言，这种“已进入 top-k，但尚未成为 top-1”的阶段很重要，因为它表示模型并非完全缺失答案，而是在候选竞争里还没稳定压过其他词。

第三，后续层不仅在“增加正确词概率”，也在“重排竞争对手”。例如 `France` 和 `Paris` 都是语义相关候选，但最终层会结合句法、搭配和实体粒度，把“地点填空里更自然的 token”收束到更具体的答案。模型内部并不是单调累加一个答案，而是在反复调整候选排序。

换一个更适合新手的理解方式。  
假设你在做一道填空题：

`The capital of Japan is ___`

你可能先排除动词、冠词这类明显不合适的词；再想到“应该是一个地名”；再缩小到“Tokyo”；最后确认拼写和格式。Logit Lens 想观察的，就是模型在这个“逐步排除和收敛”的过程中，每一层大致走到了哪一步。

从机制上看，residual stream 会逐层叠加多种信息：

- 前文语法结构
- 当前位置的预测任务
- 已激活的实体或事实知识
- 句式偏好与搭配偏好
- 上下文对候选的排除信息

Logit Lens 做的，不是把这些成分拆开，而是把它们混合后的阶段性结果投影出来。因此它非常适合回答“模型到这一层时大概已经知道了什么”，但不适合直接回答“到底是哪一个模块负责把答案写进去的”。

---

## 代码实现

最小实现并不复杂。你只需要拿到某一层的 residual stream，过一次最终 LayerNorm，再乘 unembedding 矩阵，最后做 softmax。

下面先给一个可运行的玩具版本，不依赖深度学习框架，只演示数学过程。它不仅能运行，还会打印 top token 与概率，便于你直接感受 Logit Lens 的计算链路。

```python
import math

def layer_norm(x, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [(v - mean) / math.sqrt(var + eps) for v in x]

def matvec(mat, vec):
    return [sum(mij * vj for mij, vj in zip(row, vec)) for row in mat]

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk(vocab, probs, k=3):
    pairs = list(zip(vocab, probs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]

# 三个词的词表：["the", "France", "Paris"]
vocab = ["the", "France", "Paris"]

# unembedding 矩阵：shape = [vocab_size, hidden_dim]
W_U = [
    [0.2, -0.1, 0.3],   # the
    [0.4,  0.7, -0.2],  # France
    [0.8,  1.1,  0.1],  # Paris
]

# 假设这是第 6 层某个位置的 residual stream
h6 = [0.5, 1.2, -0.4]

normed = layer_norm(h6)
logits = matvec(W_U, normed)
probs = softmax(logits)

result = dict(zip(vocab, probs))

assert abs(sum(probs) - 1.0) < 1e-9
assert max(result, key=result.get) == "Paris"

print("Layer-6 hidden:", h6)
print("Layer-6 normed:", [round(x, 4) for x in normed])
print("Layer-6 logits:", [round(x, 4) for x in logits])
print("Layer-6 probs:", {k: round(v, 4) for k, v in result.items()})
print("Top-k:", [(tok, round(p, 4)) for tok, p in topk(vocab, probs, k=3)])
```

这段代码体现的就是：

`residual -> final LayerNorm -> unembedding -> softmax`

如果你希望看“逐层轨迹”，可以把多个层的 hidden state 放进列表里，循环计算每层 top-k：

```python
layers = {
    0: [0.1, 0.0, 0.2],
    2: [0.0, 0.3, 0.1],
    4: [0.3, 0.8, -0.2],
    6: [0.5, 1.2, -0.4],
}

for layer_idx, h in layers.items():
    probs = softmax(matvec(W_U, layer_norm(h)))
    winners = topk(vocab, probs, k=2)
    print(f"Layer {layer_idx}: {[(tok, round(p, 4)) for tok, p in winners]}")
```

如果换成真实模型，思路不变，只是 $h_\ell$ 来自模型某层缓存。一个更接近工程实践的 PyTorch 版本如下。它假设你已经能拿到各层 hidden states，且模型有最终 LayerNorm 与输出头。

```python
import torch

@torch.no_grad()
def logit_lens_for_position(hidden_states, ln_f, unembed, token_pos=-1, topk=5):
    """
    hidden_states: list[Tensor], 每个元素 shape = [batch, seq, d_model]
    ln_f: final LayerNorm module
    unembed: Linear layer 或等价权重映射到 vocab
    token_pos: 观察哪个位置的下一 token 预测
    """
    trajectories = []

    for layer_idx, h in enumerate(hidden_states):
        h_pos = h[:, token_pos, :]              # [batch, d_model]
        normed = ln_f(h_pos)                    # [batch, d_model]
        logits = unembed(normed)                # [batch, vocab_size]
        probs = torch.softmax(logits, dim=-1)   # [batch, vocab_size]

        values, indices = torch.topk(probs, k=topk, dim=-1)
        trajectories.append({
            "layer": layer_idx,
            "top_probs": values[0].tolist(),
            "top_token_ids": indices[0].tolist(),
        })

    return trajectories
```

真实工程里，你通常会做四件事。

第一，固定一个 token 位置。  
Logit Lens 观察的是“某个位置的下一 token 分布”，如果不同位置混着看，图会很乱。最常见的是看最后一个位置，因为那通常对应“当前回答要输出的下一个 token”。

第二，对所有层重复计算 top-k。  
常见输出不是单一 top-1，而是每层前 5 或前 10 个候选。因为 top-1 很容易跳来跳去，而 top-k 更能反映候选集合何时开始稳定。

第三，记录 rank、probability 和 logit 三种量。  
- `rank` 看“正确答案什么时候进入前列”
- `probability` 看“概率质量什么时候开始集中”
- `logit` 看“候选之间的原始分差”

第四，把中间层结果与最终输出对照。  
例如观察：
- 正确答案何时进入 top-k
- 错误答案在哪层达到峰值
- 哪一层发生最大幅度的候选重排
- 某个 prompt 修改后，收敛层数是否提前或延后

下面给一个更具体的调试例子。  
假设用户问：“某公司 2024 年 CEO 是谁？”模型最终答错。你对答案位置做 Logit Lens 后发现：

| 层区间 | 现象 | 解释方向 |
| --- | --- | --- |
| 1 到 4 层 | 高频人名、公司名混杂 | 还在建立任务语境 |
| 5 到 9 层 | 错误姓名进入 top-3 并稳定存在 | 早期候选池已有偏差 |
| 10 到 14 层 | 正确姓名短暂升到 top-1 | 模型并非完全不知道答案 |
| 最终几层 | 错误姓名重新回到 top-1 | 后期重排或偏置把错误答案保留下来 |

这种轨迹比“模型最后答错了”更有信息量。它告诉你：问题不一定是完全缺失知识，也可能是后续层没有把正确候选稳定压住。这会直接影响你下一步该查什么，是检索文档质量、提示词约束、还是后层对回答风格的偏好。

---

## 工程权衡与常见坑

Logit Lens 很直观，但工程上有几个坑不能忽略。

第一，早期层和最终 unembedding 往往并不对齐。所谓“不对齐”，就是早期层的表示还处在一种内部语义空间里，直接拿最终输出头去翻译，会产生系统误差。白话说，像拿专门给终点站校准的翻译器，去翻译半路上的草稿语言，结果自然会偏。

第二，它不是因果工具。如果第 6 层已经很像错误词，不代表“错误就是第 6 层引入的”。更合理的表述是：到第 6 层为止，这个错误词已经能被最终输出头读出来了。它可能是更早累积的结果，也可能只是到这一层才变得可见。

第三，不同模型、不同归一化方式下的失真程度不同。对于某些模型，中后层轨迹很直观；对于另一些模型，早层会出现大量看似无意义或过于频繁的词。如果不先了解模型架构细节，就直接做跨层数值比较，很容易误判。

第四，只看 top-1 很容易误导。很多时候真正有价值的是“候选集合如何变化”，不是冠军词是谁。一个正确答案如果从第 3 层开始稳定处于 top-5，往往比“第 9 层突然冲到 top-1”更说明模型早已具备相关知识。

第五，位置混淆会让结论失真。你必须先说清楚在看哪个 token 位置。分析“生成第一个答案 token”的轨迹，与分析“长答案中某个中间 token”的轨迹，含义并不一样。

第六，tokenization 会影响解释。比如 `Paris` 可能是一个 token，也可能被拆成多个子词；中文词语更可能被拆成多个 token。如果你忽略这一点，就会把“词没有出现”误读成“模型没想到这个词”。

常见问题和缓解方式可以整理成表：

| 问题 | 现象 | 风险 | 缓解措施 |
| --- | --- | --- | --- |
| 低层失真 | 早期层出现奇怪高频词或无关子词 | 误以为模型完全没学到知识 | 重点看中后层趋势，必要时用 Tuned Lens |
| 系统偏差 | 跨层 logit 绝对值差异很大 | 误判“哪层更强” | 多看排名变化、相对差值和最终对照 |
| 非因果性 | 某层首次出现错误词 | 错把相关性当因果 | 结合干预实验、ablation 或 activation patching |
| top-1 误导 | 冠军词频繁跳变 | 忽略稳定候选集合 | 固定观察 top-k、rank 和累计概率 |
| 位置混淆 | 不同 token 位置混在一起 | 结论不稳定 | 每次只分析一个目标位置 |
| 分词影响 | 目标词被拆成多个 token | 误判答案是否浮现 | 同时跟踪完整词与组成子词 |
| Prompt 敏感 | 微小提示改动导致轨迹变化 | 错把单次现象当规律 | 对多个提示做对照实验 |

再看一个面向初学者很常见的误判。

有人在调试幻觉时发现：Layer 6 已经把错误词推上去，最终层还是错，于是断言“第 6 层引入了幻觉”。这结论过强。更合理的说法是：到第 6 层为止，错误词已经成为强候选，且后续层没有把它稳定压下去。工程上，这意味着你应优先检查早层上下文编码、检索注入或提示格式，但不能仅凭 Logit Lens 给某一层做因果定罪。

还有一个很容易忽略的问题，是“正确答案是否真的只有一个 token”。  
例如英文里的 `New York`、`San Francisco`，中文里的复合词和专有名词，都可能跨多个 token。此时只盯住第一个 token，可能会高估模型的掌握程度。更稳妥的做法是同时观察：
- 第一个 token 何时进入 top-k
- 后续 token 是否也同步变强
- 整个短语的联合生成是否稳定

这也是为什么 Logit Lens 适合“看趋势”，但不适合独立承担全部评估任务。

---

## 替代方案与适用边界

Logit Lens 最大的问题，是默认所有层都能直接共用同一个输出头。现实里这往往不成立。于是就有了 Tuned Lens。

Tuned Lens 的思路是在每一层加一个可学习的仿射校正。所谓“仿射”，就是先做线性变换，再加偏置；白话说，相当于给每层单独做一次坐标校准，再送入最终词表空间。一个常见写法是：

$$
\tilde{h}_\ell = A_\ell \,\mathrm{LN}_f(h_\ell) + b_\ell
$$

再读出为：

$$
\mathrm{logits}_\ell^\text{tuned} = W_U \cdot \tilde{h}_\ell
$$

其中：

- $A_\ell$ 是第 $\ell$ 层专属线性变换
- $b_\ell$ 是第 $\ell$ 层专属偏置
- 它们通过训练得到，目标是让中间层解码分布更接近模型继续跑到最后时的真实输出分布

这比“每层只乘一个缩放再加一个偏置”的简化写法更一般，也更符合 Tuned Lens 的标准做法。

两者对比如下：

| 方案 | 做法 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- | --- |
| Logit Lens | 直接复用最终 LayerNorm + unembedding | 简单、直观、零训练成本 | 低层失真大，层间不完全可比 | 机制教学、快速可视化、粗粒度调试 |
| Tuned Lens | 每层先做可学习仿射校正再解码 | 更接近真实输出分布，层间更可比 | 需要训练校准参数，成本更高 | 严肃评估、跨层比较、跨模型分析 |

什么时候用 Logit Lens 就够了？

如果你的目标是回答下面这些问题，它通常已经足够：

- 正确答案大概从哪几层开始浮现
- 错误答案是在早层就存在，还是后层才被放大
- 改了 prompt 之后，逐层收敛轨迹有没有明显变化
- 某次错误更像“早期知识没激活”还是“后期没压住错误候选”

什么时候更适合 Tuned Lens？

- 你需要更准确地比较不同层概率大小
- 你要跨模型比较中间表示
- 你希望把逐层预测作为较正式的评估指标
- 你关心低层表示，而不是只观察中后层趋势

可以用一个简单比喻理解：Logit Lens 像拿统一标尺去量每层；Tuned Lens 则承认每层坐标系有偏差，先校准标尺再测。前者快，后者准。

所以边界很明确。Logit Lens 适合“看趋势、看节奏、看候选重排”；Tuned Lens 更适合“做较严肃的定量比较”。如果你只是想看模型推理从模糊到清晰的过程，Logit Lens 已经非常有价值；如果你想把这些中间结果拿去做更严格的工程判断，最好引入校准。

最后再补一句容易被忽略的话。  
Tuned Lens 也不是万能因果工具。它解决的是“中间层读出失真”问题，不是“解释等于因果”问题。即便 tuned 后的逐层预测更接近最终输出，也仍然主要是在回答“这一层已经能被读出什么”，而不是“这一层独立决定了什么”。

---

## 参考资料

| 来源 | 年份 | 重点 |
| --- | --- | --- |
| [nostalgebraist, *interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | 2020 | Logit Lens 的原始直觉、逐层预测轨迹和可视化方式 |
| [Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*](https://arxiv.org/abs/2303.08112) | 2023 | Tuned Lens 的正式方法、训练目标、层间校准与实验结果 |
| [Alignment Research, `tuned-lens` GitHub 仓库](https://github.com/AlignmentResearch/tuned-lens) | 2023-至今 | 可复现实验代码、工具实现与实际使用接口 |
| [Vaswani et al., *Attention Is All You Need*](https://arxiv.org/abs/1706.03762) | 2017 | Transformer、residual connection、LayerNorm 等基础架构背景 |
