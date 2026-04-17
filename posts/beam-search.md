## 核心结论

Beam Search，中文常译为“束搜索”，可以理解为“每一步只保留少量最有希望的候选路径”。它解决的不是“模型怎么预测下一个词”，而是“整句话应该怎么选”。

如果把序列生成看成一棵树，根节点是起始符号，往下每一层表示再生成一个 token，那么：

- 贪心解码只看当前一步最优，速度快，但容易因为早期一步选错，后面整句都变差。
- 完全穷举会检查所有路径，理论上最好，但词表一大就不可算。
- Beam Search 取中间路线：每一层只保留 `beam size = k` 条当前得分最高的部分序列，继续向下扩展。

它的核心目标通常写成：

$$
y^* = \arg\max_y \log p(y|x)
= \arg\max_y \sum_{t=1}^{|y|}\log p(y_t \mid y_{<t}, x)
$$

这里的“对数似然”可以先直白理解为“模型对整条输出路径的累计信心分数”。之所以用对数，是因为多个概率相乘会非常小，取对数后变成求和，计算更稳定。

结论上，Beam Search 常常比贪心解码质量更高，因为它不是只押一条路；但它也一定比贪心更贵，因为每一步都要同时维护多条候选。对初级工程师最重要的认识是：它不是保证全局最优，而是在固定计算预算下，尽量逼近更好的完整序列。

---

## 问题定义与边界

Beam Search 主要用于“序列到序列”或“自回归生成”任务。自回归的白话解释是：模型每次生成一个 token，下一个 token 要依赖前面已经生成的内容。常见场景包括机器翻译、语音识别、摘要生成、图像描述。

它要解决的问题是：

在每一步都有很多候选 token 的情况下，如何在可接受的时间和内存内，找到一条整体分数较高、并且能正常结束的输出序列。

如果不用搜索，直接贪心地每步取最大概率 token，容易出现“局部最好，整体一般”。如果完全搜索，候选数量会按指数增长。Beam Search 的边界条件通常由三类参数决定：

| 参数 | 含义 | 对搜索边界的影响 |
| --- | --- | --- |
| `beam size` | 每一轮最多保留多少条候选前缀 | 决定搜索广度 |
| 终止准则 | 碰到 `eos` 或达到最大长度是否停止 | 决定搜索深度 |
| 评分函数 | 用什么分数比较不同候选 | 决定“好路径”标准 |

这里“前缀”指还没生成完的部分序列，白话就是“句子写到一半的草稿”。

玩具例子可以这样理解。假设你从起点出发，每一层有很多走廊，但你规定“每层只允许 k 个人继续前进”。那么 Beam Search 不是找一条唯一通道，也不是把所有通道都看完，而是每层挑出最宽的几条继续走。这个限制就是它的边界。

真实工程里，这些边界很重要。比如翻译系统里，如果 `beam size=1`，它就退化成贪心解码；如果 `beam size=20`，输出可能更稳，但延迟和显存占用也会上去。再比如，若没有 `eos` 终止规则或最大长度限制，模型可能一直生成下去，形成“停不下来”的坏输出。

---

## 核心机制与推导

Beam Search 的一轮更新，可以拆成五步：

1. 取出当前 beam 里的所有候选前缀。
2. 对每个前缀，尝试接上词表中的所有可能 token。
3. 计算每个新候选的累计得分。
4. 把所有扩展后的候选放在一起排序。
5. 只保留 top-k，作为下一轮 beam。

如果当前 beam 大小是 $k$，词表大小是 $|V|$，那么一轮扩展后，临时候选数是：

$$
k \times |V|
$$

这也是 Beam Search 比贪心贵、但比穷举便宜的直接原因。

更具体地看分数。假设当前已有前缀 $y_{<t}$，它的累计分数是：

$$
s(y_{<t})=\sum_{i=1}^{t-1}\log p(y_i|y_{<i},x)
$$

当我们在末尾接一个新 token $y_t$ 时，新分数变成：

$$
s(y_{\le t}) = s(y_{<t}) + \log p(y_t|y_{<t},x)
$$

也就是说，路径分数是一步一步累加出来的，不是最后一次性算。

下面给一个最小玩具例子。设：

- `beam size = 2`
- 词表为 `{A, B, C, <eos>}`
- 起点为 `<sos>`

第一步模型给出的对数分数如下：

| 候选 | 分数 |
| --- | --- |
| A | -0.3 |
| B | -0.5 |
| C | -1.2 |
| `<eos>` | -2.0 |

保留 top-2，所以 beam 变成：

- `[A]`，分数 `-0.3`
- `[B]`，分数 `-0.5`

第二步继续扩展。假设：

从 `[A]` 扩展得到：

| 候选 | 新增分数 | 累计分数 |
| --- | --- | --- |
| `[A, A]` | -0.4 | -0.7 |
| `[A, B]` | -0.6 | -0.9 |
| `[A, C]` | -1.0 | -1.3 |
| `[A, <eos>]` | -0.8 | -1.1 |

从 `[B]` 扩展得到：

| 候选 | 新增分数 | 累计分数 |
| --- | --- | --- |
| `[B, A]` | -0.3 | -0.8 |
| `[B, B]` | -0.7 | -1.2 |
| `[B, C]` | -1.1 | -1.6 |
| `[B, <eos>]` | -0.9 | -1.4 |

把两组放在一起排序，top-2 是：

- `[A, A]`，`-0.7`
- `[B, A]`，`-0.8`

其余都被剪枝。这里“剪枝”就是把不够好的路径直接丢掉，不再继续扩展。

这说明一个重要点：Beam Search 比较的不是“刚刚生成的那个 token 谁最大”，而是“到当前为止整条路径谁更强”。所以它有机会修正贪心解码的早期错误。

但还要注意一个现实问题：对数概率连乘后，长句子通常更吃亏，因为每多一步都会再加一个负数。因此工程上常加“长度归一化”。一个常见形式是：

$$
\text{score}(y)=\frac{1}{|y|^\alpha}\sum_{t=1}^{|y|}\log p(y_t|y_{<t},x)
$$

这里的 $\alpha$ 是调节项。白话说，它是在防止系统因为“短句分数更容易高”而过早输出 `<eos>`。

真实工程例子：机器翻译模型把 “The method is simple but effective.” 翻成中文时，贪心解码可能先选“这个”，后续越来越别扭；Beam Search 则会同时保留“该方法”“这个方法”等候选前缀，后续继续比较整句累计分数，最后更可能得到“该方法简单但有效”这样的完整结果。

---

## 代码实现

下面给一个可运行的 Python 版本。它不是深度学习框架代码，而是一个最小可运行实现，重点展示 Beam Search 的控制流程。

```python
from dataclasses import dataclass
from math import inf

@dataclass
class Hypothesis:
    tokens: list
    score: float
    finished: bool = False

VOCAB = ["A", "B", "C", "<eos>"]

# 一个玩具模型：输入前缀，返回下一步各 token 的对数分数
TRANSITIONS = {
    (): {"A": -0.3, "B": -0.5, "C": -1.2, "<eos>": -2.0},
    ("A",): {"A": -0.4, "B": -0.6, "C": -1.0, "<eos>": -0.8},
    ("B",): {"A": -0.3, "B": -0.7, "C": -1.1, "<eos>": -0.9},
    ("A", "A"): {"<eos>": -0.2, "B": -0.9, "C": -1.5, "A": -1.0},
    ("B", "A"): {"<eos>": -0.1, "B": -1.2, "C": -1.0, "A": -0.8},
}

def next_log_probs(prefix):
    # 未定义状态时，强制尽快结束
    return TRANSITIONS.get(tuple(prefix), {"<eos>": -0.1, "A": -2.0, "B": -2.0, "C": -2.0})

def beam_search(beam_size=2, max_len=3):
    beam = [Hypothesis(tokens=[], score=0.0, finished=False)]

    for _ in range(max_len):
        candidates = []

        for hyp in beam:
            if hyp.finished:
                candidates.append(hyp)
                continue

            for token, logp in next_log_probs(hyp.tokens).items():
                new_tokens = hyp.tokens + [token]
                finished = (token == "<eos>")
                candidates.append(
                    Hypothesis(
                        tokens=new_tokens,
                        score=hyp.score + logp,
                        finished=finished
                    )
                )

        candidates.sort(key=lambda h: h.score, reverse=True)
        beam = candidates[:beam_size]

        if all(h.finished for h in beam):
            break

    finished_hyps = [h for h in beam if h.finished]
    if finished_hyps:
        return max(finished_hyps, key=lambda h: h.score)
    return max(beam, key=lambda h: h.score)

best = beam_search(beam_size=2, max_len=3)

assert best.tokens[-1] == "<eos>"
assert best.score > -1.0
print(best)
```

这段代码里，几个输入输出要看清：

| 名称 | 类型 | 作用 |
| --- | --- | --- |
| `beam` | 候选前缀列表 | 当前保留的 top-k 路径 |
| `token` | 单个词元 | 当前步尝试接上的新符号 |
| `score` | 浮点数 | 路径累计对数分数 |
| `finished` | 布尔值 | 是否已经生成 `<eos>` |

实现上最容易理解的伪流程就是：

1. 初始化 `beam=[起始状态]`
2. 扩展所有候选
3. 重新排序
4. 截断 top-k
5. 检查是否全部结束

如果接入真实模型，`next_log_probs(prefix)` 不再是手写字典，而是模型前向计算后输出的 log-prob 分布。工程中还常见两个细节：

- 已结束的序列不再扩展，只原样保留参与排序。
- 最终返回时优先从已结束序列中选最佳，否则可能拿到“还没收尾”的半句。

---

## 工程权衡与常见坑

Beam Search 的工程价值很高，但坑也很集中。

第一个坑是“beam 越大不一定越好”。很多初学者直觉上会认为 `beam size` 从 4 提到 16，结果一定更优。实际未必。因为模型训练时通常优化的是逐 token 概率，不是最终任务指标。beam 变大后，它更容易把“模型打分高但人类体验一般”的句子保留下来，这就是常说的 beam search curse。

第二个坑是长度偏差。由于累计的是对数概率，而每一步概率都小于 1，所以对数通常为负，句子越长，总分越低。表现出来就是输出偏短、过早结束。典型调节方式是长度归一化。

第三个坑是重复。尤其在摘要和开放式生成里，beam 可能保留多条非常相似的路径，最后得到“意思一样、措辞重复”的候选，浪费搜索预算。

第四个坑是早停策略不当。有的实现一看到某条路径生成 `<eos>` 就提前返回，这是错的。正确做法通常是继续比较 beam 中其他路径，或者满足更严格的结束条件，否则很可能错过更好的完整句子。

下面给一个“问题-调节”对照表：

| 问题 | 常见表现 | 常见调节项 |
| --- | --- | --- |
| 长度偏短 | 很快输出 `<eos>` | 长度归一化、最小长度约束 |
| 重复严重 | 同一短语反复出现 | 重复惩罚、多样性惩罚 |
| 结果单一 | 多条候选几乎相同 | Diverse Beam Search |
| 延迟过高 | 推理太慢、吞吐下降 | 减小 beam size、缩小词表候选 |
| 分数高但质量差 | 自动指标和人工观感不一致 | 重设评分函数、重排序 |

真实工程例子：语音识别里，声学模型会给出多个相近的词序列假设。如果 `beam=1`，系统可能因为某个高频词在局部更强，就把整句带偏；如果 `beam=5`，就能同时保留几条转录假设，后续再结合语言模型或结束条件选出更完整的一句。这就是 Beam Search 在工业系统里长期被保留的原因。

还有一个常见误区是把 Beam Search 当成“提高模型能力”的方法。它提高的是解码阶段的搜索质量，不会改变模型本身的知识边界。如果模型根本不会某种映射，再大的 beam 也救不回来。

---

## 替代方案与适用边界

Beam Search 不是默认永远最佳，它适合“希望质量稳定、允许一定额外推理成本、输出多为单一高质量答案”的场景，例如机器翻译、ASR、图像字幕。

如果你更关心速度，直接用贪心解码即可。贪心可以看成 `beam size = 1` 的特殊情况。它的优点是快、实现简单、延迟低，缺点是容易错过整体更优路径。

如果你更关心多样性，标准 Beam Search 就不够了。因为它天然偏向保留高分且相似的路径。这时可以考虑：

| 方法 | 搜索容量 | 适用场景 | 代价 |
| --- | --- | --- | --- |
| Greedy | 最小 | 低延迟在线推理 | 质量波动大 |
| Beam Search | 中等 | 翻译、识别、摘要 | 计算量随 beam 增长 |
| Diverse Beam Search | 中等偏高 | 需要多个差异化候选 | 实现更复杂 |
| Stochastic Beam Search | 中等偏高 | 希望引入随机探索 | 结果稳定性下降 |
| Sampling | 可变 | 开放式生成、创作类任务 | 质量可控性较差 |

Diverse Beam Search 的白话解释是：不是只看高分，还要惩罚候选之间太像，从而让不同路径都有机会活下来。Stochastic Beam Search 则是在保留束结构的同时加入随机性，不完全按确定性 top-k 截断。

一个很实用的判断标准是：

- 目标是“找一个最稳妥的答案”，优先 Beam Search。
- 目标是“生成多个不同但都还行的答案”，优先 diverse 或 stochastic 变种。
- 目标是“必须极低延迟”，优先 greedy。
- 目标不是 token 概率最大，而是某个复杂规划目标时，Beam Search 可能就不够，需要更强的搜索或重排序方法。

所以，Beam Search 的适用边界非常清楚：它是固定预算下的近似搜索器，不是万能解码器，也不是全局最优保证器。

---

## 参考资料

- Wikipedia, “Beam search”: https://en.wikipedia.org/wiki/Beam_search
- Emergent Mind, “Beam Search Decoding Overview”: https://www.emergentmind.com/topics/beam-search-decoding
- Built In, “Introduction to the Beam Search Algorithm”: https://builtin.com/software-engineering-perspectives/beam-search
