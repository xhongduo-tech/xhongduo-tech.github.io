## 核心结论

Attention sink 指的是这样一种现象：在自回归 Transformer 的注意力分布里，序列最前面的少数几个位置，会长期吸收一部分本来难以分配的注意力权重。它们未必承载了最重要的语义，但在概率分布上常常扮演“固定落点”的角色。对长文本做流式推理时，真正不能轻易删掉的，往往不是最老的一段语义内容，而是最前面 1 到 4 个作为分布锚点存在的 sink token。

StreamingLLM 的核心策略正是利用这个现象。它不再保留全部历史，而是固定保留前几个 sink token，再拼接最近的一段滑动窗口：

$$
\text{Context}_t = S \cup W_t
$$

其中，$S$ 是固定保留的 sink token 集合，$W_t$ 是时刻 $t$ 最近的 $L$ 个 token。这样做的结果是：KV cache 的大小近似固定，但模型看到的注意力结构仍接近训练期的常见形态。论文和后续复现实验都表明，纯滑动窗口在越过窗口边界后往往迅速失稳；而保留前 1 到 4 个 sink token 后，perplexity 通常能维持在接近正常窗口推理的水平。

| 方案 | 保留前几个 token | 超出窗口后的表现 | 工程结论 |
|---|---:|---|---|
| 纯滑窗 | 0 | 很快失稳，perplexity 急剧上升 | 不适合长流式生成 |
| 仅保留 sink | 1~4 | 稳定性明显恢复，但近因信息不够 | 只解决一半问题 |
| sink + 滑窗 | 1~4 + 最近 $L$ 个 | 可持续长序列生成 | 是 StreamingLLM 标准配方 |

如果用一个直观例子理解，可以把上下文看成一条持续前进的传送带。纯滑窗不仅裁掉旧货物，还把传送带最前面的定位基准一起裁掉；于是后续位置虽然还在，但整个坐标系突然变了。StreamingLLM 则保留最前面几个固定基准点，让后面的区域正常滑动。这样，模型每一步都还能看到一个“像训练时那样”的序列开头。

---

## 问题定义与边界

这个问题的本质，不是“模型完全记不住很远的内容”，而是“你在推理时人为破坏了模型训练期间学到的注意力分布结构”。

自回归模型在生成第 $t$ 个 token 时，会对历史位置 $0,\dots,t-1$ 计算注意力分数，再经 softmax 变成一个总和为 1 的分布：

$$
\alpha_{t,i}=\frac{\exp(q_t^\top k_i / \sqrt{d})}{\sum_{j < t}\exp(q_t^\top k_j / \sqrt{d})}
$$

这里：

| 符号 | 含义 | 新手可直接理解为 |
|---|---|---|
| $q_t$ | 当前 token 的 query | “我现在想找什么信息” |
| $k_i$ | 第 $i$ 个历史 token 的 key | “这个历史位置能提供什么线索” |
| $\alpha_{t,i}$ | 对第 $i$ 个位置的注意力权重 | “当前 token 分多少注意力给它” |

softmax 的关键约束是：无论相关不相关，权重总和都必须等于 1。于是当大量历史 token 对当前预测都不太关键时，模型仍然必须把概率质量分出去。训练中最早的几个位置几乎总是存在，因此它们容易成为稳定的“默认接收点”。这就是 sink 出现的统计原因。

这也解释了为什么纯滑窗会崩。训练期间，位置 0、1、2、3 通常长期可见；而推理时一旦窗口满了，纯滑窗会直接把这些位置删掉。模型随后不得不在一个它几乎没见过的上下文结构里重新分配注意力，导致分布突变。论文与复盘材料中，一个常被引用的现象是：在 `Llama-2-13B` 上，删除初始 token 后，纯滑窗的 perplexity 可从约 `5.40` 飙升到 `5158.07`，这已经不是“略差”，而是几乎立即失控。

这里的边界需要说清楚：

| 参数 | 含义 | 常见取值 | 作用 |
|---|---|---:|---|
| $|S|$ | sink token 数量 | 1~4 | 保持注意力分布锚点 |
| $L$ | 滑窗长度 | 512 / 1024 / 4096 | 保留最近语义上下文 |
| 位置编码 | RoPE / ALiBi 等 | 现代 LLM 常见 | 决定位置不连续时失真如何放大 |
| 任务类型 | 对“久远细节”的依赖程度 | 对话 / 摘要 / 检索 / 代码 | 决定窗口够不够用 |

新手最常见的误解有两个。

第一，以为“前几个 token 重要，是因为它们语义最重要”。这通常不对。attention sink 的主要价值是位置层面的，不是知识层面的。它们更像概率分布的泄压点，而不是早期内容本身很有信息量。

第二，以为“只要保留 sink，就等于有无限上下文”。这也不对。StreamingLLM 保住的是生成稳定性，不是无限记忆能力。模型仍然主要只能利用窗口内的近因内容；窗口外的细节如果没有通过摘要、检索或外部记忆重新引入，仍然会丢失。

---

## 核心机制与推导

StreamingLLM 可以拆成两个核心动作：保留固定前缀，重建连续位置。

第一步是构造推理时刻 $t$ 的上下文：

$$
\text{Context}_t = S \cup W_t
$$

其中

$$
S = \{x_0, x_1, \dots, x_{m-1}\}
$$

表示固定保留的前 $m$ 个 sink token，$W_t$ 表示最近的 $L$ 个 token。于是有效 cache 长度近似为：

$$
|\text{Context}_t| \approx m + L
$$

当新 token 到来时，只在 $W_t$ 内执行淘汰。$S$ 永远不参与 eviction。

第二步是位置重编号。很多新手第一次看到这里会困惑：为什么保留 token 还不够，还要改位置？

原因是大多数 LLM 并不只看 token 本身，还看位置编码。RoPE、ALiBi 等机制都隐含了“当前位置与历史位置的相对结构”。如果你把中间很长一段历史删掉，但继续沿用原始绝对编号，那么模型看到的就是一个有大洞的坐标轴。例如：位置 `0,1,2,3,1001,1002,1003`。这在训练中不是主流分布，容易导致位置相关模式失真。

因此需要对当前 cache 中仍然活着的 token 重新编号：

$$
\text{Reindex}(k)=k', \quad k' \in [0,\ |S|+|W_t|-1]
$$

更直白地说，就是不要问“它原来在第几位”，而要问“它现在在 cache 里排第几位”。

对于 RoPE，可以把这个想法理解成：旋转位置只基于“当前保留序列的连续坐标”来计算，而不是沿用已经断裂的旧坐标。对于 ALiBi，也同样要基于当前可见 token 的相对位置构造偏置，否则 attention bias 会落在不自然的跨度上。

下面用一个最小例子说明。

假设 sink 数量为 2，窗口大小为 4。原始序列走到第 8 个 token 时：

| 原始序列位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| token | A | B | C | D | E | F | G | H |
| 是否保留 | 是 | 是 | 否 | 否 | 是 | 是 | 是 | 是 |
| 重编号后位置 | 0 | 1 | - | - | 2 | 3 | 4 | 5 |

于是模型当前实际看到的上下文是：

$$
[A, B, E, F, G, H]
$$

而不是完整历史 `[A,B,C,D,E,F,G,H]`。关键在于，这个新上下文的坐标仍然连续：`0,1,2,3,4,5`。

再看注意力层面的意义。原本某个头可能习惯把一部分注意力分给最前面的固定位置。如果纯滑窗把 `A,B` 删掉，那么这些“原本有固定落点”的权重就会突然被迫改分配。这个变化不是小扰动，而是整张 softmax 分布的归一化结构变化。StreamingLLM 的作用，就是保留这部分结构稳定性。

真实工程里，这类效应常见于长会话客服、多轮代理调用、长摘要和文档问答。只用纯滑窗时，模型经常在长序列后出现以下现象：

| 现象 | 常见表现 |
|---|---|
| 重复生成 | 连续复述同一句或同一段模板 |
| 语义漂移 | 回答从原主题慢慢偏掉 |
| 指令失效 | 系统提示早期约束不再稳定生效 |
| 格式崩溃 | 原本要求的 JSON / 列表 / 代码格式开始变形 |

采用 `sink + 滑窗` 后，这些问题通常不能完全消失，但会显著减弱。

---

## 代码实现

实现时需要保证三个不变量：

1. 前 `m` 个 sink token 永不淘汰。
2. 最近窗口采用 FIFO 淘汰。
3. 所有仍在 cache 中的 token 每一步都按当前存活顺序连续编号。

下面给出一个可以直接运行的最小实现。它不依赖任何第三方库，用来演示 StreamingLLM 的数据结构与更新逻辑。

```python
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple


@dataclass(frozen=True)
class CacheEntry:
    token: str
    position: int
    is_sink: bool


class StreamingCache:
    def __init__(self, sink_size: int = 4, window_size: int = 8):
        if sink_size < 0 or window_size < 0:
            raise ValueError("sink_size and window_size must be non-negative")
        self.sink_size = sink_size
        self.window_size = window_size
        self.sink: List[str] = []
        self.window: Deque[str] = deque()

    def append(self, token: str) -> None:
        if len(self.sink) < self.sink_size:
            self.sink.append(token)
            return

        if self.window_size == 0:
            return

        if len(self.window) >= self.window_size:
            self.window.popleft()
        self.window.append(token)

    def tokens(self) -> List[str]:
        return self.sink + list(self.window)

    def entries(self) -> List[CacheEntry]:
        tokens = self.tokens()
        return [
            CacheEntry(token=tok, position=i, is_sink=(i < len(self.sink)))
            for i, tok in enumerate(tokens)
        ]

    def debug_state(self) -> Tuple[List[str], List[int], List[bool]]:
        entries = self.entries()
        return (
            [e.token for e in entries],
            [e.position for e in entries],
            [e.is_sink for e in entries],
        )


def demo() -> None:
    cache = StreamingCache(sink_size=2, window_size=3)

    for tok in ["A", "B", "C", "D", "E", "F"]:
        cache.append(tok)

    tokens, positions, is_sink = cache.debug_state()

    assert tokens == ["A", "B", "D", "E", "F"]
    assert positions == [0, 1, 2, 3, 4]
    assert is_sink == [True, True, False, False, False]

    print("tokens   :", tokens)
    print("positions:", positions)
    print("is_sink  :", is_sink)


if __name__ == "__main__":
    demo()
```

这段代码的执行结果应为：

```text
tokens   : ['A', 'B', 'D', 'E', 'F']
positions: [0, 1, 2, 3, 4]
is_sink  : [True, True, False, False, False]
```

这个例子只演示“存什么”和“怎么编号”，没有真正计算 attention。若映射到真实推理系统，可以把每个 token 替换成对应的 KV 状态。抽象结构如下：

| 字段 | 含义 | 说明 |
|---|---|---|
| `token` | token 或其对应状态 | 这里可替换成 K/V 向量 |
| `position` | cache 内连续位置 | 用于构造位置编码 |
| `is_sink` | 是否属于固定前缀 | 决定能否被 eviction |

如果写成更贴近推理引擎的伪代码，流程如下：

```text
initialize sink = first m tokens
initialize window = empty queue

for each new token x_t:
    if sink not full:
        append x_t to sink
    else:
        if len(window) == L:
            evict oldest token from window
        append x_t to window

    alive_tokens = sink + window
    positions = [0, 1, ..., len(alive_tokens)-1]
    run attention on alive_tokens with these positions
```

如果进一步映射到 KV cache 实现，通常需要处理下面三个工程细节：

| 细节 | 说明 |
|---|---|
| K/V 分区存储 | sink 区和 window 区最好物理分开，避免误删 |
| 批处理一致性 | batch 内不同样本长度不同，重编号逻辑要独立 |
| 位置编码适配 | RoPE / ALiBi 的位置输入必须基于当前 cache 连续坐标 |

新手实现里最容易出错的是“逻辑上保留了 sink，但位置还沿用原编号”。这种实现表面上像 StreamingLLM，实际上只做了一半。

---

## 工程权衡与常见坑

StreamingLLM 的优势很明确，但它不是“打开开关就完事”。实际部署中，最常见的问题都出在实现细节上。

| 坑 | 现象 | 原因 | 对策 |
|---|---|---|---|
| 把 sink 当普通旧 token 一起淘汰 | perplexity 激增，输出失稳 | 分布锚点消失 | sink 单独存放，永不淘汰 |
| 保留了 sink，但位置仍用原始绝对编号 | 生成质量仍明显下降 | cache 坐标不连续 | 位置按当前 cache 连续重编号 |
| sink 太少 | 稳定性恢复不完全 | 不足以提供固定落点 | 通常优先试 4 个 |
| 窗口太短 | 答案忽略刚提过的信息 | 近因上下文不够 | 按任务增大 $L$ |
| 把 sink 当“重要语义 token”做人工挑选 | 规则复杂但收益很小 | sink 主要是位置效应 | 默认保留最前几个 token |
| 只关注显存，不看输出行为 | 表面能跑，实际质量差 | 稳定性问题在长序列才暴露 | 做长轮次回归测试 |

还有两个重要权衡。

第一，StreamingLLM 节省的是内存增长速度。全量 KV cache 的空间复杂度近似随序列长度线性增长，而 `sink + 滑窗` 的 cache 长度近似固定为 $m + L$。这对长会话和持续生成非常有价值。

第二，它节省不了“窗口外精确记忆”的需求。假设窗口只有 1024，那么 5000 token 之前的某个具体变量名、某段代码、某个合同条款，如果没有被重新提及或外部检索拉回，模型大概率无法稳定利用。StreamingLLM 保住的是结构稳定性，而不是跨无限长度的细节可访问性。

工程上常见的一个经验配置如下：

| 场景 | 典型配置 | 说明 |
|---|---|---|
| 长对话机器人 | `sink=4, window=1024~4096` | 保住系统提示和对话稳定性 |
| 流式摘要 | `sink=4, window=2048+` | 依赖近因上下文较多 |
| 日志/代理流处理 | `sink=4, window=512~2048` | 更强调持续吞吐 |
| 长代码问答 | `sink=4 + 检索/摘要` | 单靠滑窗通常不够 |

如果要给一个简洁判断标准，可以这样理解：

- 任务主要依赖最近信息，且需要长时间稳定生成：StreamingLLM 很适合。
- 任务强依赖很久以前的精确细节：StreamingLLM 不够，需要配合检索、摘要或外部记忆。

---

## 替代方案与适用边界

StreamingLLM 不是唯一选择。要理解它的价值，最好把它放进一组常见策略里比较。

| 策略 | 内存占用 | 计算开销 | 长序列稳定性 | 优点 | 局限 |
|---|---|---|---|---|---|
| 全量 KV cache | 高，随长度增长 | 逐步增加 | 最强 | 不丢历史，最直接 | 显存和延迟压力大 |
| 纯滑窗 | 低，近似固定 | 稳定 | 差 | 实现简单 | 容易破坏训练分布 |
| sink + 滑窗 | 低到中，近似固定 | 稳定 | 好 | 在成本和稳定性之间平衡最好 | 仍丢窗口外细节 |
| 滑窗重算 | 低显存 | 高计算 | 好 | 可在低显存下逼近更好精度 | 很慢，不适合在线服务 |
| 检索增强 | 取决于外部系统 | 有额外检索开销 | 视实现而定 | 能找回久远信息 | 不是原生流式 cache 方案 |
| 分块摘要/外部记忆 | 中 | 中 | 中到好 | 能压缩历史关键信息 | 摘要会引入信息损失 |

实际选择取决于任务目标。

如果机器资源足够，且序列长度可控，全量 KV cache 仍然是最稳的基线。如果资源紧张、只关心最近上下文，纯滑窗最简单，但要接受长序列容易崩的代价。如果要在线处理长会话、大文档流或代理执行日志，`sink + 滑窗` 往往是最实用的折中。

StreamingLLM 的适用边界也需要明确：

| 适用 | 不适用或不足 |
|---|---|
| 长会话生成 | 必须保留极久远的逐 token 细节 |
| 流式文档摘要 | 法律条文逐段精确对齐 |
| 长日志解释 | 超长代码库全局精确引用 |
| 代理系统事件流 | 所有历史细节都必须无损可访问 |

因此，在复杂系统中，StreamingLLM 常常不是单独使用，而是与其他机制配合：

$$
\text{Long-context system} \approx \text{StreamingLLM} + \text{Retrieval} + \text{Summarization} + \text{External Memory}
$$

它解决的是“流式推理如何稳定”这个问题，而不是“无限上下文如何完整记忆”这个问题。

---

## 参考资料

1. Xiao, Guangxuan et al. *Efficient Streaming Language Models with Attention Sinks*. ICLR 2024. 贡献：正式提出 StreamingLLM，系统展示 attention sink 现象，并报告长达 4M token 级别的稳定流式生成结果。  
2. MIT Han Lab 项目页：*Efficient Streaming Language Models with Attention Sinks*. 贡献：提供方法示意图、实验摘要、实现说明与部署视角，便于从论文过渡到工程理解。  
3. Michael Brenndoerfer, *Attention Sinks: Enabling Infinite-Length LLM Generation with StreamingLLM*, 2025. 贡献：用更易读的方式解释了 softmax 泄压、分布锚点与位置连续编号之间的关系。  
4. Liner 论文复盘：*Efficient Streaming Language Models with Attention Sinks*. 贡献：整理了若干关键实验数值，包括 `Llama-2-13B` 在纯滑窗删除初始 token 后 perplexity 从约 `5.40` 跳升到 `5158.07` 的现象。  
5. GoKawiil 文章：*How Attention Sinks Keep Language Models Stable*. 贡献：从实现角度解释了为什么实际部署中常用“前 1~4 个 token + 最近窗口”的结构。  
6. 补充阅读建议：如果需要进一步理解位置机制，建议同时回看 RoPE 与 ALiBi 的原始论文或技术解读。原因不是它们本身提出了 attention sink，而是 StreamingLLM 的“重编号”步骤只有放在位置编码机制下才容易真正看懂。
