## 核心结论

长上下文的最新突破，核心不是“把窗口从 8K 改成 128K、1M 就结束了”，而是让模型在更长输入下仍然能**稳定引用远处信息**。这里的“稳定引用”可以白话理解为：前面第 3 页写过的关键条件，模型到第 300 页还能正确提起，而不是只会复述最近几段。

目前主流做法大致分成五类，而且往往是组合使用：

| 方向 | 解决的问题 | 代表思路 | 优点 | 主要代价 |
| --- | --- | --- | --- | --- |
| 直接扩窗 | 让模型一次装下更多 token | 更长位置编码、更大 KV cache | 接口最直接 | 成本高，效果不一定线性提升 |
| 位置编码扩展 | 让模型“认得”更远的位置 | RoPE 插值、YaRN | 对已有模型改动相对小 | 超出训练分布后仍可能漂移 |
| 稀疏/分块注意力 | 降低全局注意力平方复杂度 | chunk、block、sliding window | 更省算力 | 全局依赖可能被切断 |
| 外部记忆/交叉注意力 | 保留窗外信息 | CEPE、memory module | 不必完全重训主模型 | 系统更复杂 |
| 检索混合 | 不把全文都塞进去 | RAG + 长上下文 | 成本可控，更新灵活 | 检索错了就全错 |

真正的突破点是：这些方法开始能在工程上协同工作。比如，位置编码负责“把坐标系拉长”，分块注意力负责“让计算可承受”，滑窗缓存负责“保留最近局部细节”，检索或外部记忆负责“把窗外关键信息再拉回来”。

一个最小结论是：**长上下文突破解决的是“可用性”，不是“无限记忆”**。模型能读 1M token，不等于它在 1M token 下做跨段推理仍然准确。对代码库分析、法律文档、长会议记录，这个区别非常关键。

---

## 问题定义与边界

“上下文窗口”指模型单次推理时可见的输入长度。白话讲，就是模型一次能摊开阅读的文本页数。长上下文问题不是单纯的存储问题，而是下面三个问题同时成立：

1. 模型是否能看见这么长的输入。
2. 模型是否还能区分远近位置。
3. 模型是否还能在这么长的输入里准确找到相关信息并完成推理。

很多误解出在第 1 点和第 3 点被混为一谈。官方说某模型支持 1M token，通常先说明“能喂进去”；但工程上更关心的是“在 1M token 内完成问答、对比、归因、证据定位时，结果是否还可靠”。

下面这个表更接近实际工程判断：

| 训练窗口 | 任务窗口 | 典型风险 | 现象 |
| --- | --- | --- | --- |
| 接近训练窗口 | 接近训练窗口 | 风险较低 | 表现通常稳定 |
| 远超训练窗口 | 只做简单检索 | 中等风险 | 能找到片段，但推理未必稳 |
| 远超训练窗口 | 做跨段比较/归纳 | 高风险 | 中段遗忘、引用错位 |
| 极长窗口 | 需要精确定位证据 | 很高风险 | 页码、条款、函数关系容易漂移 |

这里的“分布外”可以白话理解为：模型训练时主要见过 8K 或 32K 长度的样本，但推理时突然让它处理 128K 或 1M。即使架构支持，模型也可能不知道如何在这么长的距离上分配注意力。

玩具例子可以说明这个边界。

假设有一篇 120 段的长文：

- 第 5 段写“合同金额上限为 200 万”
- 第 63 段写“付款分三期”
- 第 117 段问“是否存在与付款条款相关的金额上限约束”

如果模型只会盯着最近几十段，它可能只看见“付款分三期”，却漏掉第 5 段的金额限制。于是它不是完全没看见全文，而是**跨远距离关联失败**。

所以问题定义必须先收紧：你要解决的是哪一种长上下文任务？

- 全文检索：找到相关段落即可。
- 跨段问答：找到后还要正确组合答案。
- 全文归纳：要综合许多分散信息。
- 精确审计：不仅要结论，还要条款级证据定位。

这四类任务对窗口的真实要求完全不同。很多场景根本不需要 1M token，而是需要“在 50K 到 200K 内可靠完成跨段引用”。

---

## 核心机制与推导

先看最基础的障碍：Transformer 的自注意力会计算每个 token 与其他 token 的相关性。如果序列长度是 $n$，全局注意力的计算和存储通常随 $n^2$ 增长。也就是说，长度翻 10 倍，代价可能接近翻 100 倍。这解释了为什么“直接堆令牌”很快会变得昂贵。

### 1. 位置编码扩展：先把“坐标系”拉长

位置编码是模型区分“第 10 个 token”和“第 10000 个 token”的方法。RoPE 可以理解为一种把位置信息编码进向量旋转角度的技术。白话讲，它像给每个 token 发一个带周期规律的坐标标签。

长上下文下，经典问题是原本训练时只学会了较短距离的坐标规律，到了超长距离后，这套规律开始失真。YaRN 的思路就是按维度给 RoPE 做缩放，让不同频率分量承担不同职责：

$$
\theta'_d = \lambda_d \cdot \theta_d
$$

其中：

- $\theta_d$ 是原始第 $d$ 个维度的位置角频率
- $\lambda_d$ 是缩放因子
- 白话解释：不是所有维度都同样拉长，而是有的维度多拉一点，有的少拉一点

再配合 attention temperature $\tau$ 调整 softmax 的敏感度，可以写成：

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\tau \sqrt{d}}\right)V
$$

这里的 $\tau$ 可以白话理解为“注意力温度旋钮”：

- $\tau$ 小一些，分配更尖锐，更容易盯住少数位置
- $\tau$ 大一些，分配更平滑，但也更可能稀释重点

YaRN 的价值不在于某个单一公式，而在于它承认一件事：**长上下文不是简单按比例拉伸原坐标，而是要兼顾近处细节和远处结构**。高频部分保留局部细节，低频部分承担长距离延展，这样模型才不至于一扩窗就把局部顺序感弄坏。

### 2. 分块与滑窗：不要让所有 token 彼此全连

“滑动窗口注意力”可以白话理解为：每个 token 只重点看附近一段，而不是全文所有 token。这样复杂度会大幅下降，因为模型不再做完全全局连接。

如果窗口大小为 $w$，则粗略上计算量更接近 $O(n \cdot w)$，当 $w \ll n$ 时，成本显著低于 $O(n^2)$。

但滑窗有天然缺点：远处信息可能掉出窗口。于是工程上常再加“缓存”或“外部记忆”。最近内容走滑窗，历史内容进入摘要记忆、状态记忆或额外 KV 表示。

### 3. CEPE：把“长历史”交给外接编码器

CEPE 的思路很重要，因为它说明长上下文不一定要重训整个大模型。它把系统分成两部分：

- 主解码器：负责正常生成，参数尽量冻结
- 小编码器：负责先把长文本分块编码成额外的 key/value

然后在解码阶段插入 cross-attention。交叉注意力可以白话理解为：主模型当前正在写答案时，额外去查询另一份“长历史笔记”。

公式上可写成：

$$
\text{CrossAttn}(H, K_{ext}, V_{ext})=\text{softmax}\left(\frac{H K_{ext}^\top}{\sqrt{d}}\right)V_{ext}
$$

其中：

- $H$ 是主模型当前隐藏状态
- $K_{ext}, V_{ext}$ 是小编码器提供的长上下文表示

这相当于说：主模型不必把 128K 全部直接塞进原生自注意力里，而是把远处信息先压成另一套可查询的记忆，再按需取用。

玩具例子：

- 原模型训练窗口 8K
- 一份 120K 的文档被切成 60 个 chunk
- 每个 chunk 先由小编码器处理，形成 chunk 级表示
- 主模型回答问题时，不是重新全读 120K，而是对这 60 个 chunk 表示做选择性访问

这就像考试时桌上除了当前草稿纸，还有一本做过标记的参考册。你不需要每次从头翻全书，只需要在生成时去查相关页。

### 4. 记忆模块：窗外信息不直接丢弃

AllMem 一类方法进一步强调：即使 token 已经滑出当前窗口，也不要简单删除，而是写入一个可更新的记忆模块。这里的“test-time memory”可以白话理解为：模型在推理时边读边记，把早先的重要信息压缩成后续还可访问的状态。

因此，长上下文的真实机制已经从“更长的单块序列”转向“局部窗口 + 可查询外部记忆 + 必要时检索补充”的组合系统。

---

## 代码实现

下面用一个可运行的 Python 玩具实现说明 CEPE 风格的核心流程：先分块，再生成块级记忆，再在回答时做简单打分检索。它不是论文级实现，但足够说明“主模型不必直接重读全文，而是通过额外表示访问长历史”。

```python
from collections import Counter
from math import log

def tokenize(text: str):
    return [w.strip(".,:;!?()[]").lower() for w in text.split() if w.strip()]

def chunk_text(text: str, chunk_size: int = 20):
    tokens = tokenize(text)
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

def encode_chunk(chunk_tokens):
    # 用词频向量模拟“小编码器输出的记忆表示”
    return Counter(chunk_tokens)

def score_query(memory, query_tokens):
    # 用简单的 tf 加权打分模拟 cross-attn 前的相关性选择
    return sum(memory.get(tok, 0) for tok in query_tokens)

def cepe_expand_answer(long_text: str, question: str, topk: int = 2):
    chunks = chunk_text(long_text, chunk_size=12)
    memories = [encode_chunk(chunk) for chunk in chunks]

    q_tokens = tokenize(question)
    ranked = sorted(
        enumerate(memories),
        key=lambda x: score_query(x[1], q_tokens),
        reverse=True
    )

    selected_ids = [idx for idx, _ in ranked[:topk]]
    selected_chunks = [" ".join(chunks[idx]) for idx in selected_ids]
    return selected_ids, selected_chunks

doc = """
Section A contract amount cap is two million yuan.
Section B payment is made in three stages after acceptance.
Section C liability survives termination for twelve months.
Section D payment disputes must still respect the amount cap in Section A.
"""

ids, chunks = cepe_expand_answer(doc, "What limits payment disputes?")
joined = " ".join(chunks)

assert len(ids) == 2
assert "payment" in joined
assert "cap" in joined or "amount" in joined
print(ids, chunks)
```

这个实现故意做得很朴素，重点有三个：

1. `chunk_text` 对应长文本分块。
2. `encode_chunk` 对应小编码器输出额外记忆。
3. `score_query` + `topk` 对应主模型在回答时只访问最相关的块。

真实工程例子可以看代码库问答。

假设一个仓库有 400 个文件，总长 180K token。用户问：“为什么订单超时后，库存会被回滚，但优惠券状态没有恢复？”这类问题通常需要跨文件追踪：

- `order_service.py` 里有超时关闭逻辑
- `inventory_worker.py` 里有库存补偿
- `coupon_service.py` 里缺少对应回滚
- `event_schema.py` 里还能看到消息字段是否带 coupon id

如果只做短上下文 RAG，可能检索到单个函数，却漏掉异步任务和事件链路。长上下文系统更适合把整个相关子仓库一次装入，再借助 chunk memory 做跨文件引用，回答“库存回滚了，但优惠券没恢复，因为超时补偿链路只订阅了库存事件，未订阅券状态恢复事件”。

下面是更接近工程结构的伪代码：

```python
def cepe_infer(chunks, small_encoder, decoder, question):
    kv_extra = []
    for chunk in chunks:
        k, v = small_encoder(chunk)
        kv_extra.append((k, v))

    state = decoder.init_state(question)
    outputs = []

    for _ in range(decoder.max_steps):
        x = decoder.self_attn(state)
        x = decoder.cross_attn(x, kv_extra)
        state = decoder.feed_forward(x)
        token = decoder.sample(state)
        outputs.append(token)
        if token == "<eos>":
            break

    return outputs
```

这类实现的关键工程价值是：**扩展模块可单独优化，主模型尽量不动**。当解码器被冻结时，适配成本和显存成本通常比“从头训练一个超长窗口模型”更低。

---

## 工程权衡与常见坑

长上下文最容易犯的错误，是把“最大窗口”当成“稳定工作窗口”。两者往往不是一回事。

| 常见坑 | 白话解释 | 典型后果 | 常见规避方法 |
| --- | --- | --- | --- |
| 中段遗忘 | 模型容易记住开头和结尾，忽略中间 | 条款漏读、代码依赖漏链 | 滑窗重排、关键段前置、额外 memory |
| 注意力稀释 | 文本太长，相关线索被噪声淹没 | 回答泛泛而谈 | chunk 检索、top-k 过滤、temperature 调整 |
| 位置漂移 | 远距离位置信号失真 | 引错页码、引错函数 | RoPE 扩展、长度专项训练 |
| 成本飙升 | token 多导致时延和显存上升 | 推理慢、费用高 | 分块注意力、缓存、外接编码器 |
| 检索噪声 | 混合 RAG 时召回不准 | 看似“有依据”但依据错 | rerank、结构化索引、多路召回 |
| 评测失真 | demo 能跑，真实任务不稳 | 上线后错误率高 | 用真实数据反复压测 |

其中“中段遗忘”尤其值得强调。很多长文任务中，模型对首尾信息比较敏感，对中间部分反而容易丢。这和人读长文只记开头结尾有点像，但模型的问题更结构化：它的注意力分布、位置编码、检索机制都可能在中段共同退化。

真实工程例子：法律合同审查。

你把一个并购项目的材料包一次喂给模型，总长 150K token，目标是找出“控制权变更后触发提前还款”的所有条款。常见坑有：

- 条款分散在主协议、补充协议、担保协议中
- 提前还款可能不用完全一致的表述
- 真正关键的定义项在前面 20K，触发条件却在后面 110K

如果只看官方 200K 窗口就盲目全塞，模型可能会给出一份“看起来完整”的结论，但漏掉中段的补充协议条款。更稳的做法通常是：

1. 先做 50K、100K、150K 三档压测。
2. 设计必须命中的证据题，而不是只看主观回答是否顺。
3. 若高窗口下开始漂移，就退回较短窗口，叠加 chunk memory 或检索。
4. 对关键结论强制要求“结论 + 原文定位 + 交叉来源”。

这里的原则是：**窗口不是越大越好，而是越大越需要验证。**

---

## 替代方案与适用边界

并不是所有长文任务都该用长上下文模型。选择时至少看四件事：文档是否固定、信息是否经常更新、是否需要跨段推理、是否必须给出精确证据。

| 方案 | 适合场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| 纯长上下文 | 固定大文档、跨段依赖强 | 读全局方便，少做外部调度 | 成本高，可靠长度有限 |
| 纯 RAG | 数据常更新、目标片段明确 | 便宜、快、可扩展 | 检索错就答错，跨段推理弱 |
| Hybrid memory | 既长又复杂，预算有限 | 兼顾全局和成本 | 系统实现更复杂 |
| 分层摘要 | 先归纳再问答 | 结构清晰 | 摘要过程可能丢信息 |

可以用一个简单判断：

- 如果任务是“从持续变化的知识库里找到最新规则”，优先 RAG。
- 如果任务是“审完整份固定档案并做跨章节比较”，优先长上下文或 hybrid。
- 如果任务是“长日志中找局部异常”，通常检索 + 局部窗口更划算。
- 如果任务是“要逐条给出证据并承担审计风险”，必须做专门评测，不能只信窗口规格。

玩具例子：

- 医院每日新增病历，问“患者今天最新的肾功能指标是多少”。
- 这类问题的关键是“最新”，不是“全文推理”。
- 最优解通常不是把半年病历全塞进 1M 上下文，而是先检索最近相关记录，再把短片段送入模型。

真实工程例子：

- 对一个固定的超大代码库做架构梳理，问题是“支付链路失败时，哪些模块会参与补偿，是否存在重复扣减风险？”
- 这需要跨目录、跨异步任务、跨配置和测试文件做关联。
- 如果只靠 RAG，可能召回零散片段却缺少全链路视角。
- 这里更适合长上下文 + chunk memory：先把整个关键子系统装入，再让模型在统一视野下归纳依赖关系。

因此，长上下文的适用边界不是“文档长就上”，而是“需要在长输入中保留远距依赖并做稳定推理时才上”。一旦任务主要矛盾变成“资料更新快”或“目标片段很窄”，RAG 往往更优。

---

## 参考资料

- Long-Context Extrapolation in LLMs: https://www.emergentmind.com/topics/long-context-extrapolation
- Long-Context Modeling: https://www.emergentmind.com/topics/long-context-modeling
- CEPE 论文信息页: https://bohrium.dp.tech/paper/arxiv/2401.07004
- YaRN 与长上下文机制解析: https://www.letsdatascience.com/blog/long-context-models-working-with-1m-token-windows
- AllMem 论文信息页: https://huggingface.co/papers/2602.13680
- 长上下文应用笔记: https://learn-prompting.fr/blog/long-context-windows-guide
- Thomson Reuters CoCounsel 长上下文评测文章: https://www.thomsonreuters.com/en-us/posts/innovation/legal-ai-benchmarking-evaluating-long-context-performance-for-llms/
