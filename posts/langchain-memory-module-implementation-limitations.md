## 核心结论

LangChain 的 Memory，本质上不是“模型真的记住了过去”，而是“在下一次调用模型前，把过去的一部分信息重新注入 prompt”。这里的 prompt 可以直白理解为“本轮发给大模型的完整输入”。因此，Memory 的核心价值不是存储，而是**选择什么历史、以什么形式、在什么时机重新提供给模型**。

对初级工程师来说，最重要的结论有四个。

第一，`ConversationBufferMemory` 这类 Buffer 记忆最容易理解：把历史对话原样保留，再拼回下一轮输入。它适合短对话和调试，因为行为最直观，但 token 成本增长最快。token 可以理解为“模型计费和上下文长度使用的基本单位”。

第二，`ConversationSummaryMemory` 和 `ConversationSummaryBufferMemory` 通过摘要压缩历史，用更少 token 保留大意。它们能延长对话长度，但风险是关键信息会在压缩中丢失，尤其是人名、订单号、时间、配置值这类低频但关键的细节。

第三，`VectorStoreRetrieverMemory` 与 `ConversationVectorStoreTokenBufferMemory` 把旧对话转成向量后检索。向量可以白话理解为“把文本变成可计算相似度的数字表示”。这类方案适合跨会话、跨天、跨线程保留长期信息，但效果强依赖 embedding 质量、切分粒度和检索阈值。

第四，从 2025 到 2026 的官方路线看，LangChain 这套经典 memory 抽象已经进入 deprecated 路线，新项目更推荐用 LangGraph 的短期状态与长期存储，或用 LangMem 做长期记忆管理。也就是说，这些类仍然有学习价值，也能维护旧项目，但不应该再被当成长期主线架构。

| Memory 类型 | 历史来源 | 优点 | 主要问题 | 适合场景 |
|---|---|---|---|---|
| ConversationBufferMemory | 全量最近对话 | 简单、可解释 | token 快速膨胀 | 短对话、调试 |
| ConversationBufferWindowMemory | 最近 N 轮 | 控制长度稳定 | 旧信息直接丢失 | 只关心上下文邻近信息 |
| ConversationTokenBufferMemory | 最近消息，按 token 裁剪 | 比按轮数更稳 | 仍会丢旧事实 | 中等长度对话 |
| ConversationSummaryMemory | 历史摘要 | 节省 token | 摘要失真 | 长对话但细节不重要 |
| ConversationSummaryBufferMemory | 摘要 + 近期消息 | 兼顾大意和近因 | 实现更复杂 | 中长对话 |
| VectorStoreRetrieverMemory | 向量检索历史片段 | 可跨会话 | 检索可能错召回 | 长期事实记忆 |
| ConversationVectorStoreTokenBufferMemory | 近期消息 + 向量检索 | 短期与长期结合 | 参数难调 | 生产中的折中方案 |

---

## 问题定义与边界

如果没有 Memory，模型每一轮只看到当前输入，看不到之前聊过什么。于是“上文提到的保修政策”“刚才那段代码里的变量名”“你之前说你用的是 Windows”这些信息都会丢失。Memory 要解决的不是数据库问题，而是**上下文恢复问题**。

这件事有三个边界必须先说清。

第一，Memory 不等于长期知识库。它只是在调用时把部分历史拼回去，不会改变基础模型参数，也不会让模型“永久学会”新事实。模型参数可以理解为“训练阶段固化进模型内部的权重”；Memory 只是推理阶段的外部上下文。

第二，Memory 不等于真实用户画像系统。比如用户手机号、会员等级、权限范围，这些高价值字段不能只放在摘要或向量检索里。它们应该有结构化主存储，Memory 只是消费这些信息的一层上下文适配。

第三，Memory 不等于任意长度上下文。无论用哪种 Memory，最终都要落到上下文窗口限制和推理成本。可以把总预算近似写成：

$$
B = T_{system} + T_{input} + T_{history} + T_{tools} + T_{output}
$$

其中 $B$ 是模型上下文上限，$T_{history}$ 才是 Memory 能使用的预算。预算不够时，系统提示词、工具描述、检索文档和历史消息会互相抢空间。

从实现角度，经典 Memory 的调用过程可以抽象成：

$$
history = f(\text{recent messages},\ \text{summary},\ \text{retrieved memories})
$$

然后把 `history` 拼进 prompt。也就是：

```text
prompt = system + history + current_user_input
```

因此，真正的问题不是“要不要记忆”，而是：

1. 保留原文还是保留摘要。
2. 丢弃旧消息还是外部存储后再检索。
3. 用轮数裁剪、token 裁剪，还是语义召回。
4. 哪些事实必须结构化持久保存，不能只靠自然语言记忆。

一个玩具例子可以说明边界。

用户第 1 轮说：“我叫小王，预算 3000，想买轻薄本。”
第 8 轮说：“我还是想要能跑 Docker 的。”
第 20 轮问：“按我刚才的条件推荐。”

如果只保留最近 5 轮，预算 3000 可能丢了。
如果只保留摘要，`Docker` 这种后来补充的关键条件可能被压缩掉。
如果只做向量检索，“预算 3000”可能因为词不够显著没有被召回。
所以，Memory 的边界从一开始就决定了它不是单一方案能覆盖全部情况。

---

## 核心机制与推导

经典 LangChain Memory 的共同接口很像：调用前 `load_memory_variables()`，调用后 `save_context()`。前者负责“取出历史”，后者负责“写入本轮”。

最朴素的 Buffer 机制可以写成：

$$
H_t = H_{t-1} \cup \{(u_t, a_t)\}
$$

其中 $u_t$ 是第 $t$ 轮用户输入，$a_t$ 是模型回答，$H_t$ 是当前历史。它的优点是信息不失真，缺点也直接来自公式本身：历史只增不减。

Summary 的思路是把历史映射成更短的文本：

$$
S_t = g(S_{t-1}, u_t, a_t)
$$

其中 $g$ 通常还是一次 LLM 调用。问题在于，摘要并不满足严格的信息守恒。只要压缩，就可能失去重建原文所需的信息。工程上这意味着：摘要适合保留“发生过什么”，不适合保留“精确字段”。

VectorStore 方案再进一步，把旧消息转成向量并按相似度召回。简化后可以写成：

$$
R(q) = \operatorname{topK}\_{\text{sim}(e(q), e(m_i))}
$$

这里 $e(\cdot)$ 是 embedding 模型，白话就是“把文本编码成向量”；$\text{sim}$ 是相似度函数；$R(q)$ 是对当前问题 $q$ 检索到的旧记忆片段。它解决的是“不是所有旧内容都要重放，只取和当前问题相关的”。

`ConversationVectorStoreTokenBufferMemory` 则是经典方案里最接近生产折中的一种。它把近期消息放在 buffer 里，把超过 token 限制的旧消息转存到向量库。官方文档和源码都说明，`load_memory_variables()` 返回的 `history` 不只是当前 buffer，还会包含从向量库取回的带时间戳片段，以及当前时间信息；`save_context()` 会先写入当前消息，再在超过 `max_token_limit` 后把旧消息移出并存入 retriever 背后的向量库。

这意味着它的实际 prompt 结构更接近：

```text
history =
  current_time
  + retrieved_old_excerpts
  + recent_buffer
```

而不是“单纯的聊天记录拼接”。

下面用一个可运行的玩具实现说明三种机制差异。这个例子没有依赖 LangChain，本质上模拟它们的选择逻辑：

```python
from collections import deque
from math import sqrt

class BufferMemory:
    def __init__(self):
        self.messages = []

    def save(self, user, ai):
        self.messages.append(("Human", user))
        self.messages.append(("AI", ai))

    def load(self):
        return self.messages[:]

class WindowMemory:
    def __init__(self, k=4):
        self.messages = deque(maxlen=k)

    def save(self, user, ai):
        self.messages.append(("Human", user))
        self.messages.append(("AI", ai))

    def load(self):
        return list(self.messages)

class TinyVectorMemory:
    def __init__(self):
        self.docs = []

    def _embed(self, text):
        # 词袋向量：只是玩具实现，用于说明“检索依赖表示质量”
        keywords = ["保修", "预算", "Docker", "轻薄", "游戏", "学生"]
        return [text.count(k) for k in keywords]

    def _cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sqrt(sum(x * x for x in a))
        nb = sqrt(sum(x * x for x in b))
        return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

    def save(self, text):
        self.docs.append((text, self._embed(text)))

    def search(self, query, k=2):
        qv = self._embed(query)
        ranked = sorted(
            self.docs,
            key=lambda item: self._cosine(qv, item[1]),
            reverse=True,
        )
        return [text for text, _ in ranked[:k] if self._cosine(qv, self._embed(text)) > 0]

buffer = BufferMemory()
window = WindowMemory(k=4)
vector = TinyVectorMemory()

turns = [
    ("我叫小王，预算3000，想买轻薄本", "明白，你的预算是3000，偏向轻薄本。"),
    ("我主要是学生使用", "好的，使用场景是学生学习。"),
    ("我还想偶尔跑Docker", "收到，这意味着不能只看最轻，还要考虑CPU和内存。"),
    ("之前提过的预算别忘了", "记得，预算是3000。"),
]

for u, a in turns:
    buffer.save(u, a)
    window.save(u, a)
    vector.save(u)
    vector.save(a)

all_history = " ".join(x[1] for x in buffer.load())
recent_history = " ".join(x[1] for x in window.load())
retrieved = " ".join(vector.search("按我之前的预算和Docker需求推荐"))

assert "预算3000" in all_history
assert "预算3000" not in recent_history  # 最近窗口可能已经把最早事实丢掉
assert "Docker" in retrieved
print("Buffer保留全量，Window可能丢旧事实，Vector能按语义召回部分旧信息。")
```

这个例子里，`WindowMemory` 会因为窗口裁剪而忘掉“预算 3000”，而 `TinyVectorMemory` 有机会按查询把 `Docker` 相关内容找回来。它也顺便暴露了一个事实：检索效果完全取决于表示质量。玩具词袋都这么脆弱，真实工程里 embedding 模型、分块策略、metadata 过滤只会更重要。

真实工程例子更能说明问题。一个售后客服机器人通常有三层上下文：

1. 当次会话的最近消息，用 Buffer 或 TokenBuffer 保留。
2. 当天或当前工单的摘要，用 SummaryBuffer 压缩。
3. 跨天、跨工单的用户历史与FAQ命中记录，用 VectorStore 或业务数据库检索。

用户今天问“我上次那个保修单后来怎么处理了”，系统不能只靠最近聊天记录。它需要先按用户 ID 和工单号查结构化数据，再把必要字段与检索到的自然语言历史一起拼进 prompt。这才是 Memory 在生产里真正可用的形式。

---

## 代码实现

如果站在经典 LangChain API 的角度，理解 `ConversationVectorStoreTokenBufferMemory` 最重要，因为它最能体现“短期 + 长期”的组合设计。

它的工作流可以概括成四步：

1. 当前轮开始前，调用 `load_memory_variables()`。
2. 这个方法先从向量库检索与当前输入相关的旧片段，再与近期 buffer 组合成 `history`。
3. 把 `history` 塞进 prompt，调用模型。
4. 模型回答后，调用 `save_context()` 写入本轮消息；如果超出 `max_token_limit`，就把旧消息转存到向量库。

伪代码如下：

```python
history = memory.load_memory_variables({"input": user_input})["history"]
prompt = template.format(history=history, input=user_input)
response = llm.invoke(prompt)
memory.save_context({"input": user_input}, {"output": response})
```

如果你维护的是旧版或经典项目，典型初始化方式类似下面这样：

```python
from langchain.memory import ConversationVectorStoreTokenBufferMemory
from langchain_openai import OpenAI

retriever = chroma.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.75},
)

memory = ConversationVectorStoreTokenBufferMemory(
    llm=OpenAI(),
    retriever=retriever,
    max_token_limit=1000,
    return_messages=True,
)
```

这里有三个参数决定了大部分行为。

| 参数 | 作用 | 调大后的效果 | 调小后的效果 |
|---|---|---|---|
| `max_token_limit` | 近期 buffer 的 token 上限 | 近期细节更多，成本更高 | 更早把消息转存到向量库 |
| `k` | 检索返回条数 | 召回更多，噪声也更多 | 更精简，但容易漏信息 |
| `score_threshold` | 相似度阈值 | 严格过滤，减少错召回 | 宽松召回，增加噪声 |

但如果你是新项目，应该先知道一个重要边界：官方现在把这套 memory 归在 `langchain_classic` 路线，文档明确标注为 deprecated，且强调 `BaseChatMemory` 这类抽象不适合新代码，尤其和原生 tool calling 聊天模型结合时可能静默失败。新项目更推荐把“短期状态”交给 LangGraph 的 thread state，把“长期事实”交给 store 或 LangMem。

因此，经典 Memory 的代码实现更适合理解设计思想，或者维护老项目；如果要做新系统，最好把架构拆成：

1. 会话内状态：线程级状态或 checkpoint。
2. 长期事实：结构化存储或长期 memory store。
3. 检索增强：按 query 动态召回。
4. 提示词拼接层：明确哪些字段一定进入 prompt。

---

## 工程权衡与常见坑

最常见的误区，是把 Memory 当作“只要接上就更聪明”。事实正相反。Memory 一旦设计不当，最先出现的是成本上涨、延迟变大和错误上下文污染。

第一类坑是 Buffer 过长。对话越长，prompt 越大，延迟和费用越高，而且模型更难聚焦。很多人以为“历史越多越好”，实际上模型常常被无关旧内容分散注意力。短对话里 Buffer 很好用，长对话里它通常只是临时过渡方案。

第二类坑是摘要失真。摘要看起来节省 token，但它引入了第二次生成误差。尤其是用户偏好、数值、版本号、报错栈、路径名、SKU、时间戳，这些内容在摘要里很容易被省略或改写。解决方法不是“写更长摘要”，而是把这类字段单独结构化保存。

第三类坑是向量检索错召回。用户问“保修”，结果召回到别的商品；问“退款”，结果召回到过期工单。这类问题的根因通常不是 Memory 类本身，而是 embedding 模型不匹配、文本切块太粗、metadata 过滤缺失、阈值设置过宽。

第四类坑是把“会话记忆”和“业务真相”混在一起。比如“用户会员等级是 VIP”不能只存在自然语言摘要里，因为一旦摘要错了，系统会给出错误权限。业务真相必须来自可靠主存储，Memory 只能辅助生成回答。

第五类坑是忽略时间衰减。很多旧信息应该过期，例如“今天下午发货”“当前工单处理中”“临时故障已恢复”。如果长期记忆没有 TTL、版本号或最后更新时间，模型会把过时事实当成当前事实继续回答。

下面这个表可以作为排错清单：

| 问题 | 表现 | 根因 | 应对 |
|---|---|---|---|
| token 爆炸 | 成本和延迟快速上升 | Buffer 无上限增长 | 改为 TokenBuffer 或 SummaryBuffer |
| 丢关键细节 | 用户说“你忘了我刚才说的” | 摘要压缩过度 | 关键字段结构化存储 |
| 检索答非所问 | 召回了不相关旧记录 | embedding、切块、阈值不合适 | 调整 chunk、阈值、metadata filter |
| 记住了过时信息 | 回答沿用旧状态 | 没有 TTL 或时间约束 | 加时间戳、版本号、过期策略 |
| 工具调用异常 | 记忆与工具状态不一致 | 经典 memory 与新式 agent/tool calling 耦合差 | 新项目迁移到 LangGraph 状态管理 |

真实工程中，一个电商客服系统通常会这样分层：

1. 最近 3 到 8 轮消息保留原文，保证当前语气和问答链不断裂。
2. 工单摘要只保留状态变化，如“已退款”“等待补件”。
3. 用户等级、订单号、物流状态走结构化查询。
4. 跨天历史投诉、偏好和常问问题走向量检索。
5. 拼 prompt 时明确字段优先级，结构化事实优先于自然语言记忆。

这样做的原因很简单：Memory 负责“补足上下文”，数据库负责“给出真相”。

---

## 替代方案与适用边界

如果你的目标只是做一个能连续聊几轮的 demo，`ConversationBufferMemory` 或 `ConversationBufferWindowMemory` 足够。它们最适合教学、调试和低成本原型，因为你能直接看到传给模型的历史是什么。

如果你的目标是中等长度对话，比如教学助理、代码答疑、销售咨询，`ConversationSummaryBufferMemory` 往往比单纯 Buffer 更实用。它保留最近细节，同时用摘要保留远处大意，是经典 API 里更均衡的方案。

如果你的目标是跨会话、跨天、跨工单的长期记忆，`VectorStoreRetrieverMemory` 或 `ConversationVectorStoreTokenBufferMemory` 更接近需求，但它们不应该单独承担全部事实记忆。长期事实最好分成两类：

1. 结构化且必须准确的信息，进入数据库。
2. 模糊、偏好型、可检索的信息，进入向量记忆。

而如果你做的是新系统，替代方案实际上不是“换一个 Memory 类”，而是直接换抽象层。当前官方路线更推荐：

| 目标 | 更推荐的方向 | 适用边界 |
|---|---|---|
| 会话内连续状态 | LangGraph thread state / checkpointer | 单线程或会话级短期状态 |
| 长期用户偏好 | LangGraph store / LangMem | 跨会话长期记忆 |
| 精确业务事实 | 结构化数据库 | 订单、权限、余额、工单状态 |
| 语义回忆 | 向量库检索 | FAQ、历史偏好、相似问题 |

所以，经典 LangChain Memory 的适用边界可以直接下结论：

1. 学习原理，合适。
2. 维护旧项目，合适。
3. 快速做 demo，合适。
4. 新建复杂 agent 系统，不应再把它当主架构。

---

## 参考资料

| 资料 | 用途 |
|---|---|
| [LangChain API: ConversationBufferMemory](https://api.python.langchain.com/en/latest/langchain/memory/langchain.memory.buffer.ConversationBufferMemory.html) | Buffer 记忆接口与行为 |
| [LangChain API: ConversationVectorStoreTokenBufferMemory](https://api.python.langchain.com/en/latest/memory/langchain.memory.vectorstore_token_buffer_memory.ConversationVectorStoreTokenBufferMemory.html) | 近期消息 + 向量库的组合记忆 |
| [LangChain 源码: vectorstore_token_buffer_memory](https://api.python.langchain.com/en/latest/_modules/langchain/memory/vectorstore_token_buffer_memory.html) | `load_memory_variables()` 与 `save_context()` 的实现细节 |
| [LangChain API 索引: memory](https://api.python.langchain.com/en/latest/langchain_api_reference.html) | 经典 memory 类列表与 deprecated 信息 |
| [LangChain Reference: BaseMemory](https://reference.langchain.com/python/langchain-classic/base_memory/) | 经典 memory 抽象已进入 deprecated 路线 |
| [LangChain Reference: BaseChatMemory](https://reference.langchain.com/python/langchain-classic/memory/chat_memory/BaseChatMemory/) | 新代码不建议继续使用该抽象的官方说明 |
| [LangGraph Memory Overview](https://docs.langchain.com/oss/javascript/langgraph/memory) | 新一代短期与长期记忆的官方思路 |
| [LangMem Introduction](https://langchain-ai.github.io/langmem/) | 长期记忆与记忆管理的新官方方向 |
