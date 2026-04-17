## 核心结论

MemGPT 的核心不是“把模型变得更聪明”，而是把“有限上下文窗口”变成“可管理的分层记忆系统”。

这里先定义两个词：

- **上下文窗口**：模型这一次回答时，眼前能直接看到的全部文本。白话说，就是模型当前桌面上摊开的材料。
- **分层记忆**：把信息按“正在用”“最近用过”“长期保存”分开放。白话说，就是桌面、文件柜、仓库三层。

MemGPT 借鉴操作系统里的**虚拟内存**思想。虚拟内存的白话解释是：程序看到的内存空间可以比真实内存大，因为不常用的数据会被临时放到磁盘里，需要时再读回来。映射到大模型里，就是：

| 操作系统概念 | MemGPT 对应层 | 作用 | 访问方式 |
|---|---|---|---|
| RAM | Main Context | 当前推理直接可见的工作记忆 | 模型直接读写 |
| 最近页面/交换区 | Recall Storage | 可回溯的对话历史 | 通过函数检索 |
| Disk/长期存储 | Archival Storage | 长期知识、事实、偏好 | 通过函数检索 |

因此，MemGPT 不是简单“存更多文本”，而是让模型具备一种**自主换入换出**能力：什么时候把旧内容迁出，什么时候把关键事实保留，什么时候把外部信息重新拉回主上下文。

它解决的问题也很明确：即使模型只有有限窗口，也能在多轮任务里表现出接近“无限上下文”的行为。

---

## 问题定义与边界

先把问题说清楚。

大语言模型的每次推理都受限于窗口容量 $C$。如果当前上下文占用为 $T$，那么只有 $T \le C$ 时，模型才能把这些内容一起纳入推理。一旦长期对话、代码协作、客服工单、知识问答不断累积，旧信息就会把新信息挤掉。

这里的关键边界有三个：

| 区域 | 含义 | 是否能被模型直接看到 | 典型内容 |
|---|---|---|---|
| Main Context | 主上下文 | 是 | 系统指令、当前任务、最近消息 |
| Recall Storage | 召回存储 | 否 | 历史对话、过去交互片段 |
| Archival Storage | 归档存储 | 否 | 长期事实、稳定知识、用户偏好 |

三个术语首次出现时可以这样理解：

- **Main Context**：模型当前正在处理的工作区。白话说，就是眼前这一页。
- **Recall Storage**：过去对话的可检索历史。白话说，就是翻聊天记录。
- **Archival Storage**：长期保存的重要知识。白话说，就是存档案，不是每次都摆在桌上。

MemGPT 不是“把所有历史都塞进提示词”，而是规定了只有 Main Context 能直接参与当前推理，Recall 和 Archival 都必须通过显式函数调入。这一点很重要，因为它决定了系统边界：

1. 主上下文容量始终有限。
2. 外部存储容量可以很大，但访问有延迟和检索成本。
3. 模型需要自己管理迁移策略，而不是依赖开发者手工拼 prompt。

一个玩具例子最容易说明边界。

假设你和助手进行了 50 轮对话。前 10 轮里你说过“我对花生过敏”，第 45 轮你又问“晚餐推荐什么？”  
如果系统没有分层记忆，那么第 45 轮时，前 10 轮内容可能已经不在窗口里，模型就可能推荐含花生的菜。  
如果系统有 MemGPT 式分层结构，那么“我对花生过敏”这种稳定事实更适合被整理后放入 Archival，而不是赌它一直留在主上下文里。

所以，MemGPT 的问题边界不是“所有内容永久可见”，而是“所有内容可被管理、可被找回、可按优先级进入当前可见区”。

---

## 核心机制与推导

MemGPT 的机制可以抽象成一个简单的容量控制问题。

设：

- $C$：上下文窗口总容量
- $T$：当前已占用 token 数
- $W_{warn}$：预警阈值
- $W_{flush}$：强制清理阈值

常见设定是：

$$
W_{warn} = 0.7 \cdot C
$$

$$
W_{flush} = 1.0 \cdot C
$$

含义是：

- 当 $T < W_{warn}$，系统还比较宽松，继续在主上下文里工作。
- 当 $T \ge W_{warn}$，系统开始出现“内存压力”，需要主动把旧内容摘要并迁出。
- 当 $T \ge W_{flush}$，系统必须执行强制清理，否则新输入无法进入窗口。

把它写成流程更直观：

$$
\text{if } T \ge W_{warn},\ \text{store(oldest\_chunk)}
$$

$$
\text{if } T \ge W_{flush},\ \text{flush(FIFO)} + \text{summarize}
$$

而主上下文在检索后的更新可以表示为：

$$
C' = C_{active} \cup f_{\text{retrieve}}(q)
$$

这里的 $f_{\text{retrieve}}(q)$ 表示：根据当前问题 $q$，从 Recall 或 Archival 中取回相关内容，再拼接到当前活跃上下文里。

### 为什么必须分层

原因不是“存储空间不够”，而是“注意力预算不够”。

**注意力**这个词的白话解释是：模型在一次推理里，能把多少文本彼此关联起来。即使外部有海量数据，如果不先筛选并调回当前窗口，这些数据也不会自动影响当前回答。

因此，MemGPT 的关键不是单纯存，而是四个显式动作：

| 动作 | 含义 | 白话解释 | 典型触发时机 |
|---|---|---|---|
| store | 写入外部层 | 把暂时不用的内容收起来 | 快接近预警线 |
| retrieve | 从外部取回 | 把需要的旧内容调回来 | 当前问题依赖历史 |
| summarize | 生成摘要 | 用更短文本保留关键信息 | 窗口压力大 |
| update | 更新已存记忆 | 修正已有事实或偏好 | 用户状态变化 |

### 玩具例子：32k 窗口如何工作

设窗口容量 $C = 32000$，则：

$$
W_{warn} = 0.7 \cdot 32000 = 22400
$$

$$
W_{flush} = 32000
$$

假设当前对话已经累计到 23000 token，超过预警线。系统不会等到完全塞满，而是先做一轮迁移：

1. 选出最早的一段历史，例如 1000 token。
2. 对这 1000 token 做摘要。
3. 如果这段主要是过程性对话，则写入 Recall。
4. 如果其中包含稳定事实，比如“用户常住上海”“偏好英文技术资料”，则抽取后写入 Archival。
5. 主上下文只保留摘要或干脆移除原文，腾出空间。

如果继续增长到 32000 token，就要强制 flush。此时 FIFO 历史队列会被清理，系统依赖外部记忆和摘要来维持连续性。

### 真实工程例子：客服助手

考虑一个 SaaS 客服助手：

- 当前工单上下文：放在 Main Context
- 用户过去 3 次工单记录：放在 Recall
- 用户长期偏好与套餐信息：放在 Archival

当用户说“还是上次那个发票问题”时，系统先判断这句话缺少完整上下文，于是调用 `retrieve` 去 Recall 找最近工单；如果又发现此用户是企业客户、需要专票，则去 Archival 取稳定账户事实。

这样做有两个直接收益：

1. 当前窗口只装与本轮工单最相关的信息，减少噪声。
2. 长期事实不需要每轮都重输，但又不会丢失。

这比“把最近 30 轮聊天全部贴进去”更稳定，因为后者会让大量无关细节占满窗口，降低当前问题的有效注意力密度。

---

## 代码实现

下面先用一个最小 Python 程序模拟 MemGPT 的阈值管理。它不是完整实现，但能跑通“预警、摘要、迁移、检索”的核心逻辑。

```python
from dataclasses import dataclass, field

@dataclass
class Chunk:
    text: str
    tokens: int
    kind: str = "dialogue"  # dialogue / fact

@dataclass
class MiniMemGPT:
    capacity: int
    warn_ratio: float = 0.7
    main_context: list = field(default_factory=list)
    recall_storage: list = field(default_factory=list)
    archival_storage: list = field(default_factory=list)

    @property
    def warn_threshold(self):
        return int(self.capacity * self.warn_ratio)

    @property
    def used_tokens(self):
        return sum(chunk.tokens for chunk in self.main_context)

    def add_to_main(self, chunk: Chunk):
        self.main_context.append(chunk)
        self._maybe_evict()

    def _summarize(self, chunk: Chunk) -> Chunk:
        # 用更短摘要表示旧内容
        summary_tokens = max(1, chunk.tokens // 5)
        summary_text = f"summary({chunk.text[:20]})"
        return Chunk(summary_text, summary_tokens, kind=chunk.kind)

    def _maybe_evict(self):
        while self.used_tokens >= self.warn_threshold and self.main_context:
            oldest = self.main_context.pop(0)
            summary = self._summarize(oldest)
            if oldest.kind == "fact":
                self.archival_storage.append(summary)
            else:
                self.recall_storage.append(summary)

    def retrieve(self, keyword: str):
        results = []
        for store in (self.recall_storage, self.archival_storage):
            for chunk in store:
                if keyword in chunk.text:
                    results.append(chunk)
        return results

mem = MiniMemGPT(capacity=100, warn_ratio=0.7)
mem.add_to_main(Chunk("user says name is Sarah", 30, "fact"))
mem.add_to_main(Chunk("small talk about travel", 25, "dialogue"))
assert mem.used_tokens == 55

# 加入新内容后触发预警迁移
mem.add_to_main(Chunk("long discussion about invoice issue", 20, "dialogue"))
assert mem.used_tokens < mem.warn_threshold

# 稳定事实应进入 archival，普通对话进入 recall
assert len(mem.archival_storage) + len(mem.recall_storage) >= 1
assert any(c.kind == "fact" for c in mem.archival_storage) or any(c.kind == "dialogue" for c in mem.recall_storage)

# 检索可以找回被迁出的摘要
results = mem.retrieve("summary")
assert len(results) >= 1
print("ok")
```

上面的程序体现了三个关键点：

1. 主上下文有硬容量。
2. 一旦逼近阈值，就优先迁走最旧内容。
3. 稳定事实与临时对话走不同存储层。

如果把它映射到实际系统，例如 Letta 风格接口，主思路通常是先给 agent 配置核心记忆块，再让模型通过工具函数访问外部层：

```python
from letta_client import Letta, CreateBlock

client = Letta(base_url="http://localhost:8283")

agent_state = client.agents.create(
    model="openai/gpt-4o-mini-2024-07-18",
    embedding="openai/text-embedding-3-small",
    memory_blocks=[
        CreateBlock(label="human", value="My name is Sarah."),
        CreateBlock(label="persona", value="You are a helpful assistant."),
    ],
)
```

这个例子里：

- `memory_blocks` 可以理解为一部分核心记忆初始化。
- `embedding` 的白话解释是：把文本变成向量，方便做语义检索。
- 真正的 Recall/Archival 管理，通常依赖运行时工具调用，例如搜索历史对话、写入归档记忆、更新长期事实。

一个真实工程流程通常是这样：

| 阶段 | 系统动作 | 目的 |
|---|---|---|
| 用户发问 | 写入 Main Context | 让当前问题可见 |
| 达到预警线 | summarize + store | 给窗口减压 |
| 需要旧信息 | retrieve from Recall | 补回历史过程 |
| 需要长期事实 | retrieve from Archival | 补回稳定知识 |
| 用户信息变化 | update Archival | 避免记忆过时 |

例如，用户先说“我以后默认用英文回答”，系统应把它整理成稳定偏好写入 Archival。之后即使隔了很多轮，模型也不必重新看完所有旧聊天，只需要在需要时把这个偏好调回主上下文。

---

## 工程权衡与常见坑

MemGPT 的价值很高，但工程上不是“免费午餐”。

### 1. 显式检索带来额外延迟

每次 `retrieve` 或 `store` 都可能触发额外运行时事件，包括向量检索、数据库读写、摘要生成。这意味着响应链路会变长。

| 常见坑 | 现象 | 原因 | 对策 |
|---|---|---|---|
| 延迟升高 | 回答变慢 | 频繁调用外部记忆函数 | 提前摘要，减少临界时检索 |
| 重复检索 | 同一事实被多次拉回 | Recall 和 Archival 职责混乱 | 稳定事实固定归档 |
| 记忆冲突 | 用户偏好前后矛盾 | update 机制缺失 | 对长期事实做版本更新 |
| 过度摘要 | 细节丢失 | 摘要粒度太粗 | 关键字段结构化存储 |
| 污染主上下文 | 无关旧内容混入 | 检索召回过宽 | 检索后再做过滤与重排 |

### 2. Recall 和 Archival 不要混用

这两个层级虽然都在“外部”，但语义不同。

- **Recall** 更像经历过什么。白话说，是事件记录。
- **Archival** 更像知道什么。白话说，是稳定事实。

如果把“上周聊过退款流程”和“用户默认语言是英文”都放在同一个桶里，就会发生两个问题：

1. 检索命中很多无关内容。
2. 稳定事实不能被快速、确定地找回。

更合理的做法是：

- 时间性、过程性内容放 Recall。
- 稳定性、可复用事实放 Archival。
- 当前正在影响决策的状态放 Main Context。

### 3. 不要等窗口满了再处理

很多实现失败，不是因为架构错，而是因为迁移太晚。

如果等到 $T \approx C$ 才开始摘要，系统通常已经来不及做平滑迁移，只能执行粗暴 flush。这样会导致：

- 当前会话链被突然截断
- 需要立刻再去 Recall 检索
- 响应延迟和错误率一起升高

因此，预警阈值的意义不是“提醒快满了”，而是给系统留出**主动整理记忆**的时间窗口。

### 4. 摘要不是越短越好

摘要的白话解释是：用更少文字保留关键事实。但如果压得太狠，会出现“连续性还在，细节已经死了”的问题。

比如把“客户在 2025 年 9 月、11 月、12 月连续三次遇到发票税号校验错误，且仅企业版租户受影响”摘要成“客户多次遇到发票问题”，那么真正排障时最关键的时间和范围信息就没了。

所以工程上常见的改进是：

- 对事实型信息做结构化提取，而不只做自然语言摘要。
- 对重要历史保留摘要 + 原始片段定位信息。
- 对高价值字段单独建索引，如用户偏好、账户级配置、故障编号。

---

## 替代方案与适用边界

MemGPT 不是唯一方案。它适合“长周期、多轮、跨会话、状态持续演化”的任务，但不是所有问题都需要它。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| MemGPT 分层记忆 | 能跨会话管理状态，记忆可持续演化 | 系统复杂，检索和更新有延迟 | 智能体、客服、长期协作 |
| 直接使用大窗口模型 | 实现简单，少写记忆管理逻辑 | 成本高，窗口仍然有限，噪声大 | 短期复杂任务、一次性分析 |
| 传统 RAG | 外部知识接入成熟，检索链路清晰 | 更偏“查资料”，不擅长持续用户状态 | 文档问答、知识库搜索 |
| 手工摘要历史 | 实现最简单 | 质量依赖人工规则，难以自适应 | 小型原型、低复杂度产品 |

### MemGPT 和大窗口模型的关系

更大窗口当然有帮助，但它解决的是“能看更多”，不是“能长期管理”。

如果一个模型支持 128k 甚至更大窗口，那么很多短中期任务确实可以直接塞进去完成。但在长期系统里，问题仍然存在：

1. 成本会随着输入长度显著上升。
2. 无关历史会稀释当前问题的有效信息密度。
3. 跨会话状态仍需要持久化机制。

所以大窗口更像是“更大的 RAM”，而不是完整的“内存层级”。

### MemGPT 和传统 RAG 的区别

**RAG** 的白话解释是：先从外部知识库检索，再把检索结果拼进提示词。

它擅长补充外部知识，但通常默认“每一轮按当前问题查资料”，不天然负责“我和这个用户长期互动后，应该记住什么”。  
因此，RAG 更像“外部知识访问系统”，而 MemGPT 更像“带生命周期管理的记忆系统”。

一个真实对比场景：

- 法规问答机器人：主要任务是查准文档，传统 RAG 足够。
- 长期客户成功助手：要记住客户偏好、历史故障、升级节奏，MemGPT 更合适。
- 代码协作代理：要记住当前任务状态、上次失败原因、仓库长期约束，MemGPT 或其变体更合理。

所以适用边界可以概括为：

- 如果问题主要是“资料在哪”，优先考虑 RAG。
- 如果问题主要是“这个系统长期记住了什么，并且何时该想起来”，优先考虑 MemGPT。
- 如果任务短、一次性强、成本不敏感，大窗口模型可能更简单。

---

## 参考资料

1. MemGPT 官方项目与论文概览：<https://research.memgpt.ai/>
2. Emergent Mind 对 MemGPT 分层记忆管理的整理：<https://www.emergentmind.com/topics/memgpt-style-memory-management>
3. Mehul Arora, The Emerging Agent Memory Stack：<https://www.mehularora.me/memory-stack.html>
4. Leonie Monigatti 对 MemGPT/Letta 的复盘与示例：<https://www.leoniemonigatti.com/papers/memgpt.html>
5. Neeraj Kumar 关于 MemGPT 虚拟上下文与阈值机制的介绍：<https://neerajku.medium.com/memgpt-extending-llm-context-through-os-inspired-virtual-memory-and-hierarchical-storage-c5cc96f9818a>
