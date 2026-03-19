## 核心结论

CrewAI 的“记忆”现在更适合理解成一个统一的 `Memory` 系统，而不是三个完全独立、需要你分别接线的模块。你在旧资料里常见的 `short_term_memory`、`long_term_memory`、`entity_memory` 三层划分，到了 2026 年官方文档语境里，已经被统一入口封装了：`memory=True` 或直接传入 `Memory(...)`，Crew 会在任务前检索相关上下文，在任务后把输出拆成离散事实再写回。

对初学者可以用一句白话概括：**Crew 记忆像一本团队共用笔记本，Agent 开工前先翻和当前任务最相关的几页，做完后再把新的结论写回去。**

但要注意一个容易混淆的点：很多文章把 CrewAI 记忆直接讲成“短期=Chroma、长期=SQLite、实体=RAG”。这更接近旧版实现心智模型。**截至 2026-03-19 可查到的官方文档，公开 API 已经收敛为统一 `Memory` 类，默认强调的是统一存储、统一作用域树、统一检索评分，而不是让你直接操作三个低层对象。** 从学习角度，三层划分仍然有用；从工程角度，应该优先按统一 `Memory` 来理解和配置。

下面这张表适合先建立全局图景：

| 记忆层 | 白话解释 | 主要作用 | 典型内容 | 当前 API 视角 |
|---|---|---|---|---|
| short-term memory | 当前流程里的临时上下文 | 保持多步任务连续性 | 上一轮任务结论、刚提取出的事实 | 已被统一 `Memory` 封装 |
| long-term memory | 跨任务、跨会话保留的经验 | 让系统下次还能记得 | 稳定决策、经验规则、历史结论 | 已被统一 `Memory` 封装 |
| entity memory | 针对“人、地点、系统、概念”等实体的记忆 | 让系统记住对象及关系 | “客户 A 用企业版”“服务 B 依赖 Redis” | 已被统一 `Memory` 封装 |

---

## 问题定义与边界

先把问题定义清楚：**CrewAI 的记忆不是“某个 Agent 自己脑子里的一段缓存”，而是一个可被 Crew、Agent、Flow 共同访问的记忆存储与检索系统。**

这里有三个核心边界。

第一，`memory=True` 的作用对象首先是 **Crew 级共享记忆**。也就是说，在默认情况下，同一个 Crew 里的 Agent 读到的是同一套记忆库，而不是每个 Agent 各管各的。官方文档明确说明：当 Crew 使用 `memory=True` 时，如果 Agent 没有单独记忆，所有 Agent 默认共享 Crew 的记忆。

第二，共享不等于完全裸奔。CrewAI 用 **scope** 和 **slice** 管边界。

- `scope`，作用域，白话讲就是“给记忆分文件夹路径”。
- `slice`，切片视图，白话讲就是“把几个文件夹拼成一个只读或读写视图”。

最常见的作用域树长这样：

```text
/
  /company
    /company/knowledge
  /project
    /project/blog
  /agent
    /agent/researcher
    /agent/writer
```

这个设计解决的是两个现实问题：

| 工程问题 | 如果没有 scope/slice 会怎样 | CrewAI 的做法 |
|---|---|---|
| 多 Agent 都在写记忆 | 信息容易串台 | 用 `/agent/...`、`/project/...` 分树存放 |
| 某个 Agent 只该看一部分内容 | 提示词被无关内容污染 | 用 `scope()` 限定单棵子树 |
| 某个 Agent 需要“自己内容 + 公共知识” | 单一子树不够用 | 用 `slice()` 合并多个 scope |
| 共享知识不希望被误写 | 下游 Agent 可能污染公共库 | `read_only=True` |

一个新手版例子：

- researcher 只看 `/agent/researcher`
- writer 既看 `/agent/writer`，又看 `/company/knowledge`
- 公共知识对 writer 只读，避免 writer 把草稿错误写回共享区

这时就能把“共享”理解成**共享底层存储**，而不是“所有人看到完全同样的东西”。

---

## 核心机制与推导

CrewAI 记忆真正关键的不是“存了没有”，而是**怎么取出来**。

### 1. 任务前检索，任务后写回

官方文档给出的工作流很明确：

1. 任务开始前，系统从记忆中 `recall()` 与当前任务最相关的内容。
2. 这些内容会被注入到任务提示词里。
3. 任务完成后，Crew 会自动从输出中抽取离散事实。
4. 这些事实被写回记忆，供后续任务继续使用。

这就是所谓“学习机制”的工程含义。它不是参数微调，也不是模型再训练，而是**把执行结果转成可检索事实**，形成持续累积的外部记忆。

### 2. 评分不是只看语义相似度

CrewAI 官方文档把召回分数定义为三部分加权和：

$$
Score = w_s \cdot similarity + w_r \cdot decay + w_i \cdot importance
$$

默认权重是：

- 语义权重 `semantic_weight = 0.5`
- 时间权重 `recency_weight = 0.3`
- 重要性权重 `importance_weight = 0.2`

其中：

- `similarity`：语义相似度，白话讲就是“这段记忆和当前问题像不像”
- `decay`：时间衰减，白话讲就是“这条记忆离现在多久了”
- `importance`：重要性，白话讲就是“系统判断这条记忆值不值得长期保留”

时间衰减的官方公式是：

$$
decay = 0.5^{age\_days / half\_life\_days}
$$

这表示当记忆年龄等于半衰期时，时间分数会降到 0.5。

### 3. 玩具例子

假设现在有一条候选记忆：

- semantic = 0.9
- recency = 0.6
- importance = 0.8

代入默认权重：

$$
Score = 0.5 \times 0.9 + 0.3 \times 0.6 + 0.2 \times 0.8 = 0.79
$$

这个结果的意义不是“79 分就一定被选中”，而是**它会和其他候选片段一起排序，分数更高的更可能进入 prompt**。

### 4. 真实工程例子

假设你做一个“技术博客生产 Crew”，有两个 Agent：

- researcher：负责查资料、抽事实
- writer：负责组织结构、生成初稿

一次真实流程可能是：

1. researcher 完成“CrewAI 记忆机制调研”，输出 10 条事实
2. Crew 自动把这些事实写入 `/agent/researcher` 或 `/project/blog`
3. writer 开始写作前，系统检索“CrewAI memory、scope、recall、importance”等相关记忆
4. writer 的 prompt 被补上这些事实，所以它不会从零开始“胡猜”
5. writer 产出的术语解释、边界条件、坑点，又会被写回，为后续审稿 Agent 使用

这个机制的价值不在“让每次回答更长”，而在于**让多 Agent 流程有连续上下文，而不是每一步都重新失忆。**

---

## 代码实现

先给一个可运行的玩具实现，用纯 Python 模拟 CrewAI 的复合评分逻辑。

```python
from math import pow

def recency_decay(age_days: float, half_life_days: float) -> float:
    return pow(0.5, age_days / half_life_days)

def composite_score(
    similarity: float,
    age_days: float,
    importance: float,
    semantic_weight: float = 0.5,
    recency_weight: float = 0.3,
    importance_weight: float = 0.2,
    half_life_days: float = 30.0,
) -> float:
    decay = recency_decay(age_days, half_life_days)
    return (
        semantic_weight * similarity
        + recency_weight * decay
        + importance_weight * importance
    )

score = composite_score(similarity=0.9, age_days=22.1, importance=0.8, half_life_days=30)
assert 0.78 < score < 0.80

today_score = composite_score(similarity=0.7, age_days=0, importance=0.6)
old_score = composite_score(similarity=0.7, age_days=60, importance=0.6)
assert today_score > old_score

print(round(score, 3))
```

这个例子说明两件事：

- 相似度高的片段更容易被召回
- 在相似度接近时，更新的记忆会因为时间衰减更有优势

接着看 CrewAI 的典型接法。下面这段代码体现的是**统一 Memory + scope/slice** 的正确心智模型：

```python
from crewai import Agent, Crew, Memory

memory = Memory(
    semantic_weight=0.5,
    recency_weight=0.3,
    importance_weight=0.2,
    recency_half_life_days=30,
)

researcher = Agent(
    role="Researcher",
    goal="Collect accurate facts",
    memory=memory.scope("/agent/researcher"),
)

writer = Agent(
    role="Writer",
    goal="Write a beginner-friendly technical article",
    memory=memory.slice(
        scopes=["/agent/writer", "/company/knowledge", "/project/blog"],
        read_only=True,
    ),
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[...],
    memory=memory,
    embedder={
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    },
)
```

这段代码里有几个关键点。

第一，`memory=memory` 传给 Crew，表示 Crew 使用统一记忆实例。  
第二，researcher 用 `scope("/agent/researcher")`，只能看自己那棵子树。  
第三，writer 用 `slice(...)`，能同时读多个分支，但因为 `read_only=True`，不能把内容误写回公共知识区。

如果你要在可写 slice 里写入记忆，必须显式指定 `scope=`，因为系统需要知道这条信息该落到哪棵子树：

```python
view = memory.slice(
    scopes=["/team/research", "/team/writing"],
    read_only=False,
)

view.remember(
    "最终采用统一 Memory API 讲解，不再把三层记忆当成三个独立入口。",
    scope="/team/writing",
    categories=["decisions"],
)
```

这就是工程上最重要的实践：**读取可以合并视图，写入必须指明归属。**

---

## 工程权衡与常见坑

CrewAI 记忆不是“开了就更聪明”，而是“开了以后开始进入检索系统设计问题”。

### 1. 共享记忆的优点和成本

共享记忆最大的优点是**即时一致性**。即时一致性，白话讲就是“前一个 Agent 刚写完，后一个 Agent 立刻能读到”。这让串行任务链很顺。

但代价也直接：

| 问题 | 表现 | 解决手段 |
|---|---|---|
| 噪声过多 | Prompt 被不相关历史污染 | 设计好 scope/slice |
| 写入太随意 | 记忆树越来越乱 | 关键写入显式指定 `scope` |
| 检索太重 | 任务前召回变慢 | 常规任务用浅层召回，复杂问题再深召回 |
| 公共区被误写 | 团队知识库被草稿污染 | 公共 slice 设为 `read_only=True` |

### 2. 深度召回不是免费午餐

官方文档说明，`recall()` 有两种深度：

- `depth="shallow"`：直接向量检索，快，不走 LLM 分析
- `depth="deep"`：默认模式，会做查询分析、作用域选择、并行搜索、低置信度时递归探索

这意味着一个现实选择：

- 聊天式短任务、流水线步骤，优先 `shallow`
- 总结型、跨项目、范围不清晰的问题，才值得 `deep`

否则你会把检索层自己变成性能瓶颈。

### 3. `memory=True` 带来的隐式依赖

社区里一个非常常见的问题是：任务本身跑完了，但日志里报

`Failed to add to long term memory`

根因通常不是“输出错了”，而是**记忆写回阶段依赖的 LLM、embedder、环境变量、存储路径出了问题**。尤其在 Gemini 相关配置里，社区帖子提到常见情况是同时需要：

- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

因为不同依赖链路读的环境变量名不一样。

排查时可以直接按下面这个清单过：

| 检查项 | 为什么要查 |
|---|---|
| API Key 是否齐全 | 写回和检索可能走不同依赖 |
| `embedder` 是否明确配置 | `memory=True` 会自动带上记忆系统，向量化配置不能含糊 |
| 存储路径是否可写 | 官方文档指出默认存储目录需要可写 |
| 是否误以为只启用了部分记忆 | 社区讨论表明 `memory=True` 会带起整套默认记忆能力 |
| 是否需要先清理旧记忆 | 旧数据或错误状态可能持续影响后续运行 |

官方文档在 2026 版里给出的重置方式是：

```python
crew.reset_memories(command_type="memory")
# 或
memory.reset()
```

这比旧资料里按 `long`、`short` 分开清更符合现在的统一 Memory 模型。

### 4. 作用域设计是第一控制杆，不是“可选优化”

很多人把 scope 当成高级特性，这是错的。  
**在多 Agent 系统里，scope 是控制噪声、权限和可解释性的第一手段。**

经验上有三条足够实用：

- 路径结构保持浅，通常 2 到 3 层够用
- 已知归属的信息尽量显式传 `scope`
- 需要跨分支读取时再用 `slice`，不要一开始就让所有 Agent 读根目录 `/`

---

## 替代方案与适用边界

CrewAI 的统一 Memory 适合的是：**需要多 Agent 协作、希望快速得到共享上下文、又不想自己从零搭状态存储和召回逻辑。**

但它不是唯一方案。

| 方案 | 可见性模型 | 合并策略 | 基础设施复杂度 | 适合什么场景 |
|---|---|---|---|---|
| CrewAI Memory | 统一记忆库 + scope/slice 视图 | 以检索排序和作用域控制为主 | 中等 | 多 Agent 协作、快速原型、共享上下文 |
| LangGraph StateGraph | 共享状态对象 | reducer 聚合状态更新 | 中等偏高 | 需要确定性状态合并、强可追踪工作流 |
| Anthropic Claude Code Memory | 分层文件记忆 | 通过层级覆盖与导入组合 | 低 | 项目规范、团队约定、持久指令管理 |
| Mem0 | 向量记忆 + 可选图记忆 | 语义检索为主，图关系做补充上下文 | 中等 | 长会话用户画像、关系型记忆增强 |
| MemOS | MemCube 统一封装多类记忆 | 以模块化记忆容器组织 | 较高 | 想把文本、图、参数、缓存纳入统一内存体系 |

这里最值得对比的是 CrewAI 和 LangGraph。

- CrewAI 的核心是“统一记忆库 + 自动事实写回 + 召回注入 prompt”
- LangGraph 的核心是“共享状态 + reducer 决定如何合并”

如果你最关心的是**流程确定性、每一步状态如何合并必须可控**，LangGraph 通常更强，因为它把状态更新写成显式程序逻辑。  
如果你最关心的是**多 Agent 先跑起来，并且能自动积累上下文**，CrewAI 的统一 Memory 更省工程量。

再看 Anthropic 的官方记忆文档，它强调的是 `CLAUDE.md` 这类层级化文件记忆，本质更像“持久项目指令”和“团队共享约束”，不是 CrewAI 这种运行时共享检索记忆。所以它适合管理规范，不适合直接替代多 Agent 的任务记忆流水线。

Mem0 和 MemOS 更适合“记忆本身就是产品能力”的场景。比如长期用户画像、跨会话偏好、图关系记忆、专门的 memory service。它们比 CrewAI 更像“把记忆做成单独基础设施”，而不是“附着在 Agent 编排里的一个内建能力”。

---

## 参考资料

| 来源 | 时间 | 主要内容 |
|---|---|---|
| [CrewAI 官方文档：Memory](https://docs.crewai.com/en/concepts/memory) | 2026 | 统一 `Memory` API、`memory=True`、scope/slice、复合评分、浅层/深层召回、重置与排障 |
| [CrewAI 官方文档：Memory（旧概念页）](https://docs.crewai.com/concepts/memory) | 2025 前后仍可访问 | 旧版“三层记忆”心智模型：short-term、long-term、entity、contextual |
| [CrewAI Community: Is long term memory activated by default?](https://community.crewai.com/t/is-long-term-memory-activated-by-default/5321) | 2025-04-14 | 社区对 `memory=True` 默认带起整套记忆能力的讨论 |
| [CrewAI Community: Memory issue when using the Gemini API](https://community.crewai.com/t/memory-issue-when-using-the-gemini-api/3517) | 2025-02-04 至 2025-03-18 | Gemini/EmbedChain 相关 API Key 报错与排查思路 |
| [LangGraph 官方文档：StateGraph](https://langchain-ai.github.io/langgraphjs/reference/classes/langgraph.StateGraph.html) | 2026 | 节点通过共享状态通信，状态键可用 reducer 聚合 |
| [Mem0 官方文档：Graph Memory](https://docs.mem0.ai/platform/features/graph-memory) | 2026 | 图记忆如何在向量检索之外补充实体关系上下文 |
| [MemOS 官方文档：MemCube Overview](https://memos-docs.openmem.net/open_source/modules/mem_cube) | 2026 | MemCube 如何统一封装文本记忆、激活记忆、参数记忆 |
| [Anthropic 官方文档：Manage Claude's memory](https://docs.anthropic.com/en/docs/claude-code/memory) | 2025 | 文件层级的项目/用户/组织记忆模型 |
