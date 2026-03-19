## 核心结论

记忆感知的 Prompt 构建，本质是把“本轮对话之外的信息”接到当前一次 LLM 调用里。这里的“记忆”就是持久化保存、可被再次取回的上下文，比如用户偏好、历史任务结果、结构化事实。完整链路通常是：检索、格式化、注入。

三种常见注入位点分别是：

| 注入方式 | 白话解释 | 模型权重 | 固定令牌占用 | 典型额外延迟 | 治理难度 | 适合内容 |
|---|---|---:|---:|---:|---:|---|
| 前置注入到 System Prompt | 先把记忆写进最高优先级说明书 | 高 | 高 | 低 | 高 | 稳定用户画像、长期规则、结构化偏好 |
| 上下文内嵌到对话历史 | 把记忆贴在最近几轮对话旁边 | 中 | 中 | 低 | 中 | 本轮强相关、短期事件、会话摘要 |
| 工具调用返回记忆 | 先回答，再按需去外部查 | 可变 | 低 | 约 50-200ms | 中到高 | 低频但重要、跨度大、需要按需取回的信息 |

如果只看“被模型真正利用的概率”，前置注入通常最高，因为它直接进入系统层指令。你给出的研究摘要中把这一利用率写成约 85%，这可以作为工程上“高质量固定记忆”的经验目标，而不是普适定律。工具调用最灵活，但会引入一次检索或服务调用的等待，常见量级是几十到两百毫秒。

对新手最直观的理解是：

- 前置注入像把用户档案贴在操作手册第一页。
- 上下文内嵌像把便签贴在当前聊天窗口旁边。
- 工具调用像需要时再去档案室翻资料。

真正的关键不是“有没有记忆”，而是“把哪类记忆放到哪里”。放错位置，最常见结果不是完全失效，而是两种更糟的问题：一是白白占用窗口，二是让模型被低相关旧记忆带偏。

---

## 问题定义与边界

问题定义可以压缩成一句话：让 LLM 在同一次回答里，同时利用“最新会话信息”和“跨会话持久记忆”，且不过度消耗上下文窗口。

这里的“上下文窗口”就是模型一次请求里能看到的总令牌数。记忆系统不是无限加料，而是在有限预算里做分配。边界要先说清楚，否则“加记忆”会变成“堆文本”。

一个实用的预算模型是：

$$
B_{\text{total}} = B_{\text{system}} + B_{\text{memory}} + B_{\text{history}} + B_{\text{query}} + B_{\text{answer}}
$$

其中：

- $B_{\text{system}}$：系统提示预算
- $B_{\text{memory}}$：记忆注入预算
- $B_{\text{history}}$：最近对话历史预算
- $B_{\text{query}}$：用户本轮输入预算
- $B_{\text{answer}}$：预留给模型输出的预算

玩具例子：假设一次调用总预算是 4000 tokens，你固定给 System Prompt 500 tokens，再给最近对话 1200，用户输入 300，模型输出预留 1000，那么留给记忆的空间只有：

$$
B_{\text{memory}} = 4000 - 500 - 1200 - 300 - 1000 = 1000
$$

这 1000 还不是“有效知识量”。如果检索精度只有 85%，等价有效记忆大约是：

$$
1000 \times 0.85 = 850
$$

你给的大纲里提到另一个边界例子：System Prompt 固定注入 500 令牌，结构化记忆精度 85%，则有效约 425 令牌。这个数字的意义是，预算上限和检索精度共同决定“提前注入是否值得”。如果你只能塞 500 tokens，但真正相关的只有 200，剩下 300 就在污染上下文。

因此，适合提前注入的记忆通常满足三条：

| 条件 | 含义 | 例子 |
|---|---|---|
| 稳定 | 不会在短时间频繁变化 | 用户常用语言、输出格式偏好 |
| 高复用 | 多轮、多任务都会用到 | 安全约束、企业术语表 |
| 高可信 | 结构清晰，冲突少 | 已验证的用户档案、审批规则 |

不适合提前注入的记忆也有三条：

| 条件 | 风险 | 例子 |
|---|---|---|
| 时效短 | 容易过期误导 | “今天下午三点开会” |
| 相关性不稳定 | 很多轮对话都用不上 | 某次临时购物需求 |
| 内容大且松散 | 吃掉窗口却不聚焦 | 长篇历史摘要、原始日志 |

所以边界不是“System Prompt 最强，所以全放进去”，而是“只有稳定且高复用的记忆，才值得拿固定窗口换高权重”。

---

## 核心机制与推导

记忆系统可以抽象为四步：

1. 检索：从长期记忆库找候选记忆。
2. 格式化：把候选记忆整理成模型容易消费的文本或结构。
3. 注入：把整理后的记忆放到 System、对话历史或工具返回里。
4. 利用：模型在注意力计算中真正引用这些信息。

这里的“注意力”可以先白话理解为：模型在生成下一个词时，会给上下文里不同片段分配不同关注度。注入位置不同，关注度分布通常不同。

为了衡量“记忆有没有真的被用到”，可以定义记忆利用率：

$$
U_{\text{memory}}=\frac{\text{LLM 实际消耗的检索记忆令牌}}{\text{注入的记忆令牌}}
$$

这不是模型 API 直接给出的官方指标，而是工程上常用的评估思想。可以通过归因分析、消融实验、回答引用率或 attention proxy 近似估计。

例如你给出的 OrbitalAI 目标值是 60%。如果某次前置注入了 400 tokens 记忆，模型最终回答中可验证地使用了其中约 240 tokens 对应的信息，那么：

$$
U_{\text{memory}}=\frac{240}{400}=60\%
$$

这个指标帮助我们解释三种注入方式的差异。

第一，前置注入。  
它进入系统层，通常能稳定影响后续生成，因此 $U_{\text{memory}}$ 往往较高。但缺点是它每轮都要带着走，即使当前问题不需要，也会长期占预算。

第二，上下文内嵌。  
它离当前用户问题更近，若主题高度相关，利用率也可以不错；但它受最近轮对话干扰更大，且在长对话里容易被后续内容淹没。

第三，工具调用。  
它的特点不是天然高利用率，而是“只在需要时注入”，所以分母更小，预算更灵活。代价是每次调用都要多一个检索或服务往返，常见增加约 50-200ms。对于文本客服，这个延迟通常能接受；对语音 Agent，这个量级已经会明显影响流畅度。

玩具例子可以更直观。设用户问：“我下周去东京出差，继续按我之前的偏好订酒店。”  
系统有三类记忆：

- 长期偏好：喜欢安静酒店、预算每晚 1200 元内
- 上次任务：曾住过新宿某酒店，评价一般
- 临时上下文：这次还要求靠近客户办公室

如果把长期偏好放入 System Prompt，把“上次住店评价”放进对话历史，把“客户办公室位置”由工具实时查询，那么模型会同时获得三层信息：稳定规则、近期经验、外部最新数据。这就是“分层注入”优于“所有内容塞一个位置”的原因。

真实工程例子是企业客服或企业 Copilot。Google 的 Context Engineering 白皮书把 memory 描述为长期个性化机制，并明确区分了系统指令中的记忆、会话历史中的记忆、以及 memory-as-a-tool。Amazon Bedrock AgentCore 进一步把记忆拆成 extraction、consolidation、reflection 等步骤，说明企业场景真正关心的不只是“取回”，还包括“抽取什么、怎么合并、何时更新”。

因此，机制推导的落点是：

- 注入位点决定注意力优先级。
- 检索精度决定有效知识密度。
- 格式化质量决定模型能否稳定消费这些记忆。
- 是否同步调用决定延迟成本。

---

## 代码实现

实现上不要先写“大而全的记忆系统”，先把接口拆开。最低可用版本至少有三个组件：

- `retriever`：按查询找候选记忆
- `formatter`：把记忆整理成紧凑文本
- `injector`：按配置决定注入到哪里

先看一个可运行的 Python 玩具实现。它不依赖外部库，只演示“检索→格式化→注入”的核心逻辑。

```python
from dataclasses import dataclass
from typing import List


@dataclass
class MemoryRecord:
    text: str
    score: float
    kind: str   # "stable" | "episodic" | "tool"


def retrieve(memories: List[MemoryRecord], min_score: float = 0.6) -> List[MemoryRecord]:
    result = [m for m in memories if m.score >= min_score]
    result.sort(key=lambda x: x.score, reverse=True)
    return result


def format_memories(memories: List[MemoryRecord], max_items: int = 3) -> str:
    selected = memories[:max_items]
    if not selected:
        return ""
    lines = ["[MEMORY]"]
    for i, m in enumerate(selected, start=1):
        lines.append(f"{i}. ({m.kind}) {m.text}")
    return "\n".join(lines)


def inject(base_system: str, history: List[str], memory_payload: str, mode: str):
    if mode == "system":
        system_prompt = f"{base_system}\n\n{memory_payload}" if memory_payload else base_system
        return {"system": system_prompt, "history": history}
    if mode == "context":
        new_history = history[:]
        if memory_payload:
            new_history.insert(0, memory_payload)
        return {"system": base_system, "history": new_history}
    if mode == "tool":
        return {"system": base_system, "history": history, "tool_hint": "call_memory_tool_if_needed"}
    raise ValueError("unsupported mode")


memories = [
    MemoryRecord("用户偏好正式、简洁的回答风格", 0.92, "stable"),
    MemoryRecord("用户东京出差时更偏好新宿区域酒店", 0.81, "episodic"),
    MemoryRecord("用户今天下午 3 点要开评审会", 0.45, "episodic"),
]

retrieved = retrieve(memories, min_score=0.6)
payload = format_memories(retrieved)
result = inject(
    base_system="你是企业差旅助手，优先保证信息准确。",
    history=["用户：帮我订下周东京的酒店"],
    memory_payload=payload,
    mode="system",
)

assert "正式、简洁" in result["system"]
assert "今天下午 3 点" not in result["system"]
assert result["history"] == ["用户：帮我订下周东京的酒店"]
```

这段代码做了三件重要的事：

1. 用 `score` 过滤低相关记忆，避免过度注入。
2. 用 `kind` 区分稳定记忆和事件记忆，后面可以做差异化路由。
3. 把注入位点做成 `mode` 配置，而不是写死在业务里。

如果用 JavaScript/TypeScript 写在线服务，结构通常类似下面这样：

```ts
type InjectionMode = "system" | "context" | "tool";

interface MemoryRecord {
  text: string;
  score: number;
  type: "stable" | "episodic" | "preference";
}

function format(records: MemoryRecord[]): string {
  const top = records
    .filter(r => r.score >= 0.6)
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  if (top.length === 0) return "";

  return [
    "[Relevant Memories]",
    ...top.map((r, i) => `${i + 1}. [${r.type}] ${r.text}`)
  ].join("\n");
}

function buildPrompt(
  baseSystem: string,
  conversation: string[],
  memoryRecords: MemoryRecord[],
  mode: InjectionMode
) {
  const memoryPayload = format(memoryRecords);

  if (mode === "system") {
    const systemPrompt = memoryPayload
      ? `${baseSystem}\n\n${memoryPayload}`
      : baseSystem;
    return { systemPrompt, messages: conversation };
  }

  if (mode === "context") {
    const messages = memoryPayload
      ? [`Memory:\n${memoryPayload}`, ...conversation]
      : conversation;
    return { systemPrompt: baseSystem, messages };
  }

  return {
    systemPrompt: baseSystem,
    messages: conversation,
    tools: [{ name: "retrieve_memory", description: "Fetch long-term user memory on demand" }]
  };
}
```

新手要抓住一个实现原则：先决定“哪些记忆属于稳定层”，再决定“这些稳定层是不是值得固定放进 System Prompt”。不是所有 memory 都应该进最高层。

真实工程例子可以是一个学习助手：

- System 注入：用户偏好中文解释、代码示例优先、回答简洁
- Context 注入：本次课程主题是二叉树、上一轮刚讨论过 DFS
- Tool 注入：当用户问“我上周错过哪道题”时，再去查作业记录库

这样设计后，系统既不会每轮都携带全部作业记录，也不会丢掉稳定偏好。

---

## 工程权衡与常见坑

第一类坑是过度关联。  
也就是模型看到固定记忆后，倾向于把当前任务也往这段记忆上靠。比如用户只是问一个通用 Python 问题，系统却因为长期偏好里有“旅行助手”相关记忆，不断往差旅场景上解释。这类问题在前置注入最常见。

常用应对措施：

| 风险 | 表现 | 应对措施 |
|---|---|---|
| 过度关联 | 什么问题都套到旧记忆上 | 相关性阈值、分类路由、最大注入条数 |
| 记忆老化 | 过期信息持续影响回答 | TTL、时间衰减、最近一次确认时间 |
| 记忆冲突 | 新旧偏好互相打架 | consolidation 合并策略、版本号 |
| 幻觉性补全 | 模型把模糊记忆说成确定事实 | 结构化 schema、来源字段、置信度 |
| 工具延迟 | 每轮都卡一次检索 | 预热、缓存、异步并发、只在需要时调用 |

第二类坑是记忆老化。  
“老化”就是旧信息还在库里，但现实已经变了。比如用户半年前喜欢某框架，现在项目已经迁移。如果你仍然把这段偏好固定注入，模型会持续给出落后建议。解决方法不是简单删除，而是给记忆加时间和来源元数据，例如：`updated_at`、`confirmed_by_user`、`confidence`。

第三类坑是格式松散。  
很多系统检索出来的是一堆原始句子，直接拼进 Prompt。这样做短期可用，但稳定性差，因为模型很难判断哪些是规则、哪些是样例、哪些只是历史事件。更稳的做法是把记忆格式化成统一结构，例如：

- `Profile`: 长期稳定用户画像
- `Current Task Facts`: 本任务临时事实
- `Past Relevant Episodes`: 高相似历史片段

企业场景里，Amazon Bedrock AgentCore 的做法有代表性。它不是简单“存向量、再召回”，而是把语义记忆流程拆成抽取、整合、存储，再在后续推理时把整理后的记忆重新放入系统或推理上下文。这里的“整合”就是把新旧记忆做去重、更新或跳过，避免无限增长和冲突扩散。这类流程特别适合金融、客服、企业助理，因为这些场景要求格式稳定、审计友好、尽量减少幻觉。

第四类坑是把所有检索都放同步热路径。  
热路径就是会直接拖慢用户响应的关键链路。对文本聊天，额外 100ms 可能问题不大；对语音 Agent，50-200ms 的记忆查询就可能把整体交互从“自然”拖成“明显停顿”。Google 的 Context Engineering 资料也强调，memory generation 更适合放在响应后的后台流程，而 retrieval 才是每轮是否阻塞的关键决定。

一个实用原则是：

- 写入记忆尽量异步
- 检索记忆按需同步
- 核心偏好可预加载
- 临时大块信息不要固定前置

---

## 替代方案与适用边界

如果你的场景对格式一致性要求不高，其实不一定要上“前置注入 + 结构化长期记忆 + 工具调用”的全套架构。

第一种替代方案是纯上下文注入。  
做法是把检索到的记忆写到最近几轮消息前后，例如“补充背景：用户更喜欢分步骤解释”。这很像把便签贴在最新对话旁边。优点是实现简单、改动小；缺点是权重不如 System 强，在长对话里更容易被后续内容挤掉。它适合内容运营助手、普通知识问答助手这类中等复杂度场景。

第二种替代方案是纯工具调用。  
可以把它理解为“先聊天，需要时再去图书馆找资料”。优点是节省固定窗口、治理上更灵活；缺点是增加时延，而且模型要先决定“是否调用”，因此可能漏用记忆。它适合动态多变、低频调用、信息体量大的场景，比如查历史工单、查跨项目档案、查用户行为日志。

第三种替代方案是会话摘要替代细粒度记忆。  
也就是不保存很多原子事实，而是每个 session 结束时生成一段摘要，下次只检索摘要。这种方式治理成本低，但精细度差，不适合需要强一致用户画像的系统。

可以用一个简单边界表来选：

| 方案 | 适用场景 | 不适用场景 |
|---|---|---|
| System 前置注入 | 企业助理、强格式客服、长期稳定用户偏好 | 高频变化、信息量巨大、时效极强的记忆 |
| 上下文内嵌 | 普通聊天助手、单任务会话、短中程记忆 | 超长会话、强一致性要求、复杂多任务 |
| 工具调用 | 动态知识、跨系统查询、低频高价值记忆 | 严格低延迟语音交互、每轮都必须用记忆 |
| 会话摘要 | 长对话压缩、低成本维持连续性 | 需要原子级偏好与精确事实更新 |

最终判断标准不是“哪种更先进”，而是两个问题：

1. 这段记忆是否值得占用固定窗口？
2. 这段记忆是否必须在当前轮立即可见？

如果两个答案都是“是”，优先考虑 System 前置注入。  
如果第一个是“否”，第二个是“是”，更适合上下文内嵌。  
如果两个都不是强约束，工具调用通常更经济。

---

## 参考资料

1. Emergent Mind, *Memory-Augmented Prompting*  
   核心贡献：给出 memory-augmented prompting 的总体定义，说明外部记忆如何参与 prompt 构建。  
   https://www.emergentmind.com/topics/memory-augmented-prompting

2. Google/Tool.lu 转载, *Context Engineering: Sessions & Memory*  
   核心贡献：把 memory 放到完整的 context engineering 生命周期中，区分 system instructions、conversation history、memory-as-a-tool 等注入方式。  
   https://tool.lu/index.php/deck/Wx/detail

3. Amazon Bedrock AgentCore, *Built-in strategies* 与 *System prompt for semantic memory strategy*  
   核心贡献：展示企业级语义记忆的 extraction、consolidation、reflection 设计，以及用 system prompt 约束记忆提取和合并。  
   https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-strategies.html  
   https://docs.aws.amazon.com/zh_cn/bedrock-agentcore/latest/devguide/memory-system-prompt.html

4. Amazon Bedrock AgentCore, *User preference memory strategy*  
   核心贡献：展示“用户偏好记忆”如何被结构化为可持续更新的长期记忆。  
   https://docs.aws.amazon.com/zh_cn/bedrock-agentcore/latest/devguide/user-preference-memory-strategy.html

5. PromptBase, *Memory System Blueprint*  
   核心贡献：提供面向预算、上下文分配和延迟权衡的工程化示例，可作为预算估算参考，但不应视为通用基准。  
   https://promptbase.com/prompt/memory-system-blueprint

6. Reddit, *Adding Memory to Voice Agents: 4 Architectural Decisions That Actually Matter*  
   核心贡献：给出语音 Agent 中内联检索的典型额外延迟量级，并讨论预加载、语义搜索和后处理的取舍。  
   https://www.reddit.com/r/AI_Agents/comments/1rffve1/adding_memory_to_voice_agents_4_architectural/

7. OrbitalAI, *Memory Management*  
   核心贡献：提供记忆利用效率这类工程评估视角，可作为设置目标利用率的启发性材料。  
   https://orbitalai.in/Orbitalai-memory-management.html
