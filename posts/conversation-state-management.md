## 核心结论

对话状态管理，白话说就是：聊天越来越长时，系统要决定“哪些内容继续原样带着走，哪些内容压缩后再带着走，哪些内容只缓存计算结果、不再重复算”。

它解决的是一个硬约束问题，不是文风问题。模型单次请求能看到的上下文长度有限，记为 $L$。系统提示长度记为 $S$，检索补充内容长度记为 $R$，预留给输出的长度记为 $O$，第 $i$ 轮历史消息长度记为 $h_i$。当累计长度持续增长，系统迟早会撞到边界。超过边界的内容，不是“模型看得不够仔细”，而是根本没有进入这一次推理。

最常见的生产级方案不是单一策略，而是三件事同时做：

1. 永远保留系统提示、工具约束、硬规则。
2. 最近若干轮使用滑动窗口保留原文。
3. 更早的历史先压缩成摘要或结构化记忆，再与固定前缀一起复用。

可以把它理解成“有限纸面管理”。纸张大小固定，新增内容不断写入，系统只能做三件事：保留最关键的原文、把旧内容压缩成更短的表示、让稳定前缀复用已算过的中间结果。

| 必要条件 | 作用 | 不满足会怎样 |
| --- | --- | --- |
| 保留固定系统提示 | 维持角色、规则、工具边界 | 模型行为漂移，输出不稳定 |
| 保留最近窗口 | 保留当前任务的局部连续性 | 回答脱离当前轮次 |
| 压缩旧历史 | 给后续轮次留空间 | 对话越长越容易爆窗 |
| 复用固定前缀缓存 | 降低重复计算成本 | TTFT 和输入成本上升 |

一行总结：多轮对话的状态管理，本质是在有限上下文窗口内做“保真、压缩、复用”的联合优化。

---

## 问题定义与边界

先把几个术语讲清楚。

| 术语 | 记号 | 含义 |
| --- | --- | --- |
| 上下文窗口 | $L$ | 模型单次请求最多能处理的 token 总量 |
| 系统提示 | $S$ | 每次请求都要重复发送的固定说明 |
| 历史消息 | $h_i$ | 第 $i$ 轮用户与助手消息占用的 token |
| 检索上下文 | $R$ | 从知识库、工具结果、外部文档补进来的内容 |
| 输出预算 | $O$ | 为本次回答预留的 token 空间 |

如果系统只保留最近 $N$ 轮历史，那么最基本的约束是：

$$
S+\sum_{i=n-N+1}^{n} h_i \le L
$$

这就是最简单的滑动窗口。它的优点是稳定、容易实现、可预测；缺点也很直接：它默认“越早的信息越不重要”，而真实任务里这条假设经常不成立。

看一个最小例子。

- 第 1 轮：`我叫小林`
- 第 2 轮：`我对花生过敏`
- 第 3 轮：`帮我学 Python`
- 第 4 轮：`继续讲列表`
- 第 5 轮：`给我写一段早餐建议`

如果系统只保留最近 3 轮，那么当前窗口里只剩第 3、4、5 轮。第 2 轮里“花生过敏”这个硬约束已经消失。第 5 轮再问早餐建议时，模型完全可能给出花生酱三明治。问题不在模型“没记牢”，而在系统根本没把这条事实发给它。

所以真正的问题不是“保留多少轮”，而是三件事：

1. 哪些信息必须原样保留。
2. 哪些信息可以压缩表示。
3. 哪些前缀只要内容不变，就应该复用已算过的结果。

在工程里，容量边界通常由下面三部分共同决定：

1. 系统提示、工具定义、输出格式约束，本身就会占用大量预算。
2. 用户消息长度波动很大，代码、日志、表格、文档片段都比普通自然语言更耗 token。
3. 输出必须预留空间，不能把窗口全部塞给输入，否则回答会被截断。

因此，更实用的约束应写成：

$$
S+\sum h_i+R+O \le L
$$

这里 $O$ 不能省略。因为“输入塞满窗口”不等于“系统可用”，真正可用的前提是模型还有空间生成足够完整的输出。

所以，“截断与压缩”不是锦上添花，而是任何长对话系统迟早都要实现的容量管理。

---

## 核心机制与推导

常见实现是“滑动窗口 + 摘要压缩 + 固定前缀复用”。

先看滑动窗口。若保留最近 $N$ 轮原文，则当前窗口长度为：

$$
T_{\text{window}}=S+\sum_{i=n-N+1}^{n} h_i
$$

只要 $T_{\text{window}}+O \le L$，系统就还能继续工作。这种方式适合短对话、弱历史依赖任务，或者前面历史几乎没有关键事实的场景。

一旦越界，系统就需要把更早的一段历史 $h_j,\dots,h_k$ 压缩成摘要 $Z$。压缩的基本要求是：

$$
|Z| < \sum_{r=j}^{k} h_r
$$

压缩后总长度变成：

$$
T' = S + Z + \sum_{i=k+1}^{n} h_i
$$

只要满足：

$$
T' + O \le L
$$

就说明这次压缩有效。

但真实工程里，“压缩哪一段”比“怎么压缩”更关键。如果只按时间先后直接删除最老消息，系统会把早期但关键的信息一起删掉。因此通常要给历史片段打重要性分数。最简单的写法是把“时间衰减”和“语义重要度”分开：

$$
I_r = W_r \cdot \exp\left(-\frac{\Delta t_r}{\tau}\right)
$$

其中：

- $\Delta t_r$：距离当前轮次的时间或轮次数差
- $\tau$：衰减尺度，越小表示“旧消息掉得越快”
- $W_r$：语义权重，例如身份信息、禁忌项、已确认决策、当前任务状态可以给更高权重

这样，系统不是简单地“越旧越删”，而是“越旧且越不重要的内容越优先压缩”。

再看一个数值例子。设：

- 上下文上限 $L=8192$
- 系统提示 $S=512$
- 预留输出 $O=1024$
- 最近 3 轮各占 $1600$ token

此时输入部分长度是：

$$
512 + 3\times1600 = 5312
$$

加上输出预算后为：

$$
5312 + 1024 = 6336
$$

还没有超限。

假设接下来新增一轮，用户贴来一段日志和代码，共 $1800$ token；同时系统又要把更早两轮一并带上，那么输入会变成：

$$
512 + 2\times1600 + 3\times1600 + 1800 = 10312
$$

这时总量已经明显超过 $8192$。如果把前两轮合并压缩成一个 400 token 的摘要，则输入改写为：

$$
512 + 400 + 3\times1600 + 1800 = 7512
$$

如果这次回答只需要几百 token 输出，请求就重新回到安全区；如果要预留更长输出，还得继续压缩、减少检索内容，或缩小最近窗口。

下面把关键量放在一起：

| 量 | 公式 | 含义 |
| --- | --- | --- |
| 窗口长度 | $S+\sum_{i=n-N+1}^{n} h_i$ | 当前原样保留的上下文 |
| 压缩后长度 | $S+Z+\sum_{i=k+1}^{n} h_i$ | 用摘要替换旧历史后的长度 |
| 重要性分数 | $I_r=W_r \cdot e^{-\Delta t_r/\tau}$ | 决定哪段历史更适合被压缩 |

这里还要区分两个很容易混淆的概念。

第一，摘要压缩是“内容层”优化。它减少的是本次请求里实际发送的 token 数，也决定哪些信息能继续进入模型视野。

第二，KV Cache 或 Prompt Caching 是“计算层”优化。它减少的是重复前缀的重复编码开销，改善的是首 token 延迟和输入成本，但它不会自动替你保存已经被删掉的内容。

所以：

- 摘要解决“旧内容还能不能留下来”。
- 缓存解决“相同前缀还要不要重新算”。

两者互补，不能互相替代。

---

## 代码实现

下面给一个可直接运行的 Python 示例，演示四件事：

1. 始终保留系统提示。
2. 最近若干轮保留原文。
3. 较早历史超限时压缩成摘要。
4. 把“不能丢的事实”同步写入结构化记忆，避免只靠自然语言摘要。

这不是生产代码，但流程是完整的，复制后可以直接运行。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import json
import re

LIMIT = 120
OUTPUT_BUDGET = 24
KEEP_LAST = 4

SYSTEM_PROMPT = (
    "你是技术助理。必须保留用户硬约束、已确认事实、当前任务目标。"
)

@dataclass
class Turn:
    role: str
    content: str

def count_tokens(text: str) -> int:
    """
    玩具版 token 估算：
    - 连续英文/数字算 1 个单位
    - 每个中文字符算 1 个单位
    """
    units = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text)
    return len(units)

def turns_tokens(turns: List[Turn]) -> int:
    return sum(count_tokens(t.content) for t in turns)

def extract_memory(turns: List[Turn]) -> Dict[str, str]:
    memory: Dict[str, str] = {}
    all_text = "\n".join(t.content for t in turns)

    name_match = re.search(r"我叫([一-龥]{1,8})", all_text)
    if name_match:
        memory["name"] = name_match.group(1)

    if "花生过敏" in all_text:
        memory["allergy"] = "花生过敏"

    if "Python" in all_text and "数据处理" in all_text:
        memory["goal"] = "学习Python用于数据处理"

    return memory

def summarize(turns: List[Turn], memory: Dict[str, str]) -> Turn:
    bullets = []
    if "name" in memory:
        bullets.append(f"用户姓名：{memory['name']}")
    if "allergy" in memory:
        bullets.append(f"饮食约束：{memory['allergy']}")
    if "goal" in memory:
        bullets.append(f"学习目标：{memory['goal']}")

    recent_topics = []
    for t in turns[-4:]:
        snippet = t.content.strip().replace("\n", " ")[:20]
        recent_topics.append(f"{t.role}:{snippet}")

    summary_text = "历史摘要；" + "；".join(bullets + recent_topics)
    return Turn(role="system", content=summary_text)

def total_input_tokens(system_prompt: str, history: List[Turn]) -> int:
    return count_tokens(system_prompt) + turns_tokens(history)

def fits_limit(system_prompt: str, history: List[Turn]) -> bool:
    return total_input_tokens(system_prompt, history) + OUTPUT_BUDGET <= LIMIT

def compress_history(history: List[Turn], keep_last: int = KEEP_LAST) -> List[Turn]:
    if len(history) <= keep_last:
        return history

    old_part = history[:-keep_last]
    recent_part = history[-keep_last:]

    memory = extract_memory(old_part)
    summary_turn = summarize(old_part, memory)
    return [summary_turn] + recent_part

def build_context(system_prompt: str, history: List[Turn]) -> List[Turn]:
    current = history[:]

    while not fits_limit(system_prompt, current):
        compressed = compress_history(current, keep_last=KEEP_LAST)
        if compressed == current:
            # 如果已经无法继续压缩，就继续丢弃摘要后的最旧非系统轮次
            if len(current) <= 1:
                break
            current = current[1:]
        else:
            current = compressed

    return [Turn(role="system", content=system_prompt)] + current

def main() -> None:
    history = [
        Turn("user", "我叫小林，我对花生过敏。"),
        Turn("assistant", "记住了，你对花生过敏。"),
        Turn("user", "我要学 Python，用于数据处理。"),
        Turn("assistant", "可以，先学列表、字典和函数。"),
        Turn("user", "继续讲列表推导式，并结合早餐建议举例。"),
        Turn("assistant", "可以，但早餐示例必须避开你的过敏项。"),
        Turn("user", "顺便给我一个一周学习计划。"),
        Turn("assistant", "可以，先安排基础语法，再安排数据处理练习。"),
    ]

    context = build_context(SYSTEM_PROMPT, history)
    joined = "\n".join(f"{t.role}: {t.content}" for t in context)

    assert context[0].role == "system"
    assert "花生过敏" in joined
    assert total_input_tokens(SYSTEM_PROMPT, context[1:]) + OUTPUT_BUDGET <= LIMIT

    print("=== final context ===")
    print(joined)
    print("input_tokens =", total_input_tokens(SYSTEM_PROMPT, context[1:]))
    print("reserved_output =", OUTPUT_BUDGET)
    print("limit =", LIMIT)

if __name__ == "__main__":
    main()
```

这段代码里，真正值得看的是三个动作：

1. `extract_memory`：把姓名、过敏项、学习目标抽成结构化字段。
2. `summarize`：把旧历史压缩成一条摘要消息。
3. `build_context`：先检查是否超限，超限就压缩；还不够，再继续缩减。

这比“单纯删最老消息”更稳，因为它把不能丢的事实单独抽出来了。对于新手来说，这里最重要的理解不是“摘要写得多漂亮”，而是“哪些信息必须进入下一轮”。

同样的流程，用 JavaScript 可以写成下面这个最小版本：

```javascript
const LIMIT = 120;
const OUTPUT_BUDGET = 24;
const KEEP_LAST = 4;
const systemPrompt = "你是技术助理。必须保留用户硬约束、已确认事实、当前任务目标。";

function countTokens(text) {
  const units = text.match(/[\u4e00-\u9fff]|[A-Za-z0-9_]+/g) || [];
  return units.length;
}

function totalInputTokens(system, history) {
  return countTokens(system) + history.reduce((n, t) => n + countTokens(t.content), 0);
}

function extractMemory(turns) {
  const text = turns.map(t => t.content).join("\n");
  const memory = {};
  const nameMatch = text.match(/我叫([一-龥]{1,8})/);
  if (nameMatch) memory.name = nameMatch[1];
  if (text.includes("花生过敏")) memory.allergy = "花生过敏";
  if (text.includes("Python") && text.includes("数据处理")) {
    memory.goal = "学习Python用于数据处理";
  }
  return memory;
}

function summarize(turns, memory) {
  const parts = [];
  if (memory.name) parts.push(`用户姓名：${memory.name}`);
  if (memory.allergy) parts.push(`饮食约束：${memory.allergy}`);
  if (memory.goal) parts.push(`学习目标：${memory.goal}`);

  const topics = turns.slice(-4).map(t => `${t.role}:${t.content.slice(0, 20)}`);
  return { role: "system", content: `历史摘要；${[...parts, ...topics].join("；")}` };
}

function compressHistory(history) {
  if (history.length <= KEEP_LAST) return history;
  const oldPart = history.slice(0, -KEEP_LAST);
  const recentPart = history.slice(-KEEP_LAST);
  const summary = summarize(oldPart, extractMemory(oldPart));
  return [summary, ...recentPart];
}

function buildContext(system, history) {
  let current = [...history];
  while (totalInputTokens(system, current) + OUTPUT_BUDGET > LIMIT) {
    const compressed = compressHistory(current);
    if (JSON.stringify(compressed) === JSON.stringify(current)) {
      current = current.slice(1);
    } else {
      current = compressed;
    }
  }
  return [{ role: "system", content: system }, ...current];
}
```

生产系统通常还会再补三层能力：

| 能力 | 作用 | 为什么需要 |
| --- | --- | --- |
| 结构化 memory | 保存硬约束、实体状态、已确认决策 | 避免摘要措辞变化导致事实丢失 |
| 分层摘要 | 区分“事实摘要”和“过程摘要” | 既保结论，也保任务进度 |
| 预算分配器 | 动态给历史、检索、输出分配长度 | 防止某一部分独占上下文 |

如果是企业客服，结构化 memory 常见字段是：用户身份、订单号、禁忌项、工单状态、已执行操作。如果是编码代理，字段则常见为：目标文件、失败测试、已确认 bug、当前 patch 状态。

---

## 工程权衡与常见坑

摘要不是免费午餐，缓存也不是必然命中。

第一类风险是摘要误差。摘要本质上是“模型替系统记笔记”。只要笔记写错一次，后面多轮都会沿着这份错误记忆继续推理。最容易出问题的内容有三类：

1. 否定表达，例如“不要改数据库 schema”。
2. 条件表达，例如“仅在测试环境启用”。
3. 时序表达，例如“昨天失败的是 A，今天修复的是 B”。

因此，关键事实不能只放在自然语言摘要里，最好同时写入结构化字段。换句话说，摘要适合保留背景，结构化 memory 适合保留不能错的约束。

第二类风险是“只按时间截断”。早期信息不一定不重要。用户身份、过敏项、法律事实、已确认需求，往往出现在前面几轮，但影响贯穿整个会话。只做 FIFO 截断，系统早晚会删掉这些关键事实。

第三类风险是“只盯输入，不预留输出”。很多系统把输入塞到接近 $L$ 才发送请求，结果模型输出刚开始就触发截断。对话看起来像“模型答不全”，其实是系统根本没留足输出预算。

第四类风险是“前缀不稳定，导致缓存命中率低”。缓存依赖的是精确前缀匹配。下面这些微小变化都可能让缓存收益明显下降：

| 破坏因素 | 例子 | 结果 |
| --- | --- | --- |
| system prompt 每轮重写 | 同义改写、顺序调整 | 前缀变了，缓存失效 |
| tools 顺序变化 | 工具 schema 字段顺序不同 | 相同语义但不是相同前缀 |
| 历史摘要措辞漂移 | 每次总结格式不同 | 复用率下降 |
| 图片/文档块变化 | 同文档不同切片顺序 | 前缀不再相同 |

第五类风险是把 KV Cache 和 Prompt Caching 混为一谈。

- KV Cache 常见于自托管推理。它复用的是模型前缀已经计算出的 key/value 状态。
- Prompt Caching 常见于 API 服务。它复用的是服务端已见过的长前缀计算结果或其等价优化路径。

两者的共同点是都要求“前缀足够稳定”，但约束细节并不相同。

从官方文档看，OpenAI 的 Prompt Caching 目前强调：

- 提示词达到 `1024` token 才有缓存收益。
- 命中依赖精确公共前缀。
- 缓存通常在 `5-10` 分钟不活跃后清理，最长一般不超过 `1` 小时。
- 新文档还提供 `prompt_cache_key`，用于改善共享前缀请求的路由与命中率。

Anthropic 的 Prompt Caching 则更偏显式控制：

- 前缀层次按 `tools -> system -> messages` 组织。
- 默认是 `5m` 的 `ephemeral` 缓存，可显式改成 `1h`。
- 命中前提是缓存断点之前的内容完全一致。
- 并发请求里，通常要等首个请求开始返回后，后续请求才能读到这段缓存。

TensorRT-LLM 的 KV cache reuse 文档则明确指出：跨请求复用要等前一个请求先完成计算并留下可复用状态；高并发下，如果很多请求同时起跑，后发请求未必赶得上复用，缓存块还会因内存压力被逐出。

工程里最常见的坑，可以直接列成检查表：

| 问题 | 表现 | 缓解策略 |
| --- | --- | --- |
| 摘要误解 | 错误信息在多轮中持续扩散 | 关键事实双写到结构化 memory，定期回源校验 |
| 只按时间截断 | 早期约束消失 | 引入重要性分数，优先保留身份、禁忌、已确认决策 |
| 输出预算缺失 | 回答经常半截 | 每次请求固定预留输出空间 |
| 前缀不稳定 | 缓存命中率低 | 固定 system、tool schema、摘要模板 |
| 并发过高 | KV/Prompt cache 复用下降 | 控制扇出，等待首请求写入缓存后再并行 |
| 摘要过短 | 任务进度和依赖链丢失 | 分层摘要：事实摘要和过程摘要分开保存 |

还有两个容易被忽略的设计点。

第一，摘要最好模板化。不要每次让模型自由发挥“写一段总结”，而要固定字段，例如：

| 字段 | 示例 |
| --- | --- |
| 用户硬约束 | 花生过敏、预算上限、不可联网 |
| 已确认事实 | 用户姓名、设备型号、合同签署日期 |
| 当前目标 | 学 Python、修复登录 bug、生成报销单 |
| 当前状态 | 已完成 A，卡在 B，待确认 C |
| 未完成事项 | 等用户上传日志、等工具返回、等审批 |

第二，TTL 要贴近业务节奏，而不是越长越好。若同一前缀在 5 分钟内会反复命中，短 TTL 通常就够；若用户可能十几分钟后回来，或者代理链本身持续较久，1 小时 TTL 才有意义。目标不是“缓存活得越久越好”，而是“命中率、成本、内存占用三者平衡”。

---

## 替代方案与适用边界

不是所有系统都需要“滑动窗口 + 摘要 + 结构化记忆”。

第一种方案是纯滑动窗口。它最简单，系统可预测性最好，适合短对话、一次性问答、轻任务助手。只要历史依赖弱，这就是成本最低的方案。

第二种方案是“固定前缀 + 缓存，不做摘要”。它适合重型 system prompt、长工具定义、固定模板特别多，但历史本身不长的场景，例如企业知识问答模板、代码审查助手、固定 SOP 助手。这里核心问题不是“历史放不下”，而是“相同前缀重复计算太贵”。

第三种方案是“滑动窗口 + 摘要”。这是长流程任务最常见的做法。它适合需要跨多轮保留上下文、但又不值得为每一条旧消息都原样付费的任务，例如售后排障、报销协助、项目协同、编码代理。

第四种方案是“结构化 memory + 少量窗口”。它适合有明确状态机的任务，例如表单填写、案件初筛、客服工单、问诊分诊。系统把关键状态存在固定字段里，窗口只负责最近交互。这比单纯自然语言摘要更稳定。

第五种方案是“外部记忆/RAG + 少量窗口”。这里旧内容不是一直塞在 prompt 里，而是放进数据库、向量库或文档存储，当前轮次再按需检索回来。它适合跨会话、跨天、跨任务的长期记忆，但一致性控制和召回质量会变成新的难点。

可以用一个简化决策表判断：

| 场景 | 推荐策略 | 原因 |
| --- | --- | --- |
| 简短聊天机器人 | 仅最近 2-4 轮滑动窗口 | 实现简单，历史依赖弱 |
| 固定政策很长的企业客服 | 固定前缀 + Prompt Caching | 重复前缀多，缓存收益明显 |
| 长流程任务助手 | 滑动窗口 + 摘要 | 既要连续性，也要控制长度 |
| 法律/医疗初筛 | 摘要 + 结构化关键事实 | 不能只靠最近几轮 |
| 跨会话长期任务 | 外部存储 + 检索 | 不能把全部历史长期塞进 prompt |
| 代码代理/多工具 Agent | 固定前缀 + 摘要 + 结构化状态 | 工具约束重，任务状态复杂 |

看两个对比例子。

玩具例子：一个只负责天气、翻译、笑话的闲聊机器人，完全可以只保留最后 2 轮。因为早期消息对当前回答几乎没有决定性影响，做摘要的收益很低。

真实工程例子：一个法律咨询机器人在第 2 轮确认“合同签署日期”，第 6 轮确认“是否书面通知”，第 11 轮确认“付款是否逾期”，第 15 轮才问“是否构成违约”。如果只保留最近窗口，前面已确认的案件事实很容易消失；如果只做缓存，又只能减少重复计算，不能保证事实继续被带入当前推理。所以这类任务更适合：

- 最近窗口保留原文
- 关键事实写入结构化案件状态
- 旧轮次压缩为事实摘要和过程摘要

结论可以压成一句话：KV cache 解决的是“重复算不算”，摘要和记忆解决的是“内容留不留”，RAG 解决的是“旧内容何时再取回来”。三者解决的问题不同，不能互相替代。

---

## 参考资料

| 出处 | 核心内容 | 适用章节 |
| --- | --- | --- |
| Craig Trim, *The Invisible Boundaries of AI Conversation* | 用直观方式解释上下文窗口、截断和摘要的必要性 | 核心结论、问题定义 |
| OpenAI Docs, *Prompt caching* | `1024` token 起步、精确前缀匹配、`prompt_cache_key`、缓存保留与监控字段 | 工程权衡、代码实现 |
| OpenAI, *Prompt Caching in the API* | 最长公共前缀命中、早期公开机制说明、`cached_tokens` 用法 | 工程权衡 |
| Anthropic Docs, *Prompt caching* | `tools -> system -> messages` 层次、`cache_control`、默认 `5m` 与 `1h` TTL | 工程权衡、替代方案 |
| NVIDIA TensorRT-LLM, *KV cache reuse* | 跨请求复用条件、并发对命中的影响、TTFT 改善边界 | 核心机制、工程权衡 |
| NVIDIA TensorRT-LLM, *KV Cache System* | 优先级 LRU、块复用、部分复用与 offloading 机制 | 工程权衡 |
| Emergent Mind, *Multi-turn RAG Conversations* | 多轮状态维护、压缩与检索协同思路 | 替代方案 |

1. Craig Trim 的文章适合建立一个基础直觉：上下文窗口不是“提示强弱”，而是模型单次推理真正能看到的硬边界。链接：https://medium.com/%40craigtrim/the-invisible-boundaries-of-ai-conversation-702a02ab16e5
2. OpenAI 当前官方文档将 Prompt Caching 写得更细，除了 `1024` token 起步和精确前缀匹配外，还补充了 `prompt_cache_key`、典型清理时间和 `cached_tokens` 监控字段。链接：https://platform.openai.com/docs/guides/prompt-caching
3. OpenAI 2024 年的产品说明文保留了“最长公共前缀”“从 1024 token 起按 128 token 递增命中”的公开描述，适合解释早期机制。链接：https://openai.com/index/api-prompt-caching/
4. Anthropic 的官方文档提供了更显式的缓存控制方式，包括 `cache_control`、默认 `5m` TTL 和可选 `1h` TTL，并明确前缀层次为 `tools -> system -> messages`。链接：https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
5. NVIDIA TensorRT-LLM 的两篇官方文档分别解释了跨请求 KV cache 复用的成立条件，以及缓存块如何在 LRU 和优先级策略下被保留或淘汰。链接：https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html 和 https://nvidia.github.io/TensorRT-LLM/features/kvcache.html
6. 如果系统不是单会话，而是跨轮次、跨文档、跨天任务，RAG 和状态压缩的组合会变得更重要。可参考综述页：https://www.emergentmind.com/topics/multi-turn-rag-conversations
