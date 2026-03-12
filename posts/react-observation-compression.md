## 核心结论

ReAct 是“先想一步，再调用工具，再读取反馈”的代理循环。这里的 Observation 指工具返回给模型的外部反馈，比如搜索结果、日志、文件内容。真正先把上下文窗口吃满的，通常不是 Thought，而是 Observation。

原始 ReAct 每走一步，都会把 `Thought / Action / Observation` 继续附加到下一轮 prompt 里，因此上下文长度近似线性增长。设第 $t$ 步的上下文长度为 $L_t$，单步新增长度为 $\ell_t$，则有

$$
L_t = L_{t-1} + \ell_t,\quad \ell_t \approx |r_t| + |a_t| + |o_t|
$$

其中 $r_t$ 是推理文本，$a_t$ 是动作描述，$o_t$ 是观察结果。工程里通常有 $|o_t| \gg |r_t| + |a_t|$，所以 Observation 管理决定了 ReAct 能走多远。

结论很直接：

| 策略 | 核心做法 | 适合场景 | 主要风险 |
|---|---|---|---|
| 滑动窗口 | 只保留最近 $N$ 轮原文 | 短任务、低风险状态 | 忘掉早期关键事实 |
| 摘要压缩 | 把旧轮次改写成结构化摘要 | 长任务、需要保留语义 | 摘要丢细节 |
| Observation Masking | 旧 Observation 用占位符替换，保留推理链和最近细节 | 工具输出很长、日志很多 | 占位符设计差会误导模型 |

公开资料给出的信号是一致的。Microsoft 的 MetaReflection 实验里，ReAct 配置常直接设置“最多 6 个 action”；Google ADK 官方文档把 compaction 当成内置能力；ACON 论文则专门在 15+ interaction 的长程任务上评估压缩框架。严格说，这些不是同一个实验里的直接 A/B，但可以得到工程判断：不做压缩时，5 到 6 步后就容易进入窗口、成本或注意力利用率瓶颈；引入智能压缩后，面向 15+ 步任务才开始变得可操作。

玩具例子可以先这样理解：不要把“前六天日记全文”每天都粘到新的一页，而是保留“最近三天原文 + 更早内容的目录和摘要”。模型仍然知道之前发生过什么，但不必重新阅读全文。

---

## 问题定义与边界

问题不是“要不要记忆”，而是“哪些历史必须原文保留，哪些历史只需可追溯”。

设第 $t$ 轮之前的轨迹为 $\tau_{t-1}$。原始 ReAct 可写成：

$$
\tau_{t-1} = (o_{sys}, o_{user}, T_1, T_2, ..., T_{t-1})
$$

其中 $T_i=(r_i,a_i,o_i)$，即第 $i$ 轮的推理、动作、观察三元组。

压缩发生的边界要提前定义，否则系统行为不稳定。至少要明确四个参数：

| 参数 | 含义 | 常见取值思路 |
|---|---|---|
| `trigger_threshold` | 何时触发压缩 | 超过 token 阈值或轮次阈值 |
| `compaction_interval` | 多久压缩一次 | 每 3 到 5 次工具调用 |
| `keep_last_n` | 原文保留最近几轮 | 2 到 5 轮 |
| `overlap_size` | 相邻摘要窗口保留的重叠 | 1 到 2 轮，防止摘要断层 |

压缩前后的长度变化可以写成：

$$
\Delta L = L_t^{raw} - L_t^{compact}
$$

如果旧 Observation 总长为 $O_{old}$，压缩后只保留一个摘要 $S$ 和最近窗口 $W$，那么近似有

$$
L_t^{compact} \approx L_{fixed} + |S| + |W|
$$

只要 $|S| + |W| \ll O_{old}$，压缩就是有效的。

这里有一个很容易混淆的边界：压缩不等于删除。删除是“模型再也看不到”；压缩是“模型仍知道这里发生过一段历史，只是表达形式变了”。这两者对后续推理的影响完全不同。

简单流程图如下：

```text
原始 ReAct Prompt
系统指令
用户任务
T1 原文
T2 原文
T3 原文
...
Tn 原文

压缩后 Prompt
系统指令
用户任务
[结构化摘要：任务意图 / 已做决策 / 待办]
[更早 Observation 已压缩的占位信息]
T(n-2) 原文
T(n-1) 原文
Tn 原文
```

---

## 核心机制与推导

Observation Masking 的思想很朴素：旧观察不再原文展开，但保留“这里有过观察”这个事实。论文《The Complexity Trap》把它形式化为：

$$
\tau'_{t-1} = (o_{sys}, o_{user}, (r_1,a_1,o'_1),...,(r_{t-1},a_{t-1},o'_{t-1}))
$$

其中

$$
o'_i=
\begin{cases}
p_i, & i < t-M \\
o_i, & i \ge t-M
\end{cases}
$$

白话解释：最近 $M$ 个 Observation 保留原文，更早的 Observation 用占位符 $p_i$ 替代，比如“此前 8 行输出已省略”。

这比“只删旧轮次”更稳，因为推理链还在。模型仍能看到：

1. 之前做过什么动作。
2. 这些动作确实返回过结果。
3. 最近若干轮的细节还完整存在。

玩具例子：

假设已经有 6 条 Observation，取 $M=3$。压缩后保留 $o_4,o_5,o_6$ 的原文，把 $o_1,o_2,o_3$ 换成：

```text
[Observation summarized/masked: previous diagnostic outputs omitted]
```

这样模型下一轮依然知道“前面排查过网络、检查过配置、查看过日志”，但只精读最近三次结果。

摘要压缩更进一步。它不是只说“这里省略了”，而是把旧内容改写成结构化语义：

| 字段 | 作用 |
|---|---|
| 任务意图 | 当前到底在完成什么 |
| 关键决策 | 已经排除或确认了哪些路径 |
| 待办项 | 下一轮最该继续检查什么 |

这三段是最低保真集合。少了任务意图，模型容易偏题；少了关键决策，模型会重复试错；少了待办项，模型会失去推进方向。

真实工程例子：一个代码审查代理连续调用 `grep`、`git diff`、`pytest`、`sed`、`cat`。如果每次测试日志和文件全文都塞回上下文，十几轮后 prompt 主要是在重复携带历史输出。合理做法是：

- 最近 2 到 3 次失败日志保留原文。
- 更早日志改成“已验证 X、Y，无需重查”。
- 重要状态单独结构化保留，例如“当前分支已修改 4 个文件，失败集中在 parser 模块”。

这样压缩的不是“事实”，而是“事实的表达方式”。

---

## 代码实现

工程实现通常分成两层：

1. 每轮结束后把 `thought/action/observation` 追加进 `history`。
2. 当历史超过阈值时，调用 `compact(history)`，生成“摘要 + 最近窗口 + 掩码”。

下面先给一个可运行的 Python 玩具实现，它展示窗口保留和旧 Observation 掩码，并用 `assert` 保证行为可验证。

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Turn:
    thought: str
    action: str
    observation: str

def compact_history(history: List[Turn], keep_last_n: int = 3) -> List[Turn]:
    if len(history) <= keep_last_n:
        return history[:]

    compacted = []
    cutoff = len(history) - keep_last_n

    for i, turn in enumerate(history):
        if i < cutoff:
            compacted.append(
                Turn(
                    thought=turn.thought,
                    action=turn.action,
                    observation="[Observation masked: earlier tool output omitted]"
                )
            )
        else:
            compacted.append(turn)
    return compacted

history = [
    Turn("检查网络", "ping db", "ping output ..."),
    Turn("检查路由", "traceroute db", "trace output ..."),
    Turn("检查端口", "ss -lntp", "socket output ..."),
    Turn("检查配置", "cat config", "config content ..."),
    Turn("检查日志", "tail app.log", "error stack ..."),
]

new_history = compact_history(history, keep_last_n=2)

assert len(new_history) == 5
assert new_history[0].observation.startswith("[Observation masked")
assert new_history[1].observation.startswith("[Observation masked")
assert new_history[2].observation.startswith("[Observation masked")
assert new_history[3].observation == "config content ..."
assert new_history[4].observation == "error stack ..."
```

再看更接近 ReActAgent 的 TypeScript 伪实现：

```ts
type Turn = {
  thought: string;
  action: string;
  observation: string;
};

type CompactConfig = {
  triggerThreshold: number;
  keepLastN: number;
  summaryPrefix: string;
};

function shouldCompact(history: Turn[], cfg: CompactConfig): boolean {
  return history.length >= cfg.triggerThreshold;
}

function buildSummary(oldTurns: Turn[]): string {
  const actions = oldTurns.map((t) => t.action).join(" -> ");
  return [
    "任务意图：继续完成当前多步任务",
    `已做决策：${actions}`,
    "待办：优先处理最近一次失败对应的问题"
  ].join("\n");
}

function compact(history: Turn[], cfg: CompactConfig) {
  if (!shouldCompact(history, cfg)) {
    return { summary: "", turns: history };
  }

  const split = Math.max(0, history.length - cfg.keepLastN);
  const oldTurns = history.slice(0, split);
  const recentTurns = history.slice(split);

  const maskedOldTurns = oldTurns.map((t) => ({
    ...t,
    observation: "[Observation masked: earlier output summarized]"
  }));

  return {
    summary: `${cfg.summaryPrefix}\n${buildSummary(oldTurns)}`,
    turns: [...maskedOldTurns, ...recentTurns]
  };
}
```

参数可以这样理解：

| 参数 | 作用 | 设计建议 |
|---|---|---|
| `compaction_interval` | 多少轮后做一次压缩 | 工具调用频繁时设小一点 |
| `keep_last_n` | 保留最近几轮细节 | 让模型能直接利用最新证据 |
| `summary_prefix` | 告诉模型这是一段压缩历史 | 避免模型误判为用户新输入 |
| `overlap_size` | 新旧压缩窗口的重叠 | 防止摘要交界处丢状态 |

如果需要更稳，摘要内容最好固定模板，而不是让模型自由发挥。自由摘要容易“写得像总结”，但缺少后续真正有用的状态字段。

---

## 工程权衡与常见坑

三种常见坑，基本都来自“压得太猛”或“压得太随意”。

| 坑 | 为什么会出问题 | 缓解方式 |
|---|---|---|
| 只用滑动窗口 | 旧信息直接消失，早期约束被忘掉 | 给旧阶段补一段结构化摘要 |
| 摘要过度自由 | LLM 会漏掉当下看似不重要、后面却关键的细节 | 固定摘要模板，强制写任务意图/决策/待办 |
| 掩码过短 | 模型只看到“省略了”，却不知道省略的是哪类信息 | 占位符带上类型，如“earlier test logs omitted” |

为什么摘要至少要包含“任务意图、最近决策、待办”三部分？因为它们分别对应三个推理问题：

1. 我现在到底在解什么问题。
2. 哪些路径已经试过，不要重复。
3. 下一步最有信息增益的动作是什么。

真实工程例子：代码审查代理只保留最近 3 条 review comment，结果忘了更早一轮已经确认“根因在权限检查顺序而不是 SQL 查询”。后续模型又回头审数据库层，导致重复排查。加入一段结构化摘要后：

```text
Earlier review summary:
- 任务意图：定位 403 错误根因
- 已确认：SQL 查询结果正确
- 已决策：问题更可能出在 auth middleware 顺序
- 待办：检查 route guard 与 role resolution
```

系统马上稳定很多。原因不是模型更聪明了，而是上下文的“决策状态”被保留下来了。

另一个现实权衡是成本。摘要压缩本身通常需要额外一次模型调用，Masking 不需要。所以在高频工具调用、长日志场景里，Masking 往往是性价比最高的第一步；摘要更像是在 Masking 之上补语义骨架。

---

## 替代方案与适用边界

压缩不是唯一答案。另一类路线是把历史移出 prompt，只在需要时召回。

| 方案 | 适用边界 | 优点 | 局限 |
|---|---|---|---|
| 滑动窗口 | 任务少于 4 到 6 步 | 实现最简单 | 忘早期状态 |
| 摘要压缩 | 10 到 20 步任务 | 保留任务语义 | 额外模型调用 |
| Observation Masking | 长日志、多工具任务 | 便宜、稳定、实现简单 | 对隐藏细节不可直接回读 |
| 外部记忆/检索 | 超长任务、跨会话任务 | 不占主 prompt | 召回失败会漏关键状态 |

简单切换逻辑可以画成：

```text
任务步数少，工具输出短
-> 只用滑动窗口

任务变长，旧状态仍重要
-> 滑动窗口 + 结构化摘要

工具输出极长，日志占大头
-> Masking + 最近窗口

跨会话、跨天、跨任务复用
-> 外部记忆/检索
```

所以没有“最强单一策略”，只有“当前任务最合适的组合”。

对零基础读者，可以记成一句话：短任务用窗口，长任务加摘要，日志特别长时优先掩码，跨会话记忆再上外部存储。

---

## 参考资料

| 来源 | 核心结论 |
|---|---|
| ReAct 原论文: https://arxiv.org/abs/2210.03629 | ReAct 把推理与行动交替组织成多轮轨迹，是后续上下文膨胀问题的起点 |
| DEV Community, ReAct Pattern: https://dev.to/seahjs/react-pattern-38d1 | 工程上常用“步间摘要 + 滑动窗口”来控制上下文增长 |
| Microsoft MetaReflection PDF: https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/MetaReflection__CR__final.pdf | 论文实验里 ReAct 常设最多 6 个 action，说明长链代理很快碰到步数与上下文瓶颈 |
| Google ADK Context Compaction: https://google.github.io/adk-docs/context/compaction/ | 官方提供 `compaction_interval`、`overlap_size` 和可定制 summarizer，证明压缩已是工程内置能力 |
| OpenReview, The Complexity Trap: https://openreview.net/pdf?id=OHVzruJl5k | 给出 Observation Masking 形式化定义，显示简单掩码可在成本上显著优于原始全量历史，并常与摘要法持平 |
| ACON 论文索引: https://huggingface.co/papers/2510.00615 | 在 15+ interaction 的长程任务上研究上下文压缩，报告 26% 到 54% 的峰值 token 降低 |
| Arun Baby, Context Window Management: https://www.arunbaby.com/ai-agents/0012-context-window-management/ | 总结了摘要、滑窗、长期记忆的工程权衡，适合作为实现层参考 |

文中使用的 Observation Masking 公式可追溯为：

$$
o'_i=
\begin{cases}
p_i, & i < t-M \\
o_i, & i \ge t-M
\end{cases}
$$

它表达的不是“删除过去”，而是“保留轨迹结构，只压缩旧观察的展开长度”。
