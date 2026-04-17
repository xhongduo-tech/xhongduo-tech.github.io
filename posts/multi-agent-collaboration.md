## 核心结论

多 Agent 协作模式，本质上是把一个复杂目标拆成多个可管理的子任务，让多个具备不同专长的代理分别处理，再通过协调机制把结果重新合成。这里的 Agent 可以先理解为“会接任务、会产出结果、会按规则和别人配合的自动执行单元”。

如果目标内部存在明显的前后依赖，顺序模式最稳；如果多个子任务彼此独立，并行模式吞吐最高；如果任务规模大、结构复杂、需要统一治理，层次模式最容易控制质量。工程里通常不是三选一，而是混合使用：先由上层做拆分，再让中层并行执行，最后顺序汇总和校验。

一个新手容易理解的例子是“写一篇博客”。协调者先拆成“调研”“写草稿”“润色”三个子任务。最简单做法是串联：调研 Agent 先产出资料，写作 Agent 根据资料写草稿，润色 Agent 最后改写表达。进一步优化时，“调研”这一步可以再拆成多个并行主题搜索，最后由整合者统一去重、校对和定稿。

| 模式 | 典型结构 | 优势 | 常见瓶颈 | 适合场景 |
|---|---|---|---|---|
| 顺序 | A → B → C | 简单、稳定、易追踪 | 总耗时长，前一步卡住全局卡住 | 强依赖流程 |
| 并行 | A/B/C 同时执行，再归并 | 吞吐高、响应快 | 合并冲突、同步成本高 | 独立子任务 |
| 层次 | Supervisor → Workers → Merger | 适合复杂任务、治理清晰 | 协调者可能成为瓶颈 | 大任务拆解 |

文字小结：多 Agent 的收益不来自“人数变多”，而来自“拆分正确、通信正确、冲突可控”。模式选错，系统只会更复杂，不会更快。

---

## 问题定义与边界

本文讨论的不是“多个模型同时调用”这么简单，而是“多个代理围绕同一目标协作完成任务”的工程模式。这里的协作，至少要回答四个问题：目标怎么拆、依赖怎么表达、结果怎么交换、冲突怎么解决。

问题边界可以先用一个写作场景理解。比如“文献写作”可以拆成“资料检索、资料整理、正文写作、事实校验”。其中“多个主题的资料检索”通常可并行，因为它们互不阻塞；但“正文写作”往往要在核心材料齐全后再开始，因此通常是串行依赖。也就是说，是否能并行，不由工具决定，而由任务本身的依赖结构决定。

通信方式通常有两类。

- 消息传递（message passing，白话说就是“你把结果直接发给下一个人”）
- 共享记忆（shared memory，白话说就是“大家把结果写到同一本公共记录里，再按需读取”）

两者都能工作，但边界不同。消息传递更容易保证顺序和责任归属，适合明确的流程链路；共享记忆更灵活，适合多方反复读写同一上下文，但必须处理版本一致性。

还要明确资源约束。多 Agent 不是无限扩展的，至少会受到三类限制。

| 目标 | 依赖 | 通信方式 | 可并行度 |
|---|---|---|---|
| 写一篇短博客 | 调研后才能起草 | `msg` 为主 | 低到中 |
| 多主题资料调研 | 主题之间独立 | `mem` 或 `msg` | 高 |
| 代码评审流程 | 生成后才能审查 | `msg` + `mem` | 中 |
| 大型研究助手 | 上层拆分、下层执行 | `mem` 为主 | 中到高 |

第一类限制是时间和算力。并行 Agent 越多，理论上越快，但同步、调度、上下文构造也会额外耗时。第二类限制是冲突概率。两个 Agent 同时修改同一段共享草稿，冲突就会增加。第三类限制是自主度。协调者如果把每一步都规定死，系统更稳，但灵活性差；如果 worker 完全自主，吞吐可能更高，但一致性风险会升高。

因此，多 Agent 问题的正确定义不是“怎么多开几个代理”，而是“如何在约束内，让拆分、通信、合并三件事形成闭环”。

---

## 核心机制与推导

顺序、并行、层次三种模式可以用统一的时间模型理解。

顺序流水线中，任务必须一个接一个完成，总耗时近似为：

$$
T_{seq} = \sum_{i=1}^{n} t_i + \sum_{i=1}^{n-1} \delta_i
$$

其中 $t_i$ 是第 $i$ 个 Agent 的执行时间，$\delta_i$ 是第 $i$ 次交接消息的延迟。白话说，前一个人做完，下一个人才能开始，所以时间基本就是一段一段累加。

并行扇出归并中，多个 Agent 同时工作，总耗时近似为：

$$
T_{par} \approx \max_i t_i + t_{sync}
$$

其中 $t_{sync}$ 是归并和对齐开销。白话说，最后要等最慢那个分支回来，再花一点时间合并结果。

层次监督中，上层负责拆分与验收，下层负责执行，总耗时常写成：

$$
T_{hier} = T_{super} + \sum_j T_{worker_j}^{local} + T_{merge}
$$

如果下层 worker 之间也并行，那么中间那部分在实际系统中更接近“分层 max + merge”的组合，而不是简单累加。

玩具例子最容易说明差异。假设三个子任务分别耗时 2 秒、3 秒、1 秒。

- 顺序模式：$T_{seq}=2+3+1=6s$
- 并行模式：若同步开销是 0.5 秒，则 $T_{par}\approx \max(2,3,1)+0.5=3.5s$

这个例子说明，并行不是“把 6 秒变成 2 秒”，因为你还要支付协调和归并成本，但仍然可能明显下降。

可以把三种结构想成下面三种图示。

```text
顺序流水线:
[Coordinator] -> [Researcher] -> [Writer] -> [Editor]

并行扇出归并:
                 -> [Research-A]
[Coordinator] --- -> [Research-B] ---> [Merger]
                 -> [Research-C]

层次监督:
[Supervisor]
   |-- [Lead-1] -> [Worker-1A][Worker-1B]
   |-- [Lead-2] -> [Worker-2A][Worker-2B]
   ---> [Final Merger]
```

仅有时间公式还不够，还必须处理一致性。这里可以定义一个冲突检测函数：

$$
f(drafts) \rightarrow \{0,1\}
$$

当 $f(drafts)=1$ 时，表示多个草稿之间存在冲突，需要再协商。这个冲突可能是事实不一致、术语不统一，也可能是代码改动覆盖同一位置。

通信架构会直接影响这个函数的触发频率。

- 消息队列适合单向责任链，冲突少，但灵活度低。
- 共享记忆适合多人协同编辑，灵活度高，但 stale state 更常见。stale state 可以理解为“你看到的是旧版本，不是最新版本”。

真实工程例子是代码评审 Agent。一个生成器 Agent 先写补丁，安全审查 Agent 检查危险调用，风格审查 Agent 检查命名和结构，最后整合器 Agent 合并意见。如果三个评审 Agent 各自独立读代码，这是并行；如果整合器发现“安全修改”和“风格修改”都改了同一行，就触发冲突函数，再要求相关 Agent 重做或协商。

---

## 代码实现

工程实现里，最常见的角色有三个：协调者、执行者、整合者。

- 协调者：负责拆任务、分配任务、设置约束
- 执行者：负责处理局部任务并产出结果
- 整合者：负责合并结果、做一致性检查、决定是否返工

先看一个最小可运行的 Python 示例。它不依赖外部库，模拟共享记忆、版本号写入和冲突检测。

```python
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Draft:
    agent: str
    content: str
    version: int


class SharedMemory:
    def __init__(self):
        self.store: Dict[str, Draft] = {}
        self.version = 0

    def write(self, key: str, agent: str, content: str) -> Draft:
        self.version += 1
        draft = Draft(agent=agent, content=content, version=self.version)
        self.store[key] = draft
        return draft

    def read(self, key: str) -> Draft | None:
        return self.store.get(key)


def detect_conflict(drafts: List[Draft]) -> bool:
    normalized = {d.content.strip().lower() for d in drafts}
    return len(normalized) > 1


def sequential_blog():
    mem = SharedMemory()
    research = mem.write("research", "researcher", "multi-agent has sequential parallel hierarchical modes")
    draft = mem.write("draft", "writer", f"based on: {research.content}")
    final = mem.write("final", "editor", draft.content + " [polished]")
    return final


def parallel_research():
    mem = SharedMemory()
    d1 = mem.write("topic_a", "agent_a", "code review agent uses generator and critic")
    d2 = mem.write("topic_b", "agent_b", "research assistant uses planner and writer")
    return [d1, d2], mem.version


def merge_drafts(drafts: List[Draft]) -> str:
    if detect_conflict(drafts):
        return "needs renegotiation"
    return "\n".join(d.content for d in drafts)


final = sequential_blog()
drafts, version = parallel_research()

assert final.version == 3
assert version == 2
assert detect_conflict([Draft("a", "same", 1), Draft("b", "same", 2)]) is False
assert detect_conflict([Draft("a", "x", 1), Draft("b", "y", 2)]) is True
assert merge_drafts([Draft("a", "same", 1), Draft("b", "same", 2)]) == "same\nsame"
```

这个例子里，版本号的作用是标记“谁在什么时刻写入了什么结果”。真实系统里，整合者通常会检查“我现在合并的草稿，是否仍然基于最新上下文”，否则就会把旧结果误合并进来。

下面给出三种 dispatch 伪代码。

```text
顺序模式
1. tasks = decompose(goal)
2. for task in tasks:
3.     result = send(task)
4.     context = append(context, result)
5. return finalize(context)
```

```text
并行模式
1. tasks = split_independent(goal)
2. futures = parallel_send(tasks)
3. drafts = collect(futures)
4. if detect_conflict(drafts):
5.     drafts = renegotiate(drafts)
6. return merge(drafts)
```

```text
层次模式
1. subplans = supervisor.decompose(goal)
2. for each lead in subplans:
3.     lead.dispatch_to_workers()
4.     lead_summary = lead.aggregate()
5. final = supervisor.merge(lead_summaries)
6. return final
```

真实工程例子可以看“研究助手 Agent”。上层 Planner 先把目标拆成“检索论文、提取论点、生成提纲、起草正文”；检索可以并行，提纲必须在论点基本稳定后生成，正文写作再顺序展开。这里常见实现是：检索结果进入共享记忆，提取 Agent 按版本读取，写作 Agent 只接受“已冻结版本”的摘要输入，避免边写边变。

如果是消息队列方案，重点是顺序和重试。比如：

- 同一任务链路上的消息要有顺序编号
- worker 失败后要支持重试
- 重试要避免重复消费，通常需要幂等键。幂等可以理解为“同一个请求重复执行，结果仍只算一次”

如果是共享记忆方案，重点是版本控制：

- 写入时增加版本号或时间戳
- 读取时标记读到的版本
- 合并时检查版本是否过期
- 冲突时支持回滚或重新生成

---

## 工程权衡与常见坑

多 Agent 系统的第一类风险是协调者瓶颈。很多团队一开始设计一个超强协调者，让它负责拆分、调度、审计、合并，结果所有请求都堵在它身上。表面看系统是多 Agent，实际上还是单点串行。

写博客场景里，如果只有一个协调者负责“给研究 Agent 发任务、收结果、再发给写作 Agent、再发给润色 Agent”，那只要协调者响应慢，全链路就慢。改进方法通常不是继续强化协调者，而是允许 worker 在局部范围内互相反馈。例如调研 Agent 发现术语定义不一致，可以直接通知整理 Agent 修正，而不必每次都回到总协调者。

第二类风险是共享记忆无版本。两个 Agent 先后读取同一份草稿，一个在旧版本上修改，一个在新版本上修改，最后后写入者覆盖前写入者，造成 silent overwrite。silent overwrite 可以理解为“没有报错，但正确内容被悄悄盖掉”。

第三类风险是模式选择错误。一个典型误区是：凡事都顺序执行，因为实现最简单。结果本来可以并行的文献检索、规则检查、单元测试都被串起来，整体耗时被人为拉长。另一个误区相反：强行并行，把高度耦合的任务切开，最后归并成本比执行成本还高。

| 常见坑 | 表现 | 后果 | 规避策略 |
|---|---|---|---|
| 协调者单点 | 所有决策都集中在一个 Agent | 吞吐下降，单点失效 | 下放局部自治，分层协调 |
| 共享记忆无版本 | 后写覆盖先写 | 内容错乱，难追踪 | 版本号、时间戳、CAS 检查 |
| 错误模式选择 | 可并行任务被串行化 | 延迟增加 | 先做依赖分析 |
| 归并策略过弱 | 多个结果直接拼接 | 事实冲突、术语漂移 | 引入冲突检测和重写 |
| worker 自主度过高 | 各写各的 | 风格不一致，目标漂移 | 明确 schema、提示模板和验收标准 |

还要注意两个常见的隐藏成本。

第一是上下文构造成本。每个 Agent 都需要输入上下文，上下文越大，调用越贵，延迟越高。第二是观测成本。系统越复杂，越需要日志、trace、任务 ID、版本链，否则一旦结果出错，很难定位是拆分错了、通信错了，还是合并错了。

工程上一个实用原则是：先做最小闭环，再逐步加并行。不要一开始就设计“全局自治、多层分发、自动重协商”的复杂系统。先验证任务是否真的可拆，再决定要不要引入更多角色。

---

## 替代方案与适用边界

不是所有任务都需要多 Agent。低复杂度、低依赖、输出结构简单的任务，单 Agent 或纯顺序流程通常更划算。比如“写一个小脚本把 CSV 转成 JSON”，需求清楚、步骤短、冲突少，用一个顺序 Agent 就足够；如果硬拆成“读需求 Agent、写代码 Agent、改代码 Agent、审查 Agent”，开销往往比收益更大。

当任务天然是地图式拆分，也就是“多个块可以独立做完再汇总”，并行扇出归并最有效。典型例子是多主题资料调研、批量代码扫描、测试矩阵执行。

当任务具有明显的组织结构，且需要中间层负责领域治理，层次监督更合适。比如“资料调研 + 实验复现 + 报告写作”，上层需要统一研究问题，中层分别管理实验和写作，下层执行具体子任务。这时候单纯顺序或单层并行都不够稳定。

| 条件 | 推荐模式 | 推荐通信 |
|---|---|---|
| 任务短、依赖强 | 顺序 | `msg` |
| 子任务独立、量大 | 并行 | `msg` 或 `mem` |
| 任务复杂、需要治理 | 层次 | `mem` 为主，辅以 `msg` |
| 一致性要求高 | 顺序或层次 | 带版本的 `mem` |
| 吞吐优先 | 并行 | 队列化 `msg` |

也可以把选择过程理解成一个简单决策树。

1. 任务能否拆成若干彼此独立的块？
2. 如果能，归并成本是否低于并行收益？
3. 如果不能，是否存在明确的上级治理关系？
4. 如果有，选层次；如果没有，选顺序。

通信方案也有替代路线。除了消息传递和共享记忆，还可以用事件驱动。事件驱动可以理解为“谁完成了什么，不直接找下家，而是发一个事件，让订阅者自行响应”。这种方法适合松耦合系统，但调试难度更高。另一种方案是“共享记忆 + 回滚机制”，即先允许并发写入，再用版本冲突检查和回滚补偿修正错误，适合高吞吐但可接受重算的场景。若一致性要求极高，还可以引入锁或协商式提交，但代价是并发能力下降。

因此，适用边界可以概括为一句话：多 Agent 不是默认答案，而是复杂任务的结构化答案。拆分性不足、冲突成本过高、观测能力不够时，少做角色、少做并行，通常更对。

---

## 参考资料

1. 《Multi-Agent Collaboration》，Springer。本文关于顺序、并行、层次三类协作形式，以及任务分解视角，主要对应这一来源的分类框架。链接：`link.springer.com/chapter/10.1007/978-3-032-01402-3_7`
2. AI Agents Guide: Multi-Agent Systems Guide。本文中的时间公式、消息传递与共享记忆两类通信方式、扇出归并结构，主要参考这一手册式说明。链接：`www.aiagentlearn.site/tutorials/multi-agent-systems-guide/`
3. ShShell: Multi-Agent Systems Collaboration。本文中的“代码评审 Agent”“研究助手 Agent”两个真实工程例子，主要对应这一来源给出的实践案例。链接：`www.shshell.com/blog/multi-agent-systems-collaboration`
4. Cognition Commons: Multi-Agent Coordination。本文关于协调者单点、共享记忆 stale、模式误选等工程风险与规避建议，参考了该材料中的协调问题讨论。链接：`cognitioncommons.org/research/multi-agent-coordination`
