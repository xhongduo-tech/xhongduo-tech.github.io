## 核心结论

多智能体协作的本质，不是“很多模型一起聊天”，而是把一个大任务拆成多个自治角色，再用明确协议固定输入、输出、责任、状态和异常处理。这里的“自治角色”可以先白话理解为：它能独立接活、做事、交付结果的一个执行单元。

协作能否成立，取决于一个简单判断：专长带来的质量收益，能不能覆盖额外的沟通、等待和交接成本。如果边界不清、状态不落盘、责任不唯一，多代理不会让系统更聪明，只会让系统更乱。

可以先用一个统一判断框架看问题：

$$
U = Q - 0.1L - 0.2H
$$

其中，$Q$ 是结果质量，$L$ 是总时延，$H$ 是交接次数。这个式子不是物理定律，而是工程上的简化打分函数，意思是：质量越高越好，但等待和交接都要付成本。

一个玩具例子：

- 单体代理独自完成“检索资料 + 写成文章 + 校对格式”，设 $Q=0.78, L=8, H=0$，则  
  $$
  U=0.78-0.8=-0.02
  $$
- 三个代理分工完成同一任务，设 $Q=0.93, L=6, H=1$，则  
  $$
  U=0.93-0.6-0.2=0.13
  $$

这个例子说明，多代理不是天然更优，而是在“少量交接换来明显质量提升”时更优。

下表先给出最重要的对比：

| 方案 | 优点 | 主要成本 | 适合场景 |
|---|---|---|---|
| 单体代理 | 简单、快、状态集中 | 专长不足、上下文混杂 | 小任务、低延迟任务 |
| 多智能体协作 | 可分工、可隔离、可扩展 | 协议设计、交接成本、状态同步 | 可拆分、可验收、跨边界任务 |

---

## 问题定义与边界

多智能体协作，指的是：多个代理围绕同一目标工作，每个代理有明确职责，并通过可检查的协议完成任务交接、状态流转和结果沉淀。这里的“协议”可以先白话理解为：一套提前写清楚的合作规则，而不是临时靠聊天猜对方意思。

它不等于以下几种情况：

| 看起来像协作 | 实际上不算 |
|---|---|
| 两个模型轮流发消息 | 如果没有任务单元、状态机和产物管理，本质还是聊天 |
| 一个代理把同一句话改写三遍 | 这更像采样或重试，不是角色分工 |
| 多个函数串联调用 | 如果函数没有自治决策和协议边界，更像普通流水线 |

边界要先划清。适合多智能体的任务，通常有四个特征：

| 判断维度 | 适合多智能体 | 不适合多智能体 |
|---|---|---|
| 可拆分性 | 可以拆成若干相对独立子任务 | 步骤强耦合，必须频繁来回确认 |
| 验收方式 | 每一步都有明确输入输出 | 很难判断某一步是否完成 |
| 交接频率 | 交接点少，但价值高 | 每一步都要交接 |
| 错误代价 | 需要隔离责任、可追踪回滚 | 出错后只能整体重来 |

例如，“写一篇带引用的技术文章”适合拆成检索代理、写作代理、校验代理。因为每个环节都有清晰产物：资料列表、正文草稿、检查报告。

但“把一句文案润色得更顺口”通常不适合多代理。因为它本身短、低风险、强耦合，拆分以后交接成本大于收益。

真实工程里，边界还包括部署边界。跨团队、跨进程、跨语言的协作，往往需要正式协议；同进程内部的几个紧耦合步骤，通常不值得上远程多代理。

---

## 核心机制与推导

多智能体协作可以抽象成一个任务分配问题。设代理集合为 $a_i$，任务集合为 $t_j$，则一个最小化表达是：

$$
U=\sum_{i,j} x_{ij}(q_{ij}-\lambda c_{ij})-\mu \sum h_{ij}
$$

变量含义如下：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $x_{ij}\in\{0,1\}$ | 任务 $t_j$ 是否分给代理 $a_i$ | 这个活是不是由它来干 |
| $q_{ij}$ | 预期质量 | 它干这件事大概率做得多好 |
| $c_{ij}$ | 执行成本或时延 | 它完成这件事要花多少资源 |
| $h_{ij}$ | 交接成本 | 把任务从一个代理转给另一个代理的损耗 |
| $\lambda$ | 成本权重 | 把时延或资源折算成损失 |
| $\mu$ | 协作权重 | 把交接开销折算成损失 |

这个式子的关键不是数学复杂，而是把一个常见误区说清楚：多代理系统失败，往往不是模型不够强，而是 $h_{ij}$ 太大，或者任务分配 $x_{ij}$ 设计得太碎。

再看一个直观数值推导。

假设要做“资料检索 + 摘要整理 + 事实核验”。

- 单体代理：$Q=0.80, L=7, H=0$  
  $$
  U=0.80-0.7=0.10
  $$
- 双代理：检索代理负责找资料，写作代理负责总结，$Q=0.91, L=6, H=1$  
  $$
  U=0.91-0.6-0.2=0.11
  $$
- 三代理：再加一个核验代理，但交接增至 4 次，$Q=0.95, L=8, H=4$  
  $$
  U=0.95-0.8-0.8=-0.65
  $$

结论很直接：第三个代理虽然让质量继续上升，但额外交接把收益吃掉了。

为了让协作不是“口头约定”，协议层通常要固定几个对象：

| 协议对象 | 作用 | 工程意义 |
|---|---|---|
| `AgentCard` | 声明能力、接口、支持的模式 | 防止错误调用不支持的能力 |
| `Task` | 表示一个工作单元 | 协作不再依赖模糊聊天记录 |
| `contextId` | 维持上下文标识 | 重试、续跑、并发时不串线 |
| `TaskState` | 管理状态流转 | 明确待处理、执行中、完成、失败 |
| `Artifact` | 保存最终结果或中间产物 | 让关键结果可追踪、可重放 |

这里的核心思想是：状态机比聊天记录更可靠。“状态机”可以先白话理解为：一组有限状态和合法跳转规则，比如 `PENDING -> RUNNING -> DONE`。

---

## 代码实现

实现时，最容易犯的错是把多智能体写成“几段 prompt 相互调用”。真正可维护的写法，应该先定义协议数据结构，再写调度逻辑。

下面给出一个可运行的 Python 玩具实现，演示能力检查、任务分配、状态流转和唯一 owner 原则：

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict


class TaskState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class AgentCard:
    name: str
    skills: List[str]


@dataclass
class Artifact:
    task_id: str
    content: str


@dataclass
class Task:
    task_id: str
    skill_required: str
    context_id: str
    owner: str | None = None
    state: TaskState = TaskState.PENDING
    artifact: Artifact | None = None


def assign_task(task: Task, agents: List[AgentCard]) -> str:
    if task.owner is not None:
        raise ValueError("task already has an owner")
    for agent in agents:
        if task.skill_required in agent.skills:
            task.owner = agent.name
            task.state = TaskState.RUNNING
            return agent.name
    raise LookupError("no capable agent found")


def finish_task(task: Task, content: str) -> None:
    if task.owner is None or task.state != TaskState.RUNNING:
        raise ValueError("task is not running")
    task.artifact = Artifact(task_id=task.task_id, content=content)
    task.state = TaskState.DONE


agents = [
    AgentCard(name="researcher", skills=["search", "retrieve"]),
    AgentCard(name="writer", skills=["write", "summarize"]),
    AgentCard(name="checker", skills=["verify"]),
]

task = Task(task_id="t1", skill_required="write", context_id="ctx-article-001")

owner = assign_task(task, agents)
assert owner == "writer"
assert task.state == TaskState.RUNNING
assert task.owner == "writer"

finish_task(task, "Draft finished.")
assert task.state == TaskState.DONE
assert task.artifact is not None
assert task.artifact.content == "Draft finished."
```

这个例子很小，但已经体现了四个工程约束：

| 协议字段/规则 | 工程职责 |
|---|---|
| `skills` | 能力发现，避免盲发任务 |
| `owner` | 单一责任人，防止并发写冲突 |
| `context_id` | 同一业务链路下的任务归组 |
| `artifact` | 结果落盘，而不是只留在消息流中 |

真实工程例子可以看客服系统。假设“客服代理”负责接用户问题，“商品目录代理”负责查询库存和属性。一个典型流程是：

1. 客服代理收到“这个型号还有货吗”。
2. 它识别出这是 `inventory_lookup` 任务。
3. 先读取目录代理的 `AgentCard`，确认支持这个能力。
4. 创建 `Task`，写入 `contextId`，派给目录代理。
5. 目录代理返回库存 `Artifact`。
6. 客服代理再基于该产物组织最终回复。

这个流程的价值不在“用了两个代理”，而在于目录查询职责可以独立部署、独立升级、独立审计。跨团队边界时，这比把所有逻辑塞进一个大代理更稳。

---

## 工程权衡与常见坑

多智能体系统最常见的问题，不是推理能力不够，而是协作纪律不够。

| 常见坑 | 表现 | 规避方法 |
|---|---|---|
| 角色重叠 | 两个代理都改同一份结果 | 每个任务只允许一个 owner |
| 只聊天不落盘 | 重试后上下文丢失 | 关键结果写入 `Artifact` |
| 忽略能力声明 | 调用了对方不支持的模式 | 先读 `AgentCard` 再发任务 |
| 过度切分 | 每一步都要远程交接 | 把低延迟步骤留在本地 |
| 冲突不升级 | 局部协商来回打转 | 设置超时、失败阈值、升级路径 |

其中最容易低估的是“过度切分”。很多系统把“查字段”“改格式”“补一句话”都拆成独立代理，看起来模块化，实际上把 $H$ 拉得很高。系统整体时延上升，故障点变多，日志也更难排查。

还要注意“状态不一致”。例如一个代理认为任务已完成，另一个代理仍认为任务在执行中。这通常来自两类问题：

- 状态机没有统一来源，消息和数据库各写一套。
- 允许多个代理同时更新同一任务。

更稳的做法是让 `TaskState` 有单一真源，并限制状态跳转规则，例如只允许：

$$
PENDING \rightarrow RUNNING \rightarrow DONE
$$

或

$$
PENDING \rightarrow RUNNING \rightarrow FAILED
$$

如果失败，需要清楚定义“重试”与“升级”的区别。重试是相同目标、相同责任、再执行一次；升级是把问题上抬到更高层，重新分配目标或约束。

---

## 替代方案与适用边界

多智能体不是默认答案。实际工程里，至少有三种常见方案：

| 方案 | 推荐场景 | 不足 |
|---|---|---|
| 单体代理 | 小任务、低延迟、上下文集中 | 职责混杂，难做隔离 |
| 本地子代理 | 同进程内可分工任务 | 仍共享故障域 |
| 远程多智能体协议 | 跨团队、跨服务、跨语言协作 | 成本最高，设计要求也最高 |

可以用一个简单决策表来选型：

| 任务条件 | 推荐方案 |
|---|---|
| 任务短、低风险、一次完成 | 单体代理 |
| 可拆分，但步骤强耦合、低延迟要求高 | 本地子代理 |
| 需要正式契约、跨边界调用、独立审计 | 远程多智能体协议 |

再用一句决策树式的话概括：

1. 能否拆成独立可验收的子任务？
2. 如果不能，优先单体代理。
3. 如果能，再问是否跨边界、是否需要正式契约。
4. 如果不跨边界，优先本地子代理。
5. 只有在跨团队、跨服务、跨进程且收益明确时，才值得上远程多代理协议。

所以，“改一个短文案”通常用单体代理就够；“跨多个内部系统查库存、生成风险报告、再做人工审核记录”才是更典型的多智能体协作场景。

---

## 参考资料

下表说明这些来源各自提供什么：

| 来源名称 | 作用 | 对本文贡献 |
|---|---|---|
| A2A Protocol Specification | 协议字段与状态对象定义 | 支撑 `AgentCard`、`Task`、`Artifact` 等协议概念 |
| ADK A2A 文档 | 协议落地方式 | 支撑“何时用远程 A2A、何时用本地子代理”的工程边界 |
| AutoGen 文档 | 多代理会话组织方式 | 支撑多角色协同的实现视角 |
| UMass 技术报告 | 冲突升级与协商层次 | 支撑“局部失败后升级”的机制解释 |

1. [A2A Protocol Specification v1.0.0](https://a2a-protocol.org/dev/specification/)
2. [ADK: Introduction to A2A](https://adk.dev/a2a/intro/)
3. [AutoGen: Agent and Multi-Agent Applications](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/core-concepts/agent-and-multi-agent-application.html)
4. [AutoGen: Multi-agent Conversation Framework](https://microsoft.github.io/autogen/0.2/docs/Use-Cases/agent_chat/)
5. [UMass Technical Report UM-CS-1999-017](https://web.cs.umass.edu/publication/docs/1999/UM-CS-1999-017.pdf)
