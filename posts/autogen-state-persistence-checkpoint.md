## 核心结论

AutoGen 的“断点恢复”本质上不是魔法，而是两层状态的组合保存。

第一层是**消息序列**。消息序列就是“按时间顺序排好的对话记录”，每条消息都带 `type`、`source`、`content`、`created_at` 等字段。旧版 GroupChat 体系里，`GroupChatManager.messages_to_string()` 会把这串消息转成 JSON 字符串，`resume()` 再把它读回来并按顺序重放。

第二层是**团队状态**。团队状态就是“每个 agent 当前脑子里记住了什么、轮到谁发言、已经跑了几轮”。在较新的 AgentChat stable 文档里，这层状态通过 `team.save_state()` / `team.load_state()` 管理，核心结构是 `TeamState`，里面保存 `agent_states`，每个 agent 和 manager 都有自己的 `StateDict`。

工程上要记住一条主线：**保存时收集 `TeamState`，必要时再把关键消息序列单独转成字符串；恢复时先 `load_state()`，再做消息恢复或继续运行。**这样才能把计划、轮次、终止计数、上下文记忆一起接回来，而不是只恢复一半。

一个最小玩具例子是：

```text
[
  {"type":"TextMessage","source":"user","content":{"text":"Hi"}},
  {"type":"TextMessage","source":"assistant","content":{"text":"Hello"}}
]
```

这两条消息转成 JSON 字符串后，重启进程仍可反序列化并依序重放，所以“Hello”之前确实见过“Hi”这一事实不会丢。

---

## 问题定义与边界

先定义问题。**状态持久化**就是“把运行中的内存状态写到外部存储，稍后还能读回来”。对 AutoGen 来说，单 agent 还能勉强只记模型上下文；到了多 agent 长对话，必须同时保存：

| 层级 | 作用 | 典型内容 |
|---|---|---|
| `MessageSequence` | 保存对话轨迹 | 每条消息的类型、来源、内容、时间 |
| `StateDict` | 保存单个 agent 或 manager 的内部状态 | 模型上下文、消息缓冲、当前 speaker、轮次 |
| `TeamState` | 保存整个团队快照 | `agent_states` 映射表 |

可以把它写成：

$$
\text{TeamState} \triangleq \{
  "agent\_states": \{\,agent\_name \mapsto StateDict\,\}
\}
$$

这里的 `StateDict` 可以继续展开为“agent 自己的状态 + manager 的线程信息 + 轮次信息 + 当前发言者信息”。不同团队类型内部字段不完全一样，比如 RoundRobin 会关心 `next_speaker_index`，Selector 会关心 `current_speaker`，更复杂的 orchestrator 还会记录 `plan`、`n_rounds`、`n_stalls`。

边界也要说清楚。

第一，**不要在 team 正在运行时保存**。官方 stable 文档明确提示，运行中 `save_state()` 可能得到不一致状态。原因不复杂：某个 agent 可能刚生成一半消息，manager 的轮次计数已经变了，但其他 agent 还没同步到。

第二，**新旧格式不完全等价**。stable 文档说明从 `v0.4.9` 开始，`TeamState` 用 agent 名称而不是运行时 agent ID 作为 key，并移除了 `team_id`。这让状态更可移植，但也意味着旧 checkpoint 可能需要迁移脚本。

第三，**终止条件会污染恢复入口**。如果最后一条消息里还带着终止字符串，比如 `TERMINATE`，那么 `resume()` 后很可能立刻再次命中终止条件，表现成“刚恢复就结束”。

真实工程例子可以看审批链路。用户提交工单后，`planner` 负责拆任务，`reviewer` 审核风险，`executor` 生成处理建议。中途容器重启，只要在停机前定期保存 `team.save_state()` 和消息串，重启后先 `load_state()` 再恢复，系统就能从“已审批到第几步、谁下一轮发言、当前计划是什么”继续，而不是重新问用户一遍。

---

## 核心机制与推导

AutoGen 的恢复机制可以拆成“结构正确”和“顺序正确”两件事。

**结构正确**依赖序列化 schema。首次出现的术语 **schema** 可以理解为“数据长什么样的固定模板”。消息对象在 `autogen_agentchat.messages` 中定义，核心思路是：消息先 `dump()` 成 JSON 可序列化字典，之后再 `load()` 回对象。因此 `MessageSequence` 本质上就是：

$$
\text{MessageSequence} = [m_1, m_2, \dots, m_n], \quad
m_i \in \text{MessageDict}
$$

其中每个 `MessageDict` 至少要让系统知道“这是什么类型、谁发的、内容是什么、何时创建”。对新手来说，可以把它理解为“每条消息都带身份证”。

下面这个表足够把层级看清：

| 结构 | 关键字段 | 含义 |
|---|---|---|
| `MessageDict` | `type` | 消息类型，如 `TextMessage`、事件类消息 |
| `MessageDict` | `source` | 谁发的消息 |
| `MessageDict` | `content` | 具体文本或结构化内容 |
| `MessageDict` | `created_at` | 创建时间，常用于排序和审计 |
| `StateDict` | `type` / `version` | 状态类型和版本 |
| `StateDict` | `llm_context` / `message_thread` / `current_turn` | agent 或 manager 的内部运行状态 |
| `TeamState` | `agent_states` | 整个团队中每个成员的状态映射 |

**顺序正确**依赖消息重放。这里的“重放”就是“按原顺序重新喂回系统”。如果消息顺序错了，计划推导就会错，因为很多 agent 的内部状态更新并不交换律。用数学写法表示，恢复后的上下文不是只看消息集合 $\{m_i\}$，而是看有序序列 $(m_1,m_2,\dots,m_n)$：

$$
S_n = F(F(\cdots F(S_0, m_1), m_2)\cdots, m_n)
$$

也就是说，最终状态 $S_n$ 不只是“有哪些消息”，而是“这些消息以什么顺序作用到初始状态 $S_0$ 上”。

玩具例子最容易理解。假设原会话只有两条消息：

1. `user: Hi`
2. `assistant: Hello`

若重启后只记得第二条，不记得第一条，那么 assistant 的“Hello”就成了悬空响应。只有把两条消息按原顺序恢复，后续问“你刚才在向谁打招呼”时，系统才有完整因果链。

更真实的工程例子是协同写作。`planner` 先生成提纲，`writer` 写正文，`reviewer` 打回缺陷，`planner` 再修订计划。此时真正关键的不是“最后一句话”，而是“当前 plan 形成的路径”。如果只存最终输出，不存消息序列和 manager 状态，那么恢复后虽然看见成稿片段，却不知道为什么下一轮该由 `planner` 还是 `reviewer` 发言，也不知道 termination 已累计到第几次检查。

因此，`load_state()` 和 `resume()` 的推荐顺序是有原因的：先让每个 agent 和 manager 恢复自己的内部状态，再让消息链路和会话入口对齐。前者解决“谁记得什么”，后者解决“对话从哪接上”。

---

## 代码实现

下面先给一个可运行的玩具实现。它不依赖 AutoGen 包，但把“消息转字符串并恢复”的核心动作完整模拟出来。

```python
import json
from copy import deepcopy

def messages_to_string(messages):
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))

def messages_from_string(payload):
    return json.loads(payload)

def replay(messages):
    state = {"history": [], "last_user_text": None, "last_assistant_text": None}
    for msg in messages:
        state["history"].append((msg["source"], msg["content"]["text"]))
        if msg["source"] == "user":
            state["last_user_text"] = msg["content"]["text"]
        elif msg["source"] == "assistant":
            state["last_assistant_text"] = msg["content"]["text"]
    return state

chat_messages = [
    {"type": "TextMessage", "source": "user", "content": {"text": "Hi"}},
    {"type": "TextMessage", "source": "assistant", "content": {"text": "Hello"}},
]

payload = messages_to_string(chat_messages)
restored_messages = messages_from_string(payload)
restored_state = replay(restored_messages)

assert restored_messages == chat_messages
assert restored_state["last_user_text"] == "Hi"
assert restored_state["last_assistant_text"] == "Hello"
assert restored_state["history"][0] == ("user", "Hi")
```

这段代码说明一件事：**持久化先保证可逆，再谈继续执行。**如果序列化后反序列化都不相等，恢复一定不可靠。

再看贴近 AutoGen 的简化伪代码，新手版只保留主干：

```python
# 伪代码：强调调用顺序
team_state = await team.save_state()

assistant_messages = team_state["agent_states"]["assistant"]["chat_messages"]
token = manager.messages_to_string(assistant_messages)

# ... 把 team_state 和 token 保存到数据库或对象存储 ...

await team.load_state(team_state)
last_agent, last_message = manager.resume(token)
```

工程版则要多做三件事：校验版本、事务写入、处理终止标记。

```python
# 伪代码：工程版
team_state = await team.save_state()

record = {
    "checkpoint_version": "autogen-stable-2026-03",
    "team_state": team_state,
    "message_payload": manager.messages_to_string(previous_messages),
    "saved_at": "2026-03-19T10:00:00Z",
}

# 和业务状态一起原子提交，避免“数据库写了，checkpoint 没写完”
repo.save_checkpoint(record)

loaded = repo.load_checkpoint(task_id)

validate_checkpoint_version(loaded["checkpoint_version"])
await team.load_state(loaded["team_state"])

payload = loaded["message_payload"]
payload = strip_termination_if_needed(payload, marker="TERMINATE")

last_agent, last_message = manager.resume(payload)
```

这里的关键不是 API 名字，而是两条原则。

第一，**`save_state` 保存的是团队内部状态，不等于自动帮你做业务级 checkpoint**。你仍然要自己决定存哪里、何时存、如何和业务主表保持一致。

第二，**消息串与团队状态最好一起保存**。只存 `TeamState`，你可能缺可审计的轨迹；只存消息串，你可能缺 manager 的轮次、speaker 索引或 plan。

---

## 工程权衡与常见坑

先看常见坑和规避方式：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| 运行中直接 `save_state()` | 得到半更新状态 | 等团队停止，或用可控暂停点保存 |
| 只存消息，不存 `TeamState` | 恢复后轮次、speaker、plan 丢失 | 消息串和团队状态一起存 |
| 只存 `TaskResult` | 轨迹不完整，难以继续多 agent 协作 | 长对话优先存完整 `MessageSequence` |
| 新旧版本混用 | `load_state()` 失败或字段不匹配 | 为 checkpoint 标记版本并写迁移脚本 |
| 末尾有终止字符串 | `resume()` 后立刻结束 | 恢复前移除 termination 标记 |
| 异常捕获里立刻保存 | 保存到未落盘的中间状态 | 在外层确认线程静止后再保存 |

这里最容易被忽略的是**一致性边界**。很多人会在 `except` 里写：

```python
except Exception:
    await team.save_state()
```

这很危险。因为异常抛出时，可能某个 agent 已更新内部 `llm_context`，但 manager 还没把这轮消息广播给其他参与者。你以为自己保存了“最新状态”，实际保存的是“不完整状态”。

审批链路里常见的正确做法是：让外层调度器先把 team 停在安全点，再统一做 checkpoint。安全点可以是“本轮 agent 完成响应后”“termination 检查后”“消息已写入审计日志后”。

还有一个现实权衡是**存储成本**。完整 `MessageSequence` 很占空间，尤其是代码生成、工具调用、流式输出多的时候。但这笔成本换来的是高恢复准确性和可审计性。对于需要回放、排障、合规追踪的系统，这通常值得。

---

## 替代方案与适用边界

不是所有场景都必须做完整状态持久化。你可以按恢复精度选方案。

| 方案 | 数据量 | 恢复准确性 | 适用场景 |
|---|---|---|---|
| 完整状态持久化：`TeamState` + `MessageSequence` | 高 | 高 | 多 agent 协作、长任务、可审计系统 |
| Plan-only：只存计划或节点状态 | 低 | 中 | 流程固定、上下文可重算 |
| 只存 `TaskResult` 或最终产物 | 很低 | 低 | 一次性任务、失败后可整轮重跑 |

**Plan-only** 的意思是“只记任务推进到哪一步”，白话讲就是“只记流程节点，不记完整对话”。例如协同写作系统里，只要知道当前提纲版本号、待修订段落 ID、下一个负责人是谁，就能继续推进，那确实可以不保存所有消息。

但代码生成场景通常不适合这么做。因为 reviewer 为什么打回、executor 为什么改成某个实现、planner 为什么调整 plan，这些都依赖完整消息链。只存一个 `plan_id`，恢复后经常出现“知道现在该写什么，但不知道为什么这样写”的问题，最终导致重复调用模型、结果漂移，甚至和前一轮设计相矛盾。

所以适用边界很简单：

- 如果任务允许“从节点重新问一遍”，Plan-only 可以。
- 如果任务要求“上下文连续且可追责”，必须保存完整消息和团队状态。
- 如果系统存在人工审批、工具调用、副作用操作，优先选完整 checkpoint。

---

## 参考资料

- [AutoGen stable: Managing State](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/state.html)  
  说明 `save_state()` / `load_state()` 的官方用法，适合理解新版 AgentChat 团队级 checkpoint。

- [AutoGen stable: `autogen_agentchat.state`](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.state.html)  
  给出 `TeamState`、`RoundRobinManagerState`、`SelectorManagerState` 等状态结构，适合核对字段层级。

- [AutoGen stable: `autogen_agentchat.messages`](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.messages.html)  
  定义消息类型、`dump()` / `load()` 语义，以及消息的 JSON 可序列化形式。

- [AutoGen stable 源码: `_base_group_chat.py`](https://microsoft.github.io/autogen/stable/_modules/autogen_agentchat/teams/_group_chat/_base_group_chat.html)  
  可直接看到 `save_state()`、`load_state()`、运行中保存不一致警告，以及 `v0.4.9` 后状态 key 变化说明。

- [AutoGen 0.2: `agentchat.groupchat`](https://microsoft.github.io/autogen/0.2/docs/reference/agentchat/groupchat)  
  旧版 `GroupChatManager.resume()`、`messages_to_string()`、`messages_from_string()` 的参考文档，适合理解字符串化恢复链路。

- [AutoGen 0.2: Resuming a GroupChat](https://microsoft.github.io/autogen/0.2/docs/topics/groupchat/resuming_groupchat/)  
  解释 `resume()` 如何基于历史消息继续群聊，以及 `remove_termination_string` 的必要性。

- [Leeroopedia: Microsoft AutoGen State Persistence](https://leeroopedia.com/index.php/Principle%3AMicrosoft_Autogen_State_Persistence)  
  对官方机制做了二次整理，便于从工程视角理解 “消息轨迹 + 团队状态” 的双层持久化。
