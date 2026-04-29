## 核心结论

Paged KV Cache 的多租户隔离，不是把每个租户的 KV cache 完全切成独立孤岛，而是在共享 GPU 显存的前提下，精确定义“哪些缓存可以复用、谁可以复用、什么时候应该回收”。KV cache 可以理解为“模型已经算过的上下文中间结果”，下次遇到同样前缀时直接复用，少做一遍前向计算。

一句话概括这件事：`块级 KV 复用 + 租户盐 + 调度优先级 + 回收/驱逐策略`。

如果只追求“省显存”，系统会倾向于让相同前缀尽量复用；但在多租户场景里，只看“前缀相同”是不够的，因为还会同时出现三类问题：

| 目标/风险 | 含义 | 典型对策 |
| --- | --- | --- |
| 省显存 | 相同前缀尽量复用已有块 | Paged KV Cache、前缀命中 |
| 越权命中 | 不该共享的内容被别的租户复用 | `tenant_salt`、命名空间隔离 |
| 时延侧信道 | 通过响应时延推测别人是否命中过缓存 | 限制共享边界、降低跨租户可见性 |
| 资源抢占 | 某个大租户占满显存，拖慢其他租户 | `priority`、配额、回收策略 |

玩具例子很直观。租户 X 和租户 Y 都发来同一段开头：“请总结下面内容：”。如果系统只看 token 前缀，它会认为两边可以共用同一份缓存；如果系统把“属于哪个租户”也放进缓存键，那么即使文本一样，也不会跨租户命中。这说明多租户隔离的核心不是“内容是否一样”，而是“命中边界是否一样”。

真实工程里，这个问题直接决定共享推理集群能不能稳定运行。一个面向多个企业客户的 LLM 服务，往往同时跑交互式聊天、批处理总结、RAG 检索增强问答。没有隔离时，缓存复用率可能很好看，但延迟抖动、显存争抢、跨租户命中风险也会一起上来。多租户隔离因此不是附属优化，而是服务设计的一部分。

---

## 问题定义与边界

这里说的“隔离”，只讨论 KV cache 命中和调度层面的隔离，不展开三类更底层问题：

1. 不讨论模型权重隔离。
2. 不讨论 GPU 进程级安全或驱动级安全。
3. 只讨论共享显存中的 KV block 如何被命中、复用、调度和驱逐。

要把问题说清楚，必须区分三层对象：

| 对象 | 白话解释 | 作用 |
| --- | --- | --- |
| 物理显存块 | GPU 上真实分配出来的一小块内存 | 存放某些 token 的 K/V 向量 |
| 逻辑缓存条目 | “这一段前缀对应哪些块”的索引记录 | 决定查找和复用 |
| 租户命名空间 | “这些缓存属于谁、谁能看到” | 决定命中边界 |

很多实现出问题，是因为把这三层混成了一层：以为显存块分开了，就等于缓存隔离了；或者以为请求里带了 `tenant_id`，就等于命中已经隔离了。实际上不成立。真正的复用条件应该写成：

$$
\text{可复用} = (\text{前缀相同}) \land (\text{tenant salt 相同}) \land (\text{元信息一致})
$$

这里的“元信息一致”，指的是所有会改变模型行为、进而改变 KV 内容的附加条件。例如 LoRA ID。LoRA 可以理解为“在原模型上挂一个小型增量适配器”，不同 LoRA 会让同样的输入产生不同中间状态，因此不能混用缓存。多模态输入的图像哈希也是同理。

边界可以这样划分：

| 数据类型 | 是否可共享 | 说明 |
| --- | --- | --- |
| 系统提示词、公共模板、公开 FAQ 前缀 | 可全局共享 | 内容公开，风险较低 |
| 同组织知识库前缀 | 仅租户内共享 | 组织内可复用，组织间不可复用 |
| 用户私有对话、敏感文档、内部问答 | 不可跨租户共享 | 应严格限制命中范围 |

玩具例子：同样是前 34 个 token 一样，如果缓存键只由 token 构成，那么租户 A 的块会被租户 B 命中；如果缓存键包含 `tenant_salt`，那它们会被视为两个不同条目。也就是说，隔离不是“看起来有租户概念”，而是“租户信息真正进入命中判定”。

---

## 核心机制与推导

Paged KV Cache 的关键思想，是把原本连续的大段 KV 内存切成固定大小的块。块大小记为 `s`，序列长度记为 `L``，则需要的块数是：

$$
B = \lceil L / s \rceil
$$

内部碎片，也就是“最后一个块没用满但仍然占位”的浪费量，是：

$$
waste = B \cdot s - L,\quad 0 \le waste < s
$$

这两个式子解释了为什么块大小是个工程权衡。块越大，管理更简单，但尾部浪费可能更高；块越小，浪费更少，但哈希、索引、元数据维护成本会上升。

看一个玩具例子。设块大小 `s = 16`，序列长度 `L = 34`，则：

- `B = ceil(34 / 16) = 3`
- 总共占位 `3 * 16 = 48`
- 浪费 `48 - 34 = 14`

对应表如下：

| 参数 | 数值 | 含义 |
| --- | --- | --- |
| `s` | 16 | 每块可容纳 16 个 token |
| `L` | 34 | 当前前缀长度 |
| `B` | 3 | 需要 3 个块 |
| `waste` | 14 | 最后一个块内部浪费 14 个位置 |

多租户隔离的核心不是这个切块本身，而是“块哈希链”里要不要带租户边界信息。一个简化表示是：

$$
h_i = H(h_{i-1} \Vert T_i \Vert E_i)
$$

其中：

- $T_i$ 是第 $i$ 个块对应的 token 序列。
- $E_i$ 是额外元信息。
- $H$ 是哈希函数。
- $\Vert$ 表示拼接。

`E_i` 通常至少应包含这些内容：

| 字段 | 作用 |
| --- | --- |
| `tenant_salt` | 限制缓存只能在允许的租户范围内命中 |
| `lora_id` | 防止不同 LoRA 共用错误缓存 |
| `modal_hash` | 防止不同图像或多模态输入误复用 |
| 模板/消息边界信息 | 防止拼接方式不同却被当成同一前缀 |

为什么要用“链式哈希”，而不是只对当前块做哈希？因为前缀缓存的本质是“直到当前块为止，前面的内容都一致”。如果第 3 块相同，但前 2 块不同，那么它们实际上不是同一前缀，不能复用。链式写法把“前文一致”也编码进去了。

再看一个命中例子。租户 X 和租户 Y 的前 34 个 token 完全一致：

- 若不加 `tenant_salt`，两边前两个完整块的哈希相同，Y 可以命中 X 的缓存。
- 若 `tenant_salt(X) != tenant_salt(Y)`，则从第一个块开始哈希链就不同，后续块也都不同，跨租户命中失效。

这正是“相同内容，按租户决定能不能共享”的数学表达。

除了命中边界，还要讨论调度。调度器负责决定“谁先占用显存、谁先做 prefill、谁可以继续 decode”。`priority` 可以理解为“请求优先级”，通常数值越小，越先处理；`FCFS` 则是“先来先服务”。在共享推理集群里，常见策略是：

1. 默认按 `FCFS` 进队。
2. 若高优请求到来且资源不足，允许抢占低优请求。
3. 被抢占的请求回到 waiting queue，等待下一轮调度。
4. 若显存持续紧张，再配合驱逐策略回收低价值缓存。

真实工程例子：一个公司内部集群同时跑两类流量。

- 在线客服助手：低延迟、高优先级。
- 夜间批量文档总结：高吞吐、低优先级。

合理做法往往不是“所有请求平等共享所有缓存”，而是：

- 公共系统提示词可全局复用。
- 同组织知识库前缀带 `org_salt`，组织内复用。
- 敏感会话再带 `user_salt`，只在用户级别命中。
- 交互请求的 `priority` 更高。
- 批量任务在资源紧张时可被抢占或更快驱逐其缓存。

---

## 代码实现

代码层面最重要的原则是分层：把“缓存键生成”“块分配”“命中查询”“调度策略”拆开，不要把多租户隔离写成零散的 if 判断。否则随着 LoRA、多模态、分级租户策略加入，代码会很快失控。

下面是一个可运行的简化 Python 实现。它不是完整推理框架，但足够展示四个关键点：

1. `tenant_salt` 进入缓存键。
2. 只有完整块可复用。
3. 块池独立于请求对象。
4. 调度优先级独立于缓存命中逻辑。

```python
import hashlib
import math
from dataclasses import dataclass

BLOCK_SIZE = 16

def H(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def chunk(tokens, size):
    return [tokens[i:i + size] for i in range(0, len(tokens), size)]

def cache_key(prev_hash, block_tokens, tenant_salt, lora_id="", modal_hash=""):
    meta = f"{tenant_salt}|{lora_id}|{modal_hash}"
    payload = prev_hash + "|" + " ".join(block_tokens) + "|" + meta
    return H(payload)

@dataclass
class Request:
    tenant_salt: str
    tokens: list[str]
    priority: int = 10
    lora_id: str = ""
    modal_hash: str = ""

class BlockPool:
    def __init__(self):
        self.pool = {}

    def lookup_or_alloc_full_blocks(self, req: Request):
        prev = "ROOT"
        hits = 0
        misses = 0
        keys = []
        for block in chunk(req.tokens, BLOCK_SIZE):
            if len(block) < BLOCK_SIZE:
                break  # 只缓存完整块
            key = cache_key(prev, block, req.tenant_salt, req.lora_id, req.modal_hash)
            if key in self.pool:
                hits += 1
            else:
                self.pool[key] = {"tokens": block}
                misses += 1
            keys.append(key)
            prev = key
        return hits, misses, keys

def blocks_needed(L, s):
    B = math.ceil(L / s)
    waste = B * s - L
    return B, waste

# 玩具例子：34 个 token，块大小 16
B, waste = blocks_needed(34, 16)
assert B == 3
assert waste == 14

# 两个租户内容一样，但 salt 不同，不能跨租户复用
tokens = [f"t{i}" for i in range(34)]
pool = BlockPool()

req_x = Request(tenant_salt="tenant-X", tokens=tokens)
req_y = Request(tenant_salt="tenant-Y", tokens=tokens)
req_x2 = Request(tenant_salt="tenant-X", tokens=tokens)

hits1, misses1, _ = pool.lookup_or_alloc_full_blocks(req_x)
hits2, misses2, _ = pool.lookup_or_alloc_full_blocks(req_y)
hits3, misses3, _ = pool.lookup_or_alloc_full_blocks(req_x2)

assert (hits1, misses1) == (0, 2)   # 首次写入两个完整块
assert (hits2, misses2) == (0, 2)   # 不同租户，不命中
assert (hits3, misses3) == (2, 0)   # 同租户同前缀，可复用

print("all assertions passed")
```

这段代码里有几个实现检查点：

| 检查点 | 为什么重要 |
| --- | --- |
| `tenant_salt` 是否进 hash | 这是命中隔离，不是展示字段 |
| 消息边界是否进 key | 不同模板拼接可能产生相同 token 片段假象 |
| full block 与 tail block 是否一致处理 | 很多系统只缓存完整块 |
| preempt 后状态是否可恢复 | 被抢占请求不能丢失调度状态 |

调度侧可以再写一个极简伪代码：

```python
def maybe_preempt(running_low, incoming_high):
    if incoming_high.priority < running_low.priority:
        running_low.state = "waiting"
        incoming_high.state = "running"
        return "preempt"
    return "keep"
```

这段逻辑说明一件事：优先级影响的是“谁先跑”，不是“谁能命中谁的缓存”。很多系统把这两个概念混写，最终既不好测，也不好审计。

真实工程里，代码通常会拆成四类模块：

| 模块 | 职责 |
| --- | --- |
| block pool | 块分配、引用计数、回收 |
| prefix cache | 哈希链、命中查找、索引结构 |
| scheduler | `FCFS`、`priority`、preemption、waiting queue |
| request metadata | `tenant_salt`、LoRA、模态信息、配额标签 |

---

## 工程权衡与常见坑

多租户隔离最容易犯的错误，不是算法不会写，而是边界定义不完整。

第一类坑：`salt` 只在 API 层传了，但没有真正进入 block hash。这样日志里看起来“系统支持租户”，但缓存命中仍然是跨租户的。判断标准很简单：不是看请求对象里有没有 `tenant_id`，而是看命中键里有没有它。

第二类坑：只隔离显存，不隔离命中。比如给每个租户做了配额限制，却仍然允许共享同一个 prefix cache 索引。这样租户 A 仍可能从租户 B 的历史缓存中受益，隔离仍不成立。

第三类坑：把 `priority` 当成公平机制。优先级只解决排序，不解决硬公平。一个高优租户如果持续注入请求，低优租户仍可能长期饿死。公平通常还要配合 token budget、并发上限、速率限制。

第四类坑：块大小只按平均上下文长度拍脑袋决定。块太大，内部碎片严重；块太小，哈希和块管理开销上升。正确方法是按真实长度分布压测，例如看 P50、P95、P99 的请求长度，而不是只看平均值。

第五类坑：尾块处理不一致。有些实现只缓存完整块，但 chunked prefill 的切分点又和块大小不对齐，结果是命中率忽高忽低，定位很难。

下面用表格汇总：

| 常见坑 | 后果 | 规避方式 |
| --- | --- | --- |
| `salt` 只传 API，不进 hash | 伪隔离，仍会跨租户命中 | 确认 `salt` 进入 block hash |
| 只隔离显存，不隔离缓存命中 | 安全边界失效 | 先做命名空间隔离，再谈共享 |
| 只有 `priority`，没有配额 | 低优租户长期饿死 | 增加 token budget、并发上限、限流 |
| 块过大或过小 | 浪费或管理开销过高 | 结合真实长度分布压测 |
| 尾块处理不一致 | 命中率不稳定 | 让 chunked prefill 与 block size 对齐 |

这里有两个最重要的工程权衡。

第一，块大小 vs 内部碎片。块大时，单块索引更少，管理简单，但 $waste$ 的上界也更大；块小时，空间更细粒度，但需要更多哈希、更多索引、更多调度元数据。

第二，复用率 vs 隔离强度。把 `tenant_salt` 设得很细，比如细到用户级，安全边界最清晰，但复用率会下降；如果按组织级别共享，复用率更高，但风险边界也更宽。没有全局最优，只有按业务风险选择的局部最优。

结论可以压缩成一句话：多租户隔离不是“完全禁止共享”，而是“对共享设边界、设优先级、设回收机制”。

---

## 替代方案与适用边界

不是所有场景都适合做跨租户缓存复用。如果业务本身高度敏感，或者请求上下文普遍很短，复用收益很小，那么复杂的隔离设计未必值得。

常见方案可以并排比较：

| 方案 | 复用率 | 实现复杂度 | 安全性 | 适用场景 |
| --- | --- | --- | --- | --- |
| 全局共享 prefix cache | 高 | 低 | 低 | 公共内容很多、风险低 |
| 按租户独立 cache pool | 低到中 | 中 | 高 | 强隔离、多企业环境 |
| 按组织/用户分级 salt | 中到高 | 高 | 中到高 | 想兼顾复用与隔离 |
| 完全不做 prefix cache | 最低 | 低 | 最高 | 强隐私、短请求、实现简化优先 |

玩具例子：一个内部公共 FAQ 机器人，大量请求都带着相同系统提示词和公开知识库开头，这时做全局共享往往很划算；但如果是金融、医疗、法务系统，租户边界很强，敏感上下文很多，那么宁可牺牲部分复用率，也应限制共享范围，甚至直接禁用跨租户复用。

真实工程里，分级 salt 往往更实用。例如：

- 系统提示词：不加盐，全局共享。
- 组织文档：加 `org_salt`，组织内共享。
- 私人会话：加 `user_salt`，只在用户内命中。
- 特殊敏感任务：直接禁用 prefix cache。

这种做法适合的边界是：

1. 多租户共享推理集群。
2. 交互式低延迟请求和批处理混跑。
3. 既要吞吐和成本，又不能放弃隔离。

不适合的边界也要说清楚：

1. 超强隔离要求，任何侧信道都不可接受。
2. 请求极短，复用收益很低。
3. 旧系统无法可靠把 `salt`、LoRA、模板边界等信息注入缓存键。

如果系统连“谁能命中谁”都无法稳定表达，那么继续做共享优化只会把问题藏得更深。

---

## 参考资料

1. [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) 用于说明 prefix cache 如何按完整块复用，以及命中设计的基本思路。
2. [vLLM: BlockPool / KVCacheBlocks](https://docs.vllm.ai/en/stable/api/vllm/v1/core/block_pool/) 用于说明块池、块管理和 KV block 数据结构。
3. [vLLM: Scheduler Config](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/) 用于说明 `fcfs`、`priority` 等调度配置含义。
4. [vLLM: Scheduler Source Docs](https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/scheduler/) 用于说明 priority preemption、waiting queue 等实际调度行为。
5. [vLLM RFC: Cache Salting for Secure and Flexible Prefix Caching](https://github.com/vllm-project/vllm/issues/16016) 用于说明为什么需要给缓存键引入 salt，以及安全边界设计。
6. [PagedAttention: Efficient Memory Management for LLM Serving with PagedAttention](https://huggingface.co/papers/2309.06180) 用于说明 PagedAttention 和分页式 KV 管理的基本原理。
