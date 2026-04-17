## 核心结论

SGLang 解决的不是“模型本身更聪明”，而是“同一批请求里哪些计算其实不用再做一遍”。它把多次生成、多分支推理、多 Agent 调用写成一种结构化 DSL，即 SGL，白话说就是“先把调用路径写清楚，让系统知道哪些步骤一定共享上下文”。

RadixAttention 是 SGLang 的核心执行机制之一。它把 KV Cache 存进一棵 Radix Tree，也叫压缩前缀树，白话说就是“把很多请求共同开头的 token 合并存储”。新请求到来时，系统先在树里找到最长公共前缀，再只计算没命中的尾部 token。对多轮对话、Agent 模板、Tree-of-Thought 这类“前缀高度相似”的任务，这种复用会直接提高吞吐并降低延迟。

一个最重要的判断标准是前缀共享度。前缀共享度高，RadixAttention 收益大；前缀共享度低，收益会接近普通 KV Cache。它不是对所有请求都神奇加速，而是对“很多请求共享同一段开头”的场景特别有效。

下面先看一个最小对比：

| 场景 | 每个请求总 token 数 | 共享前缀 token 数 | 需要新算的 token 数 | 预期吞吐 |
|---|---:|---:|---:|---|
| 常规多调用 | 12 | 0 | 12 | 基线 |
| RadixAttention 复用 | 12 | 8 | 4 | 明显提升 |
| 多 Agent 共用系统提示 | 200 | 140 | 60 | 提升更明显 |
| Tree-of-Thought 分支搜索 | 400 | 300 | 100 | 常见为高收益场景 |

玩具例子很直观：如果 3 个请求都以“你是一个有帮助的助理，请按 JSON 输出结果”开头，那么这段前缀只需要计算一次，后面各自不同的问题再分别补算。真实工程里，多个 Agent 共享同一 system prompt、工具说明、输出 schema 时，收益通常比这个玩具例子更大，因为共享部分往往更长。

---

## 问题定义与边界

问题的本质是重复预填充。预填充，白话说就是“模型先把整段输入读一遍并建立注意力状态”。在多轮对话、Agent 工作流、批量评测中，大量请求会重复同一段系统提示、角色说明、历史消息或工具描述。如果每次都从头计算，这部分 attention 成本会被反复支付。

最典型的新手版例子是：

- 请求 A：`你是个有帮助的助理。请总结这篇文章。`
- 请求 B：`你是个有帮助的助理。请翻译这段文本。`
- 请求 C：`你是个有帮助的助理。请提取关键词。`

如果服务系统把它们当成 3 个完全独立请求，那么“你是个有帮助的助理”这一段会被算 3 次。问题不在显存里有没有缓存，而在于默认 KV Cache 往往只绑定单个请求生命周期，不能自动跨请求共享。

可以用一个简化示意图表示：

```text
请求A: [系统提示][任务A尾部]
请求B: [系统提示][任务B尾部]
请求C: [系统提示][任务C尾部]

常规处理:
A 全算
B 全算
C 全算

共享前缀处理:
系统提示 算一次并缓存
A 只算任务A尾部
B 只算任务B尾部
C 只算任务C尾部
```

对应的简化公式是：

$$
\text{prefill\_cost}_{\text{normal}} \propto L
$$

$$
\text{prefill\_cost}_{\text{reuse}} \propto L - l_{\text{shared}}
$$

其中，$L$ 是请求总长度，$l_{\text{shared}}$ 是命中的最长共享前缀长度。这个式子表达的就是：共享越长，重算越少。

但这里有明确边界：

| 边界问题 | 结论 |
|---|---|
| 请求前缀几乎都不同 | 收益有限 |
| 只是 decode 阶段慢，而不是 prefill 重 | 收益不一定大 |
| 单轮短 prompt、无模板共享 | 可能不值得引入复杂缓存树 |
| 多轮聊天、Agent 模板、分支推理 | 最适合 |

所以，SGLang 和 RadixAttention 主要优化的是“跨请求重复前缀”，不是所有推理成本。

---

## 核心机制与推导

SGL 是 Structured Generation Language，白话说就是“把生成流程写成程序结构，而不是零散发很多次请求”。它的关键价值不只是语法优雅，而是让运行时知道多个调用之间的依赖关系、共享上下文和分支结构。

例如一个多 Agent 任务可以抽象成：

1. 共享系统提示与工具规范。
2. 派生出 `planner`、`coder`、`critic` 三个分支。
3. 三个分支各自继续生成。
4. 最后再汇总。

从执行器角度看，这不是 4 次毫无关系的调用，而是一棵有共同根节点的生成树。这个“结构”正好对应 RadixAttention 的“前缀树”复用方式。

Radix Tree 是压缩前缀树，白话说就是“把相同开头合并成一条边，而不是每个 token 都单独复制一份”。假设有两个请求：

- Prompt A：`[S1, S2, S3, U1, U2, A1]`
- Prompt B：`[S1, S2, S3, U1, U2, B1]`

它们共享前 5 个 token，于是树结构可以理解为：

```text
root
└── S1 S2 S3 U1 U2
    ├── A1
    └── B1
```

当请求 B 到来时，系统会做三件事：

1. 沿树查找最长匹配前缀。
2. 直接复用该路径上的 KV Cache。
3. 只对剩余未命中的 token 继续 prefill。

因此，请求总长度为 $L$、共享前缀为 $l_{\text{shared}}$ 时，新增计算量近似为：

$$
\Delta L = L - l_{\text{shared}}
$$

若 attention 代价随 token 数增长，则复用后的额外 prefill 成本从“与 $L$ 成正比”下降为“与 $\Delta L$ 成正比”。当 $l_{\text{shared}}$ 接近 $L$ 时，增量成本就很小。

玩具例子：

- A：12 个 token
- B：12 个 token
- 共享前缀：8 个 token

则：

$$
\Delta L = 12 - 8 = 4
$$

也就是说，第二个请求不是重算 12 个 token，而是只补 4 个 token。节省比例近似为：

$$
1 - \frac{4}{12} = 66.7\%
$$

真实工程例子更接近下面这种情况：

- system prompt：80 token
- 工具说明：120 token
- 输出 schema：60 token
- 对话历史：200 token
- 当前用户问题：40 token

如果同一工作流里 10 个 Agent 分支共享前 460 token，只在最后几十个 token 上不同，那么 RadixAttention 复用的不是一句提示词，而是一大段昂贵上下文。这也是它在 Agentic 和 Tree-of-Thought 场景收益明显的根本原因。

简化伪代码如下：

```text
for request in incoming_requests:
    prefix_len = radix_tree.longest_prefix_match(request.tokens)
    reuse_kv(prefix_len)
    compute(request.tokens[prefix_len:])
    radix_tree.insert(request.tokens, new_kv)
```

这里还有两个关键配套机制：

- 路径压缩：把连续单分支节点压成一段，减少树节点数量。
- LRU 驱逐：显存不够时按最近最少使用原则淘汰缓存。
- Cache-aware 调度：优先把共享前缀近的请求排在一起，增加命中率。

如果没有调度配合，树里即使有缓存，也可能因为请求顺序混乱而频繁驱逐最有价值的公共前缀。

---

## 代码实现

下面先给一个“能跑的玩具实现”。它不是 SGLang 源码，而是一个最小化版本，用来说明“插入 token 序列并返回共享前缀长度”的核心逻辑。

```python
class RadixNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class SimpleRadixCache:
    def __init__(self):
        self.root = RadixNode()

    def longest_prefix_and_insert(self, tokens):
        node = self.root
        matched = 0

        for tok in tokens:
            if tok in node.children:
                node = node.children[tok]
                matched += 1
            else:
                break

        # 从未匹配部分开始插入
        for tok in tokens[matched:]:
            new_node = RadixNode()
            node.children[tok] = new_node
            node = new_node

        node.is_end = True
        return matched

cache = SimpleRadixCache()

a = ["你是", "有帮助的", "助理", "请", "总结", "文章"]
b = ["你是", "有帮助的", "助理", "请", "翻译", "文本"]

m1 = cache.longest_prefix_and_insert(a)
m2 = cache.longest_prefix_and_insert(b)

assert m1 == 0
assert m2 == 4
assert len(cache.root.children) == 1
print("ok")
```

这段代码表达的是最小思想：

- 第一次插入 `a`，没有前缀可复用，所以匹配长度是 0。
- 第二次插入 `b`，前 4 个 token 相同，所以直接命中 4。
- 真正需要新增的只是后面的差异部分。

如果把它映射到实际推理系统，接口通常会包含更多字段：

| 字段 | 含义 | 作用 |
|---|---|---|
| `tokens` | token 序列 | 用于前缀匹配 |
| `kv_ptr` | KV 指针 | 指向显存中的 KV 块 |
| `prefix_len` | 共享前缀长度 | 决定从哪里继续算 |
| `ref_count` | 引用计数 | 判断节点是否仍被请求使用 |
| `last_access_ts` | 最近访问时间 | 支持 LRU |
| `children` | 子节点 | 形成前缀树 |

再看一个更贴近工程的伪代码，体现查询、插入和驱逐：

```python
class CacheEntry:
    def __init__(self, kv_ptr=None):
        self.children = {}
        self.kv_ptr = kv_ptr
        self.ref_count = 0
        self.last_access_ts = 0

def serve_request(tokens, now):
    node = ROOT
    prefix_len = 0

    for tok in tokens:
        if tok not in node.children:
            break
        node = node.children[tok]
        prefix_len += 1
        node.last_access_ts = now

    reused_kv = node.kv_ptr if prefix_len > 0 else None
    new_tokens = tokens[prefix_len:]
    new_kv = prefill_from(reused_kv, new_tokens)

    insert_suffix(node, new_tokens, new_kv, now)
    maybe_evict_by_lru()
    return prefix_len, new_kv
```

SGL 层则负责把“哪些请求共享前缀”显式表达出来。可以把它理解成这样的结构：

```text
state = shared_system_prompt
branch_1 = state + user_query_1
branch_2 = state + user_query_2
branch_3 = state + user_query_3
parallel_generate(branch_1, branch_2, branch_3)
```

这里最关键的不是 DSL 长什么样，而是执行器知道 `branch_1/2/3` 来自同一个 `state`。一旦知道这一点，底层 RadixAttention 就能把共享路径对齐到同一棵前缀树上。

真实工程例子可以是“代码修复 Agent”：

- 共享前缀：仓库规则、输出 JSON Schema、修复策略说明、报错栈摘要。
- 分支 1：尝试改依赖版本。
- 分支 2：尝试改配置。
- 分支 3：尝试改源码。

这 3 个分支在业务上不同，在模型输入上却高度重合。SGLang 会把它们组织成结构化分支；RadixAttention 则把重合部分变成可复用 KV。

---

## 工程权衡与常见坑

第一个权衡是复杂度换吞吐。普通服务框架把每个请求独立处理，逻辑简单；RadixAttention 要维护树、引用关系、驱逐策略和调度策略，实现复杂得多。但只要场景里有高共享前缀，这个复杂度通常是值得的。

第二个权衡是“命中率”比“是否开缓存”更重要。很多人以为开了 KV Cache 就一定快，实际上跨请求复用依赖命中率。如果请求顺序把本该连续命中的前缀打散，或者 LRU 把公共祖先节点过早驱逐，吞吐会明显下降。

一个新手容易理解的故事是：

你有 50 个客服机器人，都先读同一份 300 token 的服务规范，再回答各自用户问题。系统刚开始很快，因为前缀被复用了。后来你把调度改成随机混排，又把缓存容量压得很小，于是公共前缀节点不断被挤掉。现象上看就是“模型没变、QPS 却突然掉下去”。根因不是模型退化，而是高价值前缀被驱逐后，系统开始反复重算那 300 token。

常见问题如下：

| 问题 | 原因 | 规避策略 |
|---|---|---|
| 吞吐突然下降 | 高共享前缀被 LRU 驱逐 | 增大缓存预算，保留公共祖先节点 |
| 开了缓存仍不快 | 请求前缀共享度低 | 先做 trace 分析，确认是否真有共享 |
| 多分支任务收益不稳定 | 调度没有按前缀亲缘度聚类 | 做 cache-aware 调度 |
| 显存占用波动大 | 树节点多、生命周期复杂 | 监控节点数、KV 块数、命中率 |
| 命中率高但延迟仍高 | decode 成本主导而非 prefill | 区分 prefill 优化和 decode 优化 |
| 共享不生效 | prompt 模板有细小差异 | 固定系统提示、工具描述、序列化格式 |

工程上应至少监控 4 个指标：

| 指标 | 解释 | 观察目的 |
|---|---|---|
| prefix hit rate | 前缀命中率 | 判断 Radix 树是否发挥作用 |
| avg shared length | 平均共享长度 | 判断复用深度 |
| eviction rate | 驱逐频率 | 判断缓存是否过小 |
| prefill latency | 预填充延迟 | 判断收益是否真的落地 |

一个经验判断是：如果关闭 RadixAttention 或缺少 cache-aware 调度后，吞吐下降很明显，说明系统确实依赖共享前缀复用；如果几乎没变化，通常意味着业务请求本来就不共享前缀，或者模板没有被标准化。

---

## 替代方案与适用边界

替代方案并不是“有没有 KV Cache”，而是“KV Cache 能不能跨请求、跨分支、按前缀共享”。很多系统也有缓存，但只在单请求内部生效，无法像 RadixAttention 那样显式管理共享前缀。

下面是常见对比：

| 方案 | 是否要求高前缀共享度 | 适用场景 | 局限 |
|---|---|---|---|
| 普通逐请求 KV Cache | 低 | 单轮普通推理 | 无法跨请求共享 |
| SGLang + RadixAttention | 高 | 多轮对话、Agent、ToT、模板化批量任务 | 实现与调度复杂 |
| vLLM 等通用高吞吐服务 | 中 | 大多数在线推理 | 对结构化多分支复用不如前缀树显式 |
| 手工 prompt 合批 | 中 | 简单批处理 | 可维护性差，难表达复杂工作流 |

新手可以把它理解成“缓存 vs 每次从头算”的升级版：

- 普通缓存：我记住了同一个请求已经算过什么。
- RadixAttention：我还能发现“不同请求其实共享同一段开头”，所以也能复用。

但它也有清晰的适用边界。以下场景不应高估收益：

1. 用户请求都很短，且彼此完全不同。
2. 主要瓶颈在 decode，而不是 prefill。
3. 模板经常变化，导致共享前缀很难稳定命中。
4. 显存紧张到连公共前缀都无法保留。

相反，以下场景最适合：

1. 多 Agent 共用同一 system prompt 和工具规范。
2. 多轮聊天共享长历史上下文。
3. Tree-of-Thought 或 beam-like 分支搜索。
4. 批量评测、批量抽取、批量结构化生成。

所以，不是“RadixAttention 一定比别的框架快”，而是“在高共享前缀任务上，它的设计更对路”。

---

## 参考资料

1. NeurIPS 2024 论文《SGLang: Efficient Execution of Structured Language Model Programs》。先看论文中的问题定义、SGL 执行模型和 RadixAttention 设计，这是最原始依据。
2. SGLang 官方 README 与官方文档。适合确认工程接口、运行方式和功能边界。
3. 官方或社区的性能对比文章。重点看多轮聊天、Agentic workflow、Tree-of-Thought 场景，而不是只看单轮短 prompt。
4. 学习型解读文章。适合辅助理解“最长公共前缀”“树上复用”“为什么调度会影响命中率”。

建议查阅顺序：

1. 先读论文，确认定义和机制。
2. 再读官方 README，确认系统实现和接口。
3. 再看社区分析，理解实际部署中的吞吐差异。
4. 最后回到自己的业务日志，验证是否真的存在高共享前缀。

如果要进一步理解 RadixAttention 的细节，最值得追的是两条线：一条是“树怎么存 KV”，另一条是“缓存什么时候被驱逐”。前者决定能不能复用，后者决定复用能不能持续。
