## 核心结论

构建代码补全服务，第一约束不是“模型够不够强”，而是“开发者是否愿意等”。代码补全是指：开发者停顿输入时，系统自动给出下一段代码建议。对这类交互，整体响应最好落在 200~300ms，超过 400ms 就会明显打断输入节奏。用户不是先看补全再决定是否继续输入，而是继续打字；所以一旦慢，功能就等于不存在。

这直接推导出系统设计原则：预算先给延迟，再给模型。一个可用系统通常拆成 IDE 插件、上下文组装器、候选生成器、排序与校验器四层。IDE 插件负责捕获光标附近代码和输入事件；上下文是指模型决策所需的代码片段集合；候选生成器通常是 LLM 或规则引擎；排序器负责把“像样但不一定能用”的候选，筛成“能直接接受”的候选。

经验上可以先记住下面这张表：

| 端到端延迟 | 用户感受 | 工程判断 |
|---|---|---|
| `<300ms` | 接近瞬时 | 可以积极触发补全 |
| `300~500ms` | 还能维持流畅 | 需要更强取消和缓存 |
| `>400ms` | 明显破坏节奏 | 用户会继续输入，补全被忽视 |

真正可用的系统，不会在每次按键后都重新“从零开始”请求。它会做三件事：一是节流与取消，避免废弃请求堆积；二是上下文预算控制，避免把整仓库塞给模型；三是长连接复用，避免每次请求都付出完整网络建立成本。GitHub Copilot 一类系统能工作，不是因为“模型神奇”，而是因为把这三件基础工程做到了位。

---

## 问题定义与边界

代码补全服务解决的问题，可以精确定义为：在用户当前编辑位置，基于有限上下文，在极短时间内生成一个或多个高可接受率的候选片段。这里的“高可接受率”不是文学意义上的自然，而是开发者按下 `Tab` 或点击接受的概率高。

边界要先划清。补全服务不是聊天助手，不追求长推理；也不是全仓库重构器，不应动辄分析整个项目。它只回答一个局部问题：此刻光标后面最可能接什么代码。这个定义决定了它的输入和输出都必须短、快、可取消。

一个常见误区是把“模型上下文窗口很大”理解成“应该尽量多塞内容”。这是错的。上下文窗口是模型可容纳的上限，不是你的默认目标。补全场景里，最有价值的通常只有三类内容：

| 上下文部分 | 含义 | 典型作用 |
|---|---|---|
| Prefix | 光标前代码，用户刚刚写过的内容 | 决定当前意图 |
| Suffix | 光标后代码，尚未被覆盖的剩余内容 | 保证补全能接得上 |
| Retrieved Context | 检索出的相关定义、示例、接口说明 | 提供项目知识 |

一个实用的预算分配可以写成：`prefix 50% + suffix 20% + retrieved 30%`。这不是固定公式，但表达了优先级：当前局部代码通常比远处资料更重要。

玩具例子很直观。假设新手在 VS Code 里输入：

```python
user.
```

系统不需要扫描整个仓库。它只要快速拿到当前文件中 `user` 的类型线索、前几行赋值、可能的类定义，必要时再补一段该类所在文件的属性列表，就足够生成 `user.name`、`user.email` 之类候选。若此时还去检索十几个无关模块，延迟上升，命中率反而下降。

再看一个稍复杂但仍典型的例子：

```python
fetchData(
```

系统应优先收集当前函数、光标前几行、同模块内 `fetchData` 的签名、最近访问的服务定义，而不是把整个 `api/` 目录拼进去。因为补全的目标，多半只是建议参数名、回调形式或返回值处理，而不是理解系统全貌。

---

## 核心机制与推导

补全候选为什么能排序？底层依据通常是生成概率。生成概率可以白话理解为：模型觉得这段代码在当前上下文下有多“顺”。如果候选由 token 序列 $c_1, c_2, ..., c_T$ 构成，那么其对数概率可以写成：

$$
\log P(c)=\sum_{t=1}^{T}\log P(c_t \mid \text{context}, c_{1:t-1})
$$

这表示整段候选的得分，等于每一步生成该 token 的条件概率对数之和。工程上常进一步取平均，避免长候选天然吃亏：

$$
\text{avg\_score}(c)=\frac{1}{T}\sum_{t=1}^{T}\log P(c_t \mid \text{context}, c_{1:t-1})
$$

这里的“对数概率”可以理解成一种更稳定的分数表示法，便于累加与比较。

但只看概率还不够，因为“看起来顺”不等于“在当前项目里可用”。所以排序通常是“两阶段”：

1. 用模型概率做第一轮粗排，筛掉不自然的候选。
2. 用作用域、类型、语法约束做第二轮校验，筛掉虽然自然但不可执行的候选。

举一个玩具例子。当前上下文里已有：

```python
import math
variance = 9
data = [1, 2, 3]
```

模型给出两个候选：

1. `std = math.sqrt(variance)`
2. `std = compute_std(data)`

如果只看语言自然度，两者都像样；但当前作用域中 `math` 和 `variance` 已定义，`compute_std` 没定义。那么第一个候选更应排前。也就是说，平均对数概率加上标识符合法性检查，往往已经足够构成一个基础可用的排序器。

延迟也能做公式化拆解。设总延迟为：

$$
L = L_{debounce} + L_{retrieve} + L_{assemble} + L_{network} + L_{ttft} + L_{decode}
$$

其中：

- `debounce` 是防抖等待，意思是等用户短暂停一下再发请求。
- `retrieve` 是检索相关片段。
- `assemble` 是上下文拼装。
- `network` 是网络来回时间。
- `ttft` 是 time to first token，指模型开始吐出第一个 token 的时间。
- `decode` 是后续生成时间。

如果前端防抖设为 100ms，模型推理相关时间总和是 180ms，那么用户大约在 280ms 看见结果，还在合理区间。若为了减压把防抖调到 200ms，总时延就变成 380ms，已经接近危险线。这说明很多系统并不是“模型太慢”，而是把前端和后端各自多加一点保守延迟，最后把体验叠没了。

真实工程例子更明显。大型在线补全系统常出现“用户继续输入，旧请求已无意义”的情况。这类被放弃的请求，在一些生产环境中可占到 45%~50%。如果服务端不支持流级取消，就会继续为这些旧请求做检索、推理、解码，既浪费 GPU，也挤占新请求时延。所以补全系统的核心机制，不只是生成候选，更是快速承认“这个候选已经过时”。

---

## 代码实现

下面给出一个极简但可运行的补全评分原型。它不调用真实模型，只模拟两个关键步骤：候选打分与标识符合法性检查。这样可以先把排序框架搭起来，再替换成真实推理服务。

```python
import math
import re

PYTHON_BUILTINS = {
    "len", "sum", "min", "max", "print", "range", "str", "int", "float", "list", "dict"
}

PYTHON_STDLIB_HINTS = {
    "math", "json", "re", "datetime"
}

def score_completion(token_log_probs):
    assert token_log_probs, "token_log_probs cannot be empty"
    return sum(token_log_probs) / len(token_log_probs)

def extract_identifiers(code):
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
    keywords = {
        "def", "class", "return", "if", "else", "for", "while",
        "import", "from", "as", "try", "except", "with", "in"
    }
    return {t for t in tokens if t not in keywords}

def check_identifier_validity(completion, context):
    context_ids = extract_identifiers(context)
    completion_ids = extract_identifiers(completion)
    allowed = context_ids | PYTHON_BUILTINS | PYTHON_STDLIB_HINTS
    unknown = completion_ids - allowed
    return len(unknown) == 0, unknown

def rank_candidates(context, candidates):
    ranked = []
    for text, token_log_probs in candidates:
        score = score_completion(token_log_probs)
        valid, unknown = check_identifier_validity(text, context)
        penalty = -5.0 if not valid else 0.0
        ranked.append({
            "text": text,
            "score": score + penalty,
            "valid": valid,
            "unknown": sorted(unknown),
        })
    return sorted(ranked, key=lambda x: x["score"], reverse=True)

context = """
import math
variance = 9
data = [1, 2, 3]
"""

candidates = [
    ("std = math.sqrt(variance)", [-0.05, -0.08, -0.03, -0.06]),
    ("std = compute_std(data)", [-0.04, -0.04, -0.05, -0.04]),
]

ranked = rank_candidates(context, candidates)

assert ranked[0]["text"] == "std = math.sqrt(variance)"
assert ranked[0]["valid"] is True
assert ranked[1]["valid"] is False
assert "compute_std" in ranked[1]["unknown"]

print(ranked[0])
```

这段代码表达了三个工程上非常重要的点。

第一，`score_completion` 用平均对数概率给候选做基础排序。真实系统里，这些分数来自模型返回的 token logprob，或来自一个轻量 reranker。reranker 可以理解为“二次排序器”，专门做更细的候选比较。

第二，`check_identifier_validity` 用上下文中出现过的标识符，加上内置函数和标准库提示，过滤明显越界的候选。这里的“标识符”就是变量名、函数名、类名这类可引用名称。真正的生产系统会接入语法树、类型系统、LSP 或静态分析器，但最小系统先做词法级检查，也比裸输出强很多。

第三，候选排序不是“谁分数高就完事”，而是“高分且合法”优先。很多新手实现会直接把模型 top-1 输出给 IDE，这样命中率很不稳定。正确做法是至少保留多候选，做一次后验验证，再决定最终展示顺序。

如果把它放进真实服务链路，一个最小架构可以是：

| 组件 | 输入 | 输出 | 关键职责 |
|---|---|---|---|
| IDE 插件 | 光标、文件内容、按键事件 | 补全请求 | 节流、取消、展示 |
| Context Assembler | 文件片段、检索结果 | 模型提示词 | 控制上下文预算 |
| Generator | 提示词 | 多个候选 + logprob | 生成候选 |
| Validator/Reranker | 候选、上下文 | 排序后的候选 | 语法、作用域、类型检查 |

真实工程例子可以设成这样：用户在 TypeScript 项目里输入 `user.`。插件立刻发送当前文件前后片段，并附带最近编辑过的 `types/user.ts` 和 `services/auth.ts` 的摘要。服务端若 120ms 内能完成检索与组装，模型再在 80ms 内返回首个 token，那么开发者约在 250~300ms 看到 `user.id`、`user.email`、`user.roles` 等候选。这时体验是“像 IDE 本地能力自然冒出来”，而不是“我触发了一个远程 AI 请求”。

---

## 工程权衡与常见坑

第一个权衡是触发频率与资源消耗。防抖太短，请求风暴严重；防抖太长，用户等不到结果。多数系统会把前端防抖放在 100~150ms 左右，并且一旦用户继续输入，立刻取消旧请求。这里的“取消”必须落到服务端连接级别，而不是前端本地假装忽略结果；否则 GPU 和网络仍在消耗。

第二个权衡是上下文长度与首 token 延迟。上下文越长，prefill 越慢。prefill 可以理解为模型在正式生成前，先把输入整段“读一遍”的计算阶段。很多补全服务慢，不是 decode 慢，而是 prefill 慢。所以应尽量缓存稳定片段，比如文件头 import、类定义摘要、最近未变更的接口描述，而不是每次全量重算。

第三个权衡是通用知识与项目知识。通用模型擅长补全常见语法和公共库，但一到公司内部 SDK、私有 API、约定式命名，命中率会明显下降。这时要加项目级检索，甚至构建轻量知识图谱。知识图谱可以白话理解为：把“模块、函数、调用关系、常用搭配”组织成可查询的结构化关系网。

常见坑通常集中在下面三类：

| 常见坑 | 直接后果 | 修正方法 |
|---|---|---|
| 防抖不当，且不支持取消 | 40% 以上请求被废弃，服务器空转 | `100~150ms` 防抖 + 服务端流级取消 |
| 上下文贪多 | prefill 变慢，整体超时 | 严格做 prefix/suffix/retrieved 预算 |
| 只靠通用模型 | 内部 API 建议错误、变量名脱轨 | 加项目检索、示例索引、接口摘要 |

还要特别注意网络层。在线补全不是普通 Web 表单，它是高频、小包、强时延敏感场景。如果没有 HTTP/2 多路复用和长连接，请求会频繁重建 TLS 连接；一旦再叠加大量废弃请求，成本和延迟都会失控。很多团队先优化 prompt，最后发现真正的瓶颈在连接管理，这是典型的工程认知偏差。

---

## 替代方案与适用边界

不是所有团队都应该直接上“远端 LLM + 全局在线服务”。如果场景高度隐私敏感，或者网络不稳定，本地推理加静态分析可能更合适。本地推理的意思是模型直接在开发者机器或企业内网设备上跑，避免代码出域。它的优点是隐私和可控，缺点是模型尺寸、显存、升级成本受限。

可以把主要路线放进一个决策矩阵：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 远端 LLM | 模型强、易统一升级 | 网络敏感、隐私压力大 | 公共代码、云研发环境 |
| 本地推理 | 低出域风险、弱网可用 | 机器资源受限 | 金融、政企、离线开发 |
| 项目检索增强 | 贴合私有 API、改造成本低 | 需要维护索引 | 大型内部项目、命名复杂系统 |

一个现实可行的替代流程是“渐进增强”：

1. 本地静态分析先给出最便宜的补全，比如字段名、局部变量、函数签名。
2. 后台并行发起项目知识检索和 LLM 请求。
3. 若在预算内返回高置信候选，就替换或补充本地候选。
4. 本地 reranker 最终决定展示顺序。

这条路线的价值在于，最坏情况下用户仍有基础补全；最好情况下又能获得更智能的建议。它不是把所有赌注都压在远端模型上，而是把体验拆成分层退化结构。对“零基础到初级工程师”常见的团队项目，这种设计通常比单纯追求最大模型更稳。

适用边界也要讲清。代码补全擅长的是“局部续写”和“短距离 API 提示”，不擅长“跨文件大规模设计决策”。当任务变成“重构模块边界”或“根据需求生成完整子系统”时，补全服务就不再是核心工具，而应切换到更长上下文、更强交互的代码助手模式。

---

## 参考资料

- Michael Brenndoerfer, *Code Completion: Context, Ranking, Latency & UX*：讨论上下文预算、排序信号、延迟分解与输入节奏控制。<https://mbrenndoerfer.com/writing/code-completion-context-ranking-latency-ux-llm>
- ZenML, *Building a Low-Latency Global Code Completion Service*：总结 GitHub Copilot 代理层、HTTP/2 多路复用、请求取消和全球部署经验。<https://www.zenml.io/llmops-database/building-a-low-latency-global-code-completion-service>
- EmergentMind, *Project-Specific Code Completion*：讨论项目级知识检索、私有 API 对齐与补全增强。<https://www.emergentmind.com/topics/project-specific-code-completion>
- EmergentMind, *AI Code Completion*：总结概率评分、本地推理与多阶段补全思路。<https://www.emergentmind.com/topics/ai-code-completion>
- Alibaba Cloud, *Qoder NEXT Performance Optimization*：给出代码补全的体验阈值与延迟分级经验。<https://www.alibabacloud.com/blog/602787>
