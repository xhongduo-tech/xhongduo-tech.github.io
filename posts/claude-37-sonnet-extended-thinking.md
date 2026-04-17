## 核心结论

Claude 3.7 Sonnet 的 `Extended Thinking` 可以理解为“给同一个模型一笔额外的思考预算”。`Token` 是模型处理文本时使用的最小计数单位，近似可理解为“文本片段的计费与容量单位”。开启后，模型不是直接吐出答案，而是先消耗一部分预算生成 `thinking block`，再用剩余预算给最终回答。

这件事的价值不在“更像人思考”，而在工程上更可控。默认模式偏向低延迟，适合直接问答；`Extended Thinking` 偏向复杂任务拆解，适合 Agent。`Agent` 是“会多步执行任务、会调工具的程序”。官方文档给出的约束是：思考预算也算在本轮 `max_tokens` 里，而且 Claude 3.7 不会自动帮你压缩超额预算，超了就直接报校验错误。

对工程场景更关键的是基准表现。公开资料中，Claude 3.7 Sonnet 在 SWE-bench Verified 上为 62.3%，配合自定义 scaffold 可到 70.3%；同口径资料里，o3-mini(high) 为 49.3%。`Scaffold` 是“围在模型外面的一层固定流程代码，比如补充测试、重试、文件筛选”。这说明 Claude 3.7 的优势不是单条回答更花哨，而是在多步编码与工具调用链条里更稳。

| 模式 | 响应时间 | 是否展示思考链路 | 适合场景 | Agent 稳定性 |
|---|---:|---|---|---|
| 默认模式 | 较快 | 通常不展示 | FAQ、摘要、改写 | 中 |
| Extended Thinking | 较慢 | 可返回 thinking block | 调试、规划、多步编码 | 较高 |

玩具例子可以这样理解：默认模式像聊天机器人直接快答；打开 `Extended Thinking` 后，Claude 会先写出“先检查缓存、再看 DB、再看重试链路”这类步骤，再给结论。它不是切到另一套模型，而是在一个模型里切换“速答档”和“深思档”。

---

## 问题定义与边界

问题不是“模型能不能想得更久”，而是“在 Agent 场景里，怎样同时拿到速度、可解释性和可执行性”。如果一个模型只追求快，它容易在多步任务里漏步骤；如果只追求深思，它会变慢、变贵，还可能把预算浪费在无关分支上。

Claude 3.7 的边界非常明确：所有内容都要塞进上下文窗口。`上下文窗口` 是“本轮请求里模型能看到并处理的总容量上限”。设：

- $C_w$：上下文窗口容量
- $P$：prompt tokens
- $B_t$：thinking budget
- $T_o$：最终输出 tokens

则约束是：

$$
P + B_t + T_o \leq C_w
$$

在 Claude 3.7/4 的文档口径里，更接近 API 实际行为的写法是：

$$
P + \texttt{max\_tokens} \leq C_w,\quad B_t < \texttt{max\_tokens}
$$

因为 `max_tokens` 已经把思考块和最终回答一起包进去了。你可以把它理解成：

- `max_tokens` 是“本轮输出总预算”
- `thinking_budget` 是“总预算里专门预留给思考的上限”

最小数值例子：

- prompt 占 2000 tokens
- `max_tokens = 10000`
- `thinking_budget = 4000`

那么 Claude 最多可以用 4000 token 做 thinking block，剩余最多 6000 token 输出最终答案。如果这时上下文窗口只允许 11000 token，那么 `2000 + 10000 = 12000`，请求会被 API 直接拒绝，不会像旧模型那样自动替你缩小输出预算。

这里还有一个容易忽视的边界：前一轮的 thinking block 不会一直原样堆积进后续上下文。官方文档说明，历史 thinking blocks 会被剥离，不按普通聊天文本那样持续累加。这是为了避免“越聊越爆 token”。

---

## 核心机制与推导

`混合推理` 可以理解为“同一模型内的双工作模式”。不是一个模型先回答、另一个模型再审题，而是一个模型在同一轮请求里，先走内部推导，再生成最终输出。

流程可以简化成：

```text
prompt
  ↓
thinking block（消耗 B_t）
  ↓
final answer（消耗 max_tokens - thinking 部分）
```

如果写成 token 序列，更接近工程实现：

```text
[system + user prompt = P]
        ↓
[thinking tokens <= B_t]
        ↓
[text tokens <= max_tokens - used_thinking]
```

这里的关键不是“思考越多越好”，而是“把思考花在高价值步骤上”。例如一个高级问题是“为什么缓存穿透会把数据库打挂，以及该怎么修”。默认模式可能直接列措施：布隆过滤器、空值缓存、限流。开启 `Extended Thinking` 后，模型更可能先做子任务拆解：

1. 识别异常流量是否绕过缓存。
2. 判断空查询是否持续命中数据库。
3. 区分缓存击穿、缓存穿透、缓存雪崩。
4. 再输出治理方案与优先级。

这个“先拆解再回答”的过程，就是 thinking block 的工程价值。它能提升多步任务的正确率，因为模型先把步骤排顺，再生成最终文本或工具调用。

玩具例子：

- 问题：`9 个请求里有 8 个查不存在的商品 ID，接口为什么慢？`
- 默认模式：直接回答“可能缓存穿透，建议加布隆过滤器”。
- Extended Thinking：先判断“是否大量 miss”“是否存在空值缓存缺失”“是否每次 miss 都落 DB”，最后再得出“缓存穿透导致 DB 被无效请求放大”的结论。

真实工程例子：

一个故障 Agent 负责排查“缓存穿透 + 数据库超时”。它收到报警后，不是立刻打开页面乱点，而是先用 thinking block 规划顺序：先查网关错误码分布，再拉 Redis miss 比例，再看 MySQL 慢查询，再决定是否执行 `Computer Use` 去点开监控面板、滚动日志、运行脚本。这种“先深后动”的节奏，正是 Claude 3.7 被拿来做 Agent 的原因。

---

## 代码实现

下面用一个可运行的 Python 玩具实现模拟预算校验。它不调用真实 API，但完整表达了 Claude 3.7 的预算约束。

```python
from dataclasses import dataclass

@dataclass
class ThinkingConfig:
    context_window: int
    prompt_tokens: int
    max_tokens: int
    thinking_budget: int

def validate_budget(cfg: ThinkingConfig) -> tuple[int, int]:
    # Claude 3.7/4 的核心约束：prompt + max_tokens 不能超过上下文窗口
    if cfg.prompt_tokens + cfg.max_tokens > cfg.context_window:
        raise ValueError("context overflow: prompt_tokens + max_tokens exceeds context window")

    # thinking budget 必须小于 max_tokens，且官方建议最小 1024
    if cfg.thinking_budget >= cfg.max_tokens:
        raise ValueError("invalid budget: thinking_budget must be less than max_tokens")
    if cfg.thinking_budget < 1024:
        raise ValueError("invalid budget: thinking_budget should be at least 1024")

    final_answer_budget = cfg.max_tokens - cfg.thinking_budget
    return cfg.thinking_budget, final_answer_budget

cfg = ThinkingConfig(
    context_window=200000,
    prompt_tokens=2000,
    max_tokens=10000,
    thinking_budget=4000,
)

thinking_tokens, answer_tokens = validate_budget(cfg)

assert thinking_tokens == 4000
assert answer_tokens == 6000

# 一个会失败的例子
try:
    validate_budget(ThinkingConfig(
        context_window=11000,
        prompt_tokens=2000,
        max_tokens=10000,
        thinking_budget=4000,
    ))
    assert False, "should not pass"
except ValueError as e:
    assert "context overflow" in str(e)
```

如果把它映射到 API 结构，思路通常是：

```python
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=10000,
    thinking={
        "type": "enabled",
        "budget_tokens": 4000,
    },
    messages=[
        {"role": "user", "content": "分析缓存穿透与数据库超时的根因"}
    ],
)
```

实现顺序建议固定为：

1. 先做 token 预算估算。
2. 再发送请求。
3. 从响应里分离 `thinking` 与最终 `text`。
4. 在 Agent 控制台展示思考块摘要。
5. 只有当规划结果足够明确时，再进入工具调用或 `Computer Use`。

真实工程例子里，控制台可以这么展示：

- `thinking`: 先检查 Redis miss，再看 DB 连接池等待，再比较 404/5xx 分布
- `action`: 打开 Kibana，筛选 10 分钟窗口，执行聚合查询
- `final`: 根因是无效 ID 流量造成缓存穿透，DB 超时是二次效应

这样做的价值是，人类工程师能审阅“它为什么这么做”，而不是只看到最后一句结论。

---

## 工程权衡与常见坑

`Extended Thinking` 的收益来自更强的中间规划，但代价同样真实：更慢、更贵，而且有时会“过度思考”。`过度思考` 是“模型把预算花在不必要的分支上，导致耗时增加，结果却不一定更好”。

| 坑点 | 典型表现 | 原因 | 规避措施 |
|---|---|---|---|
| 过度思考 | 简单问题也变慢 | `B_t` 设太高 | 从 1024 起步，小步上调 |
| 预算未估算 | API 直接报错 | `prompt + max_tokens` 超窗 | 先做 token counting |
| 输出被挤压 | 思考很多，结论太短 | `thinking_budget` 吃掉输出预算 | 给最终答案保底长度 |
| Agent 动作不稳 | 工具执行顺序混乱 | thinking 没转成明确 action plan | 强制分离“计划”和“执行” |
| Computer Use 风险 | 点错、输错、误操作 | Beta 工具直接控制桌面 | 放进 VM/容器，保留人工复核 |
| 历史会话失控 | 聊久后成本上升 | prompt 持续膨胀 | 截断旧上下文，摘要化历史 |

一个常见错误是：开发者只看到“thinking 能增强推理”，于是把所有请求都开到很高预算。结果是 FAQ、格式改写、短摘要这种任务也走深思模式，延迟上升，吞吐下降，用户只会觉得“系统变卡了”。

真实工程例子更典型。处理“缓存穿透 + 数据库超时”时，Agent 先用思考块整理调用顺序，再通过 `Computer Use` 打开日志平台。如果此时忘了压缩 prompt，把告警文本、整段日志、上一轮思考、整套 SOP 全塞进去，就可能在发请求前已经接近窗口上限。再加上高 `max_tokens`，请求会被直接拒绝，根本走不到排障动作那一步。

所以工程上的正确做法不是“默认开大”，而是按任务分级：

- 轻任务：默认模式
- 中任务：低 thinking budget
- 重任务：Extended Thinking + 工具调用 + 人工复核

---

## 替代方案与适用边界

如果任务只是“快速问答、简单解释、短文本生成”，默认模式通常更经济。你真正买到的是低延迟，而不是更高的多步正确率。

如果对比 o3-mini，这里要区分“单轮快答效率”和“复杂 Agent 成功率”。公开资料里，o3-mini(high) 在 SWE-bench Verified 的数值约 49.3%，而 Claude 3.7 Sonnet 为 62.3%，加自定义 scaffold 可到 70.3%。这不代表 o3-mini 没价值，而是说明它更适合快问快答、短链路推理、预算敏感场景；当任务进入“读代码仓库、改多文件、跑测试、根据反馈回修”这类闭环时，Claude 3.7 的混合推理更占优。

| 模型/模式 | 适合场景 | 编码成功率口径 | 预算策略 |
|---|---|---:|---|
| Claude 3.7 默认模式 | 一般问答、普通代码解释 | 低于带 scaffold 的 Agent 配置 | 低 `max_tokens`，不开 thinking |
| Claude 3.7 Extended Thinking | 多步调试、复杂 Agent、工具链协同 | SWE-bench Verified 62.3%，scaffold 70.3% | 预留 thinking budget，先验算窗口 |
| o3-mini(high) | 快速推理、短任务、高吞吐 | SWE-bench Verified 49.3% | 控制成本，追求响应速度 |

替代方案大致有三类：

1. 不开 thinking，直接用默认模式。
2. 用更快的小模型做第一层分类，再把复杂任务路由到 Claude 3.7。
3. 不依赖模型内部思考，而是在外部写死流程，也就是用 scaffold 把“先读文件、再跑测试、再修复”做成程序。

第三类很重要，因为很多工程稳定性其实不是模型自己“想明白”，而是框架帮它“少犯错”。所以实际系统里，最稳的做法往往是：模型负责局部推理，程序负责预算、重试、权限和执行顺序。

---

## 参考资料

- Anthropic 官方文档，`Building with extended thinking`：说明 `thinking` 的 API 形态、`max_tokens` 的严格限制、历史 thinking block 的剥离规则，以及预算建议。  
  https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking

- Anthropic 官方文档，`Computer use tool`：说明 `Computer Use` 的能力边界、beta 状态，以及应放在受限 VM/容器中运行的安全建议。  
  https://docs.anthropic.com/en/docs/build-with-claude/computer-use

- Anthropic Help Center，`Using extended thinking`：说明产品界面中的 Extended Thinking 开关与可视化“Thinking”区域。  
  https://support.anthropic.com/en/articles/10574485-using-extended-thinking

- DataCamp，`Claude 3.7 Sonnet: How it Works, Use Cases & More`：提供 SWE-bench Verified、TAU-bench，以及与 o3-mini 的对比数值，适合用来解释工程 Agent 的公开基准表现。  
  https://www.datacamp.com/blog/claude-3-7-sonnet

- IBM Think，关于 Claude 3.7 混合推理的报道：适合辅助解释“同一模型在快答与深思之间切换”的产品定位。  
  https://www.ibm.com/think/news/claude-sonnet-hybrid-reasoning

- Business Insider，对 Claude 3.7 的实测报道：补充“可能过度思考”的使用体验风险，适合放入工程权衡部分。  
  https://www.businessinsider.com/anthropic-claude-3-7-sonnet-test-thinking-grok-chatgpt-comparison-2025-2
