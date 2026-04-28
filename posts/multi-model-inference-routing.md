## 核心结论

多模型推理路由，指的是在一次线上请求到来时，先判断这条请求的难度、风险、时延要求和预算，再从一组候选模型里选出“当前最划算”的那个去执行。

它解决的不是“哪个模型最强”，而是“在给定质量目标下，哪个模型最值得用”。这里的“值得”通常同时包含三件事：回答质量、推理成本、响应延迟。对初级工程师来说，可以先记成一句话：**先让便宜模型处理大多数简单请求，只有在它不够可靠时才升级到更强模型。**

最小决策公式可以写成：

$$
m^*(x)=\arg\max_{m\in \mathcal M}\Big(\hat q_m(x)-\lambda c_m-\mu \ell_m(x)\Big)
$$

其中，$\hat q_m(x)$ 是模型 $m$ 对请求 $x$ 的预期质量，白话说就是“这题它大概能答多好”；$c_m$ 是成本；$\ell_m(x)$ 是延迟；$\lambda,\mu$ 是把成本和延迟换算成“惩罚分”的系数。

下面这张表可以把目标看清楚：

| 目标 | 希望方向 | 常见代价 |
|---|---|---|
| 质量 | 更高正确率、更稳定输出 | 往往需要更大模型、更高成本 |
| 成本 | 每次请求更便宜 | 可能牺牲复杂任务表现 |
| 延迟 | 更快返回 | 可能限制上下文长度或模型能力 |

玩具例子很直接：用户问“HTTP 404 是什么”，小模型通常足够；用户上传一篇很长的技术设计文档，要求找出前后矛盾并给出修改建议，这时大模型更合适。路由器真正做的，就是把这两类请求区分开，而不是让所有请求都走同一条昂贵路径。

---

## 问题定义与边界

把问题严格写出来，多模型推理路由的输入、输出和约束如下。

| 元素 | 定义 | 白话解释 |
|---|---|---|
| 请求特征 $x$ | 请求文本、上下文长度、任务类型、历史反馈等 | 路由器看到的题目信息 |
| 候选模型集合 $\mathcal M$ | 可调用的模型列表 | 你系统里现成能用的老师名单 |
| 业务目标 | 质量、成本、延迟、SLA、安全 | 你到底想优化什么 |
| 约束条件 | 预算上限、超时、配额、地域、合规 | 哪些线不能碰 |
| 输出动作 $a$ | 选哪个模型、是否升级、是否回退 | 这次到底怎么处理 |

这里有几个边界必须划清。

第一，它不是 MoE 路由。MoE，Mixture of Experts，白话说是“一个大模型内部把不同 token 分给不同专家网络”。那是模型内部计算图的一部分。本文讲的是**系统层**路由，是在多个独立模型服务之间做选择。

第二，它不是负载均衡。负载均衡关注的是把流量分到多台机器，避免单机过载；多模型路由关注的是“这条请求该由哪种能力的模型回答”。两者可能同时存在，但目标不同。

第三，它不等于 fallback。fallback，白话说是“失败后兜底重试”。如果你只有“小模型失败了再换大模型”这一条规则，这只是路由的一种最简单形式，不代表完整的路由系统。

第四，它也不等于模型集成。集成是让多个模型同时给答案再融合，提升稳定性；路由通常只选一个模型，重点是省钱和控延迟。

真实工程里，一批请求经常天然分层：短文本分类、客服 FAQ、复杂数学推理、长文总结，它们最优模型常常不同。路由成立的前提，不是“模型越多越好”，而是“不同请求之间的难度和价值差异足够大”。

---

## 核心机制与推导

核心机制是：先估计每个候选模型在当前请求上的收益，再扣掉成本和延迟惩罚，最后选总分最高的模型。

统一打分函数已经给出：

$$
s(m,x)=\hat q_m(x)-\lambda c_m-\mu \ell_m(x)
$$

然后选择：

$$
m^*(x)=\arg\max_{m\in \mathcal M}s(m,x)
$$

如果系统采用两阶段级联，还要再加一个“是否升级”的阈值规则。设小模型输出后，我们能计算一个置信度 $\hat r(x)$，那么：

$$
a(x)=
\begin{cases}
\text{accept small model}, & \hat r(x)\ge \tau \\
\text{escalate to large model}, & \hat r(x)<\tau
\end{cases}
$$

这里的 $\tau$ 不是拍脑袋定的。它应该来自验证集上的收益比较。一个常见思路是：当“小模型直接放行的预期损失”大于“升级一次的额外成本”时，就升级。形式上可写成：

$$
\text{upgrade if } \Delta \text{quality} > \lambda \Delta \text{cost} + \mu \Delta \text{latency}
$$

这句话翻译成白话就是：**大模型带来的质量提升，必须值得那笔额外的钱和等待时间。**

先看一个 2 模型玩具例子。模型 A 是小模型，模型 B 是大模型。

| 请求 | $\hat q_A(x)$ | $\hat q_B(x)$ | $c_A$ | $c_B$ | $\ell_A(x)$ | $\ell_B(x)$ |
|---|---:|---:|---:|---:|---:|---:|
| 简单 FAQ | 0.86 | 0.90 | 0.1 | 1.0 | 0.2 | 1.2 |
| 长文推理 | 0.48 | 0.83 | 0.1 | 1.0 | 0.3 | 1.5 |

假设 $\lambda=0.2,\mu=0.1$。

对简单 FAQ：

- A 分数：$0.86-0.2\times0.1-0.1\times0.2=0.82$
- B 分数：$0.90-0.2\times1.0-0.1\times1.2=0.58$

所以选 A。

对长文推理：

- A 分数：$0.48-0.2\times0.1-0.1\times0.3=0.43$
- B 分数：$0.83-0.2\times1.0-0.1\times1.5=0.48$

所以选 B。

这说明路由的关键不是“大模型总分高”，而是“在当前请求上，大模型是否高到足以覆盖它更贵、更慢的代价”。

真实工程例子：一个面向企业客服的问答系统，白天有大量“退款政策、发票下载、密码重置”这类标准问题，也有少量“合同条款冲突解释、跨轮上下文追责、长附件摘要”这类复杂任务。如果所有流量都打到大模型，成本会失控；如果全部用小模型，复杂问题又会明显误答。最合理的方案通常是：先用规则和轻量特征做首轮过滤，再用一个路由器估计难度，对难请求升级。

要注意一个经常被忽视的问题：$\hat q_m(x)$ 本身是预测值，不是真实质量。它需要校准。校准，白话说就是“让模型说的 80% 置信度，真的大致对应 80% 成功率”。没有校准，阈值就会乱跳，离线好看，线上失真。

---

## 代码实现

工程实现最好拆成 4 层：特征提取、路由打分、模型执行、结果回传。不要把这些逻辑揉在一次模型调用里，否则你既无法解释路由原因，也很难后续调参。

一个最小请求结构可以这样想：

| 字段 | 含义 |
|---|---|
| `task_type` | 问答、总结、分类、代码解释等 |
| `prompt_len` | 输入长度 |
| `context_len` | 附加上下文长度 |
| `risk_level` | 业务风险级别 |
| `latency_sla_ms` | 时延要求 |
| `user_tier` | 用户等级或套餐 |

执行流程可以概括为：

`请求进入 -> 特征提取 -> 对每个模型打分 -> 选择模型 -> 执行推理 -> 记录日志与反馈`

下面给一个可运行的 Python 玩具实现：

```python
from dataclasses import dataclass

@dataclass
class Request:
    task_type: str
    prompt_len: int
    context_len: int
    risk_level: int  # 1 low, 2 medium, 3 high

MODELS = {
    "small": {"cost": 0.1, "base_latency": 0.2},
    "large": {"cost": 1.0, "base_latency": 1.2},
}

def predict_quality(model_id: str, req: Request) -> float:
    if model_id == "small":
        score = 0.88
        if req.prompt_len + req.context_len > 4000:
            score -= 0.30
        if req.task_type in {"math_reasoning", "long_summary"}:
            score -= 0.20
        if req.risk_level >= 3:
            score -= 0.10
        return max(0.0, min(1.0, score))
    else:
        score = 0.93
        if req.task_type in {"math_reasoning", "long_summary"}:
            score += 0.03
        return max(0.0, min(1.0, score))

def predict_latency(model_id: str, req: Request) -> float:
    base = MODELS[model_id]["base_latency"]
    size_penalty = (req.prompt_len + req.context_len) / 10000.0
    return base + size_penalty

def route(req: Request, lambda_cost: float = 0.2, mu_latency: float = 0.1) -> str:
    best_model = None
    best_score = float("-inf")
    for model_id, info in MODELS.items():
        q = predict_quality(model_id, req)
        l = predict_latency(model_id, req)
        score = q - lambda_cost * info["cost"] - mu_latency * l
        if score > best_score:
            best_score = score
            best_model = model_id
    return best_model

faq = Request(task_type="faq", prompt_len=80, context_len=50, risk_level=1)
hard_task = Request(task_type="long_summary", prompt_len=2500, context_len=3000, risk_level=2)

assert route(faq) == "small"
assert route(hard_task) == "large"
```

这个例子虽然简单，但结构是对的：

1. `predict_quality` 负责估计质量，不直接做实际推理。
2. `predict_latency` 负责估计时延。
3. `route` 统一做打分决策。
4. 真正的 `infer()` 可以在选定模型后再调用。
5. 线上必须额外记录 `route_reason`、超时、异常、fallback 次数。

真实工程里，日志至少要保留：请求特征摘要、选中的模型、每个候选模型的得分、是否升级、最终用户反馈。没有这些字段，后面就无法回答“为什么这类请求成本突然涨了”。

---

## 工程权衡与常见坑

多模型路由最常见的失败，不是公式错，而是工程闭环不完整。下面这张表覆盖高频问题。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 阈值过低 | 大量请求升级到大模型，成本失控 | 用验证集画成本-质量曲线，按预算选阈值 |
| 阈值过高 | 复杂请求被小模型误处理，质量下降 | 对高风险任务设单独阈值或强制升级 |
| 没有 fallback | 目标模型超时或报错时直接失败 | 设计超时回退、默认模型和重试策略 |
| 只看 benchmark | 离线分数好，线上场景不匹配 | 用真实流量回放和人工抽检补评估 |
| 路由器漂移 | 模型版本变了，旧规则失效 | 做周期重标注、校准和在线监控 |

还有几个现实约束不能省略。

第一，只看平均延迟不够，必须看尾延迟。尾延迟，白话说是“最慢那一小部分请求的延迟”，常用 P95、P99 表示。如果路由把大量边缘请求都送进慢模型，哪怕平均值还行，SLA 也可能已经坏掉。

第二，模型升级会改变路由器本身。比如你把小模型从旧版替换成新版，它的回答风格、长度、置信度分布都可能变化，原来的阈值不一定还能用。路由器不是一次训练完就永远稳定的组件，它和被路由的模型是联动的。

第三，不同时间段的最优策略可能不同。白天高峰时，慢模型队列更长，延迟惩罚应该更大；夜间流量低时，可以更愿意升级。也就是说，$\ell_m(x)$ 不只是模型固有属性，还跟系统负载有关。

第四，路由器本身会过拟合。比如你把“输入越长越难”学成硬规律，结果线上出现很多“长但简单”的模板化文本，系统就会被误导。解决方法通常不是把路由模型做得无限复杂，而是增加更稳的特征、加校准、做在线 A/B。

---

## 替代方案与适用边界

多模型路由不是默认答案。很多时候，更简单的方案更合适。

| 方案 | 优点 | 缺点 | 适用条件 | 实现成本 |
|---|---|---|---|---|
| 单一默认模型 | 最简单、最稳定 | 成本或质量难两全 | 业务量小、任务单一 | 低 |
| 规则路由 | 可解释、易审计 | 规则维护成本高 | 高合规、任务边界清晰 | 低到中 |
| 两阶段级联 | 降本明显，结构清楚 | 阈值调优敏感 | 多数请求简单、少数困难 | 中 |
| 人工白名单 | 风险可控 | 扩展性差 | 少量关键客户或高价值任务 | 低 |
| 模型集成 | 质量上限高 | 成本和延迟最高 | 高风险、低吞吐任务 | 高 |
| 学习型路由 | 潜在最优，能持续改进 | 训练和监控复杂 | 流量大、难度分层明显 | 高 |

可以把选择条件压缩成一个决策矩阵：

| 场景特征 | 更适合的方案 |
|---|---|
| 请求类型少、预算不敏感 | 单一默认模型 |
| 规则很明确，例如按业务线分流 | 规则路由 |
| 80% 请求简单、20% 请求困难 | 两阶段级联 |
| 单次误答代价极高 | 模型集成或强制大模型 |
| 请求差异大、流量足够大 | 学习型路由 |

多模型路由最值得上的场景，通常满足三条：

1. 候选模型之间在成本和能力上存在明显梯度。
2. 请求难度分布明显分层，不是所有请求都一样难。
3. 业务可以容忍一定系统复杂度，愿意为降本或提效增加监控、评估和调参。

反过来，如果几个候选模型表现差不多，只是名字不同，那路由收益通常有限；如果业务第一优先级是稳定可解释，而不是极致降本，那么固定模型加少量规则往往更稳。

---

## 参考资料

1. [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176) 这篇资料对应正文里“把质量、成本、延迟放到统一决策里”的核心思路。
2. [RouteLLM: Learning to Route LLMs with Preference Data](https://arxiv.org/abs/2406.18665) 这篇资料对应正文里“学习型路由器”和“离线评估阈值”的部分。
3. [Amazon Bedrock Intelligent Prompt Routing](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html) 这篇资料对应正文里“真实系统中的提示路由”和工程落地边界。
4. [Azure Architecture Center: Rate Limiting pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/rate-limiting-pattern) 这篇资料用于区分“系统调度/限流”和“模型路由”不是同一层问题。
5. [Google SRE Book: Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/) 这篇资料对应正文里“没有 fallback、只看平均时延会出问题”的工程风险。
