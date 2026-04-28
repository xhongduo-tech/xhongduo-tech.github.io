## 核心结论

多模型编排架构，指的是把多个能力和成本不同的模型接到同一条服务链路里，再由一个“路由器”决定每个请求应该走哪条路径。这里的“路由器”可以理解成分诊器：它不直接回答问题，只负责判断该把请求交给谁处理。

它解决的核心问题不是“怎样让单次回答最强”，而是“怎样在可接受质量下，把平均成本和平均延迟压低”。对多数线上系统，真正稀缺的资源不是模型数量，而是高质量推理预算。把所有请求都送进最强模型，通常能工作，但往往浪费。

一个最直观的玩具例子是客服问答。用户问“今天订单到哪了”，这类问题结构固定、上下文短，模板或小模型就足够；用户问“帮我分析这段合同里的违约责任是否对乙方不利”，这类问题需要长上下文、术语理解和条件推理，才值得升级到大模型。多模型编排的价值，就在于把这两类请求区分开。

| 请求类型 | 典型特征 | 适合路径 | 原因 |
| --- | --- | --- | --- |
| 简单请求 | 短问题、固定意图、答案模式稳定 | 模板 / 小模型 | 成本低、延迟短 |
| 复杂请求 | 长上下文、多条件、风险高 | 大模型 | 推理和稳健性更强 |

所以，多模型编排不是“多调几个模型试一试”，而是把不同模型串成分层决策链：小模型先过滤、分类、重写或草拟，再把少量高价值请求升级到大模型。路由准确时，系统整体效率会显著提升；路由错误时，质量会直接回退，甚至出现成本失控。

---

## 问题定义与边界

要设计多模型编排，第一步不是选模型，而是定义“什么样的请求值得升级”。这里的“升级”是指从便宜路径切换到更贵、通常也更强的路径。这个判断一般依赖五类信号：请求难度、业务风险、上下文长度、模型置信度、请求价值。

这说明多模型编排本质上是资源分配问题。系统面对的不是“能不能回答”，而是“这道题值不值得花更贵的推理成本”。如果 100 个请求里只有 20 个真的复杂，让 100 个都走大模型就是浪费；但如果路由器把那 20 个复杂请求漏掉，用户会首先感知到质量下降，而不是账单变便宜。

一个简化流程可以写成：

`输入请求 -> 特征提取 -> 路由判断 -> 小模型/大模型 -> 结果返回或兜底回退`

这里的“特征提取”是把原始请求转成可判断的信号，比如字数、是否包含法律条款、是否需要多轮上下文、是否命中高风险标签。它不一定依赖神经网络，规则系统也可以完成第一版。

边界同样重要。多模型编排适合请求复杂度分布明显不均匀的业务，不适合所有链路都混进去。

| 场景 | 是否适合 | 原因 |
| --- | --- | --- |
| 客服问答 | 适合 | 简单问题多，复杂问题少 |
| 检索增强问答 | 适合 | 可先判断证据充分性再升级 |
| 内容审核 | 适合 | 常规过滤与高风险复核可分层 |
| 代码助手 | 适合 | 简单补全与复杂推理差异大 |
| 低风险草稿生成 | 适合 | 容忍一定回退，可换成本 |
| 强一致性交易 | 不适合 | 路由误判代价过高 |
| 必须单模型可复现的核心链路 | 不适合 | 路径变化会影响稳定性 |
| 无法容忍误判的高风险场景 | 不适合 | 小模型误放行会造成严重后果 |

真实工程例子是企业知识库助手。员工问“报销单上传入口在哪”，命中 FAQ，直接返回模板答案；员工问“这个采购流程是否符合跨境数据合规要求”，就需要检索、法规理解和风险措辞，应该直接升级到更强模型，甚至附带人工复核。前者追求低延迟，后者追求低误判，这是两种完全不同的优化目标。

---

## 核心机制与推导

多模型编排里最核心的对象是路由函数。设输入请求为 $x$，路由器输出升级概率 $r(x)$，其中：

$$
r(x) \in [0,1]
$$

如果 $r(x) \ge \tau$，则请求走大模型 $L$；否则走小模型 $S$。这里的 $\tau$ 是阈值，可以理解成“多大把握才值得升级”。

$$
\text{if } r(x) \ge \tau,\ \text{use } L;\quad \text{else use } S
$$

从系统视角，平均成本和平均延迟都可以写成加权期望。设升级比例为 $q$，路由器本身也有调用开销 $C_r$ 和延迟 $T_r$，则：

$$
E[C] = C_r + (1-q)C_S + qC_L
$$

$$
E[T] = T_r + (1-q)T_S + qT_L
$$

其中 $C_S, C_L$ 分别是小模型和大模型单次成本，$T_S, T_L$ 分别是对应延迟。

工程上只最小化成本通常不够，因为低成本可能意味着质量损失。于是常把质量损失项也放进目标函数：

$$
E[L] = E[C] + \lambda \cdot E[Q_{loss}]
$$

这里的 $\lambda$ 是权重，表示业务愿意为了质量多花多少钱。白话说，$\lambda$ 越大，系统越保守，越倾向把请求送去强模型。

用题目给的数值做一次推导：

- 小模型：`0.002` 美元，`150 ms`
- 大模型：`0.020` 美元，`1200 ms`
- 路由器：`0.001` 美元，`30 ms`
- 升级率：$q = 0.2$

则：

$$
E[C] = 0.001 + 0.8 \times 0.002 + 0.2 \times 0.020 = 0.0066
$$

$$
E[T] = 30 + 0.8 \times 150 + 0.2 \times 1200 = 390
$$

也就是平均每次请求成本 `0.0066` 美元，平均延迟 `390 ms`。如果全部直接走大模型，则是 `0.020` 美元与 `1200 ms`。两者对比：

| 路径 | 平均成本（美元） | 平均延迟（ms） |
| --- | ---: | ---: |
| 全量大模型 | 0.0200 | 1200 |
| 20% 升级的多模型编排 | 0.0066 | 390 |

这相当于平均成本下降约 `67%`，平均延迟下降约 `67.5%`。这就是多模型编排能成立的基本数学前提：复杂请求只占少数，而且路由成本远小于大模型成本。

阈值 $\tau$ 的变化会直接改变升级率 $q$。阈值低，更多请求走大模型，质量上升但成本和延迟也上升；阈值高，系统更省钱，但更容易把复杂请求误分到小模型。

| 阈值变化 | 升级率趋势 | 成本 | 延迟 | 质量风险 |
| --- | --- | --- | --- | --- |
| 阈值变低 | 升级率上升 | 上升 | 上升 | 下降 |
| 阈值变高 | 升级率下降 | 下降 | 下降 | 上升 |

分诊台这个比喻在这里成立：不是所有病人都挂专家号，而是先分轻重缓急。问题不在于“专家好不好”，而在于“分诊准不准”。

---

## 代码实现

最小可用版本通常分成三段：特征提取、路由决策、模型执行与兜底。第一版不必先训练复杂路由器，规则加阈值就能跑通主链路。

下面是一个可运行的 Python 玩具实现。它不调用真实模型，而是模拟路由逻辑、升级率和超时兜底。`assert` 用来验证预期行为。

```python
from dataclasses import dataclass

SMALL_COST = 0.002
LARGE_COST = 0.020
ROUTER_COST = 0.001

@dataclass
class Request:
    text: str
    context_tokens: int = 0
    risk: str = "low"   # low / medium / high
    confidence: float = 1.0  # small-model confidence estimate

def extract_features(req: Request) -> dict:
    text = req.text.lower()
    return {
        "length": len(req.text),
        "context_tokens": req.context_tokens,
        "has_contract": "合同" in req.text or "违约" in req.text,
        "has_code": "bug" in text or "traceback" in text or "代码" in req.text,
        "risk_high": req.risk == "high",
        "low_confidence": req.confidence < 0.65,
    }

def predict_upgrade_prob(features: dict) -> float:
    score = 0.0
    if features["length"] > 40:
        score += 0.15
    if features["context_tokens"] > 2000:
        score += 0.30
    if features["has_contract"]:
        score += 0.35
    if features["has_code"]:
        score += 0.20
    if features["risk_high"]:
        score += 0.25
    if features["low_confidence"]:
        score += 0.25
    return min(score, 1.0)

def route_request(req: Request, tau: float = 0.5) -> str:
    features = extract_features(req)
    score = predict_upgrade_prob(features)
    return "large" if score >= tau else "small"

def small_model_generate(req: Request) -> str:
    if "超时" in req.text:
        raise TimeoutError("small model timeout")
    return "small:" + req.text[:20]

def large_model_generate(req: Request) -> str:
    return "large:" + req.text[:20]

def fallback_model_generate(req: Request) -> str:
    return "fallback:" + req.text[:20]

def handle_request(req: Request, tau: float = 0.5) -> str:
    try:
        route = route_request(req, tau=tau)
        if route == "large":
            return large_model_generate(req)
        return small_model_generate(req)
    except TimeoutError:
        return fallback_model_generate(req)

simple_req = Request(text="今天订单到哪了", context_tokens=20, risk="low", confidence=0.95)
complex_req = Request(text="请分析这段合同中的违约责任是否对乙方不利，并指出风险条款", context_tokens=3200, risk="high", confidence=0.40)
timeout_req = Request(text="订单查询超时", context_tokens=10, risk="low", confidence=0.99)

assert route_request(simple_req) == "small"
assert route_request(complex_req) == "large"
assert handle_request(simple_req).startswith("small:")
assert handle_request(complex_req).startswith("large:")
assert handle_request(timeout_req).startswith("fallback:")

def expected_cost(q: float) -> float:
    return ROUTER_COST + (1 - q) * SMALL_COST + q * LARGE_COST

assert round(expected_cost(0.2), 4) == 0.0066
```

这个版本体现了三个关键点。

第一，特征不必复杂。字数、上下文长度、是否涉及合同或代码、风险等级、置信度，已经足够构成一个早期路由器。所谓“置信度”，就是系统对当前模型答案可靠性的估计，不是用户满意度。

第二，路由器可以是规则，也可以是学习器。最开始常见写法就是：

```python
def route_request(x):
    score = router.predict_upgrade_prob(x)
    if score >= tau:
        return large_model.generate(x)
    return small_model.generate(x)
```

第三，执行层必须有 fallback。fallback 就是兜底路径，指主路径失败时的替代处理方式。因为线上系统的真实故障不是“答得不好”而已，还包括超时、接口错误、上下文过长、配额耗尽。

```python
try:
    if route_request(x) == "large":
        return large_model.generate(x, timeout=2.0)
    return small_model.generate(x, timeout=0.5)
except TimeoutError:
    return fallback_model.generate(x)
```

真实工程例子可以看企业 Copilot 或客服系统的部署方式。第一层先做意图分类与风险识别；第二层简单 FAQ 走缓存、模板或小模型；第三层复杂问答走检索增强加大模型；第四层在超时或低置信度时回退到固定文案或人工接管。这样设计的关键不是模型越多越好，而是每一层职责明确。

实现清单通常至少包含以下几项：

| 模块 | 最低要求 | 作用 |
| --- | --- | --- |
| 输入特征 | 长度、风险、上下文、置信度 | 判断是否升级 |
| 路由策略 | 规则 / 分类器 / 打分阈值 | 控制分流 |
| 超时控制 | 小模型与大模型各自超时 | 防止链路阻塞 |
| fallback 机制 | 备用模型 / 模板 / 人工接管 | 保证可用性 |
| 日志与监控 | 路由分布、错误、延迟、质量 | 支持调参与排障 |

---

## 工程权衡与常见坑

多模型编排最常见的误区，是把它理解成“加一层路由就自动省钱”。实际并不是。只有当路由足够准，且复杂请求比例确实不高时，它才成立。否则可能得到一个比单大模型更复杂、却并不更便宜的系统。

第一类坑是阈值没校准。很多团队直接设一个 `0.5` 就上线，但路由器输出分数不一定可比较。所谓“校准”，就是让 0.8 尽量真的接近 80% 概率，而不是一个任意分数。阈值不校准时，高风险请求会被误送到小模型，质量会直接塌陷。做法是用独立验证集按业务分桶调参，而不是只看整体均值。

第二类坑是只看平均值。平均延迟下降，不代表用户体验更好。少量请求如果掉到特别慢的尾部，用户照样会投诉，所以必须同时看 `p50/p95/p99`。这里的 `p95/p99` 可以理解成“最慢的后 5% / 1% 请求处在什么水平”。

第三类坑是升级链太长。比如小模型失败后升中模型，中模型再升大模型，再走人工审核。链路一长，累计超时、重复 token 开销和调试复杂度都会迅速增加。多数业务应限制最多一到两次升级。

第四类坑是分布漂移。分布漂移是指线上请求类型变了，路由器却还按旧数据判断。比如原本系统处理普通客服问答，后来大量接入政策问答和代码问答，旧路由器的判断边界就会失效。解决方法是定期重训、做回放评估和 shadow traffic。

常见坑可以汇总为：

| 常见坑 | 直接后果 | 常见做法 |
| --- | --- | --- |
| 阈值没校准 | 高风险请求误分流 | 用独立验证集调参 |
| 只看均值 | 掩盖尾部慢请求 | 同时看 `p95/p99` |
| 升级链过长 | 延迟和复杂度失控 | 限制最多一到两次升级 |
| 分布漂移 | 线上质量下降 | 定期重训与回放评估 |

监控指标至少要覆盖下面这些项：

| 指标 | 说明 |
| --- | --- |
| 升级率 | 有多少请求被送去大模型 |
| 命中率 | 路由是否把复杂请求送对路径 |
| `p50/p95/p99` 延迟 | 平均与尾部性能 |
| 人工接管率 | 自动链路失败后需要人工处理的比例 |
| 回退率 | fallback 被触发的比例 |
| 质量评分 | 人评、自动评测或业务 KPI |

工程上还有一个重要权衡：路由器越复杂，本身越贵。若路由器成本接近小模型，甚至需要额外大模型来判断是否升级，那么编排收益会被吃掉。路由器应尽量便宜、稳定、可解释，否则它自己会成为新的瓶颈。

---

## 替代方案与适用边界

多模型编排不是唯一方案，它只是“按请求分流”的一种外部架构。实际系统里，至少还有三类常见替代路线。

第一类是单大模型。优点是简单、稳定、调试成本低，行为也更一致；缺点是成本高、延迟高。若你的请求类型非常统一，而且几乎都需要复杂推理，单大模型反而更稳。

第二类是规则系统。优点是便宜、可控、可解释；缺点是泛化差，覆盖范围有限。很多 FAQ、表单校验、固定审核场景，用规则就足够，不一定值得引入模型路由。

第三类是 MoE。MoE 即 Mixture of Experts，可以理解为模型内部的专家分流机制。它和多模型编排相似之处在于“按输入选择专家”，不同之处在于 MoE 发生在单个模型内部，是训练和推理架构的一部分；多模型编排则是系统外部的服务编排。

还要把缓存和检索增强区分开。缓存解决的是重复请求；检索增强解决的是知识注入；多模型编排解决的是算力分配。三者可以组合，但不是一回事。

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 单大模型 | 简单稳定 | 成本高 | 请求统一且普遍复杂 |
| 规则系统 | 便宜可控 | 泛化差 | 流程固定、答案模式稳定 |
| 多模型编排 | 兼顾成本和质量 | 依赖路由准确性 | 请求复杂度分布不均 |
| MoE | 分流能力内置于模型 | 训练和部署复杂 | 大规模模型体系 |
| 缓存 / 检索增强 | 降低重复成本 / 提升知识性 | 不能替代路由 | 与其他方案配合使用 |

什么时候不该用多模型编排，可以用一张判断清单概括：

| 判断项 | 若成立，通常不建议优先采用 |
| --- | --- |
| 请求类型单一 | 分流价值低 |
| 错误代价极高 | 误路由不可接受 |
| 无法积累足够路由数据 | 阈值难校准，学习器难训练 |
| 线上链路已足够便宜 | 节省空间有限，复杂度不值得 |

因此，多模型编排最适合“简单请求占大头，复杂请求占少数，但复杂请求又不能答错”的系统。它的前提不是模型多，而是请求分布有明显层次；它的上限不是模型最强，而是路由足够准。

---

## 参考资料

下表给出一组覆盖理论、路由训练、开源实现、工程配置和架构类比的资料：

| 资料 | 用途说明 |
| --- | --- |
| FrugalGPT | 理论背景，说明为什么多模型链路可以同时降成本并维持效果 |
| RouteLLM | 路由训练方法，说明如何学习“该不该升级” |
| RouteLLM GitHub 源码 | 开源实现，便于看真实代码结构与实验配置 |
| Hugging Face Chat UI - LLM Router 文档 | 工程配置，便于理解 `primary_model`、`fallback_models` 等落地方式 |
| Sparsely-Gated Mixture-of-Experts Layer | 架构类比，帮助区分系统外部编排与模型内部专家分流 |

1. [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://huggingface.co/papers/2305.05176) - 适合建立多模型级联为何能省钱的理论直觉。
2. [RouteLLM: Learning to Route LLMs with Preference Data](https://huggingface.co/papers/2406.18665) - 适合理解路由器如何用偏好数据学习升级策略。
3. [RouteLLM GitHub 源码](https://github.com/lm-sys/RouteLLM) - 适合查看可复现实验、数据处理和评测实现。
4. [Hugging Face Chat UI - LLM Router 文档](https://huggingface.co/docs/chat-ui/en/configuration/llm-router) - 适合理解真实产品里如何配置主模型、路由和 fallback。
5. [Google Research: Sparsely-Gated Mixture-of-Experts Layer](https://research.google/pubs/outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer/) - 适合理解与 MoE 的关系和边界。
