## 核心结论

多模型服务的本质，是把多个模型当成一组可以被统一调度的能力单元。路由负责回答“这次请求该给谁处理”，版本管理负责回答“当前线上到底跑的是哪一版、怎么安全升级、失败后怎么退回去”。

对初级工程师来说，可以先记住三个判断：

| 维度 | 要解决的问题 | 直接价值 |
| --- | --- | --- |
| 智能选型 | 同一个入口下，哪一个模型最适合当前请求 | 提升质量、降低延迟、控制成本 |
| 并发治理 | 多个模型同时在线时，如何分流、限流、故障切换 | 提升可用性，避免单点过载 |
| 统一认证 | 谁可以调用模型、调用了什么、是否留痕 | 降低凭证泄漏风险，方便审计 |

一个新手友好的例子是客服平台。用户发来“我要修改订单地址”，系统不会先盲目调用最大最贵的模型，而是先判断这是“对话 + 订单知识查询”任务，再结合延迟要求、成功率、预算、合规要求，对候选模型打分，最后把请求送到最合适的一路。外层网关再统一做认证、权限、审计日志记录。这样做的意义不在于“接了很多模型”，而在于“把模型当成可治理的生产资源”。

再把这件事压缩成一句更容易记住的话：

$$
\text{多模型服务} = \text{可解释路由} + \text{可回滚版本} + \text{统一治理}
$$

三个部分缺一不可：

| 组成 | 如果缺失会怎样 |
| --- | --- |
| 可解释路由 | 线上请求为什么命中某模型说不清，问题难排查 |
| 可回滚版本 | 新版出问题时退不回去，故障持续扩大 |
| 统一治理 | 鉴权、审计、密钥、配额散落在各业务里，后续失控 |

---

## 问题定义与边界

多模型服务讨论的不是“能不能接多个 API”，而是“多个模型如何在一个稳定入口下被可控地选择、发布和回退”。这里的“路由”是指根据请求特征把请求送到合适模型；这里的“版本管理”是指明确记录每个模型版本及其切换方式，保证线上可验证、可回滚。

术语先做一句白话解释：

- 请求特征：这次请求本身长什么样，比如它是客服问答、代码生成，还是文档总结。
- 路由策略：决定“选哪个模型”的规则。
- 语义化版本：用 `major.minor.patch` 这样的编号表示大改动、小功能、修复补丁。
- 回滚：上线后发现有问题时，快速切回旧版本。
- 模型注册表：记录模型资产信息的台账，至少要知道“这版模型从哪里来、现在在哪跑、能不能上线”。
- 灰度发布：只让一小部分流量先走新版本，先观察，再决定是否放量。

这类系统的核心边界有四个。

第一，单次请求通常只应命中一个最终执行模型。可以有预处理模型、分类器、守卫模型，但真正产出主结果的执行路径必须清晰，否则难以解释错误来源。

第二，路由不是任意分发，而是在明确目标函数下做选择。目标函数可能是质量优先、成本优先、延迟优先，不能三者都说重要却没有排序。

第三，版本管理不只管理“权重文件”。如果只存模型文件、不存特征定义、提示模板、依赖版本、数据快照，那么“同一版本”在不同机器上可能不是同一个东西。

第四，多模型服务强调统一治理。也就是说，认证、访问控制、审计、限流、配额最好集中在网关层，而不是散落在每个业务服务里。

下面这张表可以把决策输入看清楚：

| 决策输入 | 典型示例 | 处理方式 |
| --- | --- | --- |
| 任务特征 | 对话/摘要/代码、上下文长度、行业领域 | 先做分类，再缩小候选模型集合 |
| 性能指标 | P95 延迟、成功率、超时率、单位成本 | 实时打分，动态调整权重 |
| 业务规则 | 优先国内模型、预算上限、VIP 用户优先 | 作为硬约束或加分项插入路由 |
| 合规要求 | 数据不得出境、敏感请求必须审计 | 先过滤不合规模型，再参与评分 |

玩具例子可以非常简单。假设一个系统只有三个模型：

- `model-a`：快，但质量一般
- `model-b`：慢一点，但质量高
- `model-c`：便宜，但只能处理短文本

如果请求是“50 字以内的简单分类”，那就优先考虑便宜模型；如果请求是“长上下文合同总结”，那就把 `model-c` 直接排除。这个过程不是玄学，而是先过滤，再排序。

真实工程里，规则会更具体。例如一家企业内部 AI 平台同时接入代码模型、通用问答模型、知识检索增强模型。平台要求：

- 代码问题优先走代码专用模型
- 响应时间必须低于 300 ms 的请求不能走慢模型
- 含敏感字段的请求只能走私有部署模型
- 普通用户默认走成本更低的模型，VIP 可以走质量更高的模型

这就是多模型服务的实际边界：不是追求“模型越多越好”，而是让每个请求在可解释、可约束、可回退的前提下落到最合适的模型。

为了避免新手把“路由”和“负载均衡”混为一谈，可以直接区分：

| 概念 | 回答的问题 | 决策依据 |
| --- | --- | --- |
| 模型路由 | 这次请求应该选哪一类模型 | 任务类型、预算、延迟、合规、质量目标 |
| 负载均衡 | 已经选中的这类模型，具体打到哪个实例 | 实例健康状态、连接数、流量分布 |
| 版本发布 | 当前流量应该落到哪个版本 | 发布计划、验证结果、回滚条件 |

这三个概念经常出现在同一套系统里，但不是同一层问题。

---

## 核心机制与推导

多模型服务通常可以拆成两条主线：模型路由和版本管理。

先看路由。最常见的做法不是直接写死 `if-else`，而是把多个因素量化成一个分数。分数越高，模型越优先。一个常见形式是：

$$
score(m, r) = w_s \cdot S_m + w_l \cdot L_m + w_q \cdot Q_m + B(m, r) - P(m, r)
$$

其中：

- `m` 表示候选模型，`r` 表示当前请求
- `S_m` 表示模型成功率
- `L_m` 表示延迟得分
- `Q_m` 表示质量得分
- `B(m, r)` 表示业务加分，例如 VIP 请求偏好高质量模型
- `P(m, r)` 表示惩罚项，例如超预算、超时风险、跨境限制
- `w_s + w_l + w_q = 1`，表示主指标权重之和为 1

如果把它写成一个初学者更容易代入的具体式子，可以得到：

$$
score = success\_rate \times 0.4 + latency\_score \times 0.3 + quality\_score \times 0.2 + priority\_bonus - risk\_penalty
$$

这里各项含义分别是：

- `success_rate`：成功率，白话说就是请求能不能稳定返回结果。
- `latency_score`：延迟得分，白话说就是模型响应够不够快。
- `quality_score`：质量得分，白话说就是结果是否更准确、更符合业务要求。
- `priority_bonus`：优先级加成，白话说就是业务规则给某些模型额外加分。
- `risk_penalty`：风险惩罚，白话说就是不完全合适但又没到必须排除时，要扣一点分。

为什么不直接比延迟或直接比质量？因为生产系统里目标是多目标优化。一个极快但经常失败的模型没有价值，一个极准但价格过高、延迟过大的模型也不一定适合所有请求。路由的本质，是在业务目标下对多个指标做加权排序。

为了让公式真正能落地，通常还要先做“硬过滤”，再做“软排序”：

$$
Candidate(r) = \{m \mid constraint_i(m, r)=\text{true}, \forall i\}
$$

白话说就是：先把一定不能用的模型排除掉，只在剩余候选集里比较分数。这样做有两个好处：

- 合规条件不会被高分模型“意外冲掉”
- 候选集变小后，后续打分更稳定、更容易解释

看一个玩具数值例子。假设请求是“普通用户的 500 字知识问答”，目标是“延迟不超过 300 ms，预算偏低”。现在有两个候选模型：

| 指标 | `cheap-general` | `private-rag` |
| --- | --- | --- |
| 成功率 `S` | `0.96` | `0.98` |
| 延迟得分 `L` | `0.92` | `0.84` |
| 质量得分 `Q` | `0.78` | `0.88` |
| 优先级加成 `B` | `0.07` | `0.02` |
| 风险惩罚 `P` | `0.00` | `0.03` |

代入公式：

$$
score(\text{cheap-general}) = 0.96 \times 0.4 + 0.92 \times 0.3 + 0.78 \times 0.2 + 0.07 - 0.00 = 0.886
$$

$$
score(\text{private-rag}) = 0.98 \times 0.4 + 0.84 \times 0.3 + 0.88 \times 0.2 + 0.02 - 0.03 = 0.810
$$

所以这次请求会命中 `cheap-general`。这个例子说明了两个关键点：

第一，打分必须可解释。你需要知道某模型为什么赢。  
第二，指标必须可更新。线上表现变差后，分数要能自动下降，而不是永远写死。

再看版本管理。语义化版本常写成：

$$
V = major.minor.patch
$$

含义如下：

| 字段 | 含义 | 典型场景 |
| --- | --- | --- |
| `major` | 破坏性变更 | 输入输出协议变化、提示模板结构变化 |
| `minor` | 向后兼容的新能力 | 新增工具调用、支持更长上下文 |
| `patch` | 修复与小改动 | 修 bug、修参数、调阈值 |

但在模型服务里，仅有 `major.minor.patch` 还不够。因为一个“模型版本”往往由多个对象共同组成，可以把它表示为：

$$
Release = (weights,\ tokenizer,\ prompt,\ feature\_code,\ runtime,\ index,\ config)
$$

这条式子的意思是：真正需要被发布和回滚的，不是单个权重文件，而是一整套可复现运行单元。只要其中一个组件变化，行为就可能变化。

可以把一个完整版本对象拆成下面这些字段：

| 组成部分 | 为什么必须纳入版本 |
| --- | --- |
| 权重文件 | 直接决定推理能力 |
| tokenizer | 同样输入可能被切成不同 token |
| 提示模板 | 指令结构变化会直接改变输出 |
| 特征预处理代码 | 输入格式和字段处理逻辑可能变化 |
| 推理镜像与依赖 | 同一权重在不同运行环境里结果可能不同 |
| 检索索引版本 | RAG 系统的答案质量依赖索引内容 |
| 路由配置 | 阈值、候选集、权重变化会改变命中结果 |

所以，多模型服务里的版本管理更接近“发布单元管理”，而不是“文件命名管理”。

线上切换策略通常有三种：

| 切换策略 | 做法 | 适合场景 | 回滚条件 |
| --- | --- | --- | --- |
| A/B 测试 | 新旧版本按流量比例并行 | 比较质量、点击、转化 | 新版关键指标显著变差 |
| 蓝绿发布 | 两套环境并行，一次性切换流量 | 需要快速整体切换 | 切换后立即异常 |
| 滚动更新 | 分批替换实例 | 在线服务实例较多 | 任一批次验证失败 |

这三种方式不是互斥的。很多团队的实际做法是：

1. 先灰度 5% 流量观察错误率和 P95 延迟  
2. 再扩到 20% 或 50%  
3. 指标稳定后再切到全量  
4. 一旦核心指标越过阈值，立即回滚到稳定版

如果把发布流程抽象成状态机，可以写成：

$$
Stable \rightarrow Canary \rightarrow Partial \rightarrow Full
$$

发生异常时：

$$
Canary \rightarrow Stable,\quad Partial \rightarrow Stable,\quad Full \rightarrow Stable
$$

这里最重要的不是状态名字，而是“每一步都必须有进入条件和退出条件”。例如：

| 阶段 | 进入条件 | 退出条件 |
| --- | --- | --- |
| Canary | 离线评估通过，镜像构建完成 | 错误率、P95、质量抽检全部达标 |
| Partial | Canary 稳定运行一段时间 | 指标继续稳定，无新增严重告警 |
| Full | Partial 达标 | 保持监控，准备归档旧版本 |
| Rollback | 任一关键指标越界 | 恢复稳定版并保留事故证据 |

真实工程例子可以这样理解。某平台提供统一 AI Gateway，入口后面挂多个模型，网关先做 Round Robin 或 Failover。Round Robin 是轮询，白话说就是把请求依次分给多个后端；Failover 是故障转移，白话说就是主模型挂了就切到备模型。后端运行时再用多模型推理框架切分 CPU/GPU 资源，避免一个热门模型把所有资源吃光。这样，路由、负载和资源分配形成三层机制：

- 业务层决定该选哪个模型
- 网关层决定如何分发和故障切换
- 运行时决定资源怎么切给每个模型实例

这三层缺一不可。只做路由、不做资源配额，热门模型仍然会把机器拖死；只做部署、不做版本验证，新版模型出错时仍然会全量影响用户。

---

## 代码实现

下面给一个可运行的最小 Python 例子，演示“请求过滤 + 模型打分 + 版本滚动更新 + 回滚”。这段代码可以直接保存为 `multi_model_demo.py` 后执行：

```python
from dataclasses import dataclass, field
from typing import List, Dict, Iterable


@dataclass(frozen=True)
class Request:
    task_type: str
    text_length: int
    max_latency_ms: int
    budget_level: str
    user_tier: str
    contains_sensitive_data: bool = False


@dataclass
class Model:
    name: str
    task_types: List[str]
    max_text_length: int
    deployment_scope: str  # "public" or "private"
    success_rate: float
    p95_latency_ms: int
    quality_score: float
    unit_cost: float
    current_load: float
    version: str


@dataclass
class Instance:
    instance_id: str
    version: str
    healthy: bool = True


@dataclass
class ReleaseRecord:
    version: str
    weights_uri: str
    tokenizer_version: str
    prompt_template_version: str
    runtime_image: str
    feature_schema_version: str
    retrieval_index_version: str
    approved: bool = False
    notes: List[str] = field(default_factory=list)


def latency_score(p95_latency_ms: int, max_latency_ms: int) -> float:
    """
    把延迟转换为 0~1 分数。
    越接近目标延迟，得分越高；明显超出目标，得分快速下降。
    """
    ratio = p95_latency_ms / max_latency_ms
    if ratio <= 0.5:
        return 1.0
    if ratio <= 0.8:
        return 0.9
    if ratio <= 1.0:
        return 0.75
    if ratio <= 1.2:
        return 0.4
    return 0.1


def cost_penalty(unit_cost: float, budget_level: str) -> float:
    if budget_level == "low":
        if unit_cost <= 0.002:
            return 0.0
        if unit_cost <= 0.005:
            return 0.05
        return 0.12
    if budget_level == "medium":
        if unit_cost <= 0.005:
            return 0.0
        return 0.05
    return 0.0


def load_penalty(current_load: float) -> float:
    """
    current_load 取值 0~1，越接近 1 说明越繁忙。
    """
    if current_load < 0.6:
        return 0.0
    if current_load < 0.8:
        return 0.03
    if current_load < 0.9:
        return 0.08
    return 0.15


def eligible(model: Model, req: Request) -> bool:
    if req.task_type not in model.task_types:
        return False
    if req.text_length > model.max_text_length:
        return False
    if req.contains_sensitive_data and model.deployment_scope != "private":
        return False
    return True


def score_model(model: Model, req: Request) -> float:
    l_score = latency_score(model.p95_latency_ms, req.max_latency_ms)
    bonus = 0.0

    # 低预算请求优先低成本模型
    if req.budget_level == "low" and model.unit_cost <= 0.002:
        bonus += 0.05

    # VIP 更偏向质量
    if req.user_tier == "vip" and model.quality_score >= 0.9:
        bonus += 0.05

    # 私有部署模型在敏感请求里额外加分
    if req.contains_sensitive_data and model.deployment_scope == "private":
        bonus += 0.08

    penalty = cost_penalty(model.unit_cost, req.budget_level) + load_penalty(model.current_load)

    score = (
        model.success_rate * 0.4
        + l_score * 0.3
        + model.quality_score * 0.2
        + bonus
        - penalty
    )
    return round(score, 4)


def route_request(req: Request, models: Iterable[Model]) -> Model:
    candidates = [m for m in models if eligible(m, req)]
    if not candidates:
        raise ValueError("no eligible model found")

    ranked = sorted(
        candidates,
        key=lambda model: score_model(model, req),
        reverse=True,
    )
    return ranked[0]


def validate_release(record: ReleaseRecord) -> bool:
    """
    生产系统里这里通常会接离线评估、冒烟测试、依赖检查和审批状态。
    这里做最小可运行示例。
    """
    required_fields = [
        record.weights_uri,
        record.tokenizer_version,
        record.prompt_template_version,
        record.runtime_image,
        record.feature_schema_version,
        record.retrieval_index_version,
    ]
    return record.approved and all(required_fields)


def validate_instances(instances: Iterable[Instance]) -> bool:
    return all(instance.healthy for instance in instances)


def rolling_update(
    instances: List[Instance],
    target_release: ReleaseRecord,
    batch_size: int = 1,
) -> bool:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not validate_release(target_release):
        return False

    original_versions = {instance.instance_id: instance.version for instance in instances}

    for start in range(0, len(instances), batch_size):
        batch = instances[start:start + batch_size]

        for instance in batch:
            instance.version = target_release.version

        if not validate_instances(batch):
            for instance in instances:
                instance.version = original_versions[instance.instance_id]
            return False

    return True


def print_route_result(req: Request, models: List[Model]) -> None:
    print(f"request={req}")
    for model in models:
        if eligible(model, req):
            print(f"  candidate={model.name:<16} version={model.version:<8} score={score_model(model, req)}")
        else:
            print(f"  candidate={model.name:<16} version={model.version:<8} score=SKIP")
    winner = route_request(req, models)
    print(f"winner={winner.name}:{winner.version}")
    print("-" * 60)


def main() -> None:
    models = [
        Model(
            name="cheap-general",
            task_types=["chat", "summary"],
            max_text_length=2000,
            deployment_scope="public",
            success_rate=0.96,
            p95_latency_ms=220,
            quality_score=0.78,
            unit_cost=0.0015,
            current_load=0.45,
            version="1.2.3",
        ),
        Model(
            name="private-rag",
            task_types=["chat", "qa"],
            max_text_length=8000,
            deployment_scope="private",
            success_rate=0.98,
            p95_latency_ms=280,
            quality_score=0.88,
            unit_cost=0.0040,
            current_load=0.35,
            version="2.1.0",
        ),
        Model(
            name="code-specialist",
            task_types=["code"],
            max_text_length=12000,
            deployment_scope="private",
            success_rate=0.97,
            p95_latency_ms=340,
            quality_score=0.93,
            unit_cost=0.0060,
            current_load=0.55,
            version="3.0.1",
        ),
    ]

    normal_chat = Request(
        task_type="chat",
        text_length=500,
        max_latency_ms=300,
        budget_level="low",
        user_tier="normal",
        contains_sensitive_data=False,
    )
    print_route_result(normal_chat, models)

    sensitive_qa = Request(
        task_type="qa",
        text_length=1500,
        max_latency_ms=400,
        budget_level="medium",
        user_tier="vip",
        contains_sensitive_data=True,
    )
    print_route_result(sensitive_qa, models)

    instances = [
        Instance(instance_id="pod-1", version="1.2.3"),
        Instance(instance_id="pod-2", version="1.2.3"),
        Instance(instance_id="pod-3", version="1.2.3"),
    ]

    new_release = ReleaseRecord(
        version="1.2.4",
        weights_uri="s3://models/cheap-general/1.2.4/weights.bin",
        tokenizer_version="tok-v5",
        prompt_template_version="prompt-v3",
        runtime_image="registry.local/infer:1.2.4",
        feature_schema_version="schema-v2",
        retrieval_index_version="index-v7",
        approved=True,
        notes=["offline_eval_passed", "smoke_test_passed"],
    )

    ok = rolling_update(instances, new_release, batch_size=1)
    print(f"rolling_update_success={ok}")
    print(f"versions_after_update={[instance.version for instance in instances]}")

    # 模拟失败回滚
    bad_instances = [
        Instance(instance_id="pod-a", version="2.1.0"),
        Instance(instance_id="pod-b", version="2.1.0", healthy=False),
    ]
    private_release = ReleaseRecord(
        version="2.2.0",
        weights_uri="s3://models/private-rag/2.2.0/weights.bin",
        tokenizer_version="tok-v8",
        prompt_template_version="prompt-v6",
        runtime_image="registry.local/infer:2.2.0",
        feature_schema_version="schema-v4",
        retrieval_index_version="index-v12",
        approved=True,
    )

    ok2 = rolling_update(bad_instances, private_release, batch_size=1)
    print(f"rolling_update_success={ok2}")
    print(f"versions_after_rollback={[instance.version for instance in bad_instances]}")


if __name__ == "__main__":
    main()
```

这段代码表达了六件事。

第一，先过滤再打分。不是所有模型都能处理所有请求，必须先按任务类型、文本长度、敏感数据约束等条件排除不合格模型。

第二，分数里同时编码指标与业务规则。比如低预算请求给低成本模型加分，VIP 请求给高质量模型加分，高负载实例对应模型会被扣分。

第三，延迟不是直接拿毫秒值比较，而是先转换成可统一加权的得分。这样它才能和成功率、质量分一起参与排序。

第四，滚动更新是一批一批替换，而不是瞬间全量替换。只要某一批验证失败，就立即停止并回滚。

第五，版本发布前必须校验发布单元是否完整。示例里的 `ReleaseRecord` 故意把权重、tokenizer、模板、镜像、特征、索引都放进来，就是为了说明“版本”是一个组合对象。

第六，主程序里同时演示了两类请求。普通低预算请求会偏向 `cheap-general`，敏感请求会强制进入私有模型候选集。这比单个断言更接近真实系统。

如果你第一次接触这类代码，可以按下面顺序理解：

| 代码块 | 作用 | 初学者要看懂什么 |
| --- | --- | --- |
| `eligible` | 做硬过滤 | 哪些模型根本不能参与竞争 |
| `score_model` | 做软排序 | 多指标如何合并成一个分数 |
| `route_request` | 选赢家 | 为什么最终只返回一个执行模型 |
| `ReleaseRecord` | 描述版本单元 | 为什么版本不等于单个权重文件 |
| `rolling_update` | 发布与回滚 | 为什么升级不能一步到位 |

真实工程里，还需要一个模型注册表。注册表可以理解成“模型资产台账”，至少要存这些字段：

| 元数据字段 | 用途 |
| --- | --- |
| 模型名 | 作为路由与监控中的唯一标识 |
| 版本号 | 明确线上具体运行版本 |
| 训练代码版本 | 保证重新训练和排查可追溯 |
| 特征定义 | 保证预处理与训练时一致 |
| 数据快照 | 明确这版模型基于哪份数据产生 |
| tokenizer/依赖版本 | 避免同权重不同运行结果 |
| 部署镜像 | 保证运行环境一致 |
| 路由标签 | 声明该模型适合哪些任务与约束 |
| 审批状态 | 标识是否允许进入生产环境 |
| 回滚目标 | 指明故障时退回哪一个稳定版本 |

一个更贴近业务的真实工程例子是企业知识助手。入口请求先由网关鉴权，再做敏感等级识别。如果请求涉及内部数据，路由只允许私有模型候选集参与评分；如果是公开问答，则可以放开到低成本外部模型。模型选定后，系统把请求 ID、调用者、模型版本、耗时、结果状态统一打入审计日志。这样，当某天出现投诉“同样的问题昨天能答、今天不能答”时，工程团队能直接查到：昨天命中的是 `private-rag:2.0.4`，今天因为延迟阈值变更命中了 `cheap-general:1.2.3`。这就是可观测性，白话说就是“出问题时能追得回去”。

建议最少记录下面这些审计字段：

| 字段 | 作用 |
| --- | --- |
| `request_id` | 串起一整次调用链 |
| `caller_id` | 知道是谁在调用 |
| `route_decision` | 知道路由为什么这么选 |
| `model_name` | 知道命中了哪个模型 |
| `model_version` | 知道命中了哪个具体版本 |
| `latency_ms` | 判断性能是否异常 |
| `result_status` | 成功、失败、超时、降级 |
| `trace_id` | 与链路追踪系统对齐 |

---

## 工程权衡与常见坑

多模型服务最大的误区，是把它理解成“接更多模型就更先进”。实际上，它引入的是治理能力，同时也引入复杂度。复杂度来自更多指标、更多版本、更多资源类型、更多失败路径。

先看典型权衡：

| 权衡点 | 倾向 A | 倾向 B | 典型决策 |
| --- | --- | --- | --- |
| 质量 vs 成本 | 高质量模型更贵 | 低成本模型质量可能一般 | VIP、关键任务走高质量，普通请求走低成本 |
| 延迟 vs 稳定性 | 快模型可能波动大 | 稳模型可能偏慢 | 实时场景优先快，后台任务优先稳 |
| 统一治理 vs 灵活接入 | 统一网关更规范 | 各团队直连更快上线 | 生产环境优先统一治理 |
| 发布速度 vs 风险控制 | 快速全量 | 分阶段验证 | 默认滚动或 A/B，少用一次性全量 |

常见坑主要有三类。

第一类坑是血统缺失。血统就是模型从数据、代码到部署的一整条来源链。白话说，就是“这版模型到底怎么来的”。很多团队回滚时只回滚了权重，没有回滚特征预处理、提示模板或 tokenizer，结果旧权重配了新预处理，输出仍然异常。正确做法是把模型、特征代码、依赖版本、数据快照作为一个整体版本登记，而不是拆开管理。

第二类坑是凭证散落。每个业务服务自己保存模型 API Key，看起来接入快，实际后续会失控。人员变动时很难统一回收；某个服务被入侵时，也很难判断密钥影响范围。正确做法是让统一网关持有凭证，业务服务只访问内部网关，不直接碰外部模型密钥。

第三类坑是审计断层。很多团队只记录“某接口被调用了”，但不记录“调用者是谁、命中了哪个模型版本、输入输出是否脱敏、失败原因是什么”。出了事故后只能猜。对生产系统来说，审计日志不是附属品，而是治理主干。

可以把这些坑与规避方式对应起来：

| 常见坑 | 典型后果 | 规避措施 |
| --- | --- | --- |
| 血统缺失 | 回滚后结果仍错，无法复现历史行为 | 模型注册表记录权重、特征、数据、依赖、模板 |
| 凭证散落 | 密钥泄漏面大，难统一禁用 | 统一网关做访问控制和密钥托管 |
| 审计断层 | 出问题后无法追责与排障 | 全链路日志记录请求、路由、版本、结果 |
| 只看平均延迟 | 峰值时用户体验很差 | 监控 P95/P99，而不只看均值 |
| 全量替换新版本 | 单次事故影响全体用户 | 先灰度、再放量，失败立即回滚 |
| 路由规则过多 | 维护困难，行为不可解释 | 先硬过滤，再少量核心评分项 |

这里再给一个新手容易理解的失败例子。某团队升级了文本分类模型，只替换了推理权重文件，没有同步旧版分词器。结果上线后短文本看起来没问题，长文本错误率突然上升。表面看像“新模型变差”，实际是输入处理链条不一致。这个问题本质上不是算法问题，而是版本治理问题。

再给两个真实工程里很常见、但初学者经常忽略的坑。

第四类坑是“路由分数漂移”。离线评估时某模型质量高，于是它长期被设为高分；但线上数据分布变了，新请求类型开始偏向另一模型，原来的高分已经不成立。如果路由权重长期不更新，系统会持续选错。正确做法是定期用线上指标校正评分参数，并把“最近 1 小时/1 天”的表现纳入动态特征。

第五类坑是“资源治理缺席”。你可能选对了模型，但这个模型所在 GPU 节点已经很满，排队时间远超推理时间。用户看到的是“模型慢”，实际问题在资源调度。正确做法是把负载、实例健康、队列长度等运行时信号纳入路由或负载均衡层，而不是只盯模型能力本身。

可以把常见故障拆成“在哪一层出错”来理解：

| 层级 | 常见问题 | 表现 |
| --- | --- | --- |
| 业务路由层 | 候选集或评分权重错误 | 请求稳定命中错误模型 |
| 网关层 | 限流、鉴权、故障切换异常 | 大量 401、429、超时 |
| 运行时层 | GPU 饱和、实例不健康、队列积压 | 延迟抖动、吞吐下降 |
| 版本治理层 | 回滚对象不完整、灰度条件缺失 | 发布失败后无法恢复 |

这样拆开后，排障思路会清晰很多：先定位故障层，再定位具体配置或代码，而不是把所有问题都归因于“模型效果不好”。

---

## 替代方案与适用边界

多模型服务不是唯一答案。是否上多模型路由，要看任务种类、变更频率、治理要求和团队工程能力。

如果你的系统只有一个主要任务，比如单一的文档摘要，而且单模型已经满足质量、延迟和成本要求，那么直接使用单模型通常更合适。因为每增加一个路由层、版本层、监控层，都会带来新的维护成本。

如果你的系统已经出现明显的任务分化，比如同时有对话、知识问答、代码生成、搜索增强，并且不同请求对质量、成本、延迟要求差异很大，那么多模型路由通常值得做。

还有一种替代方案是“外部编排优先”。也就是先由一个统一器负责拆解任务，再调用外部模型或工具。这适合工具链复杂但模型种类不多的场景。

下面做一个直接对比：

| 方案 | 适用条件 | 运维复杂度 | 治理能力 | 局限 |
| --- | --- | --- | --- | --- |
| 多模型路由 | 多任务、多约束、追求高可用和成本优化 | 高 | 强 | 路由与版本管理复杂 |
| 单模型方案 | 任务单一、变更少、团队小 | 低 | 中 | 难兼顾多种任务目标 |
| 外部编排方案 | 工具调用复杂、模型本身差异不大 | 中 | 中 | 对模型版本治理覆盖有限 |

可以用一个初学者视角的判断方法：

- 只有一个模型，而且它能稳定满足大多数场景，先不要急着做路由。
- 开始出现“不同请求明显适合不同模型”的情况，再引入候选集过滤和评分。
- 当上线失败、权限混乱、版本不可追溯成为问题时，再补齐注册表、网关治理和回滚机制。

如果希望判断得更具体，可以再加一张“是否值得上多模型服务”的检查表：

| 判断问题 | 如果答案是“是” | 结论倾向 |
| --- | --- | --- |
| 是否同时存在多种任务类型 | 对话、代码、总结、检索明显不同 | 更适合多模型 |
| 是否存在明显的成本分层 | 免费用户、付费用户、内部员工要求不同 | 更适合多模型 |
| 是否有合规隔离要求 | 敏感数据只能走私有部署 | 更适合多模型 |
| 是否已经频繁发布模型版本 | 每周甚至每天更新 | 更需要版本治理 |
| 团队是否缺少平台能力 | 没有统一网关、没有监控、没有审计 | 先别急着上复杂路由 |

也就是说，多模型服务最适合中大型平台，尤其是同时对接多个模型能力、需要集中治理、还必须保证线上稳定性的场景。它不适合“为了显得先进而堆架构”的项目。架构应该服务于问题规模，而不是反过来制造问题。

一个实用落地顺序通常是：

1. 先做单模型 + 基础监控  
2. 再做候选集过滤  
3. 再做评分路由  
4. 再做统一网关、注册表、灰度发布  
5. 最后再做更复杂的动态权重和自动回滚

这样推进的好处是，每一步都对应明确问题，不会在还没有流量和治理痛点时就先把系统做复杂。

---

## 参考资料

| 资料 | 用途 |
| --- | --- |
| [Semantic Versioning 2.0.0](https://semver.org/) | 语义化版本的正式规范，明确 `major.minor.patch` 的定义与边界 |
| [WSO2 AI Gateway: Multi-Model Routing](https://apim.docs.wso2.com/en/4.6.0/ai-gateway/multi-model-routing/overview/) | 展示网关层多模型路由的常见策略，包括 Round Robin、Weighted Round Robin、Failover |
| [KServe ModelMesh Overview](https://kserve.github.io/website/latest/modelserving/mms/modelmesh/overview/) | 说明多模型推理服务中的模型放置、请求转发、资源利用与稳定端点 |
| [KServe ModelMesh Installation / Admin Guide](https://kserve.github.io/website/docs/admin-guide/modelmesh) | 补充多模型运行时治理、元数据存储和高密度服务的工程背景 |
| [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry) | 说明模型注册表、版本、别名、血统追踪与生命周期治理 |
| [Argo Rollouts Canary Strategy](https://argo-rollouts.readthedocs.io/en/stable/features/canary/) | 说明灰度发布、逐步放量、暂停观察与失败中止的典型做法 |
| [Istio Canary Upgrades](https://istio.io/latest/docs/setup/upgrade/canary/) | 从服务网格视角解释为什么生产升级应先小流量验证再迁移 |
| [OpenTelemetry Logs Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/) | 说明审计与可观测性中日志字段、TraceId、SpanId 等统一记录方式 |
| [OpenTelemetry Logging](https://opentelemetry.io/docs/specs/otel/logs/) | 说明日志、指标、链路如何关联，适合作为审计链路设计参考 |
