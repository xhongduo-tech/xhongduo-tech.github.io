## 核心结论

层级式多 Agent 的管理者路由策略，本质上是在“准确率、延迟、成本、可控性”之间做约束下的最优化。Manager Agent 可以理解为系统里的总调度器，白话讲，就是专门决定“任务怎么拆、谁来做、做完怎么验收”的那个上层代理。

它通常做三件事：

1. 把一个复杂请求拆成若干边界清晰的子任务。
2. 在多个子 Agent 之间做路由决策。
3. 汇聚结果、检测冲突、触发回退或重试。

如果只看路由，常见有三类策略：

| 路由策略 | 决策依据 | 典型延迟 | 典型优点 | 典型缺点 |
|---|---|---:|---|---|
| 基于规则 | 关键词、正则、枚举条件 | $<1\text{ ms}$ 到几 ms | 快、稳定、可审计 | 覆盖面窄，维护成本随场景增长 |
| 基于语义相似度 | 向量嵌入距离 | 5-15 ms | 比规则灵活，成本低 | 对短句、否定句、边界语义敏感 |
| 基于 LLM 判断 | 大模型读任务后直接分类/分派 | 200-800 ms | 泛化强，适合复杂任务 | 延迟高、成本高、输出不稳定 |

因此，工程上最常用的不是“单一最好路由器”，而是分层路由：

$$
a^*=\arg\max_{i\in A}(c_i-\lambda w_i)
$$

这里 $c_i$ 是候选 Agent 的置信度，白话讲，就是“它做这件事大概率有多靠谱”；$w_i$ 是负载或成本，白话讲，就是“它现在忙不忙、贵不贵、慢不慢”；$\lambda$ 是权衡系数。

结论很直接：低复杂度任务优先规则或语义路由，高模糊度任务再升级到 LLM 路由；当多个子 Agent 结果互相冲突时，Manager 不能直接拼接，而要做评分、校验和冲突检测。

---

## 问题定义与边界

讨论“层级式多 Agent 的管理者路由策略”时，问题边界必须先收紧。这里不讨论完全去中心化协商，也不讨论端到端单模型直接生成全部答案。这里只讨论一种结构：上层有 Manager，下层有若干能力不同的 Worker Agent。

这类系统的输入通常包含四类信息：

| 输入维度 | 含义 | 例子 |
|---|---|---|
| 用户目标 | 最终要交付什么 | “生成一份带引用的竞品分析” |
| 全局约束 | 整个任务必须遵守的限制 | 延迟上限 2 秒，预算 0.02 美元 |
| 子 Agent 能力画像 | 每个 Agent 擅长什么 | 检索型、代码型、写作型、审校型 |
| 系统状态 | 当前资源和历史表现 | 当前负载、失败率、缓存命中率 |

Manager 的目标不是“找到最聪明的 Agent”，而是“在约束内找到最合适的执行路径”。如果任务拆分错误，后面所有路由都会建立在错误前提上；如果路由不考虑预算和时延，系统可能局部最优、整体失控；如果汇聚时没有契约校验，错误会级联放大。

一个玩具例子可以说明边界。

用户说：“帮我写一篇关于 PostgreSQL 索引的入门文章。”

Manager 不能把它直接丢给“写作 Agent”。更合理的拆法是：

1. 检索 Agent 收集概念、索引类型、适用场景。
2. 结构化 Agent 生成提纲。
3. 写作 Agent 生成初稿。
4. 审校 Agent 检查事实一致性与术语统一。

如果系统只做“一个请求对一个 Agent”的扁平路由，那它不是层级式多 Agent，而只是普通分发器。

真实工程里，边界更明显。比如 API 网关每天处理 10 万条请求，Manager 需要在首跳就做预算分级：

| 请求复杂度 | 比例 | 首选路由 | 延迟预算 | 模型成本策略 |
|---|---:|---|---:|---|
| 简单问答 | 60% | 规则/语义路由到轻量模型 | $<300$ ms | 最低成本 |
| 中等复杂任务 | 30% | 语义路由到中型模型 | $<800$ ms | 平衡准确率与成本 |
| 高模糊或高风险任务 | 10% | 升级到 LLM 判断 + 并行评审 | $<2$ s | 允许高成本 |

你可以把这看成一个受 SLO 约束的调度问题。SLO 是服务等级目标，白话讲，就是系统答应业务方“多快、多稳、多准”的那条线。只要 SLO 很严，LLM 路由就不可能成为默认首选。

---

## 核心机制与推导

层级式 Manager 的工作流通常分成四步：任务表征、候选筛选、执行调度、结果汇聚。

### 1. 任务表征

Manager 先把输入转成可计算特征，例如：

- 意图类别
- 是否需要外部检索
- 是否涉及代码执行
- 时延预算
- 风险等级
- 历史相似任务的成功路径

这一步的输出，不是自然语言结论，而是一组结构化信号。因为只有结构化之后，后续路由才可比较、可回放、可优化。

### 2. 候选路由评分

设候选 Agent 集合为 $A=\{a_1,a_2,\dots,a_n\}$，则最常见的一类选择函数是：

$$
a^*=\arg\max_{i\in A}(c_i-\lambda w_i)
$$

其中：

- $c_i$：候选 Agent 对该任务的适配度或置信度
- $w_i$：负载、成本、预计时延的归一化组合
- $\lambda$：系统配置的惩罚系数

这个公式的意思不是“选最强 Agent”，而是“选当前最划算的 Agent”。

例如一个玩具例子：

| Agent | 擅长 | $c_i$ | $w_i$ | $c_i-\lambda w_i$，取 $\lambda=0.3$ |
|---|---|---:|---:|---:|
| 检索 Agent | 找资料 | 0.92 | 0.30 | 0.83 |
| 写作 Agent | 生成文本 | 0.70 | 0.10 | 0.67 |
| 通用 LLM Agent | 泛化强 | 0.95 | 0.90 | 0.68 |

这里最优不是通用 LLM，而是检索 Agent。原因不是它“更聪明”，而是它在这个任务上“高适配且负载可接受”。

### 3. 低置信度升级与并行评审

当最佳候选的置信度仍低于阈值时，单路由不够安全。此时 Manager 会触发并行执行，再用复合评分选最终结果：

$$
s_i=\alpha\cdot Coherence+\beta\cdot Factuality+\gamma\cdot Relevance
$$

这里：

- Coherence：连贯性，白话讲，就是输出前后是否自洽
- Factuality：事实性，白话讲，就是是否和证据或知识库一致
- Relevance：相关性，白话讲，就是是否真正回答了当前子任务

这个阶段常见做法是：先让两个或三个 Agent 并行完成，再让评估器统一打分，而不是直接相信第一个返回结果。

### 4. 汇聚与冲突检测

多 Agent 的难点不在“把结果收回来”，而在“判断这些结果能不能拼在一起”。

如果两个子 Agent 对同一个事实给出不同判断，Manager 需要识别冲突。常见方法之一是用分布差异度量，例如：

$$
D_{KL}(P_i \parallel P_j)=\sum_x P_i(x)\log\frac{P_i(x)}{P_j(x)}
$$

$D_{KL}$ 是 KL 散度，白话讲，就是“两个概率分布差得有多远”。在文本系统里，它不一定直接作用在原始词分布上，也可以作用在结构化标签分布、证据支持分布、候选结论分布上。

如果两个结果的 $D_{KL}$ 很高，说明它们背后的判断模式差异很大，Manager 应该：

1. 不直接汇总。
2. 回到证据层重新核验。
3. 触发某一子任务重跑或引入仲裁 Agent。

真实工程例子是客服自动化平台。用户问：“我想退订昨天升级的企业套餐，但保留历史数据。”  
这个请求至少涉及：

- 账户识别
- 套餐变更规则
- 数据保留政策
- 实际执行动作

如果检索 Agent 说“可以直接退订且保留数据”，策略 Agent 说“该套餐降级会触发数据冻结”，两者不能简单拼接成一句客服话术。Manager 必须先发现冲突，再回查策略文档，否则系统会生成表面流畅、实际违规的结果。

---

## 代码实现

下面给出一个可运行的最小实现。它不是完整生产系统，但覆盖了三个核心动作：效用路由、并行候选评分、冲突检测。

```python
from math import log
from typing import List, Dict


def utility(confidence: float, load: float, lam: float) -> float:
    return confidence - lam * load


def choose_agent(agents: List[Dict], lam: float = 0.3) -> Dict:
    best = max(agents, key=lambda a: utility(a["confidence"], a["load"], lam))
    return best


def composite_score(result: Dict, alpha=0.3, beta=0.5, gamma=0.2) -> float:
    return (
        alpha * result["coherence"]
        + beta * result["factuality"]
        + gamma * result["relevance"]
    )


def kl_divergence(p: List[float], q: List[float]) -> float:
    eps = 1e-12
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        total += pi * log(pi / qi)
    return total


def manager_route(task: Dict, agents: List[Dict]) -> Dict:
    # 第一层：按效用选最优候选
    best = choose_agent(agents, lam=task["lam"])

    # 置信度不足时，并行执行并按复合分数选优
    if best["confidence"] < task["confidence_threshold"]:
        results = task["parallel_results"]
        best_result = max(results, key=composite_score)

        # 汇聚前做简单冲突检测
        for other in results:
            if other is best_result:
                continue
            if kl_divergence(best_result["belief"], other["belief"]) > task["kl_threshold"]:
                return {
                    "status": "conflict",
                    "action": "rerun_or_review",
                    "winner": best_result["agent"]
                }

        return {
            "status": "ok",
            "action": "accept_parallel_result",
            "winner": best_result["agent"]
        }

    return {
        "status": "ok",
        "action": "accept_direct_route",
        "winner": best["name"]
    }


if __name__ == "__main__":
    agents = [
        {"name": "rule_router", "confidence": 0.62, "load": 0.05},
        {"name": "semantic_router", "confidence": 0.76, "load": 0.20},
        {"name": "llm_router", "confidence": 0.91, "load": 0.95},
    ]

    task = {
        "lam": 0.3,
        "confidence_threshold": 0.80,
        "kl_threshold": 0.25,
        "parallel_results": [
            {
                "agent": "writer_a",
                "coherence": 0.88,
                "factuality": 0.82,
                "relevance": 0.90,
                "belief": [0.70, 0.20, 0.10],
            },
            {
                "agent": "writer_b",
                "coherence": 0.84,
                "factuality": 0.89,
                "relevance": 0.86,
                "belief": [0.68, 0.22, 0.10],
            },
        ],
    }

    decision = manager_route(task, agents)
    assert decision["status"] == "ok"
    assert decision["winner"] in {"writer_a", "writer_b", "semantic_router", "llm_router", "rule_router"}
```

这个实现表达了一个重要工程原则：Manager 不应该只返回“谁来做”，还应该返回“为什么这样做，以及接下来要不要复审”。

如果把它映射到真实系统，一般会再加四类能力：

| 模块 | 作用 | 生产环境常见补充 |
|---|---|---|
| Router | 选 Agent | 加缓存、灰度、在线指标 |
| Planner | 拆任务 | 输出 DAG 或任务树 |
| Aggregator | 汇总结果 | 强制 schema 校验 |
| Evaluator | 评分与冲突仲裁 | 接证据库、规则库、审计日志 |

真实工程例子可以是研发 Copilot。一个需求“修复支付回调幂等问题并补测试”会被拆成：

1. 代码检索 Agent 找相关模块。
2. 静态分析 Agent 定位竞态风险。
3. 代码生成 Agent 产出补丁。
4. 测试 Agent 生成回归测试。
5. 审校 Agent 检查补丁是否破坏原有逻辑。

这时 Manager 的价值不在“生成代码”，而在“决定先查什么、谁来改、谁来验、冲突时谁说了算”。

---

## 工程权衡与常见坑

层级式 Manager 最常见的问题，不是理论上的路由公式，而是工程上的失真。

| 常见坑 | 具体表现 | 防护方式 |
|---|---|---|
| 忽略输出契约 | 子 Agent 返回自由文本，后续无法稳定消费 | 强制 schema、字段校验、缺失字段回退 |
| 语义路由样本不足 | 短句、否定句、跨域句子误分派 | 定期补样本，设置低置信度升级 |
| LLM 路由过慢 | 高峰期排队，SLO 超时 | 先规则/语义，再升级到 LLM |
| 子任务语义漂移 | 局部答案看似合理，但整体互相冲突 | 汇聚前做一致性检查和 KL 监控 |
| Manager 单点瓶颈 | 所有请求都卡在上层 | 缓存首跳决策，增加局部子管理者 |

最容易被低估的坑是“格式正确不等于事实正确”。很多团队做了 JSON schema 校验，就以为结果可用了。实际上 schema 只能保证结构稳定，不能保证内容成立。

另一个常见误区是把语义路由当成“便宜版 LLM 路由”。二者不是强弱关系，而是适用边界不同。语义路由擅长把相似任务快速归类，但一旦用户请求包含否定、例外、组合约束，单纯靠向量相似度就容易误判。例如：

- “不要退款，只想取消自动续费”
- “不是要重装数据库，只要清理 WAL 日志”

这种句子在向量空间可能和“退款”“重装数据库”很近，但业务语义刚好相反。解决办法不是把阈值调得更激进，而是把这类请求升级给 LLM 判断或规则白名单处理。

还有一个工程现实：Manager 越聪明，越容易变成瓶颈。因为所有决策都依赖它。解决思路通常不是“让一个更大的 Manager 处理一切”，而是把系统拆成局部子图，例如检索子图、代码子图、写作子图，每个子图再有自己的局部管理者。总 Manager 只决定进入哪个子图，不处理所有细节。

---

## 替代方案与适用边界

层级式多 Agent 不是唯一组织方式。它适合任务可拆解、职责边界清晰、需要统一审计与回退的场景，但并不适合所有问题。

| 组织模式 | 典型场景 | 优点 | 缺点 |
|---|---|---|---|
| Pipeline 顺序流 | 固定流程，如“检索→写作→审校” | 简单、稳定、好维护 | 一步错步步错，回滚差 |
| Hierarchical 层级管理 | 复杂任务拆解、企业流程自动化 | 可控、可审计、易加约束 | Manager 可能成瓶颈 |
| Decentralized 去中心化 | 高动态协同、开放式探索 | 灵活、容错高 | 难统一策略，调试困难 |

如果任务天然是固定流水线，比如“转写音频 -> 摘要 -> 存档”，那么引入复杂 Manager 往往过度设计。  
如果任务高度开放、节点之间需要频繁协商，比如自动科研探索，去中心化结构可能更自然。  
只有当你同时需要“任务拆解、预算控制、统一验收、失败回退”时，层级式 Manager 才最有价值。

可以用一个直观对比来判断：

- Sequential：适合步骤固定的问题。
- Hierarchical：适合目标明确、子任务可拆、需要总控的问题。
- Decentralized：适合环境变化快、局部自治强的问题。

因此，层级式管理者路由并不是“最先进就最好”，而是“在复杂但可治理的任务里最合适”。

---

## 参考资料

- Springer 2024, *A survey on LLM-based multi-agent systems*：介绍层级式多 Agent、Manager 的任务分解与协调作用。  
  https://link.springer.com/article/10.1007/s44336-024-00009-2

- Emergent Mind, *Manager Agent*：总结 Manager 作为顶层编排者的职责，包括协议约束、监控和工作流控制。  
  https://www.emergentmind.com/topics/manager-agent

- Emergent Mind, *Multi-agent LLM Frameworks*：给出路由效用函数、复合评分与冲突检测等形式化描述。  
  https://www.emergentmind.com/topics/multi-agent-llm-frameworks

- Zylos Research 2026, *AI Agent Model Routing*：讨论规则、语义、LLM 路由的延迟与回退策略。  
  https://zylos.ai/research/2026-03-02-ai-agent-model-routing

- LLM Semantic Router 文档：提供多模型分流与成本优化案例。  
  https://llm-semantic-router.readthedocs.io/en/latest/overview/semantic-router-overview/

- Next.gr, *Multi-agent systems with LLM communication*：讨论多 Agent 通信中的冲突、漂移与一致性问题。  
  https://www.next.gr/ai/autonomous-systems/multi-agent-systems-with-llm-communication

- Field Guide to AI, *Multi-agent Systems*：对比 hierarchical、sequential、decentralized 等组织模式。  
  https://fieldguidetoai.com/guides/multi-agent-systems
