## 核心结论

Agent 评测不能只看成功率。成功率高，意思是“做对的比例高”；但在生产环境里，真正要付钱的是 Token、API 调用次数、等待时间，以及失败后的重试和人工复核。

把评测从“单指标排行榜”改成“多指标决策”后，最有用的视角是帕累托前沿。帕累托前沿可以理解为“一组谁都不能在所有维度同时更好”的候选：如果一个 Agent 成功率更高、Token 更少、API 调用更少、延迟还更低，那么另一个 Agent 就没有继续比较的必要。

一个实用的起点是成本效率指标：

$$
E=\frac{SR}{T/1000}
$$

其中 $SR$ 是成功率，$T$ 是平均 Token 消耗。它表示“每千 Token 带来多少有效成功”。这个指标不是为了替代成功率，而是为了回答另一个问题：同样的预算，谁更值。

在 2026 年 2 月公开讨论的 SWE-bench Verified 相关数据里，Claude Opus 4.6 与 MiniMax M2.5 的成功率大约是 80.8% 和 80.2%，差距只有 0.6 个百分点；但 AgentPMT 文中给出的输入 Token 价格量级分别约为每百万 $5 与 $0.15，成本差约 33 倍。这类例子说明，成功率微增并不自动等于更优，尤其当代价是数量级级别的成本放大时。

下面这张表把“性能”和“资源”放到同一张纸上看：

| Agent/模型 | 成功率 SR | Token 成本示意 | API 调用 | 平均延迟 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Claude Opus 4.6 | 80.8% | 高 | 中 | 中 | 成功率高，但成本压力大 |
| MiniMax M2.5 | 80.2% | 很低 | 中 | 中 | 成功率略低，但成本效率可能显著更好 |
| 某低配 Agent | 72% | 中 | 高 | 高 | 排行不高，且资源消耗不优 |
| 某优化型 Agent | 79% | 低 | 低 | 低 | 常落在预算友好的前沿上 |

结论可以压缩成一句话：Agent 评测要同时比较成功、Token、API、延迟，真正的“最优”不是单点最高成功率，而是在预算约束下位于帕累托前沿的方案。

---

## 问题定义与边界

先把问题说清楚。这里讨论的不是“哪个 Agent 学术上最强”，而是“在给定预算下，哪个 Agent 更适合上线”。预算包括四类：

| 预算边界 | 含义 | 直接影响 |
| --- | --- | --- |
| Token cap | 单次任务最多可消耗多少 Token | 限制上下文长度、重试次数 |
| API 限额 | 单任务最多允许多少次模型/工具调用 | 限制链路深度与多 Agent 协调 |
| 超时 | 最大允许延迟 | 限制搜索、反思、长链推理 |
| 人工复核成本 | 失败后的人力补救成本 | 决定低成功率方案是否真的便宜 |

这四项经常一起作用。因果链可以写成：

1. 成功率决定失败比例。
2. 失败比例决定重试次数和人工介入次数。
3. 重试和人工介入反过来放大 Token、API 与延迟。
4. 最终决定的不是“单次调用便不便宜”，而是“每个成功结果总共花多少钱”。

玩具例子先看一个最小版本。

假设 Agent A 成功率 80%，每次用 100k Token；Agent B 成功率 82%，每次用 500k Token。只看成功率，B 更好；但按每千 Token 的成功贡献：

- A: $80 / 100 = 0.8$
- B: $82 / 500 = 0.164$

也就是说，B 多拿了 2 个百分点成功率，却把 Token 开销放大到 5 倍，效率反而明显下降。

真实工程例子更能说明边界。客服 Agent 中，低成功率模型看起来 Token 花费低，但如果每次失败都需要人工复审 15 分钟，而人工成本按 $100/小时 计算，那么一次失败就是 $25 的人工成本。Agents Squads 给出的案例更保守，按 15 分钟、$100/小时、每次失败约 $10 的复核成本估算，低成功率方案会让“便宜的 Token”变成“昂贵的结果”。因此，这套方法适用于明确受预算约束的生产环境，不适用于完全忽略成本、只追求最高上限的研究性测试。

---

## 核心机制与推导

### 1. 基础指标

设：

- $SR$：成功率，表示任务完成比例
- $T$：平均 Token 消耗，表示一次任务平均使用的输入加输出 Token
- $C$：平均 API 调用次数，表示模型与工具的总交互轮数
- $L$：平均延迟，表示任务端到端完成时间

最基础的成本效率指标是：

$$
E=\frac{SR}{T/1000}
$$

如果 $SR$ 用百分数表示，$E$ 的含义就是“每千 Token 支撑的成功率点数”。

但生产里光看 Token 不够，因为 API 次数和延迟也会占预算，所以可扩展为：

$$
E'=\frac{SR}{(T+w_cC+w_lL)/1000}
$$

这里 $w_c$ 和 $w_l$ 是权重，用来把 API 次数和延迟折算进同一成本尺度。它们不是物理常数，而是预算偏好：

- 如果你的瓶颈是供应商 QPS 限额，$w_c$ 应该更大。
- 如果你的瓶颈是 SLA 超时，$w_l$ 应该更大。
- 如果你做离线批处理，$w_l$ 可以更小。

### 2. 手算一个例子

给定模型 A：

- $SR=80$
- $T=200000$
- $C=30$
- $L=1.2$
- $w_c=0.5$
- $w_l=0.2$

则：

$$
E'_A=\frac{80}{(200000+0.5\times30+0.2\times1.2)/1000}
$$

$$
E'_A \approx \frac{80}{200.01524}\approx 0.400
$$

模型 B：

- $SR=81$
- $T=400000$
- $C=20$
- $L=0.9$

则：

$$
E'_B=\frac{81}{(400000+0.5\times20+0.2\times0.9)/1000}
$$

$$
E'_B \approx \frac{81}{400.01018}\approx 0.202
$$

虽然 B 的成功率更高 1 个点，但它的综合效率只有 A 的约一半。若你的预算主要受 Token 约束，A 更可能位于前沿；若你极端重视成功率且不在乎 Token，B 才有机会胜出。

### 3. 帕累托前沿怎么判定

一个模型被另一个模型“支配”，意思是后者至少在一个维度更好，并且在其他维度不差。对我们的问题，可以理解成：

- 成功率更高或相等
- Token 更低或相等
- API 次数更低或相等
- 延迟更低或相等
- 且至少一项严格更好

满足这个条件时，被支配模型就不该出现在候选集中。

可以把前沿想成下面这种草图：

| 模型 | SR 高 | Token 低 | API 低 | 延迟低 | 是否可能在前沿 |
| --- | --- | --- | --- | --- | --- |
| A | 是 | 是 | 否 | 是 | 是 |
| B | 是 | 否 | 是 | 是 | 是 |
| C | 否 | 否 | 否 | 否 | 否 |
| D | 否 | 是 | 是 | 否 | 视预算而定 |

这就是“无赢家区域”的含义：A 和 B 谁都不能全面压过对方，所以两者都可能在前沿上，只是适用于不同预算。

---

## 代码实现

下面给一个可运行的 Python 例子。输入假设是一个 CSV，包含字段：

- `model`
- `success_rate`
- `token_usage`
- `api_calls`
- `latency`

代码会做三件事：

1. 计算 $E'$。
2. 找出四维意义上的非支配解。
3. 输出按效率排序的帕累托前沿。

```python
import csv
import io
from math import isclose

CSV_TEXT = """model,success_rate,token_usage,api_calls,latency
Claude Opus 4.6,80.8,420000,28,1.30
MiniMax M2.5,80.2,180000,30,1.25
Agent-X,79.0,160000,18,0.95
Agent-Y,81.0,400000,20,0.90
Budget-Agent,76.0,90000,12,0.80
"""

W_C = 500.0   # 把一次 API 调用折算成多少“等价 token”
W_L = 20000.0 # 把 1 秒延迟折算成多少“等价 token”

def load_rows(text):
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        sr = float(row["success_rate"])
        t = float(row["token_usage"])
        c = float(row["api_calls"])
        l = float(row["latency"])
        effective_cost = t + W_C * c + W_L * l
        efficiency = sr / (effective_cost / 1000.0)
        rows.append({
            "model": row["model"],
            "success_rate": sr,
            "token_usage": t,
            "api_calls": c,
            "latency": l,
            "effective_cost": effective_cost,
            "efficiency": efficiency,
        })
    return rows

def dominates(a, b):
    better_or_equal = (
        a["success_rate"] >= b["success_rate"] and
        a["token_usage"] <= b["token_usage"] and
        a["api_calls"] <= b["api_calls"] and
        a["latency"] <= b["latency"]
    )
    strictly_better = (
        a["success_rate"] > b["success_rate"] or
        a["token_usage"] < b["token_usage"] or
        a["api_calls"] < b["api_calls"] or
        a["latency"] < b["latency"]
    )
    return better_or_equal and strictly_better

def pareto_frontier(rows):
    frontier = []
    for i, candidate in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i != j and dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier

rows = load_rows(CSV_TEXT)
frontier = pareto_frontier(rows)
frontier_sorted = sorted(frontier, key=lambda x: x["efficiency"], reverse=True)

# 基本断言
assert len(rows) == 5
assert any(r["model"] == "MiniMax M2.5" for r in frontier_sorted)
assert all(r["efficiency"] > 0 for r in rows)
assert isclose(rows[0]["success_rate"], 80.8)

for row in frontier_sorted:
    print(
        row["model"],
        f"SR={row['success_rate']:.1f}%",
        f"E'={row['efficiency']:.3f}",
        f"T={int(row['token_usage'])}",
        f"API={int(row['api_calls'])}",
        f"L={row['latency']:.2f}s",
    )
```

伪输出可以理解为：

| 模型 | SR | E' | 说明 |
| --- | ---: | ---: | --- |
| Budget-Agent | 76.0% | 0.650 | 成功率不最高，但预算效率极强 |
| Agent-X | 79.0% | 0.427 | 综合表现均衡 |
| MiniMax M2.5 | 80.2% | 0.390 | 高成功率且成本相对友好 |
| Agent-Y | 81.0% | 0.195 | 成功率高，但成本压力明显 |

实现层面有两个新手常漏掉的点。

第一，要记录“每次实验”的原始日志，而不是只保留平均值。因为延迟和 Token 往往有长尾，单次极端值会影响实际 SLA。第二，要做多次采样。Agent CI 文档就强调性能评测要关注多次迭代、95 分位延迟、Token 趋势，而不是只看一次跑分。

---

## 工程权衡与常见坑

最常见的误区，是把 Token 成本当成唯一成本。实际工程里，隐性成本通常更大。

### 1. 多 Agent 协调开销

多 Agent 不是免费并行。Agents Squads 的工程文章给出的经验范围是：一次 handoff，也就是上下文交接，常带来 500 到 2000 Token 的额外消耗；路由、摘要也会继续加价。

看一个真实感更强的例子。假设一个 4 步工作流：

1. 检索资料
2. 生成方案
3. 执行变更
4. 审核输出

如果第 3 步失败，就从第 2 步重试，并且每次重试都伴随 2 次上下文切换，共 1500 Token。若失败率是 20%，那么平均到每次请求上的额外 Token 是：

$$
1500 \times 20\% = 300
$$

300 Token 看起来不大，但这是“只算协调，不算重跑正文”的额外成本。如果主链本身也要重跑，真实开销会进一步放大。工作流越长，这类隐性成本越接近“乘法”而不是“加法”。

### 2. 低成功率导致人工接管

低成功率模型有时在账面 API 成本上很漂亮，但失败后会把问题甩给人。客服、代码审查、财务对账这类场景尤其明显。一旦失败需要人工判断，Token 节省很容易被人力成本淹没。

### 3. 没有预算告警和分段重试

如果系统只设置“总超时”，没设置 Token 上限、调用次数上限、checkpointed retry，就会发生一个典型问题：最后一步出错，整条链从头再跑。checkpointed retry 的意思是“在关键节点保存状态，失败时从最近检查点恢复”，而不是整链回放。

下面这张表是最常见的坑：

| 常见坑 | 表现 | 结果 | 缓解措施 |
| --- | --- | --- | --- |
| 多 Agent context switch 过多 | 每步都传整段历史 | Token 暴涨、延迟累积 | 只传摘要，缩短 handoff |
| 失败重试无检查点 | 一步失败整链重跑 | 成本成倍放大 | checkpointed retry |
| 只看成功率 | 上线后 API 账单超标 | 预算失控 | 同时监控 SR、Token、API、Latency |
| 只看平均值 | 长尾请求拖垮 SLA | 用户体验不稳 | 看 P95/P99 延迟 |
| 只看单 Agent | 忽略协同成本 | 误判多 Agent 收益 | 记录每次 handoff 的 Token |

生产监控面板至少应同时展示四个指标：成功率、平均/分位 Token、API 次数、延迟。少一个，决策都会偏。

---

## 替代方案与适用边界

并不是所有系统都能完整拿到 Token、调用次数和延迟。有些平台只给你账单，有些只给你请求日志。这时可以退一步。

### 1. Token 数据不完整时

如果只有 API 调用次数和平均每次响应 Token，可以近似估算：

$$
T \approx N_{call} \times \bar{t}
$$

其中 $N_{call}$ 是调用次数，$\bar{t}$ 是单次平均 Token。再代入近似效率：

$$
\hat{E}=\frac{SR}{(N_{call}\times \bar{t})/1000}
$$

这对单链路 Agent 还算有效，但对多 Agent 系统偏差会变大，因为 handoff、路由、总结这类中间开销往往不会被平均值很好描述。

### 2. 延迟优先场景

如果你做的是超低延迟系统，比如实时语音中断判断、100ms 内必须响应的在线风控，那么 Token 成本可能不是主导项。这时更合适的策略是：

- 先用成功率过滤掉明显不可用的方案
- 再用 P99 latency 做主排序
- 成本效率指标只作为参考，不做主决策

### 3. 不适用的边界

成本效率指标不适合下面两类任务：

- 研究性突破实验：目标是探索能力上限，不关心预算。
- 极低频高价值任务：比如一年只跑几十次，但每次都值几万美元，这时成功率本身权重更大。

三种常见策略可以放在一起比较：

| 策略 | 核心指标 | 适用场景 | 局限 |
| --- | --- | --- | --- |
| 完整版帕累托 | SR + Token + API + Latency | 生产级 Agent 选型 | 采集成本高 |
| Token-estimated | SR + 估算 Token + API | 日志不完整的平台 | 多 Agent 偏差较大 |
| Latency-first | SR + P99 latency | 实时系统 | 可能忽略长期账单 |

所以，帕累托前沿不是唯一方法，但它是“预算约束下做上线决策”最稳妥的方法之一。

---

## 参考资料

1. [AgentPMT, *Twelve Frontier Models. 0.8 Points Apart. The Moat Moved.*, 2026-02-20](https://www.agentpmt.com/articles/twelve-frontier-models-0-8-points-apart-the-moat-moved-2)  
用于说明 2026 年 2 月 SWE-bench Verified 头部模型成功率已高度接近，而价格差距仍显著存在。

2. [Agent CI Documentation, *Performance Evaluations*, 获取于 2026-03-20](https://agent-ci.com/docs/evaluations/performance/)  
用于支持生产评测应同时记录 latency、token usage、迭代次数与分位统计，而不是只看功能是否通过。

3. [MGX, *Token-Efficient Agent Planning: Foundational Concepts, Advanced Techniques, and Future Directions*, 获取于 2026-03-20](https://mgx.dev/insights/token-efficient-agent-planning-foundational-concepts-advanced-techniques-and-future-directions/1d13cc92cab043b4b8049c73a9238739)  
用于支持 token efficiency、cost-of-pass、成功率与资源消耗联立分析的思路。

4. [Agents Squads, *Token Economics: From Cost Centers to Value Centers*, 获取于 2026-03-20](https://agents-squads.com/research/token-economics-value-centers/)  
用于说明低成功率方案会因为人工复核与重做成本而在真实业务里变贵，并提供多 Agent 协调的开销经验值。

5. [Agents Squads, *Context Engineering for Multi-Agent Systems*, 获取于 2026-03-20](https://agents-squads.com/research/context-engineering-multi-agent/)  
用于支持多 Agent handoff、上下文预算、按角色分配预算等工程实践。
