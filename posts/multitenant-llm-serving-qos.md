## 核心结论

多租户 LLM Serving 的 QoS 控制，本质是在共享 GPU、KV cache 和内存带宽的前提下，把不同租户的延迟、吞吐和公平性控制在可接受区间，而不是让所有请求都同样快。

先给一句总定义：**QoS（Quality of Service，服务质量）控制，是把“谁先跑、每次跑多少、谁能继续进系统、谁该被限速”变成显式策略。** 对 LLM 来说，真正被控制的不是“请求个数”，而是 **token 级资源消耗**，因为不同请求的 prompt 长度、生成长度、KV cache 占用差异极大。

一个最直观的玩具例子是：租户 A 发来一个 30 token 的短问答，租户 B 发来一个 2000 token 的长 RAG 请求。如果系统只按 FCFS（First Come First Serve，先来先服务）排队，B 的长 prompt prefill 会先把 GPU 顶住，A 虽然很短，也可能要等很久才能看到首个 token。QoS 的目标不是让 B 变慢，而是避免 B 把 A 长时间挡死。

可以先把问题压成一张表：

| 维度 | 实际受限资源 | 目标指标 | 常用手段 |
|---|---|---|---|
| 交互延迟 | GPU 计算、调度时隙 | TTFT、ITL | 优先级、切块、准入控制 |
| 持续输出 | decode 算力、KV cache | TPOT、吞吐 | 连续批处理、按 token 调度 |
| 多租户公平 | GPU 时间、缓存份额 | fairness | 权重、公平队列、aging |
| 系统稳定性 | KV cache、显存余量 | goodput、尾延迟 | 配额、限流、资源预留 |

这里的几个指标含义后文会正式定义，但先记住主线：**LLM 服务的 QoS 不等于普通 Web 请求的排队控制，因为它是一个会持续占用资源、并且资源占用随 token 生成而演化的过程。**

---

## 问题定义与边界

先定义讨论范围：本文讨论的是**多租户共享推理服务**，也就是多个业务、团队或用户共同使用一组 LLM 推理节点的场景；不讨论训练，也不讨论主要以离线批处理为主、对交互延迟不敏感的系统。

LLM Serving 和普通 HTTP 服务最大的区别，是一个请求通常分成两个阶段：

1. **prefill**：把输入 prompt 一次性编码进模型，并建立 KV cache。白话说，就是“先把问题完整读一遍并记住上下文”。
2. **decode**：模型基于已有上下文逐 token 生成输出。白话说，就是“接下来一个字一个字往下写”。

这两个阶段的资源特征不同。prefill 往往更吃大块计算，尤其对长 prompt 很重；decode 则是很多小步迭代，每个活跃请求都持续占着 KV cache，并不断消耗调度机会。所以 QoS 的控制点也不同：prefill 影响首 token 时间，decode 影响持续输出平滑度和总吞吐。

下面是几个核心术语：

| 术语 | 定义 | 白话解释 |
|---|---|---|
| `TTFT` | Time To First Token | 从请求进入到收到第一个输出 token 的时间 |
| `ITL` / `TPOT` | Inter-Token Latency / Time Per Output Token | 相邻输出 token 之间的延迟 |
| `prefill` | 输入阶段计算 | 先把整段 prompt 读进去 |
| `decode` | 生成阶段计算 | 再一个 token 一个 token 地写出来 |
| `goodput` | 满足 SLO 的有效吞吐 | 不只是跑得多，而是“满足延迟约束地跑得多” |
| `fairness` | 公平性 | 不同租户拿到的服务份额是否符合预期 |

可以把一次请求抽象成下面这个流程：

| 阶段 | 主要动作 | 主要资源压力 |
|---|---|---|
| 请求进入 | 鉴权、路由、打标签 | 网关、元数据 |
| 预处理 | tokenize、请求分类 | CPU、内存 |
| prefill | 读入 prompt，建立 KV | GPU 计算、显存 |
| decode | 逐 token 生成 | GPU、KV cache、调度时隙 |
| 输出返回 | streaming 返回结果 | 网络、队列管理 |

为什么普通队列策略不够用？因为普通请求常被近似为“一个请求对应一次服务”；但 LLM 请求更像“一个请求包含一长串 token 级小服务”。如果还只按请求数做公平，会出现两个失真：

1. 两个请求数量相同，但 token 成本可能相差 100 倍。
2. 一个长请求在 prefill 阶段可能瞬间吞掉大量计算，在 decode 阶段又长期占着缓存。

所以“每人一个请求槽位”看起来公平，实际上对资源完全不公平。

这里还要区分**软 QoS**和**硬 QoS**。软 QoS 的意思是“尽量保证”，主要靠调度、限流、配额和反馈控制；硬 QoS 的意思是“必须保证”，通常还需要显式资源预留、GPU partition、MIG（Multi-Instance GPU，多实例 GPU）或者独占实例。调度器再聪明，也不能在资源已被挤满时凭空创造确定性。

---

## 核心机制与推导

先看为什么“按请求数公平”不成立。

假设租户 A 和租户 B 各提交 10 个请求。A 的每个请求是 100 token prompt、50 token 输出；B 的每个请求是 4000 token prompt、500 token 输出。如果系统只说“两个租户各有 10 个请求，所以应该平等”，那就忽略了每个请求真实资源成本完全不同。

更合理的抽象是给请求定义 token 成本：

$$
cost(r) = \alpha \cdot p_r + \beta \cdot d_r
$$

其中：

- $p_r$ 是请求 $r$ 的 prompt token 数
- $d_r$ 是请求 $r$ 的 decode token 数
- $\alpha, \beta$ 是权重，表示 prefill 和 decode 的单位成本不一定相同

白话解释是：**一个请求值多少钱，不按“来了一个请求”算，而按“它会吃掉多少输入计算和输出计算”算。**

如果按租户做加权公平调度，可以定义：

$$
next = \arg\min_i \frac{C_i(t)}{w_i}
$$

其中：

- $C_i(t)$ 是租户 $i$ 到时间 $t$ 为止已经拿到的累计服务量
- $w_i$ 是租户权重，权重越大，理论上应拿到越多份额

这条规则的意思很直接：**谁当前“拿到的服务 / 应得份额”更少，下一轮优先给谁。**

如果系统还声明了延迟目标，则可以把租户或请求的 SLO（Service Level Objective，服务级目标）写成：

$$
TTFT_r \leq B_r^{ttft}
$$

$$
ITL_r \leq B_r^{itl}
$$

即请求 $r$ 的首 token 延迟和 token 间延迟都要不超过对应预算。

### 玩具例子：为什么 chunked prefill 有用

假设一张 GPU 的有效吞吐约为 `1000 tokens/s`。现在有两个租户：

- A：短问答，`p_A = 200`，输出较短
- B：长 RAG，`p_B = 2000`，输出较长

如果 B 先到，并且 prefill 不切块，那么 A 的 TTFT 至少要等 B 的大段 prefill 跑完一大截，量级可能接近 2 秒。对聊天业务来说，这已经明显变差。

如果把 B 的 prefill 拆成每次 `250 token` 的小块，即 **chunked prefill**，那么调度器就有了更多“插入点”。B 不再一次吃完整个 prefill，而是每做完一小块，就允许调度器重新决定下一轮给谁。这样 A 往往能在下一个调度点插队，TTFT 可能下降到 `0.3s~0.5s` 量级。

为什么这个机制成立？因为长请求的问题不是“总成本高”本身，而是“它以不可中断的大块形式占用资源”。切块相当于把长时间独占，改成多次可重排的短时占用。

### 几类机制分别解决什么问题

| 机制 | 主要解决的问题 | 为什么有效 | 代价 |
|---|---|---|---|
| `FCFS` | 实现最简单 | 无额外调度复杂度 | 容易被长请求拖垮 |
| `priority` | 关键租户低延迟 | 让高优租户先拿资源 | 低优可能饥饿 |
| `weighted fair scheduling` | 按份额公平 | 按累计服务量补偿“欠服务”租户 | 统计和调度复杂度更高 |
| `continuous batching` | 提高 GPU 利用率 | 动态把活跃请求拼进批次 | 可能伤害尾延迟 |
| `chunked prefill` | 避免长 prompt 阻塞短请求 | 给调度器更多重排机会 | 切太细会增加调度开销 |
| `prefill/decode disaggregation` | 避免两阶段互相干扰 | 让长输入与持续生成分离 | 系统更复杂，数据搬运更重 |

这里特别容易混淆的是，`continuous batching` 和 `chunked prefill` 不解决同一个问题。

- `continuous batching` 的目标是**提高利用率**。只要某些请求进入 decode、某些请求完成一轮，就尽快把新的工作补进 batch。
- `chunked prefill` 的目标是**拆散长输入独占**。它首先是延迟控制手段，其次才影响利用率。
- `priority / rate limit` 控制的是**谁更该先跑、谁该被限制进入**。
- `admission control` 控制的是**系统是否还应该继续接活**。

### 真实工程例子：统一企业 LLM 网关

假设一个企业统一 API 服务三类业务：

| 业务 | prompt 特征 | QoS 重点 |
|---|---|---|
| 客服聊天 | 短 prompt，中短输出 | `TTFT` 和尾延迟 |
| 代码补全 | 中等 prompt，要求流式顺滑 | `ITL` 稳定 |
| 批量总结 | 长 prompt，吞吐优先 | 总 tokens/s |

如果把三类请求都扔进一个无区分队列，批量总结最容易把客服聊天拖慢。更合理的做法是：

1. 客服聊天高优先级，并有更严格的 TTFT 预算。
2. 代码补全保持中优先级，但限制单租户并发，避免 decode 阶段拖太久。
3. 批量总结低优先级，允许在系统空闲时吃掉剩余算力。
4. 长 prompt 强制 chunked prefill。
5. KV cache 留出 headroom（安全余量），避免系统接满后频繁抖动。

这时系统追求的不是“所有人同时更快”，而是“客服稳定快、代码补全平稳、总结任务吃剩余资源”。这就是多租户 QoS 的工程目标。

---

## 代码实现

QoS 不是一个单独的调度函数，而是一组联动模块。至少需要下面这些组件：

| 模块 | 作用 | 关键状态 |
|---|---|---|
| API Gateway | 接入、鉴权、限流 | tenant id、请求元数据 |
| Tenant Classifier | 识别租户和业务类型 | 权重、优先级、SLO 档位 |
| Admission Controller | 决定要不要接进系统 | 并发数、KV 预算、队列深度 |
| Scheduler | 决定下一轮跑谁 | 累计服务量、活跃请求阶段 |
| KV Cache Manager | 管理缓存占用和回收 | token 占用、headroom |
| Metrics / Feedback Loop | 观测并调参 | TTFT、ITL、goodput、drop rate |

实现的关键不是“写一个最聪明的分数函数”，而是**把租户身份、token 预算和阶段状态贯穿请求整个生命周期**。如果网关打了优先级标签，但调度器最终只看请求到达顺序，那前面的策略就白做了。

一个典型的调度循环可以写成伪代码：

```text
for each scheduling round:
    collect active tenants
    compute C_i(t) / w_i
    pick tenant with smallest value
    admit if KV cache budget allows
    run one chunk of prefill or a decode step
    update metrics and weights
```

下面给一个可运行的 Python 玩具实现。它不是完整 serving 系统，只演示“按权重补偿欠服务租户，并用 chunked prefill 切长请求”的核心思想。

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Request:
    tenant: str
    prompt_tokens: int
    decode_tokens: int
    chunk_size: int
    pref_done: int = 0
    dec_done: int = 0

    def stage(self) -> str:
        if self.pref_done < self.prompt_tokens:
            return "prefill"
        if self.dec_done < self.decode_tokens:
            return "decode"
        return "done"

    def step(self) -> int:
        if self.stage() == "prefill":
            n = min(self.chunk_size, self.prompt_tokens - self.pref_done)
            self.pref_done += n
            return n
        if self.stage() == "decode":
            self.dec_done += 1
            return 1
        return 0

@dataclass
class TenantState:
    weight: float
    served: float = 0.0
    queue: List[Request] = field(default_factory=list)

def pick_tenant(tenants):
    active = [(name, st) for name, st in tenants.items() if any(r.stage() != "done" for r in st.queue)]
    assert active, "no active tenants"
    return min(active, key=lambda x: x[1].served / x[1].weight)[0]

def run_rounds(tenants, rounds: int):
    history = []
    for _ in range(rounds):
        active_exists = any(any(r.stage() != "done" for r in st.queue) for st in tenants.values())
        if not active_exists:
            break
        tenant_name = pick_tenant(tenants)
        st = tenants[tenant_name]
        req = next(r for r in st.queue if r.stage() != "done")
        consumed = req.step()
        st.served += consumed
        history.append((tenant_name, req.stage(), consumed))
    return history

tenants = {
    "A": TenantState(
        weight=2.0,
        queue=[Request(tenant="A", prompt_tokens=200, decode_tokens=4, chunk_size=50)]
    ),
    "B": TenantState(
        weight=1.0,
        queue=[Request(tenant="B", prompt_tokens=2000, decode_tokens=4, chunk_size=250)]
    ),
}

hist = run_rounds(tenants, rounds=40)

# A 的权重大于 B，应更早拿到服务
first_10 = [x[0] for x in hist[:10]]
assert first_10.count("A") >= first_10.count("B")

# A 不会被 B 的 2000-token prompt 一次性挡死
a_req = tenants["A"].queue[0]
assert a_req.pref_done > 0

# B 的 prefill 确实是分块推进的，而不是一次跑完
b_req = tenants["B"].queue[0]
assert b_req.pref_done < 2000 or b_req.stage() != "done"

print("ok")
```

这个例子刻意保留了几个真实系统中的关键点：

1. **服务量按 token 累计**，而不是按请求次数累计。
2. **prefill 和 decode 分阶段推进**，不能把整次请求视为不可分对象。
3. **长 prompt 切块**，让调度器在块与块之间重新做决定。

实际工程里通常还要有一层配置，例如：

| 配置项 | 示例值 | 作用 |
|---|---|---|
| `tenant_weight` | `gold=4, silver=2, bronze=1` | 定义租户应得份额 |
| `max_prefill_chunk` | `256` | 限制长 prompt 单轮占用 |
| `max_concurrent_prefill` | `2` | 避免多个长 prefill 同时顶满 GPU |
| `kv_cache_headroom` | `15%` | 预留缓存余量，避免抖动 |
| `priority_level` | `interactive > standard > batch` | 决定抢占顺序 |

如果落到现有推理框架，思路通常是：

- 在 vLLM 一类系统中，关注 scheduler 配置、并发 token 预算和 prefill 行为。
- 在 Triton 一类系统中，关注实例组、队列策略、速率限制和模型级资源约束。

但重点不是记某个参数名，而是确认这些参数是否真的映射到了你的 QoS 目标：控制的是 TTFT、ITL、公平性，还是只是让 GPU 看起来更忙。

---

## 工程权衡与常见坑

QoS 设计的难点不在于“有没有算法名字”，而在于每个机制都带副作用。工程上常见的矛盾是：

- 提高吞吐，通常会伤害尾延迟。
- 提高公平性，可能会破坏缓存局部性。
- 加强隔离，通常会降低总体利用率。

先看一张常见坑位表：

| 问题 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| 只按请求数公平 | 长 prompt 租户吃穿系统 | 请求数不代表 token 成本 | 按 token 或服务量做公平 |
| 只调大 batch 不切 prefill | 吞吐上升但 TTFT 变差 | 长请求更容易压住短请求 | 对长 prompt 做 chunked prefill |
| chunked prefill 但不做 admission check | KV cache 抖动、OOM | 只拆调度，不控总占用 | 进入系统前检查全长预算 |
| 忽略 prefix locality | cache 命中率下降 | 过度打散相似前缀请求 | 在公平和局部性之间加约束 |
| 静态优先级没有 aging | 低优租户长期饥饿 | 高优队列一直压制低优 | 用 aging 逐步提升等待过久者 |
| 只靠调度想做硬 QoS | 指标偶发失守 | 资源竞争本身未消失 | 对关键业务做预留或物理隔离 |

### 1. 只看吞吐，会把交互体验做坏

很多系统在压测时先看 `tokens/s`，因为它最容易量化。但在线交互系统首先感知到的不是吞吐，而是首 token 是否够快、生成过程是否平滑。

举个对比例子：把 batch 一路调大，GPU 利用率可能更高，总吞吐也更漂亮；但短请求可能要在更长队列里等“合批”或等大 prefill 完成，结果是 P95 TTFT 明显上升。用户体感会觉得“系统变卡了”，即使后台图表显示吞吐更高。

### 2. chunked prefill 不是越细越好

切块的直觉是对的，但切得太细会带来三类额外成本：

1. 调度轮次变多，调度开销上升。
2. 批次更碎，算子效率下降。
3. 系统更频繁地在租户间切换，缓存局部性变差。

所以 chunk size 不是越小越公平，而是要找到“足以打断长阻塞，但不至于把系统切碎”的平衡点。这个点和模型大小、GPU 类型、prompt 分布、业务 SLO 都相关，通常需要压测。

### 3. admission control 比调度更基础

很多初学者容易把注意力全放在“下一轮该跑谁”，但如果系统已经把请求收得过满，后面的调度只能在拥塞里做局部优化。

admission control 的作用是：**当 KV cache、显存或活跃 decode 数接近上限时，拒绝或延后新请求进入。** 这是在防止系统进入抖动区。一旦进入抖动区，常见现象是：

- TTFT 突然陡增
- ITL 不稳定
- cache 回收变频繁
- 有效吞吐下降，甚至不如保守配置

### 4. prefix locality 和公平性会冲突

prefix locality 指很多请求前缀相同，例如都带同一段系统提示词或检索模板。白话说，就是“如果相似请求挨在一起跑，缓存更容易复用”。

问题在于，严格公平调度可能把这些本来相似的请求打散，让缓存命中率下降。于是系统表面上更公平，实际上整体更慢。工程上常见折中是：

- 在同一优先级层内，优先保留相似前缀的局部性。
- 用“软公平”而不是每轮绝对公平。
- 对高复用前缀的业务单独做路由或池化。

### 5. 静态优先级必须配 aging

如果“客服永远高优、批处理永远低优”而没有 aging（老化补偿），那么一旦高优流量持续满载，低优租户可能长期几乎拿不到资源。这在企业内部平台上常会引发更大的业务冲突，因为“低优”不等于“不重要”。

aging 的思想很简单：等待越久，隐含优先级越高。它不是取消优先级，而是防止永久饥饿。

这一节的小结可以压成一句话：**QoS 的目标不是把所有租户都变快，而是把延迟、吞吐和公平性控制在业务可以接受的边界内。**

---

## 替代方案与适用边界

并不是所有多租户 LLM 系统都该用同一套 QoS 策略。关键不是“哪个算法先进”，而是你的业务要求是“尽量保证”，还是“必须保证”。

下面给出一张替代方案对照表：

| 方案 | 适合什么业务 | 解决什么问题 | 会牺牲什么 |
|---|---|---|---|
| `priority + rate limit` | 交互优先、业务分层明显 | 快速保护关键租户 | 低优可能受压制 |
| `weighted fair queue` | 多团队共享平台 | 按份额分资源 | 实现和调参更复杂 |
| `static reservation` | 关键业务有稳定基线负载 | 保证最低资源份额 | 低负载时可能浪费 |
| `GPU partition / MIG` | 必须隔离、强 SLO | 降低互扰，实现硬边界 | 利用率下降，灵活性差 |
| `dedicated instance` | 大客户、专属服务 | 最强隔离和最清晰计费 | 成本高 |
| `offline batch queue` | 长总结、离线生成 | 把吞吐型任务与在线流量隔离 | 批任务时延更高 |

### 什么时候用软 QoS

软 QoS 适用于“尽量保证”的场景，例如企业统一 API、共享研发平台、通用聊天入口。这类系统最重要的是：

- 不让少数长请求拖垮整体
- 让高优业务延迟稳定
- 在总体利用率和体验之间找平衡

此时通常优先选择：

- 优先级分层
- 按 token 的公平队列
- chunked prefill
- admission control
- KV cache 余量管理

### 什么时候必须上硬隔离

如果业务要求接近“合同级保证”，例如：

- 某客户必须独占容量
- 某在线业务必须稳定在很低 TTFT
- 某监管场景不能接受资源互扰导致的性能波动

那单靠调度通常不够。因为调度只能在共享资源里重排顺序，不能消除根本竞争。这时更合理的是：

- 做静态预留
- 做 GPU partition / MIG
- 甚至给租户独立实例

### 场景化判断

再看一个真实工程场景。企业统一 API 同时服务客服聊天、代码补全、批量总结。

- 客服聊天：`TTFT < 1.5s`，要强交互。
- 代码补全：要稳定流式输出，不能一会快一会慢。
- 批量总结：对单请求延迟不敏感，更关心总吞吐。

合理方案通常不是“全部平权”，而是：

1. 客服高优先级，严格限并发和 TTFT 预算。
2. 代码补全中优先级，重点控制 decode 平稳性。
3. 批量总结低优先级，长 prompt 做 chunked prefill。
4. 整体保留 KV cache headroom，避免服务抖动。
5. 如果客服业务规模足够大，再考虑单独资源池。

所以边界判断可以压成三句：

- **软 QoS** 适用于“尽量保证”。
- **硬隔离** 适用于“必须保证”。
- 介于两者之间时，先做调度和配额，再决定是否加资源预留。

---

## 参考资料

| 标题 | 类型 | 作用 | 文章中对应章节 |
|---|---|---|---|
| Orca | 论文 | 解释连续批处理和 LLM serving 基本模型 | 核心机制与推导、代码实现 |
| Fairness in Serving Large Language Models | 论文 | 支撑按服务量而非请求数做公平的思路 | 核心机制与推导、工程权衡与常见坑 |
| DistServe | 论文 | 讨论 prefill / decode 解耦与 goodput | 核心机制与推导、替代方案与适用边界 |
| NVIDIA Triton Rate Limiter | 工程文档 | 对应准入控制、限流和资源约束 | 代码实现、替代方案与适用边界 |
| NVIDIA Triton Model Configuration | 工程文档 | 对应部署时的实例与队列配置 | 代码实现 |
| vLLM Scheduler Config | 工程文档 | 对应调度器和 token 预算概念 | 代码实现、工程权衡与常见坑 |

1. [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
2. [Fairness in Serving Large Language Models](https://www.usenix.org/conference/osdi24/presentation/sheng)
3. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://huggingface.co/papers/2401.09670)
4. [NVIDIA Triton Rate Limiter](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2470/user-guide/docs/user_guide/rate_limiter.html)
5. [NVIDIA Triton Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2510/user-guide/docs/user_guide/model_configuration.html)
6. [vLLM Scheduler Config](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/)
