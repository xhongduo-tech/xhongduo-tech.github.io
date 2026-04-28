## 核心结论

推理服务成本预测的核心，不是看“模型有多少参数”，而是看“在既定延迟约束下，单实例一年能稳定产出多少有效 token”。这里的“单实例”可以理解为一台实际对外提供推理服务的部署单元，通常包含 1 张或多张 GPU。这里的“延迟约束”是服务等级目标，例如 `TTFT < 300ms` 或 `p95 < 2s`，也就是用户体验不能被突破的硬边界。

同一个模型，部署方式不同，单位成本可以差很多。原因不在公式变了，而在于单实例吞吐 $q(L,m)$ 变了。这里的“吞吐”就是单位时间内系统能完成多少请求或生成多少 token；“工作负载分布” $m$ 指真实请求的输入长度、输出长度、请求类型混合比例。continuous batching、动态批处理、PagedAttention 这些机制的作用，都是把满足延迟约束时的可用吞吐往上推。

下面这张表先把完整推导链路钉住：

| 输入或中间量 | 含义 | 示例 |
|---|---|---:|
| `λ_peak` | 峰值请求率，单位 `req/s` | `120` |
| `q(L,m)` | 满足延迟约束时单实例吞吐 | `30 req/s` |
| `n` | 需要的实例数 | `ceil(120/30)=4` |
| `G` | 每实例 GPU 数 | `1` |
| `p_gpu` | GPU 小时单价 | `$2/hr` |
| `C_year` | 年总成本 | `$70,080 + 其他成本` |
| `R_year` | 年请求数 | 由真实业务估算 |
| `T_year` | 年 token 数 | 由真实输入输出分布估算 |
| `c_req` | 单请求成本 | `C_year / R_year` |
| `c_tok` | 单 token 成本 | `C_year / T_year` |

玩具例子很简单。假设峰值流量是 `120 req/s`，满足延迟目标时，单实例只能稳定跑到 `30 req/s`，那么需要的实例数就是：

$$
n = \lceil \lambda_{peak} / q(L,m) \rceil = \lceil 120 / 30 \rceil = 4
$$

若每实例 1 张 GPU，GPU 单价 `$2/hr``，只看算力成本：

$$
C_{year} = 4 \times 1 \times 2 \times 8760 = 70080
$$

这个结论已经说明重点：成本预测本质上是“容量规划问题”，不是“拍脑袋报价问题”。

---

## 问题定义与边界

这篇文章讨论的是在线推理服务。所谓“在线”，就是请求到来后系统要立刻返回结果，核心约束是延迟和稳定性；它不讨论离线批处理、训练成本，也不讨论只追求极限吞吐、不关心响应时间的压测场景。

输入和输出必须分清，否则公式再漂亮也没有意义。

| 符号 | 定义 | 说明 |
|---|---|---|
| `L` | 延迟预算 | 例如 `TTFT < 300ms, p95 < 2s` |
| `m` | 工作负载分布 | 输入长度、输出长度、请求类型、到达模式 |
| `λ_peak` | 峰值请求率 | 规划时应看高峰，不看全年平均 |
| `q(L,m)` | 单实例吞吐 | 在满足 `L` 时实测得到 |
| `G` | 每实例 GPU 数 | 例如 1、2、4 |
| `p_gpu` | GPU 小时单价 | 云上或自建折算单价 |
| `c_host` | 年化主机成本 | CPU、内存、磁盘等 |
| `c_lic` | 年化软件授权成本 | 商业引擎或平台费用 |
| `c_net` | 年化网络与带宽成本 | 出网、LB、跨区流量等 |
| `C_year` | 年总成本 | 全部实例全年总成本 |
| `R_year` | 年请求数 | 总请求量 |
| `T_year` | 年 token 数 | 总输入加总输出 token 量 |
| `c_req` | 单请求成本 | 成本按请求折算 |
| `c_tok` | 单 token 成本 | 成本按 token 折算 |

新手最容易犯的错，是把平均 QPS 当成规划依据。平均值能描述账面负载，不能描述线上风险。真实系统常被三类因素击穿：

| 风险来源 | 为什么平均值失效 |
|---|---|
| 白天高峰 | 平均值会稀释峰值，导致实例数低估 |
| 长上下文请求 | 少量长请求会显著拉低吞吐 |
| 突发流量 | 短时间 burst 会把队列迅速堆高 |

因此，规划输入通常要包含 `p95/p99` 级别的流量观察值，以及一个 burst factor。这里的“burst factor”可以理解为突发放大倍数，例如平时 `60 req/s`，但一分钟内可能冲到 `120 req/s`，那就不能按 60 规划。

边界也要说清。成本预测回答的是：“为了满足业务延迟目标，需要多少实例，全年大约花多少钱，单位请求和单位 token 成本是多少。”它不直接回答“模型值不值得用”，也不回答“算法上最优不最优”。

---

## 核心机制与推导

核心流程可以压缩成四步：

1. 定义 workload 分布，也就是 $m$。
2. 基于真实 workload 做 benchmark，测出吞吐-时延前沿。
3. 在满足延迟约束 $L$ 的点上，取最高吞吐 $q(L,m)$。
4. 用峰值流量反推实例数，再折算年成本与单位成本。

这条链路可以写成：

$$
m \rightarrow benchmark \rightarrow q(L,m) \rightarrow n \rightarrow C_{year} \rightarrow c_{req}, c_{tok}
$$

最关键的式子有四个。

实例数：

$$
n = \lceil \lambda_{peak} / q(L,m) \rceil
$$

年总成本：

$$
C_{year} = n \cdot (G \cdot p_{gpu} \cdot 8760 + c_{host} + c_{lic} + c_{net})
$$

单请求成本：

$$
c_{req} = C_{year} / R_{year}
$$

单 token 成本：

$$
c_{tok} = C_{year} / T_{year}
$$

为什么这套方法成立？因为在线推理服务本质上是排队系统。请求不是均匀到来的，而是随机到来、长度不一、互相争抢显存和计算。只要系统负载接近上限，排队延迟就会突然放大。也就是说，吞吐不能脱离延迟单独讨论；任何“单卡能跑多少”都必须附带延迟条件。

这里给一个玩具例子。假设一个模型在单请求模式下，为了满足 `TTFT < 300ms`，单实例只能跑 `18 req/s`。改成 continuous batching 后，在同样延迟目标下可以跑到 `30 req/s`。那么对同一个峰值 `120 req/s`：

| 方案 | 满足延迟时单实例吞吐 | 需要实例数 |
|---|---:|---:|
| 单请求推理 | `18 req/s` | `7` |
| continuous batching | `30 req/s` | `4` |

模型没变，需求没变，成本却明显下降。原因不是“batch 让公式失效”，而是它提高了 $q(L,m)$。

再看真实工程例子。在线客服助手的请求往往分成两类：短问题和长会话。短问题可能是 `128` 输入 token、`64` 输出 token；长会话可能是 `2048` 输入 token、`256` 输出 token。长请求的 prefill 阶段会占用大量 KV cache。这里的“KV cache”可以理解为模型为后续生成保留的历史上下文记忆区，占显存且与上下文长度强相关。它一旦占满，batch 就会下降，吞吐也会掉。

因此，正确做法不是拿一个平均长度 `600 token` 去测一次，而是按桶测。例如：

| 输入桶 | 输出桶 | 满足延迟时单实例吞吐 |
|---|---|---:|
| `128` | `64` | `52 req/s` |
| `512` | `128` | `31 req/s` |
| `2048` | `256` | `11 req/s` |

然后按各桶流量占比合成整体容量，而不是用一个“平均请求”代替全部请求。平均值最大的问题，是会把长尾工作负载洗掉，而系统通常正是被长尾打穿。

动态 batching 和 PagedAttention 也要准确理解。动态 batching 是把接近时间窗口内的请求合并执行，以减少 GPU 空洞；continuous batching 是在生成过程中持续接入新请求，而不是一整批必须同时开始同时结束；PagedAttention 的本质是更高效地管理 KV cache，减少显存碎片和浪费。它们都不是直接改成本公式，而是改善实现方式，从而把可用前沿向右上移动。

---

## 代码实现

代码实现的目标不是造一个复杂平台，而是把“可复算”这件事先落地。最低要求是：给定峰值流量、满足延迟时的吞吐、每实例资源成本、年请求数和年 token 数，能稳定算出实例数、年成本、单请求成本和单 token 成本。

下面是一个最小可运行脚本：

```python
import math
from dataclasses import dataclass


@dataclass
class CostInput:
    lambda_peak: float   # 峰值 req/s
    q: float             # 满足延迟时的单实例吞吐 req/s
    g: int               # 每实例 GPU 数
    p_gpu: float         # GPU 单价 $/hr
    c_host: float = 0.0  # 每实例年化主机成本
    c_lic: float = 0.0   # 每实例年化授权成本
    c_net: float = 0.0   # 每实例年化网络成本
    r_year: float = 1.0  # 年请求数
    t_year: float = 1.0  # 年 token 数


def estimate_cost(x: CostInput):
    assert x.lambda_peak > 0
    assert x.q > 0
    assert x.g >= 1
    assert x.p_gpu >= 0
    assert x.r_year > 0
    assert x.t_year > 0

    n = math.ceil(x.lambda_peak / x.q)
    annual_per_instance = x.g * x.p_gpu * 8760 + x.c_host + x.c_lic + x.c_net
    annual_cost = n * annual_per_instance
    cost_per_req = annual_cost / x.r_year
    cost_per_token = annual_cost / x.t_year

    return {
        "instances": n,
        "annual_cost": annual_cost,
        "cost_per_req": cost_per_req,
        "cost_per_token": cost_per_token,
    }


demo = CostInput(
    lambda_peak=120,
    q=30,
    g=1,
    p_gpu=2.0,
    r_year=1.0e9,
    t_year=7.56e9,
)

result = estimate_cost(demo)

assert result["instances"] == 4
assert abs(result["annual_cost"] - 70080) < 1e-9
assert abs(result["cost_per_token"] - (70080 / 7.56e9)) < 1e-12

print(result)
```

这段代码只解决了最小问题，但已经比“口头估算”强很多。真正上线时，通常要按分桶输入 `q(L,m)`，而不是只传一个总吞吐。一个实用的简化做法，是先按 prompt 长度和输出长度分桶，再给每个桶配置实测吞吐与流量占比。

例如：

| 桶 ID | prompt tokens | output tokens | 流量占比 | `q(L,m)` |
|---|---:|---:|---:|---:|
| A | `128` | `64` | `50%` | `52 req/s` |
| B | `512` | `128` | `35%` | `31 req/s` |
| C | `2048` | `256` | `15%` | `11 req/s` |

更进一步，工程里还会拆成 prefill 和 decode 两段。prefill 是“先把输入上下文读进去”的阶段，计算量对输入长度敏感；decode 是“逐 token 生成输出”的阶段，速度更多受 batch 与 KV cache 状态影响。长 prompt 系统常常是 prefill 先卡住，长输出系统则常常是 decode 拖慢尾延迟。

因此，脚本的真正价值不在于公式本身，而在于把每次 benchmark 的结果沉淀成配置化数据。只要分桶、单价、SLO 没变，你就能重复计算，而不是每次重新拍脑袋。

---

## 工程权衡与常见坑

成本和延迟是硬耦合关系。batch 变大通常更省钱，但前提是 `TTFT`、`p95`、`p99` 仍然过线。这里的 `p95` 指 95% 请求的延迟不超过某阈值，是看尾部体验的指标。如果为了省卡把队列压到接近饱和，平均吞吐也许更好看，但线上用户会先感知到变慢。

下面这张坑位表比抽象讨论更有用：

| 常见坑 | 失败原因 | 规避方法 |
|---|---|---|
| 只看平均 QPS | 峰值和突发被平均值抹平 | 按 `p95/p99` 流量和 burst factor 规划 |
| 只看单请求吞吐 | 忽略 batch、排队、混合长度 | 用真实 workload 做分桶 benchmark |
| 忽略长上下文请求 | KV cache 被顶满后吞吐会塌 | 按上下文长度分桶，单独评估长尾 |
| 只算 GPU 成本 | CPU、内存、带宽、授权、空转未计入 | 将 `c_host/c_lic/c_net` 并入总成本 |
| 把静态公式外推到饱和区 | 接近满载时队列延迟非线性上升 | 在高负载区直接测 `p95/p99` |
| 认为 batch 越大越省 | 大 batch 可能突破 TTFT 目标 | 只在满足 SLO 的点上取最高吞吐 |
| 用平均 token 长度代表全部请求 | 长尾请求被低估 | 使用输入桶和输出桶联合统计 |

真实工程例子可以看在线客服系统。白天流量突刺时，大量会话型请求会带着长聊天历史进入系统，prompt 迅速变长。结果往往不是“略微变慢”，而是 KV cache 占用抬升，单实例可容纳 batch 降低，吞吐下降，队列堆积，随后为了保住 SLO 被迫临时扩容，单位成本突然上升。

这类问题常见的治理方式有三种：

| 方案 | 作用 | 代价 |
|---|---|---|
| prefill / decode 拆分 | 把两类瓶颈分开优化 | 系统复杂度更高 |
| 限制最大上下文长度 | 直接控制 KV 占用上界 | 可能伤害回答质量 |
| 更省内存的 serving 引擎 | 提高同卡可承载并发 | 迁移与验证成本高 |

一个容易被忽略的工程事实是：空转也要钱。很多服务夜间负载低，但为了白天高峰必须常驻一部分容量。若没有自动伸缩，`C_year` 里会包含大量低利用率时间；即使有伸缩，也可能受冷启动、镜像加载、权重装载时间限制，不能无限贴近流量曲线。因此预算时不要只看“忙时成本”，还要看“全年驻留成本”。

---

## 替代方案与适用边界

这套方法适合在线推理服务的容量规划，特别适合以下场景：有明确延迟 SLO、请求长度分布复杂、业务存在峰谷、单卡成本高、需要对外报预算或做方案比较。

但它不是唯一方法。实际中常见三种估算方式：

| 方法 | 适用场景 | 优点 | 缺点 | 风险 |
|---|---|---|---|---|
| 平均 QPS 法 | 内部低频工具、请求稳定 | 快，数据需求少 | 忽略高峰与长尾 | 常低估实例数 |
| 固定并发法 | 已知并发上限的封闭系统 | 与线程池或连接池思路接近 | 对 token 长度变化不敏感 | 长输出时容易失真 |
| 前沿法 | 对外在线 API、严格 SLO | 最贴近真实服务约束 | 需要 benchmark 成本 | 前期工作量更高 |

如果只是内部小工具，例如每天几百次调用、请求长度接近、超时也可接受，那么按平均流量乘一个保守系数通常就够了。原因是错误代价低，估算精度的收益不大。

但如果是对外在线 API，这种粗略方法通常不够。因为用户不会按平均值访问你的系统，他们会在高峰时同时打进来。只按平均值算的结果，往往是日常看起来没事，一到高峰就开始排队、超时、重试，最后不仅体验差，成本还因重试和扩容进一步变高。

还要强调一个边界：动态 batching 和 PagedAttention 不是降低公式本身，而是改变 $q(L,m)$ 的实现方式。也就是说，它们优化的是“同样延迟条件下单实例能扛多少负载”，不是把容量规划这件事变没了。只要你是在线服务，就逃不开工作负载分布、排队延迟、显存占用和实例数反推这些基本约束。

---

## 参考资料

1. [LLM Inference Benchmarking: How Much Does Your LLM Inference Cost?](https://developer.nvidia.com/blog/llm-inference-benchmarking-how-much-does-your-llm-inference-cost/) 用于建立“先测吞吐-时延前沿，再做成本换算”的 benchmark 思路。
2. [NVIDIA Triton Docs: Dynamic Batching](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) 用于说明动态批处理如何提高资源利用率，以及它与延迟约束的关系。
3. [vLLM Documentation](https://docs.vllm.ai/en/v0.10.0/) 用于说明在线 LLM serving 中的 continuous batching、调度和工程实现方式。
4. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) 用于说明 PagedAttention 如何通过更高效的 KV cache 管理改善吞吐与显存利用率。
5. [A Queueing Theoretic Perspective on Low-Latency LLM Inference with Variable Token Length](https://arxiv.org/abs/2407.05347) 用于说明为什么变长 token 工作负载下，排队理论对低延迟推理成本预测是必要视角。
