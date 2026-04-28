## 核心结论

模型推理成本，不是“租一张 GPU 每小时多少钱”，而是“在目标 SLA 下，稳定交付 1 个有效 token 需要多少钱”。SLA 是服务等级目标，白话说就是你答应用户的延迟、可用性和稳定性标准。有效 token 也不是离线压测里偶尔跑出的峰值 token，而是在真实流量、真实排队、真实故障条件下，系统持续交付出去的 token。

把这个定义写成公式就是：

$$
C_{tok}=\frac{C_{gpu}+C_{store}+C_{net}+C_{ops}+C_{loss}}{T_{eff}}
$$

其中，$C_{gpu}$ 是算力成本，$C_{store}$ 是存储成本，$C_{net}$ 是网络成本，$C_{ops}$ 是运维成本，$C_{loss}$ 是故障、冗余、滚动发布、空转带来的损失，$T_{eff}$ 是有效交付 token 总量。

结论可以压缩成一句话：看 `tokens/$`，不要只看 GPU 价格。

对初学者最容易误判的一点是：高端 GPU 单价更高，不代表单位 token 成本一定更高。只要它的稳定吞吐更高、KV cache 更省、为了满足 SLA 需要的副本更少，它最后的每 100 万 token 成本反而可能更低。这里的 KV cache 是“保存历史上下文中间结果的显存区域”，白话说就是模型为了不重复计算之前内容而保留下来的缓存。

一个最小直觉例子：

- 方案 A：单卡便宜，但一台只能稳定跑 `1000 tok/s`，为了扛住高峰和冗余要上 4 个副本。
- 方案 B：单卡更贵，但一台能稳定跑 `2200 tok/s`，只要 2 个副本就够。

如果 B 的总副本更少、空转更少、运维更简单，那么它不只是“性能更强”，而是“生意上更便宜”。

---

## 问题定义与边界

先把“这篇文章到底算什么成本”说清楚。这里讨论的不是单卡账单，也不是实验室里的 benchmark 跑分，而是一个已经上线、需要对外提供服务的推理系统总成本。

下面这张表先定边界：

| 范围 | 是否纳入 |
|---|---|
| GPU/加速卡 | 是 |
| 存储/KV cache | 是 |
| 网络 | 是 |
| 运维/人力 | 是 |
| 故障冗余/空转 | 是 |
| 纯训练成本 | 否 |
| 纯算法效果（准确率） | 只作为间接影响因素 |

几个核心变量先定义：

| 符号 | 含义 |
|---|---|
| `C_tok` | 单位 token 成本 |
| `C_gpu` | GPU 算力成本 |
| `C_store` | 存储与缓存成本 |
| `C_net` | 网络与跨机通信成本 |
| `C_ops` | 运维、人力、监控、发布成本 |
| `C_loss` | 可用性损失、冗余空转、失败重试成本 |
| `T_eff` | 有效交付 token 数 |

这里“有效交付 token”是重点。它不是单机理论峰值，而是满足 SLA 后，系统真正稳定产出的 token。举例说，两套系统都测到过 `1000 tok/s` 的峰值：

- 系统 X 在 p95 延迟约束下只能长期稳定跑 `600 tok/s`
- 系统 Y 在同样约束下可以长期稳定跑 `900 tok/s`

那么算成本时，X 应该按 `600 tok/s` 算，Y 应该按 `900 tok/s` 算。p95 延迟是“95% 请求的延迟不超过某个值”，白话说就是大多数用户实际感受到的慢不慢。

这篇文章也明确不讨论三件事：

- 不讨论训练成本。训练是另一套成本模型，硬件利用方式、时间尺度、目标函数都不同。
- 不讨论模型效果本身。效果会影响业务价值，但这里先只分析交付成本。
- 不讨论离线 benchmark 的绝对跑分。离线跑分适合粗筛，不适合直接当上线决策。

所以，本文真正回答的问题是：在相同服务目标下，怎样比较不同推理部署方案的真实单位成本。

---

## 核心机制与推导

推理成本的关键，不是单点性能，而是“稳定吞吐 × 副本数 × 可用性”共同决定最终产出。很多人盯着一台机器的峰值 `tok/s`，但业务成本是系统级问题，不是单卡问题。

先定义几个核心量：

$$
T_{eff}=u \times N \times 3600 \times A
$$

- `u`：单副本稳定吞吐，单位 `tok/s`
- `N`：上线副本数
- `A`：可用性系数，范围 `0~1`，表示故障、重试、发布扰动之后真正可交付的比例

再看副本数怎么来：

$$
N=\lceil \frac{\text{目标峰值 token 速率}}{\text{单副本稳定吞吐}} \rceil + \text{冗余副本}
$$

这里的冗余副本不是浪费，而是为了 SLA 必须预留的保险。比如要支持一台机器故障后仍不中断服务，就不能按“平均刚好够”来配。

最后回到总成本：

$$
C_{tok}=\frac{C_{gpu}+C_{store}+C_{net}+C_{ops}+C_{loss}}{T_{eff}}
$$

这三步的顺序很重要：

1. 先从 SLA 推出需要多少副本 `N`
2. 再算有效交付 token `T_eff`
3. 再把总成本除以 `T_eff`

### 玩具例子

假设你要做一个小型在线问答服务，目标是高峰时段稳定交付 token，并且允许一份冗余。现在有两个方案：

| 方案 | 单副本全成本 | 稳定吞吐 | SLA 需要副本 | 每小时有效 token | 1M token 成本 |
|---|---:|---:|---:|---:|---:|
| A | `$6/h` | `1000 tok/s` | `4` | `14.4M` | `$1.67` |
| B | `$9/h` | `2200 tok/s` | `2` | `15.84M` | `$1.14` |

这个表的意思是：

- A 每台便宜，但为了扛住业务和冗余，总共要 4 台
- B 每台更贵，但 2 台就够，而且总有效交付 token 更高

为什么 A 每小时有效 token 是 `14.4M`？因为：

$$
1000 \times 4 \times 3600=14{,}400{,}000
$$

为什么 B 是 `15.84M`？

$$
2200 \times 2 \times 3600=15{,}840{,}000
$$

进一步看总小时成本：

- A：`4 × $6 = $24/h`
- B：`2 × $9 = $18/h`

所以：

- A 的 `1M token` 成本约为 `$24 / 14.4 = $1.67`
- B 的 `1M token` 成本约为 `$18 / 15.84 = $1.14`

这就是“贵卡未必贵”的最小验证。

### 为什么 `MFU` 不能单独代表成本

`MFU` 是 Model FLOPs Utilization，白话说就是“模型把理论算力真正用起来了多少”。常见定义是：

$$
MFU=\frac{achieved\_flops}{peak\_flops}
$$

它有用，但它只是中间指标。原因有三点：

- `MFU` 高，不代表排队短。你可能把卡跑得很满，但用户已经排队排到超时。
- `MFU` 高，不代表副本少。某些方案单机很满，但为了 p99 延迟仍然要堆很多副本。
- `MFU` 高，不代表总系统简单。跨机通信、KV cache 碎片、滚动发布损失，都不会被 `MFU` 直接表达。

所以 `MFU` 更像“算力利用是否健康”的诊断指标，而不是“最终商业成本”的判定指标。最终还是要回到 `T_eff` 和 `C_tok`。

### 真实工程例子

假设你在做一个在线聊天 API，流量同时包含两类请求：

- 短问答：prompt 短，输出也短
- 长上下文问答：prompt 长，历史消息很多，输出不一定长

这时候瓶颈常常不在“矩阵乘法算得够不够快”，而在：

- prefill 阶段是否被长 prompt 拉爆
- decode 阶段是否因为 KV cache 占满显存而降低并发
- scheduler 是否能把不同长度请求稳定批处理

prefill 是“先把输入上下文吃进去的阶段”，decode 是“一个 token 一个 token 往后生成的阶段”。白话说，前者更像一次性读题，后者更像逐字写答案。

在这类场景里，工程团队通常不会只看单卡跑分，而会看：

- 相同 p95/p99 下，能稳定支撑多少并发
- 相同显存约束下，KV cache 能容纳多少活跃会话
- 为了满足高峰流量和故障切换，总共要开多少副本

这也是为什么 `vLLM` 的 continuous batching、`PagedAttention`，以及 `TensorRT-LLM` 的 inflight batching 会直接影响成本。它们提升的不是“理论峰值”，而是上线后可持续的稳定吞吐。

---

## 代码实现

下面给一个可运行的 Python 小脚本，把上面的公式落成计算逻辑。它做三件事：

- 输入各类小时成本
- 输入稳定吞吐、副本数、可用性
- 输出每小时有效 token 和每 100 万 token 成本

```python
from math import ceil

def token_cost_per_1m(
    c_gpu_per_replica,
    c_store_per_replica,
    c_net_per_replica,
    c_ops_per_replica,
    c_loss_total,
    stable_throughput_tok_s,
    replicas,
    availability,
):
    total_hour_cost = replicas * (
        c_gpu_per_replica
        + c_store_per_replica
        + c_net_per_replica
        + c_ops_per_replica
    ) + c_loss_total

    t_eff = stable_throughput_tok_s * replicas * 3600 * availability
    c_tok = total_hour_cost / t_eff
    return c_tok * 1_000_000, t_eff, total_hour_cost


def required_replicas(peak_token_rate, stable_throughput_tok_s, redundancy):
    return ceil(peak_token_rate / stable_throughput_tok_s) + redundancy


# 玩具例子 A
replicas_a = 4
cost_a, t_eff_a, total_a = token_cost_per_1m(
    c_gpu_per_replica=5.0,
    c_store_per_replica=0.4,
    c_net_per_replica=0.3,
    c_ops_per_replica=0.3,
    c_loss_total=0.0,
    stable_throughput_tok_s=1000,
    replicas=replicas_a,
    availability=1.0,
)

# 玩具例子 B
replicas_b = 2
cost_b, t_eff_b, total_b = token_cost_per_1m(
    c_gpu_per_replica=8.0,
    c_store_per_replica=0.4,
    c_net_per_replica=0.3,
    c_ops_per_replica=0.3,
    c_loss_total=0.0,
    stable_throughput_tok_s=2200,
    replicas=replicas_b,
    availability=1.0,
)

assert round(total_a, 2) == 24.0
assert round(total_b, 2) == 18.0
assert int(t_eff_a) == 14_400_000
assert int(t_eff_b) == 15_840_000
assert round(cost_a, 2) == 1.67
assert round(cost_b, 2) == 1.14
assert cost_b < cost_a

# 一个按峰值需求反推副本数的小检查
assert required_replicas(peak_token_rate=3500, stable_throughput_tok_s=1000, redundancy=1) == 5
assert required_replicas(peak_token_rate=3500, stable_throughput_tok_s=2200, redundancy=1) == 3

print("A 方案每 1M token 成本:", round(cost_a, 2))
print("B 方案每 1M token 成本:", round(cost_b, 2))
```

如果你要把它变成实际评估脚本，建议采集这些指标：

| 指标 | 用途 |
|---|---|
| p95/p99 延迟 | 判断是否满足 SLA |
| 稳定吞吐 `tok/s` | 计算 `T_eff` |
| 副本数 `N` | 计算总产能 |
| 利用率 / MFU | 辅助定位瓶颈 |
| 故障率 / 重试率 | 估算 `C_loss` |

真实生产里还要再细一步：把 prefill 和 decode 分开统计，不要只看平均值。因为长 prompt 可能把 prefill 压得很重，而长输出可能把 decode 压得很重。两段瓶颈不同，调参方向也不同。

---

## 工程权衡与常见坑

推理优化的目标，不是“把单卡跑满”，而是“在 SLA 下把有效 token 成本压低”。这句话听起来像口号，但它会直接改变你的调参方法。

最常见的误区如下：

| 常见坑 | 为什么会错 | 规避方法 |
|---|---|---|
| 只看峰值 `tok/s` | 峰值不等于稳定吞吐 | 按 SLA 测稳定吞吐 |
| 只看 GPU 价格 | 忽略副本和空转 | 统一算 `tokens/$` |
| 混算 prefill 和 decode | 两段瓶颈不同 | 分开测、分开调 |
| 忽略故障与滚动发布 | 低估 `C_loss` | 把可用性纳入预算 |
| 只看 `MFU` | 中间指标不等于最终成本 | 回到 `T_eff` 和 `C_tok` |

再看几个常见工程权衡。

第一，`max_batch_size` 不是越大越好。它决定单次可合批的请求规模，白话说就是“一锅能煮多少请求”。批更大通常吞吐更高，但等待合批的排队时间也会变长，显存占用也会上升，最后可能把 p99 延迟拉坏。

第二，KV cache 不是“有就行”，而是决定你能不能稳定扛住长上下文。长上下文请求一多，显存碎片和 cache 管理效率就会开始影响并发。如果 cache 策略差，单卡理论算力再高，也会因为装不下活跃会话而被迫降并发。

第三，副本数不是简单按平均流量配。真实服务要考虑：

- 高峰比平均高多少
- 单副本故障后剩余容量够不够
- 滚动发布时是否还满足 SLA
- 是否跨可用区部署

这些都会反映到 `C_loss` 和最终需要的总副本数上。

一个真实工程里很典型的情况是：方案 A 离线压测跑得比方案 B 高，但 A 的 p99 抖动更大，滚动升级期间容量掉得更明显，于是上线后不得不多加 30% 副本兜底。结果是 A 的“单机性能更强”，但系统总成本更高。

下面这张“调参影响链”表更贴近实际：

| 参数 | 影响 |
|---|---|
| `max_batch_size` | 吞吐、延迟、显存占用 |
| `max_num_tokens` | 调度效率、排队长度 |
| `instance_count` | 容灾能力和单位 token 成本 |
| KV cache 策略 | 显存利用率和长上下文吞吐 |

因此，很多在线聊天 API 的成本瓶颈，最终不是纯算力，而是调度、KV cache、排队和冗余策略。`vLLM`、`TensorRT-LLM` 这类系统之所以重要，不只是“快”，而是它们能把稳定吞吐做高，把为 SLA 付出的额外副本压低。

---

## 替代方案与适用边界

成本模型没有单一最优答案，不同业务场景的主导项不同。你必须先问：我的系统是在争取最低延迟、最高吞吐，还是最稳的上线复杂度？

下面先看场景差异：

| 场景 | 主要瓶颈 | 优化重点 |
|---|---|---|
| 在线聊天 API | 延迟、调度、KV cache | 稳定吞吐、批处理、冗余控制 |
| 长上下文推理 | 显存、cache、碎片 | cache 复用、分段调度 |
| 离线批处理 | 总吞吐 | 极限 batching、吞吐最大化 |
| 低延迟交互 | p95/p99 | 预热、副本冗余、快速路由 |

### 什么时候可以只看 `tok/s`

在方案粗筛阶段可以。比如你先排除明显太慢的卡型、明显不适合的部署框架，这时直接按 `tok/s` 看是高效的。但它只能做第一轮过滤，不能直接作为上线决策。

### 什么时候 `tokens/$` 才有意义

只有在 SLA 一致时，`tokens/$` 才能公平比较。如果一个方案的 p95 是 500ms，另一个方案的 p95 是 2s，它们的业务体验已经不同，不能只看单位 token 成本。

### 什么场景更适合追求极限吞吐

离线批处理通常更适合。比如 nightly 批量摘要、日志分析、文档分类，这类任务没有强 p95/p99 约束，可以把批处理开得更激进，把系统往总吞吐最大化方向调。

### 什么场景不能只追求低成本

面向用户的在线聊天、代码补全、语音助手都属于这一类。因为一旦延迟抖动明显，或者高峰期错误率上升，业务损失往往比节省的 GPU 钱更大。这时最低 `C_tok` 不一定是最优决策，稳定性和上线复杂度本身就是成本。

所以可以把几种常见比较方法总结成这样：

- 直接按 `tok/s` 排序：适合粗筛，不适合作为最终决策。
- 只按 `MFU` 比较：适合看算力利用率，不适合作为成本决策。
- 按 `tokens/$` 比较：更接近真实经营指标，但前提是 SLA 一致。
- 按“达到目标 SLA 所需总副本数”比较：最接近上线实际，因为它把冗余和稳定性一起考虑进来了。

最终的判断标准仍然是同一句话：不是谁卡便宜，也不是谁峰值高，而是谁能在相同 SLA 下，用更少的钱交付更多有效 token。

---

## 参考资料

1. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/vllm)
2. [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
3. [NVIDIA TensorRT Performance Optimization](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)
4. [NVIDIA TensorRT-LLM Triton Backend Model Config](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2540/user-guide/docs/tensorrtllm_backend/docs/model_config.html)
5. [NVIDIA TensorRT-LLM Scheduler](https://nvidia.github.io/TensorRT-LLM/torch/scheduler.html)
6. [NVIDIA NeMo AutoModel Performance Summary](https://docs.nvidia.com/nemo/automodel/latest/performance-summary.html)
