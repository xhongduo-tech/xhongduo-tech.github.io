## 核心结论

流水线并行（Pipeline Parallel, PP）是把模型按层切成多个 stage，再把这些 stage 放到不同 GPU 上，让一个 batch 拆出来的多个 microbatch 依次流过整条线。白话说，它像把一条很长的装配线拆给多个人做，目标首先是“这台机器终于能装下模型”，其次才是“吞吐别太差”。

它最核心的价值不是让单个请求天然更快，而是把原本单卡放不下的大模型拆开部署。以 70B 级 decoder 模型为例，单卡显存通常不够，先做 `PP=4` 才有“能不能上线”的基础，然后才轮到吞吐、延迟、稳定性这些二级问题。

PP 的收益高度依赖三个量：微批数量 `M`、stage 是否均衡、stage 之间的激活传输成本。微批太少，流水线头尾会有明显空转；某个 stage 慢，整条线都被它卡住；跨慢网络传激活，理论收益会被通信直接吃掉。

下面先给一句判断标准：PP 更适合“模型太大单卡放不下，但业务又需要稳定吞吐”的场景；它不适合把最低延迟当首要目标的在线推理。

| 场景 | PP 是否合适 | 原因 |
|---|---|---|
| 70B 模型单卡放不下 | 适合 | 先解决容量问题 |
| 多请求并发、可做微批 | 适合 | 可以用更多 `M` 填满流水线 |
| 小 batch 在线推理 | 往往不理想 | 气泡比例高，卡容易空转 |
| 跨节点慢网络部署 | 风险高 | 激活传输可能吞掉收益 |
| 追求极低 `p99` 延迟 | 通常不适合 | PP 更偏容量与吞吐优化 |

---

## 问题定义与边界

本文只讨论推理场景的流水线并行，不讨论训练里完整的前向加反向调度。推理里只有前向传播，关注点是吞吐、首 token 延迟、尾延迟，以及 stage 之间传输激活的成本。

先统一术语。stage 指模型被切开的一个连续层段，白话说就是“某张卡负责的一段层”。microbatch 指把一个大 batch 再切成更小的数据块，白话说就是“为了让流水线连续流动，先把货拆成多箱”。bubble 指流水线中的空转时间，也就是某些 stage 没活干、只能等待的时间。

| 符号 | 含义 | 作用 |
|---|---|---|
| `P` | stage 数 | 决定模型被切成几段 |
| `M` | microbatch 数 | 决定流水线能否被填满 |
| `t_i` | 第 `i` 个 stage 的计算时间 | 决定谁是慢 stage |
| `c_i` | 第 `i` 到 `i+1` 个 stage 的传输时间 | 决定通信代价 |
| `bubble` | 流水线空转时间 | 直接降低利用率 |

一个玩具例子：假设有 `P=4` 个 stage，把一个 batch 拆成 `M=8` 个 microbatch。`mb1` 先在 stage 0 跑，下一拍它流到 stage 1，同时 `mb2` 进入 stage 0。看起来四张卡都在干活，但开始几拍只有前面的卡忙，结束几拍只有后面的卡忙，这些头尾空档就是 bubble。

所以“层切得越多越好”是误解。`P` 变大确实能分摊显存，但如果 `M` 没同步变大，或者 `c_i` 很高，流水线只会更长、更空、更难调。

---

## 核心机制与推导

PP 的执行过程可以概括成三步：

1. 把模型层按顺序切成 `P` 个 stage。
2. 把一个 batch 拆成 `M` 个 microbatch。
3. 让每个 microbatch 按 stage 顺序前进，前一个 stage 算完就把激活传给下一个 stage。

如果先看最理想情况，假设每个 stage 的计算时间都一样，且通信可以忽略，那么流水线总长度近似由“启动阶段 + 稳态阶段 + 排空阶段”组成。启动和排空造成的那部分空转，常用近似公式表示为：

$$
bubble\_ratio \approx \frac{P-1}{M+P-1}
$$

对应利用率近似是：

$$
utilization \approx \frac{M}{M+P-1}
$$

这两个式子的直觉很直接：`P-1` 是头尾必然存在的空档数，`M` 越大，真正干活的部分越多，所以气泡占比会下降。当 `M \gg P` 时，

$$
bubble\_ratio \approx \frac{P-1}{M}
$$

也就是气泡大致按 `1/M` 衰减。

用题目要求的数值例子代入：

- 当 `P=4, M=8` 时：

$$
bubble\_ratio \approx \frac{4-1}{8+4-1} = \frac{3}{11} \approx 27.3\%
$$

- 当 `P=4, M=2` 时：

$$
bubble\_ratio \approx \frac{3}{5} = 60\%
$$

这说明同样是四段流水线，微批从 8 降到 2，空转会从大约四分之一直接升到六成。在线推理里如果并发不高，这就是为什么“看上去用了 4 张卡，吞吐却不线性增长”。

下面给一个简化时间轴。`S0~S3` 表示四个 stage，`m1~m8` 表示八个 microbatch。

| 时间片 | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| 1 | m1 | - | - | - |
| 2 | m2 | m1 | - | - |
| 3 | m3 | m2 | m1 | - |
| 4 | m4 | m3 | m2 | m1 |
| 5 | m5 | m4 | m3 | m2 |
| 6 | m6 | m5 | m4 | m3 |
| 7 | m7 | m6 | m5 | m4 |
| 8 | m8 | m7 | m6 | m5 |
| 9 | - | m8 | m7 | m6 |
| 10 | - | - | m8 | m7 |
| 11 | - | - | - | m8 |

第 1 到第 3 个时间片是“灌满流水线”，第 9 到第 11 个时间片是“排空流水线”。这些 `-` 就是 bubble。

但真实系统比公式复杂，因为总耗时更接近下面这个思路：

$$
T \approx (M + P - 1)\cdot \max_i(t_i) + \sum c_i + \text{imbalance penalty}
$$

这里 `imbalance penalty` 指 stage 不均衡额外造成的等待。白话说，最慢的那张卡决定节拍，传输越慢、各段越不均匀，理想公式越不准。

真实工程例子是 70B 级 decoder 推理。单卡放不下，于是把层切成 `PP=4`。如果某次 continuous batching 后只攒出 2 个 microbatch，那么即使模型能跑，四张卡里也会有很多等待时间；如果这四段还分布在两个节点之间，激活跨节点传输又会再加一层损耗。此时 PP 解决了“能部署”，但不一定解决“高效率”。

---

## 代码实现

实现 PP 时，真正要做的不是“把模型按层切开”这么简单，而是两件事一起成立：

1. 每个 stage 只持有自己负责的层。
2. microbatch 能在 stage 之间稳定 send/recv，形成连续流。

先看一个可运行的 Python 玩具实现。它不依赖 GPU，只模拟流水线节拍和气泡公式，用来验证直觉。

```python
from math import isclose

def bubble_ratio(P: int, M: int) -> float:
    assert P >= 1 and M >= 1
    return (P - 1) / (M + P - 1)

def utilization(P: int, M: int) -> float:
    assert P >= 1 and M >= 1
    return M / (M + P - 1)

def pipeline_schedule(P: int, M: int):
    steps = M + P - 1
    table = []
    for t in range(steps):
        row = []
        for s in range(P):
            mb = t - s
            row.append(mb + 1 if 0 <= mb < M else None)
        table.append(row)
    return table

assert isclose(bubble_ratio(4, 8), 3 / 11, rel_tol=1e-9)
assert isclose(bubble_ratio(4, 2), 3 / 5, rel_tol=1e-9)
assert isclose(utilization(4, 8), 8 / 11, rel_tol=1e-9)

sched = pipeline_schedule(4, 4)
assert sched[0] == [1, None, None, None]
assert sched[3] == [4, 3, 2, 1]
assert sched[-1] == [None, None, None, 4]
```

如果把它翻译成工程中的伪代码，结构大致如下：

```text
microbatches = split_batch(batch, M)

for microbatch in microbatches:
    x = recv_from_prev_stage_or_input(microbatch)
    y = forward_layers_for_this_stage(x)
    send_to_next_stage(y)
```

真实系统还要处理两个问题。第一，stage 0 从输入队列拿 token embedding，最后一个 stage 输出 logits；中间 stage 则只收发激活。第二，调度不是只跑一轮 `for`，而是要处理灌满、稳态、排空三个阶段。

在 Megatron 一类实现里，常见配置长这样：

| 配置项 | 作用 | 常见含义 |
|---|---|---|
| `pipeline_model_parallel_size` | PP 规模 | 模型切成多少个 stage |
| `virtual_pipeline_model_parallel_size` | 虚拟 PP 规模 | 一个物理 stage 再切成多个虚拟 chunk，用交错调度减 bubble |
| `tensor_model_parallel_size` | TP 规模 | 单层内部再切分张量 |

一个典型组合可以写成下面这样理解：

| `PP` | `virtual PP` | `TP` | 含义 |
|---|---|---|---|
| 1 | 1 | 4 | 只做张量并行 |
| 4 | 1 | 1 | 只做流水线并行 |
| 4 | 2 | 2 | 4 个物理 stage，每个 stage 再切成 2 个虚拟段，同时每层做 2 路 TP |

交错流水线（interleaved pipeline）的核心直觉是：把每个物理 stage 再拆细，让不同 microbatch 在更细粒度的子段上穿插执行。这样不能消灭通信，但通常能减少头尾气泡，让流水线更接近稳态。

---

## 工程权衡与常见坑

PP 最大的误区是把“分层”理解成纯结构问题，实际上它首先是调度和系统问题。下面这张表比口头提醒更有用。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 按层数切，不按耗时切 | 某个 stage 成为瓶颈，整条线等它 | 先 profile 每层耗时、KV cache、attention 开销，按时间切 |
| 微批太小 | bubble 高，卡利用率低 | 提高并发，做 continuous batching，让 `M` 明显大于 `P` |
| 跨慢网络 | 激活传输吞掉收益 | stage 尽量放在同机或同一高速互联域 |
| 忽略激活通信 | 理论吞吐和实测差很多 | 估算 `c_i`，尽量 overlap p2p 通信与计算 |
| 把 PP 当成低延迟手段 | `p99` 反而变差 | 把 PP 用于容量和稳定吞吐，不把它当首选低延迟方案 |

这里再强调一次 `M=2, P=4` 的例子。光看公式，bubble 已经到 `60%`。如果再叠加通信和慢 stage，实际可用吞吐可能比“更强 batching 的 TP 方案”还差。也就是说，PP 能让模型跑起来，不代表它一定是当下最优解。

另一个常见坑是忽略 stage 间激活大小。推理时虽然不做反向，但 decoder 的中间激活、KV cache 访问、不同序列长度带来的 attention 成本，都会让 `t_i` 和 `c_i` 偏离“每层差不多”这个假设。特别是长上下文场景，后半段层的注意力成本常常更重，按层数平均切很容易出问题。

---

## 替代方案与适用边界

PP 不是并行方案的总开关，它只解决“模型太大、单卡装不下”这一类问题。工程上更稳的思路通常是按问题类型选工具，而不是先决定“我要上 PP”。

| 方案 | 主要收益 | 主要代价 | 适用场景 |
|---|---|---|---|
| TP | 单层加速、分摊单层参数 | all-reduce 通信频繁 | 中等到较大模型，机内高速互联 |
| PP | 装下更大模型、稳定吞吐 | 气泡、调度复杂、激活传输 | 大模型推理，单卡放不下 |
| DP | 扩副本提总吞吐 | 显存按副本重复 | 多请求、模型单卡能放下 |
| 量化 | 显著降显存 | 精度和算子兼容风险 | 显存紧张、追求成本 |
| 连续 batching | 提升吞吐和卡利用率 | 调度复杂 | 在线推理服务 |
| speculative decoding | 降解码延迟 | 需要草稿模型和接受机制 | 高吞吐、低延迟生成 |

选择顺序通常可以写成一句工程规则：

1. 先看单卡能不能放下。
2. 放不下时，先评估量化和 TP 是否足够。
3. 单层也难承载、总参数仍超预算时，再引入 PP。
4. 在线推理如果 batch 很小且跨节点，优先考虑 `TP + 更强 batching`，不要一上来就假设 PP 会更快。

一个真实工程判断例子：如果服务流量低、请求零散、每轮 dynamic batching 后只形成很小的 batch，而机器还是跨节点部署，那么引入 PP 往往先带来复杂度，再带来很有限的吞吐收益。相反，如果是单机多卡、NVLink 互联、70B 模型又必须上线，PP 就是合理选择，因为它先解决容量边界，再通过 continuous batching 和虚拟 stage 慢慢把利用率拉回来。

所以 PP 的适用边界可以总结成一句话：它最擅长解决“模型太大”的问题，不擅长单独解决“请求必须极低延迟”的问题。

---

## 参考资料

1. [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://papers.nips.cc/paper_files/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf)  
支持本文的气泡、微批和流水线利用率直觉，是经典理论来源。

2. [NVIDIA Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)  
支持本文关于 `pipeline_model_parallel_size`、交错流水线和多种并行组合方式的工程说明。

3. [Megatron Core model_parallel_config API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.model_parallel_config.html)  
支持本文对 `virtual_pipeline_model_parallel_size` 等关键配置项的解释。

4. [Megatron-LM pipeline schedules source](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)  
支持本文关于实际调度、灌满与排空阶段、P2P 发送接收的实现层描述。

5. [DeepSpeed Inference Overview](https://www.deepspeed.ai/inference/)  
支持本文关于多 GPU 推理目标、吞吐优化和在线服务背景的工程视角。

6. [DeepSpeed Inference: Multi-GPU inference and kernel optimization](https://www.deepspeed.ai/2021/03/15/inference-kernel-optimization.html)  
支持本文关于小 batch 推理、系统瓶颈和工业部署优化方向的说明。
