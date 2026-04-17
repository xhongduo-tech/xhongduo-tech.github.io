## 核心结论

ZeRO++ 是 ZeRO Stage 3 的通信优化版本。它不改变“参数、梯度、优化器状态分片”这个基本框架，而是专门压缩三类最贵的跨机通信：

1. `qwZ`：把前向阶段的权重 `all-gather` 从 FP16 临时量化到低精度再传输。
2. `hpZ`：在每个节点内保留一份完整权重副本，把反向阶段原本跨机的权重 `all-gather` 变成机内通信。
3. `qgZ`：把梯度 `reduce-scatter` 改写成分层 `all-to-all` 流程，在通信前量化、在规约前反量化。

如果把某一层需要跨机交换的参数量记为 $M$，ZeRO 的三段通信大致是：

$$
C_{\text{ZeRO}} = M_{\text{fw}} + M_{\text{bw}} + M_{\text{grad}} = 3M
$$

ZeRO++ 的目标是把它压到：

$$
C_{\text{ZeRO++}} = 0.5M_{\text{fw}} + 0M_{\text{bw}} + 0.25M_{\text{grad}} = 0.75M
$$

也就是把跨机通信量从 `3M` 降到 `0.75M`，总量变成原来的四分之一。论文报告在 384 张 GPU 规模下，吞吐最高可提升到原来的 2.16 倍；官方博客在 400 Gbps 跨机互联场景下给出的实测提升是最高 1.56 倍。两组数字对应的实验条件不同，但结论一致：当训练受网络带宽限制时，ZeRO++ 的收益非常明显。

一个直观记法是：ZeRO 原来每层要“跨机搬三次货”，ZeRO++ 变成“前向搬半箱、反向第二次不跨机搬、梯度只搬四分之一箱”。

---

## 问题定义与边界

先定义问题。ZeRO 是“零冗余优化器”，白话解释就是：不要让每张 GPU 都完整保存优化器状态、梯度和参数副本，而是把这些大对象拆开存到不同 GPU 上。这样能把单卡显存压力降下来，所以大模型训练经常依赖 ZeRO Stage 3。

但 ZeRO Stage 3 有一个代价：显存省下来了，通信变多了。尤其在两种场景里，这个代价会变成主瓶颈：

| 场景 | 为什么容易卡住 | 表现 |
|---|---|---|
| 跨机带宽有限 | 参数和梯度要频繁跨节点交换 | GPU 算得不慢，但经常等网络 |
| micro-batch 很小 | 每次算的 token 少，计算时间短 | 通信时间占比被放大 |

ZeRO 的核心三段通信可以粗略写成下面这张表：

| 通信阶段 | ZeRO 跨机数据量 | ZeRO++ 对应优化 | 误差风险 |
|---|---:|---|---|
| 前向权重 `all-gather` | $M$ | `qwZ` 压到 `0.5M` | 有量化误差，需要块级量化 |
| 反向权重 `all-gather` | $M$ | `hpZ` 变成 `0` 跨机流量 | 主要是内存开销，不是数值误差 |
| 梯度 `reduce-scatter` | $M$ | `qgZ` 压到 `0.25M` | 不能直接低精度规约，否则误差累积 |

所以 ZeRO++ 的边界很明确：它不是重新设计训练器，而是在 ZeRO-3 上替换通信路径。也因此它的使用边界也很明确：

- 适合：多机多卡、大模型、带宽相对紧张、micro-batch 偏小的训练。
- 不一定值得：单机训练、节点间网络很强且计算更重、显存已经非常紧张的环境。
- 工程前提：需要 DeepSpeed 支持对应配置项，包括 `zero_quantized_weights`、`zero_quantized_gradients`、`zero_hpz_partition_size`。

对初学者可以这样理解。原本每个 GPU 每层都要来回搬三箱数据：前向一箱、反向一箱、梯度一箱。ZeRO++ 做了三件事：

- 第一箱先压缩再搬。
- 第二箱只在本机内部搬，不跨机。
- 第三箱先拆块、重排、压缩后只跨机搬一部分。

因此真正走慢链路的货量大幅下降。

---

## 核心机制与推导

### 1. qwZ：压缩前向权重通信

`qwZ` 是 quantized weights for ZeRO，白话解释就是“传权重前先量化”。ZeRO 在前向阶段需要为当前层临时收集完整权重，默认是 FP16，也就是每个参数 2 字节。ZeRO++ 在传输前先把它压成低精度，例如 INT8，每个参数 1 字节，收完再反量化。

如果某层有 $M$ 个参数，那么前向通信量从：

$$
2M \text{ bytes (FP16)}
$$

近似降到：

$$
1M \text{ bytes (INT8)}
$$

所以按参数个数计，跨机通信从 $M$ 降为 $0.5M$。

这里不能简单粗暴地全局量化，因为不同权重分布差异很大。ZeRO++ 用的是块级量化，白话解释就是“把大张量切成很多小块，每块单独算缩放系数”，这样量化误差更可控。

### 2. hpZ：把反向跨机 `all-gather` 变成机内通信

`hpZ` 是 hierarchical partitioning for ZeRO，白话解释就是“做两级分片”。普通 ZeRO-3 会把完整模型权重分散到整个数据并行组，因此反向计算每层梯度时，也要跨所有机器重新收集这一层的权重。

hpZ 的做法是：在每个节点内再保留一份完整模型副本。代价是机内显存占用变大，好处是反向需要的权重不再从别的节点拿，而是只在本节点内部拿。因为 NVLink、PCIe 或机内拓扑带宽远高于跨机网络，所以这一步基本把跨机反向权重通信消掉了。

因此：

$$
C_{\text{bw}} = M \rightarrow 0
$$

注意，这里是“跨机通信”降到 0，不是“完全没有通信”。机内还是要通信，但成本小很多。

### 3. qgZ：先量化通信，再高精度规约

`qgZ` 是 quantized gradients for ZeRO，白话解释就是“梯度也压缩传，但不能直接用低精度做最终求和”。梯度通信比权重通信难，因为 `reduce-scatter` 不只是搬运，还包含规约。若直接在 INT4 或 INT8 上连续做加法，误差会累积并放大。

ZeRO++ 的解决思路不是“低精度 reduce-scatter”，而是把它改写成分层 `all-to-all`：

1. 梯度切片并重排，保证最终每张卡拿到正确分片。
2. 机内先量化、通信、反量化，再做局部高精度规约。
3. 跨机再次量化、通信、反量化，再做最终高精度规约。

核心点只有一句：**低精度只用于传输，不用于最终规约。**

论文与博客给出的推导是：设每个节点有 $N$ 张 GPU，量化压缩比为 $Z$，模型大小为 $M$。

如果不做分层，单跳跨机 `all-to-all` 的总流量可写成：

$$
\frac{M \times N}{Z}
$$

采用分层方案后，每张 GPU 的跨机流量从 $\frac{M}{Z}$ 降到 $\frac{M}{Z \times N}$，于是总跨机流量变成：

$$
\frac{M \times N}{Z \times N} = \frac{M}{Z}
$$

这就是 qgZ 为什么能把梯度跨机通信压缩到 `M/Z` 量级。若取 4 倍压缩，可以近似写成 `0.25M`。

### 4. 玩具例子：4 卡、单层参数量为 M

先看最小理解模型。假设只有一层，参数量是 $M$，并且把 ZeRO 的三段跨机通信都按 `M` 记。

| 方案 | 前向权重 | 反向权重 | 梯度规约 | 合计 |
|---|---:|---:|---:|---:|
| ZeRO | $M$ | $M$ | $M$ | $3M$ |
| ZeRO++ | $0.5M$ | $0$ | $0.25M$ | $0.75M$ |

因此：

$$
\frac{C_{\text{ZeRO++}}}{C_{\text{ZeRO}}} = \frac{0.75M}{3M} = 0.25
$$

这就是“4 倍通信削减”的来源。

如果用一个更白话的版本描述：原来你每层要跨机搬 3 次文件，现在变成只搬 `0.5 + 0 + 0.25 = 0.75` 次等价文件。

### 5. 真实工程例子：384 张 V100 的多机训练

官方博客给的是 384 张 NVIDIA V100，跨机网络为 4 路 100 Gbps InfiniBand，也就是 400 Gbps。这个场景下，1k token 每卡时，ZeRO++ 相比 ZeRO-3 的吞吐提升约 28% 到 36%；图中最高可到 1.56 倍。论文页面则总结为在 384 GPU 规模下最高 2.16 倍吞吐，并指出 RLHF 相比 vanilla ZeRO 可加速 3.3 倍。

这说明两件事：

- ZeRO++ 的收益主要取决于“通信是不是瓶颈”。
- 带宽越紧、batch 越小、模型越大，收益通常越高。

---

## 代码实现

ZeRO++ 的优点是大多数情况下不需要改模型代码，重点是 DeepSpeed 配置。下面先给一个最小可复制的配置片段，再用一个 Python 小脚本演示通信量计算。

### 1. DeepSpeed 配置示例

```json
{
  "zero_optimization": {
    "stage": 3,
    "reduce_bucket_size": 10000000,
    "reduce_scatter": true,

    "zero_quantized_weights": true,
    "zero_hpz_partition_size": 8,
    "zero_quantized_gradients": true,

    "contiguous_gradients": true,
    "overlap_comm": true
  }
}
```

这几个字段的含义如下：

| 配置项 | 作用 | 常见取值 |
|---|---|---|
| `stage` | 启用 ZeRO Stage 3 | `3` |
| `zero_quantized_weights` | 打开 `qwZ` | `true` |
| `zero_quantized_gradients` | 打开 `qgZ` | `true` |
| `zero_hpz_partition_size` | 指定 hpZ 次级分组大小 | 通常设为“每节点 GPU 数” |
| `reduce_scatter` | 保持 ZeRO 梯度分片规约路径 | `true` |
| `contiguous_gradients` | 让梯度内存更连续，减少碎片和调度开销 | `true` |
| `overlap_comm` | 尝试把通信和计算重叠 | `true` |

如果你的单节点有 8 张卡，`zero_hpz_partition_size` 一般就设为 `8`；如果单节点有 16 张卡，就常见设为 `16`。它的本质是告诉 DeepSpeed：“hpZ 的机内组有多大”。

### 2. 可运行的通信量计算脚本

```python
def zero_volume(M: float) -> float:
    # ZeRO: forward all-gather + backward all-gather + gradient reduce-scatter
    return M + M + M


def zeropp_volume(M: float, fw_ratio: float = 0.5, bw_ratio: float = 0.0, grad_ratio: float = 0.25) -> float:
    return fw_ratio * M + bw_ratio * M + grad_ratio * M


def speedup_from_bandwidth_bound(M: float) -> float:
    # 在纯带宽受限的理想模型下，速度提升约等于通信量倒数比例
    return zero_volume(M) / zeropp_volume(M)


M = 100.0
assert zero_volume(M) == 300.0
assert zeropp_volume(M) == 75.0
assert zeropp_volume(M) / zero_volume(M) == 0.25
assert speedup_from_bandwidth_bound(M) == 4.0

print("ZeRO volume:", zero_volume(M))
print("ZeRO++ volume:", zeropp_volume(M))
print("Ideal bandwidth-bound speedup:", speedup_from_bandwidth_bound(M))
```

这个脚本不是 DeepSpeed 实现，而是一个“玩具计算器”。它表达的工程含义是：如果训练几乎完全被跨机通信卡住，那么通信量缩成四分之一，理想上吞吐上限可以接近 4 倍；但真实系统里还受算力、调度、kernel、机内通信、重叠效率等因素影响，所以实际通常远小于 4 倍。

### 3. 真实训练中的接入方式

如果你已经有标准 ZeRO-3 训练脚本，接入 ZeRO++ 常见只要两步：

1. 在 DeepSpeed 配置中打开三个选项。
2. 按原来的方式启动训练，例如：

```bash
deepspeed train.py --deepspeed_config deepspeed_config.json
```

模型前向、反向、优化器更新的 Python 代码通常不需要改。真正需要确认的是：你的 DeepSpeed 版本、CUDA 环境、分布式启动参数和节点拓扑是否支持这些低精度通信路径。

---

## 工程权衡与常见坑

ZeRO++ 不是“免费午餐”。它用更复杂的通信换吞吐，因此有明确代价。

| 风险点 | 具体问题 | 缓解方式 |
|---|---|---|
| 量化误差 | 低精度传输可能影响收敛 | 用块级量化，不要把通信量化和低精度规约混为一谈 |
| 额外显存 | hpZ 会在节点内保留完整权重副本 | 先算清单节点剩余显存，再设置 `zero_hpz_partition_size` |
| 配置不一致 | 只开一个开关，收益不完整 | 按 `qwZ -> qgZ -> hpZ` 逐步启用并测吞吐 |
| 调优错位 | 网络不慢却强行上 ZeRO++ | 先 profile，确认瓶颈确实在跨机通信 |
| 误读收益 | 把“4 倍通信削减”误当“4 倍训练提速” | 吞吐受计算、重叠、内核实现共同影响 |

最容易踩的坑有三个。

第一，**不要把“量化通信”理解成“低精度训练”**。ZeRO++ 的重点是“传输时变小，规约时回到高精度”，尤其 qgZ 必须在规约前反量化。如果你把整个梯度规约都留在低精度，误差会明显变坏。

第二，**hpZ 的收益和代价是一一对应的**。它之所以能把反向跨机 `all-gather` 消掉，是因为你在每个节点内多存了一份完整权重副本。对显存已经很紧的任务，这一步可能根本放不下。

第三，**收益和拓扑强相关**。在 384 张 V100、400 Gbps 互联、micro-batch 很小的场景，ZeRO++ 明显有效；如果你的训练本来主要卡在矩阵乘法、attention kernel 或数据加载，那么通信压缩不会给你同样级别的收益。

一个典型的真实工程判断流程是：

- 如果你看 profile 发现大量时间在 `all-gather` 和 `reduce-scatter` 上，ZeRO++ 值得试。
- 如果只开 `qwZ` 后吞吐提升有限，常见原因是反向跨机 `all-gather` 还在，此时要看 hpZ。
- 如果开了 qgZ 但 loss 波动明显，要优先检查是否是版本、配置或量化路径问题，而不是先怀疑模型结构。

---

## 替代方案与适用边界

不是所有训练都该直接全开 ZeRO++。更稳妥的办法是按瓶颈选择。

| 条件 | 推荐方案 | 原因 |
|---|---|---|
| 单机或跨机通信很少 | 标准 ZeRO-3 | 通信压缩收益有限，复杂度不值 |
| 带宽一般，先求低风险 | ZeRO-3 + `qwZ` | 先压前向权重通信，验证收敛稳定性 |
| 带宽明显受限，节点内显存有余量 | `qwZ + hpZ` | 先消掉最痛的反向跨机 `all-gather` |
| 强带宽瓶颈，目标是最大吞吐 | `qwZ + hpZ + qgZ` | 才能接近 4 倍通信缩减 |
| 显存很紧 | 谨慎使用 hpZ | hpZ 可能因副本成本而不可行 |

对初学者，最实用的启用顺序通常是：

1. 先开 `zero_quantized_weights`，检查 loss 曲线和吞吐。
2. 再开 `zero_quantized_gradients`，观察梯度通信收益。
3. 最后按每节点 GPU 数设置 `zero_hpz_partition_size`，评估显存是否够。

这样做的原因很简单：量化问题和内存问题最好分开排查。一次性全开，出了问题你很难判断到底是数值误差、版本兼容、还是 hpZ 分组导致的。

还有一点要讲清楚。ZeRO++ 不是唯一的“通信优化”方向。它适合的前提是：你已经决定使用 ZeRO-3 做分片训练，并且主要瓶颈确实在跨机 collectives。如果你的系统更适合张量并行、流水线并行、FSDP 或更激进的混合并行，那么最优解可能不是 ZeRO++，而是先换并行策略，再考虑是否需要通信压缩。

---

## 参考资料

1. [ZeRO++: Extremely Efficient Collective Communication for Large Model Training](https://www.microsoft.com/en-us/research/publication/zero-extremely-efficient-collective-communication-for-large-model-training/)
   重点：论文摘要、三项优化总览、`4x` 通信削减、`384 GPU` 下最高 `2.16x` 吞吐，以及 RLHF `3.3x` 加速结论。

2. [DeepSpeed ZeRO++: A leap in speed for LLM and chat model training with 4X less communication](https://www.microsoft.com/en-us/research/blog/deepspeed-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/)
   重点：`qwZ`、`hpZ`、`qgZ` 的机制图与通信量表，`3M -> 0.75M` 的推导，以及 `400 Gbps` 场景下的吞吐实测。

3. [DeepSpeed ZeRO++ Tutorial](https://www.deepspeed.ai/tutorials/zeropp/)
   重点：工程配置项说明，包括 `zero_quantized_weights`、`zero_quantized_gradients`、`zero_hpz_partition_size` 以及 ZeRO++ 的最小配置示例。
