## 核心结论

MFU、HFU 和 Roofline 模型解决的是同一个问题：训练为什么没有跑满硬件，以及应该先优化哪一部分。

先给结论。

| 指标/概念 | 定义 | 解决的问题 |
|---|---|---|
| MFU | 模型理论 FLOPs 利用率，即“模型本来需要做的计算”占峰值算力的比例 | 这次训练任务本身把硬件吃满了多少 |
| HFU | 硬件实际 FLOPs 利用率，即“硬件实际上做过的计算”占峰值算力的比例 | 硬件是不是一直很忙 |
| Roofline | 用算术强度与峰值带宽/峰值算力共同判断瓶颈 | 当前算子到底卡算力还是卡带宽 |

三组公式是全文的主线：

$$
\text{MFU}=\frac{F_{\text{model}}/t}{F_{\text{peak}}}
$$

$$
\text{HFU}=\frac{(F_{\text{model}}+F_{\text{recompute}})/t}{F_{\text{peak}}}
$$

$$
I=\frac{\text{FLOPs}}{\text{Bytes}}, \quad
P_{\text{attainable}}=\min(F_{\text{peak}}, I \cdot BW_{\text{peak}})
$$

$$
I_{\text{ridge}}=\frac{F_{\text{peak}}}{BW_{\text{peak}}}
$$

这里的“算术强度” $I$ 可以理解成“每搬 1 字节数据，顺手做了多少次浮点运算”。如果这个值很低，说明大部分时间都花在搬数据，属于 memory-bound；如果这个值很高，说明数据搬运成本被摊薄，更可能进入 compute-bound。

一个最小例子就够说明问题。假设 H100 的 BF16 峰值算力记为 989 TFLOPS，某训练循环实测只有 400 TFLOPS，那么：

$$
\text{MFU} \approx \frac{400}{989} \approx 40.4\%
$$

如果没有激活重算，HFU 也约等于 40.4%。但如果启用了 activation checkpointing，硬件会额外做一部分“重新算一遍”的 FLOPs，HFU 会升高，而 MFU 不变。于是通常有：

$$
\text{MFU} \le \text{HFU}
$$

所以，MFU 更适合看“训练算法和实现还差多少效率空间”，HFU 更适合看“卡有没有闲着”。

---

## 问题定义与边界

先把三个边界说清楚，否则后面很容易混淆。

第一，MFU 的分子不是“GPU 实际跑过的所有 FLOPs”，而是模型理论上完成一次前向和反向应该做的 FLOPs。白话说，MFU 统计的是“应付账单”，不统计重复付款。HFU 才会把激活重算、重复 kernel、低效调度带来的额外 FLOPs 也算进去。

第二，Roofline 不关心某个模型叫 Transformer 还是 CNN，它只关心两件事：这段计算总共做了多少 FLOPs，搬了多少字节。二者一除就是算术强度：

$$
I = \frac{\text{FLOPs}}{\text{Bytes}}
$$

如果某硬件峰值算力是 $F_{\text{peak}}$，峰值带宽是 $BW_{\text{peak}}$，那么脊点是：

$$
I_{\text{ridge}} = \frac{F_{\text{peak}}}{BW_{\text{peak}}}
$$

当 $I < I_{\text{ridge}}$ 时，性能近似受 $I \cdot BW_{\text{peak}}$ 限制；当 $I > I_{\text{ridge}}$ 时，性能才可能接近 $F_{\text{peak}}$。

第三，MFU/HFU 是整体视角，Roofline 是局部视角。整体视角回答“这一轮训练效率高不高”，局部视角回答“具体哪个算子卡住了”。

下面这张表把问题边界压缩到最小。

| 维度 | MFU | HFU |
|---|---|---|
| 统计对象 | 理论 forward + backward FLOPs | 理论 FLOPs + 重算 FLOPs + 其他额外计算 |
| 是否受 checkpointing 影响 | 基本不变 | 会升高 |
| 更适合比较什么 | 模型、batch、并行策略、训练成本 | 硬件繁忙程度、系统层调度效果 |
| 常见误读 | 把低 MFU 误解为“GPU 没干活” | 把高 HFU 误解为“训练效率已经很好” |

对应到 Roofline，还要补三项硬件边界参数：

| 参数 | 含义 | 用途 |
|---|---|---|
| $F_{\text{peak}}$ | 峰值 FLOPS | 算力天花板 |
| $BW_{\text{peak}}$ | 峰值带宽 | 带宽斜率 |
| $I_{\text{ridge}}$ | 脊点 | 判断 compute-bound / memory-bound |

玩具例子可以这样看。某算子做了 120 GFLOPs，搬了 60 GB 数据，则 $I=2$ FLOP/byte。若某卡的脊点是 100 FLOP/byte，那么这个算子远在脊点左侧，说明你优化乘法次数意义不大，先减少数据搬运更有效。

---

## 核心机制与推导

从模型训练到 Roofline，可以连成一条公式链。

一次训练迭代中，模型理论 FLOPs 记为 $F_{\text{model}}$，迭代耗时记为 $t$。那么理论平均吞吐是：

$$
P_{\text{model}}=\frac{F_{\text{model}}}{t}
$$

MFU 就是把它除以峰值算力：

$$
\text{MFU}=\frac{P_{\text{model}}}{F_{\text{peak}}}
$$

如果启用了激活重算，额外多做了 $F_{\text{recompute}}$ FLOPs，那么硬件实际吞吐是：

$$
P_{\text{hw}}=\frac{F_{\text{model}}+F_{\text{recompute}}}{t}
$$

HFU 则是：

$$
\text{HFU}=\frac{P_{\text{hw}}}{F_{\text{peak}}}
$$

这解释了为什么重算会把 HFU 拉高，但不会改善 MFU。因为它只是让硬件更忙，不是让“完成同样训练目标所需的理论工作量”减少。

接下来是 Roofline。设某个算子做了总 FLOPs 为 $F$，搬运字节数为 $B$，则：

$$
I=\frac{F}{B}
$$

这时该算子在这块硬件上的可达性能上界为：

$$
P_{\text{attainable}}=\min(F_{\text{peak}}, I \cdot BW_{\text{peak}})
$$

这个公式非常重要。它说的是：真正决定上界的，不只是“卡最快能算多少”，还包括“你是否能用足够高的算术强度把带宽成本摊掉”。

把它放进 Transformer，差别就出来了。

以自注意力为例，朴素实现的主要 FLOPs 大约是 $O(N^2 d)$，其中 $N$ 是序列长度，$d$ 是 head 维度。白话说，序列越长，token 两两交互的成本越高。问题在于，朴素实现往往会把完整 attention score 矩阵写回高带宽显存 HBM，再读出来做 softmax、再写回、再读回参与后续乘法。这样 FLOPs 没少，但字节搬运极大，于是 $I$ 偏低。

一个常见估算是：普通 attention 的算术强度可能在约 64 FLOP/byte，而经过 FlashAttention 这类 IO-aware 重排后，FLOPs 量级近似不变，但中间矩阵不再完整 materialize 到 HBM，字节搬运显著下降，算术强度可提升到约 506 FLOP/byte。若 A100 的脊点约在 156 FLOP/byte 左右，那么前者在脊点左边，后者已跨过脊点。

这就是“减少 IO 比减少 FLOPs 更重要”的来源。不是说 FLOPs 不重要，而是说在 memory-bound 区域，优化乘法次数常常没有先优化访存来得有效。

再看 FFN。Transformer 的前馈网络主要是大矩阵乘法，数据复用度更高。白话说，同一批权重和激活在片上缓存中可被反复使用，因此每搬一次数据能做更多乘法。于是 FFN 的算术强度通常高于 naive attention，更容易落在 Roofline 的平顶附近，也更接近 compute-bound。

真实工程例子就是 FlashAttention。它没有改变注意力的数学定义，也没有把 $O(N^2)$ 改成更低复杂度；它做的是重排计算与访存路径，让更多中间量只在 SRAM 或寄存器级别短暂停留，通过 tiling 和 online softmax 降低 HBM 往返。结果是 attention 从“带宽受限”向“算力受限”移动，这时 MFU 和端到端吞吐都更容易提升。

---

## 代码实现

下面给一个最小可运行版本，用来把前面的公式落成数字。单位统一采用 TFLOPs、秒、TB/s，便于直接代入常见 GPU 参数。

```python
def compute_utilizations(measured_flops, measured_time, peak_flops, recompute_flops=0.0):
    """
    measured_flops: 理论模型 FLOPs，单位 TFLOPs per step
    measured_time: 单步耗时，单位 s
    peak_flops: 硬件峰值算力，单位 TFLOPS
    recompute_flops: 额外重算 FLOPs，单位 TFLOPs per step
    """
    actual_model_tflops = measured_flops / measured_time
    actual_hw_tflops = (measured_flops + recompute_flops) / measured_time
    mfu = actual_model_tflops / peak_flops
    hfu = actual_hw_tflops / peak_flops
    return actual_model_tflops, actual_hw_tflops, mfu, hfu


def roofline_intensity(total_flops, bytes_moved, peak_flops, peak_bw):
    """
    total_flops: 总 FLOPs，单位 TFLOPs
    bytes_moved: 总数据搬运，单位 TB
    peak_flops: 峰值算力，单位 TFLOPS
    peak_bw: 峰值带宽，单位 TB/s
    """
    intensity = total_flops / bytes_moved
    attainable = min(peak_flops, intensity * peak_bw)
    ridge = peak_flops / peak_bw
    return intensity, attainable, ridge


# 玩具例子：A100 BF16 峰值算力近似记为 312 TFLOPS
model_flops = 624.0     # 一步理论需要 624 TFLOPs
step_time = 4.0         # 跑了 4 秒
peak_flops = 312.0      # A100 BF16 峰值 TFLOPS
recompute = 312.0       # 额外重算 312 TFLOPs

model_tflops, hw_tflops, mfu, hfu = compute_utilizations(
    model_flops, step_time, peak_flops, recompute
)

assert round(model_tflops, 2) == 156.00
assert round(mfu, 2) == 0.50
assert round(hw_tflops, 2) == 234.00
assert round(hfu, 2) == 0.75

# Roofline 例子：同一个算子做 4 TFLOPs，搬 0.0625 TB 数据
# 则 intensity = 64 FLOP/byte（这里把 T 与 T 相消，数值等价）
intensity, attainable, ridge = roofline_intensity(
    total_flops=4.0,
    bytes_moved=0.0625,
    peak_flops=312.0,
    peak_bw=2.0
)

assert round(intensity, 2) == 64.00
assert round(ridge, 2) == 156.00
assert round(attainable, 2) == 128.00
```

这个例子说明三件事。

第一，单步理论 FLOPs 是 624 TFLOPs，4 秒跑完，所以模型实际吞吐是 156 TFLOPS，MFU 为 50%。这表示“从模型理论需求看”，只吃到了峰值算力的一半。

第二，如果额外又做了 312 TFLOPs 的重算，那么硬件实际吞吐是 234 TFLOPS，HFU 为 75%。这表示“从硬件是否繁忙看”，卡其实更忙，但训练任务本身并没有更省。

第三，若某算子算术强度只有 64 FLOP/byte，而脊点是 156，那么它最多只能达到 $64 \times 2 = 128$ TFLOPS，离 312 TFLOPS 的算力平顶还远。这时你再纠结 Tensor Core 理论峰值，意义不大，因为瓶颈根本不在那。

真实工程中，流程通常是：

| 步骤 | 需要的数据 | 产出 |
|---|---|---|
| 统计模型理论 FLOPs | 参数规模、层数、序列长度、batch | $F_{\text{model}}$ |
| 记录单步时间 | profiler 或训练日志 | $t$ |
| 查硬件峰值 | GPU 规格表 | $F_{\text{peak}}, BW_{\text{peak}}$ |
| 估算重算 FLOPs | checkpointing 配置、profile | $F_{\text{recompute}}$ |
| 估算 bytes moved | kernel 读写量、HBM 访问量 | $I$ 与 Roofline 位置 |

---

## 工程权衡与常见坑

最常见的错误，不是公式算错，而是问题边界设错。

| 常见坑 | 现象 | 优化意图 | 规避策略 |
|---|---|---|---|
| 物化 attention 矩阵 | attention 很慢，HBM 读写高 | 降低 IO、提高算术强度 | 用 tiling、online softmax、FlashAttention 类实现 |
| 忽略非 matmul 开销 | 只盯 GEMM，端到端提速不明显 | 找真实瓶颈 | 把 softmax、dropout、LayerNorm、残差读写一并计入带宽预算 |
| 只看 MFU 不看 HFU | 以为低 MFU 就是卡没跑满 | 区分“硬件忙”和“模型效率高” | 同时报告 MFU、HFU 与 step time |

第一个坑最典型。很多人觉得 attention 的问题是 $N^2$ FLOPs 太大，于是只盯着“减少乘法次数”。但在中等长度甚至较长序列下，朴素 attention 经常先被 HBM 带宽卡住。也就是说，问题不是“算太多”，而是“搬太多”。

第二个坑是忽略非 matmul 操作。新手容易把 Transformer 理解成“基本都是矩阵乘法”。这句话方向没错，但在性能分析里不够精确。softmax、mask、残差连接、LayerNorm 这些操作的 FLOPs 占比可能不大，但字节搬运占比并不低。Roofline 里真正限制性能的是 FLOPs 和 bytes 的比值，因此“FLOPs 少”不等于“成本小”。

第三个坑是把 HFU 高误读成训练效率高。比如你打开 checkpointing 后，HFU 可能上升，因为 GPU 一直在重新计算激活；但从训练成本角度看，单位样本需要的理论有效计算并没有下降。若只看 HFU，会误以为优化成功。若同时看 MFU 和 step time，问题才清楚。

还有一个常见误判：把所有场景都按训练看。实际上，推理里的 prefill 和 decode 在 Roofline 上位置差很多。prefill 序列长、并行度高，更可能接近 compute-bound；decode 尤其是 batch=1 时，算术强度很低，经常稳定落在 memory-bound 区域。两种场景的优化抓手完全不同。

---

## 替代方案与适用边界

MFU、HFU 不是唯一指标，也不应该被当成唯一指标。

如果你要比较不同训练配置的端到端成本，MFU 很有价值，因为它把“理论模型工作量”和“硬件峰值能力”直接联系起来，适合估算扩展效率、并行策略效果和训练预算。

如果你要判断 GPU 是否空转，HFU 更直接。尤其在重算较多、流水线气泡明显、通信等待复杂的场景，HFU 能告诉你硬件繁忙程度。

如果你要决定“下一步是改 kernel，还是改 batch，还是改并行策略”，Roofline 更有效，因为它能区分瓶颈属于算力还是带宽。

把三者放到推理场景，边界更清楚。

| 场景 | 典型算术强度 | Roofline 位置 | 常见优化 |
|---|---|---|---|
| Decode，batch=1 | 很低，常接近 1 FLOP/byte | 脊点左侧，强 memory-bound | 量化、增大 batch、改进 KV cache 访存 |
| Prefill，长序列 | 随序列增长而上升 | 可能接近或越过脊点 | 提升 matmul 利用率、用更高效 attention kernel |
| FFN 训练 | 通常较高 | 更接近平顶 | Tensor Core 利用、融合 kernel、并行调度 |
| Naive attention | 中低强度 | 常在斜坡区 | 降 IO、减少 materialization |

decode 与 prefill 的区别可以用一句话概括：decode 每次只生成一个 token，计算量小但每步都要读大量历史 KV，算术强度低；prefill 一次处理整段上下文，能把更多计算摊在同一批数据搬运上，因此更有机会跨过脊点。

所以，适用边界可以总结为：

1. 想比较“模型训练效率”，优先看 MFU。
2. 想比较“硬件是否忙”，补看 HFU。
3. 想定位“为什么忙但仍然慢”，用 Roofline 分析算子级强度。
4. 想优化 Transformer，不要默认 attention 一定是算力瓶颈，先确认它是不是 IO 瓶颈。
5. 想解释 FlashAttention 的收益，不要只说“更快”，要说清它本质上是把 attention 从低算术强度区域往脊点右侧推。

---

## 参考资料

- Model FLOPs Utilization 定义与用途（YMSHICI 技术专栏）：https://www.ymshici.com/tech/2345.html
- MFU vs HFU、重算影响与实测公式（ML Engineering 文章）：https://saforem2.github.io/ml-engineering/qmd/training/performance/index.html
- Roofline 原理、脊点与算术强度推导（GPU Fundamentals & LLM Mental Models）：https://shekkari1999.github.io/blog/gpu-fundamentals.html
- FlashAttention IO 分析与 Roofline 位移案例（Hugging Face 文章）：https://huggingface.co/blog/atharv6f/flash-attention-io-analysis
- Transformer attention 内存瓶颈与 softmax/LayerNorm 带宽成本分析（OA Quantum Labs）：https://oaqlabs.com/2025/10/12/kernel-level-gpu-optimization-for-transformer-attention-a-technical-deep-dive/
