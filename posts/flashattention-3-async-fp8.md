## 核心结论

FlashAttention-3不是“把 Attention 再写快一点”，而是把 Hopper GPU 上两个原本容易互相等待的硬件单元真正并行起来：`TMA` 负责搬数据，白话就是“专门的高速搬运工”；`wgmma` 负责 Tensor Core 矩阵乘，白话就是“专门的高吞吐计算工”。两者通过异步流水线重叠后，Attention 不再是“先搬完再计算”的串行流程，而变成“边搬边算”的流水线流程。

在 H100 上，这带来两个直接结果：

| 路径 | 代表吞吐 | 对 H100 理论峰值利用率 | 相对 FlashAttention-2 |
| --- | ---: | ---: | ---: |
| FlashAttention-2 / H100 / FP16 | 约受限于 35% 利用率 | 约 35% | 基线 |
| FlashAttention-3 / H100 / FP16 | 最高约 740 TFLOPs/s | 约 75% | 1.5-2.0x |
| FlashAttention-3 / H100 / FP8 | 接近 1.2 PFLOPs/s | 接近 FP8 峰值的大比例 | 明显高于 FP16 |

核心瓶颈可以先记成一个近似式：

$$
\text{effective throughput} \approx \min(\text{wgmma\_rate}, \text{TMA\_rate})
$$

意思很直接：计算再快，搬运跟不上也白搭；搬运再快，Tensor Core 吃不满也白搭。FlashAttention-3的设计重点，就是把这两个速率尽量拉平，并把 softmax 插到流水线空隙里执行。

---

## 问题定义与边界

标准缩放点积注意力是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

这里的 `softmax` 是“把一组分数变成概率分布”的函数。问题不在公式本身，而在硬件执行路径：

1. `QK^T` 和 `PV` 是大矩阵乘，适合 Tensor Core。
2. `softmax` 需要 `max`、`exp`、归一化，不适合 Tensor Core。
3. 如果每一步都等上一步完全结束，H100 的计算单元和访存单元会互相空转。

玩具例子可以这样理解。假设有两队人：

| 角色 | 旧流程 | 新流程 |
| --- | --- | --- |
| 搬运队 | 先把原料全部搬完 | 一边搬一边交下一块 |
| 计算队 | 等原料全到齐再开工 | 拿到一块就开始算 |

旧流程的总时间近似是“搬运时间 + 计算时间”，新流程更接近“二者最大值”。这就是异步流水线的价值。

边界也很明确：

| 条件 | FlashAttention-3收益 |
| --- | --- |
| Hopper H100/H800，长上下文，较大 batch | 高 |
| 非 Hopper 平台 | 关键硬件能力缺失，收益有限 |
| 很短序列，如 512 token | 流水线很难铺满，收益可能不显著 |
| 极度敏感的数值任务 | 仍需优先验证 FP16/BF16 |

---

## 核心机制与推导

FlashAttention-3的核心不是单一技巧，而是三个机制同时成立。

第一，`warp specialization`。`warp` 是 GPU 上一小组并行线程；`warpgroup` 是 4 个 warp 组成的更大执行单元。FlashAttention-3让不同 warpgroup 分工：

- producer warpgroup 负责 `TMA` 预取，把下一块 Q/K/V 从 HBM 拉到 shared memory
- consumer warpgroup 负责 `wgmma`，消费 shared memory 里的 tile 做矩阵乘

第二，`ping-pong scheduling`。也就是双缓冲。一个缓冲区正在被计算时，另一个缓冲区同时被填充。这样就把“等数据”和“等计算”的时间叠起来。

第三，`interleave softmax`。softmax 不再被当成两个 GEMM 中间的独立大阶段，而是按 block 拆开，插入流水线的空档执行。因为 Hopper 上 special function 单元和 Tensor Core 是不同资源，所以可以并行消化。

一个简化状态机可以写成：

```text
stage 0: producer -> load tile A
stage 1: consumer -> compute QK^T on tile A
stage 2: producer -> load tile B
stage 3: consumer -> softmax(tile A) + compute on tile B
stage 4: consumer -> accumulate P*V for tile A
repeat
```

所以总吞吐近似不再是：

$$
T_{\text{old}} \approx T_{\text{load}} + T_{\text{gemm}} + T_{\text{softmax}}
$$

而更接近：

$$
T_{\text{new}} \approx \max(T_{\text{load}}, T_{\text{gemm}}, T_{\text{softmax-overlapped}})
$$

低精度部分也不是“直接把 FP16 改成 FP8”这么简单。`FP8` 是 8 位浮点数，白话就是“指数和尾数都更短，所以更省带宽、吞吐更高，但更容易失真”。Hopper 支持两种常见 FP8 格式：

| 格式 | 特点 | 常见用途 |
| --- | --- | --- |
| E4M3 | 精度稍好，动态范围较小 | 更适合多数前向激活/权重 |
| E5M2 | 动态范围更大，精度更粗 | 更适合更大幅值数据 |

工程里常说“E4M3/E5M2 混合”，本质是按张量特征选格式，不让 `QK^T` 因个别大值直接爆掉。FlashAttention-3再配合两件事降低误差：

1. 块级量化：不是整张量只用一个 scale，而是每个 block 单独 scale。
2. incoherent processing：对 Q/K 做 Hadamard + random sign 之类的正交混合，把离群值打散。

因为正交变换满足 $MM^\top = I$，理论上不会改变注意力结果，只改变数值分布，使量化更稳定。论文和官方博客给出的结果是：FP8 路径误差比基线 FP8 attention 小约 2.6 倍。

---

## 代码实现

下面先给一个可运行的玩具例子，不是 CUDA 内核，而是用 Python 模拟“块级 softmax + 双缓冲思路”，验证分块 Attention 和一次性计算结果一致。`assert` 用来做结果校验，白话就是“程序自己检查自己没算错”。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def attention_naive(q, ks, vs):
    scores = [sum(q[i] * k[i] for i in range(len(q))) / math.sqrt(len(q)) for k in ks]
    probs = softmax(scores)
    out = [0.0] * len(vs[0])
    for p, v in zip(probs, vs):
        for i in range(len(out)):
            out[i] += p * v[i]
    return out

def attention_blockwise(q, ks, vs, block_size=2):
    d = len(q)
    running_max = float("-inf")
    running_sum = 0.0
    running_out = [0.0] * len(vs[0])

    for start in range(0, len(ks), block_size):
        block_k = ks[start:start + block_size]
        block_v = vs[start:start + block_size]
        scores = [sum(q[i] * k[i] for i in range(d)) / math.sqrt(d) for k in block_k]

        block_max = max(scores)
        new_max = max(running_max, block_max)

        old_scale = 0.0 if running_max == float("-inf") else math.exp(running_max - new_max)
        block_exps = [math.exp(s - new_max) for s in scores]

        running_out = [x * old_scale for x in running_out]
        running_sum *= old_scale

        for w, v in zip(block_exps, block_v):
            for i in range(len(running_out)):
                running_out[i] += w * v[i]
            running_sum += w

        running_max = new_max

    return [x / running_sum for x in running_out]

q = [1.0, 0.5]
ks = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]]
vs = [[1.0, 2.0], [0.0, 1.0], [3.0, 1.0], [2.0, 0.0]]

a = attention_naive(q, ks, vs)
b = attention_blockwise(q, ks, vs, block_size=2)

for x, y in zip(a, b):
    assert abs(x - y) < 1e-9, (a, b)
print("ok")
```

真实工程里，CUDA 伪代码更接近下面这样：

```python
# pseudo code
for tile_id in range(num_tiles):
    if warpgroup_id < producer_groups:
        TMA_load(smem_buffer[next_buffer], gmem_qkv[tile_id + prefetch_distance])
        barrier_arrive(next_buffer)
    else:
        barrier_wait(curr_buffer)
        scores = wgmma_qk(smem_buffer[curr_buffer])
        stats = online_softmax_update(scores, stats)
        out = wgmma_pv(stats, smem_buffer[curr_buffer])
        if done(curr_buffer):
            store(out)
    curr_buffer, next_buffer = next_buffer, curr_buffer
```

一个常见 kernel 配置可以概括成：

| 参数 | 作用 | 常见取值方向 |
| --- | --- | --- |
| warpgroup size | 一组协同执行线程 | 4 warps |
| ping-pong buffers | 双缓冲 shared memory | 2 |
| tile size | 每次搬运/计算块大小 | 依 head dim、寄存器压力调 |
| accumulator dtype | 累加精度 | softmax 统计通常保留 FP32 |
| input dtype | 输入精度 | FP16/BF16 或 FP8 |

要点是：输入可以低精度，但 softmax 的统计量和关键累加通常不能盲目全降到 FP8，否则误差会迅速放大。

---

## 工程权衡与常见坑

第一类坑是“只看 FLOPs，不看流水线”。如果没有 producer/consumer 分工，或者 barrier 放错位置，TMA 和 wgmma 仍然会互相等，最后性能退化成“看上去用了 Hopper 新指令，实测却没提多少”。

第二类坑是“只做 FP8 存储，不做稳定量化”。如果整块张量共用一个全局 scale，`QK^T` 很容易被离群值主导，softmax 前的分数尺度失真，最终变成概率塌缩。经验上要优先做块级 scale，并保留在线 softmax 的高精度统计。

第三类坑是寄存器压力。异步重叠并不是零成本。你同时保留更多 tile、更多中间统计量，就会吃掉更多寄存器和 shared memory，occupancy 可能下降。工程上不是“重叠越多越好”，而是找到寄存器、shared memory、吞吐三者的平衡点。

| 问题 | 现象 | 处理方式 |
| --- | --- | --- |
| barrier 过多 | 算子正确但速度上不去 | 缩短同步路径，只同步必要阶段 |
| barrier 过少 | 读到未完成数据，结果错乱 | 对 TMA 完成和 buffer 切换做显式保护 |
| 全局 FP8 scale | 误差明显上升，困惑度恶化 | 块级量化 + amax/scale 管理 |
| 全部中间值都降精度 | softmax 不稳定 | 关键统计保留 FP32 |
| 小 batch、短序列强上 FA-3 | 调优复杂但收益小 | 先测 FA-2 或 cuDNN |

真实工程例子：如果你在 H100 上做 4K 甚至更长上下文的 LLM 推理，prefill 阶段 Attention 往往是大头。此时 FlashAttention-3 的意义不是“单个 kernel 漂亮”，而是同等 GPU 数量下提升 tokens/s，或者在同样吞吐目标下减少 GPU 占用。反过来，如果只是 512 token、小 batch 的在线请求，流水线很难铺满，部署复杂度可能比收益更高。

---

## 替代方案与适用边界

FlashAttention-3不是所有场景的默认最优解。

| 方案 | 适用硬件 | 优点 | 边界 |
| --- | --- | --- | --- |
| FlashAttention-3 FP8 | H100/Hopper | 吞吐最高 | 对硬件、编译链、量化稳定性要求高 |
| FlashAttention-3 FP16/BF16 | H100/Hopper | 性能高且更稳 | 吞吐低于 FP8 |
| FlashAttention-2 | A100、非 Hopper | 生态成熟，移植简单 | 吃不到 TMA/wgmma 红利 |
| cuDNN SDPA | 多平台 | 集成方便 | 对特定长上下文场景未必最优 |
| 常规 Attention | 任意平台 | 实现最简单 | 长序列下性能和显存都差 |

可以用一个简单指标估算收益：

$$
\text{speedup} = \frac{\text{throughput}_{\text{FA3, FP8}}}{\text{throughput}_{\text{baseline, FP16}}}
$$

如果你的任务满足下面三条中的两条以上，FlashAttention-3通常值得上：

1. Hopper 平台。
2. 序列长度至少到 2K-4K。
3. 推理或训练足够大，能把流水线铺满。

如果不满足，优先级通常是：先用 FlashAttention-2 或 cuDNN 跑稳定，再决定是否为 Hopper 单独维护 FA-3 路径。

---

## 参考资料

- [PyTorch 官方博客：FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://pytorch.org/blog/flashattention-3/)
- [NVIDIA 官方博客：Next Generation of FlashAttention](https://developer.nvidia.com/blog/next-generation-of-flashattention/)
- [FlashAttention-3 论文摘要页（arXiv 2407.08608）](https://arxiv.org/abs/2407.08608)
- [NVIDIA cuDNN Frontend 文档：FP8 Attention 与 E4M3/E5M2](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html)
- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core/nvidia-h100-tensor-core-gpu-architecture)
