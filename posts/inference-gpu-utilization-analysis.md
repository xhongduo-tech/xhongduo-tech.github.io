## 核心结论

推理场景里的 GPU 利用率，不能只看一个“百分比”。更准确的拆法有两层：第一层看 kernel 之间有没有空洞，也就是 GPU 有没有在等活；第二层看 kernel 内部到底是谁在限制吞吐，是 SM、Tensor Core，还是显存带宽。

这里的 **kernel** 可以理解为“一次被 GPU 执行的小计算任务”。**SM** 是 GPU 的流式多处理器，可以把它理解为“真正执行大量线程的核心计算单元”。**Tensor Core** 是专门做矩阵乘加的硬件单元，主要用于深度学习里的高吞吐矩阵计算。**HBM 带宽** 指显存读写速度上限，白话讲就是“数据从显存进出 GPU 计算单元有多快”。

在线推理里，`nvidia-smi` 常见的高利用率，很多时候只是“设备在一段时间里经常被调度到”，不等于“算力被高效吃满”。尤其在 batch 很小、前处理在 CPU、请求到达不均匀的服务里，GPU 很可能处于“忙着等待”“忙着搬数据”或者“忙着启动很多短小 kernel”的状态。

下面这组定义是文章的主线：

$$
U_{SM}=\frac{T_{sm}}{T}, \quad
U_{TC}=\frac{T_{tc}}{T}, \quad
G=\frac{T_{gap}}{T}, \quad
U_{HBM}=\frac{((B_r+B_w)/T)}{BW_{peak}}
$$

它们分别表示：SM 活跃占比、Tensor Core 活跃占比、空洞占比、显存带宽利用占比。

最常见的误判是：看到 GPU 80% 占用，就认为模型已经吃满算力。对在线推理来说，这个结论往往不成立。更常见的真实情况是：CPU 送数慢、batch 太小、kernel 太碎、访存不顺，最后表现为“GPU 看起来很忙，但有效吞吐不高”。

| 指标 | 它真正表示什么 | 常见误读 |
|---|---|---|
| `nvidia-smi` 利用率 | 设备在采样周期内有工作发生的时间占比 | 误当成算力利用率 |
| SM 利用率 | SM 在观测窗口内活跃的时间占比 | 误当成模型整体效率 |
| Tensor Core 利用率 | Tensor Core 真正在工作的时间占比 | 误以为低就一定有问题 |
| HBM 带宽利用率 | 显存带宽被用掉的比例 | 忽略了带宽满但算力没满 |
| occupancy | 驻留线程束比例 | 误当成吞吐或性能结论 |

一个新手版直观例子：如果某个服务里 GPU 显示 80% 占用，但请求要先在 CPU 做分词、拼 batch、准备输入，再从主存拷到显存，那么这 80% 很可能混入了大量等待和搬运，并不说明 GPU 的数学计算被充分利用。

一个真实工程例子：LLM 在线推理里，`batch=1~4` 时，经常会出现大量短 GEMM 和 attention kernel。时间线上可以看到 kernel 之间有明显空白，平均 GPU 利用率却不低。此时单看一个平均值，很容易误判系统“已经很健康”。

---

## 问题定义与边界

这篇文章讨论的是 **推理场景** 的 GPU 利用率分析，不讨论训练，也不讨论大规模离线批处理。推理的约束通常是在线延迟、请求抖动、batch 小、CPU/GPU 协同复杂，所以它和训练的性能图景不一样。

我们先定义一组术语。**观测窗口** 就是你拿来统计的一段时间。**空洞时间** 是 GPU 时间线上没有有效 kernel 在跑的部分。**算术强度** 指每读写 1 字节数据，能做多少浮点运算，白话讲就是“这个算子更像在拼命算，还是更像在拼命搬数据”。

| 术语 | 符号 | 定义 | 白话解释 |
|---|---|---|---|
| 观测窗口 | $T$ | 一段统计时间 | 你拿来做分析的时间尺子 |
| 空洞时间 | $T_{gap}$ | kernel 之间或前后等待时间 | GPU 没真干活的空白 |
| SM 活跃时间 | $T_{sm}$ | SM 有效执行的时间 | 通用计算单元在工作 |
| Tensor Core 活跃时间 | $T_{tc}$ | Tensor Core 执行时间 | 矩阵硬件在工作 |
| occupancy | $O$ | active warps / max warps | 能驻留多少线程束 |
| 算术强度 | $I$ | $F/(B_r+B_w)$ | 算得多还是搬得多 |

算术强度公式是：

$$
I=\frac{F}{B_r+B_w}
$$

其中 $F$ 是浮点运算量，$B_r+B_w$ 是总读写字节数。

为什么它重要？因为 roofline 模型里有一个 **ridge point**，可以理解为“从带宽受限切到算力受限的分界线”。当 $I$ 低于 ridge point，算子通常更偏 **memory-bound**，也就是“受内存带宽限制”；当 $I$ 高于 ridge point，才更可能是 **compute-bound**，也就是“受计算单元限制”。

边界也要说清楚：

| 情况 | 是否能只靠 GPU 利用率解释 |
|---|---|
| 模型主要耗时在 GPU kernel | 可以，且收益高 |
| 大量前后处理在 CPU | 不行，必须看端到端时间线 |
| 算子不走 Tensor Core 路径 | 不能用 Tensor Core 低活跃直接下结论 |
| 服务瓶颈在网络、存储、后处理 | GPU 利用率不是主解释变量 |

一个边界例子：同样是 GPU 90%“忙”，训练可能是一个长时间的大矩阵乘法把卡持续打满；推理则可能是请求断断续续、CPU 排队慢、GPU 频繁跑小 kernel。两个 90% 的含义完全不同。

---

## 核心机制与推导

推理 GPU 利用率的分析顺序，应该先按 **时间线** 拆，再按 **资源** 拆。

第一步看时间线。把观测窗口 $T$ 切成三类时间：kernel 之间的空洞、kernel 内部通用计算活跃时间、kernel 内部特定单元活跃时间。最直观的指标是：

$$
G=\frac{T_{gap}}{T}
$$

如果 $G$ 很高，优先怀疑“GPU 没有持续吃到活”，而不是怀疑“GPU 算力不够”。

第二步看 kernel 内部资源。常用指标是：

$$
U_{SM}=\frac{T_{sm}}{T}
$$

$$
U_{TC}=\frac{T_{tc}}{T}
$$

$$
B_{eff}=\frac{B_r+B_w}{T}
$$

$$
U_{HBM}=\frac{B_{eff}}{BW_{peak}}
$$

这四个量放在一起，才能判断问题在哪。比如：

- $G$ 高，说明先优化调度、拷贝、batch、同步点。
- $G$ 不高但 $U_{HBM}$ 高，说明更像带宽瓶颈。
- $G$ 不高、$U_{HBM}$ 不高、$U_{TC}$ 也不高，常见于 kernel 太小、形状不友好、启动开销偏大。
- occupancy 高但 $U_{HBM}$ 已经接近上限，说明它虽然“驻留得多”，但本质仍是 memory-bound。

这里要特别强调 **occupancy**。它的定义是：

$$
O=\frac{\text{active warps}}{\text{max warps}}
$$

它只表示“最多能同时挂多少线程束在线”，不表示这些线程束都在高效做计算。高 occupancy 也可能只是大量线程在等数据回来，所以它不能直接当作吞吐结论。

一个玩具例子：把一个请求看成 10 ms 的路程。假设 2 ms 在等 CPU 预处理结束，4 ms 在跑普通 SM 计算，1 ms 在跑 Tensor Core，剩下 3 ms 用于数据搬运和其他杂项。这个请求并不是“10 ms 全在算”，而是混合了等待、访存和计算。

再看数值版推导。设：

- $T=10$ ms
- $T_{sm}=4$ ms
- $T_{tc}=1$ ms
- $T_{gap}=2$ ms
- $B_r+B_w=3$ GB
- $BW_{peak}=1$ TB/s

则：

$$
U_{SM}=4/10=40\%
$$

$$
U_{TC}=1/10=10\%
$$

$$
G=2/10=20\%
$$

$$
U_{HBM}=\frac{3\text{GB}/0.01\text{s}}{1000\text{GB/s}}=30\%
$$

结论很直接：这张卡既没有被算力打满，也没有被带宽打满，同时还存在 20% 的时间空洞。此时继续纠结“换更大 GPU”通常不是第一优先级。

真实工程例子更典型。在线推荐或 LLM 服务里，`batch=1~4` 时主干往往是很多短 GEMM 和小 attention kernel。Nsight Systems 常能看到 CPU 侧请求准备与 GPU kernel 之间脱节；Nsight Compute 再往下看，会发现 occupancy 不低，但 SM、Tensor Core、DRAM 利用都没到高位。这说明瓶颈更可能在“小而碎”的执行形态，而不是单纯算力不足。

| 指标组合 | 更可能的瓶颈 | 优先动作 |
|---|---|---|
| $G$ 高，其他都不高 | 时间线空洞 | 查 CPU 送数、同步点、batch |
| $G$ 低，$U_{HBM}$ 高 | memory-bound | 优化访存、融合、减少重复读写 |
| $G$ 低，$U_{TC}$ 低，kernel 很碎 | 启动开销/形状不友好 | 融合算子、增大 batch |
| occupancy 高但吞吐低 | 驻留高但等待多 | 检查带宽、访存模式、依赖链 |

---

## 代码实现

真正可复用的分析流程，不是“跑一次测速脚本”，而是建立固定的两段式 profiling：先找空洞，再解释热点 kernel。

第一段用 Nsight Systems，看请求到 CPU、CPU 到 GPU、GPU 内 kernel 排布之间是否连续。第二段用 Nsight Compute，看热点 kernel 的 SM、Tensor Core、HBM、occupancy 指标。

工具分工可以这样记：

| 工具 | 主要看什么 | 典型问题 |
|---|---|---|
| Nsight Systems | 时间线、空洞、CPU/GPU 交接 | GPU 为什么没持续吃到活 |
| Nsight Compute | 单个 kernel 的资源利用 | kernel 为什么没跑满 |

命令通常像这样：

```bash
nsys profile -t cuda,nvtx -o trace ./serve
ncu --set full --target-processes all ./serve
```

如果服务里有请求边界，建议打 NVTX 标记。**NVTX** 可以理解为“给代码时间线贴标签”，方便把请求、预处理、拷贝、推理阶段对齐到 GPU 活动上。

```python
from dataclasses import dataclass

@dataclass
class GpuWindow:
    T_ms: float
    T_sm_ms: float
    T_tc_ms: float
    T_gap_ms: float
    bytes_rw: float
    bw_peak_bytes_s: float

def analyze(window: GpuWindow):
    T_s = window.T_ms / 1000.0
    u_sm = window.T_sm_ms / window.T_ms
    u_tc = window.T_tc_ms / window.T_ms
    g = window.T_gap_ms / window.T_ms
    b_eff = window.bytes_rw / T_s
    u_hbm = b_eff / window.bw_peak_bytes_s
    return {
        "U_SM": u_sm,
        "U_TC": u_tc,
        "G": g,
        "B_eff": b_eff,
        "U_HBM": u_hbm,
    }

toy = GpuWindow(
    T_ms=10.0,
    T_sm_ms=4.0,
    T_tc_ms=1.0,
    T_gap_ms=2.0,
    bytes_rw=3e9,
    bw_peak_bytes_s=1e12,
)

m = analyze(toy)
assert round(m["U_SM"], 2) == 0.40
assert round(m["U_TC"], 2) == 0.10
assert round(m["G"], 2) == 0.20
assert round(m["U_HBM"], 2) == 0.30
print(m)
```

这段代码不是 profiler，本质上是把你从 trace 和 kernel 报告里读到的数据，统一映射成可比较指标，避免团队里每个人都凭感觉解释“GPU 很忙”。

真实工程例子可以按下面流程落地在 Triton 或自研服务上：

1. 用 `nsys` 抓一段真实流量，先确认 CPU 预处理、请求合批、H2D 拷贝、kernel 执行之间是否有明显空洞。
2. 找到耗时最高或调用最频繁的 kernel。
3. 用 `ncu` 查看这些 kernel 的 SM、Tensor Core、DRAM、occupancy。
4. 回到代码验证是否能通过动态 batching、Pinned memory、异步拷贝或算子融合减少空洞。

---

## 工程权衡与常见坑

最常见的坑是把单一指标当结论。尤其是把 `nvidia-smi` 当性能指标，把 occupancy 当优化结果。

| 常见坑 | 为什么错 | 规避方式 |
|---|---|---|
| `nvidia-smi` 高就认为高效 | 它更接近时间占用，不是算力吃满 | 必须结合时间线和 kernel 指标 |
| occupancy 高就认为性能高 | 驻留高不代表执行单元高效工作 | 同时看带宽和 SM/Tensor Core |
| 只看服务平均值 | 平均值会抹掉长尾空洞 | 按请求、按阶段、按 kernel 看 |
| 忽略 pageable memory | 可能触发隐式同步 | 用 pinned memory 和异步拷贝 |
| 只盯 GPU 不看 CPU | CPU 可能是送数瓶颈 | 一起看 CPU 线程和 CUDA 时间线 |

再给一个现象排查表：

| 现象 | 可能原因 | 优先排查项 |
|---|---|---|
| GPU 利用率不低，但吞吐上不去 | 小 batch、kernel 太碎 | 是否能做动态 batching |
| GPU 常有空白段 | CPU 预处理慢、同步点多 | 时间线上的 H2D、tokenize、postprocess |
| occupancy 不低但速度仍慢 | memory-bound | DRAM 吞吐、访存模式、数据复用 |
| kernel 很短且很多 | 启动开销显著 | 融合算子、减少 launch 次数 |

新手版例子：两个服务都显示 GPU 70% 占用。A 服务是在稳定跑大 GEMM，B 服务是在 CPU 预处理和隐式同步之间不断让 GPU 短暂工作。数值一样，但 A 的有效吞吐通常更高。

真实工程例子：如果输入来自 pageable memory，也就是普通可分页内存，H2D 拷贝可能引入额外 staging 和同步。结果是 kernel 本身很快，但 GPU 前后总有空洞，吞吐下降，用户却误以为“模型算子太慢”。这类问题常常优先于“换更快的卡”。

---

## 替代方案与适用边界

如果目标是提升端到端吞吐，未必先优化单个 GPU kernel。在线推理更常见的高收益动作，是让 GPU 更连续地拿到工作，而不是盲目追求某个 kernel 的极限 FLOPS。

| 方案 | 适用场景 | 代价 |
|---|---|---|
| 动态 batching | 请求密集、吞吐优先 | 可能增加延迟 |
| Pinned memory + 异步拷贝 | H2D 拷贝明显 | 实现复杂度上升 |
| 算子融合 | kernel 太碎、launch 太多 | 开发和验证成本高 |
| Tensor Core 友好精度路径 | 矩阵算子多、硬件支持好 | 可能影响数值精度 |
| 直接换更大 GPU | 工程时间紧 | 成本最高，且可能掩盖系统问题 |

判断顺序可以用一个简单准则：

- 如果 $G$ 高，先解决空洞。
- 如果 $G$ 不高但 $U_{HBM}$ 高，先按 memory-bound 思路优化。
- 如果 $I$ 高于 ridge point，才更值得怀疑 compute-bound。
- 如果 batch 很小且延迟敏感，不要机械追求更大 batch，而要优先减少拷贝和同步。

新手版例子：每次只来 1 个请求时，强行增大 batch 可能伤害延迟目标。这时更现实的路线是减少 CPU 阻塞、减少 H2D 等待、减少零碎 kernel。

真实工程例子：对小 attention kernel 很多的在线服务，算子融合和调度优化往往比单纯升级 GPU 更有效。相反，如果模型本身已经是大矩阵乘法主导、且 Tensor Core 路径正确、$G$ 很低、$U_{HBM}$ 也不高，这时换更强 GPU 才更有确定性。

---

## 参考资料

| 资料名 | 用途 | 对应章节 |
|---|---|---|
| Nsight Systems Post-Collection Analysis Guide | 看时间线与空洞 | 核心机制、代码实现 |
| Nsight Compute Profiling Guide | 看 kernel 内部指标 | 核心机制、工程权衡 |
| CUDA C++ Best Practices Guide | 理解访存、拷贝、异步 | 工程权衡、替代方案 |
| NVIDIA Nsight Deep Learning Designer User Guide | 深度学习推理分析辅助 | 代码实现 |
| Triton Inference Server Batchers | 理解服务侧 batching | 替代方案 |

1. [Nsight Systems Post-Collection Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
2. [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
3. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
4. [NVIDIA Nsight Deep Learning Designer User Guide](https://docs.nvidia.com/nsight-dl-designer/2024.1/UserGuide/index.html)
5. [Triton Inference Server Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
