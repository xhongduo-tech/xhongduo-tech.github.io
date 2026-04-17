## 核心结论

Nsight Systems、Nsight Compute 与 `torch.profiler` 不是三个互相替代的工具，而是三层不同粒度的观测面。

第一层是 Nsight Systems。它解决“时间到底花在哪里”的问题。时间线是把 CPU 调用、CUDA API、内存拷贝和 GPU kernel 执行放到同一张图里看。新手最容易从这里建立直觉：如果上方 CPU 线程很忙、下方 GPU 轨道却有空白，通常说明 launch、同步或数据准备没有接上；如果 memcpy 与 kernel 没有重叠，通常说明流水线没有铺开。

第二层是 Nsight Compute。它解决“这个 kernel 为什么慢”的问题。它不会先告诉你是不是 Python 写得差，而是直接把单个 kernel 放到 roofline 模型里，看它更接近带宽上限还是算力上限。roofline 可以先做一个粗分类：memory bound，白话说是“主要卡在搬数据”；compute bound，白话说是“主要卡在算得不够快”。

第三层是 `torch.profiler`。它解决“如何把 Python 代码段、算子和 trace 文件连起来”的问题。它最适合 PyTorch 工程里的日常迭代：先用 `schedule` 控制采样窗口，再用 `trace_handler` 导出 Chrome Trace 或 TensorBoard Trace，让一次训练 step 的 Python 调用、CUDA 活动和算子耗时能对上号。

这三者组合起来，才能从“怀疑哪里慢”走到“确认是哪一层慢、为什么慢、该怎么改”。

---

## 问题定义与边界

本文讨论的是 GPU 程序的性能分析，不是模型精度，也不是算法效果。目标很具体：定位训练或推理过程中，CPU 与 GPU 是否不同步、数据传输是否与计算重叠、单个 kernel 是带宽瓶颈还是算力瓶颈。

这三个工具的边界不同：

| 工具 | 关注颗粒 | 输出格式 | 典型用途 |
| --- | --- | --- | --- |
| Nsight Systems | 进程、线程、CUDA API、memcpy、kernel 时间线 | 时间线报告 | 看整体流水线、同步、launch 延迟、传输重叠 |
| Nsight Compute | 单个 kernel 的硬件指标 | roofline、metric 报告 | 判断 memory bound / compute bound，分析 kernel 微观瓶颈 |
| `torch.profiler` | Python 函数、PyTorch 算子、CUDA 活动 | Chrome Trace / TensorBoard Trace | 把 Python 调用栈与 GPU 活动关联起来 |

边界要先说清楚，否则容易拿错工具。

如果你在训练 ResNet，发现一轮 step 比预期慢，这时先看 Nsight Systems，因为你先要判断是“整体流程没接起来”，还是“某个 kernel 本身差”。假设时间线上看到 CUDA launch 的紫色条、同步事件的绿色条紧贴在 kernel 前后，而 kernel 之间有明显空白，同时 H2D 拷贝也没有覆盖到计算阶段，那就可以先定义问题为：CPU launch 与数据传输没有和 GPU 计算形成有效重叠。

如果你已经知道大部分时间耗在某个卷积或 attention kernel 上，这时再切到 Nsight Compute，因为问题已经从“流程编排”缩小成“单个 kernel 受什么硬件上限约束”。

如果你只想知道“哪段 Python 代码触发了这些 GPU 活动”，并且项目本身就是 PyTorch 训练脚本，那先上 `torch.profiler` 更省成本。

---

## 核心机制与推导

性能分析要有一个统一的判断框架。roofline 模型就是这个框架。

它的横轴是 Arithmetic Intensity，简称 AI，白话说是“每搬 1 字节数据，做了多少浮点运算”。定义是：

$$
AI=\frac{\text{Work (FLOP)}}{\text{Memory Traffic (Bytes)}}
$$

纵轴是实际达到的浮点性能，也就是 FLOP/s。对于一块 GPU，理论上会有两条上限：

1. 带宽上限形成一条斜线，因为在固定带宽下，AI 越高，单位时间可支撑的 FLOP/s 越高。
2. 峰值算力形成一条平线，因为算力再强也不可能无限增长。

一个 kernel 在图上的位置，决定了它更像哪种瓶颈。

玩具例子：某个 kernel 做了 $1\times10^9$ 次 FLOP，访问了 $0.5$ GiB 数据，那么

$$
AI\approx\frac{1\times10^9}{0.5\times2^{30}}\approx1.86\ \text{FLOP/byte}
$$

如果该点落在 roofline 斜面附近，说明它主要受内存带宽限制；如果已经接近平顶，则主要受算力限制。这个判断很重要，因为优化方向完全不同。memory bound 的首选动作是提高数据复用、减少重复访存、减少不必要拷贝；compute bound 的首选动作才是考虑指令效率、并行度、张量核心利用率。

Nsight Systems 的机制则是时间因果关系。你可以把它理解成一张“谁触发了谁”的流程图，只不过图是按时间展开的。CPU 线程发起 CUDA API 调用，经过 runtime/driver，把异步工作提交给 GPU。NVIDIA 文档里常用 ac2g flow 来描述这类 CPU 到 GPU 的关联，意思是你不只看到“发生了一个 kernel”，还能看到“它是被哪次 CPU 调用发起的”。

新手在时间线里可以先只看三类对象：

1. CPU 线程上的 CUDA API 调用。
2. 内存拷贝轨道上的 H2D/D2H。
3. GPU 轨道上的 kernel 执行块。

一个简化的文本示意如下：

```text
CPU Thread:  [launch k1]   [launch k2]      [sync]
Memcpy(H2D):      [copy batch1] [copy batch2]
GPU Kernel:          [k1 running]   [k2 running]
```

如果 `launch k2` 明显晚于 `k1 running` 结束，GPU 中间会出现空白，这往往不是 GPU 算得慢，而是 CPU 没及时把下一份工作送上去。  
如果 `copy batch2` 总是在 `k1` 完成后才开始，也说明传输和计算没有重叠。

真实工程例子：训练 ResNet 时，DataLoader、host 到 device 拷贝、前向卷积、反向传播和优化器更新本应形成稳定流水线。如果 Nsight Systems 里看到每个 step 的前半段 CPU 很忙，后半段 GPU 很忙，但两者很少同时忙，那么问题就不是某个单点，而是整个生产者-消费者节奏失配。常见原因包括小 batch 导致 launch 过密、`torch.cuda.synchronize()` 被误用、DataLoader worker 不足、Pinned Memory 没开导致 H2D 拷贝不顺畅。

`torch.profiler` 的价值在于补上“Python 代码位置”这一层。它的 `trace_handler` 会按设定窗口导出 trace 文件，然后可以用 Chrome Trace 或 TensorBoard Trace Viewer 打开。这样一来，某个 Python 训练 step、某个 ATen 算子、某次 CUDA kernel 执行就能串起来看，不需要一开始就进入更重的 Nsight 工具链。

---

## 代码实现

下面先给一个最小可运行的 Python 例子，用来计算 roofline 所需的 AI，并用 `assert` 固定预期。这个例子不依赖 GPU，目的是把判断逻辑先讲清楚。

```python
def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    assert flops >= 0
    assert bytes_moved > 0
    return flops / bytes_moved


def bound_type(ai: float, achieved_flops_per_s: float,
               peak_bandwidth_bytes_per_s: float,
               peak_flops_per_s: float) -> str:
    memory_roof = ai * peak_bandwidth_bytes_per_s
    compute_roof = peak_flops_per_s
    effective_roof = min(memory_roof, compute_roof)
    assert effective_roof > 0
    return "memory-bound" if memory_roof < compute_roof else "compute-bound"


# 玩具例子：1e9 FLOP, 0.5 GiB traffic
flops = 1e9
bytes_moved = 0.5 * (2 ** 30)
ai = arithmetic_intensity(flops, bytes_moved)

assert round(ai, 2) == 1.86

# 假设峰值带宽 900 GB/s，峰值算力 19.5 TFLOP/s
kind = bound_type(
    ai=ai,
    achieved_flops_per_s=1e12,
    peak_bandwidth_bytes_per_s=900 * (10 ** 9),
    peak_flops_per_s=19.5 * (10 ** 12),
)
assert kind in {"memory-bound", "compute-bound"}
print(ai, kind)
```

接着是 PyTorch 里的最小 profiling 片段。`schedule` 的意思是“按轮次控制何时采样”；`trace_handler` 的意思是“采样窗口结束后，如何导出结果”。

```python
import torch
from torch.profiler import profile, schedule, ProfilerActivity

def train_one_step(model, x):
    y = model(x)
    loss = y.sum()
    loss.backward()

model = torch.nn.Linear(1024, 1024).cuda()
x = torch.randn(64, 1024, device="cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        train_one_step(model, x)
        prof.step()
```

这段代码的关键不是 API 名字，而是采样边界清晰：

1. `wait=1` 表示先跳过第一轮，避免冷启动噪声。
2. `warmup=1` 表示做一轮预热。
3. `active=3` 表示连续记录三轮。
4. `prof.step()` 把“这一轮训练结束”显式告诉 profiler。

这样导出的 trace 文件更容易和训练 step 对齐。打开 `./log` 下的 trace 后，你能看到 Python 侧函数、ATen 算子和 CUDA 活动，适合先在工程内快速定位热点。

如果之后确认瓶颈主要在 GPU 时间线组织上，再去用 Nsight Systems 观察 launch、sync、memcpy 和 kernel 的整体关系；如果已经确认热点集中在一个 kernel，再交给 Nsight Compute 做 roofline 和硬件指标分析。这就是三层工具的实际串联顺序。

---

## 工程权衡与常见坑

第一个常见坑是把 tracing 开销当成真实性能问题。Nsight Systems 对极短事件会引入固定开销，因此看到 10 微秒级别的空隙时，不要立刻断言是同步问题。做法是交叉验证：同时用 `time.perf_counter()` 量 step 级别耗时，再看关闭 profiling 后间隙是否仍然存在。如果只有开 profiler 才出现，优先怀疑观测扰动。

第二个常见坑是 roofline 参数设错。roofline 的结论依赖峰值带宽和峰值算力。如果 GPU 型号识别错、频率状态不同、或者混用了不同精度模式，memory-bound 与 compute-bound 的分界线会整体偏移。结论不是“roofline 没用”，而是“输入参数必须和实际硬件、实际精度一致”。

第三个常见坑是 trace 文件过大。粗略看，trace 大小和“事件数 × 每事件记录字段数”近似正相关。训练轮次太多、开启 `with_stack=True`、记录过多 shape 信息时，JSON 会迅速膨胀。文件过大后，Chrome Trace 可能加载很慢甚至失败。实务上通常只抓几个稳定 step，而不是整轮训练全量导出。

第四个常见坑是误把同步写进业务代码。比如为了打印某个张量值、统计某步耗时，随手插入 `.item()`、`torch.cuda.synchronize()` 或频繁 CPU-GPU 数据回传，这些都会改变原本异步的执行节奏。你以为是在“测量”，实际上已经在“干预”。

第五个常见坑是只看单个工具的局部结论。Nsight Compute 看到一个 kernel memory bound，不代表整个训练就是带宽瓶颈；也可能真正的大头是 CPU 数据准备、kernel launch 过碎，或者多个小 kernel 之间空转太多。先用 Systems 定位全局，再用 Compute 确认局部，这是更稳妥的顺序。

---

## 替代方案与适用边界

如果问题只在 Python 层，例如怀疑数据预处理、模块调用顺序、某个自定义 loss 太慢，那么只用 `torch.profiler` 就够了。它部署成本最低，和训练代码贴得最近，适合日常迭代。

如果已经确认问题在 GPU 调度层，比如 kernel 之间空白很多、memcpy 没和计算重叠、同步事件过多，那么应该优先用 Nsight Systems。因为这是“流程编排问题”，不是“单 kernel 指标问题”。

如果已经定位到一个明确的热点 kernel，并且你需要知道它是访存瓶颈、寄存器压力、SM 利用率不足还是张量核心没吃满，那么进入 Nsight Compute 更合适。它比 `torch.profiler` 更底层，比 Systems 更聚焦。

可以用一个简单决策树理解：

先问“是否只关心 Python/算子层？”
如果是，用 `torch.profiler`。  
如果不是，再问“是否需要看 CPU、memcpy、kernel 的整体时间关系？”
如果是，用 Nsight Systems。  
如果已经知道具体慢的是哪个 kernel，并且要判断 compute bound 还是 memory bound，用 Nsight Compute。

还有一个实践边界：在多人协作项目里，`torch.profiler` 更适合做常驻诊断；Nsight Systems 和 Nsight Compute 更适合做专项分析。原因很简单，前者轻，后者深。轻工具负责高频发现问题，深工具负责低频确认问题。

---

## 参考资料

1. NVIDIA 技术博客《Getting Started with CUDA Graphs》：适合理解 CPU 发起 CUDA 工作、launch 开销以及为什么要减少碎片化提交，对本文“时间线与 launch 关系”部分有直接帮助。
2. NVIDIA 技术博客《Understanding the Visualization of Overhead and Latency in Nsight Systems》：重点解释 Nsight Systems 里 overhead、latency、时间线可视化的含义，对“不要误判间隙”这一节最关键。
3. Nsight Compute Profiling Guide 的 roofline 章节：说明 Arithmetic Intensity、roofline 图和 memory bound / compute bound 的判断逻辑，是本文公式与推导部分的主要依据。
4. PyTorch 官方文档中 `torch.profiler.schedule` 与 `on_trace_ready` 相关说明：用于理解采样窗口和 trace 导出方式，对代码实现部分最直接。
5. PyTorch Profiler 与 TensorBoard Trace Viewer 的官方使用文档：适合理解 trace 文件如何落地、如何在可视化界面中关联 Python 调用和 CUDA 活动。
