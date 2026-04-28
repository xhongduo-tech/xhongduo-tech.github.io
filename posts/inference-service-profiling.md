## 核心结论

推理服务性能剖析的目标，不是笼统地回答“GPU 忙不忙”，而是把一次在线请求的端到端时延拆成可解释的时间分量，判断瓶颈究竟落在应用层、框架层还是 GPU 层。应用层就是服务代码、请求排队、前后处理这一层；框架层就是 PyTorch 这类深度学习框架负责算子调度和执行的这一层；GPU 层就是实际执行 kernel、做显存拷贝和消耗带宽算力的这一层。

结论先行可以压缩成一张表：

| 层级 | 常见慢点 | 典型现象 | 主要工具 |
| --- | --- | --- | --- |
| 应用层 | 排队、序列化、Python 前后处理 | GPU 有时空闲，请求总时延却高 | 服务日志、应用埋点、PyTorch Profiler |
| 框架层 | 算子调度、小算子碎片化、shape 抖动 | CPU 侧调度重，算子很多但单个很短 | PyTorch Profiler |
| GPU 层 | H2D/D2H 拷贝、kernel 空洞、单核低效 | 时间线上有空白，拷贝多，热点 kernel 慢 | Nsight Systems、Nsight Compute |

一次请求的总时延可以写成：

$$
T = T_{py} + T_{dispatch} + T_{h2d} + T_{kernel} + T_{d2h} + T_{gap}
$$

其中，$T_{py}$ 是 Python 侧开销，$T_{dispatch}$ 是框架调度开销，$T_{h2d}$ 和 $T_{d2h}$ 是主机与显卡之间的数据搬运，$T_{kernel}$ 是 GPU 真正算数的时间，$T_{gap}$ 是 kernel 之间的空洞时间。

优化顺序也应固定：先看空洞，再看搬运，最后看 kernel。原因很直接：如果 GPU 大量时间在等活，先优化单个 kernel 通常收益最小；如果搬运比例很高，算力升级也很难线性改善总时延；只有当前两项都压得比较低时，单核优化才是主路径。

玩具例子：一次请求总共 10 ms，看起来“模型推理 10 ms”。拆开后可能是 Python 调度 1 ms、框架分发 0.8 ms、H2D 拷贝 2 ms、kernel 计算 4 ms、D2H 拷贝 0.4 ms、空洞 1.8 ms。这个例子说明，总时延不等于“模型算子真的在算”的时间。

---

## 问题定义与边界

本文讨论的是在线推理服务的性能剖析。在线推理指请求来了就要尽快返回结果的服务路径，典型指标是时延、吞吐和尾延迟。尾延迟就是最慢那部分请求的延迟，比如 P95、P99，常用来衡量系统在压力下是否稳定。

先把边界讲清楚：

| 本文讨论 | 不讨论 |
| --- | --- |
| 单次请求或小批请求的在线推理链路 | 训练性能与收敛问题 |
| 应用层、框架层、GPU 层的分层定位 | 模型结构创新本身 |
| P50、P95、P99、QPS、GPU 利用率、带宽、occupancy | 业务策略、AB 实验收益 |
| PyTorch Profiler、Nsight Systems、Nsight Compute 的配合 | 大规模分布式训练调优 |

occupancy 可以白话理解为“GPU 执行单元被填满的程度”。它高不一定代表系统整体快，但它低时往往提示单个 kernel 没有把硬件吃满。

常看指标可以总结如下：

| 指标 | 含义 | 适合回答的问题 |
| --- | --- | --- |
| P50 | 一半请求不超过这个时延 | 常态时延如何 |
| P95 / P99 | 最慢 5% / 1% 请求的时延 | 长尾是否严重 |
| QPS | 每秒处理请求数 | 系统吞吐上限 |
| GPU 利用率 | 某时间窗内 GPU 是否在忙 | 粗看资源忙闲 |
| 带宽利用 | 显存或总线搬运是否接近上限 | 是否搬运受限 |
| occupancy | 单 kernel 的并发填充程度 | kernel 是否吃满硬件 |

新手常见误区是把“慢”当成一个问题。实际上至少有三种完全不同的慢法。第一种是请求排队久，模型本身未必慢；第二种是输入输出搬运多，GPU 在等数据；第三种是算子真的重，单个 kernel 已经成为主瓶颈。三种情况都表现为“接口慢”，但优化动作完全不同。

---

## 核心机制与推导

性能剖析的核心方法是分解。只看总时延，等于把所有阶段混成一个数字，无法知道该优化哪一段。

分解公式仍然是：

$$
T = T_{py} + T_{dispatch} + T_{h2d} + T_{kernel} + T_{d2h} + T_{gap}
$$

为了判断优先级，再定义每段占比：

$$
U_x = \frac{T_x}{T}
$$

这个占比非常重要，因为优化收益的理论上限由它决定。若某段只占 5%，你把它优化掉一半，总时延也只改善 2.5%。这就是为什么很多优化做了很多事，却几乎不降时延。

最小数值例子如下：

| 阶段 | 时延 ms | 占比 |
| --- | ---: | ---: |
| Python 前后处理 $T_{py}$ | 1.2 | 12% |
| 框架分发 $T_{dispatch}$ | 0.8 | 8% |
| H2D 拷贝 $T_{h2d}$ | 2.0 | 20% |
| kernel 计算 $T_{kernel}$ | 4.0 | 40% |
| D2H 拷贝 $T_{d2h}$ | 0.4 | 4% |
| 空洞 $T_{gap}$ | 1.6 | 16% |
| 总计 $T$ | 10.0 | 100% |

从这张表可以直接推出决策顺序。先看 $T_{gap}$，因为 16% 的空洞说明 GPU 没有连续工作，继续深挖单核优化很可能抓不到主因。再看搬运，因为 H2D 和 D2H 加起来有 24%，这已经足以决定不少小批量场景的时延上限。最后才看 kernel，因为只有当前两者都比较可控时，单核优化才会成为最高收益项。

工具分工也必须分层理解：

| 工具 | 主要回答的问题 | 看不到什么 |
| --- | --- | --- |
| PyTorch Profiler | 哪些算子慢、CPU 调度是否重、shape 是否抖动 | GPU 时间线空洞的根因细节不完整 |
| Nsight Systems | kernel、拷贝、同步、空洞在时间线上如何排列 | 单个 kernel 的底层硬件瓶颈不够细 |
| Nsight Compute | 某个热点 kernel 是算力受限还是带宽受限 | 整个请求链路和排队关系 |

真实工程例子：一个 LLM 在线服务在 `batch=1~4` 下，PyTorch Profiler 看到 `aten::to`、`copy_` 和很多短小 `matmul`。这只能说明算子碎、拷贝不少。接着看 Nsight Systems，如果发现 GPU 时间线上 kernel 之间有明显白块，就知道还有调度等待或数据没准备好。最后再用 Nsight Compute 看热点 kernel，若发现带宽接近上限而 SM 利用一般，说明它更像 memory-bound，也就是“被数据读写速度卡住”，不是纯算力不够。

---

## 代码实现

代码实现的核心不是“把 profiler 打开”，而是让请求边界、阶段边界和热点算子都可观测。可观测，白话说，就是让你事后能把一条慢请求拆开看清楚。

先给一个可运行的玩具分析代码，它不依赖 GPU，但能演示如何按阶段做分解和断言：

```python
from math import isclose

parts = {
    "py": 1.2,
    "dispatch": 0.8,
    "h2d": 2.0,
    "kernel": 4.0,
    "d2h": 0.4,
    "gap": 1.6,
}

total = sum(parts.values())
ratios = {k: v / total for k, v in parts.items()}

assert isclose(total, 10.0, rel_tol=1e-9)
assert ratios["kernel"] == 0.4
assert ratios["h2d"] + ratios["d2h"] == 0.24
assert ratios["gap"] > ratios["dispatch"]

priority = sorted(parts.items(), key=lambda kv: kv[1], reverse=True)
assert priority[0][0] == "kernel"
assert priority[1][0] == "h2d"
print("total:", total, "ratios:", ratios)
```

真正落到 PyTorch 服务里，至少要有这几个阶段：请求入口、前处理、模型前向、后处理、结果导出。结构可以理解成下面这样：

| 阶段 | 作用 | 推荐标记方式 |
| --- | --- | --- |
| 请求入口 | 绑定 request id，记录起点 | 应用日志 + NVTX range |
| 前处理 | tokenizer、特征组装、tensor 构造 | `record_function` + NVTX |
| 推理 | `model(...)` 或 `generate(...)` | profiler + NVTX |
| 后处理 | decode、格式转换、裁剪 | `record_function` + NVTX |
| 结果导出 | 汇总统计、落盘 trace | profiler 输出 |

最小 PyTorch Profiler 示例：

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

model = torch.nn.Linear(8, 4).eval()
x = torch.randn(2, 8)

with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("request"):
        with record_function("preprocess"):
            inp = x.float()
        with record_function("forward"):
            with torch.no_grad():
                y = model(inp)
        with record_function("postprocess"):
            z = y.softmax(dim=-1)

table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
print(table)
assert z.shape == (2, 4)
```

如果环境有 CUDA，还应加 NVTX。NVTX 可以白话理解成“给 GPU 时间线贴标签”，这样你在 Nsight Systems 里能把一段时间对应回具体请求阶段。

```python
import torch

torch.cuda.nvtx.range_push("request:42/preprocess")
# preprocess ...
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("request:42/forward")
# model forward ...
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("request:42/postprocess")
# postprocess ...
torch.cuda.nvtx.range_pop()
```

结果读取时，不要只盯着一个最慢算子，而要把热点和阶段边界一起看。常见做法是先从 `key_averages()` 找到热点算子，再去 Nsight Systems 对齐同一段时间线，判断它慢在调度、拷贝还是 kernel 本身。

---

## 工程权衡与常见坑

profiling 本身有开销。开销就是“为了测量而额外引入的成本”。因此 profiling 不能无脑长期开着，尤其在线上高流量服务里，采样窗口、warmup 和同步点设置不合理时，测出来的结论可能比不测更误导。

常见坑可以先列成表：

| 坑 | 后果 | 规避方式 |
| --- | --- | --- |
| 只看 `nvidia-smi` 利用率 | 误以为 GPU 已吃满，忽略空洞 | 必看时间线 |
| 只跑一种 profiler | 只能看到局部，不知道瓶颈层级 | PyTorch Profiler 与 Nsight 配合 |
| 忽略 warmup | 冷启动、缓存未命中污染结果 | 固定 warmup 轮次后再采样 |
| 只测单一 shape | 线上真实分布被掩盖 | 按输入长度或分辨率分桶 |
| 没有 NVTX 标记 | 请求边界对不齐，难以复盘 | 给关键阶段统一打标签 |

还有两个非常常见的误判。

第一，只看平均值。平均值可能很好看，但 P99 很差，线上用户感受到的往往是尾延迟而不是平均值。第二，把离线微基准结果直接套到线上。微基准就是把某个算子单独拿出来测，它适合回答“这个 kernel 理论上能跑多快”，但不适合直接回答“整个服务为什么慢”。因为线上还有排队、调度、数据准备、碎片化输入这些因素。

一个典型的新手坑是：`nvidia-smi` 显示 GPU 利用率 90%，于是判断“GPU 已经满了”。这并不可靠。利用率是时间窗内的粗指标，可能只是某一小段时间里有大 kernel 在跑，但请求之间仍有大量等待和碎片化拷贝。真正能看出空洞的，是时间线而不是单个百分比。

---

## 替代方案与适用边界

不是所有问题都值得完整做三层剖析。分层是方法，不是仪式。简单问题用简单工具，复杂问题再下钻。

方案选择可以这样看：

| 方法 | 适用场景 | 优点 | 局限 |
| --- | --- | --- | --- |
| 只看应用埋点 | 怀疑排队、超时、序列化 | 成本低，最贴近线上 | 看不到框架和 GPU 细节 |
| PyTorch Profiler | 怀疑算子热点、调度重、shape 抖动 | 上手快，直接贴近代码 | GPU 底层细节有限 |
| Nsight Systems | 怀疑空洞、同步、拷贝问题 | 能看完整 GPU 时间线 | 单 kernel 指标不够细 |
| Nsight Compute | 已锁定某个热点 kernel | 能看 occupancy、带宽、stall | 无法解释整条请求链路 |

可以把决策流程简化成三步：

1. 如果请求大部分时间花在排队、前后处理或 CPU 调度，先停在应用层和框架层。
2. 如果 PyTorch Profiler 发现 CUDA 时间不集中、算子之间关系不清楚，进入 Nsight Systems。
3. 如果 Nsight Systems 已确认瓶颈集中在单个或少数 kernel，再进入 Nsight Compute。

这也对应“微基准优化”和“在线服务优化”的边界。微基准优化关注单算子或单 kernel 的极限性能；在线服务优化关注真实请求在真实链路里的总收益。前者可以指导底层实现，后者决定用户实际体验。两者有关联，但不能直接替代。

---

## 参考资料

1. [PyTorch Profiler 官方文档](https://docs.pytorch.org/docs/stable/profiler.html) 用于看框架层算子热点、CPU 调度和 shape 信息。
2. [PyTorch `record_function` 文档](https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.record_function.html) 用于给前处理、前向和后处理打阶段标记。
3. [PyTorch CUDA / NVTX 文档](https://docs.pytorch.org/docs/stable/cuda) 用于给 GPU 时间线添加请求与阶段标签。
4. [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/) 用于看时间线空洞、同步关系、拷贝与 kernel 排布。
5. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) 用于看单个 kernel 的 occupancy、带宽、stall 原因。
6. [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) 用于对照常见框架级性能优化项，避免先改次要路径。
