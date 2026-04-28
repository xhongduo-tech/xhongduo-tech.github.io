## 核心结论

ONNX Runtime（简称 ORT，白话说就是“专门负责执行 ONNX 模型的运行时”）的推理性能，主要由三件事决定：执行提供器分配、图优化与融合、内存复用。

执行提供器（Execution Provider，简称 EP，白话说就是“具体替你执行算子的后端”）决定一个算子最终落在 CPU、CUDA 还是 TensorRT 上。图融合（fusion，白话说就是“把多个小算子合并成一个更大的执行单元”）决定 kernel 启动次数和跨设备搬运次数。arena 分配器（白话说就是“预留并复用内存的分配器”）决定临时张量能否少申请、少释放、少碎片。

可以用两个式子抓住主线：

$$
L_{request} \approx \sum_i (t_{compute,i}+t_{launch,i}) + t_{copy,H2D} + t_{copy,D2H}
$$

$$
P_{peak} \approx M_{live} + M_{arena\ slack} + M_{workspace}
$$

第一个式子描述单请求延迟，第二个式子描述峰值显存。很多优化收益并不是“算得更快”，而是“少启动、少搬运、少碎片”。

| 变量 | 本质作用 | 主要影响指标 | 常见现象 |
|---|---|---|---|
| 执行提供器分配 | 决定算子落点与子图下沉范围 | 延迟、吞吐、稳定性 | EP 顺序不对时大量节点回落 CPU |
| 图优化与融合 | 减少节点数、kernel 启动、调度开销 | 延迟、吞吐 | 计算时间几乎不变，但总延迟下降 |
| 内存复用 | 复用中间张量与工作区 | 显存峰值、并发上限 | 峰值更稳，但可能多一些预留空间 |

玩具例子先看一个。假设原图有 12 个 GPU kernel，每个启动开销是 $8\ \mu s$，总计算时间是 $0.90\ ms$，主机到设备与设备到主机的搬运一共 $0.25\ ms$。那么总延迟约为：

$$
12 \times 0.008 + 0.90 + 0.25 = 1.246\ ms
$$

如果图融合后只剩 5 个 kernel，并且更多节点被同一个 GPU EP 吃掉，搬运降到 $0.10\ ms$，则总延迟变成：

$$
5 \times 0.008 + 0.90 + 0.10 = 1.04\ ms
$$

纯计算时间没变，但总延迟下降了。这就是 ORT 优化最常见的收益来源。

---

## 问题定义与边界

“推理性能”不能只理解成“更快”。在工程里，至少要拆成四个维度：

| 指标 | 定义 | 为什么重要 |
|---|---|---|
| 单请求延迟 | 一个请求从输入到输出的耗时 | 影响 P50/P95 响应时间 |
| 批量吞吐 | 单位时间能处理多少样本 | 影响成本与资源利用率 |
| 峰值显存/内存 | 一次推理中最高占用 | 影响单卡并发上限 |
| Session 创建时间 | `InferenceSession` 初始化耗时 | 影响冷启动与弹性扩容 |

本文只讨论 ORT 的推理优化，不讨论以下内容：

| 本文关注 | 本文不关注 |
|---|---|
| EP 分区、图优化、内存复用、I/O binding | 训练加速、量化精度误差分析、模型结构改写、蒸馏与剪枝 |
| 单机在线推理与离线批推理 | 分布式训练、参数服务器、数据并行 |

几个术语先定清楚。

EP：执行提供器，也就是具体执行算子的后端，比如 `CPUExecutionProvider`、`CUDAExecutionProvider`、`TensorRTExecutionProvider`。  
subgraph：子图，白话说就是“从整张计算图里切出来、交给某个 EP 的一部分节点集合”。  
fusion：融合，把多个节点合成一个更大的节点或可执行单元。  
arena：内存池式分配器，通过预留和复用内存减少频繁申请释放。  
fallback：回退，指某个 EP 吃不下某些节点后，交给后面的 EP 兜底。

一个边界很重要：同一个 BERT 模型，在 CPU EP、CUDA EP、TensorRT EP 下表现不同，模型数学定义没变，变的是“谁来执行哪些节点，以及数据是否需要来回搬运”。所以 ORT 性能优化，调的不是模型公式本身，而是执行路径。

---

## 核心机制与推导

ORT 的运行流程可以按顺序理解为：

`ONNX 模型 -> 图预处理/图优化 -> EP 能力查询 -> 图分区 -> 子图编译/融合 -> 执行`

其中关键步骤有三个。

第一步，图优化。ORT 会先做通用优化，比如常量折叠、冗余节点消除、某些模式融合。常量折叠（白话说就是“把编译期就能算出的结果提前算好”）能减少运行期工作量。

第二步，EP 分区。每个 EP 会通过类似 `GetCapability()` 的机制声明“我能吃哪些节点”。ORT 根据能力把图切成不同子图。一个连续的大子图越能整体下沉到同一个 EP，上下文切换和数据拷贝通常越少。

第三步，子图编译。某些 EP 会把子图进一步编译成更大的执行单元。TensorRT 这类后端尤其明显，它更喜欢固定 shape、连续且可融合的子图。

图优化等级可以粗分为三类：

| 优化级别 | 典型内容 | 发生时机 | 对性能的主要作用 |
|---|---|---|---|
| Basic | 常量折叠、无效节点删除 | 较早阶段 | 先减节点数 |
| Extended | 更复杂的模式融合 | 分区前后都可能有影响 | 降低 kernel 数和调度成本 |
| Layout | 与数据布局相关的优化 | 依赖后端能力 | 提升特定 EP 执行效率 |

为什么图融合有效，可以直接从延迟式子推导。若一个请求被拆成很多小算子，则每个算子除了计算时间 $t_{compute}$，还要支付启动时间 $t_{launch}$。当模型存在大量小 kernel 时，$\sum t_{launch,i}$ 可能并不小。融合后，即使 $\sum t_{compute,i}$ 基本不变，$\sum t_{launch,i}$ 会明显下降。

再看显存。推理时的峰值显存并不只由“当前活跃张量”决定，还包括 arena 预留空间和后端工作区：

$$
P_{peak} \approx M_{live} + M_{arena\ slack} + M_{workspace}
$$

这里 `workspace` 可以理解成 cuDNN、TensorRT 在执行时需要的临时工作区。`arena_slack` 可以理解成“为了更快复用而暂时保留的空余空间”。所以一个常见现象是：开启 arena 后，平均分配更快，但瞬时看到的占用可能并不总是最低。

真实工程例子看 BERT 在线服务。若 attention、GELU、layer norm 等模式能被 TensorRT EP 连续吃掉，剩余少量 GPU 算子由 CUDA EP 接手，CPU EP 只做最后兜底，那么一次请求通常会经历更少的子图边界、更少的 H2D/D2H 拷贝和更少的 fallback。反过来，如果 dynamic shape 太多，导致 TensorRT cache 命中差、可融合子图变碎，延迟就可能明显抖动。

---

## 代码实现

下面给一个最小可复现的 Python 示例，重点不在业务逻辑，而在 session 配置。前面的 EP 先尝试吃图，后面的 EP 负责兜底。

```python
import onnxruntime as ort

def build_session(model_path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_profiling = True

    providers = [
        ("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
            "trt_fp16_enable": True,
        }),
        ("CUDAExecutionProvider", {
            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
            "arena_extend_strategy": "kNextPowerOfTwo",
        }),
        "CPUExecutionProvider",
    ]

    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    active = sess.get_providers()

    # 兜底断言：至少要保留 CPU EP，避免模型完全不可执行
    assert "CPUExecutionProvider" in active

    # 用户请求了 TensorRT / CUDA 时，最终激活列表通常应包含至少一个 GPU EP
    want_gpu = any(
        (isinstance(p, tuple) and "ExecutionProvider" in p[0] and p[0] != "CPUExecutionProvider")
        or (isinstance(p, str) and p != "CPUExecutionProvider")
        for p in providers
    )
    if want_gpu:
        assert any(p in active for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider"])

    return sess

def estimate_latency_ms(num_kernels, launch_us, compute_ms, copy_ms):
    total = num_kernels * (launch_us / 1000.0) + compute_ms + copy_ms
    return round(total, 3)

before = estimate_latency_ms(12, 8, 0.90, 0.25)
after = estimate_latency_ms(5, 8, 0.90, 0.10)

assert before == 1.246
assert after == 1.04
assert after < before
```

如果输入输出本来就在 GPU 上，I/O binding 也很关键。I/O binding（白话说就是“显式指定输入输出绑在哪个设备上”）可以减少不必要的主机设备拷贝。新手常见问题是：模型明明跑在 GPU，结果前后还在 CPU 和 GPU 之间搬来搬去。

概念上可以这样理解：

```python
# 伪代码：展示意图，不保证直接运行
io = sess.io_binding()
# 把 GPU 上已有的输入 buffer 直接绑给输入名
# io.bind_input(...)
# 把输出也绑定到 GPU 或预分配 buffer
# io.bind_output(...)
sess.run_with_iobinding(io)
```

如果是服务启动时做预热，还应关注两件事。第一，提前创建 session，让图优化和 EP 编译成本不要落到首个请求上。第二，如果使用 TensorRT cache，缓存目录要稳定，否则每次重建 engine 都会拉高冷启动时间。

---

## 工程权衡与常见坑

工程里最常见的误区，不是“不知道参数名”，而是“只调参数，不看 profile 和分区结果”。

| 坑点 | 典型症状 | 排查方式 | 规避策略 |
|---|---|---|---|
| EP 顺序错误 | 模型能跑，但延迟更差 | 看 provider 列表、profile、节点分配 | 把更强的 GPU EP 放前面，CPU 放最后 |
| dynamic shape 过多 | TensorRT cache 命中差，延迟抖动 | 看 engine 是否频繁重建 | 尽量收敛 shape 桶，固定常见输入尺寸 |
| 盲目关闭 arena | 峰值下降一点，但吞吐明显变差 | 对比 profile 与内存曲线 | 先用于排障，再决定是否长期关闭 |
| 误解 `gpu_mem_limit` | 以为显存绝不会超过该值 | 观察实际显存和 workspace | 理解它主要约束 allocator，不覆盖全部占用 |
| 离线优化跨机器复用 | 换机后性能异常甚至不可用 | 核对硬件、驱动、EP 版本 | 把离线产物与环境绑定管理 |

真实工程例子可以看一个线上文本分类服务。模型是 RoBERTa，部署在单张 GPU 上，要求 P95 延迟小于 20ms，同时单卡支持稳定并发。最开始团队只做了“开 CUDA EP”，结果很多 layer 仍在 CPU fallback，profile 里能看到多次设备拷贝。后来改成 `TensorRT -> CUDA -> CPU` 的 provider 顺序，收敛输入长度桶，开启 engine cache，并把输出也做 I/O binding，P95 降得最明显的部分并不是 matmul 计算本身，而是 fallback 和 copy 的减少。

还要明确几个权衡：

| 优化方向 | 对延迟 | 对吞吐 | 对显存 | 对排障复杂度 |
|---|---|---|---|---|
| 更激进的图融合 | 通常更好 | 通常更好 | 中性 | 更高 |
| 更强的 GPU EP 下沉 | 通常更好 | 通常更好 | 可能更高 | 更高 |
| 更积极的 arena 复用 | 通常更好 | 更好 | 峰值未必最低 | 中等 |
| 更严格的固定 shape | 更稳 | 更稳 | 更可控 | 业务约束更强 |

所以正确顺序通常是：先看 profile，再看节点分区，再看 copy 路径，最后才是细调 `gpu_mem_limit`、arena 策略和 cache 开关。

---

## 替代方案与适用边界

ORT 不是所有场景的唯一最优解。它的优势是统一接口、多 EP、跨硬件部署便利；它的代价是多一层运行时抽象，极限性能不一定总能压到最低。

| 方案 | 性能上限 | 部署复杂度 | 跨硬件能力 | 维护成本 | 适用场景 |
|---|---|---|---|---|---|
| ORT + CUDA | 高 | 中 | 较强 | 中 | 多模型、需要统一接口、GPU 推理 |
| ORT + TensorRT | 很高 | 中到高 | 中 | 中到高 | 需要高性能且仍想保留 ORT 统一入口 |
| 纯 TensorRT | 最高之一 | 高 | 较弱 | 高 | 固定硬件、固定 shape、追求极致低延迟 |
| 纯 CPU runtime | 中到低 | 低 | 强 | 低 | 小模型、CPU 服务、成本敏感场景 |

判断边界可以用三条经验。

第一，固定 shape、硬件固定、目标是极致低延迟时，纯 TensorRT 往往更值得深挖。  
第二，多模型、多环境、需要快速上线和统一推理入口时，ORT 更合适。  
第三，如果本来就是 CPU 推理，不要一开始就跳到 GPU 方案，先把 CPU EP、线程配置、图优化看明白，常常更划算。

换句话说，ORT 更像“工程上平衡性能、可移植性和维护成本”的方案；纯 TensorRT 更像“为单一硬件压榨极限性能”的方案。

---

## 参考资料

资料导航表：

| 资料 | 主题 | 建议阅读时机 | 在本文中的作用 |
|---|---|---|---|
| Architecture | 整体架构 | 最先读 | 建立“优化-分区-执行”的主流程 |
| Graph Optimizations | 图优化 | 读完架构后 | 理解为什么节点数和融合会影响延迟 |
| Execution Providers | EP 体系 | 机制章节后 | 理解为什么同一模型会有不同执行路径 |
| CUDA EP | CUDA 参数与行为 | 代码实现前后 | 对应 provider 配置与显存相关参数 |
| TensorRT EP | TensorRT 子图与 cache | 工程优化时 | 对应高性能 GPU 场景 |
| Memory Consumption | 内存机制 | 调优与排障时 | 理解 arena、峰值内存与共享策略 |

1. [ONNX Runtime Architecture](https://onnxruntime.ai/docs/reference/high-level-design.html)
2. [Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
3. [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
4. [CUDA Execution Provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
5. [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
6. [Memory Consumption](https://onnxruntime.ai/docs/performance/tune-performance/memory.html)
