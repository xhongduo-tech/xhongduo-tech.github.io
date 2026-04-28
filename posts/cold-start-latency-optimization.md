## 核心结论

大模型推理服务的冷启动延迟，不是一次“慢请求”这么简单，而是三段成本叠加：权重加载、CUDA context 初始化、KV Cache 准备。用一个直接的量级例子说明：如果一个 70B 模型用 FP16 或 BF16 存储，权重大约是 140GB，首次拉起服务时，仅把权重从 NVMe SSD 读到内存并搬到 GPU，就可能花掉 3 到 5 秒；再加上 CUDA runtime 和 driver 建立执行环境约 0.5 秒，以及缓存结构准备时间，用户首请求看到的“卡几秒”是完全可能的。

冷启动优化的目标，不是把启动成本从物理上消灭，而是把它从用户首请求路径里移走。更准确地说，应该把冷启动变成部署阶段、预热阶段、灰度阶段承担的后台成本，而不是线上首批用户承担的前台成本。

总公式可以先记住：

$$
T_{cold} \approx T_{load} + T_{cuda} + T_{kv}
$$

| 符号 | 含义 | 典型来源 |
|---|---|---|
| $T_{load}$ | 权重加载时间 | 从 NVMe、网络盘或对象存储读取模型文件 |
| $T_{cuda}$ | CUDA context 初始化时间 | GPU 驱动、上下文、kernel 首次准备 |
| $T_{kv}$ | KV Cache 准备时间 | 推理缓存页、内存池、调度结构建立 |
| $T_{cold}$ | 总冷启动时间 | 用户真正能发起可用推理前的总等待 |

结论很明确：如果只盯着某一项，例如只做“空请求预热”，通常不够。真正有效的优化，是把加载、初始化、缓存准备和切流流程整体设计成“先准备，再接流量”。

---

## 问题定义与边界

冷启动延迟，指的是“一个推理实例从未就绪状态进入可服务状态所需的时间”。这里有两个很容易混淆的阶段。

第一，服务进程已启动。意思是进程活着，端口也许已经监听，健康检查可能返回存活。第二，模型已进入可服务状态。意思是模型权重已经可读、GPU 执行环境已经可用、推理缓存已经能承接真实请求。前者只是“机器开了门”，后者才是“厨房真的能出菜”。

给零基础读者一个白话解释：刚开机的 GPU 还没准备好，可以暂时把它想成厨房刚开张，但锅、火、食材都没摆好。不过工程上不能停留在比喻里，真正的定义是：服务只有在模型和依赖资源都准备完成后，才算 ready，而不是进程一启动就算 ready。

本文边界很窄，只讨论 LLM 推理服务的冷启动，不展开以下问题：

| 术语 | 白话解释 | 本文是否展开 |
|---|---|---|
| 权重加载 | 把模型参数文件搬进可计算位置 | 是 |
| CUDA context | GPU 的执行上下文，类似“GPU 的进程工作环境” | 是 |
| KV Cache | 保存历史 token 中间结果的缓存，用于避免重复计算 | 是 |
| 预热 | 在正式接流量前先跑一组请求，把慢路径走一遍 | 是 |
| 热实例 | 已经准备好，首个真实请求不会明显变慢的实例 | 是 |
| 冷实例 | 刚启动，还没完成准备的实例 | 是 |
| 容器调度 | Pod 拉起、镜像拉取、节点分配 | 否 |
| 网络握手 | 网关、TLS、长连接建立 | 否 |
| 前端等待 | 页面轮询、重试、用户感知层逻辑 | 否 |

这篇文章讨论的不是所有启动延迟，而是模型服务内部那部分最贵、最常见、最容易误判的启动成本。

---

## 核心机制与推导

先拆公式：

$$
T_{cold} \approx T_{load} + T_{cuda} + T_{kv}
$$

### 1. 权重加载为什么慢

权重加载就是把模型参数搬进推理所需的内存层级。模型越大、精度越高、磁盘越慢，$T_{load}$ 越长。

在 FP16 或 BF16 下，每个参数通常占 2 字节，因此：

$$
W \approx 2N_p
$$

其中 $N_p$ 是参数个数，$W$ 是权重字节数。70B 模型大约有 $70 \times 10^9$ 个参数，所以：

$$
W \approx 2 \times 70 \times 10^9 = 140 \times 10^9 \text{ bytes}
$$

也就是约 140GB。

如果 NVMe 有效带宽按 30GB/s 粗估，那么：

$$
T_{load} \approx \frac{W}{BW_{nvme}} \approx \frac{140}{30} \approx 4.7s
$$

这就是“70B 首次请求慢几秒”的最小推导。真实工程里还要考虑文件切片、CPU 内存拷贝、PCIe 传输、校验和格式转换，所以 3 到 5 秒是很常见的数量级，不夸张。

### 2. CUDA context 为什么不是零成本

CUDA context 可以白话理解为“GPU 开始干活前需要建立的运行现场”。第一次调用 CUDA API、第一次分配显存、第一次启动 kernel，常常都会触发初始化。这个阶段通常不是几秒级大头，但 0.3 到 0.8 秒并不少见，尤其是在实例刚拉起、driver 首次介入时。

很多团队的问题不是不知道这件事，而是把它漏在健康检查之后。结果是实例对外显示 ready，但第一个真实用户请求才去付这 0.5 秒成本。

### 3. KV Cache 为什么也会影响冷启动

KV Cache 是自回归生成里的历史状态缓存。白话说，它保存前面 token 的“记忆”，这样模型生成下一个 token 时，不用把前文全部再算一遍。

如果服务支持长上下文、多并发、流式输出，那么 KV Cache 不是一个小结构。它会涉及显存池、页表、块分配器、调度元数据。一次性整块预留简单，但浪费大；分页式分配更灵活，能降低显存碎片和无效占用。

PagedAttention 的核心思路，是把 KV Cache 按块管理。假设一个序列长度为 $L$，每块能容纳 $B$ 个 token，则所需块数近似为：

$$
N_{block} = \lceil \frac{L}{B} \rceil
$$

这和操作系统按页管理内存类似。优点是请求长度不同、生命周期不同的时候，不需要每个请求都预留一整段连续大块显存，更适合长上下文和多租户场景。

### 玩具例子

假设你有一个教学模型，参数量只有 1B，采用 FP16，则权重大约 2GB。如果本地 SSD 有效带宽 2GB/s，那么理论加载时间就是 1 秒左右。这个例子说明，冷启动不是“大模型专属概念”，而是所有模型都有，只是大模型把问题放大到了用户可感知级别。

### 真实工程例子

一个在线问答服务平时保留 2 个热副本，流量高峰时自动扩到 4 个副本。运维以为“扩容成功”就结束了，但新副本只是进程起来了，没有完成权重加载和真实 prompt 预热。结果扩容后的首批用户正好被路由到冷实例，请求延迟从 800ms 突然升到 6s。问题不在扩容本身，而在“ready 的定义错了”。

| 阶段 | 作用 | 常见瓶颈 | 可优化手段 |
|---|---|---|---|
| 权重加载 | 让模型参数进入可计算位置 | NVMe 带宽、格式转换、CPU/GPU 传输 | 权重常驻、分层加载、异步 I/O |
| CUDA 初始化 | 建立 GPU 执行环境 | 首次 API 调用、kernel 首次编译/加载 | 启动期显式初始化、预热 kernel |
| KV Cache 准备 | 为真实推理分配缓存与页表 | 显存分配、碎片、元数据建立 | 分页式缓存、内存池预建、真实长度预热 |

---

## 代码实现

真正有效的预热，不是发一个“hello”空请求，而是覆盖真实路径上的关键 kernel、真实 batch、真实 prompt 长度、真实采样参数。否则你预热的是一条假路径，线上仍然会在第一批真实请求上补交成本。

建议的启动顺序如下：

1. 加载权重
2. 建立 CUDA context
3. 初始化 KV Cache 或内存池
4. 执行 warmup 请求
5. 通过 readiness 检查
6. 再切流量

下面给一个可运行的 Python 启动流程玩具实现，用来表达控制顺序，而不是替代真实框架：

```python
import time
from dataclasses import dataclass

@dataclass
class ServiceState:
    weights_loaded: bool = False
    cuda_ready: bool = False
    kv_ready: bool = False
    warmed_up: bool = False

class ModelWarmup:
    def __init__(self, prompt_tokens: int, batch_size: int):
        self.prompt_tokens = prompt_tokens
        self.batch_size = batch_size

def load_weights(weight_gb: float, bw_gbps: float) -> float:
    assert weight_gb > 0
    assert bw_gbps > 0
    return weight_gb / bw_gbps

def init_cuda() -> float:
    return 0.5

def init_kv_cache(max_seq_len: int, block_size: int) -> int:
    assert max_seq_len > 0 and block_size > 0
    blocks = (max_seq_len + block_size - 1) // block_size
    return blocks

def readiness(state: ServiceState) -> bool:
    return all([
        state.weights_loaded,
        state.cuda_ready,
        state.kv_ready,
        state.warmed_up,
    ])

def startup():
    state = ServiceState()
    warmup = ModelWarmup(prompt_tokens=1024, batch_size=4)

    t_load = load_weights(weight_gb=140, bw_gbps=30)
    state.weights_loaded = t_load > 0

    t_cuda = init_cuda()
    state.cuda_ready = t_cuda >= 0.5

    kv_blocks = init_kv_cache(max_seq_len=8192, block_size=16)
    state.kv_ready = kv_blocks == 512

    # 模拟真实预热：覆盖真实长度与 batch，而不是空请求
    if warmup.prompt_tokens >= 1024 and warmup.batch_size >= 4:
        state.warmed_up = True

    assert readiness(state) is True
    return round(t_load + t_cuda, 2), kv_blocks

total_prefix_cost, kv_blocks = startup()
assert total_prefix_cost == 5.17
assert kv_blocks == 512
```

这个例子体现三件事：顺序要固定、预热要像真实请求、readiness 不能只看进程是否存活。

### Triton：`model_warmup` 怎么落地

Triton Inference Server 提供 `model_warmup` 配置，作用是在模型标记可用前，先向模型发送预定义请求。工程上要注意两点：第一，预热输入不要太小；第二，最好覆盖和线上主流流量接近的输入长度，否则只能触发部分 kernel。

### vLLM：PagedAttention 怎么落地

vLLM 的价值不是“帮你消灭冷启动”，而是把 KV Cache 的使用方式从粗糙的大块预留，变成更接近分页内存管理的模式。代码层的关键不是手写分页算法，而是确保初始化时把缓存管理器、显存池和目标上下文长度一起纳入启动流程，而不是在首个大请求到达时再动态拉起。

### 异步加载：怎么做分层启动

当单卡或单节点不适合全量常驻时，可以考虑分层加载。直白说，就是先把最小可服务能力拉起来，再继续后台补齐后续层、额外副本或次级缓存。要点有两个：ready 标志必须绑定“当前已承诺的服务能力”；后台加载不能和前台流量抢到不可控。

### 滚动发布：怎么避免用户撞冷实例

滚动发布不是“新副本起来就替换旧副本”，而是“新副本预热完成后再接流，再下线旧副本”。如果顺序反了，本质上只是把冷启动从启动阶段挪到了线上。

---

## 工程权衡与常见坑

冷启动优化本质上是在启动速度、显存占用、实例数量和实现复杂度之间做交换。最容易犯的错误，是只追求“启动快”，却牺牲了稳定供给。

一个常见真实运维场景是：为了节省显存，只保留单副本；发布时先停旧副本，再起新副本。结果新副本虽然启动更快，但用户请求全部撞到冷实例，甚至因为 KV Cache 临时扩容触发 OOM。看起来“优化了启动”，实际上恶化了服务。

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 单副本预热 | 首批用户直接撞冷启动 | 至少保留一个热副本承接流量 |
| 只测空请求 | 真正业务请求仍触发慢路径 | 用真实长度、真实 batch、真实采样参数预热 |
| readiness 只看进程活着 | 未就绪实例提前接流 | readiness 绑定权重、CUDA、KV、warmup 完成 |
| 没有 I/O 与计算重叠 | 启动链路串行过长 | 异步加载、后台搬运、分阶段准备 |
| 一次性大块预留 KV | 显存浪费、碎片严重 | 分页式 KV 或内存池策略 |
| 热实例过少 | 扩容或发布时抖动明显 | 多副本轮换，不让服务窗口归零 |

优先级可以这样排：

| 优先级 | 建议 |
|---|---|
| 必须做 | 正确的 readiness、真实预热、滚动切流 |
| 建议做 | 多热副本、KV 分页管理、启动顺序显式化 |
| 可选做 | 分层流式加载、后台异步补齐、复杂缓存调度 |

---

## 替代方案与适用边界

不是所有服务都值得上同一套冷启动优化。模型规模、流量模式、成本约束不同，策略就不同。

| 方案 | 启动速度 | 显存占用 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|
| 权重常驻显存（热备） | 最快 | 最高 | 低 | 高价值在线服务、稳定高流量 |
| 多副本轮换 | 快 | 高 | 中 | 需要滚动发布、弹性扩容 |
| 分层流式加载 | 中 | 中 | 高 | 单卡放不下全量热备、超大模型 |
| 按需加载 | 慢 | 低 | 中 | 低频内部服务、成本敏感 |
| 分页式 KV | 间接加速 | 中 | 中 | 长上下文、多并发推理 |

边界案例很典型：如果单卡显存根本放不下全量热备权重，你就不能靠“强行全量预热”解决问题，因为它在资源上不可成立。这时必须考虑分层加载、张量并行、磁盘到显存的流水化搬运，或者降低单实例承诺能力。

什么时候不值得优化？如果模型很小，比如几百 MB 到 1GB；冷启动频率极低，比如一天一次；或者用户对秒级抖动不敏感，比如离线内部工具，那么复杂的热备、分层加载、滚动池化可能不划算。工程优化不是把所有高级方案都用上，而是让收益大于复杂度。

本文中的主要结论来源可以这样对应：`model_warmup` 的 ready 前预热语义来自 Triton 文档；分页式 KV 的块管理思路来自 PagedAttention 论文和 vLLM 官方资料；权重大小与加载时间的数字是基于参数量、精度和磁盘带宽的工程推导；CUDA context 初始化的概念来自 CUDA Driver API 文档。

---

## 参考资料

机制：
1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
2. [CUDA Driver API - Context Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)

实现：
1. [NVIDIA Triton Inference Server User Guide - Model Warmup](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_navigator/docs/inference_deployment/triton/api/warmup.html)
2. [vLLM Documentation](https://docs.vllm.ai/)

工程实践：
1. [vLLM Blog: PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
2. [NVIDIA Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/)
