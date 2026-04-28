## 核心结论

模型压缩率不是单一数字，而是一组部署指标的联合评估。参数量是“模型里有多少权重”的计数，白话讲就是模型要记住多少个数；但部署时真正决定值不值得压缩的，不只是这个计数，还包括存储体积、峰值显存、延迟、吞吐和精度。

必须先给出判定标准。统一记号后，有：

$$
R_{param} = \frac{N_0}{N_1},\quad
R_{store} = \frac{S_0}{S_1},\quad
R_{mem} = \frac{M_0}{M_1},\quad
S_{lat} = \frac{L_0}{L_1},\quad
S_{thr} = \frac{T_1}{T_0}
$$

其中：

- $N$ 是参数量
- $S$ 是模型落盘存储大小
- $M$ 是推理时峰值显存
- $L$ 是延迟
- $T$ 是吞吐

结论句可以直接写成：

**压缩有效 = 指标改善 + 精度达标 + 目标硬件真实成立**

更严格一点，可以写成：

$$
A_1 \ge A_{min}
\quad \text{且} \quad
R_{store}, R_{mem}, S_{lat}, S_{thr} \text{ 至少一项在目标硬件上真实成立}
$$

这里的 $A$ 是任务指标，白话讲就是模型在真实任务上的效果分数。

先看一个玩具例子。一个模型从 `100M` 参数压到 `25M` 参数，看起来是 `4x` 参数压缩；但如果推理时主要占内存的是激活值和 `KV cache`，或者低精度算子没有被后端真正加速，那么服务延迟可能只从 `20ms` 变到 `18ms`。这说明“模型更小”和“服务更快”不是同一个命题。

再看真实工程例子。LLM 在 GPU 上做推理时，把权重从 `FP16` 换成 `INT8`，模型文件通常会变小，显存通常也会下降；但如果目标引擎里有部分算子回退到更高精度，或者 batch 很小导致计算根本没打满，延迟收益就可能很有限。论文里写“压缩很多倍”，不等于你的线上服务一定更快。

---

## 问题定义与边界

这篇文章讨论的是**面向部署的模型压缩评估**，不是压缩算法论文中的理论压缩比，也不是训练阶段如何得到一个压缩模型。重点不是“某个压缩算法是否先进”，而是“压缩后的模型是否真的适合部署”。

“压缩率”在这里不是一个单指标，而是多个部署指标的联合观察。原因很简单：模型部署的目标不是把一个文件做小，而是在指定硬件、指定后端、指定请求模式下，把资源占用和推理性能做出可接受的改善，同时保持精度不掉出业务门槛。

可以先用一个生活化但不偷换概念的理解方式：像打包行李，箱子体积变小，不代表旅程更轻松。真正影响体验的是体积、重量、是否超重、搬运是否方便。模型压缩也是一样，参数变少，不代表显存一定更低，延迟一定更短。

先把核心指标摆清楚：

| 指标 | 含义 | 是否必须看 |
|---|---|---|
| 参数量 | 模型权重数量 | 是 |
| 存储体积 | 文件落盘大小 | 是 |
| 峰值显存 | 推理时内存峰值 | 是 |
| 延迟 | 单次请求耗时 | 是 |
| 吞吐 | 单位时间处理能力 | 是 |
| 精度 | 任务效果 | 是 |

这些指标为什么都必须看，可以压缩成一句话：**参数量回答“模型抽象上变没变小”，存储回答“文件占没占更少磁盘”，显存回答“设备装不装得下”，延迟和吞吐回答“线上跑得快不快”，精度回答“还有没有业务价值”。**

本文的边界也要明确：

- 不把压缩算法本身优劣作为重点
- 不讨论只在论文环境成立、但未映射到部署路径上的结论
- 不讨论训练成本、蒸馏数据构造、剪枝策略搜索等实现细节
- 重点放在压缩后模型的部署评估方法

如果没有这个边界，文章很容易把“学术上的压缩比”和“工程上的部署收益”混为一谈。那种结论通常对上线决策没有帮助。

---

## 核心机制与推导

理解模型压缩率，最容易出错的地方是把不同层次的指标混在一起。更清晰的方式是分四层看：

1. 参数压缩
2. 存储压缩
3. 显存下降
4. 真实推理收益

可以把它理解成一条链路：

**参数 -> 存储 -> 运行时显存 -> 真实推理性能**

这四层不是等价关系，而是“上游变化可能影响下游，但不保证线性传递”。

### 1. 参数压缩

参数是模型中的可学习权重。参数压缩率定义为：

$$
R_{param} = \frac{N_0}{N_1}
$$

如果一个模型从 `100M` 参数变成 `25M` 参数，那么：

$$
R_{param} = \frac{100}{25} = 4
$$

这只说明权重个数减少了 `4x`。它只对“模型抽象大小”负责，不直接承诺文件大小、显存占用和延迟一定同步下降。

### 2. 存储压缩

模型存储大小受参数格式影响。比如 `FP32` 每个参数常按 `4` 字节存，`FP16` 常按 `2` 字节存，`INT8` 常按 `1` 字节存。直觉上，量化后存储会变小，但实际存储并不只等于“参数数目 × 每参数字节数”。

存储压缩率定义为：

$$
R_{store} = \frac{S_0}{S_1}
$$

这里 $S$ 还会受到这些因素影响：

- 元数据：白话讲就是为了描述压缩格式额外保存的信息
- 对齐：白话讲就是硬件或文件格式为了方便读取，把数据补齐到固定边界
- 编码开销：比如码本、索引、稀疏位置等附加数据

所以 `INT8` 不一定正好是 `FP32` 的 `4x` 存储收益。比如权重量化后，可能还要存 scale、zero-point，或者稀疏结构的索引表。

### 3. 显存下降

峰值显存是推理时 GPU 最忙那一刻占用的显存。它不是只由权重决定，而是：

$$
M \approx M_{weight} + M_{activation} + M_{cache} + M_{workspace}
$$

其中：

- `activation` 是激活值，白话讲就是网络中间层临时算出来的数据
- `cache` 在 LLM 场景里常指 `KV cache`，白话讲就是为了加速后续 token 生成保存的上下文状态
- `workspace` 是临时工作区，白话讲就是库或算子执行时额外借用的一块内存

显存下降率定义为：

$$
R_{mem} = \frac{M_0}{M_1}
$$

关键点在这里：权重缩小，只能保证 $M_{weight}$ 有机会变小；但如果 $M_{activation}$ 或 $M_{cache}$ 才是主要开销，整体 $M$ 的下降就会有限。LLM 长上下文场景尤其典型，很多时候权重不是显存的唯一大头。

### 4. 真实推理收益

真实部署最关心的是延迟和吞吐。

- 延迟是单个请求从进来到出结果花了多久
- 吞吐是单位时间能处理多少请求或多少 token

定义如下：

$$
S_{lat} = \frac{L_0}{L_1},\qquad
S_{thr} = \frac{T_1}{T_0}
$$

这里最容易产生误解。参数少、文件小、显存降，不自动推出延迟一定更短、吞吐一定更高。原因通常有六类：

| 原因 | 影响 |
|---|---|
| 元数据与索引 | 压缩格式附加开销抵消部分收益 |
| 对齐与打包 | 实际落盘或加载大小大于理论值 |
| 激活与 KV cache | 权重变小，但运行时主开销不在权重 |
| kernel 支持不足 | 硬件没有高效执行对应低精度或稀疏算子 |
| 算子回退 | 某些层退回高精度路径，速度收益消失 |
| batch 与并发模式变化 | 小 batch 和大 batch 的收益规律不同 |

这里的 `kernel` 可以理解为“硬件真正执行矩阵乘、卷积等操作的底层实现”。

### 玩具例子：为什么 4x 参数压缩不等于 4x 部署收益

假设：

| 指标 | 压缩前 | 压缩后 |
|---|---:|---:|
| 参数量 | 100M | 25M |
| 权重格式 | FP32 | INT8 + 元数据 |
| 纯权重体积 | 400MB | 25MB |
| 实际存储 | 400MB | 30MB |
| 峰值显存 | 1.8GB | 1.3GB |
| P95 延迟 | 20ms | 18ms |
| 吞吐 | 50 QPS | 56 QPS |
| 任务指标 | 90.0 | 89.6 |

可以算出：

- 参数压缩率：$100 / 25 = 4x$
- 存储压缩率：$400 / 30 \approx 13.3x$
- 显存下降率：$1.8 / 1.3 \approx 1.38x$
- 延迟加速比：$20 / 18 \approx 1.11x$
- 吞吐加速比：$56 / 50 = 1.12x$

这里最重要的信息不是“有 13.3x 存储压缩”，而是“只有约 1.1x 的真实推理收益”。如果你的上线目标是降低延迟，这个压缩就未必值；如果你的上线目标是把模型塞进更小显存的卡，这个压缩可能又是值的。

### 真实工程例子：LLM 量化后为什么不一定明显更快

在 NVIDIA GPU 上部署 LLM 时，把权重从 `FP16` 改成 `INT8` 或 `INT4`，理论上有三个潜在收益：

- 权重文件更小
- 权重显存更低
- 某些矩阵乘可以更快

但工程上常见的结果是：文件变小明显，显存下降也明显，延迟却只改善一点点，甚至几乎不变。原因往往是：

- 长上下文下 `KV cache` 主导显存
- 某些层没有低精度 kernel
- 引擎没有选到最佳 tactic，白话讲就是没有选到最优执行策略
- 小 batch 下 GPU 本来就没跑满，量化后也难以线性提速
- 某些前后处理仍在 CPU 或高精度路径上

所以压缩评估的核心不是看论文给出的压缩倍数，而是把压缩结果放到你的目标硬件、目标 batch、目标引擎上重新测。

最终判定条件可以写成：

$$
\text{压缩有效}
=
(A_1 \ge A_{min})
\land
(R_{store} > 1 \;\text{or}\; R_{mem} > 1 \;\text{or}\; S_{lat} > 1 \;\text{or}\; S_{thr} > 1)
$$

如果没有精度约束，这个结论不完整；如果没有目标硬件实测，这个结论也不完整。

---

## 代码实现

压缩评估的代码重点不是“怎么训练一个压缩模型”，而是“怎么在相同条件下采集指标”。同条件比较的意思是：同一台机器、同一个 batch、同一个输入长度、同一个后端、同一个并发模式，分别跑原模型和压缩模型。

下面给一个最小可运行的 Python 评估骨架。它不依赖特定深度学习框架，先把评估逻辑本身写清楚。真实工程里，你只需要把 `benchmark_model` 和 `evaluate_accuracy` 接到自己的推理代码上。

```python
from dataclasses import dataclass
from statistics import quantiles
from typing import List


@dataclass
class ModelMetrics:
    name: str
    params_million: float
    storage_mb: float
    peak_mem_mb: float
    latencies_ms: List[float]
    throughput_qps: float
    accuracy: float

    @property
    def p95_latency_ms(self) -> float:
        if len(self.latencies_ms) < 2:
            return self.latencies_ms[0]
        # 近似 P95；真实系统可换成更严谨的分位数实现
        return quantiles(self.latencies_ms, n=20, method="inclusive")[18]


def compression_report(baseline: ModelMetrics, compressed: ModelMetrics, min_accuracy: float):
    r_param = baseline.params_million / compressed.params_million
    r_store = baseline.storage_mb / compressed.storage_mb
    r_mem = baseline.peak_mem_mb / compressed.peak_mem_mb
    s_lat = baseline.p95_latency_ms / compressed.p95_latency_ms
    s_thr = compressed.throughput_qps / baseline.throughput_qps

    is_effective = (
        compressed.accuracy >= min_accuracy
        and (r_store > 1 or r_mem > 1 or s_lat > 1 or s_thr > 1)
    )

    return {
        "baseline": baseline.name,
        "compressed": compressed.name,
        "R_param": round(r_param, 3),
        "R_store": round(r_store, 3),
        "R_mem": round(r_mem, 3),
        "S_lat": round(s_lat, 3),
        "S_thr": round(s_thr, 3),
        "acc_before": baseline.accuracy,
        "acc_after": compressed.accuracy,
        "effective": is_effective,
    }


# 玩具数据：模拟同一硬件、同一 batch、同一输入长度下的测量结果
baseline = ModelMetrics(
    name="fp32_baseline",
    params_million=100.0,
    storage_mb=400.0,
    peak_mem_mb=1800.0,
    latencies_ms=[19, 20, 20, 21, 18, 20, 19, 22, 20, 21],
    throughput_qps=50.0,
    accuracy=90.0,
)

compressed = ModelMetrics(
    name="int8_sparse",
    params_million=25.0,
    storage_mb=30.0,
    peak_mem_mb=1300.0,
    latencies_ms=[17, 18, 18, 19, 18, 17, 20, 18, 19, 18],
    throughput_qps=56.0,
    accuracy=89.6,
)

report = compression_report(baseline, compressed, min_accuracy=89.0)

assert report["R_param"] == 4.0
assert report["R_store"] > 10
assert report["R_mem"] > 1
assert report["S_lat"] > 1
assert report["S_thr"] > 1
assert report["effective"] is True

headers = ["模型", "参数量(M)", "存储(MB)", "峰值显存(MB)", "P95延迟(ms)", "吞吐(QPS)", "任务指标"]
rows = [
    [
        baseline.name,
        baseline.params_million,
        baseline.storage_mb,
        baseline.peak_mem_mb,
        round(baseline.p95_latency_ms, 2),
        baseline.throughput_qps,
        baseline.accuracy,
    ],
    [
        compressed.name,
        compressed.params_million,
        compressed.storage_mb,
        compressed.peak_mem_mb,
        round(compressed.p95_latency_ms, 2),
        compressed.throughput_qps,
        compressed.accuracy,
    ],
]

print("| " + " | ".join(headers) + " |")
print("|" + "|".join(["---"] * len(headers)) + "|")
for row in rows:
    print("| " + " | ".join(map(str, row)) + " |")

print(report)
```

这段代码传达的是评估流程，而不是框架技巧。真实工程里，完整流程应当是：

1. 加载原模型和压缩模型
2. 固定设备、batch、输入长度、并发数、精度模式
3. 预热若干次，避免首次加载和缓存抖动污染结果
4. 统计 `P95` 延迟、吞吐、峰值显存
5. 跑验证集得到任务指标
6. 输出对比表，再做结论判断

结果表建议统一成下面这种格式：

| 模型 | 参数量 | 存储 | 峰值显存 | P95 延迟 | 吞吐 | 任务指标 |
|---|---:|---:|---:|---:|---:|---:|
| 原模型 | 100M | 400MB | 1.8GB | 20ms | 50 QPS | 90.0 |
| 压缩模型 | 25M | 30MB | 1.3GB | 18ms | 56 QPS | 89.6 |

如果你用的是 PyTorch、TensorRT 或 ONNX Runtime，真实代码会多出设备同步、显存统计 API、引擎构建参数这些细节，但评估骨架本质上不变。核心要求只有一个：**同条件对比**。

---

## 工程权衡与常见坑

工程里最常见的问题不是“没做压缩”，而是“测错了指标”。压缩评估如果只挑对自己有利的那一项，最后很容易得到一个在部署上没有意义的结论。

先看高频坑点：

| 常见坑 | 为什么会错 | 怎么规避 |
|---|---|---|
| 只看模型文件大小 | 忽略元数据和运行时开销 | 看实际存储和峰值显存 |
| 只看参数量 | 忽略激活和 KV cache | 统计运行时内存峰值 |
| 只看平均延迟 | 忽略尾延迟和 batch 敏感性 | 报 P95 / P99 |
| 只看论文结果 | 环境不一致 | 目标硬件复测 |
| 只看压缩率 | 精度可能掉太多 | 加入任务指标门槛 |

### 1. 只看文件大小

文件变小，只能说明落盘更省空间。对于边缘设备、模型分发、镜像体积，这很重要；但对在线推理延迟，它只是一个相关因素，不是最终答案。

### 2. 只看参数量

很多新手最容易把“参数量下降”直接等同于“显存下降”和“延迟下降”。这在小模型、短输入、简单前向里有时还勉强接近，但到了 LLM、长序列、流式生成场景，这种近似会很快失效。因为 `KV cache` 和激活值常常成为主要开销。

### 3. 只看平均延迟

平均值容易掩盖抖动。线上系统更关心尾延迟，也就是慢请求那部分。`P95` 的意思是 95% 请求都不超过这个时间。一个压缩模型如果平均延迟略好，但 `P95` 更差，线上体验可能反而变坏。

### 4. 只看论文结果

论文环境和你的部署环境通常至少有一项不同：

- GPU 型号不同
- 驱动和 CUDA 版本不同
- 推理引擎不同
- batch 不同
- 输入长度不同
- 数据分布不同

其中任一项变化，都可能让理论收益消失。比如论文中的稀疏剪枝，在你的 GPU 和后端上如果没有对应稀疏 kernel，压缩后的结构就只是“数学上更稀疏”，不是“执行上更快”。

### 5. 只看压缩率，不看精度门槛

压缩不是越狠越好。业务需要的是“足够快且效果不掉出门槛”，不是“压得最多”。如果一个模型存储缩小 `8x`，但任务指标从 `90.0` 掉到 `82.0`，而业务最低要求是 `88.0`，那它对部署没有价值。

评估时建议固定下面这些条件：

- 固定硬件
- 固定 batch
- 固定输入长度
- 固定并发数
- 固定后端版本

这份约束清单的目的只有一个：减少无关变量，让“压缩前后”的差异真正来自压缩本身，而不是环境波动。

真实工程例子可以再明确一点。假设你在一张支持 Tensor Core 的 NVIDIA GPU 上测试一个稀疏模型。论文说能 `2x` 加速，但你的结果只有 `1.02x`。常见原因不是论文错了，而是：

- 你的稀疏模式不是硬件支持的结构化稀疏
- 引擎没有命中对应优化路径
- 模型里真正耗时的层不在被稀疏加速的部分
- 你的 batch 太小，调度和访存开销主导了总延迟

所以工程结论必须绑定环境。脱离目标硬件、目标后端、目标 batch 的压缩收益，参考价值有限。

---

## 替代方案与适用边界

压缩不是唯一优化路径，而且不同压缩手段解决的问题也不同。不能拿同一套判断标准去看所有方案，更不能默认“先剪枝再说”。

先看常见方案的对比：

| 方案 | 主要收益 | 典型风险 | 适用场景 |
|---|---|---|---|
| 剪枝 | 降参数量 | 稀疏不一定被硬件加速 | 结构化稀疏支持好 |
| 量化 | 降存储和显存 | 精度掉点、算子回退 | 硬件支持 INT8/INT4 |
| 低秩分解 | 降计算和参数 | 结构改动较大 | 权重矩阵冗余明显 |
| 蒸馏 | 小模型替代大模型 | 需要重新训练 | 可接受训练成本 |

### 剪枝

剪枝是去掉不重要权重或结构。它最容易带来漂亮的参数压缩率，但如果剪出来的是“非结构化稀疏”，硬件往往很难直接加速。也就是说，参数量可能变少，真实延迟却不一定变短。

### 量化

量化是把高精度数字换成低精度表示。白话讲，就是用更少的字节去存同样的权重。它通常最直接地改善存储和显存，因此部署上最常见。但量化也最依赖硬件和后端支持。如果某些层回退，理论收益会缩水。

### 低秩分解

低秩分解本质上是把一个大矩阵拆成几个更小矩阵，用更少参数近似原始计算。它对某些冗余明显的权重矩阵有效，但会改模型结构，工程接入成本比纯量化高。

### 蒸馏

蒸馏不是把原模型直接压小，而是让一个更小的学生模型学习大模型行为。它的优势是最终得到一个天然更小、结构更规整的模型；缺点是通常需要重新训练，开发周期更长。

这里要强调一个部署判断原则：

**先看硬件支持，再看压缩算法；先看部署目标，再看理论压缩比。**

如果目标是“把模型放进更小显存的卡”，量化通常比剪枝更直接；如果目标是“降低在线延迟”，可能换推理引擎、调 batch、改缓存策略的收益比做复杂剪枝更稳定；如果目标是“在边缘设备离线部署”，文件体积和内存上限可能比吞吐更重要。

再给一个新手常见误区的修正版本。很多人会说：“我要降延迟，所以我要做压缩。”这个逻辑不完整。更准确的顺序应该是：

1. 明确目标是降延迟、降显存、降存储，还是降成本
2. 确认目标硬件支持哪些低精度或稀疏路径
3. 再选择量化、剪枝、蒸馏或其他优化方案
4. 最后在真实部署条件下做统一评估

如果目标硬件不支持对应优化路径，那么相关压缩方案的收益可能主要停留在离线指标上，比如文件更小、参数更少，但线上性能收益很有限。这不是压缩方法一定无效，而是它和当前部署环境不匹配。

---

## 参考资料

这些资料的用途不是证明“某个压缩算法一定最好”，而是帮助你区分理论指标和工程指标，建立正确的评估顺序。阅读时建议按这个顺序走：先看量化和压缩基础，再看推理测量方法，最后看硬件对稀疏和量化类型的支持边界。

1. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://hanlab.mit.edu/projects/deep-compression)
2. [PyTorch Quantization 文档](https://docs.pytorch.org/docs/stable/quantization.html)
3. [PyTorch Quantization Recipe](https://docs.pytorch.org/tutorials/recipes/quantization.html)
4. [NVIDIA TensorRT: Advanced Performance Measurement Techniques](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/measurement-techniques.html)
5. [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.15.1/inference-library/work-quantized-types.html)
6. [NVIDIA TensorRT: Structured Sparsity](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/io-formats-sparsity.html)
