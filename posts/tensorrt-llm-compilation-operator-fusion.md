## 核心结论

TensorRT-LLM 的“编译与算子融合”，本质上是在**推理阶段**重写 GPU 的执行方式，而不是改写模型本身的预测逻辑。编译，白话说就是把原本运行时才临时决定的执行图，提前固化成更适合 NVIDIA GPU 的 engine；算子融合，白话说就是把原来需要多次调用 GPU 的小步骤，合并成更少、更大的 kernel 一次执行。

它主要改动的是 Transformer 前向链路中的三类成本：

1. `kernel launch` 成本，也就是 CPU/驱动发起一次 GPU 任务的固定开销。
2. 中间张量回写显存的成本，也就是算完一步后把半成品搬回 HBM 再读回来。
3. padding 带来的无效计算，也就是为了对齐 batch 长度而算了很多不存在的 token。

因此，TensorRT-LLM 最直接影响的指标是**延迟、吞吐、显存占用**，而不是“模型更聪明”或“输出更准”。

单层时延可以粗略写成：

$$
T_{layer} \approx \sum_i (t_{launch,i} + t_{compute,i} + t_{HBM,i})
$$

融合后更接近：

$$
T'_{layer} \approx t_{launch,fused} + t_{compute,fused} + t_{HBM,fused}
$$

这里的意思很直接：如果原来一层里要启动很多 kernel、反复读写显存，那么就算每一步计算量没变，总时延也会偏大；如果把多步合并，固定开销和内存往返就会下降。

对零基础读者，可以把 eager 推理理解成“每做一步就叫一次工人，还要把中间半成品搬回仓库”；TensorRT-LLM 则是“先把工序编排好，让一个工人连续做完更多步骤，少跑腿”。

---

## 问题定义与边界

先把边界说清楚。TensorRT-LLM 优化的是**LLM 在 GPU 上的推理执行链路**，不是整个应用的所有阶段。

| 范围 | TensorRT-LLM 是否直接优化 |
|---|---|
| tokenizer / detokenizer | 否 |
| 自回归解码顺序 | 否 |
| Transformer 前向计算 | 是 |
| kernel launch 与中间张量搬运 | 是 |
| 长上下文 attention 成本 | 只能部分缓解，不能改变复杂度本质 |

这里的“自回归解码顺序”指的是模型一次生成一个 token，后一个 token 依赖前一个 token 的结果。这个顺序约束没有被取消，所以 TensorRT-LLM 不是把串行生成 magically 变成完全并行。

它最适合解决的问题是：

- 在线服务中的低延迟和高吞吐需求
- 长上下文带来的显存压力
- 多请求并发时的调度效率
- 重复前缀带来的重复计算
- 量化后如何把带宽优势真正转化成吞吐

不太适合的场景也很明确：

- 本地单卡调试
- `batch=1` 且 prompt 很短
- QPS 很低，GPU 大量空闲
- 请求长度分布极乱，几乎没有可复用前缀

玩具例子：如果只有一个请求，prompt 就 16 个 token，生成 8 个 token，那么多数时间本来就消耗在非常少量的算子上。此时做复杂编译，收益可能小于 build 成本和维护复杂度。

真实工程例子：一个客服聊天服务，system prompt 固定，用户问题长度在 200 到 2000 tokens 之间波动，且同一时刻有很多并发请求。这里优化前向链路、减少 padding、复用共享前缀的价值会被成倍放大。

---

## 核心机制与推导

TensorRT-LLM 的提速，通常可以拆成三层：**图编译、算子融合、运行时调度优化**。

图编译的作用，是把模型结构映射为 TensorRT engine。engine 可以理解成已经为目标 GPU、目标 shape 范围和目标精度做过选择与优化的可执行计划。这样运行时不需要每次都临时决定“下一步调用哪个实现”。

算子融合的作用，是把多个高频小步骤拼成更少的 kernel。例如 attention 附近常见的步骤包括 QKV 处理、RoPE、bias、量化/反量化、mask、softmax、KV cache 读写等。如果这些步骤分散执行，就会产生很多 launch 和很多中间张量。融合后，这些中间结果会尽量停留在寄存器或共享内存，而不是频繁落到 HBM。

运行时调度优化的作用，是减少“明明 GPU 很强，但实际吃不满”的问题，例如：

- packed input：把不同长度样本紧凑排布，减少 padding
- paged context attention：把 KV cache 分页管理，降低长上下文的内存压力
- KV cache reuse：对相同前缀复用已经算过的 KV
- in-flight batching：让上下文阶段和生成阶段更高效地穿插调度

可以用一个表格看收益来源：

| 项目 | 优化前 | 优化后 | 作用 |
|---|---|---|---|
| kernel launch | 多次 | 更少 | 降低调度开销 |
| 中间张量 | 多次回写 | 尽量在片上完成 | 降低 HBM 访问 |
| padding | 高 | packed 后接近 0 | 减少无效计算 |
| KV cache | 逐 token 追加 | paged / reuse | 降低显存压力与重复计算 |

### 1. 为什么“少 launch”有价值

很多初学者直觉上只盯着 FLOPs，但真实推理不只看乘加次数。一次 kernel launch 有固定成本；如果你把一个 layer 拆成十几个小 kernel，哪怕每个都很快，累计起来也会显著拖慢端到端延迟。尤其在 batch 小、每步生成一个 token 的解码阶段，这种固定成本更明显。

### 2. 为什么“少 HBM 往返”有价值

HBM 指 GPU 的高带宽显存。它很快，但比寄存器和共享内存慢得多。很多 LLM 推理并不是纯算力受限，而是**内存带宽受限**，白话说就是“不是算不过来，而是数据搬不过来”。算子融合减少中间张量回写，本质上是在减少搬运。

### 3. packed input 为什么经常有效

假设一个 batch 里有 3 个请求，长度分别是 `64`、`128`、`256`。如果按传统 padded 方式对齐到 `256`，总共会算：

$$
3 \times 256 = 768
$$

但真实需要的只有：

$$
64 + 128 + 256 = 448
$$

浪费的 token 数是：

$$
W = \sum (L_{max} - L_i) = 320
$$

浪费比例是 `320 / 768 ≈ 41.7%`。这就是为什么长度分布越分散，packed mode 越值得做。

### 4. KV cache 为什么是显存大户

KV cache，白话说就是“把前面 token 的 attention 中间结果存起来，后面直接复用，不再重算”。它能省计算，但会吃显存。粗略估算：

$$
M_{kv} \approx 2 \cdot B \cdot L \cdot N_{kv} \cdot d_{head} \cdot b
$$

其中：

- $B$ 是 batch size
- $L$ 是序列长度
- $N_{kv}$ 是 KV heads 数
- $d_{head}$ 是每个 head 的维度
- $b$ 是每个元素字节数

如果 `B=1, L=2048, N_kv=8, d_head=128, b=2`（FP16），那么单层 KV cache 大约是：

$$
2 \cdot 1 \cdot 2048 \cdot 8 \cdot 128 \cdot 2 \approx 8 \text{ MiB}
$$

32 层就是约 `256 MiB`。这还只是 batch=1。长上下文和并发一上来，KV cache 很快就成为主要显存负担。所以 paged attention、block reuse、prefix reuse 都不是可有可无的小优化，而是系统能不能稳住的关键。

---

## 代码实现

代码层面最容易误解的一点是：TensorRT-LLM 的核心工作不是“手写一个新模型”，而是**把现有模型导入、配置、编译并交给特定 runtime 执行**。真正决定性能的，通常不是最上层 `generate()` 调用，而是 build 配置、profile、plugin 和 runtime 调度策略。

一个简化后的工作流可以写成下面这样：

```python
from dataclasses import dataclass

@dataclass
class BuildConfig:
    max_seq_len: int
    use_packed_input: bool
    use_paged_context_attention: bool
    enable_kv_cache_reuse: bool
    quantization: str

def estimate_padding_waste(lengths):
    lmax = max(lengths)
    total_padded = len(lengths) * lmax
    total_real = sum(lengths)
    waste = total_padded - total_real
    return waste, waste / total_padded

def estimate_kv_cache_bytes(batch, seq_len, n_kv, d_head, bytes_per_elem, n_layers):
    per_layer = 2 * batch * seq_len * n_kv * d_head * bytes_per_elem
    total = per_layer * n_layers
    return per_layer, total

config = BuildConfig(
    max_seq_len=2048,
    use_packed_input=True,
    use_paged_context_attention=True,
    enable_kv_cache_reuse=True,
    quantization="fp16",
)

lengths = [64, 128, 256]
waste, waste_ratio = estimate_padding_waste(lengths)
assert waste == 320
assert round(waste_ratio, 4) == round(320 / 768, 4)

per_layer, total = estimate_kv_cache_bytes(
    batch=1,
    seq_len=2048,
    n_kv=8,
    d_head=128,
    bytes_per_elem=2,
    n_layers=32,
)
assert per_layer == 8 * 1024 * 1024
assert total == 256 * 1024 * 1024

print(config)
print("padding waste tokens:", waste)
print("per-layer kv bytes:", per_layer)
print("total kv bytes:", total)
```

这段代码不是 TensorRT-LLM 官方 API 的直接替身，而是一个可运行的“成本估算玩具模型”。它帮助你在真正 build engine 前，先理解两个最重要的问题：

1. 你的长度分布里 padding 浪费有多大。
2. 你的 KV cache 会不会先把显存打满。

如果继续往真实配置靠近，通常会关心这些项：

| 配置项 | 作用 | 风险 |
|---|---|---|
| `max_seq_len` | 决定支持的上下文上限 | 设太大增大 engine 和显存 |
| optimization profile | 给动态 shape 设范围和最优点 | 范围过宽会削弱优化 |
| packed input | 去掉 padding 浪费 | 输入整理复杂度上升 |
| paged context attention | 降低长上下文内存压力 | 依赖支持路径 |
| KV cache reuse | 复用共享前缀计算结果 | 只有前缀重合时收益大 |
| quantization | 降低显存和带宽 | 精度与兼容性要权衡 |

真实工程例子：假设你在 H100 上部署一个 13B 聊天模型，用户共享同一段 system prompt，并且请求长度集中在 `512` 到 `4096`。这时合理做法通常不是无脑把 `max_seq_len` 拉到一个极大值，而是根据输入分布设计 profile，并开启 packed input、paged context attention 和 KV cache reuse。这样收益通常体现在首 token 延迟更低、长尾更稳、同卡并发更高。

---

## 工程权衡与常见坑

TensorRT-LLM 不是“装上就自动加速”的按钮。它的收益高度依赖负载形态。

第一类坑是**工作负载不匹配**。如果 `batch=1`、prompt 很短、QPS 很低，那么 launch 和显存搬运本来就不是主要瓶颈。此时编译、profile 调优、engine 管理这些额外复杂度，可能换不来明显收益。说得直白一点：为了跑一句“你好”，没必要修一条高速公路。

第二类坑是**profile 设得过大**。TensorRT 的动态 shape 依赖 optimization profile，也就是提前声明输入尺寸范围和优化目标点。如果你的 `max_seq_len` 远大于真实分布，或者把 batch 范围设得过宽，engine 体积、build 时间和显存保留都会上涨，实际运行还不一定更快。

第三类坑是**请求分布很散**。packed input 的收益来自减少 padding；KV cache reuse 的收益来自共享前缀。如果每个请求长度差异极大，且几乎没有共同前缀，那么这两类优化的边际收益会明显下降。

第四类坑是**支持路径限制**。很多读者把“TensorRT-LLM 支持某模型”理解成“所有精度、所有插件、所有融合路径都支持”。这是两回事。模型能跑，不代表它能走到最佳 plugin 和最佳 kernel。只要某些结构或精度不在支持矩阵里，就可能退回更通用的实现，性能会打折。

第五类坑是**误以为它能消灭 attention 的复杂度问题**。对标准 full attention 而言，长上下文预填充阶段的核心成本并没有从根本上脱离平方级趋势。融合、paged、chunked prefill 主要是在优化常数项、访存模式和调度方式，而不是把复杂度定律改掉。

| 坑 | 表现 | 规避方式 |
|---|---|---|
| `batch=1` 且 prompt 很短 | 提速不明显 | 只在高频服务端场景投入 |
| 输入长度分布很散 | packed / reuse 收益下降 | 分桶、分 profile |
| `max_seq_len` 过大 | engine 变大、build 变慢 | 按真实分布设定 |
| 模型/精度不支持 | 回退通用 kernel | 先确认支持矩阵 |
| 长上下文 attention 成瓶颈 | 只能优化常数项 | 接受复杂度约束 |

---

## 替代方案与适用边界

TensorRT-LLM 不是唯一答案。是否采用它，核心看三件事：**是否需要极致性能、是否运行在 NVIDIA GPU、是否能接受更高部署复杂度**。

| 方案 | 优势 | 不足 | 适合场景 |
|---|---|---|---|
| TensorRT-LLM | 高性能、强融合、对 LLM 特化 | 配置复杂、支持路径受限 | NVIDIA GPU 在线服务 |
| PyTorch eager | 开发简单、调试方便 | launch 多、性能一般 | 原型验证、研究实验 |
| vLLM | 调度强、服务化成熟、上手快 | 极限 kernel 级性能未必最高 | 通用在线推理 |
| 通用 ONNX/TensorRT | 部署链路统一 | 对 LLM 专门优化较少 | 多模型统一部署 |

对初学者，一个实用判断标准是：

- 如果你在做离线实验、功能验证、prompt 调试，优先 PyTorch eager。
- 如果你需要快速起一个可用的在线服务，且希望少踩工程坑，vLLM 往往更省事。
- 如果你明确运行在 NVIDIA GPU 上，流量大、上下文长、GPU 成本敏感，而且愿意投入 build/profile/plugin 调优，那么 TensorRT-LLM 才最值得。

换句话说，TensorRT-LLM 更像“追求单位 GPU 产出最大化”的方案，而不是“所有人默认先用”的方案。它最划算的时间点，通常不是模型刚能跑起来的时候，而是服务已经进入性能和成本优化阶段的时候。

---

## 参考资料

1. [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)
2. [TensorRT-LLM Architecture Overview](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html)
3. [TensorRT Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html)
4. [TensorRT-LLM GPT Attention / Packed Mode 文档](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/gpt-attention.md)
5. [TensorRT-LLM KV Cache Reuse 文档](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/kv-cache-reuse.md)
