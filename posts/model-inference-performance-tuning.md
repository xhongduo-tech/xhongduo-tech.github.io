## 核心结论

模型推理性能调优，不是单纯把模型“跑快一点”，而是在**不明显影响输出质量**的前提下，重新分配 `算力 / 显存 / 带宽 / 调度` 预算，把 `吞吐`、`TTFT`、`TPOT` 和 `p95/p99` 调到目标区间。

给零基础读者一个直观类比：训练像造车，推理像开车接单。调优不是把车造得更大，而是让同一辆车在同样油量下接更多单、少堵车、少空跑。这里的“油量”对应 GPU 算力、显存容量和显存带宽。

推理阶段最常见的瓶颈，往往不是一次前向计算本身，而是三件事：

1. `KV cache`：也就是“把历史上下文算过的 Key/Value 结果存起来，后续生成时直接复用”的缓存，它会随上下文长度线性增长，先吃掉显存。
2. 带宽压力：也就是“数据搬运速度”，长上下文下，每生成一个 token 都要把大量历史缓存重新读出来，显存读写会变贵。
3. 批处理不充分：也就是“GPU 没吃满”，请求太散、长度差异太大、调度不合理时，GPU 会空转。

先建立全局认知，可以看这张总览表：

| 目标指标 | 主要瓶颈 | 典型手段 | 适用场景 |
|---|---|---|---|
| `TTFT` 首 token 延迟 | 长 prompt 预填充阻塞 | `chunked prefill`、优先级调度、prefix caching | 长输入在线问答 |
| `TPOT` 每 token 时间 | 串行解码、KV 读取成本 | `speculative decoding`、高效 attention kernel | 流式生成 |
| 吞吐量 | 显存装不下更多并发、batch 不满 | `continuous batching`、`PagedAttention`、量化 | 高并发在线服务 |
| `p95/p99` 尾延迟 | 长短请求混跑、排队 | 分层队列、请求分类、预填充分离 | SLA 强约束系统 |
| 单卡成本 | 显存大、带宽吃紧 | FP8/INT4 量化、KV 压缩、离线批处理 | 成本敏感任务 |

结论先说透：**推理优化本质上是系统层面的资源再分配问题，不是只盯某个算子，也不是默认上量化就能解决。**

---

## 问题定义与边界

先定义“推理性能”到底在看什么。它不是只看一条请求多久结束，而是同时看多个指标，因为不同角色关心的不是同一件事。用户最先感知的是首个字什么时候出来，服务端更关心的是一块 GPU 一小时能处理多少 token。

| 术语 | 白话解释 | 关注对象 |
|---|---|---|
| `TTFT`（Time To First Token） | 从请求到第一个输出 token 的时间 | 用户首感知 |
| `TPOT`（Time Per Output Token） | 后续每个输出 token 平均花多久 | 流式体验 |
| 吞吐量 | 单位时间处理多少 token 或多少请求 | 资源利用率 |
| `p95/p99` | 最慢那 5% / 1% 请求的延迟水平 | 高峰体验、SLA |
| 排队时间 | 请求还没上 GPU 前等了多久 | 调度健康度 |
| `KV cache` | 历史 token 的中间结果缓存 | 显存与带宽压力核心来源 |

本文的范围要收清楚。这里说的“性能调优”，重点是**推理系统层**，不是重新训练模型，也不是换模型架构。

| 本文范围 | 非本文范围 |
|---|---|
| 批处理策略 | 重新训练模型 |
| `KV cache` 管理 | 改 Transformer 结构 |
| 量化部署 | 数据清洗与微调 |
| `prefill / decode` 分离 | 提升模型知识能力 |
| 并行解码与调度 | Prompt 工程本身 |

一个常见误区是把“模型更快”理解成“单次前向更快”。实际在线系统里，决定体验的通常是**等待 + 缓存读取 + 批次组织**。所以同样一个聊天接口，用户盯的是 `TTFT`，平台盯的是吞吐和尾延迟，两者必须一起看。

---

## 核心机制与推导

先从注意力机制开始。多头注意力的基本形式可以写成：

$$
\text{MHA} = \text{softmax}(QK^T \otimes mask)V
$$

这里 `Q/K/V` 可以先粗理解成“当前 token 拿去查询历史信息所需的三个表示”。训练时整段序列一起算；推理时是**自回归**，也就是“一次生成一个 token，再接着生成下一个”。如果每次都从头重算全部历史，成本会非常高，所以系统会把历史 token 的 `K/V` 存进 `KV cache`。

这就是推理优化的第一原则：**不是每次都从头算全部历史，而是把历史记在缓存里；但历史越长，缓存越大，读缓存也越贵。**

`KV cache` 大小可近似估算为：

$$
M_{kv} \approx 2 \times L \times H_{kv} \times d \times b \times T
$$

其中：

- `L`：层数
- `H_kv`：KV 头数
- `d`：每个 head 的维度
- `b`：单元素字节数，比如 FP16 是 2 字节
- `T`：上下文长度

“玩具例子”先算一个最小量级。设 `L=32, H_kv=32, d=128, b=2, T=2000`，那么：

- 每层每 token 的 KV 大小约为 `2 × 32 × 128 × 2 = 16384` 字节，约 `16KB`
- 全模型每 token 约 `16KB × 32 = 512KB`
- `2000` 个 token 约 `1GB`

这说明一个初学者经常忽视的事实：**长上下文场景里，先卡住 batch size 的常常不是算不动，而是显存装不下。**

接着看为什么要有几种常见优化：

`PagedAttention`：可以把它理解成“把 KV cache 从一整段连续大内存，改成按页管理的小块内存”。这样做的价值不是改变数学结果，而是减少显存碎片。请求长度动态变化时，如果每个请求都占一块连续大区域，增长和释放会让空洞越来越多；分页后，系统更容易复用零散空间。

`continuous batching`：可以理解成“不是等整批请求凑齐再发车，而是系统每一步都动态把准备好的请求塞进当前批次”。这样 GPU 更容易保持忙碌，减少短请求等长请求、空位不能补的问题。

`chunked prefill`：`prefill` 就是“先把整段输入 prompt 编码并写入 KV cache 的阶段”。长 prompt 会占住 GPU 很久，拖慢其他请求的首 token 输出。把它拆成小块后，可以让长输入少一点“独占时间”。

`speculative decoding`：可以理解成“小模型先草拟几个 token，大模型一次性验证”。如果验证通过率高，就把本来一个个串行生成的过程，变成“先猜一串，再批量确认”，从而降低逐 token 串行成本。

可以把机制串成一个流程图：

```text
prompt prefill
    ↓
write KV cache
    ↓
decode one/few tokens
    ↓
scheduler regroup batch
    ↓
paging / batching / verification
    ↓
emit tokens
```

真实工程例子：一个在线聊天服务同时接“短问答”和“长文总结”。短问答需要低 `TTFT`，长文总结有长 `prefill` 和大 `KV cache`。这时常见组合不是单点技巧，而是 `continuous batching + paged KV cache + chunked prefill`。如果硬件和算子链路支持，再叠加 FP8/INT4 量化或 `speculative decoding`。目标不是让某一条请求绝对最短，而是让**单位 GPU 时间完成更多有效 token，同时压住 p95/p99**。

---

## 代码实现

代码层面最关键的不是某个神奇函数，而是把 `prefill / decode` 分离、请求调度、KV 分配和指标监控连成闭环。

先看一个简化版推理主循环：

```python
from dataclasses import dataclass

@dataclass
class Request:
    req_id: str
    prompt_len: int
    output_len: int
    generated: int = 0

def pack_by_phase(requests):
    prefill = [r for r in requests if r.generated == 0]
    decode = [r for r in requests if r.generated > 0 and r.generated < r.output_len]
    return prefill, decode

def estimate_kv_bytes(layers, kv_heads, head_dim, bytes_per_elem, tokens):
    return 2 * layers * kv_heads * head_dim * bytes_per_elem * tokens

def scheduler_step(requests, max_decode_batch):
    prefill, decode = pack_by_phase(requests)
    decode_batch = decode[:max_decode_batch]
    for r in decode_batch:
        r.generated += 1
    return prefill, decode_batch

# 玩具例子：KV cache 估算
kv = estimate_kv_bytes(layers=32, kv_heads=32, head_dim=128, bytes_per_elem=2, tokens=2000)
assert kv == 1048576000  # 约 1.0 GB

# 玩具例子：prefill / decode 分离
reqs = [
    Request("a", prompt_len=100, output_len=3, generated=0),
    Request("b", prompt_len=20, output_len=4, generated=2),
    Request("c", prompt_len=10, output_len=2, generated=1),
]
prefill, decode_batch = scheduler_step(reqs, max_decode_batch=2)
assert len(prefill) == 1
assert len(decode_batch) == 2
assert reqs[1].generated == 3
assert reqs[2].generated == 2
```

上面代码虽然简化，但已经体现两个核心点：

1. 调度策略决定 GPU 利用率。把 `prefill` 和 `decode` 混成一锅，长输入会拖慢流式输出；分开后才能针对不同阶段做不同打包策略。
2. 缓存分配策略决定显存碎片和搬运成本。真实系统里不会按“每请求一整块连续内存”粗放分配，而会做 page/block 管理。

可以用伪代码概括真实服务路径：

```python
while True:
    requests = scheduler.collect_ready_requests()
    prefill_batch, decode_batch = scheduler.pack_by_phase(requests)
    kv_pages = kv_manager.allocate(prefill_batch, decode_batch)
    prefill_out = model.prefill(prefill_batch, kv_pages)
    decode_out = model.decode(decode_batch, kv_pages)
    metrics.observe(prefill_out, decode_out, kv_pages)
    scheduler.emit(prefill_out, decode_out)
```

再给一张“机制到代码”的对照表：

| 优化点 | 实现位置 | 主要收益 | 风险 |
|---|---|---|---|
| `continuous batching` | scheduler | 提高 GPU 利用率、增吞吐 | 调度逻辑复杂，可能恶化尾延迟 |
| `PagedAttention` / paged KV | kv manager / attention kernel | 减少碎片、提升驻留并发 | 实现复杂，对 kernel 适配有要求 |
| `chunked prefill` | prefill pipeline | 降低长 prompt 阻塞 | 配置不当会增加总调度开销 |
| 量化 | model runner / kernel | 降显存、可能增吞吐 | 算子不支持时收益落空 |
| `speculative decoding` | decode engine | 降低逐 token 串行成本 | 验证拒绝率高时反而变慢 |

---

## 工程权衡与常见坑

性能优化最容易犯的错，是只看平均值。平均延迟下降，不代表用户体验更好。高峰期真正伤人的，往往是 `p95/p99` 和排队时间。

可以继续用开车类比，但要说清机制：把车速调高不等于出行更快。如果路口一直堵、红灯一直等，平均时速再高也没用。推理服务里，红灯就是排队，堵车就是长短请求混跑和显存不够导致的调度拥塞。

常见坑可以直接对照看：

| 坑点 | 为什么会出问题 | 典型表现 | 规避方式 |
|---|---|---|---|
| 只看平均延迟 | 长尾被平均值掩盖 | 高峰期投诉多 | 同时看 `p50/p95/p99` 和排队时间 |
| 量化后算子不支持 | 权重省了，但 kernel 没提速 | 显存降了，吞吐没升 | 先验证硬件、框架、算子链路 |
| `draft model` 过慢 | 草拟本身就不划算 | `speculative decoding` 无收益 | 量 `acceptance rate` 和端到端时延 |
| `KV cache` 碎片化 | 连续内存反复分配释放 | batch 上不去，OOM 提前出现 | 分页式 KV 管理 |
| 长短请求混跑 | 长 prefill 阻塞短 decode | `TTFT` 和 `p99` 恶化 | 分队列、分 phase、分优先级 |

还要注意几个条件依赖：

- 量化不是默认有效。它是否真正提升吞吐，取决于硬件是否支持对应精度、框架是否有高效 kernel、模型算子是否走到了优化路径。
- `speculative decoding` 不是白送加速。只有当草拟模型足够快、接受率足够高时，它才有净收益。
- 长上下文下，瓶颈常从算力切到带宽。此时继续盯 FLOPS 没意义，要看 attention kernel 和 KV 读写路径。

所以正确顺序通常是：**先测 `TTFT / TPOT / 吞吐 / p95/p99 / 显存占用 / acceptance rate`，再选手段。**

---

## 替代方案与适用边界

没有一套“万能组合拳”适合所有场景。不同业务的优先级完全不同，关键是先判断你是在和什么约束打架。

| 场景 | 主要矛盾 | 推荐方案 |
|---|---|---|
| `短请求 / 低并发` | 复杂优化收益小 | 基础 batching、简单缓存、稳定监控 |
| `高并发在线服务` | 吞吐与尾延迟冲突 | `continuous batching + paged KV + 分层调度` |
| `长上下文总结` | 显存与带宽压力大 | `chunked prefill + 高效 attention kernel + KV 管理` |
| `成本敏感离线任务` | 单位 token 成本 | 大 batch、量化、异步队列 |
| `低时延强约束场景` | `TTFT/p99` 最重要 | 小批次、短队列、请求隔离、必要时保守配置 |

这部分可以用另一个新手版比喻理解：同样是开车，市区通勤要少堵车，高速长途要少耗油，拉货场景要看载重，不能用一套标准解决所有路况。推理也是一样。

适用边界可以再说得更具体一些：

- 如果是低并发、短输入、短输出，复杂分页和推测解码未必值得，系统复杂度可能超过收益。
- 如果是超长上下文，总体瓶颈往往先落在 `KV cache` 和带宽，优先级高于纯算力优化。
- 如果业务对少量精度损失可容忍，再考虑 FP8/INT4 量化；如果输出稳定性要求极高，要先做质量回归。
- 如果 SLA 极强，比如客服首字必须很快，就应优先保证 `TTFT` 和 `p99`，不应为了平均吞吐把批次堆得过满。

最终判断标准不是“用了多少热门技术”，而是：**这套组合是否在你的流量分布、上下文长度分布和硬件条件下，稳定改善目标指标。**

---

## 参考资料

下表先说明每份资料主要解决什么问题，避免只堆链接：

| 来源 | 关注点 | 对应章节 | 推荐阅读顺序 |
|---|---|---|---|
| vLLM README | `PagedAttention`、`continuous batching`、量化、推测解码的工程入口 | 核心机制、代码实现 | 1 |
| PagedAttention 论文 | `KV cache` 碎片、分页管理、吞吐提升原因 | 核心机制 | 2 |
| TensorRT Transformers 文档 | KV cache 张量形状、更新方式、工程接口 | 代码实现 | 3 |
| Speculative Decoding 论文 | draft-verify 原理、何时能加速 | 核心机制、工程权衡 | 4 |
| FlashAttention 论文 | 为什么带宽会成为瓶颈、IO-aware attention | 核心机制、适用边界 | 5 |

1. [vLLM GitHub README](https://github.com/vllm-project/vllm)
2. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
3. [Working with Transformers - NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-transformers.html)
4. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
5. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
