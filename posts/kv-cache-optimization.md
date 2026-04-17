## 核心结论

KV Cache 优化的本质，是把推理阶段已经算过的 Key 和 Value 缓存下来，并进一步把这份缓存做“瘦身”。Key/Value 可以理解为注意力机制在每个位置留下的检索线索和内容载荷；模型生成下一个 token 时，不必重算整段历史，只需要读取历史缓存并补上最新位置即可。

如果不做优化，KV Cache 的显存占用会随着上下文长度线性增长。对自回归生成来说，自回归的意思是“每次只生成一个新 token，再把它接回上下文继续生成”，所以缓存会越来越长，最后常常不是算力先耗尽，而是显存容量和显存带宽先成为瓶颈。

最核心的结论只有两条：

1. 缓存体积与 `dtype_bytes` 近似成线性关系，把 FP16 改成 INT8，缓存约减半；改成 INT4，缓存约变成四分之一。
2. 量化、压缩、分页分配、重要性剪枝可以叠加使用，其中“量化 + 分页”是当前工程里最常见、收益最稳定的一组组合。

KV Cache 大小常用近似公式是：

$$
\#bytes \approx 2 \times N_{layers} \times H_{kv} \times d_{head} \times (L_{prompt} + L_{gen}) \times dtype\_bytes
$$

其中：

- $N_{layers}$ 是 Transformer 层数
- $H_{kv}$ 是参与缓存的 KV 头数。标准多头注意力里通常等于注意力头数；GQA/MQA 中会小于查询头数
- $d_{head}$ 是每个 KV 头的维度
- $L_{prompt}$ 是输入长度
- $L_{gen}$ 是已生成长度
- 前面的 `2` 表示要同时存 Key 和 Value

很多资料会把 $H_{kv} \times d_{head}$ 合并写成总维度 $d$，于是得到更简洁的版本：

$$
\#bytes \approx 2 \times d \times (L_{prompt} + L_{gen}) \times dtype\_bytes \times N_{layers}
$$

这两个公式表达的是同一件事，只是拆分粒度不同。对新手来说，先记住一句话即可：**KV Cache 的体积，正比于层数、上下文长度、每层缓存维度，以及每个元素的字节数。**

一个新手版本的玩具例子最直观。假设某层只有一组总维度为 2048 的缓存，历史总长度是 640 个 token，那么：

- FP16：`2 × 1 × 2048 × 640 × 2 = 5,242,880 bytes ≈ 5.24 MB`
- INT8：约 `2.62 MB`
- INT4：约 `1.31 MB`

同一层里，单纯把数值精度从 FP16 降到 INT8/INT4，就能直接换来 2 倍到 4 倍的缓存空间。对显存紧张的推理服务，这通常比“继续堆更大显卡”更直接。

再把视角拉回真实模型。假设一个 32 层模型，每层都像上面这样缓存，那么总缓存大约是：

- FP16：`5.24 MB × 32 ≈ 167.7 MB`
- INT8：`83.9 MB`
- INT4：`41.9 MB`

这个数字还只是单请求、单时刻的近似值。在线服务往往同时跑很多请求，因此 KV Cache 很容易成为吞吐上限的决定因素。

---

## 问题定义与边界

问题定义很明确：在 Transformer 推理中，KV Cache 会随着上下文长度增加而线性膨胀，而模型每生成一个新 token，都要读取前面所有位置的缓存参与注意力计算。结果是两个瓶颈同时出现：

- 显存容量不够，直接 OOM
- 显存带宽压力过大，吞吐下降、时延上升

这里要先划清边界。KV Cache 优化解决的是“推理阶段历史上下文存储与读取”的问题，不是模型参数压缩本身。模型权重是模型本体，KV Cache 是推理过程中的中间状态，两者都占显存，但优化手段和收益位置不同。

下面这张表给出一个近似直觉。为了便于理解，假设缓存元素个数不变，只比较不同数据类型的空间变化：

| 数据类型 | `dtype_bytes` | 相对 FP16 压缩倍数 | 同等长度下缓存体积 |
|---|---:|---:|---:|
| FP16 | 2 | 1x | 100% |
| BF16 | 2 | 1x | 100% |
| FP8 | 1 | 2x | 50% |
| INT8 | 1 | 2x | 50% |
| INT4 | 0.5 | 4x | 25% |

再看长度增长的影响。仍用 `H=1, d=2048` 这个玩具配置：

| `L_prompt` | `L_gen` | 总长度 | FP16 | INT8 | INT4 |
|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 640 | 5.24 MB | 2.62 MB | 1.31 MB |
| 2048 | 512 | 2560 | 20.97 MB | 10.49 MB | 5.24 MB |
| 8192 | 1024 | 9216 | 75.50 MB | 37.75 MB | 18.87 MB |

这个表说明一个关键事实：长度翻 4 倍，缓存也几乎翻 4 倍；数据类型减半，缓存也近似减半。这种线性关系决定了 KV Cache 优化非常适合长上下文推理。

但真正的工程压力不只来自“占多少显存”，还来自“每一步要读多少显存”。生成第 $t$ 个 token 时，当前 Query 需要和前 $t-1$ 个位置的 Key 计算相关性，再用这些相关性去聚合前 $t-1$ 个位置的 Value。因此：

- 缓存越长，每一步需要读取的历史越多
- 读取量越大，对显存带宽越敏感
- 带宽一旦成为瓶颈，GPU 算力可能处于“等数据”的状态

这就是为什么有时 GPU 利用率看上去不低，但系统整体吞吐仍然上不去。问题不一定在算子本身，而在缓存读取链路。

真实工程例子里，70B 量级模型在长上下文条件下，原始 FP16 KV Cache 很容易吃掉几十 GB 显存。即使模型权重已经分片到多卡，单卡上也仍可能因为缓存峰值而没有余量给激活、调度缓冲区、通信区和运行时分配器。启用 INT8 KV Cache 再配合分页后，往往能明显留出额外 20% 到 30% 的空间，让系统不至于在高峰期因上下文波动直接 OOM。

但边界也很明确：

- 量化不会免费带来收益，它会引入量化误差和额外的量化/反量化开销
- 短 prompt、短生成场景下，缓存本来就不大，量化收益可能抵不过新增开销
- 对数值敏感的任务，例如部分高精度问答、代码生成、医学和法律场景，INT4 风险通常高于 INT8
- 如果模型已经使用 GQA/MQA，KV Cache 天然更小，此时进一步量化的边际收益需要单独评估
- 如果主要瓶颈不是显存而是采样、网络或请求排队，KV Cache 优化不会直接解决系统全局瓶颈

可以把这件事拆成一句更准确的话：**KV Cache 优化是推理系统里的“内存与带宽优化”，不是模型能力优化。**

---

## 核心机制与推导

Transformer 的注意力计算要用到当前 Query 与历史所有 Key 的匹配分数，再用这些分数对历史所有 Value 做加权求和。注意力可以理解为“当前 token 去历史里查信息并汇总”。如果每次生成新 token 都把历史所有位置的 K/V 重新前向一遍，成本会非常高，因此推理时通常采取增量方式：

1. 首次处理 prompt，得到每层所有位置的 K/V
2. 把这些 K/V 存进缓存
3. 生成新 token 时，只计算这个新位置对应的 K/V
4. 把新位置追加进缓存
5. 下一步直接复用整段历史缓存

这就是 KV Cache 的基本机制。

为了让新手更容易建立直觉，先看不使用缓存和使用缓存的区别：

| 方案 | 第 1 步生成前 | 第 2 步生成前 | 第 100 步生成前 |
|---|---|---|---|
| 不用 KV Cache | 重算全部 prompt | 重算 prompt + 第 1 个生成 token | 重算 prompt + 前 99 个生成 token |
| 使用 KV Cache | 计算并保存全部 prompt 的 K/V | 只计算第 1 个新 token 的 K/V | 只计算第 99 个新 token 的 K/V |

所以，KV Cache 省掉的首先是**重复计算**；而量化、省页、剪枝做的，是在“已经决定缓存”的前提下，继续降低**存储成本和读取成本**。

为什么量化能直接省显存？因为缓存本质上就是大张量，张量占用近似等于“元素个数 × 每个元素字节数”。在元素个数不变时，`dtype_bytes` 从 2 变成 1，再变成 0.5，显存就线性下降。

把公式展开看更直观。若按“每层总维度 $d$”来写：

$$
\#bytes = 2 \times d \times L \times dtype\_bytes
$$

其中 $L = L_{prompt} + L_{gen}$。

若按“层数、头数、每头维度”来写：

$$
\#bytes = 2 \times N_{layers} \times H_{kv} \times d_{head} \times L \times dtype\_bytes
$$

对单次请求而言，变量主要只有三类：

- 总长度 $L$
- 数据类型字节数 `dtype_bytes`
- 模型结构参数 `N_layers`、`H_kv`、`d_head`

所以从 FP16 到 INT8 的压缩因子是：

$$
\frac{1}{2}
$$

从 FP16 到 INT4 的压缩因子是：

$$
\frac{0.5}{2} = \frac{1}{4}
$$

继续用玩具例子推导。令 $H=1, d=2048, L=640$：

$$
\#bytes_{FP16} = 2 \times 1 \times 2048 \times 640 \times 2 = 5{,}242{,}880
$$

$$
\#bytes_{INT8} = 2 \times 1 \times 2048 \times 640 \times 1 = 2{,}621{,}440
$$

$$
\#bytes_{INT4} = 2 \times 1 \times 2048 \times 640 \times 0.5 = 1{,}310{,}720
$$

物理意义很简单：你没有减少缓存位置数，也没有减少每个位置的逻辑信息结构，只是把“每个数占几字节”压小了。

但这里有一个容易被忽略的点：**量化不是只改“存储格式”，还会改变“读写路径”。** 因为缓存写入时要量化，读取时要反量化，或者在内核里边解码边参与计算。所以工程收益通常来自两个来源：

| 收益来源 | 直接效果 | 间接效果 |
|---|---|---|
| 缓存更小 | 单请求显存下降 | 能容纳更长上下文或更多并发 |
| 读写数据更少 | 带宽压力下降 | 每步注意力读取更轻，吞吐更稳定 |

但代价也来自两个方向：

| 代价来源 | 体现方式 |
|---|---|
| 数值误差 | 注意力分数和聚合结果会有偏差 |
| 编解码开销 | 写缓存和读缓存都多了一步 |

量化不是唯一手段。工程上通常会和以下机制配合：

| 机制 | 作用 | 解决的问题 |
|---|---|---|
| KV 量化 | 降低每个元素字节数 | 显存容量、部分带宽压力 |
| PagedAttention | 按页管理缓存 | 避免大块连续显存碎片，便于回收 |
| Head/Layer 剪枝 | 删除低重要性部分缓存 | 进一步压缩体积 |
| 低秩/压缩表示 | 用更小表示近似原缓存 | 用额外计算换存储 |
| 滑动窗口/局部注意力 | 只保留部分历史 | 限制超长上下文的读取成本 |

PagedAttention 可以理解为“把大块缓存拆成很多固定页，再按需映射”。它不直接减少每个元素的体积，但能减少内存碎片，并支持动态分配与回收。和量化结合后，峰值显存更容易控制。简化流程如下：

| 阶段 | 动作 | 结果 |
|---|---|---|
| 写入缓存 | 生成新 K/V 后先量化 | 单页容量变大 |
| 分配页面 | 按 token 块映射到页 | 避免一次性申请超大连续内存 |
| 读取缓存 | 注意力计算前按需反量化或在线解码 | 保持兼容原注意力逻辑 |
| 释放页面 | 会话结束或旧页淘汰时回收 | 降低长期驻留显存 |

可以把它理解成磁盘文件系统里的“分页和索引”思想，只不过这里管理的是 GPU 上的 KV 张量页，而不是磁盘块。这个类比只用于帮助建立直觉，真正的实现远比文件系统更强调并行访问和带宽局部性。

---

## 代码实现

下面先给一个**可直接运行**的 Python 玩具实现，只演示三件事：

1. 估算 KV Cache 体积
2. 对浮点数组做对称 INT8 量化与反量化
3. 用一个最小化注意力例子说明“量化前后结果会接近，但不完全相同”

对称量化可以理解为“用一个缩放系数把浮点数映射到整数区间，再在读取时乘回来”。

```python
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import List, Sequence, Tuple


def kv_cache_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    prompt_len: int,
    gen_len: int,
    dtype_bytes: float,
) -> float:
    total_tokens = prompt_len + gen_len
    return 2 * num_layers * num_kv_heads * head_dim * total_tokens * dtype_bytes


def format_mb(num_bytes: float) -> str:
    return f"{num_bytes / 1_000_000:.2f} MB"


@dataclass
class QuantizedInt8:
    values: List[int]
    scale: float


def quantize_int8(xs: Sequence[float]) -> QuantizedInt8:
    if not xs:
        return QuantizedInt8(values=[], scale=1.0)

    max_abs = max(abs(x) for x in xs)
    if max_abs == 0:
        return QuantizedInt8(values=[0 for _ in xs], scale=1.0)

    scale = max_abs / 127.0
    q = []
    for x in xs:
        v = int(round(x / scale))
        v = max(-127, min(127, v))
        q.append(v)

    return QuantizedInt8(values=q, scale=scale)


def dequantize_int8(q: QuantizedInt8) -> List[float]:
    return [v * q.scale for v in q.values]


def dot(xs: Sequence[float], ys: Sequence[float]) -> float:
    assert len(xs) == len(ys)
    return sum(x * y for x, y in zip(xs, ys))


def softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    exps = [exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def attention_scores(query: Sequence[float], keys: Sequence[Sequence[float]]) -> List[float]:
    scale = sqrt(len(query))
    return [dot(query, k) / scale for k in keys]


def attention_output(
    query: Sequence[float],
    keys: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
) -> Tuple[List[float], List[float]]:
    scores = attention_scores(query, keys)
    probs = softmax(scores)

    out_dim = len(values[0])
    out = [0.0] * out_dim
    for p, v in zip(probs, values):
        for i in range(out_dim):
            out[i] += p * v[i]
    return probs, out


def max_abs_error(xs: Sequence[float], ys: Sequence[float]) -> float:
    assert len(xs) == len(ys)
    return max(abs(x - y) for x, y in zip(xs, ys))


def main() -> None:
    fp16_bytes = kv_cache_bytes(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2048,
        prompt_len=512,
        gen_len=128,
        dtype_bytes=2,
    )
    int8_bytes = kv_cache_bytes(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2048,
        prompt_len=512,
        gen_len=128,
        dtype_bytes=1,
    )
    int4_bytes = kv_cache_bytes(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2048,
        prompt_len=512,
        gen_len=128,
        dtype_bytes=0.5,
    )

    assert fp16_bytes == 5_242_880
    assert int8_bytes == fp16_bytes / 2
    assert int4_bytes == fp16_bytes / 4

    xs = [0.2, -0.7, 1.3, -1.1, 0.0]
    qx = quantize_int8(xs)
    restored = dequantize_int8(qx)
    err = max_abs_error(xs, restored)

    assert err < 0.01

    query = [0.8, -0.1, 0.3, 0.5]
    keys = [
        [0.7, -0.2, 0.4, 0.3],
        [0.1, 0.9, -0.3, 0.2],
        [0.6, -0.1, 0.2, 0.8],
    ]
    values = [
        [1.0, 0.0, 0.5],
        [0.2, 0.9, 0.1],
        [0.8, 0.3, 0.7],
    ]

    probs_fp, out_fp = attention_output(query, keys, values)

    q_keys = [quantize_int8(k) for k in keys]
    q_values = [quantize_int8(v) for v in values]
    dq_keys = [dequantize_int8(k) for k in q_keys]
    dq_values = [dequantize_int8(v) for v in q_values]

    probs_q, out_q = attention_output(query, dq_keys, dq_values)

    print("KV cache size")
    print("  FP16:", format_mb(fp16_bytes))
    print("  INT8:", format_mb(int8_bytes))
    print("  INT4:", format_mb(int4_bytes))
    print()

    print("Quantization demo")
    print("  original:", xs)
    print("  quantized:", qx.values)
    print("  restored:", [round(v, 4) for v in restored])
    print("  max_abs_error:", round(err, 6))
    print()

    print("Attention demo")
    print("  probs_fp :", [round(x, 6) for x in probs_fp])
    print("  probs_q  :", [round(x, 6) for x in probs_q])
    print("  out_fp   :", [round(x, 6) for x in out_fp])
    print("  out_q    :", [round(x, 6) for x in out_q])
    print("  out_error:", round(max_abs_error(out_fp, out_q), 6))


if __name__ == "__main__":
    main()
```

如果运行成功，你会看到三类输出：

- 缓存体积估算：验证 FP16、INT8、INT4 的线性关系
- 量化恢复误差：说明反量化结果接近原值，但不是完全一致
- 注意力输出误差：说明量化后的 K/V 会把最终注意力输出推偏一点点

这比只展示“算字节数”的例子更接近真实部署中的核心矛盾：**空间变小了，但数值会有偏差。**

如果把它放进真实生成循环，伪代码通常长这样：

```python
dtype_bytes = 1.0  # INT8
kv_pages = []

for step in range(max_new_tokens):
    q, k_new, v_new = model.forward_one_token(input_token, kv_pages)

    k_q = quantize_int8(k_new)
    v_q = quantize_int8(v_new)

    kv_pages.append({
        "k": k_q.values,
        "v": v_q.values,
        "k_scale": k_q.scale,
        "v_scale": v_q.scale,
        "dtype_bytes": dtype_bytes,
    })

    k_hist = [
        dequantize_int8(QuantizedInt8(page["k"], page["k_scale"]))
        for page in kv_pages
    ]
    v_hist = [
        dequantize_int8(QuantizedInt8(page["v"], page["v_scale"]))
        for page in kv_pages
    ]

    logits = attention(q, k_hist, v_hist)
    input_token = sample_next_token(logits)
```

真实框架当然不会在 Python 列表层面这样做，而是会在 CUDA kernel、页表管理器、allocator 和 runtime schedule 上做更细的优化。但思路一致：缓存写入路径做压缩，读取路径做兼容。

下面这张表可以把“量化前后 API 心智模型”看得更清楚：

| 场景 | 原始缓存 API | 量化后缓存 API | 额外信息 |
|---|---|---|---|
| 写入 | `append(k, v)` | `append(qk, qv, scale_k, scale_v)` | 需要保存缩放参数 |
| 读取 | `read()` 返回浮点张量 | `read()` 后反量化或内核内解码 | 读取路径更复杂 |
| 统计 | 只关心 token 数 | 还要关心页大小、量化粒度 | 调度器更重要 |
| 回收 | 释放连续块 | 回收页或块 | 更适合动态请求 |

真实工程例子是 vLLM、LMDeploy 这类推理系统。它们通常不会把量化当作孤立优化，而是与分页缓存、请求并发调度、连续批处理一起设计。服务高峰时，不同请求的 prompt 长度变化很大，分页缓存能减少碎片，INT8/INT4 又能降低每页体积，两者叠加后，单卡可承载的并发数和上下文总量会明显提升。

如果你是第一次接触这类系统，可以把整个链路记成一句话：

$$
\text{新 token} \rightarrow \text{算出 } K,V \rightarrow \text{压缩后写入缓存} \rightarrow \text{下次读取历史缓存参与注意力}
$$

理解这条链路后，后面的量化、分页、剪枝都只是在这个框架上做不同位置的优化。

---

## 工程权衡与常见坑

工程上最常见的误区，是只看“显存省了多少”，不看“精度损失和时延开销”。这会导致离线评测很漂亮，线上体验却变差。

先给一个经验公式，虽然是近似的，但足够说明问题：

$$
latency \approx base\_latency + quant\_dequant\_overhead \times tokens
$$

如果 `tokens` 很短，或者缓存本来就小，那么后面那部分新增开销可能比省下来的显存收益更显眼，结果就是“更省显存，但更慢”。

还可以把收益和成本拆成两个维度看：

| 维度 | 不做 KV 优化 | 做 INT8 KV 量化后的常见变化 |
|---|---|---|
| 显存占用 | 高 | 明显下降 |
| 带宽压力 | 高 | 通常下降 |
| 单步计算复杂度 | 较简单 | 增加量化/反量化逻辑 |
| 数值稳定性 | 最稳 | 有轻微风险 |
| 系统实现复杂度 | 低 | 更高 |

常见坑与缓解方式如下：

| 问题 | 表现 | 原因 | 缓解措施 |
|---|---|---|---|
| 短序列负收益 | 延迟反而升高 | 量化/反量化开销盖过收益 | 只在长上下文启用量化 |
| INT4 精度下降 | 回复质量、置信度下降 | 量化误差更大 | 先用 INT8，再局部试 INT4 |
| 某些层特别敏感 | 少数任务明显退化 | 不同层/头对误差容忍不同 | 分层量化，敏感层保留 FP16 |
| 显存碎片仍高 | 仍出现 OOM | 只量化不分页 | 配合页式缓存 |
| 吞吐不稳定 | 高并发抖动 | 动态长度请求互相挤压 | 做页级调度与预算隔离 |
| 量化粒度过粗 | 长文本后段质量恶化 | 一个缩放参数覆盖太多元素 | 改成 per-head、per-channel 或 per-block 量化 |
| 校准样本不匹配 | 线上效果差于离线 | 离线测试文本太短或分布单一 | 用真实请求分布评估 |
| 指标只看平均值 | 少量坏例子被淹没 | 平均分掩盖长尾退化 | 同时监控 P95/P99 时延和失败样本 |

这里的“敏感层”可以理解为“对最终输出影响更大的层”。实际部署里，经常不是全层统一量化，而是采用混合策略：高层或关键头保留 FP16/FP8，低层优先做 INT8，最不敏感部分才考虑 INT4。

一个真实工程场景是对话模型上线。团队把 KV Cache 全部切到 INT4 后，长上下文的确不再 OOM，但部分复杂问答的答案稳定性下降，尤其是在长链推理和代码补全里更明显。最后可行的方案通常不是“回退全部优化”，而是：

- 默认 INT8
- 对低影响层试 INT4
- 对关键层或关键头保留高精度
- 只在长上下文请求上启用更激进策略

这类方案不如“一刀切 INT4”理论压缩率高，但更符合上线要求。工程目标从来不是单指标最优，而是整体吞吐、延迟、质量、稳定性之间的可接受平衡。

为了让判断更可执行，可以用下面这张决策表：

| 场景 | 推荐策略 | 原因 |
|---|---|---|
| 上下文短、追求最低延迟 | 先保留 FP16/BF16 | 避免额外编解码开销 |
| 中长上下文、质量要求稳 | 优先 INT8 + 分页 | 收益稳定，风险较低 |
| 超长上下文、显存非常紧 | INT8 基线后局部尝试 INT4 | 先保住稳定性，再挖极限空间 |
| 高风险行业任务 | 混合精度或仅低层量化 | 降低关键输出退化概率 |

这比“统一追求最高压缩率”更符合实际部署逻辑。

---

## 替代方案与适用边界

KV 量化不是唯一道路。它适合“显存紧张且上下文较长”的推理场景，但并不总是最优。常见替代方案或配套方案如下：

| 方案 | 优势 | 代价 | 适用场景 |
|---|---|---|---|
| KV 量化 | 直接按比例省显存 | 有量化误差与解码开销 | 长上下文、显存紧张 |
| PagedAttention | 降低碎片，易扩展并发 | 实现复杂 | 动态请求、多会话服务 |
| 稀疏 Attention | 减少历史读取量 | 可能改模型行为 | 超长上下文、结构性任务 |
| 动态层剪枝 | 降低每步计算和缓存压力 | 可能损失质量 | 延迟敏感、可接受轻微退化 |
| 混合精度缓存 | 精度更稳 | 压缩率低于全量化 | 高精度任务 |
| 滑动窗口缓存 | 控制缓存上限 | 丢弃远距离信息 | 对远程依赖不敏感的场景 |
| GQA/MQA 结构优化 | 天然减小 KV 头数 | 需要模型结构支持 | 新模型设计或可重新训练场景 |

适用边界可以概括成三条：

1. 如果你主要瓶颈是显存容量，优先考虑 `INT8 + 分页`。
2. 如果你主要瓶颈是极致精度，不要直接上 INT4，先做分层或混合精度。
3. 如果你主要瓶颈是短请求延迟，先验证量化开销是否值得，不要默认它一定更快。

还可以再补一句常被忽视的边界：如果模型本身已经采用 GQA/MQA，KV Cache 的规模比传统多头注意力更小，这时“继续压缩 KV”与“优化调度和分页”谁更优，需要重新测，而不是沿用旧结论。

再给一个真实工程例子。某在线服务部署在 A100 上，白天请求长度波动很大，短问题和长文档问答混在一起。只做 FP16 KV Cache 时，高峰期容易因为几个超长请求把缓存顶满。改成“分页 + INT8”后，系统能按页回收旧会话缓存，也能让同样显存容纳更多活动请求，OOM 频率明显下降。相反，在医学问答模型里，由于答案稳定性要求更高，团队通常只量化低层缓存，高层仍保留更高精度，这就是典型的“边界驱动设计”。

所以更准确的说法不是“INT4 一定最好”，而是“根据请求长度分布、模型敏感性、硬件预算选择组合策略”。对大多数团队而言，稳定的起点通常是：

- 先测 FP16 基线
- 再测 INT8 KV Cache
- 最后在局部尝试 INT4、剪枝或更激进压缩

如果把这三步翻译成更工程化的检查清单，可以写成：

| 检查项 | 需要回答的问题 |
|---|---|
| 请求分布 | 长请求占比是多少，P95/P99 多长 |
| 显存水位 | 峰值时是权重占多，还是 KV 占多 |
| 质量红线 | 哪些任务不能接受轻微退化 |
| 时延红线 | 是否必须压低首 token 或单 token 延迟 |
| 部署形态 | 单卡、多卡、连续批处理还是离线批推理 |

没有这张表，很多优化方案都只是在“凭感觉替换数据类型”。

---

## 参考资料

| 资料 | 主要内容 | 用途 |
|---|---|---|
| Hugging Face `KV cache strategies` 文档 | 解释 KV Cache 的定义、缓存策略和接口思路 | 建立概念与工程入口 |
| LMDeploy `KV Cache Quantization` 指南 | 介绍 INT8/INT4 KV 量化方式与部署经验 | 评估量化收益与风险 |
| KIVI / MiniKV 一类缓存量化与压缩论文 | 给出缓存压缩动机、误差控制思路和更激进的压缩方向 | 理解理论压缩空间 |
| vLLM / PagedAttention 相关论文与文档 | 解释页式缓存和高并发推理调度 | 理解分页与吞吐优化的关系 |
| MQA / GQA 相关论文 | 解释为什么减少 KV 头数可以天然降低缓存体积 | 理解模型结构层面的替代方案 |

如果按阅读顺序安排，建议这样看：

1. 先看 Hugging Face 文档，建立“KV Cache 是什么、为什么能省重算”的基本概念。
2. 再看 vLLM / PagedAttention，理解为什么线上服务要把缓存做成页，而不是简单的连续大数组。
3. 接着看 LMDeploy 的量化实践，理解 INT8/INT4 在部署里的真实收益和风险。
4. 最后再读 KIVI、MiniKV 一类论文，理解更激进压缩方法在误差控制和理论空间上的差异。

下面补一版更具体的参考方向，便于继续深挖：

| 方向 | 应重点关注的问题 |
|---|---|
| 基础文档 | KV Cache 的生命周期、prefill 和 decode 的差别是什么 |
| 量化实践 | 量化粒度是 per-tensor、per-channel 还是 per-token，误差如何传播 |
| 分页调度 | 页面大小如何选，碎片如何统计，并发时如何回收 |
| 学术论文 | 压缩率如何定义，评测任务是否覆盖长上下文和高敏感任务 |
| 模型结构 | MQA/GQA 为什么能减少缓存，代价转移到了哪里 |

Hugging Face 提供的是“KV Cache 是什么、接口怎么理解”的基础框架；KIVI、MiniKV 一类工作更强调“为什么可以压缩、压缩到什么程度”；LMDeploy 和类似推理框架文档更接近部署实践，重点在“如何把 INT8/INT4 和运行时调度结合起来”；MQA/GQA 相关工作则提醒我们，**不是所有优化都必须发生在推理系统层，模型结构本身也会改变 KV Cache 的规模。**

如果只保留一句最实用的结论，那就是：**先用文档理解机制，再用框架文档理解实现，最后用论文判断极限方向是否值得投入。**
