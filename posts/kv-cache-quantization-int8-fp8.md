## 核心结论

`KV Cache 量化 = 将推理阶段的 K/V 缓存从高精度压缩到低精度表示。`

这里的 `K/V` 是注意力机制里保存历史信息的两类张量，白话说，就是模型为了“记住前面已经看过什么”而留下的历史记录。`cache` 是缓存，白话说，就是 decode 阶段会不断增长的一块显存区。

KV Cache 量化的目标不是把模型权重变小，而是把**推理阶段持续增长的缓存显存**压下去。在线上长对话、RAG、代码助手、多轮问答场景里，真正先把 GPU 顶满的，常常不是权重，而是越来越长的 KV cache。

新手版玩具例子可以这样理解：一个模型权重已经能完整装进 80GB GPU，但每生成一个新 token，历史 token 的 `K/V` 都要继续累积。如果原来用 `FP16/BF16` 存，显存会线性增长；改成 `INT8` 或 `FP8` 后，相当于把“历史记录”换成更省空间的格式，生成能继续往后走，或者同一张卡能同时接更多请求。

结论表如下：

| 方案 | 主要收益 | 主要代价 | 适用场景 |
|---|---|---|---|
| FP16/BF16 KV | 精度高 | 显存占用大 | 小上下文、低并发 |
| INT8 KV | 显存减半级别 | 需校准、可能损精度 | 通用推理优化 |
| FP8 KV | 显存更省、硬件友好时效果好 | 依赖 GPU / kernel 支持 | 新 GPU、长上下文服务 |

如果问题是“模型本身装不下”，优先级通常是权重量化；如果问题是“历史越长，显存越爆”，KV Cache 量化通常更直接。

---

## 问题定义与边界

本文讨论的对象只包括**推理时**的 KV cache，不包括训练权重，也不包括完整激活量化。`prefill` 是模型先把整段输入一次性读完并建立初始 cache，白话说，就是“先把上下文吃进去”；`decode` 是之后每次只生成一个 token，同时把新 token 的 `K/V` 追加进 cache，白话说，就是“边生成边记历史”。

边界说明如下：

| 项目 | 是否属于本文 |
|---|---|
| 权重量化 | 否，作为对比背景 |
| 激活量化 | 否，除非与 KV cache 方案比较 |
| KV Cache 量化 | 是，核心主题 |
| 训练期低精度 | 否 |

为什么重点放在 decode 阶段？因为 prefill 的计算重、但生命周期短；decode 的单步计算轻很多，可它会把历史缓存越积越多。单个请求的上下文越长，或者并发请求越多，这块显存就越大。一个很常见的线上现象是：模型权重早已稳定驻留显存，真正随着业务量上升而持续膨胀的是 KV cache。

可以把流程抽象成两步：

| 阶段 | 发生的事 |
|---|---|
| `prefill` | 输入 token 进入模型，生成初始 cache |
| `decode` | 每步生成 1 个 token，cache 持续增长 |

最小术语集也很简单：

| 术语 | 含义 |
|---|---|
| `K` = Key | 注意力里用于匹配“我该关注谁”的向量 |
| `V` = Value | 注意力里真正被取出的内容向量 |
| `cache` | 历史 token 的 `K/V` 存储 |

一个真实工程例子：长上下文代码审查服务，用户贴入几千行代码后继续多轮追问。首轮输入完成后，后续每次生成解释、修改建议、diff 分析，都要依赖此前所有 token 的 KV cache。如果服务目标是高并发，权重是否量化只决定“模型能不能上卡”，KV cache 是否量化更决定“单卡能扛多少会话”。

---

## 核心机制与推导

统一写法先抽象成两步：

$$
Q = quantize(X, s)
$$

$$
\hat{X} = dequantize(Q, s)
$$

其中 `X` 是原始 K 或 V 张量，`s` 是 scale，白话说，就是“把数值缩放到适合低精度存储的比例尺”；`Q` 是量化后的表示；$\hat{X}$ 是反量化后的近似值。

### INT8 路线

`INT8` 的核心是“整数编码 + scale”：

$$
Q = clip(round(X / s), -127, 127)
$$

$$
\hat{X} = Q \cdot s
$$

直观理解是：先除以 `s`，把浮点数压到接近 `[-127,127]` 的整数区间；存储时只记整数；读取时再乘回 `s`。由于 `round` 和 `clip` 会丢信息，$\hat{X}$ 只能逼近 `X`，不能完全相等。

最小数值例子如下。设最大绝对值为 `1.2`，取 $s = 1.2 / 127 \approx 0.00945$：

| 原值 `X` | 缩放后 | `INT8` 量化值 `Q` | 反量化 `X_hat` |
|---|---:|---:|---:|
| 0.8 | 84.7 | 85 | 0.803 |
| -0.3 | -31.7 | -32 | -0.302 |
| 1.2 | 127.0 | 127 | 1.200 |

这个玩具例子说明两件事。第一，量化误差通常是“小偏差”，不是完全乱掉。第二，scale 选得好不好决定误差大小。如果 `s` 太小，大值会被截断；如果 `s` 太大，小值会被压得太粗。

### FP8 路线

`FP8` 不是把数值变成整数，而是先缩放，再映射到更低位的浮点格式，比如 `E4M3` 或 `E5M2`。白话说，`E4M3` 表示指数位 4 位、尾数位 3 位；`E5M2` 则指数范围更大、尾数更粗。

它的抽象过程是：

1. 先做缩放：`Y = X / s`
2. 再 cast 到 FP8：`Q = cast_fp8(Y)`
3. 读取时乘回：$\hat{X} = Q \cdot s$

因此 FP8 的重点不是“有没有 scale”，而是“scale 是否让数值分布落在 FP8 容易表达的区间里”。这也是为什么 FP8 常被说成更依赖硬件和 kernel 支持，因为实际收益高度依赖底层实现。

### 为什么 scale 要按最大值或统计分布选

scale 本质是在解决“有限表示范围”和“原始数据分布”之间的对齐问题。若张量中少量异常大值存在，直接按最大值定标会保护大值，但小值分辨率下降；如果按百分位数或校准分布定标，小值更准，但极端值可能溢出。这就是量化里经典的动态范围与精度分辨率权衡。

### 为什么不同 head / layer 敏感度不同

`head` 是多头注意力里的一个独立注意力子空间，白话说，就是“模型从不同角度看历史信息”。不同层、不同 head 的数值分布并不一样。有的 head 值域集中，适合粗粒度量化；有的 head 对细小变化很敏感，量化误差更容易放大到注意力分数里。所以工程上经常出现同样是 INT8 KV，某些模型几乎无损，某些模型在长上下文下质量明显回落。

### 为什么 per-tensor 是最常见工程折中

`per-tensor scale` 是整个张量共用一个 scale，白话说，就是“一整块缓存只配一个比例尺”。它实现简单、元数据少、kernel 好做，所以常被用作第一版基线。更细粒度的 `per-head` 或 `per-channel` 可能更准，但需要更多 scale、更多读写和更复杂的融合 kernel。工程里最终不是只看误差最小，而是看“显存节省 + kernel 可落地 + 吞吐收益”是否整体最优。

---

## 代码实现

下面用纯 Python 写一个可运行的玩具版本，展示四步：数据准备、量化、反量化、接入 attention。这里不用框架 API 黑盒，目的是让量化发生的位置一眼可见。

```python
import math

def max_abs(xs):
    return max(abs(x) for x in xs) if xs else 0.0

def quantize_int8(xs):
    scale = max_abs(xs) / 127.0 if max_abs(xs) > 0 else 1.0
    q = []
    for x in xs:
        v = round(x / scale)
        v = max(-127, min(127, v))
        q.append(int(v))
    return q, scale

def dequantize_int8(qs, scale):
    return [q * scale for q in qs]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [x / s for x in exps]

def attention(query, keys, values):
    scores = [dot(query, k) for k in keys]
    probs = softmax(scores)
    out = [0.0 for _ in values[0]]
    for p, v in zip(probs, values):
        for i, x in enumerate(v):
            out[i] += p * x
    return out

# 1. 数据准备：两条历史 token 的 K/V
keys = [
    [0.8, -0.3, 1.2],
    [0.1, 0.4, -0.5],
]
values = [
    [1.0, 0.0],
    [0.2, 0.9],
]
query = [0.7, -0.2, 0.6]

# 2. cache 写入前量化
qk = []
sk = []
for k in keys:
    q, s = quantize_int8(k)
    qk.append(q)
    sk.append(s)

qv = []
sv = []
for v in values:
    q, s = quantize_int8(v)
    qv.append(q)
    sv.append(s)

# 3. decode 读取时反量化
keys_hat = [dequantize_int8(q, s) for q, s in zip(qk, sk)]
values_hat = [dequantize_int8(q, s) for q, s in zip(qv, sv)]

# 4. 接入 attention
out_fp = attention(query, keys, values)
out_q = attention(query, keys_hat, values_hat)

# 允许少量量化误差
assert abs(out_fp[0] - out_q[0]) < 0.02
assert abs(out_fp[1] - out_q[1]) < 0.02
assert qk[0] == [85, -32, 127]
```

如果把这段逻辑映射回真实推理系统，接口形态通常类似下面这样：

```python
def quantize_int8(x):
    s = x.abs().max() / 127
    q = torch.clamp((x / s).round(), -127, 127).to(torch.int8)
    return q, s

def dequantize_int8(q, s):
    return q.float() * s

def quantize_fp8(x, fp8_dtype):
    s = x.abs().max()
    y = x / s
    q = y.to(fp8_dtype)
    return q, s
```

接入位置示意如下：

| 阶段 | 输入 | 输出 |
|---|---|---|
| cache 写入 | FP16/BF16 `K/V` | INT8/FP8 cache |
| cache 读取 | 低精度 cache | 还原后的 `K/V` |
| attention 计算 | 还原后的 `K/V` | logits |

工程里的关键不是“能不能量化”，而是“是否能把反量化与 attention 融合”。如果每一步 decode 都先把大量低精度 cache 拉出来、单独反量化、再做 attention，那么收益可能主要体现在容量，而不是延迟。

真实工程例子：一个企业知识库问答服务需要 32k 上下文，并发 64。原始 BF16 KV cache 让单卡并发上不去，服务在 decode 阶段 OOM。改成 `INT8_KV_CACHE` 后，单请求可用上下文更长，或者在相同上下文下提升并发；如果底层有 fused dequant + attention，吞吐量还能进一步改善。这类收益在“长上下文 + 多并发”服务里非常现实。

---

## 工程权衡与常见坑

量化不是只看“显存省了多少”，还必须同时看精度、硬件支持、kernel 融合程度和真实延迟。很多方案在 PPT 上很好看，落到生产环境后，真正有价值的指标只有三个：能不能稳定跑、质量掉多少、吞吐到底升没升。

常见坑如下：

| 坑 | 结果 | 规避方式 |
|---|---|---|
| 只看显存不看精度 | 回答质量下降 | 用代表性数据集评测 |
| scale 过粗 | 某些 head 更敏感 | 先从 per-tensor 做基线，再看 per-head |
| 没有 fused kernel | 延迟不降 | 检查是否有 fused dequant + attention |
| 忽略硬件限制 | 功能不可用或性能差 | 确认 GPU / TensorRT / kernel 支持 |
| FP8 校准不足 | 数值溢出或失真 | 使用校准数据选 scale |

收益判定表建议固定查看：

| 指标 | 是否必看 |
|---|---|
| 显存占用 | 是 |
| 吞吐量 | 是 |
| 首 token 延迟 | 视场景 |
| decode 延迟 | 是 |
| 任务质量 | 是 |

这里最容易误判的是 FP8。很多人把 FP8 理解成“比 INT8 更先进，所以一定更好”，这是错误的。FP8 的优势常常建立在两个前提上：第一，GPU 和 kernel 对 FP8 路线支持足够成熟；第二，scale 与校准足够好。否则它可能既没明显降时延，也没有比 INT8 更稳。

另一个常见坑是把评测做得太短。KV cache 的价值主要体现在长上下文和多步 decode，如果只测短 prompt、只看首 token 延迟，结论会失真。正确做法是把评测拉到真实业务分布：短问答、长问答、连续多轮、并发压测都要覆盖。

---

## 替代方案与适用边界

KV Cache 量化不是唯一手段，它只是“推理显存优化”谱系里的一个分支。是否优先做它，要先判断瓶颈到底在哪。

对比如下：

| 方案 | 解决的问题 | 优点 | 限制 |
|---|---|---|---|
| KV Cache 量化 | 推理缓存显存 | 直接降低 cache 占用 | 依赖校准和 kernel |
| 权重量化 | 模型参数显存 | 更广泛通用 | 不直接解决长上下文 cache |
| 稀疏注意力 | 长上下文计算与存储 | 潜在更强压缩 | 改动更大 |
| 分页 / 管理式 cache | 显存调度 | 更利于并发 | 不是数值压缩 |

边界判断可以直接按下面的清单看：

1. 是否主要是 decode 阶段显存紧张。
2. 是否有新 GPU 支持 FP8。
3. 是否已有 fused kernel。
4. 是否允许少量精度损失。

如果你的问题是“7B 模型都放不进卡”，优先级更可能是权重量化；如果你的问题是“模型能跑，但长对话时 cache 把显存吃光”，KV Cache 量化通常更值得先做。若业务上下文本来很短、并发也不高，那做 KV 量化的收益会明显下降，甚至不如先做更简单的权重压缩或缓存调度优化。

还要看到一个现实边界：KV Cache 量化主要优化的是**容量**。它能让你装下更长上下文、扛住更多并发，但不保证必然降低延迟。只有当低精度读写、反量化和 attention 被很好地融合进 kernel 时，容量收益才更容易转成吞吐收益。

---

## 参考资料

| 资料 | 用途 |
|---|---|
| TensorRT-LLM `INT8/FP8 KV Caches` | 官方实现与接口说明 |
| TensorRT-LLM `Quantized KV-Cache` | 性能与量化配置参考 |
| vLLM `Quantized KV Cache` | 另一个推理框架的实现对照 |
| NVIDIA Transformer Engine `FP8 Current Scaling` | FP8 缩放机制背景 |
| KIVI | 低比特 KV cache 研究方向 |
| KVQuant | 长上下文 KV cache 量化实践 |

1. [TensorRT-LLM: INT8/FP8 KV Caches](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html) 说明官方在推理侧如何接入 `INT8_KV_CACHE` 和 `FP8_KV_CACHE`，支撑本文对工程落地点的描述。
2. [TensorRT-LLM: Quantized KV-Cache](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html) 给出量化配置与性能调优信息，支撑本文对“容量收益不等于时延收益”的讨论。
3. [vLLM: Quantized KV Cache](https://docs.vllm.ai/en/v0.10.1/features/quantization/quantized_kvcache.html) 提供另一套推理框架的实现视角，适合对照理解 KV cache 量化在不同系统中的接口形态。
4. [NVIDIA Transformer Engine: FP8 Current Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html) 解释 FP8 缩放机制，支撑本文关于 scale 与数值范围的推导背景。
5. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://huggingface.co/papers/2402.02750) 提供更低比特 KV cache 研究方向，帮助读者理解 per-channel、非对称量化等进一步优化空间。
6. [KVQuant Project Page](https://slice.eecs.berkeley.edu/papers/kvquant-towards-10-million-context-length-llm-inference-with-kv-cache-quantization/) 展示长上下文 KV cache 量化的研究实践，支撑本文对超长上下文场景的适用边界说明。
