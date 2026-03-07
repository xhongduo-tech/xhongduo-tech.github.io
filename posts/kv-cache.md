## 核心结论

KV Cache 是自回归推理中的一种“空间换时间”机制。空间换时间，白话说，就是多占一部分显存，换取后续每一步生成更快。它的做法是：把每一层、每个历史 token 的 Key 和 Value 先存起来，后面生成新 token 时直接复用，不再把整段上下文重新算一遍。

如果不做缓存，生成第 $t$ 个 token 时，模型通常要再次处理前 $t-1$ 个 token，对应的总计算量随序列长度呈平方级增长，可记为 $O(T^2)$。如果做了 KV Cache，那么生成第 $t$ 个 token 时，只需要：

1. 为当前新 token 计算一次新的 $Q,K,V$
2. 用当前 $Q$ 去和历史缓存的 $K$ 做点积
3. 用注意力权重去聚合历史缓存的 $V$

这时新增 token 的计算只和“新来的这一个 token”有关，整段生成过程的重复计算被消掉，总体可近似看成 $O(T)$ 的增量过程。

可以先记住一个最实用的结论：

$$
B_{\text{token}} = 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times b_{\text{prec}}
$$

其中 $B_{\text{token}}$ 表示“每个 token 需要缓存多少字节”。这个公式说明，KV Cache 的内存成本会随着层数、KV 头数、每头维度、精度字节数线性增长。

一个最直观的例子是：生成第 100 个 token 时，只需要用第 100 个 token 的 Query 去和前 99 个 token 已缓存的 Key/Value 交互，不需要重新扫一遍 prompt。也就是说，历史信息不再“重复编码”，而是“重复使用”。

简化图示可以写成：

`新 Query -> 点乘历史 K Cache -> 得到权重 -> 聚合历史 V Cache -> 输出当前结果`

---

## 问题定义与边界

问题定义很简单：自回归生成时，模型每一步都要“看见”之前全部上下文。自回归，白话说，就是一个 token 一个 token 往后接着生成。注意力机制要求当前 token 能访问前面的历史 token，因此如果没有缓存，历史部分会被重复计算很多次。

设最终生成长度为 $T$。如果第 1 步看 1 个 token，第 2 步看 2 个 token，直到第 $T$ 步看 $T$ 个 token，那么总工作量近似是：

$$
1 + 2 + 3 + \cdots + T = \frac{T(T+1)}{2}
$$

这就是平方级增长的来源。

KV Cache 解决的是“重复计算历史 K/V”这个问题，但它不解决所有推理瓶颈。它的边界主要有三个：

| 边界 | 含义 | 影响 |
|---|---|---|
| 显存容量 | 历史 token 越长，缓存越大 | 长上下文容易爆显存 |
| 并发请求数 | 多个请求会各自持有缓存 | 高并发下显存线性累加 |
| 缓存管理方式 | 连续分配、回收、碎片整理都要成本 | 管理不好会浪费显存 |

这里最容易让人误判的是：KV Cache 不是一次性固定开销，而是随着上下文长度持续增长。即使 `batch=1`，缓存曲线也是线性上升；如果 batch 增大，曲线会整体按倍数抬高。

以 LLaMA-2 70B 的典型参数估算，`batch=1` 时：

| 上下文长度 | KV Cache 近似占用 |
|---|---:|
| 4K | 约 10 GB |
| 8K | 约 20 GB |
| 16K | 约 40 GB |
| 32K | 约 80 GB |

这张表的含义很直接：长度翻倍，缓存占用基本也翻倍。很多人会觉得“4K 不算长”，但对 70B 这类大模型来说，4K 已经是不可忽略的显存预算。

玩具例子可以这样理解。假设你有一个只有 4 个 token 的输入：`A B C D`。当模型生成第 5 个 token 时，如果没有 KV Cache，就要重新把 `A B C D` 的注意力相关中间结果再算一遍；如果有 KV Cache，则 `A B C D` 的 K/V 早已存好，只需为当前第 5 个 token 新算一个 Query，然后直接去查历史缓存。

真实工程例子更能说明边界：如果某个 70B 服务上，一个 8K 请求大约需要 20GB 的 KV Cache，那么 10 个并发请求就接近 200GB。即使模型权重已经做了量化，单机显存通常也扛不住。这就是为什么“模型能跑起来”和“服务能稳定接多用户请求”是两件不同的事。

---

## 核心机制与推导

先把符号统一：

| 符号 | 含义 |
|---|---|
| $L$ | Transformer 层数 |
| $H_{\text{kv}}$ | KV 头数 |
| $D_{\text{head}}$ | 每个头的维度 |
| $b_{\text{prec}}$ | 每个元素占用的字节数，例如 FP16 是 2 |
| $B$ | batch size，请求批大小 |
| $S$ | 序列长度 |
| $B_{\text{token}}$ | 单个 token 的 KV Cache 字节数 |
| $M$ | 总 KV Cache 字节数 |

为什么单个 token 的缓存大小是

$$
B_{\text{token}} = 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times b_{\text{prec}}
$$

因为对每一层来说，一个 token 需要保存一份 Key 和一份 Value，所以有前面的系数 2。每份张量的元素数是“KV 头数乘以每头维度”，再乘上层数和每元素字节数，就是总字节数。

于是总缓存大小为：

$$
M = B \times S \times B_{\text{token}}
$$

这就是 KV Cache 的核心推导。它告诉我们两件事：

1. 时间上，历史 K/V 被重用，不再重复计算。
2. 空间上，历史 K/V 被完整保存，因此显存随 $B$ 和 $S$ 线性增长。

以 LLaMA-2 70B 的一个常见估算为例，取：

- $L = 80$
- $H_{\text{kv}} = 64$
- $D_{\text{head}} = 128$
- $b_{\text{prec}} = 2$ 字节（FP16）

代入得到：

$$
B_{\text{token}} = 2 \times 80 \times 64 \times 128 \times 2
$$

即：

$$
B_{\text{token}} = 2{,}621{,}440 \text{ bytes} \approx 2.5 \text{ MB}
$$

如果 `batch=1`，上下文长度 $S=4096$，那么：

$$
M = 1 \times 4096 \times 2.5\text{MB} \approx 10\text{GB}
$$

如果长度变成 32768，那么：

$$
M = 1 \times 32768 \times 2.5\text{MB} \approx 80\text{GB}
$$

这个推导非常重要，因为它说明“即使只有一个请求”，只要上下文足够长，KV Cache 也会把显存吃满。很多部署失败不是算力不够，而是内存预算从一开始就算错了。

再看机制本身。对当前时刻 $t$，注意力可以写成：

$$
\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^\top}{\sqrt{d}}\right)V_{1:t}
$$

有了 KV Cache 之后，$K_{1:t-1}$ 和 $V_{1:t-1}$ 来自缓存，当前只新增 $K_t, V_t$。因此每一步的主要工作不是“重算前缀”，而是“把新 token 接到历史后面”。

从实现角度看，流程可以概括为：

1. 预填充阶段，把 prompt 整段送进去，生成首批 K/V，并写入 cache。
2. 解码阶段，每次只输入一个新 token。
3. 每层拿当前 Query 去和历史 K Cache 做点积。
4. 算完新的 K/V 后，将其 append 到 cache 尾部。
5. 下一步继续复用。

这就是“前缀不变，缓存复用”的本质。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不是完整 Transformer，而是把 KV Cache 的最关键部分抽出来：初始化、追加、查询。Query 可以理解为“当前 token 想找什么信息的向量表示”，Key 可以理解为“历史 token 提供什么索引线索”，Value 则是“被取回的内容”。

```python
import math

class KVCache:
    def __init__(self, n_layers):
        self.k = [[] for _ in range(n_layers)]
        self.v = [[] for _ in range(n_layers)]

    def append(self, layer_idx, k_new, v_new):
        assert len(k_new) == len(v_new)
        self.k[layer_idx].append(k_new)
        self.v[layer_idx].append(v_new)

    def get_layer_cache(self, layer_idx):
        return self.k[layer_idx], self.v[layer_idx]


def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]


def attention(query, keys, values):
    assert len(keys) == len(values)
    scores = [dot(query, k) / math.sqrt(len(query)) for k in keys]
    weights = softmax(scores)

    out = [0.0] * len(values[0])
    for w, v in zip(weights, values):
        for i in range(len(v)):
            out[i] += w * v[i]
    return out


# 玩具例子：1层、历史3个token，生成第4个token
cache = KVCache(n_layers=1)

# 先把前三个 token 的 K/V 放入缓存
cache.append(0, [1.0, 0.0], [10.0, 0.0])   # token A
cache.append(0, [0.0, 1.0], [0.0, 20.0])   # token B
cache.append(0, [1.0, 1.0], [5.0, 5.0])    # token C

# 当前第4个 token 的 Query
q_new = [1.0, 0.2]

keys, values = cache.get_layer_cache(0)
out = attention(q_new, keys, values)

# 输出维度与 value 维度一致
assert len(out) == 2

# 缓存长度没有被重算，只是复用已有的 3 个历史 token
assert len(keys) == 3
assert len(values) == 3

# 当前 token 算出自己的 K/V 后再追加
k_new = [0.3, 0.7]
v_new = [3.0, 7.0]
cache.append(0, k_new, v_new)

keys2, values2 = cache.get_layer_cache(0)
assert len(keys2) == 4
assert len(values2) == 4
```

这段代码体现了三个动作：

1. `KVCache(...)` 初始化缓存结构。
2. `append(...)` 在每一步把新 token 的 K/V 追加进去。
3. `attention(...)` 只拿当前 Query 和历史缓存交互。

如果写成更贴近工程实现的伪代码，可以概括为：

```python
cache = init_empty_cache()

# prefill: 一次处理整个 prompt
for layer in model.layers:
    q, k, v = layer(prompt_hidden_states)
    cache[layer].append(k, v)

# decode: 每次只输入一个新 token
while not stop:
    hidden = embed(last_token)

    for layer in model.layers:
        q, k_new, v_new = layer(hidden)

        k_hist, v_hist = cache[layer].read()
        hidden = attend(q, concat(k_hist, k_new), concat(v_hist, v_new))

        cache[layer].append(k_new, v_new)

    last_token = sample(hidden)
```

真实工程里不会像玩具代码一样用 Python 列表逐项 `append`，而是通常预分配一块连续显存，再通过位置指针写入，原因有两个：

1. 避免频繁重新分配显存。
2. 保持张量布局稳定，方便 GPU 高效读取。

真实工程例子：在线推理服务收到一个 2048-token 的 prompt 后，会先做一次 prefill，把 2048 个位置的各层 K/V 全部写入 cache。之后进入逐 token 解码，每生成一个 token，只扩展一格 cache，而不是重建前 2048 格。

---

## 工程权衡与常见坑

KV Cache 几乎是大模型在线推理的标准配置，但它不是“开了就一定更好”。工程上真正难的是如何把它控制在显存预算内。

先看几个常见坑：

| 问题 | 现象 | 原因 | 常见处理方式 |
|---|---|---|---|
| 长上下文爆显存 | 请求一长就 OOM | $M = B \times S \times B_{\text{token}}$ 线性增长 | 限制最大上下文、做容量预估 |
| 高并发放大占用 | 单请求能跑，多请求挂掉 | 每个请求各自持有 cache | admission control、分级队列 |
| 显存碎片 | 理论够用，实际分配失败 | cache 长短不一，反复申请释放 | 分页管理、固定块分配 |
| 被最长请求拖累 | 小请求也占大块空间 | 粗粒度预留按最大长度算 | page-based cache |
| 模型权重和 cache 混算错误 | 只算权重没算 cache | 部署预算不完整 | 总量估算时权重、激活、cache 分开算 |

最常见的误区是“模型权重放得下，服务就能跑”。这是错误的。在线推理至少要同时考虑：

$$
\text{总显存} \approx \text{模型权重} + \text{KV Cache} + \text{运行时临时张量} + \text{框架额外开销}
$$

如果只盯着权重大小，很容易在高峰并发时突然 OOM。

一个真实工程例子：假设某 70B 服务的单请求 8K 上下文需要约 20GB KV Cache。如果你希望同时接 10 个活跃请求，仅 cache 就接近 200GB。再加上模型权重、临时激活和通信缓冲区，单机几乎不现实。此时要么降低上下文上限，要么减少并发，要么切到多卡/多机方案。

另一个典型问题是显存碎片。碎片，白话说，就是显存虽然总量还有空余，但因为空闲块不连续，大块张量申请失败。KV Cache 的请求长度不一致，生命周期也不同，特别容易造成碎片。vLLM 的 PagedAttention 之所以重要，就在于它把 cache 切成页来管理，而不是让每个请求独占一大块连续内存。这样更容易复用空闲页，也更适合动态长度请求。

实践中可执行的避坑策略通常有四条：

| 策略 | 作用 |
|---|---|
| 分页缓存 | 降低碎片，便于回收和复用 |
| 空闲页回收 | 请求结束后尽快归还显存块 |
| 上下文长度监控 | 防止单请求占满整卡 |
| cache-aware batching | 把长度接近的请求放一起，减少浪费 |

怎么量化“会不会爆显存”？一个简单方法是先算单请求：

$$
M_{\text{req}} = S \times B_{\text{token}}
$$

再乘并发请求数 $N$：

$$
M_{\text{total}} \approx N \times M_{\text{req}}
$$

最后检查：

$$
\text{模型权重} + M_{\text{total}} + \text{余量} < \text{可用显存}
$$

这里“余量”不能省。真实系统里要给 allocator、通信 buffer、算子工作区留空间。经验上，不把卡打到 100% 才是稳定服务的前提。

---

## 替代方案与适用边界

KV Cache 不是唯一办法，只是在大多数自回归推理场景中，它是默认最划算的办法。是否适合，要看上下文长度、并发规模和硬件条件。

下面做一个对比：

| 方案 | 核心思路 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|---|
| KV Cache | 保存历史 K/V，后续直接复用 | 单步解码快，适合在线生成 | 吃显存，长上下文和高并发压力大 | 中长上下文、低到中高并发 |
| 分布式/模型并行缓存 | 把层、头或 cache 拆到多卡/多机 | 单卡放不下时仍可运行 | 实现复杂，通信开销高 | 超大模型、超长上下文 |
| 延迟加载上下文 | 不一次性持有全部上下文，按需取回 | 节省本地显存 | 时延变大，实现复杂 | 极长上下文、吞吐优先场景 |
| 不缓存，每步重算 | 每一步重新计算全部前缀 | 实现最简单，占用更低 | 速度慢，序列长时不可接受 | 很短上下文、离线低频任务 |

如果单卡只有 30GB 可用显存，而业务又要求 32K 上下文，那么单卡 KV Cache 方案就已经越界了。因为仅 cache 就可能逼近甚至超过这个预算，更别说权重和运行时开销。此时更现实的路线是：

1. 多卡分摊层或头。
2. 使用支持分页缓存的推理框架。
3. 必要时把一部分上下文用外部存储或分段检索方式处理。

相反，如果任务只是一个短上下文、低并发的离线脚本，例如每次只处理 128 到 256 token，而且并不关心交互时延，那么不做复杂的 cache 管理也未必是错的。因为工程复杂度本身也是成本。

所以适用边界可以概括为：

- 短上下文、低并发：KV Cache 不是必须，收益可能有限。
- 中长上下文、在线生成：KV Cache 通常是标准答案。
- 超长上下文、超大并发：仅靠 KV Cache 不够，需要分页、调度、分布式一起上。
- 极限显存场景：先做容量预算，再谈是否开启 cache，而不是反过来。

---

## 参考资料

1. Saeed Mehrang, *KV-Caching in LLMs: The Optimization That Makes Inference Practical*  
   用途：定义与基本机制，适合作为“为什么缓存能加速自回归推理”的入门材料。

2. Ayushi Gupta 相关 KV Cache 工程文章，以及 vLLM 的 PagedAttention 资料  
   用途：工程实现与显存管理，重点看分页、碎片控制、请求复用。

3. Rajan Sethi 等关于 LLM 推理内存估算的文章  
   用途：公式与容量估算，适合推导单 token cache 大小、总 cache 大小与上下文长度关系。
