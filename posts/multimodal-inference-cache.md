## 核心结论

多模态推理缓存的本质，不是“让模型生成更快”，而是“避免把同一段输入重复算很多次”。这里的输入包含两部分：

1. 媒体输入的编码结果，例如图片经过视觉编码器后得到的 embedding。embedding 可以理解为“模型内部能处理的数字表示”。
2. 前缀的 KV cache。KV cache 可以理解为“模型已经看过前文后留下的注意力中间状态”。

因此，多模态推理缓存通常是两层缓存一起工作：

```text
媒体输入 -> encoder -> embedding -> prefill -> KV cache -> decode
              |            |            |
              |            |            +-- 命中 prefix cache 时跳过
              |            +--------------- 作为前缀输入的一部分
              +---------------------------- 命中 encoder cache 时跳过
```

它最有效的场景不是“每个请求都不一样”，而是“同一媒体和同一固定前缀被高频重复使用”。例如同一张商品主图被 100 个用户反复问“有没有划痕”，第一次请求需要：

1. 编码图片。
2. 把图片 embedding 和固定提示词一起做 prefill。
3. 建立对应的 KV cache。

后续请求如果图片相同、模板相同、前缀相同，就可以直接复用图片编码和前缀 KV，只计算“用户新问的问题”那部分。

结论可以压缩成一句话：多模态推理缓存节省的是重复 prefill 成本，收益来源于重复输入，边界条件是缓存 key 必须完整。

---

## 问题定义与边界

讨论多模态推理缓存时，先把三个容易混淆的概念分开：

| 类型 | 缓存对象 | 命中后跳过什么 | 典型用途 | 风险 |
| --- | --- | --- | --- | --- |
| 文本前缀缓存 | 文本 token 对应的 KV cache | 重复文本 prefill | 固定 system prompt、模板化问答 | 只适合文本前缀一致 |
| 媒体编码缓存 | 图片/视频/音频的 encoder 输出 | 重复媒体编码 | 同图问答、同视频片段摘要 | 预处理漂移会导致失效 |
| 完整推理缓存 | 最终答案 | 整个推理过程 | 完全确定性、重复问答 | 条件稍变就不能复用 |

本文讨论的是前两者如何组合，而不是缓存最终答案。原因很简单：最终答案缓存要求“问题和上下文都完全一样”，适用面很窄；而媒体编码缓存 + 前缀缓存可以覆盖更多真实请求。

边界也要说清楚。缓存命中不是“语义看起来差不多”就行，而是“影响模型输入的内容必须完全一致”。对白话解释就是：只要模型真正看到的内容变了，就不应该复用旧缓存。

下面这个玩具例子最能说明问题：

- 请求 A：图片是白鞋，问题是“这张图里有没有瑕疵？”
- 请求 B：图片是黑鞋，问题也是“这张图里有没有瑕疵？”

文本完全一样，但模型输入并不一样，因为图片不同。如果这两个请求命中同一个缓存，系统就可能把白鞋上的划痕结论错误地套到黑鞋上。这不是缓存收益问题，而是结果错误问题。

所以，多模态缓存的 key 至少要覆盖下面这些变化项：

| 可缓存对象 | 不能忽略的变化项 | 典型风险 |
| --- | --- | --- |
| 文本前缀 | chat template 版本、tokenizer 版本、system prompt | 同一句文本切词不同，KV 不可复用 |
| 图像编码 | 原图内容、resize/crop 参数、processor 版本 | 同图不同预处理，embedding 变化 |
| 视频编码 | 帧序列、采样率、起止时间、帧顺序 | 同视频不同抽帧，视觉 token 不同 |
| 音频编码 | 音频切片、采样率、特征提取器版本 | 同一句音频不同切片，声学表示不同 |
| 组合前缀 KV | 多模态输入 hash、LoRA id、`cache_salt`、租户信息 | 同文本不同图误命中，或多租户污染 |

这里的 `cache_salt` 可以理解为“人为加入的隔离因子”，用于主动打断不该共享的缓存，例如不同租户、不同实验版本。

---

## 核心机制与推导

缓存为什么有价值，核心原因在于多模态请求的前缀往往很长。

设文本 token 数为 $N_t$，多模态 token 数为 $N_m$，总长度为：

$$
N = N_t + N_m
$$

在 Transformer 中，KV cache 近似按层数、序列长度和隐藏维度线性增长。一个常见近似公式是：

$$
M_{KV} \approx 2 \times L \times N \times d_{model} \times b
$$

其中：

- $L$ 是层数。
- $d_{model}$ 是隐藏维度。
- $b$ 是每个元素字节数，bf16 通常取 2。
- 前面的 2 表示要存 K 和 V 两份。

这个公式的白话意思是：输入越长，层越多，模型越大，缓存占用越大；同样地，重复 prefill 的代价也越高。

看一个最小数值例子。假设：

- $L = 32$
- $d_{model} = 4096$
- $b = 2$ 字节
- 文本 token 数 $N_t = 128$
- 视觉 token 数 $N_m = 576$

则总长度：

$$
N = 128 + 576 = 704
$$

KV cache 大小近似为：

$$
M_{KV} \approx 2 \times 32 \times 704 \times 4096 \times 2
= 369{,}098{,}752 \text{ bytes}
\approx 352 \text{ MiB}
$$

这说明什么？说明一次请求仅前缀阶段就可能对应数百 MiB 的中间状态。如果这些前缀被重复使用，重复算它们会非常浪费。

再看收益。如果 704 个 token 里有 512 个 token 的前缀命中缓存，那么理论上跳过的前缀计算比例约为：

$$
\frac{512}{704} \approx 72.7\%
$$

通常会近似说成节省约 73% 的前缀计算。但要注意，这不是总推理耗时减少 73%。因为生成阶段，也就是 decode，仍然要继续执行。缓存优化的是 prefill，不是后续每一步生成。

为了防止误命中，前缀缓存通常不是按“整段字符串”做 key，而是按 block 粒度做哈希链。block 可以理解为“固定长度的一段 token 块”。典型形式是：

$$
h_i = H(h_{i-1}, block\_tokens_i, extra\_hashes)
$$

其中：

- $h_i$ 是第 $i$ 个 block 的哈希。
- $h_{i-1}$ 把前面所有 block 的历史带进来。
- `block_tokens_i` 是当前 block 的 token 内容。
- `extra_hashes` 是额外上下文，例如多模态输入 hash、LoRA id、`cache_salt`。

哈希链的意义是：只有“前面所有块都一样，当前块也一样，额外上下文也一样”时，当前块才认为相同。

可以用一个示意图表示：

```text
block_1: [文本 token...]
h1 = H(seed, block_1, mm_hash, template_v, lora_id, cache_salt)

block_2: [图片占位符 + 文本 token...]
h2 = H(h1, block_2, mm_hash, template_v, lora_id, cache_salt)

block_3: [继续的固定前缀...]
h3 = H(h2, block_3, mm_hash, template_v, lora_id, cache_salt)
```

只要 `mm_hash` 不同，即使文本完全一样，`h1 h2 h3` 也会不同，于是“同文本不同图”不会误命中。

下面这个表能更直观地把 token 数、显存和收益联系起来：

| 场景 | 文本 token | 多模态 token | 总 token | 近似 KV 占用 | 若命中前缀 token | 理论节省的前缀比例 |
| --- | --- | --- | --- | --- | --- | --- |
| 纯文本 FAQ | 256 | 0 | 256 | 较低 | 192 | 75% |
| 单图问答 | 128 | 576 | 704 | 约 352 MiB | 512 | 72.7% |
| 多图报告 | 256 | 1152 | 1408 | 约 704 MiB | 1024 | 72.7% |
| 短视频片段问答 | 128 | 2048 | 2176 | 更高 | 1536 | 70.6% |

真实工程例子是电商视觉问答。用户问法不同，但大部分流量集中在少量热门商品图上，例如：

- “鞋头有没有开胶”
- “左右两只颜色一致吗”
- “鞋面有划痕吗”

这些请求的问题尾部不同，但媒体输入相同，且系统提示词、输出格式、商品质检模板往往固定。于是 encoder cache 和 prefix cache 都有较高复用率。

---

## 代码实现

工程上建议把缓存拆成两层，而不是做成一个大而全的黑盒。

1. `encoder_cache`：按媒体内容或 `multi_modal_uuids` 缓存编码结果。
2. `prefix_cache`：按 block hash 缓存前缀 KV。

下面是一个简化但可运行的 Python 玩具实现。它不真的跑模型，只模拟 key 设计、命中逻辑和“同文本不同图不能误命中”这个关键约束。

```python
import hashlib
from dataclasses import dataclass

def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class Request:
    image_bytes: bytes
    prompt_prefix: str
    user_question: str
    template_version: str
    lora_id: str
    cache_salt: str

class EncoderCache:
    def __init__(self):
        self.store = {}

    def get_or_compute(self, mm_hash: str):
        if mm_hash not in self.store:
            self.store[mm_hash] = f"image_embedding:{mm_hash[:8]}"
        return self.store[mm_hash]

class PrefixCache:
    def __init__(self):
        self.store = {}

    def get(self, block_hash: str):
        return self.store.get(block_hash)

    def put(self, block_hash: str, kv_value: str):
        self.store[block_hash] = kv_value

def compute_mm_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

def build_prefix_key(req: Request, mm_hash: str, embedding: str) -> str:
    normalized = "|".join([
        req.prompt_prefix.strip(),
        req.template_version,
        req.lora_id,
        req.cache_salt,
        mm_hash,
        embedding,
    ])
    return sha256_hex(normalized)

def handle_request(req: Request, encoder_cache: EncoderCache, prefix_cache: PrefixCache):
    mm_hash = compute_mm_hash(req.image_bytes)
    embedding = encoder_cache.get_or_compute(mm_hash)

    prefix_key = build_prefix_key(req, mm_hash, embedding)
    kv = prefix_cache.get(prefix_key)

    if kv is None:
        kv = f"kv_for:{prefix_key[:8]}"
        prefix_cache.put(prefix_key, kv)
        prefetched = False
    else:
        prefetched = True

    answer_stub = f"use {kv}, decode question={req.user_question}"
    return {
        "mm_hash": mm_hash,
        "embedding": embedding,
        "prefix_key": prefix_key,
        "prefix_hit": prefetched,
        "answer_stub": answer_stub,
    }

encoder_cache = EncoderCache()
prefix_cache = PrefixCache()

req1 = Request(
    image_bytes=b"white-shoe-image",
    prompt_prefix="请判断商品主图是否存在瑕疵，并按JSON输出",
    user_question="鞋头有没有划痕？",
    template_version="v1",
    lora_id="base",
    cache_salt="tenant-a",
)

req2 = Request(
    image_bytes=b"white-shoe-image",
    prompt_prefix="请判断商品主图是否存在瑕疵，并按JSON输出",
    user_question="鞋面有没有脏污？",
    template_version="v1",
    lora_id="base",
    cache_salt="tenant-a",
)

req3 = Request(
    image_bytes=b"black-shoe-image",
    prompt_prefix="请判断商品主图是否存在瑕疵，并按JSON输出",
    user_question="鞋面有没有脏污？",
    template_version="v1",
    lora_id="base",
    cache_salt="tenant-a",
)

r1 = handle_request(req1, encoder_cache, prefix_cache)
r2 = handle_request(req2, encoder_cache, prefix_cache)
r3 = handle_request(req3, encoder_cache, prefix_cache)

assert r1["prefix_hit"] is False   # 第一次没有命中
assert r2["prefix_hit"] is True    # 同图同前缀命中
assert r1["prefix_key"] == r2["prefix_key"]
assert r3["prefix_key"] != r2["prefix_key"]  # 同文本不同图，不能命中
```

这段代码体现了真实系统中的关键顺序：

```text
normalize input
-> compute mm_hash
-> encoder_cache.get(mm_hash)
-> build block/prefix hash with mm_hash + template_version + lora_id + cache_salt
-> prefix_cache.get(block_hash)
-> prefill if miss
-> decode user suffix
```

如果写成更接近生产系统的伪代码，可以是：

```text
request -> normalize input
        -> compute media hashes / multi_modal_uuids
        -> lookup encoder_cache by media hash
        -> if miss: run encoder and store embedding
        -> build token blocks for fixed prefix
        -> compute block hashes with extra_hashes
        -> lookup prefix_cache
        -> if miss: run prefill, materialize KV blocks, store cache
        -> append user-specific suffix
        -> decode
```

实现时最容易错的不是“缓存结构怎么写”，而是“key 是否完整”。少一个关键字段，系统就可能在低流量时看起来没问题，在高流量时悄悄给出错答案。

---

## 工程权衡与常见坑

多模态推理缓存不是“加上就赚”的优化，它本质上是“拿更多状态管理复杂度换吞吐收益”。下面这些坑最常见。

| 坑 | 现象 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 错误 key 设计 | 同文本不同图误命中 | key 没包含媒体 hash 或 `multi_modal_uuids` | 把媒体 hash、模板版本、LoRA、`cache_salt` 一起纳入 |
| block 边界 miss | 看起来前缀几乎相同却不命中 | prefix cache 只缓存完整 block | 让常见模板对齐 block，接受局部 miss |
| 预处理版本漂移 | 发布后缓存命中率骤降 | resize/crop/tokenizer/processor 改了 | 固定预处理版本，把版本号写进 key |
| 高并发 LRU 抖动 | 热点缓存刚写入就被淘汰 | 热点大、容量小、并发高 | 分租户、分热点池、调大 block 池 |
| 多租户污染 | A 租户命中 B 租户缓存 | 隔离字段缺失 | 把 tenant id 或 `cache_salt` 放进 key |

先看第一个坑。只按文本做 key，是最危险的错误，因为它不是“少赚点性能”，而是“直接答错”。这是业务事故，不是优化失误。

第二个坑是 block 边界 miss。很多前缀缓存系统按固定 block 长度缓存，只有完整 block 才能复用。于是两个请求虽然前 500 个 token 一样，但如果系统 block 大小是 128，而其中一个请求在第 384 到 511 token 之间插入了少量差异，后续块就都可能 miss。这个问题通常不能完全消除，只能通过模板设计、输入归一化和 block 对齐来降低损失。

第三个坑是版本漂移。电商场景很典型：用户都在问同一商品的不同细节，原本命中率很高；结果某次发布把图片 resize 从 448 改成 512，或者把 processor 升级了一个小版本，缓存瞬间整体失效。这里的教训是：缓存复用依赖“输入规范稳定”，不是只依赖“原始媒体看起来一样”。

第四个坑是高并发 LRU 抖动。LRU 可以理解为“最近最少使用淘汰”。在热点媒体很多、每个前缀又很大时，缓存池会反复把刚热起来的 KV 挤掉，形成抖动。结果是 GPU 显存和 CPU 管理开销都上去了，但命中率不稳定。常见做法是：

- 给热门模板单独保留容量。
- 按租户或场景分片。
- 对明显一次性请求不入缓存。
- 对媒体 hash 做热点统计，只缓存高复用对象。

真实工程例子还是电商质检服务。你会发现收益最大的，不一定是“模型最贵的请求”，而是“重复率最高的那批请求”。热门商品图、固定模板、固定输出 schema，往往远比长尾用户上传图更适合缓存。

---

## 替代方案与适用边界

不是所有多模态系统都值得做推理缓存。判断标准不是“模型慢不慢”，而是“重复输入多不多”。

下面这个表可以直接用于初筛：

| 维度 | 高 | 低 | 适用性判断 |
| --- | --- | --- | --- |
| 重复度高 | 同媒体被反复问 | 每次都是新媒体 | 重复度高更适合 |
| 媒体稳定 | 图片/视频片段固定 | 每次上传新内容 | 媒体稳定更适合 |
| prompt 固定 | 模板统一、输出格式固定 | 提示词经常变化 | prompt 固定更适合 |
| 并发高 | 热点对象有集中流量 | 请求分散 | 并发高更能摊薄缓存成本 |
| 延迟敏感 | 需要快速首 token | 允许稍高延迟 | 命中后可明显改善首包延迟 |

可以进一步给出结论：

| 场景 | 是否适合多模态推理缓存 | 原因 |
| --- | --- | --- |
| 电商商品图问答 | 很适合 | 同图高频复用，模板稳定 |
| 视频监控固定片段告警解释 | 适合 | 热点片段可能反复分析 |
| OCR 后固定表单抽取 | 适合 | 版式和前缀高度稳定 |
| 创作类图片助手 | 不太适合 | 每次新图新问题，命中率低 |
| 个性化多轮开放对话 | 边际有限 | 上下文变化快，前缀不稳定 |

一次性创作类应用就是反例。用户每次都上传新图，再问新的开放问题，例如“把这张旅行照改写成电影海报文案”。这种场景里：

- 媒体几乎每次都变。
- 提示词也常常不稳定。
- 业务更在意生成质量和交互速度，而不是高并发吞吐。

此时多模态推理缓存命中率通常很低，反而增加实现复杂度、显存占用和排障成本。

如果业务目标是低延迟，而不是高重复吞吐，那么更直接的优化手段往往是：

- 减少输入长度。
- 做更激进的量化。
- 优化批处理和调度。
- 缩短多模态 token 长度，例如减少帧数、降分辨率、裁剪无关区域。

这些方案未必比缓存“高级”，但在低复用场景里经常更有效。

---

## 参考资料

下表按“机制层 -> 实现层 -> 系统层”组织，用来对应本文各章节。

| 来源名称 | 对应章节 | 可引用结论 | 适合放在哪一段 |
| --- | --- | --- | --- |
| vLLM Automatic Prefix Caching | 核心机制与推导、工程权衡 | block hash、`extra_hashes`、完整 block 命中规则 | 讲哈希链与 block 边界时 |
| vLLM Multimodal Inputs / Cached Inputs | 问题定义与边界、代码实现 | 多模态输入可按内容哈希，支持 `multi_modal_uuids` | 讲媒体 hash 和 key 完整性时 |
| vLLM `EncoderCacheManager` API | 代码实现 | encoder cache 可按多模态 item 级别管理 | 讲两层缓存拆分时 |
| PagedAttention 论文 | 核心机制与推导 | KV cache 分块管理和共享的系统基础 | 讲 KV 为什么值得缓存时 |
| vLLM-Omni Prefix Caching | 代码实现、工程权衡 | 多模态 block 边界、encoder cache 与 prefix cache 的配合 | 讲真实多模态实现时 |

1. [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)：对应本文“核心机制与推导”，重点支撑 block hash、`extra_hashes` 和完整 block 命中规则。
2. [vLLM Multimodal Inputs / Cached Inputs](https://docs.vllm.ai/en/v0.15.0/features/multimodal_inputs/)：对应本文“问题定义与边界”和“代码实现”，重点支撑媒体内容哈希、`multi_modal_uuids` 与多模态输入缓存。
3. [vLLM EncoderCacheManager API](https://docs.vllm.ai/en/latest/api/vllm/v1/core/encoder_cache_manager/)：对应本文“代码实现”，重点支撑 encoder cache 与 prefix cache 分层设计。
4. [PagedAttention: Efficient Memory Management for LLM Serving with PagedAttention](https://openreview.net/forum?id=QBUHGSFTid)：对应本文“核心机制与推导”，重点支撑 KV cache 分块共享的系统背景。
5. [vLLM-Omni Prefix Caching Design](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/prefix_caching/)：对应本文“工程权衡与常见坑”，重点支撑多模态场景中的 block 边界和缓存协同问题。
