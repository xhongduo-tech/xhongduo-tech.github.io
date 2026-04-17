## 核心结论

VLM，Vision-Language Model，指“同时处理图像和文本的模型”。这类模型在推理时，真正容易卡住的地方通常不是逐字生成答案的 Decode，而是第一次把图像和提示词一起送进模型的 Prefill。原因很直接：一张图往往会被切成数百到上千个视觉 token，token 可以理解为“模型内部处理的最小片段”；这些 token 与文本 token 一起做全注意力，计算量会近似增长为

$$
\text{Prefill FLOPs} \approx O((N_{img}+N_{text})^2)
$$

当 $N_{img}\approx 1000$ 时，哪怕文本只有一百多个 token，Prefill 仍然是主要瓶颈。

如果图像在多轮对话里不变，最有效的优化不是“让模型更快看图”，而是“不要重复看图”。具体做法是把图像 token 在 Prefill 阶段生成的 KV Cache 保存下来。KV Cache 可以理解为“注意力机制后续查表要用的中间结果”。下一轮用户继续问同一张图时，系统直接复用这部分缓存，只让新文本参与计算，跳过图像 Prefill。

新手版玩具例子：把一张图拆成 1000 块，Prefill 就像第一次把 1000 块拼图的边角和颜色关系都整理好；KV Cache 就是把这份整理结果保留下来。后面再问“图里左上角是什么”“再比较一下右下角”，不需要重新整理整张图，只要在已有结果上继续回答。

只做 KV Cache 复用还不够。工程上还要把 Prefill 和 Decode 分开调度。Prefill 是 compute-heavy，也就是更吃计算；Decode 是 memory-heavy，也就是更吃显存和缓存访问。两者混在同一批 GPU worker 上，会出现计算和内存抢占，导致 TTFT，Time To First Token，指“用户看到第一个输出 token 的时间”，以及尾延迟一起变差。把图像 Prefill、文本 Decode 做成异构队列，通常能同时改善吞吐和稳定性。

手工流程可以概括为：

`图像 chunk -> 生成视觉 KV -> 写入缓存 -> 后续请求命中缓存 -> 直接进入 Decode`

---

## 问题定义与边界

先明确问题。本文讨论的不是训练，也不是图像编码器本身如何设计，而是 VLM 在线推理时，如何降低“固定图像 + 多轮问答”场景下的延迟和资源浪费。

典型流程里，图像先被视觉编码器切成视觉 token。例如高分辨率图像经过 patch 切分后，最终可能形成约 1k 个 token。文本提示再形成几十到几百个 token。进入语言模型后，模型在 Prefill 阶段一次性处理这两部分上下文，并为每一层建立 KV Cache。这里的核心代价有两个：

1. 全注意力的平方级计算。
2. 大量 KV Cache 的构建和写入。

新手版理解：把 1000 个视觉 token 想成 1000 个人，Prefill 时每个人都要“看见”其他人，这就是全注意力；人数一多，互相打招呼的总次数就暴涨。

下面这张表可以直接看出两个阶段的资源特征差异：

| 阶段 | 主要输入 | 主要瓶颈 | 资源特征 | 典型现象 |
|---|---|---|---|---|
| Prefill | 图像 token + 历史文本 + 当前提示 | 全注意力计算、KV 构建 | Compute-heavy | GPU 算力吃满，TTFT 上升 |
| Decode | 已有 KV + 新生成 token | KV 读取、显存带宽、调度 | Memory-heavy | 单 token 延迟敏感，吞吐受显存约束 |
| 多轮复问同图 | 图像不变、文本新增 | 重复视觉 Prefill 浪费 | 可缓存复用 | 明明图没变，却再次付出同样成本 |

本文的边界也要说清楚：

| 场景 | 是否适合图像 KV 复用 |
|---|---|
| 同一张图，多轮问答 | 非常适合 |
| 图片频繁变化 | 收益有限 |
| 每次都是单轮请求 | 可能不值得 |
| 视频逐帧理解 | 需要更细粒度策略，不能直接套用静态图像缓存 |
| 模型参数频繁更新 | 旧 KV 可能失效，需要额外版本管理 |

所以本文默认一个重要前提：图像内容在一段会话内是静态的，或者至少在缓存生命周期内不变。

---

## 核心机制与推导

核心机制有两层：第一层是“把图像 Prefill 结果缓存下来”，第二层是“把 Prefill 与 Decode 分开调度”。

先看复杂度。设视觉 token 数量为 $N_{img}$，文本 token 数量为 $N_{text}$。在标准全注意力里，Prefill 需要处理总长度 $N=N_{img}+N_{text}$ 的上下文，因此主要计算量近似是：

$$
\text{Cost}_{prefill} \propto N^2 = (N_{img}+N_{text})^2
$$

展开后是：

$$
(N_{img}+N_{text})^2 = N_{img}^2 + 2N_{img}N_{text} + N_{text}^2
$$

当 $N_{img}$ 很大时，主导项通常是 $N_{img}^2$ 和 $2N_{img}N_{text}$。这就是为什么“图像 token 很多”会把 Prefill 直接拉成大头。

举一个玩具例子。假设一张图生成 1024 个视觉 token，提示词和系统词一共 128 个文本 token，那么 Prefill 总长度约为 1152。此时全注意力规模和 $1152^2$ 成正比。如果下一轮用户只新增 32 个文本 token，但图像不变，重复做完整 1024 个视觉 token 的 Prefill 就很浪费。

这时就引出 KV Cache。KV Cache 指每一层注意力模块里保存的 Key 和 Value。白话说，它是“后面生成答案时继续参考上下文所需的中间记忆”。可以抽象写成：

$$
KV_{cache} \approx [layer][token]
$$

更细一点，它本质上是按层、按头、按 token 位置组织的张量，但对理解机制来说，记成 `[layer][token]` 已经足够。只要图像输入不变、模型参数不变、位置编码处理一致，那么视觉 token 对应的 KV 理论上可以复用。

Chunked Prefill 的思路就是：把视觉 token 作为一个固定块提前处理，并把这块生成的 KV 存起来。后续请求来到时，先查缓存。如果 `image_key` 命中，就把这部分 KV 直接挂到当前请求上下文里，再只对新增文本做处理。这样后续回合的成本更接近：

$$
\text{Cost}_{reuse} \approx O(N_{text,new}^2 + N_{text,new}\cdot N_{cached})
$$

它仍然不是零成本，因为新 token 还是要对已缓存上下文做注意力，但最重的“重新构建整张图的 KV”这一步被省掉了。

新手版理解：Prefill 是“做卡片”，Decode 是“翻卡片”。图像卡片只要第一次做好，后面不用再做。

再看第二层机制：异构调度。Prefill 和 Decode 在硬件上的理想运行条件不同。Prefill 更适合大矩阵计算密集的设备时间片，Decode 更在乎 KV 的驻留和显存访问稳定性。如果两者混在同一个 worker 上，会出现一种常见现象：大图像 Prefill 把计算资源占满，小请求的 Decode 被拖慢；或者 Decode 的长尾请求把显存挤住，影响后续 Prefill 排队。

因此会把系统拆成两类 worker：

1. Prefill worker：专门处理图像+文本的首轮上下文，适合 chunked prefill、批量算大矩阵。
2. Decode worker：专门处理增量生成，依赖已建好的 KV，追求稳定 token/s 和低尾延迟。

真实工程例子：产品手册问答系统里，用户上传一张固定设备结构图，然后连续追问“这个接口在哪”“散热片和风扇关系是什么”“再解释红框区域”。如果每轮都重跑图像 Prefill，TTFT 会显著上升，GPU 也会被重复工作浪费。更好的方式是首轮构建视觉 KV，后续轮次通过 `mm_hash` 命中缓存，并把 Decode 放到另一组 worker 上执行。

---

## 代码实现

实现上至少要解决三个问题：

1. 怎么唯一标识一张图。
2. 怎么在 Prefill 前查缓存。
3. 怎么把 Prefill 与 Decode 分到不同执行路径。

`mm_hash` 可以理解为“多模态输入的指纹”。白话说，就是给每张图算一个不会轻易撞上的唯一 ID。不能只看 `<image>` 占位符，因为占位符相同不代表图片相同。

下面是一个可运行的简化 Python 示例，模拟“命中图像 KV 就跳过视觉 Prefill”的逻辑：

```python
import hashlib
from dataclasses import dataclass

def mm_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

@dataclass
class KVEntry:
    image_key: str
    num_img_tokens: int
    kv_ready: bool = True

class LMCache:
    def __init__(self):
        self.store = {}

    def has(self, key: str) -> bool:
        return key in self.store and self.store[key].kv_ready

    def get(self, key: str) -> KVEntry:
        return self.store[key]

    def put(self, key: str, entry: KVEntry) -> None:
        self.store[key] = entry

def prefill_image_and_cache(cache: LMCache, image_bytes: bytes, num_img_tokens: int) -> KVEntry:
    key = mm_hash(image_bytes)
    entry = KVEntry(image_key=key, num_img_tokens=num_img_tokens, kv_ready=True)
    cache.put(key, entry)
    return entry

def infer_round(cache: LMCache, image_bytes: bytes, text_tokens: int) -> str:
    key = mm_hash(image_bytes)
    if cache.has(key):
        entry = cache.get(key)
        return f"decode_with_cached_kv(img_tokens={entry.num_img_tokens}, text_tokens={text_tokens})"
    entry = prefill_image_and_cache(cache, image_bytes, num_img_tokens=1024)
    return f"prefill_and_cache_then_decode(img_tokens={entry.num_img_tokens}, text_tokens={text_tokens})"

cache = LMCache()
img = b"same_image_content"

first = infer_round(cache, img, text_tokens=128)
second = infer_round(cache, img, text_tokens=32)

assert first.startswith("prefill_and_cache_then_decode")
assert second.startswith("decode_with_cached_kv")
assert mm_hash(img) == mm_hash(b"same_image_content")
```

如果写成伪代码，主流程通常是这样：

```python
def handle_vlm_request(image, text, cache, prefill_worker, decode_worker):
    image_key = mm_hash(image)

    if cache.has(image_key):
        visual_kv = cache.get(image_key)
        return decode_worker.generate(text=text, reused_visual_kv=visual_kv)

    visual_chunk = encode_image_to_embeddings(image)
    visual_kv = prefill_worker.chunked_prefill(visual_chunk)

    cache.put(image_key, visual_kv)
    return decode_worker.generate(text=text, reused_visual_kv=visual_kv)
```

这里的关键点有两个。

第一，`chunked_prefill` 要保证视觉 embedding 按系统约定整块提交。因为视觉 token 的位置、shape、跨层布局通常是固定假设的一部分。如果把一整块视觉 embedding 随意拆开，后面在拼接 KV 或恢复位置时容易出错。

第二，Prefill 和 Decode 最好在调度层就是两条路径。比如：

- 路径 A：接收“新图像 + 文本”的请求，发往 Prefill worker。
- 路径 B：接收“已有图像缓存 + 新文本”的请求，直接发往 Decode worker。
- 中间层：负责 `mm_hash -> KV entry` 的查找、注册、过期和一致性校验。

一个更接近线上系统的流程可写成：

1. 接收请求，提取图片 bytes 和文本。
2. 计算 `mm_hash`。
3. 查询 LMCache 或外部 KV 服务。
4. 命中则构造“仅文本增量”的 decode request。
5. 未命中则走视觉编码与 chunked prefill。
6. 写回视觉 KV 元数据与版本信息。
7. 将后续轮次统一路由到 decode 队列。

---

## 工程权衡与常见坑

真正上线时，问题通常不在“能不能缓存”，而在“缓存会不会错用”。

最常见的坑如下：

| 坑 | 问题表现 | 规避措施 |
|---|---|---|
| 只按 `<image>` 占位符做 prefix cache | 不同图片命中同一缓存，回答明显串图 | 必须使用 `mm_hash` 或 per-image fingerprint |
| 视觉 embedding 非整块提交 | shape mismatch、位置错位、KV 拼接失败 | 强制 chunked prefill 的输入协议固定 |
| 模型版本变化后复用旧 KV | 输出异常、隐性错误 | cache key 加入模型版本、分辨率、位置编码配置 |
| Prefill/Decode worker 间 KV 映射不一致 | 明明命中缓存却读取失败或复用错对象 | 统一 KV 元数据格式与跨 worker 注册流程 |
| 缓存过大 | 显存或外部缓存压力升高 | 设置 TTL、LRU、按会话热度回收 |
| 多副本调度无一致性监控 | 长尾偶发错误难排查 | 增加 KV consistency monitor 和命中率监控 |

新手版理解：这和“同名文件覆盖”很像。你不能因为文件名都叫 `image.png`，就认为内容一样。缓存系统也一样，必须认内容指纹，不能只认占位符。

再说几个容易被忽略的权衡。

第一，缓存不是免费午餐。KV Cache 占显存或外部缓存带宽。如果并发很低、图像复用率很差，维持缓存本身的成本可能超过节省的 Prefill 成本。

第二，命中率决定价值。如果用户每轮都换图，那么 `mm_hash` 每次都不同，缓存几乎不命中，这时系统复杂度上升了，收益却接近零。

第三，PD，Prefill-Decode，disaggregation，也就是“把 Prefill 与 Decode 拆开”的架构，会引入 KV 传输与注册开销。尤其当 KV 需要跨进程、跨机甚至跨网络传递时，调度收益必须覆盖通信成本。

第四，要做一致性监控。所谓 `KV consistency monitor`，可以简单理解为“检查 cache key、模型版本、shape、token 位置是否一致的守门器”。没有这层监控，错误可能不会直接报错，而是变成输出质量下降，这种问题最难查。

一个真实工程例子：客服助手接入产品结构图后，白天高峰有大量“同图多问”的请求。如果没有异构调度，大图的 Prefill 会把整卡拖入高占用区，导致小文本问答也排队。拆分成 Prefill worker 和 Decode worker 后，系统可以让首轮图像请求集中消化，后续文本轮次则稳定运行在更适合缓存读取的队列里，TTFT 和尾延迟都会更平稳。

---

## 替代方案与适用边界

图像 KV Cache 复用不是唯一方案，它只是“静态图像、多轮问答、高并发”场景下最值钱的一种。

先看决策表：

| 场景 | 推荐方案 | 优点 | 缺点 |
|---|---|---|---|
| 固定图像，多轮追问，高并发 | 图像 KV 复用 + Chunked Prefill + PD 异构调度 | TTFT 下降明显，吞吐更稳 | 系统复杂度高 |
| 图像几乎每次都变 | 不缓存或只做普通批处理 | 实现简单，不会错用缓存 | 无法节省重复视觉 Prefill |
| 低并发、单轮问答 | 仅做基础优化 | 运维简单 | 性能收益有限 |
| 模型频繁热更新 | 部分层缓存或短 TTL 缓存 | 降低失效风险 | 复用收益打折 |
| 视频/流式画面 | 增量视觉缓存或窗口化处理 | 更贴近时序场景 | 实现难度更高 |

可以把替代方案分成三类。

第一类，完全不缓存。适合图片每次都不同，或者系统还在验证阶段，不值得引入额外一致性与缓存管理逻辑。新手版理解：如果每轮都换图，缓存就像把旧照片锁进柜子，后面根本用不上。

第二类，只缓存部分层。原因是高层语义更接近任务理解，底层可能对分辨率、编码细节更敏感；或者在模型更新频繁时，只保留相对稳定的一部分缓存，降低全量失效风险。这种方案更复杂，但在一些频繁热更新的部署体系里更容易落地。

第三类，只做调度优化，不做图像 KV 复用。也就是把 Prefill/Decode 拆开，但每次仍然重新处理图像。这在“图像变化频繁但请求量大”的场景依然有价值，因为至少避免了 compute-heavy 和 memory-heavy 任务互相拖累。

因此适用边界可以简化为一句话：当图像是静态资产，而问题是围绕这张图反复展开时，图像 KV Cache 复用最有效；当图像本身频繁变化时，重点应转向调度、批处理和模型结构优化，而不是缓存复用。

---

## 参考资料

1. LMCache 博客，2025/07/03，多模态 KV Cache 扩展。支持点：说明如何为静态图像生成并复用视觉 KV，减少多轮问答中的重复 Prefill。
2. vLLM 多模态支持文档，2025。支持点：多模态输入、前缀缓存及相关实现边界，尤其是图像占位与缓存键设计问题。
3. vLLM GitHub Issue `#21175`，2025。支持点：相同 `<image>` placeholder 但图像不同导致 prefix cache 污染的风险，以及 `mm_hash`/fingerprint 的必要性。
4. SGLang / NVIDIA Dynamo 关于 PD Disaggregation 的文档，2026。支持点：Prefill 与 Decode 的异构 worker 拆分、资源解耦和尾延迟优化。
5. Introl 多模态基础设施文章，2025。支持点：VLM 推理中 Prefill 的平方级成本、视觉 token 带来的资源压力，以及调度策略的工程意义。
6. LinkedIn 工程实践分享，2025。支持点：多模态推理上线后 TTFT 和延迟飙升的现象，以及图像 KV 复用和任务拆分带来的收益。
