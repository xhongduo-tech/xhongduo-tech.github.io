## 核心结论

多模态推理流水线，不是“把图片丢给大模型”这么简单，而是一个有明确阶段划分的串联系统：预处理、特征提取、对齐融合、解码生成。这里的“串联系统”可以直接理解为前一段没完成，后一段通常就不能完整开始。于是，整体体验不只由模型本身决定，还由排队、缓存、批处理和资源分配共同决定。

在线请求的总耗时可以先写成：

$$
T_{\text{req}} = T_{\text{pre}} + T_{\text{enc}} + T_{\text{fus}} + T_{\text{dec}} + W_q
$$

其中，$T_{\text{pre}}$ 是预处理时间，$T_{\text{enc}}$ 是模态编码时间，$T_{\text{fus}}$ 是融合时间，$T_{\text{dec}}$ 是解码生成时间，$W_q$ 是队列等待时间。这里的“队列等待”就是请求没开始算，而是在等 CPU 线程、GPU 卡位或 batch 拼满。

稳态吞吐上限通常近似受最慢阶段限制：

$$
Q_{\max} \approx \min\left(\frac{1}{T_{\text{pre}}}, \frac{1}{T_{\text{enc}}}, \frac{1}{T_{\text{fus}}}, \frac{1}{T_{\text{dec}}}\right)
$$

这不是严格定理，而是工程近似：哪一段最慢，整条链路就最容易堵在那一段。

下面这个表可以先建立直觉：

| 阶段 | 主要工作 | 主要资源 | 常见瓶颈 |
| --- | --- | --- | --- |
| 预处理 | 解码、resize、采样、归一化 | CPU、内存带宽 | 图片/音频解码慢，主线程阻塞 |
| 编码 | 图像/音频/视频变 embedding | GPU、显存 | encoder 重，显存占用高 |
| 融合 | 把多模态特征接入 LLM | GPU、显存 | token 对齐、跨模态桥接 |
| 解码 | 自回归生成文本 | GPU、KV cache | 长输出拉高尾延迟 |
| 排队 | 等 worker、等 batch、等 GPU | 调度器、队列 | TTFT 和 p95/p99 恶化 |

一个新手可理解的玩具例子：用户上传一张商品图并问“这双鞋适合跑步吗”。系统先把 JPEG 解码，再做 resize，再送入视觉编码器得到向量特征，再把这些特征和用户问题一起交给语言模型生成回答。只要其中一段排队，用户感受到的就是“第一下很慢”。

---

## 问题定义与边界

本文讨论的是在线多模态推理流水线。这里的“在线”指用户发出请求后，系统需要在当前会话里尽快返回结果，而不是离线慢慢跑完再写入数据库。

更具体地说，本文讨论的对象是：

| 项目 | 是否包含 | 说明 |
| --- | --- | --- |
| 图像/音频/视频输入读取 | 包含 | 输入进入模型前的第一步 |
| 预处理与标准化 | 包含 | 例如 resize、重采样、分帧 |
| 模态编码 | 包含 | 用 encoder 提取特征 |
| 融合与语言生成 | 包含 | 把特征接到 LLM 并输出文本 |
| 在线缓存 | 包含 | 例如 embedding cache |
| 离线训练 | 不包含 | 与线上推理目标不同 |
| 纯检索系统 | 不包含 | 没有真正的模态编码与生成链路 |
| 业务后处理编排 | 部分不包含 | 只在影响核心推理时提及 |

再看阶段定义：

| 阶段 | 输入 | 输出 | 主资源 | 是否可并行 |
| --- | --- | --- | --- | --- |
| 预处理 | 原始文件或字节流 | 模型输入张量 | CPU | 常可并行 |
| 编码 | 张量 | embedding | GPU | 可做批处理 |
| 融合 | embedding + prompt | LLM 可消费表示 | GPU | 部分可并行 |
| 解码 | 融合后上下文 | 文本 token | GPU | 批量受限 |
| 缓存 | 输入或中间结果 key | 命中后的复用结果 | CPU/内存/存储 | 可独立扩展 |

几个术语先定清楚。

`TTFT`，first token time，首 token 时间，指用户从发起请求到看到第一个输出 token 的时间。对聊天系统来说，它比“全部生成完的总时间”更接近用户的主观感受。

`batching`，批处理，指把多个请求拼成一批一起算。它的白话含义是“牺牲一点等待时间，换更高硬件利用率”。

`embedding cache`，向量缓存，指把已经算过的模态特征保存起来，下次遇到相同或等价输入时直接复用，避免重复编码。

所以，本文不是讨论“哪个多模态模型更强”，而是讨论“一个多模态请求从进来到出去，链路为什么慢、慢在哪、如何拆开优化”。

---

## 核心机制与推导

核心机制有两个：串行依赖，和排队换吞吐。

先看串行依赖。假设单请求耗时如下：

- 预处理：8 ms
- 编码：24 ms
- 融合：4 ms
- 解码：14 ms

那么理想情况下：

$$
T_{\text{req}} = 8 + 24 + 4 + 14 = 50 \text{ ms}
$$

如果动态 batch 为了攒够请求又额外等了 20 ms，那么：

$$
T_{\text{req}} = 50 + 20 = 70 \text{ ms}
$$

这说明一个关键事实：算得快，不等于返回快。因为用户感受到的是总链路，而不是单个 kernel 的 benchmark。

为什么最慢阶段决定吞吐上限？因为稳态下每段都像一个服务台。若编码阶段每个请求平均要 24 ms，那么它最多每秒处理大约 $1 / 0.024 \approx 41.7$ 个请求。即使预处理和融合更快，流量也会在编码前堆积。这就是瓶颈传递。

再看批处理。批处理提升吞吐的原因是 GPU 更喜欢“大块连续工作”，不喜欢频繁切换小任务。但批处理的代价是等待。于是就出现典型交换：

- batch 小：等待少，TTFT 好，吞吐可能偏低
- batch 大：吞吐高，等待多，TTFT 变差

这也是为什么工程里不能只报平均延迟。平均值可能很好看，但 p95 和 p99 很差。这里的 `p95/p99` 指 95% 或 99% 请求都不超过的延迟阈值，用来描述尾部慢请求。

模型结构也会影响流水线形态。下面用两个代表性方案做对比：

| 方案 | 关键机制 | 优点 | 代价 |
| --- | --- | --- | --- |
| BLIP-2 | 冻结视觉编码器和 LLM，中间加轻量 Querying Transformer | 训练与迁移成本低，桥接清晰 | 融合能力受中间桥接容量影响 |
| Flamingo | 通过跨注意力处理交错视觉-文本序列 | 适合图文交错上下文 | 系统更复杂，推理侧资源更敏感 |

这里的“冻结”可以理解为原模型参数不再大规模改动，只训练连接模块。对于部署者来说，这通常意味着系统模块边界更清楚，更容易拆开看性能。

真实工程例子：电商图文客服。高峰期用户不断上传商品图并追问细节。图像预处理在 CPU 上完成，视觉 encoder 占用 GPU，融合后进入 LLM 解码。如果同一张商品主图被不同用户反复询问，那么视觉编码其实不该每次重算，而应走 embedding cache。否则 GPU 很快被重复工作吞掉，TTFT 和队列长度都会恶化。

---

## 代码实现

实现时最忌讳把全部逻辑塞进一个 `infer()` 黑盒。正确做法是按阶段拆模块，并保证每段都可观测、可缓存、可替换。

下面是一个可运行的玩具实现。它不依赖真实模型，但保留了流水线结构、缓存逻辑和时延统计方式。

```python
import hashlib
from dataclasses import dataclass

@dataclass
class Request:
    image_bytes: bytes
    text: str

class EmbedCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

def stable_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def preprocess(image_bytes: bytes):
    # 玩具实现：把原始字节长度当作“已解码尺寸”
    assert isinstance(image_bytes, (bytes, bytearray))
    return {"pixels": len(image_bytes), "normalized": True}

def encode(mm_tensor):
    # 玩具实现：把输入压成一个稳定 embedding
    pixels = mm_tensor["pixels"]
    return [pixels % 7, pixels % 11, pixels % 13]

def fuse(text: str, mm_embed):
    # 融合：把文本和多模态特征拼成统一输入
    return {
        "prompt": text,
        "image_tokens": mm_embed,
        "token_count": len(text.split()) + len(mm_embed),
    }

def decode(fused):
    # 玩具实现：模拟语言模型输出
    return f"answer(tokens={fused['token_count']}, image_tokens={fused['image_tokens']})"

def infer(request: Request, cache: EmbedCache):
    key = stable_hash(request.image_bytes)
    mm_embed = cache.get(key)
    cache_hit = mm_embed is not None

    if not cache_hit:
        mm_tensor = preprocess(request.image_bytes)
        mm_embed = encode(mm_tensor)
        cache.set(key, mm_embed)

    fused = fuse(request.text, mm_embed)
    answer = decode(fused)
    return {
        "answer": answer,
        "cache_hit": cache_hit,
        "embedding": mm_embed,
    }

cache = EmbedCache()
req = Request(image_bytes=b"fake-image-content", text="describe this product")

r1 = infer(req, cache)
r2 = infer(req, cache)

assert r1["cache_hit"] is False
assert r2["cache_hit"] is True
assert r1["embedding"] == r2["embedding"]
assert "answer(" in r1["answer"]
```

这个例子虽然简单，但已经体现了线上系统最重要的拆分方式：

| 模块 | 职责 | 典型优化方向 |
| --- | --- | --- |
| `stable_hash` | 生成稳定输入标识 | 避免误判与重复计算 |
| `preprocess` | 输入标准化 | 异步化、并行化、CPU 池 |
| `encode` | 提取模态 embedding | 批处理、半精度、拆分部署 |
| `fuse` | 构造统一模型输入 | 校验 token 对齐 |
| `decode` | 生成文本结果 | 降低 TTFT、优化 KV cache |
| `cache` | 复用中间结果 | TTL、容量控制、一致性 |

真实工程里，这个骨架通常会扩展成：

1. 输入层负责文件校验、解码、resize、采样。
2. 编码层单独部署，必要时与解码层分离。
3. 调度层控制 `max_queue_delay`、batch 上限和优先级。
4. 观测层按阶段记录 `T_pre`、`T_enc`、`T_fus`、`T_dec`、`W_q`。
5. 缓存层对重复图、重复音频片段或重复视频帧做复用。

如果是视频理解，预处理还会多出“抽帧”这一步。抽帧本质上是在长序列里选一小部分代表性帧，否则 encoder 和显存压力会直接爆炸。

---

## 工程权衡与常见坑

工程上最常见的问题，不是模型精度，而是链路组织错误。

| 常见坑 | 表现 | 原因 | 规避方法 |
| --- | --- | --- | --- |
| resize/OCR/重采样塞进主线程 | TTFT 偏大 | CPU 阻塞前置路径 | 独立 worker，异步队列 |
| 只看平均延迟 | 线上偶发卡顿 | 尾延迟被均值掩盖 | 必看 p95/p99 和队列长度 |
| batch 过大 | 首 token 很慢 | 等待时间 $W_q$ 变长 | 设 `max_queue_delay` |
| `<image>` 与特征数错位 | 输出异常或直接报错 | 融合协议不一致 | 使用模型专用 processor |
| 重复图像重复计算 | GPU 浪费严重 | 没有 embedding cache | 用稳定 hash 复用编码结果 |

这里最容易被忽视的是“平均值陷阱”。例如平均 TTFT 只有 180 ms，看起来不错，但 p99 达到 2 s，用户仍会觉得系统经常卡顿。原因通常不是模型突然变慢，而是队列在峰值期爆了。

再看缓存。缓存不是无条件越大越好。缓存 key 设计如果太粗，会把不同输入误判为相同；如果太细，又命中率过低。工程上一般至少要保证输入内容哈希稳定，并考虑 processor 版本、采样参数、模型版本是否进入 key。

再看资源拆分。多模态系统经常出现“CPU 很满，GPU 也很满，但整体吞吐并不高”的情况。根因通常是阶段之间没有解耦，导致 CPU 解码抖动把 GPU 喂不饱，或者 GPU 编码把解码阶段挤占。拆分部署和异步队列的价值，就在于把抖动隔离开。

---

## 替代方案与适用边界

没有一种多模态架构适合所有业务。应按请求类型、复用率、交互时延目标来选。

| 方案 | 适用场景 | 优点 | 缺点 | 对延迟影响 | 对吞吐影响 |
| --- | --- | --- | --- | --- | --- |
| BLIP-2 风格桥接 | 图文问答、轻量接入 | 模块边界清晰，易迁移 | 融合能力受桥接模块限制 | 中等 | 中等偏好 |
| Flamingo 风格交错建模 | 图文交错上下文、多轮理解 | 跨模态上下文更强 | 系统复杂，部署更重 | 偏高 | 依实现而定 |
| 端到端单体式方案 | 希望统一优化、结构简洁 | 模型内部协同强 | 难拆分，调优空间少 | 可能较低 | 扩缩容不灵活 |
| encoder-decoder 拆分部署 | 高并发、输入复用高 | 可独立扩展，适合缓存 | 系统调度更复杂 | TTFT 可明显改善 | 吞吐通常更好 |

什么时候选哪类方案，可以直接按业务约束判断。

如果业务主要是图文客服、商品审核、知识问答，输入图像重复率高，优先考虑桥接式架构加 embedding cache，必要时做 encoder-decoder 拆分部署。

如果业务是长视频理解或多帧跨时间关系建模，真正的难点通常不在文本生成，而在视频采样、跨帧对齐和显存控制。这时单纯优化 LLM 解码帮助有限。

如果业务强调强交互、低首包延迟，比如实时助手或边看边问，那么应优先压低 $W_q$ 和 `TTFT`，而不是一味追求极致吞吐。因为用户首先感知到的是“多久看到第一句”，不是 GPU 利用率报表。

---

## 参考资料

| 来源类型 | 用途 | 对应章节 |
| --- | --- | --- |
| 论文 | 理解桥接式多模态架构 | 核心机制与推导 |
| 论文 | 理解交错视觉-文本建模 | 核心机制与推导 |
| 工程文档 | 理解动态 batching 与排队权衡 | 工程权衡与常见坑 |
| 工程文档 | 理解多模态输入处理 | 代码实现 |
| 工程文档 | 理解 encoder 拆分部署 | 替代方案与适用边界 |

1. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)
2. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
3. [NVIDIA Triton Inference Server: Dynamic Batcher](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)
4. [vLLM Multi-Modal Data Processing](https://docs.vllm.ai/en/v0.13.0/design/mm_processing/)
5. [vLLM Disaggregated Encoder](https://docs.vllm.ai/en/v0.12.0/features/disagg_encoder/)
6. [vLLM Input Processing](https://docs.vllm.ai/en/v0.5.1/dev/input_processing/model_inputs_index.html)
