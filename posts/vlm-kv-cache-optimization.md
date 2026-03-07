## 核心结论

视觉语言模型（VLM，Vision-Language Model，指同时处理图像与文本的模型）的 KV Cache 优化，核心不是“让模型少生成几个字”，而是“不要为同一批视觉 token 重复做昂贵计算”。KV Cache 的作用，是把历史 token 的 key/value 张量保存在显存或外部缓存里，后续解码时直接读取，不再为旧上下文重复执行注意力投影。对纯文本模型，这已经是标准工程手段；对 VLM，难点在于图像会额外引入大批视觉 token。单张图常见会展开成 256 到 1024 个视觉 token，在某些实现中，经过占位和拼接后会形成更长的多模态前缀，因此显著抬高 prefill 延迟、KV 占用和首 token 时间（TTFT）。

现阶段最直接、最工程化的优化路径，基本可以分成三类：

| 路径 | 解决对象 | 核心动作 | 适合场景 |
| --- | --- | --- | --- |
| 图像级复用 | 相同图片被反复问 | 用 `mm_hash` 或等价图像身份标识，直接复用历史 KV | 多轮对话、图像客服、审核平台 |
| token 级压缩 | 视觉 token 太多 | 合并相似 token，缩短进入解码器的序列长度 | 新图首问也要加速 |
| hidden 级压缩 | KV 本身太大 | 对 K/V 做低秩投影、量化或混合压缩 | 显存紧张、长上下文、高并发 |

这三类方案分别对应三个不同问题：

1. 同一张图重复出现，浪费了重复 prefill。
2. 单张图 token 太长，第一次请求本身就慢。
3. 即使不重复、不裁 token，KV 宽度也已经大到拖慢系统。

LMCache 在 vLLM V1 的多模态栈上，利用图像 `mm_hash` 让相同图片能够命中 KV Cache。其公开示例中，同一张图第二次请求时，KV 命中数接近总 token 数，TTFT 从约 18 秒下降到约 1 秒。这说明在很多真实业务里，最值钱的优化不是重新训练模型，而是避免把同一张图再编码一遍。

SGLang 的 RadixAttention 更进一步。它可以理解成“把公共前缀组织成树结构的前缀缓存”，因此不仅能复用系统提示词，还能复用示例图像、历史轮次和共享模板。SGLang 论文报告，在多模态基准上可达到最高约 6 倍吞吐；在 Chatbot Arena 的线上部署中，观察到 52.4% 到 74.1% 的缓存命中率。这里的重点不是具体数字本身，而是一个工程事实：多模态请求并不是天然“每次都全新”，它们往往存在可重复前缀。

一个直观例子是发票问答。用户连续两次上传同一张发票，第一次系统需要执行视觉编码器，把图像切块、投影成视觉 token，再进入语言解码器建立 KV；第二次如果图片完全一致，而且图像插入位置与前缀结构不变，那么系统完全可以直接取回旧 KV。这和浏览器缓存同一网页资源是同一类思想，只是缓存对象从静态文件变成了注意力状态。

---

## 问题定义与边界

先把问题严格定义清楚。VLM 推理一般分为两个阶段：

1. 预填充（prefill）：一次性处理输入图像与文本，建立初始 KV Cache。
2. 解码（decode）：逐 token 生成输出，每一步读取已有 KV 并只为新 token 追加 K/V。

真正昂贵的部分通常在 prefill，尤其是图像很多或分辨率较高时。原因不复杂：文本前缀长度通常几十到几百个 token，而一张图可以轻易带来数百个视觉 token。若每次都重新编码图像，那么 prefill 计算量大致随视觉 token 数量增长。

把问题写成一个粗略公式更容易理解。设：

- 文本 token 数为 $T$
- 视觉 token 数为 $V$
- 层数为 $L$
- 每层 KV 宽度为 $d$
- 数据类型字节数为 $b$，例如 FP16 时 $b=2$

则 KV Cache 的近似显存占用可写成：

$$
\text{KV bytes} \approx 2 \times L \times (T+V) \times d \times b
$$

前面的 `2` 表示 K 和 V 两份。这个公式不是精确实现细节，而是足够有用的工程近似。它告诉我们：在 VLM 里，问题往往不是文本太长，而是 $V$ 突然很大。

如果进一步把 prefill 的注意力代价看成序列长度的函数，那么视觉 token 过多还会抬高激活访存和 kernel 执行时间。实际系统里，prefill 延迟通常可粗略理解为：

$$
\text{Prefill latency} \approx \text{VisionEncode}(V) + \text{LLM\_Prefill}(T+V)
$$

因此，VLM 的 KV Cache 优化至少有两层含义：

1. 避免重复执行 `VisionEncode(V)`。
2. 减少进入 `LLM_Prefill(T+V)` 的有效长度或宽度。

这类优化的边界也必须说清楚。不是“看起来差不多”就能复用。

| 边界问题 | 能否直接复用 | 原因 |
| --- | --- | --- |
| 同一张图、相同前缀、相同位置 | 可以 | 图像内容、顺序、位置编码都一致 |
| 同一张图、系统提示变了 | 通常不能整段复用 | 文本前缀变化会影响后续 token 对齐 |
| 同一张图、插入位置不同 | 不能直接整段复用 | 位置编码和前缀切分变化 |
| 不同图、文本恰好相同 | 绝对不能复用图像 KV | 会把错误视觉信息带入当前请求 |
| 相似图、裁剪图、压缩图 | 一般不能做精确复用 | 当前主流方案大多要求内容完全一致 |

因此，多模态缓存键通常不能只看文本前缀。更合理的抽象是：

$$
\text{cache\_key} = f(\text{text prefix}, \text{mm\_hash}, \text{position}, \text{model config})
$$

其中：

- `text prefix` 表示图像之前和周围的可复用文本结构
- `mm_hash` 表示图像身份
- `position` 表示图像 token 在整段序列中的插入位置
- `model config` 表示模型版本、图像处理器参数、分辨率策略等上下文

新手最容易忽略的是“占位符问题”。很多 VLM 在 prompt 里并不是直接写入真实图像内容，而是先插入类似 `<image>` 的占位符，再由底层把图像编码结果拼接进去。如果缓存系统只看这些占位符，而没有把真实图像身份映射进去，那么“用户 A 上传猫图”和“用户 B 上传狗图”在 prefix match 阶段看起来可能是同一个前缀。这不仅会导致结果错误，还可能造成跨请求的数据泄露。

一个更工程化的判断规则是：只要你的缓存系统无法回答“这个 `<image>` 到底是不是同一张图”，那它就还没有真正支持多模态 KV 复用。

---

## 核心机制与推导

第一类机制是图像级复用。它不减少 token，也不压缩 hidden 维度，而是直接避免重复计算。核心前提是“同图、同位置、同前缀结构”。这类方法对重复图像场景收益最大，因为一旦命中，可以直接跳过视觉编码器和大部分 prefill。

LMCache 的公开示例很有代表性：

| Query | Total tokens | KV hits | Hit rate | 结果解读 |
| --- | --- | --- | --- | --- |
| First image | 16,178 | 0 | 0% | 冷启动，必须完整 prefill |
| Same image (2nd) | 16,178 | 16,177 | 约 100% | 基本全命中，TTFT 约从 18s 降到 1s |
| New image (3rd) | 4,669 | 0 | 0% | 图片变了，不能复用 |

这个结果说明，图像级复用本质上是在压“请求次数”维度。假设同一张图被问 $N$ 次，单次完整 prefill 成本为 $C$，缓存命中后成本为 $c$，且 $c \ll C$，则总成本从：

$$
N \cdot C
$$

下降为：

$$
C + (N-1)\cdot c
$$

当 $N$ 稍大时，收益非常直接。

第二类机制是 token 级压缩。以 LightVLM 的 Pyramid Token Merging 为代表，这类方案在视觉编码或跨层传播阶段，把相似视觉 token 合并成更少的代表 token，从而减少后续层和解码器前缀长度。它压缩的是“序列有多长”。

一个简化表达可以写成：

$$
X_l' = \text{Merge}(X_l, S_l)
$$

其中：

- $X_l$ 表示第 $l$ 层的 token 表示
- $S_l$ 表示相似度、重要性或聚合分数
- $X_l'$ 表示压缩后的 token 序列

若进一步展开到注意力缓存层面，可以把其效果理解为：

$$
K_l' = P_l K_l,\qquad V_l' = P_l V_l
$$

这里 $P_l$ 不是普通的方阵，而是一个“聚合/选择矩阵”，把原来较长的 token 序列映射到更短的序列。压缩后，序列长度由 $V$ 降为 $\hat V$，其中 $\hat V < V$。那么后续 KV 占用也会从：

$$
2 \times L \times (T+V) \times d
$$

下降为：

$$
2 \times L \times (T+\hat V) \times d
$$

这类方法的优点是即使是第一次见到的新图，也能提速。缺点是如果合并过猛，图像局部细节、小目标、OCR 信息往往先受损。

第三类机制是 hidden 级压缩。Palu 代表的路线不减少 token 数量，而是压缩每个 token 的表示宽度。它处理的是“每个 token 有多宽”，而不是“有多少个 token”。

其基本思想可以用低秩近似来理解：

$$
K \approx \tilde K = A_k B_k,\qquad V \approx \tilde V = A_v B_v
$$

或者写成熟悉的分解形式：

$$
K \approx U_k \Sigma_k V_k^\top,\qquad V \approx U_v \Sigma_v V_v^\top
$$

这里的要点不是必须做标准 SVD，而是通过低秩投影把原本宽度为 $d$ 的 K/V 压到更小的中间表示 $r$，其中 $r \ll d$。于是 KV 占用可以从与 $d$ 成正比，降到近似与 $r$ 成正比。若压缩比为 $\rho = r/d$，则缓存占用近似变为：

$$
\text{KV compressed} \propto 2 \times L \times (T+V) \times r
$$

这类方法的优点是对长上下文和大模型很有吸引力，因为它不依赖“是否同图重复”。缺点是仅有数学压缩还不够，执行内核通常要一起改，否则恢复近似 K/V 的开销会抵消收益。

把三类方法放在一起看，就很清楚了：

| 方法 | 压缩对象 | 本质 | 直接收益 |
| --- | --- | --- | --- |
| `mm_hash` 复用 | 重复请求次数 | 同图不再重算 | 降低 TTFT、减少视觉重复 prefill |
| Token Merging | 序列长度 | 相似 token 合并 | 新图首问也能提速 |
| Low-Rank / Quantization | hidden 宽度 | 每个 token 的 K/V 更小 | 降低显存、提高并发 |

它们并不互斥。一个实际系统完全可以按下面顺序叠加：

1. 先做精确缓存复用，处理重复图像。
2. 对未命中的新图做 token 压缩，降低首问成本。
3. 对剩余 KV 做低秩或量化，压显存并提高并发上限。

这也是工程上最稳妥的组合顺序，因为它从“零精度风险的复用”开始，再逐步走向“可能有精度代价的压缩”。

---

## 代码实现

下面给出一个可直接运行的玩具实现。它不依赖任何推理框架，只模拟三件事：

1. 如何为图像生成稳定的 `mm_hash`
2. 如何把 `prompt_id + position + mm_hash` 组合成缓存键
3. 如何在同图重试时跳过“视觉编码”

```python
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class CacheKey:
    prompt_id: str
    position: int
    mm_hash: str


@dataclass
class KVEntry:
    mm_hash: str
    position: int
    vision_tokens: int
    kv_bytes: int
    build_ms: int
    reuse_count: int = 0


class ToyVLMCache:
    """
    一个最小可运行示例：
    - encode_image() 模拟视觉编码成本
    - get_or_build() 模拟 KV 复用
    - 这里只演示“命中/失配”的缓存逻辑，不涉及真实 attention
    """

    def __init__(self, num_layers: int = 24, hidden_dim: int = 1024, bytes_per_elem: int = 2):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bytes_per_elem = bytes_per_elem
        self.store: Dict[CacheKey, KVEntry] = {}

    def hash_image(self, image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()[:16]

    def build_cache_key(self, prompt_id: str, position: int, mm_hash: str) -> CacheKey:
        return CacheKey(prompt_id=prompt_id, position=position, mm_hash=mm_hash)

    def encode_image(self, image_bytes: bytes) -> int:
        """
        用输入字节长度模拟视觉 token 数。
        真实系统里，这一步通常由 vision encoder + projector 完成。
        """
        size = len(image_bytes)
        return max(32, size // 12)

    def estimate_kv_bytes(self, total_tokens: int) -> int:
        # 近似公式：2 * L * seq_len * d * bytes
        return 2 * self.num_layers * total_tokens * self.hidden_dim * self.bytes_per_elem

    def get_or_build(self, image_bytes: bytes, prompt_id: str, position: int, text_tokens: int) -> Tuple[str, KVEntry]:
        mm_hash = self.hash_image(image_bytes)
        key = self.build_cache_key(prompt_id, position, mm_hash)

        if key in self.store:
            entry = self.store[key]
            entry.reuse_count += 1
            return "hit", entry

        start = time.perf_counter()

        vision_tokens = self.encode_image(image_bytes)
        total_tokens = text_tokens + vision_tokens

        # 用 sleep 模拟视觉编码和 prefill 的耗时
        time.sleep(0.02)

        entry = KVEntry(
            mm_hash=mm_hash,
            position=position,
            vision_tokens=vision_tokens,
            kv_bytes=self.estimate_kv_bytes(total_tokens),
            build_ms=int((time.perf_counter() - start) * 1000),
        )
        self.store[key] = entry
        return "miss", entry


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def main() -> None:
    cache = ToyVLMCache(num_layers=32, hidden_dim=1280, bytes_per_elem=2)

    img_a = b"invoice-image-content-v1" * 200
    img_b = b"another-image-content-v1" * 200

    text_tokens = 180
    prompt_id = "customer-support-template-v1"
    position = 128

    runs = [
        ("first_a", img_a, prompt_id, position),
        ("second_a", img_a, prompt_id, position),
        ("same_image_wrong_pos", img_a, prompt_id, position + 64),
        ("new_image", img_b, prompt_id, position),
    ]

    for name, image, pid, pos in runs:
        status, entry = cache.get_or_build(
            image_bytes=image,
            prompt_id=pid,
            position=pos,
            text_tokens=text_tokens,
        )
        print(
            f"{name:>20} | {status:>4} | "
            f"mm_hash={entry.mm_hash} | pos={entry.position} | "
            f"vision_tokens={entry.vision_tokens} | "
            f"kv={format_bytes(entry.kv_bytes)} | "
            f"build_ms={entry.build_ms} | reuse_count={entry.reuse_count}"
        )

    # 基本断言：确保缓存逻辑符合预期
    status1, entry1 = cache.get_or_build(img_a, prompt_id, position, text_tokens)
    status2, entry2 = cache.get_or_build(img_a, prompt_id, position, text_tokens)
    status3, entry3 = cache.get_or_build(img_b, prompt_id, position, text_tokens)

    assert status1 == "hit"
    assert status2 == "hit"
    assert status3 in {"hit", "miss"}  # 第三次调用前已运行过 new_image，因此这里允许 hit
    assert entry1.mm_hash == entry2.mm_hash
    assert entry1.mm_hash != entry3.mm_hash

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
```

这个脚本可以直接运行。执行后你会观察到四种情况：

| 场景 | 预期结果 | 原因 |
| --- | --- | --- |
| 第一次请求 `img_a` | `miss` | 冷启动，必须建立 KV |
| 第二次请求 `img_a` | `hit` | 图像、模板、位置完全一致 |
| 同图但位置变化 | `miss` | `position` 进入 key 后不再视为同一前缀 |
| 换成 `img_b` | `miss` | `mm_hash` 改变，不允许误复用 |

这个玩具实现虽然简单，但已经说明了多模态缓存复用最关键的三条工程原则：

1. 图像身份必须进入缓存键。
2. 图像插入位置必须进入缓存键。
3. 命中缓存后，系统应当跳过视觉编码，而不只是“命中后再验证一下”。

更接近真实系统的伪代码如下。它把 LMCache 的 `mm_hash` 思路和 SGLang 的 RadixAttention 思路统一到一个流程里：

```python
def handle_request(
    text_prefix,
    image_bytes,
    image_pos,
    tokenizer,
    vision_encoder,
    projector,
    radix_tree,
    kv_store,
):
    # 1. 计算图像身份
    mm_hash = sha256(image_bytes)[:16]

    # 2. 把多模态占位符替换成带身份的 token 序列
    token_ids = apply_mm_hashes_to_token_ids(
        text_prefix=text_prefix,
        mm_hash=mm_hash,
        image_pos=image_pos,
        tokenizer=tokenizer,
    )

    # 3. 尝试在 radix tree 中查找最长可复用前缀
    prefix_node = radix_tree.longest_prefix_match(token_ids)

    if prefix_node is not None and prefix_node.matches_image(mm_hash, image_pos):
        kv = kv_store.load(prefix_node.cache_id)
        return generate_from_cached_prefix(kv)

    # 4. 未命中时才真正走视觉编码
    vision_feats = vision_encoder(image_bytes)
    vision_tokens = projector(vision_feats)

    # 5. 进入 LLM prefill，构建当前请求的 KV
    kv = prefill(text_prefix=text_prefix, vision_tokens=vision_tokens)

    # 6. 写回缓存索引
    cache_id = kv_store.save(kv)
    radix_tree.insert(token_ids, cache_id, mm_hash=mm_hash, image_pos=image_pos)

    # 7. 继续 decode
    return generate_from_cached_prefix(kv)
```

这里最值得解释的是 `apply_mm_hashes_to_token_ids()`。它的作用不是“真的拿图像哈希去当模型输入”，而是在缓存管理阶段，把原本语义空洞的占位符变成“可以唯一识别图像身份的索引表示”。这样一来，底层前缀匹配系统就不需要为多模态单独发明一整套缓存协议，而是可以沿用文本缓存的匹配、插入和淘汰机制。

如果把这个流程拆开看，真实的 VLM 缓存系统一般至少会经过下面几个阶段：

| 阶段 | 输入 | 输出 | 是否昂贵 |
| --- | --- | --- | --- |
| 图像身份计算 | 原始图像字节 | `mm_hash` | 很便宜 |
| 前缀匹配 | 文本前缀 + 图像身份 + 位置 | 最长可复用块 | 很便宜 |
| 视觉编码 | 图像像素 | 视觉特征 / token | 很昂贵 |
| LLM prefill | 文本 token + 视觉 token | 当前请求 KV | 很昂贵 |
| decode | 已有 KV + 新 token | 生成输出 | 单步较便宜，但次数多 |

所以，图像缓存命中的真正价值，不是“少查一次字典”，而是跳过后两项最贵的环节。

---

## 工程权衡与常见坑

真正上线时，难点往往不在“有没有论文”，而在“论文中的压缩手段能否与缓存系统、调度器、kernel 和隔离策略一起工作”。

下面这些坑最常见：

| 问题 | 冲击 | 应对措施 |
| --- | --- | --- |
| 没有 `mm_hash` 精确匹配 | 错复用图像 KV，严重时产生隐私泄露 | 缓存键必须包含图像身份和位置 |
| 只按图像匹配，不看位置 | 同图插入不同上下文位置时错配 | 把 `position` 纳入 key 或前缀结构 |
| 忽略模型与预处理配置 | 同图在不同分辨率策略下复用错误 | key 中加入模型版本、processor 参数 |
| 只做量化，不改 kernel | 解压和访存开销吞掉收益 | 量化格式与 attention kernel 联合设计 |
| token 合并过猛 | 小目标、OCR、表格细节精度下降 | 分层设置预算，对视觉精细任务保守裁剪 |
| 复用粒度太粗 | 命中率高，但误差累积不可控 | 允许局部重算，特别是前层关键 token |
| 多租户共用缓存 | 命中率提升，但隔离边界变弱 | 租户级 namespace、TTL、权限隔离 |
| 只看命中率，不看 TTFT | 指标好看，但用户无感 | 重点观察视觉块命中后的首 token 改善 |

一个典型误区，是把“缓存命中率”当作唯一目标。对 VLM 来说，高命中率不等于高收益。原因是文本前缀命中和图像块命中的价值差异很大。命中几句系统提示，可能只省下很少的计算；命中一整段图像前缀，省下的却是视觉编码与大块 prefill。换句话说，VLM 更应该关注“命中了什么”，而不是只看“命中了多少次”。

这个差异可以用一个很简单的拆分来理解。设一次请求的总延迟近似为：

$$
\text{TTFT} \approx t_{\text{text-prefill}} + t_{\text{vision-prefill}} + t_{\text{scheduler}} + t_{\text{first-decode}}
$$

如果缓存只命中了文本模板，那么真正减少的是 $t_{\text{text-prefill}}$；而在图像较多的 VLM 里，大头常常在 $t_{\text{vision-prefill}}$。因此业务监控最好同时记录：

1. 文本块命中率
2. 图像块命中率
3. 命中后的 TTFT 改善
4. 每类命中带来的 GPU 时间节省

另一个经常被低估的问题是“压缩后的执行路径”。量化、低秩、稀疏都可以让 KV 数字上变小，但如果 attention kernel 仍按原格式工作，就可能在运行前发生大量重排、还原、解压，从而把收益吃掉。Palu 特别强调 low-rank-aware 和 quantization-friendly，本质上就是在提醒：压缩方法和执行内核必须一起设计，否则论文里的压缩率不一定转化为线上吞吐。

新手还容易忽略一个现实问题：缓存是一种状态，而状态意味着生命周期管理。你至少要回答下面这些问题：

| 生命周期问题 | 如果不处理会怎样 |
| --- | --- |
| 缓存保留多久 | 长时间占用显存或外部存储 |
| 谁可以命中谁的缓存 | 多租户数据串用 |
| 模型升级后是否失效 | 老缓存污染新模型输出 |
| 图像预处理器参数变化怎么办 | 同图产生不同视觉 token |
| eviction 按什么策略做 | 热图被挤掉，命中率波动 |

真实工程里，一个图片审核平台经常会遇到“同一张图被不同规则重复检查”的流量模式。若只升级 GPU 或只改模型，仍然要为每次请求重新做视觉编码；而加入 `mm_hash` 复用后，同图的第二次、第三次检查几乎都能直接读取缓存。对平台型服务来说，这种优化往往比单纯追逐更大显卡更稳定，因为它减少的是重复工作，而不是被动堆硬件。

---

## 替代方案与适用边界

如果业务中“完全相同的图片重复出现”很多，那么 LMCache 这类基于 `mm_hash` 的图像级复用，通常是性价比最高的第一步。它改动相对可控，对精度几乎没有额外风险，收益又直接落在 TTFT 上。但它不是全能方案，因为第一次见到的新图，它几乎帮不上忙。

这时就要看另外几条路线。

VL-Cache 的思路可以概括为“预算化保留”。这里的 budget，就是你允许 KV Cache 占用多少容量。它的核心观察是：VLM 中视觉 token 和文本 token 的重要性并不一致，不同层在 prefill 和 decode 阶段也表现出不同稀疏模式。因此，它不是简单地“统一删掉 90% token”，而是按层、按模态分配缓存预算，再用不同策略打分保留哪些 token。ICLR 2025 论文给出的结果是：只保留约 10% 的 KV Cache，也能接近全缓存精度，并把 GPU 上 KV 显存占用降低约 90%，解码速度最高提升到约 7.08 倍。它适合“显存是首要瓶颈”的场景，比如高并发服务或长视频/多图输入。

VLCache 走的是更激进的路线。它不仅复用 KV，还复用视觉编码器输出，并引入“少量关键 token 重算”的机制。它的关键观点是：复用不是免费的，误差会累积；但并不需要把所有视觉 token 都重算，只需要在关键层对少量关键 token 做纠偏。论文摘要报告，在仅重算 2% 到 5% token 的情况下，可以获得接近全量重算的精度，并带来 1.2 倍到 16 倍 TTFT 提升。它更适合“重复输入极多、同时团队愿意投入更复杂调度逻辑”的平台型部署。

LightVLM 适合“新图首问也很慢”的情况。因为它针对的是 token 级长度，而不是复用问题。只要视觉 token 太长，不论图像是不是第一次出现，token merging 都有价值。但它对任务类型比较敏感，OCR、图表解析、细粒度定位这类任务往往比粗粒度问答更怕信息丢失，因此裁剪策略要保守。

Palu 更偏向“宽度压缩”，适合显存紧、上下文长、模型较大的系统。它并不依赖重复输入，因此适用面更广；但实现门槛也更高，因为你往往需要配套内核优化，才能把低秩压缩转化为真实吞吐。

把这些方案放在一起，可以得到一个更实用的选择表：

| 方案 | 主要动作 | Cache budget / 重算 | 最适合 | 边界 |
| --- | --- | --- | --- | --- |
| LMCache + `mm_hash` | 同图直接复用 KV | 几乎不重算，依赖精确命中 | 多轮同图问答、图像客服、审核平台 | 新图首问收益有限 |
| SGLang RadixAttention | 复用共享前缀，含文本和图像 | 依赖前缀结构复用 | 多轮、多模板、共享示例图像 | 前缀变动大时收益下降 |
| VL-Cache | 只保留重要 KV | 约保留 10%，其余丢弃 | 显存极紧、高并发部署 | 需要稳定的重要性评分 |
| VLCache | 复用 encoder + KV，并少量关键 token 重算 | 仅重算 2% 到 5% token | 高频重复输入平台 | 系统复杂度高 |
| LightVLM | 编码阶段合并 token，并压缩后续 cache | 压序列长度 | 新图首问也要提速 | 过度合并会伤细节 |
| Palu | 低秩压 hidden 维度 | 压每个 token 的 K/V 大小 | 长上下文、大模型、显存紧张 | 需要 kernel 配合 |

实际选型时，可以按下面的判断规则走：

- 重复图像很多，先做 `mm_hash` 复用。
- 模板和多轮上下文复用明显，再考虑 RadixAttention。
- 新图居多、首问慢，优先看 token 压缩，如 LightVLM。
- 显存是第一约束，优先看 VL-Cache、Palu。
- 平台型服务、重复输入密集且能接受复杂实现，再评估 VLCache。

还有一个容易被忽略的现实边界：这些方案并不是都已经变成成熟产品能力。图像级精确复用目前最接近可落地工程；而少量重算、模态感知预算分配这类路线，虽然论文结果很好，但要在生产环境稳定落地，通常还需要额外解决调度、隔离、缓存失效和观测问题。

---

## 参考资料

- LMCache Blog, “LMCache Extends Its Turbo-Boost to Multimodal Models in vLLM V1”  
  https://blog.lmcache.ai/2025-07-03-multimodal-models/

- SGLang, NeurIPS 2024, “SGLang: Efficient Execution of Structured Language Model Programs”  
  https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf

- LightVLM, arXiv:2509.00419, “LightVLM: Acceleraing Large Multimodal Models with Pyramid Token Merging and KV Cache Compression”  
  https://arxiv.org/abs/2509.00419

- ICLR 2025 Poster, “Palu: KV-Cache Compression with Low-Rank Projection”  
  https://iclr.cc/virtual/2025/poster/29993

- ICLR 2025, “VL-Cache: Sparsity and Modality-Aware KV Cache Compression for Vision-Language Model Inference Acceleration”  
  https://proceedings.iclr.cc/paper_files/paper/2025/hash/00db17c36b5435195760520efa96d99c-Abstract-Conference.html

- Amazon Science, “VL-Cache: Sparsity and modality-aware KV cache compression for vision-language model inference acceleration”  
  https://www.amazon.science/publications/vl-cache-sparsity-and-modality-aware-kv-cache-compression-for-vision-language-model-inference-acceleration

- arXiv:2512.12977, “VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference”  
  https://arxiv.org/abs/2512.12977
