## 核心结论

视频 LLM 推理的核心不是“把整段视频直接塞进模型”，而是先做压缩，再做预算管理。压缩分两层：第一层是帧压缩，也就是先删掉重复帧和变化很小的帧；第二层是 Token 压缩，也就是在视觉编码后只保留最重要的一小部分表示。这样做的目标很明确：把原本会撑爆上下文窗口的视觉输入，压到 LLM 能稳定处理的范围内。

对长视频，最实用的工程路径通常是四步：

| 阶段 | 作用 | 典型方法 | 结果 |
|---|---|---|---|
| 帧压缩 | 先减少冗余画面 | 均匀降采样、运动感知关键帧 | 帧数下降 |
| Token 压缩 | 再减少视觉表示 | 时序池化、注意力分数筛选 | Token 数下降 |
| 分块推理 | 把长视频拆成可调度的小段 | chunk processing | 单次推理可控 |
| 缓存复用 | 避免重复做前缀计算 | prefix cache、chunk cache | 延迟下降 |

一个常用的近似公式是：

$$
N_{llm}=\alpha \cdot N_{clip}
$$

其中 $N_{clip}$ 是一个 clip 原始视觉 token 数，$\alpha$ 是保留比例。若 $\alpha \approx 6\%$，就表示只把最重要的约百分之六送入 LLM。

玩具例子：32 帧 clip 原始有 3136 个视觉 token，如果只保留 6%，则送入 LLM 的 token 约为：

$$
3136 \times 0.06 \approx 188
$$

这时模型看到的不是完整视频，而是“已经过筛选的关键视觉证据”。这正是长视频可运行的前提。

---

## 问题定义与边界

这里先定义问题。视频 LLM 指“能够同时处理视频内容和文本提示的大语言模型系统”。白话说，它不是只看字，也要看画面。问题在于，视频天然是高冗余输入，尤其长视频会持续产生大量视觉 token。

如果一个系统上下文窗口是 128k token，而 10 分钟以上视频经过视觉编码后轻易达到数十万 token，那么直接拼接输入几乎一定失败。失败形式通常有三种：

| 失败类型 | 表现 | 根因 |
|---|---|---|
| 上下文溢出 | 输入放不下 | 原始视觉 token 太多 |
| 延迟失控 | 首 token 时间过长 | prefill 过重 |
| 记忆断裂 | 前后语义接不上 | 分块之间没有稳定记忆机制 |

这里的 prefill 是“模型先把整段输入编码进 KV Cache 的阶段”。白话说，就是模型在真正回答前，先把上下文读一遍并存起来。长视频最大的问题不是 decode，decode 是逐 token 生成；真正重的是 prefill，因为每多一批视觉 token，KV Cache 就会继续膨胀。

边界也要说清楚。本文讨论的是“长视频推理预算管理”，不是视频训练，不讨论多机训练策略，也不讨论端到端视频编码器如何训练。重点只放在在线推理链路：

`frame -> visual token -> selected token -> chunk -> scheduler -> LLM`

再看一个长视频数值例子。假设 64 个 clip，每个 clip 来自 32 帧，那么总帧数约 2048。若每个 clip 原始为 3136 个 token，总 token 数是：

$$
64 \times 3136 = 200704
$$

这已经远超 128k。若每个 clip 压缩到 188 个 token，则总输入仅为：

$$
64 \times 188 = 12032
$$

从二十万级降到一万级，这时长视频推理才从“不可能”变成“可调度”。

---

## 核心机制与推导

第一层机制是帧级压缩。均匀降采样的意思是按固定时间间隔抽帧，白话说就是“先粗略地隔几帧看一次”。它简单、稳定，但会漏掉短暂关键动作。运动感知关键帧选取会更进一步：根据相邻帧差异、光流或者编码器特征变化，优先保留发生明显变化的位置。白话说，就是“只有画面真的变了，才值得多看”。

第二层机制是 Token 级压缩。视觉编码器会把一段 clip 变成很多 token，这些 token 不同程度地影响后续语言生成。系统通常会给 token 打分，再只保留高分 token。这里的注意力分数可以理解为“模型认为哪些视觉位置更值得关注的信号”。

压缩后的核心公式就是：

$$
N_{llm}=\alpha \cdot N_{clip}, \quad \alpha \approx 6\%
$$

其中 $\alpha$ 越小，输入越省，但信息损失越大。工程上不是越小越好，而是要找到“信息保留”和“上下文成本”之间的平衡点。

再往下是调度。长视频不能只看“总 token 数”，还要看“每次送多少”。这就引出 token budget，也就是“单次微批允许占用的 token 上限”。白话说，它像一张临时配额表，规定这一轮最多能塞多少内容。可写成：

$$
\sum_{i \in \text{micro-batch}} n_i \le B_t
$$

其中 $B_t$ 是当前时刻的 token 预算。注意这里常见的工程策略是：每个 micro-batch 只允许一个 prefill chunk。原因是 prefill 很重，如果多个 prefill chunk 同时抢资源，会压住 decode，导致流水线出现空泡，也就是 GPU 有时在等，有时在堵，不平稳。

可以把它理解成一个简单原则：

| 调度规则 | 目的 |
|---|---|
| 每批 token 不超过 $B_t$ | 防止瞬时过载 |
| 每批只进一个 prefill chunk | 避免 prefill 抢占 decode |
| 已缓存前缀优先复用 | 减少重复 KV 计算 |

玩具例子可以这样看。现在系统里已经有若干请求在 decode，每轮还剩 2000 token 预算。新到一个视频 chunk，prefill 需要 1600 token；另一个 chunk 需要 900 token。虽然两者加起来 2500 超预算，而且两个都是 prefill，因此调度器不会一起放入。它更可能只放一个 1600 的 prefill，留剩余空间给 decode。这样做不是最“满”，但更稳，因为 decode 被拖慢会直接拉高用户可见延迟。

真实工程里还会引入一个目标：尽量让不同流水线阶段的耗时接近，减少空转。无论公式写成吞吐目标还是阶段差最小化，本质都一样，即让 prefill 与 decode 的负载分配更平衡。

---

## 代码实现

下面给一个可运行的简化 Python 示例，演示“压缩 -> 分块 -> 按预算调度”的核心流程。这个例子不依赖真实模型，目的是把预算逻辑讲清楚。

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Clip:
    clip_id: int
    raw_tokens: int
    selected_ratio: float = 0.06

    @property
    def selected_tokens(self) -> int:
        return max(1, int(self.raw_tokens * self.selected_ratio))

@dataclass
class Chunk:
    clip_id: int
    tokens: int
    cached: bool = False

def build_chunks(clips: List[Clip]) -> List[Chunk]:
    return [Chunk(clip_id=c.clip_id, tokens=c.selected_tokens) for c in clips]

def schedule_micro_batches(chunks: List[Chunk], budget: int) -> List[List[Chunk]]:
    """
    规则：
    1. 每个 micro-batch 总 token <= budget
    2. 每个 micro-batch 最多一个非 cached 的 prefill chunk
    3. cached chunk 视为复用前缀，调度成本近似为 0
    """
    batches = []
    current = []
    current_tokens = 0
    has_prefill = False

    for ch in chunks:
        cost = 0 if ch.cached else ch.tokens
        need_new_batch = False

        if current_tokens + cost > budget:
            need_new_batch = True
        if (not ch.cached) and has_prefill:
            need_new_batch = True

        if need_new_batch:
            batches.append(current)
            current = []
            current_tokens = 0
            has_prefill = False

        current.append(ch)
        current_tokens += cost
        if not ch.cached:
            has_prefill = True

    if current:
        batches.append(current)

    return batches

# 玩具数据：32帧clip -> 3136 raw tokens -> 188 selected tokens
clips = [Clip(clip_id=i, raw_tokens=3136) for i in range(4)]
chunks = build_chunks(clips)

assert chunks[0].tokens == int(3136 * 0.06)

# 假设前两个 chunk 已命中 cache
chunks[0].cached = True
chunks[1].cached = True

batches = schedule_micro_batches(chunks, budget=250)

# 校验：每个 batch 最多一个非 cached chunk，且预算不超
for batch in batches:
    prefill_count = sum(1 for x in batch if not x.cached)
    token_cost = sum(0 if x.cached else x.tokens for x in batch)
    assert prefill_count <= 1
    assert token_cost <= 250

# 4个clip原始总token vs 压缩后总token
raw_total = sum(c.raw_tokens for c in clips)
selected_total = sum(c.selected_tokens for c in clips)

assert raw_total == 12544
assert selected_total == 752
print(raw_total, selected_total, len(batches))
```

这个代码省略了真实视觉编码器，但保留了三个最关键的约束：

1. `selected_ratio` 模拟注意力筛选后的保留比例。
2. `cached=True` 模拟 prefix cache 或 chunk cache 命中。
3. `schedule_micro_batches` 明确表达“预算约束 + 每批一个 prefill”的调度规则。

真实工程例子通常会再加两层接口：

| API | 作用 |
|---|---|
| `alloc_budget(chunk)` | 判断当前轮还能不能放入该 chunk |
| `enqueue_prefill(chunk)` | 把新 chunk 送去做 prefill |
| `assemble_prefix_cache(seq)` | 将历史 chunk 的缓存拼成当前前缀 |

如果系统采用 LMCache 一类外部 chunk cache，那么缓存命中时，不需要重新计算整个历史前缀的 KV，只需要取回可复用片段，再对新来的 chunk 做补充 prefill。对于连续视频问答，这个收益很大。比如监控场景中，用户先问“刚才谁进入房间”，接着又问“他手里拿了什么”，第二问不应该把前 9 分钟视频全部重新 prefill 一次，否则延迟和显存压力都会明显上升。

---

## 工程权衡与常见坑

第一类坑是只做均匀抽帧，不做动态筛选。这样实现简单，但在“长时间静止 + 短暂关键动作”的视频里容易丢信息。比如仓库监控中，前面 5 分钟几乎没变化，只有最后 3 秒有人拿走箱子。均匀抽帧很可能把关键动作抽稀到不可判别。

第二类坑是只压帧，不压 token。很多人以为抽帧后问题就解决了，但视觉编码后每帧仍会生成较多 token，KV Cache 仍然会快速增长。帧压缩解决的是“输入长度”，Token 压缩解决的是“上下文成本”，两者不能互相替代。

第三类坑是静态预算。也就是无论当前 decode 压力如何，永远按固定大小塞 prefill。这会导致系统在高并发下反复抖动，表现为某些 chunk 被推迟、重排，甚至重新做 KV 填充。

第四类坑是缓存淘汰策略过于粗糙。LRU 是“最近最少使用淘汰”，白话说就是很久没用的先删。它实现简单，但不知道一个会话是否还会继续。如果刚刚结束的一轮问答在几秒后还会追问，LRU 很可能过早删掉有价值的前缀。学习型 prefix caching 的思路是预测“这个前缀未来还会不会被继续用”，再决定是否保留。

| 策略 | 优点 | 劣势 | 典型失误 |
|---|---|---|---|
| LRU | 简单，易实现 | 不理解会话续写概率 | 过早淘汰还会复用的前缀 |
| LPC | 更懂哪些前缀值得留 | 需要训练或额外模型 | 预测失误时收益不稳定 |
| Dynamic Budget | 负载更平滑 | 实现复杂，参数多 | 预算估计过保守导致吞吐下降 |
| Chunk Cache | 降低重复 prefill | 命中依赖 chunk 切分稳定 | chunk 粒度设计不当，命中率低 |

工程上建议重点监控三个指标：

| 指标 | 含义 | 作用 |
|---|---|---|
| KV fill time | prefill 填充耗时 | 判断视觉输入是否过重 |
| Cache hit ratio | 缓存命中率 | 判断 chunk 切分和缓存策略是否有效 |
| GPU idle rate | GPU 空闲比例 | 判断调度是否失衡 |

如果 `KV fill time` 居高不下，先检查 token 压缩是否不足；如果 `cache hit ratio` 低，先检查 chunk 设计是否稳定；如果 `GPU idle rate` 高，优先排查是否有多个大 prefill 同时挤占 decode。

---

## 替代方案与适用边界

不是所有视频任务都需要这套完整机制。若输入只是几十秒短视频，而且交互轮数少，直接用滑动窗口加普通缓存就够了。滑动窗口的意思是“只保留最近的一段上下文”。白话说，就是旧内容逐渐滑出视野，不追求完整长时记忆。

如果场景是大量重复素材的检索问答，还可以考虑更粗粒度的 prompt-level cache 或 semantic cache。semantic cache 是“按语义相似度复用结果的缓存”。白话说，就是用户问法不同，但其实问的是同类内容，可以直接重用先前结果。这对文档问答很有效，但对视频逐帧细节不一定可靠，因为两段视频语义接近，不代表关键动作发生在同一时刻。

下面给一个简化决策表：

| 场景 | 推荐机制 | 原因 |
|---|---|---|
| 短视频，低并发 | 静态 chunk + 普通 prefix cache | 实现成本低，已经够用 |
| 长视频，单轮总结 | 帧压缩 + token 压缩 + chunk processing | 先把内容塞进窗口 |
| 长视频，多轮交互 | 上述方案 + chunk-level cache | 需要复用历史前缀 |
| 高并发流式服务 | 动态 token budget + 单 prefill 调度 | 避免 prefill 与 decode 打架 |
| 高频续问会话 | LPC 或更智能的前缀保留策略 | 减少错误淘汰 |

真实工程例子：一个安防平台连续接入 30 分钟监控视频，用户会不断追问“上一分钟发生了什么”“刚刚那个人离开时带了什么”。这种场景里，短视频策略基本不够，因为每次问题都可能依赖前文。系统通常需要同时具备三点：对视频做时间分块、对视觉 token 做压缩、对已处理 chunk 做前缀缓存。否则要么记不住，要么答得慢，要么成本失控。

适用边界也要明确。如果硬件极其充裕，或者业务只关心粗粒度事件摘要，那么可以牺牲部分预算优化，直接增加上下文或减少筛选强度。但只要进入“分钟级以上视频 + 多轮追问 + 并发请求”的组合，预算管理就不再是优化项，而是系统是否能上线的基础能力。

---

## 参考资料

- Streaming Video LLM Architecture（Token Efficiency）: https://www.emergentmind.com/topics/streaming-video-large-language-model-llm?utm_source=openai
- Dynamic Micro-Batch and Token-Budget Scheduling for IoT-Scale Pipeline-Parallel LLM Inference（Sensors 2026）: https://www.mdpi.com/1424-8220/26/4/1101?utm_source=openai
- Chunk-Level Caching and LMCache: Accelerating LLM Inference（Pynomial 2025）: https://pynomial.com/2025/09/chunk-level-caching-and-lmcache-accelerating-llm-inference/?utm_source=openai
- Learned Prefix Caching for Efficient LLM Inference（NeurIPS 2025）: https://openreview.net/forum?id=Vj48eXaQDM&utm_source=openai
