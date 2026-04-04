## 核心结论

LLM 应用的延迟优化，目标不是把某一个环节做快，而是把用户真正感知到的“从发出请求到看到第一个字”的时间，也就是 **TTFT，Time To First Token，首个 token 时间**，稳定压到接近实时交互的区间。对聊天、搜索问答、Copilot、游戏 NPC 这类场景，工程上通常把 **200 到 300 毫秒**看作非常顺滑的目标，把 **500 毫秒以内**看作可接受，把 **1 秒以上**看作明显有等待感。

结论先给出：

1. LLM 延迟的核心指标应先盯住 TTFT，再盯住后续 token 吞吐。
2. 真正有效的优化手段主要是三类：**压缩、并行、缓存**。
3. 这三类手段必须同时作用在 **客户端、基础设施、模型推理** 三层，否则一层变快，另一层会立刻变成新瓶颈。
4. 延迟优化不是单纯追求最短时间，而是在 **响应时间、输出质量、显存占用、并发能力** 之间做约束内最优。

一个玩具例子最容易理解。假设玩家在游戏聊天界面里问一句“这关 boss 怎么打”，原始链路是：

| 阶段 | 优化前 | 优化后 | 变化原因 |
|---|---:|---:|---|
| 网络往返 | 150ms | 60ms | 接入点更近，连接复用 |
| 客户端处理 | 300ms | 40ms | prompt 模板预编译，序列化更轻 |
| 服务端排队+prefill | 1300ms | 170ms | 动态批次、静态前缀缓存、请求整形 |
| 首 token 解码 | 250ms | 30ms | KV cache 复用，推理内核优化 |
| 总 TTFT | 2000ms | 300ms | 用户体感从“卡顿”变“实时” |

这里的 **prefill，预填充**，可以理解成“模型先把整段上下文读一遍并建好内部状态”；**KV cache，键值缓存**，可以理解成“把已经算过的上下文中间结果存起来，下次不要重算”。  
所以，“慢”变“快”通常不是靠某个单点魔法，而是靠更近的网络、智能批处理、可复用缓存，以及更稳定的推理流水线。

---

## 问题定义与边界

延迟问题先要拆开，否则工程讨论会失真。对一次普通 LLM 请求，可以写成：

$$
TTFT = T_{net} + T_{prompt} + T_{prefill} + T_{decode}
$$

其中：

- $T_{net}$：网络往返时间。白话说，就是请求从用户设备到服务端再返回首包的时间。
- $T_{prompt}$：客户端和网关侧的请求处理时间。白话说，就是把输入整理成模型能吃的格式所花的时间。
- $T_{prefill}$：模型读完整段上下文并建立注意力状态的时间。
- $T_{decode}$：生成第一个 token 的时间，也就是第一次自回归推理的时间。

这个公式有两个重要含义。

第一，GPU 很快，不代表 TTFT 很短。  
举一个简化场景：用户在海外，客户端先把大段对话历史发到远端网关，再经过负载均衡器转到 GPU 节点。即使 GPU 上的 prefill 只要 120ms，但如果网络来回 220ms、网关排队 180ms、前端模板拼接 150ms，总 TTFT 仍然会超过 600ms。也就是说，**延迟是链路指标，不是单卡指标**。

第二，任何一层失衡都会把优化收益吃掉。  
例如模型侧已经做了 KV cache，但请求经常跨机器漂移，导致缓存命中率很低；或者客户端把几千行历史消息全量上传，结果省下来的推理时间又被网络和序列化吃掉。工程里最常见的问题不是“不会优化”，而是“只优化自己熟悉的一层”。

边界也必须说清楚：

| 边界条件 | 为什么重要 | 典型后果 |
|---|---|---|
| 长上下文 | prefill 近似随上下文长度增长 | TTFT 被上下文拖垮 |
| 高并发 | 调度与排队时间增加 | P99 延迟明显恶化 |
| 显存上限 | KV cache 占用会迅速膨胀 | OOM 或被迫降批次 |
| 跨区域访问 | 网络抖动高于单机优化收益 | 用户感知不稳定 |
| 输出长度 | decode 总时长变大 | 首 token 快但总响应慢 |

因此，本文讨论的“延迟优化”边界是：**在真实线上系统中，以 TTFT 和稳定性为优先目标，在显存、成本和质量约束内做全链路优化**。它不等同于离线 benchmark 跑分，也不等同于只看 tokens/s。

---

## 核心机制与推导

延迟优化常被说成很多零散技巧，但本质上可以归为三种作用方式：

1. **压缩**：减少必须传输、必须计算、必须保存的数据量。
2. **并行**：让原来串行的阶段尽量重叠执行。
3. **缓存**：让重复工作不再重复计算。

把它放到三层结构里看会更清楚：

| 层级 | 主要延迟 | 常用手段 | 直接作用 |
|---|---|---|---|
| 客户端 | $T_{net}$、$T_{prompt}$ | prompt 压缩、历史裁剪、连接复用、流式渲染 | 更快送达，更快展示 |
| 基础设施 | 排队、路由、批次形成 | 动态批次、粘性路由、近源接入、异步流水线 | 降低抖动，减少排队 |
| 模型推理 | $T_{prefill}$、$T_{decode}$ | 静态前缀缓存、KV 共享、量化、张量/序列并行 | 降低单请求计算量 |

可以把一次请求想成三步：**准备、处理、输出**。  
新手版理解是：不要让这三步互相等，而是让它们尽量重叠。

一个简单的推导思路如下。

假设原始链路是完全串行：

$$
T_{serial} = T_{net} + T_{prompt} + T_{prefill} + T_{decode}
$$

如果客户端能边序列化边上传，服务端能边收包边做前置校验，网关能提前把请求归入合适批次，那么实际等待时间更接近：

$$
T_{overlap} \approx \max(T_{net}, T_{prompt}) + T_{prefill}^{'} + T_{decode}^{'}
$$

这里的 $T_{prefill}^{'}$ 和 $T_{decode}^{'}$ 又会因为缓存和量化进一步下降。  
所以延迟优化不是简单相加减法，而是把一部分串行路径改成重叠路径，再把剩下的关键路径压短。

### 玩具例子

用户问：“Python 列表和元组有什么区别？”  
如果系统每次都把完整聊天历史、系统提示词、用户画像、工具定义全量送入模型，那么即使问题只有一句话，prefill 也要读几千 token。  
更合理的做法是：

- 系统提示词做静态前缀缓存。
- 历史消息做窗口裁剪。
- 客户端把 metadata 与正文分开发送，正文优先上行。
- 服务端优先形成小批次，避免等大批次凑齐。

结果是，首 token 时间从“等整车装满再发车”变成“先把最关键的一箱送上路”。

### 真实工程例子

以一个 13B 模型在线聊天服务为例，业务高峰期每秒几十到上百个请求。系统可能同时采用以下策略：

- 网关把相同系统提示词的请求打到同一组 GPU，提升静态前缀缓存命中率。
- 推理层把短 prompt 请求优先组成 prefill batch，避免被超长上下文拖慢。
- decode 阶段对活跃会话保留 KV cache，继续追问时只追加新 token。
- 客户端在首 token 到达后立刻开始渲染，而不是等完整句子返回。

这类组合优化的意义在于：**不是单次请求最快，而是整体系统在高并发下仍然稳定地快**。很多线上系统中位数 TTFT 不差，但 P95、P99 很差，原因通常就是批次策略、缓存路由和长上下文请求互相干扰。

---

## 代码实现

下面用一个可运行的 Python 玩具实现说明思路。它不是真实推理框架代码，但结构上包含了客户端预处理、服务端 KV 缓存复用、动态批次调度与流式发送。

```python
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Request:
    session_id: str
    prefix_id: str
    prompt_tokens: int
    new_tokens: int

STATIC_PREFIX_CACHE = {"chat_v1": 600}  # 已缓存的静态前缀 token 数
SESSION_KV_CACHE = defaultdict(int)      # 已为某会话保留的 KV token 数

def client_prepare(history_tokens: int, question_tokens: int) -> int:
    # 客户端先裁剪历史，再发送必要内容
    kept_history = min(history_tokens, 200)
    return kept_history + question_tokens

def server_prefill_cost(prefix_id: str, prompt_tokens: int) -> int:
    cached = STATIC_PREFIX_CACHE.get(prefix_id, 0)
    uncached_tokens = max(prompt_tokens - cached, 0)
    return uncached_tokens

def decode_step(session_id: str, new_tokens: int) -> int:
    # KV cache 复用后，只需要为新增 token 追加状态
    SESSION_KV_CACHE[session_id] += new_tokens
    return new_tokens

def estimate_ttft(network_ms: int, client_tokens: int, prefill_tokens: int, first_decode_tokens: int = 1) -> int:
    t_prompt = client_tokens // 5
    t_prefill = prefill_tokens // 4
    t_decode = first_decode_tokens * 10
    return network_ms + t_prompt + t_prefill + t_decode

# 玩具例子：优化前
raw_prompt = 1200 + 20
ttft_before = estimate_ttft(
    network_ms=150,
    client_tokens=raw_prompt,
    prefill_tokens=raw_prompt,
)

# 优化后：客户端裁剪 + 静态前缀缓存 + 会话 KV 复用
prepared = client_prepare(history_tokens=1200, question_tokens=20)
prefill = server_prefill_cost(prefix_id="chat_v1", prompt_tokens=prepared)
decode_step("s1", new_tokens=30)  # 假设上一轮已经建立会话缓存
ttft_after = estimate_ttft(
    network_ms=60,
    client_tokens=prepared,
    prefill_tokens=prefill,
)

assert prepared == 220
assert prefill == 0  # 因为静态前缀缓存已覆盖
assert ttft_after < ttft_before
print(ttft_before, ttft_after)
```

这段代码分别模拟了几个关键动作：

- `client_prepare`：客户端裁剪历史，降低 $T_{prompt}$ 与 $T_{net}$。
- `server_prefill_cost`：静态前缀缓存命中后，prefill 只处理未缓存部分。
- `decode_step`：会话继续追问时复用 KV cache，降低后续 decode 成本。
- `estimate_ttft`：用简化模型估算 TTFT 变化趋势。

如果把它改写成更接近线上服务的伪代码，结构通常类似这样：

```python
async def handle_request(req):
    packed = serialize_prompt(req.messages, req.metadata)   # 减少传输体积
    await send_to_gateway(packed)

async def scheduler_loop():
    while True:
        batch = build_prefill_batch(max_wait_ms=8, max_tokens=8192)
        grouped = group_by_prefix(batch)                    # 提升静态缓存命中
        for g in grouped:
            kv = load_prefix_cache(g.prefix_key)
            run_prefill(g.requests, kv)
        run_decode_interleaved()                           # 与网络发送交错
        for token in ready_tokens():
            await async_send(token)                        # 首 token 先发
```

这里每一步都对应延迟控制：

- `serialize_prompt`：减少请求大小，降低网络与解析时间。
- `build_prefill_batch`：动态批次，但等待窗口要小，否则排队收益反而变损失。
- `group_by_prefix`：把相同前缀的请求放一起，提升缓存复用率。
- `run_decode_interleaved`：prefill 和 decode 流水化，避免 GPU 阶段空转。
- `async_send`：一旦首 token 可用就立即回传，不等完整回答结束。

真正的工程实现会更复杂，比如需要处理取消请求、超时、批次拆分、长短序列混排、GPU 内存回收等，但核心思想就是：**把 TTFT 路径上的重复计算去掉，把必须计算的部分并起来，把能先返回的结果先返回**。

---

## 工程权衡与常见坑

延迟优化最大的误区，是把“优化手段”误当成“无代价收益”。实际上每一种收益都对应成本。

先看常见坑：

| 常见坑 | 现象 | 规避策略 |
|---|---|---|
| KV cache 爆显存 | 长上下文+高并发时直接 OOM | 做缓存压缩、限制保留窗口、配合序列并行 |
| 只优化模型不优化路由 | benchmark 很好，线上体感一般 | 做粘性路由，提升缓存命中率 |
| 批次过大 | 吞吐上升但 TTFT 变差 | 对交互式流量限制最大等候窗口 |
| 过度压缩 prompt | 延迟下降但回答质量下降 | 对摘要/裁剪做质量回归测试 |
| 长短请求混跑 | 短请求被长请求拖慢 | 建立长度分层队列 |
| 只看平均值 | 中位数好看，尾延迟很差 | 重点看 P95/P99 与抖动 |

一个典型工程权衡是 KV cache。  
它的好处非常直接：decode 不必反复重算历史上下文，所以后续 token 更快，连续对话更顺滑。  
但它的代价也非常直接：**吃显存**。上下文越长、层数越多、并发越高，KV cache 占用越大。粗略理解，缓存占用近似与“层数 × token 数 × hidden size × 批次”成正相关。

因此在 64K 长上下文场景里，常见做法不是“全量保留一切缓存”，而是组合策略：

- 对 KV cache 做压缩，比如低比特量化或分层保留。
- 开启序列并行，把超长上下文拆到多卡。
- 限制活跃会话的缓存驻留时间。
- 对低价值历史做摘要替代，而不是盲目保留原文。

如果这些措施没有跟上，就会出现一个很典型的失败模式：  
系统为了降低 decode 延迟保留了大量缓存，但一到高峰期显存不够，只能降批次，结果 prefill 排队时间暴涨，TTFT 反而更差。

另一个常见问题是网络与总线迁移瓶颈。  
很多团队优化模型内核后发现收益有限，原因不是模型没变快，而是瓶颈已经转移到 **PCIe/NVLink 数据搬运、跨机路由、网关排队**。这类问题的信号是：GPU 利用率看着不低，但用户仍然觉得慢，而且不同机房、不同时间段波动很大。  
所以延迟优化必须带着链路追踪做，不能只看单点 profile。

---

## 替代方案与适用边界

不是所有场景都需要同一套优化组合。选型要看交互模式、硬件预算、上下文长度和部署位置。

| 方案 | 适用场景 | 权衡 |
|---|---|---|
| 动态批次 + KV cache + 粘性路由 | 在线聊天、Copilot、客服 | 交互体验好，但系统复杂度高 |
| 流水线并行 + 动态编译 | 资源充裕的大模型服务 | 吞吐高，部署与调优成本高 |
| Prompt 压缩 + 小批次推理 | 边缘设备、移动端 SDK | 实现简单，但质量可能下降 |
| 分布式缓存 + 前缀复用 | 系统提示高度重复的业务 | 命中时收益大，命中不稳时一般 |
| 检索摘要替代长上下文 | 文档问答、知识库 | 降低 prefill，但依赖检索质量 |

可以给两个对比鲜明的例子。

**移动端 SDK 场景**：  
手机侧算力弱、网络不稳定，最有效的策略往往不是复杂多卡并行，而是：

- 在边缘或客户端先压缩 prompt。
- 只上传必要上下文和简化 metadata。
- 服务端保持小批次推理，优先保证首 token 快速返回。

这里的边界是：如果你过度压缩，模型可能丢掉关键上下文，答案质量下降。

**高性能机房场景**：  
如果硬件资源充裕，且请求模式稳定，可以优先使用：

- 流水线并行或张量并行；
- 动态编译和推理内核融合；
- 大规模 KV cache 与前缀缓存复用。

这里的边界是：系统调度复杂度会显著提升，路由、缓存一致性、故障恢复都更难处理。

因此，延迟优化没有“银弹配置”。  
可以用一句更工程化的话总结：**交互型业务优先 TTFT，批处理型业务优先吞吐，长上下文业务优先 prefill 控制，高并发业务优先尾延迟稳定性**。

---

## 参考资料

1. Inference.net，《What Is Inference Latency & How Can You Optimize It?》  
   侧重点：解释什么是推理延迟，以及为什么用户感知主要受首响应时间影响。

2. MLJourney，《Latency Optimization Techniques for Real-Time LLM Inference》  
   侧重点：延迟拆解与公式，适合理解 $TTFT = T_{net} + T_{prompt} + T_{prefill} + T_{decode}$ 这类工程分析框架。

3. Latentforce，《LLM Inference Optimization: 5 Techniques》  
   侧重点：生产环境案例数据，适合理解缓存、动态批次、压缩的组合收益。

4. NVIDIA Blog，《Mastering LLM Techniques: Inference Optimization》  
   侧重点：推理优化常见坑，尤其是 KV cache、显存、并行与硬件链路约束。

5. NanoGPT，《Latency Profiling for Large Language Models》  
   侧重点：如何做延迟剖析，帮助把“慢”准确定位到网络、调度还是模型内部。
