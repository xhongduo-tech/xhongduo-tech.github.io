---
title: 前后端分离架构：Router 与 Runtime
date: 2026-08-07
---

# 前后端分离架构：Router 与 Runtime

<div class="epigraph">
<p>计算机科学中的所有问题，都可以通过增加一层间接层来解决。</p>
<footer>—— 戴维 · 惠勒（David Wheeler）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 第四章 SGLang ｜ 2026-08-07</p>
</div>

## 为什么从 Router 与 Runtime 开始

前两节给 SGLang 配齐了两件武器：**RadixAttention** 让共享前缀的 KV 只算一次，**约束解码**让输出必然合法。但这两件武器都假设「模型就在这台机器上」。真实服务不是这样的——一个大模型服务要扛住成百上千并发，通常横跨多张 GPU、多台机器；请求从哪进、往哪台机器去，这个决定直接决定第二件武器（RadixAttention）能不能命中。<span class="marginnote">想想看：一条请求如果被轮询到一个「从未见过它前缀」的机器上，那台机器就得从零开始算一遍 Prefill——上一节树里辛辛苦苦攒下的共享 KV 全白费。路由策略不是锦上添花，它决定缓存命中率，进而决定 TTFT 与吞吐。</span>

SGLang 的答案是**前后端分离**：把「接客」与「干活」拆成两个角色——Router（前端）负责接请求、做决策；Runtime（后端）负责真正跑模型。这就是惠勒说的那层「间接层」：它让我们能在「尽量命中缓存」与「尽量均分负载」之间做文章。

**核心概念：Router 与 Runtime（前后端分离）**：Router 是面向客户的前端进程，负责 HTTP/OpenAI 兼容 API、tokenize、会话管理与**路由决策**；Runtime 是实际执行推理的后端进程，内含调度器、RadixAttention 缓存与模型执行器。二者通过进程边界分离，一个 Router 可以指挥多个 Runtime。

## 1 一张架构图：谁在前、谁在后

把两个角色摆开，各自的职责就清楚了。

**Router（前端）**：接收 HTTP 请求，做 tokenize，维护每个会话的上下文记账，然后**决定这条请求去哪台 Runtime**。它手里不拿模型权重，只拿「各 Runtime 的缓存概况」与「各 Runtime 的负载概况」。
**Runtime（后端）**：拿一条已经 tokenize 好的请求，做调度（连续批处理）、走 RadixAttention 复用前缀、执行 Prefill/Decode、做约束解码与采样，最后把 token 流送回 Router 转给客户端。它手里拿着模型权重与 KV 缓存池。

部署形态有两种：

**单进程**：直接把 Router 与 Runtime 合在同一个进程里跑，单机单卡最省事，也是大多数实验的起点。
**分离部署**：Router 与 Runtime 各自独立成进程，一个 Router 背后挂多个 Runtime（每张卡一个 Runtime），各自拥有独立的 RadixAttention 前缀树。<span class="marginnote">分离带来的第一个好处是<strong>独立扩缩</strong>：Router 是纯 CPU 进程（只做 tokenize 与决策），卡脖子的 Runtime 可以横向加卡；第二个好处是<strong>隔离</strong>：一台 Runtime 崩了，Router 可以把请求切到别的 Runtime，而不是整个服务宕机。</span>

把一条请求在分离架构下的完整旅程走一遍，两个角色就彻底活了：

1. 客户端发送一段多轮对话到 Router，Router 先做 **tokenize**，得到输入序列，并按会话记账（这轮输入 = 上一轮历史 + 新消息）。
2. Router 拿着这段输入跑路由策略（下一节细讲），选定一台 Runtime，把 **token id**（而不是原始文本）交过去。
3. Runtime 的调度器把它排进连续批处理，RadixAttention 在新请求的前缀与缓存树之间做最长前缀匹配，能复用就复用，只算后缀。
4. 生成的 token 流逐片送回 Router，Router 边收边流式转发给客户端（SSE），会话状态在 Router 侧累计。

注意第 2 步的细节：**tokenize 发生在 Router，Runtime 只收 token id**。这让 Router 可以精确估算每台 Runtime 的输入长度与负载（容量估计按 token 数算），也把「逐次重复 tokenize」的开销集中到一处。代价是 Router 与 Runtime 必须用**同一套 tokenizer**——换模型时两者要一起换，这是「按模型路由」的又一个理由。

注意：这里的 Router 是**按模型**的——它路由的是「同一个模型的多份副本」。如果要在多个模型之间路由，那是另一层（SGLang 后来把路由抽成了独立的 sgl-model-gateway / SMG，支持多模型、多后端混合路由），不在本节范围。

## 2 路由策略家族：从轮询到缓存感知

Router 的价值全在「用什么策略选 Runtime」。SGLang 的策略是一整个家族：

| 策略 | 怎么选 | 对缓存命中 |
| --- | --- | --- |
| round-robin（轮询） | 轮流分 | **差**：同前缀请求被甩到不同机器，各自重算 |
| random（随机） | 随机分 | 差 |
| least-connections / least-loaded（最小负载） | 选负载最低的 | 不看缓存，可能命中也可能不命中 |
| power-of-two-choices（二选一） | 随机抽两台比负载 | 负载好，缓存靠运气 |
| consistent-hash（一致性哈希） | 按请求哈希稳定映射 | 稳定的前缀会去同一台，但哈希不感知负载 |
| prefix-hash（前缀哈希） | 按前缀哈希映射 | 同前缀同机器，**但太僵**：不感知负载，可能把一台机器塞爆 |
| cache-aware（缓存感知） | 看谁最可能命中缓存，再平衡负载 | **旗舰策略** |

把表格读薄：前面六种要么「只看负载、不看缓存」，要么「只看缓存、不看负载」。真正的难点是**同时看两者**——这正是 cache-aware 策略做的事。学第三级《分布式系统》时遇到的「一致性哈希」「最短队列」「二选一」这些负载均衡思想，在这里几乎原样复刻了一遍——区别只在于，这里的「代价函数」里多了一项缓存命中比，而那一项恰恰是最贵的（一次 Prefill 重算顶得上几十次转发）。

## 3 公式解析：缓存感知路由的权衡函数

cache-aware 策略的核心，是给每台 Runtime 维护一棵**近似的 radix 树**——注意，是**原始文本字符**（不是 token id）的树，用来「预测」这台 Runtime 的缓存里大概存了哪些前缀。请求到达时，Router 拿请求文本去每棵树上跑一次前缀匹配，得到命中比：

$$
h_i = \frac{\text{在 Runtime } i \text{ 树上匹配到的字符数}}{\text{请求总字符数} \, L}
$$

于是路由规则可以写成「分段」的形式：

$$
i^* =
\begin{cases}
\arg\max_i h_i, & \max_i h_i > \theta_{\text{cache}} \;\;(\text{有足够缓存命中}) \\
\arg\min_i \text{size}_i, & \text{否则（去缓存最空的机器，最有可能吸收新前缀）}
\end{cases}
$$

逐步拆解：

- **第一步，认命中比 $h_i$**：$h_i$ 接近 1 说明这台机器的缓存几乎完整覆盖了请求的前缀，去那里 Prefill 几乎免单；接近 0 说明去了也是从零算起。
- **第二步，认阈值 $\theta_{\text{cache}}$**：只有当最高命中比**超过阈值**（SGLang 默认约 0.3–0.5）才「为缓存而动」；否则不去赌缓存，而是把请求发给**缓存树最小**的机器——它空闲容量最大，最适合把这串新前缀「种」进缓存。
- **第三步，认负载钳制**：若放任「只看缓存」，会出现热点——命中率最高的那台机器被所有请求挤爆。所以策略会盯负载差：当
$$
\max_i q_i - \min_i q_i > \theta_{\text{abs}} \quad \text{且} \quad \max_i q_i > \theta_{\text{rel}} \cdot \min_i q_i
$$
（$\theta_{\text{abs}}$ 默认约 32–64，$\theta_{\text{rel}}$ 默认约 1.1–1.5）时，说明负载失衡，切到「**短队列 + 缓存亲和**」模式：在「负载不超过最轻机器加 $\theta_{\text{abs}}$」的窗口里，选缓存命中最长的那台。<span class="marginnote">这是 SGLang 对「缓存 vs 负载」这场拔河的关键修补：早期版本一旦失衡就直接切最短队列、完全丢弃缓存亲和，导致缓存抖动；改进版在负载窗口内<strong>保留缓存亲和</strong>，实测在多轮负载上把平均 TTFT 降了 51%、P90 降 61%、P99 降 67%——路由决策不是二选一，而是「先保负载，再在窗口里挑缓存」。</span>

**第四步，看近似与精确**：Router 的树是**字符级近似**，Runtime 的 RadixAttention 是 **token 级精确**。Router 用近似树只为了「快速预测」哪台更可能命中，真正命中与否由 Runtime 的精确树决定——两者之间永远有「预测误差」，这是路由层的固有成本，也是它敢把 tokenize 省到 Router 的原因（原始文本比 token id 更容易跨模型比对）。

把这三段合起来看，就是一条清晰的决策链：**先看能不能靠缓存省事（命中比），再看会不会失衡（负载差），最后在「安全窗口」里挑最省事的**。

## 4 单机多卡场景：DP 内的内容感知调度

上面的 Router 是「跨机器」的路由。还有一类常见场景是**单机多卡数据并行（DP）**：同一份权重复制 N 份，请求被分到 N 张卡上，每张卡各带一份 RadixAttention 缓存。此时没有独立 Router 进程，由 DataParallelController 在实例内部做分发——而它历史上的 round-robin 分发对缓存是灾难：Agent 风格的多轮请求，每轮都带一段 2 万 token 的共享 system prompt，轮询分发让每张卡每轮都重算一遍。<span class="marginnote">SGLang 社区实测：22,651 token 的共享前缀、tp=16/dp=16 的 MoE 模型上，round-robin 让每条请求的 TTFT 都停在约 7400 ms；换成内容感知的 prefix-hash 分发（对请求头部做稳定哈希、同前缀同 rank），稳定态 TTFT 降到约 1290 ms——<strong>约 5.7 倍提速</strong>，且零跨卡状态交换。</span>

这印证了同一件事：**只要前缀共享是收益来源，「把请求往它该去的卡上送」就与「把缓存建起来」同等重要。** 路由不是调度器的附属品，而是缓存体系的第一公里。

## 5 工程实现与易错辨析

Router 落地还有一些工程细节：各 Runtime 要在 Router 上注册为 worker，Router 维护其健康状态（失败率超限触发熔断（circuit breaker），把请求切走）；每台 Runtime 的近似树要按访问时间（last_access_time）做 LRU 剪枝，防止 Router 内存被树占满。

**辨析｜易错点：**

- **误区一：把 Router 当「万能多模型网关」。** 本节 Router 是**按模型**的，路由对象是同一模型的多份副本。多模型/多引擎混合路由是 sgl-model-gateway（SMG）那层的事，别混用。
- **误区二：以为 cache-aware 策略只认缓存。** 它有负载钳制：失衡时自动切「短队列 + 缓存亲和」。缓存命中的收益永远要跟「别把一台机器挤爆」放在同一个公式里。
- **误区三：以为 Router 的树是精确的。** Router 用字符级近似树预测命中，Runtime 用 token 级精确树定论。两者存在预测误差——这是「快速决策」与「精确计算」的分工代价。
- **误区四：以为前缀哈希就够了。** prefix-hash 保证同前缀同机器，但**不感知负载**——某台机器可能被热点前缀塞爆。cache-aware 存在的理由，正是补上这个「哈希看不到负载」的盲区。
- **误区五：单机也强行拆 Router/Runtime。** 单卡跑单进程模式就够；拆分的收益来自多卡多机与独立扩缩，小场景拆分徒增一跳网络延迟。

## 6 小结

- **前后端分离**：Router 接客决策（tokenize + 路由），Runtime 干活（调度 + RadixAttention + 推理）；单进程模式是合体版，生产可分离部署。
- **策略家族**：从轮询、随机、最小负载到前缀哈希、缓存感知；前六种只盯一面，cache-aware 同时看缓存与负载。
- **缓存感知公式**：命中比 $h_i$ 超过阈值则去命中最高者，否则去缓存树最小的机器；负载失衡时切「短队列 + 窗口内挑最长缓存前缀」。
- **实测收益**：多轮负载上平均 TTFT 降 51%（P99 降 67%）；DP 内 prefix-hash 分发在 Agent 负载上约 5.7 倍提速。
- **辨析**：Router 按模型路由、近似树 ≠ 精确树、前缀哈希不感知负载、小场景别强行拆分。

在下一节，我们离开 SGLang，进入英伟达的地盘。TensorRT-LLM 走的是另一条完全不同的路——不做 Python 里的灵活调度，而是**在编译期把整张计算图焊死**。这就是第五篇的开篇《TensorRT 图优化与算子重写》。
