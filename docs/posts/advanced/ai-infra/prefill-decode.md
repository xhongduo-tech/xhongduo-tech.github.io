---
title: 推理的两个阶段：prefill 与 decode 的资源特征
date: 2026-08-07
---

# 推理的两个阶段：prefill 与 decode 的资源特征

<div class="epigraph">
<p>一次推理，一半是阅读，一半是书写。</p>
<footer>—— 克里斯托弗 · 曼宁（Christopher Manning，斯坦福 NLP 教授）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ LLM 推理系统文献（vLLM / SGLang）· 推理基础设施篇 ｜ 2026-08-07</p>
</div>

## 为什么从 prefill 与 decode 开始

上一次推理请求，GPU 其实做了两种截然不同的活：先「读一遍你的 prompt」（**prefill**），再「一个字一个字地写回答」（**decode**）。这两阶段的计算特征、瓶颈、优化手段完全不同——**理解它们的分野，是推理性能优化的第一课**。

prefill 是「算力密集、并行度高」，decode 是「带宽密集、串行低效」。一个推理系统如果对两阶段一视同仁，就会「既慢又浪费」——这正是 PD 分离（把两阶段放到不同 GPU）等高级架构的前提。本篇把两阶段的资源特征与优化含义讲透。

## 1 prefill：并行地读你的 prompt

**prefill（预填充）** 是推理的第一步：把用户输入的整个 prompt 一次性前向计算，生成第一个 token 并缓存 KV。

它的特征：

- **并行度高**：整条 prompt 的所有 token 同时前向，像训练一样「批量」处理——**算力可以打满**。
- **计算密集**：计算量 $\propto$ prompt 长度 $P \times$ 参数量 $N$，是「算力密集型」阶段。
- **耗时短**：prompt 通常几百到几千 token，prefill 占整次请求时间的很小部分。
- **产生 KV Cache**：把 prompt 的所有 K/V 写进缓存，供 decode 复用。

**prefill 的指标是 TTFT（Time To First Token）**——用户看到第一个字要等多久。<span class="marginnote">prefill 虽然耗时短，但它「吃算力」的特性在并发高时很伤：多个请求同时 prefill，算力被瞬间瓜分，TTFT 飙升。所以推理系统常给 prefill 单独的资源配额（或干脆 PD 分离），避免 decode 的轻量请求被 prefill 的重量请求堵住。</span>

## 2 decode：串行地写你的回答

**decode（解码）** 是推理的第二步：逐个生成输出 token，每步把「新 token」拼进输入继续前向。

它的特征：

**串行依赖**：第 $k+1$ 个 token 依赖前 $k$ 个——**无法并行**，必须一步步走。
**带宽密集**：每步只处理「一个 token」，计算量极小（$2N$），但要把整个权重 $2N$ 从显存读一遍——**瓶颈在显存带宽，不在算力**。
**耗时主导**：生成 $S$ 个 token 需要 $S$ 步，decode 占整次请求时间的绝大部分。
**读多算少**：每步的「计算/读权重」比极低，GPU 算力大量闲置。

**decode 的指标是 TPOT（Time Per Output Token）**——每个输出 token 的间隔。<span class="marginnote">decode 的「带宽密集」是推理最反直觉的一点：算力最强的 GPU 在 decode 阶段可能只用到 20%–30% 的算力，因为它在「等权重从显存搬到计算单元」。这也是为什么推理选 GPU 要看显存带宽（本专题专门一篇）——decode 的速度基本由带宽决定。</span>

## 3 资源画像对比

两阶段的资源需求可以用一张表看清：

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 计算模式 | 并行（整条 prompt） | 串行（逐步） |
| 瓶颈资源 | 算力（FLOPs） | 显存带宽 |
| 计算/带宽比 | 高 | 低 |
| 耗时占比 | 小（<10%） | 大（>90%） |
| 关键指标 | TTFT | TPOT |
| 显存增量 | 一次性写 KV | 每步追加 KV |

**核心矛盾**：同一块 GPU 要同时满足「prefill 要算力」与「decode 要带宽」——而这两者很难在同一张卡上同时最优。<span class="marginnote">「一张卡同时干两件资源特征相反的事」是推理系统的深层痛点：prefill 卡住时算力紧张、decode 卡住时带宽闲置。要么用调度把两类请求错峰（continuous batching 的分层调度），要么物理分离（PD disaggregation）——后者是本专题后文的主角。</span>

## 4 为什么两阶段不能一视同仁

如果推理系统对 prefill 与 decode 用同一套策略，会出现两败俱伤：

**让 decode 等 prefill**：一个长 prompt 的 prefill 占住 GPU 算力，所有在 decode 的短请求被拖慢——TPOT 飙升。
**让 prefill 等 decode**：decode 的每一步都很短，但它「占着」GPU 的调度槽，prefill 的并行优势发挥不出来——TTFT 飙升。
**混合排队**：两者竞争同一资源池，谁的延迟都无法保证。

于是现代推理系统都做**分层调度**：把 prefill 请求与 decode 请求分开排队、分优先级、甚至分 GPU。**「两类负载分开处理」是推理系统设计的核心原则**。<span class="marginnote">连续批处理（continuous batching）里最常见的实现是「prefill 优先 + decode 交错」：新请求的 prefill 可以插入到 decode 的「token 间隙」里执行——因为 decode 每步只有几毫秒，插一个 prefill 进去刚好填满带宽空闲。这种「见缝插针」是两阶段共存的基础技巧。</span>

## 5 公式解析：两阶段的耗时模型

设参数量 $N$、prompt 长度 $P$、输出长度 $S$、单卡算力 $C$（FLOPs）、显存带宽 $B$（字节/秒）。

**Prefill 耗时**（计算密集，由算力主导）：

$$T_{\text{prefill}} \approx \frac{2NP}{C}$$

**Decode 每步耗时**（带宽密集，由带宽主导）：

$$T_{\text{decode}} \approx \frac{2N}{B}$$

- **$\frac{2NP}{C}$（prefill）**：计算量 $2NP$ 除以算力。prompt 越长越重，靠并行算力摊薄。
- **$\frac{2N}{B}$（decode 每步）**：每步读一遍权重 $2N$ 除以带宽。**与输出长度无关，只与模型大小与带宽有关**——这就是「decode 瓶颈在带宽」的数学表达。
- **总延迟**：$T_{\text{total}} = \frac{2NP}{C} + S \cdot \frac{2N}{B}$。<span class="marginnote">代入数字感受：70B 模型、BF16 权重 140GB，H100 带宽约 3.35TB/s，decode 每步 ≈ 140GB/3.35TB/s ≈ 42ms——即每 token 约 42ms，约 24 token/s。要提高这个数字，要么量化权重（减 $2N$）、要么上更高带宽的卡——算力再强也帮不上 decode 的忙。</span>

## 6 辨析｜易错点：两阶段的常见误区

**辨析｜易错点：**
- **「GPU 算力强推理就快」是错觉**：decode 由带宽决定，算力再强也白搭；prefill 才吃算力。
- **「两阶段一起优化」是次优**：资源特征相反，分开调度/分离部署才是正解。
- **「TTFT 好就是延迟好」不完整**：TTFT 只反映 prefill；总体验由 TTFT + S·TPOT 共同决定。
- **「decode 不占算力所以不用管」**：decode 虽然算力利用率低，但它占「带宽」，并发高时带宽被打满，一样变慢。
- **别把「prefill 的耗时」当「延迟的主体」**：长输出下 decode 才是延迟主体；prompt 极长时才反过来。

## 7 小结

- **Prefill**：并行读 prompt、算力密集、TTFT 指标、耗时占比小。
- **Decode**：串行写输出、带宽密集、TPOT 指标、耗时占比大。
- **资源画像相反**：prefill 要算力、decode 要带宽，一张卡难两全。
- **核心原则**：两类负载分开排队/调度/部署，避免互相拖累。
- **耗时模型**：$T = \frac{2NP}{C} + S \cdot \frac{2N}{B}$——prefill 看算力、decode 看带宽。

## 8 进阶与延伸

**动手实测两阶段的耗时**：用一个 LLM 推理引擎（如 vLLM），对一个长 prompt（如 2000 token）发请求，用 profiler 拆出 prefill 与 decode 各自的时间——你会看到「prefill 几毫秒、decode 占 90%+」的结构，并验证本篇的耗时公式。

**几个值得进一步挖的方向**：

- **TTFT 与 TPOT 的权衡**：continuous batching 里「prefill 优先」保 TTFT 但伤 TPOT，「decode 优先」反之——怎么用「SLA 加权」在两个指标间找平衡？
- **长 prompt 的 prefill 危机**：RAG 场景 prompt 有几千 token，prefill 的算力需求暴增——「prefill 占 20%+ 时该上 PD 分离」的判据怎么量化？
- **speculative decoding 的视角**：投机解码（小模型猜、大模型验）改变的是「decode 的步数」——它优化的是 TPOT 还是 TTFT？想清楚这个，你就理解了投机解码的定位。

**自测题**：为什么 decode 每步耗时与「输出长度」无关？如果你能说清「每步都读一遍权重、与读到哪无关」，就抓住了 decode 的带宽本质。

## 9 动手实践清单

- 用推理引擎对长 prompt 发请求，拆出 prefill 与 decode 的耗时。
- 用「$T = 2NP/C + S \cdot 2N/B$」算你的模型总延迟。
- 观察 decode 每步耗时是否「与输出长度无关」。
- 用 continuous batching 里「prefill 优先 vs decode 优先」对比 TTFT 与 TPOT。
- 测 RAG 长 prompt 场景的 prefill 占比，判断是否该上 PD 分离。
- 用 profiler 确认 decode 阶段的算力闲置（带宽瓶颈）。
- 画「prefill 要算力、decode 要带宽」的资源画像图。

在下一节，我们把推理系统放大到集群——**推理集群架构**：路由层、推理实例与扩缩容策略。
