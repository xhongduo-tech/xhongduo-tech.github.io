## 核心结论

vLLM 的高吞吐推理，本质是把“大批次一次性跑完”改成“请求和 token 按需流动地跑”。这里的**吞吐**，白话说就是“单位时间里总共能吐出多少个 token”。它依赖三件事同时成立：

1. **Continuous Batching（连续分批）**：白话说，不等一整批请求都结束，谁准备好了谁先进入下一轮计算。
2. **PagedAttention（分页注意力）**：白话说，把每个请求的 KV Cache 拆成固定大小的小页，用页表管理，像操作系统管内存页一样复用显存。
3. **异步调度**：白话说，prefill 和 decode 不再硬绑成一个整批的同步步骤，而是由调度器持续把空出来的算力塞满。

玩具例子可以把它想成多个请求排队过安检。传统静态批次像“凑满一队再一起进”，其中一个人行李很多，整队都慢；vLLM 则像“短请求先过、长请求分段过”，闸机一直有人在走，所以总流量更高。

在相同模型和上下文长度下，传统静态批次常见只有 15 到 20 tok/s，而 vLLM 的连续分批可以做到 60+ tok/s。这个量级差异说明，它优化的不是单个请求的理论算子效率，而是多请求混合场景下 GPU 的空转和等待。

| 方案 | 模型 | 上下文长度 | 调度方式 | 平均吞吐 |
|---|---|---:|---|---:|
| 传统静态批次 | 同一模型 | 长上下文混合 | 固定 batch，同步推进 | 15-20 tok/s |
| vLLM 连续分批 | 同一模型 | 长上下文混合 | 请求动态进出，token 级推进 | 60+ tok/s |

---

## 问题定义与边界

这个问题不是“怎么把 batch 调大”，而是“在请求到达时间不一致、上下文长度差异很大、输出长度也不可预测时，怎么让 GPU 尽量一直忙”。

**上下文长度**，白话说，就是模型在当前推理时需要看到的历史 token 数。一个用户只问一句话，可能只有几十个 token；另一个用户贴了 20 页文档，可能就是几千到上万个 token。传统静态批次的问题在这里暴露得很明显：

- 短请求会被长请求拖住
- batch 一旦固定，长度波动会让大量算力浪费在 padding 和等待上
- KV Cache 反复申请和释放容易造成显存碎片
- 请求流量一旦突发，排队时间会迅速放大

玩具例子可以把 GPU 想成餐厅厨房。传统方式像“等一整桌菜点齐才开炒”，其中有一道慢炖菜，整桌都得等；vLLM 更像“点一道做一道，灶台不断切换”，长菜继续炖，快菜先出，厨房利用率更高。

这里的边界也要说清楚。vLLM 主要解决的是：

- 多请求并发
- 上下文长度差异大
- 输出长度不可预测
- 需要把总 token/s 做高

它不直接保证：

- 单个请求的首 token 延迟一定最低
- 所有场景都比简单 batch 更划算
- 所有硬件上默认都能拿到最佳性能

如果你的服务只有少量请求、上下文很短、长度也很整齐，那么固定 batch 反而更简单，调试成本更低。

下面这个伪代码展示了问题本质：调度器不能按“请求”为单位同步推进，而要按“当前可计算的 token 片段”推进。

```python
# 伪代码：根据请求状态异步调度
while True:
    new_reqs = poll_new_requests()
    waiting.extend(new_reqs)

    # prefill 阶段：把新 prompt 编进 KV Cache
    while has_prefill_capacity() and waiting:
        req = pick_shortest_or_oldest(waiting)
        launch_prefill(req)

    # decode 阶段：只推进仍未结束的请求一个或多个 token
    runnable = [r for r in running if r.ready_for_decode()]
    batch = pack_by_token_budget(runnable)
    launch_decode(batch)

    reclaim_finished_requests()
```

---

## 核心机制与推导

Transformer 的核心计算没有变，还是注意力公式：

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这里的 $Q,K,V$ 分别是 Query、Key、Value。白话说，Query 是“当前要找什么”，Key 是“每段历史内容的索引”，Value 是“历史内容本体”。模型每生成一个新 token，都要拿当前 Query 去历史的 Key/Value 里查一遍相关内容。

问题在于，推理时历史会越来越长，所以每个请求都会积累一份 **KV Cache**。**KV Cache**，白话说，就是把历史 token 的 Key 和 Value 保存下来，避免每次从头重算。传统实现常把这块显存当连续大块来管，结果是：

- 长短请求混跑时，申请大小变化剧烈
- 请求结束后留下很多不规则空洞
- 新请求可能有总空间却拿不到连续空间
- 为了整理空间还会发生额外拷贝

vLLM 的 PagedAttention 直接借用了“分页”的思路。它不要求每个请求的 KV Cache 在物理显存里连续，只要求在逻辑上连续。可以写成：

$$
\text{logical\_block}_{i,j} \rightarrow \text{physical\_page}_{p}
$$

其中 $i$ 表示第 $i$ 个请求，$j$ 表示这个请求的第 $j$ 个逻辑块，页表记录它实际映射到哪个物理页。这样做的结果是：

- 分配单位固定，碎片更容易复用
- 请求增长时只需要追加页
- 请求结束时只回收对应页，不影响别的请求
- 避免了大块连续显存的强依赖

可以把它理解成翻书查资料。传统做法要求“这本书必须整本放在一张完整桌子上”；PagedAttention 则是“把书拆成页，页码单独索引，需要哪页就跳到哪页”。

第二个核心机制是 Continuous Batching。传统固定 batch 的推进方式大致是：

1. 收集一批请求
2. 一起做 prefill
3. 一起做 decode step 1
4. 一起做 decode step 2
5. 直到这一整批都结束，才换下一批

它的问题是批内最慢请求决定整体节奏。连续分批改成：

1. 新请求一来就尽快进入 prefill
2. decode 队列里谁还没结束，谁就参与下一轮
3. 某请求结束后立刻退出，不占后续 batch
4. 新请求可在中途加入，不必等老请求清空

于是 GPU 不再围绕“固定批次”转，而是围绕“当前可执行 token 集合”转。吞吐提升可以粗略理解为：

$$
\text{Throughput} \approx \frac{\text{单位时间完成的总 token 数}}{\text{计算时间 + 等待时间 + 内存管理开销}}
$$

vLLM 不是改变分子里的模型能力，而是显著压低分母中的等待和内存管理开销。

| 维度 | 传统固定 batch | 连续分批 |
|---|---|---|
| 请求进入时机 | 要等凑批 | 到达即可插入 |
| 短请求影响 | 被长请求拖慢 | 可提前退出 |
| 显存分配 | 常依赖连续大块 | 固定页复用 |
| 延迟表现 | 尾延迟容易放大 | 平均更稳，尾部更可控 |
| 吞吐表现 | 对长度波动敏感 | 对混合流量更友好 |

真实工程例子是检索增强问答系统。用户 A 只问“总结这篇文章”，用户 B 上传了长 PDF 并要求逐段分析，用户 C 在做多轮对话。三类请求上下文长度差异很大。如果采用固定 batch，B 会不断拖慢 A 和 C；如果采用 vLLM，B 的长上下文会占更多页，但 A 和 C 仍可在 decode 队列中快速进出，从而保证整体 token/s。

---

## 代码实现

工程上要把这个机制落地，核心不是几行 API，而是“请求生命周期”和“KV 页生命周期”是否被正确解耦。一个典型流程可以概括成：

请求进入 → 调度器判定是 prefill 还是 decode → 分配或复用 KV 页 → 进入当前 token 预算允许的 batch → 执行 attention/backend kernel → 更新页表和请求状态 → 已完成请求释放页

下面先给一个可运行的玩具代码，用 Python 模拟“连续分批比固定分批更少等待”。它不是 vLLM 源码，只是帮助理解调度行为。

```python
from dataclasses import dataclass

@dataclass
class Req:
    name: str
    total_tokens: int
    done: int = 0

    def finished(self) -> bool:
        return self.done >= self.total_tokens

def static_batch_steps(requests):
    # 固定 batch：所有请求都要跟着最慢的那个走完
    max_tokens = max(r.total_tokens for r in requests)
    return max_tokens * len(requests)

def continuous_batch_steps(requests):
    # 连续分批：每轮只处理仍未结束的请求
    active = [Req(r.name, r.total_tokens, 0) for r in requests]
    steps = 0
    while not all(r.finished() for r in active):
        for r in active:
            if not r.finished():
                r.done += 1
                steps += 1
    return steps

reqs = [Req("short", 2), Req("mid", 4), Req("long", 8)]

static_cost = static_batch_steps(reqs)
continuous_cost = continuous_batch_steps(reqs)

assert static_cost == 24
assert continuous_cost == 14
assert continuous_cost < static_cost

print(static_cost, continuous_cost)
```

这个玩具例子里，固定 batch 的“总步数”是 $8 \times 3 = 24$，因为三个请求都被最慢的那个 8 token 请求拖着走；连续分批只做真实需要的 14 次推进，差值就是等待和空转。

再看一个更接近工程的伪代码，重点放在调度器和分页 KV：

```python
class PageTable:
    def __init__(self, page_size, total_pages):
        self.page_size = page_size
        self.free_pages = list(range(total_pages))
        self.mapping = {}  # req_id -> [physical_page_id, ...]

    def alloc_for_tokens(self, req_id, num_tokens):
        need = (num_tokens + self.page_size - 1) // self.page_size
        pages = [self.free_pages.pop() for _ in range(need)]
        self.mapping.setdefault(req_id, []).extend(pages)

    def release(self, req_id):
        for p in self.mapping.pop(req_id, []):
            self.free_pages.append(p)

class Scheduler:
    def __init__(self, token_budget, backend):
        self.waiting = []
        self.running = []
        self.token_budget = token_budget
        self.backend = backend

    def schedule(self, request):
        self.waiting.append(request)

    def tick(self):
        self._launch_prefill()
        batch = self._build_decode_batch()
        self.backend.decode(batch)
        self._reclaim_finished()

    def _launch_prefill(self):
        while self.waiting and self._has_prefill_slot():
            req = self.waiting.pop(0)
            req.prefill()
            self.running.append(req)

    def _build_decode_batch(self):
        batch, used = [], 0
        for req in self.running:
            if req.finished:
                continue
            if used + 1 <= self.token_budget:
                batch.append(req)
                used += 1
        return batch
```

如果把它映射到真实系统，关键配置通常集中在三类：

| 配置项 | 作用 | 配错后果 |
|---|---|---|
| async scheduling | 让 prefill 与 decode 更连续地交错 | 短请求排队，GPU 阶段性空转 |
| PagedAttention / paged KV cache | 固定页管理 KV，减少碎片 | 长时间运行后显存效率下降 |
| hardware-specific backend | 针对 CUDA、ROCm 等选择合适 kernel | 理论可跑，但吞吐明显偏低 |

真实工程例子可以看生产检索问答服务。比如电商问答或职场搜索这类系统，白天流量有明显峰值，且问题长度分布极不均匀。服务端常见做法是：

- 新请求先做 tokenizer 和长度估计
- 调度器根据 token budget 决定进入 prefill 还是排队
- KV Cache 按页分配到 GPU 显存
- decode 以 token 为粒度持续填充当前 batch
- 请求完成后立即释放页，供后续请求复用

这类模式已经被真实产品采用，原因不是“概念先进”，而是它更适合长期在线、混合负载、响应波动大的服务环境。

---

## 工程权衡与常见坑

第一类坑是只开了连续分批，却没有把 prefill 和 decode 的调度真正拆开。**prefill**，白话说，就是先把用户输入 prompt 编成首批 KV；**decode** 是后续逐 token 生成。如果这两阶段仍然大步同步推进，短请求还是会卡在长 prompt 后面。

第二类坑是没有分页 KV。系统刚启动时往往看不出问题，但跑久了以后，显存碎片会让“理论剩余显存很多，实际可用空间却不足”的情况出现。表现通常是：

- 可并发请求数逐渐下降
- 尾延迟变差
- 频繁触发 OOM 或回退

第三类坑是忽略硬件后端。attention kernel 对不同硬件差异很大，特别是 CUDA 和 ROCm 的实现路径不一样。如果后端没有针对硬件调优，吞吐损失可能达到 1.2x 甚至更多。这里的**backend**，白话说，就是真正执行 attention 和采样等核心算子的底层实现。

玩具例子可以继续用“备料+上菜”理解。若厨房必须等所有备料都完成才开始上菜，短单就会被饿住；若食材没有标准盒分装，而是散着堆，找料和清空位置都会越来越慢。

| 缺少的策略 | 典型症状 | 开启后的改善 |
|---|---|---|
| 不开 async scheduling | 短请求排队，阶段性空转 | prefill/decode 可交错，平均吞吐提升 |
| 不开 PagedAttention | 显存碎片，长期运行退化 | 页级回收复用，稳定性更好 |
| 不选专用 backend | GPU 利用率上不去 | attention kernel 更贴合硬件 |

建议设置可以直接列成检查清单：

- 开启 `async scheduling`
- 开启 `PagedAttention` 或等价 paged KV 管理
- 按硬件选择对应 attention backend
- 对请求设置 token budget，而不是只看请求数
- 监控 `throughput`、`TTFT`、`tail latency`、`KV cache usage`

其中 **TTFT**，白话说，就是“首个 token 返回时间”。吞吐高不等于 TTFT 一定低，工程上必须同时看两组指标，否则很容易把“平均 token/s 很好看”误判成“用户体验很好”。

---

## 替代方案与适用边界

vLLM 不是所有推理服务的默认答案。它适合的是“请求多、长度杂、流量波动大”的场景。如果业务很简单，替代方案可能更省事。

| 方案 | 适用场景 | 优点 | 局限 | 硬件要求 |
|---|---|---|---|---|
| vLLM | 高并发、长短上下文混合、多模型服务 | 吞吐高，动态调度强，显存利用率好 | 系统复杂度更高，调优项更多 | 对 backend 和显存管理更敏感 |
| 静态 batch | 请求少、长度接近、离线任务 | 实现简单，行为稳定 | 被最长请求拖慢，对波动不友好 | 通用 GPU 即可 |
| 简单并发 + FP16 | 单模型、短上下文、成本敏感 | 开发快，部署简单 | 吞吐上限较低，长上下文退化明显 | 要求最低 |

玩具例子可以看小餐厅和大型快餐连锁。小餐厅顾客少，菜单固定，手工排单就够；大型快餐连锁高峰期订单不断，还分堂食、外卖、自提，必须靠流水线和标准化周转。

因此边界可以这样划：

- 如果请求数量有限、上下文长度一致，静态 batch 足够稳定
- 如果只有单一模型、短上下文、低预算，简单并发可能更合适
- 如果是在线产品、长上下文混合、多租户并发，vLLM 的优势才会显著放大

判断标准不要只看“能不能跑”，而要看“在真实流量下是否还能稳定把 GPU 吃满”。

---

## 参考资料

下表列的是入门和工程实现最值得看的几类资料，分别覆盖机制、调度、生态和硬件支持。

| 文献名称 | 发布日期 | 关注点 |
|---|---|---|
| Inside vLLM: Anatomy of a High-Throughput LLM Serving System | 2025-09-05 | 核心架构、PagedAttention、调度机制 |
| vLLM Continuous Batching for High-Throughput Serving for Long Contexts | 2025-11-23 | 连续分批原理、长上下文吞吐数据 |
| vLLM 2024 Retrospective & 2025 Vision | 2024/2025 | 生态落地、生产案例、多模型支持 |
| Beyond Porting: AMD ROCm for vLLM | 2026-02-27 | 硬件后端、ROCm 优化、跨厂商支持 |
| vLLM 官方博客与文档 | 持续更新 | 部署参数、后端支持、版本演进 |

如果是零基础到初级工程师，阅读顺序建议是：

1. 先看 Anatomy，建立“连续分批 + 分页 KV”的主框架。
2. 再看 Continuous Batching 的数据案例，理解为什么吞吐能翻倍。
3. 最后看 Vision 和 ROCm 文章，把机制映射到真实硬件和生产部署。
