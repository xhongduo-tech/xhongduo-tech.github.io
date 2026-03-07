## 核心结论

Continuous Batching，中文常叫“连续批处理”或“持续批处理”，本质是**把批处理的调度粒度从“整批结束”改成“每轮生成一个 token 后都检查一次”**。白话说，它不再要求一组请求一起开始、一起结束，而是谁先结束就立刻腾出位置，马上塞入下一个等待请求。

这个变化解决的是 LLM 在线推理里最常见的低效来源：**短请求被长请求拖住**。传统静态批处理里，只要一个 batch 中混入长序列，整批的释放时间就被最长序列锁死，短序列即使早就生成完，也只能空等。Continuous Batching 则让 GPU 上的 slot 像停车位一样动态流转，尽量保持“满位运行”。

先看一个玩具例子。假设 batch 最多容纳 3 条请求，当前有 5 个请求要生成的 token 数分别是：

| 请求 | 需生成 token 数 |
|---|---:|
| A | 20 |
| B | 80 |
| C | 80 |
| D | 20 |
| E | 20 |

静态批处理先跑 A、B、C。A 在第 20 轮就结束了，但 D 不能进来，必须等 B、C 也都跑完到第 80 轮。于是第 21 到 80 轮里，GPU 实际只跑 2 条活跃序列。Continuous Batching 在第 20 轮后立刻把 D 塞进去，第 40 轮再把 E 塞进去，GPU 长时间都保持接近满载。

它的价值不是“单条请求更快”，而是**整体吞吐量更高**。吞吐量，白话说就是单位时间内系统能完成多少输出 token 或多少请求。在长短请求混合的在线服务里，Continuous Batching 往往比静态批处理高出约 2 到 4 倍，这也是 Orca、vLLM 这类现代 LLM serving 系统把它当成核心机制的原因。

---

## 问题定义与边界

先定义问题。LLM 推理通常分两段：

1. **Prefill**：把输入 prompt 一次性喂进去，建立初始 KV cache。
2. **Decode**：每轮生成一个新 token，并把它继续追加回上下文。

KV cache，白话说就是模型为了避免重复计算，把前面 token 的中间结果缓存起来。在线对话系统的大头通常在 decode，因为它是逐 token 推进的。

传统静态批处理的问题出现在 decode 阶段最明显。它的规则很简单：先凑一批请求，一起跑；这一批没全部结束之前，不接新请求进入这个 batch。于是：

- 最长序列决定这一批的释放时刻；
- 短序列提前完成也无法被替换；
- 活跃 batch 大小会随时间下降；
- GPU 利用率跟着掉下来。

可以用一个文字图示表示：

| 调度方式 | 流程 |
|---|---|
| 静态批处理 | 组 batch → 一直跑到整批完成 → 再装入下一批 |
| Continuous Batching | 组初始 batch → 每轮 decode 后检查完成项 → 立即补入 waiting queue 中的新请求 |

新手可以把它理解成餐厅叫号系统。静态批处理像“这一桌 4 个人必须全部吃完，才允许下一桌任何一个人入场”；Continuous Batching 像“只要空出一个座位，就立刻叫下一个人进来”。如果客人用餐时间差异很大，后者明显更高效。

但它也有边界，不是任何场景都收益巨大。

| 场景 | Continuous Batching 收益 |
|---|---|
| 在线推理、请求长度差异大 | 高 |
| 短 prompt + 长 decode 的对话场景 | 高 |
| waiting queue 经常为空 | 低 |
| 所有请求长度接近 | 中等或偏低 |
| 单个超长任务独占系统 | 很低 |

所以问题边界很明确：Continuous Batching 主要解决的是**有等待队列、请求长度分布不均、希望提升整体吞吐**的在线 serving 场景。它不是离线单任务推理的万能加速器，也不是所有延迟指标都会自动变好。

---

## 核心机制与推导

可以把连续批处理抽象成一个随时间变化的系统。

设：

- $B_{\max}$：系统允许的最大 batch slot 数；
- $B(t)$：时刻 $t$ 的活跃序列数；
- $R(t)$：时刻 $t$ 的等待队列长度；
- $I$：一次 decode iteration 的平均耗时。

静态批处理的核心问题是，虽然初始时 $B(t)=B_{\max}$，但随着短序列陆续完成，$B(t)$ 会不断下降，直到整批彻底结束才能回升。于是很多时间片里 GPU 并没有满载。

Continuous Batching 的目标则是尽量保持：

$$
B(t)=\min(B_{\max},\ \text{running}(t)+R(t))
$$

更直观地说，只要 waiting queue 里还有请求，且 batch 有空槽，就应该立刻补满。

吞吐量可以粗略写成对时间的累计：

$$
T \propto \int \frac{B(t)}{I}\,dt
$$

这里的意思不是严格物理公式，而是一个系统视角：在 iteration 成本 $I$ 近似稳定时，吞吐量主要由“每个时刻有多少活跃序列在跑”决定。让 $B(t)$ 更长时间贴近 $B_{\max}$，总吞吐自然更高。

下面看一个更具体的时间线。

batch 容量为 3，请求长度依次为：20、80、80、20、20。

| 时间区间 | 静态批处理活跃序列数 | Continuous Batching 活跃序列数 |
|---|---:|---:|
| 1-20 | 3 | 3 |
| 21-40 | 2 | 3 |
| 41-60 | 2 | 3 |
| 61-80 | 2 | 2 |

解释如下：

- 第 1 到 20 轮，A/B/C 都在跑，两者一样。
- 第 20 轮后，A 完成。
- 静态批处理不能插入 D，所以只剩 B/C 两条。
- Continuous Batching 立刻插入 D，继续保持 3 条。
- 第 40 轮后，D 又完成，再插入 E。

如果把“活跃序列数”看作 GPU 的有效占用，那么 Continuous Batching 明显更高。

这个机制和停车场补位非常像。停车场容量是 100 辆车，传统静态批处理像“早上先放 100 辆进去，直到全部离开前不准新车进场”；Continuous Batching 则是“任何一辆开走，门口等待区马上补进一辆”。停车位的周转率越高，停车场的总服务量越大。

这里还要区分两个概念：

| 概念 | 含义 |
|---|---|
| 静态批处理 | 批大小在整个 batch 生命周期内固定，不替换成员 |
| 迭代级调度 | 每轮 token 生成后都允许调整 batch 成员 |
| Continuous Batching | 迭代级调度在 LLM serving 中的典型实现方式 |

Orca 论文强调的是 iteration-level scheduling，即“按迭代而不是按请求整体来调度”；vLLM 则把这件事和高效的 KV cache 管理结合起来，形成工程上真正可扩展的系统。

---

## 代码实现

下面先给一个最小玩具实现。它不做真实神经网络计算，只模拟“每轮每个请求消耗 1 个 token 配额”，用来观察静态批处理和 Continuous Batching 的完成时间差异。

```python
from collections import deque

def simulate_static(lengths, batch_size):
    waiting = deque({"id": i, "remain": x} for i, x in enumerate(lengths))
    time = 0
    finished = []

    while waiting:
        batch = []
        while waiting and len(batch) < batch_size:
            batch.append(waiting.popleft())

        while batch:
            time += 1
            next_batch = []
            for req in batch:
                req["remain"] -= 1
                if req["remain"] == 0:
                    finished.append((req["id"], time))
                else:
                    next_batch.append(req)
            batch = next_batch

    return finished

def simulate_continuous(lengths, batch_size):
    waiting = deque({"id": i, "remain": x} for i, x in enumerate(lengths))
    running = []
    time = 0
    finished = []

    while waiting or running:
        while waiting and len(running) < batch_size:
            running.append(waiting.popleft())

        time += 1
        survivors = []
        for req in running:
            req["remain"] -= 1
            if req["remain"] == 0:
                finished.append((req["id"], time))
            else:
                survivors.append(req)
        running = survivors

        while waiting and len(running) < batch_size:
            running.append(waiting.popleft())

    return finished

lengths = [20, 80, 80, 20, 20]
static_res = dict(simulate_static(lengths, 3))
cont_res = dict(simulate_continuous(lengths, 3))

assert static_res[3] == 100
assert cont_res[3] == 40
assert cont_res[4] == 60
assert cont_res[3] < static_res[3]
```

这段代码里最关键的区别只有一句话：**在每轮结束后，Continuous Batching 会再次检查 waiting queue，把空槽补满**。这就是它的本质。

如果写成更接近真实 serving 系统的伪代码，大致如下：

```python
def scheduler_loop(waiting_queue, running_batch, max_batch_size):
    while waiting_queue or running_batch:
        while len(running_batch) < max_batch_size and waiting_queue:
            req = waiting_queue.pop_left()
            allocate_kv_blocks(req)
            running_batch.add(req)

        outputs = model.decode_one_step(running_batch)

        finished = []
        for req, token in outputs:
            req.append_token(token)
            if req.is_finished():
                finished.append(req)

        for req in finished:
            running_batch.remove(req)
            release_or_recycle_kv_blocks(req)

        while len(running_batch) < max_batch_size and waiting_queue:
            req = waiting_queue.pop_left()
            allocate_kv_blocks(req)
            running_batch.add(req)
```

这段伪代码对应三个工程动作：

| 步骤 | 作用 |
|---|---|
| `allocate_kv_blocks(req)` | 为请求分配 KV cache 空间 |
| `decode_one_step` | 对当前 running batch 做一轮 token 生成 |
| `release_or_recycle_kv_blocks(req)` | 请求结束后回收或复用缓存页 |

真实工程例子是 vLLM。它不是简单地把所有序列 pad 到一样长后再做大矩阵运算，而是结合 **PagedAttention** 管理 KV cache。PagedAttention 可以理解成“把连续上下文拆成固定大小的数据页进行映射”，类似操作系统按页管理内存。这样做的价值是：

- 不必为每个请求预留一整段很长的连续 KV cache；
- 不同长度请求可以按需增长；
- 序列结束后，页块可以回收给新请求；
- 长短请求混跑时，内存碎片和浪费明显减少。

这就是为什么现代系统里常说：**Continuous Batching 解决调度层面的浪费，PagedAttention 解决内存层面的浪费**。两者合起来，才构成完整的高吞吐 LLM serving 方案。

---

## 工程权衡与常见坑

Continuous Batching 不是“写一个 while 循环补位”就结束了，真正难点在工程细节，尤其是 KV cache、调度粒度和资源上限。

先看常见坑：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| 对可变长度序列统一大 padding | KV cache 和计算量暴涨 | 改用按需分配或分页式 KV 管理 |
| 每轮不检查 waiting queue | 退化回接近静态批处理 | 在每次 decode 前后都做补位检查 |
| 只看请求数，不看 token 预算 | 显存被长上下文压爆 | 用 token budget 或 block budget 限流 |
| 结束请求后不及时回收 cache | 可用 slot 变少，吞吐下降 | 做页块回收和复用 |
| 把 prefill 和 decode 混成一个统一策略 | 短请求延迟抖动大 | 对 prefill/decode 分阶段调度 |
| 调度锁过重 | CPU 调度开销反而上升 | 用轻量队列和低争用数据结构 |

第一类坑是最常见的：**只解决了调度，没有解决内存**。  
如果你让不同长度的序列都 pad 到同一长度，那么短序列虽然能动态补入，但它们仍然会占着大量无效的 KV cache 空间。尤其在长上下文场景里，内存很快就成瓶颈。也就是说，调度补位了，显存却先满了，系统还是跑不快。

第二类坑是调度退化。很多人实现“动态批处理”时，只在一批完全结束后再收新请求，或者只在若干轮后统一收一次。这种做法名字可能叫动态，但如果空槽不能立即补位，效果就和静态差不多。

第三类坑是指标理解错误。Continuous Batching 提升的首先是**整体吞吐**，不是必然降低每个请求的首 token 延迟。首 token 延迟，白话说就是用户发出请求后，第一次看到输出的等待时间。如果系统为了提高吞吐一直倾向于补满 batch，那么某些短请求的首 token 时间未必最优。因此生产系统通常要在吞吐、首 token 延迟、尾延迟之间做权衡。

可以用一个真实工程判断标准来理解：

| 指标 | 更关注什么 | Continuous Batching 表现 |
|---|---|---|
| 吞吐量 | 单位时间处理多少 token/请求 | 通常显著更好 |
| 平均延迟 | 单请求整体完成时间 | 常有改善，但依赖负载 |
| 首 token 延迟 | 第一个输出出现多快 | 不一定最优 |
| 尾延迟 | 最慢请求的延迟 | 依赖调度策略，可能波动 |

所以在生产上，Continuous Batching 往往不会单独存在，而会和优先级队列、最大并发 token 数、请求分类路由一起使用。

---

## 替代方案与适用边界

Continuous Batching 不是唯一方案。它的替代方案主要有三类。

| 方案 | 核心思路 | 优点 | 缺点 |
|---|---|---|---|
| 静态批处理 | 一批请求固定到结束 | 实现简单，控制稳定 | 长短混跑效率差 |
| Selective Batching | 只对部分阶段或部分算子做合并 | 改造成本较低 | 仍难解决整批等待问题 |
| Continuous Batching | 每轮都可补入新请求 | 吞吐高，适合在线混合负载 | 调度和内存管理复杂 |

Selective Batching，白话说就是“并不是所有东西都一起批，只把适合合并的那部分算子拿来合并”。Orca 论文中除了 iteration-level scheduling，也讨论了 selective batching 的思路。它比纯静态批处理更灵活，但如果系统仍然不能在序列结束后立刻补位，那么最长序列仍然会成为主要瓶颈。

什么时候不必强上 Continuous Batching？

| 场景 | 更合适的做法 |
|---|---|
| waiting queue 长期为空 | 简单静态批处理即可 |
| 请求长度非常接近 | 静态批处理收益损失不大 |
| 单租户离线批量任务 | 直接按吞吐最优的固定批次跑 |
| 调度线程已是瓶颈 | 先优化队列和锁，再谈连续批处理 |
| 超长上下文任务为主 | 先解决显存和 KV cache 问题 |

这里有一个容易忽视的边界：如果线上几乎只有一个长请求在跑，Continuous Batching 无法凭空制造并行度。没有等待队列，就没有补位对象。这时维护复杂调度器反而增加额外开销。换句话说，Continuous Batching 的前提是“有足够多、长度差异明显、可被动态插入的请求”。

因此更准确的结论应该是：

- 当负载是**多请求、长度不均、持续到达**时，Continuous Batching 很值得；
- 当负载是**稀疏、单一、长度接近**时，静态批处理可能已经足够；
- 当负载是**超长上下文、显存紧张**时，PagedAttention 一类内存机制甚至比调度本身更关键。

---

## 参考资料

| 资料 | 类型 | 核心贡献 | 链接 |
|---|---|---|---|
| Orca: A Distributed Serving System for Transformer-Based Generative Models | 论文 / OSDI 2022 | 提出 iteration-level scheduling 与 selective batching，奠定 Continuous Batching 的系统思路 | https://www.usenix.org/conference/osdi22/presentation/yu |
| vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention | 论文 | 将 Continuous Batching 与 PagedAttention 结合，解决高吞吐与 KV cache 管理问题 | https://arxiv.org/abs/2309.06180 |
| vLLM Official Documentation | 项目文档 | 展示工程实现、API 与 serving 架构细节 | https://docs.vllm.ai/ |
| Optimization for Modern LLM Online Serving: Continuous Batching and PagedAttention | 技术博客 | 用系统视角解释 Continuous Batching 与 PagedAttention 的作用和适用场景 | https://llmsys.net/2023/12/10/Optimization-for-Modern-LLM-Online-Serving-Continuous-Batching-and-Paged-Attention/ |
| Emergent Mind 对 vLLM 论文的整理页 | 资料汇总 | 汇总论文摘要、背景和工程收益数据，便于快速建立全局认知 | https://www.emergentmind.com/papers/2309.06180 |

1. 先读 Orca，理解“为什么调度粒度必须从整请求下沉到单 iteration”。
2. 再读 vLLM，理解“为什么只有调度还不够，还必须同时重做 KV cache 管理”。
3. 最后结合 vLLM 文档和 LLMSYS 博客，把论文概念映射到真实 serving 系统。
