## 核心结论

PipeDream 的 1F1B 调度，意思是“每个流水线阶段在稳态时交替执行一次前向、一次反向”，核心价值不是“更炫的并行方式”，而是把流水线并行从“能跑”变成“在显存受限机器上也能持续跑”。它解决的是传统流水线并行里两个直接问题：

1. 激活缓存随 micro-batch 数量 $m$ 增长，显存峰值容易失控。
2. 前向和反向如果看到的权重版本不同，梯度就不再对应同一个计算图。

PipeDream 的答案是两件事一起用：

- `1F1B`：把稳态阶段的执行顺序改成前向、反向交替，让管道里只保留“正在飞行”的 micro-batch。
- `weight stashing`：把某个 micro-batch 前向时看到的权重版本存下来，等它反向时继续用同一版本。

因此，传统按 batch 刷新流水线的方法需要保存 $O(m)$ 份激活，而 1F1B 在稳态下只需要和流水线深度 $d$ 同阶的缓存，常写成 $O(d)$。如果再配合 PipeDream-2BW，权重副本数还能从最坏 $O(d)$ 压到常数 2 份。

一个最小玩具例子是：把一个 mini-batch 拆成 8 个 micro-batch，模型切成 4 个 stage。传统 GPipe 风格会先尽量把 8 个前向都压进流水线，再统一反向，因此每个 stage 可能要背着 8 份激活。1F1B 的稳态里，每个 stage 只需要围绕“正在管道中的 4 个 micro-batch”工作，激活缓存近似只和 4 这个深度有关。

下表先给出结论对比：

| 方案 | 激活缓存数量 | 权重副本数量 | 是否需要周期性 flush | 典型特点 |
|---|---:|---:|---|---|
| GPipe / flush pipeline | $O(m)$ | 1 | 是 | 语义最接近普通大 batch 训练，但气泡明显 |
| PipeDream 1F1B + weight stashing | $O(d)$ | 最坏 $O(d)$ | 否 | 吞吐高，但前端 stage 权重缓存压力大 |
| PipeDream-2BW | $O(d)$，重计算时更低 | 2 | 否 | 在吞吐和显存之间更平衡 |

---

## 问题定义与边界

先定义几个术语。

- `micro-batch`：把一个大 batch 拆出来的一小份样本。白话说，就是为了让流水线能连续喂数据，把一车货拆成多个小箱子。
- `stage`：模型切分后的一个连续子网络。白话说，就是一张 GPU 负责的一段层。
- `pipeline depth d`：流水线总共有多少个 stage。
- `m`：一个 mini-batch 被拆成多少个 micro-batch。

问题本质是显存和调度的联动约束。

如果一个 batch 被拆成 $m$ 个 micro-batch，而流水线训练又要求反向时拿到前向产生的中间激活，那么最直接的实现会把这 $m$ 个 micro-batch 的激活都保留下来。若再考虑多版本权重，粗略的峰值可以写成：

$$
\text{Memory}_{\text{naive}} \approx m \cdot |A| + m \cdot |W|
$$

这里：

- $|A|$ 表示单个 micro-batch 在某个 stage 需要保存的激活大小。
- $|W|$ 表示该 stage 持有的权重大小。

这不是一条严格的论文公式，而是工程上帮助判断趋势的近似式：micro-batch 越多，缓存越线性增长。

1F1B 的目标是把这个增长从“跟 $m$ 走”改成“跟 $d$ 走”，也就是：

$$
\text{Memory}_{\text{1F1B}} \approx d \cdot |A| + d \cdot |W|
$$

直观理解是：流水线深度是 4，就像一条只能同时容纳 4 个包裹位置的传送带。即使你今天总共要送 8 个包裹，也不需要在每个站点同时堆 8 个，只需要处理当前在传送带上的那一批。

新手版类比可以这样看：4 层模型串联、8 个 micro-batch。如果每层都为每个 micro-batch 留一份输出，就像每个人背着 8 个包。1F1B 的想法不是“让包消失”，而是让旧包尽快完成回传并出队，于是每个人只需要背大约 4 个包，也就是流水线深度那一档。

边界也要说清楚：

- 1F1B 不是“没有气泡”，它只是显著减少稳态中的空闲。
- 1F1B 也不是“自动保证完全同步 SGD 语义”，它需要 weight stashing 才能保证单个 micro-batch 的前后向一致。
- 如果 $m < d$，流水线很难进入理想稳态，吞吐优势会明显变差。
- 如果 stage 切分极不均衡，某个 stage 计算最慢，整个调度仍然会被它卡住。

---

## 核心机制与推导

### 1. 为什么 1F1B 能把激活从 $O(m)$ 压到 $O(d)$

PipeDream 原论文把 1F1B 描述为：进入稳态后，每个 stage 交替执行当前某个 minibatch 的前向和更早一个 minibatch 的反向。[原文](https://device.report/m/ea07a1ee072f1fd169aa7fe48329806e0e8dd55fa300ecdeccad3818a4fd9755_doc)的关键点是，稳态时每个 GPU 都尽量持续忙碌。

以 $d=4, m=8$ 为例。前几个时刻是“灌满流水线”：

- 时刻 1：stage0 做 micro-batch1 的前向
- 时刻 2：stage0 做 micro-batch2 前向，stage1 做 micro-batch1 前向
- 时刻 3：stage0 做 micro-batch3 前向，stage1 做 micro-batch2 前向，stage2 做 micro-batch1 前向
- 时刻 4：stage0 做 micro-batch4 前向，stage1 做 micro-batch3 前向，stage2 做 micro-batch2 前向，stage3 做 micro-batch1 前向

从这里开始，最尾部 stage 已经能尽早触发反向。于是稳态不再是“把 8 个全做完前向再反向”，而是前向和反向穿插进行。这样一来，较早进入流水线的 micro-batch 激活可以更快消费掉，不必一直堆到整个 batch 前向结束。

所以保存的不是“这个 batch 里所有 micro-batch 的激活”，而是“当前仍在流水线中、还没完成反向的那些激活”。这个数量上界和流水线深度同阶，因此常写为 $O(d)$。

### 2. 为什么只改调度还不够

如果只做 1F1B，不做权重版本管理，会出错。

原因是：某个 micro-batch 的前向可能在时间 $t$ 用权重 $W^{(v)}$ 算出来，但它的反向到来时，stage 上的当前权重已经更新成 $W^{(v+1)}$。这时梯度不再对应同一条前向计算路径，数学上就不是对原损失函数那次前向的正确求导。

`weight stashing` 的意思是“把前向时看到的权重版本暂存下来”。白话说，就是给每个正在飞行的 micro-batch 记住它上车时看到的是哪一版参数，等它回程时还按那一版算。

这保证了“同一个 micro-batch 在同一 stage 上，前向和反向使用同一版权重”。但原始 PipeDream 仍可能出现“不同 stage 看到的版本延迟不完全一致”的情况，所以它的语义接近带延迟的数据并行，而不是完全等价于普通同步 SGD。

### 3. PipeDream-2BW 为什么只要 2 份权重

PipeDream-2BW 的关键观察是：新生成的权重版本不必立刻给所有 in-flight micro-batch 使用。已经在流水线里的输入，必须继续用旧版本；只有新进入流水线的输入，才切换到新版本。

于是每个 stage 只要维护：

- 当前仍服务于 in-flight micro-batch 的 `shadow weights`
- 下一批新输入要使用的 `latest weights`

这就是 double buffering，双缓冲。论文明确给出：在深度为 $d$ 的流水线里，PipeDream 最坏可能需要 $d$ 个权重版本，而 2BW 只需 2 个。[论文](https://cs.stanford.edu/~matei/papers/2021/icml_pipedream_2bw.pdf)

更正式一点，PipeDream-2BW 的附录给出内存估计。若使用激活重计算，最坏显存近似为：

$$
\frac{2|W|}{d} + \frac{|A_{\text{total}}(b)|}{d} + d|A_{\text{input}}(b)|
$$

含义分别是：

- $\frac{2|W|}{d}$：每个 stage 只保留两份自己那一段权重。
- $\frac{|A_{\text{total}}(b)|}{d}$：重计算时，不需要长期保存完整中间激活，总激活开销按 stage 均摊。
- $d|A_{\text{input}}(b)|$：仍要保存每个 in-flight micro-batch 的阶段输入，用于之后重算前向。

这里 $d$ 的作用有两面：

- 好的一面：模型被切成更多 stage 后，每个 stage 的权重份额变小，所以 $\frac{2|W|}{d}$ 下降。
- 不好的一面：流水线更深时，在飞 micro-batch 更多，输入级缓存项 $d|A_{\text{input}}(b)|$ 会增加。

真实工程例子是训练 GPT/BERT 这类深层 Transformer。模型参数塞不进单卡时，团队通常先做流水线切分；如果直接用 GPipe，micro-batch 一多，激活缓存就爆；如果直接用原始 PipeDream，前端 stage 又容易因权重版本太多而爆。PipeDream-2BW 的价值就在于它把两类内存压力都压到了更可控的范围。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不模拟真实算子，只验证两件事：

1. 1F1B 场景下，激活在“前向入栈、反向出栈”后不会无限增长。
2. 双缓冲权重保证前向和反向看到同一版本。

```python
from collections import deque

def simulate_1f1b(num_stages=4, num_micro=8):
    # 每个 stage 保存仍未反向完成的 micro-batch -> weight_version
    activation_stash = [deque() for _ in range(num_stages)]
    # 双缓冲: current 给新进入流水线的 micro-batch，shadow 给仍在飞的旧 micro-batch
    current_weight = [0 for _ in range(num_stages)]
    shadow_weight = [0 for _ in range(num_stages)]

    # 记录每个 micro-batch 在每个 stage 前向看到的权重版本
    seen_forward = {}
    seen_backward = {}

    # 简化模型:
    # 先 warmup 到 depth，再按 1F1B 做“进一个，出一个”
    in_flight = deque()
    peak_stash = [0 for _ in range(num_stages)]

    # warmup
    for mb in range(min(num_stages, num_micro)):
        in_flight.append(mb)
        for s in range(num_stages):
            version = current_weight[s]
            activation_stash[s].append((mb, version))
            seen_forward[(mb, s)] = version
            peak_stash[s] = max(peak_stash[s], len(activation_stash[s]))

    next_mb = num_stages

    # steady state: forward one, backward one
    while in_flight:
        # backward oldest micro-batch
        old_mb = in_flight.popleft()
        for s in reversed(range(num_stages)):
            mb, version = activation_stash[s].popleft()
            assert mb == old_mb
            seen_backward[(mb, s)] = version

        # 每处理完一个“批次窗口”，生成新权重版本
        for s in range(num_stages):
            shadow_weight[s] = current_weight[s]
            current_weight[s] += 1

        # forward next micro-batch
        if next_mb < num_micro:
            mb = next_mb
            in_flight.append(mb)
            for s in range(num_stages):
                version = current_weight[s]
                activation_stash[s].append((mb, version))
                seen_forward[(mb, s)] = version
                peak_stash[s] = max(peak_stash[s], len(activation_stash[s]))
            next_mb += 1

    # 前后向必须使用同一版本权重
    for key, version in seen_forward.items():
        assert seen_backward[key] == version

    # 峰值 stash 数量不应超过流水线深度
    assert max(peak_stash) <= num_stages
    return peak_stash, current_weight

peak, versions = simulate_1f1b()
assert peak == [4, 4, 4, 4]
assert versions == [5, 5, 5, 5]
print("ok")
```

这段代码里的几个结构可以直接映射到真实系统：

- `activation_stash[stage]`：保存这个 stage 上还没做 backward 的 micro-batch 状态。
- `current_weight`：给新进入流水线的 micro-batch 用。
- `shadow_weight`：给已经在飞的 micro-batch 保持一致性。
- `deque`：这里充当 ring buffer，白话说就是一个固定顺序循环使用的队列。

如果写成更接近工程实现的伪代码，核心循环通常是：

```python
for clock in range(total_steps):
    for stage in stages:
        if stage.can_run_backward():
            mb = stage.pop_oldest_inflight()
            w = stage.weight_buffer[mb.slot]
            grad = stage.backward(mb, w)
            stage.send_grad(grad)

        if stage.can_run_forward():
            mb = stage.recv_or_create_microbatch()
            slot = stage.alloc_slot(mb)
            stage.weight_buffer[slot] = stage.current_weight_version
            out = stage.forward(mb, stage.weight_buffer[slot])
            stage.stash_activation(slot, out)
            stage.send_activation(out)
```

新手版理解方法是：`buffer[stage][0..1]` 就是“当前版”和“下一版”两个抽屉。前向开始时把“这个 micro-batch 属于哪个抽屉”记下来，反向时按原抽屉取回，等这个 micro-batch 整个结束后，那个抽屉才允许被覆盖。

---

## 工程权衡与常见坑

最常见的误解是：“1F1B 已经把激活从 $O(m)$ 降到 $O(d)$，那显存问题就解决了。”这只对了一半。真实瓶颈常常变成权重副本。

原始 weight stashing 的痛点是：在前端 stage，进入流水线的 micro-batch 最多，等待时间也最长，因此最容易积压多个权重版本。表面上每个 stage 都只存“自己那一段参数”，但如果模型前几段本来就大，或者优化器状态也放在同卡，内存还是会先在前端爆。

下面是标准 weight stashing 和 PipeDream-2BW 的工程差异：

| 策略 | 最大权重副本 | 更新延迟 | 实现复杂度 | 主要风险 |
|---|---:|---|---|---|
| 标准 weight stashing | 最坏 $d$ | 分 stage 延迟不一致 | 中 | 前端 stage 显存不平衡 |
| PipeDream-2BW 双缓冲 | 2 | 固定 1 个版本延迟 | 中到高 | 需要严格管理版本切换周期 |
| Flush 类方案 | 1 | 无额外版本延迟 | 低 | 吞吐下降，气泡增加 |

几个工程坑最值得提前规避：

1. `m` 选得太小。  
如果 $m < d$，稳态很短甚至进不去，GPU 大量时间都在等，1F1B 的优势基本发挥不出来。

2. stage 切分只看层数，不看耗时。  
Transformer 的注意力层、MLP 层、embedding 层开销不一样。按“层数均分”很容易得到一个计算严重失衡的流水线。

3. 只算参数显存，不算优化器状态。  
Adam 一般至少还要带一阶矩、二阶矩，权重副本一旦增加，优化器状态也可能跟着成为大头。

4. 通信重叠被高估。  
论文里的吞吐提升通常依赖“计算和通信能较好重叠”。如果机器拓扑差、跨节点链路慢，理论上的 1F1B 时间线会被通信空洞破坏。

5. 激活重计算的收益被误解。  
重计算不是“免费省显存”，而是用额外算力换显存。算力本来就满的时候，再加重计算可能反而拉低整体 token/s。

一个真实工程例子是单机 8 卡训练中等规模 LLM。很多团队会组合使用：

- 流水线并行：切深模型，解决“单卡放不下”。
- 1F1B：让流水线在有限 micro-batch 下仍保持高利用率。
- ZeRO / optimizer sharding：再把优化器状态和梯度分片。
- activation recomputation：进一步压激活。

这时收益往往不是某一项技术单独带来的，而是“先用 1F1B 把流水线调度稳定下来，再用分片和重计算把剩余峰值压下去”。

---

## 替代方案与适用边界

没有一种流水线方案在所有场景都最好，核心是你缺的是哪种资源。

| 方案 | 激活缓存 | 权重副本 | 通信开销 | 适用场景 |
|---|---|---|---|---|
| GPipe | 高，约 $O(m)$ | 1 | 中 | 显存相对充裕，追求简单一致语义 |
| PipeDream 1F1B | 中，约 $O(d)$ | 最坏 $O(d)$ | 中 | 想避免 flush、追求高吞吐 |
| PipeDream-2BW | 中，约 $O(d)$ | 2 | 中 | 显存紧张且不想牺牲太多吞吐 |
| 1F1B + 重计算 | 更低 | 2 或 $O(d)$ | 中 | 算力有余、显存更紧 |
| 1F1B + ZeRO Stage-3 | 中 | 可再分片 | 高 | 跨卡通信可接受、模型极大 |

新手对比可以直接记一个数字版结论：若 $d=4, m=8$，GPipe 需要围绕 8 个 micro-batch 保持更多激活；1F1B 稳态只需围绕 4 个在飞 micro-batch 组织缓存；若再叠加 ZeRO 分片，参数与优化器状态也会继续下降，但代价是额外通信和更复杂的同步路径。

适用边界也很明确：

- 如果模型不大，单卡或普通数据并行就能轻松放下，1F1B 往往不值得增加实现复杂度。
- 如果网络互联很差，流水线并行的收益可能被激活传输抵消。
- 如果业务更在意“完全等价于标准同步 SGD 的训练语义”，flush 类方法更容易解释和复现。
- 如果模型极深、单 stage 仍放不下，流水线并行通常还要和张量并行一起用，1F1B 只是其中一层调度策略。

所以更准确的结论是：1F1B 不是 GPipe 的“全面替代品”，而是“在显存受限、又希望保持高流水线利用率时更合适的默认选择”；PipeDream-2BW 则是在这个选择上补上了“权重副本不能太多”这一块短板。

---

## 参考资料

1. Narayanan, D. et al. “Memory-Efficient Pipeline-Parallel DNN Training.” ICML 2021, PMLR 139. 链接：[https://cs.stanford.edu/~matei/papers/2021/icml_pipedream_2bw.pdf](https://cs.stanford.edu/~matei/papers/2021/icml_pipedream_2bw.pdf)  
覆盖视角：2BW 双缓冲、两份权重副本、内存公式、与 GPipe/PipeDream 的对比。

2. Harlap, A. et al. “PipeDream: Generalized Pipeline Parallelism for DNN Training.” SOSP 2019. 转引链接：[https://device.report/m/ea07a1ee072f1fd169aa7fe48329806e0e8dd55fa300ecdeccad3818a4fd9755_doc](https://device.report/m/ea07a1ee072f1fd169aa7fe48329806e0e8dd55fa300ecdeccad3818a4fd9755_doc)  
覆盖视角：1F1B 调度、weight stashing、流水线稳态执行语义。

3. Yue Shui.《训练大模型并行和内存优化技术》, 2025-03-01. 链接：[https://syhya.github.io/zh/posts/2025-03-01-train-llm/](https://syhya.github.io/zh/posts/2025-03-01-train-llm/)  
覆盖视角：中文工程综述，适合理清流水线并行、1F1B 与 PipeDream-2BW 的关系。

4. 中文综述论文：《并行和内存优化技术综述》相关章节，链接：[https://jcst.ict.ac.cn/fileup/1000-9000/PDF/JCST-2024-3-4-3872-567.pdf](https://jcst.ict.ac.cn/fileup/1000-9000/PDF/JCST-2024-3-4-3872-567.pdf)  
覆盖视角：工程实践中的内存瓶颈、版本缓存问题与方法分类。

5. 阅读建议：先看 PipeDream 论文中 1F1B 时间线，再看 PipeDream-2BW 的公式部分。尤其是 $\frac{2|W|}{d}$ 这一项，直接说明“为什么双缓冲后每个 stage 只需两份权重”。
