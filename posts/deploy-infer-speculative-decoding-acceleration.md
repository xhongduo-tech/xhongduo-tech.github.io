## 核心结论

投机采样（Speculative Sampling，也常被叫作 Speculative Decoding）是一种**在不改变目标大模型输出分布的前提下，减少大模型串行推理轮数**的方法。白话解释：先让一个更快的小模型提前“打草稿”，再让大模型一次性检查这段草稿，能直接用的就整段收下，不能用的再按大模型自己的概率补采。

它成立的关键，不是“小模型猜得准所以大模型省事”，而是：**接受与拒绝规则经过专门设计，最终采样结果严格等价于原始目标模型的自回归采样**。这里的“自回归”是指模型每次只生成下一个 token，然后把这个 token 接回上下文，继续生成下一步。

因此，投机采样解决的是一个很具体的问题：

1. 大模型单次 forward 很贵。
2. 生成长文本时，要重复很多次“只为了拿下一个 token”的 forward。
3. 如果能用便宜的小模型先提议多个 token，再用一次大模型 forward 验证多个位置，就能把“多轮昂贵调用”压缩成“少轮昂贵调用 + 少量便宜调用”。

一个新手版直觉例子：

- 目标模型一次 forward 要 `50ms`
- 草稿模型一次只要 `5ms`
- 草稿模型先连续提议 5 个 token
- 目标模型一次 forward 同时给出这 5 个位置上的真实概率
- 如果前 4 个都被接受，只在第 5 个位置拒绝，那么这一轮只跑了 1 次目标模型，却产出了 4 个 token

这和纯自回归的差别，可以先看一个最小对比：

| 方案 | 生成 4 个 token 需要的大模型 forward 次数 | 小模型额外成本 | 最终分布是否等价于目标模型 |
|---|---:|---:|---|
| 纯 autoregressive | 4 | 0 | 是 |
| speculative sampling | 通常接近 1 到 2 | 有 | 是 |

如果记草稿总成本相对目标模型的比例为 $C$，记草稿分布和目标分布的重叠率为 $a$，常见近似写法是：

$$
\text{Speedup} \approx \frac{1}{C + (1-a)}
$$

这条式子的含义很直接：

- $C$ 越小，草稿越便宜，越有利。
- $a$ 越大，草稿越像目标模型，接受率越高，越有利。
- 当 $a \ge 0.7$ 且 $C \ll 1$ 时，2x 到 4x 的加速是现实可见的。

---

## 问题定义与边界

先把问题说清楚：投机采样的目标不是“生成近似文本”，而是**生成分布仍然等于目标模型 $P$，同时减少目标模型的顺序执行次数**。

这里有两个模型：

- 目标模型 $P(w \mid h)$：真正想保留其输出行为的大模型。$w$ 是下一个 token，$h$ 是当前上下文。
- 草稿模型 $Q(w \mid h)$：更小、更快的模型，用来先提出候选 token。

“分布”这个词第一次出现时可以理解成：**模型认为每个候选 token 应该出现的概率表**。

如果只是让小模型直接替代大模型，那叫近似推理，不叫投机采样。投机采样要求更强：**即使中间用了小模型提议，最终抽出来的 token 仍然要像是直接从大模型 $P$ 中一步一步采样出来的**。

这决定了它的适用边界。

| 变量 | 作用 | 典型限制 |
|---|---|---|
| 草稿速度 | 决定额外成本 $C$ | 草稿如果不够快，收益被吃掉 |
| 分布对齐程度 | 决定接受率与重采概率 | 模型家族差太远会导致接受率低 |
| block 长度 | 决定一轮最多跳过多少步 | 过长会让后段 token 更容易被拒绝 |
| 采样温度 | 决定输出发散程度 | 温度高时草稿和目标更难对齐 |
| 任务类型 | 决定可预测性 | 创意写作通常比事实问答更难加速 |

一个真实工程语境下的例子可以这样理解：

- 目标模型：GPT-4.1 级别的大模型
- 草稿模型：同家族蒸馏出来的更小模型
- 场景：低温问答、代码补全、结构化抽取

流程是：

1. 小模型先快速给出一段草稿。
2. 大模型一次性评估这段草稿中每个位置的概率。
3. 被接受的位置直接保留。
4. 第一个被拒绝的位置，按目标模型的真实分布重新采样。
5. 继续下一轮。

只要拒绝规则正确，最后得到的样本仍然等价于目标模型原始输出。

但它并不是“任何场景都加速”。下面这些情况要格外小心：

- 草稿模型和目标模型不是同家族，分布错位明显。
- 温度太高，候选 token 发散，导致重叠率低。
- prompt 很开放，比如诗歌、小说、营销文案，后续 token 本身就高度不确定。
- 草稿模型虽然小，但部署链路不够轻，实际延迟并不低。
- block 太长，后面 token 累积偏差变大，接受率下降。

所以，投机采样的本质边界不是“能不能做”，而是：**在保持分布正确的前提下，接受率是否足够高，草稿是否足够便宜**。

---

## 核心机制与推导

### 1. 为什么能一次验证多个 token

普通自回归生成的瓶颈在于：每多生成一个 token，就要再跑一次目标模型。即使 KV Cache 已经缓存了历史状态，仍然要进行一次新的解码步。

投机采样的关键观察是：

- 草稿模型可以先基于当前上下文 $h$ 连续采样出一段候选：$x_1, x_2, \dots, x_k$
- 然后把这段候选当作“未来轨迹”送给目标模型
- 目标模型在一次 forward 中，同时给出这些位置上的条件概率：
  $$
  P(x_1 \mid h),\ P(x_2 \mid h,x_1),\ \dots,\ P(x_k \mid h,x_1,\dots,x_{k-1})
  $$

于是，一次目标模型调用就不再只服务于“下一个 token”，而是服务于“一整段候选路径的验证”。

### 2. 接受规则为什么不会改分布

假设在某一步的上下文是 $h$，草稿模型采出了 token $w$，它在草稿模型下的概率是 $Q(w \mid h)$，在目标模型下的概率是 $P(w \mid h)$。

接受概率定义为：

$$
\alpha(w,h) = \min\left(1, \frac{P(w \mid h)}{Q(w \mid h)}\right)
$$

白话解释：

- 如果草稿模型低估了这个 token，也就是 $P > Q$，那它一定被接受。
- 如果草稿模型高估了这个 token，也就是 $P < Q$，那它只能以 $P/Q$ 的概率被接受。
- 一旦拒绝，就不能继续相信这条草稿路径，必须从目标分布修正采样。

为什么这样设计是对的？因为它在做一件类似接受-拒绝采样的事。接受部分已经按 $P$ 与 $Q$ 的重叠区域保留，剩下不重叠的概率质量，再通过目标模型补回来，整体就恢复成目标分布。

更直观一点，定义重叠率：

$$
a(h) = \sum_w \min\{Q(w \mid h), P(w \mid h)\}
$$

它表示在上下文 $h$ 下，两份概率分布“共同覆盖”的那一部分质量。这个量越大，说明草稿越像目标，草稿 token 越容易被接受。

### 3. 玩具例子

假设某一步只有 3 个候选 token：`A`、`B`、`C`。

| token | 目标分布 $P$ | 草稿分布 $Q$ | $\min(P,Q)$ |
|---|---:|---:|---:|
| A | 0.6 | 0.5 | 0.5 |
| B | 0.3 | 0.4 | 0.3 |
| C | 0.1 | 0.1 | 0.1 |

那么重叠率：

$$
a = 0.5 + 0.3 + 0.1 = 0.9
$$

这表示 90% 的概率质量是对齐的。此时平均来说，大部分草稿 token 都能被直接接受。

进一步看接受率：

- 对 `A`：$\min(1, 0.6/0.5)=1$，必接受
- 对 `B`：$\min(1, 0.3/0.4)=0.75$
- 对 `C`：$\min(1, 0.1/0.1)=1$

所以哪怕 `B` 会有一部分被拒绝，整体修正后仍然回到目标分布。

### 4. 加速比从哪里来

现在看时间成本。

设：

- $T_{\text{target}}$：目标模型一次验证 block 的时间
- $T_{\text{draft}}$：草稿阶段总时间
- $C = T_{\text{draft}} / T_{\text{target}}$

如果一轮平均能接受的 token 数与重叠率相关，那么“每产生一个有效 token 的目标模型摊销成本”会下降。常见近似可写成：

$$
\text{Speedup} \approx \frac{1}{C + (1-a)}
$$

这不是一个严格适用于所有实现细节的精确公式，但它抓住了核心关系：

- 草稿成本占比是 $C$
- 因拒绝与重采造成的额外损失，近似与 $(1-a)$ 成正比

数值例子：

- $a = 0.8$
- $C = 0.1$

则

$$
\text{Speedup} \approx \frac{1}{0.1+0.2} = 3.33
$$

这意味着同样的硬件预算下，理论上吞吐可以提高到原来的 3.3 倍左右。

再给一个更工程化的估算：

- 目标模型：`50ms / forward`
- 草稿模型：`5ms / step`
- block 长度：5
- 平均接受 3 个草稿 token，随后补 1 个真实 token
- 一轮总成本：$50 + 5 \times 5 = 75 \text{ms}$

如果这一轮平均产出 4 个 token，那么单 token 平均成本是：

$$
75 / 4 = 18.75\text{ms/token}
$$

相比纯目标模型 `50ms/token`，提速约：

$$
50 / 18.75 \approx 2.67
$$

### 5. 一个真实工程例子

在 TensorRT-LLM 或类似 GPU 推理框架中，目标模型通常已经做了 KV Cache、张量并行、融合 kernel 等优化。此时单 token 解码仍然受限于串行依赖。投机采样的价值在于：**不需要改变输出分布，却能减少“必须由大模型亲自完成”的解码轮数**。

例如在低温问答场景中：

- 用户问的是结构化问题，如“把这段日志解析成 JSON”
- 目标模型的输出模式稳定
- 草稿模型与目标模型来自同一架构家族
- 草稿模型很便宜，且能给出高重叠率候选

这时，系统可能看到：

- 平均接受率 75% 到 90%
- 平均每次目标模型 forward 验证 4 到 8 个位置
- 端到端 latency 降低
- GPU 吞吐提升 2x 左右甚至更高

---

## 代码实现

下面先给一个可以运行的玩具实现。它不是完整的 LLM 系统，而是用离散分布模拟“草稿提议 + 目标验证 + 拒绝后补采”的核心逻辑。

```python
import random

VOCAB = ["A", "B", "C"]

def sample_from_dist(dist):
    r = random.random()
    acc = 0.0
    for token, p in dist.items():
        acc += p
        if r <= acc:
            return token
    return token

def speculative_accept(token, p_dist, q_dist):
    p = p_dist[token]
    q = q_dist[token]
    ratio = 1.0 if q == 0 else min(1.0, p / q)
    return random.random() < ratio

def toy_target_dist(context):
    # 用上下文长度制造一点变化，模拟真实条件分布
    if len(context) % 2 == 0:
        return {"A": 0.6, "B": 0.3, "C": 0.1}
    return {"A": 0.2, "B": 0.5, "C": 0.3}

def toy_draft_dist(context):
    # 草稿分布与目标相近，但不完全一致
    if len(context) % 2 == 0:
        return {"A": 0.5, "B": 0.4, "C": 0.1}
    return {"A": 0.25, "B": 0.45, "C": 0.30}

def speculative_step(context, block_len=4):
    draft_tokens = []
    q_probs = []
    cur = list(context)

    # 1. 草稿模型先提议一个 block
    for _ in range(block_len):
        q_dist = toy_draft_dist(cur)
        tok = sample_from_dist(q_dist)
        draft_tokens.append(tok)
        q_probs.append(q_dist[tok])
        cur.append(tok)

    # 2. 目标模型逐位置验证
    accepted = []
    cur = list(context)
    for tok in draft_tokens:
        p_dist = toy_target_dist(cur)
        q_dist = toy_draft_dist(cur)

        if speculative_accept(tok, p_dist, q_dist):
            accepted.append(tok)
            cur.append(tok)
        else:
            repaired = sample_from_dist(p_dist)  # 从目标分布补采
            accepted.append(repaired)
            cur.append(repaired)
            break

    return accepted

# 基本正确性检查
random.seed(0)
out = speculative_step(["<bos>"], block_len=5)
assert 1 <= len(out) <= 5
assert all(t in VOCAB for t in out)

# 接受概率边界检查
p = {"A": 0.6, "B": 0.3, "C": 0.1}
q = {"A": 0.5, "B": 0.4, "C": 0.1}
assert min(1.0, p["A"] / q["A"]) == 1.0
assert abs(min(1.0, p["B"] / q["B"]) - 0.75) < 1e-9
```

上面这段代码对应的组件职责，可以压成一个工程表：

| 组件 | 职责 |
|---|---|
| `draft_model.sample` | 基于当前上下文连续生成 block 候选 |
| `target_model.forward` | 一次性给出 block 各位置上的真实条件概率 |
| `accept/reject` | 按 $P/Q$ 规则决定草稿 token 是否保留 |
| `repair_sample` | 第一个拒绝位置从目标分布补采 |
| `scheduler` | 控制 block 长度、终止条件、状态推进 |

如果把它写成更接近真实系统的伪代码，主循环大致如下：

```python
def decode(context, max_new_tokens, block_len, draft_model, target_model):
    output = []
    while len(output) < max_new_tokens:
        # 1. 草稿阶段
        draft_tokens, q_probs = draft_model.sample_block(context, block_len)

        # 2. 目标模型一次 forward 验证整段
        p_probs = target_model.score_block(context, draft_tokens)

        accepted_any = False
        for token, q, p in zip(draft_tokens, q_probs, p_probs):
            accept_ratio = min(1.0, p / q) if q > 0 else 1.0

            if random.random() < accept_ratio:
                output.append(token)
                context = context + [token]
                accepted_any = True
            else:
                repaired = target_model.sample_next(context)
                output.append(repaired)
                context = context + [repaired]
                accepted_any = True
                break

        if not accepted_any:
            repaired = target_model.sample_next(context)
            output.append(repaired)
            context = context + [repaired]

    return output
```

实现时真正容易出错的，不是这段逻辑本身，而是下面三件事：

1. **上下文推进必须严格一致**  
   每接受一个 token，后续位置的条件分布就变了，所以验证时的位置对齐不能错。

2. **拒绝后要立即截断 block**  
   第一个拒绝位置之后，原草稿后续 token 的条件前提已经失效，不能继续复用。

3. **目标模型要能高效返回整段位置的 logits**  
   否则你只是把多步采样写成了更复杂的控制流，并没有真正降低目标模型成本。

一个真实工程例子是代码补全服务。用户输入函数签名后，模型继续补实现。由于这类文本局部确定性很强，比如常见的缩进、括号、库调用模式、类型标注格式都很稳定，所以同家族小模型给出的草稿常常和大模型高度重合。这类场景比开放式聊天更适合投机采样。

---

## 工程权衡与常见坑

投机采样在论文里看起来很干净，但工程上最常见的问题是：**理论成立，不等于系统一定变快**。

先看典型坑位：

| 坑 | 触发条件 | 结果 | 规避措施 |
|---|---|---|---|
| 草稿和目标分布错位 | 不同模型家族、不同 tokenizer、蒸馏不足 | 接受率低，频繁重采 | 优先选同家族草稿模型 |
| 草稿温度过高 | 创意生成、开放式对话 | 候选发散，$a$ 下降 | 降低温度或改成贪心草稿 |
| block 过长 | 后续位置误差累积 | 前几位通过，后几位频繁拒绝 | 从短 block 开始调参 |
| 草稿不够快 | $C$ 接近 0.5 甚至更高 | 几乎无加速 | 用更小模型或更轻推理链路 |
| 验证实现低效 | target 不能高效批量算 block | 理论省 forward，实际没省时延 | 优化 kernel、KV 布局、批处理 |
| 只看平均接受率 | 某些 prompt 很好，某些很差 | 延迟抖动大 | 监控分位数，不只看均值 |

一个新手容易忽视的点是：**接受率不是唯一指标**。例如两个方案都 80% 接受率，但如果其中一个草稿模型成本是目标模型的 30%，另一个只有 5%，后者收益会明显更高。因为真正进入公式的是 $C + (1-a)$，而不是只看 $a$。

再看一个失败例子。假设：

- 目标模型：GPT-4.1 级别
- 草稿模型：同家族小模型
- 任务：写诗
- 温度：1.2
- 实测接受率：35%

这时系统会发生什么？

1. 草稿模型先生成一段很发散的候选。
2. 大模型验证时很快在前几个位置就拒绝。
3. 每轮都要补采。
4. 草稿阶段还额外花掉了成本。
5. 结果变成“多跑了一个小模型，还没有减少多少大模型轮数”。

这就是为什么很多团队上线时会加一个非常朴素但有效的控制策略：

1. 监控接受率。
2. 如果接受率持续偏低，先缩短 block。
3. 还不够就降低草稿温度。
4. 仍然不行，就对该类请求退回纯目标模型解码。

可以把这条调优逻辑理解为：

- 先保住分布正确性
- 再保住系统收益
- 不要为了“用了投机采样”而强行一直开着

还有一个容易踩的坑是 tokenizer 不一致。tokenizer 可以白话理解成“把文本切成 token 的规则”。如果草稿模型和目标模型的切分方式不同，那么“同一个文本前缀下的下一 token”都不在一个坐标系里，验证与接受逻辑会非常麻烦，甚至直接不成立。现实工程里通常要求两者共享 tokenizer，或者采用专门的映射机制。

---

## 替代方案与适用边界

投机采样不是唯一的推理优化手段。它的独特点在于：**针对单条序列，把原本必须串行发生的多次目标模型 forward，压缩成更少次验证 forward**。

和其他方案对比更容易看清边界：

| 方案 | 主要优化点 | 是否减少单条序列的大模型串行步数 | 优势 | 限制 |
|---|---|---|---|---|
| Speculative Sampling | 草稿提议 + 目标验证 | 是 | 单请求低延迟收益明显 | 依赖高接受率与低草稿成本 |
| Prefix Caching | 复用相同前缀的 KV Cache | 否 | 多请求共享前缀时很有效 | 对单条新生成帮助有限 |
| Batch Decoding | 多请求一起跑 GPU | 否 | 提高整体吞吐 | 单请求尾延迟未必下降 |
| Beam Search | 同时保留多个候选路径 | 否，通常更贵 | 提升搜索质量 | 不是加速手段 |
| Medusa/多头草稿 | 同模型内部多分支预测 | 部分是 | 减少外部草稿模型依赖 | 需要训练或改模型结构 |

适用边界可以直接记成三句话：

1. **接受率高的场景适合。**  
   比如低温问答、代码补全、结构化抽取、格式稳定的 agent 输出。

2. **草稿模型极便宜的场景适合。**  
   如果小模型并不“小”，那就只是把系统变复杂。

3. **输出高确定性的场景适合。**  
   比如“把这段 SQL 改成参数化查询”通常比“写一首现代诗”更适合。

给一个对比例子：

- 场景 A：客服问答  
  用户问“退款流程是什么”。答案模板化、术语稳定、分布重叠高，投机采样通常有效。

- 场景 B：诗歌生成  
  用户要求“模仿某位作家风格写一首朦胧诗”。下一个 token 的不确定性很高，草稿和目标更难对齐，此时接受率可能很差，更适合直接用目标模型，或者先把草稿模型做更强蒸馏与任务对齐。

所以，判断要不要上投机采样，工程上通常先看这三个量：

- 接受率是否稳定高于 60% 到 70%
- 草稿成本占比 $C$ 是否足够低
- 任务输出是否具有强结构或强局部确定性

如果接受率长期不到 30% 到 40%，一般就不该硬上。此时更合理的路线通常是：

- 继续用目标模型
- 改善 KV Cache / batching / kernel
- 或者训练一个更对齐的草稿模型

---

## 参考资料

| 标题 | 链接 | 重点摘要 |
|---|---|---|
| NVIDIA: An Introduction to Speculative Decoding for Reducing Latency in AI Inference | https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/ | 通俗介绍整体架构，适合初学者先建立直觉 |
| Inference.net: Speculative Decoding | https://inference.net/content/speculative-decoding | 用较清晰的步骤解释草稿、验证、拒绝与补采流程 |
| Emergent Mind: Speculative Sampling Efficiency | https://www.emergentmind.com/topics/speculative-sampling-efficiency | 汇总重叠率、接受率与加速比等理论视角 |
| Adaptive ML: Speculative Decoding Visualized | https://www.adaptive-ml.com/post/speculative-decoding-visualized | 用图示和数值例子帮助理解 block 验证过程 |
| TensorRT-LLM: Speculative Decoding | https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html | 偏工程实现，适合理解 GPU 推理框架中的落地方式 |
