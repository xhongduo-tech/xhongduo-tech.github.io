## 核心结论

GShard 的随机 Top-2 路由不是“选两个最大专家然后都执行”，而是“第一专家必选，第二专家按概率抽样”。这里的路由器（router，负责决定 token 去哪个专家的小网络）先对每个 token 产生一个 gate 向量，取前两名得到 $g_1,g_2$ 与对应专家 $e_1,e_2$。随后执行下面的规则：

$$
g(x)=\mathrm{softmax}(xW_g)
$$

$$
\hat g_1=\frac{g_1}{g_1+g_2},\qquad \hat g_2=\frac{g_2}{g_1+g_2}
$$

$$
P(e_2\mid x)=\min\left(1,\ 2\cdot \hat g_2\right)
=\min\left(1,\ 2\cdot \frac{g_2}{g_1+g_2}\right)
$$

含义很直接：

1. $e_1$ 一定接收这个 token，只要容量没满。
2. $e_2$ 不是必接收，而是按概率接收。
3. $g_2$ 越接近 $g_1$，第二专家越可能被激活；$g_2$ 很小时，就少做一次专家计算。

这套设计的价值在于同时解决两个矛盾：

| 目标 | 纯确定性 Top-2 | GShard 随机 Top-2 |
|---|---|---|
| 保留主路径稳定性 | 可以 | 可以，且 Top-1 更明确 |
| 让次优专家得到训练信号 | 可以，但成本固定 | 可以，且成本随不确定性变化 |
| 控制计算量 | 较差，每次都算两位专家 | 更好，不确定时才更常激活第二专家 |
| 负载均衡 | 依赖辅助 loss | 依赖辅助 loss，同时随机性提供额外探索 |

最小玩具例子：某个 token 的路由权重为 $g_1=0.6,\ g_2=0.2$。归一化后：

$$
\hat g_2=\frac{0.2}{0.6+0.2}=0.25
$$

所以：

$$
P(e_2\mid x)=2\times 0.25=0.5
$$

也就是训练中大约一半 step 会激活第二专家，另一半只走第一专家。这比“第二专家永远不参与”更有探索性，也比“每次都跑第二专家”更省容量和算力。

可以把流程压缩成一句话：

`Top-1 确定分配 -> Top-2 按概率采样 -> 容量检查 -> 超出容量则 overflow`

---

## 问题定义与边界

先定义几个术语。

token：模型处理中最小的离散输入单位，常见为一个词片段。  
专家（expert）：MoE 层里的一小块前馈网络，只处理被分给自己的 token。  
容量（capacity）：每个专家在一个 batch 或 group 中最多能接收多少 token 的上限。  
overflow：某个 token 本来想发给某专家，但该专家容量已满，只能跳过该专家计算。

GShard 讨论的是 MoE 层内部的路由问题，而不是整个 Transformer 的全部训练问题。它的边界主要有三层。

第一层边界：只在少数专家上计算。  
如果一个 MoE 层有 $E$ 个专家，最朴素的方法是每个 token 都跑完全部 $E$ 个专家，再做加权平均。这当然最平滑，但计算成本太高。GShard 的做法是只看 Top-2，也就是每个 token 最多只访问两个专家。

第二层边界：每个专家有容量上限。  
假设一组里有 $S$ 个 token、$E$ 个专家，平均每个专家应接收 $S/E$ 个 token。若容量因子为 $c_f$，则常见容量设定可写成：

$$
C=\left\lceil c_f\cdot \frac{S}{E}\right\rceil
$$

这里容量因子（capacity factor，表示容量相对平均值放大的倍数）控制“允许多不均匀”。  
$c_f=1$ 表示每位专家只能接平均负载；$c_f=2$ 表示允许暂时承受两倍平均负载。

第三层边界：overflow token 通常不再强行塞进专家。  
这类 token 会绕过 MoE 子层，沿残差路径继续往下传。直观上，这等于告诉系统：“这一层专家已经挤满，别再硬算了。” 这也是为什么容量参数太小会直接损失训练信号。

下面用一个边界表说明行为：

| 输入 token 数 $S$ | 专家数 $E$ | 容量因子 $c_f$ | 单专家容量 $C$ | 可能结果 |
|---|---:|---:|---:|---|
| 64 | 8 | 1.0 | 8 | 负载稍不均就可能 overflow |
| 64 | 8 | 1.25 | 10 | 允许轻度倾斜 |
| 64 | 8 | 2.0 | 16 | 更能容忍热门专家短时拥堵 |
| 100 | 4 | 1.0 | 25 | 若 60 个 token 都想去同一专家，大量 overflow |
| 100 | 4 | 2.0 | 50 | 同样场景下仅少量 overflow |

必须强调一个常见误解：随机 Top-2 不是为了“平均随机分配 token”。它仍然以 gate 大小为核心依据，只是在第二专家上加入概率化执行。也就是说，路由主导权仍然来自 softmax 权重，而不是均匀随机。

真实工程边界可以这样理解：如果一个 batch 中 100 个 token 里有 60 个都强烈偏向同一专家，而该专家容量只有 25，那么哪怕路由公式再漂亮，也会出现 35 个以上 token 无法进入该专家。这时真正决定效果的，不只是采样公式，还有容量因子、辅助 loss、分组策略和 batch 统计稳定性。

---

## 核心机制与推导

先看纯数学对象。

给定 token 表示向量 $x$，router 用线性层产生 logits，再做 softmax：

$$
g(x)=\mathrm{softmax}(xW_g)
$$

其中 $g(x)\in \mathbb{R}^E$，每一维表示当前 token 对某个专家的偏好强度。softmax 可以理解成“把一组分数变成总和为 1 的概率权重”。

然后取最大的两个分量：

$$
(g_1,e_1),(g_2,e_2)=\mathrm{Top2}(g(x))
$$

这里的核心不是直接拿原始 $g_1,g_2$ 输出，而是先在二者内部重新归一化：

$$
\hat g_1=\frac{g_1}{g_1+g_2},\qquad \hat g_2=\frac{g_2}{g_1+g_2}
$$

为什么要这样做？因为执行层面只保留两个专家，其他专家都被截断了。若不重新归一化，剩余权重和会小于 1，输出尺度会漂移。归一化后的组合输出可写成：

$$
y=\hat g_1\cdot \mathrm{Expert}_{e_1}(x)+m\cdot \hat g_2\cdot \mathrm{Expert}_{e_2}(x)
$$

其中 $m\in\{0,1\}$ 是第二专家是否被采样到的随机变量：

$$
m\sim \mathrm{Bernoulli}\left(\min(1,2\hat g_2)\right)
$$

这一步是 GShard 的关键。它不是固定让第二专家参与，而是让第二专家的参与概率与 $\hat g_2$ 成正比。这个设计有两个直接后果。

第一，模型不确定时，第二专家更容易被激活。  
如果 $g_1$ 与 $g_2$ 接近，说明 router 对“到底发给谁”没有绝对把握，此时探索第二专家更有价值。

第二，模型非常确定时，第二专家会被抑制。  
如果 $g_1 \gg g_2$，说明主专家已经非常明确，再去执行第二专家通常收益有限，却占用容量和算力。

### 玩具例子

假设一个 token 是单词 `bank`，它可能表示“银行”，也可能表示“河岸”。  
router 输出四位专家分数：

| 专家 | 含义 | gate |
|---|---|---:|
| $e_0$ | 金融语义专家 | 0.60 |
| $e_1$ | 地理语义专家 | 0.20 |
| $e_2$ | 时间表达专家 | 0.12 |
| $e_3$ | 语法修复专家 | 0.08 |

此时 Top-1 是金融专家，Top-2 是地理专家。归一化后：

$$
\hat g_1=\frac{0.6}{0.8}=0.75,\qquad \hat g_2=\frac{0.2}{0.8}=0.25
$$

所以第二专家的采样概率是：

$$
P(e_2\mid x)=2\times 0.25=0.5
$$

这意味着在训练的很多步中，这个歧义 token 仍有机会进入“河岸语义专家”。如果改成纯 Top-1，地理专家可能长期看不到这类样本；如果改成确定性 Top-2，则每次都要多做一次专家计算。随机 Top-2 取的是中间点。

### 为什么这不会完全破坏可训练性

初学者容易担心：“既然第二专家是抽样的，那梯度不是变得不稳定吗？”  
答案是：会增加方差，但不会失去训练价值，因为稳定部分和不稳定部分被拆开了。

稳定部分：Top-1 始终存在，主路径不会消失。  
不稳定部分：Top-2 用抽样引入探索，让边缘专家也能拿到样本。  

此外，GShard 通常还配合辅助损失（auxiliary loss）。直观上，它是在惩罚“少数专家太忙、其他专家太闲”的路由分布。一个常见写法可概念化为：

$$
\ell_{aux}=\frac{1}{E}\sum_{e=1}^{E}\left(\frac{c_e}{S}\right)m_e
$$

其中：

- $c_e$ 表示专家 $e$ 实际接收到的 token 数或负载统计；
- $m_e$ 表示 router 给专家 $e$ 的平均门控权重；
- $S$ 是组内 token 数；
- $E$ 是专家数。

它的目标不是直接提高主任务精度，而是维持专家利用率更均衡。没有这类约束时，少数专家可能因为早期偶然优势快速变成“热门专家”，形成专家塌缩。专家塌缩就是大部分 token 总挤向少数专家，其他专家几乎不学习。

### 随机性如何帮助负载均衡

负载均衡不只靠辅助 loss，也受执行路径影响。设想两个专家权重接近，但若你总是只走 Top-1，那么第二专家永远得不到真实输入，router 更难修正早期偏差。随机 Top-2 给了次优专家“偶尔上场”的机会，这种机会本质上是一种探索信号。

可以把它理解为：

- 确定性 Top-1：开发利用最强，但探索最弱。
- 确定性 Top-2：探索存在，但代价固定。
- 随机 Top-2：按不确定性大小自适应探索。

这就是它比“纯 Top-2 恒定双路执行”更细的地方。

---

## 代码实现

下面给出一个可运行的 Python 版本，只演示路由逻辑，不包含真实神经网络计算。代码目标是把论文中的“Top-1 确定 + Top-2 随机采样 + 容量检查”翻译成工程上容易验证的形式。

```python
import math
import random

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def gshard_top2_route(logits_batch, capacity, seed=0):
    """
    logits_batch: List[List[float]], shape = [num_tokens, num_experts]
    capacity: int, per-expert capacity
    return:
      assignments: each token -> list of (expert_id, normalized_gate)
      counts: per-expert accepted token count
      overflow_tokens: token indices that failed at least one desired route
    """
    random.seed(seed)
    num_tokens = len(logits_batch)
    num_experts = len(logits_batch[0])
    counts = [0] * num_experts
    assignments = [[] for _ in range(num_tokens)]
    overflow_tokens = set()

    for token_id, logits in enumerate(logits_batch):
        probs = softmax(logits)
        top2 = sorted(range(num_experts), key=lambda i: probs[i], reverse=True)[:2]
        e1, e2 = top2[0], top2[1]
        g1, g2 = probs[e1], probs[e2]
        norm = g1 + g2
        ng1, ng2 = g1 / norm, g2 / norm

        # Top-1: deterministic
        if counts[e1] < capacity:
            assignments[token_id].append((e1, ng1))
            counts[e1] += 1
        else:
            overflow_tokens.add(token_id)

        # Top-2: stochastic
        p2 = min(1.0, 2.0 * ng2)
        rnd = random.random()
        if rnd < p2:
            if counts[e2] < capacity:
                assignments[token_id].append((e2, ng2))
                counts[e2] += 1
            else:
                overflow_tokens.add(token_id)

    return assignments, counts, sorted(overflow_tokens)

# 玩具例子：希望出现 g1=0.6, g2=0.2 的效果
g1, g2 = 0.6, 0.2
p2 = min(1.0, 2.0 * g2 / (g1 + g2))
assert abs(p2 - 0.5) < 1e-9

# 小批量路由测试
batch = [
    [3.0, 1.9, 0.1, -0.3],  # expert 0 strongest, expert 1 second
    [2.8, 2.7, 0.2, -1.0],  # expert 0 and 1 close
    [0.1, 2.9, 2.8, 0.0],   # expert 1 strongest, expert 2 second
    [3.2, 0.5, 0.4, 0.3],   # expert 0 dominant
]

assignments, counts, overflow = gshard_top2_route(batch, capacity=2, seed=42)

# 基本正确性检查
assert len(assignments) == 4
assert all(c <= 2 for c in counts)
assert isinstance(overflow, list)
assert sum(len(x) for x in assignments) >= 4  # 至少有 Top-1 成功
```

这段代码对应两段核心逻辑：

1. 先保证 Top-1。只要 $e_1$ 还有容量，就一定接收 token。
2. 再处理 Top-2。只有当随机数满足概率条件，且 $e_2$ 还有容量时，才接收 token。

如果把它写成流程图，其实就是：

| 步骤 | 动作 |
|---|---|
| 1 | 对 token 计算 router logits 和 softmax |
| 2 | 取 $e_1,e_2$ 与 $g_1,g_2$ |
| 3 | 归一化得到 $\hat g_1,\hat g_2$ |
| 4 | 若 $e_1$ 未满，则分配给 Top-1 |
| 5 | 采样 $u\sim U(0,1)$ |
| 6 | 若 $u<\min(1,2\hat g_2)$ 且 $e_2$ 未满，则分配给 Top-2 |
| 7 | 超出容量的部分记为 overflow |

真实框架里通常不会用 Python `for` 循环逐 token 写，而会做向量化实现。原因很简单：真实训练时 token 数成千上万，逐个循环会严重拖慢 GPU 或 TPU 吞吐。

### 真实工程例子

以多语种机器翻译为例，很多语言对共享一套编码器与解码器，但不同语言的形态变化、词序和词表分布差异很大。MoE 层的目标就是让不同专家分担不同语言模式。

假设某一层有 64 个专家，batch 里同时混入英语、德语、阿拉伯语、芬兰语句子。若用纯 Top-1，某些高资源语言可能快速占据多数专家；若用确定性 Top-2，每个 token 又固定多做一次专家计算，通信与容量压力都会上升。随机 Top-2 的效果是：

- 明确偏向某专家的 token，大多只跑 Top-1。
- 边界模糊的 token，更容易顺带激活第二专家。
- 长尾语言可以借由随机探索更早接触到合适专家。
- 配合 capacity factor=2 和 aux loss，overflow 能控制在较低水平。

这类场景里，随机性不是“增加噪声而已”，而是帮助不同语言样本在训练初期更快找到可用专家分区。

---

## 工程权衡与常见坑

随机 Top-2 的优点很明确，但它不是“打开就赢”。真正落地时，问题往往出在容量、统计监控和训练稳定性上。

先给出一个问题表：

| 坑 | 原因 | 监控指标 | 对策 |
|---|---|---|---|
| overflow 过高 | capacity factor 太低，热门专家装不下 | `overflow_rate` | 提高容量因子，优化 batch 混排 |
| 专家塌缩 | 少数专家过早占优 | `expert_routed_fraction`、gate entropy | 增强 aux loss，加入噪声或 z-loss |
| 第二专家几乎不触发 | $g_2$ 长期过小 | `second_expert_accept_rate` | 检查 router 初始化与负载均衡损失 |
| 计算变慢 | 向量化差或跨设备通信重 | step time、all-to-all 时间 | 合并 dispatch，做 expert parallel 优化 |
| 训练波动大 | 抽样方差与容量丢弃同时存在 | loss variance | 调大学习稳定项，减少早期过强稀疏性 |

### 1. capacity factor 太低时，随机也救不了拥堵

假设 4 个专家、100 个 token、capacity factor=1，则每位专家容量只有 25。  
如果 60% token 都偏向同一专家，那么光 Top-1 路径就已经至少有 35 个 token 无法进入。再加上部分 token 还可能想去相同的 Top-2 专家，拥堵会更严重。

这时经常发生两个现象：

1. overflow token 直接走残差，MoE 对它们等于没工作。
2. 第二专家虽然有概率采样，但很多采样命中后仍因容量满而失败。

所以随机 Top-2 的收益建立在“容量至少不是过低”之上。工程里常见做法是从 $c_f=1.25$ 或 $2.0$ 起调，而不是一开始就压到 1。

### 2. 第二专家的随机性不是越大越好

公式里的系数 2 并不是“越大越探索越好”的随意参数。它代表一种受控放大：让第二专家在 $\hat g_2$ 不太小时较容易被触发，但又不把低权重候选普遍变成高频路径。

如果你把它改成更大的常数，短期看似让更多专家得到训练，实际上会带来三类副作用：

- 容量更快打满；
- all-to-all 通信更重；
- 路由器更难收敛到稳定分工。

所以这类概率放大是一个平衡点，不是纯探索奖励。

### 3. 监控“被选中”不够，还要监控“被成功接收”

新手常只看 gate 分布，例如“专家 7 的平均 gate 不低，所以它没有问题”。这不够。真正要看的是两层统计：

1. router 想把多少 token 送到某专家；
2. 该专家实际成功接收了多少 token。

如果第二个数显著低于第一个数，说明系统不是“不会选”，而是“选了也塞不进去”。这类问题必须从容量与并行调度上处理，而不是只调 loss。

### 4. deterministic Top-2 与随机 Top-2 的实际差异

两者都能让第二专家得到训练信号，但方式不同。

- deterministic Top-2：所有 token 的第二专家都执行，梯度覆盖稳定，但成本固定更高。
- 随机 Top-2：只有“不确定性较高”的 token 更常激活第二专家，信号更稀疏，但资源利用更灵活。

如果你的系统已经被通信开销卡住，随机 Top-2 常比确定性 Top-2 更现实。

---

## 替代方案与适用边界

随机 Top-2 不是唯一选择。是否采用它，取决于你的目标是“更稳的训练”、“更低的延迟”还是“更强的负载均衡”。

下面给出对比：

| 策略 | 激活专家数 | 负载均衡机制 | 计算开销 | 适用场景 |
|---|---:|---|---|---|
| 随机 Top-2 | 1 到 2 | aux loss + 第二专家概率采样 | 中等 | 训练期希望兼顾探索与成本 |
| 确定性 Top-2 | 固定 2 | aux loss | 较高 | 资源足够，追求更稳定双路训练 |
| Switch Top-1 | 固定 1 | 强依赖 aux loss | 低 | 推理延迟敏感、系统简单优先 |
| Expert Choice | token 由专家反向挑选 | 从机制上约束容量 | 中等到高 | 对负载均衡极敏感的多任务场景 |

### 随机 Top-2 与 Switch Top-1

Switch Top-1 可以理解为“只保留第一专家”。优点是简单、快、通信少。缺点也很明显：没有第二条探索路径，router 若早期偏了，次优专家更难拿到样本。  
所以当目标是在线推理、低时延服务或硬件受限部署时，Top-1 很有吸引力；但若目标是大规模预训练、特别是多任务和多语言训练，随机 Top-2 通常更稳妥。

### 随机 Top-2 与 Expert Choice

Expert Choice 的思路反过来，不是 token 选专家，而是专家在候选 token 中主动挑自己要处理的样本。它对容量约束更直接，能在一些负载极不均的场景里做得更整齐。  
但它的系统实现和调度逻辑也更复杂，不一定适合所有训练栈。

### 一个简单判断规则

如果你的主要瓶颈是：

- 推理延迟：优先考虑 Top-1。
- 训练初期专家塌缩：优先考虑随机 Top-2。
- 负载均衡极差、通信已经被打爆：评估 Expert Choice。
- 算力充足且追求实现直接：可考虑确定性 Top-2。

换句话说，随机 Top-2 最适合的边界是：你希望比 Top-1 更有探索性，但又不想承受确定性 Top-2 的固定双专家成本。

---

## 参考资料

- Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*  
  https://openreview.net/pdf/cdb90e31da8446076492c5aef0c8c6ae35dd472a.pdf

- Bruno Magalhaes, *Distributed Mixture-of-Experts and Expert Parallelism*  
  https://brunomaga.github.io/Mixture-of-Experts

- Facebook AI WMT2021 多语种翻译论文  
  https://www.statmt.org/wmt21/pdf/2021.wmt-1.19.pdf

- IntelliTechTribe, *Mixture of Experts: A Comprehensive Guide*  
  https://intellitechtribe.com/mixture-of-experts/

- *Mixture-of-Experts with Expert Choice Routing*  
  https://www.researchgate.net/publication/358764387_Mixture-of-Experts_with_Expert_Choice_Routing
