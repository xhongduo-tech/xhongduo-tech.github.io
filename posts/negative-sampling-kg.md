## 核心结论

知识图谱负采样的本质，是在没有天然负例的前提下，为每个真实三元组构造“应该被判低分的假三元组”。负采样策略决定了训练信号的质量，进而直接影响链接预测的排序指标，例如 MRR 和 Hits@K。

只用均匀随机替换头实体或尾实体，通常能把训练跑起来，但很快会遇到两个问题：第一，很多负例过于明显，模型一眼就能分开，梯度很弱；第二，负例质量不稳定，容易把计算预算浪费在“毫无挑战”的样本上。类型约束采样先解决第一个问题，它用实体类型过滤候选集合，避免生成明显荒谬的负例。对抗负采样和自对抗负采样进一步解决第二个问题，它们的目标都是持续提供 hard negative，也就是“当前模型最容易搞混的假样本”。

两者的区别在于负例由谁来挑。对抗负采样通常有一个额外的生成器，专门产出难例；自对抗负采样不引入额外模型，而是直接用当前知识图谱嵌入模型自己的得分，对一批负例做重加权。实现上，自对抗更简单，收益通常更稳定，因此在工程里更常见。

对新手最重要的判断标准只有一个：如果你的负例大多是 easy negative，训练就会被稀释；如果你的负例在类型上合理、在得分上接近正例，模型才会学到真正有用的判别边界。实践里，`type constraint + self-adversarial` 往往是最有性价比的组合。在 FB15k-237、WN18RR 这类基准上，这类 hard negative 机制常见到约 2 到 6 个点的 MRR 改善，若换算成相对增益，常落在 1% 到 3% 左右。

下面这张表可以先建立一个直观印象，数值是示意，不同模型会有偏差：

| 策略 | 负例来源 | 合理性 | 难度控制 | 训练成本 | 常见效果 |
|---|---|---:|---:|---:|---|
| 均匀随机 | 任意替换头/尾 | 低 | 低 | 低 | 能训练，但易被 easy negative 稀释 |
| 类型约束 | 同类型实体中替换 | 中高 | 低 | 低到中 | 指标更稳，减少无效负例 |
| 对抗负采样 | 生成器挑难例 | 高 | 高 | 高 | 可能提升明显，但训练更复杂 |
| 自对抗负采样 | 模型自身得分重加权 | 高 | 高 | 中 | 工程上常用，收益稳定 |

---

## 问题定义与边界

知识图谱是“实体和关系组成的网络”，例如 `(Paris, capitalOf, France)`。知识图谱嵌入训练的目标，是让真实三元组得分高，让错误三元组得分低。但数据集通常只有正例，没有明确标注“这是错的”。所以训练必须自己造负例。

最基础的方法是均匀随机替换。给定正例 $(h,r,t)$，可以随机把头实体替换成 $h'$，得到 $(h',r,t)$；也可以随机把尾实体替换成 $t'$，得到 $(h,r,t')$。这叫 corruption，也就是“把真实事实改坏”。问题是，这种方法很容易造出太明显的错误。例如 `(Paris, capitalOf, France)` 若替换成 `(Car, capitalOf, France)`，这里 `Car` 是交通工具，不是地点，负例虽然是错的，但错得太明显，训练价值很低。

这就是类型约束的边界条件。类型约束的意思是：候选实体先按语义类别过滤。白话说，就是“先保证替换对象像个样子”。如果关系 `capitalOf` 的头实体应该是城市，那么候选池就只从城市中采样，不从国家、人物、动物里采。这样生成的负例更合理，也更接近真实决策边界。

玩具例子可以用“出生地”来理解。正例是 `(Obama, bornIn, Honolulu)`。  
如果用均匀随机，可能得到 `(Obama, bornIn, PacificOcean)` 或 `(Obama, bornIn, Microsoft)`。前者语义勉强接近，后者则完全离谱。  
如果加类型约束，候选池只保留“地理位置”类实体，就更可能得到 `(Obama, bornIn, Chicago)`、`(Obama, bornIn, Nairobi)` 这类难例。模型真正需要区分的，恰恰是这种“类型没错，但事实不对”的样本。

再往前一步，对抗和自对抗的边界在于是否需要外部生成器。  
对抗负采样：额外训练一个模块，让它专门生成最像真的负例。  
自对抗负采样：不加新模块，只用当前模型给候选负例打分，再把高分负例赋予更大权重。

可以把流程抽象成下面这个链条：

`正例三元组 -> 构造候选实体集合 -> 采样策略筛选/加权 -> 进入损失函数 -> 产生梯度`

真实工程里，这个问题还有一个边界：假负例。假负例是“你以为它错了，其实它是真的”。比如 `(Obama, livedIn, Chicago)` 如果数据集中没标，但现实中可能成立。负采样策略越激进，越可能采到这类未标注真值。因此工程上需要在“难例强度”和“假负例风险”之间做平衡，而不是无限追求 hardest negative。

---

## 核心机制与推导

自对抗负采样的关键思想很直接：不是所有负例都同等重要。得分越高的负例，说明模型越容易把它误判成真，越应该重点优化。

设模型打分函数为 $f_\theta(h,r,t)$。白话说，它输出一个分数，表示模型认为这个三元组“像不像真”。分数越高，越像真。对一个正例 $(h,r,t)$，我们先采样出 $k$ 个负例：
$$
\{(h'_1,r,t'_1), (h'_2,r,t'_2), \dots, (h'_k,r,t'_k)\}
$$

然后用当前模型给每个负例打分，并通过 softmax 计算权重：
$$
p_\theta(h'_j,r,t'_j)=\frac{\exp(\beta f_\theta(h'_j,r,t'_j))}{\sum_i \exp(\beta f_\theta(h'_i,r,t'_i))}
$$

这里的 $\beta$ 是温度反比参数。白话说，它控制“你有多偏爱最难的负例”。$\beta$ 越大，权重越集中在高分负例上；$\beta$ 越小，权重越平均。

损失通常写成：
$$
\mathcal{L}=
-\log \sigma(\gamma + f_\theta(h,r,t))
-\sum_j p_\theta(h'_j,r,t'_j)\log \sigma(-\gamma - f_\theta(h'_j,r,t'_j))
$$

其中 $\sigma$ 是 sigmoid，$\gamma$ 是 margin，也就是“正负样本之间至少拉开的安全距离”。这个式子的含义是：正例要拿高分，负例要拿低分；但负例不是平均处理，而是按当前困难程度加权处理。

数值玩具例子最能看清这个机制。假设某个正例对应 3 个负例，它们当前得分分别是：

$$
\{0.8,\ 0.2,\ -0.4\}
$$

若 $\beta = 5$，则 softmax 权重大约是：

$$
\{0.95,\ 0.05,\ 0.00\}
$$

这意味着训练几乎把精力都集中在第一个 hardest negative 上。若 $\beta = 1$，权重大约会更平缓，类似：

$$
\{0.54,\ 0.30,\ 0.16\}
$$

下面的表格展示了 $\beta$ 的直觉作用：

| 负例得分 | $\beta=1$ 权重示意 | $\beta=5$ 权重示意 | 解释 |
|---|---:|---:|---|
| 0.8 | 0.54 | 0.95 | 最难负例被重点优化 |
| 0.2 | 0.30 | 0.05 | 中等负例影响被压缩 |
| -0.4 | 0.16 | 0.00 | 明显 easy negative 几乎无贡献 |

这就是“避免训练被 easy negative 稀释”的数学来源。均匀采样时，每个负例对损失贡献一样；自对抗时，贡献和当前错误概率近似挂钩。

再看一个更贴近业务的真实工程例子。假设你做问答系统里的实体链接预测，正例是 `(Aspirin, mayTreat, Headache)`。  
如果尾实体候选里有 `Fever`、`Pain`、`SQLDatabase`。  
`SQLDatabase` 虽然一定是负例，但没有价值；`Fever` 和 `Pain` 与真实语义更近，是更好的 hard negative。自对抗会因为它们得分更高而自动给更大权重，不需要你手写复杂规则去指定“什么叫难”。

所以，自对抗的核心不是“生成更多负例”，而是“让有限的负例预算优先作用于当前最有学习价值的部分”。

---

## 代码实现

实现上，建议先分两层处理：

1. 用类型约束构建候选池，先去掉明显不合理的实体。  
2. 在每个 batch 内，对候选负例用当前模型重新打分，再做 self-adversarial 加权。

下面给一个可运行的 Python 玩具实现。它不依赖深度学习框架，只演示采样和加权逻辑，便于先把机制讲清楚。

```python
import math

def softmax(xs, beta=1.0):
    scaled = [beta * x for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [x / s for x in exps]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def self_adversarial_loss(pos_score, neg_scores, beta=1.0, gamma=0.0):
    weights = softmax(neg_scores, beta=beta)
    pos_loss = -math.log(sigmoid(gamma + pos_score))
    neg_loss = 0.0
    for w, s in zip(weights, neg_scores):
        neg_loss += -w * math.log(sigmoid(-gamma - s))
    return pos_loss + neg_loss, weights

# 玩具例子：一个正例，三个负例
pos_score = 1.2
neg_scores = [0.8, 0.2, -0.4]

loss1, w1 = self_adversarial_loss(pos_score, neg_scores, beta=1.0)
loss5, w5 = self_adversarial_loss(pos_score, neg_scores, beta=5.0)

# beta 越大，最高分负例权重越集中
assert w5[0] > w1[0]
assert abs(sum(w1) - 1.0) < 1e-9
assert abs(sum(w5) - 1.0) < 1e-9

# 最难负例应该拿到最高权重
assert w5[0] > w5[1] > w5[2]

print("beta=1 weights:", [round(x, 4) for x in w1])
print("beta=5 weights:", [round(x, 4) for x in w5])
print("loss(beta=1):", round(loss1, 4))
print("loss(beta=5):", round(loss5, 4))
```

如果换成训练循环，伪代码可以写成这样：

```python
for batch in train_loader:
    positives = batch["triples"]  # [(h, r, t), ...]

    neg_candidates = []
    for h, r, t in positives:
        # 先按关系角色取类型约束候选池
        if random_head_or_tail():
            pool = typed_entity_pool[(r, "head")]
            sampled_heads = sample_k(pool, k)
            neg_candidates.extend([(h_neg, r, t) for h_neg in sampled_heads])
        else:
            pool = typed_entity_pool[(r, "tail")]
            sampled_tails = sample_k(pool, k)
            neg_candidates.extend([(h, r, t_neg) for t_neg in sampled_tails])

    pos_scores = model.score(positives)
    neg_scores = model.score(neg_candidates).reshape(len(positives), k)

    weights = softmax(beta * neg_scores, dim=-1).detach()
    pos_loss = -logsigmoid(gamma + pos_scores).mean()
    neg_loss = -(weights * logsigmoid(-gamma - neg_scores)).sum(dim=-1).mean()

    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
```

这里有两个工程要点。

第一，`weights` 常常会 `detach()`。原因是自对抗负采样通常只把它当作“重加权系数”，而不是让梯度继续穿过 softmax 权重本身，否则优化行为会更复杂，也更容易不稳定。

第二，候选生成和模型评分最好分工。CPU 侧负责做类型过滤、采样和缓存维护，GPU 侧负责大批量打分。大图谱里，如果每一步都全量检索候选，吞吐会明显下降。因此常见做法是维护一个 adversarial buffer，也就是“最近一段时间内较难的候选缓存”，周期性刷新，而不是每次从头全算。

---

## 工程权衡与常见坑

负采样不是越“聪明”越好，而是要在计算成本、训练稳定性和假负例风险之间取平衡。最常见的误区，是把策略理解成单纯的“提升难度竞赛”。

先看几类典型问题：

| 问题 | 现象 | 根因 | 常见处理 |
|---|---|---|---|
| uniform 导致梯度弱 | loss 很快接近 0 | easy negative 太多 | 加类型约束或自对抗重加权 |
| 类型约束后仍提升有限 | 指标停滞 | 候选虽合理，但不够难 | 在候选池上做 self-adversarial |
| generator oscillation | 指标来回波动 | 生成器和判别器互相追逐 | 降低更新频率、引入缓存、分阶段训练 |
| $\beta$ 太大 | 只盯少数负例，过拟合 | 权重过度集中 | 降低 $\beta$，或做 schedule |
| 假负例增多 | 训练变慢甚至退化 | 候选太接近真实事实 | 过滤已知邻居、多跳相关实体去重 |

均匀采样的第一个坑，是“能学，但学不深”。很多新手看到 loss 在下降，就以为策略有效。实际上如果负例过于简单，loss 下降只说明模型在区分“显然错误”的三元组，并不说明它能区分“语义接近但事实不同”的候选。最终在 MRR 这类排序指标上，常表现为前期涨得快，后期卡住。

对抗负采样的典型坑是震荡。生成器如果更新过快，会不断追着判别器最薄弱的地方打；判别器若还没来得及收敛，训练分布就已经变了。结果是两边都不稳定。这类问题在 GAN 风格方案里尤其明显，所以很多工程团队会优先用自对抗，而不是完整的生成器方案。

自对抗的主要坑则是 $\beta$。$\beta$ 太小，权重接近平均，自对抗退化成普通采样；$\beta$ 太大，模型只盯着一两个最难负例，既可能过拟合，也可能放大假负例噪声。一个实用做法是 schedule，也就是分阶段调整：训练早期用较小 $\beta$ 保证覆盖面，训练后期再逐渐加大，让模型更聚焦 hardest negative。

真实工程例子里，这个问题很常见。比如医疗知识图谱中 `(DrugA, mayTreat, DiseaseB)`，如果候选尾实体里有很多高相关疾病，hard negative 的确更有训练价值；但有些疾病之间存在并发症、共病或别名问题，数据未标注完整时，越“难”的负例越可能是假负例。这里如果直接把 $\beta$ 拉满，模型会被错误监督带偏。因此高质量系统通常会叠加实体别名归一化、已知邻居过滤、规则黑白名单等保护措施。

一个实用结论是：  
先解决“负例是否合理”，再解决“负例是否够难”。  
也就是先上类型约束，再调 self-adversarial。顺序反过来，通常不稳。

---

## 替代方案与适用边界

不同负采样方案没有绝对优劣，关键看图谱规模、算力预算和稳定性要求。

| 方案 | 适用条件 | 优点 | 缺点 |
|---|---|---|---|
| 均匀随机 | 小实验、基线复现 | 简单、便宜 | 负例质量低，后期收益差 |
| 类型约束 | 有实体类型或 ontology | 负例更合理，稳定 | 难度仍可能不足 |
| 对抗生成器 | 追求极致 hard negative，算力充足 | 难例强，潜在收益高 | 复杂、易震荡 |
| 自对抗 | 大多数实际训练任务 | 无需额外模型，效果稳 | 仍需调 $\beta$ 和候选池 |
| 规则过滤 + 重排序 | 领域知识强、误杀成本高 | 可控性好 | 规则维护成本高 |

对小型知识图谱，最推荐的起点不是直接上对抗，而是 `type constraint + uniform`。原因很简单：数据规模小，很多时候瓶颈不是“难例不够”，而是“样本和标注本身不够稳定”。这时先把明显错误的负例过滤掉，通常已经能拿到足够清晰的收益。

对中大型知识图谱，尤其是 FB15k-237、WN18RR 这类标准链接预测任务，`type constraint + self-adversarial` 是更稳的主线。它不要求额外生成器，也不需要重构整体训练框架，只是在已有负采样流程上多一步打分和加权。

对在线问答或推荐系统场景，如果你需要持续更新模型，完整对抗生成器未必合适，因为它训练和维护成本高。更实际的做法是：  
类型约束候选池 + 候选缓存 + 自对抗重加权。  
白话说，就是“先筛合理候选，再从里面优先训最难的”。这样既能保留 hard negative 的优势，又不会把系统复杂度推得过高。

还要强调一个适用边界：如果图谱没有可靠类型信息，类型约束的收益会下降；如果图谱存在大量未标注真值，自对抗和对抗采样都可能放大假负例风险。此时可以改用更保守的 score-based re-ranking，也就是先均匀采样，再只对部分高分候选重排，而不是强行全量 hard negative 化。

因此，工程上的推荐顺序通常是：

1. 先做均匀随机，得到可复现实验基线。  
2. 再加类型约束，提升负例合理性。  
3. 然后引入自对抗，提升负例有效性。  
4. 只有在资源允许且收益明确时，再考虑外部对抗生成器。

这个顺序不是保守，而是符合训练系统的复杂度增长规律。每加一层策略，都应该能解释“为什么它带来的额外复杂度值得”。

---

## 参考资料

- MDPI: *Universal Knowledge Graph Embedding Framework Based on High-Quality Negative Sampling and Weighting*  
  主要内容：系统讨论高质量负采样、类型约束、负样本加权，对工程组合策略有直接参考价值。

- Emergent Mind: *Self-Adversarial Negative Sampling Methods*  
  主要内容：总结自对抗负采样的核心公式、softmax 重加权机制，以及温度参数对 hard negative 聚焦程度的影响。

- Emergent Mind: *Adversarial Negative Sampling*  
  主要内容：概述外部生成器式对抗负采样、hard negative 的收益来源，以及在排序指标上的典型改进范围。
