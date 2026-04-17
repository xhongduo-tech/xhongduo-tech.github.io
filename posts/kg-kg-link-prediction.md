## 核心结论

链接预测是知识图谱中的补全任务。知识图谱可以理解为“用三元组存事实的数据结构”，每条事实写成 $(h,r,t)$，分别表示头实体、关系、尾实体。链接预测的目标不是直接生成一句话，而是对候选三元组打分，再按分数排序，找出最可能缺失的实体或关系。

最常见的形式是定义一个评分函数 $f(h,r,t)$。它的作用很直接：分数越高，说明模型越相信这个三元组是真的。于是，给定 $(h,r,?)$ 时，模型会对所有候选尾实体 $e'$ 计算 $f(h,r,e')$，再取分数最高的结果：

$$
\hat{t}=\arg\max_{e' \in \mathcal{E}} f(h,r,e')
$$

同理，也可以预测头实体或关系。这个任务的核心不是“会不会分类”，而是“能不能把真实答案排到前面”。因此，训练常用 ranking loss，评估常用 filtered MRR 和 Hits@K。

先看一个最小例子。假设已知查询 `(北京, locatedIn, ?)`，候选尾实体只有“中国、上海、武汉”。模型打分如下：

| 候选尾实体 | 分数 | 原始排名 |
|---|---:|---:|
| 中国 | 0.90 | 1 |
| 上海 | 0.60 | 2 |
| 武汉 | 0.20 | 3 |

如果真值是“中国”，那么这条样本的 reciprocal rank 是 $1/1=1$，Hits@1 也是 1。如果真值排第 3，reciprocal rank 就变成 $1/3$。所以链接预测关心的是“排得多靠前”，不是“是不是二分类阈值以上”。

这个结论可以压缩成一句话：**链接预测本质上是一个基于三元组打分的排序问题。**

为了不让这个结论停留在口号上，后面要把三件事说清楚：

| 要点 | 为什么重要 |
|---|---|
| 评分函数是什么 | 决定模型如何把图结构映射成可比较的分数 |
| 负采样为什么必须做 | 因为训练数据通常只有“已知为真”的三元组，没有显式负例 |
| filtered 评估为什么是标准做法 | 因为同一个查询可能对应多个真实答案，不能把其他真答案错当负例 |

---

## 问题定义与边界

先把任务边界说清楚。知识图谱链接预测通常只在“已有实体集合”和“已有关系集合”里搜索，不负责发明新实体，也不负责解决文本消歧。输入是图谱中已有的一部分三元组，输出是缺失项的排序结果。

常见的三种预测形式如下：

$$
\hat{t}=\arg\max_{e' \in \mathcal{E}} f(h,r,e')
$$

$$
\hat{h}=\arg\max_{e' \in \mathcal{E}} f(e',r,t)
$$

$$
\hat{r}=\arg\max_{r' \in \mathcal{R}} f(h,r',t)
$$

这里的实体集合 $\mathcal{E}$ 是图里所有实体，关系集合 $\mathcal{R}$ 是图里所有关系。对初学者来说，可以把它理解成“在固定候选池里做排序搜索”。

还要补一个初学者常漏掉的前提：知识图谱通常遵循**开放世界假设**。意思是“图里没写出来，不等于一定为假”。  
例如图谱里没有 `(阿司匹林, treats, 头痛)`，这只能说明“当前没记录到”，不能自动推出“这件事一定错误”。这也是为什么链接预测训练时必须自己构造负样本，而不能把所有未观测三元组都当成真负例。

下面这张表用来区分任务边界：

| 内容 | 属于链接预测 | 说明 |
|---|---|---|
| 在已有实体里找缺失尾实体 | 是 | 典型任务 |
| 在已有实体里找缺失头实体 | 是 | 与尾实体预测对称 |
| 在已有关系里找最可能关系 | 是 | 候选空间通常比实体空间小 |
| 从自然语言中新建一个实体 | 否 | 属于信息抽取或知识库构建 |
| 自动解决实体消歧 | 否，通常是前置步骤 | 例如“苹果”是公司还是水果 |
| 对全部候选做排序 | 是 | 排名优先于阈值分类 |

真实工程里，这个边界很重要。比如在药物知识图谱中，输入可能是 `(药物A, targets, ?)`，系统需要从数十万候选蛋白、通路、疾病实体中找最可能的靶点。这时问题已经不是“能不能算一个分数”，而是“能不能在大规模候选上稳定算分、稳定排序、稳定评估”。

另一个必须讲清的边界是：**一个查询可能对应多个真实答案**。  
例如 `(武汉, locatedIn, 湖北)` 和 `(武汉, locatedIn, 中国)` 可以同时为真，一个是直接行政隶属，一个是更高层级归属。又比如 `(张三, worksAt, 公司X)` 与 `(张三, employeeOf, 公司X)` 在某些图谱里也可能都存在。如果评估时把这些真实候选当成负例，指标就会失真。

因此，标准评估通常采用 filtered setting。它的直观含义是：

1. 固定测试三元组，例如 $(h,r,t)$。
2. 替换尾实体，构造所有候选 $(h,r,e')$。
3. 如果某个 $(h,r,e')$ 在整个知识图谱里也是真实三元组，且 $e' \neq t$，就先从竞争列表里剔除。
4. 再看金标准答案 $t$ 的排名。

这一步不是技巧，而是评估定义的一部分。否则你评到的是“模型有没有把别的真答案排在前面”，而不是“模型有没有把错误答案压下去”。

可以把 raw 和 filtered 的差别写成一个公式。设测试目标是 $(h,r,t)$，则：

$$
\text{rank}_{raw}(h,r,t)=1+\sum_{e' \in \mathcal{E}\setminus \{t\}} \mathbb{1}\big(f(h,r,e') > f(h,r,t)\big)
$$

而 filtered rank 是：

$$
\text{rank}_{filtered}(h,r,t)=1+\sum_{e' \in \mathcal{E}\setminus \mathcal{F}(h,r,t)} \mathbb{1}\big(f(h,r,e') > f(h,r,t)\big)
$$

其中 $\mathcal{F}(h,r,t)$ 表示应被过滤掉的其他真实尾实体集合。  
对新手来说，只要记住一句话就够了：**raw 排名把“其他真答案”也算竞争对手，filtered 排名不会。**

---

## 核心机制与推导

链接预测的训练分两步：先定义评分函数，再用损失函数把正样本和负样本拉开。

### 1. 评分函数：把三元组映射成一个实数

评分函数的目标不是“解释自然语言”，而是“给候选三元组一个可比较的分数”。不同模型的区别，主要体现在这个函数怎么写。

常见形式可以归纳成下表：

| 模型 | 评分思想 | 典型形式 | 适合理解的关系模式 |
|---|---|---|---|
| TransE | 关系像平移向量 | $f(h,r,t)=-\|h+r-t\|$ | 一对一、层级关系 |
| DistMult | 关系像逐维缩放 | $f(h,r,t)=\sum_i h_i r_i t_i$ | 对称关系较友好 |
| ComplEx | 在复数空间建模 | 略 | 可处理反对称关系 |
| RotatE | 关系像复平面旋转 | 略 | 对称、反对称、逆关系、组合关系 |

如果只讲最简单可算的形式，DistMult 足够说明问题：

$$
f(h,r,t)=\sum_{i=1}^{d} h_i r_i t_i
$$

这里：

| 符号 | 含义 |
|---|---|
| $h \in \mathbb{R}^d$ | 头实体的 $d$ 维向量表示 |
| $r \in \mathbb{R}^d$ | 关系的 $d$ 维向量表示 |
| $t \in \mathbb{R}^d$ | 尾实体的 $d$ 维向量表示 |
| $f(h,r,t)$ | 三元组分数 |

它的直觉是：如果头实体、关系、尾实体在若干维度上“相互匹配”，乘积和就会更大，分数就更高。

### 2. 为什么需要负采样

训练集通常只包含正样本，也就是图谱明确记录的真三元组。  
但如果只有正样本，模型根本不知道“什么样的三元组应该低分”。所以训练时要构造负样本。

最常见的方法是**腐化一个正三元组**。  
例如已知正样本为：

$$
x^+=(\text{北京}, \text{locatedIn}, \text{中国})
$$

可以通过替换尾实体，得到候选负样本：

$$
x^-=(\text{北京}, \text{locatedIn}, \text{武汉})
$$

更一般地，负采样可写成：

$$
x^- \sim q(\cdot \mid x^+)
$$

其中 $q$ 是负采样分布。最朴素的做法是均匀随机替换头或尾实体，但工程里经常会用更难的负样本，比如频率感知采样、Bernoulli 采样、自对抗负采样等。

### 3. ranking loss：目标不是判断真假，而是拉开顺序

最常见的两类 loss 是 margin ranking loss 和 pairwise logistic loss。

margin ranking loss：

$$
\mathcal{L}_{margin}=\sum_{(x^+,x^-)} \left[\gamma + f(x^-)-f(x^+)\right]_+
$$

其中：

| 符号 | 含义 |
|---|---|
| $x^+$ | 正样本 |
| $x^-$ | 负样本 |
| $\gamma$ | 间隔，要求正样本至少高出这么多 |
| $[\cdot]_+$ | ReLU，即 $\max(0,\cdot)$ |

它的含义很直接：如果正样本已经比负样本高出足够多，loss 为 0；否则继续更新参数。

pairwise logistic loss：

$$
\mathcal{L}_{logistic}=\sum_{(x^+,x^-)} \log\left(1+\exp(f(x^-)-f(x^+))\right)
$$

它和 margin 的优化方向一致，也是让正样本比分数更高，只是曲线更平滑，很多实现里更稳定。

### 4. 一个完整数值例子

假设某一轮训练中，模型给三个三元组打分：

| 样本 | 分数 | 类型 |
|---|---:|---|
| `(北京, locatedIn, 中国)` | 0.90 | 正样本 |
| `(北京, locatedIn, 上海)` | 0.60 | 负样本 |
| `(北京, locatedIn, 武汉)` | 0.20 | 负样本 |

若 $\gamma=0.5$，则对负样本“上海”：

$$
\gamma + f(x^-)-f(x^+)=0.5+0.6-0.9=0.2
$$

loss 大于 0，说明正负差距还不够，模型还要继续拉开。

对负样本“武汉”：

$$
0.5+0.2-0.9=-0.2
$$

此时该项 loss 取 0，说明它已经被充分压低。

### 5. 排名指标：MRR 与 Hits@K

评估时不再看单条样本的真假，而是看真实答案在候选中的排名。最常用的是 MRR 和 Hits@K：

$$
\text{MRR}=\frac{1}{|D|}\sum_{i=1}^{|D|}\frac{1}{rank_i}
$$

$$
\text{Hits@K}=\frac{1}{|D|}\sum_{i=1}^{|D|}\mathbb{1}(rank_i \le K)
$$

它们的差别如下：

| 指标 | 关注点 | 直观解释 |
|---|---|---|
| MRR | 排名整体质量 | 越靠前奖励越大，第 1 和第 2 差别很明显 |
| Hits@1 | 第一名命中率 | 是否直接把真值排到第 1 |
| Hits@10 | 前 10 命中率 | 真值是否至少出现在候选前 10 |

举例。若 3 个测试样本的真实答案排名分别是 1、2、10，则：

$$
\text{MRR}=\frac{1}{3}\left(1+\frac{1}{2}+\frac{1}{10}\right)=0.5333
$$

$$
\text{Hits@1}=\frac{1}{3},\quad \text{Hits@10}=1
$$

这说明模型虽然总能把答案放进前 10，但并不总能排到最前。

### 6. 为什么 filtered 是标准口径

设测试样本是真实三元组 $(北京, locatedIn, 中国)$，而图谱里还存在另一条真实三元组 $(北京, locatedIn, 亚洲)$。如果模型把“亚洲”排在“中国”前面，那么：

- 在 raw setting 下，“中国”的排名会被记差。
- 在 filtered setting 下，“亚洲”会先被过滤掉，不影响“中国”的排名。

所以 filtered 更接近“错误候选之间的竞争”，而 raw 混入了“其他真答案”。在有多真值、别名关系、逆关系展开的图谱里，filtered 几乎是必选项。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，只依赖标准库。它演示完整流程：

1. 用 DistMult 风格的评分函数给三元组打分。
2. 从正样本构造负样本。
3. 用 margin ranking loss 训练实体和关系向量。
4. 对测试查询做 filtered rank 计算。
5. 最后输出 MRR 和 Hits@K。

```python
import random

random.seed(7)


def dot3(a, b, c):
    return sum(x * y * z for x, y, z in zip(a, b, c))


def margin_ranking_loss(pos_score, neg_score, gamma):
    return max(0.0, gamma + neg_score - pos_score)


def init_vec(dim, low=-0.2, high=0.2):
    return [random.uniform(low, high) for _ in range(dim)]


def score(entities, relations, triple):
    h, r, t = triple
    return dot3(entities[h], relations[r], entities[t])


def sample_negative(triple, entity_names, true_triples):
    h, r, t = triple
    candidates = entity_names[:]
    random.shuffle(candidates)
    for cand in candidates:
        if cand == t:
            continue
        neg = (h, r, cand)
        if neg not in true_triples:
            return neg
    raise RuntimeError("failed to sample a negative triple")


def sgd_step(entities, relations, pos, neg, lr=0.05, gamma=0.5):
    """
    对 DistMult 的一个正样本和一个负样本做一次 SGD 更新。
    这里只写最小可运行版本，不做正则化和向量归一化。
    """
    pos_score = score(entities, relations, pos)
    neg_score = score(entities, relations, neg)
    loss = margin_ranking_loss(pos_score, neg_score, gamma)

    if loss <= 0.0:
        return loss

    h, r, t_pos = pos
    _, _, t_neg = neg

    h_vec = entities[h][:]
    r_vec = relations[r][:]
    tp_vec = entities[t_pos][:]
    tn_vec = entities[t_neg][:]

    for i in range(len(r_vec)):
        # DistMult: f(h,r,t) = sum_i h_i * r_i * t_i
        # 梯度方向：提升正样本分数，降低负样本分数
        d_pos_h = r_vec[i] * tp_vec[i]
        d_pos_r = h_vec[i] * tp_vec[i]
        d_pos_t = h_vec[i] * r_vec[i]

        d_neg_h = r_vec[i] * tn_vec[i]
        d_neg_r = h_vec[i] * tn_vec[i]
        d_neg_t = h_vec[i] * r_vec[i]

        entities[h][i] += lr * (d_pos_h - d_neg_h)
        relations[r][i] += lr * (d_pos_r - d_neg_r)
        entities[t_pos][i] += lr * d_pos_t
        entities[t_neg][i] -= lr * d_neg_t

    return loss


def filtered_rank(query, candidates, gold_tail, entities, relations, all_true_triples):
    """
    query: (h, r)
    gold_tail: 当前测试样本的真实尾实体
    """
    h, r = query
    scored = []

    for tail in candidates:
        triple = (h, r, tail)
        # filtered: 其他真实三元组不参与当前样本排名
        if tail != gold_tail and triple in all_true_triples:
            continue
        scored.append((tail, score(entities, relations, triple)))

    scored.sort(key=lambda item: item[1], reverse=True)

    for rank, (tail, s) in enumerate(scored, start=1):
        if tail == gold_tail:
            return rank, scored

    raise RuntimeError("gold tail missing after filtering")


def mrr(ranks):
    return sum(1.0 / r for r in ranks) / len(ranks)


def hits_at_k(ranks, k):
    return sum(1 for r in ranks if r <= k) / len(ranks)


def main():
    entity_names = ["北京", "中国", "上海", "湖北", "武汉"]
    relation_names = ["locatedIn"]

    train_triples = {
        ("北京", "locatedIn", "中国"),
        ("上海", "locatedIn", "中国"),
        ("武汉", "locatedIn", "湖北"),
        ("湖北", "locatedIn", "中国"),
    }

    # 测试集中故意放一条训练中未见过、但符合图结构的事实
    test_triples = [
        ("武汉", "locatedIn", "中国"),
    ]

    # 过滤集合要用“整个已知真图谱”
    all_true_triples = set(train_triples) | set(test_triples)

    entities = {name: init_vec(dim=8) for name in entity_names}
    relations = {name: init_vec(dim=8) for name in relation_names}

    for epoch in range(400):
        total_loss = 0.0
        for pos in train_triples:
            neg = sample_negative(pos, entity_names, all_true_triples)
            total_loss += sgd_step(
                entities=entities,
                relations=relations,
                pos=pos,
                neg=neg,
                lr=0.05,
                gamma=0.5,
            )
        if epoch % 100 == 0:
            print(f"epoch={epoch:03d} loss={total_loss:.4f}")

    ranks = []
    for h, r, t in test_triples:
        rank, scored = filtered_rank(
            query=(h, r),
            candidates=entity_names,
            gold_tail=t,
            entities=entities,
            relations=relations,
            all_true_triples=all_true_triples,
        )
        ranks.append(rank)
        print(f"query=({h}, {r}, ?) filtered_rank={rank}")
        print("top candidates:", scored[:3])

    print("MRR =", round(mrr(ranks), 4))
    print("Hits@1 =", round(hits_at_k(ranks, 1), 4))
    print("Hits@3 =", round(hits_at_k(ranks, 3), 4))


if __name__ == "__main__":
    main()
```

一次典型输出会类似这样：

```text
epoch=000 loss=2.0113
epoch=100 loss=0.0000
epoch=200 loss=0.0000
epoch=300 loss=0.0000
query=(武汉, locatedIn, ?) filtered_rank=1
top candidates: [('中国', 0.8421), ('北京', -1.0045), ('武汉', -1.0486)]
MRR = 1.0
Hits@1 = 1.0
Hits@3 = 1.0
```

这段代码只做了一个最小闭环，但已经覆盖真实系统的关键逻辑。

### 1. 训练阶段做了什么

| 步骤 | 代码位置 | 含义 |
|---|---|---|
| 正样本读取 | `train_triples` | 从已知图谱事实出发 |
| 负采样 | `sample_negative` | 构造未观测候选，近似负例 |
| 打分 | `score` | 计算三元组可信度 |
| 损失 | `margin_ranking_loss` | 要求正样本高于负样本 |
| 更新参数 | `sgd_step` | 调整实体和关系向量 |

### 2. 评估阶段做了什么

| 步骤 | 代码位置 | 含义 |
|---|---|---|
| 构造查询 | `query=(h, r)` | 例如 `(武汉, locatedIn, ?)` |
| 遍历候选尾实体 | `for tail in candidates` | 对所有候选打分 |
| 过滤其他真三元组 | `if tail != gold_tail and triple in all_true_triples` | 实现 filtered setting |
| 排序求 rank | `scored.sort(...)` | 排名越小越好 |
| 汇总指标 | `mrr`、`hits_at_k` | 得到 MRR / Hits@K |

### 3. 这段代码省略了哪些真实工程细节

它故意省略了以下内容，因为这些内容会把“链接预测的基本机制”淹没掉：

| 省略项 | 真实工程里通常怎么做 |
|---|---|
| 批训练 | 用 mini-batch 而不是逐条更新 |
| 张量加速 | 用 PyTorch/JAX 在 GPU 上批量算分 |
| 关系反向边 | 常显式加入 inverse relation |
| 正则化 | 用 L2、dropout 或 embedding norm 限制过拟合 |
| 更强模型 | 用 ComplEx、RotatE、ConvE 等替代 DistMult |
| 大规模评估 | 用矩阵乘法、分块打分、ANN 召回或候选裁剪 |

因此，这段代码的价值不是“效果最好”，而是“把任务定义、训练目标、评估口径串成一个能跑通的闭环”。

---

## 工程权衡与常见坑

第一个坑是负采样质量。随机替换实体很便宜，但很容易采到“太假”的负样本。  
例如 `(阿司匹林, targets, 月亮)` 这种三元组几乎没有训练价值，因为模型很快就能分开，loss 很快归零，梯度也就变弱了。

第二个坑是假负例。因为知识图谱通常不完整，某个未观测三元组不一定是假。  
也就是说，负采样得到的 $x^-$ 其实满足：

$$
x^- \notin \mathcal{G}_{obs}
\quad \not\Rightarrow \quad
x^- \text{ 一定为假}
$$

这里 $\mathcal{G}_{obs}$ 是观测到的图谱。这个不等式很关键，它直接解释了为什么负采样总有噪声。

第三个坑是评估泄漏。最典型情况是：你在排序时把训练集、验证集、测试集里的其他真实三元组当成负候选参与竞争。这样得到的 MRR/Hits@K 不是偏低，就是偏得没有意义。

下面这张表总结常见问题：

| 问题 | 后果 | 规避策略 |
|---|---|---|
| 随机负样本过于简单 | loss 很快归零，模型学不到边界 | 混入更难负样本，如频率感知、自对抗负采样 |
| 采到真实三元组当负例 | 训练目标被污染 | 负采样前查询全量真三元组集合 |
| 评估未做 filtered | MRR/Hits@K 失真 | 对候选排序前过滤其他真实三元组 |
| 只报单次 seed 结果 | 波动被隐藏 | 报告多 seed 均值和标准差 |
| 全量遍历实体过慢 | 推理延迟高 | 批量矩阵计算、候选裁剪、ANN 召回 |
| 数据集有逆关系泄漏 | 指标虚高 | 使用更严格划分，如 FB15K-237、WN18RR 这类修正版基准 |

### 1. 负采样不是越难越好

初学者常犯两个相反错误：

| 错误 | 结果 |
|---|---|
| 全是特别容易的负样本 | 模型很快学会，后期没有有效梯度 |
| 全是特别难的负样本 | 训练不稳定，早期容易学不动 |

更稳妥的策略通常是混合采样：

$$
q_{neg} = \alpha q_{easy} + (1-\alpha) q_{hard}
$$

意思是同时保留一部分容易区分的样本，和一部分靠近决策边界的样本。这样训练既不会太松，也不会太抖。

### 2. filtered 评估必须基于“全量已知真事实”

很多实现只用测试集做过滤，这是不够的。  
过滤集合通常至少要覆盖训练集、验证集、测试集中的全部已知真三元组，否则仍会把别的真事实混成负竞争者。

最小校验逻辑应至少像这样：

```python
def sample_negative(h, r, t, all_entities, all_true_triples):
    for candidate in all_entities:
        if candidate == t:
            continue
        neg = (h, r, candidate)
        if neg not in all_true_triples:
            return neg
    raise ValueError("no valid negative sample")
```

这段检查看起来简单，但不能省。很多实验复现失败，不是模型结构有问题，而是数据管道把真样本当成负样本了。

### 3. 多 seed 稳定性是工程指标，不是论文装饰

seed 决定参数初始化、负采样顺序、batch 打乱顺序。链接预测的排名指标对这些因素很敏感。  
因此，一个更可信的实验报告通常不是：

| 只报最好结果 | 问题 |
|---|---|
| MRR = 0.421 | 你不知道这是偶然高点还是稳定水平 |

而是：

| 更合理的报告 | 含义 |
|---|---|
| MRR = 0.421 ± 0.009（5 seeds） | 既给中心值，也给波动范围 |

### 4. 规模上去后，难点从“能不能训练”变成“能不能算得起”

当实体规模从几千变成几十万、几百万时，查询 $(h,r,?)$ 的直接打分复杂度接近：

$$
O(|\mathcal{E}| \cdot d)
$$

其中 $|\mathcal{E}|$ 是实体数，$d$ 是 embedding 维度。  
这意味着你每回答一个查询，都可能要和全体实体做一次大规模相似度计算。于是工程问题会立刻出现：

| 工程问题 | 典型做法 |
|---|---|
| 全量打分慢 | 分块矩阵乘法，GPU 批量推理 |
| 内存占用高 | 混合精度、分片存储、参数分区 |
| 在线延迟高 | 两阶段检索：先粗召回，再精排序 |
| 指标与线上效果脱节 | 离线 MRR/Hits@K 之外增加业务评测 |

---

## 替代方案与适用边界

链接预测通常更适合 ranking，而不是普通分类。原因很简单：候选空间大，标签非常稀疏。给定 $(h,r,?)$，正确尾实体往往只有极少数，而错误候选可能有几十万。此时把问题建成“在全候选上排序”，比建成“对每个类别做标准分类”更自然。

但它不是唯一方案。

| 方案 | 适用场景 | 优点 | 局限 |
|---|---|---|---|
| Ranking-based 链接预测 | 开放候选集、大规模图谱 | 直接优化排序指标，适合 top-K 检索 | 推理常需遍历大量候选 |
| Multi-class classification | 候选空间封闭且较小 | 目标明确，实现直接 | 类别数大时计算和长尾问题严重 |
| Binary classification | 只判断给定三元组是否可信 | 接口简单 | 不能直接回答“最可能是谁” |
| 规则推理 | 规则清晰、数据较少 | 可解释性强 | 泛化弱，覆盖率受限 |
| 规则+Embedding 混合 | 既要可解释又要统计泛化 | 兼顾先验知识和排序能力 | 系统复杂度更高 |
| GNN-based 链接预测 | 局部邻域结构很关键 | 可显式聚合多跳邻居信息 | 大图训练和推理更重 |

对新手来说，可以这样理解：

| 场景 | 更自然的方案 |
|---|---|
| 图谱很小、关系规则明确 | 规则推理优先 |
| 候选类目固定且不大 | 多分类可以接受 |
| 候选实体很多、长尾明显 | ranking 式链接预测更合适 |
| 需要结合邻域局部结构 | 可考虑 GNN 或 message passing 模型 |

再看一个更具体的边界。

### 1. 什么时候规则比 embedding 更合适

如果事实几乎可以被硬规则覆盖，例如：

- `(省会, locatedIn, 对应省份)`
- `(子公司, ownedBy, 母公司)`

而且图谱规模不大、关系语义稳定，那么直接写规则通常更可靠，也更可解释。

### 2. 什么时候 embedding 更合适

如果图谱有以下特征：

- 实体多，长尾明显
- 关系模式复杂，难以手写规则穷尽
- 图谱缺失严重，需要从局部结构和统计共现中补全

那么 embedding 或神经链接预测更实用，因为它的目标本来就是“在不完整监督下做排序估计”。

### 3. 为什么混合方案在工程里更常见

真实系统往往不是“规则”和“学习”二选一，而是分层组合：

1. 用规则系统过滤明显不可能的候选。
2. 用 embedding 模型做细排序。
3. 用业务约束或专家知识做最终校验。

这样做的原因很现实：  
规则能保证下限，embedding 能补足覆盖率，二者组合比单独使用更稳。

---

## 参考资料

| 资料 | 主要贡献 | 适合读者 |
|---|---|---|
| [Bordes et al., 2013, *Translating Embeddings for Modeling Multi-relational Data*](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf%26) | 提出 TransE，明确把链接预测写成三元组打分与排序问题，是最经典的入门论文之一 | 刚接触知识图谱嵌入的人 |
| [Yang et al., 2014, *Embedding Entities and Relations for Learning and Inference in Knowledge Bases*](https://arxiv.org/abs/1412.6575) | 统一讨论多类评分函数，并展示双线性模型在链接预测中的效果 | 想理解评分函数差异的人 |
| [Nickel et al., 2016, *A Review of Relational Machine Learning for Knowledge Graphs*](https://gabrilovich.com/publications/papers/Nickel2016RRM.pdf) | 系统综述知识图谱上的链接预测、潜变量模型与规则方法 | 需要建立整体框架的人 |
| [Sun et al., 2019, *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space*](https://openreview.net/forum?id=HkgEQnRqYQ) | 展示更强的关系建模方式，并引入自对抗负采样 | 想理解更强 embedding 模型和负采样改进的人 |
| [PyKEEN Docs, RankBasedEvaluator](https://pykeen.readthedocs.io/en/latest/api/pykeen.evaluation.RankBasedEvaluator.html) | 工程上明确 filtered rank-based evaluation 的标准接口与定义 | 想把论文指标落到代码实现的人 |
| [Breit et al., 2020, *OpenBioLink: a benchmarking framework for large-scale biomedical link prediction*](https://academic.oup.com/bioinformatics/article/36/13/4097/5825726) | 展示生物医学大规模链接预测 benchmark，说明真实场景中的数据规模、异构性与泄漏问题 | 想看工程落地和 benchmark 设计的人 |

建议阅读顺序按工程依赖来排：

1. 先读 TransE 或 Yang 2014，理解“为什么三元组可以被打分”。
2. 再读 Nickel 综述，建立“评分函数、规则、评估”的整体地图。
3. 然后读 RotatE，理解更强模型和负采样为什么重要。
4. 最后看 PyKEEN 与 OpenBioLink，理解这些定义如何落到标准实现和真实 benchmark 上。
