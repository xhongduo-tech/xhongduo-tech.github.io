## 核心结论

记忆蒸馏的目标，不是让 Agent 保存更多历史，而是把大量一次性的经历压缩成少量可复用规则。这里的“蒸馏”指的是：从许多具体事件中，提取出跨场景仍然成立的稳定处理规律。

当 Agent 已经积累了上千条 episodic memory 时，episodic memory 指“某一次具体发生过的交互记录”，继续逐条检索通常会遇到三个直接问题：

1. 检索延迟上升，因为候选集合越来越大。
2. 上下文膨胀，因为需要携带更多原始片段进入提示词。
3. 噪声增加，因为许多历史样本只是同一模式的重复表达。

因此，更稳妥的做法通常不是继续扩展原始记忆库，而是建立一层规则层：

1. 先把语义相近的经验向量做聚类。
2. 再让 LLM 为每个簇归纳一条可复用规则。
3. 在线检索时优先查规则，只在必要时补少量原始记忆作为证据。

一个常见配置是用 DBSCAN 做聚类。DBSCAN 是一种基于密度的聚类方法，它的判断逻辑不是“必须分成几类”，而是“距离足够近且数量足够多的点形成一个簇，孤立点视为噪声”。在 embedding 已经做过归一化、语义空间相对稳定时，`eps=0.3`、`min_samples=3~5` 往往可以作为初始试验区间。

最直接的收益是压缩比。假设系统中有 1000 条原始记忆，聚类并归纳后只保留 50 条规则，那么推理阶段不必再先扫描 1000 条日志，而是先在 50 条规则中做召回。许多高重复场景中的典型结果是：任务表现只下降约 0%~2%，但检索速度可以提升到原来的 5 到 20 倍。

下面这个玩具例子最容易理解：

| 方案 | 存储单元 | 检索时加载 | 典型延迟 | 准确率变化 |
|---|---:|---:|---:|---:|
| 原始记忆检索 | 1000 条对话 | 20~100 条候选 | 高 | 基线 |
| 规则蒸馏检索 | 50 条规则 + 少量原始记忆 | 3~10 条规则 + 2~5 条原始记忆 | 低 | 约下降 0%~2% |

客服 Agent 的 1000 条对话里，很多问题其实只是“退款时效”“发票开具”“地址修改”几类场景的重复变体。把这些对话按语义聚成若干簇，再把每个簇总结成一句规则，例如：

> 当用户询问退款到账时间时，先确认支付渠道，再告知不同渠道的预计时效。

后续检索时，只要命中这条规则，再补充少量该簇中的原始对话作为证据即可，不需要重新回看全部历史。

数学上，规则检索本质上仍然是相似度检索，只是候选对象从“原始记忆”换成了“规则”：

$$
s(q,m)=\frac{q\cdot m}{\|q\|\,\|m\|}
$$

其中：

| 符号 | 含义 |
|---|---|
| $q$ | 当前问题的 embedding |
| $m$ | 某条规则或某条原始记忆的 embedding |
| $q \cdot m$ | 两个向量的点积 |
| $\|q\|,\|m\|$ | 向量的模长，用于归一化 |

embedding 指“把文本映射成向量的数值表示”。相似度越高，说明当前问题越可能适用这条规则。

这个结论可以压缩成一句话：**记忆蒸馏不是扩容存储，而是把历史经验从日志形态转成知识形态。**

---

## 问题定义与边界

问题的本质不是“记忆不够多”，而是“记忆类型不对”。大量原始 episodic memory 更像日志，而不是知识。日志适合审计、复盘、回溯；知识适合在新任务到来时直接指导行为。Agent 一旦长期运行，就会不断写入“措辞不同但处理步骤相近”的经历，检索系统会越来越像“在海量近似样本里重复寻找共同模式”。

因此，记忆蒸馏要解决的是两个边界非常明确的问题：

1. 如何把高重复经验压缩成规则。
2. 如何保证压缩之后，不损伤关键行为和关键证据。

这里的“规则”不是硬编码的 if-else，也不是业务代码中的固定分支，而是一种可复用的行为摘要。例如：

- 当用户提供错误订单号时，先要求校验订单 ID，再继续退款流程。
- 当用户投诉物流延迟时，先确认承运商和发货状态，再决定是否补偿。
- 当用户情绪激烈但诉求不清时，先复述问题并确认目标，再进入具体处理步骤。

这些规则必须满足一个要求：**它们应该能跨多个样本成立，而不是只是在复述某一条对话。**

适用边界可以用下表概括：

| 条件 | 是否适合做规则蒸馏 | 推荐策略 |
|---|---|---|
| 记忆条目少于 100 条 | 通常不急需 | 直接原始检索即可 |
| 记忆条目 100~1000 条，场景重复明显 | 适合 | 聚类 + 规则归纳 |
| 记忆条目超过 1000 条，多轮任务反复出现 | 很适合 | 规则层 + 原始层双层检索 |
| 高保真审计场景 | 谨慎 | 保留原始记忆优先，规则仅作辅助 |
| 多用户共享系统 | 必须分层 | 按 user/type 隔离后再蒸馏 |
| 时效性强的业务 | 必须加 TTL | 规则可保留，原始事实要过期 |

“边界”最容易被误解的地方有两个。

第一，规则蒸馏不等于删除原始记忆。更准确地说，它是在原始记忆之上增加一个摘要层。原始记忆仍然要保留，尤其是以下三类内容不能只靠规则替代：

| 类型 | 为什么不能只保留规则 |
|---|---|
| 低频但高风险事件 | 样本太少，无法稳定归纳 |
| 合规与审计记录 | 需要保留可追溯证据 |
| 反常案例与异常输入 | 往往是改进系统的重要信号 |

第二，规则蒸馏并不适合所有任务。如果任务本身变化快、样本少、每个案例都高度独特，那么聚类结果通常不是规则，而是噪声平均值。比如法律合同审查、复杂研发故障定位、一次性高价值企业谈判，这类任务里每条案例都包含独特事实，直接检索原始证据往往更可靠。

一个简单判断标准是：如果你观察到大量记忆只是“换了说法，但处理流程基本不变”，就说明系统已经进入适合做蒸馏的阶段。反过来，如果大多数样本的处理方式都依赖具体上下文细节，那么规则层的收益会明显下降。

---

## 核心机制与推导

完整机制可以写成一条流水线：

原始 episodic 记忆  
$\rightarrow$ embedding 向量化  
$\rightarrow$ DBSCAN 聚类  
$\rightarrow$ 每簇交给 LLM 归纳规则  
$\rightarrow$ 规则 embedding 入库  
$\rightarrow$ 在线检索规则  
$\rightarrow$ rerank  
$\rightarrow$ 必要时补原始记忆

这里的 rerank 指“二次排序”，即先用便宜的检索模型做粗召回，再用更强的模型或更复杂的打分方法做精排。

### 1. 为什么先聚类再归纳

如果直接把 1000 条原始记忆全部喂给 LLM，让它“总结经验”，通常会出现两个问题：

1. 成本高，因为上下文太长。
2. 质量不稳，因为不同场景会相互污染。

举一个新手更容易理解的例子。假设原始记忆里同时包含下面三类样本：

- “支付宝退款多久到”
- “发货前如何修改地址”
- “发票能否补开专票”

如果不先聚类，LLM 在总结时很容易给出一种很空泛的描述，例如：

> 先确认订单状态，再根据用户诉求进入对应流程。

这句话并不完全错误，但可执行性很差，因为它没有保留每个场景真正关键的触发条件和操作步骤。聚类的作用，就是先把相似问题、相似处理方式放在一起，让每次总结只针对一组内部一致的样本进行。

DBSCAN 特别适合这个场景，原因有三点：

| 原因 | 说明 |
|---|---|
| 不要求预先指定簇数 | 记忆系统通常无法提前知道会有多少种模式 |
| 能显式标记噪声点 | 低频、偶发、不可泛化的经验可直接排除 |
| 对不规则簇较友好 | 真实语义空间中的簇往往不是标准球形 |

DBSCAN 的两个核心参数如下：

| 参数 | 含义 | 调大后的效果 | 调小后的效果 |
|---|---|---|---|
| `eps` | 邻域半径，表示“多近算相似” | 更容易合并成大簇 | 更容易碎成小簇 |
| `min_samples` | 成簇最小样本数 | 更严格，噪声更多 | 更宽松，簇更多 |

经验上，如果 embedding 已归一化，`eps=0.3` 可以作为起点，但绝不是通用真理。你真正应该关心的是三项输出指标：

| 指标 | 观察目的 |
|---|---|
| 簇数 | 是否压缩过度或压缩不足 |
| 簇内一致性 | 同一簇里的样本处理方式是否真的接近 |
| 噪声比例 | 是否把太多有用样本误判成噪声 |

下面给一个参数敏感性示意：

| `eps` | `min_samples` | 簇数 | 噪声比例 | 解释 |
|---:|---:|---:|---:|---|
| 0.20 | 4 | 95 | 28% | 太严，很多相近样本没聚到一起 |
| 0.30 | 4 | 52 | 11% | 较平衡，适合先试 |
| 0.40 | 4 | 21 | 4% | 太松，容易把不同场景混成大簇 |

如果你是第一次落地，最稳妥的办法不是盲目调参，而是人工抽查若干簇，看簇内样本是否满足下面两个条件：

1. 触发条件接近。
2. 处理步骤接近。

只要这两点不同时成立，该簇就不应该直接生成一条规则。

### 2. 为什么规则检索比原始记忆检索更快

假设当前查询向量为 $q$，规则集合为 $\{r_1,r_2,\dots,r_n\}$，原始记忆集合为 $\{e_1,e_2,\dots,e_m\}$，且 $n \ll m$。在线推理时先比较 $q$ 与规则的相似度：

$$
s(q,r_i)=\frac{q\cdot r_i}{\|q\|\,\|r_i\|}
$$

取 top-k 规则后，再决定是否展开对应簇中的少量原始记忆。这样做的收益不在于公式变了，而在于候选集规模显著下降。假设：

- 原始记忆数 $m=1000$
- 规则数 $n=50$
- 每次最终只需要查看 top-5 规则

那么粗检索阶段的比较规模从 1000 下降到 50，减少了 20 倍。对于 ANN 索引、缓存命中、rerank 成本、上下文长度来说，这都会进一步放大收益。

还可以把它写成一个更直观的两层检索过程：

$$
\text{Cost}_{raw} \approx O(m)
$$

$$
\text{Cost}_{distilled} \approx O(n) + O(k \cdot c)
$$

其中：

| 符号 | 含义 |
|---|---|
| $m$ | 原始记忆总数 |
| $n$ | 规则总数 |
| $k$ | 命中的规则数 |
| $c$ | 每条规则需要展开的少量原始证据数 |

当 $n \ll m$ 且 $k \cdot c$ 很小的时候，双层检索的总成本会明显低于原始记忆直接检索。

### 3. 玩具例子

假设系统里有 12 条原始记忆：

- 4 条关于“退款到账时间”
- 3 条关于“修改配送地址”
- 3 条关于“发票补开”
- 2 条关于“用户情绪激烈但需求不清晰”

聚类后得到 4 个簇，再分别总结成 4 条规则：

| 簇 | 原始主题 | 归纳后的规则 |
|---|---|---|
| C1 | 退款到账时间 | 先确认支付渠道，再给出对应到账时效 |
| C2 | 修改地址 | 若订单未出库，允许修改；已出库则转人工 |
| C3 | 发票补开 | 先校验订单完成状态，再确认发票类型 |
| C4 | 情绪激烈但需求不清 | 先复述诉求并澄清目标，再执行具体流程 |

当用户问“支付宝退款多久到”，系统不需要回看全部 4 条类似历史对话，只需先命中 C1 规则，再按需补一两条该簇中的原始记忆作为参考。

这个例子里要注意一个关键点：**规则不是替代事实，而是替代重复表达。**  
“支付宝退款多久到”与“微信退款多久到”在语言上不同，但在处理流程上接近，所以适合被压缩到同一条规则下。

### 4. 真实工程例子

一个多租户工单 Agent 每天处理几千次“账号解封、支付失败、发票纠错、工单升级”。如果直接把所有工单记录都作为长时记忆，随着时间增长会出现三类典型问题：

1. 同一个问题被数百条近似记录重复表示。
2. 检索会混入其他租户或旧版本流程。
3. LLM 每次都看到大量相似背景，浪费 token。

工程上更合理的做法通常是：

1. 按租户、产品线、记忆类型分库。
2. 每晚对新增记忆做局部聚类。
3. 对稳定高频簇生成规则。
4. 规则进入“热层索引”，原始记忆进入“冷层索引”。
5. 查询时先搜热层，再按需补冷层。

“热层”和“冷层”可以简单理解为：

| 层级 | 内容 | 查询优先级 | 作用 |
|---|---|---|---|
| 热层 | 蒸馏后的规则 | 高 | 快速指导当前决策 |
| 冷层 | 原始记忆与审计记录 | 低 | 提供证据、细节和回溯能力 |

这套结构的本质，是把“日志系统”改造成“知识系统”。前者擅长保存发生过什么，后者擅长回答下一步该怎么做。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖外部模型，只用手写二维向量模拟 embedding，用一个简化但完整可运行的 DBSCAN 演示“聚类后生成规则，再用相似度检索规则”的流程。

代码的目标不是复刻生产环境，而是把流程拆到新手可以逐步验证的程度。

```python
from math import sqrt
from collections import defaultdict


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def euclidean_distance(a, b):
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def region_query(points, idx, eps):
    center = points[idx]["embedding"]
    neighbors = []
    for j, point in enumerate(points):
        if euclidean_distance(center, point["embedding"]) <= eps:
            neighbors.append(j)
    return neighbors


def expand_cluster(points, labels, idx, neighbors, cluster_id, eps, min_samples):
    labels[idx] = cluster_id
    seed_queue = list(neighbors)
    seen = set(seed_queue)

    while seed_queue:
        current = seed_queue.pop(0)

        if labels[current] == -1:
            # 之前被当成噪声，但现在落入某个有效簇，应当并入该簇
            labels[current] = cluster_id

        if labels[current] is not None:
            continue

        labels[current] = cluster_id
        current_neighbors = region_query(points, current, eps)

        if len(current_neighbors) >= min_samples:
            for n in current_neighbors:
                if n not in seen:
                    seed_queue.append(n)
                    seen.add(n)


def dbscan(points, eps=0.3, min_samples=3):
    labels = [None] * len(points)
    cluster_id = 0

    for i in range(len(points)):
        if labels[i] is not None:
            continue

        neighbors = region_query(points, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        expand_cluster(
            points=points,
            labels=labels,
            idx=i,
            neighbors=neighbors,
            cluster_id=cluster_id,
            eps=eps,
            min_samples=min_samples,
        )
        cluster_id += 1

    return labels


def summarize_cluster(texts):
    joined = " ".join(texts)
    if any(keyword in joined for keyword in ["退款", "到账", "银行卡", "支付宝", "微信"]):
        return {
            "rule": "当用户询问退款到账时间时，先确认支付渠道，再说明不同渠道的预计到账时效。",
            "trigger": "退款时效咨询",
            "action": "确认渠道 -> 告知时效 -> 必要时补充异常说明",
        }
    if any(keyword in joined for keyword in ["地址", "收货地", "配送"]):
        return {
            "rule": "当用户要求修改配送地址时，先判断订单是否出库；未出库可修改，已出库则转人工或走拦截流程。",
            "trigger": "地址修改请求",
            "action": "确认出库状态 -> 可改则修改 -> 不可改则转人工",
        }
    if any(keyword in joined for keyword in ["发票", "专票", "补开"]):
        return {
            "rule": "当用户申请补开发票时，先校验订单完成状态，再确认发票类型和抬头信息。",
            "trigger": "发票补开请求",
            "action": "校验订单 -> 确认类型 -> 收集抬头信息",
        }
    return {
        "rule": "先识别用户意图，再执行对应标准流程。",
        "trigger": "通用场景",
        "action": "意图识别 -> 进入标准处理流程",
    }


def average_embedding(items):
    dim = len(items[0]["embedding"])
    sums = [0.0] * dim
    for item in items:
        for i, value in enumerate(item["embedding"]):
            sums[i] += value
    return [value / len(items) for value in sums]


def build_rule_store(records, eps=0.06, min_samples=2):
    labels = dbscan(records, eps=eps, min_samples=min_samples)

    clusters = defaultdict(list)
    for record, label in zip(records, labels):
        if label == -1:
            continue
        clusters[label].append(record)

    rules = []
    for cluster_id, cluster_records in clusters.items():
        summary = summarize_cluster([r["text"] for r in cluster_records])
        rules.append({
            "cluster_id": cluster_id,
            "rule": summary["rule"],
            "trigger": summary["trigger"],
            "action": summary["action"],
            "embedding": average_embedding(cluster_records),
            "evidence": [r["text"] for r in cluster_records],
        })

    return labels, rules


def search_rules(query_embedding, rules, top_k=3):
    scored = []
    for rule in rules:
        score = cosine_similarity(query_embedding, rule["embedding"])
        scored.append((score, rule))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


records = [
    {"text": "用户问支付宝退款多久到账", "embedding": [0.10, 0.20]},
    {"text": "用户咨询微信退款时间", "embedding": [0.12, 0.18]},
    {"text": "退款到银行卡需要几天", "embedding": [0.11, 0.21]},
    {"text": "订单没出库能否修改地址", "embedding": [1.00, 1.10]},
    {"text": "发货前修改配送地址", "embedding": [0.98, 1.08]},
    {"text": "包裹未出库如何改收货地", "embedding": [1.02, 1.12]},
    {"text": "订单完成后怎么补开发票", "embedding": [2.00, 2.10]},
    {"text": "能否补开专票", "embedding": [1.98, 2.08]},
]

labels, rules = build_rule_store(records, eps=0.06, min_samples=2)

refund_query = [0.09, 0.19]
refund_results = search_rules(refund_query, rules, top_k=2)

address_query = [1.01, 1.09]
address_results = search_rules(address_query, rules, top_k=2)

assert len(rules) == 3
assert labels.count(-1) == 0

best_refund_rule = refund_results[0][1]["rule"]
best_address_rule = address_results[0][1]["rule"]

assert "退款" in best_refund_rule
assert "地址" in best_address_rule

for score, rule in refund_results:
    print(f"[refund] score={score:.4f} rule={rule['rule']}")

for score, rule in address_results:
    print(f"[address] score={score:.4f} rule={rule['rule']}")
```

这段代码包含了完整的最小闭环：

1. 准备原始记忆及其 embedding。
2. 运行 DBSCAN 聚类。
3. 对每个簇生成规则摘要。
4. 为规则生成中心向量。
5. 用查询向量检索规则。
6. 返回最匹配的规则。

如果你要自己跑，直接保存为 `distill_demo.py` 并执行：

```bash
python3 distill_demo.py
```

你应该看到两类查询分别命中退款规则和地址修改规则。

为了帮助新手理解，下面把示例中的数据结构也展开说明：

| 字段 | 含义 |
|---|---|
| `text` | 原始记忆文本 |
| `embedding` | 文本对应的二维向量，真实系统中通常是高维向量 |
| `labels` | DBSCAN 输出的簇编号，`-1` 表示噪声 |
| `rules` | 蒸馏后的规则集合 |
| `evidence` | 与规则绑定的原始样本，用于审计和补充上下文 |

上面的代码省略了真实系统中的三部分：

1. embedding 模型，用于把文本转成高维向量。
2. LLM 摘要器，用于把一个簇归纳成规则。
3. 向量索引库，用于高效搜索。

如果换成接近生产环境的伪代码，结构通常是这样：

```python
# 1. 为新增记忆生成 embedding
memory_embeddings = embedder.encode(memory_texts)

# 2. 做聚类
labels = DBSCAN(eps=0.3, min_samples=4, metric="cosine").fit_predict(memory_embeddings)

# 3. 每个簇调用 LLM 归纳规则
rules = []
for cluster_id in unique_non_noise(labels):
    cluster_text = collect_cluster_text(memory_texts, labels, cluster_id)
    rule_text = llm.summarize(
        "请从以下相似经验中归纳一条可复用处理规则，只保留稳定步骤和触发条件；"
        "不要保留一次性事实、个别异常或具体用户信息。\n\n"
        + cluster_text
    )
    rules.append(rule_text)

# 4. 为规则生成 embedding 并入向量库
rule_embeddings = embedder.encode(rules)
rule_index.add(rule_embeddings)

# 5. 在线检索
q = embedder.encode([query])[0]
top_rules = rule_index.search(q, top_k=5)
reranked = rerank_with_cross_encoder(query, top_rules)

# 6. 必要时补充对应簇中的少量原始记忆
final_context = inject_rules_and_evidence(reranked, raw_memory_store)
```

模块映射关系通常如下：

| 模块 | 作用 | 常见依赖 |
|---|---|---|
| 聚类 | 找到相似经验簇 | `sklearn.cluster.DBSCAN` |
| 摘要 | 把簇归纳成规则 | LLM API |
| 向量化 | 生成 embedding | OpenAI embedding / bge / e5 |
| 索引 | 近似最近邻搜索 | FAISS / Milvus / pgvector |
| 精排 | 提高规则命中质量 | cross-encoder / 二次 LLM 判断 |

查询流程可以概括为：

$$
query \rightarrow embedding \rightarrow top\text{-}k\ rules \rightarrow rerank \rightarrow inject
$$

其中 inject 指“把选中的规则注入当前上下文”。

如果写成更完整的上下文构造公式，可以近似表示为：

$$
C = \{r_1,\dots,r_k\} \cup \{e_{1},\dots,e_{c}\}
$$

其中：

- $\{r_1,\dots,r_k\}$ 是召回并精排后的规则
- $\{e_{1},\dots,e_{c}\}$ 是这些规则对应的少量原始证据
- $C$ 是最终送入推理模型的上下文

这说明规则蒸馏并不是“只看规则”，而是“先看规则，再按需补证据”。

---

## 工程权衡与常见坑

规则蒸馏最大的风险，不是压缩率不够，而是把不该合并的经验错误合并。规则一旦生成，就会影响后续大量推理，因此必须把重点放在“规则的可靠性”而不是“规则的数量”。

下表列出常见问题：

| 常见坑 | 结果 | 规避方法 |
|---|---|---|
| 未按用户分层 | 检索到别的用户历史 | 按 `user_id + memory_type` 分库或分 namespace |
| 未设 TTL | 旧流程污染新流程 | 对事实型记忆设置过期时间 |
| 写入不去重 | 重复样本越来越多 | 写入前先做相似度去重 |
| 聚类参数过松 | 不同场景被混成一条规则 | 调小 `eps`，增加簇内一致性检查 |
| 聚类参数过严 | 规则数量过多，压缩失败 | 调大 `eps` 或降低 `min_samples` |
| 只存规则不存证据 | 无法审计和回溯 | 保留规则到原始簇的映射 |
| 每次全量重聚类 | 成本和延迟不可控 | 改成增量聚类或周期性批处理 |

### 写入侧的成本控制

新手最容易踩的坑，是把“规则蒸馏”做成“每新增一条记忆，就把全部历史重新聚类一遍”。这样做在样本量小时似乎没问题，但一旦历史超过 1000 条，成本会迅速失控：

- embedding 成本增加
- 聚类时间增加
- LLM 摘要调用次数增加
- 规则重建导致索引频繁更新

更合理的思路是先限制候选集规模：

$$
batch\_size=\min(\text{new\_records}, MAX\_LOADED)
$$

如果每天新增 5 条交互，不应该每天重新读取全部 1000 条旧记忆，而应该采用下面的写入策略：

1. 先把新增记录和现有规则做相似度比较。
2. 与已有规则高度相似的，直接挂到已有簇。
3. 只有新场景或低相似条目，才进入待聚类队列。
4. 到达批量阈值后，再统一蒸馏。

一个实用的写入判定表如下：

| 条件 | 处理方式 |
|---|---|
| 与现有规则高相似 | 直接并入对应簇 |
| 与现有规则中等相似 | 进入待人工抽查或待二次判断 |
| 与现有规则低相似 | 放入待聚类缓冲区 |
| 明显是临时事实 | 进入 TTL 事实层，不参与规则生成 |

### 规则质量的评估

规则不是“总结得像不像”，而是“召回后是否真的有用”。因此评估时至少要看三类指标：

| 指标 | 关注点 | 典型问题 |
|---|---|---|
| 规则命中率 | 当前问题是否能召回正确规则 | 规则表述过于抽象 |
| 任务成功率 | 引入规则后任务是否更稳定 | 规则丢失关键条件 |
| 压缩收益 | token、延迟、索引规模是否下降 | 规则层过大，收益不足 |

进一步说，可以把线上评估拆成四个维度：

| 维度 | 说明 |
|---|---|
| Recall@k | 正确规则是否出现在前 k 个召回结果中 |
| Success Rate | 实际任务是否完成 |
| Avg Context Tokens | 单次推理平均上下文长度 |
| P95 Latency | 高分位延迟是否明显下降 |

如果规则让上下文变短了，但成功率明显下降，说明规则过于泛化，决策条件被压缩掉了。如果规则召回率很高，但延迟没有改善，说明你可能只是把原始记忆换了名字，并没有真正减少候选规模。

### 真实工程例子

在一个售后工单系统里，原始记忆中既有“退款规则”，也有“某地区银行通道异常”的临时事实。如果把这两类内容放在同一层并一起蒸馏，系统可能错误地产生一句永久规则：

> 某银行退款总是慢。

这显然是错误的，因为它只是某个时间窗口内的临时事件，而不是稳定规律。这个例子说明：**蒸馏前的分层，比蒸馏算法本身更重要。**

因此工程上通常要把记忆拆成三类：

| 类型 | 是否参与规则蒸馏 | 生命周期 |
|---|---|---|
| 规则型知识 | 是 | 中长期保留 |
| 事实型知识 | 通常否 | 有 TTL，按时间过期 |
| 审计型记录 | 否 | 长期保存，仅供回溯 |

如果这一步没有做好，后续无论用 DBSCAN、HDBSCAN 还是其他聚类算法，最终生成的规则都可能不可靠。

---

## 替代方案与适用边界

规则蒸馏不是唯一答案，只是“高重复经验场景”里性价比较高的一种答案。

最直接的替代方案，是继续使用原始 episodic retrieval。它适合以下情况：

- 记忆量小
- 每条案例都很独特
- 任务强调高保真
- 证据细节比抽象规则更重要

优点是信息损失少，缺点是检索成本、上下文成本和噪声会随着历史规模近似线性上升。

另一类方案是结构化记忆，例如 PlugMem 的 fact-skill graph。这里的 fact 指“事实节点”，skill 指“技能节点”。这种方法不是把经验压缩成一句自然语言规则，而是把经验拆成“已知事实”和“可执行技能”，再通过图结构组合。

三种方案可以对比为：

| 方案 | 上下文开销 | 响应速度 | 维护成本 | 适用场景 |
|---|---|---|---|---|
| Rule Distillation | 低 | 快 | 中 | 高重复、长周期任务 |
| Bare Episodic Retrieval | 高 | 慢 | 低 | 小规模、高保真任务 |
| PlugMem fact-skill graph | 很低到中 | 快 | 高 | 复杂工作流、多类型知识融合 |

如果从“系统复杂度”角度看，它们分别解决不同问题：

| 方案 | 主要解决的问题 |
|---|---|
| 原始记忆检索 | 如何尽量不丢失历史细节 |
| 规则蒸馏 | 如何把重复经验压缩成可复用策略 |
| fact-skill graph | 如何把事实、动作和关系显式结构化 |

一个实用判断规则如下：

- 当记忆少于几十条时，直接检索原始记忆通常更简单。
- 当记忆超过数百条且重复明显时，规则蒸馏通常性价比最高。
- 当任务不仅需要“回忆”，还需要“拆分事实与技能”时，图结构更合适。

规则蒸馏也可以与 PlugMem 结合。组合方式通常是：

1. 先检索事实节点和技能节点。
2. 再检索蒸馏规则作为高层策略。
3. 最后由 reasoning 层把事实、技能、规则组合成当前建议。

伪流程如下：

```python
facts = fact_index.search(query, top_k=5)
skills = skill_index.search(query, top_k=3)
rules = rule_index.search(query, top_k=3)

plan = reasoning_llm.compose(
    query=query,
    facts=facts,
    skills=skills,
    rules=rules,
)
```

这种做法适合复杂 Agent，因为三层内容解决的是三个不同问题：

| 层 | 负责内容 | 回答的问题 |
|---|---|---|
| facts | 当前状态与客观约束 | 现在已知什么 |
| skills | 可执行动作 | 系统能做什么 |
| rules | 经验策略 | 在这种场景下通常该怎么做 |

如果把三者混成一个统一文本层，系统通常会更难控制，也更难评估。

因此，规则蒸馏的适用边界可以再压缩成一句话：**它适合处理“重复模式”，不适合替代“关键事实”。**

---

## 参考资料

下表不是简单列名，而是按“阅读目的”整理的使用方式：

| 资料 | 作用 |
|---|---|
| ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory | 用于理解“从经验轨迹中提炼推理记忆”为什么能提升任务成功率并减少无效步骤 |
| Microsoft PlugMem 研究博客 | 用于理解把原始交互拆成 fact + skill 结构后，系统为何更容易复用和组合知识 |
| `agentmemory` 的 DBSCAN 相关文档 | 用于理解 `eps`、`min_samples`、噪声点处理和增量聚类的工程含义 |
| GeeksforGeeks 的 DBSCAN 教程 | 用于快速复习密度聚类概念、参数直觉和基础公式 |
| Emergent Mind 关于 memory agent 的综述 | 用于理解向量检索、余弦相似度、分层记忆与 Agent memory 的整体结构 |
| Agentic AI / Memory Management 工程实践文章 | 用于补充 TTL、租户隔离、写入去重、冷热分层与成本控制等部署问题 |

如果按阅读顺序组织，建议这样使用：

1. 先看 DBSCAN 资料，建立“为什么能把重复样本聚成簇”的直觉。
2. 再看 memory agent 综述，理解规则层在整体记忆架构中的位置。
3. 最后看 PlugMem 与工程实践材料，理解结构化记忆和生产部署中的差异。

对于第一次实现这类系统的人，最重要的不是一开始追求最复杂的架构，而是先回答下面四个问题：

| 问题 | 目的 |
|---|---|
| 哪些记忆是高重复的 | 判断是否值得蒸馏 |
| 哪些内容是临时事实 | 避免把临时现象蒸馏成永久规则 |
| 规则召回后是否真的提升成功率 | 防止“压缩有效，效果无效” |
| 原始证据是否仍可追溯 | 保证系统可审计、可复盘 |

最终可以把全文再收束成一句更工程化的结论：

**记忆蒸馏的核心不是“总结历史”，而是把高重复经验转成低成本、可检索、可复用的行为规则，同时保留必要证据，避免把日志层误当知识层。**
