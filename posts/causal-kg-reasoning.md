## 核心结论

传统知识图谱（Knowledge Graph, KG，白话说就是“把实体和关系存成图的数据结构”）擅长表示“谁和谁有关”，但默认只回答相关性问题，不直接回答“如果我强行改变一个因素，结果会不会变”。因果推理（Causal Inference，白话说就是“研究改变原因后结果如何变化的方法”）补上了这一层：它把图谱中的事实关系进一步组织成因果图和结构方程系统，让系统不仅能做关联式推断，还能做干预式推断和反事实解释。

最核心的变化只有两点。第一，查询目标从“观察到 $X$ 时 $Y$ 多大概率发生”变成“把 $X$ 强制设成某个值后，$Y$ 会怎样”，即：

$$
P(Y \mid do(X=x))
$$

第二，在已经观察到现实结果后，还能追问“如果当时不是这个决策，会怎样”，即反事实查询：

$$
P(Y_{X=x'} \mid X=x, Y=y)
$$

这两种能力对工程系统很重要。决策支持场景里，系统需要知道“采取动作”是否真会改善结果，而不是只看历史上它们经常同时出现。公平性分析场景里，系统需要回答“如果把敏感属性或代理变量改变，结论是否会变化”，否则很难判断模型是否依赖了不合理路径。

可以把整体流程压缩成一条线：

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| KG 表示 | 实体、关系、属性 | 构建图结构 | 相关性知识 |
| 因果化 | 补充方向、混淆变量、结构方程 | 定义谁影响谁 | 因果图 |
| 干预推断 | 指定 `do(X=x)` | 替换结构方程并传播 | 干预效果 |
| 反事实解释 | 观测事实 `(X=x, Y=y)` | 回溯个体噪声并改写原因 | “如果没做 X” 的结果 |

如果只记一句话：知识图谱解决“世界里有哪些连接”，因果推理解决“改动一个连接背后的原因后，世界会怎样变”。

---

## 问题定义与边界

先定义问题。知识图谱中的三元组（triplet，白话说就是“头实体-关系-尾实体”的一条事实）例如“用户A-点击-商品B”“药物X-治疗-疾病Y”，本身只说明事实或统计模式。即使某条路径频繁出现，也不能推出它具有因果作用。

一个典型错误是把共现当成因果。假设图中有变量 $A$ 表示“是否接受某种干预”，$B$ 表示“是否成功”。如果只看到：

$$
P(B=1)=0.3
$$

这只能说明总体成功率是 0.3，不能说明干预 $A$ 是否有效。因为可能存在混淆变量（Confounder，白话说就是“同时影响原因和结果的第三个因素”）$C$，例如用户基础能力、疾病严重程度、地区资源水平。此时观察数据中的 $A$ 和 $B$ 相关，可能只是因为 $C$ 同时影响了两者。

如果进一步得到：

$$
P(B=1 \mid do(A=1))=0.6
$$

含义就变了。`do(A=1)` 不是“看到 A=1 的样本”，而是“把 A 强制设为 1”。这相当于切断进入 $A$ 的上游原因，只保留 $A$ 对下游的影响。因此从 0.3 到 0.6 的变化才可被解释为干预效应。

再往前一步，反事实问题不是问群体平均效果，而是问单个样本：某个用户这次成功了，如果当时没给他干预，会不会仍然成功？这类问题形式上写作：

$$
P(B_{A=0}=1 \mid A=1, B=1)
$$

这里的 $B_{A=0}$ 表示“在把 $A$ 设为 0 的那个假想世界里，$B$ 的值”。

下面给一个最小玩具例子。

设：
- $A$：是否参加训练营
- $B$：是否通过面试
- $C$：原始基础水平

如果训练营通常只录取基础好的人，那么观察数据里“参加训练营的人更容易通过面试”并不能证明训练营本身有效，因为基础水平 $C$ 同时影响了 $A$ 和 $B$。

| 估计方式 | 使用的信息 | 结论 | 是否能解释因果 |
|---|---|---|---|
| 相关性估计 | $P(B \mid A)$ | 参加训练营的人通过率更高 | 不能 |
| 调整混淆后估计 | $P(B \mid A, C)$ | 在相同基础水平下比较 | 部分可以 |
| 干预估计 | $P(B \mid do(A))$ | 强制参加训练营后通过率变化 | 可以 |
| 反事实估计 | $P(B_{A=0} \mid A=1, B=1, C)$ | 某个已通过的人若未参加会怎样 | 可以解释个体 |

这类方法也有明确边界，不是“把 KG 加上几条边就自动因果化”。

| 边界条件 | 含义 | 做不到会怎样 |
|---|---|---|
| 因果方向合理 | 需要知道谁先影响谁 | 公式可写，结论不可信 |
| 混淆变量尽量完备 | 至少封闭关键 back-door 路径 | 干预估计偏移 |
| 数据支持结构学习或专家知识 | 不能只靠直觉连图 | 图结构可能错 |
| 反事实需个体层信息 | 需要更细粒度状态或噪声建模 | 只能做平均效应，难做个体解释 |

back-door（白话说就是“从原因后门绕进来的混淆路径”）如果没有被可观测变量封闭，`do` 的结果就会混入伪影响。这是因果 KG 落地时最常见的失败点。

---

## 核心机制与推导

因果化 KG 的核心不是“把图画得更复杂”，而是给图上的节点配上结构方程（Structural Equation，白话说就是“每个变量由哪些上游变量决定的公式”）。例如在一个简化图中：

$$
A \rightarrow B \rightarrow C
$$

可以写成：

$$
A := U_A
$$

$$
B := f_B(A, U_B)
$$

$$
C := f_C(B, U_C)
$$

其中 $U_A,U_B,U_C$ 是外生噪声（白话说就是“系统外部带来的个体差异”）。

### 1. 干预如何发生

当我们执行 `do(A=1)` 时，不是简单地在条件概率里加一个条件，而是把 $A$ 的方程替换掉：

$$
A := 1
$$

然后重新传播到下游：

$$
B := f_B(1, U_B), \quad C := f_C(B, U_C)
$$

这就是为什么 $P(C \mid do(A=1))$ 与 $P(C \mid A=1)$ 一般不同。后者仍然保留“谁更可能取到 $A=1$”的信息，前者则把这个过程切掉了。

### 2. do-演算为什么有用

do-演算（do-calculus，白话说就是“把干预查询变形为可由数据估计的规则系统”）解决的问题是：现实中我们通常没有大规模随机试验，只能用观测数据。于是需要把含 `do()` 的表达式，在满足图条件时转成不含 `do()` 的形式。

三条规则可概括为：

1. 插入/删除观测  
如果图条件满足，某些观测变量可以加入或移除，不改变干预查询。

2. 动作与观测互换  
如果图结构允许，可以把 `do(Z)` 换成条件 `Z`，或反过来。

3. 插入/删除动作  
如果某个动作对目标在图上已经“阻断”，可以删掉该动作。

对初学者来说，不需要先背完整图论条件，先记住目的：把难估计的 $P(Y \mid do(X))$ 变成容易从样本频率估计的东西。

### 3. 在链式图上的推导

继续看链：

$$
A \rightarrow B \rightarrow C
$$

因为 $A$ 对 $C$ 的影响完全通过 $B$ 传递，且没有额外混淆，干预后有：

$$
P(C \mid do(A=a)) = \sum_b P(C \mid b) P(b \mid do(A=a))
$$

如果 $B$ 与 $A$ 之间也没有混淆，则：

$$
P(b \mid do(A=a)) = P(b \mid A=a)
$$

于是：

$$
P(C \mid do(A=a)) = \sum_b P(C \mid b) P(b \mid A=a)
$$

这就是“先强制设定 A，再看 B 如何响应，最后看 C 如何随 B 改变”。

### 4. 从 KG 到因果推理链

可以把可追踪流程写成下面这样：

| 步骤 | 目标 | 结果 |
|---|---|---|
| Traceability | 从 KG 抽取候选路径 | 找出可能影响目标的关系链 |
| Intervention | 对关键节点施加 `do` | 判断路径是否仍有效 |
| Prediction | 计算下游分布变化 | 得到干预效果 |
| Counterfactual | 回到具体样本改写局部世界 | 生成“如果当时没这样做”的解释 |

### 5. 玩具例子

图谱中有三类节点：
- `Student`
- `Training`
- `Offer`

关系有：
- `attend_training`
- `has_skill`
- `gets_offer`

如果只看图谱统计，会发现参加训练的学生更容易拿到 offer。但加入因果图后，我们还要补上 `has_skill` 对 `attend_training` 和 `gets_offer` 的共同影响。这样就能区分两种说法：

- 相关性说法：参加训练的人拿到 offer 的概率更高。
- 因果说法：在控制技能基础后，强制安排训练，offer 概率是否仍然提升。

前者是模式识别，后者才是决策依据。

### 6. 真实工程例子

在推荐或风控系统里，知识图谱常用于路径推理，例如：

`用户 -> 浏览 -> 品类 -> 品牌 -> 商品`

若系统发现“某地区用户更常点击某类高风险借贷产品”，纯 KG 或纯路径打分模型可能把“地区”相关路径权重提得很高。但“地区”常是收入、教育、营销投放强度等因素的代理变量。此时如果直接把路径权重当证据，模型可能学到不公平甚至违规的决策逻辑。

因果化后要做两件事：
- 用因果图区分“真实业务因果链”与“由混淆变量带出的伪路径”。
- 对敏感属性或代理变量做干预/反事实分析，检查推荐或风控结果是否显著变化。

这时系统输出的不再只是“这条路径分高”，而是“在固定用户真实还款能力后，去掉地区路径，风险评分是否仍保持”。这才是工程上可审计的解释。

---

## 代码实现

下面给一个最小可运行示例。目标不是复现完整论文系统，而是展示“相关性估计”和“干预估计”为什么不同，以及如何把“因果先验”用于图路径打分。

```python
from collections import defaultdict

# 玩具数据：
# skill: 基础能力（混淆变量）
# train: 是否参加训练
# offer: 是否获得 offer
data = [
    {"skill": 1, "train": 1, "offer": 1},
    {"skill": 1, "train": 1, "offer": 1},
    {"skill": 1, "train": 0, "offer": 1},
    {"skill": 1, "train": 0, "offer": 0},
    {"skill": 0, "train": 1, "offer": 1},
    {"skill": 0, "train": 1, "offer": 0},
    {"skill": 0, "train": 0, "offer": 0},
    {"skill": 0, "train": 0, "offer": 0},
]

def prob_offer_given_train(rows, train_value):
    subset = [r for r in rows if r["train"] == train_value]
    return sum(r["offer"] for r in subset) / len(subset)

def prob_skill(rows, skill_value):
    subset = [r for r in rows if r["skill"] == skill_value]
    return len(subset) / len(rows)

def prob_offer_given_train_and_skill(rows, train_value, skill_value):
    subset = [r for r in rows if r["train"] == train_value and r["skill"] == skill_value]
    return sum(r["offer"] for r in subset) / len(subset)

# back-door adjustment:
# P(Y|do(X)) = sum_z P(Y|X,z) P(z)
def prob_offer_do_train(rows, train_value):
    total = 0.0
    for skill_value in [0, 1]:
        total += (
            prob_offer_given_train_and_skill(rows, train_value, skill_value)
            * prob_skill(rows, skill_value)
        )
    return total

obs = prob_offer_given_train(data, 1)
dov = prob_offer_do_train(data, 1)

# 观察概率与干预概率一般可以不同
assert round(obs, 4) == 0.75
assert round(dov, 4) == 0.625

# 一个简单的“因果先验路径权重”示意
paths = {
    ("user", "region", "loan"): 0.20,   # 容易受混淆影响
    ("user", "income", "loan"): 0.70,   # 更接近真实因果链
    ("user", "repay_hist", "loan"): 0.90,
}

counterfactual_penalty = {
    ("user", "region", "loan"): 0.8,    # 反事实下贡献不稳定，强惩罚
    ("user", "income", "loan"): 0.1,
    ("user", "repay_hist", "loan"): 0.05,
}

def causal_weight(base_weight, penalty):
    return base_weight * (1 - penalty)

weighted = {
    path: causal_weight(paths[path], counterfactual_penalty[path])
    for path in paths
}

assert weighted[("user", "region", "loan")] < weighted[("user", "income", "loan")]
assert weighted[("user", "repay_hist", "loan")] > weighted[("user", "income", "loan")]

print("P(offer=1 | train=1) =", obs)
print("P(offer=1 | do(train=1)) =", dov)
print("causal path weights =", weighted)
```

这段代码展示了两件事。

第一，back-door adjustment（后门调整，白话说就是“把混淆变量按总体分布重新加权”）能把观测相关性变成干预估计。上例中：

- $P(\text{offer}=1 \mid \text{train}=1)=0.75$
- $P(\text{offer}=1 \mid do(\text{train}=1))=0.625$

观测值更高，是因为高能力人群本来就更容易参加训练，也更容易拿 offer。

第二，可以把反事实稳定性转成路径先验，修正图推理策略。真实系统里常见的结构大致如下：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Prior Table 构建 | 关系路径、历史样本 | relation weight / entity weight | 给路径一个因果先验 |
| Policy Network | 当前节点、候选边、prior | 下一跳动作概率 | 控制图上搜索 |
| Counterfactual Estimator | 路径、干预设置 | 路径在反事实世界的贡献变化 | 判断路径是否稳定 |
| RL Loop | 奖励信号、轨迹 | 参数更新 | 学出更可靠的推理策略 |

一个简化伪代码如下：

```python
for episode in episodes:
    state = query_head
    path = []

    while not stop(state):
        candidates = extract_edges(state)
        scored = []

        for edge in candidates:
            prior = prior_table.get(edge.relation, 0.0)
            cf_weight = counterfactual_weight(edge, query_context)
            score = policy_network(state, edge) * combine(prior, cf_weight)
            scored.append((edge, score))

        action = sample(scored)
        state = action.next_entity
        path.append(action)

    reward = evaluate(path, query_tail)
    update_policy(path, reward)
    update_prior(path, reward)
```

这里的 `prior_table` 可以理解为“先给每类关系一个可信度底分”，`counterfactual_weight` 则回答“如果把关键因素改掉，这条路径是否还站得住”。两者结合后，策略网络不再只追逐历史上最常见的路径，而更偏向选择因果上稳定的路径。

真实工程里，这种设计特别适合两个场景：

- 决策支持：例如医疗问答、风控审批、策略推荐，需要解释“为什么建议这个动作”。
- 公平性分析：例如招聘、贷款、推荐，需要评估某个属性或代理变量是否通过不合理路径影响输出。

---

## 工程权衡与常见坑

第一类坑是图结构错。很多团队把现有 KG 路径直接当因果路径，这通常不成立。KG 中的边可能来自知识库抽取、日志共现、规则归纳，并不自动表示“前者导致后者”。如果方向弄反，后续所有 `do()` 查询都只是形式正确，结论错误。

第二类坑是未观测混淆变量。即使你知道要做后门调整，也不代表能做对。只有关键混淆变量被观测并纳入图中，back-door 才能被真正封闭。否则系统会产生很有说服力但实际偏移的解释。

第三类坑是把反事实当成“随便改一个字段重新跑模型”。真正的反事实不是简单替换输入值，而是要保持个体其余条件和因果机制一致。比如在招聘系统里，把“毕业院校”改掉后，不能把其对技能、项目经历、推荐关系的潜在影响全都忽略，否则得到的是不自洽样本。

第四类坑是规则学习只做共现压缩，不做干预重估。很多 rule learner 会得出类似：

`地区 -> 渠道偏好 -> 贷款通过率`

这种规则在观测数据上可能拟合很好，但如果其中包含敏感代理路径，工程上会产生严重公平性风险。引入干预后，规则应被重新评估：当固定真实偿付能力并删除地区影响时，这条规则是否仍有贡献？

下面用表格压缩这些风险。

| 风险点 | 只做相关性时的后果 | 引入因果后的改进 |
|---|---|---|
| 把共现路径当因果链 | 学到伪规则 | 通过结构方程和干预检查路径有效性 |
| 忽略混淆变量 | 干预效果高估或低估 | 用后门调整控制偏移 |
| 不做反事实 | 难解释个体决策 | 可回答“如果没做 X 会怎样” |
| 不审查敏感代理路径 | 公平性风险高 | 可对敏感属性做干预/反事实分析 |

工程检查清单可以很短，但必须真的执行：

| 检查项 | 要问的问题 |
|---|---|
| 因果图完整性 | 关键变量和方向是否有业务依据或结构学习证据？ |
| 混淆变量可观测性 | 重要 back-door 路径是否已封闭？ |
| 干预定义 | `do(X)` 改写的是变量本身，还是只改了输入特征？ |
| 反事实一致性 | 改写后的样本是否仍符合因果结构？ |
| 公平性审计 | 敏感属性及代理变量路径是否被单独评估？ |

真实数据集如 NELL、FB15k 一类 KG 常见问题是：路径分高不代表路径真有效。因为高频关系往往混有采样偏差、流行实体偏差、语义近邻偏差。因果先验和反事实重估的价值就在这里，它不是让模型“更聪明”，而是让模型少学一些不该学的捷径。

---

## 替代方案与适用边界

并不是所有 KG 推理问题都必须上因果层。若目标只是链接预测（Link Prediction，白话说就是“补全图中缺失边”）且不关心决策解释、干预效果、公平性，那么纯 embedding、图神经网络、path ranking 往往更直接，训练成本也更低。

但一旦问题变成下面这些形式，因果方法就开始有必要：
- “如果我采取这个动作，结果会改善多少？”
- “这个推荐为什么成立，是因为真实偏好还是代理变量？”
- “如果去掉敏感属性的影响，结论还成立吗？”

可以用一张表看适用边界。

| 方案 | 数据需求 | 可解释性 | 能否回答干预问题 | 适合场景 |
|---|---|---|---|---|
| 纯 KG embedding | 中到高 | 低 | 否 | 大规模补全、召回 |
| Path ranking / RL reasoning | 中等 | 中 | 弱 | 路径可解释推理 |
| 规则学习 | 中等 | 高 | 弱到中 | 人类可读规则抽取 |
| CausalRL / 因果规则学习 | 高 | 高 | 是 | 决策支持、公平性分析、策略优化 |

再看两个对比。

### 1. Standard RL reasoning vs. CausalRL + prior

前者是在图上学“哪条路径更容易命中答案”；后者是在这个基础上再问“这条路径在干预或反事实世界里是否仍稳定”。如果 KG 稀疏、样本少、偏差重，因果先验通常更稳，因为它会压低那些只在历史数据里碰巧高频的边。

### 2. Rule learning without intervention vs. with do

不带干预的规则学习，更像在总结“历史上常见模式”；带 `do()` 的规则学习，才是在判断“如果主动改变前件，后件是否真的跟着变”。前者适合知识整理，后者适合策略制定和合规审计。

因果 KG 的适用边界也很明确：
- 如果拿不到关键混淆变量，只能得到弱结论。
- 如果图结构缺乏业务依据，因果解释会变成伪精确。
- 如果业务不需要干预和公平性分析，上因果层可能性价比不高。
- 如果目标是高风险决策系统，因果层往往不是可选项，而是必要约束。

---

## 参考资料

- “Causal Reinforcement Learning for Knowledge Graph Reasoning”, *Applied Sciences*, 2024. 讨论如何用 counterfactual weight 与 prior table 引导 KG 上的强化学习推理。  
  https://www.mdpi.com/2076-3417/14/6/2498

- “Rule learning with causal intervention for knowledge graph reasoning”, *Applied Soft Computing*, 2026. 讨论在知识图谱规则学习中引入显式干预，以降低混淆影响并提升可解释性。  
  https://www.sciencedirect.com/science/article/abs/pii/S1568494626002504

- “Explainable AI and Causal Understanding: Counterfactual Approaches Considered”, *Minds and Machines*, 2023. 讨论可解释 AI 中的 do-演算与反事实理解框架。  
  https://link.springer.com/article/10.1007/s11023-023-09637-x
