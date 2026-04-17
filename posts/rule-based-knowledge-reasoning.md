## 核心结论

规则推理的目标，是把知识图谱里的离散事实，归纳成可以重复使用的一阶逻辑规则。这里的“一阶逻辑”可以先理解成“用变量表示实体、用关系表示连接方式的形式化语言”。最典型的例子是：

$$
grandparent(X, Z) \leftarrow parent(X, Y) \land parent(Y, Z)
$$

它的含义很直接：如果 `X` 是 `Y` 的父母，`Y` 是 `Z` 的父母，那么 `X` 就是 `Z` 的祖父母。这个规则的价值不在于重写常识，而在于它能把大量三元组压缩成一个可复用模式，用来做事实补全、错误检测和知识清洗。

规则方法的核心优势是可解释性。可解释性就是“模型为什么得出这个结论，人可以直接看懂”。当系统预测 `grandparent(A, C)` 时，它不仅给结果，还能给出证据链：`parent(A, B)` 与 `parent(B, C)`。这和只输出一个分数的黑盒嵌入模型不同。

目前常见方法可以分成两类：

| 方法 | 代表 | 搜索方式 | 训练方式 | 主要优点 | 主要缺点 |
|---|---|---|---|---|---|
| 离散规则挖掘 | AMIE / AMIE+ / AMIE 3 | 显式枚举规则体并剪枝 | 非梯度式统计搜索 | 规则清晰、解释强、适合大规模稀疏 KG | 搜索空间大，覆盖率常偏低 |
| 可微规则学习 | Neural LP / DRUM | 把关系组合变成连续权重 | 梯度训练 | 能和神经模型联合优化，端到端学习 | 解释性比纯离散规则弱，训练更复杂 |

玩具例子先看最小场景。已知：

- `parent(张三, 李四)`
- `parent(李四, 王五)`

那么按上面的规则，可以推出：

- `grandparent(张三, 王五)`

这就是规则推理最核心的工作模式：把局部事实链，提升为可复用推理模板。

真实工程里，规则挖掘常用于 DBpedia、YAGO、Wikidata 这类大知识库。比如从大规模人物和地点三元组中，可能挖出类似“配偶关系 + 居住地相同”的规则，用于发现缺失属性或可疑冲突数据。规则不是万能答案，但在“需要证据链”的场景里非常有价值。

---

## 问题定义与边界

知识图谱里的基本数据单位是三元组 `(head, relation, tail)`，可以理解成“头实体、关系、尾实体”。例如 `(Alice, parent, Bob)` 表示 Alice 是 Bob 的父母。

规则挖掘的问题可以写成：

- 输入：一组已有三元组
- 输出：若干条 Horn 规则及其统计质量分数

“Horn 规则”可以先理解成“规则头只有一个结论的逻辑规则”。一般写成：

$$
H \leftarrow B_1 \land B_2 \land \dots \land B_n
$$

其中 `H` 是要预测的关系，`B_i` 是条件关系。

在知识图谱里，这个任务并不是“把所有关系随便拼起来”。它有明确边界，否则组合数会爆炸，规则会大多无意义。常见边界如下：

| 项目 | 定义 | 例子 |
|---|---|---|
| 输入 | 三元组集合 | `parent(A,B)`、`parent(B,C)` |
| 输出 | Horn 规则 + 置信度 | `grandparent(X,Z) <- parent(X,Y) & parent(Y,Z)` |
| 闭合规则 | 规则体中的变量要和头部连通 | 避免出现无关变量 `W` |
| 支持度阈值 | 规则至少命中若干实例 | 少于 2 次通常可视为偶然 |
| 覆盖率阈值 | 规则解释头关系的比例 | 只覆盖极少样本的规则价值有限 |
| 置信度阈值 | 规则成立的可靠程度 | 太低说明噪声或巧合太多 |

新手最容易忽略的一点，是开放世界假设。开放世界假设就是“知识库里没写，不等于事实不存在”。比如 KB 中没有 `grandparent(A,C)`，不能简单认为它为假，因为它可能只是没录入。这也是为什么经典规则挖掘常用 PCA confidence，而不是普通分类里的精确率。

一个新手版任务可以这样描述：给定若干 `parent` 三元组，只允许规则体里有两个关系，问能不能自动发现 `grandparent` 规则，并计算这条规则的可靠性。

一个简化伪代码如下：

```text
for each target relation H:
    initialize rules with head H
    while rule body can still expand:
        enumerate candidate expansions
        compute support
        compute head_coverage
        compute PCA_confidence
        keep rules passing thresholds
```

这里常见两个指标是：

$$
head\ coverage(r)=\frac{support(r)}{|H|}
$$

它表示这条规则解释了多少头关系实例。另一个是 PCA confidence，可以粗略理解成“在头实体已知有某种对应对象的前提下，这条规则有多常成立”。它比普通置信度更适合知识图谱这种“不完整数据库”。

---

## 核心机制与推导

### 1. AMIE / AMIE+：离散搜索 + 剪枝

AMIE 的基本思路不是直接猜整条规则，而是从短规则开始，按广度优先搜索逐步扩展规则体。广度优先搜索就是“先看所有长度为 1 的候选，再看长度为 2，再看长度为 3”。

例如以 `grandparent(X,Z)` 为头部时，系统会尝试加入原子条件：

- `parent(X,Y)`
- `parent(Y,Z)`
- `child(Z,Y)`
- 其他可能关系

然后检查哪些组合满足闭合性、支持度和置信度要求。

可以把它画成一棵扩展树：

```text
head: grandparent(X,Z)
|
+-- add parent(X,Y)
|   |
|   +-- add parent(Y,Z)      -> 有意义候选
|   +-- add sibling(Y,Z)     -> 可能低支持
|
+-- add married(X,Y)
    |
    +-- add parent(Y,Z)      -> 另一个候选
```

AMIE+ 和后续 AMIE 3 的重点，是在“尽量不漏掉高质量规则”的前提下，用更强的剪枝减少无效搜索。剪枝就是“提前丢掉明显不可能通过阈值的候选”。

对一个玩具例子，假设知识库中：

- `parent(A,B)`
- `parent(B,C)`
- `parent(D,E)`
- `parent(E,F)`
- `grandparent(A,C)`
- `grandparent(D,F)`

那么规则

$$
grandparent(X, Z) \leftarrow parent(X, Y) \land parent(Y, Z)
$$

会在两个实例上命中：`(A,C)` 和 `(D,F)`。如果数据库里一共有 4 条 `grandparent` 事实，则：

$$
head\ coverage = \frac{2}{4}=0.5
$$

如果规则体一共匹配出 3 次，而其中只有 2 次在 PCA 分母定义下被确认支持头关系，则：

$$
PCA\ confidence = \frac{2}{3}
$$

### 2. Neural LP：把规则搜索改成矩阵链乘

Neural LP 的思路，是把“选哪条关系组成规则”变成一系列连续权重。连续权重就是“不是硬选某个关系，而是给每个关系一个概率或注意力分数”。

设每个关系 `R_k` 对应一个邻接矩阵 `M_{R_k}`。矩阵中的 `1` 表示实体间存在该关系。给定起点实体 `x` 的 one-hot 向量 `v_x`，长度为 `T` 的规则链可写成：

$$
\left(\prod_{t=1}^{T}\sum_{k=1}^{|R|} a_t^k M_{R_k}\right) v_x
$$

其中：

- `a_t^k` 是第 `t` 步选择关系 `R_k` 的注意权重
- `M_{R_k}` 是关系矩阵
- `v_x` 是起始实体向量

直观解释：第 1 步系统不是死板地说“必须走 `parent`”，而是对所有关系打分；第 2 步再打一次分；最后把这些关系矩阵连乘，得到哪些目标实体最可能成立。

对初学者，可以看一个最小矩阵例子。实体顺序是 `A, B, C`，关系 `parent` 的邻接矩阵为：

$$
M_{parent}=
\begin{bmatrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

起点 `A` 的 one-hot 向量是：

$$
v_A=
\begin{bmatrix}
1\\0\\0
\end{bmatrix}
$$

那么：

$$
M_{parent}v_A=
\begin{bmatrix}
0\\1\\0
\end{bmatrix}
$$

表示一步到达 `B`；再乘一次：

$$
M_{parent}^2v_A=
\begin{bmatrix}
0\\0\\1
\end{bmatrix}
$$

表示两步到达 `C`。如果控制器在两步里都把 `parent` 权重打高，那么模型就等价于学到了 `grandparent` 规则链。

### 3. DRUM：神经控制器 + 可共享规则链

DRUM 可以看成更强调“规则组合共享”的神经规则模型。它通过双向 RNN 生成关系链的分数，再用低秩张量近似减少参数量。低秩张量可以先理解成“用更少参数表示大规模关系组合”。

它的工程意义在于：很多规则链前缀是共享的，比如

- `parent -> parent`
- `parent -> sibling`
- `parent -> spouse`

这些都以 `parent` 开头。如果每条都独立学习，代价很高；如果共享链式表示，训练会更高效。

可以把离散方法和可微方法对比成两个流程：

```text
AMIE:
候选规则生成 -> 统计支持度/置信度 -> 剪枝 -> 输出显式规则

Neural LP / DRUM:
关系矩阵表示 -> 控制器产生权重 -> 连续组合关系链 -> 梯度更新 -> 提取高权重规则
```

两者本质上都在回答同一个问题：哪些关系组合能稳定推出目标关系。差别在于，一个靠显式枚举和统计，一个靠连续优化和参数学习。

---

## 代码实现

工程上可以先实现一个最小可运行版本：只处理长度为 2 的规则体，先从三元组构图，再计算支持度、head coverage 和一个简化版 PCA confidence。下面这段 Python 可以直接运行：

```python
from collections import defaultdict

triples = [
    ("A", "parent", "B"),
    ("B", "parent", "C"),
    ("D", "parent", "E"),
    ("E", "parent", "F"),
    ("A", "grandparent", "C"),
    ("D", "grandparent", "F"),
    ("X", "grandparent", "Y"),  # 额外头事实，用来让 coverage 不是 1.0
]

def build_index(triples):
    rel_to_pairs = defaultdict(set)
    for h, r, t in triples:
        rel_to_pairs[r].add((h, t))
    return rel_to_pairs

def infer_two_hop(rel_pairs_left, rel_pairs_right):
    left_map = defaultdict(set)
    for x, y in rel_pairs_left:
        left_map[y].add(x)

    inferred = set()
    for y, z in rel_pairs_right:
        for x in left_map.get(y, set()):
            inferred.add((x, z))
    return inferred

def rule_stats(triples, head_rel, body_rel_1, body_rel_2):
    rel_to_pairs = build_index(triples)
    head_pairs = rel_to_pairs[head_rel]
    body_pairs_1 = rel_to_pairs[body_rel_1]
    body_pairs_2 = rel_to_pairs[body_rel_2]

    inferred = infer_two_hop(body_pairs_1, body_pairs_2)
    support = len(inferred & head_pairs)
    head_coverage = support / len(head_pairs) if head_pairs else 0.0

    # 简化版 PCA：分母近似为所有可推出的候选数
    pca_denominator = len(inferred) if inferred else 1
    pca_confidence = support / pca_denominator

    return {
        "inferred": inferred,
        "support": support,
        "head_coverage": head_coverage,
        "pca_confidence": pca_confidence,
    }

stats = rule_stats(
    triples,
    head_rel="grandparent",
    body_rel_1="parent",
    body_rel_2="parent",
)

assert ("A", "C") in stats["inferred"]
assert ("D", "F") in stats["inferred"]
assert stats["support"] == 2
assert abs(stats["head_coverage"] - (2 / 3)) < 1e-9
assert abs(stats["pca_confidence"] - 1.0) < 1e-9

print(stats)
```

这段代码没有实现完整 AMIE，但已经覆盖了一个 Horn 规则从三元组生成到统计判断的核心流程。

对应模块职责可以拆成这样：

| 模块 | 职责 | 最小实现 |
|---|---|---|
| 数据输入 | 读取三元组或 CSV/JSON | `load_triples()` |
| 图索引 | 按关系组织实体对 | `build_index()` |
| 规则枚举 | 枚举头关系与规则体模板 | `enumerate_rules()` |
| 规则求值 | 计算 support / coverage / confidence | `rule_stats()` |
| 剪枝 | 过滤低质量候选 | `if metric >= threshold` |
| 验证 | 用留出集评估命中率 | `evaluate_rules()` |

如果继续往 AMIE 风格扩展，一般会按以下步骤组织：

1. 枚举目标头关系。
2. 从长度 1 的规则体开始扩展。
3. 每加一个原子条件，就做一次支持度和上界估计。
4. 低于阈值的候选立刻剪掉。
5. 对保留下来的闭合规则做精确统计。
6. 把通过阈值的规则写入结果集。

真实工程例子里，如果你在 Wikidata 上做人物知识补全，可以先针对 `place_of_residence`、`spouse`、`employer` 这类关系做规则挖掘，再把高置信规则作为候选生成器，交给后续的排序模型复核。这样做的价值是：候选来自规则，排序来自统计模型，解释性和覆盖率都比单独用一种方法更平衡。

---

## 工程权衡与常见坑

规则方法最突出的优点，是人能直接审查输出；最常见的问题，是高质量规则往往只覆盖局部子集。覆盖率低，就是“规则很准，但只能命中少量样本”。

常见问题和规避方式如下：

| 问题 | 表现 | 常见原因 | 规避方式 |
|---|---|---|---|
| 覆盖率低 | 规则很准但命中少 | 真实关系过于多样 | 与嵌入模型融合做补全 |
| 噪声敏感 | 少量脏数据误导规则 | KG 抽取错误、关系混淆 | 设支持度下限，做规则人工抽检 |
| 搜索空间爆炸 | 候选规则数量极大 | 关系种类多、链长增加 | BFS 剪枝、多线程、限制体长度 |
| 规则过拟合 | 只适用于个别实体模式 | 样本少且头关系稀疏 | 留出验证集，限制常量和过长链 |
| 置信度误判 | 把缺失当负例 | 开放世界假设 | 使用 PCA confidence 而非简单 precision |

新手常见误区，是看到一条高置信规则，就默认它“适合全图谱”。这通常不成立。比如某条规则只在“欧洲王室人物”子图里成立得很好，但放到全体人物实体上就几乎没用。原因不是规则错，而是它只描述了一个窄子分布。

另一个常见坑，是把规则系统当成完整推理引擎。实际上，很多知识图谱关系是模糊的、统计性的，未必能被短 Horn 规则稳定表示。比如“可能合作过”“风格相近”这类关系，更适合嵌入模型或图神经网络。

AMIE 3 这类系统在工程上常用多线程和更精确的统计计算来提升速度。一个简短流程可以理解为：

```text
按头关系切分任务 -> 并行扩展候选规则 -> 用上界先剪枝 -> 对剩余候选精确计算 -> 合并结果
```

这类优化的意义，不是改变算法本质，而是把“原本搜索太慢”变成“可以在大图谱上跑完”。

---

## 替代方案与适用边界

如果你的目标是“让人看懂为什么预测成立”，离散规则挖掘通常优先级更高；如果你的目标是“把规则学习嵌入端到端训练流程”，Neural LP 或 DRUM 更自然；如果你的目标只是“预测尽量准”，很多场景下嵌入模型会更省事。

| 方案 | 适合场景 | 不适合场景 | 核心边界 |
|---|---|---|---|
| AMIE / AMIE 3 | 稀疏 KG、需要显式规则、需要人工审查 | 关系特别密集且链很长 | 解释性强，训练非端到端 |
| Neural LP | 想保留规则结构，又要可微训练 | 资源有限、模型调参能力弱 | 适合中等规模规则学习 |
| DRUM | 想共享规则链、提高神经规则建模效率 | 只需要少量简单规则 | 适合复杂关系组合 |
| 嵌入模型 | 大规模补全、模糊关系预测 | 需要强解释性 | 准确率常较好，但解释弱 |

一个新手能直观看懂的真实例子，是在 Wikidata 挖掘类似“配偶 + 居住地”相关规则，用于补全人物地址信息。它的工程用途不是直接替代人工，而是先生成“值得检查的候选事实”。规则命中的候选，可以交给审核系统或排序模型二次过滤。

在复合任务里，混合使用规则与统计推理通常更实用：

- 规则负责生成高精度候选和证据链
- 嵌入模型负责补足长尾和模糊模式
- 最终排序器负责综合多源信号

简化说，规则像“精确但窄的模板”，嵌入像“宽覆盖但弱解释的相似性机器”。当知识图谱稀疏、噪声高、又需要审计时，规则方法价值最高；当关系复杂、数据密集、目标偏预测效果时，神经和嵌入方法更合适。

---

## 参考资料

| 类型 | 作者 / 会议 | 主要贡献 | 链接 |
|---|---|---|---|
| 规则挖掘综述 | 相关知识图谱规则推理综述论文 | 总结 AMIE、Neural LP、DRUM 等方法及评价指标 | https://pmc.ncbi.nlm.nih.gov/articles/PMC7250613/ |
| 规则挖掘 | AMIE / AMIE+ / AMIE 3 相关论文 | 用 BFS、head coverage、PCA confidence 挖掘 Horn 规则 | 可从综述中的参考文献继续追溯 |
| 神经可微规则 | Neural LP, NeurIPS 2017 | 用关系矩阵和控制器注意力做可微规则学习 | 可从综述与论文索引页检索 |
| 神经规则模型 | DRUM, NeurIPS 2019 | 用 RNN 和低秩表示学习可解释规则 | 可从综述与论文索引页检索 |
| 泛化与局限 | EARR 等后续讨论 | 讨论规则模型覆盖率、泛化和融合方向 | https://www.sciencedirect.com/science/article/pii/S2666651021000061 |
| 工程应用讨论 | 后续工程论文 | 讨论规则与嵌入模型混合、低覆盖问题 | https://www.sciencedirect.com/science/article/abs/pii/S0957417423013337 |
