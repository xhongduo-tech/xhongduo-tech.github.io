## 核心结论

模式层与数据层融合，指的是把 `schema 对齐` 和 `instance 对齐` 放进同一个决策过程里，而不是先把一层做死、再把另一层被动接上。这里的 `schema` 可以先理解为“字段、类、关系这些结构定义”，`instance` 可以先理解为“具体的一条条对象数据”。

只做模式层，常见错误是“名字像就当成一个东西”。例如 `brand`、`maker`、`manufacturer` 看起来都像品牌字段，但其中有的系统里 `manufacturer` 可能记录的是生产企业法人主体，有的系统里 `brand` 记录的是消费者可见品牌，这两者在业务上并不总是等价。只做数据层，常见错误是“值像就当成同一个对象”。例如两个商品都叫 “Apple”，一个可能是品牌，一个可能是水果分类，离开类型约束就很容易误配。

联合优化的核心价值是双向约束：

| 证据类型 | 作用 | 风险 |
|---|---|---|
| schema | 提供结构、类型、命名先验 | 名字像但语义不同 |
| instance | 提供值分布、共现、重复匹配证据 | 样本少时容易不稳定 |

可以把它抽象成一个总分：

$$
Score(C_i,C_j,\pi)=\alpha \, sim_{schema}(C_i,C_j,\pi)+\beta \, sim_{inst}(M)-\gamma \, Pen_{cons}
$$

其中，$(C_i, C_j)$ 是候选类对，$\pi$ 是属性映射，$M$ 是实例匹配集合。直观上：

- `sim_schema` 负责回答“结构上像不像”
- `sim_inst` 负责回答“数据上支不支持”
- `Pen_cons` 负责回答“有没有明显冲突”

真正有效的系统，不是让三项随便相加，而是让它们形成闭环：模式层缩小候选空间，数据层验证候选；数据层积累到足够强时，又可以反过来修正模式层。

---

## 问题定义与边界

这类问题不是单纯的“字段名相似度计算”，而是三个对象一起动：

| 对齐对象 | 含义 | 例子 |
|---|---|---|
| 类对齐 | 判断两个概念是否等价或接近 | `Product` vs `Item` |
| 属性对齐 | 判断两个字段是否对应 | `brand` vs `maker` |
| 实例对齐 | 判断两个具体对象是否相同 | 两个站点中的同一商品 |

因此更准确的任务定义是：

$$
\text{Task} = \arg\max_{\pi, M} \; \text{Consistency}(\text{schema}, \text{instance})
$$

这表示我们不是单独求一个字段映射 $\pi$，也不是单独求一个实例集合 $M$，而是要让“结构解释”和“数据解释”尽量一致。

边界也要说清楚。

第一，这不是说任何知识融合问题都应该塞进一个统一分数。统一分数是工程上的抽象，不是万能答案。某些领域，比如医疗编码、金融合规、工业控制，外部规则比统计证据更强，必须把硬约束放在更高优先级。

第二，这里讨论的是“辅助判定与联合修正”，不是“自动替代人工”。尤其在冷启动阶段，样本少、字段脏、命名混乱时，人工规则仍然是必要部分。

第三，这里默认两个数据源之间至少同时存在“结构信息”和“实例数据”。如果只剩下字段名字和类型，问题更接近传统 schema matching；如果 schema 非常弱，只剩大规模记录，问题更接近 entity resolution，也就是“判断不同记录是否指向同一实体”。

一个新手容易理解的边界例子是 `title`。两个系统里都有这个字段：

- 系统 A：`title` 表示商品标题
- 系统 B：`title` 表示网页 SEO 标题

名字完全一样，但实例分布明显不同。商品标题里会出现容量、规格、型号；SEO 标题里会出现站点名、营销词、页面栏目名。只看 schema，会误合并；看实例分布后，冲突会暴露出来。

所以本文说的“融合”，不是把两层粗暴拼起来，而是明确：类、属性、实例三个层次必须互相校验。

---

## 核心机制与推导

一个可落地的联合过程，通常按下面五步走：

1. 候选生成
2. schema 打分
3. instance 验证
4. 一致性惩罚
5. 迭代更新

### 1. 候选生成

候选生成也叫 `blocking`，可以先理解为“先筛掉明显不可能的组合，避免全量爆炸”。如果两个图谱各有 1000 个类、每类几十个属性，不做 blocking，后面组合数会直接失控。

常见 blocking 线索有：

- 名称词形接近，比如 `manufacturer` 和 `maker`
- 值类型接近，比如都主要是字符串品牌名
- 邻接结构接近，比如都经常和 `price`、`category` 同时出现
- 层次位置接近，比如都挂在 `Product` 下面

这一步的目标不是求真，而是“宁可少接受，不要多放错”。

### 2. schema 打分

`schema 打分` 是对候选对的结构性评估。它通常由多项组成：命名相似、类型兼容、父类关系兼容、约束兼容、邻居属性相似。

例如：

- `brand` vs `maker`：名称不完全一样，但都属于产品属性，取值常为短文本
- `title` vs `seo_title`：名称相近，但上下文和使用位置不一致

因此 `schema` 更像“先验分数”。它能告诉你哪些候选值得继续看，但它本身不能保证语义正确。

### 3. instance 验证

`instance 验证` 指的是用具体记录去支持或反驳前面的候选。直观公式可以写成：

$$
sim_{inst}(M)=\frac{1}{|M|}\sum_{(x,y)\in M} sim(x,y)
$$

这里的 $sim(x,y)$ 不是固定只能做字符串相等，它可以综合：

- 主键或外部 ID 是否一致
- 品牌、型号、容量等字段是否共现一致
- 文本相似度是否足够高
- 图结构邻居是否相似

一个“玩具例子”可以说明为什么实例证据重要。

假设两个源都有一个字段候选：

- 源 A：`brand`
- 源 B：`maker`

抽出 5 对已经大概率是同一商品的实例后，发现：

| 商品对 | A.brand | B.maker | 是否支持 |
|---|---|---|---|
| 1 | Apple | Apple | 支持 |
| 2 | Sony | Sony | 支持 |
| 3 | Huawei | Huawei | 支持 |
| 4 | Nike | Nike | 支持 |
| 5 | Lenovo | Lenovo | 支持 |

这时实例支持很强，说明 `brand -> maker` 很可能成立。

如果换成：

| 商品对 | A.brand | B.manufacturer | 是否支持 |
|---|---|---|---|
| 1 | Apple | Foxconn | 冲突 |
| 2 | Sony | Sony Group | 部分冲突 |
| 3 | Huawei | Huawei Device | 部分冲突 |
| 4 | Nike | Yue Yuen | 冲突 |
| 5 | Lenovo | Lenovo | 支持 |

那就说明 `brand` 与 `manufacturer` 不能直接视为等价字段，即使名字在商业语境里看起来接近。

### 4. 一致性惩罚

只有相似度还不够，因为有些错误恰好会拿到不错的局部分数，所以需要惩罚项：

$$
Pen_{cons} \approx D(\text{value distribution}_i,\text{value distribution}_j)
$$

这里的 $D$ 是“分布距离”，可以先理解为“两个字段值的统计形状差多远”。常见选择是总变差距离 $D_{TV}$ 或 Jensen-Shannon 距离 $D_{JS}$。

如果两个字段真对应，它们的值分布通常会在以下方面接近：

- 值长度分布接近
- 唯一值个数级别接近
- 是否更像枚举型接近
- 是否常出现单位、公司后缀、数字串接近

权重的含义也可以直接表出来：

| 项 | 含义 | 直观作用 |
|---|---|---|
| $\alpha$ | schema 权重 | 控制先验强度 |
| $\beta$ | instance 权重 | 控制数据证据强度 |
| $\gamma$ | 惩罚权重 | 控制冲突惩罚力度 |

### 5. 迭代更新

联合优化最关键的地方在“回传”。也就是：

- schema 给出候选
- instance 给出支持或反驳
- 结果再回去修正 schema 决策

如果一批实例已经稳定对齐，那么它们的共同属性分布会反过来强化属性映射；如果某个属性映射被反复证伪，就要降低对应类对的可信度，甚至拆掉已有假设。

一个“真实工程例子”是多电商源商品融合。站点 A 有字段 `brand`、`price`、`category`；站点 B 有 `maker`、`sale_price`、`leaf_category`。工程系统常先用 schema 候选把 `brand-maker`、`price-sale_price`、`category-leaf_category` 拉出来，再根据重复商品对齐的结果验证。若大量匹配商品显示 `brand` 和 `maker` 高度一致，而 `category` 与 `leaf_category` 只在上层类目接近，则系统可能接受前者为等价、后者为层级映射，而不是简单一一合并。

这就是“实例支持 schema，对 schema 的选择产生反馈”。

---

## 代码实现

工程上一般不会一次性全局求最优，而是做“blocking + 候选打分 + 迭代更新”。原因很简单：全局最优通常太贵，而且数据源一脏，精确模型会先死在候选爆炸上。

核心伪代码可以写成：

```python
candidates = blocking(schema_graph)
for (Ci, Cj) in candidates:
    pi = propose_attribute_alignment(Ci, Cj)
    M = match_instances(Ci, Cj, pi)
    s_schema = schema_score(Ci, Cj, pi)
    s_inst = instance_score(M)
    penalty = consistency_penalty(Ci, Cj, M)
    score = a * s_schema + b * s_inst - g * penalty
    if score >= tau:
        accept_alignment(Ci, Cj, pi)
    else:
        reject_or_keep_soft(Ci, Cj, pi)
```

各模块职责最好拆开：

| 模块 | 输入 | 输出 |
|---|---|---|
| blocking | schema 结构 | 候选对 |
| attribute proposal | 候选类对 | 属性映射 $\pi$ |
| instance matching | 属性映射 + 数据 | 实例集合 $M$ |
| scorer | schema/instance/penalty | 总分 |
| decision | 总分 | 接受 / 拒绝 / 软保留 |

下面给一个可运行的最小 Python 例子。它不是完整图谱系统，而是把“schema 分数 + 实例支持 + 分布惩罚”这个思想压成最小实现。

```python
from math import isclose

def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def normalized_name_similarity(x, y):
    x, y = x.lower(), y.lower()
    tokens_x = x.replace("_", " ").split()
    tokens_y = y.replace("_", " ").split()
    return jaccard(tokens_x, tokens_y)

def total_variation_distance(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)

def empirical_dist(values):
    n = len(values)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return {k: c / n for k, c in counts.items()}

def schema_score(field_a, field_b, type_a, type_b):
    name_score = normalized_name_similarity(field_a, field_b)
    type_score = 1.0 if type_a == type_b else 0.2
    return 0.6 * name_score + 0.4 * type_score

def instance_score(pairs):
    if not pairs:
        return 0.0
    hits = sum(1 for x, y in pairs if x == y)
    return hits / len(pairs)

def consistency_penalty(values_a, values_b):
    dist_a = empirical_dist(values_a)
    dist_b = empirical_dist(values_b)
    return total_variation_distance(dist_a, dist_b)

def fused_score(s_schema, s_inst, penalty, a=0.4, b=0.5, g=0.1):
    return a * s_schema + b * s_inst - g * penalty

brand_values = ["Apple", "Sony", "Huawei", "Nike", "Lenovo"]
maker_values = ["Apple", "Sony", "Huawei", "Nike", "Lenovo"]
manufacturer_values = ["Foxconn", "Sony Group", "Huawei Device", "Yue Yuen", "Lenovo"]

pairs_brand_maker = list(zip(brand_values, maker_values))
pairs_brand_manufacturer = list(zip(brand_values, manufacturer_values))

s_schema_1 = schema_score("brand", "maker", "string", "string")
s_inst_1 = instance_score(pairs_brand_maker)
penalty_1 = consistency_penalty(brand_values, maker_values)
score_1 = fused_score(s_schema_1, s_inst_1, penalty_1)

s_schema_2 = schema_score("brand", "manufacturer", "string", "string")
s_inst_2 = instance_score(pairs_brand_manufacturer)
penalty_2 = consistency_penalty(brand_values, manufacturer_values)
score_2 = fused_score(s_schema_2, s_inst_2, penalty_2)

assert isclose(s_inst_1, 1.0)
assert s_inst_2 < s_inst_1
assert penalty_1 == 0.0
assert score_1 > score_2
assert score_1 > 0.5

print("brand-maker:", round(score_1, 4))
print("brand-manufacturer:", round(score_2, 4))
```

这个例子故意保持简单，但它已经体现出三点工程原则：

- 不把总分写死成一个黑盒
- schema、instance、penalty 分开计算，方便调参
- 保留软状态，不要只有“合并/不合并”两种结果

真实系统中，还会加入更多细节：

- blocking 阶段先按类、语言、单位、来源过滤
- 实例匹配先做候选召回，再做精排
- 分布惩罚不只看原值，还看长度、数字比例、单位模式
- 每轮迭代只放行高置信映射，低置信先保留观察

---

## 工程权衡与常见坑

联合方法通常比单层方法稳，但也更容易因为实现粗糙而把错误放大。核心问题不在“能不能联合”，而在“联合的节奏和门控怎么设”。

常见坑可以直接看表：

| 常见坑 | 结果 | 规避方式 |
|---|---|---|
| 实例太少 | 分布不稳定 | 先保留强 schema 约束 |
| 早期错配 | 错误传播 | 软约束 + 置信度门控 |
| 纯字符串相似 | 语义混淆 | 加类型、单位、共现约束 |
| 候选过多 | 计算失控 | blocking + 局部联合优化 |

### 1. 实例太少时，别让数据证据主导

如果某个字段只出现过 3 次，且取值全都相同，实例统计几乎没有信息量。这时强行让 `instance score` 主导，会把偶然性当规律。工程上更稳的做法是：

1. 先做候选过滤
2. 再做局部联合匹配
3. 最后做一致性回查

也就是说，前期宁可保守，不要急着全局合并。

### 2. 早期错配会被迭代放大

联合优化最大的风险是“自我强化”。一旦某个错误类对齐被接受，它会诱导错误属性映射；错误属性映射再去支持错误实例匹配；错误实例匹配又反过来给前两者投票。最后系统会越来越自信，但方向错了。

解决思路不是取消迭代，而是加“门”：

- 高分直接接受
- 中间分数进入软保留
- 低分直接拒绝

这比二值化强很多，因为中间态给了系统“以后反悔”的空间。

### 3. 纯字符串相似度不够

很多系统最开始都爱做字段名相似度，因为实现便宜。但它只适合做非常弱的先验。比如：

- `author` 可能是作者名，也可能是内容来源
- `model` 可能是产品型号，也可能是机器学习模型名
- `title` 可能是商品标题，也可能是网页标题

所以字段名只应该参与召回，不应该单独决定融合。

### 4. 候选空间控制是硬成本问题

联合优化常常被讲成算法问题，但在工程里它先是资源问题。如果没有 blocking，任何“类 x 属性 x 实例”的三层组合都会让复杂度暴涨。真正可用的系统，往往不是最复杂的打分器，而是最懂得在哪一步砍搜索空间。

---

## 替代方案与适用边界

不是所有场景都值得上联合优化。是否采用，取决于 schema 强不强、实例多不多、错误成本高不高。

| 方法 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| schema-only | 结构清晰、数据少 | 简单稳定 | 易错配 |
| instance-only | 数据多、模式明显 | 能发现隐含对应 | 容易受噪声影响 |
| 联合优化 | 数据与结构都可用 | 鲁棒性最好 | 实现更复杂 |

可以用一个很粗的判断式表示：

$$
\text{Choose Method} =
\begin{cases}
\text{schema-only}, & |M| \text{ 很小} \\
\text{instance-only}, & schema \text{ 很弱} \\
\text{joint}, & \text{其余大多数情况}
\end{cases}
$$

具体理解如下。

如果是新系统冷启动，字段定义比较清楚，但历史记录很少，那么 schema-only 往往更稳。比如企业内部两个新服务要打通，字段字典、类型说明、枚举规则都很完整，但线上数据还没积累，这时先靠 schema 和人工映射更合理。

如果是老系统遗留整合，文档几乎没有，但实例量极大，instance-only 会更有价值。比如日志平台、用户行为埋点、广告素材库，字段命名混乱但样本很多，这时统计规律能帮你发现隐藏对应。

而大多数真实知识融合项目，既不是“只有文档”，也不是“只有数据”。它们通常有一些 schema，又有一批实例，因此联合优化最能体现价值。它不是因为“更高级”而值得做，而是因为它能降低两个典型误差：

- schema 脱离数据语义导致的静态错配
- instance 缺少类型约束导致的局部误判

最后要明确边界：并不是所有 ontology matching 或 entity resolution 都必须联合求解。但只要 schema 和 instance 两类证据同时存在，而且错误传播代价不低，联合通常比割裂处理更优。

---

## 参考资料

1. [Doan, Domingos, Halevy, 2003, Learning to Match Schemas of Data Sources: A Multistrategy Approach](https://researchgate.net/publication/2548228_Learning_to_Match_Schemas_of_Data_Sources_A_Multistrategy_Approach)
2. [Schopman et al., 2012, Instance-Based Ontology Matching by Instance Enrichment](https://link.springer.com/article/10.1007/s13740-012-0011-z)
3. [Leme et al., 2010, OWL schema matching](https://link.springer.com/article/10.1007/s13173-010-0005-3)
4. [Bhattacharya & Getoor, 2007, Online Collective Entity Resolution](https://research.ibm.com/publications/online-collective-entity-resolution)
