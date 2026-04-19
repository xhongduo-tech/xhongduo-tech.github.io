## 核心结论

语义解析是把自然语言问题 $x$ 转换成形式化表示 $y$ 的任务。形式化表示，就是计算机可以执行、查询或验证的结构，例如 SQL、$\lambda$ 表达式、知识图谱查询或一段程序。

它的目标不是“把原句换个说法”，而是生成一个和原问题语义等价、并且能在目标环境中运行的表示。

例如用户问：

> 北京出生的人有多少？

系统正确的输出不是：

> 这句话是在询问出生地为北京的人数。

而应该是能执行的查询：

```sql
SELECT COUNT(*) FROM people WHERE birth_city='北京';
```

核心判断标准是：生成的 $y$ 是否表达了“筛选北京 + 计数”这两个条件，并且能在数据库里得到正确结果。

统一形式可以写成：

$$
y^* = \arg\max_y p_\theta(y \mid x)
$$

意思是：在所有可能的形式化表示 $y$ 中，模型选择在当前输入 $x$ 下概率最高的那个。

| 输入自然语言 $x$ | 输出形式化表示 $y$ | 判断标准 |
|---|---|---|
| 北京出生的人有多少 | `SELECT COUNT(*) FROM people WHERE birth_city='北京';` | 是否正确筛选北京并计数 |
| 上周华东地区退款金额最高的渠道是什么 | 聚合、排序、过滤后的 SQL | 是否正确处理时间、地区、退款金额和渠道 |
| 找出导演过《盗梦空间》的人的出生地 | 知识图谱查询 | 是否沿着正确实体关系查询 |

语义解析的关键不在“语言像不像”，而在“结构对不对、语义准不准、结果能不能验证”。

---

## 问题定义与边界

语义解析研究的是从 $x \rightarrow y$ 的映射。这里的 $x$ 是自然语言输入，$y$ 是形式化输出。

术语先说明清楚：

| 术语 | 白话解释 |
|---|---|
| $x$ | 用户输入的自然语言问题，例如“北京出生的人有多少” |
| $y$ | 机器可执行的结构，例如 SQL 或程序 |
| `schema` | 数据库或知识库的结构说明，例如有哪些表、字段、字段类型 |
| `execution` | 执行生成结果，例如运行 SQL 得到答案 |

典型的输出类型包括：

| 输出类型 | 示例 | 常见场景 |
|---|---|---|
| SQL | `SELECT COUNT(*) FROM people...` | 数据库问答、BI 分析 |
| $\lambda$ 表达式 | `count(λx.birth_city(x, 北京))` | 逻辑语义表示 |
| 图查询 | SPARQL、Cypher | 知识图谱问答 |
| 程序 | Python、JavaScript、DSL | 代码生成、自动化任务 |
| API 调用 | `search_hotels(city="上海")` | 对话系统、工具调用 |

它和普通文本生成的区别很大。文本生成关注语言是否自然，语义解析关注输出是否能执行、是否和原问题严格对应。

| 任务 | 输入 | 输出 | 主要目标 |
|---|---|---|---|
| 自然语言理解 | 一句话或一段文本 | 意图、实体、标签 | 判断文本含义 |
| 语义解析 | 自然语言问题 | SQL、程序、逻辑表达式 | 生成可执行结构 |
| 文本生成 | 提示词或上下文 | 自然语言文本 | 生成流畅文本 |
| 问答 | 问题 | 答案文本 | 返回用户可读答案 |

新手容易把“理解一句话”都归为语义解析。这个边界需要划清：如果任务只是解释一句话的意思，更接近摘要、分类或问答；如果任务要落到数据库、知识库、程序或 API 上执行，就属于语义解析。

例如：

> 上周华东地区退款金额最高的渠道是什么？

如果系统只回答“这是在询问退款渠道”，不是语义解析。真正的语义解析要把它拆成结构化操作：

| 语义片段 | 对应操作 |
|---|---|
| 上周 | 时间过滤 |
| 华东地区 | 区域过滤 |
| 退款金额 | 聚合指标 |
| 最高 | 排序取最大值 |
| 渠道 | 分组或返回字段 |

真实工程里，这类问题通常会变成类似 SQL：

```sql
SELECT channel
FROM refunds
WHERE region = '华东'
  AND refund_date >= '2026-04-06'
  AND refund_date < '2026-04-13'
GROUP BY channel
ORDER BY SUM(refund_amount) DESC
LIMIT 1;
```

语义解析的边界就是：自然语言必须被落到一个可计算的目标系统里。

---

## 核心机制与推导

语义解析可以分成两条主要路线：基于语法的方法和神经网络方法。

基于语法的方法，会显式定义自然语言片段和目标表示片段之间的对应关系。同步文法是一类同时生成自然语言和逻辑表示的规则系统；CCG，即组合范畴语法，是一种把句子结构和语义组合绑定起来的语法框架。

神经方法通常把语义解析看成条件生成任务：输入一句话，输出一串目标表示 token。训练目标通常是最大化标注样本上的条件概率：

$$
\max_\theta \sum_i \log p_\theta(y_i \mid x_i)
$$

如果 $y$ 被看成一个序列 $y=(y_1,\dots,y_T)$，则可以分解为：

$$
p_\theta(y \mid x)=\prod_t p_\theta(y_t \mid y_{<t}, x)
$$

意思是：模型一步一步生成 SQL 或程序，每一步都依赖输入问题和前面已经生成的 token。

玩具例子：

输入：

> 北京出生的人有多少？

目标输出：

```sql
SELECT COUNT(*) FROM people WHERE birth_city='北京';
```

模型要学到的不是“北京”和“出生”这两个词，而是下面这个组合关系：

| 自然语言片段 | 目标结构 |
|---|---|
| 北京出生 | `WHERE birth_city='北京'` |
| 有多少 | `COUNT(*)` |
| 人 | `FROM people` |

推理时，神经模型可能按如下顺序生成：

```text
SELECT -> COUNT -> ( -> * -> ) -> FROM -> people -> WHERE -> birth_city -> = -> '北京'
```

但纯生成有一个明显问题：它可能生成非法 SQL。例如：

```sql
SELECT COUNT FROM people WHERE = '北京';
```

这在语法上就是错的。因此工程上通常会加入约束。

| 方法 | 核心思想 | 优点 | 风险 |
|---|---|---|---|
| 语法派方法 | 用同步文法、CCG 等规则推导目标表示 | 可解释、结构稳定 | 规则设计复杂，迁移成本高 |
| 神经方法 | 直接从输入生成目标序列 | 覆盖面广，适合大规模数据 | 容易生成非法或语义错误输出 |
| 约束解码 | 生成时限制候选 token 或结构 | 减少非法 SQL、非法程序 | 需要额外实现 schema 和语法约束 |

一个常见流程是：

```text
输入 x -> 编码 -> 生成 y -> 语法检查/执行校验 -> 输出
```

这里的“编码”是把自然语言转成模型内部向量表示；“生成”是输出 SQL、逻辑表达式或程序；“校验”是检查语法是否正确、字段是否存在、执行结果是否符合预期。

语义解析的难点不只是生成一个字符串，而是生成一个在目标世界里成立的结构。自然语言是松散的，数据库 schema 是严格的，中间必须有可靠的对齐机制。

---

## 代码实现

最小实现可以拆成三步：

1. 解析输入问题里的实体和操作。
2. 生成结构化查询。
3. 执行查询并验证结果。

下面是一个可运行的玩具 Text-to-SQL 示例。它没有使用大模型，只用规则演示语义解析的基本流程。

```python
import sqlite3

def parse_to_sql(question: str) -> str:
    if "北京" in question and ("多少" in question or "几" in question):
        return "SELECT COUNT(*) FROM people WHERE birth_city = '北京';"
    raise ValueError("无法解析该问题")

conn = sqlite3.connect(":memory:")
cur = conn.cursor()

cur.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, birth_city TEXT)")
cur.executemany(
    "INSERT INTO people (id, name, birth_city) VALUES (?, ?, ?)",
    [
        (1, "张三", "北京"),
        (2, "李四", "上海"),
        (3, "王五", "北京"),
    ],
)

question = "北京出生的人有多少"
sql = parse_to_sql(question)
result = cur.execute(sql).fetchone()[0]

assert sql == "SELECT COUNT(*) FROM people WHERE birth_city = '北京';"
assert result == 2

print(sql)
print(result)
```

这个例子很小，但已经包含语义解析的核心结构：

| 输入问题 | schema linking | SQL | 执行结果 |
|---|---|---|---|
| 北京出生的人有多少 | `北京 -> birth_city`，`多少 -> COUNT` | `SELECT COUNT(*) FROM people WHERE birth_city = '北京';` | `2` |

`schema linking` 是把自然语言里的词和数据库结构对齐。比如“出生地”“出生城市”“哪里出生”都可能对应字段 `birth_city`。如果这一步错了，后面的 SQL 即使语法正确，也会语义错误。

真实工程中的 Text-to-SQL 流程通常是：

```text
自然语言问题
-> schema linking
-> SQL 生成
-> SQL 语法检查
-> 数据库执行
-> 结果校验
-> 自然语言回答
```

真实工程例子：

用户问：

> 上周华东地区退款金额最高的渠道是什么？

系统需要识别：

| 问题片段 | 结构化含义 |
|---|---|
| 上周 | 时间范围 |
| 华东地区 | `region = '华东'` |
| 退款金额 | `refund_amount` |
| 最高 | `ORDER BY SUM(...) DESC LIMIT 1` |
| 渠道 | 返回 `channel` |

可能生成：

```sql
SELECT channel
FROM refunds
WHERE region = '华东'
  AND refund_date >= '2026-04-06'
  AND refund_date < '2026-04-13'
GROUP BY channel
ORDER BY SUM(refund_amount) DESC
LIMIT 1;
```

如果执行结果是：

| channel |
|---|
| 小程序 |

最终回答才可以是：

> 上周华东地区退款金额最高的渠道是小程序。

注意顺序：答案文本是执行结果的表达，不是模型凭空生成的结论。语义解析负责把问题落到可验证的结构上。

---

## 工程权衡与常见坑

语义解析最常见的问题是“格式对了但语义错了”。SQL 能运行，不代表 SQL 正确。

例如用户问：

> 退款金额最高的渠道是什么？

正确 SQL 应该按退款金额排序：

```sql
SELECT channel
FROM refunds
GROUP BY channel
ORDER BY SUM(refund_amount) DESC
LIMIT 1;
```

错误 SQL 可能按订单数排序：

```sql
SELECT channel
FROM refunds
GROUP BY channel
ORDER BY COUNT(*) DESC
LIMIT 1;
```

第二个 SQL 语法完全合法，也能返回结果，但它回答的是“退款订单数最多的渠道”，不是“退款金额最高的渠道”。

| 问题类型 | 风险 | 规避手段 |
|---|---|---|
| 格式正确但语义错误 | SQL 能跑但答案不对 | 执行校验、单测、人工抽检 |
| 字段对齐错误 | 把 `refund_amount` 选成 `order_amount` | schema linking、字段描述、样例值 |
| 表连接错误 | join 条件缺失或重复计算 | 显式建模外键、限制 join 路径 |
| 时间表达错误 | “上周”边界算错 | 统一时间函数和时区策略 |
| 组合泛化差 | 见过“华东”和“退款”，没见过组合问题 | 中间表示、约束解码、覆盖更多组合样本 |
| 标注数据少 | 模型学不到稳定映射 | 弱监督、伪标注、模板扩增、执行反馈 |

一个简单检查清单：

```text
[ ] SQL 语法是否合法
[ ] 表名和字段名是否存在
[ ] 字段含义是否和问题一致
[ ] 时间范围是否符合业务口径
[ ] 聚合函数是否正确
[ ] 排序方向是否正确
[ ] LIMIT 是否符合问题要求
[ ] 执行结果是否可解释
[ ] 是否有单测覆盖典型问法
```

工程上还要注意数据权限。语义解析系统如果能生成 SQL，就可能访问敏感字段。因此生产系统通常不会让模型自由生成任意 SQL，而是限制可访问表、字段、函数和查询范围。

另一个常见坑是把自然语言字段名当作数据库字段名。用户说“销售额”，schema 里可能叫 `gmv`；用户说“退款金额”，schema 里可能叫 `refund_amt`。没有 schema linking，模型很容易选错列。

更稳的方式是先把字段候选列出来：

| 用户表达 | 候选字段 | 最终选择 |
|---|---|---|
| 退款金额 | `refund_amount`, `order_amount`, `paid_amount` | `refund_amount` |
| 渠道 | `channel`, `source`, `platform` | `channel` |
| 华东地区 | `region`, `province`, `city` | `region` |

然后再进入 SQL 生成阶段。这样系统把“理解业务词”和“生成查询结构”分开，错误更容易定位。

---

## 替代方案与适用边界

不是所有自然语言到结构化输出的任务都必须做完整语义解析。关键要看任务是否需要可执行、可验证的结构。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 规则模板 | 稳定、可控、容易测试 | 覆盖范围窄，维护成本随模板增长 | 固定问法、固定报表 |
| 同步文法 | 推导过程清晰，可解释性强 | 规则设计复杂 | 结构严格、标注少的任务 |
| CCG | 句法和语义组合能力强 | 学习门槛高，工程实现复杂 | 需要组合语义解释的场景 |
| seq2seq | 实现简单，迁移快 | 可能生成非法结构 | 数据充足、格式相对简单 |
| 约束解码 | 能减少非法 SQL 或程序 | 需要接入语法和 schema | 生产级 Text-to-SQL |
| 预训练模型 | 泛化强，少样本效果好 | 成本高，稳定性和可控性要额外处理 | 用户表达自由、schema 多变的系统 |

同步文法、CCG、seq2seq、约束解码、预训练模型可以理解为一条从“规则更强”到“模型更强”的路线：

```text
同步文法 -> CCG -> seq2seq -> 约束解码 -> 预训练模型
```

新手可以用一个判断标准来选方案：

| 任务特征 | 更合适的方案 |
|---|---|
| 用户问法固定，只有几十种 | 规则或模板 |
| 结构严格，标注数据少 | 文法方法 |
| 有大量问句和 SQL 标注 | 神经语义解析 |
| 用户表达自由，schema 经常变化 | 预训练模型 + schema linking + 约束解码 |
| 只需要判断意图 | 分类模型即可 |
| 只需要抽取槽位 | NER 或 slot filling 即可 |

例如一个客服机器人只需要判断“查物流”“退货”“改地址”，这不需要完整语义解析，意图分类就够了。

但如果用户问：

> 查询过去 30 天每个渠道的退款率，并按退款率降序排列。

这就不只是分类。系统必须生成查询，计算退款率，处理时间范围、分组、排序和字段映射。此时应该选择可解释、可执行的语义解析方案。

结论是：当结果必须进入数据库、知识库、程序或 API，并且需要被执行验证时，优先选择语义解析；当任务只需要粗粒度判断或自然语言回答时，生成式近似或分类方案可能更简单。

---

## 参考资料

| 论文 | 贡献 | 适合阅读阶段 |
|---|---|---|
| Context Dependent Semantic Parsing: A Survey | 梳理上下文相关语义解析任务和方法 | 入门后建立任务地图 |
| Learning Synchronous Grammars for Semantic Parsing with Lambda Calculus | 展示同步文法到 $\lambda$ 表达式的经典路线 | 理解语法派方法 |
| Neural Shift-Reduce CCG Semantic Parsing | 将神经模型与 CCG 语义解析结合 | 理解神经语法方法 |
| Few-Shot Semantic Parsing with Language Models Trained on Code | 说明代码预训练模型在少样本语义解析中的效果 | 理解大模型路线 |
| On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model | 分析基于提示的大模型语义解析鲁棒性 | 理解工程风险 |

1. [Context Dependent Semantic Parsing: A Survey](https://aclanthology.org/2020.coling-main.226/)
2. [Learning Synchronous Grammars for Semantic Parsing with Lambda Calculus](https://aclanthology.org/P07-1121/)
3. [Neural Shift-Reduce CCG Semantic Parsing](https://aclanthology.org/D16-1183/)
4. [Few-Shot Semantic Parsing with Language Models Trained on Code](https://aclanthology.org/2022.naacl-main.396/)
5. [On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex](https://aclanthology.org/2023.eacl-main.77/)
