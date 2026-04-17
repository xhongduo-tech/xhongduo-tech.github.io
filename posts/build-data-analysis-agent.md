## 核心结论

构建数据分析 Agent，本质上是在做一条可审计的自动化分析链路：把自然语言问题转成受约束的 SQL，执行后再把结果转换成图表、结论和报告。这里的“受约束”很重要，意思是模型不能自由碰数据库，而只能在给定 schema、只读权限、脱敏规则和历史上下文内工作。

这类系统对初级工程师最有价值的点，不是“模型会写 SQL”，而是它把完整分析流程串起来了。一个可用的 Pipeline 至少包含五步：

| 阶段 | 输入 | 核心动作 | 输出 |
|---|---|---|---|
| 自然语言输入 | “上季哪个区域销售最好” | 识别意图、时间、指标、维度 | 结构化分析意图 |
| 意图理解 | 用户问题 + 对话历史 | 抽取时间范围、聚合方式、筛选条件 | 查询计划 |
| SQL 生成 | 查询计划 + schema | 生成只读、可执行、可验证 SQL | SQL + 参数 |
| 执行与校验 | SQL + 数据库连接 | 语法检查、权限检查、超时限制 | 结果集 |
| 可视化与叙述 | 结果集 + 上下文 | 选图表、生成洞察、输出报告 | 图表 + 文字结论 |

玩具例子很直观。用户问：“上季哪个区域销售好？” Agent 会先把“上季”解释成一个明确时间窗口，再把“销售好”解释成一个指标，比如总销售额。然后生成类似“按区域求和并排序”的 SQL，执行后给出柱状图和一句话总结。如果用户继续问“按城市拆解”，系统不能重新猜时间，而要继承刚才的“上季”和“销售额”上下文。

可以把整个过程写成一个公式：

$$
Insight = Narrative( Viz( Exec( SQL(NL, Schema, History), SafetyRules ) ) )
$$

这里的 `NL` 是自然语言问题，`Schema` 是数据库元数据，`History` 是对话历史，`SafetyRules` 是安全规则。少任何一层，系统都容易从“可用”退化成“看起来聪明但不可靠”。

---

## 问题定义与边界

数据分析 Agent 解决的是结构化数据查询与解释问题，不是通用数据库操作系统。结构化数据，白话讲，就是表格化、字段固定、适合 SQL 查询的数据，比如订单表、用户表、广告投放表。它的边界应该在设计时写死，而不是运行时“希望模型自觉”。

最小边界可以先定成下面这样：

| 边界项 | 允许 | 禁止 | 为什么 |
|---|---|---|---|
| 数据类型 | 结构化表、视图、物化视图 | 任意文件系统扫描 | 降低数据源不确定性 |
| SQL 类型 | `SELECT`、只读聚合 | `INSERT`、`UPDATE`、`DELETE`、DDL | 防止误写库 |
| 字段访问 | 白名单字段 | PII、密钥、内部备注 | 防止泄露敏感信息 |
| 上下文 | 显式保存时间、地区、指标 | 让模型自己猜 | 避免多轮问答漂移 |
| 输入接口 | 必传 schema 与权限配置 | 裸 prompt 直接问库 | 防止幻觉表名列名 |

所谓 PII，是“个人可识别信息”，白话讲就是能定位到具体人的敏感数据，比如手机号、邮箱、身份证号。数据分析 Agent 最常见的失败，不是 SQL 语法错，而是越权读了不该看的列，或者在多轮对话里把上一轮的时间范围忘掉了。

例如第一轮用户问：“Q3 东北区销售是多少？” 第二轮问：“那按城市拆解呢？” 第二句里的“那”没有任何数据库意义，真正有意义的是它继承了上一轮的两个条件：`Q3` 和 `东北区`。如果系统没有显式存储上下文，就可能错误地按“当前默认季度”或“全国范围”去查，最后得到一份语法正确但业务错误的结果。

所以问题定义必须先于模型调用。你不是在“让模型分析数据”，而是在“让模型在严格边界内参与查询规划和结果表达”。

---

## 核心机制与推导

一个工程上可落地的数据分析 Agent，常见结构是：

`Router -> Schema Manager -> SQL Generator -> Executor -> AI Data Analyzer`

这些名字看起来抽象，拆开后并不复杂。

`Router` 是路由器，白话讲就是先判断用户要什么。是查数、做分析、追问上文，还是要生成报告。不同请求应该走不同分支，不要一股脑都丢给同一个 prompt。

`Schema Manager` 是 schema 管理器，白话讲就是给模型提供“它现在能看到哪些表、哪些列、它们是什么意思”。Oracle 的方案里，这一层不仅保存 schema，还会做语义检索，从很多表里筛出“本次问题真正相关的 restricted schema”。这一步非常关键，因为模型如果面对几百张表，命中率会明显下降。

`SQL Generator` 才是把自然语言翻译成 SQL 的模块。它不应该直接吃完整数据库 schema，而是吃被收缩过的 restricted schema、业务规则和上下文。这样做的推导逻辑很简单：候选表越少，错误表名和错误 join 的概率越低。

`Executor` 负责两件事：校验和执行。校验包括只允许 `SELECT`、禁止敏感列、禁止越权表、限制 join 数、限制返回行数、限制执行时间。执行是最后一步，而不是第一步。

`AI Data Analyzer` 负责把结果集转成人能读懂的话，并决定用什么图。这里的“图”不是装饰，而是压缩信息的方式。比如时间序列适合折线图，Top N 排名适合条形图，组成占比适合堆叠图或饼图。

可以把这条链路继续细化成一次推导：

1. 从用户问题中抽取意图 $I$、约束 $C$、上下文 $H$
2. 从全量 schema 中选择相关子集 $S' \subseteq S$
3. 生成候选 SQL：$q = f(I, C, H, S')$
4. 用规则集 $R$ 做验证：$valid = check(q, R)$
5. 执行得到结果集 $D = exec(q)$
6. 选择图表 $v = select\_viz(D, I)$
7. 生成结论 $n = summarize(D, H)$

真实工程例子可以参考 Oracle 的 SQL Agent 架构。用户提交自然语言请求后，Router 先判断是“查数”还是“分析”。Schema Manager 再基于语义检索选出相关表和表摘要。SQL Generator 用这些上下文生成 SQL。Executor 做语法验证与只读执行。最后 AI Data Analyzer 根据查询结果写总结，必要时再触发报告生成。这个设计的重点不是模块多，而是把“理解问题”“写查询”“执行数据库”“解释结果”明确拆层，便于审计和替换。

---

## 代码实现

下面给一个最小可运行的 Python 版本，用来演示三个核心点：上下文继承、只读 SQL 校验、结果解释。它不是完整生产实现，但可以把思路跑通。

```python
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class AgentContext:
    time_range: str | None = None
    region: str | None = None
    metric: str | None = None

@dataclass
class DataAnalysisAgent:
    allowed_tables: set = field(default_factory=lambda: {"sales"})
    blocked_columns: set = field(default_factory=lambda: {"email", "phone"})
    history: List[Dict[str, Any]] = field(default_factory=list)

    def understand_query(self, question: str, ctx: AgentContext) -> AgentContext:
        q = question.lower()

        if "q3" in q:
            ctx.time_range = "2025-Q3"
        elif "上季" in question:
            ctx.time_range = "2025-Q4"  # 玩具例子：假设当前分析口径定义如此

        if "东北" in question or "northeast" in q:
            ctx.region = "Northeast"

        if "销售" in question or "sales" in q:
            ctx.metric = "revenue"

        return ctx

    def generate_sql(self, question: str, ctx: AgentContext) -> str:
        if "城市" in question:
            group_by = "city"
        else:
            group_by = "region"

        where = []
        if ctx.time_range:
            where.append(f"quarter = '{ctx.time_range}'")
        if ctx.region:
            where.append(f"region = '{ctx.region}'")

        where_sql = " AND ".join(where) if where else "1=1"

        return f"""
        SELECT {group_by}, SUM(amount) AS revenue
        FROM sales
        WHERE {where_sql}
        GROUP BY {group_by}
        ORDER BY revenue DESC
        LIMIT 20
        """.strip()

    def validate_sql(self, sql: str) -> None:
        upper = sql.upper()

        assert upper.startswith("SELECT"), "Only SELECT is allowed"
        banned = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
        assert not any(word in upper for word in banned), "Mutating SQL is forbidden"

        table_match = re.search(r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.I)
        assert table_match, "Missing FROM table"
        assert table_match.group(1) in self.allowed_tables, "Unauthorized table"

        for col in self.blocked_columns:
            assert re.search(rf"\b{col}\b", sql, re.I) is None, f"Blocked column: {col}"

    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        # 玩具结果：真实工程里这里应接数据库，并设置超时、只读账号、行数限制
        if "GROUP BY city" in sql:
            return [
                {"city": "Boston", "revenue": 1100000},
                {"city": "Harbin", "revenue": 900000},
                {"city": "Shenyang", "revenue": 700000},
            ]
        return [
            {"region": "Northeast", "revenue": 4200000},
            {"region": "East", "revenue": 3900000},
        ]

    def interpret_results(self, rows: List[Dict[str, Any]]) -> str:
        top = rows[0]
        key = next(k for k in top.keys() if k != "revenue")
        return f"最高项是 {top[key]}，销售额为 {top['revenue']:,}"

    def analyze(self, question: str, ctx: AgentContext | None = None) -> Dict[str, Any]:
        ctx = ctx or AgentContext()
        ctx = self.understand_query(question, ctx)
        sql = self.generate_sql(question, ctx)
        self.validate_sql(sql)
        rows = self.execute_sql(sql)
        summary = self.interpret_results(rows)

        result = {"sql": sql, "rows": rows, "summary": summary, "context": ctx}
        self.history.append({"question": question, "result": result})
        return result


agent = DataAnalysisAgent()

r1 = agent.analyze("Q3 东北区销售是多少")
assert "quarter = '2025-Q3'" in r1["sql"]
assert "region = 'Northeast'" in r1["sql"]
assert r1["rows"][0]["revenue"] == 4200000

r2 = agent.analyze("那按城市拆解呢？", ctx=r1["context"])
assert "GROUP BY city" in r2["sql"]
assert "quarter = '2025-Q3'" in r2["sql"]
assert "region = 'Northeast'" in r2["sql"]
assert r2["rows"][0]["city"] == "Boston"
```

这段代码故意把流程拆成 `understand_query -> generate_sql -> validate_sql -> execute_sql -> interpret_results` 五段，原因是每一段都需要单独测。生产系统里最常见的错误，并不是模型表现差，而是所有逻辑糊在一个 `analyze()` 里，最后无法定位是哪一层错了。

如果用 Node.js + PostgreSQL 实现，核心对象也应该有类似配置：schema 描述、缓存、历史对话、最大返回行数、SQL 超时、只读连接池。Grizzly Peak 的实现思路就是如此：先检查缓存，再理解问题，再生成 SQL，再执行，再解释结果。这个顺序的价值在于，越昂贵、越危险的动作越往后放。

---

## 工程权衡与常见坑

数据分析 Agent 真正难的是工程约束，不是 demo 效果。下面这张表比“提示词怎么写”更重要：

| 问题 | 典型表现 | 对策 |
|---|---|---|
| schema 幻觉 | 生成不存在的表或列 | 提供 restricted schema，SQL 执行前做程序化校验 |
| 多轮上下文漂移 | “按城市拆解”丢失上一轮时间范围 | 显式保存时间、地区、指标，不靠模型记忆 |
| PII 泄露 | 误查 `email`、`phone` | 列级白名单/黑名单 + 脱敏流水线 |
| 慢查询 | 大表全扫，成本高 | 强制 `LIMIT`、要求 `WHERE`、设置超时 |
| 结果不可解释 | 给出原始表格，没有业务结论 | 单独做结果摘要与图表选择 |
| 审计缺失 | 不知道模型查了什么 | 记录问题、SQL、耗时、行数、错误 |

一个典型坑是“不给 schema，只给数据库连接”。模型看起来也能生成 SQL，但只要表稍微复杂，就会开始猜列名、猜 join 键。这个问题不会随着模型变大自动消失，因为它不是推理强弱问题，而是输入信息不完整。

另一个高风险坑是把数据库安全寄托在 prompt 上。比如你写一句“不要访问敏感字段”，这只是一条自然语言建议，不是强约束。真正可靠的做法是：

1. 数据库账号只读
2. 执行前做 SQL AST 或正则级校验
3. 列级权限与脱敏先于模型输出
4. 每次请求写审计日志

例如模型生成：

```sql
SELECT customer_name, email, SUM(amount)
FROM sales
GROUP BY customer_name, email
```

如果系统提前定义了 `email` 为受限列，那么执行前就应该直接报错：`Access to column 'email' is restricted`。这比“执行后再提醒”更正确，因为泄露应该在访问前阻断。

还有一个容易被忽视的点是“结果验证”。如果返回 10 万行明细，再把它整个发给 LLM，总成本和时延都会失控。正确做法通常是先做字段统计、采样、异常值检测，再把压缩后的结果交给叙述模块。比如对数值列可以先算均值、最大值、分位数；对类别列可以先取 Top N。工程上这是必要的，不是优化项。

---

## 替代方案与适用边界

不是所有团队都需要同一种数据分析 Agent。单体式 SQL Agent 和多 Agent 协作方案，适用边界不同。

| 方案 | 适用场景 | 优点 | 复杂度 | 数据安全策略 |
|---|---|---|---|---|
| 单体流程 Agent | 内部 BI 问答、快速上线 | 部署快、链路短、调试简单 | 低到中 | 只读账号、schema 限制、审计 |
| 多 Agent 协作 | 报表、图表、洞察、调度都要自动化 | 分工清晰、模块复用高 | 中到高 | 每个 Agent 分权，统一编排审计 |
| 传统 BI + 模板 SQL | 指标稳定、问题固定 | 可控、低风险 | 低 | 静态权限控制最简单 |
| 人工分析师流程 | 高歧义、探索性强 | 解释力强、能处理模糊需求 | 高人力成本 | 靠制度和流程 |

TeamDay 的方案更偏多 Agent 编排。它把职责分成 Query Writer、Insight Generator、Dashboard Builder、Report Assembler 四个角色。这样的好处是每个模块可以单独替换，也更适合“查数后立刻做可视化，再定时出报告”的企业场景。

而 Oracle 或 Grizzly Peak 这类方案更适合先落一个单体流程：自然语言转 SQL，执行后生成结论，必要时补图表。它的前提是 schema 相对稳定，安全边界清楚，目标是尽快让内部用户实现自助查询。

所以选择标准不是“哪个更先进”，而是“你的系统到底自动化到哪一步”。如果只是想解决“业务问一句话，系统给一个可靠答案”，单体流程通常更务实。如果你要做的是“查询、洞察、仪表盘、周报月报全自动”，多 Agent 才开始有明显价值。

---

## 参考资料

1. Grizzly Peak Software, *Building Data Analysis Agents*（2026）  
   https://www.grizzlypeaksoftware.com/library/building-data-analysis-agents-eathhc14

2. Katonic AI, *Data Analyst Agent | Natural Language to SQL*  
   https://www.katonic.ai/agent-data-analyst

3. Oracle, *Design an OCI Generative AI powered SQL agent for databases and applications*  
   https://docs.oracle.com/en/solutions/oci-gen-ai-sql/index.html

4. TeamDay, *AI Data Analytics Team — Natural Language SQL, Dashboards & Reports*  
   https://www.teamday.ai/teams/data-analyst
