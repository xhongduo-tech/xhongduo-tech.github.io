## 核心结论

多 Agent 协作时，最容易坏掉的不是“推理能力”，而是“数据接口”。这里的接口，指一个 Agent 给下一个 Agent 传什么字段、字段是什么类型、哪些字段必填、值域是否合法。自由文本没有这些约束，所以一旦链路变长，就会出现字段丢失、命名不一致、类型漂移、单位混乱等问题。

解决办法不是让所有 Agent 都改成“只会说 JSON”，而是只在 **Agent 的输入输出边界** 上强制结构化。具体做法是：

1. 上游 Agent 声明输出 Schema，约定自己必须产出什么结构。
2. 下游 Agent 声明输入 Schema，约定自己只接受什么结构。
3. 在两者之间执行 `SchemaMatch(S_out, S_in)`。
4. 如果不兼容，就走 Adapter 做字段映射、类型转换、缺省补全或重试。
5. 只有通过校验的数据才进入下游 Agent。

这里的 Schema，直白地说，就是“数据长什么样的合同”。常见实现有两类：

| 方案 | 作用层 | 优点 | 局限 |
| --- | --- | --- | --- |
| JSON Schema | 跨语言通信协议 | 标准化、语言无关、适合传输与校验 | 手写较繁琐，可读性一般 |
| Pydantic BaseModel | Python 内部类型定义 | 写法简洁、类型提示强、运行时校验方便 | 更偏 Python，本身不是跨语言协议 |

工程上通常不是二选一，而是 **Pydantic 定义模型，导出 JSON Schema 作为通信合同**。前者方便开发，后者方便跨 Agent、跨服务、跨语言协作。

---

## 问题定义与边界

问题定义很具体：多个 Agent 串联执行任务时，如果它们之间传的是自然语言文本，那么每一跳都可能发生信息损耗。这里的信息损耗，不只是“少说一句”，而是下游无法稳定解析上游结果，最终导致流程失败或结果悄悄变错。

一个最小玩具例子：

- Agent A 负责找菜谱材料
- Agent B 负责计算购物清单

如果 Agent A 返回：

```text
这道菜需要鸡蛋、面粉、牛奶，差不多两三个人吃。
```

人能读懂，但 Agent B 未必知道：
- `鸡蛋` 是字段 `ingredients` 里的一个元素
- “两三个人” 是 `servings=2` 还是 `servings=3`
- 是否还有单位、数量、顺序要求

如果 A 改成返回：

```json
{
  "name": "松饼",
  "ingredients": ["鸡蛋", "面粉", "牛奶"],
  "servings": 3
}
```

B 就不需要“猜”。

这就是边界：**我们要结构化的是 Agent 间的数据交换，不是强迫 Agent 内部思考过程也结构化。** 一个规划 Agent 内部可以仍然写长段推理，但对外输出时必须收敛成约定好的数据对象。

自由文本与结构化输出的差别可以直接比较：

| 维度 | 自由文本 | 结构化输出 |
| --- | --- | --- |
| 可解析性 | 依赖 prompt、正则或二次 LLM 解析 | 直接按字段读取 |
| 类型安全 | 没有保证 | 可声明 `string`、`array`、`integer` 等 |
| 缺字段处理 | 常常静默失败 | 校验阶段直接报错 |
| 多跳稳定性 | 跳数越多越脆弱 | 通过契约降低漂移 |
| 可落库/可渲染 | 常需额外清洗 | 可直接写数据库或 UI |

如果把每一跳误读或漏传的概率记为 $\varepsilon$，链路有 $n$ 跳，则“至少发生一次损耗”的概率近似为：

$$
P(\text{loss}) \approx 1 - (1-\varepsilon)^n
$$

这不是精确物理定律，而是一个工程直觉模型：单跳看起来还行的自由文本，经过多跳后，整体失败率会上升得很快。

边界也要说清楚。结构化输出并不意味着：

- 所有字段都必须极细粒度建模
- 所有任务都必须严格 JSON 化
- 一旦校验失败系统就不可用

它真正适合的是：**输出要被程序继续消费的场景**。例如写数据库、驱动 UI、调用工具、触发审批、拼接 API 请求。这些场景下，结构不明确，后面一定会出问题。

---

## 核心机制与推导

可以把每个 Agent 看成一个带输入输出接口的函数：

$$
A: S_{in}^{(A)} \rightarrow S_{out}^{(A)}
$$

其中：

- $S_{in}$：输入 Schema，下游要求拿到什么结构
- $S_{out}$：输出 Schema，上游承诺产出什么结构

当 Agent A 的输出要送给 Agent B 时，需要检查：

$$
\perp(S_{out}^{(A)}, S_{in}^{(B)}) \in \{true, false\}
$$

这里的 $\perp$ 不是数学里唯一标准符号，本文把它当作“Schema 是否兼容”的布尔判断。若返回 `true`，表示可以直接传；若返回 `false`，表示需要适配。

一个常见的兼容失败不是“完全错”，而是“结构接近但不一致”。例如：

- A 输出 `title`
- B 需要 `name`

或者：

- A 输出 `ingredients: "鸡蛋, 面粉, 牛奶"`
- B 需要 `ingredients: ["鸡蛋", "面粉", "牛奶"]`

这种情况不应该直接丢弃，而应进入 Adapter。Adapter 直白地说，就是“格式翻译器”。它做的事一般包括：

| 适配类型 | 例子 | 风险 |
| --- | --- | --- |
| 字段重命名 | `title -> name` | 语义可能不完全等价 |
| 类型转换 | `"3" -> 3` | 字符串可能不是合法数字 |
| 结构提升 | `"a,b,c" -> ["a","b","c"]` | 分隔符规则不稳定 |
| 范围裁剪 | `12 -> 10` | 会修改原始语义 |
| 缺省补全 | 缺失 `currency` 时补 `CNY` | 默认值可能不适用 |

因此一个更实用的链路不是“只校验一次”，而是：

```python
S_out = schema_of(agent_a)
S_in = schema_of(agent_b)

if SchemaMatch(S_out, S_in):
    data_for_b = validate(raw_output_a, S_in)
else:
    candidate = Adapter(raw_output_a, source=S_out, target=S_in)
    data_for_b = validate(candidate, S_in)
```

如果 `validate` 仍然失败，再决定：
- 让上游 Agent 重试
- 降级成人工审核
- 退回自由文本并中止自动链路

一个简单但很关键的值域例子是评分字段：

```json
{"score": 12}
```

如果 Schema 规定 `score` 必须是 0 到 10 的整数，那么这条数据应该在边界上被拦住，而不是让下游继续计算。因为下游一旦把这个值写入数据库或参与排序，错误就已经扩散了。

所以真正有用的不是“让模型尽量听话”，而是建立一条 **强校验链**：

```text
Agent 输出
  -> Schema Validator
  -> Adapter / Retry
  -> 下游 Agent
```

这个机制的核心收益有两个：

1. 错误尽量早暴露  
早在接口边界上失败，比在第 4 个 Agent 内部炸掉容易定位得多。

2. 错误从“隐性”变“显性”  
自由文本常见的问题不是立即报错，而是默默传错字段。结构化校验能把这种静默错误改成显式错误。

真实工程里，一个典型例子是“检索 Agent + 审核 Agent + UI Agent”链路。检索 Agent 找到商品信息后，如果只返回一段描述文本，审核 Agent 需要再抽取字段，UI Agent 还要再格式化一次。每多一层解析，就多一层不确定性。若检索阶段就产出结构化对象，后面两个 Agent 基本只做校验和消费，链路稳定性会高很多。

---

## 代码实现

下面给一个可运行的 Python 最小实现。它不依赖真实大模型，只模拟“上游输出可能不规范，下游要求严格 Schema”的场景。

```python
from typing import List, Any
from pydantic import BaseModel, Field, ValidationError


class RecipeOut(BaseModel):
    name: str = Field(description="菜名")
    ingredients: List[str] = Field(description="材料列表，每个元素是一个字符串")
    score: int = Field(ge=0, le=10, description="推荐分，0到10之间的整数")


def adapter(payload: dict[str, Any]) -> dict[str, Any]:
    adapted = dict(payload)

    # 字段重命名
    if "title" in adapted and "name" not in adapted:
        adapted["name"] = adapted.pop("title")

    # 字符串转列表
    if isinstance(adapted.get("ingredients"), str):
        adapted["ingredients"] = [
            x.strip() for x in adapted["ingredients"].split(",") if x.strip()
        ]

    # 分数转整数并裁剪到合法区间
    if "score" in adapted:
        try:
            adapted["score"] = int(adapted["score"])
        except (TypeError, ValueError):
            pass

        if isinstance(adapted["score"], int):
            adapted["score"] = max(0, min(10, adapted["score"]))

    return adapted


def validate_or_adapt(payload: dict[str, Any]) -> RecipeOut:
    try:
        return RecipeOut.model_validate(payload)
    except ValidationError:
        adapted = adapter(payload)
        return RecipeOut.model_validate(adapted)


# 玩具例子：字段名不一致、ingredients 是字符串、score 超范围
raw = {
    "title": "松饼",
    "ingredients": "鸡蛋, 面粉, 牛奶",
    "score": 12,
}

obj = validate_or_adapt(raw)

assert obj.name == "松饼"
assert obj.ingredients == ["鸡蛋", "面粉", "牛奶"]
assert obj.score == 10

print(obj.model_dump())
print(RecipeOut.model_json_schema())
```

这段代码里有三层含义：

1. `RecipeOut` 是 Python 内部的强类型模型。  
术语“强类型”，白话说就是字段种类和范围先写死，运行时不让乱来。

2. `model_json_schema()` 可以导出 JSON Schema。  
这意味着你可以在 Python 里用 Pydantic 写得舒服，再把结果给别的服务或别的 Agent 做通用校验。

3. `adapter()` 只处理“可恢复的不兼容”。  
例如字段改名、字符串转列表、分数裁剪。若缺核心字段，仍然应该报错，而不是瞎补。

如果把这个模式放进多 Agent 流程，文本流程图大概是这样：

```text
用户请求
  -> 检索 Agent
  -> 输出 Schema 校验
  -> Adapter / Retry
  -> QA Agent
  -> 输入 Schema 校验
  -> UI Agent / DB Writer
```

再给一个更接近真实工程的例子：商品卡片生成系统。

- Agent A：抓取商品页，返回标题、价格、卖点、图片
- Agent B：做合规审核，判断是否含禁用词
- Agent C：生成前端卡片 JSON，直接给页面渲染

如果 A 返回自由文本：

```text
这是一款轻薄笔记本，价格大概五千多，适合学生和办公。
```

B 和 C 都会遇到问题：价格是 5000、5999 还是 5299？图片在哪？标题是什么？卖点是数组还是长句？

如果 A 返回结构化对象：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `title` | `string` | 商品标题 |
| `price` | `number` | 当前价格 |
| `highlights` | `array[string]` | 卖点列表 |
| `image_url` | `string` | 主图地址 |
| `compliance_flags` | `array[string]` | 审核标记 |

那么 B 可以只关心 `highlights` 和 `compliance_flags`，C 可以直接渲染卡片，甚至把结果直接写入缓存或数据库。结构化输出在这里不是“更优雅”，而是“能不能稳定上线”的差别。

---

## 工程权衡与常见坑

结构化输出不是没有成本。Schema 越严格，前期定义和维护成本越高；但越宽松，链路漂移风险越大。关键不是一味追求最严，而是看任务风险和后续消费方式。

一个常见权衡如下：

| 选择 | 好处 | 代价 | 适合场景 |
| --- | --- | --- | --- |
| 严格 Schema | 错误早暴露，便于自动化 | 模型更容易校验失败 | 写库、调用 API、审批链 |
| 宽松 Schema | 更容易生成成功 | 下游仍可能二次解析 | 探索性分析、低风险摘要 |

常见坑主要有五类。

| 常见坑 | 具体表现 | 规避策略 |
| --- | --- | --- |
| 字段描述过短 | 模型不知道字段语义，容易“自由发挥” | 给每个字段写清用途、格式、单位 |
| required 太多 | 一缺信息就整体失败 | 对可能缺失的信息设为可选 |
| 只校验输出，不校验输入 | 中间层改坏了数据却没人发现 | 每个 Agent 边界都校验 |
| Adapter 过于激进 | 把错误数据“修得像对的” | 只做低风险转换，高风险直接重试 |
| 把自然语言正文也强行塞进严格结构 | 可读性差，模型容易失败 | 仅结构化关键字段，正文作为一个文本字段 |

字段描述为什么重要，可以用一个简化伪代码理解：

```python
score: int = Field(ge=0, le=10, description="推荐分，只能是整数，不允许小数")
```

这里 `ge` 和 `le` 是边界约束，白话说就是“最小值”和“最大值”。`description` 虽然不是硬校验，但它会影响模型理解字段语义。很多失败不是模型不会输出 JSON，而是它不知道字段到底想表达什么。

另一个常见误区是把“Schema 校验”和“Prompt 约束”混为一谈。Prompt 只能告诉模型“尽量这样做”，Schema 校验才是真正的边界控制。前者是软约束，后者是硬约束。工程上两者都要有，但职责不同：

- Prompt 负责引导生成
- Schema 负责验收结果

如果只写“请用 JSON 输出”，通常还不够。更稳的写法是明确列出字段、类型、是否必填、是否允许额外字段。例如：

```text
只能输出以下 JSON 对象：
- name: string，必填
- ingredients: string[]，必填
- score: integer，0 到 10，必填
- 不允许输出额外字段
```

这类说明对零基础工程师也很重要，因为它把“让模型输出结构化内容”从玄学 prompt 技巧，变成了接口定义问题。

---

## 替代方案与适用边界

不是所有多 Agent 场景都要上完整 Schema 协商。更轻量的方案是：只结构化关键字段，其余保留自由文本。

例如一个审核 Agent 只需要返回：
- `passed: boolean`
- `reason: string`

这里 `passed` 是硬字段，`reason` 可以自由发挥。这样做的本质是把高风险决策结构化，把解释性内容保留为自然语言。

严格 Schema 与轻量校验可以对比如下：

| 方案 | 适用场景 | 优点 | 风险 |
| --- | --- | --- | --- |
| 严格 Schema | 数据要入库、调用工具、驱动 UI | 可自动化、稳定 | 开发与维护成本更高 |
| 轻量校验 | 只需少量控制字段 | 实现简单、成功率更高 | 自由文本部分仍有歧义 |
| 全自由文本 | 聊天、开放式创作 | 灵活、自然 | 几乎不可编程消费 |

一个实用的模块划分方式是：

```text
开放式理解模块 -> 结构化决策模块 -> 工具执行模块 -> 自然语言呈现模块
```

也就是说，开放式任务可以留给自由文本，但一旦进入“要交给程序继续执行”的阶段，就应该切到结构化协议。

还有一个边界要注意：结构化输出解决的是“格式可靠”，不是“事实正确”。一个符合 Schema 的 JSON 仍然可能内容是错的。比如：

```json
{
  "name": "松饼",
  "ingredients": ["鸡蛋", "面粉"],
  "score": 9
}
```

这个对象格式完全合法，但可能漏了牛奶，或者 `score` 是凭空编的。所以 Schema 校验通常要和其他机制配合：

- 检索增强，保证字段来源可追溯
- 业务规则校验，保证数值和状态合理
- 人工抽检，防止“格式正确但语义错误”

如果任务的主要输出就是长篇自然语言，例如写博客、写小说、写报告正文，那么不适合把全文硬拆成极细字段。更合理的方式是：

- 结构化：标题、摘要、标签、风险级别、引用列表
- 非结构化：正文主体

这样既保留自然语言表达能力，又保证关键控制面可程序化处理。

---

## 参考资料

| 来源 | 主要内容 | 解决的问题 |
| --- | --- | --- |
| Agno Structured Output for Agents | 用 Pydantic 作为 `output_schema`，让 Agent 返回已验证对象 | 如何把 Agent 输出从文本变成可直接消费的对象 |
| Claude Agent SDK Structured Outputs | 用 JSON Schema、Zod、Pydantic 获取验证后的 `structured_output` | 多步 Agent 工作流如何返回稳定 JSON |
| Pydantic JSON Schema 文档 | 从模型生成 JSON Schema，并声明字段类型、约束、描述 | 如何把 Python 类型系统变成通用校验协议 |
| Jaiqu GitHub 项目 | 自动把一种 JSON 结构映射到另一种 Schema | Schema 不兼容时如何做自动转换 |

- Agno: https://docs.agno.com/basics/input-output/structured-output/agent  
说明结构化输出的核心价值是“返回已验证对象而不是原始文本”，适合作为上游 Agent 的输出约束。

- Claude Agent SDK Structured Outputs: https://platform.claude.com/docs/en/agent-sdk/structured-outputs  
强调 Agent 工作流默认返回自由文本，而结构化输出可以通过 JSON Schema 保证返回结果可直接被程序使用。

- Pydantic JSON Schema: https://docs.pydantic.dev/2.0/usage/json_schema/  
给出从 Python 模型生成 JSON Schema 的方式，适合把内部模型定义和外部通信协议打通。

- Jaiqu: https://github.com/AgentOps-AI/Jaiqu  
展示“把任意 JSON 转成目标 Schema”的思路，适合用来理解 Adapter 层在多 Agent 链路里的角色。
