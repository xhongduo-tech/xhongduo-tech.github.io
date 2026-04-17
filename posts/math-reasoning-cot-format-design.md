## 核心结论

数学推理里的“思维链格式设计”，核心不是把步骤写得更长，而是把“哪一部分由模型负责，哪一部分交给外部系统负责”切清楚。

传统 CoT，Chain-of-Thought，意思是“模型用自然语言把中间步骤一步步说出来”。它的优点是可读，缺点是推理和计算混在同一条文本链里，任何一步算错、抄错、改写错，后面都会连锁偏移。

PoT，Program-of-Thought，意思是“模型把关键思路写成可执行程序，再由解释器执行”。它的提升机制不神秘，本质是把自然语言擅长的“拆解问题”与程序擅长的“精确计算”解耦。模型负责生成控制流、变量关系和公式结构，解释器负责给出确定数值。这样做后，算术误差不再主要由语言生成承担。

如果再叠加 self-verifier，自验证器，也就是“专门检查结果是否和程序、约束一致的模块”，以及 JSON schema 这类结构校验，就会形成“推理→执行→验证”的闭环。这个闭环通常比单纯延长 CoT 更有效，因为它不是继续相信模型，而是在关键节点引入确定性检查。

一个最小玩具例子就能说明差别：

- CoT：先说“先把 $a+b$ 算出来，再除以 $c$”，然后模型自己口算。
- PoT：先生成 `result = (a + b) / c`，再由执行器算出结果，最后自然语言只解释这段代码在做什么。

对数学、财务、表格计算这类任务，PoT 往往比纯自然语言 CoT 更稳。文献中常见的趋势是：执行成功率越高，最终准确率越高；而一旦输出格式错、代码无法执行、验证器无法通过，准确率就会明显回落。

---

## 问题定义与边界

这里讨论的“思维链格式”不是提示词修辞，而是中间表示形式。所谓中间表示，就是“模型在得到最终答案前，先产出的那一层内容”。

常见格式可以分成四类：

| 格式 | 白话解释 | 优势 | 主要问题 | 更适合的任务 |
|---|---|---|---|---|
| 自然语言 CoT | 用人话逐步解释 | 可读性高 | 算术易漂移 | 概念解释、证明草稿 |
| 代码式推理 PoT/PAL | 用程序表达步骤 | 可执行、可验证 | 需要执行环境 | 数值题、表格题、财务题 |
| LaTeX 推理 | 用公式表达过程 | 符号精确、展示好 | 不可直接执行 | 公式推导、教学展示 |
| 结构化 JSON 推理 | 用固定字段表达步骤 | 便于系统接入 | 容易被格式约束拖慢 | 工作流编排、审计流水线 |

问题的核心边界是：不是所有任务都该改成 PoT。

如果任务以“解释概念”为主，比如为什么梯度下降会收敛、为什么哈希表平均查找快，CoT 仍然自然，因为这里主要矛盾不是数值精度，而是叙述清晰度。

如果任务以“数值求解”为主，比如联立方程、税费结算、财报指标复核，那么把中间层改成代码或结构化格式通常更合适。原因很简单：这类任务的失败，往往不是思路完全不会，而是最后几步算错、抄错、单位混了。

真实工程例子是金融审计。模型读取一段财报文本后，不应直接口头说“营业利润率大约是多少”，而应先输出结构化描述，例如收入、成本、期间费用、口径说明，再由程序做核算。此时 JSON 的作用不是“更聪明”，而是“更容易被程序校验”。

$$
y = f_\text{LLM}(x; \theta), \quad r = g_\text{exec}(y)
$$

这里 $f_\text{LLM}$ 表示模型生成中间结果，$g_\text{exec}$ 表示执行器对中间结果求值。PoT 的重点是让 $y$ 更接近“可执行表示”，而不是“更长的解释文本”。

---

## 核心机制与推导

把 CoT 换成 PoT，真正改变的是误差传播路径。

在自然语言 CoT 中，可以粗略写成：

$$
y_\text{cot} = f_\text{LLM}(x;\theta), \quad a = h_\text{text}(y_\text{cot})
$$

这里 $h_\text{text}$ 表示“从文字步骤中读出答案”的过程。因为步骤和答案都在文本里，模型既要负责逻辑拆解，也要负责精确计算，所以误差会耦合。

在 PoT 中则更像：

$$
c = f_\text{LLM}(x;\theta), \quad a = g_\text{exec}(c), \quad v = \text{check}(c, a)
$$

- $c$ 是代码或可执行表达式。
- $g_\text{exec}$ 是解释器或工具调用。
- $\text{check}$ 是验证环节，用来检查执行结果、字段约束、单位一致性等。

这带来三个直接效果。

第一，算术从“语言问题”变成“执行问题”。  
语言模型不再需要在生成时顺带完成精确乘除法，只需要把关系写对。

第二，验证可以独立建层。  
如果执行失败、输出为空、变量未定义，系统可以判定这次求解失败，而不是把一段看似流畅但实际错误的解释直接交给用户。

第三，结果质量与执行成功率建立了可观测关系。  
只做 CoT 时，很难知道“这段话错在哪一步”；改成 PoT 后，可以统计代码执行率、schema 通过率、重试成功率，这些都是可监控指标。

下面用一个玩具例子说明。

题目：已知 $x=5$，求 $2x+3$。

自然语言 CoT 可能写成：

1. 先把 5 乘以 2。
2. 得到 12。
3. 再加 3 得到 15。

逻辑看起来通顺，但中间一步已经错了。

PoT 会写成：

```python
x = 5
y = 2 * x + 3
print(y)
```

解释器输出 13。模型最后只需要补一句：“代入 $x=5$，得到 $2x+3=13$。”

这就是“推理”和“计算”分工后的效果。

从公开结果看，PoT 在数学类 benchmark 上常见到明显提升。一个常被引用的数字是某些 DeepMind Math 子集上，Llama3-70B 从 CoT+验证的 58.6% 提升到 PoT+验证的 73.9%。这类提升不该被理解为“代码天生更聪明”，而应理解为“代码执行减少了文本链中的计算噪声”。

可以把机制压缩成下表：

| 环节 | CoT 主要风险 | PoT 对应改进 |
|---|---|---|
| 步骤生成 | 文字看似合理但细节漂移 | 用变量和表达式固定关系 |
| 数值计算 | 模型口算误差累积 | 交给解释器确定执行 |
| 结果检查 | 难以自动校验 | 可检查执行状态和输出 |
| 系统集成 | 难接外部工具 | 可接沙箱、数据库、公式引擎 |

---

## 代码实现

工程上不要把“生成代码”理解成“直接让模型一次性吐出最终 Python”。更稳的做法通常是两段式：

1. 先让模型自由生成解题草稿，目的是把问题拆开。
2. 再把草稿压缩成受约束的代码或 JSON，目的是执行和校验。

下面是一个可运行的最小实现，演示“草稿→程序→执行→校验”的核心逻辑：

```python
def solve_linear_word_problem(apples_per_bag: int, bag_count: int, extra: int) -> dict:
    # 模型阶段本应输出程序，这里直接手写模拟 PoT 结果
    code = f"result = {apples_per_bag} * {bag_count} + {extra}"
    env = {}
    exec(code, {"__builtins__": {}}, env)
    result = env["result"]

    explanation = (
        f"共有 {bag_count} 袋，每袋 {apples_per_bag} 个，再加上额外的 {extra} 个，"
        f"所以总数是 {result}。"
    )

    return {
        "code": code,
        "result": result,
        "explanation": explanation,
    }

toy = solve_linear_word_problem(4, 3, 2)
assert toy["result"] == 14
assert "result =" in toy["code"]
assert "总数是 14" in toy["explanation"]
print(toy)
```

这个例子很小，但已经体现了 PoT 的关键分工：

- “关系建模”体现在 `4 * 3 + 2`
- “精确求值”由 `exec` 执行
- “用户可读解释”留到最后再写

实际系统里不会直接裸用 `exec`，而会放进沙箱。沙箱就是“受限执行环境”，只允许少量安全操作，避免恶意代码访问文件、网络或系统命令。

如果任务还要与外部工作流集成，JSON 会比纯代码更容易控。比如财报核算可以先让模型输出：

```json
{
  "task": "gross_margin",
  "inputs": {
    "revenue": 1250000,
    "cost": 800000
  },
  "formula": "(revenue - cost) / revenue",
  "unit": "ratio"
}
```

随后由程序检查字段是否齐全、数值是否为正、单位是否符合预期，再把 `formula` 放入受限解释器执行。

一个简化流程可以写成：

```python
draft = llm.solve(problem, mode="free_form")
structured = formatter.to_schema(draft)
result = sandbox.execute(structured)

if not verifier.check(structured, result):
    answer = cot_fallback(problem)
else:
    answer = renderer.render(structured, result)
```

这里每一层都有明确职责：

- `draft` 负责想清楚
- `structured` 负责格式正确
- `sandbox.execute` 负责精确计算
- `verifier.check` 负责结果可信
- `cot_fallback` 负责兜底

真实工程里，这种分层比“让一个 prompt 同时完成推理、格式化、执行说明、最终作答”稳定得多。

---

## 工程权衡与常见坑

PoT 不是没有成本。它把错误从“答案错了”转成“代码错了、格式错了、执行失败了”，所以工程质量要求更高。

最常见的问题如下：

| 常见坑 | 具体表现 | 规避策略 |
|---|---|---|
| 过早强制 JSON/LaTeX | 模型一直修括号、引号、字段名 | 先自由草稿，再二次结构化 |
| 代码不可执行 | 变量未定义、缩进错误、非法函数 | 加 schema 校验和静态检查 |
| 执行器不安全 | 读文件、发请求、死循环 | 使用沙箱、超时、白名单 |
| 结果对了但口径错 | 单位、币种、含税口径不一致 | 在验证层检查元数据 |
| 回退缺失 | 一旦执行失败直接报错 | 保留 CoT 或模板化兜底 |

很多团队第一次接入结构化推理时，会直接要求模型“输出严格 JSON，不许多一个字”。这通常会把模型的注意力从“解题”拉到“补格式”上。结果是 JSON 很整齐，但推理质量下降。更稳的做法是先允许自由思考，再做 constrained decoding，约束解码，也就是“只在最后一层强制格式”。

另一个常见坑是把“能执行”误当成“正确”。代码跑通只说明语法上成立，不说明业务上成立。比如财务任务里，`(revenue - cost) / revenue` 执行没问题，但如果 `revenue` 和 `cost` 来自不同会计口径，结果仍然无效。所以验证器不能只看 `exit_code == 0`，还要看字段来源、单位、边界值。

下面是一个最小回退逻辑：

```python
def safe_compute(expr: str):
    try:
        env = {}
        exec(f"result = {expr}", {"__builtins__": {}}, env)
        return {"ok": True, "result": env["result"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

out = safe_compute("(12 + 8) / 5")
assert out["ok"] is True
assert abs(out["result"] - 4.0) < 1e-9

bad = safe_compute("unknown_var + 1")
assert bad["ok"] is False
```

这段代码体现的不是高级技巧，而是工程底线：执行失败必须被显式捕获，并进入回退路径，而不是让系统静默输出一段错误解释。

---

## 替代方案与适用边界

如果按任务类型选择格式，可以直接记住一个原则：叙述任务优先 CoT，数值任务优先 PoT，复杂工作流优先混合方案。

| 方案 | 典型任务 | 延迟 | 资源成本 | 适用边界 |
|---|---|---|---|---|
| CoT | 概念说明、证明思路、教学解释 | 低 | 低 | 对精确数值不敏感 |
| PoT | 数学求值、表格推导、财务核算 | 中 | 中 | 需要执行器和沙箱 |
| 混合方案 | 文档理解后做计算 | 中到高 | 中到高 | 既要解释又要精算 |

LaTeX 也是一种重要替代形式。它不直接执行，但适合表达符号结构。例如在代数推导、概率公式展开里，LaTeX 比自然语言更紧凑，也比 JSON 更适合人读。不过 LaTeX 本身不解决“最后一步算错”的问题，所以常见做法是：用 LaTeX 展示，用 PoT 计算，用 JSON 传递。

对小模型或低延迟系统，还可以用“轻量级 PoT”。所谓轻量级，就是不让模型生成完整程序，只让它输出关键算式或字段，再交给后端微服务求值。例如只输出：

```json
{"expression": "(principal * rate * days) / 365"}
```

这样做牺牲了一部分表达力，但降低了执行风险，也更容易做缓存和监控。

因此，最常见也最实用的架构不是“CoT 和 PoT 二选一”，而是：

1. 用 CoT 或自由文本做草稿，保证拆题质量。
2. 用 PoT 或 JSON 固化关键变量和公式。
3. 用执行器给出确定值。
4. 用验证器检查结果是否可接受。
5. 再把结果翻译回用户可读答案。

这类混合设计，通常比单押某一种格式更符合工程现实。

---

## 参考资料

| 来源 | 关键贡献 | 链接提示 |
|---|---|---|
| Program-of-Thoughts Prompting, TMLR 2023 | 系统提出 PoT，把推理与计算解耦 | https://www.emergentmind.com/papers/2211.12588 |
| ICLR 2025 DOTS 论文 | 给出 CoT 与 PoT 在数学任务上的精度对照，如 58.6% 到 73.9% 的提升案例 | https://proceedings.iclr.cc/paper_files/paper/2025/file/5e5d6f9ac33ba9349ba7b2be9f21bad9-Paper-Conference.pdf |
| DocMath-Eval 相关综述 | 展示长文档、财报、表格计算中 PoT 与验证层的工程价值 | https://www.emergentmind.com/topics/docmath-eval-benchmark |
| Program-of-Thought Prompting 词条 | 提供 PoT、PAL、程序化推理等概念脉络 | https://en.wikipedia.org/wiki/Program_of_Thought_Prompting |
| Program CoT / 执行与验证综述 | 总结代码质量、执行成功率与最终精度之间的关系 | https://www.emergentmind.com/topics/program-of-thought-program-cot |
| 格式设计实践文章 | 讨论自由草稿、受约束格式、JSON schema 等工程策略 | https://medium.com/%40michael.hannecke/beyond-json-picking-the-right-format-for-llm-pipelines-b65f15f77f7d |
