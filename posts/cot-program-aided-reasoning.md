## 核心结论

PAL，Program-Aided Language Models，中文可理解为“程序辅助语言模型”。它的基本做法不是让大语言模型直接在自然语言里完成全部推理，而是先把题目翻译成程序，再把程序交给解释器执行，最后根据执行结果组织答案。

它的最小流程可以压缩成一行：

`题目 -> LLM 生成代码 -> 解释器执行 -> 输出答案`

这套机制的关键，不是“代码比文字更高级”，而是“职责拆分更清晰”：

- 语言模型负责理解题意、抽取变量、确定步骤。
- 解释器负责执行加减乘除、循环、条件判断等确定性过程。

一个最小例子就能看出差异。题目是：

“7 个披萨，每个 8 片，4 个人平均分，每人几片？”

纯 CoT，Chain-of-Thought，思维链，通常会让模型直接在自然语言中写出：

1. 总片数是 $7 \times 8 = 56$
2. 每人分到 $56 \div 4 = 14$

PAL 的写法则是让模型只输出程序：

```python
num_pizzas = 7
slices_per_pizza = 8
total_slices = num_pizzas * slices_per_pizza

num_people = 4
answer = total_slices / num_people
```

解释器执行后，结果稳定得到：

$$
answer = \frac{7 \times 8}{4} = 14
$$

这里最重要的变化是计算职责发生了转移。模型不再亲自完成乘除法，而是只负责把题目映射成结构化程序。于是，很多原本出现在 CoT 中的错误会减少，比如：

- 中间算错数
- 抄错前文数字
- 某一步漏写
- 步骤描述正确但最终答案不一致

可以先用一张表看 PAL 和纯 CoT 的差异：

| 环节 | 纯 CoT | PAL |
|---|---|---|
| 题意理解 | LLM | LLM |
| 中间步骤表达 | 自然语言 | Python 代码 |
| 数值计算 | LLM 自己算 | 解释器执行 |
| 结果稳定性 | 受采样波动影响较大 | 运算部分更稳定 |
| 错误暴露方式 | 错误常埋在文字里 | 报错、变量值、日志更容易观察 |
| 常见失败模式 | 算错、抄错、步数漂移 | 代码生成错、语法错、运行异常 |

因此，核心结论可以收紧成两点：

1. PAL 的价值不是“让模型会编程”，而是“把不擅长的精确执行外包给解释器”。
2. 当任务包含明确数值关系、公式、条件判断、循环或枚举时，PAL 往往比纯自然语言 CoT 更稳。原始论文在 GSM8K 等数学推理任务上也报告了明显提升。

---

## 问题定义与边界

先明确问题本身。

CoT 的作用，是把“直接输出答案”改成“先展开步骤，再得出答案”。这对复杂问题很有帮助，因为模型不再只生成一句结论，而是先显式写出中间过程。

但 CoT 有一个天然边界：它写出来的是文字，不是可执行过程。文字能表达思路，却不能保证计算正确。一个看起来很合理的思维链，仍然可能在某一步把

$$
48 + 27
$$

算错，也可能在后续步骤里把 `48` 抄成 `84`。

PAL 要解决的，不是“模型完全不会推理”，而是下面两个更具体的问题：

1. 模型能不能把自然语言题目稳定映射成程序。
2. 程序中那些需要确定性执行的步骤，能不能交给解释器完成。

对新手可以这样理解：

- CoT 是“模型一边想，一边自己算”。
- PAL 是“模型负责把思路写成代码，再让解释器去算”。

所以，PAL 不是否定思维链，而是把思维链拆成两部分：

| 部分 | 谁负责 | 作用 |
|---|---|---|
| 语义推理 | LLM | 理解题意、识别关系、决定步骤 |
| 符号执行 | 解释器 | 进行精确计算、循环、分支、枚举 |

这个边界很重要，因为 PAL 并不是万能框架。不同任务，适用性差异很大：

| 问题类型 | PAL 是否合适 | 原因 |
|---|---|---|
| 小学到中学数学应用题 | 很合适 | 变量、数量关系、目标函数都清晰 |
| 需要循环、枚举、条件判断的问题 | 合适 | 可直接写成程序执行 |
| 表格统计、折扣计算、单位换算 | 合适 | 规则明确，计算链条稳定 |
| 开放式写作、摘要、情感分析 | 不太合适 | 重点不是确定性计算 |
| 强依赖最新外部事实的问题 | 适用性有限 | 程序能算，但不能替代知识检索 |
| 审美评价、模糊判断类任务 | 不合适 | 没有明确可执行目标 |

看两个对比例子更直观。

适合 PAL 的题：

“一个盒子里有 12 支铅笔，老师又发了 5 支，然后平均分给 17 个学生，每人几支？”

这类题具有三个特征：

- 实体明确：铅笔、学生
- 操作明确：加法、除法
- 目标明确：每人数量

因此它很适合翻译成程序：

```python
pencils = 12
extra_pencils = 5
students = 17
answer = (pencils + extra_pencils) / students
```

不适合 PAL 的题：

“请评价这篇文章是否有启发性，并给出修改建议。”

这个任务没有唯一答案，也不存在确定的程序执行路径。即使硬写成代码，也只是把开放判断包装成程序外壳，并不会因此更准确。

所以，PAL 的真实定位不是“通用推理终极解法”，而是“把结构化、可执行的那一部分从自然语言推理中拆出来”。

---

## 核心机制与推导

PAL 的核心机制可以用一个简单符号系统描述：

- $x$：输入问题
- $c$：LLM 生成的代码
- $e = \text{interp}(c)$：解释器执行代码得到的结果
- $y$：最终输出给用户的答案

一种常见的形式化写法是：

$$
P(y \mid x) = \sum_c P_{\text{LLM}}(c \mid x)\cdot \delta\!\big(e=\text{interp}(c)\big)\cdot P_{\text{LLM}}(y \mid x,c,e)
$$

这条式子看起来复杂，但意思并不复杂。它只是把整个系统拆成三个阶段。

### 第一段：生成程序

$$
P_{\text{LLM}}(c \mid x)
$$

意思是：给定题目 $x$，模型生成某段代码 $c$ 的概率是多少。

这一步仍然是概率性的。也就是说，同一道题在不同采样条件下，模型可能写出不同的程序。比如：

- 变量名不同
- 步骤拆分不同
- 有时写中间变量，有时直接一行算完

所以，PAL 并没有消除生成的不确定性，它只是把不确定性集中到了“程序生成”阶段。

### 第二段：执行程序

$$
e=\text{interp}(c)
$$

这一步表示：代码一旦生成，后面的执行交给解释器。

这里的关键变化是，系统从“概率生成”切换到了“确定性执行”。如果代码不变、输入不变、解释器环境不变，那么执行结果就应当保持一致。

对比 CoT，这一步的价值非常直接：

- CoT：模型既要理解题意，又要自己完成运算
- PAL：模型只负责写出运算规则，真正的运算交给解释器

### 第三段：组织最终回答

$$
P_{\text{LLM}}(y \mid x,c,e)
$$

这一步表示：模型拿到原问题、程序、执行结果后，再组织最终输出。

有些任务中，这一步很轻，直接输出 `e` 就够了；有些任务中，还需要把执行结果转成自然语言。例如：

- “答案是 14”
- “每人分到 14 片披萨”
- “总计 312 件，还剩 312 件”

因此，PAL 不是“程序执行后就彻底不需要语言模型”，而是把语言模型的职责缩到更擅长的部分。

### 为什么说它是“互补机制”

可以把整个流程写成：

`x -> LLM -> c -> Interpreter -> e -> LLM -> y`

真正发生能力切换的位置，是：

`c -> e`

在这一步之前，系统处理的是自然语言理解与结构化表达；在这一步之后，系统处理的是确定性执行。

继续用披萨题说明：

题目：
“7 个披萨，每个 8 片，4 个人平均分，每人多少片？”

对应关系是：

- $x$：题目文本
- $c$：模型生成的代码
- $e$：解释器执行结果 `14`
- $y$：自然语言答案“每人 14 片”

如果只用 CoT，模型需要同时承担两件事：

1. 理解“7 个披萨，每个 8 片，4 个人平均分”这组语义关系
2. 正确计算 $7 \times 8 \div 4$

PAL 则把第二件事剥离出去。换句话说，PAL 不是让模型“更会算”，而是让模型“少算”。

### 一个更实用的误差视角

可以把系统总误差粗略分解成：

$$
\text{总误差} \approx \text{语义映射误差} + \text{执行误差}
$$

其中：

- 语义映射误差：题意理解错、变量抽取错、关系建模错
- 执行误差：中间算错、抄错、分支执行错、枚举漏项

纯 CoT 中，这两类误差都由 LLM 承担。

PAL 中，执行误差的大部分被解释器接管，因此系统主要风险收缩到“程序生成是否正确”。这并不代表 PAL 不会错，而是它的错法更集中、更容易被定位。

这也是它在工程上有价值的原因。因为下面两类错误的可处理性差异很大：

| 错误类型 | 是否容易发现 | 是否容易重试 | 是否容易做日志分析 |
|---|---|---|---|
| 自然语言里某一步算错 | 较难 | 较难 | 较难 |
| 程序生成语法错或变量错 | 容易 | 容易 | 容易 |

所以，PAL 的优势不只是“可能更准”，还包括“更可观测、更可调试、更可测试”。

---

## 代码实现

下面给一个最小可运行实现。它不调用真实 LLM，只模拟 PAL 的后两步：拿到代码后执行，并返回统一的 `answer`。

这个示例可以直接运行：

```python
from typing import Any, Dict


def run_pal_python(code: str) -> Any:
    namespace: Dict[str, Any] = {}

    # 只保留极少量安全内建，避免 demo 因完全禁用而无法表达基础计算
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "range": range,
        "len": len,
    }

    exec(code, {"__builtins__": safe_builtins}, namespace)

    if "answer" not in namespace:
        raise ValueError("generated code must assign the final result to `answer`")

    return namespace["answer"]


toy_code = """
# 7 个披萨，每个 8 片
num_pizzas = 7
slices_per_pizza = 8

# 计算总片数
total_slices = num_pizzas * slices_per_pizza

# 4 个人平均分
num_people = 4
answer = total_slices / num_people
"""

result = run_pal_python(toy_code)
assert result == 14
print(result)
```

这段代码里有两个工程要点。

第一，接口必须统一。  
无论模型怎么写中间步骤，最终都要求：

```python
answer = ...
```

第二，执行环境必须受控。  
示例里只保留了少量安全内建，目的是说明一个原则：PAL 不是“随便执行模型代码”，而是“在受限环境里执行结构化程序”。

### 一个更完整的接口形态

如果把它扩展成更接近真实服务的样子，可以写成：

```python
from typing import Any, Dict


class ProgramInterface:
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

    def build_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question)

    def execute(self, code: str) -> Any:
        namespace: Dict[str, Any] = {}
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "range": range,
            "len": len,
        }

        exec(code, {"__builtins__": safe_builtins}, namespace)

        if "answer" not in namespace:
            raise ValueError("generated program must assign `answer`")

        return namespace["answer"]


MATH_PROMPT = """
You are a reasoning system that solves math word problems by writing Python.

Rules:
1. Output Python code only.
2. Use clear variable names.
3. Use comments starting with # when needed.
4. Store the final result in a variable named `answer`.
5. Do not print anything.

Question:
{question}

Python:
""".strip()


def demo():
    interface = ProgramInterface(MATH_PROMPT)
    prompt = interface.build_prompt("7个披萨，每个8片，4个人平均分，每人几片？")

    generated_code = """
# number of pizzas
num_pizzas = 7

# slices in each pizza
slices_per_pizza = 8

# total number of slices
total_slices = num_pizzas * slices_per_pizza

# number of people
num_people = 4

# final answer
answer = total_slices / num_people
"""

    value = interface.execute(generated_code)

    assert "Question:" in prompt
    assert value == 14
    return prompt, value


prompt_text, value = demo()
print(prompt_text)
print(value)
```

对新手来说，这个接口可以拆成三层：

| 层级 | 作用 | 对应代码 |
|---|---|---|
| Prompt 层 | 告诉模型该如何输出程序 | `build_prompt` |
| Program 层 | 模型产出可执行 Python | `generated_code` |
| Execution 层 | 在受限环境里执行程序并取 `answer` | `execute` |

### 一个包含条件判断的例子

PAL 的价值，不只体现在四则运算上。只要问题包含条件、循环、枚举，它就开始比纯自然语言更有优势。

例如：

“一个商店买满 200 元打 9 折。某人买了 3 件商品，单价分别是 45、60、80 元，请问实际支付多少？”

可执行程序是：

```python
prices = [45, 60, 80]
total = sum(prices)

if total >= 200:
    answer = total * 0.9
else:
    answer = total

print(answer)
```

上面代码有一个工程问题：它用了 `print(answer)`，但接口不应依赖打印结果。更稳的写法是：

```python
prices = [45, 60, 80]
total = sum(prices)

if total >= 200:
    answer = total * 0.9
else:
    answer = total
```

计算过程为：

$$
45 + 60 + 80 = 185
$$

因为 $185 < 200$，所以不打折，最终：

$$
answer = 185
$$

这个例子说明：只要题目里出现规则分支，PAL 的收益就开始扩大。因为它不只是“替模型算数”，而是把“按规则执行”的部分整体外包给了解释器。

### 一个更接近服务端的伪代码

真实系统中，数学问答服务通常不会只做“生成代码然后执行”这么简单，而是还会加入日志、异常处理、重试、回退策略。

一个常见流程是：

1. API 接收问题文本
2. 用模板构造 prompt
3. LLM 返回一段 Python
4. 服务端检查代码是否完整
5. 在受限解释器中执行
6. 记录代码、结果、错误类型
7. 返回答案，必要时回退到其他方案

伪代码如下：

```python
def solve_math_question(question: str, llm, program_interface):
    prompt = program_interface.build_prompt(question)
    code = llm.generate(prompt)

    try:
        value = program_interface.execute(code)
        return {
            "question": question,
            "code": code,
            "answer": value,
            "status": "success",
        }
    except Exception as exc:
        return {
            "question": question,
            "code": code,
            "answer": None,
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
```

这里真正重要的，不是“多写了几行代码”，而是系统从“只拿到最终答案”变成“能看到完整中间产物”：

- 原问题
- prompt
- 生成代码
- 执行结果
- 异常类型
- 是否需要重试

这会直接影响后续评估、调优和监控。

---

## 工程权衡与常见坑

PAL 很适合做结构化推理，但工程落地时，失败点也比较集中。问题通常不在“论文想法是否成立”，而在“程序接口是否收敛”。

先看一张总表：

| 项目 | 错误做法 | 更稳的做法 |
|---|---|---|
| 最终输出 | 让模型自由 `print(...)` | 强制写入 `answer` |
| 变量命名 | `a`, `b`, `c` 这类无语义名 | 使用 `total_slices`、`num_people` 等语义名 |
| 输出内容 | 混合解释文字和代码 | 只输出合法 Python |
| 代码完整性 | 不检查截断 | 检查语法与结尾完整性 |
| 失败处理 | 一次执行失败就结束 | 捕获异常、重试、必要时回退 |
| 执行环境 | 直接 `exec` 全权限代码 | 沙箱、白名单、超时、资源限制 |
| 结果校验 | 只信任一次执行 | 结果范围检查、类型检查、单元测试 |

下面按常见坑展开。

### 1. 代码根本不是合法 Python

这是最常见的问题之一。模型可能把说明文字混到代码里：

```python
先算总片数
total = 7 * 8
然后除以 4
answer = total / 4
```

这段代码不可运行，因为：

- `先算总片数` 不是注释
- `然后除以 4` 也不是注释

正确写法必须是：

```python
# 先算总片数
total = 7 * 8

# 然后除以 4
answer = total / 4
```

因此，prompt 中最好明确写清：

- 只输出 Python
- 注释必须以 `#` 开头
- 不要输出解释文字、Markdown 围栏、自然语言结论

### 2. 最终输出接口不统一

如果模型有时写：

```python
result = 14
```

有时写：

```python
ans = 14
```

有时写：

```python
final_value = 14
```

那么调用方就必须猜“哪个变量才是答案”。这是不必要的不确定性。

更稳的做法是统一协议：

```python
answer = ...
```

这相当于把 PAL 从“开放式代码生成”收敛成“带固定接口的程序生成”。

### 3. 自然语言和计算混在一起

模型有时会生成这种内容：

```python
x = "7 pizzas times 8 slices"
answer = x / 4
```

这里的问题不是解释器不够强，而是语义映射失败。模型没有把自然语言关系正确落到数值变量上。

这类问题往往要通过 prompt 约束来修复，例如要求：

- 所有数字必须显式赋值
- 每个中间量都要定义变量
- 不要把自然语言字符串直接参与运算

### 4. 输出被截断

模型生成到一半停止，会留下不可执行代码：

```python
total_slices = num_pizzas *
```

这种错误和“计算稍微错一点”不同，它会让执行完全失败。

工程里通常有三种处理方式：

| 方式 | 做法 | 作用 |
|---|---|---|
| 停止符控制 | 设置停止 token | 降低混入多余内容的概率 |
| 语法检查 | 执行前先 parse | 提前拦截不完整代码 |
| 自动重试 | 失败后重新生成 | 提高成功率 |

### 5. 安全边界不清

这是 PAL 与普通 CoT 最大的工程差异之一。CoT 只是生成文本，PAL 则要执行代码，所以安全边界必须前置设计。

至少要考虑：

- 禁止文件系统访问
- 禁止网络访问
- 禁止系统命令执行
- 限制可用内建函数
- 限制执行时间
- 限制内存和 CPU
- 限制导入模块

如果这些约束缺失，那么你得到的不是“程序化推理”，而是“把不可信代码直接交给运行环境”。

### 6. 题意理解错了，程序会稳定地产生错答案

这是新手最容易误解的地方。PAL 确实能降低执行误差，但它不能自动修复建模错误。

例如题目是：

“每箱 24 件，有 18 箱，卖出 120 件后还剩多少件？”

如果模型错误理解成“进货 120 件后还剩多少件”，程序可能写成：

```python
boxes = 18
items_per_box = 24
total_items = boxes * items_per_box
restocked = 120
answer = total_items + restocked
```

这段程序执行完全正确，但问题理解错了，所以答案仍然错。

因此，PAL 提升的是“按给定程序精确执行”的能力，不是“自动纠正所有语义错误”的能力。

### 一个实用 checklist

PAL 落地前，至少应检查下面几项：

- 模板是否要求只输出 Python
- 是否固定使用 `answer` 作为统一出口
- 是否要求显式变量赋值，而不是隐式计算
- 是否对语法错误、名称错误、除零错误做异常捕获
- 是否做执行前语法检查
- 是否设置超时和资源限制
- 是否在沙箱环境中执行
- 是否保留问题、代码、结果、异常的日志

如果把工程难点压缩成一句话，那就是：

PAL 的难点从来不是“把 demo 跑起来”，而是“让生成程序的接口足够稳定、足够安全、足够可观测”。

---

## 替代方案与适用边界

PAL 不是唯一方案。至少可以和三类方法对比：

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| 直接答案生成 | 最简单，延迟低 | 不可解释，复杂题容易错 | 简单问答、低风险场景 |
| 纯 CoT | 有中间步骤，可读性强 | 计算仍由模型承担，容易算错 | 解释过程重要，但计算不重 |
| PAL | 计算和规则执行更稳，可观测性强 | 依赖代码生成质量，需安全执行环境 | 数学题、结构化推理、规则执行 |
| 外部工具调用器 | 可接搜索、数据库、计算器等多个工具 | 系统复杂度更高，编排更难 | 多工具工作流、真实生产系统 |

把同一道题分别交给 CoT 和 PAL，看差异会更清楚。

题目：

“一个仓库有 18 箱货，每箱 24 件，卖出 120 件后还剩多少件？”

### 纯 CoT 路径

理想步骤是：

1. 总件数  
   $$
   18 \times 24 = 432
   $$

2. 剩余件数  
   $$
   432 - 120 = 312
   $$

最终答案是 312。

这种方式的优点是可读性强，但问题在于：如果第一步 `18 × 24` 算错，后面即使逻辑正确，结论也会错。

### PAL 路径

PAL 会把题目翻译成程序：

```python
boxes = 18
items_per_box = 24
total_items = boxes * items_per_box

sold_items = 120
answer = total_items - sold_items
```

解释器执行后得到：

$$
answer = 312
$$

这里模型只需完成两件事：

- 识别变量：`boxes`、`items_per_box`、`sold_items`
- 建立关系：先乘后减

只要程序语义正确，乘法和减法就由解释器稳定完成。

### 什么时候继续用解释器，什么时候没必要

适合继续使用解释器的情况：

| 判断标准 | 说明 |
|---|---|
| 最终目标是数值、布尔值、集合、排序结果 | 可直接映射为程序输出 |
| 中间过程存在公式、规则、循环或分支 | 程序执行比自然语言更稳定 |
| 错误代价较高 | 希望把计算错误压到最低 |
| 需要保留可审计中间过程 | 程序比自然语言更容易记录和回放 |

不值得强行程序化的情况：

| 判断标准 | 说明 |
|---|---|
| 任务本身是开放式文本生成 | 程序执行不能提升“文风正确性” |
| 结果标准不唯一 | 没有单一可执行目标函数 |
| 代码生成成本高于收益 | 简单问题没必要多一层执行 |
| 重点是写作风格、观点组织、修辞表达 | PAL 不解决这类问题 |

### PAL 的上限在哪里

PAL 能显著降低执行误差，但不能自动消除建模误差。

可以把它理解成下面这个判断：

- 如果错在“算错了”，PAL 往往有帮助。
- 如果错在“理解错了题”，PAL 帮助有限。

因此更准确的说法是：

PAL 不是“让模型从不出错”，而是“把错误集中到更容易观测、诊断和修复的位置”。

这也是为什么它与 CoT 不是互斥关系，而是互补关系：

- CoT 强在显式展示推理路径
- PAL 强在把可执行部分从自然语言中剥离出来

在很多系统里，二者甚至可以组合使用：

1. 先让模型用自然语言规划步骤
2. 再让模型把关键步骤落成代码
3. 最后用解释器执行并校验结果

从这个角度看，PAL 不是取代思维链，而是把思维链中的“理解”和“执行”分层处理。

---

## 参考资料

| 来源 | 用途/重点 |
|---|---|
| Gao et al., “PAL: Program-Aided Language Models”, ICML 2023, https://proceedings.mlr.press/v202/gao23f.html | PAL 原始论文。核心贡献是把自然语言推理与程序执行拆开，并在 GSM8K、SVAMP 等任务上验证效果。适合看方法定义、实验设置和准确率对比。 |
| reasoning-machines/pal, https://github.com/reasoning-machines/pal | 官方实现。可直接看到 prompt 模板、`ProgramInterface` 风格的接口设计、数学问题和代码执行的组织方式。适合理解论文如何落成工程代码。 |
| Emergent Mind: Program-Aided Language Models / Program-Aided Language Modeling, https://www.emergentmind.com/topics/program-aided-language-models-pal | 适合快速建立整体认识，尤其是概率分解、生成程序再执行的工作流、优缺点总结。适合作为综述入口。 |
| Northeastern Math Word Problems 页面, https://pages.github.khoury.northeastern.edu/4130/2024F/hw/math_word_problems/ | 教学视角的数学应用题拆解材料。适合新手练习“把自然语言题目翻译成变量、公式和 Python 步骤”。 |
| Coursera 对 PAL 的科普页面, https://www.coursera.org/articles/program-aided-language-models | 面向初学者的概念解释。适合理解“模型负责理解，解释器负责计算”的直观分工。 |
| Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, https://arxiv.org/abs/2201.11903 | 理解 CoT 背景时很重要。PAL 的价值只有放在 CoT 的局限下看，才容易看清。适合补足“为什么先有思维链，再有程序化推理”。 |
| Toolformer, https://arxiv.org/abs/2302.04761 | 帮助理解“把外部工具接入语言模型”这一更大框架。PAL 可以看作工具调用思想在程序执行场景中的一个具体实现。 |
