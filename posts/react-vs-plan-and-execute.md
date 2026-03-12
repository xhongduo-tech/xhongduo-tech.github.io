## 核心结论

ReAct、Plan-and-Execute、ADaPT 解决的是同一件事：让大语言模型把一个目标拆成可执行动作，并根据环境反馈继续推进。区别不在“能不能做”，而在“什么时候规划、规划多深、失败后怎么恢复”。

ReAct 的核心循环是“思考 -> 行动 -> 观察”。这里的“观察”就是模型拿到工具返回值或环境状态，再决定下一步。它的优点是灵活，适合目标路径不明确、必须边看边做的任务；缺点是全局计划隐含在上下文里，任务一长就容易局部贪心，反复重想，token 开销高。

Plan-and-Execute 的核心思路是“先规划，再执行”。“规划器”就是先产出步骤列表的模块，“执行器”就是按步骤调用工具的模块。它更像把复杂任务先画成流程图，因此在可预测、多步骤、依赖关系清晰的场景里更稳定，也更省上下文；但如果某个关键步骤本身就太难，整条计划会卡死，必须额外引入重规划机制。

ADaPT 可以看作两者之间的按需混合架构。它不是一开始把所有层级都拆完，而是先让执行器试；只有执行失败，才把当前这一步递归拆小。这个“递归”就是函数调用自己，把一个难任务继续变成更小子任务。在 ALFWorld 基准上，论文给出的总体成功率是 ReAct 56.7%，Plan-and-Execute 63.4%，ADaPT 79.8%，相对 ReAct 提升 23.1 个百分点。这个数字说明：长链任务里，是否具备“局部失败后局部重拆”的能力，往往比是否一开始就有计划更重要。

| 架构 | 计划频度 | 失败恢复 | 典型成功率（ALFWorld） | 典型代价 |
| --- | --- | --- | --- | --- |
| ReAct | 每一步都隐式重想 | 靠下一轮思考自然修正 | 56.7% | token 高，路径灵活 |
| Plan-and-Execute | 起始时统一规划一次 | 需要显式重规划 | 63.4% | token 较省，执行更稳 |
| ADaPT | 仅在失败节点局部规划 | 对失败子任务递归拆解 | 79.8% | 控制逻辑更复杂 |

---

## 问题定义与边界

这里讨论的不是“聊天机器人如何回答问题”，而是“带工具的 Agent 如何完成多步任务”。“Agent”可以简单理解为：不仅会生成文字，还会调用搜索、数据库、代码解释器、浏览器等工具的系统。

这三类架构的边界，主要由三个变量决定：

1. 任务复杂度  
复杂度指完成目标所需的中间步骤数量，以及步骤之间是否存在依赖。

2. 任务可预测性  
可预测性指你是否能在执行前大致知道路线。比如“读 CSV 后算总销量再出报告”很可预测；“去陌生网页上找隐藏按钮再提交表单”则不可预测。

3. token 预算  
token 可以粗看成模型处理文本的成本单位。上下文越长、每步重想越多，成本越高。

Onyx 的对照表可以压缩成一句话：ReAct 适合简单到中等复杂、路径不确定的任务；Plan-and-Execute 适合复杂、多步、流程可预期的任务。ADaPT 则适合第三类情况：整体任务很长，但真正困难的是其中少数局部步骤，而且这些难点往往要到执行时才暴露。

先看一个玩具例子：把干净杯子放到桌上。

如果用 ReAct，系统会这样工作：

1. 看当前环境，猜杯子可能在哪  
2. 去一个位置找  
3. 没找到，重新思考  
4. 再去另一个位置找  
5. 找到后再去清洗  
6. 清洗后再放桌上

它的优点是不用预先知道杯子在哪。缺点是“找杯子”这个局部过程会不断吃掉思考预算。

如果用 Plan-and-Execute，系统会先写出：

1. 找到并拿起杯子  
2. 清洗杯子  
3. 把干净杯子放桌上

这在结构上更清晰，但“找到并拿起杯子”本身可能就是最难的一步。一旦执行器做不到，后两步没有机会发生。

如果用 ADaPT，开始时也许只有三步；但当执行器发现“找到并拿起杯子”失败后，规划器才继续把这一步拆成：

- 去台面找杯子  
- 去柜子找杯子  
- 去抽屉找杯子

如果任一分支成功，就继续后面的清洗与摆放。这就是“按需分解”的本质。

再看一个真实工程例子：CSV 数据分析。

目标是“计算总销量 -> 找最佳商品 -> 生成报告”。  
ReAct 往往会在每次工具调用前重新读一遍目标和已知结果，像这样：

- 我先看看 CSV 列名  
- 我再想想要不要算总销量  
- 我算完后再想最佳商品怎么定义  
- 我找到商品后再想报告格式

这种模式在三步任务上也许能跑通，但在十步以上任务上，重复上下文会迅速累积。Plan-and-Execute 则先固定步骤，再逐步执行；ADaPT 会在“找最佳商品返回空结果”时，只对这一子问题继续细拆，比如先检查列名映射、再检查缺失值、再改用聚合口径，而不是整条流程全部重来。

---

## 核心机制与推导

ReAct 的机制可以写成一个循环：

$$
s_t \xrightarrow{\text{LLM}} (thought_t, action_t) \xrightarrow{\text{env/tool}} observation_t \xrightarrow{} s_{t+1}
$$

其中 $s_t$ 表示第 $t$ 步时模型可见的状态，包括历史动作、历史观察和当前目标。问题在于，长任务里模型必须把“全局计划”压缩在连续的局部决策中。于是随着轨迹变长，注意力会被大量无关中间状态稀释，容易出现两类错误：

1. 本地贪心  
“贪心”就是只看眼前最像正确答案的动作，不考虑它是否服务于全局目标。

2. 循环重试  
比如“再搜一次”“再点一次”“再检查一次”，但没有新的信息增量。

Plan-and-Execute 把问题改写成两个阶段：

$$
\text{Task} \xrightarrow{\text{Planner}} \{p_1, p_2, ..., p_n\}
$$

$$
p_i \xrightarrow{\text{Executor}} o_i,\quad i=1,\dots,n
$$

优点是执行器每次只看当前步骤，状态更短；缺点是计划质量决定上限。如果 $p_2$ 本身定义过大，执行器仍然可能失败。

ADaPT 的关键推导在于：不是所有步骤都值得提前细拆。真正高效的策略是把规划当成一种昂贵资源，只在当前执行器能力不足时才投入。论文里的 Algorithm 1 可以压缩成下面的逻辑：

```text
ADAPT(T, k):
  if k > d_max:
    return False

  completed = executor(T)

  if completed:
    return True

  P, logic = planner(T)
  O = [ADAPT(T_sub, k+1) for T_sub in P]
  return logic(O)
```

这里有三个关键点。

第一，执行优先。  
系统先问“当前任务能不能直接做”，而不是先问“要不要拆”。这保证简单任务不会被过度规划。

第二，逻辑组合。  
`logic(O)` 表示子任务结果如何合并。若任务是“先找到杯子，再清洗，再放桌上”，逻辑更像 AND；若任务是“台面找或柜子找或抽屉找”，逻辑更像 OR。也就是说，规划器不仅要给出子任务集合，还要给出这些子任务之间的关系。

第三，递归深度受限。  
$d_{\max}$ 是最大分解深度。没有这个边界，系统可能把一个模糊任务无限拆下去，成本失控。

继续用“把干净杯子放桌子”的玩具例子。顶层任务是：

- 找并拿起杯子  
- 清洗杯子  
- 放到桌上

若执行器在第一步失败，ADaPT 不会整条重做，而是把“找并拿起杯子”局部展开：

- 去台面找并拿起杯子  
- 去柜子找并拿起杯子  
- 去抽屉找并拿起杯子

此时组合逻辑是：

$$
\text{FindCup} = c_{\text{counter}} \lor c_{\text{cabinet}} \lor c_{\text{drawer}}
$$

整个任务则是：

$$
\text{Task} = \text{FindCup} \land \text{CleanCup} \land \text{PutOnDesk}
$$

这就是它比普通 Plan-and-Execute 更强的地方：计划不是一次性静态产物，而是执行失败后的局部修正结构。

---

## 代码实现

先给一个可运行的 Python 玩具实现。它不依赖真实大模型，而是用规则函数模拟三种架构的差异，重点看控制流。

```python
from dataclasses import dataclass

WORLD = {
    "cup_locations": ["drawer"],  # 真正的位置未知，只有抽屉有杯子
    "cup_clean": False,
    "cup_on_desk": False,
}

@dataclass
class Metrics:
    llm_calls: int = 0

def react_find_cup(metrics: Metrics) -> bool:
    # ReAct: 每试一个地方前都“重新思考”一次
    for place in ["counter", "cabinet", "drawer"]:
        metrics.llm_calls += 1  # thought/action generation
        if place in WORLD["cup_locations"]:
            return True
    return False

def plan_execute_find_cup(metrics: Metrics) -> bool:
    # Plan-and-Execute: 计划里只写“找到杯子”，执行器不会局部重拆
    metrics.llm_calls += 1  # planner once
    metrics.llm_calls += 1  # executor once
    return False  # 任务过大，单步执行失败

def adapt(task: str, depth: int, dmax: int, metrics: Metrics) -> bool:
    assert depth >= 1
    if depth > dmax:
        return False

    metrics.llm_calls += 1  # executor try
    if task == "find_cup_from_drawer":
        return True
    if task in {"clean_cup", "put_on_desk"}:
        return True

    if task == "find_cup":
        metrics.llm_calls += 1  # planner called only on failure
        subtasks = ["find_cup_from_counter", "find_cup_from_cabinet", "find_cup_from_drawer"]
        return any(adapt(t, depth + 1, dmax, metrics) for t in subtasks)

    if task == "serve_clean_cup":
        metrics.llm_calls += 1  # planner
        plan = ["find_cup", "clean_cup", "put_on_desk"]
        return all(adapt(t, depth + 1, dmax, metrics) for t in plan)

    return False

m1 = Metrics()
assert react_find_cup(m1) is True
assert m1.llm_calls == 3  # 三次位置尝试

m2 = Metrics()
assert plan_execute_find_cup(m2) is False
assert m2.llm_calls == 2  # 一次规划，一次失败执行

m3 = Metrics()
assert adapt("serve_clean_cup", 1, 4, m3) is True
assert m3.llm_calls >= 6  # 先试再按需拆解
```

这个例子故意做得很小，但已经能看出三种控制方式：

- ReAct 把决策分散到每一步
- Plan-and-Execute 把任务打包成固定流程
- ADaPT 把“分解”延迟到失败时发生

真实工程里，代码通常会把 Planner 和 Executor 拆成两个 prompt 或两个模型。

ReAct 风格的代码骨架通常是：

```python
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI

def analyze_with_react():
    agent = create_csv_agent(
        ChatOpenAI(temperature=0),
        "sales_data.csv",
        verbose=True,
    )
    return agent.run("""
    1. Calculate the total sales
    2. Find the best performing product
    3. Generate a summary report
    """)
```

这种写法的直觉是：把整个目标交给一个会边想边调工具的执行器。优点是实现简单。缺点是“总销量”“最佳商品”“报告生成”三件事都在一条轨迹里反复混合。

Plan-and-Execute 风格的骨架则是：

```python
from langchain.agents import PlanAndExecute
from langchain.tools import PythonAstREPLTool

def analyze_with_plan_execute():
    agent = create_plan_and_execute_agent(
        llm=ChatOpenAI(temperature=0),
        tools=[PythonAstREPLTool(), CSVTool("sales_data.csv")]
    )
    return agent.run("""
    1. Calculate the total sales
    2. Find the best performing product
    3. Generate a summary report
    """)
```

如果把它写成更接近系统设计的伪代码，会更清楚：

```text
plan = planner(task)

for step in plan:
    result = executor(step, prior_results)
    if failed(result):
        plan = replan(task, completed_results, remaining_steps)
```

真实工程例子里，做销售报表通常还有几个额外问题：

- “最佳商品”按销量、销售额还是利润定义
- CSV 列名是否规范
- 是否存在空值、异常值、币种混合

ReAct 可能在这些问题上不断回到全局任务，导致一次三步流程触发很多次 API。Plan-and-Execute 会更容易把它们稳定成“清洗 -> 聚合 -> 排序 -> 生成报告”。ADaPT 则适合加入到“聚合失败”“结果为空”“字段缺失”这种局部异常路径里，让系统只在问题暴露时细拆补救。

---

## 工程权衡与常见坑

工程上最常见的错误，不是“选错模型”，而是“把错误控制流放错层”。

| 问题 | 影响 | 规避方式 |
| --- | --- | --- |
| ReAct 在长轨迹里反复重想 | token 成本高，响应抖动大 | 给每步设预算；限制最大循环次数；把稳定子流程前置成计划 |
| ReAct 局部贪心 | 看起来一直在推进，实际上偏离目标 | 增加阶段性检查点，而不是只看最终答案 |
| Plan-and-Execute 的步骤过粗 | 某一步失败后整条流程停住 | 让 planner 输出可执行粒度，而不是业务口号 |
| Plan-and-Execute 缺少重规划 | 一处异常导致全局重跑 | 保留中间状态，只重做失败段 |
| ADaPT 递归过深 | 规划成本反噬执行成本 | 设置 `d_max`；对子任务最小粒度设停止条件 |
| 失败判断不可靠 | 明明能做却误拆，明明失败却误判成功 | 把“成功判定”单独设计成结构化输出 |

对零基础工程师最重要的一点是：不要把“推理能力”理解成一个纯模型问题。很多时候，成败取决于系统有没有把失败限制在局部。

举一个数据分析里的真实坑。目标是“从 CSV 生成日报”。如果你用纯 ReAct，模型可能在读到空列后开始重复试探：

- 先重读 CSV  
- 再怀疑编码问题  
- 再尝试换列名  
- 再重新生成汇总

这类行为看上去“有思考”，实际上是没有结构化恢复路径。更好的做法是：

1. 先做稳定主计划  
2. 给每步定义输入输出约束  
3. 在失败时只对当前步局部拆解

ADaPT 的价值就在这里。它把“失败恢复”从整条任务级别，下沉到子任务级别。论文里还提到一个关键工程技巧：只传播成功子任务里的关键信息。白话说，就是不要把所有失败轨迹一股脑塞回上下文，否则你会把局部失败再次放大全局噪声。

---

## 替代方案与适用边界

这三种架构不是互斥关系，更像三种默认控制策略。

如果任务是“快速回答一个事实问题”或“简单查一次数据库”，直接用 ReAct 往往足够，因为规划本身就是额外成本。此时任务短、交互实时、路径不确定，逐步决策比提前列计划更自然。

如果任务是“生成周报”“做 CSV 分析”“跑固定审批流程”，Plan-and-Execute 通常更合适，因为流程是稳定的，提前拆步能显著减少重复上下文。

如果任务是“长链、局部难点未知、失败只会在执行中暴露”，ADaPT 或类似混合架构更合适。例如网页自动化、复杂数据清洗、跨系统操作、开放环境导航。它们的共同特点是：大框架能预先规划，但真正卡住的点只有在跑到那一步才知道。

一个简单的选型表如下：

| 条件 | 更适合的模式 |
| --- | --- |
| 任务很短，实时响应优先 | ReAct |
| 步骤明确，可提前列流程 | Plan-and-Execute |
| 任务很长，但只有少数步骤真正困难 | ADaPT |
| token 预算非常紧 | 先用 Plan-and-Execute，再为失败步加局部重规划 |
| 环境变化快，计划很快过时 | 以 ReAct 为主，增加轻量检查点 |

因此，工程上最常见的落地方式不是“纯 ReAct”或“纯 Plan-and-Execute”，而是分层混合：

- 顶层用 planner 给出阶段目标
- 阶段内部用 ReAct 做局部探索
- 当局部探索连续失败时，再触发 ADaPT 式递归拆解

这也是所谓 Hybrid Usage Strategy 的合理含义：不是把多个名词拼在一起，而是在不同复杂度层级使用不同控制粒度。

---

## 参考资料

- Prasad, A. et al. “ADAPT: As-Needed Decomposition and Planning with Language Models”. NAACL Findings 2024. https://aclanthology.org/2024.findings-naacl.264.pdf
- OnyxLab, “Understanding Agents”. https://www.onyxlab.ai/docs/getting-started/understanding-agents
- James Lee, “ReAct vs Plan-and-Execute: A Practical Comparison of LLM Agent Patterns”. https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9
- ADAPT OpenReview 版本页面。https://openreview.net/pdf/8be2e6f128fbd59c1096d365ec417d492e9ade81.pdf
