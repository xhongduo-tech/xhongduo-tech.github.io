## 核心结论

多Agent的动态组队，本质上是把“先定岗位，再塞任务”改成“先看任务，再临时组队”。这里的“Agent”可以先理解成一个带有特定能力偏好的执行单元，比如偏分析、偏编程、偏总结、偏检索。系统先把任务转成需求向量，再把每个Agent转成能力向量，最后用相似度决定谁进入主角色。

静态角色分配的问题不在于“不能工作”，而在于它默认任务稳定。现实工程里，任务结构会变：上午是日志分析，下午是故障总结，晚上可能变成补代码和写公告。如果还让固定“摘要Agent”“编程Agent”“分析Agent”轮流上场，就会产生两类成本：一类是上下文切换成本，另一类是低匹配角色带来的冗余推理成本。

一个适合初学者理解的玩具例子是：同一条流水线今天同时收到“数据分析”“会议摘要”“修复脚本”三类任务。控制器不是固定把“摘要任务”交给某个永远负责摘要的角色，而是先从任务关键词生成向量，再去能力库里挑选四个最匹配的Agent。这样明天如果来了“SQL性能诊断”这种新任务，只要能力向量足够接近，就能快速组出新队伍，不需要先人工重画组织结构。

| 维度 | 静态团队 | 动态组队 |
| --- | --- | --- |
| 上下文切换成本 | 角色固定，常把无关历史指令一并带入 | 只加载当前角色所需上下文 |
| 任务覆盖率 | 对未预设任务覆盖差 | 可按需求向量临时扩展 |
| 扩展时延 | 新任务常要人工改流程 | 通过匹配和补员快速接入 |

结论可以压缩成一句话：只有当能力向量和需求向量足够接近时，Agent才应进入主角位；否则就补员、拆任务，或者让人类介入，而不是强行让低匹配角色“凑数”。

---

## 问题定义与边界

问题定义很直接：任务类型是流动的，角色能力是有限的，系统需要在运行时决定“谁做主角、谁做辅助、什么时候换人”。这里的“需求向量”可以理解成任务需要哪些能力、各需要多少；“能力向量”可以理解成某个Agent在不同能力维度上的擅长程度。

设任务需求向量为：

$$
T = [t_1, t_2, \dots, t_n], \quad t_k \in [0,1]
$$

设第 $i$ 个Agent的能力向量为：

$$
A_i = [a_{i1}, a_{i2}, \dots, a_{in}], \quad a_{ik} \in [0,1]
$$

这些分量通常落在 $[0,1]$ 区间，表示某项能力的相对强度。比如在一个最小系统里，可以只定义三维：分析、编程、总结。若一个任务更偏分析和编程，那么它的需求向量可能是 $T=[0.6,0.4,0.1]$。这句白话是：这个任务主要需要分析能力，其次需要编程，几乎不需要长篇总结。

边界同样重要：

| 边界项 | 允许 | 不允许 |
| --- | --- | --- |
| 角色调整 | 运行时重写角色描述 | 任务变化后仍沿用旧角色 |
| 主角色分配 | 只给高相似度Agent | 低相似度Agent强行顶主位 |
| 覆盖不足处理 | 补员、拆子任务、人工接手 | 假设现有团队总能覆盖 |
| Prompt管理 | 重写输入约束和示例 | 新旧角色指令并存冲突 |

对零基础读者，可以把它看成“需求清单匹配能力表”。例如任务摘要要求是“分析：0.6，编程：0.4”，系统就去能力表里找最接近的Agent。如果最高匹配都低于阈值，就说明当前库里没人足够合适，此时不该硬派，而该补员或人工处理。

因此，这套方法的边界不是“系统能自动解决一切”，而是“系统能可靠地识别什么时候自己能做，什么时候该换队、拆分或交还给人”。

---

## 核心机制与推导

动态组队的核心是余弦相似度。它衡量两个向量方向是否接近。这里的“方向接近”可以白话理解为：任务需要的能力结构，和某个Agent擅长的能力结构，是否像同一种形状。

标准化后，任务向量 $T$ 和能力向量 $A_i$ 的匹配度定义为：

$$
\text{sim}(T, A_i) = \frac{T \cdot A_i}{\|T\| \|A_i\|}
$$

其中：

$$
T \cdot A_i = \sum_{k=1}^{n} t_k a_{ik}
$$

当 $\text{sim} \to 1$ 时，说明两者方向几乎一致，即任务重点和能力重点高度重合；当它接近 0 时，说明能力结构与需求结构差异很大。

题目给出的数值演示可以直接展开。若：

$$
T=[0.6,0.8], \quad A=[0.7,0.6]
$$

则点积为：

$$
T \cdot A = 0.6 \times 0.7 + 0.8 \times 0.6 = 0.42 + 0.48 = 0.9
$$

模长为：

$$
\|T\|=\sqrt{0.6^2+0.8^2}=1
$$

$$
\|A\|=\sqrt{0.7^2+0.6^2}=\sqrt{0.85}
$$

所以：

$$
\text{sim}(T,A)=\frac{0.9}{1 \times \sqrt{0.85}} \approx 0.976
$$

这大于 0.8，可以直接进入主角色。若另一个Agent为：

$$
A'=[0.2,0.3]
$$

则：

$$
\text{sim}(T,A')=\frac{0.6 \times 0.2 + 0.8 \times 0.3}{1 \times \sqrt{0.13}}
=\frac{0.36}{\sqrt{0.13}} \approx 0.33
$$

这类低值意味着“不能胜任主角”。系统此时应触发两个动作之一：要么拆出更小的子任务重新匹配，要么调用Prompt Coach重写角色定义后再试。这里的“Prompt Coach”可以理解成一个专门维护角色说明书的Agent，负责把“你是谁、你能看什么、你该怎么输出”写清楚。

机制链条通常是这样的：

1. 控制器提取任务特征，生成需求向量。
2. 系统在能力库中计算相似度并排序。
3. 只有 $\text{sim}>0.8$ 的Agent进入主角色。
4. 低于阈值但仍有局部价值的Agent，只做补员或工具位。
5. 如果反馈显示延迟变高、错误率上升、出现新子任务，重新匹配并重写Prompt。
6. 若新任务在现有能力空间里找不到高匹配Agent，则触发人工干预或创建新角色模板。

这比静态角色更像“运行时组织结构”，而不是固定编制表。

真实工程例子可以看一个产品发布周的流水线。白天有用户调研摘要，下午有线上Bug triage，晚上要整理修复说明和对外文档。第一批任务偏总结，第二批偏代码定位，第三批偏结构化写作。动态组队会在每一批次重新算需求向量：调研摘要时把总结型Agent放到主位；Bug triage时把分析和编程型Agent推到前面；输出公告时再让结构化表达能力强的Agent成为主角色。队伍成员可以重叠，但岗位不应固定。

---

## 代码实现

实现上可以分四层：任务解析、相似度匹配、结果缓存与反馈采集、角色重写与补员。最小可用版本不复杂，核心是把“选人”和“改角色”拆开，不要混成一个黑盒。

下面是一个可运行的 Python 玩具实现，包含相似度计算、主角色筛选和低匹配时触发重建逻辑：

```python
from math import sqrt
from typing import List, Dict, Tuple

THRESHOLD = 0.8

agents = [
    {"name": "analyst", "vector": [0.9, 0.3, 0.6]},   # 分析、编程、总结
    {"name": "coder", "vector": [0.4, 0.95, 0.3]},
    {"name": "writer", "vector": [0.5, 0.2, 0.95]},
    {"name": "generalist", "vector": [0.7, 0.7, 0.7]},
]

def cosine_similarity(task: List[float], ability: List[float]) -> float:
    dot = sum(t * a for t, a in zip(task, ability))
    norm_t = sqrt(sum(t * t for t in task))
    norm_a = sqrt(sum(a * a for a in ability))
    if norm_t == 0 or norm_a == 0:
        return 0.0
    return dot / (norm_t * norm_a)

def sort_by_similarity(task_vector: List[float], pool: List[Dict]) -> List[Tuple[str, float]]:
    scored = []
    for agent in pool:
        sim = cosine_similarity(task_vector, agent["vector"])
        scored.append((agent["name"], sim))
    return sorted(scored, key=lambda x: x[1], reverse=True)

def rebuild_roles(task_vector: List[float]) -> Dict:
    return {
        "action": "rebuild",
        "reason": "top_similarity_below_threshold",
        "task_vector": task_vector,
        "need_prompt_coach": True,
        "need_human_review": True,
    }

def select_team(task_vector: List[float], top_k: int = 3) -> Dict:
    ranked = sort_by_similarity(task_vector, agents)
    top_name, top_sim = ranked[0]
    if top_sim > THRESHOLD:
        main_roles = [item for item in ranked if item[1] > THRESHOLD][:top_k]
        support_roles = [item for item in ranked if item[1] <= THRESHOLD][:top_k]
        return {
            "action": "select",
            "main_roles": main_roles,
            "support_roles": support_roles,
        }
    return rebuild_roles(task_vector)

# 玩具例子：偏分析+总结
task_1 = [0.8, 0.2, 0.7]
result_1 = select_team(task_1)
assert result_1["action"] == "select"
assert len(result_1["main_roles"]) >= 1

# 低匹配例子：若能力库覆盖不足，则重建
limited_agents = [{"name": "weak", "vector": [0.1, 0.1, 0.1]}]
score = sort_by_similarity([0.9, 0.8, 0.7], limited_agents)[0][1]
assert score < THRESHOLD
```

如果把它放到真实工程里，通常还要补两块逻辑。

第一块是缓存。相似任务不需要每次完整重算，可以缓存“任务类型 -> 已验证团队模板”的映射，但缓存不能跳过阈值检查，因为运行时反馈可能让原团队失效。

第二块是Prompt Coach。它不直接做业务任务，而是负责角色更新。例如原任务是“你是分析师，请总结日志问题”，后来切到“你是代码评审，请给出补丁建议”，如果不重写Prompt，模型会同时背着两套身份指令，输出就会摇摆。Prompt Coach要做的事包括：归档旧角色描述、注入新输入约束、更新示例、声明输出格式。

一个更接近生产的调度流程可以写成：

```python
def dispatch(task):
    task_vector = parse_task_to_vector(task)
    team = select_team(task_vector)

    if team["action"] == "rebuild":
        new_role_spec = prompt_coach_rewrite(task)
        sub_tasks = maybe_split_task(task, new_role_spec)
        return assign_or_escalate(sub_tasks)

    outputs = run_agents(task, team["main_roles"], team["support_roles"])
    feedback = collect_feedback(outputs)

    if feedback["quality_drop"] or feedback["latency_spike"]:
        rewritten = prompt_coach_rewrite(task, feedback=feedback)
        return dispatch(rewritten)

    return outputs
```

这里最关键的不是代码长短，而是职责清晰：匹配负责选人，Prompt Coach负责改角色，反馈模块负责判断是否需要再组织一次团队。

---

## 工程权衡与常见坑

动态组队不是“比静态更高级”这么简单，它是用更高的调度复杂度，换更好的任务适应性。如果场景本来就高度重复，引入它可能得不偿失。但在多样化流水线中，不引入动态机制，问题往往会在上下文和冲突指令上爆出来。

一个典型坑是把静态角色留下来的Prompt继续叠加。比如上午系统里某个Agent的身份还是“日志分析师”，下午任务切成“代码评审员”，如果只是往后追加一句新要求，而不归档旧说明，就会同时出现“先总结现象”和“先提出补丁”的竞争指令。这不是模型笨，而是输入本身互相打架。

另一个坑是忽略阈值。很多系统为了“提高利用率”，会把前几个相似度最高的Agent都塞进主团队。但“排第1”不等于“足够匹配”。如果第一名只有 0.62，说明整个能力库都不适合，正确动作是重建或人工接手，而不是让一个不合格的Agent担任主角。

| 常见坑 | 后果 | 规避策略 |
| --- | --- | --- |
| 静态角色长期不变 | 旧上下文污染新任务 | 用动态匹配重新定义主角色 |
| 忽略相似度阈值 | 低匹配Agent进入主位 | 只在 `sim > 0.8` 时分配主角色 |
| 未更新Prompt | 新旧身份冲突，输出摇摆 | 由Prompt Coach自动重写 |
| 不留人工入口 | 低覆盖任务反复失败 | 设置补员与人工介入按钮 |
| 反馈不回流 | 团队失效后仍继续运行 | 用质量与时延指标触发重组 |

真实工程里还有一个常见权衡：向量维度定义得越细，匹配越精确，但标注和维护成本越高。比如只用“分析/编程/总结”三维，很容易上手，但遇到“检索可靠性”“长上下文保持”“结构化输出稳定性”时就不够了；如果扩到十几维，效果更细，但你需要持续维护能力画像，否则向量会很快失真。

因此，工程上通常建议从少量稳定维度起步，等到误匹配案例积累够多，再增量扩维，而不是一开始就建一个庞大但无人维护的能力本体。

---

## 替代方案与适用边界

纯静态角色并没有过时，它只是适用面更窄。若任务高度重复、输入格式稳定、错误代价可控，例如每周固定生成经营报表、固定摘要客服工单，那么固定的“提取Agent -> 汇总Agent -> 审核Agent”管道足够简单，也更容易维护。

但当任务种类爆炸、反馈频繁、输入结构不稳定时，静态方案会越来越像硬编码。比如新产品上线周，需要同时处理用户调研、线上告警、Bug triage、修复说明和FAQ文档。这些任务共享部分上下文，但所需主能力不同。如果仍用固定角色流水线，系统会不断携带无关指令和无关示例，最终拖慢响应并降低质量。动态组队在这种场景下更合适，因为它允许同一批次里临时插入适配能力。

| 方案 | 任务多样性 | 上下文重建频率 | 调度复杂度 | 适用边界 |
| --- | --- | --- | --- | --- |
| 静态角色 | 低到中 | 低 | 低 | 周期性、格式稳定任务 |
| 动态组队 | 中到高 | 高 | 中到高 | 任务变化快、反馈强依赖场景 |

还可以再加一个替代思路：单Agent加工具调用。它的优点是系统简单，缺点是把“分工”压回一个大Prompt里，任务一复杂就容易出现规划、执行、校验混在一起的问题。若你的系统还处在验证期，单Agent是合理起点；一旦开始出现跨技能冲突和明显的上下文污染，就该考虑多Agent动态组队。

适用边界可以概括为：

1. 任务维度有限、输入高度固定时，静态管道足够。
2. 任务类型多、反馈频繁、需要运行时切换角色时，动态组队更稳。
3. 若能力库长期覆盖不到高相似度候选，说明问题不在调度，而在能力建模或Agent供给本身，此时必须补员或人工接手。

---

## 参考资料

1. Adaptive In-conversation Team Building for LLM Agents, arXiv 2405.19425  
   支撑观点：多Agent团队可在对话过程中按任务变化自适应重组，是本文“动态组队优于静态编制”的理论基础。

2. Decentralized adaptive task allocation for dynamic multi-agent systems, *Scientific Reports*, 2025  
   支撑观点：动态环境下的任务分配需要持续吸收反馈，而不是一次性绑定角色，支持本文的运行时重组与反馈回流机制。

3. Dynamic capability-to-task matching framework, Qeios / preprint, 2025  
   支撑观点：能力向量与任务需求向量的匹配可用余弦相似度建模，适合解释本文的 $ \text{sim}(T,A_i) $ 与阈值筛选逻辑。

4. Prompt rewriting / meta prompting engineering practices（如 Cobus Greyling、Rediminds 等实践资料）  
   支撑观点：角色切换不能只换名字，还要重写角色描述、输入约束和示例，支持本文的 Prompt Coach 工程做法。

对初学者来说，最值得先复现的不是完整多Agent平台，而是三件事：把任务转成向量、算相似度、在低于阈值时重写角色而不是硬派任务。只要这三步跑通，动态组队就已经有了最小工程闭环。
