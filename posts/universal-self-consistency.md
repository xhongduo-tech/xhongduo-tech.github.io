## 核心结论

Universal Self-Consistency，简称 USC，可以理解为“先生成多份候选答案，再让模型在这些候选里选出最一致的一份”。这里的“一致”不是指字符串完全相同，而是指下面几件事同时成立：

1. 更贴合原始任务。
2. 内部更少自相矛盾。
3. 覆盖更多核心信息。
4. 与其他候选共享的稳定信息更多。
5. 臆造内容更少。

标准 Self-Consistency（SC）主要服务于“答案可以抽取并多数投票”的任务，例如数学题、逻辑题、可抽取最终选项的问答。假设 5 条推理最后都落到 `42`，那么直接对 `42` 投票即可。  
但摘要、翻译、开放式问答、代码解释这类任务没有唯一的标准字符串答案。两个高质量答案可能表达不同，却都合理。因此，标准 SC 里“抽取最终答案再投票”的机制会失效。

USC 的关键改动是：不再要求候选在字面上相同，而是让 LLM 直接在语义层面对候选进行比较和选择。  
论文《Universal Self-Consistency for Large Language Models》的结果表明，这种做法能把自一致性从封闭答案任务扩展到开放式生成任务。以论文报告的 PaLM 2-L 结果为例，USC 在 GovReport 长文摘要上把 ROUGE-1 从 `38.8` 提升到 `40.2`；在 TruthfulQA 上把真实性指标从 `62.1` 提升到 `67.7`。这说明 USC 不是“保证正确”，而是在不额外训练判别器、也不依赖人工标签的情况下，用一次选择步骤换取更稳的输出质量。

| 方法 | 适用任务 | 选择依据 | 是否依赖固定答案 | 典型结果 |
| --- | --- | --- | --- | --- |
| Greedy decoding | 所有生成任务 | 每一步取当前最高概率 token | 否 | 成本最低，但容易早早锁死在局部最优路径 |
| 标准 SC | 数学、可抽取答案 QA | 抽取最终答案后多数投票 | 是 | 对封闭答案任务有效 |
| USC | 摘要、翻译、开放式 QA、代码解释 | 模型对候选做语义比较与选择 | 否 | 对开放任务更适用 |

一个最小直觉例子是新闻摘要。  
不是只保留第一条输出，而是先采样 5 条摘要，再要求模型“从这 5 条里选出最一致、无矛盾、覆盖核心事实且最少臆造的一条”。最后把被选中的那条作为结果。  
这就是 USC 的基本工作流。

---

## 问题定义与边界

USC 解决的问题不是“如何生成更多答案”，而是：

> 当任务没有唯一标准答案时，如何在多条候选中挑出更可靠的一条。

这个问题在开放式生成任务里很常见，因为这类任务有两个结构性特征：

1. 输出不是唯一字符串。
2. 质量评价通常是语义性的，不是字符级的。

摘要是最典型的例子。原文讲的是同一件事，两个摘要可能完全不同，但都准确、完整、无矛盾。翻译也一样，“准确”不等于“逐字一致”。开放式问答更明显，同一个问题可以有不同但都合格的组织方式。因此，标准 SC 中“答案抽取 + 多数投票”的做法在这些场景里通常不成立，因为候选之间往往几乎不会字面一致。

USC 的输入通常包含三部分：

1. 原始任务 $x$
2. 候选集合 $S=\{y_1,\dots,y_K\}$
3. 选择提示 $P$

输出目标不是重新生成新答案，而是在集合 $S$ 中选出一个最优候选 $y^*$。

如果把它写成决策问题，可以表示为：

$$
y^* \in S,\qquad
y^* = \arg\max_{y \in S} \text{Score}(y \mid x, S, P)
$$

其中 `Score` 不是人工标注器打的分，而是 LLM 在给定任务、候选集和选择标准后，对每个候选隐式形成的偏好分数。

USC 的适用边界也很明确：

| 任务类型 | USC 是否适合 | 原因 | 主要瓶颈 |
| --- | --- | --- | --- |
| 长文摘要 | 适合 | 没有唯一答案，语义选择有效 | 候选太长时上下文成本高 |
| 机器翻译 | 适合 | 多种表达都可能正确 | 模型可能偏好表面流畅而非术语忠实 |
| 开放式 QA | 适合 | 可用一致性抑制明显幻觉 | 如果所有候选都错，只能选“错得最稳”的 |
| 数学推理 | 可用，但不是核心优势场景 | 也能做选择 | 若能直接抽取最终答案，标准 SC 更简单 |
| 代码解释/文档改写 | 适合 | 质量标准偏语义层面 | 若原始知识缺失，选择器无力补知识 |
| 高事实约束任务 | 有限适合 | 可提升稳定性 | 不能替代检索、规则校验和人工复核 |

这里有一个必须说明的边界：

> USC 是“无标签选择”，不是“无标签事实核验”。

两者差别很大。  
USC 不会访问外部世界，也不会自动查证事实。它只是利用同一模型在多个候选之间进行比较。如果 5 个候选都带着同一个错误前提，USC 仍然可能稳定地选出那个错误。它减少的是随机生成误差、局部表达失误和候选间质量波动，不是从根本上消灭事实错误。

可以把它理解成下面这张表：

| 问题 | USC 能否直接解决 | 原因 |
| --- | --- | --- |
| 多个候选中哪条更完整 | 能 | 比较任务是 USC 的强项 |
| 多个候选中哪条更少矛盾 | 能 | 模型能利用内部语义判断 |
| 候选是否符合原始任务要求 | 能 | 任务约束可直接写进选择提示 |
| 候选中的事实是否与外部世界一致 | 不能保证 | 没有外部证据时只能依赖模型内部共识 |
| 所有候选都错时如何修正 | 不能直接修正 | USC 只做选择，不做外部纠错 |

真实工程里最常见的用法是“稳定化开放生成”。  
例如客服知识库摘要，greedy 输出常见的问题不是完全胡说，而是漏掉限定条件，如“仅适用于企业版”“默认关闭，需人工开通”“4 月起新用户不再可用”。USC 的做法是先并行生成 5 到 8 条摘要，再按“覆盖关键限制条件、无矛盾、少臆造”进行选择。  
这通常比单次生成更稳，因为第二阶段的模型任务已经从“从零写一段话”变成了“比较几段现成文本并挑出更合格的一段”。

---

## 核心机制与推导

USC 的核心机制可以写成一个直接的公式：

$$
y^*=\arg\max_{y\in S} \mathrm{LLMSel}(y \mid P, S, x)
$$

含义是：给定原始任务 $x$、候选集合 $S$ 和选择提示 $P$，让模型对每个候选的“被选中概率”做条件判断，最后取概率最高的候选作为输出。

如果进一步写成概率形式，就是：

$$
p_i = p(y_i \text{ is best} \mid x, S, P), \qquad
y^* = y_{\arg\max_i p_i}
$$

这里的 $\mathrm{LLMSel}$ 可以理解成“模型在看到完整候选集合后，对某个候选给出的偏好分数”。  
它不是单独训练出来的分类器，而是直接复用了同一个 LLM 的理解、比较和判断能力。

从流程上看，USC 至少包含两个阶段：

$$
\text{Sample: } x \rightarrow S=\{y_1,\dots,y_K\}
$$

$$
\text{Select: } (x, S, P) \rightarrow y^*
$$

也可以把整体写成：

$$
y^* = \mathrm{Select}\bigl(x,\ \mathrm{Sample}_K(x),\ P\bigr)
$$

这三个公式对应的是三个工程动作：

1. 用采样生成多个候选。
2. 把候选集合和选择标准一起喂给模型。
3. 让模型输出其中最优候选的编号或文本。

### 玩具例子

原任务：

> 总结下面一句话：公司将在 4 月关闭旧版 API，企业客户可申请延期到 6 月。

候选如下：

- A：旧版 API 将在 4 月关闭，企业客户可申请延期到 6 月。
- B：公司将在 6 月关闭所有 API。
- C：旧版 API 会继续长期保留给企业客户使用。

如果让模型按“与任务一致、无矛盾、覆盖核心事实、避免臆造”来选，合理结果应当是 A。原因很直接：

| 候选 | 是否覆盖“4 月关闭旧版 API” | 是否覆盖“企业客户可延期到 6 月” | 是否与原文冲突 | 结论 |
| --- | --- | --- | --- | --- |
| A | 是 | 是 | 否 | 最优 |
| B | 否，且把范围改成“所有 API” | 否 | 是 | 错误 |
| C | 否 | 否 | 是，直接反向陈述 | 错误 |

如果模型的选择分布是：

$$
p(A \mid P,S)=0.6,\quad p(B \mid P,S)=0.3,\quad p(C \mid P,S)=0.1
$$

那么 USC 输出就是 A。

### 为什么这种方法通常有效

直觉上，USC 的有效性来自两层机制。

第一层是“采样暴露不稳定性”。  
如果某个事实只在少数候选里出现，说明模型对它把握不稳；如果多个候选反复保留同一事实，说明这部分信息更像模型内部的稳定共识。

第二层是“选择通常比生成更简单”。  
从空白开始写一段高质量摘要，本质是一个大搜索问题；从 5 条候选里挑最合格的一条，本质是一个比较问题。比较问题通常更容易，因为候选空间已经被第一阶段缩小了。

因此 USC 可以看成：

> 先用采样做探索，再用选择做压缩。

这也是它和 greedy decoding 的本质区别。  
greedy 一开始就走一条路径，中途如果偏了，后续很难补救；USC 则允许模型先探索多条路径，再利用第二次前向计算进行筛选。

### 与标准 SC 的关系

为了不把 USC 和标准 SC 混在一起，可以把两者并排看：

| 维度 | 标准 SC | USC |
| --- | --- | --- |
| 输入候选 | 多条推理路径 | 多条完整候选输出 |
| 聚合方式 | 抽取最终答案后多数投票 | 模型直接做语义选择 |
| 是否要求答案格式一致 | 是 | 否 |
| 对开放任务是否自然适用 | 否 | 是 |
| 失败模式 | 抽取错误、答案格式不统一 | 选择偏差、位置偏好、共错放大 |

一句话说清两者关系：  
USC 不是推翻标准 SC，而是把“自一致性”的思想从“可投票答案”推广到了“不可直接投票的自由文本”。

---

## 代码实现

工程上最常见的 USC 流程只有三步：

1. 用较高温度采样出 $K$ 个候选。
2. 把原任务和候选集合拼成选择提示。
3. 用温度 0 或接近 0 的方式做选择。

下面先给一个**可直接运行**的 Python 最小实现。  
它不依赖真实 LLM，而是用“任务关键词覆盖 + 矛盾惩罚 + 重复惩罚”模拟选择器，目的只是把 USC 的数据流跑通。

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    text = text.lower()
    return any(p.lower() in text for p in phrases)


@dataclass
class ScoreDetail:
    overlap: int
    repetition_penalty: int
    contradiction_penalty: int

    @property
    def total(self) -> int:
        return self.overlap - self.repetition_penalty - self.contradiction_penalty


def heuristic_score(task: str, candidate: str) -> ScoreDetail:
    task_tokens = set(tokenize(task))
    cand_tokens = tokenize(candidate)

    overlap = sum(1 for t in cand_tokens if t in task_tokens)
    repetition_penalty = len(cand_tokens) - len(set(cand_tokens))

    contradiction_penalty = 0

    # 针对这个玩具例子的少量规则，只用于演示“选择器会惩罚明显冲突”
    if contains_any(task, ["4月关闭"]) and contains_any(candidate, ["6月关闭所有api"]):
        contradiction_penalty += 4
    if contains_any(task, ["延期到6月"]) and contains_any(candidate, ["长期保留", "继续长期保留"]):
        contradiction_penalty += 4
    if contains_any(task, ["旧版api"]) and contains_any(candidate, ["所有api"]):
        contradiction_penalty += 2

    return ScoreDetail(
        overlap=overlap,
        repetition_penalty=repetition_penalty,
        contradiction_penalty=contradiction_penalty,
    )


def usc_select(task: str, candidates: List[str]) -> Tuple[str, List[ScoreDetail]]:
    if not candidates:
        raise ValueError("candidates must not be empty")

    details = [heuristic_score(task, c) for c in candidates]
    best_idx = max(range(len(candidates)), key=lambda i: details[i].total)
    return candidates[best_idx], details


if __name__ == "__main__":
    task = "总结：旧版API在4月关闭，企业客户可申请延期到6月。"
    candidates = [
        "旧版API在4月关闭，企业客户可申请延期到6月。",
        "公司将在6月关闭所有API。",
        "旧版API将长期保留给企业客户。",
    ]

    best, details = usc_select(task, candidates)

    for idx, (candidate, score) in enumerate(zip(candidates, details)):
        print(f"[{idx}] {candidate}")
        print(
            f"    overlap={score.overlap}, "
            f"repeat_penalty={score.repetition_penalty}, "
            f"contradiction_penalty={score.contradiction_penalty}, "
            f"total={score.total}"
        )

    print(f"\nselected: {best}")

    assert best == candidates[0]
```

这段代码可以直接运行，预期输出是第 0 条候选被选中。  
它不是 USC 的真实效果上限，只是把核心思想拆成了新手也能验证的形式：

1. 先有一组候选。
2. 再有一个独立的选择步骤。
3. 最终返回“候选中的一个”，而不是生成一条新文本。

### 接入真实 LLM 的可运行版本

下面给一个更接近实际工程的版本。  
它用一个统一接口 `llm_call(prompt, temperature)` 表示模型调用，方便替换成你自己的 OpenAI、Gemini、Claude 或本地模型 SDK。示例里不绑定具体厂商，因此代码本身可运行，但需要你自己提供 `llm_call` 的实现。

```python
from __future__ import annotations

from typing import Callable, List


LLMCall = Callable[[str, float], str]


def build_selection_prompt(task_prompt: str, candidates: List[str]) -> str:
    candidate_block = "\n".join(
        f"[{i}]\n{candidates[i]}\n"
        for i in range(len(candidates))
    )

    return f"""
你要执行的是“选择任务”，不是“重写任务”。

请根据原始任务，从候选中选出最一致的一条。判断标准按优先级从高到低如下：
1. 与原始任务最贴合，不偏题
2. 不包含明显矛盾或错误改写
3. 覆盖核心信息，不遗漏关键限定条件
4. 尽量少臆造原任务中没有的信息
5. 若质量接近，优先选择表达更清楚的一条

原始任务：
{task_prompt}

候选：
{candidate_block}

输出要求：
- 只输出一个阿拉伯数字编号
- 不要解释
- 不要重复候选内容
""".strip()


def parse_selected_index(text: str, n: int) -> int:
    text = text.strip()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < n:
            return idx
    raise ValueError(f"selector returned invalid index: {text!r}")


def run_usc(
    llm_call: LLMCall,
    task_prompt: str,
    k: int = 5,
    gen_temperature: float = 0.8,
    sel_temperature: float = 0.0,
) -> str:
    if k <= 0:
        raise ValueError("k must be positive")

    candidates = [
        llm_call(task_prompt, gen_temperature).strip()
        for _ in range(k)
    ]

    selection_prompt = build_selection_prompt(task_prompt, candidates)
    selected = llm_call(selection_prompt, sel_temperature)
    idx = parse_selected_index(selected, len(candidates))
    return candidates[idx]
```

### 如果你用 OpenAI 风格接口

下面给一个更具体的接线例子。  
这段代码假设你已经设置好 `OPENAI_API_KEY`，并安装了官方 SDK。它本身是完整可运行的，只需要你填入自己的模型名。

```python
from __future__ import annotations

from openai import OpenAI


client = OpenAI()


def llm_call(prompt: str, temperature: float) -> str:
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=temperature,
    )
    return resp.output_text.strip()


if __name__ == "__main__":
    task = """请总结这段通知：
旧版 API 将于 2026-04-01 下线，企业客户可申请延长使用期至 2026-06-30。
新客户即日起默认接入新版 API。"""

    result = run_usc(llm_call, task, k=5)
    print(result)
```

### 选择提示为什么要写得这么具体

新手常犯的错误是写：

> 从下面答案里选最好的一个。

这类提示太空，模型会自行定义“最好”，常见偏差包括：

1. 偏好更长的答案。
2. 偏好文风更流畅的答案。
3. 偏好排在前面的答案。
4. 偏好表面更像“标准答案”的措辞。

因此，USC 的提示词应该把“好”的含义拆开，并尽量转成可比较的标准。  
一个稳定的模板通常至少要写清下面几点：

| 选择维度 | 是否建议显式写入提示 | 原因 |
| --- | --- | --- |
| 与原始任务贴合 | 建议 | 防止候选偏题 |
| 无矛盾 | 建议 | 防止自相矛盾或反向改写 |
| 覆盖核心信息 | 建议 | 防止漏掉关键限制条件 |
| 尽量少臆造 | 建议 | 防止模型奖励花哨补写 |
| 语言优美 | 谨慎 | 容易压过真实性和忠实度 |
| 尽量详细 | 谨慎 | 容易偏向冗长答案 |

### 一个可直接复用的 USC 选择模板

```text
你要执行的是选择任务，不是重写任务。

请从候选中选出最一致的一条。判断标准如下：
1. 与原始任务最贴合
2. 不包含明显矛盾
3. 覆盖核心信息与关键限制条件
4. 尽量少引入原任务没有的信息
5. 若差异很小，优先选表达更清楚的一条

只输出最佳候选的编号，不要解释。
```

---

## 工程权衡与常见坑

USC 的收益来自“多采样 + 再选择”，成本也来自这里。  
和 greedy 相比，它至少多了 $K$ 次生成和 1 次选择，因此延迟和费用都会上升。

如果把单次生成成本记为 $C_g$，选择成本记为 $C_s$，那么 USC 的总成本近似为：

$$
C_{\text{USC}} \approx K \cdot C_g + C_s
$$

如果单次生成延迟记为 $T_g$，选择延迟记为 $T_s$，在串行调用下总延迟近似为：

$$
T_{\text{USC}} \approx K \cdot T_g + T_s
$$

如果候选生成可以并行，延迟则更接近：

$$
T_{\text{USC-parallel}} \approx \max(T_g) + T_s
$$

所以 USC 在工程里常见的优化不是“减少选择步骤”，而是：

1. 并行生成候选。
2. 控制 $K$ 的规模。
3. 限制候选长度。
4. 只对高价值请求启用 USC。

最常见的坑有四类：

| 问题 | 表现 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| 成本过高 | 延迟和费用明显上升 | 需要多次采样 | 把 `K` 控制在 3 到 8，先做小流量 A/B |
| 位置偏好 | 模型总爱选第一条或最后一条 | 排序泄漏偏差、提示过空 | 随机打乱候选顺序，做多轮评估 |
| 上下文爆炸 | 长摘要任务塞不下原文和所有候选 | 候选和源文都很长 | 先压缩候选、分段选择，或仅传摘要依据 |
| 一致但错误 | 多条候选共享同一错误事实 | 选择只依赖模型内部共识 | 叠加检索、规则校验或人工抽检 |

### 新手最容易踩的三个坑

#### 1. 误把 USC 当“事实校验器”

这是最危险的误解。  
USC 无法自动知道外部世界里的真相。它只能在已有候选中做相对比较。对法律、医疗、金融、生产变更公告等高风险场景，USC 最多只能作为第一层筛选，后面仍然要接：

1. 检索增强（RAG）
2. 规则校验
3. 关键字段比对
4. 人工复核

#### 2. 候选没有多样性

如果 5 个候选几乎一模一样，USC 的收益会很有限。  
这通常意味着第一阶段采样没有真正探索到不同路径。常见原因包括：

1. 温度太低。
2. Top-p 太保守。
3. 提示过度约束。
4. 模型本身对任务分布过于确定。

这时需要先检查第一阶段，而不是一味增加 `K`。  
因为“更多几乎相同的候选”不会显著提升选择质量，只会增加成本。

#### 3. 选择标准与业务标准不一致

例如你在企业翻译场景里真正关心的是：

1. 术语一致
2. 禁止漏译
3. 数字、日期、单位必须精确
4. 不改变主语和条件范围

但提示词只写了“选最自然、最通顺的一条”。  
那么模型就会按文风做判断，而不是按业务规则做判断。USC 本身没错，错的是你给它的目标函数。

### 一个真实工程例子：内部技术文档翻译

假设你在做英文技术文档到中文的翻译。  
greedy 输出常见的问题不是整段错误，而是术语前后不统一，例如：

- `instance` 一会儿译成“实例”
- 一会儿又译成“节点”
- `deprecation` 有时译成“弃用”
- 有时又译成“废弃”

USC 的做法可以是：

1. 先生成 5 个翻译候选。
2. 在选择提示里强调“术语一致、忠实原文、不遗漏条件、不改变数字和时态”。
3. 让模型只输出最佳候选编号。
4. 对最终选中的文本再跑术语表和关键字段比对。

这样做的效果通常是：  
语言稳定性明显上升，术语漂移减少；但如果源文本身包含歧义或事实冲突，USC 仍不能替代术语库和规则检查。

### 一个常用的上线策略

实际系统里不一定要对所有请求都启用 USC。更常见的是分层启用：

| 请求类型 | 是否建议启用 USC | 原因 |
| --- | --- | --- |
| 普通闲聊 | 通常不必 | 成本收益比低 |
| 长文摘要 | 建议 | 候选质量波动较大 |
| 面向用户的正式说明文 | 建议 | 稳定性要求高 |
| 高风险事实问答 | 可以启用，但要配合外部校验 | USC 只能做第一层稳态筛选 |
| 固定格式抽取 | 通常不必 | 规则解析或 constrained decoding 更直接 |

---

## 替代方案与适用边界

USC 不是唯一方案，只是实现成本相对低、迁移成本相对小的一种。  
它的主要优势是：不需要额外训练判别器，不需要人工标签，直接复用同一个模型即可。

| 方案 | 额外组件 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- | --- |
| Greedy | 无 | 最便宜、最快 | 容易锁死在单一路径，稳定性有限 | 低成本、低风险任务 |
| USC | 同一 LLM 再选择 | 无标签、实现简单、适合开放任务 | 成本高于 greedy，不能保证事实正确 | 摘要、翻译、开放式 QA、代码解释 |
| 多模型投票 | 多个模型 | 模型偏差可能互补 | 成本更高，系统更复杂 | 高价值输出 |
| 外部评估器 | verifier / reward model | 可定制评价标准，可长期稳定 | 需要标注数据、训练和维护 | 有成熟标注体系的团队 |
| 检索 + 校验 | 搜索、知识库或规则引擎 | 事实性更强，可追证据 | 系统链路复杂，工程维护重 | 高事实要求场景 |
| Constrained decoding | 约束语法/结构 | 对格式和字段精度强 | 不适合自由文本质量优化 | 抽取、结构化输出 |

USC 最适合的条件有三个：

1. 任务允许产生多条候选。
2. 模型具备基本的比较和筛选能力。
3. 质量问题更多来自“生成不稳”，而不是“知识根本缺失”。

它不太适合的情况也很清楚：

1. 输出极短且几乎无歧义，例如固定字段抽取。
2. 候选之间差异极小，采样没有有效多样性。
3. 任务核心是外部真实性，而模型内部共识不可靠。
4. 上下文预算极紧，无法容纳源任务和多个候选。
5. 响应时延要求极严，不能接受多次推理。

可以把 USC 放到一个更清晰的决策框架里：

| 如果你的主要问题是 | 更优先考虑 |
| --- | --- |
| 输出偶尔跑偏，但大多数候选里有好答案 | USC |
| 事实错误多，且需要外部依据 | 检索 + 校验 |
| 需要严格结构化输出 | 约束解码或规则抽取 |
| 团队有稳定偏好数据 | 外部评估器或 reward model |
| 预算极紧 | Greedy 或少量 rerank |

一句话概括边界：

> USC 擅长“在已有候选里选出更稳的一条”，不擅长“证明哪条一定为真”。

---

## 参考资料

| 资料 | 内容 | 用途 |
| --- | --- | --- |
| Google DeepMind: Universal Self-Consistency with Large Language Models | 论文官方页面与摘要，说明 USC 的目标是用 LLM 选择多个候选中最一致的答案，覆盖数学推理、长文摘要和开放式问答等任务 | 用于确认 USC 的定义、问题设定和整体适用范围 https://deepmind.google/research/publications/universal-self-consistency-with-large-language-models/ |
| OpenReview PDF: Universal Self-Consistency for Large Language Models | 论文正文与实验表格；其中 GovReport 表格给出 ROUGE-1 从 `38.8` 到 `40.2`，TruthfulQA 表格给出 PaLM 2-L 的 truthfulness 从 `62.1` 到 `67.7` | 用于核对本文引用的核心实验数字与公式化描述 https://openreview.net/pdf?id=LjsjHF7nAN |
| arXiv: Universal Self-Consistency for Large Language Model Generation | 论文预印本版本，便于查阅方法细节、提示构造、基线设置与实验任务范围 | 用于补充 OpenReview 之外的公开版本入口 https://arxiv.org/abs/2311.17311 |
| Learn Prompting: Universal Self-Consistency | 面向实现的教程说明，给出 USC 的直观流程、提示模板和与标准 SC 的区别 | 用于补充面向工程实现的解释和新手友好示例 https://learnprompting.org/docs/advanced/ensembling/universal_self_consistency |
