## 核心结论

Few-shot CoT（少样本思维链，意思是“在提示里先给几道带解题过程的示范题，再让模型照着推理”）的效果，往往不取决于“有没有示例”，而取决于“示例是怎么选出来的”。

在多步推理任务里，随机放几个示例通常不是最优做法。更稳定的策略是把示例选择拆成三个维度：

| 维度 | 白话解释 | 作用 | 典型做法 |
| --- | --- | --- | --- |
| 相似度 | 跟当前问题像不像 | 保证示例相关 | 向量检索、kNN |
| 复杂度 | 示范推理链够不够完整 | 教模型走完整推理路径 | 按推理步数或链长度排序 |
| 多样性 | 示例之间是否过于重复 | 防止模型只学到一种局部模板 | 聚类、去重、分层采样 |

核心判断可以写成一句话：**先保证“像”，再优先“复杂”，最后控制“不重复”**。  
这比“纯随机”更接近工程上真正可复用的提示设计。

一个经常被引用的结果来自 Complexity-Based Prompting：在 GSM8K 数学推理数据集上，基线方法约为 74.4%，复杂度驱动的示例选择可到 82.6%，再加复杂性一致性投票可到 82.9%。这说明一个关键事实：对于需要多步推理的任务，**能示范完整推理结构的例子，比随手选的例子更可靠**。

---

## 问题定义与边界

问题可以表述为：

给定一个候选示例库 $E=\{e_1,e_2,\dots,e_n\}$，以及一个新问题 $q$，在上下文窗口有限的前提下，如何选出 $K$ 个示例放进 prompt，使模型在当前问题上的推理正确率最高？

这里有三个现实边界。

第一，**上下文是稀缺资源**。大模型的上下文窗口虽然越来越大，但在真实系统里，系统指令、用户输入、工具结果、输出格式约束都会占 token。示例不是越多越好，而是越“值钱”越好。

第二，**示例不能只看相似度**。只用 kNN（k-nearest neighbors，最近邻检索，意思是“先找语义上最接近当前问题的例子”）容易出现一类问题：检索出来的题都很像，但推理结构几乎相同，模型学到的是一种局部套路，而不是解决这类问题的完整思路。

第三，**示例也不能只看复杂度**。如果一味塞最复杂的解法，可能出现两种副作用：
1. 示例太长，挤占上下文，反而减少有效信息密度。
2. 示例难度高于当前输入太多，模型会模仿表面形式，但未必能稳定复用其推理结构。

所以这类问题的边界不是“找最难题”，而是找一组满足下面条件的示例：

$$
\text{Good Examples} = \text{Relevant} \cap \text{Sufficiently Complex} \cap \text{Non-redundant}
$$

也就是：**相似但不重复，复杂但不过载**。

下面给一个玩具例子。

当前问题：  
“小明买了 3 支笔，每支 4 元，又买了 2 本本子，每本 5 元，一共花多少钱？”

候选示例库里有四道题：

| 示例 | 与当前题相似度 | 推理步数 | 是否适合 |
| --- | --- | --- | --- |
| 买 2 个苹果，每个 3 元，总价多少 | 高 | 1 | 过于简单 |
| 买笔和本子，分别单价与数量不同，求总价 | 高 | 3 | 适合 |
| 先打折再满减再算税，求最终价格 | 中 | 6 | 可能过复杂 |
| 计算长方形面积 | 低 | 2 | 不相关 |

如果只能放 2 个示例，最合理的选择通常不是“最像的两个”，也不是“最复杂的两个”，而是“最像且复杂度足够”的那个，再配一个结构上略有差异但仍相关的例子。

真实工程里也是一样。比如一个金融问答系统要处理“分期、利率、手续费、提前还款”类推理问题，单靠相似度检索会得到很多“利率题”，但未必覆盖“先算本金余额，再算手续费，再判断违约金”这种完整推理链。系统需要的是“相关题型中的高信息量示例”，不是“相关但重复的例子堆”。

---

## 核心机制与推导

复杂度驱动的 CoT 示例选择，核心机制可以分成两段：**选示例**和**选输出**。

### 1. 选示例：优先复杂推理链

设一个候选示例为 $e_i$，它的复杂度记作 $c(e_i)$。  
最朴素的定义就是“推理步数”，也可以用 reasoning token 数、句子数、显式中间结论数来近似。

$$
c(e_i) = \text{number of reasoning steps in } e_i
$$

如果先用相似度检索得到一批候选集 $N(q)$，那么示例池可以这样构造：

$$
P(q) = \text{top-}K \{ e_i \in N(q) \mid c(e_i) \}
$$

意思是：  
先找与当前问题最相关的一小批，再从里面选复杂度更高的示例进入 prompt。

为什么这有效？因为 CoT 示例不仅在“告诉模型答案格式”，更在“演示中间状态如何展开”。复杂示例通常包含更多显式推理节点，例如“先求部分和，再代入约束，再做单位转换”。这些节点越完整，模型越容易把问题分解成子步骤，而不是直接猜答案。

### 2. 选输出：只让复杂推理链参与投票

第二步不是选输入，而是选模型生成的输出。

假设对同一个问题采样出 $m$ 条推理链：

$$
R = \{r_1, r_2, \dots, r_m\}
$$

每条链也能定义复杂度 $c(r_j)$。  
如果直接做 self-consistency（自一致性投票，意思是“多采样几次，让多数答案获胜”），会把一些过短、跳步严重、靠猜得到的答案也计入投票。复杂性一致性做的是再加一道筛选：

$$
\hat{y} = \text{MajorityVote}\big(\{ \text{ans}(r_j) \mid c(r_j) \ge \theta \}\big)
$$

其中 $\theta$ 是复杂度阈值。只有推理链足够完整的输出，才有资格进入最终投票。

这背后的直觉并不神秘。对于多步推理题，错误答案往往来自两类链：

1. 太短，几乎没展开中间步骤。
2. 虽然写了过程，但关键约束被跳过。

复杂性筛选的作用，就是把“看起来像推理，实际上是压缩猜测”的输出排除掉。

### 玩具例子

问题：  
“一个数加 7 得 19，这个数是多少？”

模型采样出 4 条链：

| 链 | 推理内容摘要 | 步数 | 最终答案 |
| --- | --- | --- | --- |
| r1 | 19-7=12 | 1 | 12 |
| r2 | 设原数为 x，则 x+7=19，所以 x=12 | 2 | 12 |
| r3 | 猜测是 13，因为 13 接近 19 | 1 | 13 |
| r4 | 先移项得 x=19-7，再算得 12 | 2 | 12 |

若设阈值 $\theta=2$，则只有 r2 和 r4 参与投票，结果稳定为 12。  
如果不过滤，r1、r3 也会进入投票，虽然这个例子里不一定翻车，但在复杂题里噪声会显著增加。

### 真实工程例子

在教育答题系统里，用户输入一道分数应用题。系统可以这样做：

1. 先用 embedding 检索出 6 道最相似题。
2. 统计每道示例的推理链步数，按复杂度降序排序。
3. 去掉结构重复的题，只保留 3 道。
4. 用这 3 道示例组装 few-shot CoT prompt。
5. 对当前题采样 5 条推理链。
6. 只保留步数大于阈值的链做多数投票。

这个流程的目标不是“让模型输出更长”，而是“让模型更可能走过必要的中间状态”。长度只是近似指标，真正要的是结构完整度。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，用来演示“相似度检索 -> 复杂度排序 -> 多样性去重 -> 构造 prompt -> 复杂性投票”的骨架。这里不用真实向量库，而是用词项重叠模拟相似度，重点是流程。

```python
from collections import Counter, defaultdict
import re

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def reasoning_steps(chain: str) -> int:
    # 用句号、分号、换行近似“推理步数”
    parts = [p.strip() for p in re.split(r"[。\n;；]+", chain) if p.strip()]
    return len(parts)

def structure_signature(example):
    # 用 tags 或首个关键词近似“结构类别”
    return example["tag"]

def select_examples(query, pool, knn_k=6, final_k=3):
    scored = []
    for ex in pool:
        sim = jaccard(query, ex["question"])
        comp = reasoning_steps(ex["chain"])
        scored.append({**ex, "similarity": sim, "complexity": comp})

    # 第一步：先按相似度取候选
    knn = sorted(scored, key=lambda x: x["similarity"], reverse=True)[:knn_k]

    # 第二步：再按复杂度降序
    knn = sorted(knn, key=lambda x: (x["complexity"], x["similarity"]), reverse=True)

    # 第三步：做简单多样性去重，每种结构最多取 1 个
    picked = []
    used = set()
    for ex in knn:
        sig = structure_signature(ex)
        if sig in used:
            continue
        picked.append(ex)
        used.add(sig)
        if len(picked) == final_k:
            break
    return picked

def build_prompt(examples, query):
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"示例{i}\n问题：{ex['question']}\n推理：{ex['chain']}\n答案：{ex['answer']}"
        )
    blocks.append(f"现在回答：\n问题：{query}\n推理：")
    return "\n\n".join(blocks)

def complexity_vote(sampled_chains, theta=2):
    valid_answers = []
    for chain, answer in sampled_chains:
        if reasoning_steps(chain) >= theta:
            valid_answers.append(answer)

    if not valid_answers:
        valid_answers = [answer for _, answer in sampled_chains]

    winner, count = Counter(valid_answers).most_common(1)[0]
    return winner, count

pool = [
    {
        "question": "买3支笔每支4元，再买2本本子每本5元，一共多少钱？",
        "chain": "先算笔的钱：3乘4等于12。再算本子的钱：2乘5等于10。最后相加得22。",
        "answer": "22",
        "tag": "sum_of_groups",
    },
    {
        "question": "有24个苹果，平均分给6人，每人多少个？",
        "chain": "总数24。人数6。24除以6等于4。",
        "answer": "4",
        "tag": "division_only",
    },
    {
        "question": "商品原价100元，先打8折，再减10元，最后多少钱？",
        "chain": "先算折后价：100乘0.8等于80。再减10。得到70。",
        "answer": "70",
        "tag": "discount_then_minus",
    },
    {
        "question": "长方形长8宽3，面积是多少？",
        "chain": "面积等于长乘宽。8乘3等于24。",
        "answer": "24",
        "tag": "geometry_area",
    },
    {
        "question": "买4盒牛奶每盒6元，再买1袋面包8元，一共多少钱？",
        "chain": "先算牛奶：4乘6等于24。再加面包8。得到32。",
        "answer": "32",
        "tag": "sum_of_groups",
    },
    {
        "question": "本金1000元，年利率5%，一年后本息合计多少？",
        "chain": "先算利息：1000乘0.05等于50。再加本金1000。得到1050。",
        "answer": "1050",
        "tag": "interest",
    },
]

query = "买2支钢笔每支7元，再买3本练习册每本4元，一共多少钱？"
examples = select_examples(query, pool, knn_k=4, final_k=3)
prompt = build_prompt(examples, query)

assert len(examples) == 3
assert "现在回答" in prompt

sampled = [
    ("先算钢笔：2乘7等于14。再算练习册：3乘4等于12。最后相加得26。", "26"),
    ("14加12等于26。", "26"),
    ("猜测答案是24。", "24"),
]
winner, count = complexity_vote(sampled, theta=2)

assert winner == "26"
assert count >= 1
print("chosen:", [e["question"] for e in examples])
print("vote:", winner)
```

上面这段代码体现了四个工程动作：

| 步骤 | 作用 | 对应函数 |
| --- | --- | --- |
| 相似度检索 | 先把题型对齐 | `select_examples` 中的 `jaccard` 排序 |
| 复杂度排序 | 优先保留完整推理链 | `reasoning_steps` |
| 多样性去重 | 防止示例重复 | `structure_signature` |
| 复杂性投票 | 输出阶段再过滤一次 | `complexity_vote` |

在真实系统里，这个骨架通常会替换成：
- 用 embedding 模型做相似度检索，而不是 Jaccard。
- 用标注好的推理步数、链长度、树深度或专家规则做复杂度估计。
- 用聚类或 MMR（最大边际相关性，意思是“同时兼顾相关性和去重”）做多样性控制。
- 用解析器从生成结果中抽取最终答案，再做投票。

---

## 工程权衡与常见坑

最常见的误区，不是“不会做复杂选择”，而是**只做单一维度优化**。

### 坑 1：只做 kNN，结果都很像

只按相似度检索，确实能保证“相关”，但常见副作用是示例高度同质。模型会被一种固定表述牵着走，遇到结构稍微变化的问题就容易失稳。

比如教育系统里，当前题是“先算单价乘数量，再做总和”。如果 6 个近邻题全都是“两类商品相加”，模型可能在面对“先打折再相加”的题时直接套错模板。

### 坑 2：只做随机，覆盖看似广，信息密度却低

随机抽样的好处是简单，也常被当作基线。但随机的核心问题是不可控。它可能抽到一组都很简单的例子，也可能抽到与当前题几乎无关的例子。对于多步推理任务，这种波动通常直接体现在准确率抖动上。

### 坑 3：只追求“最复杂”

复杂度高不等于一定好。一个 12 步的示例，如果其中 7 步是冗余格式、解释性废话或者超出当前问题所需的旁枝步骤，它占掉的上下文，可能比它提供的推理信息更多。

工程上真正要控制的是“有效复杂度”，不是绝对长度。

### 坑 4：复杂度指标定义得太粗

把 token 长度直接当复杂度，容易把啰嗦误判成复杂。更可靠的做法是混合指标，例如：

$$
c(e) = \alpha \cdot \text{step\_count} + \beta \cdot \text{constraint\_count} + \gamma \cdot \text{operation\_variety}
$$

其中：
- `step_count` 表示显式推理步数。
- `constraint_count` 表示用了多少条件约束。
- `operation_variety` 表示是否涉及加减乘除、比较、单位换算等多种操作。

### 坑 5：忽略模型容量

小模型和大模型，对示例复杂度的承受能力不一样。模型越弱，越容易出现“看见长链就模仿形式，但无法执行逻辑”的问题。此时与其塞 3 个超复杂示例，不如用 2 个中等复杂示例加 1 个结构说明。

下面这个表可以作为快速判断：

| 策略 | 稳定性 | 覆盖度 | 成本 | 常见问题 |
| --- | --- | --- | --- | --- |
| 仅随机 | 低 | 中 | 低 | 波动大，相关性差 |
| 仅 kNN | 中 | 低到中 | 中 | 同质化，容易回路化 |
| 仅高复杂度 | 中 | 低 | 中 | 可能过长，超出模型容量 |
| 相似度+复杂度+多样性 | 高 | 高 | 中到高 | 需要维护示例池和评分规则 |

### 真实工程例子

一个对话式教育系统，后端维护 5000 道已标注题。每次新题到来时：

1. 先检索 6 道最相似题。
2. 将它们分成低、中、高三档复杂度。
3. 选 1 个高、1 个中、1 个结构不同但仍相关的例子。
4. 如果总 token 超预算，优先删掉解释性废话，而不是删掉中间推理节点。
5. 输出后采样 3 到 5 条链，只让复杂度达标的链参与最终投票。

这种分层策略的价值在于：它不会把 prompt 塞满同一种题，也不会把所有槽位都给最难题。对于“零基础到初级工程师”更容易理解的一句话是：**把 prompt 当成一个小型教学样本包，而不是一个简单例题堆**。

---

## 替代方案与适用边界

不是所有任务都需要复杂度驱动的 CoT 选择。任务越简单、模型越强、上下文越紧，越可能存在更便宜的替代方案。

| 替代方案 | 适用边界 | 优点 | 主要短板 |
| --- | --- | --- | --- |
| 代表性抽样 | 任务模式稳定、推理浅 | 简单、便宜 | 对复杂多步题帮助有限 |
| 类别均衡采样 | 分类或结构化抽取 | 覆盖面清楚 | 不直接优化推理链质量 |
| 固定模板+少量示例 | 输出格式严格、步骤固定 | token 成本低 | 对开放式推理适应性弱 |
| 纯零样本 CoT | 模型强、任务不难 | 实现最简单 | 波动更大，难以控行为 |
| 检索增强但不做复杂度排序 | 相关性要求高、实时性优先 | 容易落地 | 仍可能选到浅层示例 |

### 什么时候适合用代表性抽样

如果任务本身不是多步推理，而是分类、抽取、改写、风格转换，代表性抽样通常就够了。因为这类任务的关键不在“推理路径是否完整”，而在“格式和标签边界是否清楚”。

### 什么时候适合用模板替代示例链

如果 prompt 只能放 2 到 3 个示例，而且输出流程高度固定，可以直接把结构显式写出来，例如：

1. 提取已知条件  
2. 建立公式  
3. 代入计算  
4. 输出最终答案

这种“结构说明 + 少量代表性示例”的方式，在 token 非常紧张时经常比“堆复杂链”更划算。

### 玩具例子

如果只能放 3 个示例，而你要做的是“简单四则运算应用题”，可以这样选：

- 1 个代表加法组合
- 1 个代表乘法组合
- 1 个代表先乘后加

再加一段固定结构说明，比盲目追求“最长示例”更稳。

### 真实工程边界

在金融风控、医疗问答、法律分析这类高风险场景里，复杂 CoT 选择能提高推理质量，但不能替代外部规则和校验器。因为即便示例选得再好，模型输出仍可能出现“过程看起来合理、结论却违反业务约束”的情况。

所以更准确的边界是：**复杂示例选择能提升推理质量，但不是事实校验机制，也不是安全机制**。

---

## 参考资料

- Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, Tushar Khot. *Complexity-Based Prompting for Multi-step Reasoning*. ICLR 2023.  
  https://openreview.net/forum?id=yf1icZHC-l9

- Learn Prompting. *Complexity-Based Prompting*. 文中汇总了 GSM8K、MathQA 等任务上的结果，给出了 74.4% -> 82.6% -> 82.9% 的示例。  
  https://learnprompting.org/docs/advanced/thought_generation/complexity_based_prompting

- Tetrate. *Few-Shot Learning: Practical Guide for LLM Applications*. 总结了代表性抽样、多样性采样、难度选择、动态检索等 few-shot 示例策略。  
  https://tetrate.io/learn/ai/few-shot-learning-guide

- Tetrate. *Few-Shot Learning for LLMs: Examples and Implementation Guide*. 对 diversity-based selection、difficulty-based selection、dynamic selection 给出了工程化说明。  
  https://tetrate.io/learn/ai/few-shot-learning-llms

- ICLR Poster Page. *Complexity-Based Prompting for Multi-step Reasoning*. 提供论文摘要，明确说明复杂度驱动提示与复杂性投票在 GSM8K、MathQA 上的提升。  
  https://iclr.cc/virtual/2023/poster/11280

- Springer. *CTI-Thinker* 相关论文页面。可作为“检索增强 + 推理示例池”工程化方向的补充阅读。  
  https://link.springer.com/article/10.1186/s42400-025-00505-y
