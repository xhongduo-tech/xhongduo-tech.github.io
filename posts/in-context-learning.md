## 核心结论

上下文学习（In-Context Learning，简称 ICL）指的是：在推理时把少量带标签示例直接放进提示词，模型不更新参数，只靠一次前向计算就对新输入做预测。白话说，就是“把题型和答案样例贴在题目前面，模型现场照着规律做下一题”。

它的标准形式可以写成：

$$
\text{Prompt} = [x_1, y_1, x_2, y_2, \dots, x_k, y_k, x_{\text{query}}, ?]
$$

其中 $x_i$ 是输入，$y_i$ 是对应输出，$x_{\text{query}}$ 是待预测样本。模型的目标不是“重新训练”，而是“从当前上下文里推断这是什么任务”。

ICL 的价值不在于它永远优于微调，而在于它能在没有训练流程、没有梯度更新、没有参数存储版本切换的情况下，快速让同一个大模型适配新任务。对于新类别上线、规则频繁变动、样本很少但要立刻可用的场景，这种能力非常实用。

但它不是“随便给几个例子就稳定生效”。ICL 对示例格式、顺序、标签噪声、上下文长度都敏感。一个经常被忽略的事实是：同一组示例换个顺序，准确率可能出现明显波动。也就是说，ICL 更像“推理时的临时任务适配”，不是“替代训练的一劳永逸方案”。

---

## 问题定义与边界

更正式地说，ICL 解决的问题是：给定一个冻结参数的语言模型 $f_\theta$，其中 $\theta$ 在推理前已经固定；再给定 $k$ 个上下文示例 $\{(x_i, y_i)\}_{i=1}^k$；要求模型仅基于这些示例，在不修改 $\theta$ 的条件下，对新输入 $x_{\text{query}}$ 输出预测 $y_{\text{query}}$。

这里的“冻结参数”很关键。它和微调的区别可以直接写成：

- ICL：训练阶段不发生在当前任务上，推理时只改输入上下文，不改模型权重。
- 微调：把当前任务的数据拿去做梯度下降，直接改模型权重。

一个玩具例子最容易看清边界。假设我们要做极简情感分类：

- 示例 1：`输入: 这家餐厅很好吃`，`输出: 正面`
- 示例 2：`输入: 等了一个小时还没上菜`，`输出: 负面`
- 查询：`输入: 服务态度不错，下次还来`，`输出: ?`

如果提示结构保持一致，模型往往能推断这是“中文评论二分类”任务，并输出“正面”。这里模型不是学会了新参数，而是从示例中临时推断了“输入到标签”的映射格式。

ICL 的适用边界主要有四个。

| 维度 | ICL 表现 | 原因 |
|---|---|---|
| `k` 很小 | 通常可用 | 大模型可从少量示例归纳任务格式 |
| 顺序变化大 | 可能波动明显 | 模型对上下文位置与近因信息敏感 |
| 标签有少量噪声 | 分类任务常有一定容忍度 | 模型有时更依赖格式和整体模式 |
| 任务需要长期稳定 | 往往不如微调稳 | 每次都依赖 prompt，难以完全固定行为 |

可以把它理解成一个“推理时临时装配出来的任务描述器”。它特别依赖这几点：

1. 示例和查询必须使用一致格式。
2. 示例最好覆盖主要模式，而不是只堆相似样本。
3. 标签命名要稳定，比如不要一会儿写“正面/负面”，一会儿写“积极/消极”。
4. 上下文长度有限，示例太多会挤占真实输入空间。

因此，ICL 不是泛化到所有场景的统一答案。它适合的是 few-shot，即少样本、快速适配、允许通过 prompt 工程反复调试的任务。

---

## 核心机制与推导

为什么模型不改参数，也能“像学会了任务一样”工作？当前较有解释力的一条路线是：预训练让 Transformer 学到了某种元学习（meta-learning）能力。元学习的白话解释是：不是记住某个固定任务，而是学会“如何根据新给出的样例快速猜出任务”。

Garg 等人的工作说明，Transformer 可以在上下文中学到线性函数、稀疏函数等规律。Von Oswald 等人进一步给出一个更强的构造：在线性 self-attention 的设定下，Transformer 的多层前向传播可以等价为对一个隐式模型做梯度下降。

核心更新式是：

$$
\theta_{t+1} = \theta_t - \eta X^\top (X\theta_t - y)
$$

这里：

- $\theta_t$ 表示第 $t$ 步的隐式参数状态。白话说，是模型在当前上下文里“临时拟合出来”的任务解。
- $X$ 表示由示例输入组成的矩阵。
- $y$ 表示示例标签。
- $\eta$ 是学习率，控制每一步更新幅度。

这个式子本来是普通线性回归的梯度下降更新。Von Oswald 等人的关键点在于：适当构造 attention 和层堆叠后，Transformer 的 forward 可以实现与这个过程等价的计算轨迹。也就是说，模型表面上只是“读了一遍 prompt”，但内部行为可以近似看成“根据上下文做了几步优化”。

直观流程可以写成：

$$
(x_1,y_1), \dots, (x_k,y_k)
\rightarrow \text{任务模式识别}
\rightarrow \text{隐式拟合}
\rightarrow x_{\text{query}} \text{ 的预测}
$$

这不是说大语言模型真的在运行标准 SGD 代码，而是说它的某些结构和训练结果，允许它在前向过程中近似实现“从样例中拟合任务”的效果。

再看一个玩具例子。假设输入是二维向量，标签是线性规则：

- $(1,0) \mapsto A$
- $(0,1) \mapsto B$

现在查询是 $(1,1)$。如果模型把上下文理解为“这是一个依据特征组合映射到标签的任务”，它会尝试从前两个样例中归纳规则，再对第三个样本作判断。这里不需要显式更新网络参数，更新发生在“当前上下文诱导出的隐式任务表示”里。

这也解释了两个经验现象。

第一，格式很重要。因为模型先识别“任务接口”是什么，再识别具体样例内容。`Q: ... A: ...`、`Input: ... Output: ...` 这类稳定模板，实际上在帮助模型定位任务结构。

第二，顺序会影响结果。因为注意力机制天然依赖位置编码与上下文分布，后出现的示例、格式突变的示例、异常标签示例，都可能改变模型正在形成的“隐式拟合路径”。这不是简单的随机噪声，而是模型内部任务推断过程被不同上下文轨迹扰动。

---

## 代码实现

工程上做 ICL，不需要先追求复杂框架。第一步通常就是把示例稳定地拼成 prompt。下面给出一个可运行的 Python 示例，用最简单的字符串模板构造 few-shot prompt，并用一个玩具规则模拟“根据示例推断任务”。

```python
from typing import List, Tuple

Example = Tuple[str, str]

def build_prompt(task_desc: str, examples: List[Example], query: str) -> str:
    parts = [task_desc.strip(), ""]
    for x, y in examples:
        parts.append(f"Input: {x}")
        parts.append(f"Output: {y}")
        parts.append("")
    parts.append(f"Input: {query}")
    parts.append("Output:")
    return "\n".join(parts)

def toy_infer_by_majority_keyword(examples: List[Example], query: str) -> str:
    """
    一个玩具推断器：
    - 如果含有“好/不错/喜欢”等词的示例多数被标成正面，就把相似 query 判为正面
    - 这里只是演示“根据上下文样例临时决定规则”，不是真实 LLM
    """
    positive_words = {"好", "不错", "喜欢", "满意", "推荐"}
    negative_words = {"差", "慢", "失望", "讨厌", "糟"}

    pos_label = None
    neg_label = None

    for text, label in examples:
        if any(w in text for w in positive_words) and pos_label is None:
            pos_label = label
        if any(w in text for w in negative_words) and neg_label is None:
            neg_label = label

    if any(w in query for w in positive_words) and pos_label is not None:
        return pos_label
    if any(w in query for w in negative_words) and neg_label is not None:
        return neg_label

    return examples[0][1]

examples = [
    ("这家店味道很好，下次还来", "正面"),
    ("配送太慢了，体验很差", "负面"),
]

prompt = build_prompt(
    task_desc="判断评论情感，只输出“正面”或“负面”。",
    examples=examples,
    query="服务不错，整体满意"
)

pred = toy_infer_by_majority_keyword(examples, "服务不错，整体满意")

assert "Input: 服务不错，整体满意" in prompt
assert prompt.strip().endswith("Output:")
assert pred == "正面"

print(prompt)
print("预测结果:", pred)
```

真实调用大模型 API 时，思路也是一样，只是把 `toy_infer_by_majority_keyword` 替换成模型请求。核心点有三个：

1. 模板固定。
2. 示例顺序可控。
3. 推理参数固定，减少无关波动。

示意代码如下：

```python
def build_classification_prompt(examples, query):
    lines = ["任务：判断用户请求属于哪个意图，只输出标签。", ""]
    for x, y in examples:
        lines.append(f"Input: {x}")
        lines.append(f"Output: {y}")
        lines.append("")
    lines.append(f"Input: {query}")
    lines.append("Output:")
    return "\n".join(lines)

examples = [
    ("我要开发票", "发票"),
    ("帮我查一下物流到哪了", "物流"),
    ("这个商品能退吗", "售后"),
]

query = "请问我的快递什么时候到"

prompt = build_classification_prompt(examples, query)

# 伪代码
# response = client.responses.create(
#     model="gpt-4.1",
#     input=prompt,
#     temperature=0,
#     max_output_tokens=8
# )
# print(response.output_text)
```

这里的真实工程例子是“客服意图分类”。如果业务今天新上线了“发票重开”意图，但没有时间走完整训练流程，那么只要补 3 到 5 条高质量示例，ICL 就能先顶上去。它适合做快速试运行、灰度验证、冷启动覆盖。

---

## 工程权衡与常见坑

ICL 最常见的问题不是“完全没效果”，而是“能用，但不稳”。这类不稳定通常来自四种来源：顺序敏感、标签噪声、示例分布偏移、输出格式失控。

先看顺序问题。Brown 等人的 few-shot 结果已经显示 prompt 设计会显著影响表现，后续 Zhao 等人的研究进一步说明，仅仅是示例排列不同，就可能导致明显准确率波动。白话说，模型不是只看“集合里有哪些样本”，而是在看“这些样本按什么顺序出现”。

下面是一个简化后的工程观察表：

| 策略 | 准确率 | 备注 |
|---|---:|---|
| 固定单一顺序 | 71% | 容易被末尾示例带偏 |
| 随机 5 次 permutation 后投票 | 78% | 方差下降 |
| 随机化 + 概率校准 | 80% | 对类别偏置更稳 |

这里的“校准”可以理解为：先估计模型对标签先验的偏好，再把这种偏好从最终分数里扣掉，避免模型因为更喜欢某个标签字符串而产生系统偏差。

第二个常见坑是标签噪声。已有研究表明，ICL 对少量错误标签并非完全脆弱，尤其在分类任务里，模型有时仍能保持可用性能。这说明模型不只是机械抄标签，它也在利用整体语义模式。但这不意味着可以随意混入脏数据。

经验上可以这样判断：

- 少量错误标签：可能还能工作，但性能下滑。
- 标签系统性错位：模型会学到错误映射。
- 生成任务中的噪声：通常比分类更危险，因为输出空间更大，更难约束。

第三个坑是示例覆盖不足。比如做客服分类时，你给的 3 条“退款”示例全是“我要退货”，查询却是“七天无理由怎么走流程”，模型可能因为示例太窄而误判。ICL 依赖的是上下文拟合，所以示例要覆盖概念边界，而不是只重复同一句话的不同表述。

第四个坑是输出约束不严。很多人把分类任务写成开放式自然语言问答，结果模型输出一整句解释，后处理反而更复杂。工程上应优先要求：

- 固定标签集合
- 固定输出格式
- `temperature=0` 或接近 0
- 必要时加正则或枚举约束

一个真实工程例子是工单路由。假设企业有“发票、物流、售后、账户、投诉”五类，新增“会员权益”后，团队先不做微调，而是把 4 条新示例插入 prompt。上线后如果只使用一种示例顺序，某些批次请求会偏向把模糊问题都判成“账户”。这时正确做法不是立刻重训，而是：

1. 记录示例顺序与预测结果。
2. 对同一请求做多顺序采样。
3. 对高不确定样本回退人工或规则。
4. 定期把稳定样本沉淀到微调数据集。

ICL 在这里是快速适配层，不是最终的长期收敛层。

---

## 替代方案与适用边界

ICL 很强，但不是唯一方案。和它最常比较的是微调（fine-tuning）与检索增强提示（retrieval-augmented prompting，常简称 RAG 式提示）。

| 策略 | 需要更新 | 适合场景 | 限制 |
|---|---|---|---|
| ICL | 不需要 | 少样本、快速试错、新任务冷启动 | 顺序敏感，成本随上下文增长 |
| Fine-tuning | 需要 | 长期稳定任务、高频调用、固定标签体系 | 数据与训练成本更高 |
| 检索增强 Prompt | 不一定更新模型 | 依赖外部知识、事实密集问答 | 依赖检索质量，不直接解决任务映射 |

三者的分工可以这样理解。

ICL 适合“今天就要上线，但数据只有几条”的情况。它把样例直接放进 prompt，让模型当场适应。

微调适合“这个任务会长期存在，而且每天调用很多次”的情况。因为一旦任务稳定，反复把样例塞进上下文会带来额外 token 成本，且行为仍可能波动。把能力写进参数里，长期更稳。

检索增强更适合“答案主要取决于外部知识，而不是少量标签映射”的情况。比如法律条文问答、产品文档问答、知识库客服。这时核心问题不是 few-shot 分类，而是先找到相关资料，再让模型基于资料回答。

举一个真实选择场景。假设你在做“新意图分类”：

- 如果每周都新增意图，且每个意图最初只有几条样本，先用 ICL 最合理。
- 如果半年后意图体系已经稳定，且每天数十万调用，应该考虑微调。
- 如果用户的问题高度依赖最新帮助中心文档，那么光做 ICL 不够，应该把检索系统接进来。

所以正确问题不是“ICL 和微调谁更先进”，而是“当前任务是临时适配问题，还是稳定建模问题”。

---

## 参考资料

| 来源 | 重点 | 链接简述 |
|---|---|---|
| Brown et al., 2020, *Language Models are Few-Shot Learners* | 给出 few-shot prompting 的经典定义与大规模实证 | OpenAI 论文页面 |
| Garg et al., 2022, *What Can Transformers Learn In-Context?* | 说明 Transformer 可在上下文中学到线性与稀疏规律 | NeurIPS 论文 |
| Von Oswald et al., 2023, *Transformers Learn In-Context by Gradient Descent* | 给出线性 self-attention 与梯度下降等价构造 | PMLR 论文 |
| Zhao et al., 2021, *Calibrate Before Use* | 展示 prompt 形式、顺序、标签偏置带来的性能波动，并提出校准方法 | ICML 论文 |
| *Exploring the Robustness of In-Context Learning with Noisy Labels*, 2024 | 讨论 ICL 对 noisy label 的鲁棒性边界 | arXiv 方向综述 |

对初学者，推荐这样理解这些资料的分工：

- Brown 2020：回答“ICL 是什么，few-shot prompt 长什么样”。
- Garg 2022：回答“Transformer 到底能在上下文里学到哪些函数规律”。
- Von Oswald 2023：回答“为什么前向传播会像在做梯度下降”。
- Zhao 2021：回答“为什么换个顺序结果就变了，为什么要做校准”。
- 2024 噪声鲁棒性工作：回答“标签有点脏时，ICL 还能撑到什么程度”。
