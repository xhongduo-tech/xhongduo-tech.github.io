## 核心结论

响应多样性增强，是在不明显损失语义正确性的前提下，让同一输入产生在措辞、结构、语气或解法上更分散的多个候选输出。

它的目标不是让模型“更随机”，而是让模型在正确边界内提供更多可选表达。对初级工程师来说，可以把它理解为：同一道题允许有多种合格答案，但每个答案都必须还在题目要求内。

单纯调高温度不是可靠方案。温度，指的是控制模型采样分布平滑程度的参数。温度越高，低概率 token 越容易被采到，输出会更分散，但也更容易跑偏。更稳的做法是两段式流程：

| 类型 | 输出特征 | 典型问题 | 是否推荐 |
|---|---|---|---|
| 低多样性 | 表达稳定、模板化、重复度高 | 用户感到机械，数据风格容易塌缩 | 只适合强约束任务 |
| 高多样性 | 说法丰富、结构变化大 | 容易事实漂移、风格不可控 | 不应直接上线 |
| 可控多样性 | 多候选、低重复、事实一致 | 需要额外筛选和评估 | 推荐 |

新手版例子：用户输入“帮我改写这段话”。如果只把温度调高，模型可能从“略有变化”变成“意思都变了”。如果先生成 6 个候选，再筛掉重复、跑题和事实错误的结果，最后拿到的才是“不同但还对”的回答。

核心流程可以写成：

```text
输入 x
  -> 使用 temperature + top_p 生成多个候选
  -> 过滤事实错误和约束违规
  -> 按相关性、差异度、风格覆盖重排
  -> 输出 top-k 个候选
```

采样温度的基本公式是：

$$
q_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $z_i$ 是第 $i$ 个 token 的 logit，logit 是模型在 softmax 前给每个 token 打的原始分数。$T$ 越大，分布越平，候选空间越宽。

---

## 问题定义与边界

这里的“多样性”，指的是同一任务下合理答案之间的差异，包括措辞、段落结构、语气、详略程度、推理路径和解法选择。它不是事实内容随意变化，也不是让模型输出变得不可预测。

边界可以用一个约束目标表示：

$$
\max Diversity(y) \quad \text{subject to} \quad Correctness(y) \ge \tau
$$

其中 $Diversity(y)$ 表示候选的差异度，$Correctness(y)$ 表示语义正确性，$\tau$ 是最低正确性阈值。白话说法是：先保证答案没错，再追求答案不同。

客服场景里，“抱歉给您带来不便”改成“很抱歉影响了您的使用体验”，属于可接受多样性；但把“订单已发货”改成“订单未处理”，就不是多样性，而是事实错误。

| 变化类型 | 示例 | 是否接受 | 原因 |
|---|---|---:|---|
| 措辞变化 | “已收到请求” -> “我们已经收到您的请求” | 是 | 含义不变 |
| 语气变化 | “请稍等” -> “麻烦您稍等片刻” | 是 | 语气更礼貌 |
| 结构变化 | 先结论后步骤 -> 先步骤后结论 | 视任务而定 | 需要保持信息完整 |
| 事实变化 | “已发货” -> “未处理” | 否 | 事实相反 |
| 约束丢失 | 要求 50 字内，输出 300 字 | 否 | 违反任务边界 |

玩具例子：输入是“把‘系统已启动’改写得更正式”。候选 A 是“系统已经完成启动”，候选 B 是“系统当前处于运行状态”。二者都可接受。候选 C 是“系统启动失败”，虽然词面相关，但语义反了，必须过滤。

工程上要把“生成候选”和“过滤错误”拆成两层逻辑：

```text
candidates = generate_many(prompt)
valid = []

for y in candidates:
    if violates_facts(y):
        continue
    if violates_constraints(y):
        continue
    valid.append(y)

final = diversify_and_rank(valid)
```

这个拆分很重要。生成阶段负责扩大搜索空间，筛选阶段负责守住边界。把这两件事混在一个提示词里，通常难以稳定复现。

---

## 核心机制与推导

语言模型每一步生成 token 时，会先给词表里的 token 打分，再通过 softmax 变成概率。温度采样通过调整 softmax 前的分数比例，改变概率分布的尖锐程度：

$$
q_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

当 $T < 1$ 时，高分 token 更占优势，输出更保守。当 $T > 1$ 时，低分 token 的相对概率上升，输出更容易出现不同表达。

新手版数值例子：三个 token 的 logits 是 $[3,2,1]$。当 $T=1$ 时，概率约为 $[0.665,0.245,0.090]$；当 $T=2$ 时，概率约为 $[0.506,0.307,0.186]$。第一个 token 仍然最高，但后两个 token 更容易被采到，所以句子不再总是走同一条路径。

`top-p` 采样，也叫 nucleus sampling，是另一种控制候选范围的方法。它先按概率从高到低排序，再取最小集合 $S_p$，满足：

$$
\sum_{i \in S_p} q_i^{(T)} \ge p
$$

然后只在这个集合内部重新归一化采样。比如候选概率是 $[0.50,0.20,0.15,0.10,0.05]$，当 $top\_p=0.80$ 时，会保留前三个 token，因为累计概率是 $0.85$。

| 机制 | 白话解释 | 增强多样性的方式 | 主要风险 |
|---|---|---|---|
| `temperature` | 放平或收紧概率分布 | 让低概率表达更容易出现 | 过高会跑题 |
| `top_p` | 只保留累计概率内的候选 | 截掉长尾噪声，同时保留变化 | 过低会模板化 |
| `beam search` | 每步保留高分路径 | 提高稳定性 | 候选容易相似 |
| `diversity_penalty` | 惩罚相似 beam | 让 beam 之间拉开距离 | 参数过大影响质量 |
| rerank | 生成后重新打分 | 同时考虑正确性和差异 | 需要额外模型或规则 |

多候选重排可以写成：

$$
J(y)=\frac{\log P(y|x)}{|y|^\alpha}+\lambda \cdot Div(y,C)
$$

其中 $P(y|x)$ 是输入 $x$ 下生成候选 $y$ 的概率，$|y|^\alpha$ 用来做长度归一化，$C$ 是已选候选集合，$Div(y,C)$ 表示候选 $y$ 和已选候选的差异度。$\lambda$ 控制多样性权重。

真实工程例子：客服助手需要给人工客服推荐 3 条不同风格回复。系统可以用 3 个 system prompt，分别要求“简短版”“正式版”“更有同理心版”，再配合两个温度生成 6 条候选。之后通过订单状态、退款金额、用户等级等事实字段校验，再用 embedding 相似度去重，最后输出 3 条差异明确的回复。

---

## 代码实现

工程实现里，推荐把“候选生成”和“最终输出”拆开。前者负责广撒网，后者负责保正确和去重。常见组合是 `temperature + top_p + num_return_sequences + rerank`，或者在 beam search 中使用 `group beam search + diversity_penalty`。

Hugging Face 风格伪代码如下：

```python
outputs = model.generate(
    input_ids,
    do_sample=True,
    temperature=1.2,
    top_p=0.9,
    num_return_sequences=6,
)

ranked = rerank(outputs, facts, similarity_penalty=0.35)
final = select_top_k(ranked, k=3)
```

参数职责要分清：

| 阶段 | 参数或策略 | 职责 |
|---|---|---|
| 生成阶段 | `temperature` | 控制分布平滑程度 |
| 生成阶段 | `top_p` | 控制候选 token 范围 |
| 生成阶段 | `num_return_sequences` | 一次生成多个候选 |
| 筛选阶段 | 事实一致性检查 | 删除事实错误答案 |
| 筛选阶段 | 相似度惩罚 | 删除重复表达 |
| 筛选阶段 | 风格标签 | 保证候选覆盖不同语气 |

下面是一个可运行的 Python 玩具实现。它不调用真实模型，只模拟候选重排逻辑，重点展示“先生成候选，再按正确性和差异度筛选”的结构。

```python
from math import exp

def softmax_with_temperature(logits, temperature):
    scores = [exp(x / temperature) for x in logits]
    total = sum(scores)
    return [x / total for x in scores]

def jaccard_distance(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return 1 - len(sa & sb) / len(sa | sb)

def rerank(candidates, required_fact, k=3, min_distance=0.35):
    valid = []
    for item in candidates:
        if required_fact not in item["text"]:
            continue
        valid.append(item)

    selected = []
    for item in sorted(valid, key=lambda x: x["quality"], reverse=True):
        if all(jaccard_distance(item["text"], old["text"]) >= min_distance for old in selected):
            selected.append(item)
        if len(selected) == k:
            break
    return selected

probs_t1 = softmax_with_temperature([3, 2, 1], 1.0)
probs_t2 = softmax_with_temperature([3, 2, 1], 2.0)

assert round(probs_t1[0], 3) == 0.665
assert round(probs_t2[0], 3) == 0.506
assert probs_t2[2] > probs_t1[2]

candidates = [
    {"text": "订单已发货，请留意物流更新", "quality": 0.92},
    {"text": "您的订单已发货，可以在订单页查看物流", "quality": 0.90},
    {"text": "订单未处理，请继续等待", "quality": 0.95},
    {"text": "已为您安排发货，物流信息会继续更新", "quality": 0.86},
]

final = rerank(candidates, required_fact="发货", k=2)
assert len(final) == 2
assert all("发货" in x["text"] for x in final)
assert all("未处理" not in x["text"] for x in final)
```

相似度惩罚也可以简单写成：

$$
Score(y)=Quality(y)-\beta \cdot \max_{c \in C} Sim(y,c)
$$

其中 $Sim(y,c)$ 表示候选之间的相似度，$\beta$ 越大，系统越不喜欢重复候选。

---

## 工程权衡与常见坑

多样性和稳定性天然存在张力。多样性增强会扩大搜索空间，但搜索空间越大，越需要更强的筛选、校验和评估。上线前不能只看“写得不一样”，还要看正确率、事实性、可控性和用户偏好。

一个实用的多目标评分可以写成：

$$
Score = Correctness + Diversity - Redundancy - Hallucination
$$

其中 Hallucination 指幻觉，也就是模型编造不存在的事实。这个公式不是固定算法，而是提醒工程上必须同时评估多个目标。

| 常见坑 | 表现 | 规避方法 |
|---|---|---|
| 只调高温度 | 输出更发散，但错误更多 | 使用候选筛选和事实校验 |
| 只做词面改写 | 看起来不同，信息量没变 | 加语义相似度去重 |
| 训练数据过于同质 | 模型学成单一模板 | 去重、分层采样、加入风格标签 |
| 只看 `self-BLEU` | 指标变好但答案变乱 | 同时看正确率和人工偏好 |
| 多样性压过约束 | 输出花哨但不符合任务 | 最终输出前做硬约束过滤 |

Too Long; Didn't Read 类任务的教训是：如果训练数据长期呈现同一种摘要风格，模型会把高频表达学成默认口吻。最后即使输入不同，输出也容易像同一个模板生成的。这不是模型没有能力表达，而是数据把它推向了单一风格。

新手版例子：“我已收到你的请求”改成“您的请求我已经看到了”，只是词面变化。如果回答开始加入“我们会在 24 小时内处理”，但原始事实里没有这个承诺，多样性就已经越界了。

简单去重逻辑可以写成：

```python
def keep_if_not_duplicate(candidate, selected, similarity_fn, threshold=0.82):
    for old in selected:
        if similarity_fn(candidate, old) > threshold:
            return False
    return True

assert keep_if_not_duplicate("正式回复版本", [], lambda a, b: 0.0)
assert not keep_if_not_duplicate("正式回复版本", ["正式回复版本"], lambda a, b: 1.0)
```

真实项目中，相似度函数通常不会用字符串完全匹配，而会用 embedding 相似度。embedding 是把文本转换成向量的表示方法，语义相近的文本向量距离更近。

---

## 替代方案与适用边界

如果任务目标是“回答正确”，而不是“表达多样”，最好的方案不一定是增强多样性。事实问答、金额计算、医疗建议、法律条款解释等场景，更需要约束生成、检索增强和可验证推理。

可以用一个简单判断式：

```python
config = {
    "marketing_copy": {"use_diversity": True, "temperature": 1.1, "top_p": 0.9},
    "customer_reply": {"use_diversity": True, "temperature": 0.9, "top_p": 0.85},
    "fact_qa": {"use_diversity": False, "temperature": 0.2, "top_p": 0.6},
    "billing_math": {"use_diversity": False, "temperature": 0.0, "top_p": 1.0},
}

assert config["marketing_copy"]["use_diversity"] is True
assert config["billing_math"]["use_diversity"] is False
```

更抽象地说：

$$
use\_diversity = (multiple\_valid\_outputs = true) \land (error\_cost \le acceptable)
$$

如果一个问题天然有多个合格答案，并且错误代价可控，就适合启用多样性。写营销文案时，多样性有价值，因为人需要挑版本；做金额计算时，答案应该唯一、稳定、可验证。

| 方案 | 适用场景 | 优点 | 边界 |
|---|---|---|---|
| 高温度采样 | 创作、文案、头脑风暴 | 简单直接 | 容易跑偏 |
| beam search | 翻译、摘要、格式稳定任务 | 输出稳定 | 多样性弱 |
| diverse beam search | 需要多个高质量候选 | 候选差异更明显 | 参数更难调 |
| 重排式候选生成 | 客服、推荐、改写 | 质量和多样性可平衡 | 成本更高 |
| 约束解码 | 事实问答、结构化输出 | 可控性强 | 表达空间受限 |
| 检索增强 | 知识密集任务 | 事实依据更强 | 依赖检索质量 |

推荐系统里的真实工程例子：给用户推荐文章时，不能只推 10 篇同一主题的高分文章。系统通常会先召回一批相关内容，再按主题、作者、难度、发布时间做多样性重排。这里的多样性不是为了“随机”，而是为了覆盖用户可能感兴趣的不同方向。

---

## 参考资料

| 文献或文档 | 核心贡献 | 对应章节 |
|---|---|---|
| The Curious Case of Neural Text Degeneration | 解释纯最大化概率生成会导致文本退化，提出 nucleus sampling | 核心机制与推导 |
| Diverse Beam Search | 提出在 beam search 中显式鼓励不同解 | 替代方案与适用边界 |
| Hugging Face Generation 文档 | 给出 `temperature`、`top_p`、`diversity_penalty` 等工程参数 | 代码实现 |

复现实验时，可以先固定如下参数清单：

```python
experiment = {
    "temperature": [0.7, 1.0, 1.2],
    "top_p": [0.8, 0.9, 0.95],
    "num_return_sequences": 6,
    "rerank": ["fact_check", "semantic_dedup", "style_balance"],
}

assert experiment["num_return_sequences"] > 1
assert "semantic_dedup" in experiment["rerank"]
```

1. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
2. [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/abs/1610.02424)
3. [Hugging Face Transformers Text Generation](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/text_generation)
4. [Hugging Face Blog: How to Generate Text](https://huggingface.co/blog/how-to-generate)
