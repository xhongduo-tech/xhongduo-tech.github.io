## 核心结论

上下文学习（in-context learning, ICL，指模型不改参数、只靠提示词里的示例临时学任务）不是单一能力，而是至少包含两种机制：小模型更像做 **PMI 模式匹配**，大模型更像做 **函数式任务学习**。

PMI（点互信息，白话说就是“两样东西在语料里一起出现得有多反常地频繁”）主导时，模型看到“好评”“满意”“推荐”这类词，就更容易输出“正面”；它依赖的是训练语料里的常见搭配。函数式学习主导时，模型会把 few-shot 示例当成一组约束，尝试恢复“输入怎么映射到标签”的规则，即使标签本身没有语义，也能继续做对。

这就是所谓的规模涌现：参数规模跨过某个门槛后，模型在 ICL 中的行为会从“记标签词”跳到“学输入到标签的映射”。最直接的证据是 **flipped-label** 实验：如果把情感分类里的标签整体翻转，小模型通常还是输出符合原始语义的标签，大模型则会跟着示例一起翻转。

| 维度 | 小模型常见行为 | 大模型常见行为 |
|---|---|---|
| 看到“好评→正面”示例 | 记住“好评”和“正面”常搭配 | 既利用语义，也尝试拟合映射 |
| 标签改成 `+1/-1` | 仍偏向语义熟悉标签 | 能学会 `+1/-1` 的新规则 |
| 标签随机翻转 | 表现变化小，说明没真跟示例 | 表现明显翻转，说明在跟示例 |
| 本质 | 检索高共现模式 | 在上下文里做临时任务学习 |

两个最简对比图可以写成：

```text
PMI 配对：
输入词 -> 检索常见搭配 -> 输出常见标签
“好评” -> “正面”共现高 -> 正面
```

```text
函数映射：
示例集 -> 拟合输入到标签的规则 -> 对新样本推断
(x1,y1),(x2,y2)... -> 学到 f(x) -> y_hat
```

---

## 问题定义与边界

本文讨论的“上下文学习规模涌现”，边界很明确：只讨论 **推理时通过示例学习任务** 的能力，不讨论监督微调（fine-tune，指用梯度更新模型参数）后的能力，也不讨论纯知识问答。

PMI 模式匹配可以形式化为：

$$
PMI(x,y)=\log \frac{P(x,y)}{P(x)P(y)}
$$

其中 $x$ 可以理解为输入中的词、短语或特征，$y$ 是标签。若“好评”和“正面”在预训练语料中常一起出现，那么它们的 PMI 会较高。模型即使没有真正理解当前 few-shot 的任务规则，也可能因为这种统计共现而输出“正面”。

函数式学习则可写成一个更接近分类器的形式：

$$
\hat y = \mathrm{sign}(w^\top x + b)
$$

这里 $x$ 是输入表示，$w$ 和 $b$ 是模型在当前上下文里“临时构造”的判别边界。白话说，模型不是在问“这个词平时配什么标签”，而是在问“这几个示例共同定义了什么规则”。

玩具例子最能看清边界。设任务是情感分类：

| 输入 | 标签 |
|---|---|
| 这家店服务很好 | 正面 |
| 物流太慢了 | 负面 |

如果新样本是“包装精致，值得推荐”，小模型和大模型都可能答“正面”，因为这时语义先验和示例方向一致，区分不出来。

现在把标签改成无语义符号：

| 输入 | 标签 |
|---|---|
| 这家店服务很好 | `+1` |
| 物流太慢了 | `-1` |

再进一步整体翻转：

| 输入 | 标签 |
|---|---|
| 这家店服务很好 | `-1` |
| 物流太慢了 | `+1` |

此时才真正测到 ICL 本身。若模型仍把“包装精致，值得推荐”判成“正面语义对应的标签”，它靠的是语料先验；若它稳定输出 `-1`，说明它在跟随当前示例定义的映射。

可以把两种流程并列看：

```text
小模型常见流程：
输入语句 -> 识别“好评”语义 -> 联想到“positive/正面” -> 输出

大模型常见流程：
读取整个 few-shot 集 -> 找到能同时解释所有样本的映射 -> 对新输入套用映射 -> 输出
```

所以本文的核心边界不是“模型懂不懂情感”，而是“模型能不能在上下文里重建任务函数”。

---

## 核心机制与推导

小模型的 ICL 更像检索，高层表现像分类，底层却可能只是激活了大量“输入模式-标签词”的共现记忆。大模型则更接近在 Transformer 内部做一次隐式拟合。所谓隐式拟合，意思不是显式调用线性回归库，而是在注意力和前馈层的组合里，近似实现了类似求解器的效果。

先看一个 4-shot 二分类玩具例子。设输入用二维特征表示：

| 样本 | $x_1$ | $x_2$ | 标签 $y$ |
|---|---:|---:|---:|
| A | 2 | 1 | 1 |
| B | 1 | 2 | 1 |
| C | -1 | -2 | -1 |
| D | -2 | -1 | -1 |

写成矩阵：

$$
X=
\begin{bmatrix}
2 & 1 \\
1 & 2 \\
-1 & -2 \\
-2 & -1
\end{bmatrix},
\quad
Y=
\begin{bmatrix}
1\\
1\\
-1\\
-1
\end{bmatrix}
$$

如果模型在上下文中近似恢复一个线性分类头，那么它等价于寻找：

$$
w \approx (X^\top X)^{-1}X^\top Y
$$

算出来的 $w$ 会接近一个把“整体为正”和“整体为负”分开的方向。新样本 $x=(1,1)$ 时，$w^\top x$ 为正，因此预测为正类；若新样本 $x=(-1,-1)$，则预测为负类。

这一步的重要点不在“它一定精确做了矩阵求逆”，而在“它的内部计算越来越像在恢复一个能解释示例的映射”。

PMI 路径与函数路径的差异可以总结为：

| 步骤 | PMI 主导 | 函数主导 |
|---|---|---|
| 读示例 | 提取高共现词和标签 | 提取样本间共同约束 |
| 处理标签 | 标签语义很重要 | 标签可为任意符号 |
| 新样本预测 | 找相似词面模式 | 代入拟合出的边界 |
| 标签翻转后 | 往往不跟随 | 往往跟随 |

把 Transformer 的直观过程写成伪代码，大致是：

```text
for each demo in prompt:
    encode(input_i, label_i)
    accumulate pair relation into hidden states

task_vector = aggregate(all demo relations)

for query:
    query_repr = encode(input_query)
    score = project(query_repr, task_vector)
    output label with highest consistency
```

更接近线性代数的写法是：

```text
H = Attention(X, X, Y)      # 从示例里聚合输入-标签关系
w_tilde = MLP(H)            # 构造临时任务向量
y_hat = sign(query @ w_tilde)
```

这里的 `task_vector` 或 `w_tilde` 就是“自我构造的线性头”。白话说，模型没有真的插入一个新分类器层，但它在隐藏状态里临时造出了一个近似分类器。

为什么这会表现出尺度临界？因为恢复映射函数比检索共现模式更难。它要求模型同时具备三件事：足够强的表示能力、足够长的示例整合能力、以及在推理中稳定执行多步组合计算的能力。参数太小，最省计算的解法就是“看到好评词，就输出正面词”；参数够大后，“从示例反推规则”才成为可行解。

---

## 代码实现

工程上最常见的误判是：拿一组 few-shot prompt 跑通了，就以为模型已经学会任务。其实它可能只是顺着标签语义在答。最基本的验证方法就是同时做正常标签实验和 flipped-label 实验。

下面给出一个可运行的 Python 玩具实现。它不调用真实 LLM，而是模拟两类行为：

1. `small_model_predict`：根据正负词典做 PMI 式近似判断。
2. `big_model_predict`：根据 few-shot 示例当前定义的标签映射来判断。

```python
from collections import Counter

train_examples = [
    ("这家店服务很好，值得推荐", "NEG"),   # flipped label
    ("物流太慢了，而且包装破损", "POS"),   # flipped label
    ("质量很差，不会再买", "POS"),
    ("客服耐心，体验满意", "NEG"),
]

test_examples = [
    ("做工精致，整体很满意", "NEG"),
    ("发货延迟，体验糟糕", "POS"),
]

positive_words = {"很好", "推荐", "满意", "耐心", "精致"}
negative_words = {"太慢", "破损", "很差", "糟糕", "延迟"}

def sentiment_score(text: str) -> int:
    score = 0
    for w in positive_words:
        if w in text:
            score += 1
    for w in negative_words:
        if w in text:
            score -= 1
    return score

def small_model_predict(text: str) -> str:
    # 近似 PMI/语义先验：看文本本身，不真正跟 few-shot 映射
    return "POS" if sentiment_score(text) >= 0 else "NEG"

def infer_mapping_from_context(examples):
    # 近似函数学习：从上下文恢复“正向情感对应哪个标签”
    votes = Counter()
    for text, label in examples:
        s = sentiment_score(text)
        if s >= 0:
            votes[("positive", label)] += 1
        else:
            votes[("negative", label)] += 1

    pos_label = votes[("positive", "POS")] >= votes[("positive", "NEG")]
    neg_label = votes[("negative", "POS")] >= votes[("negative", "NEG")]

    mapping = {
        "positive": "POS" if pos_label else "NEG",
        "negative": "POS" if neg_label else "NEG",
    }
    return mapping

def big_model_predict(text: str, examples) -> str:
    mapping = infer_mapping_from_context(examples)
    polarity = "positive" if sentiment_score(text) >= 0 else "negative"
    return mapping[polarity]

small_preds = [small_model_predict(x) for x, _ in test_examples]
big_preds = [big_model_predict(x, train_examples) for x, _ in test_examples]
gold = [y for _, y in test_examples]

small_acc = sum(p == y for p, y in zip(small_preds, gold)) / len(gold)
big_acc = sum(p == y for p, y in zip(big_preds, gold)) / len(gold)

assert small_preds == ["POS", "NEG"]
assert big_preds == ["NEG", "POS"]
assert small_acc == 0.0
assert big_acc == 1.0

print("small:", small_preds, small_acc)
print("big:", big_preds, big_acc)
```

上面的 prompt 片段若写成真实 few-shot，大致是：

```text
输入: 这家店服务很好，值得推荐
标签: NEG

输入: 物流太慢了，而且包装破损
标签: POS

输入: 质量很差，不会再买
标签: POS

输入: 客服耐心，体验满意
标签: NEG

输入: 做工精致，整体很满意
标签:
```

在真实工程里，可以把评估流程写成：

1. 准备正常标签集。
2. 构造 flipped-label 集，保持输入不变，只翻转标签。
3. 分别测模型在两种 prompt 下的准确率。
4. 若 flipped 后模型预测方向几乎不变，说明它主要依赖语义先验。
5. 若 flipped 后模型能稳定跟随，说明它具备更强的函数式 ICL。

一个结果表通常长这样：

| 模型类型 | 正常标签准确率 | flipped-label 准确率 | 解释 |
|---|---:|---:|---|
| 小模型 | 82% | 78% | 变化小，说明还在沿用语义 |
| 大模型 | 88% | 24% | 相对原标签方向显著翻转，说明在跟随示例 |

这里 flipped-label 的“低准确率”不是坏事。因为若测试集的标准答案仍按原语义记，真正跟随翻转规则的模型就会故意答反，反而证明它学了新映射。

---

## 工程权衡与常见坑

真实工程例子比情感分类更容易踩坑。假设一个运维团队把故障等级设计成两个自定义标签：`雷达` 表示高风险，`风暴` 表示低风险。然后给模型 few-shot：

- `RamError -> 雷达`
- `CPUOverheat -> 雷达`
- `DiskSlow -> 风暴`
- `MinorLatency -> 风暴`

如果模型足够大，它会把这些示例视为一个新的分类任务，学的是“错误特征到风险等级”的映射。若模型偏小，它可能完全不稳定，因为“雷达”“风暴”与故障风险在预训练语料里没有固定共现关系，甚至会被它错误联想到天气、军事或新闻语义。

常见坑可以直接列成检查表：

| 风险 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 误把 few-shot 成功当作任务学会 | 正常标签下效果不错 | 标签语义本来就强 | 必做 flipped-label 验证 |
| 自定义标签失效 | 任意符号标签时准确率暴跌 | 模型只会借语义先验 | 改用语义明确标签或换更大模型 |
| 示例冲突 | 同类输入对应多个标签 | 上下文约束不一致 | 保证 few-shot 标注干净 |
| 样本太少 | 结果不稳定 | 拟合不到规则 | 提高 shot 数并控制覆盖面 |
| 标签泄漏 | 标签名自带任务信息 | 评估被污染 | 使用 SUL，语义无关标签 |

SUL（semantically unrelated labels，语义无关标签）就是故意用和任务无关的标签，例如把“正面/负面”换成“河流/石墨”。如果模型在这种设置下还能学会，说明它确实在做函数式 ICL。

训练流程和验证流程可以这样区分：

```text
训练/接入流程：
选模型 -> 写 few-shot prompt -> 在线调用 -> 观察表面效果

正确验证流程：
选模型 -> 正常标签测试 -> flipped-label/SUL 测试 -> 再决定是否上线
```

一个很实际的原则是：**不要把“模型会按示例补全”误解成“模型能学任意标签映射”**。前者很多模型都能部分做到，后者是更高阶、且明显依赖规模的能力。

---

## 替代方案与适用边界

如果业务场景无法保证模型具备强函数式 ICL，不要硬赌“示例足够多它就会学会”。更稳妥的选择通常只有两个。

| 方案 | 适用模型 | 标签设计 | 优点 | 局限 |
|---|---|---|---|---|
| A. 小模型 + label engineering | 小到中等模型 | 用自然语言标签，如“高风险/低风险” | 成本低，表现稳定 | 依赖语义先验，难支持任意符号 |
| B. 大模型 + flipped-label 验证 | 大模型 | 可用任意标签，包括随机符号 | 能做真正的上下文任务学习 | 成本高，需严格验证 |

可以把选择逻辑写成一个简单决策表：

| 问题 | 若答案是“是” | 建议 |
|---|---|---|
| 标签必须是任意符号吗？ | 是 | 优先大模型，且先做 flipped/SUL |
| 标签能改成自然语言吗？ | 是 | 小模型可先尝试 |
| 任务规则经常变吗？ | 是 | 更需要函数式 ICL 或 fine-tune |
| 成本极敏感吗？ | 是 | 先做小模型 + 标签工程 |

当任务只是常规分类，且标签天然有语义，例如“正面/负面”“垃圾邮件/正常邮件”，小模型往往已经够用。因为此时 PMI 与任务规则方向一致，模型靠语料先验也能得到不错结果。

当任务要求“任意符号映射”，例如 `A/B`、`+1/-1`、`雷达/风暴`，或者标签定义会频繁变动时，仅靠小模型 few-shot 很不可靠。这时要么用参数量足够大的模型，并通过 flipped-label 验证确认其函数式能力；要么直接做 fine-tune，把规则固化进参数。

一句话概括选型边界：**能把标签写成人类熟悉语义时，小模型可以借先验工作；必须支持任意标签时，才真正需要大模型的函数式 ICL。**

---

## 参考资料

| 来源 | 简短摘要 | 本文引用位置 | 选择原因 |
|---|---|---|---|
| Google Research, “Larger language models do in-context learning differently” | 给出 flipped-label 与 SUL 实验，展示大模型与小模型的 ICL 行为差异 | 核心结论、代码实现、工程坑 | 这是本文最直接的实验证据来源 |
| OATML/ICLR 2024, “In-Context Learning Learns Label Relationships but Is Not Conventional Learning” | 说明 ICL 会学习标签关系，但不等同于传统参数学习 | 问题定义与边界、替代方案 | 用来澄清“ICL 不是普通训练”的边界 |
| MIT News, “Large language models may learn to retrieve and process information in a way analogous to functions” | 介绍模型可能在上下文中近似执行函数拟合 | 核心机制与推导 | 适合作为线性回归类比的背景材料 |
| EmergentMind, “Pattern Matching in Large Language Models” | 讨论大模型模式匹配能力如何扩展到更深层结构 | 核心结论、机制解释 | 用于补充“从表面模式到结构模式”的视角 |

阅读顺序建议：

| 顺序 | 资料 | 理由 |
|---|---|---|
| 1 | Google Research 博客 | 先建立 flipped-label 这个核心实验直觉 |
| 2 | ICLR 2024 论文 | 再看 ICL 到底学了什么、没学什么 |
| 3 | MIT News 对相关研究的解读 | 帮助理解“隐式函数拟合”的直观机制 |
| 4 | EmergentMind 汇总 | 用于补充尺度与模式扩展视角 |

- Google Research: https://research.google/blog/larger-language-models-do-in-context-learning-differently/
- ICLR 2024 Paper PDF: https://proceedings.iclr.cc/paper_files/paper/2024/file/0db404e9cc9d282be6dfe8538352408c-Paper-Conference.pdf
- MIT News: https://news.mit.edu/2023/large-language-models-in-context-learning-0207
- EmergentMind: https://www.emergentmind.com/papers/2601.11432
