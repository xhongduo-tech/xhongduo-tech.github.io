## 核心结论

“涌现能力”这个说法，先要分清是模型能力真的突然出现，还是评测指标把连续改进显示成了跳变。Schaeffer 等人在 2023 年讨论的核心点是：在多位数加法这类任务里，模型的每一位 token 正确率通常是平滑上升的，但如果评测只看“整条序列全对”的精确匹配率，曲线就会出现明显阶跃，看起来像“突然学会了”。

这里的“精确匹配”指预测结果必须和标准答案逐字符完全一致；“对数概率”指模型给正确 token 分配的概率取对数后再累加，它更接近模型在每一步到底有多确定。两者最大的区别不在名字，而在数学结构：精确匹配是乘法，对数概率是加法。

对一个长度为 $n$ 的序列，如果第 $i$ 位答对的概率是 $p_i$，那么

$$
\text{ExactMatch} \approx \prod_{i=1}^{n} p_i = \exp\left(\sum_{i=1}^{n}\log p_i\right)
$$

而交叉熵可以写成：

$$
\text{CrossEntropy} = - \sum_{i=1}^{n}\log p_i
$$

这意味着：精确匹配会把小幅、稳定的每位提升，通过连乘放大成肉眼可见的“跃迁”；对数概率和交叉熵则能连续追踪能力变化。

一个最直观的玩具例子是 5 位数加法。假设每一位的正确率都一样，从 $p=0.8$ 提升到 $p=0.95$，那么整题全对概率从

$$
0.8^5 \approx 0.32768
$$

变成

$$
0.95^5 \approx 0.77378
$$

也就是从约 33% 跳到约 77%。如果只看精确匹配，你会误以为模型出现了“突然飞跃”；但实际上，每一位只是从 80% 平滑提升到 95%，改进过程并不神秘。

---

## 问题定义与边界

先给出边界。本文讨论的是“评测指标是否制造了涌现幻觉”，不是否认所有真实的能力跃迁都不存在。这里的重点任务是多位数算术，尤其是输出长度固定、错误可逐位拆解的任务；重点指标是序列级精确匹配、token 级对数概率、交叉熵和编辑距离；重点对象是按模型规模或训练量逐步扩大的语言模型。

“涌现”在这里可以理解为：随着模型参数、训练数据或计算量增加，某个能力指标在图上不是平滑上升，而是在某个区间突然抬头，像从“不会”瞬间变成“会”。问题在于，这个“突然”可能来自指标本身，而不是模型内部机制。

新手可以把它理解成两个不同的评分系统：

| 任务 | 评分方式 | 数学形态 | 能否连续追踪能力 |
|---|---|---|---|
| 5 位数加法 | 5 位全对才算对 | 连乘，强非线性 | 较差 |
| 5 位数加法 | 每位是否正确分别计分 | 求和或平均，近似线性 | 较好 |
| 文本生成 | 序列完全一致 | 离散跳变 | 较差 |
| 文本生成 | token log-prob / 交叉熵 | 累加 | 较好 |
| 文本生成 | Edit Distance | 局部误差累计 | 中等 |

假设一个加法任务，模型输出 `5358`，标准答案是 `5359`。如果你用精确匹配，它就是 0 分；如果你用每位准确率，它是 75%；如果你用编辑距离，它只差 1 个字符；如果你看最后一位正确 token 的 log-prob，你还能知道模型是“非常不确定地错了”，还是“几乎答对，只差一点”。这些指标描述的是同一份输出，但强调的信息完全不同。

因此，本文的边界很明确：讨论的是长度可分解、token 可对齐的任务上，非线性指标如何把渐进提升显示成阶跃。对于开放式问答、主观写作、长上下文推理，问题会更复杂，不能直接照搬这个结论。

---

## 核心机制与推导

核心机制其实只有一句话：整题全对的概率，是每一步都答对概率的乘积；乘积对长度和局部误差都非常敏感。

先看最简单的独立同分布情形。若每位正确率相同，都是 $p$，长度为 $n$，则

$$
\text{ExactMatch} = p^n
$$

这条式子已经足够解释大部分“伪涌现”现象。因为当 $n$ 增大时，哪怕 $p$ 只提升一点点，$p^n$ 也可能变化很大。

新手版玩具例子如下。若每位正确率是 90%，那么：

- 1 位任务：$0.9^1 = 90\%$
- 3 位任务：$0.9^3 = 72.9\%$
- 5 位任务：$0.9^5 \approx 59.0\%$
- 10 位任务：$0.9^{10} \approx 34.9\%$

你会发现，模型“每一步都挺准”，但一旦要求“全部都对”，整体成绩就迅速下降。反过来，当每位正确率从 90% 提到 95% 时：

- 3 位任务：从 $0.9^3=72.9\%$ 到 $0.95^3\approx85.7\%$
- 5 位任务：从 $59.0\%$ 到 $77.4\%$
- 10 位任务：从 $34.9\%$ 到 $59.9\%$

局部提升并不剧烈，但整体看起来像跨过了某个门槛。

再把它写成对数形式：

$$
\log(\text{ExactMatch}) = \sum_{i=1}^{n}\log p_i
$$

这一步很重要。因为对数把乘法变成了加法，而加法更容易保留“平滑变化”的形状。交叉熵再取负号：

$$
\text{CrossEntropy} = - \sum_{i=1}^{n}\log p_i
$$

所以如果模型规模增加时，每个正确 token 的概率都在稳定上升，那么 $\sum \log p_i$ 会稳定变化，交叉熵会平滑下降。也就是说，从信息论视角看，模型能力可以是连续改进的；只是当你把它投影到“全对/不全对”这种硬阈值指标上时，图像会变成台阶。

这里还有一个常被忽略的点：精确匹配不只是“非线性”，它还是“离散”的。只要错一个字符，不管其他位置多接近，都直接归零。这会抹掉大量中间信息。

真实工程例子可以看金融或账务场景。假设模型输出 12 位流水金额编码，每位平均正确率从 98% 提升到 99%。单看每位，你会说只是提高了 1 个百分点；但整串全对的概率会从

$$
0.98^{12} \approx 78.5\%
$$

提高到

$$
0.99^{12} \approx 88.6\%
$$

如果长度更长，变化更明显。开发者只盯着精确匹配，可能会下结论：“模型突然可用了。”但从机制上看，它只是每一步更稳了，乘法把这种稳健性放大了。

---

## 代码实现

工程上最稳妥的做法，不是放弃精确匹配，而是同时记录多种指标。精确匹配适合最终验收，因为很多场景确实要求“全对”；但训练中和版本比较时，必须并行记录 token 级 log-prob 和编辑距离。

下面给出一个可运行的 Python 示例。它不依赖特定模型框架，只演示评估逻辑。这里把“token”简化成单字符，适合多位数加法任务。

```python
import math

def levenshtein(a: str, b: str) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost  # replace / match
            )
    return dp[-1][-1]

def evaluate_prediction(prediction: str, reference: str, correct_token_probs: list[float]) -> dict:
    """
    prediction: 模型输出，例如 "5358"
    reference: 标准答案，例如 "5359"
    correct_token_probs: 每个标准 token 的正确概率，例如 [0.99, 0.98, 0.97, 0.60]
    """
    assert len(reference) == len(correct_token_probs)
    assert all(0.0 < p <= 1.0 for p in correct_token_probs)

    exact_match = int(prediction == reference)
    edit_distance = levenshtein(prediction, reference)

    log_probs_sum = sum(math.log(p) for p in correct_token_probs)
    avg_log_prob = log_probs_sum / len(correct_token_probs)
    cross_entropy = -avg_log_prob

    token_accuracy = sum(
        1 for p_char, r_char in zip(prediction, reference) if p_char == r_char
    ) / max(len(reference), 1)

    return {
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "avg_log_prob": avg_log_prob,
        "cross_entropy": cross_entropy,
        "edit_distance": edit_distance,
    }

def sequence_exact_from_uniform_p(p: float, n: int) -> float:
    assert 0.0 <= p <= 1.0
    assert n >= 0
    return p ** n

# 玩具例子：5 位数任务中，每位正确率从 0.8 升到 0.95
em_80 = sequence_exact_from_uniform_p(0.8, 5)
em_95 = sequence_exact_from_uniform_p(0.95, 5)
assert round(em_80, 4) == 0.3277
assert round(em_95, 4) == 0.7738

# 单条样本评估
result = evaluate_prediction(
    prediction="5358",
    reference="5359",
    correct_token_probs=[0.99, 0.98, 0.97, 0.60]
)
assert result["exact_match"] == 0
assert result["edit_distance"] == 1
assert result["token_accuracy"] == 0.75
assert result["cross_entropy"] > 0

print(result)
```

这个函数输出五类信息：

| 指标 | 含义 | 适合回答的问题 |
|---|---|---|
| `exact_match` | 是否整条序列完全正确 | 最终能不能直接上线 |
| `token_accuracy` | 逐位正确比例 | 错误是局部还是整体 |
| `avg_log_prob` | 正确 token 的平均对数概率 | 模型对正确答案有多确定 |
| `cross_entropy` | 平均不确定性成本 | 训练和版本比较是否改善 |
| `edit_distance` | 最少改几个字符才能变成标准答案 | 输出离正确答案还有多远 |

如果你在真实评估脚本里接入模型推理，通常会在 `for token in reference_tokens` 的循环中不断累加 `log_probs_sum`，并在整条样本结束后把 `exact_match`、`avg_log_prob`、`edit_distance` 一起写入 CSV。后续画趋势图时，最有价值的通常不是“某一天精确匹配突然抬升”，而是“过去 10 个版本的平均 log-prob 是否持续改善”。

---

## 工程权衡与常见坑

精确匹配不是错指标，而是不完整指标。它适合做最终门槛，不适合单独用来判断能力形成过程。

常见坑可以直接列出来：

| 常见坑 | 具体表现 | 为什么会发生 | 缓解方法 |
|---|---|---|---|
| 只看精确匹配 | 曲线长期接近 0，后来突然抬升 | 连乘把渐进提升压成台阶 | 并行记录 token log-prob |
| 样本太少 | 小模型结果全是 0 或波动很大 | 离散指标方差高 | 扩大测试集，报告置信区间 |
| 序列太长 | 每位都不错，但全对率很低 | $p^n$ 随长度衰减 | 同时看长度归一化指标 |
| 错误成本不区分 | 只差 1 位和全错都记 0 | 精确匹配丢失中间信息 | 加上 edit distance |
| 过早部署 | 看到“飞跃”就认为模型刚学会 | 误把指标效应当能力突变 | 看趋势面板而非单点 |

真实工程里，金融账务是典型例子。假设一个字段有很多位，团队只看“整字段精确命中率”。某次版本升级后，精确匹配从 13% 提高到 60%，大家容易说“模型终于学会了”。但如果拆开看，可能只是每位正确率从 98% 提到 99%。问题不在提升不重要，而在结论不准确：这不是“突然学会”，而是持续改进跨过了一个由乘法制造出来的视觉门槛。

另一个常见坑是把 token 级指标当成上线指标。对数概率很适合监控趋势，但它并不直接等于“业务可接受”。模型可能平均 log-prob 变好了，但某些关键位仍然不能错。所以工程上更合理的做法是双层结构：用连续指标看趋势，用精确匹配做验收。

---

## 替代方案与适用边界

更稳妥的评测不是“换掉精确匹配”，而是做多指标面板。不同指标回答不同问题，组合起来才接近真实能力。

| 指标 | 敏感度 | 易解释性 | 适用任务 | 局限 |
|---|---|---|---|---|
| Exact Match | 对最终结果最敏感 | 很强 | 严格结构化输出 | 容易制造阶跃 |
| Avg Log-Prob | 对微小进步敏感 | 中等 | 训练趋势、版本比较 | 需要模型概率输出 |
| Cross Entropy | 数学上稳定 | 中等 | 训练与评估统一 | 对业务方不直观 |
| Token Accuracy | 易理解 | 强 | 定长结构化任务 | 不反映局部置信度 |
| Edit Distance | 能区分“差一点” | 强 | 文本纠错、算术串 | 对语义类任务有限 |
| BLEU/ROUGE | 适合文本重叠 | 中等 | 摘要、翻译 | 对严格正确性不足 |

对于多位算术、验证码解析、账单字段抽取这类任务，推荐的最小面板是：

- `Exact Match`
- `Avg Log-Prob`
- `Edit Distance`

简易策略可以这样设定：调参阶段主要盯 `avg_log_prob` 的改善幅度和 `edit_distance` 的下降速度；只有当这两个指标都稳定向好时，才期待精确匹配持续上升。最终上线前，再用精确匹配做硬验收。这种流程比“等 exact match 跳起来再判断”更稳，因为它能更早看到趋势。

适用边界也要讲清楚。若任务是开放式问答，token 对齐本身就不严格，逐位概率和编辑距离的解释力会下降；若任务允许多种正确答案，精确匹配更容易低估能力。这时应该换成语义级评估或人工审核。但在多位算术这种“答案唯一、位置严格”的任务中，本文讨论的结论很强：阶跃图像并不自动意味着真实的阶段突变。

---

## 参考资料

- Schaeffer et al. (2023), *Are Emergent Abilities of Large Language Models a Mirage?*  
  链接：https://www.scribd.com/document/859638811/NeurIPS-2023-Are-Emergent-Abilities-of-Large-Language-Models-a-Mirage-Paper-Conference  
  要点：提出“涌现”可能是评测指标的产物，核心贡献是用数学形式和实验说明 per-token 改进可以是平滑的，而 exact match 会显示台阶。

- Michael Brenndoerfer, *Emergence in Neural Networks / Phase Transitions / Scaling*  
  链接：https://mbrenndoerfer.com/writing/emergence-neural-networks-phase-transitions-scaling  
  要点：用 $p^n$ 的可视化直观展示“每位小幅提升如何变成序列级大跳变”，非常适合解释多位数任务中的乘法放大效应。

- Dhiria, *Emergent abilities in large language models: reality or mirage?*  
  链接：https://www.dhiria.com/index.php/en/blog/emergent-abilities-in-large-language-models-reality-or-mirage  
  要点：用具体算术例子解释“高位准确率与整体命中率”的关系，适合作为面向初学者的直观补充材料。
