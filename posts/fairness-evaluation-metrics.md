## 核心结论

Demographic Parity、Equalized Odds、Calibration 都在回答“模型是否公平”，但它们看的不是同一件事。

Demographic Parity（人口统计平等，白话就是“不同群体拿到正向结果的比例要接近”）只看输出分配，不看真实标签是否相同。它适合“机会配额”有硬约束的场景，比如活动推荐、面试邀请、资源分发。

Equalized Odds（等化机会误差，白话就是“在同样真实情况的人里，不同群体被模型判对或判错的概率要接近”）看的是错误率结构。它要求在 $Y=1$ 和 $Y=0$ 两种真实标签下，各组的预测行为都一致。它适合贷款、风控、医疗分诊这类“错误代价很高”的场景。

Calibration（校准，白话就是“模型打出的 0.8 分，真的大约意味着 80% 概率”）关注分数是否可解释。它通常用于风险评分，因为业务常先看分数，再决定阈值或人工复核。

下表先给出最重要的比较：

| 指标 | 数学关注点 | 直观含义 | 更适合的场景 | 典型限制 |
|---|---|---|---|---|
| Demographic Parity | $\mathbb{P}(\hat Y=1 \mid A=a)$ | 各组正类预测率接近 | 配额、曝光、机会分配 | 可能忽略真实资格差异 |
| Equalized Odds | $\mathbb{P}(\hat Y=1 \mid A=a, Y=y)$ | 各组在真阳性/假阳性上接近 | 信贷、医疗、司法风险 | 通常会牺牲部分准确率或需组别阈值 |
| Calibration | $\mathbb{P}(Y=1 \mid A=a, s)=s$ | 同分数在各组含义一致 | 风险评分、排序、人工复核 | 与前两者常发生冲突 |

玩具例子：贷款审批里，A 组和 B 组都各有 100 人。若模型给 A 组批了 60 人，给 B 组批了 60 人，则满足 Demographic Parity；但如果 A 组的错批率明显高于 B 组，就不满足 Equalized Odds。换句话说，前者管“批了多少”，后者管“批得准不准，而且各组要一样准”。

真实工程例子：信用评分系统常先输出一个违约风险分数，再由业务设定阈值决定是否放贷。此时如果业务重点是“相同风险分数代表相同违约概率”，Calibration 很重要；如果重点是“合格借款人不该因为群体属性更容易被拒”，则 Equalized Odds 更贴近业务目标。

---

## 问题定义与边界

先定义四个符号：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $A$ | 敏感属性 | 需要重点观察的群体划分，如性别、年龄段、地区 |
| $Y$ | 真实标签 | 真实结果，例如“是否会违约”“是否真的患病” |
| $\hat Y$ | 二分类预测 | 模型最终给出的通过/拒绝、阳性/阴性 |
| $s$ | 预测分数 | 模型的置信度或风险分数，通常在 $[0,1]$ |

三个常用定义可以写成：

$$
\text{Demographic Parity: } \mathbb{P}(\hat Y=1 \mid A=a)=\mathbb{P}(\hat Y=1 \mid A=b)
$$

$$
\text{Equalized Odds: } \mathbb{P}(\hat Y=1 \mid A=a, Y=y)=\mathbb{P}(\hat Y=1 \mid A=b, Y=y), \quad y \in \{0,1\}
$$

$$
\text{Calibration: } \mathbb{P}(Y=1 \mid A=a, s=r)=r
$$

这里的边界很关键：这些指标不是“普适真理”，而是“业务上选择强调哪一种公平”。如果两个群体的基准率不同，也就是 $\mathbb{P}(Y=1 \mid A=a) \neq \mathbb{P}(Y=1 \mid A=b)$，那么三个目标通常无法同时成立。

一个最小例子就能看出冲突：

| 组别 | 总人数 | 真实正例数 | 正例率 |
|---|---:|---:|---:|
| A | 10 | 8 | 0.8 |
| B | 10 | 5 | 0.5 |

如果你要求 Demographic Parity，那么两组必须拿到相同通过率，比如都通过 60%。但这会忽略真实正例率本来不同。  
如果你要求 Equalized Odds，那么你需要同时对齐：

- 真阳性率 $TPR=\mathbb{P}(\hat Y=1 \mid Y=1)$
- 假阳性率 $FPR=\mathbb{P}(\hat Y=1 \mid Y=0)$

这比只对齐通过率更严格。  
如果你再要求 Calibration，那么同样的 0.7 分在 A、B 两组都必须真的接近 70% 正例概率。只要组间基准率差异明显，这个目标往往会和 Equalized Odds 发生张力。

所以，讨论公平性前，先回答三个边界问题：

| 问题 | 如果回答“是” | 更偏向的指标 |
|---|---|---|
| 是否强调机会分配比例一致？ | 谁被看见、被邀请、被推荐更重要 | Demographic Parity |
| 是否强调错误成本在群体间对齐？ | 错杀和漏放都要控制 | Equalized Odds |
| 是否强调分数本身可解释、可比较？ | 决策者依赖风险分数而非直接分类 | Calibration |

---

## 核心机制与推导

Demographic Parity 的核心是：不条件化真实标签 $Y$。它只看每个群体最终有多少人被预测为正类。

把每个群体压成一个 $2 \times 2$ 分布表，可以更直观看：

**Demographic Parity 看的表：**

| 组别 | $\hat Y=1$ | $\hat Y=0$ |
|---|---:|---:|
| A | 60 | 40 |
| B | 60 | 40 |

只要每行的正类比例一致，它就满意。至于这 60 人里到底多少是真的合格者，它不直接关心。

Equalized Odds 的机制不同。它会先按真实标签 $Y$ 切开，再看每个切片里，不同组别的预测是否一致。也就是要同时比较两个表。

**在 $Y=1$ 条件下看的表：**

| 组别 | $\hat Y=1$ | $\hat Y=0$ | 对应指标 |
|---|---:|---:|---|
| A | 48 | 12 | TPR |
| B | 30 | 10 | TPR |

这里要求两组 TPR 接近，也就是“真正该通过的人，被通过的比例要接近”。

**在 $Y=0$ 条件下看的表：**

| 组别 | $\hat Y=1$ | $\hat Y=0$ | 对应指标 |
|---|---:|---:|---|
| A | 8 | 32 | FPR |
| B | 20 | 40 | FPR |

这里要求两组 FPR 接近，也就是“本不该通过的人，被误通过的比例要接近”。

因此，Demographic Parity 和 Equalized Odds 的根本区别是：

- 前者看输出边际分布，只关心 $\hat Y$
- 后者看条件分布，要关心 $(\hat Y \mid Y)$

这也是为什么 Equalized Odds 更难满足。它要求模型在“真实好人”和“真实坏人”两个子人群里都表现一致。

Calibration 再往前一步，它不直接看二值预测，而是看分数 $s$ 是否有概率意义。理想条件是：

$$
\mathbb{P}(Y=1 \mid A=a, s=0.8)=0.8
$$

白话解释：在 A 组中，所有被打成 0.8 分的人，最终大约 80% 真的为正例；在 B 组中也一样。这样业务才能说“0.8 分在所有组里都代表相同风险”。

但这会引出一个经典冲突。若两组基准率不同，想让分数完全校准，同时又让固定阈值下的 TPR/FPR 完全一致，通常做不到。因为校准保证“分数含义一致”，而 Equalized Odds 保证“阈值后的错误率一致”，二者在基准率不同的情况下会把分数分布往不同方向拉。

一句话概括推导逻辑：

- Demographic Parity：约束 $\hat Y$ 的组间边际分布
- Equalized Odds：约束 $\hat Y$ 在给定 $Y$ 后的组间条件分布
- Calibration：约束 $Y$ 在给定 $s$ 后的组间条件分布

它们条件化的变量不同，所以目标也不同。

---

## 代码实现

下面先给一个能跑的玩具数据，再计算三个常见量：正类预测率、TPR/FPR、分箱校准结果。

```python
from collections import defaultdict

rows = [
    # group, y_true, y_pred, score
    ("A", 1, 1, 0.92),
    ("A", 1, 1, 0.88),
    ("A", 1, 1, 0.81),
    ("A", 1, 0, 0.49),
    ("A", 0, 1, 0.74),
    ("A", 0, 0, 0.31),
    ("A", 0, 0, 0.20),
    ("A", 0, 0, 0.10),

    ("B", 1, 1, 0.91),
    ("B", 1, 0, 0.45),
    ("B", 1, 0, 0.40),
    ("B", 0, 1, 0.71),
    ("B", 0, 1, 0.69),
    ("B", 0, 0, 0.33),
    ("B", 0, 0, 0.22),
    ("B", 0, 0, 0.11),
]

def positive_rate(rows, group):
    xs = [r for r in rows if r[0] == group]
    return sum(r[2] for r in xs) / len(xs)

def tpr(rows, group):
    xs = [r for r in rows if r[0] == group and r[1] == 1]
    return sum(r[2] for r in xs) / len(xs)

def fpr(rows, group):
    xs = [r for r in rows if r[0] == group and r[1] == 0]
    return sum(r[2] for r in xs) / len(xs)

def calibration_by_bin(rows, bins=(0.0, 0.5, 0.8, 1.0)):
    out = defaultdict(list)
    for g, y, _, s in rows:
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if lo <= s < hi or (i == len(bins) - 2 and s == hi):
                out[(g, i)].append((s, y))
                break

    result = {}
    for key, vals in out.items():
        avg_score = sum(v[0] for v in vals) / len(vals)
        avg_label = sum(v[1] for v in vals) / len(vals)
        result[key] = {"avg_score": round(avg_score, 3), "avg_label": round(avg_label, 3)}
    return result

a_pr = positive_rate(rows, "A")
b_pr = positive_rate(rows, "B")
a_tpr = tpr(rows, "A")
b_tpr = tpr(rows, "B")
a_fpr = fpr(rows, "A")
b_fpr = fpr(rows, "B")

assert round(a_pr, 3) == 0.5
assert round(b_pr, 3) == 0.375
assert round(a_tpr, 3) == 0.75
assert round(b_tpr, 3) == 0.333
assert round(a_fpr, 3) == 0.25
assert round(b_fpr, 3) == 0.5

calib = calibration_by_bin(rows)
assert ("A", 2) in calib
assert ("B", 2) in calib

print("PR", a_pr, b_pr)
print("TPR", a_tpr, b_tpr)
print("FPR", a_fpr, b_fpr)
print("Calibration bins", calib)
```

输入数据可以先理解成这样：

| 组别 | 真实标签 $Y$ | 预测 $\hat Y$ | 分数 $s$ |
|---|---:|---:|---:|
| A | 1 | 1 | 0.92 |
| A | 1 | 0 | 0.49 |
| A | 0 | 1 | 0.74 |
| B | 1 | 0 | 0.45 |
| B | 0 | 1 | 0.71 |
| B | 0 | 0 | 0.22 |

从这段代码可以直接得到：

- Demographic Parity 看 `positive_rate`
- Equalized Odds 看 `tpr` 和 `fpr` 是否组间接近
- Calibration 看同一分数桶内 `avg_score` 与 `avg_label` 是否接近

真实工程里，通常有三步：

1. 训练模型，保留原始分数 $s$
2. 按组统计各项指标，不只看总体准确率
3. 决定是改阈值、改损失函数、还是只做监控告警

真实工程例子：招聘筛选系统会先给候选人打匹配分，再设置面试阈值。如果产品要求“面试邀请比例稳定”，会盯 Demographic Parity；如果法务更关注“合格候选人不该因群体属性更容易被漏掉”，就会重点统计 TPR gap；如果后续还有人工复核，则分数校准质量会直接影响复核效率。

---

## 工程权衡与常见坑

公平性指标真正难的部分不在定义，而在“你愿意让谁承担代价”。

| 常见坑 | 触发条件 | 风险 |
|---|---|---|
| 只看总体准确率 | 数据不均衡、组间样本量差异大 | 某一组可能被系统性误伤 |
| 把 Demographic Parity 当通用标准 | 各组真实正例率差异大 | 可能强行调配额，降低有效决策质量 |
| 把 Equalized Odds 当免费改造 | 模型原始分数分布差异大 | 往往需要组别阈值，甚至随机化 |
| 用少量样本做校准判断 | 稀疏分箱、长尾群体 | 结论不稳定，容易误判 |
| 忽略标签本身偏差 | 历史标签含人为歧视 | “公平评测”只是复现旧偏见 |

Equalized Odds 的工程实现常常需要后处理。经典做法来自 Hardt 等人的思路：在不同群体上设置不同阈值，必要时在两个阈值之间做随机化。白话解释是，对一段“灰区分数”不直接判定，而是按概率通过一部分样本，用来把 TPR/FPR 调到目标范围。

这听起来反直觉，但它揭示了现实：若原始模型分布已经不公平，后处理想精确满足 Equalized Odds，通常不能只靠一个统一阈值。

真实工程例子：FICO 信用评分里，若业务要求不同群体的错拒率和错放率接近，那么简单统一阈值往往不够。系统可能需要：

- A 组阈值设为 0.72
- B 组阈值设为 0.68
- 对 0.66 到 0.68 的 B 组申请人做概率性放贷或人工复核

这会带来额外成本：实现更复杂、审计更难、用户解释更难、合规压力更大。

另一个常见坑是把 Calibration 理解成“模型已经公平”。不是。Calibration 只说明“分数有概率意义”，不说明最终二值决策公平。两个组都可能是完美校准的，但在同一阈值下 TPR/FPR 差异仍然很大。

所以工程上应先定优先级，而不是同时喊所有口号。一个实用判断流程是：

| 业务问题 | 优先看的东西 |
|---|---|
| 资源发放是否均衡 | Demographic Parity gap |
| 错误是否由某组更多承担 | TPR/FPR gap, Equalized Odds |
| 分数能否给人工或下游系统解释 | Calibration curve / ECE |
| 历史标签是否可信 | 标签审计，而不是先调公平指标 |

---

## 替代方案与适用边界

如果你发现某个指标代价太高，不代表只能放弃公平性。更常见做法是换成更适合业务链路的约束。

| 方案 | 适用边界 | 优点 | 副作用 |
|---|---|---|---|
| Demographic Parity | 强调机会分配 | 目标直观，便于沟通 | 可能牺牲有效性 |
| Equalized Odds | 强调错误率公平 | 贴近高风险决策 | 实现复杂，常需组阈值 |
| Calibration | 强调分数解释性 | 适合评分系统 | 不保证阈值后公平 |
| 公平正则化 | 训练阶段可改模型 | 不必全靠后处理 | 调参难、效果不稳定 |
| 单调约束/业务规则 | 某些特征关系必须稳定 | 容易落地，便于审计 | 只能缓解，不能替代正式评测 |
| 先分数再决策 | 有人工复核或多级流程 | 将公平问题拆成评分与决策两层 | 流程更长，运营成本更高 |

一个实用替代路径是“先保证分数可解释，再单独设计决策阈值”。例如：

1. 先训练并校准分数，让 0.8 在各组都接近 80% 风险含义
2. 再根据业务目标设置阈值
3. 对阈值附近样本进入人工复核，而不是硬切

这比直接要求单一公平指标更符合很多实际系统，因为现实里决策不是一次二分类，而是“自动通过 + 自动拒绝 + 人工复核”三段式流程。

玩具例子：欺诈检测中，$s > 0.9$ 直接拦截，$s < 0.2$ 直接放行，中间交给人工。此时 Calibration 往往比 Demographic Parity 更关键，因为人工团队要相信分数含义。若后续发现某组的漏检率过高，再针对中间区间做阈值调整或加公平正则化，通常比强压整体通过率更稳。

适用边界可以这样记：

- 想控制“谁得到机会”，先看 Demographic Parity
- 想控制“谁承担错误”，先看 Equalized Odds
- 想控制“分数是否可信”，先看 Calibration
- 想兼顾多目标，通常需要训练约束、后处理和人工流程一起设计

---

## 参考资料

1. Fairlearn Documentation, *Fairness in Machine Learning*. 用途：指标定义、公式、工程解释，适合入门时建立统一符号和术语。
2. Moritz Hardt, Eric Price, Nati Srebro, *Equality of Opportunity in Supervised Learning*. 用途：Equalized Odds 与后处理方法的经典来源，尤其是组别阈值与随机化思想。
3. Geoff Pleiss et al., *On Fairness and Calibration*. 用途：解释校准与其他群体公平指标之间的冲突关系。
4. Hildeweerts, *Responsible Machine Learning: Group Fairness Metrics*. 用途：对不可能性结果、图示化解释和实践边界总结得较清楚。
5. Fairlearn 团队教程与示例。用途：快速查看如何在真实预测结果上计算 group fairness 指标。
6. 继续阅读方向：不可能性定理、Expected Calibration Error（ECE）、阈值移动与业务决策成本之间的关系。
