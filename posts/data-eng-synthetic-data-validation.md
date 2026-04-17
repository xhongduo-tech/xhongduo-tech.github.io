## 核心结论

合成数据质量验证不是“看起来像”就够，而是要回答四个独立问题：它是否覆盖了足够多的样本形态，它是否保留了关键统计关系，它是否会在真实任务上产生可用结果，它是否引入了泄露或伪规律。这里的“多样性”可以理解为样本空间有没有被压扁，“一致性”是指列分布、联合分布、相关结构是否仍然成立，“真实性”是指数据里有没有违背业务规律的假模式，“有用性”则是指它拿去训练后，真实测试集性能是否还能成立。

真正可落地的做法不是单一指标，而是一条验证流水线：先做统计分布对比，再做下游任务验证，再做人工审查和风险检查，最后输出“通过、退回、重生成”信号。只有把这条链路做成批次化、可审计、可回溯的流程，合成数据才有资格进入训练、微调或评估系统。

下面这张表可以直接当作治理清单使用：

| 质量维度 | 对应 metric | 触发动作 | 责任人 |
| --- | --- | --- | --- |
| 多样性 | 唯一值覆盖率、类别覆盖率、MMD | 覆盖不足则增大采样温度或补充条件生成 | 数据工程 |
| 一致性 | KL 散度、JSD、相关矩阵差异 | 超阈值则回退本批并重调生成参数 | 数据工程 |
| 真实性 | 规则校验、异常频率、人审 | 发现伪模式则修改模板或约束 | 领域专家 |
| 有用性 | TSTR 精度、F1、AUC、回归误差 | 低于基线则禁止入模 | 机器学习工程师 |
| 隐私/可识别性 | 重复率、近邻距离、泄露检测器 | 疑似泄露则整批废弃并复盘 | 安全/合规 |

一个金融违约分类器就是典型例子：先用合成贷款记录训练模型，再在真实保留集上测试；只有当 KL、MMD、TSTR AUC 都在阈值内，并且人工确认收入、负债、历史逾期等关键变量分布没有失真时，这批数据才能进入 fine-tune 或监督训练流程。

---

## 问题定义与边界

合成数据质量验证，本质上是一套交叉验证机制，用来判断“这批人工生成的数据，能不能在指定边界内替代或补充真实数据”。这里的“边界”非常重要，因为合成数据从来不是无条件等价于真实数据。

先把边界拆开：

| 边界问题 | 需要回答的核心问题 | 典型手段 |
| --- | --- | --- |
| 统计边界 | 分布像不像真实数据 | KL、MMD、相关矩阵、漂移报告 |
| 任务边界 | 拿去训练后是否有效 | TSTR、真实保留集评估 |
| 风险边界 | 会不会泄露、伪造、误导 | 泄露检测、重复样本扫描、人审 |
| 治理边界 | 出问题时能否定位责任和批次 | 验证日志、版本号、审批记录 |

对零基础读者，可以把它理解成“数据 QA 流程”。软件工程里，代码上线前有测试、回归、审计；同样，合成数据进入模型前也必须经过 QA。流程通常是：生成候选数据集，做统计对比，做下游训练验证，做人工审查，写入验证卡，最后才允许推进。

这一步为什么不能省略？因为很多失败不是“数据完全错了”，而是“局部规律悄悄坏了”。比如总体年龄分布看起来合理，但“高收入且低负债的老年用户违约率异常偏高”；如果只看均值和方差，问题会被掩盖，但模型训练后会在真实人群上产生错误决策。

所以，验证的目标不是证明“合成数据与真实数据完全相同”。那既做不到，也不应该追求。目标是证明“在当前任务和风险约束下，差异仍在可接受范围内”。这就是工程里的可用边界。

---

## 核心机制与推导

最常见的统计验证指标有两个：KL 散度和 MMD。KL 散度可以理解为“如果拿合成分布去近似真实分布，会多付出多少信息损失”；MMD 可以理解为“把样本映射到一个核空间后，两批样本的整体均值是否还接近”。

KL 散度定义为：

$$
D_{KL}(P \| Q)=\sum_i P(i)\log \frac{P(i)}{Q(i)}
$$

其中，$P$ 是真实分布，$Q$ 是合成分布。值越小，说明合成分布越接近真实分布。它不对称，也就是 $D_{KL}(P\|Q)$ 和 $D_{KL}(Q\|P)$ 一般不同。

MMD 的常见形式是：

$$
\mathrm{MMD}^2(P,Q)=\mathbb{E}_{x,x'}[k(x,x')] + \mathbb{E}_{y,y'}[k(y,y')] - 2\mathbb{E}_{x,y}[k(x,y)]
$$

这里的 $k(\cdot,\cdot)$ 是核函数，常用高斯核。直观上，MMD 不只看单列分布，还能在一定程度上捕捉高维联合差异，因此适合检查“边缘分布都像，但组合关系已经坏掉”的情况。

先看一个玩具例子。假设真实标签服从 Bernoulli(0.6)，也就是正样本概率为 $0.6$；合成标签服从 Bernoulli(0.4)$。那么：

$$
D_{KL}(0.6 \| 0.4)=0.6\ln\frac{0.6}{0.4}+0.4\ln\frac{0.4}{0.6}
$$

近似为：

$$
0.6\ln(1.5)+0.4\ln(0.666)\approx 0.085
$$

如果团队把单字段 KL 阈值设为 $0.05$，那么这一列就已经不合格。这个例子简单，但它说明一个基本事实：只要生成器把一个二元变量的概率推偏一点，信息损失就能被量化出来。

但仅靠 KL 还不够。原因有两个：

1. KL 更适合离散分布或离散化后的连续分布，且对分桶方式敏感。
2. 单列 KL 可能都不大，但特征之间的关系已经失真。

所以真实工程里通常会把指标组合起来：

| 指标 | 检查对象 | 适合发现的问题 |
| --- | --- | --- |
| KL / JSD | 单列或离散化分布 | 类别比例、区间比例偏移 |
| MMD | 高维联合分布 | 组合关系失真、模式坍缩 |
| 相关矩阵差异 | 特征间线性关系 | 变量关联方向错误 |
| 唯一值覆盖率 | 多样性 | 样本模板化、重复生成 |
| TSTR | 下游效用 | 统计像但训练无用 |

这里还要强调“统计像”和“任务有用”不是一回事。假设某批合成客服对话在词频、句长、标签比例上都接近真实数据，但它把少数类投诉场景写得过于模板化，那么训练出来的分类器可能在真实长尾投诉上明显退化。也就是说，统计指标过关，不等于下游性能过关。

因此，最可靠的机制是“两层验证”：

1. 统计层：判断分布和关系是否接近。
2. 任务层：判断训练后在真实保留集上是否有效。

如果统计层好、任务层差，常见原因是模式保真度不足，也就是数据表面像，决策边界却错了。这时应回溯生成策略，而不是盲目放宽阈值。

---

## 代码实现

工程上最实用的范式是 TSTR，英文是 Train on Synthetic, Test on Real，意思是“在合成数据上训练，在真实数据上测试”。它直接回答一个问题：这批合成数据能不能支撑真实任务。

下面给出一个可运行的极简 Python 示例，演示如何把 KL、MMD 和 TSTR 风格的验证串起来。为了保证代码不依赖第三方库，这里用最小实现展示思路。

```python
import math
from collections import Counter

def kl_divergence(real, synth):
    keys = sorted(set(real) | set(synth))
    real_total = sum(real.values())
    synth_total = sum(synth.values())
    eps = 1e-12
    kl = 0.0
    for k in keys:
        p = real.get(k, 0) / real_total
        q = synth.get(k, 0) / synth_total
        if p > 0:
            kl += p * math.log(p / max(q, eps))
    return kl

def rbf(x, y, gamma=1.0):
    return math.exp(-gamma * ((x - y) ** 2))

def mmd_1d(xs, ys, gamma=1.0):
    xx = sum(rbf(a, b, gamma) for a in xs for b in xs) / (len(xs) * len(xs))
    yy = sum(rbf(a, b, gamma) for a in ys for b in ys) / (len(ys) * len(ys))
    xy = sum(rbf(a, b, gamma) for a in xs for b in ys) / (len(xs) * len(ys))
    return xx + yy - 2 * xy

def fit_threshold_classifier(samples):
    # samples: [(score, label)]
    pos = [x for x, y in samples if y == 1]
    neg = [x for x, y in samples if y == 0]
    threshold = (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2
    return threshold

def eval_threshold_classifier(threshold, samples):
    correct = 0
    for x, y in samples:
        pred = 1 if x >= threshold else 0
        correct += int(pred == y)
    return correct / len(samples)

def verify_batch(real_label_counts, synth_label_counts, real_scores, synth_train, real_test,
                 kl_thresh=0.05, mmd_thresh=0.10, acc_floor=0.75):
    kl = kl_divergence(real_label_counts, synth_label_counts)
    mmd = mmd_1d(real_scores, [x for x, _ in synth_train], gamma=0.5)
    threshold = fit_threshold_classifier(synth_train)
    acc = eval_threshold_classifier(threshold, real_test)

    passed = (kl <= kl_thresh) and (mmd <= mmd_thresh) and (acc >= acc_floor)
    return passed, {"kl": kl, "mmd": mmd, "tstr_acc": acc}

# 玩具例子
real_counts = Counter({1: 60, 0: 40})
synth_counts = Counter({1: 40, 0: 60})
kl = kl_divergence(real_counts, synth_counts)
assert round(kl, 3) == 0.081 or round(kl, 3) == 0.082  # 取决于浮点误差

# 真实工程风格的简化例子：贷款评分越高越可能违约
real_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
synth_train = [(0.15, 0), (0.25, 0), (0.35, 0), (0.65, 1), (0.75, 1), (0.85, 1)]
real_test = [(0.12, 0), (0.22, 0), (0.32, 0), (0.72, 1), (0.82, 1), (0.92, 1)]

passed, metrics = verify_batch(real_counts, Counter({1: 58, 0: 42}),
                               real_scores, synth_train, real_test,
                               kl_thresh=0.05, mmd_thresh=0.10, acc_floor=0.80)

assert metrics["tstr_acc"] >= 0.80
assert isinstance(passed, bool)
print(passed, metrics)
```

这个示例故意保持简单，但结构已经足够清晰：

1. `kl_divergence` 检查离散标签分布是否偏移。
2. `mmd_1d` 用一维数值特征演示联合分布距离思想。
3. `fit_threshold_classifier` 和 `eval_threshold_classifier` 模拟 TSTR。
4. `verify_batch` 给出统一的“通过/失败”信号。

真实工程里，通常会把它扩展成批处理任务。以金融违约分类器为例，流程可以写成：

| 阶段 | 输入 | 输出 | 关键记录 |
| --- | --- | --- | --- |
| 生成候选批次 | 真实样本、生成参数 | synthetic batch v17 | 生成器版本、prompt、随机种子 |
| 统计验证 | real vs synthetic | KL/MMD/相关矩阵报告 | 指标值、阈值、是否超限 |
| 下游验证 | synthetic train + real holdout | TSTR AUC/F1 | 模型版本、基线差值 |
| 人工审查 | 报告 + 样本抽检 | 审查结论 | 审查人、日期、备注 |
| 决策 | 全部结果 | pass / reject / regenerate | 审批记录、批次状态 |

这里的“真实工程例子”可以具体化为贷款审批场景。假设真实数据里，`income`、`debt_ratio`、`delinquency_count` 三个字段共同决定违约风险。生成器做出来的一批合成数据，在单列分布上都接近真实数据，但 TSTR AUC 从真实训练时的 `0.86` 掉到 `0.79`。排查发现，合成数据中“高收入且高逾期”的样本过少，导致模型学不到关键边界。此时正确动作不是“反正 KL 很小就放行”，而是把失败批次打回，调整条件采样策略，专门补强这类长尾组合。

---

## 工程权衡与常见坑

第一个常见坑是只看单点指标，尤其是只看 accuracy。accuracy 很容易掩盖类不平衡和长尾崩溃。比如一个法规问答分类模型，大类样本很多，小类异常案例很少。合成数据把大类生成得很像，accuracy 仍然很高，但小类模式被模板化后，真实线上问题会集中出错。

第二个坑是把统计相似误当成业务真实。统计相似只说明“数值上像”，不说明“机制上对”。一个医疗合成数据集可能把年龄、性别、化验值边缘分布都拟合得很好，但如果它错误地削弱了“某药物使用史与肝功能异常”的关系，那么它依然不适合训练诊断模型。

第三个坑是没有失败溯源。很多团队能给出一个“本批不合格”的结论，但说不清是哪里坏了。工程上必须把失败类型结构化，否则无法指导下一轮生成。一个简单的失败分类就很有用：

| 问题 | 结果 | 规避策略 |
| --- | --- | --- |
| 只看 accuracy | 少数类和长尾样本失真 | 加入 KL、MMD、分组指标、人审 |
| 只看单列分布 | 联合关系损坏 | 增加 MMD、相关矩阵、条件分布检查 |
| 无真实保留集回归 | 训练有效性未知 | 强制 TSTR 或 TSRT 回归测试 |
| 无审计记录 | 无法定位版本责任 | 记录批次号、参数、审批人 |
| 生成失败后不调策略 | 同类错误反复出现 | 建立失败标签到生成器参数的映射 |

这里有一个很实用的溯源思路。验证失败后，不要只记录“失败”，而是记录“失败在哪一层”：

| 失败层 | 典型信号 | 可能原因 | 调整方向 |
| --- | --- | --- | --- |
| 分布层 | KL 超阈值 | 类别概率偏移、采样温度失衡 | 重新加权或条件采样 |
| 关系层 | MMD/相关矩阵异常 | 联合结构失真 | 增强约束生成或规则后处理 |
| 任务层 | TSTR 掉点 | 长尾样本缺失、标签噪声 | 补充难例、重做标签生成 |
| 风险层 | 重复率高、近邻过近 | 记忆训练样本 | 降低泄露风险、过滤近邻 |
| 业务层 | 人审不通过 | 伪规律、违背领域知识 | 加业务规则或人工模板 |

第四个坑是把合成数据当成真实数据替身，而不是受约束的补充。低风险内部测试场景，可以接受简化版验证；但高风险、受监管场景，例如金融、医疗、风控，不应该只靠自动指标放行，必须有人审和版本治理。

---

## 替代方案与适用边界

标准做法是“统计验证 + 下游验证 + 人审”，但不是所有场景都要同样重。适用边界取决于风险、成本和数据用途。

下面给出一个对照表：

| 机制 | 适用边界 | 触发条件 |
| --- | --- | --- |
| 统计指标 + 人审 | 高风险、受监管场景 | 进入训练、微调、客户交付前 |
| 统计指标 + TSTR | 中高风险任务型场景 | 需要证明训练有效性时 |
| 简化统计检查 | 内部原型、离线实验 | 只做探索、不直接上线 |
| 多源生成 | 单一生成器容易坍缩时 | 模式覆盖不足、长尾缺失 |
| MMD 引导采样 | 高维联合分布偏差明显时 | 单列指标正常但任务掉点 |

“多源生成”可以理解为同时用多种生成策略产出样本，例如规则模板、语言模型、扰动扩增三条通路，再混合验证。它的价值在于减少模式坍缩，也就是不要让所有样本都长得像同一类模板。

“MMD 引导采样”则更适合高维特征场景。它的核心思想是：生成时就把“与真实分布的距离”纳入优化目标，而不是等生成完再被动验收。这样做实现成本更高，但在复杂表格数据、时序数据、图像特征空间中更有效。

在治理上，也可以按风险分层：

| 场景 | 推荐策略 | 不建议做法 |
| --- | --- | --- |
| 内部测试数据 | 基础统计检查 + 少量抽检 | 完全不验证 |
| 模型预训练补料 | 统计检查 + TSTR + 漂移监控 | 只看生成量 |
| 金融/医疗/风控 | 统计 + TSTR + 泄露检查 + 人审 + 审计 | 只凭自动评分放行 |

最后给一个准生产例子。某团队要生成金融欺诈样本做 fine-tune。第一轮检查发现总体欺诈率、金额分布、地区分布都基本对齐，异常频率也接近真实数据，但人工审查发现“夜间高频小额转账”这一典型欺诈模式出现次数明显偏低。虽然自动统计未报错，这批数据仍然不应进入训练。正确动作是记录失败原因，调整生成约束，再重新出一批。这说明高风险场景下，人审不是附属环节，而是验证闭环的一部分。

---

## 参考资料

- Synthetic Data Verifier, Emergent Mind: https://www.emergentmind.com/topics/synthetic-data-verifier
- Evaluating Synthetic Data, IBM watsonx Docs: https://dataplatform.cloud.ibm.com/docs/content/wsj/synthetic/evaluate_data_sd.html?context=wx
- Synthetic Data Pipelines That Don’t Collapse, Tian Pan: https://tianpan.co/blog/2026-04-12-synthetic-data-pipelines-that-dont-collapse
- Synthetic Training Data Quality Collapse, Tian Pan: https://tianpan.co/blog/2026-04-09-synthetic-training-data-quality-collapse
- Synthetic Data Generation Validation for Enterprise AI Deployment, Ranktracker: https://www.ranktracker.com/blog/validating-synthetic-data-generation/
- Generation and evaluation of synthetic patient data, BMC Medical Research Methodology: https://pmc.ncbi.nlm.nih.gov/articles/PMC7204018/
- Maximum Mean Discrepancy Loss, Emergent Mind: https://www.emergentmind.com/topics/maximum-mean-discrepancy-mmd-loss
- TSTR: Train on Synthetic, Test on Real, Emergent Mind: https://www.emergentmind.com/topics/train-on-synthetic-test-on-real-tstr-8e8fef3e-7dfb-492f-9022-b047d071df99
