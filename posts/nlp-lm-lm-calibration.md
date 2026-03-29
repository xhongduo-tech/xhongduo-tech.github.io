## 核心结论

校准，简单说，就是“模型有多自信，就应该大致有多大概率答对”。如果一个语言模型把一批回答都打到 90% 置信度，那么这批回答里理想情况下也应有约 90% 真正正确。这个性质不解决模型是否有知识，但决定系统能不能可靠地区分“可以自动执行”和“应该停下来”。

大型语言模型普遍存在过度自信。过度自信的白话解释是：模型不知道时，仍然把答案说得像知道。它在知识盲区里继续输出高置信度文本，会直接放大幻觉风险。幻觉，指模型生成了流畅但不真实的内容。

最常见的后处理校准方法是温度缩放。温度，指 softmax 前对 logits 做统一缩放的系数；$T>1$ 会把分数拉平，让模型少一点“拍板式自信”。它实现简单、部署成本低，但只能修正输出分布，不能让模型真正学会“不知道就拒答”。

更进一步的方法是 R-Tuning。它把“知道”和“不知道”显式纳入训练目标，让模型在不确定样本上学会表达 unsure 或直接拒答，在高风险任务里通常比单纯温度缩放更有效。可以把它理解成：不是只给模型降火，而是教模型什么时候闭嘴。

一个最小直觉例子如下：

| 场景 | 模型行为 | 是否校准 |
|---|---|---|
| 熟悉知识问答 | 说“我很确定”，且多数答对 | 接近校准 |
| 超出知识边界 | 仍然说“我很确定”，但多数答错 | 明显失准 |
| 经 R-Tuning 后 | 对不确定问题说“我不确定”或拒答 | 校准更好 |

玩具例子：问模型“2 的 10 次方是多少”，它给出 1024，置信度 0.98，这没有问题；再问“某个从未发布的内部 API 返回码含义是什么”，它仍给出具体解释且置信度 0.96，这就是典型的校准失败。

真实工程例子：金融或医疗 Agent 在多步调用外部工具时，不能只看“生成得像不像”，而要看“它的高置信度是否真的对应高正确率”。否则系统会把一个错误但自信的中间结论继续传给下游执行模块。

---

## 问题定义与边界

校准讨论的不是“模型总体准确率高不高”，而是“模型主观置信度和客观正确率是否一致”。准确率高但不校准，系统仍然难以设阈值；因为你不知道 0.9 置信度究竟代表“九成靠谱”，还是“只有六成靠谱”。

最常用的度量是 ECE，Expected Calibration Error，中文常译为期望校准误差。它的白话解释是：把预测按置信度分桶，比较每个桶里的平均置信度和真实正确率，再做加权平均。公式为：

$$
\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|
$$

其中：

- $B_m$ 是第 $m$ 个置信度区间里的样本集合
- $\mathrm{acc}(B_m)$ 是这个区间里的真实正确率
- $\mathrm{conf}(B_m)$ 是这个区间里的平均置信度
- $n$ 是总样本数

看一个两桶的玩具例子：

| Bin | 置信度区间 | 样本占比 | 平均 conf | 实际 acc | gap |
|---|---|---:|---:|---:|---:|
| $B_1$ | 0.8-1.0 | 0.5 | 0.9 | 0.8 | 0.1 |
| $B_2$ | 0.4-0.8 | 0.5 | 0.6 | 0.4 | 0.2 |

此时：

$$
\mathrm{ECE}=0.5\times|0.8-0.9|+0.5\times|0.4-0.6|=0.15
$$

这表示模型平均有 15 个百分点的置信偏差，而且方向是过度自信。

这里要明确边界。

第一，校准不是准确率的替代品。一个完全无知但总是报 0.5 置信度的模型，可能看起来“更校准”，但没有工程价值。

第二，校准和生成质量不是同一个指标。语言模型输出的是整段文本，不像分类器只输出一个类别，因此“正确率”本身就依赖任务定义。开放式问答、摘要、代码生成、工具调用，评价标准并不一样。

第三，校准常在“给定一个可计算置信度”的前提下讨论。对于生成模型，置信度可以取 top token 概率、序列概率、答案一致性分数，或者一个额外训练出的 verifier 分数。不同定义会直接影响 ECE 数值。

第四，RLHF 之后问题常更严重。RLHF，指基于人类偏好的强化学习或偏好优化；它会把“回答得像样”优化得更强，但不一定让“知道时自信，不知道时收敛”同步改善，因此经常出现更流畅、更坚定、但不更真实的回答。

如果把 reliability diagram 画出来，理想情况应接近对角线 $y=x$；曲线落在对角线下方，表示置信度高于真实正确率，也就是过度自信。

---

## 核心机制与推导

### 1. 温度缩放

温度缩放的核心非常直接。logits 是 softmax 前的原始分数，可以理解为模型对每个候选 token 的“未归一化偏好”。温度缩放把它改成：

$$
p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

其中 $z_i$ 是第 $i$ 个类别的 logit，$T$ 是温度。

- 当 $T=1$，概率不变
- 当 $T>1$，分布更平，最大概率下降，自信度被压低
- 当 $T<1$，分布更尖，模型更自信

为什么它能改善过度自信？因为很多失准来自“分数太尖锐”，不是类别排序完全错了。把 logits 拉平后，top-1 预测往往不变，但置信度更接近真实正确率。这也是它适合做后处理的原因：不改模型参数，不重训，只在验证集上找一个最优 $T$。

简化推导逻辑是：

1. 收集验证集预测
2. 计算不同 $T$ 下的概率分布
3. 用 NLL、ECE 或二者组合选最优 $T$
4. 上线时固定使用这个 $T$

但它有明显局限。全局一个 $T$ 默认“所有 token、所有领域、所有提示模板的偏差形态相同”，现实中通常不成立。RLHF 后尤其明显，因为不同 token 位置、不同指令类型的偏差并不一致。

### 2. ATS / ETS

ATS，Adaptive Temperature Scaling，可理解为“自适应温度缩放”。白话解释是：不是所有样本都用同一个温度，而是根据 token 特征或上下文状态，动态预测当前该用多大的 $T$。

它解决的是“全局温度太粗”的问题。一个简化示意是：

$$
T_t = f(h_t, p_t, \text{pos}_t, \ldots)
$$

其中 $h_t$ 可以是当前位置隐藏状态，$p_t$ 可以是当前 top probability，$\text{pos}_t$ 是 token 位置。函数 $f$ 输出该 token 的温度。这样做可以针对不同区域做细粒度校准，尤其适合 RLHF 后分布漂移较大的模型。

### 3. R-Tuning

R-Tuning 的关键不是重新分配概率，而是把“不确定性表达”变成训练目标。它先区分两类数据：

- $D_1$：模型有知识、应当正常回答的样本
- $D_0$：模型缺乏知识、应当表达 unsure 或拒答的样本

再把这种标签写进训练流程。简化理解如下：

| 数据类型 | 训练提示或标签 | 目标行为 |
|---|---|---|
| $D_1$ 确定样本 | `certain` | 给出正常答案 |
| $D_0$ 不确定样本 | `uncertain` 或 “I am unsure” | 拒答或明确说明不知道 |

这件事的本质是把“知识边界”显式告诉模型。普通监督微调只教模型“看到问题就输出答案”，而 R-Tuning 额外教它“有些问题的正确输出不是答案，而是拒答”。

可以把机制概括为：

$$
\text{目标}= \text{答对已知问题} + \text{拒答未知问题}
$$

因此它改善的不只是概率意义上的 calibration，还改善了策略层面的 risk control。对于问答系统，过度自信最危险的地方不在“分数算偏了”，而在“错得还特别肯定”。R-Tuning 正是在这里比温度缩放更强。

---

## 代码实现

下面先给一个最小可运行的 Python 例子，演示如何计算 ECE，并用简单网格搜索找温度 $T$。这个例子不依赖深度学习框架，便于先理解机制。

```python
import math

def softmax(logits, T=1.0):
    scaled = [x / T for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [e / s for e in exps]

def ece_from_probs(probs, labels, n_bins=5):
    # probs: list of predicted confidence for chosen class
    # labels: list of 0/1, whether chosen class is correct
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))

    n = len(probs)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - conf)
    return ece

def predict_confidence(logits_list, labels, T=1.0):
    confs = []
    correct = []
    for logits, y in zip(logits_list, labels):
        probs = softmax(logits, T=T)
        pred = max(range(len(probs)), key=lambda i: probs[i])
        confs.append(probs[pred])
        correct.append(1 if pred == y else 0)
    return confs, correct

# 玩具验证集：模型偏过度自信
logits_list = [
    [4.0, 1.0],   # correct, high confidence
    [3.8, 1.2],   # correct, high confidence
    [3.2, 2.9],   # wrong but still confident
    [2.7, 2.5],   # wrong but confident
    [1.8, 1.2],   # correct
    [2.2, 2.0],   # wrong
]
labels = [0, 0, 1, 1, 0, 1]

base_probs, base_correct = predict_confidence(logits_list, labels, T=1.0)
base_ece = ece_from_probs(base_probs, base_correct, n_bins=4)

best_T = None
best_ece = float("inf")
for i in range(5, 31):  # 0.5 ~ 3.0
    T = i / 10
    probs, corr = predict_confidence(logits_list, labels, T=T)
    score = ece_from_probs(probs, corr, n_bins=4)
    if score < best_ece:
        best_ece = score
        best_T = T

assert best_T is not None
assert best_ece <= base_ece
print("base_ece =", round(base_ece, 4))
print("best_T =", best_T, "best_ece =", round(best_ece, 4))
```

上面代码体现两个重点：

1. 温度缩放通常不改变排序，只改变概率形状。
2. 选 $T$ 要在验证集上做，不能在测试集上直接调。

如果换成语言模型 token 级别伪码，常见写法类似这样：

```python
# validation stage
best_T = 1.0
best_ece = inf

for T in candidate_temperatures:
    confs = []
    hits = []
    for logits, gold_token in validation_tokens:
        probs = softmax(logits / T)
        pred = argmax(probs)
        confs.append(max(probs))
        hits.append(int(pred == gold_token))
    ece = compute_ece(confs, hits)
    if ece < best_ece:
        best_ece = ece
        best_T = T
```

R-Tuning 的数据处理伪码则更像下面这样：

```python
processed = []

for sample in raw_qa_data:
    question = sample["question"]
    answer = sample["answer"]
    knowledge_label = sample["knowledge_label"]  # 1: certain, 0: uncertain

    if knowledge_label == 1:
        prompt = f"{question}\n[certain]"
        target = answer
    else:
        prompt = f"{question}\n[uncertain]"
        target = "I am unsure about this question."

    processed.append({"prompt": prompt, "target": target})

# training objective:
# maximize p(target | prompt)
```

如果写得再接近真实工程一点，流程通常是：

1. 用旧模型或规则筛出可能未知的问题，形成 $D_0$
2. 可回答样本形成 $D_1$
3. 对 $D_0$ 监督拒答模板
4. 对 $D_1$ 维持正常回答能力
5. 在线上把拒答概率和业务阈值联动

真实工程例子：一个医疗问答 Agent 接收“药物剂量 + 患者肝肾功能 + 并发症”复杂问题。若模型只做温度缩放，可能仍会给出一个听起来完整的剂量建议，只是置信度从 0.97 降到 0.83；而经过 R-Tuning 后，它更可能输出“当前信息不足，需要人工审核”，这才符合高风险系统的目标。

---

## 工程权衡与常见坑

只做全局温度缩放，成本最低，但能力也最有限。它适合“模型基本会，只是置信度偏高”的场景；不适合“模型经常在未知问题上胡答”的场景。

常见坑可以直接列成表：

| 风险或坑 | 现象 | 监控信号 | 对应策略 |
|---|---|---|---|
| 全局温度过粗 | 某些领域校准变好，另一些变差 | 分领域 ECE 波动大 | 用 ATS/分领域温度 |
| bin 太少 | ECE 看起来很稳定，但掩盖局部偏差 | reliability diagram 过于平滑 | 提高 bin 数并做样本量约束 |
| bin 太多 | ECE 方差很大，结论不稳 | 各 bin 样本过少 | 合并稀疏 bin 或做平滑 |
| RLHF 后更自信 | 回答更流畅但错误更坚定 | 高置信错误占比上升 | 重新做校准评估 |
| 只校准不拒答 | 低知识样本仍输出具体答案 | Unknown set 错答率高 | 引入 R-Tuning 或拒答头 |
| 置信度定义不一致 | 离线指标好，线上阈值失效 | 不同任务曲线不可比 | 固定 score 定义和评测协议 |

一个容易被忽略的点是：ECE 只是平均误差，不反映最危险区域。比如高风险系统更关心“置信度 0.9 以上但答错”的尾部错误，而不是全局均值。因此部署前除了 ECE，还应看 calibration curve、coverage-risk curve，以及高置信错误率。

再看一个真实工程场景。金融 Agent 负责读取财报、抽取指标并决定是否触发交易策略。此时可设置规则：

- 结构化抽取置信度 < 0.85：进入人工复核
- 事实性问答命中 uncertain 模式：直接拒答
- 高置信但与外部数据库冲突：禁止自动执行

这里阈值不是为了“让模型更聪明”，而是为了把错误限制在可控流程里。校准做得越差，阈值越不可信；阈值越不可信，自动化边界就越危险。

另一个坑是把“拒答率高”误当成“模型更好”。拒答太多会损伤覆盖率。覆盖率，指系统愿意回答的问题比例。工程目标通常不是最大化拒答，而是在给定风险预算下，找到准确率、校准、覆盖率的平衡点。

---

## 替代方案与适用边界

如果只需要一个低成本补丁，温度缩放通常是第一步。它不改模型参数，不改服务协议，适合上线前快速修正过度自信。

如果问题主要来自 RLHF 后的 token 级漂移，ATS/ETS 更合适。它本质上是“让温度随上下文变化”，比全局 $T$ 更细，但需要额外特征、验证数据和更复杂的实现。

如果业务重点是“遇到不知道的问题必须收住”，R-Tuning 更适合。它不是简单调分数，而是把不确定性训练成模型行为的一部分，特别适合医疗、法律、金融、企业知识库问答等高风险场景。

下面是一个对照表：

| 方法 | 输入要求 | 主要收益 | 适用场景 | 主要限制 |
|---|---|---|---|---|
| 温度缩放 | 验证集 logits 与标签 | 实现简单，快速降过度自信 | 分类、抽取、低风险问答 | 只做后处理，不会拒答 |
| ATS/ETS | token 特征、更多验证数据 | 处理上下文相关漂移 | RLHF 后模型、复杂生成 | 实现复杂，依赖特征质量 |
| R-Tuning | 已知/未知样本划分或近似标注 | 学会不知道时拒答 | 高风险问答、Agent | 需要构造不确定样本 |
| 外部校验器/verifier | 额外模型或规则系统 | 可独立估计真实性 | 检索增强、工具调用 | 成本更高，链路更长 |

可以用一个简单决策流理解：

1. 你只是想让概率别那么夸张？先用温度缩放。
2. 你发现不同领域、不同 token 位置偏差差异很大？考虑 ATS/ETS。
3. 你需要模型明确说“不知道”，且错误代价高？优先考虑 R-Tuning。
4. 你必须对事实正确性做强保证？再叠加检索、规则、外部 verifier。

创作类任务是一个典型边界。比如写营销文案，校准依然有意义，但“拒答能力”通常不是核心目标，因为任务本身允许发散表达。相反，在财务问答里，“不知道就别编”远比“语言更自然”重要。因此 ATS 调温度更像“修仪表盘”，R-Tuning 更像“加刹车系统”，二者解决的问题层级不同。

---

## 参考资料

1. Peter Tran. *Large Language Models: Calibration and ECE*，课程博客，2024。用于理解校准定义、ECE 公式与 reliability diagram。  
2. Zhang et al. *Calibrating Language Models with Adaptive Temperature Scaling*，EMNLP 2024。用于理解全局温度缩放的局限与 ATS 的 token 级自适应思路。  
3. Zhang et al. *R-Tuning: Instructing Large Language Models to Say “I Don’t Know”*，NAACL 2024。用于理解 $D_0/D_1$ 划分、uncertain/certain 提示以及拒答训练。  
4. CallSphere / Next Electronics 工程博客，2024。用于理解 Agent、人工审核阈值、线上 calibration monitoring 等工程实践。  

如果要继续查原文，最值得先看的关键词是 `R-Tuning: Instructing Large Language Models to Say "I Don't Know"`。重点关注其中的不确定样本构造方式、提示模板和实验集设置，因为这些细节会直接影响你在线上系统里能否复现拒答效果。
