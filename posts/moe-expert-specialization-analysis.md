## 核心结论

MoE 里的 FFN 专家不会在训练结束后自动“平均分工”，而是会形成可测量的功能分化。这里的功能分化，指的是不同专家稳定接收不同类型的 token，例如某些专家更常处理名词、数字、标点、专有名词，或者某一类上下文模式更强的 token。

直接原因不是“专家天生不同”，而是**稀疏路由**改变了训练样本的分配方式。稀疏路由指每个 token 不经过所有专家，而只被送到分数最高的少数几个专家。对 token 表示 $x$，路由器先为每个专家计算一个分数 $h_m(x)$，再归一化得到路由概率：

$$
\pi_m(x)=\frac{\exp(h_m(x))}{\sum_{m'=1}^{M}\exp(h_{m'}(x))}
$$

若采用 top-$k$ 路由，则真正执行前向和接收梯度更新的，只有概率最高的 $k$ 个专家。于是训练动力学会变成：

$$
\text{相似 token}
\rightarrow
\text{相似路由}
\rightarrow
\text{相似专家反复更新}
\rightarrow
\text{该类专家更擅长该类 token}
\rightarrow
\text{路由进一步集中}
$$

这就是功能分化出现的根本机制。它本质上是一种由路由训练诱导出来的**隐式聚类**，而不是人工预先写死的“语法规则表”。

对初学者，最容易理解的表述是：路由器不是先知道“谁负责名词、谁负责数字”，而是在训练中不断试错。某一类 token 反复被送到同一小组专家后，这组专家就会越来越擅长处理它们，最终形成稳定偏好。

已有测量结果表明，这种分化通常明显高于“完全均匀分配”的基线。也就是说，专家不是随机接 token，而是在训练后形成了统计上稳定的接收偏好。一个简化对比如下：

| 模型 | Spec (%) | 均匀期望 U (%) | $\Delta U$ (%) |
| --- | ---: | ---: | ---: |
| Mixtral-8x7B-v0.1 | 50.21 | 25.0 | +25.21 |
| Phi-3.5-MoE-instruct | 48.49 | 12.5 | +35.99 |

这类结果的含义很直接：如果 top-$k$ 专家对某类 token 的覆盖率长期远高于均匀期望，那么“专家已经分化”就是一个可以统计验证的事实，而不是主观解释。

---

## 问题定义与边界

这篇文章讨论的是 **MoE 层中 FFN 专家的功能分化**，不讨论注意力头分工，也不讨论整个模型的全部可解释性。更具体地说，问题是：

1. 某类 token 最终更常被哪些专家接收。
2. 这种集中是否稳定、高于随机基线。
3. 这种集中是否具有语言学或结构性含义。

这里的“某类 token”不只可以是词性 POS，也可以是数字、标点、专有名词、代码片段、语言标签、子词长度区间，或者更细的上下文模式。工程上，最常见做法是先按一个标签体系切片，再统计每类 token 的专家覆盖分布。

一个常用指标是 specialization score。它的含义是：某一类 token 中，有多大比例被少数最常接收它的专家覆盖。设第 $l$ 层中某个类别 $c$ 的总 token 数为 $T_{c,\text{all}}^{(l)}$，该层 top-$k^\*$ 最常接收该类别的专家合计覆盖数为 $T_{c,\text{top-}k^\*}^{(l)}$，则定义：

$$
\mathrm{Spec}_{c,l}
=
\frac{T_{c,\text{top-}k^\*}^{(l)}}{T_{c,\text{all}}^{(l)}}
$$

跨层平均后可得到整体分化强度：

$$
\mathrm{Spec}_{c}
=
\frac{1}{L}\sum_{l=1}^{L}\mathrm{Spec}_{c,l}
$$

这个指标越高，说明越少数的专家覆盖了越多该类 token，分化越强。

为了避免把“高分化”误读成“任何集中都算合理”，还需要给出均匀基线。若某层有 $E$ 个专家，分析时只取 top-$k^\*$ 专家作为覆盖集合，那么完全均匀分配下的期望覆盖率是：

$$
U=\frac{k^\*}{E}
$$

因此更有信息量的不是单独看 $\mathrm{Spec}$，而是看：

$$
\Delta U = \mathrm{Spec}_{c,l} - \frac{k^\*}{E}
$$

当 $\Delta U$ 持续显著大于 0 时，才说明分化强于随机均分。

一个玩具例子足够说明这个指标。假设某层有 8 个专家、100 个名词 token，top-2 专家分别处理了 60 个和 20 个，则：

$$
\mathrm{Spec}_{\text{noun},l}
=
\frac{60+20}{100}
=
0.8
$$

而均匀期望是：

$$
U=\frac{2}{8}=0.25
$$

于是：

$$
\Delta U = 0.8-0.25=0.55
$$

这不是小幅波动，而是明显的集中。

为了让指标含义更清楚，可以把不同状态放在一张表里：

| 场景 | 8 个专家中 top-2 覆盖率 | 解释 |
| --- | ---: | --- |
| 完全均匀 | 25% | 没有明显分化 |
| 轻度分化 | 40% | 有一定偏好，但不强 |
| 明显分化 | 60% | 少数专家已覆盖大部分该类 token |
| 强分化 | 80% | 该类 token 基本被固定小群体专家处理 |

边界也需要说明清楚。

第一，分化不等于“每个专家只做一件事”。一个专家可以同时偏好名词、实体、日期和部分数字模式。

第二，分化不等于性能一定更高。适度分化能提升容量利用率，但过强分化可能引出负载失衡、容量溢出、路由塌缩和泛化变差。

第三，分化不等于“专家真的理解了语言学规则”。很多时候专家偏好的是**上下文模式**，不是单独词类本身。例如某专家可能偏好的不是“数字”，而是“数字 + 单位 + 表格局部结构”。

第四，编码器与解码器的现象不完全一致。编码器输入分布更丰富、上下文观察更充分，往往更容易看到清晰的专家偏好；解码器因目标 token 更短、更受自回归条件约束，分化通常弱一些。

---

## 核心机制与推导

专家分化不是因为系统预设了“语法学专家”“数字专家”，而是因为稀疏路由改变了梯度流向。机制可以拆成四步。

### 1. 路由器读取 token 的上下文表示

这里的 token 表示不是词表 ID，而是当前层看到上下文后的隐藏向量 $x\in\mathbb{R}^d$。路由器通常用一个线性映射得到每个专家的打分：

$$
h(x)=W_r x + b,\quad h(x)\in\mathbb{R}^{M}
$$

其中 $M$ 是专家数，$h_m(x)$ 是专家 $m$ 对 token $x$ 的路由分数。

再经过 softmax 得到概率：

$$
\pi_m(x)=\frac{\exp(h_m(x))}{\sum_{m'=1}^{M}\exp(h_{m'}(x))}
$$

这一步的作用只是把打分转换成可比较的概率分布。真正决定稀疏性的，是后续 top-$k$ 选择。

### 2. 只有 top-$k$ 专家真正执行

设路由集合为：

$$
\mathcal{R}(x)=\operatorname{TopK}\big(\{\pi_1(x),\dots,\pi_M(x)\},k\big)
$$

只有 $m\in\mathcal{R}(x)$ 的专家会处理该 token。于是 MoE 层输出常写成：

$$
y(x)=\sum_{m\in \mathcal{R}(x)} \alpha_m(x)\,E_m(x)
$$

其中 $E_m(\cdot)$ 是第 $m$ 个 FFN 专家，$\alpha_m(x)$ 是归一化后的门控权重。

这一点很关键。若一个 token 没有进入专家 $m$ 的 top-$k$，则该专家几乎不接收这个 token 的梯度。对未被选中的专家参数 $\theta_m$，可以近似理解为：

$$
\frac{\partial \mathcal{L}}{\partial \theta_m}\approx 0,
\qquad m\notin \mathcal{R}(x)
$$

这就是“路由决定谁能学习”的含义。

### 3. 相似输入导致相似更新

设某类 token 集合为 $C$。如果它们在路由器空间中的表示相近，则对任意 $x,x'\in C$，有较大概率满足：

$$
h(x)\approx h(x')
\Rightarrow
\operatorname{rank}(h(x))\approx \operatorname{rank}(h(x'))
\Rightarrow
\mathcal{R}(x)\approx \mathcal{R}(x')
$$

也就是说，相似 token 会反复进入相似专家。于是这些专家的参数更新，逐步变成对集合 $C$ 的条件优化。若把属于 $C$ 的 token 损失写成：

$$
\mathcal{L}_C = \sum_{x\in C}\ell(x)
$$

那么被高频选中的专家 $m$ 会更频繁地执行类似于：

$$
\theta_m
\leftarrow
\theta_m - \eta \sum_{x\in C\cap \mathcal{B}_m}\frac{\partial \ell(x)}{\partial \theta_m}
$$

其中 $\mathcal{B}_m$ 表示当前 batch 中实际路由到专家 $m$ 的 token 集合。若 $C$ 的占比在 $\mathcal{B}_m$ 中持续偏高，则该专家就会越来越偏向处理集合 $C$。

### 4. 正反馈闭环形成专家分化

把上面的过程连起来，就是：

$$
\text{相似表示}
\rightarrow
\text{相似 top-}k
\rightarrow
\text{相似专家反复更新}
\rightarrow
\text{该专家对这类输入更优}
\rightarrow
\text{相似表示更容易继续路由到它}
$$

这不是一次性的偏差，而是训练动力学里的闭环。闭环一旦形成，分化就会逐步稳定。

### 一个可计算的玩具例子

假设只有四类 token：标点、名词、动词、数字；总共有 4 个专家，每个 token 只选 2 个专家。训练初期，各专家能力接近；训练若干轮后，路由统计可能演化成：

| token 类别 | 更常进入的专家 | 直观原因 |
| --- | --- | --- |
| 标点 | 0 | 局部模式强、上下文变化小 |
| 名词 | 1, 2 | 与实体、修饰关系相关，分布更复杂 |
| 动词 | 1, 2 | 和时态、论元结构、句法位置相关 |
| 数字 | 3 | 结构模式稳定，如日期、金额、编号 |

这张表不是规则表，而是训练结果的统计摘要。若你在验证集上记录路由日志，再按标签汇总，就能观察到类似偏好。

### “专家路径”为什么也有信息

当 token 穿过多层 MoE 时，每层都会经过一组专家，这组专家序列构成它的跨层路径。设第 $l$ 层路由到的专家集合为 $\mathcal{R}_l(x)$，则一个 token 的路径可写成：

$$
P(x)=\big(\mathcal{R}_1(x),\mathcal{R}_2(x),\dots,\mathcal{R}_L(x)\big)
$$

如果不同词性或不同结构 token 的路径分布不同，那么路径本身就包含了可分离信息。工程上常见的验证方法是：把路径编码成特征，再训练一个小分类器预测 POS、是否数字、是否实体。如果路径可用于预测这些标签，说明路由行为已经携带了稳定结构信号。

这里的关键结论不是“专家显式学会了语法学”，而是“路由过程已经对某些语言学类别产生了稳定可分离的分配模式”。

---

## 代码实现

下面给出一个可直接运行的 Python 最小例子，复现两件事：

1. 对 token 执行 top-$k$ 路由。
2. 按 token 类别统计专家覆盖，并计算 specialization score 与均匀基线差值。

代码只依赖 Python 标准库。

```python
import math
from collections import Counter, defaultdict

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def topk_indices(scores, k):
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in ranked[:k]]

def route_tokens(tokens, k):
    routed = []
    for token in tokens:
        probs = softmax(token["scores"])
        experts = topk_indices(probs, k)
        routed.append({
            "text": token["text"],
            "label": token["label"],
            "scores": token["scores"],
            "probs": probs,
            "experts": experts,
        })
    return routed

def specialization_score(routed_tokens, label, top_experts, num_experts):
    hits = []
    unique_token_count = 0

    for token in routed_tokens:
        if token["label"] == label:
            unique_token_count += 1
            hits.extend(token["experts"])

    if unique_token_count == 0:
        raise ValueError(f"label={label!r} has no samples")

    counter = Counter(hits)
    covered = sum(count for _, count in counter.most_common(top_experts))

    # 注意：这里的分母用“该类 token 的总路由次数”，因为每个 token 会进入 k 个专家
    total_assignments = len(hits)
    spec = covered / total_assignments
    uniform = top_experts / num_experts
    delta_u = spec - uniform
    return {
        "label": label,
        "token_count": unique_token_count,
        "assignment_count": total_assignments,
        "expert_counter": counter,
        "spec": spec,
        "uniform": uniform,
        "delta_u": delta_u,
    }

def print_report(result):
    print(f"label={result['label']}")
    print(f"  token_count={result['token_count']}")
    print(f"  assignment_count={result['assignment_count']}")
    print(f"  expert_counter={dict(result['expert_counter'])}")
    print(f"  spec={result['spec']:.3f}")
    print(f"  uniform={result['uniform']:.3f}")
    print(f"  delta_u={result['delta_u']:.3f}")

def update_biases(biases, desired_load, observed_load, step=0.05):
    if not (len(biases) == len(desired_load) == len(observed_load)):
        raise ValueError("all inputs must have the same length")

    new_biases = []
    for b, d, o in zip(biases, desired_load, observed_load):
        error = d - o
        if error > 0:
            new_biases.append(b + step)
        elif error < 0:
            new_biases.append(b - step)
        else:
            new_biases.append(b)
    return new_biases

if __name__ == "__main__":
    # 4 个专家，top-2 路由
    tokens = [
        {"text": "cat", "label": "noun",   "scores": [3.2, 2.7, 0.2, 0.1]},
        {"text": "dog", "label": "noun",   "scores": [3.1, 2.5, 0.2, 0.2]},
        {"text": "city", "label": "noun",  "scores": [2.9, 2.6, 0.4, 0.1]},
        {"text": "runs", "label": "verb",  "scores": [0.4, 3.0, 2.4, 0.2]},
        {"text": "walks", "label": "verb", "scores": [0.3, 2.8, 2.2, 0.3]},
        {"text": "42", "label": "number",  "scores": [0.1, 0.2, 0.5, 3.4]},
        {"text": "100", "label": "number", "scores": [0.0, 0.2, 0.4, 3.2]},
        {"text": "2025", "label": "number","scores": [0.1, 0.1, 0.6, 3.3]},
        {"text": "!", "label": "punct",    "scores": [2.9, 0.3, 0.1, 0.0]},
        {"text": ",", "label": "punct",    "scores": [2.8, 0.2, 0.2, 0.1]},
    ]

    num_experts = 4
    k = 2
    top_experts = 2

    routed = route_tokens(tokens, k=k)

    by_label = defaultdict(list)
    for token in routed:
        by_label[token["label"]].append(token)

    for label in ["noun", "verb", "number", "punct"]:
        result = specialization_score(
            routed_tokens=routed,
            label=label,
            top_experts=top_experts,
            num_experts=num_experts,
        )
        print_report(result)

    noun_result = specialization_score(routed, "noun", top_experts=2, num_experts=4)
    number_result = specialization_score(routed, "number", top_experts=2, num_experts=4)

    assert noun_result["spec"] >= 0.80
    assert number_result["spec"] >= 0.80
    assert noun_result["expert_counter"][0] >= 3
    assert number_result["expert_counter"][3] >= 3

    # 一个极简的负载平衡 bias 更新例子
    biases = [0.0, 0.0, 0.0, 0.0]
    desired = [0.25, 0.25, 0.25, 0.25]
    observed = [0.55, 0.20, 0.10, 0.15]

    new_biases = update_biases(biases, desired, observed, step=0.05)
    print("old_biases =", biases)
    print("new_biases =", new_biases)

    assert new_biases[0] < 0.0
    assert new_biases[2] > 0.0
```

这段代码里有两个容易混淆的统计口径，需要明确。

第一，`token_count` 是某类 token 的样本数，例如名词有 3 个。

第二，`assignment_count` 是这些 token 被分配到专家的总次数。由于这里每个 token 都进入 top-2 专家，因此：

$$
T_{c,\text{assign}} = k \cdot T_{c,\text{all}}
$$

于是代码里 `spec` 的分母是该类 token 的总路由次数，而不是原始 token 数。这样定义的好处是它直接反映“某类 token 的路由分配有多集中”。

如果想严格贴近前文按“token 覆盖”定义的写法，也可以把统计改成“一个 token 只要命中 top-$k^\*$ 专家集合中的任一专家，就算被覆盖一次”。那样分母是 token 数，分子是被覆盖的 token 数。两种口径都能用，但文章里必须先固定定义，否则不同实现的数值不可直接比较。

### 为什么这个例子是可运行且有用的

它没有训练真实大模型，但复现了真实分析流程中的三个关键动作：

| 动作 | 玩具代码中的实现 | 真实工程中的对应物 |
| --- | --- | --- |
| 计算路由分数 | `scores` 字段 | 路由网络输出 logits |
| 执行 top-$k$ 选择 | `topk_indices` | 实际 MoE 路由 |
| 统计专家分布 | `specialization_score` | 离线分析脚本 |

真实系统里，`scores` 来自验证集推理日志，`label` 来自 POS 标注器、规则标签器或领域标注器，统计逻辑基本一致。

### 一个常见的工程扩展：动态 bias 调节

如果观察到少数专家过热，可以在路由打分前加入动态 bias：

$$
\tilde{h}_i(x)=h_i(x)+b_i
$$

其中 $b_i$ 会随负载变化而更新。一个最简化的更新形式是：

$$
b_i \leftarrow b_i + u\,\mathrm{sign}(e_i),
\qquad
e_i = \rho_i^{\ast}-\rho_i
$$

其中：

- $\rho_i^{\ast}$ 是期望负载。
- $\rho_i$ 是观测负载。
- $u$ 是步长。

如果某个专家太忙，$e_i<0$，则其 bias 下调；若某个专家太闲，$e_i>0$，则其 bias 上调。这样做不直接修改主任务损失，而是在路由前对专家分数做轻量校正。

### 一次完整离线分析通常包含四步

| 步骤 | 做什么 | 产出 |
| --- | --- | --- |
| 1 | 在验证集记录每层 token 的 top-$k$ 专家 | 路由日志 |
| 2 | 给 token 打标签，如 POS、数字、实体、语言、子词长度 | 分析切片 |
| 3 | 统计专家接收分布、Spec、$\Delta U$、负载熵 | 分化报告 |
| 4 | 对热点专家检查容量溢出、丢 token、延迟抖动 | 调参依据 |

如果是在线推理系统，这类分析还有一个直接价值：新版本上线后，一旦发现多语言输入的专家分布突然异常集中，就可以快速判断是“模型学到了更强分化”，还是“路由器开始塌缩”。

---

## 工程权衡与常见坑

第一个常见误区，是把“分化强”直接等同于“模型更好”。这不成立。功能分化和负载平衡是两个相关但不同的问题。适度分化能提高容量利用率，但如果少数专家长期过热，系统会更脆弱。

可以把两者区别成下面这张表：

| 现象 | 本质 | 可能后果 |
| --- | --- | --- |
| 功能分化 | 不同专家更擅长不同 token 模式 | 表达能力提升、解释性更强 |
| 负载失衡 | 少数专家接收了过多 token | 吞吐下降、容量溢出、训练不稳 |
| 路由塌缩 | 大量 token 挤到极少数专家 | 专家“死掉”、有效容量下降 |

第二个常见坑，是把“高频命中某类 token”误判为“单一语义专家”。真实情况通常更复杂。一个专家可能经常处理数字，但真正决定路由的不是数字本身，而是“数字 + 单位 + 日期格式 + 局部结构”这一组合模式。

因此做解释时，最好至少同时看以下四类信息：

| 维度 | 为什么要看 |
| --- | --- |
| token 文本 | 知道专家到底接了什么 |
| 结构标签 | 如 POS、数字、实体、代码片段 |
| 前后文窗口 | 区分“同一个词在不同语境”的路由差异 |
| 子词长度 / 语言标签 | 排查是否只是分词或多语言效应 |

第三个常见坑，是只看一层、一个 batch 或几条样例就下结论。专家分化是统计现象，不是个别样本现象。至少要满足三个条件，结论才比较可靠：

1. 在验证集而不是训练噪声很大的小 batch 上统计。
2. 跨多个层、多个 batch 看趋势是否稳定。
3. 对比均匀基线，而不是只看绝对覆盖率。

第四个常见坑，是把编码器与解码器混在一起讨论。编码器侧更容易出现“标点专家”“数字专家”“实体专家”，原因是输入更长、上下文结构更完整。解码器侧因目标 token 更受历史生成约束，很多层的分化图像不会同样清晰。

第五个常见坑，是忽略容量限制。若每个专家有固定 capacity，超过容量的 token 可能被丢弃、回退或重路由。此时你观察到的“专家偏好”可能同时混入了容量截断效应。也就是说：

$$
\text{观测到的专家分布}
\neq
\text{纯粹的路由偏好}
$$

更准确地说，实际观测分布往往是：

$$
\text{路由偏好}
+
\text{容量约束}
+
\text{负载平衡机制}
+
\text{实现细节}
$$

所以工程上做路由解释时，最好同时记录：

| 指标 | 作用 |
| --- | --- |
| 每专家 token 数 | 看是否过热 |
| 每专家容量溢出率 | 看是否有截断 |
| 平均门控概率 | 看路由器原始偏好 |
| 实际命中率 | 看偏好是否被容量机制改写 |

---

## 替代方案与适用边界

如果目标是增强或控制专家分化，路线不止一种。大体可以分成三类。

### 1. 显式分化约束

这类方法直接鼓励不同专家学到更不相似的表示或路由模式，例如正交约束、方差约束、去相关约束。它们的目标是让专家之间的功能边界更清楚。

优点是目标明确，适合研究“专家是否真的学到不同功能”。缺点是会引入额外训练目标，权重不合适时可能伤害主任务。

### 2. 负载控制方法

这类方法不直接要求专家“学得不同”，而是优先保证路由不要过度集中。典型做法包括 auxiliary load balancing loss、动态 bias、Loss-Free Balancing、ALF-LB 等。

它们的目标更偏工程：减少热点专家、避免塌缩、提升吞吐稳定性。优点是部署价值直接；缺点是它们主要解决“别太挤”，不保证一定得到更强的语义分化。

### 3. 路由机制替换

还有一类方法直接改变路由结构，例如 token-choice、expert-choice、soft routing、hash routing 或其他近似分配机制。这类方法既影响负载，也影响专家能否形成稳定偏好。

为了快速比较，可以放在一张表里：

| 方案 | 主要作用 | 优点 | 风险 / 边界 |
| --- | --- | --- | --- |
| 正交 / 方差 / 去相关约束 | 强化专家差异 | 可直接增强可解释性 | 额外损失可能扰动主任务 |
| Auxiliary load balancing loss | 平衡专家负载 | 实现简单，训练中常见 | 权重难调，可能与主损失冲突 |
| 动态 bias / Loss-Free Balancing / ALF-LB | 抑制热点专家 | 对主损失干扰较小 | 主要保负载，不直接保语义分工 |
| 改变路由机制 | 同时改变分化与负载形态 | 有机会从机制层面优化 | 实现复杂，系统代价更高 |

适用边界也应明确。

如果你研究的是“专家是否学到不同功能”，优先看分化指标、路径可预测性和显式分化约束。

如果你做的是大规模训练或在线服务，首要任务通常不是让专家标签看起来更漂亮，而是避免热点专家拖垮吞吐。

如果模型已经明显塌缩，先处理负载平衡，再讨论专家语义。否则你看到的“分化”可能只是路由偏置、容量限制或实现细节制造出来的假象。

更直接地说：

| 场景 | 先看什么 |
| --- | --- |
| 研究可解释性 | Spec、$\Delta U$、路径分类准确率 |
| 训练稳定性 | 负载方差、辅助 loss、动态 bias |
| 线上部署 | 热点专家、容量溢出、延迟与吞吐 |
| 多语言 / 多领域迁移 | 专家偏好是否变成语言或领域隔离 |

因此，“专家分化”不是单独拿出来看的美观统计，而应该和负载平衡、容量管理、系统开销一起看。只有这样，结论才既有研究价值，也有工程意义。

---

## 参考资料

| 资料 | 核心贡献 | 建议阅读方式 | 对应本文章节 |
| --- | --- | --- | --- |
| [Part-Of-Speech Sensitivity of Routers in Mixture of Experts Models (COLING 2025 PDF)](https://aclanthology.org/anthology-files/pdf/coling/2025.coling-main.431.pdf) | 直接研究路由器对词性类别的敏感性，适合看“专家分化是否可测量” | 先看实验设定和 specialization score 定义，再看不同模型与编码器/解码器差异 | 核心结论、问题定义与边界 |
| [Mixture-of-Experts (MoE) Router](https://www.emergentmind.com/topics/mixture-of-experts-moe-router) | 汇总 MoE 路由的常见形式、top-$k$、负载平衡与训练现象 | 用来快速回顾 MoE 路由术语和公式，再回到具体论文 | 核心机制与推导、工程权衡 |
| [Distributed Mixture-of-Experts and Expert Parallelism](https://brunomaga.github.io/Mixture-of-Experts) | 从系统角度解释专家并行、路由、容量和分布式代价 | 阅读时重点看 token 分发、capacity 和通信成本如何影响工程行为 | 工程权衡与常见坑 |
| [Loss-Free Load Balancing for MoE](https://www.emergentmind.com/papers/2408.15664) | 说明如何通过动态 bias 等方法改善负载，而不过多干扰主任务损失 | 先看方法目标，再看为什么它主要解决“平衡”而非“功能分化” | 代码实现、替代方案与适用边界 |
| [Auxiliary-Loss-Free Load Balancing (ALF-LB)](https://www.emergentmind.com/topics/auxiliary-loss-free-load-balancing-alf-lb) | 总结不依赖传统辅助 loss 的负载平衡思路 | 适合与传统 auxiliary loss 对照阅读，理解部署场景下的取舍 | 替代方案与适用边界 |
| [Advancing Expert Specialization for Better MoE](https://www.emergentmind.com/articles/2505.22323) | 关注如何显式提升专家分化及其收益 | 阅读时重点看“分化增强”与“性能提升”是否同时成立，以及实验边界 | 替代方案与适用边界 |
| [Decoding the Mixture of Experts in Language Processing](https://scisimple.com/en/articles/2025-02-02-decoding-the-mixture-of-experts-in-language-processing--a9nmm7n) | 适合作为入门综述，帮助建立 MoE、路由与专家功能分工的整体图景 | 先读概念，再回到论文级资料核对公式与实验细节 | 核心结论、问题定义与边界 |
| Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* | 稀疏 MoE 的经典起点，奠定 top-$k$ 路由和负载平衡讨论框架 | 重点看稀疏门控层、辅助负载项和训练稳定性问题 | 核心机制与推导、工程权衡 |
| Fedus et al., *Switch Transformers* | 展示极简 top-1 路由在大规模训练中的效果与代价 | 重点看单专家路由为什么更省算，以及为什么更需要负载平衡 | 工程权衡与常见坑、替代方案 |
