## 核心结论

Few-shot 关系抽取，意思是“每种新关系只给极少几个标注句子，模型仍要学会判断实体对之间是什么关系”。它解决的是知识图谱和信息抽取里最常见的现实问题：新领域关系类型很多，但标注样本非常少。

可以先把它理解成“带着 5 个新标签的迷你面试”。系统面前有 5 种新关系，每种关系只展示 1 条示例句。之后再给一条新句子，要求系统立刻判断它更像哪一个标签。这里真正起作用的不是“背答案”，而是两类能力：

1. 元学习。元学习的白话解释是“让模型学会怎样快速学习”，不是只学某一个固定任务。
2. 提示学习。提示学习的白话解释是“把原任务改写成大模型更擅长的填空或生成任务”。

在 FewRel 这类标准基准上，这两条路线已经证明有效。尤其在 FewRel 2.0 的 `5-way-1-shot` 设定下，基于 prototype 的度量学习与 prompt 方法结合，基线精度可以稳定到 80% 左右，常见表述约为 80.3%，一些组合方法可达到约 80.5%。这说明两件事：

- 预训练语言模型内部已经存了相当多的关系知识。
- 即使每类只有 1 条样本，只要表示空间和匹配机制设计得对，模型仍能快速泛化。

下面这张表先给出整体判断。

| 方法 | 核心思想 | 优势 | 局限 | 适合场景 |
| --- | --- | --- | --- | --- |
| 元学习 | 在 episode 上训练快速适应能力 | 对新关系收敛快 | 易过拟合训练分布 | 关系集合相对稳定、可构造大量 episode |
| 原型网络 | 每类取一个 prototype 做距离分类 | 结构简单、推理快 | 单样本时 prototype 噪声大 | N-way K-shot 标准 few-shot 场景 |
| 提示学习 | 把关系抽取改写成填空任务 | 能借用 PLM 语言知识 | 模板敏感、词表受限 | 预训练模型强、关系有文字描述 |
| 自动模板选择 | 自动搜索 prompt 模板 | 减少人工试错 | 训练成本更高 | 模板影响显著的复杂语料 |

---

## 问题定义与边界

Few-shot 关系抽取通常写成 `N-way K-shot episode`。意思是：

- `N-way`：当前任务里有 $N$ 个候选关系类别。
- `K-shot`：每个关系只给 $K$ 个支持样本。
- 查询集：模型要在这 $N$ 个关系里，为新句子里的实体对选最匹配的一个。

一个 episode 可以画成下面这样：

`支持集（N 个关系，每类 K 句） -> 编码器 -> 类别表示 / prototype -> 查询句编码 -> 距离计算 -> softmax 分类`

“Few-shot”的关键不在于训练集总量一定很小，而在于**测试时遇到的新关系，每类只有极少标注样本**。模型必须靠训练阶段学到的“快速适配能力”，而不是靠每个关系的海量监督。

如果用公式表示原型网络中的类别中心，第 $i$ 个关系的 prototype 是：

$$
p_i = \frac{1}{K}\sum_{k=1}^{K} s_k^i
$$

其中 $s_k^i$ 是第 $i$ 类第 $k$ 个支持样本的向量表示。白话讲，就是“把这一类的几个例句编码后求平均，得到这个关系在向量空间里的中心点”。

这里要明确边界：

| 讨论范围 | 本文覆盖 | 本文不展开 |
| --- | --- | --- |
| 数据设定 | FewRel、FewRel 2.0 一类标准 few-shot RE | 长尾大规模监督关系抽取 |
| 任务形式 | 已知 episode 中有 N 个候选关系 | 开放关系发现、无标签聚类 |
| 样本来源 | 每类 1 到几条标注句 | 无授权数据抓取、隐私敏感实体处理 |
| 目标 | 快速适配新关系类型 | 完整知识图谱清洗与对齐流水线 |

一个玩具例子很直观。假设当前 episode 只有 3 个关系：

- `创始人`
- `位于`
- `毕业于`

支持集只给每类 1 句：

- “马斯克 创办了 SpaceX”
- “故宫 位于 北京”
- “图灵 毕业于 剑桥大学”

查询句是：“乔布斯 创办了 Apple”。即使模型没见过这个具体句子，只要编码后它与“创始人”那个 prototype 最近，就能预测正确。这就是 few-shot 关系抽取的最小闭环。

FewRel 2.0 中常见的 `5-way-1-shot` 指标之所以常被拿来做边界说明，是因为它足够苛刻：5 个候选关系，每类只有 1 条支持样本，仍要求模型做出稳定分类。在这样的条件下还能达到约 80.3% 的准确率，已经说明方法有效，但也说明它远没有“完全解决”跨域关系抽取。

---

## 核心机制与推导

### 1. ProtoNet：先做表示，再做距离匹配

ProtoNet，即原型网络，可以理解成“每类用一个中心点代表自己，查询样本看离谁最近”。这类方法的核心不是复杂分类头，而是表示空间是否把同类关系拉近、异类关系推远。

如果查询样本向量是 $q$，类别 prototype 是 $p_i$，常见分类概率写成：

$$
P(y=i\mid q)=\frac{\exp(-d(q,p_i))}{\sum_j \exp(-d(q,p_j))}
$$

这里的 $d(\cdot,\cdot)$ 可以是欧氏距离，也可以是负点积。白话讲，距离越小，softmax 后的概率越大。

### 2. 一个小数值例子

考虑 `5-way-1-shot`。每类只有 1 个支持句，因此 prototype 就等于那一个支持向量。

设查询向量到 5 个 prototype 的欧氏距离分别是：

- $d_1 = 0.20$
- $d_2 = 1.10$
- $d_3 = 1.25$
- $d_4 = 1.40$
- $d_5 = 1.55$

则未归一化分数为 $\exp(-d_i)$：

- $e^{-0.20} \approx 0.819$
- $e^{-1.10} \approx 0.333$
- $e^{-1.25} \approx 0.287$
- $e^{-1.40} \approx 0.247$
- $e^{-1.55} \approx 0.212$

总和约为 $1.898$，于是第一类概率约为：

$$
P(y=1\mid q)=\frac{0.819}{1.898}\approx 0.431
$$

这个例子说明一个常被忽视的事实：**“最近”不等于“高置信度”**。如果想得到 80% 以上置信度，正确类别与其他类别的距离差必须更大。比如把其他 4 个距离拉到 2.5 左右，则：

- $e^{-0.20}\approx 0.819$
- $e^{-2.50}\approx 0.082$

这时

$$
P(y=1\mid q)\approx \frac{0.819}{0.819+4\times0.082}\approx 0.714
$$

还不够高。若其他距离进一步拉到 3.5：

$$
P(y=1\mid q)\approx \frac{e^{-0.2}}{e^{-0.2}+4e^{-3.5}} \approx 0.82
$$

这就是 few-shot RE 的本质难点：不是只要找到“最近”就行，而是要把**类间间隔**学出来。

### 3. 元学习：让模型学会“怎么快速适应”

MAML 的白话解释是“先学一个好的初始化参数，见到新任务后只要走几步梯度就能适配”。在关系抽取里，它把训练过程改成大量 episode。每个 episode 都像一个小任务，模型反复练习“看几条支持样本后快速判断查询样本”。

Qu 等工作进一步引入贝叶斯元学习与关系图。贝叶斯元学习的白话解释是“不只学一个固定 prototype，而是承认 prototype 本身也有不确定性”。SGLD 是一种用噪声近似后验采样的方法，可以缓解单样本 prototype 太脆弱的问题。对少样本关系抽取而言，这很重要，因为 1 条支持句很可能带有上下文偏差。

### 4. Prompt-based RE：把分类改写成填空

提示学习的思路是，把关系抽取改写成预训练语言模型更熟悉的形式。比如原句：

- `[头实体] Jobs founded [尾实体] Apple`

可以改写成模板：

- `Jobs 和 Apple 的关系是 [MASK]`

然后把 `[MASK]` 预测为代表关系的词或标签描述。这样模型不再只依赖一个线性分类器，而是能调用预训练时积累的语言和事实知识。

如果再把关系描述也编码进去，例如把 `创始人` 扩展为“X founded Y”，就等于把标签文本本身也纳入表示空间。对初学者来说，可以把它理解成：**不只让模型看样例句，也让它看“这个标签大致是什么意思”**。

### 5. 一个真实工程例子

医疗知识图谱里常见关系如：

- `药物-治疗-疾病`
- `症状-提示-疾病`
- `检查-支持诊断-疾病`

问题在于某些专科关系只在医生语料中出现 1 到 2 次标注样本。此时如果直接训练普通分类器，几乎必然过拟合。工程上更常见的做法是：

- 用医学领域 PLM 编码句子和实体对。
- 用 prompt 模板把关系改写成 mask 预测。
- 同时保留 prototype 匹配头，做联合训练。

这样做的好处是：prompt 借用语言知识，prototype 保留少样本分类的结构约束，两者结合通常比单独使用更稳。

---

## 代码实现

下面给出一个“能运行的玩具版本”。它不依赖深度学习框架，只演示 episode 内 prototype 分类的核心逻辑。真实工程里，`encode` 会换成 BERT、RoBERTa 或领域 PLM 的实体对表示。

```python
import math

def mean_vec(vectors):
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]

def sq_euclidean(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def predict(query, support_by_class):
    prototypes = {label: mean_vec(vectors) for label, vectors in support_by_class.items()}
    labels = list(prototypes.keys())
    logits = [-sq_euclidean(query, prototypes[label]) for label in labels]
    probs = softmax(logits)
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return labels[best_idx], probs[best_idx], dict(zip(labels, probs))

support = {
    "founder_of": [[1.0, 1.0]],
    "located_in": [[4.0, 4.0]],
    "graduated_from": [[-3.0, 2.0]],
    "parent_org": [[2.5, -2.0]],
    "part_of": [[-2.0, -2.0]],
}

query = [1.1, 0.9]
label, conf, prob_map = predict(query, support)

assert label == "founder_of"
assert conf > 0.80
assert abs(sum(prob_map.values()) - 1.0) < 1e-9

print(label, round(conf, 4), prob_map)
```

上面代码对应的流程就是：

1. 采样一个 episode。
2. 对支持集按关系分组。
3. 每组求平均，得到 prototype。
4. 查询向量与各 prototype 计算距离。
5. 用 softmax 得到分类概率。

如果把它扩成真实训练循环，伪代码如下：

```python
for episode in episodes:
    support_text, query_text, query_label = episode

    # 1. 编码实体对上下文
    support_repr = encoder(support_text)   # [N, K, D]
    query_repr = encoder(query_text)       # [Q, D]

    # 2. 原型网络
    prototypes = support_repr.mean(dim=1)  # [N, D]

    # 3. prompt 分支：把实体对拼进模板
    prompt_inputs = build_prompt(query_text, relation_descs)
    mask_logits = plm(prompt_inputs)

    # 4. 度量分支
    metric_logits = -distance(query_repr, prototypes)

    # 5. 融合
    logits = metric_logits + alpha * project(mask_logits)

    # 6. loss 与更新
    loss = cross_entropy(logits, query_label)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

这里有两个实现点最关键：

| 模块 | 最低要求 | 更稳的做法 |
| --- | --- | --- |
| 编码器 | 取实体对上下文向量 | 加实体标记、取 span pooling |
| prototype | 简单平均 | 加 relation 描述融合 |
| prompt | 手写模板 | 自动模板搜索 + 标签描述 |
| 训练目标 | 单一交叉熵 | 度量 loss + prompt loss 联合训练 |

对初学者来说，最容易理解的一句话是：**ProtoNet 决定“谁最像谁”，Prompt 决定“语言模型觉得这个关系像什么”**。工程上把两者联动，通常比只做一边更可靠。

---

## 工程权衡与常见坑

Few-shot 关系抽取在论文里看起来很整齐，但工程里问题集中在“样本噪声、跨域偏移、模板脆弱”这三类。

先看风险与对应处理：

| 风险 | 现象 | 原因 | 常见缓解 |
| --- | --- | --- | --- |
| 元学习过拟合训练域 | FewRel 上好看，换领域掉很多 | 训练任务分布过窄 | relation graph、SGLD、跨域 episode |
| 单样本 prototype 不稳 | 1-shot 时预测波动大 | 支持句偶然性太强 | 描述融合、fusion loss、多视角编码 |
| prompt 模板敏感 | 实体顺序一变精度下降 | 模板绑定了表层语序 | 自动模板、实体类型提示 |
| 标签词表受限 | `[MASK]` 不容易输出目标词 | 关系标签不自然 | 用 relation description 替代单个词 |
| 类别间语义接近 | `located_in` 和 `part_of` 混淆 | 表示空间间隔不足 | 对比学习、hard negative 采样 |

真实工程例子可以继续用医疗知识图谱。假设要抽取“药物治疗疾病”关系，但某个罕见病只标了 2 条句子。常见问题是：

- 如果只靠 prototype，这 2 条句子里某个写法过于特殊，prototype 会偏。
- 如果只靠 prompt，模板例如“X 用于治疗 Y”对某些病历表达并不自然，模型会受句式影响。
- 如果实体顺序被写成“Y 可使用 X 治疗”，同一关系可能被模型误判。

所以更实用的做法不是押注单一方法，而是：

1. 用领域 PLM 编码句子，保留结构化度量能力。
2. 用自动模板和关系描述增强 prompt 分支。
3. 对单样本 prototype 加不确定性建模或描述融合。
4. 评估时按 episode 统计，而不是只看单条准确率。

一个很容易踩的坑是把 few-shot RE 当成普通句分类。它们不一样。普通句分类的类别在训练和测试时通常固定；few-shot RE 的关键是**测试类别是新关系**。如果训练过程没有 episode 化，或者没有模拟“见新类再分类”的过程，最终效果通常会失真。

---

## 替代方案与适用边界

Few-shot 关系抽取没有唯一正确路线。你可以把常见方案看成三个层级：

| 策略 | 适用场景 | 主要假设 |
| --- | --- | --- |
| Prototype + Meta-learning | 任务结构清晰、可稳定采样 episode | 同类样本在向量空间可聚成簇 |
| Prompt / Prompt-tuning | PLM 很强、关系可写成自然语言描述 | 语言模型已存储足够关系知识 |
| Prototype + Description Fusion | 单样本噪声明显、标签有定义文本 | 标签语义能补足样本不足 |
| Relation Graph Regularization | 关系之间有层次或图结构 | 类间相关性可显式建模 |

对新手来说，一个非常实用的判断标准是：

- 如果你相信“句向量距离”能区分类别，优先用 prototype 路线。
- 如果你相信“预训练模型知道很多事实表达方式”，优先用 prompt 路线。
- 如果两边都不完全可靠，就做融合。

“如果无法相信单个 sentence prototype，可以在 prototype 上叠加 relation 描述 embedding”这句话可以这样理解：除了把支持句编码成一个类别中心，还把标签名或关系说明文本也编码进去，再做加权融合。它相当于给 prototype 增加一份先验。因为单条支持句可能写得很偏，但“创始人”“位于”“治疗”这些标签描述本身也带语义信息。

适用边界也要说清楚：

- 当关系完全开放、候选类别事先未知时，few-shot 分类框架就不够了，需要开放集识别或关系发现。
- 当实体识别本身错误很多时，关系抽取上层再强也救不回来。
- 当领域语言与预训练语料差异极大时，prompt 方法的收益会明显下降。
- 当每类已经有足够多标注样本时，few-shot 技术不一定比标准监督学习更划算。

所以，Few-shot RE 不是“标 1 条就万事大吉”，而是“在极低标注预算下，尽量把预训练知识、度量结构和快速适配能力都利用起来”。

---

## 参考资料

1. Hang 等，*A Survey of Few-Shot Relation Extraction Combining Meta-Learning with Prompt Learning*。
最适合先读。它回答“这件事是什么、有哪些路线、各路线如何分类”。

2. Qu 等，*Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs*，PMLR 2020。
第二篇读。它回答“元学习具体怎么做，以及为什么 prototype 需要不确定性和关系图建模”。

3. Zhao 等及相关 Prompt-based RE 论文。
第三篇读。它回答“如何把关系抽取改成 prompt / mask prediction，以及模板为何会影响性能”。

4. FewRel / FewRel 2.0 相关实验论文与工程复现文章。
最后读。它回答“基准如何设定、5-way-1-shot 指标怎么比较、80% 左右精度意味着什么”。

5. 医疗知识图谱中的 few-shot prompt-tuning 与 prototype 融合工作。
适合工程落地时查阅。它回答“在低资源专业领域，怎样把上述方法拼成可用系统”。

对初学者可以这样记忆这几类材料的分工：

- Survey 解决“是什么”。
- 贝叶斯元学习论文解决“怎么做”。
- Prompt 与行业论文解决“哪里实践、会踩什么坑”。

建议阅读顺序：`Survey -> ProtoNet/Meta-learning 论文 -> Prompt-based RE -> 领域应用论文`。
