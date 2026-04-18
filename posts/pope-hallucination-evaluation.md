## 核心结论

POPE 是一种 `yes/no` 轮询式幻觉评测方法。轮询式的意思是：不让视觉语言模型（VLM，Vision-Language Model，即能同时处理图像和文本的模型）自由生成长描述，而是反复问它“图中是否存在某个对象”，再把答案统计成二分类指标。

它评测的不是“这段描述写得好不好”，而是“模型会不会把图里没有的对象说成有”。例如一张工地图里没有安全帽，问题是 `Is there a helmet in the image?`，模型仍回答 `yes`，这就暴露了对象幻觉。对象幻觉是指模型在图像中“看见”了实际不存在的物体。

POPE 的核心价值在于把开放式幻觉问题压缩成可统计的存在性判断。这样可以直接观察模型的过度肯定倾向：模型是不是倾向于对高频、常见、容易共现的对象回答 `yes`。

| 项目 | POPE 中的含义 |
|---|---|
| 方法类型 | `yes/no` 轮询式幻觉评测 |
| 输入 | 图像 + 对象存在性问题 |
| 输出 | 模型回答 `yes` 或 `no` |
| 真值 | MSCOCO 等数据集中的对象标注 |
| 关注指标 | Precision、Recall、F1、Accuracy、YesRatio |
| 核心风险 | 模型对不存在对象过度回答 `yes` |

一个直观指标是：

$$
YesRatio = \frac{\#pred\_yes}{N}
$$

其中 `#pred_yes` 是模型回答 `yes` 的次数，`N` 是总问题数。`YesRatio` 过高通常说明模型存在“什么都说有”的偏置。

---

## 问题定义与边界

设一张图像为 $x$，图像中的真实物体集合为 $G_x$，候选对象词表为 $O$。如果对象 $o \in G_x$，那么问题“图中是否有 $o$”就是正样本；如果对象 $o \notin G_x$，这个问题就是负样本。

玩具例子如下：图里有猫和椅子，没有杯子。此时：

$$
G_x = \{cat, chair\}
$$

问题 `Is there a cat in the image?` 的真值是 `Yes`；问题 `Is there a cup in the image?` 的真值是 `No`。这里重点不是模型会不会讲故事，而是它会不会把不存在的杯子说成存在。

POPE 的边界也很明确：它评测的是对象存在性判断，不评测完整描述质量、空间关系、属性细节、多轮推理或开放式问答能力。换句话说，它测的是“有没有”，不是“在哪里”“是什么颜色”“两者是什么关系”。

| 样本类型 | 构造来源 | 真值 | 用途 |
|---|---|---|---|
| 正样本 | 从当前图像真实对象 $G_x$ 中采样 | `Yes` | 检查模型能否识别真实存在对象 |
| random negatives | 从 $O \setminus G_x$ 中随机采样 | `No` | 构造最基础的不存在对象问题 |
| popular negatives | 从全数据集中高频对象里选当前图像没有的对象 | `No` | 检查模型是否偏向常见对象 |
| adversarial negatives | 选择与当前图像真实对象高共现、但图中不存在的对象 | `No` | 检查模型是否被语义共现误导 |

例如图里有 `person` 和 `skateboard`，但没有 `helmet`。`helmet` 可能不是随机负样本里最难的对象，但在真实世界里它常与运动场景共现，所以更适合作为 `adversarial negative`。模型如果看到滑板就回答有头盔，问题不在语言表达，而在视觉证据不足时仍做了肯定判断。

---

## 核心机制与推导

POPE 的三种负样本难度通常可以理解为：

$$
random < popular < adversarial
$$

`random negatives` 最容易，因为随机抽到的对象可能和图像语义完全无关。例如厨房图里问有没有飞机，模型容易答对。`popular negatives` 更难，因为高频对象更容易被模型先验影响。例如很多图里都有 `person`，模型可能倾向于说有。`adversarial negatives` 最难，因为它们和当前图像里的真实对象经常一起出现。例如图里有 `dining table`，但没有 `fork`，模型可能根据共现经验回答 `yes`。

原论文常用每张图 $l=6$ 个问题，正负各半，所以：

$$
k = \frac{l}{2} = 3
$$

也就是每张图采样 3 个正样本问题和 3 个负样本问题。评测时不能只看 Accuracy，因为一个模型可能通过“全答 yes”拿到较高召回，但同时产生大量误报。更完整的指标是：

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = \frac{2PR}{P + R}
$$

$$
Accuracy = \frac{TP + TN}{N}
$$

$$
YesRatio = \frac{\#pred\_yes}{N}
$$

这些符号的含义很直接：`TP` 是本来有、模型也答有；`FP` 是本来没有、模型却答有；`TN` 是本来没有、模型也答没有；`FN` 是本来有、模型却答没有。

最小玩具例子：4 道题，真值是 `[Yes, No, Yes, No]`，模型全答 `Yes`。此时：

| 项目 | 数值 |
|---|---:|
| TP | 2 |
| FP | 2 |
| TN | 0 |
| FN | 0 |
| Precision | 0.50 |
| Recall | 1.00 |
| F1 | 0.67 |
| YesRatio | 1.00 |

这个例子说明：模型把所有题都答成有，所以召回满分，但误报很高。`Precision` 和 `YesRatio` 会直接暴露这种过度肯定倾向。

三种负样本的构造流程可以写成：

| 模式 | 构造流程 |
|---|---|
| random | 对每张图，计算 $O \setminus G_x$，随机采样 $k$ 个对象 |
| popular | 先统计全数据集中对象频次，再从高频对象中取不属于 $G_x$ 的前 $k$ 个 |
| adversarial | 先统计对象共现关系，再选与 $G_x$ 高共现且不在当前图中的前 $k$ 个 |

伪代码如下：

```text
for each image x:
    positives = sample objects from G_x
    if mode == "random":
        negatives = random_sample(O - G_x, k)
    if mode == "popular":
        negatives = top_k(popular_objects not in G_x)
    if mode == "adversarial":
        negatives = top_k(co_occur_objects(G_x) not in G_x)

    questions = positives + negatives
    ask VLM yes/no questions
    compute confusion matrix
```

---

## 代码实现

代码层面，POPE 的实现可以拆成五步：读取标注、生成题目、调用模型、归一化答案、统计指标。归一化答案是指把模型输出统一映射成 `yes` 或 `no`，例如把 `Yes, there is.` 归一成 `yes`。如果允许模型自由输出长句，解析器本身会引入噪声。

一个简化流程是：

```python
for image in dataset:
    questions = build_pope_questions(image, mode="popular")
    preds = [normalize(model(q, image)) for q in questions]
    update_confusion_matrix(preds, labels)

metrics = compute_metrics(tp, fp, tn, fn)
```

下面是一段可运行的 Python 最小实现，演示如何计算 POPE 风格指标：

```python
def normalize_answer(text):
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    raise ValueError(f"Cannot normalize answer: {text}")


def compute_metrics(labels, preds):
    assert len(labels) == len(preds)

    tp = fp = tn = fn = 0
    for label, pred in zip(labels, preds):
        assert label in ("yes", "no")
        assert pred in ("yes", "no")

        if label == "yes" and pred == "yes":
            tp += 1
        elif label == "no" and pred == "yes":
            fp += 1
        elif label == "no" and pred == "no":
            tn += 1
        elif label == "yes" and pred == "no":
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels)
    yes_ratio = sum(p == "yes" for p in preds) / len(preds)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "yes_ratio": yes_ratio,
    }


labels = ["yes", "no", "yes", "no"]
raw_outputs = ["Yes, there is.", "yes", "YES", "Yes."]
preds = [normalize_answer(x) for x in raw_outputs]
metrics = compute_metrics(labels, preds)

assert metrics["tp"] == 2
assert metrics["fp"] == 2
assert metrics["tn"] == 0
assert metrics["fn"] == 0
assert metrics["precision"] == 0.5
assert metrics["recall"] == 1.0
assert round(metrics["f1"], 2) == 0.67
assert metrics["yes_ratio"] == 1.0
```

模块职责可以这样划分：

| 模块 | 职责 | 关键风险 |
|---|---|---|
| 数据读取 | 读取图像、对象标注、评测索引 | slug、image id、annotation id 对不齐 |
| 负样本采样 | 构造 random、popular、adversarial negatives | 负样本实际出现在图中但未标注 |
| 模型推理 | 把图像和问题送入 VLM | prompt 不稳定导致输出格式漂移 |
| 答案归一化 | 把自由文本转成 `yes/no` | `sure`、`not visible` 等回答难解析 |
| 指标统计 | 计算混淆矩阵和指标 | 只看单一指标会误判模型能力 |

真实工程例子：商品图质检系统要判断图片中是否有充电器、品牌 logo、说明书。上线前可以构造 POPE 风格题集：正样本来自人工标注，负样本分成随机、热门、对抗三类。模型如果对“手机包装盒”频繁回答“有充电器”，但图中实际没有充电器，这就是业务误报风险。此时重点应看 `popular` 和 `adversarial` 下的 `Precision` 与 `YesRatio`，而不是只看 `random` 下的 Accuracy。

---

## 工程权衡与常见坑

POPE 的优点是简单、可复现、指标清楚。它把复杂的生成式幻觉问题转成二分类问题，降低了评测成本。但这种简化也带来边界：它只能说明模型在对象存在性判断上是否稳健，不能代表模型所有多模态能力。

最常见的坑是只看 Accuracy。假设正负样本比例不是严格均衡，或者模型有强烈 `yes` 偏置，Accuracy 可能掩盖误报。`Precision` 直接衡量“模型说有的时候，有多少是真的有”；`YesRatio` 直接衡量“模型是不是太爱说有”。这两个指标必须和 Accuracy 一起看。

| 常见坑 | 后果 | 规避策略 |
|---|---|---|
| 只看 Accuracy | 掩盖“全答 yes”的偏置 | 同时看 Precision 和 YesRatio |
| 只用 random negatives | 题目太简单，高估模型能力 | 加入 popular 和 adversarial |
| 标注不完备 | 实际存在对象被当成负样本 | 抽查负样本，必要时重标注 |
| 自由文本输出 | 答案解析不稳定 | 强制模型输出闭集 `yes/no` |
| 同义词混乱 | `bike/bicycle` 被算成不同对象 | 建立统一词表和别名映射 |
| prompt 不一致 | 不同模型输出不可比 | 固定问题模板和解析规则 |

MSCOCO 标注不完备是一个重要问题。标注不完备的意思是：图里可能有某个对象，但数据集没有把它标出来。如果评测系统把这个对象当负样本，模型答 `yes` 反而会被算成错误。因此，严肃复现时需要人工抽查一部分负样本，尤其是 `popular` 和 `adversarial` 负样本。

另一个工程坑是同义词。`couch`、`sofa`，`bike`、`bicycle`，在自然语言里可能指同一类物体，但在数据集词表里可能不是同一个标签。没有统一词表时，评测会把词汇差异误当成模型错误。

商品图质检中的风险更直接。比如业务问“是否有充电器”“是否有安全帽”“是否有 logo”。如果模型看到手机盒就默认有充电器，看到工地图就默认有安全帽，看到衣服就默认有品牌 logo，那么它的输出会造成误报。对业务来说，这不是生成文本风格问题，而是自动审核系统会错误拦截或错误放行。

---

## 替代方案与适用边界

POPE 适合检测对象存在性幻觉，不适合评测所有视觉语言任务。它测的是“有没有”，不是“准不准、红不红、在哪儿”。如果你要测“图里是不是有红色自行车”，POPE 可以覆盖“是否有自行车”，但对“红色”这个属性不一定敏感。如果你要测“杯子在桌子的左边还是右边”，POPE 也不是合适工具。

| 方法 | 输入 | 输出 | 适用场景 | 不适合场景 |
|---|---|---|---|---|
| POPE | 图像 + 对象是否存在的问题 | `yes/no` | 对象存在性幻觉检测 | 属性、关系、定位、长描述 |
| VQA / Open-ended QA | 图像 + 自然语言问题 | 自由文本答案 | 开放问答能力评估 | 难以稳定统计幻觉来源 |
| caption-based evaluation | 图像 + 模型生成描述 | 一段 caption | 描述完整性、语言质量 | 对具体对象误报不够敏感 |
| segmentation / detection evaluation | 图像 + 像素级或框级标注 | mask、box、类别 | 定位、检测、分割质量 | 不评估自然语言回答习惯 |

如果业务只关心“有没有某个风险物”，POPE 风格评测很合适。例如工业巡检中问“是否有明火”“是否有安全帽”“是否有警示牌”。这些问题可以转成闭集 `yes/no`，并且误报、漏报都能用混淆矩阵表达。

如果业务关心局部定位，就需要检测或分割评测。比如不仅要知道有没有安全帽，还要知道安全帽是否戴在人的头上，这已经超出单纯对象存在性判断。此时需要 bounding box、mask 或关系标注。

如果业务关心长文本描述质量，例如自动生成医学影像报告、商品图描述、新闻图片说明，POPE 只能作为幻觉子测试，不能替代完整的 caption 或 QA 评测。合理做法是把 POPE 放在评测矩阵中，专门回答一个问题：模型是否会把不存在的对象说成存在。

---

## 参考资料

阅读顺序建议是：先看论文定义，再看官方仓库中的数据格式，最后看 `evaluate.py` 的指标实现。也就是先看规则，再看例子，最后看代码。

- 论文：[`Evaluating Object Hallucination in Large Vision-Language Models`](https://openreview.net/forum?id=xozJw0kZXF)
- 论文 PDF：[`openreview.net/pdf?id=xozJw0kZXF`](https://openreview.net/pdf?id=xozJw0kZXF)
- 官方代码与数据仓库：[`RUCAIBox/POPE`](https://github.com/RUCAIBox/POPE)
- 评测脚本：[`evaluate.py`](https://github.com/RUCAIBox/POPE/blob/main/evaluate.py)
