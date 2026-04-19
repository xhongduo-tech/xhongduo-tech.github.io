## 核心结论

文本数据增强是对已有文本样本做语义尽量不变的变换，生成新的训练样本，用来缓解标注数据不足、模型过拟合和长尾表达覆盖不足。

统一写法是：

$$
\tilde x = T(x), \quad D_{\text{aug}} = D \cup \{(\tilde x_i, y_i)\}
$$

其中，\(x\) 是原始文本，\(y\) 是标签，\(T\) 是增强算子，也就是“把一句话改写成另一句话的规则或模型”，\(\tilde x\) 是增强后的文本。

玩具例子：

原句：`这部电影很精彩`  
标签：正面

增强后可以是：

- `这部影片十分精彩`
- `这部电影精彩`
- `这部电影很出色`

这些句子的意思基本没变，所以仍然可以沿用“正面”标签。文本数据增强的目标不是凭空造新知识，而是让模型看到更多等价表达，减少它对某几个固定词形、句式的依赖。

一个简单流程是：

```text
原始样本 -> 语义保持变换 -> 增强样本 -> 合并训练集 -> 训练模型
```

真实工程例子是低资源中文情感分类。假设只有 3,000 条人工标注评论，直接训练分类器容易记住少数高频表达，比如“好看”“垃圾”“推荐”。通过同义词替换、回译和少量生成式改写，可以让模型见到“值得看”“体验差”“不太满意”等更多表达，从而提升召回率，尤其是长尾评论和口语化评论的识别能力。

---

## 问题定义与边界

文本数据增强不是任意改写，而是标签保持型改写。标签保持是指增强前后样本的任务标签不应改变。例如情感分类里，`很好看` 改成 `精彩` 通常仍然是正面；但如果改成 `很难看`，标签就从正面变成负面，这不是有效增强。

统一记号如下：

| 记号 | 含义 | 白话解释 |
|---|---|---|
| \(x\) | 原始文本 | 原来的句子 |
| \(y\) | 标签 | 这句话对应的答案 |
| \(T\) | 增强算子 | 改写规则或改写模型 |
| \(\tilde x\) | 增强文本 | 改写后的句子 |
| \(D_{\text{aug}}\) | 增强后数据集 | 原始数据加新增数据 |

不同任务的增强边界不同：

| 任务类型 | 是否适合直接增强 | 主要风险 |
|---|---|---|
| 文本分类 | 适合 | 标签漂移 |
| 命名实体识别 | 谨慎 | 实体被改坏 |
| 问答 | 谨慎 | 答案不再成立 |
| 机器翻译 | 适合做源语言侧增强 | 对齐失真 |

标签漂移是指改写后文本的真实标签变了，但训练数据里仍然沿用旧标签。它是文本增强最常见、也最危险的问题。

例如，情感分类中：

```text
原句：这家店服务很好
增强：这家店服务不错
```

这通常是安全的，因为情感仍然是正面。

但命名实体识别中：

```text
原句：我买了苹果手机
增强：我买了香蕉手机
```

这里 `苹果` 可能是品牌实体，`香蕉` 不是同类实体，原来的实体标注就失效了。

问答任务也类似。假设问题是“北京是哪个国家的首都？”，上下文里有“北京是中国的首都”。如果增强时把“北京”改成“上海”，答案就不再成立。

所以文本数据增强的边界可以概括为一句话：如果改写后答案、实体、情感或意图变了，这个样本就不应直接沿用原标签。

---

## 核心机制与推导

文本增强可以统一看成一个变换过程：

$$
D_{\text{aug}}=D\cup\{(\tilde x_i,y_i)\}
$$

核心假设是：

$$
f(x) = y \Rightarrow f(T(x)) \approx y
$$

其中 \(f\) 是理想任务函数，也就是“文本真正应该对应什么标签”的规则。这个公式的意思是：如果原句属于某个标签，那么经过合理增强后的句子也应该属于同一个标签。

常见方法有四类：同义词替换、回译、EDA 和预训练模型生成。

| 方法 | 核心思想 | 优点 | 风险 |
|---|---|---|---|
| 同义词替换 | 局部词替换 | 简单快速 | 领域词易出错 |
| 回译 | 语序与措辞改写 | 语言自然 | 成本较高 |
| EDA | 轻量扰动 | 实现简单 | 短文本易失真 |
| 预训练生成 | 模型自动改写 | 多样性强 | 质量波动大 |

同义词替换是把句子中的某些词替换为近义词。它可以写成：

$$
T_{\text{sr}}(x)
$$

如果原句中有词 \(w_j\)，从近义词集合 \(\mathcal N(w_j)\) 中选一个词 \(w'_j\)，得到新句子：

$$
w'_j \in \mathcal N(w_j)
$$

例如：

```text
这部电影很精彩 -> 这部电影很好看
这部电影很精彩 -> 这部电影很出色
```

这种方法简单，但对领域词很敏感。比如在金融文本中，“空头”“多头”“回撤”不能随便替换；在医疗文本中，“阳性”“阴性”更不能按普通近义词处理。

回译是先把文本翻译到另一种语言，再翻译回原语言：

$$
T_{\text{bt}}(x)=g^{-1}(g(x))
$$

其中 \(g\) 表示翻译到中间语言，\(g^{-1}\) 表示翻译回原语言。例如：

```text
中文：这部电影很精彩
英文：This movie is wonderful
中文：这部影片十分精彩
```

回译的优点是句子通常更自然，能改变语序和措辞；缺点是成本较高，而且翻译模型可能误解领域词。

EDA 是 Easy Data Augmentation 的缩写，意思是“简单数据增强”。它通常组合四种轻量操作：同义词替换、随机插入、随机交换、随机删除。可以写成：

$$
T_{\text{eda}}=T_{\text{rd}}\circ T_{\text{rs}}\circ T_{\text{ri}}\circ T_{\text{sr}}
$$

其中 \(T_{\text{sr}}\) 是同义词替换，\(T_{\text{ri}}\) 是随机插入，\(T_{\text{rs}}\) 是随机交换，\(T_{\text{rd}}\) 是随机删除。符号 \(\circ\) 表示把多个操作串起来。

例如：

```text
原句：这部电影很精彩
EDA 删除：这部电影精彩
EDA 替换：这部影片很精彩
```

EDA 适合入门和小数据场景，但短文本上风险更高。比如 `不推荐` 删除 `不` 以后变成 `推荐`，标签直接反转。

预训练模型生成是使用已经在大量语料上训练过的语言模型做补全或改写。预训练模型是指先在大规模文本上学习语言规律，再迁移到具体任务中的模型。掩码填充可以写成：

$$
\tilde x \sim p_\phi(\cdot \mid x_{\setminus m})
$$

其中 \(x_{\setminus m}\) 表示把部分词遮住后的句子，\(p_\phi\) 是模型给出的生成分布。例如：

```text
输入：这部[MASK]很精彩
输出：这部电影很精彩
输出：这部影片很精彩
```

生成式增强的多样性更强，但必须过滤。模型可能生成流畅但错误的句子，也可能生成和原句几乎一样的重复样本。

---

## 代码实现

下面是一个最小可运行 Python 示例。它不依赖外部模型，用词典模拟同义词替换，用规则模拟回译和 EDA，重点展示完整工程流程：读取数据、生成增强样本、过滤、去重、构建新训练集。

```python
from typing import List, Tuple

Sample = Tuple[str, str]

SYNONYMS = {
    "电影": ["影片"],
    "精彩": ["好看", "出色"],
    "很": ["十分"],
}

def load_dataset() -> List[Sample]:
    return [
        ("这部电影很精彩", "positive"),
        ("这个产品很差", "negative"),
    ]

def synonym_replace(text: str, ratio: float = 0.3) -> List[str]:
    results = []
    for word, candidates in SYNONYMS.items():
        if word in text:
            results.append(text.replace(word, candidates[0], 1))
    return results

def back_translate(text: str) -> List[str]:
    rules = {
        "这部电影很精彩": "这部影片十分精彩",
        "这个产品很差": "这个商品体验很差",
    }
    return [rules[text]] if text in rules else []

def eda_augment(text: str) -> List[str]:
    # 玩具实现：删除程度副词“很”，真实工程中需要分词和更严格规则。
    if "很" in text:
        return [text.replace("很", "", 1)]
    return []

def filter_by_similarity(original: str, augmented: str) -> bool:
    if not augmented or augmented == original:
        return False
    overlap = len(set(original) & set(augmented)) / max(len(set(original)), 1)
    return overlap >= 0.5

def deduplicate(samples: List[Sample]) -> List[Sample]:
    seen = set()
    unique = []
    for text, label in samples:
        key = (text, label)
        if key not in seen:
            seen.add(key)
            unique.append((text, label))
    return unique

def build_augmented_dataset() -> List[Sample]:
    dataset = load_dataset()
    augmented = list(dataset)

    for text, label in dataset:
        candidates = []
        candidates += synonym_replace(text)
        candidates += back_translate(text)
        candidates += eda_augment(text)

        for new_text in candidates:
            if filter_by_similarity(text, new_text):
                augmented.append((new_text, label))

    return deduplicate(augmented)

data = build_augmented_dataset()
texts = [x for x, _ in data]

assert ("这部影片十分精彩", "positive") in data
assert ("这部电影精彩", "positive") in data
assert len(data) == len(set(data))
assert all(label in {"positive", "negative"} for _, label in data)

print(data)
```

这段代码做的事可以概括为：先为每条样本生成多个改写版本，再过滤掉明显不靠谱或重复的句子，最后把保留下来的句子加入训练集。

过滤规则在真实工程中很重要：

| 过滤项 | 作用 |
|---|---|
| 语义相似度阈值 | 去掉偏离原意的样本 |
| 分类器一致性 | 去掉可能改标签的样本 |
| 近重复去重 | 防止训练集被重复样本淹没 |
| 实体保护 | 防止专有名词被误改 |

真实工程里，`filter_by_similarity()` 通常不会只看字符重合，而会使用句向量相似度、交叉编码器打分，或者让一个已训练分类器判断增强前后预测是否一致。对于命名实体、品牌名、疾病名、金额、日期等字段，还要先做实体保护，避免被增强算法误改。

---

## 工程权衡与常见坑

文本增强不是越多越好。更合理的经验公式是：

$$
\text{final data} = \text{original} + \lambda \cdot \text{high-quality augmented}
$$

其中 \(\lambda\) 是增强比例。它不宜过大，因为增强样本本质上仍然来自原始样本分布。如果增强样本数量远远超过原始样本，模型可能学到增强算法的偏差，而不是任务本身的规律。

| 常见坑 | 现象 | 规避方式 |
|---|---|---|
| 标签漂移 | 情感、实体、答案变了 | 一致性过滤、限制任务范围 |
| 低质量样本 | 句子别扭、不通顺 | 先打分再入库 |
| 分布失真 | 增强样本压过原始样本 | 控制增强比例 |
| 术语误改 | 专有名词被替换 | 保护实体和领域词 |
| 短文本破坏 | 删除一个词就失去关键信息 | 少用强扰动 |
| 重复样本过多 | 模型学到无效重复 | 近重复去重 |

低资源中文情感分类中，少量高质量回译通常比大量低质量随机扰动更有效。例如原始数据只有 3,000 条评论时，可以先做 1 倍到 2 倍增强，再用验证集观察效果。如果验证集召回提升但精确率下降，说明增强可能引入了噪声；如果训练集分数下降而验证集分数上升，说明增强起到了正则化作用。

真实工程中还要注意类别平衡。如果正样本做了大量增强，负样本没有增强，模型会被新的类别比例影响。更稳妥的方式是按类别分别控制增强比例，尤其是意图分类、投诉识别、风险文本识别这类类别不均衡任务。

另一个常见问题是泄漏。泄漏是指训练阶段看到了本不该看到的信息。如果先把全量数据增强，再切分训练集和测试集，那么同一个原始句子的改写版本可能同时出现在训练集和测试集中，评估结果会虚高。正确顺序是先划分训练集、验证集和测试集，只对训练集做增强。

---

## 替代方案与适用边界

文本数据增强只是解决数据不足的一种方法，不是唯一方法。很多时候，补标注、迁移学习、半监督学习和主动学习更合适。

| 方案 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|
| 数据增强 | 标签稳定、数据少 | 成本低、见效快 | 易引入噪声 |
| 补标注 | 预算足 | 质量高 | 成本高 |
| 迁移学习 | 有强预训练模型 | 收敛快 | 依赖源任务 |
| 半监督学习 | 有大量无标注数据 | 利用未标注语料 | 训练复杂 |
| 主动学习 | 标注预算有限 | 标注效率高 | 需要迭代流程 |

迁移学习是指把已经学到的通用语言能力迁移到当前任务中，例如用预训练语言模型微调文本分类器。半监督学习是指同时使用少量有标签数据和大量无标签数据。主动学习是指让模型挑选最值得人工标注的样本，从而提高标注效率。

如果是客服意图分类，文本增强通常很有价值。用户表达同一个意图的方式很多，例如“怎么退货”“我要退一下”“不想要了能不能退”，这些表达可以共享同一个意图标签。

如果是医疗问答、法律问答或信息抽取任务，增强要非常保守。把“可以服用”改成“不建议服用”，或者把合同金额、时间、主体改错，都会直接污染标签。此时更适合先补高质量标注，或者只做非常受控的模板级增强。

判断是否适合做文本增强，可以看三个条件：

| 判断条件 | 适合增强 | 不适合直接增强 |
|---|---|---|
| 标签是否容易保持 | 情感、主题、意图较稳定 | 答案、实体、数值容易变化 |
| 业务是否允许噪声 | 少量错误可接受 | 错误代价高 |
| 是否能做过滤 | 有相似度、规则或人工抽检 | 无法判断改写质量 |

对于中文情感分类，可以优先考虑回译和轻量同义词替换；对于高风险任务，先做严格过滤和人工抽检，再决定是否使用增强。新手应先理解“为什么增强”，再学习“怎么增强”。

---

## 参考资料

| 阅读顺序 | 目的 |
|---|---|
| 先看 EDA | 理解最基础的文本增强思路 |
| 再看回译 | 理解自然语言改写 |
| 再看生成式方法 | 理解更强的自动改写 |
| 最后看综述 | 建立整体认知 |

1. [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://huggingface.co/papers/1901.11196)
2. [Improving Neural Machine Translation Models with Monolingual Data](https://aclanthology.org/P16-1009/)
3. [Data augmentation using back-translation for context-aware neural machine translation](https://aclanthology.org/D19-6504/)
4. [Semantically Consistent Data Augmentation for Neural Machine Translation via Conditional Masked Language Model](https://aclanthology.org/2022.coling-1.457/)
5. [A Survey of Data Augmentation Approaches for NLP](https://aclanthology.org/2021.findings-acl.84/)
