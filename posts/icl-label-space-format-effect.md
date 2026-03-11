## 核心结论

ICL，in-context learning，白话说就是“模型不改参数，只看你在提示里给的几个例子，就临时学会怎么答”。在这个过程中，示例最重要的作用，通常不是把“正确标签”逐条教给模型，而是告诉模型三件事：这是什么任务、输入长什么样、输出应该落在哪个标签空间里。

Min 等人在 2022 年的结果可以概括成一句话：**示例标签是否真实对应，影响往往没有想象中大；标签词本身、输入分布、整体格式，影响更大。** 这说明很多 ICL 成功案例，本质更接近“格式驱动的任务唤起”，而不是“靠几条样本完成小样本监督学习”。

更进一步，如果把情感分类里的 `positive/negative` 改成 `foo/bar`，性能通常会明显下降。原因不是模型突然不会分类了，而是它失去了标签词自带的语义先验。`positive/negative` 在预训练语料里本来就和情绪判断高度相关；`foo/bar` 只是符号，没有现成语义。此时模型只能依赖上下文中的输入-标签对应关系，临时学一层映射。

可以把这个现象写成一个简化分解：

$$
P(y \mid x, D)=\beta_{TR} P_{TR}(y \mid x, D)+\beta_{TL} P_{TL}(y \mid x, D)
$$

其中：

- $TR$ 是 Task Recognition，任务识别，白话说就是“模型先看懂你在做什么任务”
- $TL$ 是 Task Learning，任务学习，白话说就是“模型在当前上下文里临时学会输入到标签的映射”
- $\beta_{TR}$ 和 $\beta_{TL}$ 表示两条通路当前各占多大权重

当标签是 `positive/negative` 这类自然语言标签时，$\beta_{TR}$ 往往较大；当标签换成 `foo/bar` 这类无语义符号时，$\beta_{TR}$ 会明显变小，模型更多依赖 $\beta_{TL}$。

---

## 问题定义与边界

先把问题形式化。给定一个输入 $x$、一个标签集合 $\mathcal{Y}$、以及若干个示例组成的上下文 $D$：

$$
D=\{(x_1,y_1),(x_2,y_2),...,(x_k,y_k)\}
$$

模型要做的是：

$$
\hat{y}=\arg\max_{y \in \mathcal{Y}} P(y \mid x, D)
$$

这里有三个经常被混淆、但必须分开看的变量。

| 变量 | 定义 | 白话解释 | 对 ICL 的影响 |
|---|---|---|---|
| 标签空间 | 标签词本身是什么 | 输出槽位里允许填哪些词 | 直接影响模型能否调用预训练语义 |
| 输入分布 | 示例输入和真实输入是否同分布 | 例子像不像线上数据 | 影响模型能否把示例模式迁移到新输入 |
| 格式 | 输入、标签、分隔符、顺序如何组织 | 提示长什么样 | 影响模型是否把当前问题识别成某类任务 |

一个最小模板可以写成：

```text
文本: [示例输入1]
标签: [示例标签1]

文本: [示例输入2]
标签: [示例标签2]

文本: [待预测输入]
标签:
```

边界也要说清楚。

第一，这个结论不是“标签永远不重要”。更准确地说，是**标签真假不如标签空间和格式重要**。如果你把每条示例标签都打乱，但标签词仍然是 `positive/negative`，模型可能还能借助任务识别通道答得不错；但如果你同时把标签词换成无语义符号，又不给足够清晰的映射，效果就会显著下滑。

第二，这个结论更适合解释“少样本提示”场景，不是说参数训练没用。只要进入 fine-tuning、instruction tuning 或 symbol tuning，模型就能把原本没有语义的符号标签学成稳定映射。

第三，模型规模会影响结论强弱。大模型更容易从格式中恢复任务意图，也更容易在上下文里临时建立映射；小模型则更依赖现成的标签语义。一旦把 `positive/negative` 换成 `foo/bar`，小模型更可能直接掉到接近随机猜测。

玩具例子可以直接看二分类情感任务。

| 示例设置 | 模型看到的东西 | 主要依赖 |
|---|---|---|
| `I love it -> positive` | 标签词有语义 | 任务识别 + 少量映射 |
| `I love it -> foo` | 标签词无语义，但有映射 | 更多依赖上下文映射 |
| `I love it -> foo` 且示例极少、格式混乱 | 既无语义，格式也弱 | 容易失败 |

真实工程例子则是客服工单分类。线上请求可能是“退款不到账”“账户被锁”“物流一直没更新”。如果示例也都是类似短文本、且标签使用 `billing / account / shipping` 这类自然语言词，ICL 往往很稳定；如果改成 `A / B / C`，但没有解释每个符号代表什么，稳定性通常会下降。

---

## 核心机制与推导

可以把 ICL 过程拆成两步：先识别任务，再在当前上下文中补足映射。

第一步是任务识别。模型看到如下内容：

```text
Review: The movie was wonderful.
Sentiment: positive

Review: The plot was boring.
Sentiment: negative

Review: The acting is excellent.
Sentiment:
```

即使它不认真“检查”前两个例子的标签是否真对，它也很容易识别出：这里大概率是在做情感分类，因为有 `Review`、有 `Sentiment`、有 `positive/negative`。这就是 $P_{TR}$ 的来源。它本质上利用的是预训练中已经存在的统计关联。

第二步是任务学习。如果标签换成：

```text
Review: The movie was wonderful.
Sentiment: foo

Review: The plot was boring.
Sentiment: bar

Review: The acting is excellent.
Sentiment:
```

这时模型已经不能再直接利用 `foo/bar` 的语义，因为这两个词本身不表示“正面/负面”。它必须从示例里现学一个局部映射：`wonderful -> foo`，`boring -> bar`，然后把 `excellent` 和 `wonderful` 对齐，于是输出 `foo`。这更接近上下文内的模式匹配。

因此上面的分解式可以进一步直观理解为：

$$
P(y \mid x, D)
=
\underbrace{\beta_{TR} P_{TR}(y \mid x, D)}_{\text{借用预训练语义和格式}}
+
\underbrace{\beta_{TL} P_{TL}(y \mid x, D)}_{\text{依赖当前示例临时建映射}}
$$

如果标签有语义，两个通道都可用；如果标签无语义，则常见情况是：

$$
\beta_{TR} \to 0
$$

此时模型成败主要取决于 $P_{TL}$ 是否足够强。

Yu 和 Ananiadou 在 2024 年用更机制化的角度解释了这件事。他们指出，一部分 attention heads 会专门承担“从上下文样例里找相似输入，再把对应标签信息带回来”的作用。attention，白话说就是“当前 token 去看历史 token，给不同位置分配不同关注权重”。如果把它写成标准形式，就是：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

在这个框架里：

- $Q$ 和 $K$ 决定“当前输入和历史哪个位置更像”
- $V$ 携带“那个位置上到底存了什么信息”
- 对 ICL 来说，历史标签 token 的表征常常会被当作可复制的信息源

可以把它画成一个极简伪图：

```text
[示例输入1] ----相似度匹配----> [待预测输入]
      |                            |
      v                            |
 [标签 foo] <----value 携带信息----- 
```

也就是说，某些头先用 query-key 找到“这个新输入最像哪个示例输入”，再通过 value-output 把对应示例标签的表示带回当前预测位置。这样即使标签词是 `foo/bar`，模型仍然可能完成分类。

这解释了一个看似反直觉的现象：**ICL 不一定真的“理解了监督目标”，它可能只是成功建立了一个局部检索加复制机制。** 当格式稳定、输入分布接近、标签空间可识别时，这套机制就很强；当这三者被破坏，性能就会快速恶化。

---

## 代码实现

先看一个最小可运行的玩具实现。它不是真正的 Transformer，只是用词表重叠模拟“新输入和哪个示例更像，就继承哪个标签”。这个例子足够说明为什么 `foo/bar` 在有示例映射时仍然能工作。

```python
from collections import Counter

def tokenize(text):
    return [w.strip(".,!?").lower() for w in text.split() if w.strip()]

def jaccard(a, b):
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def icl_predict(demos, query):
    scored = [(jaccard(x, query), y) for x, y in demos]
    scored.sort(reverse=True)
    return scored[0][1]

semantic_demos = [
    ("This movie is wonderful and touching", "positive"),
    ("The plot is boring and slow", "negative"),
]

symbolic_demos = [
    ("This movie is wonderful and touching", "foo"),
    ("The plot is boring and slow", "bar"),
]

query1 = "A wonderful and touching film"
query2 = "A boring and slow story"

assert icl_predict(semantic_demos, query1) == "positive"
assert icl_predict(semantic_demos, query2) == "negative"
assert icl_predict(symbolic_demos, query1) == "foo"
assert icl_predict(symbolic_demos, query2) == "bar"
```

这个玩具例子说明两点。

第一，只要示例映射存在，符号标签也能工作。第二，符号标签能工作，不代表模型有了“情感”这个抽象概念；它可能只是把输入相似度和标签复制做对了。

真实工程里，更常见的是在推理前构造演示样本，并控制标签词。比如客服文本分类，先从历史工单里挑 few-shot 示例，再拼成 prompt。若决定使用符号标签，则最好配合 symbol tuning。symbol tuning，白话说就是“专门把自然语言标签替换成符号后继续训练，让模型学会在符号空间里做映射”。

下面是一个简化训练循环，重点不在深度学习细节，而在“标签词是可控变量”。

```python
def format_demonstrations(batch_texts, batch_labels, label_tokens):
    label_map = {0: label_tokens[0], 1: label_tokens[1]}
    prompts = []
    for text, label in zip(batch_texts, batch_labels):
        prompts.append(f"Text: {text}\nLabel: {label_map[label]}")
    return prompts

# 伪代码：symbol tuning 的核心是替换标签空间，而不是改任务本身
for batch in dataloader:
    prompts = format_demonstrations(
        batch_texts=batch.texts,
        batch_labels=batch.labels,
        label_tokens=["foo", "bar"],
    )
    logits = model(prompts)
    loss = cross_entropy(logits, batch.labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

如果你做的是推理系统，而不是训练系统，至少要把 prompt 构造层单独抽象出来。因为工程上最常改的不是模型，而是以下三项：

| 可控项 | 常见选择 | 工程目的 |
|---|---|---|
| 标签词 | `positive/negative` 或 `foo/bar` | 控制是否借助语义先验 |
| 示例输入 | 短评、客服问句、工单摘要 | 对齐线上输入分布 |
| 格式模板 | `Text/Label`、JSON、问答式 | 控制任务识别信号强度 |

一个真实工程例子是多语言客服路由。假设你要把工单路由到 `billing`、`account`、`shipping` 三个队列。若系统主要依赖 ICL，建议直接用这些自然语言标签，并确保 few-shot 示例覆盖线上常见表达，比如“退款失败”“登录异常”“物流延迟”。如果业务方要求输出匿名码，例如 `T1/T2/T3`，那就最好在上线前做一次 symbol tuning，或者在 prompt 中显式写出 `T1=billing, T2=account, T3=shipping`，不要直接裸用符号。

---

## 工程权衡与常见坑

工程决策的核心，不是“语义标签好还是符号标签好”，而是“你希望性能来自哪条通路”。

| 标签类型 | 是否有天然语义 | 是否需要额外训练 | ICL 稳定性 | 适用场景 |
|---|---|---|---|---|
| 自然语言标签 | 有 | 通常不需要 | 高 | 快速上线、零训练 |
| 符号标签 | 没有 | 往往需要 | 中到低 | 有脱敏、协议化输出要求 |
| 随机标签 | 几乎没有 | 强烈建议需要 | 低 | 实验分析，不适合直接上线 |

常见坑有五个。

第一，**把标签真假和标签空间混为一谈**。很多人看到“标签正确与否影响不大”，就误以为随便写都行。错。真正没那么敏感的是“示例中某条标签是否真对”；真正很敏感的是“你让模型输出的词本身是什么”。

第二，**示例输入分布和线上输入脱节**。比如示例全是影评长句，线上却是客服短句“退款不到账”。即使你还用 `positive/negative`，模型也可能把任务识别错，因为示例根本没告诉它真实输入长什么样。

第三，**格式不稳定**。前两个示例是 `Text:` / `Label:`，最后一个却写成 `Input:` / `Answer:`，模型看到的结构信号被打断。ICL 对这些局部格式变化往往比传统分类器更敏感。

第四，**小模型直接使用无语义标签**。在小模型上，`foo/bar` 很容易让 $\beta_{TR}$ 几乎消失，而 $\beta_{TL}$ 又不足以独立撑住性能，于是结果接近随机。大模型能靠上下文机制补回来一部分，小模型经常不行。

第五，**忽视关键 attention heads 的脆弱性**。已有研究表明，在符号标签场景里，只干预很少一部分承担 in-context 映射的 heads，准确率就可能显著下降。这说明系统虽然看起来“会做任务”，但它可能依赖一条很窄的机制通道。上线后若模板、顺序、示例长度轻微变化，都可能触发退化。

多语言客服是一个典型真实例子。假设你的底座模型主要在英文语料上预训练，但你要处理中文客服请求。此时若标签仍使用 `positive/negative` 或 `billing/account/shipping` 这类英语常见标签，模型往往还能借助标签语义和模板完成分类；但如果改成 `foo/bar/baz`，又没有训练过对应映射，效果常会明显变差。原因不是中文不行，而是你主动切断了最稳定的任务识别通道。

---

## 替代方案与适用边界

如果业务允许，自然语言标签仍然是默认优先级最高的方案。原因很简单：它训练成本最低，同时能最大化利用预训练先验。

如果业务必须使用符号标签，有两条主路。

| 方案 | 思路 | 成本 | 适用边界 |
|---|---|---|---|
| 明确 instruction + 自然语言标签 | 直接把任务写明白 | 低 | 大模型、快速验证 |
| 符号标签 + symbol tuning | 让模型学会符号映射 | 中到高 | 标签必须匿名化、协议化输出 |

第一种是 instruction 强化。instruction，白话说就是“用明确文字把任务规则直接讲给模型”。例如：

```text
任务：判断句子情绪，输出 positive 或 negative。

示例:
Text: 这个产品非常好用
Label: positive

Text: 服务太差了
Label: negative

Text: 物流速度很慢
Label:
```

这种写法的优点是，任务识别信号很强，哪怕示例少，也能让大模型更稳地走 $TR$ 通路。

第二种是 symbol tuning。比如：

```text
任务：判断句子情绪。
标签定义：foo 表示正面，bar 表示负面。

示例:
Text: 这个产品非常好用
Label: foo

Text: 服务太差了
Label: bar

Text: 物流速度很慢
Label:
```

如果只是临时 prompt，一次性解释 `foo/bar` 的含义，有时能缓解问题，但未必稳定。真正稳妥的做法，是用大量这类符号化样本继续训练，让模型把 `foo=positive`、`bar=negative` 学成参数内知识。这样 $\beta_{TL}$ 会更强，不再完全依赖即时上下文。

适用边界也要明确。

第一，若你要的是“最低训练成本 + 最快上线”，选自然语言标签。第二，若你要的是“输出受控、不能泄露业务语义”，选符号标签，但接受额外训练成本。第三，若你面对的是小模型、低资源、few-shot 数量很少的场景，不建议直接裸用随机符号标签，因为这基本是在主动移除模型最可靠的先验。

最终可以把实践建议压缩成一句话：**ICL 更像格式和标签空间驱动的推理启动器，而不是几条例子就能完成的稳定监督学习器。** 只要理解这一点，很多 prompt 设计问题都会变得可解释。

---

## 参考资料

- Sewon Min, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer. *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP 2022. https://aclanthology.org/2022.emnlp-main.759/
- Zeping Yu, Sophia Ananiadou. *How do Large Language Models Learn In-Context? A Study of Their Uniquely Human-Like Generalization Capability*. EMNLP 2024. https://aclanthology.org/2024.emnlp-main.192/
- Jerry Wei et al. *Symbol tuning improves in-context learning in language models*. arXiv 2023. 可参考摘要索引：https://www.scixplorer.org/abs/2023arXiv230508298W/abstract
- Min 等论文 PDF: https://aclanthology.org/2022.emnlp-main.759.pdf
- Yu & Ananiadou 论文 PDF: https://aclanthology.org/2024.emnlp-main.192.pdf
