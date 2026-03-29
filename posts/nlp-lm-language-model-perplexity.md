## 核心结论

困惑度（Perplexity，简称 PPL）是语言模型评估里最常见的指标之一。它本质上是“平均负对数似然”的指数化结果，用来衡量模型预测下一个 token 时有多不确定。公式是：

$$
\mathrm{PPL}=\exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i\mid x_{<i})\right)
$$

这里的“负对数似然”可以白话理解成：模型每次押注下一个 token 时，押得越准，惩罚越小；押得越散，惩罚越大。

直觉上，PPL 可以看成模型每一步平均像是在多少个等概率选项里做选择。比如 PPL=2，近似表示模型平均只在 2 个同等可能的候选里犹豫；PPL=50，说明它的不确定性大得多。

但 PPL 不是一个“脱离上下文的绝对分数”。它依赖三个条件：

| 条件 | 为什么会影响 PPL |
| --- | --- |
| 分词器 | 同一句话切成的 token 数不同，平均损失会变 |
| 评测语料 | 新闻、代码、医学文本的难度不同 |
| 评测流程 | 截断方式、padding、mask、是否泄漏测试集都会改结果 |

因此，PPL 只能在“同分词器、同语料、同评测协议”下直接比较。跨词表比较时，更稳妥的做法是换成 BPC（bits per character，每字符比特数）。它和交叉熵的关系是：

$$
\mathrm{BPC}=\frac{\mathrm{CrossEntropy}}{\ln 2}, \qquad \mathrm{PPL}=2^{\mathrm{BPC}}
$$

在常见引用的 PTB 评测设置里，GPT-2 Small（117M）PPL 大约是 65.9，GPT-3 175B 可降到约 20。业务含义不是“20 比 65.9 小 3 倍这么简单”，而是更大模型在绝大多数位置上能给正确 token 更集中的概率质量，表现为更稳定的下一个词预测能力。

| 模型 | 典型 PPL | 约对应 BPC |
| --- | ---: | ---: |
| GPT-2 Small（117M） | 65.9 | $\log_2 65.9 \approx 6.04$ |
| GPT-3（175B） | 20 | $\log_2 20 \approx 4.32$ |

---

## 问题定义与边界

语言模型的任务是给序列 $x_1,x_2,\dots,x_N$ 逐步分配概率，也就是预测：

$$
P(x_i\mid x_{<i})
$$

“自回归”这个词的白话解释是：模型只能看前文，再预测下一个 token，不能偷看后文。

于是，一个序列的平均负对数似然可写成：

$$
H = -\frac{1}{N}\sum_{i=1}^{N}\log P(x_i\mid x_{<i})
$$

这个 $H$ 常被称为交叉熵。PPL 就是它的指数化：

$$
\mathrm{PPL} = e^H
$$

为什么要做指数？因为交叉熵在对数空间里，不够直观。指数化后，数值能被解释成“等概率候选数”。

玩具例子可以把它想成“猜下一个词的骰子游戏”：

- 如果模型看到“今天天气很”，它主要在“好”“热”两个词里选，那不确定性低。
- 如果模型看到一段乱码，下一个 token 几乎什么都可能，那它像是在掷一个很多面的骰子。

边界也很明确。

第一，PPL 只适用于能输出条件概率的语言模型。生成式分类器、检索系统、排序模型不一定天然有这个指标。

第二，PPL 依赖 token 化方式。“分词器”可以白话理解成：把文本切成模型内部处理单元的规则。英文里，有的模型按词片切，有的按字符切；中文里，有的把“语言模型”拆成 2 个 token，有的拆成 4 个。切法不同，$N$ 就不同，平均损失自然也不同。

第三，PPL 依赖数据域。“域”就是文本所属的场景分布。一个在新闻语料上训练的模型，去测医学论文，PPL 往往明显变高。因为它不是不会语言，而是不熟悉这个领域的 token 分布。

真实工程例子：如果一个客服机器人主要服务电商售后，你拿通用新闻测试集测出来 PPL=18，不代表它在售后对话里同样优秀。反过来，一个在售后数据上 PPL=12 的模型，去做法律合同摘要，可能马上升到几十甚至更高。这不是评测坏了，而是目标域变了。

---

## 核心机制与推导

先看最小推导链：

$$
\text{CrossEntropy} = -\frac{1}{N}\sum_{i=1}^{N}\log P(x_i\mid x_{<i})
$$

$$
\mathrm{PPL} = \exp(\text{CrossEntropy})
$$

如果换成以 2 为底的对数，就能写成：

$$
\mathrm{BPC} = -\frac{1}{N}\sum_{i=1}^{N}\log_2 P(x_i\mid x_{<i})
$$

于是：

$$
\mathrm{PPL} = 2^{\mathrm{BPC}}
$$

这里的 BPC 可以白话理解成：平均生成一个字符需要多少 bit 信息量。它比 PPL 更适合跨 tokenizer 比较，因为“字符”或“字节”是更稳定的归一化单位。

看一个玩具例子。假设测试序列只有两个 token，模型给真实 token 的概率分别是 0.5 和 0.25，那么：

$$
H=-\frac{1}{2}(\ln 0.5+\ln 0.25)
$$

因为 $\ln 0.5\approx -0.693$，$\ln 0.25\approx -1.386$，所以：

$$
H\approx -\frac{1}{2}(-0.693-1.386)=1.04
$$

于是：

$$
\mathrm{PPL}=e^{1.04}\approx 2.83
$$

这表示模型平均等价于在 2.83 个等概率选项中做选择。

如果把同一个值换成 BPC：

$$
\mathrm{BPC}=\frac{1.04}{\ln 2}\approx 1.5
$$

两种视角都在描述同一件事：

- 用 PPL 看，是“平均候选数”
- 用 BPC 看，是“平均信息量”

为什么 tokenizer 会破坏可比性？因为 PPL 的平均单位是 token，不是字符。假设模型 A 把一句话切成 10 个 token，模型 B 切成 20 个 token。即使两者对原始文本掌握程度接近，平均到每个 token 的损失也可能完全不同。此时直接拿 PPL 对比，会把“切法差异”误当成“建模能力差异”。

所以工程上常见的推导路径是：

1. 先按统一协议算 token-level cross-entropy。
2. 在同 tokenizer 场景下报告 PPL。
3. 在不同 tokenizer 场景下转成 BPC 或 byte-level 指标再比较。

---

## 代码实现

下面给一个可运行的 Python 例子，演示如何从“真实 token 概率”计算交叉熵、PPL 和 BPC。它没有依赖深度学习框架，但逻辑和实际评估脚本一致。

```python
import math

def calc_metrics(probs, mask=None):
    """
    probs: 每个位置上真实 token 的概率，例如 [0.5, 0.25, 0.8]
    mask:  1 表示有效位置，0 表示忽略（例如 padding）
    """
    if mask is None:
        mask = [1] * len(probs)

    assert len(probs) == len(mask)
    valid = [p for p, m in zip(probs, mask) if m == 1]

    assert len(valid) > 0
    assert all(0.0 < p <= 1.0 for p in valid)

    cross_entropy = -sum(math.log(p) for p in valid) / len(valid)
    ppl = math.exp(cross_entropy)
    bpc = cross_entropy / math.log(2)

    return cross_entropy, ppl, bpc


# 玩具例子：两个 token 的真实概率分别为 0.5 和 0.25
ce, ppl, bpc = calc_metrics([0.5, 0.25])
assert round(ce, 2) == 1.04
assert round(ppl, 2) == 2.83
assert round(bpc, 2) == 1.50

# 带 padding 的例子：最后一个位置被 mask 掉，不参与评估
ce2, ppl2, bpc2 = calc_metrics([0.8, 0.5, 0.01], mask=[1, 1, 0])
assert ppl2 < 2.0  # 因为只看前两个较高概率位置

print("cross_entropy =", ce)
print("ppl =", ppl)
print("bpc =", bpc)
```

真实评估脚本里，通常不是直接拿概率列表，而是从 logits 取 `log_softmax` 后，抽出真实标签对应的对数概率，再做平均。伪代码如下：

```python
# logits: [batch, seq_len, vocab_size]
# labels: [batch, seq_len]
# attention_mask: [batch, seq_len]

log_probs = log_softmax(logits, dim=-1)
token_log_probs = gather(log_probs, index=labels)   # 取真实 token 的对数概率
nll = -token_log_probs                              # negative log-likelihood

valid_nll = nll * attention_mask                    # 屏蔽 padding
avg_ce = valid_nll.sum() / attention_mask.sum()
ppl = exp(avg_ce)
bpc = avg_ce / ln(2)
```

这里有两个实现细节不能漏。

第一，padding 必须 mask。padding 是补齐长度的占位符，不是真实文本。如果把它算进去，损失会被污染。

第二，固定长度模型评估长文本时，不能简单分块后独立计算。因为每个块的开头都缺上下文，PPL 会被人为抬高。更合理的做法是滑动窗口，让每个位置尽量看到足够长的前文。

真实工程例子：在一个在线写作助手项目里，团队想比较两个中文模型是否值得上线。做法不是直接跑“整体 PPL”，而是：

- 固定同一套 tokenizer
- 在真实用户草稿语料上切出验证集
- 统一滑窗长度与 stride
- 统计 PPL、BPC、生成延迟、GPU 成本

最后发现模型 A 的 PPL 只比模型 B 好 4%，但延迟高 35%，显存占用高 60%。如果业务目标是交互补全，A 未必值得上线。这个例子说明：PPL 是重要指标，但不是唯一指标。

---

## 工程权衡与常见坑

PPL 常被误用，不是因为公式难，而是因为评测协议很容易失真。

| 常见坑 | 影响 | 规避方法 |
| --- | --- | --- |
| 分词器不同 | PPL 不可直接比较 | 统一 tokenizer，或转成 BPC/byte-level 指标 |
| 域漂移 | 离线分数和线上体验脱节 | 用目标域验证集，分场景分别报告 |
| 数据泄漏 | PPL 虚低，产生错误乐观判断 | 做去重、近重复检测、反向基准检测 |
| padding 未屏蔽 | 损失被无效位置污染 | 使用 attention mask |
| 长文本截断粗暴 | 块首位置缺上下文，PPL 偏高 | 用滑窗评估 |
| 只看平均值 | 掩盖长尾失败样本 | 分桶统计，不同类别分别看 |

先说分词器差异。一个英文单词在 BPE 里可能是 1 个 token，也可能被拆成 3 个子词。中文里，同一个短语也会有不同切法。结果是：模型并不是在同一粒度上做预测。此时谁的 PPL 更低，未必意味着谁更懂文本，只可能意味着谁的 token 体系更有利。

再说域漂移。新闻文本往往语法稳定、格式规整，医学文本术语密集、长尾词多，代码又有大量符号与缩进模式。模型在三者上的 PPL 可能差一个数量级。拿通用语料上的低 PPL 去承诺专业场景效果，通常站不住。

最危险的是数据泄漏。所谓“泄漏”，白话讲就是测试集内容或其近重复样本已经进过训练集。这样模型看起来像“理解得很好”，其实只是“记住了”。这会让 PPL 虚低，尤其在公开 benchmark 被反复使用时更需要警惕。

工程里有一类思路是做“反向基准”或近似检测，例如 CraftGPT 讨论过的 Backwards Benchmark：如果模型对某些评测样本异常熟悉，甚至对其改写、逆序线索也表现出不合理稳定性，就要怀疑训练语料里出现过相同或高度相似内容。它不是唯一检测法，但提醒了一个事实：低 PPL 不一定来自泛化，也可能来自记忆。

因此，靠谱的评估流程通常包括：

- 训练集和验证集做精确去重与近重复去重
- 对关键 benchmark 单独做污染审计
- 报告分场景 PPL，而不是只给一个总分
- 同时看人工抽样、任务指标和成本指标

---

## 替代方案与适用边界

PPL 的优势是定义清晰、实现简单、和语言模型训练目标直接一致。但它也有边界。

当你无法统一 tokenizer 时，优先考虑 BPC，或者更进一步用按字节归一的指标。原因很直接：字符或字节是比 token 更稳定的比较单位，能减少“切词方案不同”带来的误差。

| 指标 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| PPL | 直观，和 LM 训练目标一致 | 依赖 tokenizer 和语料 | 同模型族、同词表比较 |
| BPC | 更适合跨 tokenizer 比较 | 对中文和多字节编码要说明口径 | 异构模型横向比较 |
| Accuracy/Top-k | 容易解释 | 只看命中，不看概率质量 | 封闭词表预测任务 |
| 任务指标 | 贴近业务目标 | 依赖具体任务定义 | 摘要、问答、补全上线评估 |
| FLOPS/延迟/成本 | 反映可部署性 | 不衡量语言质量 | 工程选型 |

什么时候 PPL 不够用？

第一，业务目标不是“预测下一个 token”，而是“回答是否正确”。比如分类、排序、工具调用成功率，这时任务指标更重要。

第二，模型已经很强，PPL 的微小改善未必带来用户体验改善。一个模型 PPL 从 12 降到 11，可能很难转化成可感知收益，但延迟增加一倍就很明显。

第三，tokenizer 粒度极细时，PPL 会显得很大或很小，但解释性变差。比如字符级模型和子词级模型的 PPL 不该直接拿来排高低，此时 BPC 更稳。

可以记一个简单原则：

- 同协议比较，用 PPL
- 跨协议比较，用 BPC
- 上线决策，再叠加真实任务指标与成本指标

---

## 参考资料

- Hugging Face Transformers, “Perplexity of fixed-length models”
  说明 PPL 的标准定义、固定长度模型在长文本评估时的窗口问题，以及为什么简单分块会失真。  
  https://huggingface.co/docs/transformers/en/perplexity

- TensorTonic, “Perplexity”
  解释了 PPL 的直觉含义、与交叉熵的关系，以及如何转成 BPC 做更可比的归一化。  
  https://www.tensortonic.com/ml-math/information-theory/perplexity

- CraftGPT, “GPT-3 knows about its evals”
  讨论评测集污染与数据泄漏问题，说明为什么过低的评测分数不一定代表真实泛化能力。  
  https://www.craftgpt.io/blog/gpt-3-knows-about-its-evals
