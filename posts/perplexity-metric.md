## 核心结论

困惑度（Perplexity，简称 PPL）是语言模型最常见的基础评测指标之一。它衡量的是：模型在一段未见过的文本上，平均预测下一个 token 时有多“拿不准”。这里的 token 可以理解为模型处理文本的最小单位，可能是字、词、子词，不一定等于“一个汉字”或“一个单词”。

PPL 的标准定义是：

$$
\mathrm{PPL}=\exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log P(x_t\mid x_{<t})\right)
$$

其中，$x_t$ 是第 $t$ 个 token，$x_{<t}$ 表示它前面的上下文，$N$ 是总 token 数。这个公式的含义是：先计算每个 token 的平均负对数似然，再做指数化。负对数似然可以白话理解为“模型为正确答案付出的信息代价”，越小越好。

最重要的三个结论：

| 结论 | 含义 |
|---|---|
| PPL 越低越好 | 说明模型在验证集或测试集上更能把概率集中到正确 token 上 |
| PPL 本质上等价于交叉熵的指数 | 交叉熵是平均信息损失，PPL 是它更直观的“有效候选数”版本 |
| PPL 不能脱离 tokenizer 单独比较 | 分词粒度不同，PPL 会系统性变化，跨模型比较必须控制分词器 |

一个直觉是：如果某位置上模型像是在 10 个候选里“平均瞎猜一个”，那它的 PPL 大约就是 10；如果它平均只需要在 2 个候选里选，PPL 大约就是 2。这个解释不严格等于“真实候选数”，但非常适合建立第一层理解。

玩具例子：“北京的首都叫 ___”这个句子本身就是错误的，模型如果训练充分，可能会对“首都”“中国”“北京”周围词序产生强烈警惕，正确 token 的概率分布会更尖锐或更异常。相反，对“中华人民共和国的首都是 ___”，高质量模型会把大量概率压到“中国首都相关 token”上，此时 PPL 会更低。

---

## 问题定义与边界

PPL 只适用于“按顺序预测下一个 token”的语言模型，也就是 causal language model。causal 可以白话理解为“只能看左边，不能偷看右边”。GPT 类模型属于这一类，因此 PPL 常用于 GPT、LLaMA 一类模型的验证集评估。

如果模型不是按“预测下一个 token”训练的，比如 BERT 的主要任务是 masked language modeling，也就是“把句子里被遮住的位置猜回来”，那么直接用标准 PPL 比较就不合适。因为训练目标不一致，指标对应的含义也变了。

PPL 的边界主要有两个：

1. 它评估的是概率分配质量，不直接评估任务完成质量。
2. 它依赖 tokenizer，也就是依赖“文本被切成什么粒度的 token”。

第二点尤其重要。假设同一句话：

- 模型 A 按词切分，共 3 个 token
- 模型 B 按字符切分，共 12 个字符 token

即使两者都“理解”了这句话，PPL 也可能完全不同。因为 A 每一步预测的是更大的单位，B 每一步预测的是更小的单位，平均损失天然不可直接对齐。

因此，跨 tokenizer 比较时，经常引入 BPC（Bits Per Character，每字符比特数）。bit 可以白话理解为“信息量的最小二进制单位”。BPC 的常见写法是：

$$
\mathrm{BPC}=\frac{H\times T}{C}
$$

其中：

- $H$ 是以 bit 为单位的平均 token 交叉熵
- $T$ 是 token 总数
- $C$ 是字符总数

如果交叉熵原本用自然对数计算，先要除以 $\ln 2$ 才能转成 bit：

$$
H_{\text{bit}}=\frac{H_{\text{nat}}}{\ln 2}
$$

于是也可以写成：

$$
\mathrm{BPC}=\frac{\left(-\frac{1}{T}\sum_{t=1}^{T}\log_2 P(x_t\mid x_{<t})\right)\times T}{C}
$$

这相当于把“每个 token 的平均损失”重新换算成“每个字符的平均损失”，从而削弱 tokenizer 差异。

一个小例子：

| 模型 | token 数 $T$ | 字符数 $C$ | 平均交叉熵 $H_{\text{bit}}$ | BPC |
|---|---:|---:|---:|---:|
| 词级模型 | 3 | 12 | 3.2 | $3.2\times3/12=0.8$ |
| 字符级模型 | 12 | 12 | 0.85 | $0.85\times12/12=0.85$ |

这时如果只看 PPL，两个模型可能差很多；但转成 BPC 后，信息密度更接近，就更适合比较。

---

## 核心机制与推导

理解 PPL，关键是把它拆成三层：

1. 概率
2. 对数
3. 指数还原

先看一条长度为 $N$ 的序列 $x_1,x_2,\dots,x_N$。语言模型给这条序列分配的联合概率是：

$$
P(x_1,\dots,x_N)=\prod_{t=1}^{N}P(x_t\mid x_{<t})
$$

这表示整句概率等于每一步条件概率相乘。直接连乘会很小，也不方便优化，因此通常取对数：

$$
\log P(x_1,\dots,x_N)=\sum_{t=1}^{N}\log P(x_t\mid x_{<t})
$$

再取负号并平均，就得到平均负对数似然：

$$
-\frac{1}{N}\sum_{t=1}^{N}\log P(x_t\mid x_{<t})
$$

这就是交叉熵在语言建模场景下的形式。交叉熵可以白话理解为“模型为了编码真实文本，平均每个 token 要花多少信息代价”。

最后再做指数化：

$$
\mathrm{PPL}=\exp(\text{cross-entropy})
$$

这样做的意义是把“信息代价”还原成“等效候选数”。所以 PPL 不是凭空发明的指标，而是交叉熵的直观版本。

看两个两步预测的玩具例子。

### 例子 1：完全不偏好

模型在两个位置都给正确 token 概率 0.5：

$$
-\frac{1}{2}(\log 0.5+\log 0.5)= -\frac{1}{2}(-0.693-0.693)=0.693
$$

于是：

$$
\mathrm{PPL}=e^{0.693}\approx 2
$$

这说明模型相当于平均在 2 个等可能选项里选一个。

### 例子 2：高度自信

模型在两个位置都给正确 token 概率 0.9：

$$
-\frac{1}{2}(\log 0.9+\log 0.9)= -\log 0.9\approx 0.105
$$

于是：

$$
\mathrm{PPL}=e^{0.105}\approx 1.11
$$

这表示模型几乎不需要“在多个选项中犹豫”，有效候选数接近 1。

对比表如下：

| 正确 token 概率分布 | 平均负对数似然 | PPL | 直观解释 |
|---|---:|---:|---|
| 0.5 / 0.5 | 0.693 | 2.00 | 像公平二选一 |
| 0.9 / 0.9 | 0.105 | 1.11 | 几乎确定答案 |
| 0.1 / 0.1 | 2.303 | 10.00 | 像在 10 个选项里瞎猜 |

这里能看到一个核心关系：平均 log 概率越高，负对数越低，PPL 越小；平均 log 概率越低，PPL 越大。

真实工程例子是训练 GPT 类模型时的验证流程。训练期间，模型会在 validation set 上周期性跑一次前向计算，把每个位置上正确 token 的 log probability 累加，然后除以总 token 数，再取 $\exp$。如果某次迭代的验证集 PPL 从 18.4 降到 16.9，通常可以判断模型对未见文本的下一个 token 预测更稳了。但这仍然只说明“语言建模损失下降”，不等于问答、检索增强、工具调用一定同步变强。

---

## 代码实现

下面给出一个最小可运行实现。它不依赖深度学习框架，只演示 PPL 的数学计算过程。为了避免 $\log 0$，代码里加了 `eps`，也就是一个很小的正数保护项。

```python
import math

def perplexity_from_probs(correct_token_probs, eps=1e-12):
    """
    correct_token_probs: list[float]
        每个位置上，模型给“正确 token”的概率
    """
    assert len(correct_token_probs) > 0
    neg_log_sum = 0.0
    for p in correct_token_probs:
        assert 0.0 <= p <= 1.0
        neg_log_sum += -math.log(max(p, eps))
    avg_neg_log = neg_log_sum / len(correct_token_probs)
    return math.exp(avg_neg_log)

def bpc_from_probs(correct_token_probs, char_count, eps=1e-12):
    """
    先算平均 nat 交叉熵，再转 bit，最后换算成每字符信息量
    这里假设一个概率对应一个 token
    """
    assert len(correct_token_probs) > 0
    assert char_count > 0

    neg_log_sum = 0.0
    for p in correct_token_probs:
        neg_log_sum += -math.log(max(p, eps))  # nat

    token_count = len(correct_token_probs)
    avg_cross_entropy_nat = neg_log_sum / token_count
    avg_cross_entropy_bit = avg_cross_entropy_nat / math.log(2)

    return avg_cross_entropy_bit * token_count / char_count

# 玩具例子 1：两个位置都只有 0.5 把握，PPL 应接近 2
ppl_half = perplexity_from_probs([0.5, 0.5])
assert abs(ppl_half - 2.0) < 1e-6

# 玩具例子 2：两个位置都 0.9 把握，PPL 应接近 1.111...
ppl_confident = perplexity_from_probs([0.9, 0.9])
assert 1.10 < ppl_confident < 1.12

# 一个简单的 BPC 例子
bpc = bpc_from_probs([0.8, 0.6, 0.9], char_count=12)
assert bpc > 0

print("PPL(0.5,0.5) =", round(ppl_half, 6))
print("PPL(0.9,0.9) =", round(ppl_confident, 6))
print("BPC example =", round(bpc, 6))
```

如果你用 PyTorch 或其他框架，真实实现通常不是直接传入概率，而是传入 logits。logits 可以白话理解为“还没过 softmax 的原始打分”。流程一般是：

1. 用模型得到每个位置的 logits
2. 对 logits 做 `log_softmax`
3. 取出目标 token 对应位置的 log probability
4. 累加负值
5. 除以总 token 数
6. 取 `exp`

伪代码结构如下：

```python
# logits: [batch, seq_len, vocab_size]
# targets: [batch, seq_len]

log_probs = log_softmax(logits, dim=-1)
target_log_probs = gather(log_probs, index=targets)
loss = -sum(target_log_probs * mask) / sum(mask)
ppl = exp(loss)
```

这里的 `mask` 很重要。它表示哪些位置应该计入损失，比如 padding 不应参与统计。padding 可以白话理解为“为了把不同长度样本凑成同一批而补的空位”。

真实工程例子：如果你在训练一个中文 causal LM，验证代码往往会按 batch 处理一整份验证集，累计 `total_neg_log_likelihood` 和 `total_tokens`，最终计算：

$$
\mathrm{PPL}=\exp\left(\frac{\text{total\_neg\_log\_likelihood}}{\text{total\_tokens}}\right)
$$

如果还要跨 tokenizer 对比，再额外统计 `total_characters`，换算成 BPC。

---

## 工程权衡与常见坑

PPL 很有用，但它不是“一个数字说明全部问题”的指标。工程里最常见的坑有下面几类。

| 问题 | 现象 | 规避策略 |
|---|---|---|
| Tokenizer 依赖 | 不同分词器下 PPL 不可直接比 | 尽量统一 tokenizer，或改用 BPC |
| 只看验证集 PPL | 语言建模更好，但任务效果未必更好 | 同时看下游任务指标 |
| 长文本截断 | 固定上下文窗口会高估真实困惑 | 使用滑动窗口评估 |
| 统计口径不一致 | 是否算 BOS/EOS/padding 会影响结果 | 明确评测协议并固定实现 |
| $\log 0$ 或数值下溢 | 极小概率导致 NaN 或 Inf | 用 `log_softmax` 和最小值保护 |

先看 tokenizer 依赖。假设模型 A 用较大的子词词表，模型 B 用字符级词表。A 一次预测的单位更“粗”，有时会得到更低的 PPL。这不一定说明 A 真懂得更多，可能只是 token 定义不同。所以论文里如果直接写“我们的 PPL 更低，因此模型更强”，必须先看是不是同一套 tokenizer。

再看“PPL 低不代表下游一定强”。这点对初学者很重要。PPL 关注的是“把下一个 token 的概率分布拟合得好不好”；而很多实际任务关注的是“最终答案是否正确”。两者相关，但不是同一目标。

以 GPT-3 一类工作为代表，可以看到一个典型现象：更强的自回归语言建模能力通常会带来更低的 PPL，但在某些 few-shot 任务上，它未必稳定超过专门为判别任务微调的模型。few-shot 可以白话理解为“只给几个样例，不做参数更新，直接让模型现场完成任务”。这时，提示格式、任务映射方式、输出约束都会影响表现，而这些因素不是 PPL 单独能覆盖的。

一个真实工程例子：

- 团队训练一个中文 GPT，用新闻语料做验证，PPL 从 12.8 降到 11.9
- 同时拿它做客服问答 few-shot，准确率只从 68% 升到 68.5%
- 继续训练到 PPL 11.5 后，few-shot 准确率反而跌到 67.9%

这通常不是“PPL 没价值”，而是因为：
- 验证语料分布和问答任务分布不一致
- 模型更擅长续写，不一定更擅长按格式输出答案
- few-shot 任务对 instruction-following 更敏感，而不仅是 token 概率拟合

因此，PPL 适合做“训练是否在继续学语言分布”的主指标，不适合单独做“产品效果是否提升”的总指标。

还有一个技术坑是长文本评估。很多实现会把长文本切成固定长度块，并在每块起点重新开始计算。这会丢失前文上下文，导致 PPL 偏高。更合理的方法是滑动窗口评估，即每次让模型尽量利用更多历史上下文。这样得到的结果更接近模型真实使用场景。

---

## 替代方案与适用边界

如果目标是评估 causal LM 的基础建模能力，PPL 仍然是首选指标之一。它计算直接、可复现、与训练目标一致，特别适合：

- 比较同一模型不同 checkpoint
- 做超参数筛选
- 监控是否过拟合
- 评估同 tokenizer 下的语言建模改进

但如果目标变成“模型能不能把任务做对”，就必须引入任务指标。

| 指标 | 适用环境 | 优点 | 局限 |
|---|---|---|---|
| PPL | causal LM 验证集评估 | 与训练目标直接一致，计算稳定 | 受 tokenizer 影响，不直接反映任务正确率 |
| BPC | 跨 tokenizer 或字符级比较 | 更接近 tokenizer-agnostic | 直觉不如 PPL 强，工业报告中不如 PPL 常见 |
| Accuracy | 分类、选择、结构化输出 | 结果直接，容易解释 | 不能反映概率校准质量 |
| EM/F1 | 问答、抽取 | 对最终答案质量更敏感 | 不适合纯生成式流畅度评估 |

玩具例子：你训练一个小型字符级语言模型和一个子词级语言模型来预测古诗。字符级模型 PPL 可能很高，因为它每次只预测一个字；子词级模型 PPL 可能更低，因为 token 更大。但如果转成 BPC，二者可能接近。这时 BPC 更适合回答“谁对文本压缩得更好”。

真实工程例子：你在做一个问答助手。训练阶段监控验证集 PPL 是合理的，因为它可以告诉你模型是否还在学习语言分布；上线前则必须同时看 answer accuracy、EM 或 F1，因为用户关心的是“答案对不对”，不是“模型对下一 token 有多自信”。

所以适用边界可以总结为：

- 看“语言建模本身”，优先用 PPL
- 看“跨 tokenizer 公平比较”，补充 BPC
- 看“任务完成效果”，必须加任务指标
- 看“产品体验”，还要加人工评测、延迟、成本等工程指标

PPL 是基础仪表盘，不是总裁判。

---

## 参考资料

1. Hugging Face Transformers 文档：Perplexity  
主要内容：给出 PPL 的标准定义、与交叉熵的关系、固定上下文窗口评估时的注意事项，以及 tokenizer 敏感性。  
用处：适合确认公式、实现口径和评测边界。

2. TensorTonic: Perplexity in LLMs  
主要内容：用“有效候选数”或“多面骰子”的方式解释 PPL。  
用处：适合建立直觉，尤其适合第一次接触该指标的读者。

3. Michael Brenndoerfer 关于 Bits Per Character 的说明  
主要内容：解释 BPC 与交叉熵、字符级比较之间的关系。  
用处：适合理解为什么跨 tokenizer 时不能只看 PPL，以及 BPC 如何补位。

4. Language Models are Few-Shot Learners（GPT-3 论文）  
主要内容：展示大语言模型在 few-shot 场景下的能力，并体现“语言建模目标”和“具体下游任务目标”并不完全相同。  
用处：适合理解为什么 PPL 更低不等于所有 benchmark 都更强。

5. 语言模型与信息论基础教材或课程笔记  
主要内容：交叉熵、负对数似然、编码长度之间的关系。  
用处：如果想从根上理解“为什么要取对数、为什么还要指数化”，这类材料最有帮助。
