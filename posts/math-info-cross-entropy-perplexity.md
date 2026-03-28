## 核心结论

交叉熵是衡量“模型给真实答案分配了多少概率”的指标。白话说，它看的是：真实数据明明这样出现，模型却给了多大的意外程度。对离散分布 $p$ 和模型分布 $q$，定义为

$$
H(p,q)=-\sum_x p(x)\log q(x)
$$

它可以拆成

$$
H(p,q)=H(p)+D_{KL}(p\|q)\ge H(p)
$$

其中，熵 $H(p)$ 是“数据本身有多不确定”，KL 散度 $D_{KL}(p\|q)$ 是“模型和真实分布差了多少”。等号只在 $p=q$ 时成立，所以最小化交叉熵，本质上就是让模型分布逼近真实分布。

困惑度是交叉熵的指数形式：

$$
PPL=\exp(H)
$$

对白话理解最重要的一点是：它近似表示模型每一步像在多少个等概率选项里猜。若 $PPL=4$，可以理解为模型平均像在 4 个同样可能的 token 中选一个；若 $PPL=100$，则说明模型平均每一步都像在 100 个候选里犹豫。

在语言模型训练里，常见损失

$$
L=-\frac{1}{N}\sum_{t=1}^{N}\log P_\theta(x_t\mid x_{<t})
$$

就是交叉熵，也是负对数似然。白话说，模型每次预测下一个 token 时，只要给正确 token 的概率越高，损失就越低，困惑度也越低。

---

## 问题定义与边界

问题的核心不是“模型生成得像不像人”，而是“模型给真实序列分配的概率高不高”。这里必须先明确三个对象。

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $p$ | 真实数据分布 | 语料里真正出现规律 |
| $q_\theta$ | 模型分布 | 模型学出来的概率分配 |
| $x_t, x_{<t}$ | 当前 token 与历史上下文 | 预测第 $t$ 个词时，前面已经看到的内容 |

对自回归语言模型，评估对象是整段序列的条件概率链：

$$
q_\theta(x_{1:N})=\prod_{t=1}^{N}q_\theta(x_t\mid x_{<t})
$$

于是平均交叉熵就是每个位置的平均负对数概率。

这里有一个非常重要的边界：困惑度依赖 token 粒度。tokenizer 是“把文本切成模型处理单位的规则”，可以按字、词、子词切。切法不同，$N$ 就不同，词表大小也不同，因此 PPL 不能跨粒度直接比较。

| 切分方式 | 典型特点 | PPL 是否可直接和别的粒度比较 |
| --- | --- | --- |
| 字符级 | 词表小，序列长 | 不可直接比较 |
| 词级 | 词表大，未登录词多 | 不可直接比较 |
| 子词级 | 工程上最常见 | 只能和同 tokenizer 模型比较 |

玩具例子：文本 “hello” 若按字符切成 `h e l l o`，模型只需在几十个字符里选，PPL 可能很低；若按词切成 `hello`，模型要在成千上万个词里选，PPL 会高很多。两者不是一个难度单位，因此不能说“字符级模型更强”。

真实工程例子：在 WikiText-103 上比较两个 Transformer，如果它们使用同一个 BPE tokenizer，那么 dev loss 和 PPL 可以直接比较；如果一个用 BPE、一个用 unigram tokenizer，PPL 数值本身就不再公平。

---

## 核心机制与推导

先看单步。若真实 token 是 $x$，模型给它的概率是 $q(x)$，这一步的损失就是

$$
-\log q(x)
$$

这叫负对数似然，白话说就是“模型越不信正确答案，罚得越重”。如果给对的 token 概率是 1，损失为 0；若概率只有 0.01，损失就很大。

推广到长度为 $N$ 的序列：

$$
L=-\frac{1}{N}\sum_{t=1}^{N}\log q_\theta(x_t\mid x_{<t})
$$

这里：

| 符号 | 含义 |
| --- | --- |
| $N$ | 被统计的有效 token 数 |
| $x_t$ | 第 $t$ 个真实 token |
| $x_{<t}$ | 第 $t$ 步之前的上下文 |
| $q_\theta$ | 参数为 $\theta$ 的模型概率分布 |

这就是训练时常见的交叉熵损失。若数据来自真实分布 $p$，这个经验平均在样本足够多时就逼近 $H(p,q_\theta)$。因此，训练语言模型就是在最小化真实数据和模型分布之间的交叉熵。

然后把它指数化：

$$
PPL=\exp(L)
$$

为什么要这样做？因为 $\exp$ 会把“平均对数损失”还原成“等效候选数”。如果平均损失是 $\ln 10$，那么困惑度就是 10，意思是模型平均像在 10 个等可能选项里选一个。

玩具例子：假设长度为 3 的序列，模型对正确 token 的预测概率依次是 $0.5,0.25,0.25$。则

$$
L=-\frac{1}{3}(\log 0.5+\log 0.25+\log 0.25)
$$

因为

$$
-\log 0.5\approx 0.693,\quad -\log 0.25\approx 1.386
$$

所以

$$
L\approx \frac{0.693+1.386+1.386}{3}=1.155
$$

进一步得到

$$
PPL=\exp(1.155)\approx 3.17
$$

这说明模型平均相当于在约 3.17 个等概率 token 之间做选择。

还可以用信息论角度理解。若对数以 2 为底，交叉熵单位就是 bit，表示“平均需要多少比特编码一个真实 token”。于是

$$
H_2(p,q)=H_2(p)+D_{KL}(p\|q)
$$

模型越接近真实分布，额外浪费的编码长度越少。

---

## 代码实现

下面给一个可运行的 Python 例子。它模拟了 3 个位置、4 个类别的 logits，手工屏蔽了一个 padding 位置，并计算平均交叉熵与困惑度。

```python
import math

def log_softmax(row):
    m = max(row)
    shifted = [x - m for x in row]
    exps = [math.exp(x) for x in shifted]
    s = sum(exps)
    return [x - math.log(s) for x in shifted]

# 3 个 token 位置，4 个类别
logits = [
    [2.0, 0.0, -1.0, -2.0],   # target = 0
    [0.1, 1.2, 0.0, -0.5],    # target = 1
    [1.0, 0.3, 0.2, -0.1],    # target = 3，但这个位置被 mask 掉
]
targets = [0, 1, 3]
mask = [1, 1, 0]  # 只统计前两个有效 token

total_nll = 0.0
total_tokens = 0

for row, target, m in zip(logits, targets, mask):
    if m == 0:
        continue
    lsm = log_softmax(row)
    total_nll += -lsm[target]
    total_tokens += 1

loss = total_nll / total_tokens
ppl = math.exp(loss)

assert total_tokens == 2
assert loss > 0
assert ppl >= 1.0

print("avg_cross_entropy =", round(loss, 6))
print("perplexity =", round(ppl, 6))
```

工程里常见写法是先按 `sum` 聚合，再除以有效 token 数，而不是直接用默认 `mean`。原因是 padding 会污染分母。若一个 batch 里短句很多，错误的平均方式会让 loss 和 PPL 偏小，看起来模型变好了，实际上只是统计口径错了。

真实工程例子：在 PyTorch 训练语言模型时，通常会把 `ignore_index=pad_id` 传给交叉熵，或者自己用 mask 汇总 `-log p`。验证阶段同步记录 `total_nll` 和 `total_tokens`，最后统一计算 `loss = total_nll / total_tokens`、`ppl = exp(loss)`。这样不同 batch 长度不一致时，统计仍然稳定。

---

## 工程权衡与常见坑

第一个坑是 tokenizer 依赖性。词表越小、token 越短，模型每步面对的候选空间通常越小，PPL 往往也更低。这不一定表示模型更强，可能只是度量单位变了。

第二个坑是评测集依赖性。WikiText-103、Penn Treebank、代码语料、中文新闻语料，分布都不同。PPL 反映的是“模型在这个数据分布上的拟合程度”，不是通用智力分数。跨语料比较，结论通常没有意义。

第三个坑是分母定义。序列截断、padding、特殊 token 是否计入，都影响 $N$。工程上必须保证训练、验证、不同实验的统计口径一致。

第四个坑是把 PPL 和生成质量直接画等号。较低 PPL 通常表示概率估计更准，但不必然意味着摘要更好、对话更安全、事实更准确。因为生成任务还受解码策略、指令跟随、对齐目标影响。

| 场景 | 常见误用 | 正确做法 |
| --- | --- | --- |
| 不同 tokenizer 比较 | 直接比较 PPL 数值 | 改用同 tokenizer，或用 BPB/BPC |
| 不同数据集比较 | 认为低 PPL 一定更强 | 只在同一评测集内比较 |
| 含 padding 的 batch | 用错误分母平均 | 只统计有效 token |
| 生成任务评估 | 用 PPL 代替全部指标 | 与任务指标联合使用 |

一个经验性认识是：传统 n-gram 或较早期循环/卷积语言模型，在公开英文数据集上的 PPL 常见在 50 到 100 或更低的区间；更强的预训练 Transformer 可以显著降到更低。比如 GPT-3 在经典英文基准上报告过约 20 左右的量级。这些数字只能在接近的评测设置下理解，不能脱离 tokenizer、语料和预处理单独解读。

---

## 替代方案与适用边界

当目标是比较“不同 tokenizer 的模型谁更会建模文本”，PPL 就不够公平。这时更常用的是 bits-per-byte，简称 BPB，意思是“平均每个字节需要多少 bit 编码”。

若总负对数似然用自然对数计算为 $\text{NLL}$，总字节数为 $B$，则

$$
BPB=\frac{\text{NLL}}{B\ln 2}
$$

如果按字符归一化，则得到 BPC，即 bits-per-character。它们的优点是跨 tokenizer 更可比，因为字节和字符是更稳定的底层单位。

| 指标 | 公式 | 适用场景 |
| --- | --- | --- |
| PPL | $\exp(\frac{\text{NLL}}{N})$ | 同一 tokenizer、同一数据集内比较 |
| BPB | $\frac{\text{NLL}}{B\ln 2}$ | 跨 tokenizer、跨分词策略比较 |
| BPC | $\frac{\text{NLL}}{C\ln 2}$ | 字符级建模或字符语料分析 |

玩具例子：模型 A 用字符切分，PPL=3；模型 B 用子词切分，PPL=18。不能直接说 A 更好。但若换成 BPB，可能发现 A 是 1.45，B 是 1.20，那么 B 的概率建模反而更有效。

真实工程例子：在同一篇中英混合文档上，字符级 tokenizer 和 BPE tokenizer 的 token 数差异可能达到数倍。若团队只看 token-level PPL，很容易误以为字符模型更优；若统一换成 byte 归一化，结论通常更稳定，也更适合跨系统对比。

PPL 的适用边界很明确：它适合评估语言模型对真实序列的概率估计能力，不适合单独衡量问答正确率、摘要覆盖率、检索质量或事实一致性。做生成系统时，PPL 更像底层模型健康度指标，而不是最终业务指标。

---

## 参考资料

- Wikipedia, “Cross-entropy”. 交叉熵、熵、KL 散度关系公式来源，支撑“核心结论”“核心机制与推导”章节。访问日期：2026-03-28。  
- Hugging Face Transformers Docs, “Perplexity of fixed-length models”. 语言模型中 $L=-\frac{1}{N}\sum \log P_\theta(x_t|x_{<t})$ 与 $PPL=\exp(L)$ 的说明，支撑“核心机制与推导”“代码实现”章节。访问日期：2026-03-28。  
- TensorTonic, “Perplexity”. 对“平均分支因子”这一初学者友好解释有帮助，支撑“核心结论”“问题定义与边界”章节。访问日期：2026-03-28。  
- ResearchGate 上关于 WikiText-103 模型结果的表格材料。用于说明传统语言模型在公开英文基准上的 PPL 量级，支撑“工程权衡与常见坑”章节。访问日期：2026-03-28。  
- mbrenndoerfer, “Perplexity: language model evaluation metric”. 讨论 tokenizer 依赖性、BPB/BPC 等归一化口径，支撑“问题定义与边界”“替代方案与适用边界”章节。访问日期：2026-03-28。
