## 核心结论

ELECTRA 是一种预训练编码器的方法：小生成器 `G` 先把少量词元替换成看起来合理的假词，大判别器 `D` 再对句子中每个词元判断“这是原词，还是被替换过”。

它的核心不是“换一种 `mask` 方式”，而是把主要训练目标从 `MLM` 改成 `RTD`。`MLM` 是 masked language modeling，意思是“遮住一部分词，让模型猜原词”。`RTD` 是 replaced token detection，意思是“给模型一整句可能被改过的话，让模型逐个位置判断真假”。

新手版可以这样理解：BERT 主要做“填空题”，ELECTRA 主要做“找错题”。填空题只批改被挖空的位置；找错题会检查整句话里的每个位置。

| 对比项 | BERT `MLM` | ELECTRA `RTD` |
|---|---|---|
| 任务形式 | 预测被 `[MASK]` 的原词 | 判断每个 token 是否为原词 |
| 训练信号位置 | 通常只有约 15% token | 100% token 都参与 |
| 输入是否自然 | 训练时包含 `[MASK]` | 判别器看到的是被替换后的自然 token |
| 主要训练对象 | 编码器自己预测词 | 判别器学习真假判断 |
| 训练后使用 | 使用 MLM 编码器 | 通常只保留判别器 `D` |

训练完通常只保留判别器 `D`：

```text
训练阶段:
原文 -> mask -> 生成器 G -> 替换词 -> 判别器 D -> RTD 训练信号

下游阶段:
输入文本 -> 判别器 D / 编码器 -> 分类、检索、匹配等任务
          生成器 G 通常丢弃
```

效率优势来自监督密度。BERT 在一个长度为 $n$ 的句子里，通常只对 $0.15n$ 个位置计算 `MLM` 训练信号；ELECTRA 的判别器对 $n$ 个位置都计算二分类损失。相同文本、相近 batch 下，ELECTRA 能从更多位置获得梯度，因此经常用更少计算资源达到相近或更好的下游效果。

---

## 问题定义与边界

ELECTRA 解决的问题是“预训练信号稀疏”。预训练信号稀疏，指一条训练样本里只有少数位置真正产生监督损失，其余位置虽然参与上下文计算，但不直接被监督。

BERT 的 `MLM` 通常随机选取约 15% token 做预测。假设一句话有 100 个 token，那么主要只有约 15 个位置产生预测损失。ELECTRA 仍然可以只在约 15% 位置造假，但判别器会检查整句 100% 的 token：哪些还是真词，哪些已经被生成器替换。

| 方法 | 构造方式 | 训练信号覆盖范围 |
|---|---|---|
| `BERT MLM` | 遮住部分 token，预测原词 | 主要覆盖被 mask 的位置 |
| `ELECTRA RTD` | 先替换部分 token，再逐 token 判真伪 | 覆盖整句所有位置 |
| 覆盖差异 | 一个做填空，一个做找错 | ELECTRA 的监督密度更高 |

边界也要说清楚。ELECTRA 不是要替代所有 Transformer 编码器。Transformer 编码器是一类模型结构，负责把输入序列变成上下文相关的向量表示；ELECTRA 是一种预训练目标和训练流程，可以用在编码器预训练上。

ELECTRA 也不是典型 GAN。GAN 是 generative adversarial network，常见含义是生成器和判别器互相对抗，生成器直接学习骗过判别器。ELECTRA 里生成器 `G` 用 `MLM` 学会预测被遮住的词，判别器 `D` 用 `RTD` 学会判断 token 是否被替换。两者联合训练，但不是标准 GAN 式的对抗目标。

适用范围主要包括三类：

| 场景 | 是否适合 | 原因 |
|---|---:|---|
| 预训练文本编码器 | 适合 | `D` 可以作为通用编码器 |
| 小算力领域预训练 | 适合 | 更高监督密度通常更省训练步骤 |
| 分类、匹配、检索 | 适合 | 需要文本表示，不要求逐词生成 |
| 生成式语言模型 | 不适合当主方案 | ELECTRA 的核心输出不是自回归生成 |
| 聊天、续写、长文本生成 | 不适合直接套用 | 这些任务通常需要解码器或编码-解码结构 |

真实工程例子：企业内部有大量客服工单、产品文档和知识库文章，目标是训练一个中文领域编码器，用于意图分类、相似问题召回和问答匹配。预算只有单卡或少量 GPU。此时 ELECTRA 比从零训练 BERT `MLM` 更值得优先尝试，因为它能让同样的语料产生更密集的训练信号。

---

## 核心机制与推导

ELECTRA 的数据流可以分成五步：

```text
原文 x -> mask -> x_masked -> 生成器 G 预测 -> x_hat
       -> 替换成 x_corrupt -> 判别器 D 逐 token 判断真假
```

设原始输入为 $x=(x_1,\dots,x_n)$，其中 $x_i$ 是第 $i$ 个 token。token 是模型处理文本的基本单位，可以是字、词、子词或特殊符号。设被选中 mask 的位置集合为 $m$。

先把选中位置替换成 `[MASK]`：

$$
x_{\text{masked}} = REPLACE(x, m, [MASK])
$$

生成器 `G` 对被 mask 的位置预测原词分布，并从分布中采样替换词：

$$
\hat{x}_i \sim p_G(x_i \mid x_{\text{masked}}), \quad i \in m
$$

然后把采样结果填回原句，得到被污染输入：

$$
x_{\text{corrupt}} = REPLACE(x, m, \hat{x})
$$

判别器 `D` 对每个位置输出“该 token 是否为原词”的概率：

$$
D_t = P(y_t=1 \mid x_{\text{corrupt}})
$$

其中标签定义如下：

| 条件 | 标签 $y_t$ | 含义 |
|---|---:|---|
| 最终 token 与原文 token 相同 | 1 | 真实 token |
| 最终 token 与原文 token 不同 | 0 | 被替换 token |

注意标签不是“这个位置是否被 mask 过”，而是“最终 token 是否与原文一致”。如果某个位置被 mask 后，生成器又采样回原词，那么这个位置仍然是 $y_t=1$。

生成器损失仍然是 `MLM`：

$$
L_{MLM} = - \sum_{i \in m} \log p_G(x_i \mid x_{\text{masked}})
$$

判别器损失是逐 token 二分类交叉熵。二分类交叉熵是衡量“预测概率”和“真实 0/1 标签”差距的常用损失：

$$
L_{RTD} = - \sum_{t=1}^{n} [y_t \log D_t + (1-y_t)\log(1-D_t)]
$$

总损失常写成：

$$
L = L_{MLM} + \lambda L_{RTD}
$$

其中 $\lambda$ 是权重系数，用来控制判别器损失在总损失中的比例。

玩具例子：

原句：

```text
[我, 喜欢, 这, 本书]
```

只 mask 第 2 个词。生成器把“喜欢”替换成“讨厌”，得到：

```text
[我, 讨厌, 这, 本书]
```

标签是：

```text
[1, 0, 1, 1]
```

假设判别器输出概率为：

```text
D = [0.9, 0.2, 0.8, 0.7]
```

则：

$$
L_{RTD} = -\log(0.9) - \log(1-0.2) - \log(0.8) - \log(0.7)
$$

数值约为：

$$
0.105 + 0.223 + 0.223 + 0.357 = 0.908
$$

这个例子说明，虽然只替换了 1 个位置，但 4 个位置都参与训练。第 1、3、4 个位置训练模型识别“真实 token”，第 2 个位置训练模型识别“替换 token”。

---

## 代码实现

实现 ELECTRA 时，最重要的不是先写复杂 Transformer，而是把数据构造和标签生成写对。训练代码可以拆成三块：mask 采样、生成器采样替换、判别器二分类损失。

结构示意如下：

```python
masked_input = mask_tokens(input_ids, mask_prob=0.15)
gen_logits = generator(masked_input)
sampled_tokens = sample_from_logits(gen_logits)
corrupted_input, labels = replace_and_label(input_ids, masked_input, sampled_tokens)
disc_logits = discriminator(corrupted_input)
loss = mlm_loss(gen_logits, input_ids, masked_positions) + lam * rtd_loss(disc_logits, labels)
```

训练流程表：

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 输入 | `input_ids` | 原始 token 序列 | 提供原文 |
| `mask` | 原始序列 | `masked_input` | 选中少量位置 |
| `G` 输出 | `masked_input` | `gen_logits` | 预测被 mask token |
| 替换 | 原始序列 + 采样 token | `corrupted_input` 和 `labels` | 构造 RTD 数据 |
| `D` 输出 | `corrupted_input` | 每个位置真假概率 | 做二分类 |
| `loss` | `G` 和 `D` 输出 | `L_MLM + λ L_RTD` | 联合训练 |

下面是一个可运行的最小 Python 例子，只实现 `RTD` 标签和损失计算，用来验证核心逻辑：

```python
import math

def replace_and_label(original, mask_positions, sampled):
    corrupted = list(original)
    labels = [1] * len(original)

    for pos, new_token in zip(mask_positions, sampled):
        corrupted[pos] = new_token
        labels[pos] = 1 if new_token == original[pos] else 0

    return corrupted, labels

def binary_cross_entropy(probs, labels):
    eps = 1e-12
    loss = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, eps), 1 - eps)
        loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return loss

original = ["我", "喜欢", "这", "本书"]
mask_positions = [1]
sampled = ["讨厌"]

corrupted, labels = replace_and_label(original, mask_positions, sampled)
assert corrupted == ["我", "讨厌", "这", "本书"]
assert labels == [1, 0, 1, 1]

disc_probs = [0.9, 0.2, 0.8, 0.7]
loss = binary_cross_entropy(disc_probs, labels)

assert round(loss, 3) == 0.908

sampled_same = ["喜欢"]
corrupted_same, labels_same = replace_and_label(original, mask_positions, sampled_same)
assert corrupted_same == original
assert labels_same == [1, 1, 1, 1]
```

这段代码体现两个关键点。第一，`RTD` 的损失覆盖所有位置。第二，如果生成器采样回原词，标签必须是 `1`，不能因为该位置被 mask 过就标成 `0`。

推理阶段更简单。训练结束后，通常只保留 `discriminator`。它本质上是一个预训练好的文本编码器，可以接分类头、匹配头、检索向量头，用在下游任务中。生成器 `G` 主要服务于预训练数据构造，通常不进入最终部署链路。

---

## 工程权衡与常见坑

ELECTRA 的关键超参之一是 `generator_size`。它表示生成器容量大小，通常小于判别器。容量是模型能表达复杂模式的能力，常由层数、隐藏维度、参数量共同决定。

生成器太弱时，替换词很假，判别器很快学会表面规律。例如原句是“用户提交退款申请”，弱生成器可能替换成明显不通顺的词，判别器不需要理解语义也能找出错误。生成器太强时，替换词过于接近真实上下文，判别器任务变得过难，训练也可能不稳定。合理的生成器应该能制造“有一定迷惑性但仍可学习”的负例。

| 问题 | 现象 | 规避方式 |
|---|---|---|
| 把 ELECTRA 当 GAN | 误解训练目标 | 强调 `G` 只做 `MLM`，不是直接对抗骗 `D` |
| 标签定义错误 | `RTD` 学歪 | 按最终 token 是否等于原词标注 |
| 生成器过弱/过强 | 替换太简单或太难 | 调 `generator_size` 和采样策略 |
| 只看预训练 loss | 误判模型好坏 | 以下游指标为准 |
| 忽略 mask 策略 | 负例分布异常 | 保持稳定的 mask 比例和随机性 |
| 训练后误用生成器 | 部署链路复杂 | 下游通常只使用判别器 `D` |

新手最容易错的是标签定义。假设原句是：

```text
[我, 喜欢, 这, 本书]
```

第 2 个位置被 mask。生成器如果采样出“讨厌”，最终 token 与原文不同，标签是 `0`。生成器如果采样出“喜欢”，最终 token 与原文相同，标签是 `1`。判断依据不是“是否被选中 mask”，而是“替换后的最终结果是否还等于原文”。

工程建议可以压缩成三条：

| 建议 | 原因 |
|---|---|
| 小算力优先从小生成器开始 | 减少额外训练成本，先验证收益 |
| 领域语料先做短周期验证 | 不同领域的替换难度和下游收益不同 |
| 关注下游任务而不是只盯 `RTD loss` | 预训练损失不等于业务效果 |

真实工程中，一个常见流程是：先用领域语料训练 ELECTRA-small 级别模型，跑一轮短周期预训练；再在意图分类、相似问召回、句对匹配上微调；如果下游指标稳定优于原有 BERT 或 RoBERTa baseline，再扩大训练步数和模型规模。这样比一开始就做大规模预训练更可控。

---

## 替代方案与适用边界

如果目标是通用编码器预训练，ELECTRA 通常比纯 `MLM` 更省样本、更省步骤。原因是判别器每个位置都有训练信号。如果目标是生成式建模，ELECTRA 不是首选，因为它不是按从左到右预测下一个 token 的训练方式，也不是天然的编码-解码生成框架。

| 方法 | 训练目标 | 信号密度 | 适用场景 |
|---|---|---:|---|
| BERT | `MLM` | 低 | 通用编码器 |
| RoBERTa | 强化版 `MLM` | 低 | 大语料编码器预训练 |
| ELECTRA | `RTD` + `MLM` | 高 | 低算力预训练、领域编码器 |
| T5 | span corruption | 中 | 编码-解码统一建模 |
| GPT 类模型 | next token prediction | 高 | 文本生成、对话、续写 |

新手版场景判断：

| 需求 | 是否适合 ELECTRA | 判断 |
|---|---:|---|
| 我要一个领域编码器，预算有限 | 适合 | 高监督密度有利于节省训练步骤 |
| 我要做文本分类 | 适合 | 判别器可接分类头 |
| 我要做语义检索 | 适合 | 判别器可产出文本表示 |
| 我要做文本生成 | 不适合当主方案 | 应优先考虑解码器或编码-解码模型 |
| 我要训练聊天模型 | 不适合直接作为主训练目标 | 聊天需要生成能力和指令对齐 |

ELECTRA 的优势是训练效率高，尤其适合分类、检索、匹配、领域表征这些“理解型”任务。劣势是训练流程比普通 `MLM` 更复杂：需要同时维护生成器和判别器，需要正确构造替换样本和标签，还要调生成器大小。

适用边界清单：

| 类型 | 结论 |
|---|---|
| 适合 | 分类、检索、匹配、排序、领域表征 |
| 适合 | 小算力领域预训练 |
| 适合 | 需要文本编码器的业务系统 |
| 不适合 | 把它当作生成式解码器训练主法 |
| 不适合 | 只需要直接生成长文本的任务 |
| 谨慎 | 对生成器容量、采样策略没有调参预算的项目 |

最终可以用一句话判断：如果下游任务需要“理解文本并输出表示”，ELECTRA 值得考虑；如果下游任务需要“持续生成文本”，ELECTRA 通常不是主方案。

---

## 参考资料

1. [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://research.google/pubs/electra-pre-training-text-encoders-as-discriminators-rather-than-generators/)
2. [ELECTRA PDF](https://openreview.net/attachment?id=r1xMH1BtvB&name=original_pdf)
3. [OpenReview 论文页](https://openreview.net/forum?id=r1xMH1BtvB)
4. [google-research/electra](https://github.com/google-research/electra)
