## 核心结论

多语言大模型的核心不是“会很多种语言”这件事本身，而是**怎样让不同语言在同一套表示空间里共享知识**。表示空间可以理解为模型内部存放语义关系的坐标系。词表设计决定哪些语言能被切成合适的子词，参数共享决定这些子词能否在同一套网络里互相借力。

最常见的做法是：**统一词表 + 统一 Transformer 编码器 + 多语言混合训练**。mBERT 和 XLM-R 都属于这一路线。它们把多种语言映射到同一套 token 序列，再由同一个编码器学习上下文表示，因此一个语言学到的模式，可能迁移到另一个语言。

但“统一”不等于“公平”。高资源语言通常文本更多、分词更稳定、脚本更常见，所以更容易占据词表容量，也更容易主导参数更新。结果是低资源语言、形态变化复杂语言、非拉丁脚本语言，往往被切得更碎，训练中也更难得到足够更新。芬兰语在 mBERT 中平均需要约为英语 2.4 倍的 subword，就是典型例子。

下面这张表先给出最重要的对比：

| 模型 | 语种数 | 词表 | 分词方法 | 训练语料 | 共享方式 |
|---|---:|---:|---|---|---|
| mBERT | 104 | 约110k | WordPiece | Wikipedia 多语语料 | 几乎全参数共享 |
| XLM-R | 100 | 250k | SentencePiece | 约2.5TB CommonCrawl | 几乎全参数共享 |

对新手可以直接这样理解：mBERT 相当于让 104 种语言共用一本 11 万词条的字典和同一套模型；XLM-R 把字典扩到 25 万，并换了更适合多脚本场景的切词方法，所以跨语言覆盖通常更好。

---

## 问题定义与边界

问题的本质是：**在词表有限、参数有限、训练预算有限的条件下，怎样同时服务高资源语言和低资源语言**。

这里有三个边界要先说清楚。

第一，**词表不是越统一越好**。词表就是模型把文本切成最小训练单元的“字典”。如果一个语言在词表里覆盖不足，同一句话会被切成更多、更短、更碎的 token。碎片越多，模型在固定上下文窗口内能容纳的有效语义越少。

玩具例子：

- 英语词 `unbelievable` 可能被切成 `un + believable`
- 芬兰语某个带丰富词尾变化的词，可能被切成 5 到 8 个碎片

如果两句话表达同等信息，但一种语言要占用更多 token，那么模型看到的有效上下文长度就缩短了。对初级读者来说，可以把它理解成：本来一页纸能记下 20 个词，现在因为切得太碎，只能记下 8 个完整意思。

第二，**参数共享不是越彻底越好**。参数共享就是不同语言共用同一套神经网络权重。好处是知识迁移，坏处是负迁移。负迁移指一个语言的训练更新，反而干扰另一个语言的表示学习。

第三，**训练采样必须矫正语言分布不均**。如果直接按原始语料比例训练，英语、法语、西班牙语这类高资源语言会压倒性地主导训练。工程上通常采用平滑采样：

$$
p'_i=\frac{p_i^\alpha}{\sum_j p_j^\alpha}, \quad \alpha<1
$$

其中 $p_i$ 是第 $i$ 种语言原始语料占比，$p'_i$ 是调整后的采样概率。$\alpha<1$ 的作用是“压低大头，抬高小头”。例如英语原本占 80%，低资源语言占 20%，经过平滑后，低资源语言会被更频繁地抽到。

所以本文的边界是明确的：讨论的是**多语言预训练阶段的词表设计与参数共享策略**，重点关注 mBERT、XLM-R 这类共享主干模型，以及 adapter 这类轻量局部适配方案；不展开讨论纯机器翻译架构、语音模型或完全独立单语模型。

---

## 核心机制与推导

多语言能力为什么会出现，核心靠两层机制叠加。

第一层是**子词共享**。子词可以理解为“比单词更小、比字符更有语义”的切分单位。英语、德语、法语之间常有形态或词根重叠；即使是不同脚本语言，也可能通过数字、符号、命名实体模式或相似上下文获得间接对齐。统一词表使这些模式有机会落到相同或相近的表示上。

第二层是 **Transformer attention**。attention 可以理解为“每个 token 在当前句子里决定该关注谁”的机制。公式是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

这里 $Q$、$K$、$V$ 分别是查询、键、值矩阵。白话说：每个 token 都会和其他 token 计算相关性，再按权重汇总信息。只要不同语言的 token 被送入同一编码器，它们就在同一矩阵空间里参与更新。

可以用“圆桌会议”理解这个过程。不同语言的 token 都坐在一张桌子上，attention 决定每个人该听谁。英语里“capital”和法语里“capitale”如果经常处在类似上下文，它们的表示就可能逐步靠近。模型不需要显式字典，也能学出跨语种对应关系。

再往下推一步，词表与采样共同决定“谁更有资格上桌”。

- 词表决定一个语言进入模型前会被切成多少块
- 采样决定训练时这个语言出现多少次
- 参数共享决定这些更新会不会被其他语言复用

三者的耦合关系可以概括为：

$$
\text{跨语种迁移效果} \approx f(\text{词表覆盖}, \text{采样平衡}, \text{共享参数容量})
$$

这不是严格数学定理，而是工程上非常稳定的经验关系。

玩具例子可以更直观。假设只有英语和西班牙语两种语言，任务是判断一句话是不是“天气相关”。

- 英语训练样本里经常出现 `rain`, `cloud`, `sunny`
- 西班牙语样本里经常出现 `lluvia`, `nube`, `soleado`

如果模型共享编码器，而这些词又经常出现在类似句式中，如“today is ...”“hoy está ...”，attention 会把相关上下文模式编码到相似区域。这样即使西班牙语标注更少，也可能借到英语任务知识。

真实工程例子更接近 XLM-R：做跨语言文本分类时，常见流程是只用英语标注训练，再直接测试德语、法语、印地语。之所以能零样本迁移，不是因为模型“懂翻译”，而是因为它在预训练阶段已经通过共享词表、共享上下文建模、共享参数，学到一部分跨语言结构。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，演示两件事：

1. 如何按 $\alpha$ 平滑多语言采样概率  
2. 如何在“冻结主干，只训练 adapter”时控制可训练参数

```python
from math import isclose

def normalize(xs):
    s = sum(xs)
    assert s > 0
    return [x / s for x in xs]

def smoothed_sampling(probs, alpha):
    assert 0 < alpha <= 1
    adjusted = [p ** alpha for p in probs]
    return normalize(adjusted)

def count_trainable(params):
    return sum(v["size"] for v in params.values() if v["trainable"])

# 原始语料分布：英语 80%，低资源语言 20%
p = [0.8, 0.2]
p_prime = smoothed_sampling(p, alpha=0.5)

# 平滑后，低资源语言占比应上升
assert p_prime[1] > p[1]
assert isclose(sum(p_prime), 1.0, rel_tol=1e-9)

# 一个极简“主干 + adapter”参数表
params = {
    "embeddings": {"size": 1000, "trainable": False},
    "encoder.layer1": {"size": 4000, "trainable": False},
    "encoder.layer2": {"size": 4000, "trainable": False},
    "adapter.layer1": {"size": 200, "trainable": True},
    "adapter.layer2": {"size": 200, "trainable": True},
    "task_head": {"size": 100, "trainable": True},
}

trainable = count_trainable(params)
total = sum(v["size"] for v in params.values())
ratio = trainable / total

# 只训练少量参数
assert ratio < 0.1
print("smoothed probs =", p_prime)
print("trainable ratio =", round(ratio, 4))
```

把它翻译成训练逻辑，结构大致如下：

```python
def train_loop(dataloaders, alpha, encoder, adapters, mlm_loss):
    raw_probs = [loader.data_ratio for loader in dataloaders]
    p_prime = normalize([p ** alpha for p in raw_probs])

    freeze(encoder)          # 主干冻结
    unfreeze(adapters)       # 只训练 adapter

    for batch in multilingual_sampler(dataloaders, p_prime):
        hidden = encoder(batch.input_ids)
        hidden = adapters(batch.lang_id, hidden)
        logits = hidden_to_vocab(hidden)
        loss = mlm_loss(logits, batch.labels)
        loss.backward()
        step_optimizer()
```

这里有三个关键点。

第一，**同一个 tokenizer** 负责所有语言的切分。这样不同语言都进入统一 token 空间。

第二，**同一个 encoder** 负责建模。这样不同语言的 token 都在同一组注意力层中交互。

第三，**adapter 按语言或语言家族区分**。adapter 可以理解为插在大模型主干里的小型可训练模块。主干不动，adapter 单独学习局部偏置，从而减少冲突。

真实工程例子是低资源翻译或分类迁移。比如做土耳其语到英语的低资源翻译，先加载 XLM-R 或 mBERT 主干，再插入 bottleneck adapter 或 LoRA。训练时只更新约 5% 左右参数。这样做的实际价值有三个：

- 显存更低
- 不同语言可以挂不同 adapter，部署灵活
- 不会轻易破坏共享主干里已有的跨语言能力

---

## 工程权衡与常见坑

多语言模型最常见的问题，不是“完全不会”，而是“会得不均匀”。

| 坑点 | 现象 | 根因 | 常见对策 |
|---|---|---|---|
| 漏词率高 | 一个词被切成很多碎片 | 词表对该语言覆盖不足 | 扩大词表，改用 SentencePiece |
| 上下文被压缩 | 同长度句子占更多 token | 子词过细 | 增大上下文窗口，优化词表分配 |
| 高资源偏置 | 英语效果显著更好 | 采样按原始频率过度失衡 | 用 $\alpha<1$ 做平滑采样 |
| 负迁移 | 某语言微调后另一语言下降 | 参数完全共享导致冲突 | 插入 adapter，按语言家族拆分 |
| 无意义 token 过多 | 出现很多 `##a` 之类片段 | 分词规则不适配低资源语言 | 用无监督子词方法并重训词表 |

先看词表问题。mBERT 使用 WordPiece，在多脚本、多形态语言下不是最理想选择。WordPiece 对英语等语言常常足够，但对土耳其语、芬兰语、匈牙利语这类形态变化丰富语言，容易把一个词拆成很多小段。这会直接带来两个工程后果：

- 训练和推理成本上升，因为 token 数变多了
- 有效语义密度下降，因为很多片段本身几乎不携带独立语义

再看参数共享问题。完全共享主干的好处是简单、统一、迁移强；缺点是所有语言都竞争同一组表示容量。语言差异越大，冲突越明显。尤其在任务微调阶段，如果直接全量更新，容易把原本共享好的跨语言结构破坏掉。

一个实用经验是：**预训练阶段尽量共享，适配阶段适度分化**。这也是 adapter 路线流行的原因。MAD-X、language-family adapter 这类方法，本质上是在共享主干外，再增加一层“局部可学习偏置”。

对初学者可以这样理解：大楼主体还是同一栋，但不同语言在楼里有自己的小隔间。这样既共享公共设施，又避免所有人挤一个房间。

---

## 替代方案与适用边界

如果统一词表 + 完全共享参数效果不够好，常见替代路线主要有三种。

第一种是**扩大词表并改进分词算法**。这正是 XLM-R 相比 mBERT 的关键升级。SentencePiece 更适合直接在原始文本上学习子词，对空格依赖更少，也更方便覆盖多脚本。词表从 11 万扩到 25 万，等于给低资源和非拉丁脚本语言更多“词槽”。

第二种是**按语言家族加 adapter**。语言家族可以理解为有较近历史或结构关系的一组语言，如罗曼语族、日耳曼语族。一个简单策略如下：

| 语言家族 | 示例语言 | adapter 策略 |
|---|---|---|
| 日耳曼语族 | 英语、德语、荷兰语 | 共用一组 family adapter |
| 罗曼语族 | 法语、西班牙语、意大利语 | 共用一组 family adapter |
| 突厥语族 | 土耳其语、阿塞拜疆语 | 单独或小范围共享 adapter |
| 汉藏语系 | 中文 | 单独 adapter 更常见 |

这种方案适合语言间结构相近、但又不希望完全混在一起的场景。

第三种是**任务专用 adapter 或 LoRA**。LoRA 可以理解为“不给原矩阵整体重写，只学习一个低秩增量”。它更省参数，适合低资源任务快速适配。对于小团队或部署资源有限的工程环境，这通常比全量微调更实际。

这些替代方案也有边界。

- 如果语种很少、每种语言数据都很多，独立模型可能更简单直接。
- 如果目标是极强的跨语种零样本能力，共享主干仍然是主路线。
- 如果任务非常垂直，比如只做中英法律检索，过度追求 100 语种覆盖反而浪费容量。
- 如果低资源语言脚本极特殊，先补语料和规范化编码，往往比盲目调 adapter 更有效。

所以不存在“唯一正确方案”。更合理的判断标准是：

$$
\text{方案选择} = \text{语言覆盖需求} + \text{资源预算} + \text{迁移目标} + \text{部署约束}
$$

面向初级工程师，最稳妥的工程建议是：

1. 先用共享主干模型作为底座  
2. 检查目标语言 token 长度、漏词率、采样比例  
3. 如果发现低资源语言明显吃亏，再引入 SentencePiece 重词表、采样平滑或 adapter 分化

---

## 参考资料

- [mBERT Overview, EmergentMind](https://www.emergentmind.com/topics/multilingual-bert-mbert?utm_source=openai)
  用途：mBERT 的共享词表、MLM/NSP 训练方式、多语言采样背景。

- [XLM-R, EmergentMind](https://www.emergentmind.com/topics/xlm-r?utm_source=openai)
  用途：XLM-R 的 250k SentencePiece 词表、100 语种、2.5TB 训练语料概览。

- [Low-Resource Translation with Parameter-Efficient Fine-Tuning, Mathematics (MDPI) 2024](https://www.mdpi.com/2227-7390/12/19/3149?utm_source=openai)
  用途：adapter/LoRA 在低资源翻译中的参数效率与效果对比。

- [Language Adapters and Multilingual Transfer, Electronics (MDPI)](https://www.mdpi.com/2079-9292/12/4/1022?utm_source=openai)
  用途：语言家族 adapter、局部参数隔离与负迁移缓解。

- [A Study on Subword Fragmentation in Multilingual Models, SCIRP](https://www.scirp.org/journal/doi.aspx?doi=10.4236%2Fjcc.2025.137002&utm_source=openai)
  用途：芬兰语等语言在共享词表下 subword 数偏高的问题示例。
