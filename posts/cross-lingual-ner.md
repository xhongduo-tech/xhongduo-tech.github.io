## 核心结论

跨语言 NER（Named Entity Recognition，命名实体识别，意思是把文本里的人名、地名、机构名等片段找出来并分类）要解决的核心问题，不是“目标语言会不会分词”，而是“高资源语言学到的实体边界与类别知识，能不能迁移到低资源语言”。

在零样本迁移（zero-shot，意思是目标语言没有任何人工标注样本）的设定下，多语言预训练模型是第一层基础。mBERT、XLM-R 这类模型通过多语言 MLM 预训练，把多种语言压到一个共享表示空间里。这个共享空间不保证所有语言天然对齐，但已经足够让英文里学到的一部分实体知识，直接迁移到阿姆哈拉语、斯瓦希里语、约鲁巴语这类低资源语言。公开结果里，XLM-R 在 WikiANN 一类跨语言 NER 评测上的平均 F1 常见落在 $78.2 \sim 78.5$ 附近，说明“只靠共享表示”已经不是不可用方案，而是强基线。

但工程上只靠零样本通常不够稳。要进一步提升低资源语言效果，主流做法是三类能力叠加：

| 策略 | 解决什么问题 | 典型收益 |
|---|---|---|
| 多语言预训练 | 提供跨语言共享表示 | 给零样本迁移打底 |
| 翻译-标注-投影 | 把高资源语言标注器借给目标语言 | 通常比纯零样本更稳 |
| 表示对齐/对抗训练 | 缩小不同语言的表示偏移 | 改善实体边界与类别判定 |

一个新手也能理解的玩具例子是：英文模型知道 “President Obama visited Mali” 里 `Obama` 是 PER、`Mali` 是 LOC。只要班图语句子经过模型后，`Obama` 和 `Mali` 附近的表示与英文句子里对应片段足够接近，分类头就有机会在没见过班图语标签的前提下，仍然打出正确实体类型。这就是“共享表示”和“对齐”的价值。

结论可以压缩成一句话：跨语言 NER 的主线不是重新为每种语言做标注，而是尽量把“高资源语言的监督信号”通过预训练、翻译投影和表示对齐，稳定地搬运到低资源语言。

---

## 问题定义与边界

问题定义很明确：给定高资源语言的标注数据，比如英语 NER 训练集，目标是在目标语言几乎没有标注、甚至完全没有标注的情况下，识别人名、地名、组织名等实体。

这里先把边界说清楚。跨语言 NER 不是机器翻译，不要求把整句翻对；也不是词典匹配，不是靠手工维护几万个实体表；更不是知识图谱补全，它只负责从原始文本里抽取实体 span 和类型。

下面这张表可以把资源边界看得更清楚：

| 资源维度 | 已知/可用 | 不确定性 |
|---|---|---|
| 训练语种 | 英语等高资源语言通常可用 | 目标语言可能无标注 |
| 标签体系 | PER / LOC / ORG / MISC 等一般已知 | 不同数据集标签边界不完全一致 |
| 多语言模型 | mBERT / XLM-R 可直接加载 | 目标语言在预训练中覆盖程度不同 |
| 翻译能力 | 有时可调用 NMT 或并行语料 | 低资源语言翻译质量可能很差 |
| 对齐模块 | 可加对抗损失或一致性损失 | 调参成本高，训练不稳定 |
| 推理目标 | 在目标语言直接输出实体标签 | 实体边界、词序、形态变化都会带来噪声 |

新手版理解可以这样说：你只有英文标签，但想在阿姆哈拉语里找人名。你手里没有阿姆哈拉语标注员，也不想从头训练一个新模型，所以只能借助“懂多国语言的共享编码器”以及“翻译或对齐”把英文监督信号转过去。

这件事有三个前提边界。

第一，标签体系要基本兼容。如果英文训练集把政府机构和商业机构都算 ORG，而目标语数据希望再细分，那迁移会天然受限。

第二，目标语言最好至少在多语言预训练中出现过，或者与出现过的语言有较强共享模式。否则模型连基本词形、子词分布都不熟，零样本效果会明显掉。

第三，跨语言 NER 更擅长迁移“实体边界”和“通用类别”，不擅长解决强本地文化实体、缩写歧义、领域新词这类问题。比如地方政党简称、地区俚语、拼写混合文本，往往需要额外领域适配。

---

## 核心机制与推导

跨语言迁移的第一层机制是多语言 MLM（Masked Language Modeling，掩码语言模型，意思是随机遮住一些词，让模型根据上下文猜回来）。它的基本损失是：

$$
\mathcal{L}_{MLM}=-\sum_{i\in M}\log P(x_i|\mathbf{x}_{\backslash M};\theta)
$$

这里 $M$ 是被遮住的位置集合，$\mathbf{x}_{\backslash M}$ 表示其余上下文，$\theta$ 是模型参数。这个目标本身不带实体标签，但它让模型学到一件关键事实：不同语言里相似的上下文结构，会在表示空间里形成可迁移的局部模式。

为什么这对 NER 有帮助？因为实体识别本质上是上下文判别任务。比如英文句子里 “President of Mali addressed the summit” 中，`Mali` 前后上下文提示它更像国家或地点；`President` 后面接人名时，更容易触发 PER 边界。只要模型在多语言里学到了类似模式，它就能把这种边界判断迁移出去。

第二层机制是有监督的 NER 微调。通常在英语训练集上最小化：

$$
\mathcal{L}_{NER}=-\sum_{t=1}^{T}\log P(y_t|\mathbf{h}_t)
$$

其中 $\mathbf{h}_t$ 是第 $t$ 个 token 的表示，$y_t$ 是它的实体标签。这个阶段让模型不再只是“会读句子”，而是开始学习“哪段文本应该打什么实体标签”。

第三层机制是跨语言对齐。最常见的写法是把总损失写成：

$$
\mathcal{L}=\mathcal{L}_{NER}+\lambda \mathcal{L}_{align}
$$

其中 $\lambda$ 控制对齐强度，$\mathcal{L}_{align}$ 可以是对比损失、分布匹配损失，也可以是对抗损失。直观解释是：英语里表示“总统”“国家名”“峰会”这些上下文模式的向量分布，应该尽量和目标语言中的对应模式靠近。这样分类头在英语上学到的决策边界，才更可能在目标语言上继续可用。

玩具例子可以写得很具体：

- 英文源句：`President of Mali addressed the summit.`
- 英文标签：`President/O`, `of/O`, `Mali/B-LOC`, ...
- 目标语言句子：假设翻译后表达为“马里的总统在峰会上发言”
- 如果没有对齐，模型可能只把“首字母大写 + 英文词形”当作 LOC 线索，迁移后马上失效
- 如果有对齐，模型会更多依赖“国家名出现在介词后、与政治事件共同出现”的上下文模式，从而在目标语言中继续打出 LOC

对抗训练（adversarial training，意思是让一个判别器尽量分辨“这段表示来自哪种语言”，而编码器反过来尽量骗过它）就是一种常见实现。它的目的不是让不同语言完全一样，而是让“与 NER 无关的语言身份差异”尽量被抹平，让“与实体边界相关的语义结构”保留下来。

所以从推导上看，跨语言 NER 并不是神秘技巧，而是三步叠加：

1. 用 $\mathcal{L}_{MLM}$ 学会多语言上下文建模。
2. 用 $\mathcal{L}_{NER}$ 学会实体判别。
3. 用 $\mathcal{L}_{align}$ 把高资源语言的决策边界推向低资源语言。

---

## 代码实现

工程上最常见的可落地流程是“两段训练 + 一段伪标注”：

1. 用英语标注数据微调 XLM-R。
2. 把目标语言句子翻译到英语，由英文 NER 模型打标签。
3. 把标签投影回目标语言，再用伪标签继续训练或直接推理。

下面这个 `python` 代码块不是完整训练脚本，但能运行，演示“标签投影”这个核心动作。标签投影的意思是：把源语言实体标签映射回目标语言 token 位置。

```python
from typing import List, Tuple

def project_bio_labels(
    src_tokens: List[str],
    src_labels: List[str],
    tgt_tokens: List[str],
    alignment: List[Tuple[int, int]],
) -> List[str]:
    """
    alignment: (src_idx, tgt_idx) 列表，表示源 token 与目标 token 的对齐关系
    """
    assert len(src_tokens) == len(src_labels)
    tgt_labels = ["O"] * len(tgt_tokens)

    for src_idx, tgt_idx in alignment:
        label = src_labels[src_idx]
        if label == "O":
            continue

        prefix, ent_type = label.split("-", 1)
        if tgt_labels[tgt_idx] != "O":
            continue

        if prefix == "B":
            tgt_labels[tgt_idx] = f"B-{ent_type}"
        else:
            tgt_labels[tgt_idx] = f"I-{ent_type}"

    # 修正非法 BIO：I-xxx 不能直接出现在句首或跟在别的类型后面
    for i, label in enumerate(tgt_labels):
        if label.startswith("I-"):
            ent_type = label.split("-", 1)[1]
            if i == 0 or tgt_labels[i - 1] not in {f"B-{ent_type}", f"I-{ent_type}"}:
                tgt_labels[i] = f"B-{ent_type}"

    return tgt_labels


src_tokens = ["President", "of", "Mali", "addressed", "the", "summit"]
src_labels = ["O", "O", "B-LOC", "O", "O", "O"]

tgt_tokens = ["总统", "在", "峰会", "上", "谈到", "马里"]
alignment = [(2, 5)]  # Mali -> 马里

tgt_labels = project_bio_labels(src_tokens, src_labels, tgt_tokens, alignment)
assert tgt_labels == ["O", "O", "O", "O", "O", "B-LOC"]
print(tgt_labels)
```

真正项目里，流程通常是这样：

```python
# 1. 加载多语言编码器，如 xlm-roberta-base
# 2. 用英语标注数据训练 token classification 头
# 3. 对目标语言无标签文本做机器翻译：target -> english
# 4. 用英文 NER 模型预测英文标签
# 5. 用词对齐或序列翻译模型把标签投影回 target
# 6. 将 target 伪标签数据与原英语真标签数据混合训练
# 7. 可选：加入 language discriminator 做对抗对齐
# 8. 在目标语言原文上直接 inference
```

如果用 Hugging Face，新手可以把任务拆成三个对象：

- `AutoTokenizer` 负责子词切分
- `AutoModelForTokenClassification` 负责 NER 预测
- `Trainer` 负责英语监督训练

真实工程例子是非洲多语新闻知识抽取。假设平台每天接入英语、豪萨语、约鲁巴语、阿姆哈拉语新闻，目标是抽取“人物-机构-地点”实体供知识图谱入库。做法通常不是等每种语言都标满，而是先用英语新闻训练高质量 NER，再把低资源语言新闻翻到英语做标注，最后通过投影与融合生成目标语言伪标签。像 TransFusion 一类方法在 MasakhaNER2.0、LORELEI 等低资源场景上，相比单纯英语微调再直接零样本迁移，能带来最高约 +16 F1 的提升，说明额外的翻译与融合成本在生产上通常是值得的。

---

## 工程权衡与常见坑

理论上三条路都可行，但上线时真正会拖垮效果的，往往不是模型大小，而是噪声路径。

| 坑 | 原因 | 缓解策略 |
|---|---|---|
| 标签投影错位 | 翻译后词序变化、实体拆分或合并 | 使用词对齐、序列级投影、BIO 修正 |
| 翻译把实体改写 | 人名音译、地名本地化、缩写展开 | 保留原文 span，做回译一致性检查 |
| 子词切分破坏边界 | XLM-R 的 BPE 可能把稀有词切碎 | 训练时按首子词对齐标签，推理时做 span 合并 |
| 语言分布偏移 | 目标语言在预训练里样本少 | 加领域继续预训练或对抗对齐 |
| 标签体系不一致 | 不同数据集对 ORG/MISC 定义不同 | 统一 schema，必要时映射到粗粒度标签 |
| 伪标签污染训练 | 错误标签被模型反复强化 | 置信度过滤、教师-学生训练、分阶段混训 |

几个典型坑要单独展开。

第一，翻译噪声不是“偶尔错一个词”这么简单。它会直接破坏实体 span。比如原句里的 `Mali` 被翻回目标语言后仍然写成 `Mali`，但对齐器把它错连到前一个功能词，最后标签就落到错误 token 上。这个问题不能只靠更大的编码器解决，往往要靠回译检查、词边界对齐或者序列级翻译投影来处理。

第二，零样本效果看起来高，不代表长尾实体稳。很多公开基准的文本风格较规范，实体形式也相对标准。真实日志、社媒、混合语言文本里的拼写变体更多，边界更乱，纯 XLM-R 往往会掉点。

第三，对抗训练会增加训练不稳定性。对抗训练的白话解释是“让编码器故意学到不那么像某种具体语言的表示”。问题在于，如果对抗强度过大，模型会把对 NER 有用的局部语言特征也一起抹掉，最后边界反而更差。所以 $\lambda$ 和判别器强度不能凭感觉设。

第四，翻译加融合有明显工程成本。你要维护翻译 API、对齐模块、伪标签过滤逻辑，还要考虑延迟与成本。如果业务只覆盖两三种语言、而且这些语言和英语很接近，直接零样本加少量人工校正可能更划算。

---

## 替代方案与适用边界

跨语言 NER 没有单一最优解，不同语言资源条件下，方案选择应当不同。

| 方案 | 需要什么 | 优势 | 劣势 | 典型语境 |
|---|---|---|---|---|
| 纯零样本迁移 | 多语言预训练模型 + 高资源语言标签 | 实现最简单、成本最低 | 对低资源长尾语言不稳 | 快速原型、基线系统 |
| 翻译投影 | 可用翻译系统或并行语料 | 能直接借高资源 NER 能力 | 受翻译质量强约束 | 语法相近、翻译资源较足 |
| 表示对齐/对抗训练 | 无标签目标语料 + 对齐模块 | 不依赖高质量翻译 | 训练复杂、调参难 | 无稳定翻译 API 的语言 |
| 翻译+融合（CROP/TransFusion） | 翻译、投影、融合训练 | 对翻译噪声更鲁棒，通常更强 | 工程链路最长 | 多语新闻、信息抽取平台 |
| 少量目标语监督 + 迁移 | 少量人工标注 | 往往是性价比最高的最终方案 | 需要人工标注流程 | 准生产或生产环境 |

可以把选择逻辑说得更直接。

如果翻译质量足够高，优先考虑翻译投影或翻译+融合。因为这类方法本质上是在“把目标语言问题临时变成高资源语言问题”，通常能直接吃到成熟英文 NER 模型的能力。

如果翻译质量差，或者根本没有稳定翻译 API，那就更适合依赖 shared encoder（共享编码器）加 adversarial loss（对抗损失）。这种做法不要求句子翻得通顺，只要求不同语言的语义表示尽量靠近。

如果已经有少量目标语言标注，不要执着于“纯零样本”。工程上常见的最优策略往往是：多语言预训练 + 高资源迁移 + 少量目标语增量微调。几十到几百条高质量标注，常常比复杂的对齐模块更值钱。

CROP、TransFusion 这类方案适合“有一定翻译能力，且愿意为效果多付一段训练链路”的场景。它们的核心不是简单回译，而是把翻译、标签传递、序列融合串成一个更稳的管道，减少投影误差放大。

---

## 参考资料

1. Chen, Nuo, et al. *Bridging the Gap between Language Models and Cross-Lingual Sequence Labeling*. NAACL 2022. 作用：给出跨语言序列标注的预训练改进思路，并报告了 XLM-R 等模型在 WikiANN 等任务上的强基线。  
   https://aclanthology.org/2022.naacl-main.139/

2. Yang, Jian, et al. *CROP: Zero-shot Cross-lingual Named Entity Recognition with Multilingual Labeled Sequence Translation*. Findings of EMNLP 2022. 作用：翻译-标注-回投影的代表性工程方案。  
   https://aclanthology.org/2022.findings-emnlp.34/

3. Zhao, Yichun, et al. *TransAdv: A Translation-based Adversarial Learning Framework for Zero-Resource Cross-Lingual Named Entity Recognition*. Findings of EMNLP 2022. 作用：把翻译法与对抗学习结合，缓解翻译噪声。  
   https://aclanthology.org/2022.findings-emnlp.52/

4. Chen, Yang, Vedaant Shah, and Alan Ritter. *Better Low-Resource Entity Recognition Through Translation and Annotation Fusion*. arXiv 2023. 作用：提出 TransFusion，说明翻译与标注融合在 25 种低资源语言上的工程价值。  
   https://arxiv.org/abs/2305.13582

5. Chai, Yuan, Yaobo Liang, and Nan Duan. *Cross-Lingual Ability of Multilingual Masked Language Models: A Study of Language Structure*. ACL 2022. 作用：解释为什么多语言 MLM 会产生跨语言迁移能力。  
   https://aclanthology.org/2022.acl-long.322/

阅读这些论文时，重点看三处：标签如何投影、对齐损失如何定义、实验里到底比较的是“纯零样本”“翻译投影”还是“翻译后再融合”。很多看起来类似的方法，差别就出在这三步。
