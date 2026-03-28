## 核心结论

分词器的“好坏”首先不是算法名字决定的，而是训练数据分布决定的。这里的训练数据分布，白话说就是“分词器在学词表时，平时最常见到哪些语言、哪些写法、哪些领域术语”。如果训练语料几乎全是英文，那么 BPE 或 SentencePiece 会优先把英文高频片段合并成短 token；同样一段中文放进去时，因为中文片段在训练里不常见，就只能被拆成更细的单位，甚至退化到 byte-level 表示，一个汉字可能占 1 到 3 个 token。

这会直接改变模型成本。序列更长，注意力计算更贵。自注意力的核心成本近似与序列长度平方成正比，即 $O(n^2)$。如果中文因为词表不适配而从 100 个 token 拉长到 250 个 token，注意力相关计算量不是增加 2.5 倍，而是接近增加到 $2.5^2=6.25$ 倍。

词表大小本身也是一个杠杆。词表太小，比如低于 16K，覆盖不足，只能靠更细碎的子词或字节编码补齐，结果是 token 数暴涨。词表太大，比如超过 200K，又会把嵌入矩阵和输出层参数推高，很多低频 token 长期见不到足够样本，训练不到位。工程上常见的中间带是 50K 到 150K，适合先做第一轮权衡。

一个典型例子是多语言词表。LLaMA 3 将词表从 32K 扩到 128K，目标之一就是改善多语言压缩率。直观理解是：它“多记住了更多常见片段”，因此中文、代码、符号和多语文本都更容易被压成更短序列。对中文来说，这通常意味着更少的 token、更低的推理成本、更稳定的上下文利用率。

| 词表方案 | 训练语料倾向 | 中文平均 token 压缩效果 | 主要问题 |
| --- | --- | --- | --- |
| 英文单语 32K | 英文高频片段 | 较差，常被拆细 | 中文序列长，注意力成本高 |
| 多语 128K | 多语言平衡 | 明显更好 | 嵌入更大，训练更贵 |
| 领域扩展词表 | 通用语料 + 领域子词 | 对目标域更好 | 需要维护新增 token 与嵌入 |

---

## 问题定义与边界

这里讨论的“分词器训练数据影响”，核心不是模型训练数据，而是词表学习阶段的数据分布如何决定 token 切分结果。分词器词表，白话说就是“模型提前记住的可复用文本片段清单”。哪个片段能进词表，主要取决于它在训练语料里出现得够不够频繁。

边界要先说清楚。

第一，我们讨论的是子词分词器，主要包括 BPE、SentencePiece 这一类。它们不是按“完整单词字典”工作，而是从字符或字节开始，不断把高频共现片段合并成更长的 token。英文因为空格和词形稳定，容易学出高频整词或常见词干；中文没有空格，是否能把“机器学习”“神经网络”这类连续片段学成单 token，更依赖训练语料里这些组合是否足够多。

第二，我们讨论的是“覆盖质量”，不是“绝对能否编码”。byte-level tokenizer 理论上几乎什么文本都能编码，但“能编码”不等于“编码得好”。如果一句中文能被编码，但每个汉字平均拆成 2.5 个 token，那这个分词器对中文的覆盖质量就很差。这里的覆盖质量更接近“压缩质量”和“语义保持质量”。

第三，评价不能脱离目标场景。一个只服务英文搜索排序的小模型，不一定需要花成本把中文压缩得很好；但一个面向中文问答、多语对话、法律文档生成的模型，如果还在使用英文主导训练出来的小词表，后续的上下文窗口、延迟、显存、训练吞吐量都会被拖累。

看一个玩具例子。假设句子是：

“深度学习用于医学影像诊断”

如果分词器主要在英文语料上学词表，它可能把中文拆成非常细的片段，接近逐字甚至逐字节编码；而多语或中文覆盖更好的词表，可能直接学到“深度学习”“医学”“影像”“诊断”这样的高频子词。对新手来说，可以把这理解成两种查字典方式：

1. 一种字典只记住英文整词，看到中文时只能拆成更小碎片慢慢拼。
2. 另一种字典平时也学过中文常见词组，所以一眼能认出更大块的片段。

真实工程例子更直观。假设你要微调一个英文基础模型去处理中文医疗摘要：

“患者接受冠状动脉 CTA 检查后，提示轻度狭窄，建议随访。”

如果 tokenizer 对中文和医疗缩写都不友好，那么“冠状动脉”“狭窄”“随访”“CTA”都可能被拆得很散。结果是同一段病历文本在 token 空间里变长，训练 batch 变小，推理时上下文更早耗尽。这不是下游模型结构的问题，而是 tokenizer 在上游已经把语义单位切碎了。

| 训练语料类型 | 对同一句中文医学句子的常见表现 | 平均 fertility 趋势 | 工程影响 |
| --- | --- | --- | --- |
| 英文单语 | 中文多被拆成细碎 token | 高 | 序列变长，显存与时延上升 |
| 多语言通用 | 常见中文片段能合并 | 中 | 序列较短，通用性更好 |
| 医疗领域适配 | 医学术语更容易成整体 | 更低 | 训练与推理更稳定 |

---

## 核心机制与推导

分词器为什么会受训练语料影响，本质上是“频率驱动的合并规则”在起作用。

以 BPE 为例，它最开始只认识很小的基本单位，然后统计训练语料中哪些相邻片段经常一起出现，把高频组合优先合并。白话说，BPE 的学习过程像“把总是一起出现的字块焊接起来”。如果训练集中英文占主导，那么 `th`、`ing`、`tion`、`model` 之类的组合会被反复看见，于是优先进入词表。中文组合如果出现少，就很难得到同等待遇。

因此，训练语料分布决定了“哪个语言的常见片段更容易成为短 token”。这也是为什么英文训练出来的 byte-level BPE 处理中文时，往往表现得像“能读，但读得很碎”。

常见的两个量化指标是 fertility 和 NSL。

fertility 可以写成：

$$
\text{Fertility}(T, D)=\frac{\text{total tokens produced by }T\text{ on }D}{\text{total words or semantic units in }D}
$$

直观理解：每个语言单位平均被切成多少个 token。越低越好，说明分词器更会“整块识别”，而不是把文本砸碎。

归一化序列长度 NSL 可以写成：

$$
\text{NSL}(T_\lambda, T_\beta)=
\frac{\sum_i \text{len}(T_\lambda(D_i))}
{\sum_i \text{len}(T_\beta(D_i))}
$$

这里：

- $D_i$ 是第 $i$ 个样本。
- $T_\lambda$ 是你要评估的 tokenizer。
- $T_\beta$ 是对照 tokenizer。
- $\text{len}(T(D_i))$ 是样本经过分词后的 token 数。

如果 $\text{NSL}(T_\lambda, T_\beta)=1.8$，表示在同一批数据上，$T_\lambda$ 产生的序列长度是对照的 1.8 倍。

玩具例子可以这样看。设有两种 tokenizer：

- $T_{en}$：主要用英文训练。
- $T_{multi}$：用多语数据训练。

对于一组中文句子 $D$，若

$$
\sum_i \text{len}(T_{en}(D_i)) = 1800,\quad
\sum_i \text{len}(T_{multi}(D_i)) = 1000
$$

那么：

$$
\text{NSL}(T_{en}, T_{multi}) = \frac{1800}{1000}=1.8
$$

这表示英文词表在中文上会产生 1.8 倍序列长度。因为注意力成本大致看 $n^2$，所以相关计算可能接近增加到 $1.8^2=3.24$ 倍。

这也是“序列被拉长”为什么不只是多几个 token，而是会变成系统级成本问题。

再看题目里给出的 LLaMA 3 例子。词表从 32K 扩到 128K 后，平均每个 token 可表达的字符数从 3.17 提升到 3.94。这个数字可以反过来理解为压缩率改善了：

$$
\frac{3.94}{3.17}\approx 1.24
$$

即每个 token 能承载更多字符，约提升 24%。这不代表所有语言都等比例提升，但它说明“大词表 + 多语覆盖”确实能把更多高频片段吸收到词表里，从而降低 fertility。

真实工程里，这个机制还有一个重要推论：领域适配时，通常不需要重训整个 tokenizer。原因是通用词表已经覆盖了大多数基础文本，问题往往集中在少量高频专业术语，比如医疗里的“冠状动脉粥样硬化”、法律里的“不可抗力条款”、代码里的 API 名称。此时追加少量领域子词，往往比全部重训更划算。

---

## 代码实现

下面给出一个可运行的玩具实现。它不依赖外部分词库，用“最长匹配词表”模拟不同训练语料下的词表覆盖差异，再计算字符级 fertility。它不是生产级 tokenizer，但足够说明训练语料分布如何改变序列长度。

```python
from typing import List, Set

def greedy_tokenize(text: str, vocab: Set[str]) -> List[str]:
    tokens = []
    i = 0
    max_len = max(len(x) for x in vocab) if vocab else 1

    while i < len(text):
        matched = None
        window = min(max_len, len(text) - i)
        for size in range(window, 0, -1):
            piece = text[i:i+size]
            if piece in vocab:
                matched = piece
                break
        if matched is None:
            matched = text[i]
        tokens.append(matched)
        i += len(matched)
    return tokens

def fertility_per_char(text: str, vocab: Set[str]) -> float:
    tokens = greedy_tokenize(text, vocab)
    non_space_chars = len([c for c in text if not c.isspace()])
    return len(tokens) / non_space_chars

# 英文主导词表：几乎不认识中文多字片段
vocab_en = {
    "deep", "learning", "model", "medical", "image",
    "诊", "断", "医", "学", "影", "像", "深", "度", "学", "习"
}

# 多语词表：学到了中文常见片段
vocab_multi = {
    "deep", "learning", "model", "medical", "image",
    "深度学习", "医学", "影像", "诊断", "用于"
}

text = "深度学习用于医学影像诊断"

tokens_en = greedy_tokenize(text, vocab_en)
tokens_multi = greedy_tokenize(text, vocab_multi)

fert_en = fertility_per_char(text, vocab_en)
fert_multi = fertility_per_char(text, vocab_multi)

print(tokens_en)
print(tokens_multi)
print(fert_en, fert_multi)

assert len(tokens_multi) < len(tokens_en)
assert fert_multi < fert_en
assert greedy_tokenize("医学影像", vocab_multi) == ["医学", "影像"]
```

这段代码表达的是同一个事实：如果词表学到的是更大、更高频的中文片段，那么 token 数就会下降。

再给一个更接近真实工程的示例。下面是用 SentencePiece 训练多语子词模型的最小流程，适合理解工作步骤。代码假定你本地已安装 `sentencepiece`。

```python
import sentencepiece as spm
from pathlib import Path

# 1. 准备训练语料：可以混合中文、英文、领域术语
corpus_path = Path("toy_corpus.txt")
corpus_path.write_text(
    "深度学习用于医学影像诊断\n"
    "冠状动脉狭窄需要长期随访\n"
    "deep learning for medical imaging\n",
    encoding="utf-8"
)

# 2. 训练一个子词模型
spm.SentencePieceTrainer.Train(
    input=str(corpus_path),
    model_prefix="toy_sp",
    vocab_size=64,
    model_type="bpe",
    character_coverage=0.9995
)

# 3. 加载并统计 fertility
sp = spm.SentencePieceProcessor(model_file="toy_sp.model")
text = "深度学习用于医学影像诊断"

pieces = sp.encode(text, out_type=str)
fertility = len(pieces) / len(text)

print(pieces)
print(f"fertility={fertility:.3f}")

assert len(pieces) >= 1
assert fertility > 0
```

如果你要做领域适配，常见流程不是“推翻重来”，而是“保留原词表，追加高频领域子词”。下面是一个抽象化示意，展示 `VocabAugmentor` 类工具在做什么：

```python
base_vocab_size = 128000
domain_terms = ["冠状动脉", "粥样硬化", "CTA", "病理切片", "不可抗力条款"]

def extend_vocab(base_size: int, new_terms: list[str]) -> int:
    # 工程上通常会先去重、过滤过低频术语，再扩充词表
    unique_terms = list(dict.fromkeys(new_terms))
    return base_size + len(unique_terms)

new_vocab_size = extend_vocab(base_vocab_size, domain_terms)

print(base_vocab_size, new_vocab_size)

assert new_vocab_size == 128005
assert "CTA" in domain_terms
```

真实系统中，词表扩展通常伴随两件事：

1. tokenizer 词表末尾追加新 token；
2. 模型输入嵌入和输出层矩阵同步扩行。

这样做的意义是：保留原模型绝大多数语言能力，只为新领域补上高频缺口。

| 方案 | 词表大小变化 | 是否重训 tokenizer | 是否需要扩嵌入 | 适用场景 |
| --- | --- | --- | --- | --- |
| 从零重训多语词表 | 大 | 是 | 是 | 新预训练模型 |
| 通用词表直接使用 | 无 | 否 | 否 | 通用任务、快速上线 |
| 追加领域子词 | 小到中 | 否 | 是 | 医疗、法律、代码等域适配 |

---

## 工程权衡与常见坑

第一个权衡是词表大小与序列长度。

小词表的好处是嵌入矩阵小、softmax 维度小、部署简单；坏处是文本会被切得很碎。特别是中文、多语混合、术语密集文本，在小词表下经常出现 2 到 5 倍的序列膨胀。新手容易只看到“词表小，参数更少”，却忽略 token 长度一旦翻倍，自注意力相关成本会按平方放大。

第二个权衡是大词表与低频 token 训练不足。

词表大到 200K 以上后，很多 token 的出现频率会非常低。白话说，就是“字典条目记得太多，但很多词一年也看不到几次”。这些 token 的嵌入难以被充分优化，既占参数，也不一定带来稳定收益。对中小规模训练来说，这种膨胀尤其明显。

第三个常见坑是把“跨语言可编码”误当成“跨语言可高效使用”。

很多 byte-level tokenizer 对任何语言都不会 OOV，表面上很安全。但实际使用时，中文、日文、阿拉伯文如果没有在词表学习阶段得到足够覆盖，序列长度会显著劣化。不能只看“能不能切”，还要看切完以后是否仍然适合训练和推理。

第四个坑是领域适配时直接重训整个 tokenizer。

如果你已经有一个通用模型，突然要加法律或医疗任务，直接重训 tokenizer 往往会引入兼容性问题：旧模型的 embedding 对应不上新词表，下游已有数据缓存也会失效。更现实的路线通常是固定原词表，追加少量高频领域子词，然后做继续训练或微调。

真实工程例子可以这样看。一个中文法律文档系统原本复用英文基础模型的 tokenizer。合同条款平均从 2K 字被切成 5K 到 6K token，长上下文很快被耗尽，训练时 attention 矩阵尺寸接近原预期的 9 倍。后来团队没有重训整个 tokenizer，而是统计法律语料高频短语，增补了几千个领域子词。结果是平均序列长度明显回落，吞吐量和显存占用都恢复到更可控的范围。

工程上更稳妥的流程通常是：

1. 先选 50K 到 150K 的通用词表规模作为起点。
2. 在目标语料上测 fertility、平均序列长度、P95 序列长度。
3. 如果目标域序列明显偏长，再做小规模词表扩展。
4. 只有在“目标语言完全变了”或“从头预训练新模型”时，才考虑重训 tokenizer。

| 词表规模 | 序列长度表现 | 嵌入参数量 | 训练/推理内存 | 常见风险 |
| --- | --- | --- | --- | --- |
| 小于 16K | 长，常膨胀 2 到 5 倍 | 小 | 注意力成本高 | 文本被切碎 |
| 50K 到 150K | 通常较平衡 | 中 | 综合成本可控 | 需要按语种调优 |
| 大于 200K | 短一些 | 大 | 嵌入与输出层更重 | 低频 token 训练不足 |

---

## 替代方案与适用边界

第一种替代方案是字符级或 byte-level tokenizer。它的优点是覆盖最稳，不容易遇到未登录片段；缺点是序列长。适合资源很受限、实现要极简、或者文本类型变化极大的场景。比如移动端做轻量分类，模型本身很小，接受更长序列但不愿维护复杂词表，这时字符级方案可以成立。

第二种方案是标准通用 BPE 或 SentencePiece。它适合多数组合型任务，特别是你不想维护太多特化规则时。它的核心前提是：训练语料必须足够代表未来输入分布。如果未来大量处理中文、多语或代码，而训练时几乎没覆盖，这种“标准方案”就会退化。

第三种方案是词表扩展。它适合已经有通用基础模型，但新任务集中在某个专业领域的情况。它的本质不是换一套语言系统，而是在原有字典末尾补一批“高频新词”。这通常比全量重训成本低，也更容易保持原能力。

可以用一个直白比喻帮助初学者理解：

- 字符级方案像“每次都按字母拼写”，字典最小，但阅读最慢。
- 大而合适的 BPE 词表像“直接记住常用词和词组”，字典更大，但阅读更快。
- 词表扩展像“原本字典够用，只是针对新专业再补几页附录”。

真实工程上，移动端聊天摘要和医疗生成就是两类典型边界。

移动端场景通常延迟敏感、内存紧张，可能宁可接受小词表和更长序列，也不愿让 embedding 太大。医疗生成则相反，术语密度高、上下文长，token 膨胀会直接压缩可用上下文，所以更值得通过多语词表或领域扩展降低 fertility。

| 策略 | 序列长度 | 嵌入规模 | 领域适应能力 | 适用边界 |
| --- | --- | --- | --- | --- |
| 字符级 / Byte-level | 长 | 小 | 弱，需要模型自己学组合 | 极简部署、覆盖优先 |
| 标准 BPE / SentencePiece | 中 | 中 | 中，取决于训练语料 | 通用任务 |
| 词表扩展 | 较短 | 中到偏大 | 强 | 领域微调、专业术语密集 |
| 从零重训多语大词表 | 较短 | 大 | 很强 | 新模型预训练 |

结论可以压缩成一句话：如果你是在做新模型预训练，优先用代表目标分布的多语或目标域语料来学词表；如果你是在做现有模型的领域适配，优先考虑词表扩展，而不是重训整个 tokenizer。

---

## 参考资料

| 资料 | 贡献说明 |
| --- | --- |
| [lucven.com: Byte-level Tokenizer](https://lucven.com/posts/tokenization/byte-level-tokenizer/?utm_source=openai) | 说明 byte-level tokenizer 在非英文脚本上会把字符拆成多个 token，适合理解英文训练词表为何会拉长中文序列。 |
| [emergentmind.com: Multilingual SentencePiece/BPE Tokenizer](https://www.emergentmind.com/topics/multilingual-sentencepiece-bpe-tokenizer?utm_source=openai) | 给出 multilingual tokenizer 设计背景，并提供 fertility 与 NSL 的定义，适合建立量化评估框架。 |
| [jaimeparker.github.io: LLaMA3](https://jaimeparker.github.io/tech/LLaMA3/?utm_source=openai) | 提供 LLaMA 3 词表扩展到 128K 的背景，可用来理解大词表在多语言压缩上的收益。 |
| [transformers-domain-adaptation.readthedocs.io](https://transformers-domain-adaptation.readthedocs.io/en/latest/content/domain_adaptation_components.html?utm_source=openai) | 展示领域适配中的词表扩展思路，说明为什么很多工程实践选择追加领域子词而不是完全重训。 |
| [mlsysbook.ai: Tokenization](https://mlsysbook.ai/tinytorch/modules/10_tokenization_ABOUT.html?utm_source=openai) | 总结词表大小与计算成本、嵌入规模、训练稳定性之间的工程权衡。 |
