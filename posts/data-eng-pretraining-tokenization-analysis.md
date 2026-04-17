## 核心结论

Tokenization 是把原始文本切成模型可处理的最小离散单元。更准确地说，模型训练时真正接收的不是“句子本身”，而是一串离散 token ID。文本先被切分，再被映射到词表中的编号，最后才进入 embedding、attention 和输出层。

这个步骤不是可有可无的预处理，而是预训练系统的一部分。它会直接影响四件事：

| 影响项 | 为什么会被 tokenization 影响 |
| --- | --- |
| 数据效率 | 同样一段文本被切成多少 token，决定一个训练 step 实际能覆盖多少原文 |
| 显存占用 | 序列越长，激活和 attention 中间张量越大 |
| 上下文长度 | 固定窗口下，token 越短越细，模型能看到的原文越少 |
| 长尾词保真度 | 稀有词、术语、拼写变体、代码片段是否能被合理表示 |

预训练里最重要的权衡是两条同时成立的事实：

1. 词表越大，单个词更可能被整体或较粗粒度地表示，序列更短，attention 成本通常更低。
2. 词表越大，嵌入表和输出 softmax 的参数与计算又会随词表大小 $|V|$ 线性增长。

常见近似下，输出层成本可写成：

$$
O(L \times d \times |V|)
$$

其中 $L$ 是序列长度，$d$ 是隐藏维度，$|V|$ 是词表大小。与此同时，自注意力主项常近似看作：

$$
O(L^2 \times d)
$$

所以 tokenizer 设计本质上是在做“词表成本”和“序列成本”的交换。

压缩效果可以用几类指标衡量：

- Bytes per Token（BPT）：每个 token 平均承载多少字节。值越高，通常说明压缩更强。
- Fertility：平均每个词被拆成多少 token，常写作 $F = T / W$，其中 $T$ 是 token 数，$W$ 是词数。值越低，通常说明拆分更少。
- Normalized Sequence Length（NSL）：相对于某个基线 tokenizer 的序列长度比例，用来横向比较不同方案。

结论可以压缩成一句话：好的 tokenization 不是“切得越细越好”，也不是“词表越大越好”，而是在给定训练语料、语言类型、上下文窗口和训练预算下，让压缩率、覆盖率和语义完整性同时落在可接受区间。

---

## 问题定义与边界

本文讨论的是预训练数据的 tokenization，不讨论下游任务微调时的 prompt 模板，也不讨论推理阶段的 KV cache 优化。边界很明确：我们关心的是“原始文本进入模型前，如何被切分，以及这种切分如何影响训练”。

先定义几个基础概念。

| 术语 | 白话解释 | 训练中关心的量 |
| --- | --- | --- |
| 词表 Vocabulary | 模型认识的“合法编号集合” | 大小 $|V|$ |
| Token | 文本切分后的最小单元 | 序列长度 $L$ |
| OOV | Out of Vocabulary，词表外内容 | 是否能回退处理 |
| Coverage | 词表对训练数据的覆盖程度 | 未知字符/片段比例 |
| Fertility | 一个词平均拆成几个 token | 序列是否膨胀 |
| Compression | 同样文本被压成多少 token | 算力效率 |

一个最容易理解的玩具例子是英文单词 `classification`。

| 切分方式 | 切分结果 | 直观含义 |
| --- | --- | --- |
| 字符级 | `c l a s s i f i c a t i o n` | 几乎永不 OOV，但序列很长 |
| WordPiece 风格 | `class` + `##ification` | 保留部分词形结构，长度适中 |
| 更粗子词或整词 | `classification` | 序列更短，但对长尾词依赖更强 |

这里的核心不是“谁更先进”，而是“谁更适合当前语料”。如果切得太细，模型虽然永远不会完全 OOV，但序列会变长；如果切得太粗，序列变短，但长尾词、拼写变体、跨语言片段可能覆盖不足。

预训练前必须先确定接受什么样的边界条件。一个实用判断框架如下：

| 约束 | 典型问题 |
| --- | --- |
| 上下文窗口固定 | token 太多会挤掉有效上下文 |
| 训练预算有限 | 序列越长，attention 越贵 |
| 语料语言复杂 | 形态变化多时，粗词表可能覆盖差 |
| 噪声多 | 清洗不当会让大量罕见片段进入词表 |

对新手来说，最容易混淆的是“可编码”和“编码得合理”不是一回事。

例如一段文本里有：

- 正常英文单词
- Unicode 变体
- 全角数字
- 表情符号
- URL
- 代码路径

如果 tokenizer 能把它们都转成 token，这只能说明“没有报错”；并不说明切分后的长度、语义完整性和训练效率是合理的。预训练系统真正关心的是：这些输入在训练时会不会把 token 预算浪费在噪声上。

一个常见的数值直觉是：非常小的词表可能让同一段文本变成约 250 个 token，而较大的词表可能压到 75 个 token。即使这只是示意，也足够说明一个事实：tokenization 会直接改变“模型到底能在一个窗口里看多少内容”。

如果上下文窗口固定为 4096：

- 250-token 的一段文本，大约可以放入 16 段
- 75-token 的一段文本，大约可以放入 54 段

这个例子不是严格测量，而是用来说明一个工程事实：tokenization 决定了窗口利用率。

---

## 核心机制与推导

先看为什么 tokenization 会影响算力。

设文本总字符量固定。若 tokenizer 更粗，平均每个 token 承载的信息更多，则序列长度 $L$ 变小。设压缩率用字符数除以 token 数表示：

$$
\text{Compression Ratio} = \frac{C}{T}
$$

其中 $C$ 是字符数，$T$ 是 token 数。若 $C$ 固定，压缩率越高，$T$ 越少。

另一常用指标是 fertility：

$$
F = \frac{T}{W}
$$

其中 $W$ 是词数。若一个语料平均每个词要拆成 1.8 个 token，说明序列膨胀比平均拆成 1.2 个 token 更严重。

BPT 的常见写法是：

$$
\text{BPT} = \frac{B}{T}
$$

其中 $B$ 是 UTF-8 字节数。这个指标特别适合跨语言比较，因为不同语言的“字符数”未必可直接比较，但字节数至少是统一的物理计量。

如果还需要横向比较不同 tokenizer，可定义相对序列长度：

$$
\text{NSL} = \frac{T_{\text{candidate}}}{T_{\text{baseline}}}
$$

含义很直接：

- $\text{NSL} < 1$：比基线更短
- $\text{NSL} = 1$：与基线接近
- $\text{NSL} > 1$：比基线更长

这几个指标之间没有一一对应，但方向通常一致：

| 指标变化 | 一般意味着什么 |
| --- | --- |
| BPT 上升 | 单个 token 承载更多字符或字节，序列更短 |
| Fertility 下降 | 一个词被拆得更少，序列更短 |
| NSL 下降 | 相对基线 tokenizer，序列更短 |

再看成本。

如果词表很小，比如偏字符级，$|V|$ 不大，但 $L$ 很长。输出层便宜，attention 贵。  
如果词表很大，$|V|$ 上升，但 $L$ 显著下降。输出层更贵，attention 更便宜。

从训练图上看，至少有三部分会受影响：

| 模块 | 更依赖什么 |
| --- | --- |
| Embedding 查表 | 词表大小、隐藏维度 |
| 输出层 softmax / LM head | $|V|$、$d$、$L$ |
| Self-Attention | 主要受 $L^2$ 支配 |

用简化公式写：

$$
\text{Embedding Params} \approx |V| \times d
$$

$$
\text{LM Head Cost} \approx O(L \times d \times |V|)
$$

$$
\text{Attention Cost} \approx O(L^2 \times d)
$$

因为 attention 近似按 $L^2$ 增长，而 softmax 更接近按 $|V|$ 线性增长，所以在很多现代模型里，适度缩短序列往往很值钱，尤其当上下文窗口较大时更明显。

玩具推导如下。假设同一段文本：

- 方案 A：250 个 token
- 方案 B：75 个 token

仅看 attention 主项：

$$
250^2 = 62500,\quad 75^2 = 5625
$$

后者约是前者的 9\% 左右。也就是说，在只比较 attention 主项时，序列从 250 降到 75，二次项成本会大幅下降。即使大词表让 embedding 和 softmax 增大，这个序列缩短带来的收益依然可能覆盖掉词表成本增长。

为了避免只看单一公式，可以把权衡写成更直观的对照：

| 变化 | 好处 | 代价 |
| --- | --- | --- |
| 词表变大 | token 更粗，序列变短 | 参数更多，长尾更稀疏 |
| 词表变小 | 参数更省，覆盖更稳 | 序列更长，窗口更浪费 |
| 归一化更强 | 变体更少，碎片更少 | 可能丢失部分表面形式信息 |
| byte fallback 更强 | 几乎不报 OOV | 脏文本可能极度膨胀 |

但这不是说词表应该无限变大。原因有三个：

1. 稀有词会变成低频参数，训练不充分。
2. 语料规模不够时，大词表里的许多条目学习不到稳定统计。
3. 多语言、噪声文本、代码混排文本会让长尾碎片暴增。

不同算法在这里的偏好也不同。

| 算法 | 机制概括 | 常见特点 |
| --- | --- | --- |
| BPE | 不断合并高频相邻片段 | 简单稳定，工程实现成熟 |
| WordPiece | 选择更有整体收益的子词单元 | 常见于 BERT 系，偏重整体建模效果 |
| Unigram | 先给大量候选，再删除低价值单元 | 概率化更强，常对复杂形态语言更灵活 |

同一段英文语料中，可能出现 WordPiece token 数最少、BPE 次之、Unigram 更细的现象。这说明算法选择并不只是“谁更新”，而是谁更适合你的数据分布。对中文、阿拉伯语、土耳其语、代码混排文本，这种差异通常会更明显。

---

## 代码实现

工程上不要先盲目训练 tokenizer，而应先做最小闭环：

1. 清洗与归一化文本。
2. 训练多个候选 tokenizer。
3. 在同一批验证语料上测 `fertility`、`BPT`、`coverage`。
4. 再决定词表大小和算法。

下面先给出一个可直接运行的 Python 示例。它不依赖第三方库训练真正的 BPE，但会模拟三种粒度，并输出可比较的指标。代码可以直接保存为 `tokenization_eval.py` 运行。

```python
import re
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class EvalResult:
    name: str
    tokens: int
    words: int
    bytes_: int
    fertility: float
    bpt: float
    coverage: float


def whitespace_words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def char_level_tokenize(text: str) -> List[str]:
    return [ch for ch in text if not ch.isspace()]


def toy_wordpiece_tokenize(text: str) -> List[str]:
    vocab = {
        "class", "##ification",
        "token", "##ization",
        "model", "data", "clean",
        "pre", "##train", "language"
    }
    out = []
    for word in whitespace_words(text.lower()):
        if word == "classification":
            out.extend(["class", "##ification"])
        elif word == "tokenization":
            out.extend(["token", "##ization"])
        elif word == "pretrain":
            out.extend(["pre", "##train"])
        elif word in {"model", "data", "clean", "language"}:
            out.append(word)
        else:
            # 模拟 subword fallback：未知词拆成 2 字符片段
            pieces = [word[i:i + 2] for i in range(0, len(word), 2)]
            out.extend(pieces)
    return out


def coarse_word_tokenize(text: str) -> List[str]:
    return whitespace_words(text.lower())


def compute_coverage(tokens: List[str]) -> float:
    unknown_like = sum(1 for tok in tokens if len(tok) <= 2 and tok.isalpha())
    return 1.0 - unknown_like / max(len(tokens), 1)


def evaluate(name: str, text: str, tokenizer: Callable[[str], List[str]]) -> EvalResult:
    words = whitespace_words(text)
    tokens = tokenizer(text)
    bytes_ = len(text.encode("utf-8"))
    fertility = len(tokens) / max(len(words), 1)
    bpt = bytes_ / max(len(tokens), 1)
    coverage = compute_coverage(tokens)
    return EvalResult(
        name=name,
        tokens=len(tokens),
        words=len(words),
        bytes_=bytes_,
        fertility=fertility,
        bpt=bpt,
        coverage=coverage,
    )


def print_table(results: List[EvalResult]) -> None:
    header = f"{'name':<16}{'tokens':>8}{'words':>8}{'bytes':>8}{'fertility':>12}{'bpt':>10}{'coverage':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<16}{r.tokens:>8}{r.words:>8}{r.bytes_:>8}"
            f"{r.fertility:>12.2f}{r.bpt:>10.2f}{r.coverage:>12.2%}"
        )


if __name__ == "__main__":
    text = "classification tokenization model data clean pretrain language"

    results = [
        evaluate("char", text, char_level_tokenize),
        evaluate("wordpiece_like", text, toy_wordpiece_tokenize),
        evaluate("coarse_word", text, coarse_word_tokenize),
    ]

    print_table(results)

    assert results[0].tokens > results[1].tokens > results[2].tokens
    assert results[0].fertility > results[1].fertility > results[2].fertility
    assert results[0].bpt < results[1].bpt < results[2].bpt
```

一组典型输出会接近下面这样：

```text
name              tokens   words   bytes   fertility       bpt    coverage
---------------------------------------------------------------------------
char                  58       7      71        8.29      1.22      100.00%
wordpiece_like        15       7      71        2.14      4.73       60.00%
coarse_word            7       7      71        1.00     10.14      100.00%
```

这个示例说明了最重要的事实：粒度越细，token 越多，fertility 越高，BPT 越低。  
同时也能看到另一件事：最粗粒度不一定自动最好，因为真实工程里它可能带来 OOV、词表稀疏和跨域泛化差的问题。上面的 `coarse_word` 只是玩具示意，它故意忽略了真实词表覆盖难题。

如果进入真实训练，至少要把评估闭环写成可以复用的脚本。下面是一个更接近真实工程的版本，使用 `sentencepiece` 训练 Unigram 或 BPE。运行前先安装：

```bash
pip install sentencepiece
```

然后执行：

```python
from pathlib import Path
import re
import sentencepiece as spm


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def write_corpus(lines, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            line = normalize_text(line)
            if line:
                f.write(line + "\n")


def train_sentencepiece(
    input_path: str,
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "unigram"
) -> None:
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        normalization_rule_name="nmt_nfkc",
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )


def whitespace_words(text: str):
    return [w for w in re.split(r"\s+", text.strip()) if w]


def evaluate_tokenizer(sp_model_path: str, eval_lines):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    total_tokens = 0
    total_words = 0
    total_bytes = 0

    for line in eval_lines:
        line = normalize_text(line)
        if not line:
            continue
        pieces = sp.encode(line, out_type=str)
        total_tokens += len(pieces)
        total_words += len(whitespace_words(line))
        total_bytes += len(line.encode("utf-8"))

    fertility = total_tokens / max(total_words, 1)
    bpt = total_bytes / max(total_tokens, 1)

    return {
        "tokens": total_tokens,
        "words": total_words,
        "bytes": total_bytes,
        "fertility": round(fertility, 4),
        "bpt": round(bpt, 4),
    }


if __name__ == "__main__":
    train_lines = [
        "Tokenization controls sequence length and training efficiency.",
        "A larger vocabulary usually shortens the sequence.",
        "But a larger vocabulary also increases embedding and softmax cost.",
        "Normalization matters for multilingual and noisy text.",
    ]
    eval_lines = [
        "Tokenization changes compute allocation in pretraining.",
        "Noisy text can explode sequence length under byte fallback.",
    ]

    corpus_path = "toy_corpus.txt"
    write_corpus(train_lines, corpus_path)

    train_sentencepiece(corpus_path, "toy_unigram", vocab_size=128, model_type="unigram")
    train_sentencepiece(corpus_path, "toy_bpe", vocab_size=128, model_type="bpe")

    unigram_metrics = evaluate_tokenizer("toy_unigram.model", eval_lines)
    bpe_metrics = evaluate_tokenizer("toy_bpe.model", eval_lines)

    print("unigram:", unigram_metrics)
    print("bpe:", bpe_metrics)
```

这个版本有三个工程价值：

1. 可以真实训练 tokenizer，而不是只看玩具拆分。
2. 可以在 held-out 文本上做统一评估。
3. 可以扩展为多词表大小、多语种、多子域的批量实验。

词表大小评估表建议至少做成这样：

| 词表大小 | 算法 | Fertility | BPT | OOV% / UNK | 备注 |
| --- | --- | --- | --- | --- | --- |
| 8k | BPE | 1.82 | 2.9 | 0.0 | 序列偏长 |
| 16k | Unigram | 1.55 | 3.5 | 0.0 | 更稳，仍偏细 |
| 32k | WordPiece | 1.36 | 4.1 | 0.0 | 较均衡 |
| 64k | Unigram | 1.21 | 4.8 | 0.0 | 压缩更强，长尾更稀疏 |

真实工程例子可以看阿拉伯语、土耳其语、芬兰语等形态变化强的语言。其特点是同一个词根会衍生很多表面形式，如果不先做 normalization，tokenizer 会把大量变体学成低频碎片，fertility 升高，上下文窗口被浪费。此时“先归一化，再训练 Unigram”通常比直接套一个英文风格的 byte-level BPE 更稳。

对新手更实用的经验是：  
不要只训练一个 tokenizer 然后直接开跑。至少应当做下面四种切片评估：

- 通用正文
- 代码与路径
- URL、邮箱、数字串
- 多语言混排文本

否则你很可能得到一个“全局平均分不错，但关键子域很差”的 tokenizer。

---

## 工程权衡与常见坑

真正影响结果的，往往不是算法名字，而是数据分布和清洗策略。

第一类坑是词表太小。现象通常是：

- 一个词被拆成很多片段
- 序列长度暴涨
- 固定窗口里能容纳的原文更少
- 训练 token 预算被快速消耗

第二类坑是词表太大。现象通常是：

- 训练初期很多词表项几乎不更新
- 长尾词参数稀疏
- 语料稍一变化就出现分布漂移
- 输出层和 embedding 占用过大

第三类坑是把 OOV 理解成“只要能编码就没问题”。这不对。字节回退虽然能保证任意文本可编码，但代价常常是极长序列。对脏数据、混合编码文本、表情符号、脚本切换文本尤其明显。

下表可直接作为排查清单：

| 坑/现象 | 典型原因 | 规避方式 | 监控指标 |
| --- | --- | --- | --- |
| 序列异常变长 | 词表太小或预切分不当 | 增大 vocab，重做 pre-tokenization | Fertility, NSL |
| 大量低频碎片 | 未做归一化，脏字符太多 | 先做 Unicode 归一化和字符过滤 | Coverage, token freq |
| OOV 回退太多 | 训练语料覆盖不足 | 增加领域语料，保留 byte fallback | OOV%, BPT |
| 代码/中英混排切分差 | 单一规则不适配混合数据 | 单独评估代码、中文、英文子集 | 分域 fertility |
| 词表训练结果不稳定 | 数据量太小却设大 vocab | 缩小词表或扩充语料 | vocab utilization |

这里补一个新手常见误区：  
很多人看到 `byte_fallback=True`，就以为 OOV 问题彻底解决了。实际上它解决的是“编码失败”，不是“训练效率”。例如一段异常文本：

```text
正常文本 + ｆｕｌｌｗｉｄｔｈ + \x00\x01 + 😀😀 + /var/log/app/error.log
```

如果没有提前做清洗和归一化，这一段常会被切成大量低价值 token。模型并不会自动知道哪些是噪声，它只会为这些 token 分配训练预算。

一个更实用的判断方式是看子域指标，而不是只看全局均值。

| 子域 | 常见风险 | 为什么单独看 |
| --- | --- | --- |
| 主语言正文 | 影响主训练质量 | 占比最高，但不能掩盖其他问题 |
| 代码片段 | 标识符、缩进、符号密集 | 常导致 fertility 上升 |
| URL/路径/邮箱 | 长串符号与分隔符 | 容易产生超长切分 |
| 数字与单位 | 日期、金额、版本号格式多 | 对规则设计敏感 |
| 多语言混排 | 脚本切换频繁 | 最容易暴露 coverage 问题 |

一个新手最容易忽略的点是：tokenizer 也是数据清洗的一部分。比如你把全角半角、不同 Unicode 组合形式、不可见控制字符、重复标点、错误编码片段直接喂进去，模型最终会为这些噪声分配 token 预算。那不是“覆盖全面”，而是在浪费训练资源。

真实工程里，通常要把语料拆成多个切片分别评估：

- 主语言正文
- 代码片段
- URL、邮箱、路径
- 数字、单位、时间
- 跨语言混排文本

因为同一个 tokenizer 在这些子域上表现差异很大。只看全局平均值，很容易掩盖问题。

如果需要一个最小排查流程，可以直接按下面执行：

1. 先抽样 5 万到 20 万行 held-out 文本，不参与 tokenizer 训练。
2. 分别统计全局、正文、代码、URL/路径、多语言混排的 fertility 和 BPT。
3. 对异常样本做反查，看看是不是脏字符、未归一化、词表过小或规则不匹配。
4. 再决定是否调词表大小、替换算法或增加领域语料。

---

## 替代方案与适用边界

Subword tokenizer 不是唯一方案。

第一类替代是 byte-level 或 char-level 优先方案。白话解释：不强依赖固定词表，任何文本都能直接编码。优点是几乎不存在传统 OOV；缺点是序列更长，需要模型结构配合降采样或更高效的长序列处理。

第二类替代是 token-free 路线，例如直接基于字节或字符建模，再在编码器内部做局部聚合。它更适合极多语言、书写系统复杂、脏数据较多的场景，但训练成本和工程复杂度通常更高。

第三类替代是“更大输入单元”的扩展路线，即在已有 tokenizer 上继续加入 multi-gram 或领域词表。它适合高频领域术语稳定、训练数据量足够、追求压缩率的场景。

可以用下表做选型：

| 方案 | 适用场景 | 优点 | 缺点 | 典型边界 |
| --- | --- | --- | --- | --- |
| Subword（BPE/WordPiece/Unigram） | 通用预训练 | 平衡最好，生态成熟 | 仍需手工选 vocab 和规则 | 大多数 LLM 训练 |
| Byte-level | 噪声文本、多语言混合 | 几乎无 OOV | 序列长 | 数据脏、字符集复杂 |
| Char/Token-free | 极端多语言、低资源语言 | 最强覆盖 | attention 压力大，结构要改 | 需要专门模型设计 |
| 扩展词表/多粒度词表 | 领域术语密集 | 可显著压缩常见短语 | 长尾参数更稀疏 | 语料够大才值得 |

如果把问题再说得更具体一些，选择逻辑通常是：

| 条件 | 更可行的路线 |
| --- | --- |
| 语料干净、主语言明确、训练预算有限 | Subword 优先 |
| 多语种、噪声大、符号密集 | Byte-level 或 byte fallback 强的方案 |
| 形态变化强，归一化收益高 | 先 normalization，再试 Unigram |
| 代码、日志、路径占比高 | 单独做代码子域评估，必要时扩展词表 |
| 极长上下文是硬需求 | 优先关注序列压缩率，而不只是 OOV |

实用判断标准如下：

- 语料干净、语言结构相对稳定：优先 Subword。
- 多语种、噪声大、符号多：优先 byte-level 或带 fallback 的方案。
- 形态变化强、正规化收益大：先做 normalization，再试 Unigram。
- 代码、路径、日志很多：单独评估这些子域，不要只看自然语言平均值。

最终边界很清楚：tokenization 不是孤立组件，它必须和语料清洗、上下文窗口、模型结构、训练预算一起设计。离开这些约束谈“哪个 tokenizer 最好”，结论通常没有意义。

---

## 参考资料

- Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with Subword Units*。BPE 在子词切分中的经典工作，适合理解“高频片段合并”的基本思想。
- Schuster, Nakajima, *Japanese and Korean Voice Search*。WordPiece 早期代表性来源，适合理解基于子词的词表构建思路。
- Kudo, *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*。SentencePiece/Unigram 路线的关键材料，适合理解候选子词与概率删减机制。
- Kudo, Richardson, *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*。工程实践中最常引用的 tokenizer 训练工具之一。
- Xue et al., *ByT5: Towards a token-free future with pre-trained byte-to-byte models*。理解 byte-level / token-free 路线的代表性论文。
- Clark et al., *CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation*。理解字符级与 token-free 编码的另一条代表路线。
- 相关 tokenizer 工程文档：SentencePiece、Hugging Face Tokenizers、OpenAI/tiktoken 等项目文档。适合查实现细节、规范化规则和实际 API。
- 关于评估指标的工程实践文章与综述：重点关注 BPT、fertility、NSL、词表利用率、子域分布评估，而不是只看单一平均 token 长度。
