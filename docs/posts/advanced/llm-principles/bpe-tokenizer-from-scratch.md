---
title: 动手实践：用 BPE 从零训练一个 Tokenizer
date: 2026-08-07
---

# 动手实践：用 BPE 从零训练一个 Tokenizer

<div class="epigraph">
<p>纸上得来终觉浅，绝知此事要躬行。</p>
<footer>—— 陆游（南宋诗人）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型原理 ｜ 《动手学深度学习》附录 / Karpathy《Let's build GPT》 ｜ 2026-08-07</p>
</div>

## 为什么动手写一遍

前面的概念——预分词、合并表、字节兜底——讲再多，都不如亲手实现一遍记得牢。这一篇我们用纯 Python 写一个**最小可用 BPE tokenizer**：能训练、能编码、能解码。不依赖任何 NLP 库，全程只用标准库与几行正则代码。写完你就能回答：词表是怎么从 256 长到 30k 的？`merges` 里那串合并规则到底在干什么？<span class="marginnote">Karpathy 的 minbpe 项目把这套代码打磨成了教学级实现；这里我们采用「先最简、后加固」的路线：先把主干跑通，再补上字节级细节与边界处理，与 minbpe 的精髓一致。</span>

## 1 最简骨架：字符级 BPE

先忽略字节与预分词，用最直接的方式实现核心逻辑。**输入**是一段字符串语料，**输出**是词表与合并表。

```python
from collections import defaultdict

def get_stats(ids):
    """扫描相邻对并计数——返回 {相邻对: 出现次数}。"""
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids, pair, idx):
    """一趟线性扫描，把所有匹配的相邻对替换成新记号 idx。"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            newids.append(idx)      # 用新记号替换匹配的相邻对
            i += 2                  # 合并后跳过新记号，不回头
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

`get_stats` 扫描相邻对并计数——这就是「找当前最高频对」的数据基础。
`merge` 用**一趟线性扫描**把所有匹配的相邻对替换成新记号——注意替换后继续前进，不回头。

`defaultdict` 与 `list` 是标准库容器，两个函数合计约 20 行。**BPE 的核心算法就这么点东西。**<span class="marginnote">`merge` 里 `while` 循环的细节很关键：合并是「非重叠贪心」，一对合并后，新记号与后继字符的相邻关系留到下一轮重新统计。若改成 `for` 循环（重叠贪心），就会产生交叠合并，语义完全不同。</span>

## 2 训练循环：把词表从 256 长到目标值

假设输入已经是 UTF-8 字节序列（先忽略文本，直接拿字节跑）。训练循环如下：

```python
def train_bpe(text: str, vocab_size: int):
    # 初始化：字符串 -> 字节，每个字节一个 token——字节级 BPE 的起点
    tokens = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}

    for new_id in range(256, vocab_size):                 # vocab_size - 256 次合并
        stats = get_stats(tokens)
        if not stats:
            break
        pair, _ = max(stats.items(), key=lambda kv: kv[1])   # 每次找最高频对
        tokens = merge(tokens, pair, new_id)
        merges[pair] = new_id                             # 登记合并规则
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]   # 新 token = 两旧 token 字节串拼接
    return vocab, merges
```

关键三步：

**初始化**：`text.encode("utf-8")` 把字符串变成字节，每个字节一个 token——这就是**字节级 BPE** 的起点。
**循环**：`vocab_size - 256` 次合并，每次找最高频对、替换、登记。`vocab_size` 决定合并次数，也就是最终词表大小。
**词表登记**：`merges` 记录「新 token 由哪两个旧 token 拼成」，解码时按字节串还原即可。<span class="marginnote">这个「按字节串还原」的设计是字节级 BPE 的妙处：vocab 直接存 bytes，解码只需把每个 id 映射回 bytes 再拼接，最后按 UTF-8 解码。合并表 merges 负责编码，vocab 负责解码，各司其职。</span>

## 3 编码与解码：回放与还原

训练完，编码和解码各是十行内的事：

```python
def encode(text: str, merges: dict) -> list:
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # 只选「合并表中最早被合并、且当前相邻」的对；不在合并表里的对排最后
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        tokens = merge(tokens, pair, merges[pair])
    return tokens


def decode(ids: list, vocab: dict) -> str:
    raw = b"".join(vocab[i] for i in ids)          # 每个 id 的字节串拼起来
    return raw.decode("utf-8", errors="replace")   # 按 UTF-8 还原，兜住非法字节
```

**编码**：从字节序列出发，**每一轮把「合并表中最早被合并、且当前相邻」的对替换掉**。`merges.get(p, float("inf"))` 让没出现在合并表里的对排在最后，保证只处理真实合并。<span class="marginnote">这是「合并表顺序」最自然的代码表达：字典 merges 的插入顺序就是训练顺序，`min(stats, key=…)` 按「最早合并」挑选。minbpe 里用显式排名优化性能，但语义与此完全一致。</span>
- **解码**：`vocab` 存了每个 id 的字节串，`b"".join(...)` 拼起来再按 UTF-8 还原。`errors="replace"` 兜住罕见的非法字节序列。

## 4 升级到真实可用：预分词与特殊 token

纯字节级 BPE 能跑，但真实模型不会这么用——纯字节切分会让英文单词碎成字节串。真实 tokenizer 加两步：

**第一步，预分词。** 用正则把文本切成「单词 + 空白」片段（英文按空格、标点边界），在每个词后补一个空格，然后在**片段内部**跑字节 BPE，片段之间不跨合并：

```python
import re

# GPT-2/GPT-4 风格：缩写整体保留，单词 / 数字 / 空白 / 换行分开
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")

def pretokenize(text: str) -> list:
    return PAT.findall(text)
```

这个正则约等于 GPT-4 的分词规则：`'s`、`'t`（`'re`、`'ve`）等缩写整体保留，单词、数字、空白、换行分开。**预分词的意义**：让「词」级别的自然边界不被打散，BPE 只负责「词内部」的优化。

**第二步，注册特殊 token。** `<|endoftext|>` 这类特殊记号要占用固定 id，且编码时直接整体映射：

```python
# 特殊 token：占用固定 id，编码时直接整体映射，不参与 BPE 合并
SPECIAL_TOKENS = {
    "<|endoftext|>": 50256,
    "<|im_start|>": 50257,
    "<|im_end|>": 50258,
}
```

特殊 token 在**任何文本处理之前**先被整体匹配，避免被 BPE 合并污染。

## 5 公式解析：编码的「最早合并优先」准则

编码循环每步选出的相邻对，必须满足「在合并表中出现最早」——这是编码与训练保持一致性的关键：

$$
\text{chosen pair} = \arg\min_{(a,b) \in \text{stats}} \text{rank}(a,b), \qquad
\text{rank}(a,b) = \text{合并表中 }(a,b)\text{ 的顺序号}
$$

对这条式子做三步拆解：

- **第一步，读懂 $\text{rank}$**：训练时第 $r$ 次合并的对，其 rank 就是 $r$。rank 越小表示合并越早、越「基础」。
- **第二步，理解「为什么是最小 rank」**：早期合并的是**全局最高频**的对，意味着它们在任意文本中最可能先出现。编码时先应用最基础、最通用的合并，再应用更高层、更稀有的合并——这是把训练时的「频率排序」忠实回放。
- **第三步，读出与训练对称的贪心**：训练是「每次选最高频」，编码是「每次选最早期」。两者方向相反但等价——都保证合并顺序的确定性，从而保证**同一段文本每次编码结果一致**（可复现性）。

**辨析｜易错点：** 编码时**不是**「每次选当前出现次数最高的对」。如果编码阶段再用频率选对，那么 `("t", "h")` 出现 100 次就先把 `("t", "h")` 合并，但这可能与训练时的合并顺序冲突。**编码必须按 rank 而非频率**——这是从零实现时最容易写错的地方。

## 6 小结

- BPE 核心 = **`get_stats`（统计）+ `merge`（替换）** 两个约 20 行的函数。
- 字节级起点：`text.encode("utf-8")`，词表从 256 个字节开始逐步合并到目标大小（如 30k）。
- 合并表 `merges` 管**编码**，词表 `vocab`（字节串映射）管**解码**，各司其职。
- 真实可用需加**预分词**（正则按词切）与**特殊 token 注册**（整体匹配，防污染）。
- **编码按「最早合并优先」而非频率**——这是保证可复现性的关键易错点。

到这里，Tokenizer 篇全部完成。接下来我们进入 GPT 架构的主体——从 **Decoder-only 架构总览**开始，把「从 Token 到 Logits 的完整数据流」一路走通。
