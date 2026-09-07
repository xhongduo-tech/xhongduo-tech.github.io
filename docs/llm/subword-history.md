---
title: GloVe 之后的子词
date: 2026-09-07
section: llm
---

# GloVe 之后的子词

<div class="epigraph">
    <p>把稀有词切成训练里反复出现的子词单位，翻译系统就不必为开放词表里的每一个未见词准备一套对策。</p>
    <footer>—— Sennrich, Haddow, Birch, Neural Machine Translation of Rare Words with Subword Units, ACL 2016</footer>
</div>

2014 年的 GloVe 把「一个词一个向量」推到静态词嵌入的高峰：全局共现矩阵的分解，补上 word2vec 局部窗口看不到的统计。之后十年，这条路没有在更大的静态词表上继续加宽，而是把符号本身切碎。本篇写 GloVe 之后子词如何成为默认输入，与 [Tokenizer 设计](/llm/tokenizer-design) 的算法对照、[词表压缩](/llm/vocab-compression) 的 $V$–压缩率权衡互补；ByT5 与 SuperBPE 是这条线两端的当代实验，细节见专文。

## 问题

静态词向量假定词表封闭。新闻与网页上的形态变化、拼写变体、专名与合成词会把大量质量送到未登录符，或把同一词干拆到从未共享梯度的行上。Mikolov 等人 2013 年的 Skip-gram / CBOW、Pennington、Socher 与 Manning 2014 年的 GloVe，优化的都是词型（word type）的点：GloVe 显式分解 $\log X_{ij}\approx w_i^\top \tilde{w}_j+b_i+\tilde{b}_j$，词表之外没有结构。开放词表语言、德语复合词、日韩无空格文本，会把这条假设撕开。

神经翻译最先撞墙。词级 seq2seq 要么截短词表再拷贝未登录，要么把字符卷积接到词边界上，系统仍然依赖某个「词」定义。需要一种固定大小、对任意字符串可解码、又能把高频词保留为原子的符号。子词是工程回答，不是语言学发现。

<span class="marginnote">GloVe 的「全局」指共现矩阵，不是上下文相关表示。ELMo 与 BERT 出现后，静态向量并没有在数学上被证伪，只是作为输入层被上下文编码器替代。子词切分与上下文编码是前后脚，不是同一件事。</span>

### 静态子词与上下文子词要分开

Bojanowski 等人 2017 年的 FastText 仍在静态空间里：每个词是字符 n-gram 向量之和，未见词可以组合，但每个 n-gram 仍是一份与句子无关的查找表。Schuster 与 Nakajima 2012 年的 WordPiece 为日韩语音搜索而做，优化的是词表对语料的似然，还不是深度编码器的输入协议。真正改写预训练生态的，是子词 ID 序列进入 Transformer，让同一片段在不同句子里得到不同隐状态。历史顺序是：先有切分，再有上下文；评价时不要用 BERT 的 GLUE 去回打 GloVe 的词类比。

## 方法

### 三条进入神经机器翻译的切分线

Sennrich、Haddow 与 Birch 把 Gage 1994 年的字节对编码改成词内合并：词表从字符加词尾符出发，反复合并最频繁相邻对，切分时贪心应用合并表。它不优化句子似然，优化的是压缩。英语里 `-ing`、`-ed` 会因为频次自然出现。Wu 等人 2016 年的 GNMT 采用 WordPiece：合并准则是语言模型似然而不是原始频次，词表大约 8k–32k 就在准确率与解码速度之间够用，未登录不再需要拷贝模块。Kudo 2018 年的 Unigram 从过大候选集用 EM 删词，切分可以取 Viterbi 或按概率采样；Kudo 与 Richardson 的 SentencePiece 把 BPE 与 Unigram 做成不依赖预分词的库，空格当成普通字符，对无空格语言是同一套算法。

这三条线解决的是同一张图：封闭词表。差别在目标函数（频次贪心 / 似然合并 / 似然删词）和是否允许跨空格。预分词一旦按空格切开，合并永远进不了短语，这是后来 SuperBPE 要松开的约束；当时 NMT 要的恰恰是「词内形态，词间仍是词」。

### 预训练把子词从翻译工具变成默认输入

BERT（Devlin 等人，2019）用 WordPiece，GPT-2（Radford 等人，2019）用字节级 BPE：先落到字节再合并，避开巨大 Unicode 字符表与未见码点。T5 与 mT5 用 SentencePiece。从此「词嵌入」在论文里多半指子词嵌入矩阵，GloVe 式 40 万词表不再是标配。上下文编码器让同一子词在「bank」的两种意义上分叉，静态 GloVe 做不到；但编码器仍然吃离散 ID，切分错误会变成不可逆的表示错误。字节级 BPE 是折中：符号仍是子词，回退是字节，不是未登录符。

```mermaid
flowchart LR
  G["GloVe / word2vec\n静态词型"] --> F["FastText\n字符 n-gram 静态和"]
  F --> N["NMT：BPE / WordPiece / Unigram"]
  N --> P["BERT / GPT\n上下文子词"]
  P --> B["字节级 BPE 默认"]
  P --> Y["ByT5 纯字节"]
  B --> S["SuperBPE 跨空格"]
```

## 机制

### 为什么静态全局矩阵先停步

GloVe 的窗口共现可以很大，但行仍是词型。加大词表只是多几行极少更新的向量；引入子词是把参数从稀疏词型行挪到可组合的片段。FastText 已经展示组合能救未见词，却仍把句子当成词袋式查找。Transformer 需要的是变长 ID 序列与位置编码，子词恰好给出定长词表上的变长序列。词类比任务上 GloVe 仍可强，那是词型几何；完形填空与生成要的是序列条件分布，输入层必须能表示任意字符串。

WordPiece 的似然合并与 BPE 的频次合并常得到相似的高频词，差别在长尾：似然更不愿把无信息的高频垃圾粘成原子，但也依赖初始候选。Unigram 的多切分正则直接攻击「同一词必须同一路径」，对噪声与形态变体有用，部署时通常改回确定性 Viterbi。这些取舍在 [Tokenizer 设计](/llm/tokenizer-design) 里展开；这里只需记住：GloVe 之后的创新首先是切分目标，其次才是模型深度。

<span class="marginnote">「子词＝语素」是事后解释。BPE 会把高频网址、页脚、Markdown 标记收成 token；也会把数字切碎或粘成多年份。语素分析器不是它的训练目标。评测形态任务时，应像 ByT5 那样直接比字素与屈折，而不是看词表里有没有 `-tion`。</span>

### 上下文到来之后，静态向量变成特征而不是主干

ELMo 仍以字符卷积组词，再接双向 LSTM；BERT 去掉「先组词再上下文」的两段，直接在 WordPiece 上做掩码语言建模。GloVe 向量还出现在一些分类基线与推荐模型里（例如某些 Fastformer 实验仍用 GloVe 初始化），但作为大模型输入层已经被子词嵌入取代。历史含义是：开放词表问题在进入深度上下文之前就已经用切分解决了；深度模型继承了这套符号，也继承了预分词、词表配比和压缩率全部偏见。

## 边界与工程取舍

子词没有取消词表，只是把未知从「词」下放到「字节或字符」。纯字节（ByT5）把切分责任交给网络，长度税回到注意力；跨空格 superword（SuperBPE）把切分再往短语推，压缩率的天花板才打开。两者都说明：GloVe 之后的主线不是「更大的静态词表」，而是「符号粒度与计算预算一起选」。

不要用 2014 年的词类比排名选择 2026 年的 tokenizer。类比测量的是静态几何，当前系统测量的是给定切分下的序列损失与下游。也不要把 SentencePiece 的「无预分词」写成已经跨过英语空格语义——默认实现仍常把空白当作特殊字符编码，合并是否跨词取决于训练设定，不是库名。

词表一旦与checkpoint绑定，改切分等于改嵌入形状。GloVe 时代换词表是换一份向量文件；现在换 tokenizer 通常要重训或做危险的嵌入移植。这是子词成为基础设施之后的版本负担。

<span class="marginnote">报告「训练了 N 个 token」时必须声明切分。同一 C4 用 GloVe 词型、32k BPE 与纯字节，N 可以差数倍。缩放定律不能跨符号系统直接连线。</span>

## 小结

- GloVe 是静态词型嵌入的高峰；开放词表与形态迫使符号从词切到子词。
- FastText 先在静态空间引入字符 n-gram；NMT 用 BPE、WordPiece、Unigram 把开放词表做成固定 ID 序列。
- SentencePiece 去掉对预分词的语言特定依赖；GPT-2 一类字节级 BPE 成为解码器默认。
- BERT / GPT 用上下文编码替代静态向量，但输入仍是子词 ID，切分偏见全部保留。
- 纯字节与跨空格 superword 是这条线在 2022 与 2025 年的两端实验，不是回到 GloVe。
- 静态词向量仍可用于特征与小模型，不再是大规模序列模型的输入合同。
- 出处：Pennington et al.，GloVe，EMNLP 2014；Sennrich et al.，ACL 2016；Wu et al.，GNMT，2016；Bojanowski et al.，FastText，TACL 2017；Kudo & Richardson，SentencePiece，2018。
