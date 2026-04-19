## 核心结论

BiLSTM-CRF 是一种“逐词打分 + 整句约束”的序列标注模型。序列标注是指：输入一串词元，对每个位置输出一个标签，例如把“阿司匹林 每日 100mg”标成“药名、频次、剂量”。

它由两部分组成：

```text
输入词元 -> BiLSTM -> 发射分数 -> CRF -> 最优标签序列
```

BiLSTM，中文常叫双向长短期记忆网络，作用是同时看当前位置左侧和右侧上下文，给每个词元生成上下文表示。它不直接输出最终标签，而是输出每个位置属于每个标签的分数，这个分数叫发射分数。

CRF，中文常叫条件随机场，作用是在整句层面判断“哪一整条标签路径最合理”。它会学习标签之间的转移分数，例如 `B-疾病` 后面接 `I-疾病` 通常合理，`O` 后面直接接 `I-药品` 通常不合理。

BiLSTM-CRF 的优势不在于让单个词元的局部判断一定更强，而在于让整条标签序列更一致。NER、分词、词性标注这类任务都有明显的标签连续性，因此适合使用 CRF。

玩具例子：假设句子是“发热 三天”，标签有 `B-症状`、`I-症状`、`O`。逐点分类可能在第二个词上认为 `I-症状` 分数最高，但如果第一个词被判成 `O`，整条序列就变成 `O I-症状`，这在 BIO 标注中不合法。CRF 会综合整句分数，倾向选择 `B-症状 I-症状`，因为这条路径整体更合理。

“全局最优”比“每点最优”更适合序列标注，原因是标签不是彼此独立的。一个位置的最佳标签，放进整句后可能导致非法或低一致性的路径；整句最佳路径才是模型真正要输出的结果。

---

## 问题定义与边界

序列标注的目标是：给定输入序列 $x=(x_1,\dots,x_n)$，输出同样长度的标签序列 $y=(y_1,\dots,y_n)$。这里的 $x_i$ 是第 $i$ 个词元，$y_i$ 是第 $i$ 个标签。

关键点不只是“每个词应该属于哪个类别”，还包括“标签之间是否满足任务约束”。

以 BIO 标注为例：

| 标签 | 含义 |
|---|---|
| `B-疾病` | 疾病实体开始 |
| `I-疾病` | 疾病实体内部 |
| `B-药品` | 药品实体开始 |
| `I-药品` | 药品实体内部 |
| `O` | 非实体 |

`I-疾病` 不能无条件出现在句首，因为它表示“疾病实体内部”，前面应该已经有一个疾病实体的开始。`B-药品 I-疾病` 也通常不合法，因为药品实体内部不能突然跳到疾病实体内部。这里的“合法性”不是后处理细节，而是问题定义的一部分。

不同任务对 BiLSTM-CRF 的适配程度不同：

| 任务类型 | 是否适合 BiLSTM-CRF | 原因 |
|---|---|---|
| 命名实体识别 NER | 适合 | BIO/BIOES 标签有强约束 |
| 中文分词 | 适合 | `B/M/E/S` 标签有固定转移规则 |
| 词性标注 | 较适合 | 相邻词性存在统计依赖 |
| 事件抽取中的触发词/论元标注 | 较适合 | 标签边界和类型需要一致 |
| 文本分类 | 不适合 | 输出是句子级标签，不是逐词标签 |
| 句子级情感分类 | 不适合 | 标签之间没有序列转移关系 |
| 独立图片分类 | 不适合 | 样本之间通常没有标签路径约束 |

边界很清楚：如果任务标签之间有明显依赖，CRF 有价值；如果每个样本或每个位置基本独立，CRF 的收益通常有限。

---

## 核心机制与推导

BiLSTM 的输出不是最终标签，而是发射分数 $e_i(k)$。它表示第 $i$ 个位置被标成标签 $k$ 的局部分数。CRF 再引入转移矩阵 $A$，其中 $A_{u,v}$ 表示从前一个标签 $u$ 转到当前标签 $v$ 的分数。

整条路径的分数由两部分组成：每个位置的发射分数，以及相邻标签之间的转移分数。

$$
s(x,y)=\sum_{i=1}^{n}\big(A_{y_{i-1},y_i}+e_i(y_i)\big)+A_{y_n,\text{STOP}},\quad y_0=\text{START}
$$

条件概率是正确路径分数在所有路径分数中的占比：

$$
p(y|x)=\frac{\exp(s(x,y))}{\sum_{y'}\exp(s(x,y'))}
$$

训练时最大化正确标签序列 $y^*$ 的概率，等价于最小化负对数似然：

$$
\mathcal{L}=-\log p(y^*|x)=\log\sum_{y'}\exp(s(x,y'))-s(x,y^*)
$$

解码时选择整句最高分路径：

$$
\hat y=\arg\max_y s(x,y)
$$

这里不能直接枚举所有路径，因为如果句长为 $n$、标签数为 $K$，路径数量是 $K^n$。Viterbi 算法用动态规划求最优路径，动态规划是把大问题拆成可复用的小问题，避免重复计算。

两词句子的玩具数值例子如下。标签集合为 $\{B,O\}$：

| 项 | 分数 |
|---|---:|
| $A_{\text{START},B}$ | 0.3 |
| $A_{\text{START},O}$ | 0.0 |
| $A_{B,B}$ | -0.4 |
| $A_{B,O}$ | 0.2 |
| $A_{O,B}$ | 0.1 |
| $A_{O,O}$ | 0.5 |
| $A_{B,\text{STOP}}$ | 0.0 |
| $A_{O,\text{STOP}}$ | 0.2 |

发射分数：

| 位置 | $e_i(B)$ | $e_i(O)$ |
|---|---:|---:|
| 第 1 个词 | 2.0 | 0.5 |
| 第 2 个词 | 0.1 | 1.2 |

四条路径总分：

| 路径 | 计算 | 总分 |
|---|---|---:|
| `B,O` | $0.3+2.0+0.2+1.2+0.2$ | 3.9 |
| `B,B` | $0.3+2.0-0.4+0.1+0.0$ | 2.0 |
| `O,O` | $0.0+0.5+0.5+1.2+0.2$ | 2.4 |
| `O,B` | $0.0+0.5+0.1+0.1+0.0$ | 0.7 |

因此 CRF 解码选择 `B,O`。如果只看每个位置的最高发射分数，这个例子也是 `B,O`；但在真实任务中，局部最高分经常会和全局路径约束冲突，CRF 的价值就在于用转移分数修正这种局部短视。

---

## 代码实现

实现通常分成 4 块：编码器 `BiLSTM`、线性层生成发射分数、`CRF` 层计算损失、`decode` 方法执行 Viterbi 解码。

训练流程：

```text
embedding -> BiLSTM -> linear -> CRF forward -> loss -> backward
                                      |
decode: embedding -> BiLSTM -> linear -> Viterbi -> 标签路径
```

下面是最小 PyTorch 结构。这里使用常见的第三方 `torchcrf` 接口，重点看张量流向和 `mask`。`mask` 是一个布尔矩阵，用来说明哪些位置是真实词元，哪些位置是 padding。

```python
import torch
import torch.nn as nn

try:
    from torchcrf import CRF
except ImportError:
    CRF = None


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_size, emb_dim=32, hidden_dim=64, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        if CRF is None:
            raise ImportError("Install torchcrf to run this model.")
        self.crf = CRF(tag_size, batch_first=True)

    def _emissions(self, tokens):
        emb = self.embedding(tokens)
        output, _ = self.bilstm(emb)
        return self.hidden2tag(output)

    def forward(self, tokens, tags, mask):
        feats = self._emissions(tokens)
        loss = -self.crf(feats, tags, mask=mask, reduction="mean")
        return loss

    def decode(self, tokens, mask):
        feats = self._emissions(tokens)
        return self.crf.decode(feats, mask=mask)


def toy_viterbi(emissions, trans, start, stop):
    tags = list(emissions[0].keys())
    best_score = None
    best_path = None
    for y1 in tags:
        for y2 in tags:
            score = start[y1] + emissions[0][y1] + trans[(y1, y2)] + emissions[1][y2] + stop[y2]
            if best_score is None or score > best_score:
                best_score = score
                best_path = [y1, y2]
    return best_path, best_score


emissions = [{"B": 2.0, "O": 0.5}, {"B": 0.1, "O": 1.2}]
trans = {("B", "B"): -0.4, ("B", "O"): 0.2, ("O", "B"): 0.1, ("O", "O"): 0.5}
start = {"B": 0.3, "O": 0.0}
stop = {"B": 0.0, "O": 0.2}

path, score = toy_viterbi(emissions, trans, start, stop)
assert path == ["B", "O"]
assert abs(score - 3.9) < 1e-6
```

真实工程例子：电子病历实体抽取。输入可能是“患者昨日出现胸痛，口服阿司匹林 100mg”。模型需要识别“症状、药品、剂量”等实体。BiLSTM 负责根据上下文判断“胸痛”更像症状，“阿司匹林”更像药品；CRF 负责保证 `B-药品 I-药品`、`B-剂量 I-剂量` 这类边界连续，减少 `I-疾病` 开头、实体内部类型跳变等错误。

---

## 工程权衡与常见坑

CRF 的收益来自结构约束，但代价是实现复杂度更高、训练和解码更慢。小数据集上，它通常能稳定提升，因为标签转移约束本身提供了额外先验；大数据或强预训练模型下，CRF 的边际收益可能缩小。

常见错误不是“模型不够大”，而是“约束没写对”。

| 常见坑 | 错误现象 | 修复方式 |
|---|---|---|
| `mask` 缺失 | padding 参与损失，长短句结果不稳定 | 所有 CRF loss 和 decode 都传入 mask |
| 转移矩阵方向混乱 | 训练分数和解码分数不一致 | 固定为 $A_{\text{prev},\text{cur}}$ 并全程统一 |
| 忽略 START/STOP | 句首句尾出现不合理路径 | 显式建模开始和结束转移 |
| 不加非法 BIO 转移约束 | 生成 `I-X` 开头或类型跳变 | 对非法转移置为极小值 |
| 把 CRF 当逐位置 softmax | 输出局部最优但全局不合法 | 使用 Viterbi 做整句解码 |
| padding 标签随便填 | 损失被无意义标签污染 | padding 位置必须被 mask 屏蔽 |
| 小数据无正则 | 训练集好、验证集差 | 加 dropout、预训练词向量或字符特征 |

错误实现 vs 正确实现：

| 场景 | 输出方式 | 可能结果 |
|---|---|---|
| 错误实现 | 每个位置直接 `argmax` | `I-疾病 O B-药品` |
| 正确实现 | Viterbi + 转移约束 | `B-疾病 O B-药品` 或 `O O B-药品` |

直接逐点 `argmax` 只看当前位置分数，不知道 `I-疾病` 开头不合法。Viterbi 会把发射分数和转移分数一起算，所以能避开很多结构错误。

还有一个工程细节：CRF 不会自动理解 BIO 规则。它只学习转移分数。如果训练数据少，模型可能仍然学不稳非法转移。因此很多实现会手动把非法转移分数设成一个很小的值，例如 `-10000`，让解码几乎不可能走这条路径。

---

## 替代方案与适用边界

BiLSTM-CRF 不是所有序列任务的默认答案。选择模型时，先看两个问题：上下文表示是否足够强，标签约束是否明显。

| 方案 | 复杂度 | 约束能力 | 实现难度 | 适用场景 |
|---|---|---|---|---|
| `BiLSTM+Softmax` | 低 | 弱 | 低 | 数据较多、标签依赖弱、需要快速基线 |
| `BiLSTM-CRF` | 中 | 强 | 中 | 小数据 NER、分词、词性标注、BIO 约束明显 |
| `Transformer+CRF` | 高 | 强 | 较高 | 需要强上下文表示，同时仍要结构约束 |
| `Transformer+Softmax` | 中到高 | 弱 | 中 | 大数据或预训练模型已经能学好边界 |
| 大模型提示或抽取接口 | 高 | 依赖提示和后处理 | 中到高 | 少样本原型、开放抽取、格式变化频繁 |

同一个 NER 任务中：

| 条件 | 推荐方案 | 原因 |
|---|---|---|
| 数据少，实体边界严格，BIO 错误多 | `BiLSTM-CRF` | CRF 能显式利用转移约束 |
| 数据较多，只需要快速上线基线 | `BiLSTM+Softmax` | 实现简单，训练和推理快 |
| 文本长、语义复杂、已有预训练模型 | `Transformer+CRF` | Transformer 提供强表示，CRF 保证标签路径一致 |
| BERT 类模型已经效果很高，非法标签很少 | `Transformer+Softmax` | CRF 的收益可能抵不过复杂度 |

BiLSTM-CRF 的边界在于：当上下文表示已经足够强，或者标签之间几乎没有结构依赖时，CRF 只是在增加计算和调试成本。但在规则强、数据少、标签约束明显的场景里，它仍然实用，尤其适合资源有限的垂直领域抽取任务。

---

## 参考资料

理论来源：

1. [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/handle/20.500.14332/6188)  
   CRF 原始理论来源，适合理解条件概率建模、全局归一化和序列级损失。

2. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)  
   BiLSTM-CRF 用于序列标注的经典论文，适合理解该结构为什么适合 POS、分块和 NER。

3. [Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030/)  
   NER 场景中的代表性应用，适合理解字符特征、BiLSTM 和 CRF 在实体识别中的组合方式。

工程实践来源：

4. [Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](https://docs.pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)  
   PyTorch 官方教程，适合看 BiLSTM-CRF 的可运行实现和 Viterbi 解码细节。

5. [torchcrf Documentation](https://pytorch-crf.readthedocs.io/en/stable/)  
   常用 PyTorch CRF 库文档，适合查 `mask`、`batch_first`、`decode` 和 `reduction` 的接口行为。
