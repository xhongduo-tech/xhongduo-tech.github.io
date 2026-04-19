## 核心结论

Biaffine 依存句法分析器是一种基于词对打分的神经依存解析模型：它先把句子中每个词编码成向量，再对“依存词-候选头词”两两打分，最后选出每个词的头词和依存关系标签。

它的核心不是规则匹配，而是三个步骤：

| 步骤 | 作用 | 输出 |
|---|---|---|
| 编码 | 把词变成带上下文信息的向量 | 每个词的表示 $x_i$ |
| 双仿射打分 | 对每个词和所有候选头词打分 | 弧分数矩阵、标签分数张量 |
| 树解码 | 保证输出是合法依存树 | 单根、无环、连通的树 |

依存句法分析的目标是找出“谁依赖谁”。例如句子 `ROOT 我 爱 你` 中，`爱` 会分别对 `ROOT / 我 / 你` 打分。假设对依存词 `爱` 的候选头词得分是：

```text
ROOT = 0.2
我   = 2.1
你   = 1.8
```

那么局部预测会得到 `爱 -> 我`。再在这条弧上判断标签：

```text
SBJ = 0.4
OBJ = 1.6
```

标签取 `OBJ`。这只是玩具例子，真实模型会对句中所有词同时做这种两两打分。

核心预测公式是：

$$
head(i) = \arg\max_j s_{arc}(i,j)
$$

$$
rel(i) = \arg\max_r s_{rel}(i, head(i), r)
$$

其中，`head(i)` 是第 `i` 个词的头词，`rel(i)` 是第 `i` 个词和它头词之间的依存关系标签。

句子级两两打分可以理解为一个矩阵：

| 依存词 \ 候选头词 | ROOT | 我 | 爱 | 你 |
|---|---:|---:|---:|---:|
| 我 | 0.8 | mask | 2.5 | 0.3 |
| 爱 | 0.2 | 2.1 | mask | 1.8 |
| 你 | 0.4 | 0.6 | 3.0 | mask |

`mask` 表示非法位置，例如一个词不能把自己当作头词。局部最高分不一定形成合法树，所以工程实现通常还要用 Eisner 算法或 MST 解码器做全局约束。

---

## 问题定义与边界

依存句法分析，白话说就是给句子里的每个词找“上级词”，并标明这条依赖关系的类型。这个“上级词”叫头词，依赖它的词叫依存词。

给定输入：

```text
ROOT 我 爱 你
```

一种可能输出是：

| 依存词 | 头词 | 方向 | 标签 |
|---|---|---|---|
| 我 | 爱 | 爱 -> 我 | SBJ |
| 爱 | ROOT | ROOT -> 爱 | ROOT |
| 你 | 爱 | 爱 -> 你 | OBJ |

这里 `ROOT` 是虚拟根节点，用来连接整棵句法树的中心谓词。上表中，`爱` 是句子的核心谓词，`我` 是主语，`你` 是宾语。

任务边界如下：

| 项目 | 说明 |
|---|---|
| 输入 | 已分词后的句子，通常还包含 `ROOT` |
| 输出 | 每个词的头词索引和依存关系标签 |
| 目标约束 | 单根、无环、连通，每个非 ROOT 词只有一个头词 |
| 非目标任务 | 不负责分词、词性标注、文本生成、语义问答 |

合法依存树必须满足三个基本约束。

| 约束 | 白话解释 | 非法例子 |
|---|---|---|
| 单根 | 只能有一个词直接依赖 `ROOT` | `我 -> ROOT` 且 `爱 -> ROOT` |
| 无环 | 不能互相当对方的上级 | `我 -> 爱` 且 `爱 -> 我` |
| 连通 | 所有词必须属于同一棵树 | `你` 没有连到任何词 |

合法例子：

```text
ROOT -> 爱
爱 -> 我
爱 -> 你
```

非法例子：

```text
我 -> 爱
爱 -> 我
你 -> ROOT
```

第二个结果里 `我` 和 `爱` 形成环，并且 `你` 另接 `ROOT`，不满足合法依存树约束。Biaffine 本身负责给弧打分，树解码器负责把高分弧组合成合法结构。

---

## 核心机制与推导

Biaffine Parser 的输入通常先经过编码器。编码器，白话说就是把每个词转换成包含上下文信息的向量。早期常用 BiLSTM，现在也可以换成 Transformer 或预训练语言模型。记第 `i` 个词的上下文表示为 $x_i$。

接着，模型不会直接用同一个 $x_i$ 同时当头词和依存词，而是用两组 MLP 转换出两个视角。MLP 是多层感知机，可以理解为对向量做非线性变换的小网络。

$$
d_i = MLP_{dep}(x_i)
$$

$$
h_i = MLP_{head}(x_i)
$$

其中，$d_i$ 表示第 `i` 个词作为依存词时的表示，$h_i$ 表示第 `i` 个词作为候选头词时的表示。这样做的原因是，同一个词在不同角色下需要不同特征。例如 `爱` 作为头词时要判断谁是主语、宾语；作为依存词时要判断它是否依赖 `ROOT` 或别的谓词。

弧分数使用双仿射形式：

$$
s_{arc}(i,j) = d_i^T U_{arc} h_j + w_{arc}^T [d_i; h_j] + b_{arc}
$$

这里 $s_{arc}(i,j)$ 表示“第 `i` 个词依赖第 `j` 个词”的分数。双仿射，白话说就是同时包含两类信息：一类是 $d_i^T U h_j$ 这种词对交互项，另一类是 $w^T[d_i;h_j]$ 这种拼接后的线性项。前者建模“这两个词是否适合连边”，后者补充单个词自身的倾向。

选出头词后，还要预测标签。标签分数也可以写成双仿射形式：

$$
s_{rel}(i,j,r) = d_i^T U_r h_j + w_r^T [d_i; h_j] + b_r
$$

其中 $r$ 是依存关系标签，例如 `SBJ`、`OBJ`、`ROOT`、`NMOD`。标签预测不是只看依存词，而是看“依存词、头词、标签”三者的组合。

玩具例子如下。对句子 `ROOT 我 爱 你`，假设模型正在为 `爱` 找头词：

| 候选头词 | 弧分数 |
|---|---:|
| ROOT | 0.2 |
| 我 | 2.1 |
| 你 | 1.8 |

局部选择得到：

$$
head(爱) = 我
$$

再看 `爱 -> 我` 这条弧上的标签分数：

| 标签 | 分数 |
|---|---:|
| SBJ | 0.4 |
| OBJ | 1.6 |

于是：

$$
rel(爱) = OBJ
$$

这个标签在语言学上未必符合“我 爱 你”的真实句法，因为真实结构更可能是 `爱 -> 我` 标为主语、`爱 -> 你` 标为宾语。这里的重点是说明模型如何打分和选择，不是说这个玩具数值一定正确。

训练时，模型通常把任务拆成两个分类问题：

$$
L = L_{arc} + L_{rel}
$$

$L_{arc}$ 是头词分类损失，也就是每个词的正确头词是谁；$L_{rel}$ 是标签分类损失，也就是在正确弧上预测正确关系。训练时用标注树监督模型，解码时再加入树约束，避免局部最高分组合出非法结构。

真实工程例子是中文 Universal Dependencies 依存分析服务。输入一批已经分词的句子后，模型会先批量编码，再一次性得到每个句子的 $n \times n$ 弧分数矩阵和 $n \times n \times R$ 标签分数张量。短句、项目性树库可以用 Eisner 解码；自由语序、非项目性结构较多的场景通常用 MST 或 Chu-Liu-Edmonds 解码。

---

## 代码实现

代码层面通常分成四步：

```text
encode(tokens)
  -> 得到每个词的上下文向量

biaffine_arc(dep_repr, head_repr)
  -> 得到 arc_scores[n, n]

biaffine_rel(dep_repr, head_repr)
  -> 得到 rel_scores[n, n, R]

decode(arc_scores, rel_scores, mask)
  -> 得到合法依存树
```

张量形状如下：

| 名称 | 形状 | 含义 |
|---|---|---|
| `x` | `n × hidden` | 每个词的上下文表示 |
| `dep` | `n × dep_dim` | 依存词视角表示 |
| `head` | `n × head_dim` | 头词视角表示 |
| `arc_scores` | `n × n` | 每个“依存词-候选头词”的弧分数 |
| `rel_scores` | `n × n × R` | 每条候选弧上每个标签的分数 |
| `mask` | `n × n` | 标记非法弧，例如自环和 padding |

下面是一个可运行的最小 Python 例子。它没有训练神经网络，只演示 Biaffine Parser 后半段的核心流程：弧矩阵、标签张量、mask、局部解码和断言检查。

```python
import numpy as np

tokens = ["ROOT", "我", "爱", "你"]
labels = ["ROOT", "SBJ", "OBJ"]
n = len(tokens)
R = len(labels)

# arc_scores[dep, head]：第 dep 个词依赖第 head 个词的分数
arc_scores = np.array([
    [-1e9, -1e9, -1e9, -1e9],  # ROOT 不需要头词
    [0.1,  -1e9, 2.5,  0.3],   # 我 -> 爱
    [3.0,   2.1, -1e9, 1.8],   # 爱 -> ROOT
    [0.2,   0.6, 3.2, -1e9],   # 你 -> 爱
])

# rel_scores[dep, head, rel]：候选弧上的标签分数
rel_scores = np.zeros((n, n, R))
rel_scores[1, 2] = [0.1, 2.0, 0.4]  # 我 -> 爱: SBJ
rel_scores[2, 0] = [3.0, 0.1, 0.1]  # 爱 -> ROOT: ROOT
rel_scores[3, 2] = [0.1, 0.3, 2.5]  # 你 -> 爱: OBJ

def local_decode(arc_scores, rel_scores):
    heads = [-1] * n
    rels = [None] * n

    for dep in range(1, n):  # 跳过 ROOT
        head = int(np.argmax(arc_scores[dep]))
        rel = int(np.argmax(rel_scores[dep, head]))
        heads[dep] = head
        rels[dep] = labels[rel]

    return heads, rels

heads, rels = local_decode(arc_scores, rel_scores)

assert heads[1] == 2          # 我 -> 爱
assert heads[2] == 0          # 爱 -> ROOT
assert heads[3] == 2          # 你 -> 爱
assert rels[1] == "SBJ"
assert rels[2] == "ROOT"
assert rels[3] == "OBJ"

print(list(zip(tokens, heads, rels)))
```

真实实现会比这个例子多几个关键细节。

第一，`ROOT` 本身不能有头词。第二，普通词不能指向自己，所以对角线要 mask 掉。第三，padding 位置既不能作为依存词，也不能作为候选头词。第四，如果任务要求输出严格合法树，不能只用 `argmax`，要把弧分数交给 Eisner 或 MST 解码器。

Python 风格伪代码如下：

```python
def parse_sentence(tokens):
    x = encoder(tokens)                         # [n, hidden]

    dep_arc = mlp_arc_dep(x)                    # [n, arc_dim]
    head_arc = mlp_arc_head(x)                  # [n, arc_dim]
    dep_rel = mlp_rel_dep(x)                    # [n, rel_dim]
    head_rel = mlp_rel_head(x)                  # [n, rel_dim]

    arc_scores = biaffine_arc(dep_arc, head_arc)        # [n, n]
    rel_scores = biaffine_rel(dep_rel, head_rel)        # [n, n, R]

    arc_scores = mask_invalid_arcs(arc_scores)
    heads = tree_decode(arc_scores, algorithm="mst")    # 或 "eisner"

    rels = []
    for dep, head in enumerate(heads):
        if dep == 0:
            rels.append(None)
        else:
            rels.append(argmax(rel_scores[dep, head]))

    return heads, rels
```

这段流程里，弧预测决定“连哪条边”，标签预测决定“这条边是什么关系”。工程上必须保证标签是在最终选中的弧上取，而不是对所有标签张量随便做全局最大值。

---

## 工程权衡与常见坑

只做局部 `argmax` 是最容易犯的错误。局部最高分只保证每个词自己选到高分头词，不保证整句结构合法。

错误例子：

```text
我 -> 爱
爱 -> 我
你 -> 爱
```

这里 `我` 和 `爱` 互相指向，形成环。每条弧单独看都可能分数很高，但整体不是一棵树。树解码器的作用就是在全局范围内选择总分高、同时满足结构约束的弧集合。

常见问题如下：

| 问题现象 | 原因 | 后果 | 规避方法 |
|---|---|---|---|
| 多个词指向 ROOT | 局部 `argmax` 没有限制单根 | 句子有多个根 | 用 MST/Eisner 加单根约束 |
| 两个词互相指向 | 没有全局无环约束 | 输出不是树 | 解码时禁止环 |
| padding 被选为头词 | mask 不完整 | 模型学到无意义结构 | mask 掉 padding 行列 |
| ROOT 指向 ROOT | 忘记 mask ROOT 自环 | 根节点结构错误 | 固定 ROOT 不作为依存词 |
| 标签取错弧 | 弧和标签独立乱取 | 头词对但关系错 | 只在选中弧上取标签 |
| 长句显存暴涨 | 弧分数是 $n \times n$ | batch 变小或 OOM | 分桶、截断、动态 batch |

训练时必须 mask 的项包括：

| mask 项 | 原因 |
|---|---|
| padding 作为依存词 | padding 不是真实词 |
| padding 作为候选头词 | 真实词不能依赖空位置 |
| 普通词自环 | 一个词不能依赖自己 |
| ROOT 自环 | `ROOT -> ROOT` 没有句法意义 |
| ROOT 作为依存词 | 通常 ROOT 不需要头词 |
| 非法候选头 | 某些任务设置下需要额外限制 |

长句是另一个现实问题。因为每个词都要和每个候选头词打分，弧矩阵大小是 $n^2$。如果句长从 50 增加到 200，弧分数位置数从 2500 增加到 40000，显存和计算都会明显增加。标签张量更大，是 $n \times n \times R$，其中 $R$ 是标签数。

真实工程里常见做法是按句长分桶。分桶，白话说就是把长度接近的句子放进同一个 batch，减少 padding 浪费。线上服务还可能设置最大句长，超长文本先切句，再逐句解析。

还有一个常见误解是“Biaffine 分数最高的边就一定应该保留”。依存解析是结构预测，不是独立分类。一个低一点的局部弧分数，可能让整棵树合法且总分更高；一个局部最高分，可能引入环或多根。解码器不是可有可无的后处理，而是把局部打分转换成合法结构的关键步骤。

---

## 替代方案与适用边界

Biaffine 是强基线，但不是唯一方案。它适合“句子已分词、目标是预测依存树、训练数据有头词和标签标注”的场景。是否选择 Eisner 或 MST，主要取决于树库和语言现象。

项目性，白话说就是依存弧画在句子上方时不交叉。英语短句经常可以用项目性树近似，例如：

```text
ROOT -> saw
saw -> I
saw -> her
her -> yesterday
```

这种结构多数弧不交叉，用 Eisner 算法比较合适。Eisner 是动态规划解码算法，能在项目性约束下找高分树。

非项目性，白话说就是依存弧可能交叉。中文、捷克语、德语等自由语序或长距离依赖较多的场景里，强制项目性可能删掉正确结构。例如中文句子：

```text
这本书 我 昨天 看完了
```

如果标注体系允许话题、时间状语和谓词之间出现更灵活的依存结构，非项目性解码通常更稳妥。MST 或 Chu-Liu-Edmonds 把句子看成有向图，在图上找最大生成树，不要求项目性。

对比如下：

| 方法 | 适用场景 | 优点 | 限制 |
|---|---|---|---|
| Biaffine + Eisner | 项目性树库、英语短句、弧交叉少 | 解码结构清晰，满足项目性约束 | 不能输出非项目性树 |
| Biaffine + MST | 非项目性树库、自由语序语言 | 能处理弧交叉，适用面更广 | 解码约束较弱，不显式建模项目性 |
| 规则/传统图模型 | 数据少、规则稳定、领域封闭 | 可解释，部署简单 | 覆盖率低，迁移困难 |
| 其他神经依存解析器 | 需要结合预训练模型、转移系统或更复杂约束 | 可以利用更强编码器或动作序列建模 | 实现复杂，错误传播或成本更高 |

规则方法适合非常窄的领域。例如某个日志系统里句式固定，只有“服务 调用 接口 失败”这类结构，手写规则可能够用。但开放文本里，规则会很快失效。

转移式依存解析器也是常见替代方案。转移式，白话说就是从左到右读句子，通过一系列动作逐步构造依存树。它速度快，但动作错误可能传播。Biaffine 属于图式解析器，先给所有词对打分，再统一解码，整体更适合做强基线。

低资源任务中，Biaffine 也不一定最优。如果标注数据很少，模型可能学不到稳定的词对结构，需要引入多语言迁移、预训练模型、规则约束或半监督数据。对于需要强语义理解的任务，依存树本身也只是中间结构，不能替代语义角色标注、信息抽取或生成式理解。

---

## 参考资料

1. [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734)
2. [Stanford’s Graph-based Neural Dependency Parser at the CoNLL 2017 Shared Task](https://aclanthology.org/K17-3002/)
3. [Three New Probabilistic Models for Dependency Parsing: An Exploration](https://aclanthology.org/C96-1058/)
4. [Non-Projective Dependency Parsing using Spanning Tree Algorithms](https://aclanthology.org/H05-1066/)
5. [SuPar Biaffine Dependency Parser Documentation](https://parser.yzhang.site/models/dep/biaffine.html)
6. [SuPar Affine and Biaffine Modules](https://parser.yzhang.site/modules/affine.html)
7. [SuPar GitHub Repository](https://github.com/yzhangcs/parser)

如果只看 1 篇，先看 Dozat 和 Manning 的 Biaffine 论文；如果想补工程实现，再看 SuPar 的 Biaffine 模型文档和源码。
