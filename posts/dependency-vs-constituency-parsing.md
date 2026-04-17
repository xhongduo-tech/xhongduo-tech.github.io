## 核心结论

成分句法（constituency parsing）把句子分析为“短语怎么一层层组成整句”的树；短语就是能作为一个整体参与更大结构组合的词组。依存句法（dependency parsing）把句子分析为“哪个词支配哪个词”的有向树；支配就是一个词作为另一个词的核心头词。

两者解决的是同一个任务：给句子补上结构。但它们观察结构的角度不同。

- 成分句法强调层级。它适合回答“这个句子先组成了哪些短语，再组成了哪些更大短语”。
- 依存句法强调关系。它适合回答“谓词是谁、主语是谁、宾语挂在哪个词下面”。

对初学者，最重要的判断标准不是“谁更先进”，而是“你的下游任务需要哪种结构”：

| 维度 | 成分句法 | 依存句法 |
| --- | --- | --- |
| 结构单位 | 短语与子树 | 词与词的弧 |
| 典型输出 | `S -> NP VP` 这样的树 | `eats -> she`, `eats -> fish` 这样的弧 |
| 典型算法 | CKY、chart parsing | arc-factored、MST、shift-reduce |
| 优势 | 层级清晰、解释性强 | 表示紧凑、部署快 |
| 常见场景 | 语法研究、教育、精细结构分析 | 信息抽取、关系建模、在线服务 |

玩具例子用句子 `she eats a fish`。成分句法会得到类似：

- `NP -> she`
- `V -> eats`
- `NP -> Det N`
- `VP -> V NP`
- `S -> NP VP`

依存句法会得到类似：

- `eats -> she`，主语关系
- `eats -> fish`，宾语关系
- `fish -> a`，限定词关系

如果只想抓“谁做了什么”，依存树通常更直接。如果想保留“名词短语、动词短语、从句”这些层级边界，成分树更合适。

---

## 问题定义与边界

成分解析的问题定义是：给定词串 $x_1,\dots,x_n$ 和一个上下文无关文法，找出一棵合法的短语结构树。上下文无关文法可以理解为“一组短语如何组合”的规则系统。为了让 CKY 工作，通常要求文法先转成 Chomsky Normal Form，简称 CNF，意思是每条产生式最多只含两个非终结符，或一个终结符。

CKY 的核心布尔递推是：

$$
P[i,j,A] = \bigvee_{A \to BC} \bigvee_{i < k < j} \left(P[i,k,B] \land P[k,j,C]\right)
$$

这里 $P[i,j,A]$ 表示：区间 $[i,j)$ 这段词是否能由非终结符 $A$ 生成。

依存解析的问题定义是：给每个词分配一个头词，并给每条弧分配一个依赖类型，最后形成一棵树。头词可以理解为“这个词语法上归谁管”。依存树通常引入一个虚拟根节点 `ROOT`，保证整句只有一个根。

arc-factored 模型把整棵树分解为若干独立弧的和：

$$
\text{Score}(x,y)=\sum_{(i,l,j)\in A_y}\text{Score}(i,l,j,x)
$$

这里 $A_y$ 是树 $y$ 里的弧集合，$(i,l,j)$ 表示从头词 $i$ 指向修饰词 $j$，依赖类型是 $l$。直白地说，就是“先给每条边打分，再把整棵树的边分数加起来”。

同一句 `she eats a fish`，两种表示如下。

成分表示关注“块”：

- `[S [NP she] [VP [V eats] [NP [Det a] [N fish]]]]`

依存表示关注“边”：

- `ROOT -> eats`
- `eats -> she`
- `eats -> fish`
- `fish -> a`

边界也要说明清楚。

- 成分句法不天然告诉你“中心词是谁”，除非再做 head rule，即人工规定“一个短语里哪个词算核心”。
- 依存句法不天然保留全部中间短语节点，所以像 `VP`、`NP` 这样的显式层级可能丢失。
- 两者都属于结构化预测。结构化预测就是“输出不是一个标签，而是整棵树”。

---

## 核心机制与推导

先看 CKY。它是典型的自底向上动态规划。动态规划的意思是：先解小区间，再复用小区间去构造大区间。

对 `she eats a fish`，设下标区间为：

- `0: she`
- `1: eats`
- `2: a`
- `3: fish`
- `4: 句尾位置`

词法规则先填对角线：

- `(0,1)` 可放 `NP`
- `(1,2)` 可放 `V`
- `(2,3)` 可放 `Det`
- `(3,4)` 可放 `N`

然后组合长度为 2 的区间：

- `(2,4)` 由 `Det N` 组成 `NP`

再组合长度为 3 的区间：

- `(1,4)` 由 `V NP` 组成 `VP`

最后组合整句：

- `(0,4)` 由 `NP VP` 组成 `S`

这就是 CKY 的搜索过程。它不是“从左到右猜”，而是“枚举所有切分点 $k$，检查左半边和右半边能否拼成更大的符号”。

下面给出这个玩具例子的 CKY 表。行列表示 span，单元格中是能覆盖该 span 的非终结符。

| Span | 内容 | 非终结符 |
| --- | --- | --- |
| `(0,1)` | `she` | `NP` |
| `(1,2)` | `eats` | `V` |
| `(2,3)` | `a` | `Det` |
| `(3,4)` | `fish` | `N` |
| `(2,4)` | `a fish` | `NP` |
| `(1,4)` | `eats a fish` | `VP` |
| `(0,4)` | `she eats a fish` | `S` |

如果把三角表写得更显式，可以看到它是从对角线一路推到顶端 `(0,4)` 的。顶端出现 `S`，表示整句可被该文法接受。

再看依存解析。arc-factored 的核心假设是“每条弧可以独立打分”。例如：

$$
\text{Score}(\text{eats}\to \text{she}) = f(h_{\text{eats}}, h_{\text{she}})
$$

$$
\text{Score}(\text{eats}\to \text{fish}) = f(h_{\text{eats}}, h_{\text{fish}})
$$

其中 $h_i$ 是词的向量表示，常由 BiLSTM 或 Transformer 编码器得到。白话说，就是先把每个词变成向量，再用一个打分函数判断“这个词当另一个词的头，有多合理”。

整棵树的分数是这些局部分数之和，所以如果模型算出：

- `Score(eats -> she) = 3.1`
- `Score(eats -> fish) = 2.9`
- `Score(fish -> a) = 1.8`

那么总分就是 $3.1+2.9+1.8$，再配合树约束搜索最优树。

shift-reduce 则是另一条路线。它不在整张图上全局搜索，而是维护一个状态：

- `stack`：已经读入、等待组合的元素
- `buffer`：还没处理的输入词序列

常见动作包括：

- `Shift`：从 `buffer` 取一个词压入 `stack`
- `Left-Arc`：建立“栈顶词依赖次栈顶词”的弧
- `Right-Arc`：建立“次栈顶词依赖栈顶词”的弧
- `Reduce`：把已完成的词从栈中移除

它的优点是快，很多实现接近线性时间。问题是，一步做错，后面状态就偏了，这叫错误传播。dynamic oracle 的作用是：即使当前状态已经偏离黄金路径，也继续告诉模型“在这个错误状态下，哪些动作仍然是最优或次优的”。这能减轻训练时“只会走标准路径、不会救场”的问题。

真实工程例子可以看信息抽取流水线。假设一句话是：

`OpenAI released a new model in March.`

如果你的目标是抽取事件三元组，依存结构更直接：

- 谓词：`released`
- 主体：`OpenAI`
- 客体：`model`
- 时间修饰：`in March`

如果你的目标是做教学型语法分析或复杂从句边界检测，成分树更有价值，因为你能显式看到 `NP`、`VP`、`PP` 的嵌套边界。

---

## 代码实现

下面先给一个最小可运行的 CKY 示例。它不做概率打分，只判断某个句子能否由文法生成，并验证 `she eats a fish` 是否能推出 `S`。

```python
from collections import defaultdict

def cky_parse(words, lexicon, binary_rules):
    n = len(words)
    table = [[set() for _ in range(n + 1)] for _ in range(n)]

    # 对角线：词到词性/短语
    for i, word in enumerate(words):
        for lhs in lexicon.get(word, []):
            table[i][i + 1].add(lhs)

    # 按 span 长度自底向上填表
    for span_len in range(2, n + 1):
        for i in range(0, n - span_len + 1):
            j = i + span_len
            for k in range(i + 1, j):
                left_set = table[i][k]
                right_set = table[k][j]
                for B in left_set:
                    for C in right_set:
                        for A in binary_rules.get((B, C), []):
                            table[i][j].add(A)

    return table

lexicon = {
    "she": {"NP"},
    "eats": {"V"},
    "a": {"Det"},
    "fish": {"N"},
}

binary_rules = defaultdict(set)
binary_rules[("Det", "N")].add("NP")
binary_rules[("V", "NP")].add("VP")
binary_rules[("NP", "VP")].add("S")

words = ["she", "eats", "a", "fish"]
table = cky_parse(words, lexicon, binary_rules)

assert "NP" in table[0][1]
assert "NP" in table[2][4]
assert "VP" in table[1][4]
assert "S" in table[0][4]
print("CKY parse success")
```

这段代码对应的就是经典三重循环：

1. 枚举跨度 `span_len`
2. 枚举起点 `i`，得到终点 `j`
3. 枚举切分点 `k`

如果要做概率版 PCFG，PCFG 就是“每条文法规则带概率的上下文无关文法”，则表中不再存集合，而是存“当前最优分数和回溯指针”。

再给一个简化的 shift-reduce 状态机示例。这里不训练模型，只演示状态更新。

```python
class State:
    def __init__(self, words):
        self.stack = []
        self.buffer = list(words)
        self.arcs = []

    def shift(self):
        assert self.buffer, "buffer 为空，不能 Shift"
        self.stack.append(self.buffer.pop(0))

    def right_arc(self, label="dep"):
        assert len(self.stack) >= 2, "栈中至少需要两个元素"
        head = self.stack[-2]
        dep = self.stack[-1]
        self.arcs.append((head, label, dep))
        self.stack.pop()

    def left_arc(self, label="dep"):
        assert len(self.stack) >= 2, "栈中至少需要两个元素"
        head = self.stack[-1]
        dep = self.stack[-2]
        self.arcs.append((head, label, dep))
        self.stack.pop(-2)

state = State(["she", "eats", "fish"])
state.shift()   # stack: [she]
state.shift()   # stack: [she, eats]
state.left_arc("nsubj")  # eats -> she
state.shift()   # stack: [eats, fish]
state.right_arc("obj")   # eats -> fish

assert ("eats", "nsubj", "she") in state.arcs
assert ("eats", "obj", "fish") in state.arcs
print(state.arcs)
```

工程里真正的 shift-reduce 解析器会在每一步调用分类器选动作，伪代码如下：

```python
while not finished(state):
    valid_actions = get_valid_actions(state)
    action = model.select_best(state, valid_actions)
    state = transit(state, action)
```

这里的 `model.select_best` 往往由神经网络实现，输入是当前 `stack`、`buffer` 中若干位置的表示，输出每个动作的分数。

---

## 工程权衡与常见坑

CKY、arc-factored、shift-reduce 都能做句法分析，但工程属性差异很大。

| 方法 | 时间复杂度 | 结构约束 | 优势 | 限制 |
| --- | --- | --- | --- | --- |
| CKY | $O(n^3|G|)$ 左右 | 需 CNF 或等价变形 | 结构完整、短语边界清晰 | 慢，文法预处理复杂 |
| arc-factored | 常配合全局解码，训练可批量化 | 默认弧独立 | 易和 Transformer 结合 | 忽略高阶依赖 |
| shift-reduce | 近线性 | 依赖动作系统设计 | 快，适合低延迟服务 | 错误传播明显 |

几个最常见的坑如下。

第一，CKY 不是“拿来就能跑”。它要求文法先二叉化。二叉化就是把一个规则右侧拆成最多两个符号。例如 `VP -> V NP PP` 不能直接进标准 CKY，必须改写成类似：

- `VP -> V X1`
- `X1 -> NP PP`

如果你忘了这一步，代码会看起来正确，但很多句子永远组不起来。

第二，arc-factored 的“弧独立”是假设，不是事实。真实语言里，两个修饰词之间常常有兄弟关系、路径关系、投射性约束。投射性可以理解为“树边不能在词序上交叉”的约束。只做一阶弧打分时，模型可能在局部上都觉得合理，但全局结构仍不自然。所以工程里常加 biaffine scorer、二阶特征、MST 解码或 reranking。biaffine 可以理解为“一种专门为头词和修饰词配对打分设计的双线性层”。

第三，shift-reduce 的速度来自贪心，贪心就容易把早期错误放大。比如主语一开始挂错，后面很多合法动作空间都会变化。解决办法通常有：

- dynamic oracle
- beam search
- self-training
- 预训练编码器增强状态表示

真实工程例子是在线 NLP 服务。假设你要处理每秒几千条搜索日志或客服对话，延迟预算可能只有几十毫秒。这时 CKY 的全局 chart 搜索通常太重，shift-reduce 或图依存解析更现实。相反，如果你做的是语言学研究平台、教育产品或需要展示句子成分树的可视化工具，CKY 或神经成分解析更合适，因为解释性比吞吐量更重要。

---

## 替代方案与适用边界

如果按“任务目标”来选，通常比按“算法流派”来选更稳妥。

| 用途 | 推荐范式 | 原因 |
| --- | --- | --- |
| 语法教学、语言学分析 | 成分句法 + CKY/神经 chart | 需要短语层级和可解释树 |
| 关系抽取、事件抽取 | 依存句法 | 头词关系直接可用 |
| 在线服务、低延迟推理 | shift-reduce 依存或转移式成分 | 推理快，资源占用低 |
| 批量训练、结合预训练模型 | graph-based arc-factored/biaffine | 容易并行、特征强 |
| 需要高阶一致性 | higher-order dependency / reranking | 一阶弧独立不够 |

可以把选择逻辑简化成下面这张表：

| 需求判断 | 更合适的选择 |
| --- | --- |
| 我需要 `NP/VP/CP` 这样的短语边界 | 成分句法 |
| 我需要直接知道主语、宾语、修饰关系 | 依存句法 |
| 我更在乎推理速度 | shift-reduce |
| 我更在乎全局结构与解释 | CKY 或 chart parsing |
| 我已有 Transformer 编码器并想批量训练 | arc-factored / biaffine |

还要看到边界。

- 成分句法并不落后。它只是更偏结构表达，不一定适合所有线上场景。
- 依存句法也不是“天然更简单”。如果任务依赖从句边界、协调结构、长距离嵌套，单纯依存弧未必足够。
- 现代系统里，神经网络更多是在“学特征”，不是替代结构约束本身。也就是说，Transformer 常负责把词表示好，CKY、MST、transition system 仍负责“树怎么合法地建出来”。

一个实用判断是：先问下游任务到底消费什么。如果下游最终只读取“谓词-论元”关系，依存解析一般性价比更高。如果下游要展示句法树、做教育解释或研究人类可读语法，成分解析更自然。

---

## 参考资料

- Wikipedia: CYK algorithm。介绍 CKY/CYK 的动态规划过程、CNF 前提与三角表思路。https://en.wikipedia.org/wiki/CYK_algorithm
- Stanford NLP: Shift-Reduce Parser。介绍转移式解析器及其工程实现背景。https://nlp.stanford.edu/software/srparser.html
- McDonald et al. / dependency parsing survey。介绍 graph-based、arc-factored 依存解析的经典建模思想。https://citeseerx.ist.psu.edu/document?doi=af68066315423186a89704bf18a0d796e068cc52&repid=rep1&type=pdf
- NLP with Deep Learning: Parsing overview。概述成分句法与依存句法的基本差异及神经网络结合方式。https://www.nlpwithdeeplearning.com/nlpqb/parsing.html
