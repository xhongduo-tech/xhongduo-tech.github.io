## 核心结论

Longformer 的核心改动只有一句话：把标准 Transformer 中“每个 token 看所有 token”的全连接注意力，替换成“多数 token 只看邻域，少数关键 token 保留全局视野”的稀疏注意力。

这里有三个结论：

1. 滑动窗口注意力把主成本从 $O(n^2)$ 降到 $O(n \cdot w)$。这里的 $n$ 是序列长度，$w$ 是窗口大小，白话解释就是“每个位置只和附近固定数量的位置交互”。
2. 全局 token 再补上一项 $O(|G| \cdot n)$。这里的 $|G|$ 是全局 token 数量，白话解释就是“只给极少数关键位置保留全图广播能力”。
3. 多层堆叠后，局部信息可以逐层传播；再配合 dilation，白话解释就是“故意隔几个位置再连边”，能在不扩大窗口的前提下放大感受野。

一个最直观的数字例子是：当序列长度 $n=4096$、窗口大小 $w=512$ 时，局部注意力的点积规模约为

$$
4096 \times 512 \approx 2.1 \text{M}
$$

而完整注意力约为

$$
4096^2 \approx 16.8 \text{M}
$$

主计算量大约减少 8 倍。即使再加上 2 个全局 token，额外成本也只有

$$
2 \times 4096 = 8192
$$

数量级仍然接近线性。

| 方案 | 主要连接方式 | 理论复杂度 | 当 $n=4096$ 时的粗略计算量 |
| --- | --- | --- | --- |
| 标准自注意力 | 每个 token 连接全部 token | $O(n^2)$ | $\approx 16.8$M |
| Longformer 局部注意力 | 每个 token 连接窗口内 $w$ 个 token | $O(n \cdot w)$ | $\approx 2.1$M |
| Longformer 局部 + 2 个全局 token | 局部连接 + 全局广播 | $O(n \cdot w + |G|\cdot n)$ | $\approx 2.11$M |

---

## 问题定义与边界

问题很明确：标准自注意力在长文本上太贵。

自注意力，白话解释就是“序列里每个位置都会主动读取别的位置的信息”。它的好处是表达能力强，但代价是注意力矩阵大小为 $n \times n$。当输入从 512 tokens 增长到 4096 tokens 时，显存和计算都会按平方上升。这使得长文档分类、长上下文问答、论文处理、合同分析这类任务很难直接用标准 Transformer。

Longformer 的边界条件也很明确：

1. 大多数有用依赖是局部的，至少在单层里如此。
2. 真正需要全局读写的位置很少，可以人工或规则指定。
3. 任务允许信息通过多层传播，而不是要求任意两个远距离 token 在单层里立即直接交互。

所以它不是“免费得到完整全局注意力”，而是做了结构性假设：局部关系最常见，全局关系最少量、最关键。

可以把它理解成两种视角的组合：

| 机制 | 白话解释 | 解决的问题 |
| --- | --- | --- |
| 滑动窗口 | 每次只看周围固定范围 | 控制主计算量 |
| 全局 token | 给少数位置开“全图广播权限” | 保留跨段信息桥接 |

玩具例子：假设你在读一本 300 页的技术书。大多数时候，你理解一句话只需要看前后几句，这对应局部窗口；但如果你要回答“这本书核心结论是什么”，目录、章节标题、问题描述这些位置需要能汇总全书信息，这对应全局 token。

因此，Longformer 的问题定义不是“完全替代全连接注意力”，而是“在长文本里保留足够的全局能力，同时把主要成本压到线性级别”：

$$
\text{Cost}_{\text{Longformer}} \approx O(n \cdot w) + O(|G| \cdot n)
$$

只要 $w \ll n$ 且 $|G| \ll n$，它就明显比 $O(n^2)$ 更可扩展。

---

## 核心机制与推导

Longformer 的注意力可以拆成三部分：局部连接、全局连接、跨层传播。

### 1. 局部滑动窗口

窗口，白话解释就是“每个位置只连附近若干位置”。若窗口总大小为 $w$，通常表示当前 token 只看左边 $w/2$ 个和右边 $w/2$ 个 token。这样注意力矩阵不再是满矩阵，而是一条沿主对角线展开的“带状矩阵”。

对第 $i$ 个 token，它只计算：

$$
j \in [i-\frac{w}{2},\ i+\frac{w}{2}]
$$

因此每个 token 的连接数从 $n$ 降到约 $w$，总复杂度变成：

$$
O(n \cdot w)
$$

这一步的本质不是近似 softmax，也不是减少 hidden size，而是直接减少边的数量。

### 2. 全局 token

全局 token，白话解释就是“给极少数位置额外特权，让它们能看见所有位置，也被所有位置看见”。典型位置包括：

- 分类任务中的 `[CLS]`
- 问答任务中的问题部分
- 文档中每段开头或标题 token
- 规则指定的关键锚点

如果有 $|G|$ 个全局 token，那么它们与全序列的交互额外带来：

$$
O(|G| \cdot n)
$$

由于 $|G|$ 一般远小于 $n$，这项开销通常可控。

这里要注意，全局 token 的意义不只是“自己看全局”，还包括“让别的 token 能把信息写进它”。这相当于在稀疏图里加了几个高连通度节点，起到桥梁作用。

### 3. 多层传播与 dilation

感受野，白话解释就是“一个位置经过若干层后，理论上能间接接触到多远的信息”。单层局部窗口只能覆盖邻域，但多层叠加后信息会逐层扩散。

如果每层窗口宽度固定为 $w$，层数为 $l$，那么不严格但常用的近似是：

$$
\text{Receptive Field} \approx l \cdot w
$$

如果再引入 dilation $d$，即只在窗口内按步长 $d$ 采样连接，那么感受野可以近似写成：

$$
\text{Receptive Field} \approx l \cdot d \cdot w
$$

dilation，白话解释就是“窗口宽度不变，但连接更稀，跨度更远”。它像卷积中的空洞卷积：不增加窗口参数规模，却让每层覆盖更远位置。

举例：若 $w=512$，层数 $l=12$，某些层使用 dilation 集合 $\{1,2,4\}$，则高层可形成远大于 512 的有效传播范围。虽然这个公式不是精确上界，但足够说明：Longformer 不是靠单层直接连全图，而是靠“局部边 + 少量全局边 + 多层传播”实现长距离依赖。

真实工程例子：长文档问答中，输入可能是“问题 + 文章正文”。常见做法是把问题 token 设为全局，把正文 token 设为局部窗口。这样正文内部按邻域建模，问题 token 则像一个全局收集器，负责从整篇文章吸收信息并把查询意图广播回去。这种设计比全文全连接注意力便宜得多，但仍能保留问答任务所需的跨段对齐能力。

---

## 代码实现

下面给出一个简化版实现，只演示“如何构造 Longformer 风格的注意力 mask”。mask，白话解释就是“告诉模型哪些位置允许互相看，哪些位置必须屏蔽”。

```python
import numpy as np

def build_longformer_mask(seq_len: int, window: int, global_positions=None):
    assert window % 2 == 0
    global_positions = set(global_positions or [])
    half = window // 2

    # 1 表示可见，0 表示不可见
    mask = np.zeros((seq_len, seq_len), dtype=np.int32)

    for i in range(seq_len):
        left = max(0, i - half)
        right = min(seq_len, i + half + 1)
        mask[i, left:right] = 1  # 局部窗口可见

    # 全局 token: 能看所有位置，也被所有位置看见
    for g in global_positions:
        mask[g, :] = 1
        mask[:, g] = 1

    return mask

# 玩具例子：长度 8，窗口 4，位置 0 和 5 为全局 token
m = build_longformer_mask(seq_len=8, window=4, global_positions=[0, 5])

# 局部可见性
assert m[3, 1] == 1
assert m[3, 3] == 1
assert m[3, 6] == 0  # 超出局部窗口，且 6 不是全局 token

# 全局可见性
assert m[0, 7] == 1
assert m[7, 0] == 1
assert m[5, 2] == 1
assert m[2, 5] == 1

def estimate_cost(seq_len: int, window: int, num_global: int):
    dense = seq_len * seq_len
    local = seq_len * window
    global_cost = num_global * seq_len
    return dense, local + global_cost

dense, sparse = estimate_cost(seq_len=4096, window=512, num_global=2)
assert dense == 4096 * 4096
assert sparse == 4096 * 512 + 2 * 4096
assert dense > sparse
```

这个实现省略了真实模型中的分头注意力、padding、batch、softmax 和 CUDA kernel 优化，但它已经准确表达了核心规则。

如果把这个规则映射到工程配置，可以抽象成下面这张表：

| token 类型 | 是否看全序列 | 是否被全序列看见 | 常见用途 |
| --- | --- | --- | --- |
| local token | 否，只看窗口内 | 否，只被窗口内看见 | 正文大多数 token |
| global token | 是 | 是 | `[CLS]`、问题、标题、段首 |

在 Hugging Face 的 Longformer 中，常见接口思路也是类似的：普通 `attention_mask` 控制有效 token，自定义的 global attention mask 标记哪些位置拥有全局权限。主干逻辑并不复杂，难点主要在高效实现局部窗口计算，而不是在数学定义本身。

---

## 工程权衡与常见坑

Longformer 真正难的地方不在公式，而在配置。

第一类权衡是窗口大小。窗口越大，局部建模越强，但显存和算力也越高。窗口太小会让跨句、跨段信息传播太慢；窗口太大又会稀释 Longformer 的效率优势。

第二类权衡是全局 token 的数量与位置。全局 token 太少，模型可能丢失跨段桥接能力；太多，则复杂度会逐步接近全连接注意力，而且训练稳定性和显存占用都会变差。

第三类权衡是 dilation。它能扩大感受野，但实现复杂度更高，而且并非所有开源实现都直接支持。

| 配置 | 优点 | 风险 | 典型场景 |
| --- | --- | --- | --- |
| `window=256` | 显存友好，速度更高 | 局部上下文偏短 | 长文档粗粒度分类 |
| `window=512` | 局部建模更稳，常见默认值 | 显存更高 | QA、摘要、精细分类 |
| `global=[CLS]` | 配置最简单 | 可能不足以覆盖任务关键位 | 文档分类 |
| `global=问题token` | 问答对齐效果更直接 | 需要准确标注问题区域 | 抽取式 QA |
| `global=段首/标题` | 结构化文档更有效 | 规则设计不当会引入噪声 | 论文、报告、合同 |

常见坑主要有四个。

1. 全局 token 放错位置。  
如果问答任务把全局 token 放在无关句子，而不是问题区域或段落锚点，模型就可能无法建立“问题 <-> 证据段”的桥接，最终表现为答案定位漂移。

2. 误以为线性复杂度等于“无限长度免费”。  
Longformer 只是把主项从平方降到线性乘窗口，不代表显存不增长。序列从 4k 增到 16k，显存仍然明显上升。

3. 把 dilation 当作默认能力。  
论文或一些资料会讨论扩张窗口，但实际常用库并不一定直接支持；如果你的实现只支持标准滑窗，就不能在实验结论里默认自己已经用了 dilation。

4. 忽略任务结构。  
Longformer 适合“局部相关性强、关键全局锚点明确”的任务。如果全局线索分散且没有明显锚点，仅靠少量 global token 可能不够，需要额外加段落汇聚、层级编码或检索模块。

---

## 替代方案与适用边界

Longformer 不是唯一的长文本 Transformer，它只是其中最容易理解、也最符合工程直觉的一类：稀疏连接图里保留局部带状结构，再补少量全局点。

下面是几个常见方案的对比：

| 模型 | 核心策略 | 复杂度特点 | 更适合的场景 |
| --- | --- | --- | --- |
| Longformer | 滑动窗口 + 全局 token | $O(n \cdot w + |G|\cdot n)$ | 长文档分类、QA、结构清晰文本 |
| Reformer | LSH 注意力 + 可逆层 | 近似降复杂度并节省激活内存 | 超长序列、强调内存优化 |
| BigBird | 局部 + 全局 + 随机稀疏连接 | 稀疏但保留更丰富图结构 | 需要更强全局连通性的长文本任务 |
| 标准 Transformer | 全连接注意力 | $O(n^2)$ | 中短文本、上下文长度可控任务 |

它们的主要差异不只是“谁更省”，而是“假设什么样的依赖结构更常见”。

Longformer 假设局部依赖是主流，少量位置承担全局桥接责任，所以特别适合：

- 长文档分类
- 问答
- 有明显标题、段落、问题区域的结构化文本

它不那么适合的场景包括：

- 全局信息非常分散，且没有明显关键锚点
- 任意远距离 token 都可能在单层直接强交互
- 任务更适合检索增强而不是纯序列建模

换句话说，Longformer 解决的是“长文本 Transformer 太贵”这个问题，但前提是你愿意接受一种偏结构化的稀疏假设。如果你的任务没有这样的结构，BigBird 的随机边补充可能更稳；如果你的瓶颈主要是激活内存而不是注意力图结构，Reformer 可能更合适。

---

## 参考资料

| 来源 | 重点内容 | 适合人群 |
| --- | --- | --- |
| DeepWiki: Attention Mechanisms | Longformer 中 sliding window、global attention、mask 机制说明 | 想先理解实现结构的读者 |
| Michael Brenndoerfer 博文 | 稀疏注意力与完整注意力的复杂度对比 | 想抓住数量级变化的读者 |
| Kanneganti 等课程/综述 PDF | 默认窗口、分类与 QA 中 global attention 的典型工程配置 | 想把原理落到任务配置的读者 |

1. DeepWiki, “Attention Mechanisms”, 重点是 sliding window 与 global attention 的实现解释。  
   https://deepwiki.com/allenai/longformer/2.1-attention-mechanisms

2. Michael Brenndoerfer, “Longformer: Efficient Attention for Long Documents”, 重点是复杂度分解与稀疏结构直观分析。  
   https://mbrenndoerfer.com/writing/longformer-efficient-attention-long-documents

3. Meghanarao Kanneganti 等资料，重点是窗口配置、分类/QA 中 global attention 的实际使用方式。  
   https://s3.amazonaws.com/na-st01.ext.exlibrisgroup.com/01CALS_USL/storage/alma/EC/BD/81/3E/DD/54/9F/96/CF/2A/77/E5/ED/87/69/A6/KannegantiMeghanarao_Spring2024.pdf

4. Longformer 原始论文：Beltagy, Peters, Cohan, “Longformer: The Long-Document Transformer”。  
   https://arxiv.org/abs/2004.05150
