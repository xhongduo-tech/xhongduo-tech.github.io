## 核心结论

稀疏注意力的核心目标，是在尽量保留 Transformer 长程建模能力的前提下，避免标准自注意力对所有 token 两两计算相关性。这里的 token 可以先理解为“序列被模型切分后的最小处理单元”，在文本中通常对应子词、标点或特殊符号。

标准自注意力的时间和显存复杂度通常写作 $O(n^2)$。原因很直接：长度为 $n$ 的序列中，每个位置都可能与其余 $n$ 个位置建立连接，因此需要构造并处理一个 $n \times n$ 的注意力分数矩阵。

局部+全局混合稀疏模式把这件事改写成两部分：

$$
O(n^2) \rightarrow O(nw + ng)
$$

其中：

- $n$ 是序列长度
- $w$ 是局部窗口大小
- $g$ 是全局 token 的数量

当 $w \ll n$ 且 $g \ll n$ 时，整体代价会从平方级下降到近似线性级。

这类设计的正式含义是：

1. 大多数普通 token 只与局部邻域中的 token 交互。
2. 少数被指定为 global 的 token 与全序列交互。
3. 跨窗口的信息主要通过全局 token 或多层局部传播完成。

可以把它先记成一句话：普通 token 负责近距离建模，全局 token 负责远距离通信。若只为帮助初学者建立直觉，可以把它类比成“局部讨论 + 少量广播节点”的通信系统，但后续分析仍以正式定义为准。

| 方案 | 每个普通 token 可访问范围 | 是否存在显式全局通道 | 单层复杂度 |
|---|---|---|---|
| Dense attention | 全部 $n$ 个 token | 不需要 | $O(n^2)$ |
| 纯局部窗口 | 窗口内约 $w$ 个 token | 没有 | $O(nw)$ |
| 局部 + 全局 | 窗口内约 $w$ 个 token，加上 $g$ 个全局 token | 有 | $O(nw + ng)$ |

从机制上看，局部窗口保留邻近依赖，全局 token 维持长程路径。Longformer 就属于这一类显式稀疏设计。Performer 则不是显式删边，而是通过核近似把全局 softmax attention 改写为线性形式，因此它与“局部+全局”是两条不同路线。

---

## 问题定义与边界

问题定义可以写得很明确：当输入序列足够长时，dense attention 的 $O(n^2)$ 成本会成为训练或推理瓶颈。长文档建模、代码仓库级上下文、长时间序列、多轮对话拼接，都会遇到这个问题。

这里的“长”不是某个固定阈值，而是“当前硬件预算下，注意力矩阵已经成为主要成本来源”的长度。比如在 16K、32K，甚至更长上下文中，单层 dense attention 的显存与计算都会迅速膨胀。

先区分三个术语：

| 术语 | 定义 | 初学者可先这样理解 |
|---|---|---|
| 局部注意力 | 每个位置只连接固定邻域中的 token | 只看附近上下文 |
| 全局注意力 | 某些特殊位置与全序列建立连接 | 少数节点有“广播权” |
| attention mask | 指定哪些位置允许相互注意的矩阵 | 一张“能不能看见彼此”的规则表 |

所谓“局部+全局混合设计”，就是把这两类连接规则写进同一个注意力 mask 中。这样做不是在改变 attention 的基本公式，而是在改变图结构：哪些边存在，哪些边被屏蔽。

以一个 16K token 的文档为例：

- dense attention 需要处理约 $16{,}000^2 = 256{,}000{,}000$ 个位置对
- 如果局部窗口大小 $w=512$，再设置少量全局 token，例如 $g=16$
- 单层可访问边的量级变为 $n(w+g)$，即约 $16{,}000 \times 528$

这个数量仍然不小，但与完整的 $n^2$ 相比已经下降明显，且显存访问也更规律。

| 维度 | Dense attention | 局部+全局稀疏 attention |
|---|---|---|
| 普通 token 访问范围 | 全部位置 | 局部窗口 + 全局节点 |
| 长距离依赖路径 | 单层直接存在 | 依赖全局节点或多层传播 |
| 显存压力 | 高 | 通常显著降低 |
| 对超长输入的扩展性 | 差 | 更好 |
| 主要调参对象 | 头数、层数、隐藏维度 | 窗口大小、全局 token 选择、mask 规则 |

边界也必须说清楚。

稀疏注意力不是“更先进的 dense attention”，而是一种带有明确偏置的结构化近似。它默认假设：大多数有用交互是局部的，少量节点足以承担全局通信。如果任务本身需要大量任意位置之间的细粒度匹配，例如复杂表格推理、跨页多跳证据绑定、远距离精确对齐，而你又没有选对 global token，那么性能可能不如 dense attention。

因此，这类方法解决的是“长序列上的成本问题”，代价是“连接图不再完全充分”。它的适用边界取决于任务结构是否允许这种删边。

---

## 核心机制与推导

设序列长度为 $n$，隐藏维度为 $d$。对第 $i$ 个位置，定义：

- 局部邻域集合为 $N_i$
- 全局 token 集合为 $G$
- Query、Key、Value 分别为 $Q_i, K_j, V_j$

标准 attention 的核心分数仍然来自缩放点积：

$$
s_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}}
$$

稀疏设计并没有改变这个打分公式，它改变的是“哪些 $j$ 对当前 $i$ 可见”。

### 1. 局部部分

对普通 token，通常只保留窗口内连接：

$$
A_{ij}^{local} \propto \exp\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right), \quad j \in N_i
$$

如果窗口半径记为 $r$，则窗口大小可写为：

$$
w = 2r + 1
$$

此时

$$
N_i = \{j \mid |i-j| \le r\}
$$

边界位置需要截断。例如当 $i=0$ 时，不存在左侧邻居，因此真实邻域是：

$$
N_0 = \{0,1,\dots,r\}
$$

### 2. 全局部分

若某些位置属于全局集合 $G$，则通常有两种常见规则：

1. 全局 token 能看见全序列。
2. 全序列也能看见全局 token。

这对应双向全局桥接。形式上可写为：

$$
A_{ij}^{global} \propto \exp\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right), \quad j \in G
$$

以及对全局位置本身：

$$
A_{ij}^{global-row} \propto \exp\left(\frac{Q_i K_j^\top}{\sqrt{d}}\right), \quad i \in G,\ \forall j
$$

这两条规则一起决定了“全局节点既能读全局，也能被全局读取”。

### 3. 用 mask 统一表达

工程实现通常不显式分开写局部公式和全局公式，而是构造一个掩码矩阵：

$$
M \in \{0, -\infty\}^{n \times n}
$$

定义为：

$$
M_{ij} =
\begin{cases}
0, & \text{如果 } j \in N_i \text{ 或 } i \in G \text{ 或 } j \in G \\
-\infty, & \text{否则}
\end{cases}
$$

然后把它加到原始 attention logits 上：

$$
S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + M_{ij}
$$

再做 softmax：

$$
\alpha_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^{n} \exp(S_{ik})}
$$

于是：

- 允许的边保留原始分数
- 不允许的边由于加上 $-\infty$，softmax 后权重为 0

最终输出仍是标准形式：

$$
O_i = \sum_{j=1}^{n}\alpha_{ij} V_j
$$

所以稀疏注意力的本质不是换掉 Transformer，而是把注意力图从完全图改成稀疏图。

### 4. 为什么复杂度是 $O(nw + ng)$

若每个普通 token 只访问约 $w$ 个局部邻居，再加上 $g$ 个全局 token，那么每一行可见位置数近似为：

$$
w + g
$$

因此总可见边数约为：

$$
n(w+g)=nw+ng
$$

这就是常见复杂度估计的来源。更严格地说：

- 序列边界会让一部分行的窗口更短
- 全局 token 自身的整行可能是全连接
- 不同实现对普通 token 是否一定能访问全部 global token 也略有差异

但在主流设定下，上式是足够准确的工程量级估计。

### 5. 一个 8 个 token 的玩具例子

令序列长度 $L=8$，窗口半径 $r=1$，于是每个普通 token 理论上最多看见自己及左右各 1 个位置。再设第 0 个 token 为全局 token，即：

$$
G=\{0\}
$$

则允许连接可写成下表：

| 位置 | 局部可见位置 | 额外全局可见位置 |
|---|---|---|
| 0 | 全部位置（因为是 global） | 全部位置 |
| 1 | 0,1,2 | 已含 0 |
| 2 | 1,2,3 | 0 |
| 3 | 2,3,4 | 0 |
| 4 | 3,4,5 | 0 |
| 5 | 4,5,6 | 0 |
| 6 | 5,6,7 | 0 |
| 7 | 6,7 | 0 |

如果没有全局节点，那么位置 7 想影响位置 2，至少需要多层逐步传播。加入全局节点后，路径可以缩短为：

$$
7 \rightarrow 0 \rightarrow 2
$$

这就是全局 token 的主要价值：它把“只能逐层扩散”的局部传播，改造成“存在快捷路径”的结构。

### 6. 常见稀疏模式对比

| 模式 | 连接定义 | 主要优点 | 主要风险 | 常见场景 |
|---|---|---|---|---|
| 局部窗口 | 只保留邻域连接 | 简单、稳定、易实现 | 长距离依赖弱 | 文本、语音、时间序列 |
| 带状注意力 | 保留主对角线附近若干带 | 存储和索引更规则 | 仍然缺少跨区桥接 | 与局部窗口近似 |
| 跨步/膨胀稀疏 | 间隔采样远处位置 | 在不增密的情况下扩大感受野 | 采样模式不当会漏信息 | 极长序列探索 |
| 全局 token | 少量节点全连接 | 长程路径短、任务可定制 | 全局点选错会断路 | QA、分类、摘要 |

Longformer 的核心思想，就是把 sliding window 与 task-motivated global attention 组合起来：普通 token 用局部窗口，问题 token、CLS、特殊标记或任务相关位置用 global attention。多层堆叠后，局部模式得到保留，远距离信息则通过全局节点快速流动。

---

## 代码实现

工程上最关键的部分不是公式本身，而是如何正确构造 mask。常见实现流程是：

1. 先生成局部窗口的带状 mask。
2. 再把全局 token 对应的行和列打开。
3. 最后在 softmax 之前把非法位置填成极小值。

下面给出一个可以直接运行的 Python 示例。它做三件事：

1. 构造局部+全局的布尔 mask。
2. 基于 NumPy 实现一次可运行的 masked attention。
3. 用断言检查 mask 方向和输出形状，避免“代码看起来对、实际跑不通”。

```python
import math
import numpy as np


def build_sparse_mask(seq_len: int, window_radius: int, global_indices=None) -> np.ndarray:
    """
    返回 shape = (seq_len, seq_len) 的布尔矩阵。
    mask[i, j] = True 表示第 i 个 token 可以关注第 j 个 token。
    """
    if global_indices is None:
        global_indices = set()
    else:
        global_indices = set(global_indices)

    mask = np.zeros((seq_len, seq_len), dtype=bool)

    # 1) 局部窗口
    for i in range(seq_len):
        left = max(0, i - window_radius)
        right = min(seq_len, i + window_radius + 1)
        mask[i, left:right] = True

    # 2) 全局 token 的整行和整列开放
    for g in global_indices:
        if not (0 <= g < seq_len):
            raise ValueError(f"global index {g} out of range for seq_len={seq_len}")
        mask[g, :] = True   # 全局 token 看全部
        mask[:, g] = True   # 全部 token 看全局 token

    return mask


def masked_softmax(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    对每一行做 masked softmax。
    未允许的位置会被置为极小值。
    """
    if scores.shape != mask.shape:
        raise ValueError("scores and mask must have the same shape")

    masked_scores = np.where(mask, scores, -1e9)
    row_max = masked_scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(masked_scores - row_max)
    exp_scores = exp_scores * mask
    denom = exp_scores.sum(axis=-1, keepdims=True)

    if np.any(denom == 0):
        raise ValueError("some rows have no visible tokens; check the mask")

    return exp_scores / denom


def sparse_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    单头 attention，输入 shape 均为 (seq_len, dim)
    """
    d = Q.shape[-1]
    scores = (Q @ K.T) / math.sqrt(d)
    weights = masked_softmax(scores, mask)
    return weights @ V


def count_edges(mask: np.ndarray) -> int:
    return int(mask.sum())


def demo():
    seq_len = 8
    dim = 4
    window_radius = 1
    global_indices = {0}

    rng = np.random.default_rng(42)
    X = rng.normal(size=(seq_len, dim))

    # 为了示例简单，这里直接令 Q=K=V=X
    Q = X.copy()
    K = X.copy()
    V = X.copy()

    mask = build_sparse_mask(
        seq_len=seq_len,
        window_radius=window_radius,
        global_indices=global_indices,
    )

    # 基本结构检查
    assert mask[3, 2] is True or mask[3, 2] == True
    assert mask[3, 4] is True or mask[3, 4] == True
    assert mask[3, 6] is False or mask[3, 6] == False
    assert mask[0, 7] is True or mask[0, 7] == True
    assert mask[7, 0] is True or mask[7, 0] == True

    out = sparse_attention(Q, K, V, mask)

    print("mask shape:", mask.shape)
    print("allowed edges:", count_edges(mask))
    print("dense edges:", seq_len * seq_len)
    print("output shape:", out.shape)
    print("first row output:", np.round(out[0], 4))


if __name__ == "__main__":
    demo()
```

若在本地运行，预期会看到：

- `allowed edges` 小于 `dense edges`
- `output shape` 为 `(8, 4)`
- 程序可直接完成一次稀疏 attention 前向计算

这段代码的价值不在性能，而在于把 mask 的逻辑完整跑通。对初学者来说，先确认“边是怎么开的、softmax 是怎么屏蔽的”，比直接阅读高性能 CUDA 内核更重要。

### 伪代码版 forward

实际模型中的写法通常接近下面的形式：

```python
band_mask = create_window_mask(window_radius=w)      # [n, n]
global_mask = create_global_mask(global_indices=G)   # [n, n]
attention_mask = band_mask | global_mask             # [n, n]

scores = (Q @ K.transpose(-1, -2)) / sqrt(d)         # [batch, heads, n, n]
scores = scores.masked_fill(~attention_mask, -inf)
weights = softmax(scores, dim=-1)
output = weights @ V
```

如果进一步区分“普通 token 的局部注意力路径”和“全局 token 的特殊计算路径”，实现会更复杂。例如某些库会单独处理 global attention 的索引，以避免真的构造完整的 $n \times n$ 矩阵。但不管内核怎么优化，逻辑本质不变：先定义允许边，再对这些边做标准 attention。

### 一个更贴近真实任务的例子

长文档问答是最典型的应用场景之一。输入形式通常是：

$$
[\text{question tokens}] + [\text{document tokens}]
$$

常见设计如下：

1. 文档正文 token 使用 sliding window。
2. 问题 token 全部标成 global。
3. 有时额外加入一个 `CLS` 或 `<s>` 作为汇聚节点。

这样会得到两个重要效果：

- 问题 token 能直接读取整篇文档的信息。
- 文档中任意位置也能把与问题相关的特征回写给问题 token。

这比“所有 token 只做局部窗口”强得多，因为问答任务本质上要求把一个全局查询条件传播到全文。

---

## 工程权衡与常见坑

局部+全局稀疏 attention 的效果，不只取决于“有没有用稀疏”，更取决于窗口和全局节点是否与任务结构匹配。下面按最常见的几个工程维度展开。

### 1. 窗口大小 $w$ 的权衡

窗口太小会出现两个问题：

1. 单层能看到的上下文过短。
2. 长距离信息只能依赖多层逐步扩散，路径过长。

窗口太大则会带来另一类问题：

1. 计算量和显存迅速上升。
2. 稀疏优势下降。
3. 实现上的块划分、缓存命中和 kernel 利用率可能变差。

可以把窗口理解为“模型对局部连续模式的先验假设强度”。如果任务中的有效依赖大多集中在短邻域，例如局部语法关系、局部时间平滑性，那么较小窗口可能足够。如果任务经常跨句、跨段引用，则需要更大的窗口或更多全局节点。

### 2. 全局 token 数量 $g$ 的权衡

全局 token 决定远距离路径是否足够短。它不是越多越好，而是要覆盖真正承担全局通信职责的位置。

| 全局 token 类型 | 优势 | 缺陷 | 常见任务 |
|---|---|---|---|
| 问题 token | 直接把查询条件广播到全文 | 仅适用于有显式 query 的任务 | QA、阅读理解 |
| CLS / `<s>` | 统一汇聚全局语义 | 容易形成单点瓶颈 | 分类、摘要 |
| 段落头 token | 与文档层级结构对齐 | 需要额外模板设计 | 长文档摘要、结构化阅读 |
| 特殊标记位 | 可编码章节边界、标题、表项等结构 | 设计不当会引入噪声 | 文档理解、表格混合输入 |

经验上，$g$ 的作用类似“全局通信带宽”。过少时不同窗口之间通信不畅，过多时复杂度中的 $ng$ 项会变大，稀疏设计开始向 dense 退化。

### 3. 多层传播如何补足局部限制

初学者常见误解是：“既然单层只看局部，那远处信息是不是完全看不到？”  
答案是不完全如此，因为多层堆叠会扩大有效感受野。

如果每层窗口半径是 $r$，那么在不考虑全局 token 的情况下，经过 $L$ 层后，一个 token 理论上最多能影响到约距离 $Lr$ 之外的位置。若再加入全局节点，远距离路径还会进一步缩短。

这可以写成一个直观对比：

| 机制 | 远距离信息如何传播 |
|---|---|
| 纯局部窗口 | 靠多层逐步扩散 |
| 局部 + 全局 token | 靠多层扩散 + 全局快捷路径 |
| Dense attention | 单层直接到达 |

这也是为什么同样的窗口大小，在浅层网络和深层网络中效果可能明显不同。

### 4. 常见实现错误

下面这些问题在第一次手写稀疏 mask 时非常常见。

| 错误 | 现象 | 原因 |
|---|---|---|
| 只开 global 行，不开 global 列 | 全局 token 能读别人，别人不能写回全局 token | 只实现了单向广播 |
| 只开 global 列，不开 global 行 | 所有人都能看全局 token，但全局 token 无法汇总全文 | 只实现了单向汇聚 |
| 边界窗口 off-by-one | 序列头尾少看或多看一个位置 | 左闭右闭、左闭右开的索引约定混乱 |
| 忘记处理 padding | 模型把填充位当作有效内容 | mask 规则未与 padding mask 合并 |
| 因果约束叠加错误 | 自回归任务中“看见未来”或“合法边被误删” | 局部窗口 mask 与 causal mask 合并不正确 |

一个简单但有效的做法是：无论是否使用成熟框架，都先写一个小尺寸的可视化或断言测试，确认每一行哪些位置是可见的。稀疏 attention 最容易出错的地方，往往不是公式，而是索引。

### 5. 为什么稀疏 attention 不等于线性 attention

这两个概念经常被混淆，但它们解决成本问题的方法并不一样。

稀疏 attention 的做法是：

- 仍然使用标准 attention 形式
- 但只保留一部分边
- 本质是改图结构

线性 attention 的做法是：

- 不删边，仍试图保留全局交互
- 但把 softmax attention 的计算近似或重写
- 本质是改计算形式

以 Performer 为例，它通过 FAVOR+ 随机特征把 softmax kernel 近似为可分解形式，使计算不必显式构造完整的 $n \times n$ 注意力矩阵。简化表达可写成：

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k)
$$

于是注意力可重写为更接近线性的累积形式，而不是对所有位置对逐一计算。这里的近似质量会受到随机特征维度 $r$ 的影响。$r$ 太小，方差可能较大；$r$ 提高后，效果通常更稳，但计算也会增加。

### 6. 任务差异会改变全局设计

同样是长上下文任务，QA 和摘要的全局 token 设计就不完全一样。

| 任务 | 更常见的全局节点设计 | 原因 |
|---|---|---|
| QA | 问题 token + 可能的 CLS | 问题本身就是跨段检索条件 |
| 分类 | CLS / `<s>` | 目标是把全文压缩为单个判别表示 |
| 摘要 | CLS + 段落头 token | 需要兼顾全局主线与段间结构 |
| 信息抽取 | 任务相关锚点 token | 需要把标签条件传播到全文 |

因此，“global token 怎么选”不是一个纯实现问题，而是任务建模问题。选错全局节点，稀疏图就会在错误的位置建立捷径。

---

## 替代方案与适用边界

局部+全局混合稀疏不是唯一降低长序列成本的方法。主流路线至少包括三类：dense、显式稀疏、线性近似。它们的差别最好从“如何保留长程依赖”来理解。

| 方案 | 全局依赖如何建模 | 复杂度特征 | 优点 | 缺点 | 更适合的边界 |
|---|---|---|---|---|---|
| Dense attention | 直接显式全连接 | $O(n^2)$ | 表达最直接，通常是最稳基线 | 长序列成本高 | 中短序列、精度优先 |
| Sparse local-global | 局部边 + 少量全局边 | $O(nw + ng)$ | 可解释、任务可定制、适合长文档 | 依赖 mask 设计质量 | 结构清楚的长序列任务 |
| Linear attention | 通过近似保留全局交互 | 近似 $O(n)$ 或线性级 | 对极长序列更友好 | 存在近似误差，调参与实现更敏感 | 超长上下文、吞吐优先 |

对新手来说，可以先记住下面这组对比。

### 1. Dense attention

它假设任何位置都可能与任何位置发生重要交互，因此不做任何删边。优点是表达能力最直接，缺点是序列一长，成本迅速失控。

### 2. Sparse local-global

它假设“大部分依赖是局部的，少量位置负责全局通信”。优点是结构可解释，也容易把任务先验写进 mask。缺点是如果先验写错，模型就会真的“看不见”重要路径。

### 3. Linear attention

它假设“全局交互仍然重要，但可以用更便宜的数学形式近似计算”。优点是不需要手工指定 global token；缺点是实现和调参通常更复杂，而且近似质量会影响上限。

因此，选择边界通常可以概括为：

1. 如果任务结构很明确，知道哪些 token 应该承担全局角色，例如问题、标题、段首、CLS，那么局部+全局稀疏通常更自然，也更可解释。
2. 如果序列极长，长到即使合理窗口也仍然很贵，而且任务中又很难手工指定全局节点，那么 Performer 一类线性 attention 可能更有吸引力。
3. 如果输入并不长，或者你的首要目标是建立最强基线而不是优化吞吐，那么 dense attention 往往仍然是更稳的选择。

这三类方案不是“谁绝对替代谁”，而是在不同约束下做不同取舍。真正的判断标准不是方法名，而是：

- 序列有多长
- 任务依赖是局部为主还是全局为主
- 你能否明确指定全局桥接节点
- 你更在意峰值精度、显存，还是吞吐

---

## 参考资料

| 标题 | 链接 | 用途 | 涉及章节 |
|---|---|---|---|
| Longformer: The Long-Document Transformer | https://arxiv.org/abs/2004.05150 | Longformer 原始论文，说明 sliding window 与 task-motivated global attention 的基本设计 | 核心结论 / 机制 / 工程 |
| Rethinking Attention with Performers | https://arxiv.org/abs/2009.14794 | Performer 原始论文，说明 FAVOR+ 近似与线性 attention 的思路 | 核心结论 / 工程权衡 / 替代方案 |
| Beltagy, Peters, Cohan: Longformer GitHub / Hugging Face documentation | https://huggingface.co/docs/transformers/model_doc/longformer | 工程视角的 Longformer 输入格式、global attention mask 与使用方式 | 代码实现 / 工程 |
| EmergentMind: Longformer-based Encoder | https://www.emergentmind.com/topics/longformer-based-encoder | 汇总 Longformer 的结构特征、复杂度与应用场景 | 机制 / 工程 / 边界 |
| EmergentMind: FAVOR+ Attention | https://www.emergentmind.com/topics/favor-attention | 汇总 FAVOR+ 的核方法解释、方差和随机特征实现要点 | 工程权衡 / 替代方案 |
| DEV: Longformer - Efficient Attention Mechanism for Long Documents | https://dev.to/nareshnishad/day-31-longformer-efficient-attention-mechanism-for-long-documents-475j | 面向入门读者的直观解释，可作为理解 local/global mask 的辅助材料 | 入门解释 / 机制 |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | 标准自注意力与缩放点积 attention 的原始定义来源 | 问题定义 / 机制公式 |
| Hugging Face Longformer model notes | https://github.com/huggingface/transformers/blob/main/src/transformers/models/longformer/modeling_longformer.py | 查看真实工程实现中 local/global attention 分支与 mask 处理方式 | 代码实现 / 工程坑 |
