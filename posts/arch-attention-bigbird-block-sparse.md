## 核心结论

BigBird 是一种面向长序列的稀疏注意力 Transformer。所谓“稀疏注意力”，就是不再让每个 token 和全部 token 计算注意力，而是只保留一部分连接。BigBird 保留了三类连接：

1. Local window attention：局部窗口注意力，只看附近若干块，负责短距离上下文。
2. Global token attention：全局注意力，让少数特殊 token 和全序列相连，负责全局汇总。
3. Random attention：随机注意力，给每个块补一些随机连接，负责提升全图连通性。

它的关键价值不是“随便删一些边”，而是用这三种边构造出一个既便宜又足够连通的稀疏图。这样做后，自注意力的计算和显存占用可以从原始 Transformer 的 $O(n^2)$ 降到接近 $O(n)$，其中 $n$ 是序列长度。

对初学者可以直接记住一句话：BigBird 不是把注意力做小了，而是把“全连接通信”改成了“局部通信 + 少量广播 + 少量随机捷径”的通信网络。

下面这张表先给出最重要的比较：

| 模型 | 每个 token 连接对象 | 复杂度 | 长序列可扩展性 |
|---|---|---:|---|
| 全注意力 Transformer | 全部 $n$ 个 token | $O(n^2)$ | 差 |
| BigBird | 局部窗口 + 全局 token + 随机 token | $O((g+w+r)\cdot n)$，通常视为 $O(n)$ | 强 |

其中 $g,w,r$ 分别表示每个 token 可见的全局、窗口、随机连接规模；如果这些值不随序列长度增长，它们就是常数，整体复杂度就近似线性。

一个玩具例子是班级聊天。全注意力相当于“班里每个人都和所有人同时聊天”；BigBird 改成“每个人只和左右几位同学说话，再和班长、老师说话，还随机认识几个人”。单轮通信少得多，但经过几层传播，信息仍能到达全班。

真实工程里，这种结构特别适合 4,096 token 甚至更长的输入，比如长文档问答、长摘要、法律文书分析、基因组序列建模。因为这些任务同时需要局部模式和远距离依赖，而全注意力的代价又过高。

---

## 问题定义与边界

原始 Transformer 的核心瓶颈来自注意力矩阵。若序列长度为 $n$，那么 $QK^\top$ 的形状是 $n \times n$，每个 token 都要和其余所有 token 打分，复杂度和显存都按平方增长。

注意力基本公式仍然是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

这里的 $d$ 是每个头的隐藏维度，白话说就是“每个 token 用多少维数字来表示语义”。

问题不在公式本身，而在 $QK^\top$ 太大。举例说，序列从 512 增长到 4096，长度只变成 8 倍，但注意力矩阵面积会变成 $8^2=64$ 倍。这就是为什么普通 BERT 在长文本上很快碰到算力和显存上限。

BigBird 试图解决的是一个更严格的问题：不是只想“跑得更便宜”，而是希望在便宜的同时尽量保留原始注意力的表达能力。这里的“表达能力”可以白话理解为：模型还能不能学到复杂的长距离依赖，而不是因为删边删得太狠，导致信息根本传不过去。

它的边界条件有三个。

第一，序列通常按 block 切分。所谓“block”，就是连续的一段 token，BigBird 常见实现里每个 block 是 64 个 token。若序列长度是 $n$，block 大小是 $b$，则 block 数量为：

$$
m = \frac{n}{b}
$$

这要求理想情况下 $n \bmod b = 0$。如果长度不能整除，就需要 padding，也就是补齐到 block 的整数倍。

第二，BigBird 的稀疏不是元素级稀疏，而是块级稀疏。元素级稀疏是“这个 token 看那个 token，不看另一个 token”；块级稀疏是“这一整块 query 看那几整块 key”。块级设计的主要原因不是数学更优雅，而是 GPU 更容易高效计算。

第三，随机连接不能完全删掉。只有局部窗口和少量全局 token 时，图的连通性可能不够好；随机边的作用是把局部簇连接成一个更容易全局传播的小世界图。

以常见配置为例，若 `block_size = 64`，窗口覆盖 3 个 block，随机连接 3 个 block，再给前若干 block 设为全局 block，那么每个 query block 看到的 token 数量大约是：

$$
g + w + r
$$

若按 token 计数，典型近似是：

- 全局部分：$g = 2 \times 64 = 128$
- 局部窗口：$w = 3 \times 64 = 192$
- 随机部分：$r = 3 \times 64 = 192$

于是每个位置只看大约 $128 + 192 + 192 = 512$ 个 token，而不是看全部 4096 个 token。

---

## 核心机制与推导

BigBird 的核心思想可以拆成两层：第一层是“按块组织计算”，第二层是“按图组织连边”。

### 1. 按块组织计算

输入原本形状通常是 `(batch, seq_len, hidden_dim)`。BigBird 先把序列切成块，再变形成：

$$
(batch,\ num\_blocks,\ block\_size,\ hidden\_dim)
$$

这一步的意义是：后续不再以单个 token 为单位决定连接，而是以 block 为单位决定“看哪些块”。

如果 `seq_len = 4096`，`block_size = 64`，那么：

$$
num\_blocks = \frac{4096}{64} = 64
$$

于是原来一个 $4096 \times 4096$ 的稠密注意力矩阵，不再显式计算，而是变成“64 个 query block 分别看哪些 key block”的稀疏模式。

### 2. 三种连接如何分工

BigBird 不是三种边随意叠加，而是明确分工。

| 注意力类型 | 作用 | 白话解释 |
|---|---|---|
| Local window | 建模局部连续依赖 | 看前后邻居，适合短语、句子、局部模式 |
| Global token | 建模全局汇总与广播 | 少数特殊位置能读全局、也能被全局读取 |
| Random | 补充远距离捷径 | 随机拉几条远边，防止图断裂 |

局部窗口很好理解。自然语言、DNA 序列、代码文本都存在明显的局部结构，相邻 token 往往最相关。只保留局部邻居，可以保住大量有用信息。

全局 token 的作用是给系统放少量“广播站”。例如分类任务里的 `[CLS]`，长文档里的标题位置，或者段落边界位置，都可以作为全局节点。它们能快速聚合全局信息，再把全局信息传回各处。

随机连接是 BigBird 的关键补丁。只靠窗口，信息传播距离会随层数线性扩展，太慢；只靠少量全局节点，又容易把负担过度集中在少数点上。随机边相当于在图里加“捷径”，让任意两点更可能通过少数跳数连通。

### 3. 为什么复杂度接近线性

若每个 token 只看固定数量的 token，例如只看 512 个，那么总计算量就是：

$$
O(512 \cdot n) = O(n)
$$

更一般地，若每个 token 只看 $g+w+r$ 个位置，则复杂度为：

$$
O((g+w+r)\cdot n)
$$

当 $g,w,r$ 不随 $n$ 增长时，这就是线性复杂度。

若从 block 角度看，每个 query block 只和固定个数的 key block 做乘法。设每个 block 大小是 $b$，每个 query block 连接 $c$ 个 key block，则总计算量近似为：

$$
O(m \cdot c \cdot b^2)
$$

又因为 $m = n/b$，所以：

$$
O\left(\frac{n}{b}\cdot c \cdot b^2\right)=O(c \cdot b \cdot n)
$$

只要 $b,c$ 是常数，仍然是 $O(n)$。

### 4. 玩具例子

设序列长度为 16，block size 为 4，那么共有 4 个 block：

- Block 0: token 0-3
- Block 1: token 4-7
- Block 2: token 8-11
- Block 3: token 12-15

若规定：

- Block 0 是全局 block
- 每个 block 看左右相邻 1 个 block
- 每个 block 再随机看 1 个非相邻 block

那么 Block 2 可能看到：

- 全局：Block 0
- 局部：Block 1、Block 3
- 随机：Block 0 或 Block 1 以外的额外块

这时它不需要看全部 4 个 block 里的所有 token，但依然能在少数层内接触远端信息。

### 5. 真实工程例子

以长文档问答为例，输入可能是“问题 + 多段文档 + 附加证据”，长度接近 4096 token。答案往往依赖两类信息：

- 局部信息：某个实体名称和后面一句解释的匹配。
- 全局信息：问题中的关键词需要和远处某段证据关联。

BigBird 的局部窗口负责保留句内和段内模式，全局 token 可以放在问题开头、文档边界或 `[CLS]` 上做全局汇总，随机边则增加跨段证据之间的偶然连通。这样比纯滑窗结构更容易处理“答案在文档后半段，但线索在前半段”的问题。

在基因组序列分析中，道理类似。DNA 序列很长，局部 motif 很重要，但远处调控关系也重要。BigBird 可以同时保留局部序列模式和远距离依赖，因此适合这类超长输入任务。

---

## 代码实现

BigBird 的工程实现重点不是重新发明注意力公式，而是把数据排布和稀疏模式组织好。

一个最小可运行的 Python 玩具实现如下。它不依赖深度学习框架，只演示“按 block 构造可见块集合”的逻辑。

```python
from typing import Dict, List, Set

def build_bigbird_block_graph(
    num_blocks: int,
    num_global_blocks: int = 1,
    window_blocks: int = 1,
    num_random_blocks: int = 1,
) -> Dict[int, List[int]]:
    assert num_blocks > 0
    assert num_global_blocks >= 0
    assert window_blocks >= 0
    assert num_random_blocks > 0  # BigBird 的随机块不应为 0

    global_blocks = set(range(min(num_global_blocks, num_blocks)))
    graph: Dict[int, List[int]] = {}

    for i in range(num_blocks):
        visible: Set[int] = set(global_blocks)

        # 局部窗口：看左右相邻若干块
        for j in range(max(0, i - window_blocks), min(num_blocks, i + window_blocks + 1)):
            visible.add(j)

        # 伪随机：这里用确定性规则代替真随机，便于测试
        # 实际实现会按层/头采样随机块
        candidate = (i * 3 + 1) % num_blocks
        visible.add(candidate)

        # 若需要多个随机块，可继续补
        k = 1
        while len(visible - global_blocks) < (2 * window_blocks + 1 + num_random_blocks):
            visible.add((candidate + k) % num_blocks)
            k += 1

        graph[i] = sorted(visible)

    return graph


def complexity_dense(seq_len: int) -> int:
    return seq_len * seq_len


def complexity_bigbird(seq_len: int, visible_per_token: int) -> int:
    return seq_len * visible_per_token


if __name__ == "__main__":
    graph = build_bigbird_block_graph(
        num_blocks=8,
        num_global_blocks=1,
        window_blocks=1,
        num_random_blocks=1,
    )

    # 每个 block 至少能看到自己和全局 block
    assert 0 in graph[0]
    assert 0 in graph[3]
    assert 3 in graph[3]

    dense = complexity_dense(4096)
    sparse = complexity_bigbird(4096, 512)

    # 4096 长度下，若每个 token 只看 512 个位置，则成本显著下降
    assert sparse < dense
    assert dense // sparse == 8

    print("graph:", graph)
    print("dense:", dense, "sparse:", sparse)
```

这个例子说明两个工程事实。

第一，真正要维护的是“每个 block 能看到哪些 block”，而不是直接维护一个巨大的 $n \times n$ 掩码矩阵。  
第二，复杂度收益来自“可见块数量固定”，不是来自某种神秘的数学技巧。

如果用 Hugging Face Transformers，配置层面的典型写法会是：

```python
from transformers import BigBirdConfig

config = BigBirdConfig(
    attention_type="block_sparse",
    block_size=64,
    num_random_blocks=3,
)
```

这里：

- `attention_type="block_sparse"` 表示启用块稀疏注意力。
- `block_size=64` 表示每个块 64 个 token。
- `num_random_blocks=3` 表示每个 query block 除局部和全局外，再连接 3 个随机块。

实现流程通常可概括为：

1. 把输入从 `(batch, seq_len, dim)` reshape 为 `(batch, num_blocks, block_size, dim)`。
2. 为每个 query block 收集三类 key/value block：`global_blocks`、`window_blocks`、`random_blocks`。
3. 只对这些块执行注意力乘法，而不是对全部块两两计算。
4. 把结果再拼回原序列维度。

为什么强调 block sparse GEMM？因为 GEMM 就是矩阵乘法内核。GPU 最擅长处理大而规则的矩阵块，不擅长处理高度零碎、索引跳跃很强的元素级稀疏。BigBird 选 block 级别稀疏，本质上是在“数学稀疏”和“硬件友好”之间取一个平衡点。

---

## 工程权衡与常见坑

BigBird 真正难的地方不在概念，而在落地细节。下面这张表是最常见的问题。

| 问题 | 现象 | 原因 | 规避方式 |
|---|---|---|---|
| `seq_len % block_size != 0` | 退回稠密注意力或报错 | block 切分不完整 | 先 padding 到整数倍 |
| `num_random_blocks = 0` | 长距离传播变差 | 图缺少随机捷径 | 通常至少保留 1，常见默认 3 |
| `block_size` 过小 | 块数暴增，调度开销上升 | 稀疏图更碎 | 常从 64 开始调 |
| `block_size` 过大 | 局部计算过重 | 单块乘法成本高 | 根据序列长度和 GPU 调优 |
| 逐元素稀疏实现 | 理论稀疏，实际不快 | 内存访问不连续 | 优先块稀疏 kernel |

### 1. block 对齐问题

这是最常见的坑。比如输入长度是 4100，而 `block_size = 64`，则：

$$
4100 \bmod 64 = 4
$$

说明最后一个 block 不完整。很多实现会选择 padding 到 4160，而不是为了凑整去改 block 大小。因为 block size 既影响图结构，也影响底层 kernel 的效率，随意改成 41 这类值通常不划算。

### 2. 随机块不能简单关掉

很多初学者看到随机连接，会觉得“不稳定”或“不优雅”，想把它设为 0，只保留局部窗口和全局 token。这在概念上很诱人，但通常会伤害图连通性。BigBird 理论分析强调的正是三者组合，而不是前两者任取其一。

随机块的数量也不是越多越好。太少，图可能传播不足；太多，又会抬高计算成本。常见默认值是 3 个随机 block，本质上是一个工程上比较稳妥的折中。

### 3. 块稀疏快，不等于任意稀疏都快

这是理解硬件实现时必须澄清的一点。GPU 加速依赖规则的数据布局和高吞吐矩阵运算。元素级稀疏理论上“零更多”，但实际常因为索引开销、访存不连续、线程利用率差而跑不快。

块级稀疏虽然保留了一些“不必要计算”，但换来了更规整的矩阵块乘法。对于现代 GPU，这往往是更优解。也就是说，BigBird 的优势来自“适度稀疏 + 硬件友好”，而不是“越稀疏越好”。

### 4. 全局 token 的选择会影响任务表现

全局 token 不是纯装饰。分类任务里 `[CLS]` 很自然；长文档任务里，也可以把标题、问题前缀、段首标记设为全局位置。如果全局位设计得太少，广播能力不足；设计得太多，又会抬高成本。

### 5. 真实工程例子

假设做一个法律文书问答系统，单份文档 3000 到 5000 token。若直接上全注意力，训练 batch 往往被迫很小；改成 BigBird 后，可以把输入稳定放到 4096 token，保留问题句、案件事实、法条引用、判决结果等长距离关联。

但这里有个常见误判：如果任务是检索增强生成中“已经选好少量高相关片段”的场景，序列虽然长，但有效信息已被外部检索压缩，BigBird 的优势就未必明显。这说明稀疏注意力不是越长越该用，而是要看任务是否真的需要模型自己在长上下文中找证据。

---

## 替代方案与适用边界

BigBird 不是唯一的长序列方案。理解它和其他方案的差异，比背配置参数更重要。

| 方法 | 核心机制 | 复杂度 | 优势 | 局限 |
|---|---|---:|---|---|
| BigBird | 局部 + 全局 + 随机块稀疏 | 近似 $O(n)$ | 理论保证较强，兼顾局部与远距 | 需要块对齐和专用实现 |
| Longformer | 局部滑窗 + 少量全局 | 近似 $O(n)$ | 结构直观，工程成熟 | 缺少随机捷径 |
| Performer | 核函数近似 softmax | 近似 $O(n)$ | 不依赖显式稀疏图 | 近似误差分析更重要 |
| Linformer | 低秩投影压缩注意力 | 近似 $O(n)$ | 参数化清晰 | 对长依赖结构有假设 |

### BigBird 和 Longformer 的差异

两者都保留局部窗口和全局信息，但 BigBird 多了随机连接。这个差异的直观意义是：Longformer 更像“规则道路网”，BigBird 多了一些“随机高速路入口”。当任务需要多跳推理、跨段证据聚合时，随机边往往有帮助。

### BigBird 的适用边界

更适合 BigBird 的任务：

- 长文档问答
- 长文本摘要
- 多段证据推理
- 基因组与生物序列建模
- 长篇法律、医疗、科研文本分析

不一定优先 BigBird 的任务：

- 输入本来就不长，512 或 1024 足够
- 外部检索已把上下文压缩得很短
- 任务主要依赖全局压缩而不是局部模式
- 部署环境缺少高效块稀疏 kernel，导致理论优势无法兑现

这里再给一个新手易懂的判断标准：  
如果你的任务是“在很长的一篇材料里，模型自己要找到局部证据并跨远距离拼起来”，BigBird 值得考虑。  
如果你的任务是“先检索出两三段高度相关内容，再交给模型总结”，那 BigBird 可能不是首选。

---

## 参考资料

1. Manzil Zaheer 等，《Big Bird: Transformers for Longer Sequences》  
重点：BigBird 的理论基础，包括稀疏图设计、近似全注意力能力与图灵完备性结论。

2. Google Research: BigBird 项目页面  
重点：论文入口与官方摘要，适合先确认模型目标和主要结论。

3. Hugging Face Blog，《Understanding BigBird's Block Sparse Attention》  
重点：用工程视角解释 block sparse attention，适合把论文概念映射到实现。

4. Hugging Face Transformers 文档，`BigBirdConfig`  
重点：`attention_type`、`block_size`、`num_random_blocks` 等配置项的实际含义。

5. 长文档问答与摘要相关基准介绍，如 Natural Questions、HotpotQA、TriviaQA、WikiHop 的公开说明  
重点：理解 BigBird 为什么在长上下文任务中受关注。

6. 基因组建模与 Path-BigBird 相关应用报道  
重点：理解 BigBird 在超长序列、生物医疗文本中的落地价值。

7. block sparse GEMM 与 GPU 稀疏计算相关工程资料  
重点：理解为什么 BigBird 采用块级稀疏而不是元素级稀疏，以及硬件加速的现实约束。
