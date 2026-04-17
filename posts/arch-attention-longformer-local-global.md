## 核心结论

Longformer 的核心不是“把注意力做稀疏”这么简单，而是把不同类型的信息流拆开处理。

第一部分是局部注意力（Local Attention）。局部注意力可以理解为“每个 token 只看自己附近一小段邻居”。如果窗口宽度是 $w$，那么一个普通 token 通常只关注左侧 $w/2$ 个和右侧 $w/2$ 个位置。这样单层的计算复杂度从标准自注意力的 $O(n^2)$ 降到 $O(n \cdot w)$。

第二部分是全局注意力（Global Attention）。全局注意力可以理解为“少数关键 token 拥有全序列视野”。这些 token 不是随机选的，而是按任务语义指定，例如分类任务中的 `[CLS]`，问答任务中的问题 token，或者候选答案 token。它们会与整条序列双向交互，用来弥补局部窗口看不到远距离依赖的问题。

标准自注意力公式没有变，仍然是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

变化在于：Longformer 不再为所有位置都构造完整的 $QK^T$ 稠密矩阵，而是拆成“带状局部块”和“少量全局块”分别计算。

最重要的结论有两个：

1. 如果长文档里大多数 token 只需要局部上下文，那么局部滑窗已经足够。
2. 如果任务中确实存在少量必须跨段理解的关键位置，就给这些位置分配全局注意力，而不是让所有位置都做全连接。

一个典型设置是：处理 4096 token 的文档时，窗口宽度设为 512，意味着大多数 token 只看自己左右各 256 个邻居；同时只给 10 到 50 个任务关键 token 全局注意力。这样模型既能处理长序列，又不会把算力浪费在大量无必要的远距离两两比较上。

---

## 问题定义与边界

先定义问题。标准 Transformer 的自注意力会让每个位置和所有位置做交互，因此序列长度为 $n$ 时，注意力矩阵大小是 $n \times n$。这带来两个直接后果：

1. 计算量是 $O(n^2)$。
2. 显存开销也随 $n^2$ 增长。

这在 512 token 左右还能接受，但一旦进入 4096、8192 这样的长文档场景，就很快成为瓶颈。对文档问答、长文摘要、检索增强阅读这类任务来说，输入本身就很长，截断到 512 往往直接丢失有效信息。

Longformer 解决的问题边界很明确：它不是试图让“所有 token 都高效地看全局”，而是利用一个经验事实，多数长文任务里，真正需要全局交互的位置其实很少。

以 WikiHop 这类多跳问答为例，输入可能是问题加多段证据文档，总长度到 4096 token。这里有两种不同的信息需求：

- 普通上下文 token：通常只需要局部语义与相邻句法结构。
- 问题 token、候选答案 token：需要跨段收集证据，必须和远距离位置交互。

这就是 Longformer 的适用边界。如果任务中的全局依赖是“少数关键位置驱动”，Longformer 很合适；如果任务要求大量位置之间都做精确长程匹配，局部+少量全局的模式就未必够。

下表先看几个模型的计算特征差异：

| 模型 | 典型最大长度 | 全局机制 | 单层注意力复杂度 | 长文档处理方式 |
|---|---:|---|---|---|
| RoBERTa | 512 | 无 | $O(n^2)$ | 截断或分块 |
| Longformer | 4096 及以上 | 指定少量全局 token | $O(n \cdot w + n \cdot g)$ | 单次编码长文 |
| BigBird | 4096 及以上 | 全局 + 随机 + 局部 | 近似线性 | 稀疏图连接 |

其中 $g$ 表示全局 token 数量。由于 $g \ll n$，整体仍接近线性。

一个容易误解的点是：Longformer 不是“完全没有远距离依赖”。它只是把远距离依赖集中到少量关键 token 上，让这些 token 充当全局信息路由器。

---

## 核心机制与推导

Longformer 的机制可以拆成两条路径：局部路径和全局路径。

局部路径里，每个普通 token 只和窗口中的邻居计算注意力。假设序列长度为 $n$，窗口宽度为 $w$，那么第 $i$ 个 token 只会与区间

$$
[j \mid i-\frac{w}{2} \le j \le i+\frac{w}{2}]
$$

中的位置交互。边界位置再做截断。

如果写成矩阵视角，标准注意力会产生一个稠密矩阵，而 Longformer 的局部部分只保留主对角线附近的一条“带状区域”。因此它本质上是 banded attention，也就是“带状注意力”。

再加上全局 token 后，矩阵结构就变成：

- 普通 token 的行：只在局部窗口内有非零注意力。
- 全局 token 的行：对所有列都可见。
- 所有 token 的列：对全局 token 列也可见。

这意味着全局 token 和全序列是双向连接的，而不是单向广播。

论文里一个关键设计是，把局部和全局使用不同的投影参数。也就是分别学习：

- 局部投影：$Q_s, K_s, V_s$
- 全局投影：$Q_g, K_g, V_g$

这里“投影”可以理解为“把输入隐藏状态映射到注意力空间的线性层”。为什么要拆开？因为局部关系和全局关系关注的模式不同。局部更偏短距离语义与邻近结构，全局更偏任务关键汇聚。ablation 结果表明，共用一套投影会损失效果，分开学习更稳定。

### 玩具例子

看一个 6 个 token 的最小例子，窗口宽度设为 3。为了便于描述，把中心宽度近似看成“左右各 1 个邻居，再包含自己”。

序列为：

| 位置 | token |
|---|---|
| 0 | `[Q]` |
| 1 | `猫` |
| 2 | `坐` |
| 3 | `在` |
| 4 | `垫子` |
| 5 | `上` |

如果位置 0 的 `[Q]` 是问题 token，并设为全局注意力，那么：

- 位置 3 的普通 token `在` 只能看位置 2、3、4。
- 位置 4 的普通 token `垫子` 只能看位置 3、4、5。
- 位置 0 的全局 token `[Q]` 可以看 0 到 5 的所有位置。
- 其他所有位置也都可以看 `[Q]`。

这时 `[Q]` 就像一个“全局汇聚点”。它能把远距离信息集中起来，再反馈给局部 token。

### 多层后为什么能扩展感受野

单层局部窗口只能看到附近，但层叠后感受野会扩张。感受野可以理解为“某个位置经过多层传播后，最终能间接接触到多远的信息”。

如果每层窗口半径是 $r=w/2$，那么在理想情况下，堆叠 $L$ 层后，某个位置能覆盖的大致范围会增长到：

$$
i-Lr \ \text{到} \ i+Lr
$$

这不是严格等于所有远程全连接，但足以逐层传播较长距离的上下文。再加上全局 token 作为跨段捷径，模型就能在保持稀疏计算的同时处理全局任务。

### 为什么实现上需要 chunk

真正的工程难点不在公式，而在硬件执行。GPU 擅长做大块连续矩阵运算，不擅长处理过度零散的稀疏索引。Longformer 因此没有直接用“通用稀疏矩阵乘法”去实现局部窗口，而是采用 sliding chunks 或 TVM/CUDA kernel，把 K/V 按 chunk 排布，只计算对角带附近需要的块。

一个简化伪代码如下：

```text
for each chunk in sequence:
    q_chunk = Q[chunk]
    k_chunk = K[chunk and neighbor chunks]
    v_chunk = V[chunk and neighbor chunks]

    local_scores = banded_matmul(q_chunk, k_chunk)
    local_probs = softmax(mask_invalid(local_scores))
    local_out = local_probs @ v_chunk
```

核心思想是：不用显式构造完整 $n \times n$ 注意力矩阵，而是只算会落在滑窗内的那一部分。这样既减少计算，也减少显存访问开销。

---

## 代码实现

下面先用一个可运行的 Python 玩具实现，演示“局部窗口 + 全局 token”的可见性规则。这个例子不依赖深度学习框架，只验证 mask 逻辑是否正确。

```python
from typing import List

def longformer_visibility(n: int, window: int, global_indices: List[int]):
    assert window % 2 == 0, "window 必须是偶数，方便左右对称"
    half = window // 2
    global_set = set(global_indices)

    mask = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        if i in global_set:
            for j in range(n):
                mask[i][j] = 1
        else:
            left = max(0, i - half)
            right = min(n, i + half + 1)
            for j in range(left, right):
                mask[i][j] = 1

        for g in global_set:
            mask[i][g] = 1

    return mask

m = longformer_visibility(n=6, window=2, global_indices=[0])

# 普通 token 3 只能看 2,3,4，再加上全局 token 0
visible_from_3 = [j for j, v in enumerate(m[3]) if v]
assert visible_from_3 == [0, 2, 3, 4]

# 全局 token 0 可以看所有位置
visible_from_0 = [j for j, v in enumerate(m[0]) if v]
assert visible_from_0 == [0, 1, 2, 3, 4, 5]

# 所有位置都能看全局 token 0
assert all(m[i][0] == 1 for i in range(6))
print("mask logic ok")
```

上面这个例子只验证连接模式，没有算数值注意力，但它已经把 Longformer 最关键的结构约束表达出来了。

### 真实工程例子

在 Hugging Face 的 Longformer 中，最常见的做法是：

- 通过 `attention_window` 指定局部窗口宽度。
- 通过 `global_attention_mask` 指定哪些 token 用全局注意力。
- 输入长度通常需要 pad 到窗口宽度的倍数，以避免滑窗对齐问题。

一个简化示意如下：

```python
# 伪代码，表达接口含义，不依赖具体版本细节
input_ids = tokenizer(question + context, max_length=4096, truncation=True)

global_attention_mask = [0] * len(input_ids)

# 问题部分设为全局 attention
for idx in question_token_positions:
    global_attention_mask[idx] = 1

outputs = model(
    input_ids=input_ids,
    attention_mask=[1] * len(input_ids),
    global_attention_mask=global_attention_mask,
    attention_window=512,
)
```

这里的 `global_attention_mask` 可以理解为“哪些位置需要获得全局读写权限”的标记。对问答任务，常见策略是：

- 问题 token 全局
- `[CLS]` 全局
- 候选答案 token 视任务决定是否全局

如果是 4096 token 的长文问答，工程上常见配置就是窗口宽度 512，其余几千个 token 用 local，问题相关 token 用 global。这样模型能一次编码完整上下文，而不用像 RoBERTa 那样把文档拆成多个 512 片段再做二阶段汇总。

---

## 工程权衡与常见坑

Longformer 的难点主要不在“能不能跑”，而在“跑得是否正确且划算”。

先看实现方式差异：

| 实现方式 | 优点 | 缺点 | 适配场景 |
|---|---|---|---|
| `n2` 全注意力回退实现 | 最直接，便于调试 | 退化成 $O(n^2)$，长序列很慢 | 正确性验证、小输入 |
| `sliding_chunks` | 易实现，纯张量操作 | 需要 padding，对内存布局敏感 | 通用 GPU 环境 |
| TVM/CUDA kernel | 速度最好，硬件友好 | 依赖额外内核支持，环境要求更高 | 追求吞吐的训练/推理 |

### 常见坑 1：忘记 pad 到窗口倍数

这是最常见的实现错误之一。因为 `sliding_chunks` 一般按固定 chunk 划分，如果输入长度和窗口设置不对齐，就可能出现带状块错位，导致注意力边界不正确。

例如窗口宽度是 512，而输入真实长度是 4103。此时如果不 pad，到 chunk 切分阶段最后一个块就会不完整，容易引发：

- 局部窗口覆盖缺失
- 对角带错位
- mask 与实际 chunk 布局不一致

正确做法通常是先 pad 到 4608 或下一个满足实现要求的对齐长度，再用 attention mask 屏蔽掉补齐位置。

### 常见坑 2：全局 token 选错位置

Longformer 不是自动推断谁该全局。全局 token 选得不好，模型性能会明显下降。

比如在 WikiHop 中，如果不给问题 token 全局注意力，而只让普通上下文做局部传播，那么跨文档证据很难在有限层数内汇聚到答案位置。论文中的 ablation 表明，去掉 global attention 或去掉单独的 global projection，会带来明显精度下降，问答任务尤其敏感。

经验上，全局 token 的选择应当遵循一句话：谁负责跨段聚合，就给谁全局。

### 常见坑 3：误以为窗口越大越好

窗口增大确实会提升单层可见范围，但代价是计算量线性上升。因为复杂度是 $O(n \cdot w)$，把 $w$ 从 256 提到 1024，成本会接近放大 4 倍。

所以窗口不是越大越好，而是要匹配任务局部依赖尺度：

- 句内、段内关系多：适当增大窗口
- 全局依赖主要靠问题 token 汇聚：优先保留少量 global，而不是无限增大窗口

### 常见坑 4：把 Longformer 当成任意长文本万能解

Longformer 能处理更长输入，不代表它对任意超长文本都自动有效。随着长度继续增长：

- 局部窗口仍然只能提供有限单层视野
- 信息需要跨多层传播
- 全局 token 数过少时，汇聚能力可能不足
- 全局 token 数过多时，又会抬高成本

因此它更像“长文档上的稀疏注意力工程折中”，不是无限长度下的通用最优解。

---

## 替代方案与适用边界

最直接的替代方案是继续用 RoBERTa 之类的全注意力模型，但把长文档切成多个 512 片段。这样做的优点是模型成熟、实现简单；缺点是跨段依赖容易断掉，通常还需要额外的 chunk rerank、late fusion 或检索汇总模块。

一个真实工程对比如下：

| 方案 | 输入处理 | 跨段依赖 | 复杂度特征 | 适用任务 |
|---|---|---|---|---|
| RoBERTa 分块 | 文档切成多个 512 chunk | 需要额外汇总 | 每块 $O(512^2)$ | 短文本、多段独立匹配 |
| Longformer | 单次输入 4096 | 问题 token 全局聚合 | $O(n \cdot w)$ | 长文 QA、文档级分类 |
| BigBird | 局部 + 全局 + 随机 | 依赖随机图连通 | 近似线性 | 更通用的长序列建模 |

### Longformer 与 BigBird 的边界差异

BigBird 也是稀疏注意力，但它在局部和全局之外加入了随机连接。随机连接的作用是增强图连通性，让更多位置能更快建立长程路径。Longformer 没有随机边，而是更强调“局部滑窗 + 任务特定全局”。

这带来一个取舍：

- 如果任务结构清晰，知道哪些 token 是关键汇聚点，例如问答中的问题 token，Longformer 很自然。
- 如果任务中全局依赖分布更分散，或者很难事先指定全局位置，BigBird 这类更通用的稀疏模式可能更稳。

### Longformer 不适合什么情况

Longformer 不适合以下几类场景：

1. 输入并不长，512 以内就能解决。此时引入稀疏机制只会增加实现复杂度。
2. 任务要求大量 token 两两精确比较，例如高密度全局对齐。局部+少量全局可能不够。
3. 无法可靠定义全局 token，且全局依赖分布高度分散。此时 task-specific global 的优势会变弱。

因此，Longformer 不是“比 RoBERTa 高级”，而是“在长文档、少数关键全局位置明确”的问题设定下更合适。

---

## 参考资料

- Iz Beltagy, Matthew E. Peters, Arman Cohan. *Longformer: The Long-Document Transformer*. 2020. 论文原文，重点看 section 3 的 attention 设计、实现细节与 QA 实验结果。  
  https://huggingface.co/shahrukhx01/gradient-whisperer/resolve/main/24_longformer.pdf?download=true

- Hugging Face Transformers 文档，Longformer 模型说明。适合查 `attention_window`、`global_attention_mask` 等接口含义，以及滑窗注意力的直观解释。  
  https://huggingface.co/transformers/v2.10.0/model_doc/longformer.html

- DeepWiki: *Longformer Attention Mechanisms*. 对 `sliding_chunks`、`tvm`、`n2` 三类实现路径的工程差异有较清晰总结，适合补实现层面的认知。  
  https://deepwiki.com/allenai/longformer/2.1-attention-mechanisms

- AllenAI Longformer 项目代码与文档。适合进一步查看 kernel、mask 规则与训练配置。  
  https://github.com/allenai/longformer
