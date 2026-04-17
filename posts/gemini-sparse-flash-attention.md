## 核心结论

Gemini 这类长上下文模型要同时解决两个问题：一是注意力矩阵太大，二是显存或片上 SRAM 的读写太贵。Sparse Attention 的作用是“少算”，也就是只在重要位置之间计算注意力；FlashAttention 的作用是“少搬”，也就是不把完整的中间矩阵落到慢内存里，而是在片上按块流式完成 softmax 和输出累计。Gemini 的关键不是二选一，而是把两者合在一起：先用 block mask 指定哪些块值得算，再用 FlashAttention-2 的块级流水把这些块高效算完。

对零基础读者，可以把它理解成“两层筛选”。第一层筛选是稀疏模式，只保留 local block 和少量 global/summary token；第二层筛选是 FlashAttention 的内核执行路径，只让这些保留块进入高效 kernel。结果是，原本需要对全部 $N \times N$ token 对做读写和计算，现在只对稀疏掩码允许的那部分块做处理，IO 复杂度从全连接场景的二次规模，下降为与稀疏度 $s$ 成正比的规模：
$$
\Theta(Nd + N^2 d^2 M^{-1} s)
$$
其中 $N$ 是序列长度，$d$ 是每个头的维度，$M$ 是片上 SRAM 容量，$s$ 是块稀疏度。

玩具例子可以直接看 16K token、块大小 128 的情况。总共是 $16000/128 \approx 125$ 个块。若每个 query 块只看自己附近 4 个局部块，再看 64 个 summary token，那么大部分远处块都被跳过。粗略理解就是：不是在 125 个块上都做全连接，而是只在极少数“邻居块 + 汇总通道”上做 dense attention。FlashAttention-2 再保证这些被选中的块内部仍然是精确 softmax，而不是近似 softmax。

| 方案 | 参与计算的块 | QK 读写规模 | 中间矩阵是否完整落盘 | 典型延迟趋势 |
| --- | --- | --- | --- | --- |
| 传统全 Attention | 全部块 | $O(N^2)$ | 通常会产生大中间结果 | 长上下文下快速上升 |
| 纯 FlashAttention-2 | 全部块 | 仍与全块数相关，但 IO 更优 | 不完整落盘，流式处理 | 比传统全注意力低 |
| Block-Sparse FlashAttention | 仅 mask 允许的块 | 与稀疏度 $s$ 成正比 | 不完整落盘，且只处理稀疏块 | 长上下文下更稳定 |

真实工程例子是 TPU 上的 Gemini 风格服务链路。输入可能是多轮 agent 上下文、工具调用日志、文档片段和代码片段，长度轻易到 8K、16K 甚至更长。如果仍然让每个 token 看全部 token，延迟和吞吐都会明显恶化。把 local+global 的 block mask 交给 FlashAttention kernel 后，系统可以只保留“附近上下文”和“全局摘要通道”，从而在较低延迟下维持长程信息传播。

---

## 问题定义与边界

先定义问题。注意力是“让每个 token 看其他 token 的相关性”的机制，白话讲，就是当前词在决定输出时，需要参考哪些别的词。标准自注意力会让每个位置都看所有位置，因此序列越长，计算和内存访问都会迅速膨胀。

这里的边界不是“如何做任意稀疏”，而是“如何在 Gemini 这类长上下文模型里，把 blockwise sparse attention 和 FlashAttention 融合”。所以重点有三个：

1. 序列很长，比如 16K。
2. 稀疏模式不是随意连接，而是以 block 为单位组织。
3. 硬件目标偏向 TPU，这意味着片上内存、tile 大小、编译稳定性都很重要。

一个常用配置是：
- 最大上下文：16K
- local block size：128 token
- global summary token：64
- 稀疏模式：local + sink/global

这里的 block 是“把连续 token 按固定长度打包成一格”，白话讲，就是先不在单个 token 粒度上乱连，而是先把序列切成许多小砖块。这样做的原因很直接：硬件执行 kernel 时，本来就是按 tile 或块调度；如果掩码也按块表达，就更容易和 FlashAttention 的分块流程对齐。

把注意力图看成棋盘会更直观。假设 128×128 token 构成一个 attention 子块，那么整张 16K 序列的注意力图就是很多块拼成的大棋盘。local block 表示“只和自己附近几格通信”；summary token 表示“保留一条全局广播通道”；sink block 可以理解成“固定保留的落点”，让所有块至少有一部分稳定的全局连接。

| 术语 | 白话解释 | 主要职责 | 计算频次 |
| --- | --- | --- | --- |
| local block | 当前块附近的邻居块 | 保留局部细节、短距离依赖 | 最高 |
| sink block | 固定保留的共享块 | 提供稳定的全局锚点 | 中等 |
| summary token | 专门做全局摘要的 token | 传播长距离信息 | 低于 local，但影响范围大 |

问题的真正难点不在“能不能写一个稀疏 mask”，而在“跳过 90% 以上块之后，softmax 还能不能精确且高效地算”。如果先构造完整 $QK^\top$，再乘一个稀疏掩码，虽然数学上也能做，但工程上几乎没有意义，因为最贵的 IO 已经发生了。Gemini 风格方案要求的是：从 kernel 层面就不去读写那些被跳过的块。

---

## 核心机制与推导

稀疏掩码记作 $\tilde{M}\in\{0,1\}^{n_b\times n_b}$，其中 $n_b=N/B$，$B$ 是块大小。$\tilde{M}_{ij}=1$ 表示第 $i$ 个 query 块会看第 $j$ 个 key/value 块；为 0 则整块跳过。块稀疏度定义为：
$$
s=\frac{\|\tilde{M}\|_1}{n_b^2}
$$
它表示在全部可能块里，真正参与计算的比例。

标准 dense attention 的核心是：
$$
S = QK^\top,\quad P = \mathrm{softmax}(S),\quad O = PV
$$
而 block-sparse FlashAttention 不是先生成完整 $S$，而是只对 $\tilde{M}_{ij}=1$ 的块计算：
$$
S_{ij}=Q_iK_j^\top \quad \text{if } \tilde{M}_{ij}=1
$$
然后在 query 块内部做流式 softmax。这里的“流式”是指：softmax 所需的最大值和归一化分母，不必等所有块一次性都算完再处理，而是可以随着块遍历逐步累计。对白话读者来说，这相当于“边看边更新总分”，不需要先把所有候选项全部摊在桌上。

玩具例子如下。假设有 8 个块，每个块 4 个 token。对第 5 个 query 块，我们只允许它看到：
- 自己和左右各 1 个块，一共 3 个 local block
- 第 0 个 sink block
- 2 个 summary token 所在的小全局区

那么这个 query 块不再对全部 8 个块做 $QK^\top$，而只对极少数块做。可以把它画成：

```text
Query block 5 -> [sink][local 4][local 5][local 6][summary]
                [  1 ][   1   ][   1   ][   1   ][   1   ]
其余块全为 0，直接跳过
```

这和电子表格很像：只有被“解锁”的单元格区域会参与公式计算，其余区域根本不进入运算图。

为什么 IO 复杂度会变成
$$
\Theta(Nd + N^2 d^2 M^{-1} s)
$$
可以分两部分理解。

第一项 $Nd$ 是读入和写出 $Q,K,V,O$ 这些基本张量的代价。无论是否稀疏，只要序列长度是 $N$，这些向量总得进出一次，因此线性项通常无法消掉。

第二项 $N^2 d^2 M^{-1} s$ 是块级 tiled attention 的主要 IO 成本。它来自三个因素：
- $N^2$：若做全连接，query 和 key 的两两交互是二维规模。
- $d^2 M^{-1}$：块在片上 SRAM 中复用，SRAM 越大，重复搬运越少。
- $s$：并不是所有块都参与，而只保留比例为 $s$ 的块。

如果是 dense attention，$s=1$。如果只有 10% 的块被保留，那么 $s=0.1$，第二项会随之缩小一个数量级。注意这不是说总成本严格线性，而是说在实际长上下文区间里，最贵的那部分二维读写被稀疏度压缩了。

还可以把数据流写成简图：

```text
tokens -> split into blocks -> build block mask
      -> select allowed (Q_i, K_j, V_j) block pairs
      -> FlashAttention tile loop
      -> streaming max / exp-sum / output accumulation
      -> merge block outputs
```

这正是 Gemini 类融合方案的关键：mask 决定“算哪些块”，FlashAttention 决定“这些块如何高效算”。

---

## 代码实现

下面先给一个可运行的 Python 玩具实现。它不是高性能 kernel，而是帮助理解“块掩码 + 精确 softmax”的语义。代码会构造 local+global 的 block mask，然后只在允许位置上做 attention，并用 `assert` 检查 mask 行为。

```python
import math

def build_block_mask(num_blocks, local_radius, global_blocks):
    mask = [[0] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(max(0, i - local_radius), min(num_blocks, i + local_radius + 1)):
            mask[i][j] = 1
        for g in global_blocks:
            mask[i][g] = 1
    return mask

def sparse_attention_block_scores(q_blocks, k_blocks, v_blocks, mask):
    outputs = []
    for i, q in enumerate(q_blocks):
        scores = []
        values = []
        for j, allowed in enumerate(mask[i]):
            if allowed:
                s = sum(q[t] * k_blocks[j][t] for t in range(len(q)))
                scores.append(s)
                values.append(v_blocks[j])

        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        probs = [e / z for e in exps]

        out = [0.0] * len(values[0])
        for p, v in zip(probs, values):
            for t in range(len(v)):
                out[t] += p * v[t]
        outputs.append(out)
    return outputs

# 4 个块，每块向量维度为 2
q_blocks = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [1.0, 1.0]]
k_blocks = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]]
v_blocks = [[10.0, 0.0], [0.0, 10.0], [8.0, 2.0], [2.0, 8.0]]

mask = build_block_mask(num_blocks=4, local_radius=1, global_blocks=[0])
outputs = sparse_attention_block_scores(q_blocks, k_blocks, v_blocks, mask)

assert mask[2][2] == 1       # 自己块可见
assert mask[2][1] == 1       # 邻居块可见
assert mask[2][0] == 1       # 全局块可见
assert mask[2][3] == 1       # 右邻居可见
assert len(outputs) == 4
assert all(len(o) == 2 for o in outputs)
```

上面这段代码表达了正确的数学语义，但工程里不会这样逐块用 Python 循环。真实做法通常是：

1. 先根据 `local_blocks`、`sink_blocks`、`summary_tokens` 编译 block mask。
2. 把 mask 传给定制化 FlashAttention/XLA kernel。
3. 由 Pallas 或类似框架把 kernel 编译成适合 TPU tile 的执行计划。

伪代码可以写成：

```python
# pseudo-code
mask = compile_block_mask(
    seq_len=16384,
    block_size=128,
    local_radius=3,
    sink_blocks=[0],
    summary_tokens=64,
)

attn_fn = flash_attention_xla(
    block_mask=mask,
    block_size=128,
    causal=True,
    backend="tpu",
)

output = attn_fn(Q, K, V)
```

这里的 `compile_block_mask` 可以理解为“把业务规则变成硬件友好的块索引”。`Pallas` 是一种自定义 kernel 编排工具，白话讲，就是让你能更明确地控制 TPU 上每个 tile 如何装载、计算和写回。对 Gemini 风格系统，难点不是 API 调一次就结束，而是要让 mask 结构稳定到足以被编译器高效利用。

真实工程例子是 agent 推理。假设一个 16K 上下文中，前半部分是工具调用历史，中间是检索文档，后半部分是当前用户问题。此时局部窗口保证模型对当前段落细节有足够分辨率，summary token 保证前面很长的历史不会完全断掉。如果把这些模式直接编码进 block mask，TPU kernel 就不必扫描整张 dense 注意力图。

| 维度 | 固定 mask | 动态 mask |
| --- | --- | --- |
| 编译方式 | 预编译少量模板 | 可能频繁重编译或切换 |
| 执行稳定性 | 高 | 较低 |
| 延迟波动 | 小 | 容易抖动 |
| 适合场景 | 结构稳定的长上下文服务 | 强路由、输入差异极大的任务 |

---

## 工程权衡与常见坑

第一个坑是动态 mask。动态 mask 的意思是不同输入会激活不同块，白话讲，就是每条请求都可能走不同的注意力连接图。数学上这更灵活，但工程上会带来两个直接问题：编译缓存命中率下降，尾部延迟上升。TPU 这类硬件特别在意形状和调度稳定性，因此“每次都重新画图”往往不划算。

第二个坑是 memory fragmentation，也就是内存碎片化。白话讲，就是可用内存总量看起来够，但因为分配过碎，连续大块空间不好拿。动态稀疏和专家路由一起出现时，这个问题更明显，因为每一步激活的块和权重位置都在变。

第三个坑是 tile size 选择。块大小 128 在数学表达上常见，但并不意味着对所有 TPU 配置都最优。有时把计算 tile 调到 256，虽然单块看起来更粗，但会减少调度开销和边界处理，从而让总体吞吐更稳。

top-k 约束是常见的稳定器。它表示每个头或每个位置最多只激活前 $k$ 个候选块，白话讲，就是强制“只保留最重要的少数连接”。若每个头最多激活 $k$ 个附加块，则额外激活比例近似满足：
$$
\text{active ratio} \le \frac{k}{n_b}
$$
如果从“每个头可自由激活很多块”收紧到固定 top-k，那么 mask 变化频率和内存抖动都会下降。

| 常见坑 | 现象 | 原因 | 规避手段 |
| --- | --- | --- | --- |
| 尾部延迟高 | P99 明显高于平均值 | mask 变化过多，编译/调度不稳定 | 限定 top-k，缓存模板 |
| memory fragmentation | 显存或片上内存利用率差 | 动态激活导致分配碎片化 | 固定 shape，减少动态分支 |
| kernel 编译慢 | 首次请求或冷启动慢 | 稀疏模式过多 | 预编译常见 mask |
| 吞吐不稳定 | batch 一变就波动 | tile 和 block 不匹配 | 用 Pallas 调 block/tile 大小 |
| 精度退化 | 长距离依赖丢失 | global 通道过少 | 保留 summary/sink 路径 |

对新手来说，可以把固定模板理解成“先印好几张标准表格”。请求来了，不要临时画新表，而是尽量选一张最接近的现成模板直接填。这样流水线更稳定。

---

## 替代方案与适用边界

不是所有模型都该上 block-sparse FlashAttention。若上下文只有 2K 到 4K，或者任务本身没有清晰的局部结构，那么直接用全 Attention + FlashAttention-2 往往更简单。因为这时 dense 计算规模还在可控范围内，稀疏化带来的工程复杂度可能超过收益。

另一类替代方案是 routing-based sparse attention，例如 BigBird、Routing Transformer 这类方法。它们的共同特点是：不是固定 local+global，而是用随机连接、全局 token、路由器或内容选择机制决定谁和谁通信。优点是表达力更灵活，缺点是硬件友好性不一定好，尤其在 TPU 上，如果路由导致 shape 和访问模式频繁变化，优势会被削弱。

可以用一个简单决策表来记：

| 方案 | 适合上下文长度 | 硬件友好度 | 精度影响 | 适用边界 |
| --- | --- | --- | --- | --- |
| 全 Attention + FlashAttention-2 | $\le 4K$ 最常见 | 高 | 最稳 | 短到中等上下文 |
| Block-Sparse FlashAttention | $\ge 8K$ 更明显 | 对 TPU 友好 | 取决于 mask 设计 | 长上下文、局部结构明显 |
| Routing-based Sparse | 长上下文也可用 | GPU 上常更灵活 | 依赖路由质量 | 动态内容选择更重要的任务 |

玩具层面的判断也很简单：
- 短文本分类、小模型训练：优先纯 FlashAttention。
- 长文档问答、长对话 agent：优先 block-sparse Flash + summary token。
- 内容路由极强、模式不固定：考虑 routing-based sparse，但要评估编译和延迟成本。

更具体的适配边界可以写成一句工程规则：当上下文长度已经到 8K 以上，且部署硬件是 TPU v5e 或更高代际，同时输入存在明显的局部连续性时，block-sparse FlashAttention 才通常值得引入。如果上下文并不长，或者每个位置都必须看全局，那么 dense FlashAttention 往往更合适。

---

## 参考资料

1. Emergent Mind, *Block-Sparse FlashAttention*（2025-07）。这篇资料适合看稀疏 block mask、流式 softmax 和 IO 复杂度推导。  
2. RaphaLabs, *DeepMind Gemini 3 Flash: Frontier Intelligence Built for Speed*（2025-12）。这篇资料适合看 Gemini 风格系统在 TPU 上的吞吐、延迟和工程约束。  
3. PyTorch/XLA Docs, *Pallas on TPU*。这份文档适合看如何在 TPU 上组织自定义 kernel、tile 和编译流程。  
4. MIT Han Lab, *Block-Sparse-Attention*。这份实现资料适合看 block mask、local/global 稀疏模式和最小代码示例。
