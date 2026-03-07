## 核心结论

FlashAttention-3 不是在 FlashAttention-2 上做一次局部调优，而是针对 NVIDIA Hopper 架构重新安排 attention kernel 的执行方式。它把原本容易串行化的三段工作重新组织成流水线：

1. 计算 $QK^\top$
2. 维护 online softmax 的统计量
3. 计算 $\mathrm{softmax}(QK^\top)V$

关键不在公式变化，而在执行路径变化。FlashAttention-3 让“数据搬运”和“张量核计算”尽量并行发生，减少 GPU 在阶段切换中的空转时间。

这里有两个必须先认识的 Hopper 关键词。

`WGMMA` 是 Warp-Group Matrix Multiply-Accumulate。它是 Hopper 面向张量核的大块矩阵乘指令接口，允许一组 warp 协同发起更高吞吐的矩阵乘。对 attention 来说，这意味着 $QK^\top$ 和后续的 $PV$ 都能更贴近 Hopper 的硬件峰值执行。

`TMA` 是 Tensor Memory Accelerator。它负责把全局显存中的规则大块数据异步搬到 shared memory。传统实现里，很多数据搬运要靠线程自己发 load/store 指令；TMA 的意义是把这部分工作从“很多线程手动搬”改成“硬件辅助异步搬”。

FlashAttention-3 的收益主要来自这两点的组合：矩阵乘更适配 Hopper，数据搬运和计算能够重叠。

| 方案 | 计算组织方式 | 主要空闲来源 | Hopper 上的结果特征 |
|---|---|---|---|
| 传统 attention | 先完整算 $QK^\top$，再 softmax，再乘 $V$ | 阶段切换明显，中间结果反复写回 HBM | Tensor Core 利用率偏低 |
| FlashAttention-2 | tile 化 + online softmax，已减少中间写回 | 对 Hopper 新指令利用仍不充分 | 比传统实现快，但未完全吃满 H100 |
| FlashAttention-3 | WGMMA + TMA + producer/consumer 流水线 | 剩余开销主要变成同步、buffer 切换、尾部收尾 | FP16 路径约可达 75% 利用率，约 740 TFLOPS；原生 FP8 路径接近 1.2 PFLOPS |

可以用一个最小化的流水线例子理解它。假设一个 thread block 中，部分 warp 负责“准备下一块数据”，另一些 warp 负责“消费当前块进行矩阵乘和 softmax 更新”。当 tile 0 正在计算时，tile 1 已经在后台搬运；tile 0 结束后，消费者直接切到 tile 1，而不是重新等待数据到位。这样访存和计算不再严格串行。

真实工程中的典型场景是长上下文大模型训练和推理，尤其是 H100 上的 decoder-only 模型。序列一长，attention 的 HBM 带宽消耗、kernel 启动成本和中间结果写回成本都会放大。FlashAttention-3 通过更激进的流水线化，通常能把 attention 吞吐提升到原来的约 1.5 到 2 倍，并进一步提高可承受的 context length。

---

## 问题定义与边界

标准自注意力的形式是：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

其中：

- `Q`、`K`、`V` 分别是 query、key、value  
  白话说，`Q` 表示“当前 token 要找什么”，`K` 表示“每个 token 提供什么索引特征”，`V` 表示“真正被聚合出来的内容”。
- `d` 是单个 attention head 的宽度，也叫 `head_dim`。

公式并不复杂，难点在于如何在 GPU 上执行得高效。

传统实现往往按下面三步做：

1. 计算分数矩阵  
   $$
   S = QK^\top
   $$
2. 对 $S$ 做缩放和 softmax  
   $$
   P = \mathrm{softmax}\left(\frac{S}{\sqrt{d}}\right)
   $$
3. 再和 $V$ 相乘  
   $$
   O = PV
   $$

其中 `GEMM` 指通用矩阵乘法。GPU 最擅长的是这种大块、规则、可并行的乘加计算，所以第 1 步和第 3 步天生适合张量核；问题出在第 2 步。

瓶颈主要来自两类开销。

第一类是阶段切换。矩阵乘偏向规则的吞吐型计算，而 softmax 包含逐行求最大值、指数、求和、归一化等操作。这些步骤需要不同的数据访问模式和不同的计算单元。当实现把它们切成独立阶段时，中间会出现同步、调度和访存等待。

第二类是中间结果规模过大。设序列长度为 $n$，则 $QK^\top$ 的大小是 $n \times n$。如果把它完整写回 HBM，再读回来做 softmax，再把概率矩阵写回，再读回来乘 $V$，总数据搬运量会随着 $n^2$ 快速上升。长上下文下，这部分成本会直接压住计算吞吐。

FlashAttention 系列的边界一直很明确：它不改变模型定义，不改变 attention 的数学结果，而是重写内核的数据流。FlashAttention-3 进一步把优化目标收窄到 Hopper 这一代 GPU，因为它显式依赖 Hopper 的两项新能力：

- WGMMA：更高效地发起 warp-group 级矩阵乘
- TMA：把规则块数据异步搬到 shared memory

因此，FlashAttention-3 的适用边界可以写得很直接：

| 维度 | 结论 |
|---|---|
| 数学定义 | 不变，仍然是标准 attention |
| 主要优化对象 | kernel 数据流、访存路径、流水线组织 |
| 最佳硬件 | Hopper，尤其 H100/H800 类设备 |
| 最佳数值路径 | 原生 FP8 路径收益最大 |
| 不适用情形 | 非 Hopper、短序列、小 batch、未启用对应编译路径 |

如果设备不是 Hopper，或者软件栈没有真正启用 WGMMA/TMA/FP8，那么 attention 仍然能运行，但你拿到的不是 FlashAttention-3 的主要收益，只是“兼容路径下的可运行版本”。

---

## 核心机制与推导

FlashAttention-3 的核心思想可以压缩成一句话：按块计算，并让“块的准备”和“块的消费”重叠执行。

先从一个小尺寸例子开始。设：

- `head_dim = 64`
- 每次处理一个 query tile，大小为 32
- 每次处理一个 key/value tile，大小也为 32

那么一个块可以写成：

$$
Q_t \in \mathbb{R}^{32 \times 64},\quad
K_t \in \mathbb{R}^{32 \times 64},\quad
V_t \in \mathbb{R}^{32 \times 64}
$$

对这个块，先计算局部分数矩阵：

$$
S_t = Q_t K_t^\top \in \mathbb{R}^{32 \times 32}
$$

若 `head_dim = 64`，缩放因子为：

$$
\mathrm{scale} = \frac{1}{\sqrt{64}} = \frac{1}{8} = 0.125
$$

于是块内计算的目标变成：

$$
P_t = \mathrm{softmax}(0.125 \cdot S_t),\qquad
O_t = P_t V_t
$$

如果只看这个公式，似乎和普通 attention 没区别。真正的差异在于：FlashAttention-3 不会先把完整的 $S = QK^\top$ 算出来再统一做 softmax，而是边处理 tile，边维护每一行的 softmax 统计量。

### 1. 为什么需要 online softmax

普通 softmax 对一行分数 $s_1,\dots,s_n$ 的定义是：

$$
\mathrm{softmax}(s_j)=\frac{e^{s_j}}{\sum_{k=1}^n e^{s_k}}
$$

为了避免数值溢出，实际实现通常写成：

$$
\mathrm{softmax}(s_j)=\frac{e^{s_j-m}}{\sum_{k=1}^n e^{s_k-m}},
\qquad
m=\max_k s_k
$$

问题在于，若按 tile 分块读取，就不能一开始拿到整行所有分数。online softmax 的做法是对每一行维护两个状态：

- 当前已见分数的最大值 $m_i$
- 对应的归一化项 $l_i$

当读到一个新 tile 后，假设这个 tile 中该行的最大值为 $\tilde{m}_i$，则新的最大值是：

$$
m_i' = \max(m_i, \tilde{m}_i)
$$

原来累计的归一化项需要重标定：

$$
l_i' = l_i \cdot e^{m_i - m_i'} + \sum_{j \in \text{tile}} e^{s_{ij} - m_i'}
$$

若还维护输出累积向量 $a_i$，则它也要按相同因子缩放：

$$
a_i' = a_i \cdot e^{m_i - m_i'} + \sum_{j \in \text{tile}} e^{s_{ij} - m_i'} v_j
$$

最后真正输出为：

$$
o_i = \frac{a_i}{l_i}
$$

这套更新公式的意义是：每处理一个 tile，就把它并入“到目前为止已经看到的全部 key/value”。因此不需要把整张 $QK^\top$ 矩阵落回 HBM。

### 2. 为什么这能减少显存压力

设序列长度为 $n$，单头宽度为 $d$。传统实现显式物化 $QK^\top$ 时，需要保存一个 $n \times n$ 的中间矩阵；FlashAttention 的 tile 化思路只需要保存：

- 当前 tile 的 $Q_t, K_t, V_t$
- 每一行的状态 $(m_i, l_i, a_i)$
- 若干 shared memory / register 中间量

换句话说，空间复杂度从“需要容纳完整分数矩阵”转成“只需要容纳局部块和行状态”。这是 FlashAttention 系列能处理更长序列的重要原因。

### 3. Hopper 上为什么进一步提速

FlashAttention-2 已经做了 tile 化和 online softmax。FlashAttention-3 的新增重点是让 Hopper 的硬件能力真正参与调度。

在 Hopper 上，一个 thread block 内部可以进一步分工：

- `producer warp group`：负责用 TMA 异步把下一块 Q/K/V 搬进 shared memory
- `consumer warp group`：负责用 WGMMA 发起矩阵乘，并更新 softmax 状态与输出累积

这通常会形成双缓冲，即 ping-pong buffer。

| 时间片 | Buffer 0 | Buffer 1 | 生产者状态 | 消费者状态 |
|---|---|---|---|---|
| t0 | 加载 tile 0 | 空 | 搬运 tile 0 | 等待首块 |
| t1 | 计算 tile 0 | 加载 tile 1 | 搬运 tile 1 | 处理 tile 0 的 $QK^\top$ 与 softmax |
| t2 | 加载 tile 2 | 计算 tile 1 | 搬运 tile 2 | 处理 tile 1 的 $PV$ 与累积 |
| t3 | 计算 tile 2 | 加载 tile 3 | 搬运 tile 3 | 切换到 tile 2 |

这张表想表达的不是“每个时间片只做一件事”，而是“同一时刻，搬运和计算在不同 buffer 上并行发生”。

### 4. FP8 路径为什么重要

Hopper 上，FlashAttention-3 不只是更会调度，还引入了更激进的低精度路径。`FP8` 是 8 位浮点数，相比 FP16/BF16，它的优势是：

- 占用更少显存带宽
- 可以让硬件以更高吞吐执行矩阵乘
- 更容易把 attention 推到接近 H100 的高利用率区间

代价是数值范围和精度都更紧。

Hopper 常见 FP8 格式如下：

| 格式 | 指数/尾数特征 | 动态范围特征 | 典型用途 |
|---|---|---|---|
| FP16 | 5 位指数、10 位尾数 | 范围和精度较稳 | 通用训练与推理 |
| FP8 E4M3 | 4 位指数、3 位尾数 | 精度相对更好，但范围更窄，常近似看作约 $\pm 240$ | 前向激活、部分 GEMM |
| FP8 E5M2 | 5 位指数、2 位尾数 | 范围更大，但精度更粗 | 对范围更敏感的路径 |

为了让 E4M3 可用，FlashAttention-3 常配合 block-wise scaling。可写成：

$$
x_q=\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s_b}\right), q_{\min}, q_{\max}\right),
\qquad
\hat{x}=s_b \cdot x_q
$$

其中：

- $x$ 是原值
- $s_b$ 是块级缩放因子
- $x_q$ 是量化后的低精度表示
- $\hat{x}$ 是反量化后的近似值

这套机制的直觉是：不要用一个全局固定尺度去量化所有数据，而是“每个块各自选一个尺度”。这样一个块里即使存在较大值，也不至于立刻把小值全部压没。

一些实现还会配合 `incoherent orthogonal transform`。它可以理解为量化前的预处理：先对向量做近似保范数的正交变换，把极端值更均匀地分散到多个维度，降低少数维度过大导致的 FP8 溢出风险。这个步骤不是 FlashAttention-3 的定义本体，但在工程上常被用来提高 FP8 路径的稳定性。

### 5. 一个完整的数据流视角

把以上机制合在一起，单个 query block 的计算流程可以写成：

1. producer 通过 TMA 把下一批 $K_t, V_t$ 和可能需要的 $Q_t$ 搬入 shared memory
2. consumer 用 WGMMA 计算局部 $S_t = Q_tK_t^\top$
3. 对 $S_t$ 做缩放，并更新每行的 $(m_i, l_i)$
4. 根据该 tile 的概率权重累积 $a_i$
5. 释放当前 buffer，切换到下一个 tile
6. 全部 tile 结束后输出 $o_i = a_i / l_i$

它优化的不是理论 FLOPs，而是“同样的 FLOPs，如何减少等待、减少中间写回、减少访存停顿”。

---

## 代码实现

先给一个教学版实现。它有三个目标：

1. 保留 FlashAttention 的核心数据流
2. 代码可以直接运行
3. 能和朴素 attention 的结果对齐，证明 online softmax 的逻辑是对的

下面这段 Python 代码不依赖任何第三方库，直接运行即可。

```python
import math


def matmul_attention_naive(q, k, v):
    """
    朴素 attention 实现：
    1. 显式算出每个 query 对所有 key 的分数
    2. 做稳定 softmax
    3. 用概率加权 v
    q, k, v: list[list[float]], shape = [n, d]
    """
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)
    out = [[0.0 for _ in range(d)] for _ in range(n)]

    for i in range(n):
        scores = []
        for j in range(n):
            s = 0.0
            for t in range(d):
                s += q[i][t] * k[j][t]
            scores.append(s * scale)

        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        denom = sum(exps)

        for j in range(n):
            p = exps[j] / denom
            for t in range(d):
                out[i][t] += p * v[j][t]

    return out


def flash_attention_toy(q, k, v, block=2):
    """
    教学版 FlashAttention：
    - 按 key/value 维度分块
    - 对每个 query 行维护 online softmax 状态
    - 不显式保存完整 QK^T
    """
    n = len(q)
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)
    out = [[0.0 for _ in range(d)] for _ in range(n)]

    for i in range(n):
        m_i = float("-inf")   # 当前已见分数最大值
        l_i = 0.0             # 当前归一化项
        acc = [0.0 for _ in range(d)]  # 当前加权和累积

        for start in range(0, n, block):
            end = min(start + block, n)

            scores = []
            for j in range(start, end):
                s = 0.0
                for t in range(d):
                    s += q[i][t] * k[j][t]
                scores.append(s * scale)

            block_max = max(scores)
            new_m = max(m_i, block_max)

            if m_i == float("-inf"):
                old_factor = 0.0
            else:
                old_factor = math.exp(m_i - new_m)

            # 旧累积先按新基准重标定
            for t in range(d):
                acc[t] *= old_factor
            new_l = l_i * old_factor

            # 再并入当前块
            probs_unnorm = []
            for s in scores:
                p = math.exp(s - new_m)
                probs_unnorm.append(p)
                new_l += p

            for local_idx, j in enumerate(range(start, end)):
                p = probs_unnorm[local_idx]
                for t in range(d):
                    acc[t] += p * v[j][t]

            m_i = new_m
            l_i = new_l

        out[i] = [acc[t] / l_i for t in range(d)]

    return out


def max_abs_diff(a, b):
    diff = 0.0
    for i in range(len(a)):
        for j in range(len(a[0])):
            diff = max(diff, abs(a[i][j] - b[i][j]))
    return diff


if __name__ == "__main__":
    q = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    k = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    v = [
        [10.0, 1.0],
        [1.0, 10.0],
        [5.0, 5.0],
    ]

    naive = matmul_attention_naive(q, k, v)
    flash = flash_attention_toy(q, k, v, block=2)

    diff = max_abs_diff(naive, flash)
    print("naive =", naive)
    print("flash =", flash)
    print("max_abs_diff =", diff)

    assert diff < 1e-9
    assert flash[0][0] > flash[0][1]
    assert flash[1][1] > flash[1][0]
    print("check passed")
```

这段代码有两个阅读重点。

第一，它没有构造完整的 $QK^\top$。对每个 query 行，它只是在遍历 key/value 的过程中不断更新：

- 当前最大值 $m_i$
- 当前归一化项 $l_i$
- 当前输出累积 `acc`

第二，它在每个 block 上都会做一次“旧状态重标定”。这是 online softmax 正确性的核心。如果只把新块直接拼上去，而不按新的最大值重缩放，最终结果会偏离真实 softmax。

下面用一个极小的手算例子帮助理解。设某个 query 对三条 key 的缩放后分数为：

$$
[2.0,\ 1.0,\ 3.0]
$$

若分成两个块：前两项一个块，最后一项一个块。

第一块处理后：

$$
m^{(1)} = 2.0
$$

$$
l^{(1)} = e^{2.0-2.0} + e^{1.0-2.0}
= 1 + e^{-1}
$$

第二块最大值是 $3.0$，所以新最大值变为：

$$
m^{(2)} = 3.0
$$

旧归一化项要按新基准重标定：

$$
l^{(2)} = l^{(1)} e^{2.0-3.0} + e^{3.0-3.0}
$$

即：

$$
l^{(2)} = (1 + e^{-1})e^{-1} + 1
$$

这和直接对整行按最大值 3.0 做稳定 softmax 的分母是一致的：

$$
e^{2-3} + e^{1-3} + e^{3-3}
= e^{-1} + e^{-2} + 1
$$

这就是 online softmax 能与标准 softmax 保持数学等价的原因。

真实工程里，当然不会用 Python 循环实现 attention。真正的 Hopper kernel 会把这些逻辑映射到共享内存、寄存器和 warp-group 协作上。下面给一个更接近工程结构的伪代码，重点看数据流，而不是语法。

```python
# 伪代码：展示 FlashAttention-3 风格的数据流
# 不可直接编译，但结构与真实实现接近

shared q_buf[2], k_buf[2], v_buf[2]
shared ready_flag[2]

producer_warp_group:
    for tile in tiles:
        buf = tile % 2
        wait_until_buffer_released(buf)

        tma_load(Q_tile[tile], q_buf[buf])
        tma_load(K_tile[tile], k_buf[buf])
        tma_load(V_tile[tile], v_buf[buf])

        signal_ready(buf)

consumer_warp_group:
    init row_max m_i = -inf
    init row_sum l_i = 0
    init row_acc a_i = 0

    for tile in tiles:
        buf = tile % 2
        wait_ready(buf)

        # 可选：原生 FP8 路径下的块级量化
        q_tile = block_quantize(q_buf[buf], fmt="E4M3")
        k_tile = block_quantize(k_buf[buf], fmt="E4M3")
        v_tile = block_quantize(v_buf[buf], fmt="E4M3")

        s_tile = wgmma_qk(q_tile, k_tile)   # QK^T
        s_tile = s_tile * scale

        m_i, l_i, prob_tile = update_online_softmax(s_tile, m_i, l_i)
        a_i = accumulate_pv(prob_tile, v_tile, a_i)

        release_buffer(buf)

    o_i = a_i / l_i
```

把双缓冲展开后，可以更直观看到交替关系：

| tile 编号 | 当前计算使用 | 后台搬运使用 | 流水线状态 |
|---|---|---|---|
| 0 | buffer0 | buffer1 准备 tile1 | 首块需要预热，重叠尚未充分形成 |
| 1 | buffer1 | buffer0 准备 tile2 | 开始进入稳定阶段 |
| 2 | buffer0 | buffer1 准备 tile3 | 计算与访存显著重叠 |
| 3 | buffer1 | buffer0 准备 tile4 | 接近 steady state |

真实工程例子是 H100 上的长上下文推理。假设一个 decoder-only 模型把序列长度从 4K 提升到 32K。此时 attention 的两个成本会明显上升：

- $QK^\top$ 对应的 key/value 扫描范围大幅增加
- 中间状态维护和 HBM 读写更容易成为瓶颈

如果这条链路成功走到 Hopper 原生 FP8 路径，常见现象是 attention 不再长期压住 MLP，端到端 profiler 中 bottleneck 的位置会发生转移。这也是为什么很多工程报告里会看到“attention 优化后，下一瓶颈变成别的模块”。

---

## 工程权衡与常见坑

FlashAttention-3 最容易被误解的地方，是“只要机器是 H100，就会自动变快”。这个判断不成立。性能收益至少依赖三个条件同时满足：

1. 硬件确实是 Hopper
2. 内核实现确实使用了 Hopper 特性
3. 编译和运行路径确实走到了对应 kernel

缺任何一项，都可能出现“功能上可运行，但性能没有达到预期”的结果。

### 1. 没有真正走到 native FP8 路径

很多问题出在这里。表面上框架配置写了 FP8，甚至日志里也出现了低精度关键字，但实际执行路径可能因为以下原因退回 FP16/BF16：

- CUDA 版本不匹配
- 编译参数未开启 Hopper 目标架构
- 框架封装没有调到 native FP8 kernel
- 某些 head_dim / layout / mask 组合未命中优化路径

结果就是：能跑，但吞吐优势明显缩水。

| 构建/运行状态 | 实际执行精度路径 | 典型结果 |
|---|---|---|
| 启用 Hopper 原生 FP8 + 对应 kernel | WGMMA/TMA + FP8 | 接近论文或官方实现展示的高吞吐区间 |
| 使用 FlashAttention，但未走原生 FP8 | 多为 FP16/BF16 | 仍比常规 attention 快，但达不到 FP8 峰值 |
| 回退到常规 attention/cuBLAS | FP16/BF16 | 兼容性最好，但中间写回最多、利用率最低 |

### 2. 误以为 FP8 只是“改一下 dtype”

不是。FP8 的核心难点不在 API，而在数值范围管理。

对 E4M3，常用近似可写成：

$$
|x| \le x_{\max} \approx 240
$$

若块缩放因子为 $s_b$，为了让量化后数值不溢出，需要满足近似关系：

$$
\left|\frac{x}{s_b}\right| \le 240
$$

因此：

$$
s_b \ge \frac{|x|}{240}
$$

这个式子说明了一个直接事实：块内最大值越大，所需缩放因子越大；缩放因子越大，小值被量化时损失的有效精度就越多。所以 block-wise scaling 不是锦上添花，而是 FP8 稳定运行的基础条件。

### 3. 短序列下收益有限

流水线不是免费午餐。它需要：

- 预热成本
- buffer 切换
- 同步成本
- 额外调度复杂度

若 `seq_len` 很短、batch 很小、tile 数量不够，流水线还没进入稳定状态就结束了。此时一个结构更复杂的 kernel 不一定比简单实现更划算。

可以用一个粗略判断理解：

| 场景 | FlashAttention-3 收益倾向 |
|---|---|
| 长序列、大 batch、attention 占主瓶颈 | 收益明显 |
| 中等序列，attention 仍然较重 | 可能有收益，需 profiler 验证 |
| 短序列、小 batch、服务追求稳定而非极限吞吐 | 收益可能不明显 |

### 4. kernel 快，不等于端到端一定同步翻倍

这是第二个常见误解。即使 attention kernel 吞吐提升 2 倍，整机推理或训练吞吐也未必提升 2 倍，因为端到端系统里还存在：

- KV cache 管理
- 跨卡通信
- 调度器开销
- HBM 分页和碎片管理
- MLP、layernorm、采样等其他算子

因此正确的问题不是“FlashAttention-3 的论文峰值是多少”，而是“在我的工作负载里，attention 当前占多少、优化后还剩多少”。

### 5. 可运行，不代表可维护

FlashAttention-3 的实现复杂度明显高于常规 attention，也高于很多用户熟悉的 FlashAttention-2 路径。复杂度主要来自：

- 多阶段异步流水线
- 更严格的 shared memory / register 预算
- 对 layout、tile shape、head_dim 的更细适配
- FP8 路径下的量化、缩放和误差控制

因此工程上要考虑的不只是“能不能跑”，还包括“升级 CUDA/驱动/框架后是否稳定”“fallback 路径是否明确”“性能退化是否能被 profiler 及时发现”。

---

## 替代方案与适用边界

如果你的设备不是 Hopper，或者软件栈暂时无法稳定启用 WGMMA/TMA/FP8，那么最合理的策略通常不是硬追 FlashAttention-3，而是选一个和当前约束匹配的方案。

| 方案 | 适合设备 | 适合序列长度 | 实现复杂度 | 吞吐上限特征 |
|---|---|---|---|---|
| cuBLAS / 标准 attention | 通用 GPU | 短序列、小 batch | 最低 | 易用，但中间写回多 |
| FlashAttention-2 | Ampere、Ada、部分 Hopper | 中长序列 | 中等 | 已明显优于标准 attention |
| FlashAttention-3 | Hopper，且 FP8 ready | 长序列、高吞吐场景 | 最高 | 最能吃满 H100 的新特性 |

把边界再压缩成一句话：

- 适用边界：`Hopper + long context + FP8 ready`
- 不强求场景：`短序列 + 小 batch + attention 不是主要瓶颈`

如果希望更工程化一点，可以按下面的判断顺序做决策：

1. 先看 profiler  
   attention 如果不是热点，就不要先优化它。
2. 再看硬件  
   不是 Hopper，就优先考虑 FlashAttention-2。
3. 再看软件栈  
   如果 native FP8 路径不稳定，先用稳妥的 FP16/BF16 路径。
4. 再看 workload  
   长上下文、高并发、训练或重吞吐推理场景，才更容易把 FlashAttention-3 的成本摊薄。

用一个玩具尺度理解也很直接。若序列长度只有 16，tile 只有两三块，双缓冲几乎没有足够空间形成稳定重叠；但如果序列长度达到 16K、32K，tile 数量足够多，流水线就更容易长期工作在 steady state。

因此，FlashAttention-3 不是“新的默认 attention 实现”，而是“在特定硬件和特定工作负载下，把 attention 推向更高吞吐上限的专门方案”。

---

## 参考资料

| 来源 | 类型 | 内容重点 |
|---|---|---|
| Tri Dao 等，《FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision》 | 论文 | FlashAttention-3 的核心设计、WGMMA/TMA、异步流水线、FP8 路径 |
| Dao-AILab/flash-attention 仓库 | 官方代码 | Hopper 支持状态、构建方式、内核实现与使用说明 |
| NVIDIA Hopper Architecture / PTX / CUDA 文档 | 官方文档 | WGMMA、TMA、warp-group、shared memory 异步搬运机制 |
| Hugging Face Papers 页面 | 论文索引 | 论文摘要、核心结论、与前代方法的差异概览 |
| Together.ai 技术博客 | 工程博客 | H100 上训练与推理的吞吐收益、部署经验、瓶颈迁移现象 |
| Rohan Paul 的技术文章 | 工程博客 | producer/consumer 模型、流水线和在线 softmax 的实现解释 |
| Emergent Mind 相关主题页 | 综述 | block-wise FP8、数值稳定性、正交变换等背景脉络 |
| GitHub issue / discussion（flash-attention、框架集成仓库） | 工程讨论 | native FP8 回退、编译链问题、不同 CUDA/驱动版本下的常见坑 |

阅读顺序建议如下：

1. 先读论文，明确 FlashAttention-3 到底改了什么  
2. 再读官方仓库，确认哪些路径是真正受支持的  
3. 最后看工程博客和 issue，理解部署时最常见的退化点

如果只记一个结论，可以记这个版本：

- FlashAttention-2 解决的是“不要把 attention 中间结果完整写回显存”
- FlashAttention-3 进一步解决的是“在 Hopper 上，如何让 attention 的搬运和计算真正重叠，并把 FP8 路径安全地跑起来”
