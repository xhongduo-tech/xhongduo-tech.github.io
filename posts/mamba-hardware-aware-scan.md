## 核心结论

Mamba 的硬件感知扫描算法，本质上是在解决一个很具体的问题：**Selective SSM 是时变系统，无法像传统 LTI SSM 那样提前化成固定卷积核，因此不能直接用 FFT 做全序列加速**。这里的“时变系统”可以先理解成“每个 token 都带着自己的一套状态更新规则”。

它的核心递推是：

$$
h_k = A_k h_{k-1} + B_k x_k,\qquad y_k = C_k h_k
$$

其中 $A_k,B_k,C_k$ 都依赖当前 token，$A_k$ 还由步长 $\Delta_k$ 离散化得到。因为参数随位置变化，系统不再满足“同一个核扫完整条序列”的条件，所以经典卷积化和 FFT 路线失效。

Mamba 的做法不是回到低效串行，而是把递推改写成**可做 parallel scan 的前缀合并问题**。这样总工作量仍是 $O(L)$，并行深度变成 $O(\log L)$。这里的“并行深度”可以先理解成“GPU 最少需要几轮同步才能算完整段序列”。

更关键的是，Mamba 不只改数学形式，还改执行位置：**把 $\Delta/B/C$ 的投影、离散化、scan、输出投影尽量融合进一个 kernel，在 SRAM 中完成主要计算，只把最终输出写回 HBM**。这里的 SRAM 可以先理解成“GPU 内部更快、更小的工作区”，HBM 可以先理解成“容量大但访问更慢的显存”。

结论可以压缩成三点：

| 结论 | 含义 | 直接收益 |
|---|---|---|
| 不能用 FFT | 参数随 token 变化，系统不再是 LTI | 必须放弃固定卷积核思路 |
| 必须用 scan | 把递推改写成满足结合律的前缀合并 | 保持 $O(L)$ 工作量和 $O(\log L)$ 并行深度 |
| 必须硬件感知 | 核心中间态不落到 HBM，后向尽量重算 | 降低 IO，逼近 FlashAttention 级内存开销 |

一个最直观的理解方式是：**Mamba 不是“把长序列一次卷掉”，而是“把每个 token 的小更新块，在快内存里并行合并，再只写出结果”**。

---

## 问题定义与边界

先看问题本身。传统 SSM 常写成：

$$
h_k = A h_{k-1} + B x_k,\qquad y_k = C h_k
$$

这里 $A,B,C$ 是常量，因此整条序列共享同一套动力学规则。这种模型叫 **LTI SSM**，即线性时不变状态空间模型。白话解释就是：不同位置看到的是同一台“状态机”。

Mamba 的 Selective SSM 改成：

$$
h_k = A_k h_{k-1} + B_k x_k,\qquad y_k = C_k h_k
$$

并且 $A_k$ 通常来自 $\Delta_k$ 对连续系统的离散化。也就是说，**每个 token 都能改变“记忆保留多久、写入多少、读出什么”**。这就是“选择性”的含义。

这带来两个直接后果。

第一，不能再把整个系统看成固定卷积。卷积要求核在时间轴上不变，但这里每一步核都在变。

第二，不能把问题简单理解成“理论复杂度线性就够了”。在 GPU 上，真正决定吞吐的常常不是 FLOPs，而是 IO。白话说，**不是算得慢，而是搬数据慢**。如果把中间状态、离散化后的参数、大张量展开结果都写回 HBM，再读出来继续算，理论上的线性复杂度会被显存流量拖垮。

下面这个表格可以把边界说清楚：

| 维度 | LTI SSM | Selective SSM（Mamba） |
|---|---|---|
| 参数是否随 token 变化 | 否 | 是 |
| 能否预先化为固定卷积核 | 可以 | 不可以 |
| 能否直接用 FFT 加速 | 可以 | 通常不可以 |
| 序列并行方式 | 卷积或频域乘法 | 并行 scan |
| 主要工程瓶颈 | 算法实现 | IO 与 kernel 设计 |
| 中间状态是否容易展开存储 | 相对容易 | 代价高，容易爆显存 |

一个玩具理解例子：假设你要处理字符串 `"abc"`。在 LTI SSM 里，`a`、`b`、`c` 都用同一组 $A,B,C$ 更新状态；在 Selective SSM 里，`a` 可能希望“快速忘记”、`b` 希望“保留更多历史”、`c` 希望“更强读出当前状态”。这就是“每个字符都自带一套算子”。

所以本文讨论的边界也很明确：

- 讨论对象是 Mamba 一类的 Selective SSM。
- 重点不是模型效果，而是**为什么必须改写执行算法**。
- 重点不是 CUDA 语法细节，而是**数学可并行性和硬件内存层级如何配合**。

---

## 核心机制与推导

关键在于把递推写成可结合的块合并。

先从单步状态更新开始。定义每个位置 $k$ 的局部块为：

$$
(P_k, S_k) = (A_k, B_k x_k)
$$

它表示“先把旧状态乘上 $A_k$，再加上新输入贡献 $B_kx_k$”。于是单步更新就是：

$$
h_k = P_k h_{k-1} + S_k
$$

现在考虑连续两步。先过第 1 步，再过第 2 步：

$$
h_2 = P_2(P_1 h_0 + S_1) + S_2
$$

展开得：

$$
h_2 = (P_2P_1)h_0 + (S_2 + P_2S_1)
$$

因此两段块可以合并成一个更大的块：

$$
(P_2,S_2)\otimes(P_1,S_1) = (P_2P_1,\ S_2 + P_2S_1)
$$

这就是 scan 的核心操作。它的意义是：**一段序列本身也能被表示成“一个总乘子 + 一个总偏移项”**。

更重要的是，这个操作满足结合律：

$$
((P_3,S_3)\otimes(P_2,S_2))\otimes(P_1,S_1)
=
(P_3,S_3)\otimes((P_2,S_2)\otimes(P_1,S_1))
$$

有了结合律，GPU 就能用树形规约做 prefix-scan。树形规约可以先理解成“先两两合并，再合并合并结果”，所以并行深度是 $O(\log L)$，而不是逐 token 串行的 $O(L)$ 轮依赖。

### 玩具例子

设初始状态 $h_0=0$，并且是最简单的一维状态：

- 第一个 token：$A_1=0.5,\ B_1x_1=1$
- 第二个 token：$A_2=0.4,\ B_2x_2=4$

串行算：

$$
h_1 = 0.5\times 0 + 1 = 1
$$

$$
h_2 = 0.4\times 1 + 4 = 4.4
$$

scan 写法：

- 第一个块：$(P_1,S_1)=(0.5,1)$
- 第二个块：$(P_2,S_2)=(0.4,4)$

合并后：

$$
P_{12}=P_2P_1=0.4\times 0.5=0.2
$$

$$
S_{12}=S_2 + P_2S_1=4 + 0.4\times 1=4.4
$$

于是整段的结果是：

$$
h_2 = P_{12}h_0 + S_{12} = 0.2\times 0 + 4.4 = 4.4
$$

结果与串行完全一致，但合并过程已经变成 scan 友好的形式。

### 为什么这能映射到 GPU

GPU 不怕并行乘加，怕的是频繁往 HBM 来回搬大张量。Mamba 的设计抓住了这一点：

1. 先把当前 chunk 的 $\Delta,B,C,x$ 拉到 SRAM。
2. 立刻完成离散化，得到当前步需要的 $A_k$ 与输入项。
3. 在 SRAM 里做分段 scan，得到各前缀块。
4. 用 $C_k$ 做读出，写回最终 $y_k$。
5. 不把完整状态轨迹 $h_1,\dots,h_L$ 写回 HBM。

可以把它想成一条流水线：**参数一进快内存，就在那里被“榨干”，只剩结果出去**。这正是“硬件感知”的含义。

如果画成文字流程，就是：

```text
token params -> SRAM
SRAM内离散化 Δ -> A
SRAM内形成局部块 (P,S)
SRAM内并行 prefix-scan
SRAM内乘 C 得到 y
仅输出 y 到 HBM
```

数学上，scan 解决了“能不能并行”；工程上，SRAM fusion 解决了“并行后会不会被 IO 吃掉”。

---

## 代码实现

下面先给一个可运行的 Python 版本，用来验证 scan 合并与串行递推结果一致。它不是 GPU 实现，但能把核心代数关系讲清楚。

```python
from typing import List, Tuple

def sequential_scan(P: List[float], S: List[float], h0: float = 0.0) -> List[float]:
    h = h0
    out = []
    for p, s in zip(P, S):
        h = p * h + s
        out.append(h)
    return out

def combine(block2: Tuple[float, float], block1: Tuple[float, float]) -> Tuple[float, float]:
    p2, s2 = block2
    p1, s1 = block1
    return (p2 * p1, s2 + p2 * s1)

def prefix_blocks(P: List[float], S: List[float]) -> List[Tuple[float, float]]:
    blocks = []
    acc = (1.0, 0.0)  # identity block
    for p, s in zip(P, S):
        acc = combine((p, s), acc)
        blocks.append(acc)
    return blocks

# 玩具例子
P = [0.5, 0.4]
S = [1.0, 4.0]

seq = sequential_scan(P, S, h0=0.0)
blocks = prefix_blocks(P, S)
scan_values = [p * 0.0 + s for p, s in blocks]

assert seq == [1.0, 4.4]
assert scan_values == [1.0, 4.4]

# 再测一个稍长例子
P2 = [0.8, 0.7, 0.9]
S2 = [2.0, -1.0, 3.0]
seq2 = sequential_scan(P2, S2, h0=1.5)
blocks2 = prefix_blocks(P2, S2)
scan2 = [p * 1.5 + s for p, s in blocks2]

for a, b in zip(seq2, scan2):
    assert abs(a - b) < 1e-9
```

上面这段代码说明两件事：

- 局部块合并公式是正确的。
- 前缀块一旦算好，任意位置的状态都能由 $h_k=P_{1:k}h_0+S_{1:k}$ 直接得到。

真实工程里当然不会这么写。真正的 kernel 会做更激进的融合。可以用下面的伪代码概括前向：

```python
# SRAM-resident tensors: delta_chunk, B_chunk, C_chunk, x_chunk, A_chunk, PS_chunk
for chunk in sequence_chunks:
    delta_chunk, B_chunk, C_chunk, x_chunk = load_from_hbm(chunk)
    A_chunk = discretize(delta_chunk)          # 在 SRAM 中完成
    PS_chunk = parallel_scan(A_chunk, B_chunk * x_chunk)
    y_chunk = C_chunk * PS_chunk.state_like    # 读出
    write_to_hbm(y_chunk)
```

如果把工程动作拆得更细，可以写成：

| 步骤 | 在哪里做 | 为什么 |
|---|---|---|
| 读取 $\Delta/B/C/x$ | HBM -> SRAM | 输入必须进来，但只搬一次 |
| 离散化得到 $A$ | SRAM | 避免写回中间参数 |
| 构造 $(P,S)$ 并 scan | SRAM | 核心递推必须留在快内存 |
| 乘 $C$ 生成输出 | SRAM | 避免额外 kernel 和额外访存 |
| 写回 $y$ | SRAM -> HBM | 只写最终结果 |

### 真实工程例子

假设你在训练一个长上下文语言模型，序列长度达到 $10^6$ token。此时如果像普通递推那样把每一步状态都存下来，内存规模接近：

$$
O(B \times L \times D \times N)
$$

这里 $B$ 是 batch，$L$ 是序列长度，$D$ 是通道维度，$N$ 是状态维度。这个量级在百万 token 下会很快失控。

Mamba 的工程实现会改成：

- 前向只保留少量必要边界信息，不保留完整状态轨迹。
- 反向传播时，对需要的 chunk 重新做离散化和 scan。
- 用重计算换显存，用 fusion 换 IO。

白话解释就是：**宁可重新算，也不要把一大堆中间结果存在显存里等会儿再读**。在现代 GPU 上，这通常是对的，因为算力增长速度比显存带宽更快。

---

## 工程权衡与常见坑

Mamba 的扫描算法不是“数学上能做就自然高效”，它非常依赖具体实现方式。以下几个坑最常见。

| 常见误区 | 后果 | 规避方式 |
|---|---|---|
| 先离散化出 $A/B$，再写回 HBM 等后续 kernel 读取 | 整体变成 memory-bound，吞吐明显下降 | 把投影、离散化、scan、读出尽量 fusion 到同一条流水线 |
| 前向保存全部 $h_k$ 供反向使用 | 显存和 IO 变成 $O(BLDN)$ 级别 | 前向少存，后向重算 |
| 只关注算子复杂度，不关注内存层级 | 理论是线性的，实际跑不快 | 设计时先看 HBM 读写次数，再看 FLOPs |
| 把 scan 理解成普通 for 循环并行化 | GPU 利用率低，无法扩展到长序列 | 使用满足结合律的块合并形式 |
| 忽略 chunk 边界状态传递 | 分块结果错误，长序列前后不一致 | 为每个 chunk 保存必要前缀摘要或边界状态 |

最关键的权衡是：**重计算 vs 存储**。

如果前向把所有状态都存起来，反向传播会简单一些，但会付出巨大的显存和 IO 成本。如果选择反向重算，训练时间会增加一部分计算，但显存压力显著下降。在长序列训练里，这个交换通常是划算的。

另一个权衡是：**fusion 程度 vs kernel 复杂度**。

融合越多，HBM 往返越少，性能通常越好；但 kernel 更难写，也更难调试，寄存器和共享内存压力也更大。一旦资源占用过高，反而会降低 occupancy。这里的 occupancy 可以先理解成“一个 SM 同时能挂多少活跃线程块”。

### 真实工程中的判断标准

如果你在做模型部署，而不是论文复现，判断一个 Mamba kernel 是否写对，通常看三类指标：

| 指标 | 现象 | 解释 |
|---|---|---|
| HBM 带宽占用异常高 | 算子理论线性，但吞吐上不去 | 说明中间结果落盘太多 |
| 显存随序列长度增长过快 | 长序列很快 OOM | 说明状态保存过多 |
| chunk 变大后性能不升反降 | SM 资源打满 | 说明 fusion 过度或寄存器压力过大 |

很多初学者会以为“scan 已经把串行变并行，所以问题解决了”。这只解决了一半。另一半是：**如果并行化后的中间结果仍然大规模写回 HBM，性能仍会被 IO 限死**。Mamba 真正难的地方不是发现 scan，而是把 scan 放进合适的硬件执行路径。

---

## 替代方案与适用边界

Mamba 并不是所有场景都优于替代方案。它的优势很集中，主要在**超长序列、需要内容自适应、并且必须严格控制显存和 IO**的场景。

先看几类主流路线的对比：

| 方案 | 时间复杂度 | 是否支持 token 级参数变化 | 长序列扩展性 | 工程依赖 |
|---|---|---|---|---|
| 标准 Transformer | $O(L^2)$ | 支持，通过注意力内容相关 | 中长序列强，超长序列成本高 | 生态成熟 |
| FlashAttention | 仍然是 $O(L^2)$，但更省 IO | 支持 | 比普通注意力更强，但仍受平方项限制 | 强依赖高质量 kernel |
| LTI SSM + FFT/卷积 | 典型可到近线性 | 不支持或很弱 | 很强 | 依赖系统时不变假设 |
| Selective SSM（Mamba） | $O(L)$ 工作量 | 支持 | 很强，适合超长上下文 | 强依赖 scan + fusion 实现 |

这张表说明了边界：

- 如果序列不长，比如几千 token 以内，Transformer 往往更简单，工具链也更成熟。
- 如果系统本身接近时不变，LTI SSM 的 FFT 路线更直接，工程复杂度也更低。
- 如果任务需要对当前内容做选择性记忆，并且序列非常长，Mamba 才真正体现优势。

### 一个典型适用例子

假设要做百万 token 级对话生成。

- 标准 Transformer：注意力矩阵规模是 $L \times L$，内存和计算都会被平方项拖住。
- FlashAttention：通过重排 IO 显著提升实际效率，但本质上仍没有消除 $O(L^2)$ 的结构限制。
- Mamba：把每个 token 的更新压成局部块，在 scan 中做线性扩展，因此理论上和工程上都更接近 $O(L)$。

但也要看到代价。Mamba 的代价不是数学复杂度，而是**实现复杂度**：

- 你需要可靠的 kernel fusion。
- 你需要处理 chunk 边界与反向重算。
- 你需要接受这条路线比“直接调用现成 attention kernel”更难维护。

所以适用边界可以概括为：

| 场景 | 更合适的方案 |
|---|---|
| 短序列、快速迭代、生态优先 | Transformer |
| 时不变系统、频域处理方便 | LTI SSM + FFT |
| 超长序列、显存紧张、需要内容自适应 | Mamba / Selective SSM |

最终判断标准不是“哪种方法更先进”，而是：**你的问题是否真的需要 token 级选择性记忆，以及你的硬件实现是否能把这种线性复杂度兑现出来**。如果不能兑现到 kernel 层，Selective SSM 只会停留在论文公式里。

---

## 参考资料

- Gu, Albert, Tri Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. 2024.  
- Sascha Kirch, *Here Comes Mamba: The Selective State Space Model* 系列。  
- Emergent Mind, *Mamba-Based Selective State Space Model* 综述。
