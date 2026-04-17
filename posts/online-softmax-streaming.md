## 核心结论

在线 Softmax 是一种把 Softmax 分母改写为“单次遍历即可维护”的精确算法。它在扫描输入 $x_1,x_2,\dots,x_N$ 时，只维护两个状态：

- $m_i$：前 $i$ 个元素中的最大值，即
  $$
  m_i=\max(x_1,\dots,x_i)
  $$
- $l_i$：以前缀最大值 $m_i$ 为参考点的指数和，即
  $$
  l_i=\sum_{k=1}^i e^{x_k-m_i}
  $$

它的核心递推式是：

$$
m_i=\max(m_{i-1},x_i)
$$

$$
l_i=l_{i-1}\cdot e^{m_{i-1}-m_i}+e^{x_i-m_i}
$$

这两个式子解决的不是“写法更巧”这么简单，而是把 Softmax 变成了适合流式输入、分块计算、并行归约和 GPU kernel 融合的形式。传统数值稳定 Softmax 通常需要三次逻辑扫描：

1. 先找全局最大值
2. 再求指数和
3. 最后做归一化

在线 Softmax 把第 1 步和第 2 步合并成一次遍历。这样做的收益主要是减少内存访问，而不是改变数学结果。只要递推和重标定过程正确，它与标准稳定 Softmax 的输出理论上完全一致。

先看一个最小例子，输入为 $[0,1,0.5]$。

- 初始：$m=-\infty,\; l=0$
- 读到 $0$：
  $$
  m=0,\quad l=0\cdot e^{-\infty}+e^{0-0}=1
  $$
- 读到 $1$：
  $$
  m=1,\quad l=1\cdot e^{0-1}+e^{1-1}=e^{-1}+1\approx1.368
  $$
- 读到 $0.5$：
  $$
  m=1,\quad l\approx1.368+e^{-0.5}\approx1.974
  $$

因此最终稳定分母是：

$$
\sum_{k=1}^3 e^{x_k-m}=l\approx1.974,\qquad m=1
$$

于是概率向量为：

$$
\left[
\frac{e^{-1}}{1.974},
\frac{e^0}{1.974},
\frac{e^{-0.5}}{1.974}
\right]
\approx
[0.186,\;0.507,\;0.307]
$$

这也是 FlashAttention 能成立的基础。FlashAttention 不是“把 attention 写快一点”，而是把注意力的 Softmax 归一化改写成块级在线合并。每处理完一个 tile，只保留每一行必须跨块传递的状态，而不回头重读全部历史分数。

---

## 问题定义与边界

Softmax 的定义是：

$$
\operatorname{softmax}(x_j)=\frac{e^{x_j}}{\sum_{k=1}^N e^{x_k}}
$$

直接实现这个公式会遇到两个实际问题。

第一，数值不稳定。  
如果某个 $x_j$ 很大，例如 $1000$，那么 $e^{1000}$ 在常见浮点格式下会直接溢出成 `inf`。

第二，工程上通常需要多次遍历输入。  
稳定实现会先减去最大值：

$$
\operatorname{softmax}(x_j)=\frac{e^{x_j-m}}{\sum_{k=1}^N e^{x_k-m}},\qquad m=\max_k x_k
$$

这样指数项不会爆掉，因为总有 $x_j-m\le 0$。但代价是你必须先知道全局最大值 $m$，而这通常意味着先扫一遍输入。

因此，传统稳定 Softmax 的逻辑流程一般是：

1. 扫描一遍，求 $m=\max_k x_k$
2. 再扫描一遍，求
   $$
   l=\sum_{k=1}^N e^{x_k-m}
   $$
3. 再扫描一遍，输出
   $$
   p_j=\frac{e^{x_j-m}}{l}
   $$

下面把几种写法放在一张表里：

| 方案 | 逻辑扫描次数 | 需要预先知道全局最大值 | 是否适合流式输入 | 数值稳定性 |
|---|---:|---|---|---|
| 直接 Softmax | 2 次左右 | 否 | 一般 | 差 |
| 传统稳定 Softmax | 3 次 | 是 | 差 | 好 |
| 在线 Softmax | 1 次维护分母，输出阶段再看需求 | 否 | 好 | 好 |

这里的“流式输入”要单独解释。它不是指网络流量，而是指数据可能分批到达，或者你故意按块处理，不能一次拿到完整向量。比如在 attention 中，一行 logits 往往是分块算出来的。你拿到第一个块时，并不知道后面块里会不会出现更大的值。

一个更接近工程的例子是 attention 的一行分数。设这行长度为 4096，若用传统稳定 Softmax，就至少需要：

1. 把 4096 个分数读一遍，求最大值
2. 再读一遍，求指数和
3. 再读一遍，写出归一化结果

如果这些分数还不是预先完整存在内存里，而是由 $QK^\top$ 分块生成的，那么问题更明显：第一个块算完以后，你还不能确定它对应的概率，因为后面的块可能刷新全局最大值。

在线 Softmax 正是为这个场景设计的。它的目标不是“少几行代码”，而是让你在只保留少量状态的前提下，持续维护全局正确的稳定分母。

它的边界也要说清楚：

- 它解决的是“如何在线维护归一化分母”。
- 它不自动解决“如何保存全部概率”。
- 它不自动解决“attention 输出向量如何在线合并”。

在 attention 里，除了分母 $l$，还要维护输出部分和。FlashAttention 的难点就在这里：分母和输出都必须在同一参考系下做重标定。

---

## 核心机制与推导

关键思想只有一句话：当最大值变了，旧的指数和必须整体换到新的参考系下。

先写出状态定义。假设处理到第 $i-1$ 个元素时，我们已经有：

$$
m_{i-1}=\max(x_1,\dots,x_{i-1})
$$

$$
l_{i-1}=\sum_{k=1}^{i-1} e^{x_k-m_{i-1}}
$$

现在读入新元素 $x_i$。新的最大值一定是：

$$
m_i=\max(m_{i-1},x_i)
$$

我们需要的新指数和是：

$$
l_i=\sum_{k=1}^{i} e^{x_k-m_i}
$$

把它拆成旧部分和新元素：

$$
l_i=\sum_{k=1}^{i-1} e^{x_k-m_i}+e^{x_i-m_i}
$$

注意前 $i-1$ 项原来是以 $m_{i-1}$ 为参考点保存的，而现在要改成以 $m_i$ 为参考点。对任意 $k<i$，有：

$$
e^{x_k-m_i}=e^{x_k-m_{i-1}}\cdot e^{m_{i-1}-m_i}
$$

因此：

$$
\sum_{k=1}^{i-1} e^{x_k-m_i}
=
e^{m_{i-1}-m_i}\sum_{k=1}^{i-1} e^{x_k-m_{i-1}}
=
e^{m_{i-1}-m_i}l_{i-1}
$$

代回去就得到：

$$
l_i=l_{i-1}\cdot e^{m_{i-1}-m_i}+e^{x_i-m_i}
$$

这就是在线 Softmax 的核心递推式。式子里的

$$
e^{m_{i-1}-m_i}
$$

叫作 rescaling 因子，也就是“重标定因子”。它的作用是把旧状态统一搬到新坐标系中。

### 两种情况的直观解释

这个递推式看起来抽象，其实只对应两种情况。

| 情况 | 发生了什么 | 结果 |
|---|---|---|
| $x_i\le m_{i-1}$ | 新元素没有刷新最大值 | $m_i=m_{i-1}$，旧状态不用缩放 |
| $x_i>m_{i-1}$ | 新元素成了新的最大值 | 旧的 $l_{i-1}$ 必须乘上 $e^{m_{i-1}-m_i}$ |

把这两种情况展开写出来更直观：

若 $x_i\le m_{i-1}$，则 $m_i=m_{i-1}$，所以

$$
l_i=l_{i-1}+e^{x_i-m_{i-1}}
$$

若 $x_i>m_{i-1}$，则 $m_i=x_i$，所以

$$
l_i=l_{i-1}\cdot e^{m_{i-1}-x_i}+1
$$

第二种情况说明：一旦最大值抬高，旧贡献会整体缩小，因为它们离新的最大值更远了。

### 完整算一个例子：$[2,4,1]$

初始状态：

$$
m_0=-\infty,\qquad l_0=0
$$

处理 $x_1=2$：

$$
m_1=\max(-\infty,2)=2
$$

$$
l_1=0\cdot e^{-\infty}+e^{2-2}=1
$$

处理 $x_2=4$：

$$
m_2=\max(2,4)=4
$$

$$
l_2=1\cdot e^{2-4}+e^{4-4}=e^{-2}+1\approx1.135335
$$

处理 $x_3=1$：

$$
m_3=\max(4,1)=4
$$

$$
l_3=l_2\cdot e^{4-4}+e^{1-4}
=1.135335+e^{-3}
\approx1.185122
$$

所以最终有：

$$
m=4,\qquad
l=\sum_{k=1}^3 e^{x_k-4}=e^{-2}+1+e^{-3}\approx1.185122
$$

概率向量为：

$$
\left[
\frac{e^{-2}}{1.185122},
\frac{1}{1.185122},
\frac{e^{-3}}{1.185122}
\right]
\approx
[0.1142,\;0.8438,\;0.0420]
$$

### 为什么它只在线得到了“分母”

这里有一个新手容易误解的点。在线 Softmax 在单次扫描过程中稳定维护的是：

- 当前最大值 $m$
- 当前分母对应的稳定和 $l$

它并没有自动保存每个位置的分子。  
因此，如果你最后想输出完整概率向量，仍然需要能拿到每个位置的 $x_j$，再做：

$$
p_j=\frac{e^{x_j-m}}{l}
$$

这也是为什么在普通向量 Softmax 里，你常常会看到“在线维护分母 + 最后再恢复概率”的实现。而在 FlashAttention 里，工程目标不是输出整张概率矩阵，而是直接得到：

$$
O=PV
$$

所以它不会显式保存整行概率，而是把概率和 $V$ 的乘积一起在线合并。

### 从逐元素推广到分块

如果输入不是逐个到达，而是按块到达，例如一个块记为 `chunk`，只要先求出块内统计量，再和全局状态合并即可。

设旧全局状态为 $(m_{\text{old}}, l_{\text{old}})$。  
当前块的局部状态为 $(m_{\text{chunk}}, l_{\text{chunk}})$，其中：

$$
m_{\text{chunk}}=\max_{x\in \text{chunk}} x
$$

$$
l_{\text{chunk}}=\sum_{x\in \text{chunk}} e^{x-m_{\text{chunk}}}
$$

那么合并后的新状态是：

$$
m_{\text{new}}=\max(m_{\text{old}},m_{\text{chunk}})
$$

$$
l_{\text{new}}=
l_{\text{old}}e^{m_{\text{old}}-m_{\text{new}}}
+
l_{\text{chunk}}e^{m_{\text{chunk}}-m_{\text{new}}}
$$

这个块级公式和逐元素公式本质上是同一件事。它说明在线 Softmax 不是只能顺序处理单个标量，而是支持“先块内归约，再块间合并”的并行结构。这正是 GPU kernel 需要的形式。

---

## 代码实现

下面给出一个完整、可直接运行的 Python 示例。它包含四部分：

1. `update_online_state`：逐元素更新状态
2. `chunk_stats`：计算单个块的局部状态
3. `merge_states`：合并两个状态
4. `online_softmax`：按块流式处理并恢复最终概率

```python
import math
from typing import Iterable, List, Sequence, Tuple

State = Tuple[float, float]  # (m, l)


def update_online_state(state: State, x: float) -> State:
    """用单个元素 x 更新在线 Softmax 状态。"""
    m, l = state
    m_new = max(m, x)
    l_new = l * math.exp(m - m_new) + math.exp(x - m_new)
    return m_new, l_new


def chunk_stats(chunk: Sequence[float]) -> State:
    """计算一个块的局部状态 (chunk_max, chunk_sum)。"""
    if len(chunk) == 0:
        return -math.inf, 0.0

    m = max(chunk)
    l = sum(math.exp(x - m) for x in chunk)
    return m, l


def merge_states(left: State, right: State) -> State:
    """合并两个在线 Softmax 状态。"""
    m_left, l_left = left
    m_right, l_right = right

    m_new = max(m_left, m_right)
    l_new = (
        l_left * math.exp(m_left - m_new)
        + l_right * math.exp(m_right - m_new)
    )
    return m_new, l_new


def stable_softmax(logits: Sequence[float]) -> List[float]:
    """传统数值稳定 Softmax，作为参考实现。"""
    if len(logits) == 0:
        return []

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [v / s for v in exps]


def online_softmax(logits: Sequence[float], chunk_size: int = 1) -> Tuple[List[float], State]:
    """
    用在线 Softmax 按块处理输入。
    返回:
      probs: 概率向量
      state: 最终状态 (m, l)
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if len(logits) == 0:
        return [], (-math.inf, 0.0)

    state = (-math.inf, 0.0)

    for start in range(0, len(logits), chunk_size):
        chunk = logits[start:start + chunk_size]
        state = merge_states(state, chunk_stats(chunk))

    m, l = state
    probs = [math.exp(x - m) / l for x in logits]
    return probs, state


def approx_equal_list(xs: Sequence[float], ys: Sequence[float], eps: float = 1e-12) -> bool:
    return len(xs) == len(ys) and all(abs(a - b) <= eps for a, b in zip(xs, ys))


def main() -> None:
    # 例子 1：逐元素处理 [2, 4, 1]
    logits1 = [2.0, 4.0, 1.0]
    probs_online1, state1 = online_softmax(logits1, chunk_size=1)
    probs_ref1 = stable_softmax(logits1)

    print("example 1 logits:", logits1)
    print("example 1 state :", state1)
    print("example 1 online:", probs_online1)
    print("example 1 ref   :", probs_ref1)

    assert approx_equal_list(probs_online1, probs_ref1)
    assert abs(sum(probs_online1) - 1.0) < 1e-12
    assert abs(state1[0] - 4.0) < 1e-12
    assert abs(state1[1] - (math.exp(-2.0) + 1.0 + math.exp(-3.0))) < 1e-12

    # 例子 2：按块处理长一点的向量
    logits2 = [0.2, -1.0, 3.5, 2.2, 3.7, -0.3, 1.1, 0.0]
    probs_online2, state2 = online_softmax(logits2, chunk_size=3)
    probs_ref2 = stable_softmax(logits2)

    print("\nexample 2 logits:", logits2)
    print("example 2 state :", state2)
    print("example 2 online:", probs_online2)
    print("example 2 ref   :", probs_ref2)

    assert approx_equal_list(probs_online2, probs_ref2)
    assert abs(sum(probs_online2) - 1.0) < 1e-12

    # 例子 3：验证“块大小变化不改变结果”
    logits3 = [1.5, -2.0, 0.3, 4.2, 4.1, -0.7, 3.0]
    ref3 = stable_softmax(logits3)
    for chunk_size in [1, 2, 4, 7]:
        probs3, _ = online_softmax(logits3, chunk_size=chunk_size)
        assert approx_equal_list(probs3, ref3), f"mismatch at chunk_size={chunk_size}"

    print("\nall tests passed")


if __name__ == "__main__":
    main()
```

这段代码可以直接运行。执行后应打印三组测试，并在最后输出：

```text
all tests passed
```

### 代码里每个状态的意义

| 字段 | 含义 | 是否需要跨块保存 |
|---|---|---|
| `m` | 当前全局最大值 | 是 |
| `l` | 以当前 `m` 为参考的全局指数和 | 是 |
| `chunk_max` | 当前块最大值 | 否 |
| `chunk_sum` | 当前块在 `chunk_max` 下的指数和 | 否 |

### 为什么块级合并是对的

把整个向量拆成两个不重叠的块 $A,B$。若块内统计量分别为：

$$
(m_A,l_A),\qquad (m_B,l_B)
$$

那么它们对应的真实指数和分别是：

$$
\sum_{x\in A} e^x = e^{m_A}l_A,\qquad
\sum_{x\in B} e^x = e^{m_B}l_B
$$

整体验证很简单。设

$$
m=\max(m_A,m_B)
$$

则：

$$
\sum_{x\in A\cup B} e^{x-m}
=
\sum_{x\in A} e^{x-m}+\sum_{x\in B} e^{x-m}
$$

继续写：

$$
\sum_{x\in A} e^{x-m}
=
e^{m_A-m}\sum_{x\in A} e^{x-m_A}
=
e^{m_A-m}l_A
$$

同理：

$$
\sum_{x\in B} e^{x-m}=e^{m_B-m}l_B
$$

所以：

$$
l=e^{m_A-m}l_A+e^{m_B-m}l_B
$$

这就是 `merge_states` 的数学来源。

### 与 FlashAttention 的关系

FlashAttention 里处理的不是一维 logits 向量，而是 $QK^\top$ 的一行或一个 tile。对每一行，它维护的不只是一对 $(m,l)$，通常还有输出向量的部分和。块级流程可以概括为：

1. 计算当前 query block 与 key block 的局部分数
2. 求每一行的 `chunk_max`
3. 求每一行在 `chunk_max` 下的指数和 `chunk_sum`
4. 用在线 Softmax 公式更新全局 `m,l`
5. 用同样的 rescaling 规则更新输出部分和

因此，在线 Softmax 并不是 FlashAttention 附带的小技巧，而是它能够在分块条件下保持“精确 attention”等价性的基础。

---

## 工程权衡与常见坑

在线 Softmax 的收益主要来自减少内存访问和支持 kernel 融合，不是因为它把指数计算消掉了。指数运算仍然存在，归一化仍然存在，变化的是执行顺序和状态传递方式。

下面先列最常见的错误：

| 错误做法 | 现象 | 原因 | 正确做法 |
|---|---|---|---|
| 更新了最大值 `m`，但没缩放旧 `l` | 分母错误，概率和不为 1 | 旧状态还停留在旧参考系 | 乘上 $e^{m_{\text{old}}-m_{\text{new}}}$ |
| 每个块内部算对了，块间直接把 `l` 相加 | 长序列结果明显失真 | 各块使用的最大值不同 | 先求新全局最大值，再重标定后相加 |
| 多行共享一个 `m,l` | 行之间互相污染 | Softmax 归一化轴混了 | 每一行单独维护状态 |
| 用低精度累积 `l` | 长序列误差累积 | 和项多、动态范围大 | `m,l` 常用 FP32 |
| 只更新分母，不同步更新输出部分和 | attention 输出偏移 | 分母和输出不在同一参考系 | 输出也按同样规则 rescale |

### 一个典型错误示例

假设你已经处理完第一个块，得到：

- 旧块最大值：$m_{\text{old}}=7$
- 旧块指数和：$l_{\text{old}}=1.8$

第二个块算出来：

- 新块最大值：$m_{\text{chunk}}=10$
- 新块指数和：$l_{\text{chunk}}=1.2$

正确合并方式是：

$$
m_{\text{new}}=10
$$

$$
l_{\text{new}}=1.8\cdot e^{7-10}+1.2\cdot e^{10-10}
=1.8e^{-3}+1.2
\approx1.2896
$$

如果你错写成：

$$
l_{\text{wrong}}=1.8+1.2=3.0
$$

那么旧块的贡献被夸大了约：

$$
e^3\approx20.085
$$

这不是小误差，而是量级错误。Softmax 的分母一旦错了，整行概率都会错。

### 为什么常把输入用 FP16/BF16，状态用 FP32

在真实 GPU kernel 里，常见做法是：

| 对象 | 常见精度 |
|---|---|
| 输入 logits | FP16 或 BF16 |
| 局部 matmul 累加 | FP32 |
| `m`、`l` 状态 | FP32 |
| 最终输出 | 视框架和 kernel 选择而定 |

原因很直接：

- 在线 Softmax 虽然避免了 $e^{1000}$ 这种显式溢出
- 但它没有消除长序列上的累加误差
- `l` 本质上是很多项的和，若用过低精度，误差会积累

状态只有两个标量，保存成本极低，通常不值得为了省一点寄存器或内存而牺牲稳定性。

### 在线 Softmax 省下的到底是什么

对新手来说，最容易混淆“算术量”和“访存量”。

| 指标 | 在线 Softmax 是否显著减少 |
|---|---|
| 指数运算次数 | 否，本质上仍要算 |
| 加法次数 | 否，数量级差别不大 |
| 读取整行 logits 的次数 | 是 |
| 需要完整保留中间矩阵 | 是，很多场景下可以避免 |
| 与上游/下游算子融合的可能性 | 是 |

在 attention 中，真正昂贵的往往不是多做了几个 `exp`，而是你是否把整张分数矩阵写回显存再读回来。在线 Softmax 改善的是这一层。

---

## 替代方案与适用边界

在线 Softmax 不是唯一的稳定 Softmax 实现，但它在“流式处理”和“分块合并”场景里最自然。

| 方案 | 扫描方式 | 是否精确 | 适合场景 | 主要限制 |
|---|---|---|---|---|
| 传统稳定 Softmax | 先求最大值，再求和，再归一化 | 是 | 向量较短、实现简单优先 | 多次读输入 |
| 在线 Softmax | 单遍维护状态，最终恢复输出 | 是 | 流式输入、长序列、kernel 融合 | 需要维护跨块状态 |
| 树形归约 Softmax | 分段并行归约再合并 | 是 | 并行归约结构明显的硬件 | 合并逻辑更复杂 |
| 近似 Softmax | 用近似函数或稀疏技巧替代 | 否或近似 | 极端追求吞吐 | 结果不再严格等价 |

什么时候传统稳定 Softmax 仍然够用：

- 向量很短，多扫几遍成本很低
- 数据已经在 cache 里，访存不是瓶颈
- 你的目标是可读性优先，而不是极限性能

什么时候在线 Softmax 更合适：

- 输入按流式或按块到达，不能先看到全量
- 归一化维度很长，多次读写代价高
- 你要把 matmul、softmax、加权求和尽量融合进一个 kernel
- 你要在 attention 中避免保存完整中间概率矩阵

它的适用边界也必须明确：

第一，它要求归一化轴定义明确。  
通常是一行向量，或者 attention 中某一行分数。若数据布局频繁变化，状态传递会更复杂。

第二，它解决的是“精确归一化的在线维护”，不是近似加速。  
所以它不会改变模型语义。

第三，它通常只把“分母维护”变成在线形式。  
如果你的最终目标是输出完整概率向量，仍然要能重新访问原始 logits 或保存必要中间量。

第四，在 attention 中，它必须和输出部分和一起设计。  
只把分母改在线，而不处理输出累积，不能直接得到完整的 FlashAttention。

一句话总结边界：在线 Softmax 是一个稳定、精确、可组合的归一化原语，但它不是完整 attention 算法本身。

---

## 参考资料

- Maxim Milakov, Natalia Gimelshein, *Online normalizer calculation for softmax*  
  核心贡献：给出在线维护 Softmax normalizer 的递推公式，是在线 Softmax 的直接来源。  
  建议阅读方式：先看状态定义和递推式，再看它为什么能减少访存，不必一开始就纠结全部性能细节。

- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*  
  核心贡献：把在线 Softmax 扩展到块级 attention，证明在不保存完整注意力矩阵的前提下仍可得到精确结果。  
  建议阅读方式：先抓住 IO-aware 和在线 rescaling 两个关键词，再看块调度。

- Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*  
  核心贡献：说明在线 Softmax 不只是数学正确，还要配合更好的并行划分，才能把理论收益兑现成真实吞吐。  
  建议阅读方式：把它当作“工程实现升级版”来看，重点看 work partitioning，而不是重复看基础公式。

- Peter Chng, *The basic idea behind FlashAttention*  
  核心贡献：用更直白的语言解释为什么块间不能直接拼接，为什么旧块必须 rescale。  
  建议阅读方式：如果你第一次接触 FlashAttention，这篇适合放在论文前面读。

- Damek Davis, *The basic idea behind Flash Attention*  
  核心贡献：对块级合并的推导比较清楚，尤其适合理解“分母”和“输出部分和”为什么都要重标定。  
  建议阅读方式：结合论文里的符号一起看，效果最好。

- Hugging Face 与 PyTorch 社区关于 `scaled_dot_product_attention`、FlashAttention 后端与 kernel 融合的工程文章  
  核心贡献：把在线 Softmax 放到框架实现和部署环境中看，能帮助理解真实代码里状态是如何按行维护的。  
  建议阅读方式：在已经理解数学递推后再看，否则容易只记 API，不理解为什么这样写。

实现时最值得反复核对的只有两件事：

1. 分母 `l` 是否在最大值变化时正确乘上了重标定因子
2. attention 输出部分和是否和分母使用了同一参考点做更新

这两点一旦有一处漏掉，结果通常不是“略有误差”，而是整体失真。
