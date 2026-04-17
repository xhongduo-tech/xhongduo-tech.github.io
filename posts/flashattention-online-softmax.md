## 核心结论

FlashAttention 的在线 Softmax，本质上是在“还没看完整行分数”时，提前维护 Softmax 需要的两个关键统计量：

- $m$：当前见过的最大值。白话说，就是“到目前为止最大的那个分数”。
- $l$：以当前最大值为基准的指数和。白话说，就是“Softmax 分母的运行中版本”。

标准 Softmax 通常按整行做三件事：先找最大值，再算指数和，最后归一化。在线 Softmax 把“找最大值”和“累计分母”合成到一次流式处理中，同时还能把对 $V$ 的加权和一起累计。结论是：

$$
\mathrm{softmax}(x)_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
$$

其中 $m=\max_j x_j$。在线算法虽然在处理中不断更新 $m$，但最终结果与先拿到全局 $m$ 再统一计算的标准 Softmax 完全等价。

这件事对 FlashAttention 很关键。因为 FlashAttention 的目标不是“更快算出一个 $N\times N$ 注意力矩阵”，而是“不把这个矩阵写回显存”。在线 Softmax 让每个分块在 SRAM 内就能完成局部计算、归一化修正和对 $V$ 的累加，从而把瓶颈从“反复读写大矩阵”改成“少量状态更新”。

---

## 问题定义与边界

先定义问题。设某个 query 对所有 key 的打分为一行向量 $x=[x_1,\dots,x_n]$。目标是计算这一行的 Softmax，或者进一步计算：

$$
\mathrm{Attn}(q,K,V)=\mathrm{softmax}(qK^\top)V
$$

这里：

- 注意力分数，白话说，就是 query 和每个 key 的相关性分数。
- 归一化，白话说，就是把一行分数变成总和为 1 的权重。
- 分块处理，白话说，就是不是一次看完整行，而是一段一段看。

标准稳定版 Softmax 的典型流程如下：

| Pass | 工作 | 代价 |
|---|---|---|
| 1 | 扫描整行，找到全局最大值 $m$ | 读一遍分数 |
| 2 | 计算 $e^{x_i-m}$ 并求和 | 再读一遍分数 |
| 3 | 用分母归一化，得到概率 | 再读或写一遍结果 |

如果只是单个小向量，这没问题。但在注意力里，每一行都很长，而且实际系统往往不只需要概率，还要继续乘上 $V$。这时问题变成两个层面：

1. 数学层面：能不能在没看到整行之前，就开始累积正确的分母和输出？
2. 工程层面：能不能在块级处理时，不物化整张注意力矩阵？

边界也要说清楚。在线 Softmax 解决的是“精确 Softmax 的流式计算”，不是近似注意力，也不是把复杂度从 $O(N^2)$ 改成更低。FlashAttention 的收益主要来自 IO，也就是显存和片上缓存之间的数据搬运，而不是改掉注意力本身的数学复杂度。

---

## 核心机制与推导

### 1. 先看一个玩具例子

向量为：

$$
x=[100,102,99,101]
$$

如果按标准方式，先取全局最大值 $m=102$，再算：

$$
e^{100-102}, e^{102-102}, e^{99-102}, e^{101-102}
= e^{-2},1,e^{-3},e^{-1}
$$

分母为：

$$
l=e^{-2}+1+e^{-3}+e^{-1}\approx 1.553
$$

在线算法按顺序处理时，不知道未来会不会出现更大的数，所以要维护运行状态。

处理第 1 个数 100：

- $m=100$
- $l=1$

处理第 2 个数 102 时，发现新最大值出现了。之前的分母是按 100 为基准累计的，现在必须改成按 102 为基准。修正规则是：

$$
l_{\text{new}}=l_{\text{old}}\cdot e^{m_{\text{old}}-m_{\text{new}}}+e^{x-m_{\text{new}}}
$$

代入得到：

$$
l=1\cdot e^{100-102}+1=e^{-2}+1
$$

这一步最重要。它说明“旧贡献没有失效”，只是要换一个参考系。白话说，之前按 100 对齐的指数值，要整体缩小到按 102 对齐。

继续处理 99 和 101，因为它们都不超过当前最大值 102，所以只需直接加：

$$
l=(e^{-2}+1)+e^{-3}+e^{-1}
$$

最终与标准结果完全一致。

### 2. 从单个元素推广到分块

FlashAttention 不会一个元素一个元素处理，而是按块处理。设第 $j$ 个块的分数向量为 $S^{(j)}$，局部统计量为：

- $m^{(j)}=\max S^{(j)}$
- $l^{(j)}=\sum_{k\in j} e^{S_k-m^{(j)}}$

如果前面块累计状态为 $(m,l)$，则合并新块后的状态为：

$$
m'=\max(m,m^{(j)})
$$

$$
l' = l\cdot e^{m-m'} + l^{(j)}\cdot e^{m^{(j)}-m'}
$$

这个式子就是在线 Softmax 的核心。原因很直接：旧块和新块原本各自都按自己的最大值缩放过，现在必须统一改写到同一个新基准 $m'$ 下。

### 3. 为什么还能顺手累计输出

在注意力里，我们最终不是只要 Softmax 概率，而是要：

$$
O = \sum_k \mathrm{softmax}(x)_k V_k
$$

所以实际维护的不是概率向量，而是“未归一化输出”：

$$
\tilde O=\sum_k e^{x_k-m}V_k
$$

这样最后只要除以分母：

$$
O=\frac{\tilde O}{l}
$$

分块时，输出的更新规则和分母完全同构：

$$
\tilde O'=\tilde O\cdot e^{m-m'}+\tilde O^{(j)}\cdot e^{m^{(j)}-m'}
$$

其中：

$$
\tilde O^{(j)}=\sum_{k\in j} e^{S_k-m^{(j)}}V_k
$$

这就是 FlashAttention 能边读块边累计最终输出的原因。它不需要先存下整行概率，再去乘 $V$；它只要维护 $(m,l,\tilde O)$ 三个状态。

### 4. 真实工程例子

假设长度为 16k 的上下文做推理，某一层某个 head 的一个 query 要和 16k 个 key 做点积。若直接生成一整行分数，再做三遍 Softmax，再乘 $V$，中间会有大量显存往返。

FlashAttention 的做法是：

1. 把一小块 $K,V$ 读入 SRAM。
2. 计算当前 query 对这块 $K$ 的分数。
3. 在块内求局部最大值和局部指数和。
4. 用在线公式更新全局 $(m,l,\tilde O)$。
5. 继续读下一块。

这样整行注意力从头到尾只表现为一组很小的运行状态，而不是一大片中间矩阵。这就是它在长序列下明显更快的根本原因。

---

## 代码实现

下面给出一个可运行的 Python 版本。它展示两件事：

- 在线 Softmax 与标准稳定版 Softmax 完全一致。
- 在线 attention 输出与标准 attention 输出一致。

```python
import math

def softmax_stable(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [v / s for v in exps]

def softmax_online(xs):
    m = float("-inf")
    l = 0.0

    for x in xs:
        m_new = max(m, x)
        l = l * math.exp(m - m_new) + math.exp(x - m_new)
        m = m_new

    return [math.exp(x - m) / l for x in xs]

def attention_stable(scores, values):
    probs = softmax_stable(scores)
    dim = len(values[0])
    out = [0.0] * dim
    for p, v in zip(probs, values):
        for i in range(dim):
            out[i] += p * v[i]
    return out

def attention_online_blocked(scores, values, block_size):
    dim = len(values[0])
    m = float("-inf")
    l = 0.0
    O = [0.0] * dim

    n = len(scores)
    for start in range(0, n, block_size):
        s_block = scores[start:start + block_size]
        v_block = values[start:start + block_size]

        m_block = max(s_block)
        m_new = max(m, m_block)

        old_scale = 0.0 if m == float("-inf") else math.exp(m - m_new)
        block_scale = math.exp(m_block - m_new)

        # 先缩放旧状态
        l *= old_scale
        O = [x * old_scale for x in O]

        # 再加入新块贡献
        p_block = [math.exp(s - m_block) for s in s_block]
        l += sum(p_block) * block_scale

        for p, v in zip(p_block, v_block):
            w = p * block_scale
            for i in range(dim):
                O[i] += w * v[i]

        m = m_new

    return [x / l for x in O]

# 玩具例子
xs = [100.0, 102.0, 99.0, 101.0]
a = softmax_stable(xs)
b = softmax_online(xs)
assert all(abs(x - y) < 1e-12 for x, y in zip(a, b))

# 真实工程风格的小例子：一行 attention
scores = [12.0, 9.0, 15.0, 7.0, 14.0]
values = [
    [1.0, 0.0],
    [0.0, 2.0],
    [3.0, 1.0],
    [1.0, 1.0],
    [2.0, 4.0],
]
o1 = attention_stable(scores, values)
o2 = attention_online_blocked(scores, values, block_size=2)
assert all(abs(x - y) < 1e-12 for x, y in zip(o1, o2))

print("all asserts passed")
```

这段代码里最容易忽略的一点是：块内可以先按局部最大值 `m_block` 计算，再统一乘上 `exp(m_block - m_new)` 合并到全局状态。这么写和直接按 `m_new` 算是等价的，但更符合 GPU 块内先局部规约、再全局合并的实现方式。

---

## 工程权衡与常见坑

在线 Softmax 不是“把三遍改一遍”这么简单，它带来了明确的状态管理要求。

| 问题 | 现象 | 原因 | 处理方式 |
|---|---|---|---|
| 低精度误差 | 输出接近 one-hot，尾部被吃掉 | 多次 rescale 累积舍入误差 | `m/l/O` 用 FP32 累计 |
| 块太小 | 理论省 IO，实际没更快 | 块管理、同步、exp 开销上升 | 根据 SRAM 容量选块大小 |
| mask 处理错误 | 因果注意力结果错位 | 被 mask 的位置没正确当成 $-\infty$ | 在块内先加 mask 再做 max |
| 状态初始化错误 | 第一块出现 NaN | $m=-\infty$ 时直接参与 exp | 首块单独处理或特判 |
| 调试困难 | 和普通实现难对齐 | 中间没有完整概率矩阵 | 先做单行、单头、单块验证 |

常见坑里，数值精度最值得强调。理论上，更新式保证精确等价；工程上，如果 `m`、`l`、`O` 用 BF16 直接累计，很多次缩放之后会出现明显偏差。原因不是公式错，而是：

$$
l' = l\cdot e^{m-m'} + \Delta
$$

当 $m'-m$ 很大时，前一项可能非常小，在低精度下直接下溢或被量化成 0。于是“前面块的贡献”被过早丢失。FlashAttention 的常见做法是：输入和块内乘法可以低精度，但关键累计状态尽量保留 FP32。

另一个坑是因果 mask。白话说，就是当前位置不能看未来 token。实现上必须在块内先把非法位置变成 $-\infty$，再参与局部最大值和指数和。如果顺序反了，`m_block` 就可能被未来位置污染，后续所有缩放都错。

---

## 替代方案与适用边界

在线 Softmax 并不是所有场景都必须上。

| 方案 | 额外状态 | HBM 访问压力 | 适用场景 |
|---|---|---|---|
| 标准三遍 Softmax | 几乎没有 | 较高 | 小序列、教学、调试 |
| 在线 Softmax | $m,l$ | 中等 | 流式归一化、块级融合 |
| FlashAttention 融合实现 | $m,l,\tilde O$ | 最低 | 长序列训练与推理 |

适用边界可以直接概括为三条：

1. 序列很短时，标准 Softmax 更简单。比如长度只有几百，三遍扫描的 IO 压力并不显著，而在线版本代码复杂度更高。
2. 只想验证数学正确性时，先写标准版。它更容易打印中间量，也更容易和框架原生实现对比。
3. 真正需要长序列吞吐、又能做 kernel fusion 时，在线 Softmax 才体现最大价值。FlashAttention 的收益来自“分块 + 融合 + 不物化中间矩阵”，不是单独一个在线公式就能完全带来。

还要强调一个边界：在线 Softmax 不是近似算法。它和 Linformer、Performer 这一类“改变注意力形式”的方法不同。只要实现正确，它给出的结果就是标准 Softmax attention 的精确值。

---

## 参考资料

- Milakov, M., & Gimelshein, N. “Online normalizer calculation for softmax.”
- Dao, T. et al. “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.”
- Hugging Face Blog: Flash Attention 与在线 Softmax 讲解
- Dao-AILab 官方 FlashAttention 仓库
- ICLR Blogposts: FlashAttention 演化与 IO 视角解析
