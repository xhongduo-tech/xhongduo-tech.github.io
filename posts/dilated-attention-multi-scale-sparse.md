## 核心结论

Dilated Attention 的关键，不是单纯“少算一些注意力”，而是把原本密集的全连接注意力，改写成一组按多尺度组织的稀疏连接图。具体做法是：先把长序列切成固定长度的 segment，再在每个 segment 内按不同的 dilation rate 采样 Q、K、V。若不同头或不同分支采用 $r=1,2,4,\dots$ 这样的指数级步长，模型就能同时保留局部细节和远距离跳跃连接。

它要解决的是标准自注意力在长序列上的平方级成本。标准注意力中，每个 token 都要和几乎全部 token 交互，计算与显存复杂度都接近 $O(N^2)$；Dilated Attention 则把连接限制在若干稀疏模式内，使每个 token 只参与有限次交互，总体计算更接近线性增长。直观地说，它不是让模型“看得更少”，而是让模型“按层次去看”。

它和空洞卷积的关系也很直接。空洞卷积通过在卷积核中留空位来扩大感受野；Dilated Attention 则是在注意力边上留空位，用更稀疏的方式覆盖更大范围。两者的共同点是都在用“跳步采样”换取更大的覆盖范围；不同点是卷积连接固定，而注意力仍然会在采样出来的 token 子集上根据内容分配 softmax 权重。

下面先看一个最小直觉例子。假设序列长度为 16，segment 长度 $w=4$，那么序列会被切成 4 个 segment。一个头用 $r=1$，表示段内全看，负责局部精细对齐；第二个头用 $r=2$，表示隔 1 个 token 采样，负责中程连接；第三个头用 $r=4$，表示每段只抓一个锚点，负责大跨度跳跃。多个头的结果合并后，模型并不是“丢了信息”，而是把昂贵的全连接，换成了结构化的多尺度稀疏连接。

| 方法 | 单层复杂度 | 最长依赖路径 | 局部细节 | 超长依赖 | 典型问题 |
|---|---:|---:|---|---|---|
| 标准 Attention | $O(N^2\cdot d)$ | $O(1)$ | 强 | 强 | 长序列成本过高 |
| 滑动窗口 Attention | $O(N\cdot w\cdot d)$ | $O(N/w)$ | 强 | 弱到中 | 远距传播慢 |
| Dilated Attention | 近似 $O(N\cdot d)$ | 约 $O(\log N)$ | 强 | 强 | 需要设计多尺度参数与实现细节 |

---

## 问题定义与边界

问题本身很清楚：当上下文长度 $N$ 很大时，标准 Transformer 的密集注意力无法继续线性扩展。原因在于注意力分数矩阵的大小是 $N\times N$，无论计算还是显存，都随序列长度平方增长。若 $N=32\,768$，两两交互数已经达到十亿量级；若继续扩到百万级、十亿级 token，标准做法基本无法落地。

标准自注意力通常写作：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中：

- $Q\in\mathbb{R}^{N\times d_k}$：Query，表示“当前位置想找什么”。
- $K\in\mathbb{R}^{N\times d_k}$：Key，表示“当前位置提供什么索引信息”。
- $V\in\mathbb{R}^{N\times d_v}$：Value，表示“当前位置真正携带的内容”。

因为矩阵乘法 $QK^\top$ 会生成一个 $N\times N$ 的注意力分数矩阵，所以其核心复杂度近似为：

$$
\text{Dense Attention} \approx O(N^2\cdot d_k)
$$

如果从单个 token 的视角看，这意味着它要和其余几乎所有 token 做匹配。短序列时这没问题，长序列时则会迅速变成瓶颈。

Dilated Attention 想在两个目标之间找平衡：

1. 把每个 token 的计算成本压到线性级别附近。
2. 不像纯局部窗口那样，把远距离依赖切成很长的传播链。

这里有一个边界必须明确。Dilated Attention 不是为了精确复现密集全连接注意力，而是为了在可扩展前提下保留长程依赖能力。换句话说，它是一种结构化近似，不是逐项等价替代。因此它更适合这些场景：

- 长文档建模
- 仓库级代码理解
- 超长日志分析
- 多轮长对话归纳
- 长轨迹或长上下文检索增强建模

相反，在下面这些场景里，Dilated Attention 通常不是第一选择：

- 上下文长度只有 2K 到 4K
- 显卡和显存足以直接跑全注意力
- 任务主要依赖很短的局部关系
- 你需要的是“所有 token 与所有 token 精确交互”的能力

再看一个更具体的数量级例子。若输入长度为 32K token：

- 标准注意力要处理大约 $32K^2\approx 10^9$ 个位置对。
- 若使用 segment + dilation 的稀疏模式，每个 token 只在少量采样点上参与交互，总工作量会更接近 $N$ 的线性增长。

可以把 segment 内采样理解成下面这个表：

| 段编号 | 原始位置范围 | $r=1$ 采样 | $r=2$ 采样 | $r=4$ 采样 |
|---|---|---|---|---|
| 0 | 0..7 | 0,1,2,3,4,5,6,7 | 0,2,4,6 | 0,4 |
| 1 | 8..15 | 8,9,10,11,12,13,14,15 | 8,10,12,14 | 8,12 |
| 2 | 16..23 | 16,17,18,19,20,21,22,23 | 16,18,20,22 | 16,20 |

表里的“采样”并不等于“这些没采到的位置就彻底没用了”。更准确地说，是不同头、不同模式会构成不同的稀疏子图，多个子图叠加后，模型仍能建立从近到远的信息传播链。

---

## 核心机制与推导

先定义最基本的符号：

- 序列长度为 $N$
- segment 长度为 $w$
- dilation rate 为 $r$
- 第 $i$ 个 segment 覆盖的位置区间为 $[iw,(i+1)w)$

在标准段内注意力里，一个 segment 会取全部 $w$ 个 token 参与计算；而在 Dilated Attention 里，只会按步长 $r$ 选出一部分位置。若第 $i$ 段起点为 $iw$，则它的采样位置可写成：

$$
\mathcal{I}_i^{(r)} = \{iw,\; iw+r,\; iw+2r,\; \dots\}\cap[iw,(i+1)w)
$$

也就是说，先从该段起点开始，每隔 $r$ 个 token 取一个，直到越过 segment 右边界为止。

因此，采样后的 Query、Key、Value 可以写成：

$$
\tilde{Q}_i = [Q_t]_{t\in\mathcal{I}_i^{(r)}}, \qquad
\tilde{K}_i = [K_t]_{t\in\mathcal{I}_i^{(r)}}, \qquad
\tilde{V}_i = [V_t]_{t\in\mathcal{I}_i^{(r)}}
$$

随后，仅在采样后的子序列上做注意力：

$$
\tilde{O}_i
=
\mathrm{softmax}\left(\frac{\tilde{Q}_i\tilde{K}_i^\top}{\sqrt{d_k}}\right)\tilde{V}_i
$$

最后再把结果 scatter 回原位置。这里的 scatter 可以直接理解成“把采样子序列上的输出，写回原序列对应下标”。如果某个位置在这一种 dilation 模式下没有被采样，那么它在这一模式中的输出可以保留为 0，或者由其他模式补齐。

如果只看单个 $(w,r)$ 模式，Dilated Attention 的能力仍然有限。真正起作用的是多模式组合。LongNet 的核心设计之一，就是让不同模式的 $(w_i,r_i)$ 按几何级数增长：

$$
w_i = w_0\cdot a^i,\qquad
r_i = r_0\cdot a^i,\qquad a>1
$$

这表示：

- 模式编号越大，segment 越大
- segment 越大，采样步长也越大
- 感受野随模式指数扩张，而不是线性扩张

这样做的作用可以拆成三层：

| 模式类型 | 典型 $r$ | 负责什么 |
|---|---:|---|
| 小尺度模式 | $r=1$ | 保留词级、短语级、邻近 token 的细粒度关系 |
| 中尺度模式 | $r=2,4$ | 跨小段桥接信息，减少局部窗口的传播层数 |
| 大尺度模式 | $r=8,16,\dots$ | 建立大跨度锚点连接，加快远距信息传播 |

如果只使用单一 dilation，会出现两个问题。

第一，局部与远距无法兼顾。  
只用小 $r$，远处传播仍然慢；只用大 $r$，近处分辨率会显著下降。

第二，中间尺度容易断层。  
例如一个头只看相邻位置，另一个头只看很远锚点，那么某些中等距离的依赖可能反而最难覆盖。

因此，多尺度不是“锦上添花”，而是 Dilated Attention 真正成立的前提之一。

很多实现还会给不同模式加一个动态混合权重。可写成：

$$
O = \sum_i \alpha_i O^{(w_i,r_i)}, \qquad
\alpha_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}
$$

这里：

- $O^{(w_i,r_i)}$：第 $i$ 个尺度模式的输出
- $s_i$：该模式的打分，可以是可学习参数，也可以由输入状态生成
- $\alpha_i$：softmax 归一化后的权重

这一步的意义是：不同位置、不同任务需要的依赖尺度不一样。  
例如代码补全更依赖局部精确关系，而长文档摘要更依赖跨段主题线索。如果把所有尺度简单平均，模型会被迫“每种尺度都一样重要”，这通常不合理。

下面给一个具体到数字的玩具例子。设一个 segment 的原始 token 为 `[0,1,2,3]`，选择 $w=4,r=2$，则采样索引为 `[0,2]`。于是：

$$
\tilde{Q}=[Q_0,Q_2],\qquad
\tilde{K}=[K_0,K_2],\qquad
\tilde{V}=[V_0,V_2]
$$

注意力只在两个采样点之间计算：

$$
A = \mathrm{softmax}\left(\frac{\tilde{Q}\tilde{K}^\top}{\sqrt{d_k}}\right),\qquad
\tilde{O}=A\tilde{V}
$$

如果另一个头同时用 $r=1$，它就会在 `[0,1,2,3]` 上做完整的局部交互；如果再加一个更大尺度的模式，它会负责跨更长距离的锚点连接。这样单层内部就同时具备“看细节”和“看跨度”的能力。

为什么很多文章会说它的最远依赖路径约为 $O(\log N)$？  
直觉可以类比“倍增跳表”或“二分式跳跃”：

- 若每一层或每种模式只能走固定窗口，信息只能一步一步传，路径长度接近线性
- 若连接跨度按几何级数扩张，信息就可以先走短跳，再走中跳，再走长跳
- 到达长度为 $N$ 的远端位置所需跳数，近似只和“需要扩张几次尺度”有关，因此更接近 $\log N$

这不是说单层内所有点都能直接一跳到任意远位置，而是说在多层、多尺度叠加后，跨超长距离的信息传播链会显著缩短。

一个真实工程场景是仓库级代码问答。模型既要读当前函数体的局部变量、控制流和参数传递，也要追到几千行外的接口定义、配置文件、测试样例和调用方。纯窗口注意力擅长局部语法，但跨文件关联路径太长；Dilated Attention 则能让一部分头保留局部语法对齐，另一部分头跳到远处锚点，从而更快建立“当前函数 -> 抽象基类 -> 配置项 -> 测试样例”的连接链。

---

## 代码实现

下面给一个可以直接运行的 Python 玩具实现。它不追求高性能，而是尽量把机制写清楚：segment 切分、按 dilation 采样、在采样子序列上做注意力、再 scatter 回原位置，最后用多尺度混合得到输出。

这段代码只依赖 `numpy`，可直接保存为 `dilated_attention_demo.py` 运行。

```python
import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(q, k, v):
    """
    q, k, v: [m, d]
    return:
        out: [m, d]
        attn: [m, m]
    """
    d = q.shape[-1]
    scores = q @ k.T / np.sqrt(d)
    attn = softmax(scores, axis=-1)
    out = attn @ v
    return out, attn


def dilated_attention_segment(x, w, r, offset=0):
    """
    单一 dilation 模式的段内注意力。

    参数:
        x: [N, d]
        w: segment length
        r: dilation rate
        offset: 采样偏移，满足 0 <= offset < r

    返回:
        out: [N, d]，只在被该模式采样到的位置写入结果
        mask: [N] bool，表示哪些位置被该模式覆盖
    """
    if r <= 0:
        raise ValueError("r must be positive")
    if w <= 0:
        raise ValueError("w must be positive")

    N, d = x.shape
    if N % w != 0:
        raise ValueError(f"N={N} must be divisible by w={w}")
    if offset < 0 or offset >= r:
        raise ValueError(f"offset must satisfy 0 <= offset < r, got offset={offset}, r={r}")

    out = np.zeros_like(x)
    mask = np.zeros(N, dtype=bool)

    for seg_start in range(0, N, w):
        seg_end = seg_start + w
        seg = x[seg_start:seg_end]  # [w, d]

        local_idx = np.arange(offset, w, r)
        if len(local_idx) == 0:
            continue

        sampled = seg[local_idx]  # [m, d]
        attended, _ = scaled_dot_product_attention(sampled, sampled, sampled)

        global_idx = seg_start + local_idx
        out[global_idx] = attended
        mask[global_idx] = True

    return out, mask


def multi_scale_dilated_attention(x, w, rates, offsets=None, mode="mean"):
    """
    多尺度 Dilated Attention。

    参数:
        x: [N, d]
        w: segment length
        rates: 例如 [1, 2, 4]
        offsets: 每个 rate 对应一个 offset；默认全为 0
        mode:
            - "mean": 对覆盖到同一位置的模式取平均
            - "sum": 直接求和

    返回:
        y: [N, d]
        coverage: [N]，每个位置被多少个模式覆盖
    """
    if offsets is None:
        offsets = [0] * len(rates)
    if len(offsets) != len(rates):
        raise ValueError("offsets and rates must have the same length")

    N, d = x.shape
    acc = np.zeros((N, d), dtype=x.dtype)
    coverage = np.zeros(N, dtype=np.int32)

    for r, offset in zip(rates, offsets):
        out_r, mask_r = dilated_attention_segment(x, w=w, r=r, offset=offset)
        acc += out_r
        coverage += mask_r.astype(np.int32)

    if mode == "sum":
        return acc, coverage

    if mode == "mean":
        y = np.zeros_like(acc)
        covered = coverage > 0
        y[covered] = acc[covered] / coverage[covered, None]
        return y, coverage

    raise ValueError(f"Unsupported mode: {mode}")


def dense_segment_attention(x, w):
    """
    作为对照：每个 segment 内做完整注意力。
    """
    N, d = x.shape
    if N % w != 0:
        raise ValueError(f"N={N} must be divisible by w={w}")

    out = np.zeros_like(x)
    for seg_start in range(0, N, w):
        seg = x[seg_start:seg_start + w]
        attended, _ = scaled_dot_product_attention(seg, seg, seg)
        out[seg_start:seg_start + w] = attended
    return out


def main():
    np.random.seed(0)

    N = 16
    d = 8
    w = 4

    x = np.random.randn(N, d)

    # 单一模式
    y_r1, mask_r1 = dilated_attention_segment(x, w=w, r=1, offset=0)
    y_r2, mask_r2 = dilated_attention_segment(x, w=w, r=2, offset=0)
    y_r2_shift, mask_r2_shift = dilated_attention_segment(x, w=w, r=2, offset=1)

    # 多尺度模式
    y_mix, coverage = multi_scale_dilated_attention(
        x,
        w=w,
        rates=[1, 2, 2, 4],
        offsets=[0, 0, 1, 0],
        mode="mean",
    )

    # 对照：完整段内注意力
    y_dense = dense_segment_attention(x, w=w)

    # 基本形状检查
    assert y_r1.shape == (N, d)
    assert y_r2.shape == (N, d)
    assert y_mix.shape == (N, d)
    assert y_dense.shape == (N, d)

    # r=1 时，段内所有位置都会被覆盖
    assert mask_r1.all()

    # r=2, offset=0 时，每段覆盖局部索引 0 和 2
    for seg_start in range(0, N, w):
        assert mask_r2[seg_start + 0]
        assert not mask_r2[seg_start + 1]
        assert mask_r2[seg_start + 2]
        assert not mask_r2[seg_start + 3]

    # r=2, offset=1 时，每段覆盖局部索引 1 和 3
    for seg_start in range(0, N, w):
        assert not mask_r2_shift[seg_start + 0]
        assert mask_r2_shift[seg_start + 1]
        assert not mask_r2_shift[seg_start + 2]
        assert mask_r2_shift[seg_start + 3]

    # 混合后，每个位置至少会被某个模式覆盖
    assert np.all(coverage > 0)

    # 输出不应全为 0
    assert np.abs(y_mix).sum() > 0

    print("Input shape:", x.shape)
    print("Dense output shape:", y_dense.shape)
    print("Dilated mixed output shape:", y_mix.shape)
    print("Coverage per token:", coverage.tolist())
    print("All checks passed.")


if __name__ == "__main__":
    main()
```

这段代码比“最简演示版”多做了两件重要的事。

第一，它显式加入了 `offset`。  
如果所有头都从同一个位置开始采样，例如总是取 `0, 2, 4, ...`，那么某些位置会长期落不到采样点上。加入偏移后，可以让不同头分别取：

- `0,2,4,...`
- `1,3,5,...`

这样覆盖会完整得多。

第二，它把“单一模式输出”和“多模式融合输出”分开写清楚了。  
这更接近真实实现，因为实际系统里不会只用一个 dilation。

如果你运行这段脚本，应该能看到类似输出：

```text
Input shape: (16, 8)
Dense output shape: (16, 8)
Dilated mixed output shape: (16, 8)
Coverage per token: [2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2]
All checks passed.
```

这里的 `coverage` 表示每个 token 被多少种稀疏模式覆盖。它不是注意力权重，而是一个“是否被采样到”的计数。这个概念很重要，因为实际工程中一个常见问题就是某些位置覆盖不足，导致局部信息缺失。

如果换成 PyTorch，真实实现的基本思路一般是：

1. 把输入从 `[B, N, H, D]` 切成 segment 视图
2. 按 head 绑定不同的 dilation rate 和 offset
3. 用索引张量从段内取出采样后的 Q/K/V
4. 在采样后的小张量上调用高效注意力 kernel
5. 再把结果 scatter 回原张量
6. 最后对多尺度输出做求和、平均或门控融合

示意伪代码如下：

```python
# q, k, v: [B, N, H, D]
q_seg = q.view(B, num_segments, w, H, D)
k_seg = k.view(B, num_segments, w, H, D)
v_seg = v.view(B, num_segments, w, H, D)

out = torch.zeros_like(q)

for h in range(H):
    r = dilation_rates[h]
    offset = offsets[h]
    idx = torch.arange(offset, w, r, device=q.device)

    qh = q_seg[:, :, idx, h, :]   # [B, S, M, D]
    kh = k_seg[:, :, idx, h, :]
    vh = v_seg[:, :, idx, h, :]

    oh = flash_attn(qh, kh, vh)   # 在采样子序列上计算

    out_seg = out.view(B, num_segments, w, H, D)
    out_seg[:, :, idx, h, :] = oh
```

需要注意，这段只是“说明结构”的伪代码，并不能直接代表高性能实现。工程上真正困难的地方不是数学公式本身，而是下面几件事：

- `gather/scatter` 是否会打碎内存连续性
- 每个 head 的采样布局是否规则
- 稀疏模式是否方便 kernel 融合
- 是否能和 FlashAttention、块稀疏 kernel 或张量并行策略配合

LongNet 这类实现的实际价值，不在于“理论上可以稀疏”，而在于它把稀疏结构设计成了相对规则、可扩展、适合超长上下文训练的形式。

---

## 工程权衡与常见坑

Dilated Attention 的难点，不在“概念上能不能想到隔点采样”，而在“这种采样图能否既保表达力，又能真正跑得快”。下面按工程上最常见的坑展开。

### 1. 只设一个 dilation，尺度会断层

很多新手第一次实现时，会觉得“既然 dilation 有用，那我就固定 $r=4$ 或 $r=8$”。这通常效果不好。原因很简单：单一 dilation 只覆盖一种尺度。

- 只用小 $r$，远距离传播仍然慢
- 只用大 $r$，局部精度会明显下降
- 中间距离的信息桥接也可能缺失

所以合理方案通常是多尺度组合，例如：

$$
r\in\{1,2,4,8\}
$$

配合对应的 segment 设计，形成从局部到全局的层次覆盖。

### 2. segment 太小，远距建模上不去

segment 长度 $w$ 太小时，每个段内可供采样的位置本来就少。即便 dilation 很大，也无法真正形成大跨度连接，因为跳跃仍然被限制在很短的段内。

例如：

- 若 $w=4$，即使 $r=4$，每段也只采到 1 个锚点
- 这适合玩具示例，不适合真实长序列建模

所以真实系统通常会设一个足够大的基础窗口，再按层或按模式几何扩张。

### 3. segment 太大，局部分辨率和吞吐都会受影响

segment 不是越大越好。太大时会出现两个问题：

- 对局部建模来说，最小模式的相对分辨率下降
- 对硬件来说，采样后的访存模式可能更差，kernel 更难高效执行

因此 segment 长度本质上是“表达能力”和“执行效率”的折中参数。

### 4. 忽略 offset，会造成覆盖盲区

这是最容易被忽略、但又很关键的问题。  
如果所有头都用相同的采样起点，例如都取：

$$
0,\; r,\; 2r,\; 3r,\dots
$$

那么很多位置会长期落不到采样点上。更合理的方式是让不同头使用不同偏移：

$$
s_j,\; s_j+r,\; s_j+2r,\dots
$$

其中 $s_j$ 是第 $j$ 个头的 offset。  
例如对于 $r=2$：

- 头 A 取 `0,2,4,...`
- 头 B 取 `1,3,5,...`

两者叠加后才能覆盖完整位置集。

### 5. 只看理论复杂度，不看 kernel 友好性

论文里常写近似线性复杂度，但实际速度能不能提升，取决于实现是否真的硬件友好。若你的实现存在以下问题，理论复杂度再漂亮也可能跑不快：

- 大量不规则 `gather/scatter`
- 小张量过多，kernel 启动开销偏大
- head 之间采样模式差异过大，难以批处理
- 内存布局不连续，缓存命中率差

因此，Dilated Attention 的工程成败，往往不由公式决定，而由张量布局和 kernel 设计决定。

### 6. 多尺度直接平均，容易互相冲淡

若多个尺度一律平均：

$$
O = \frac{1}{M}\sum_{i=1}^M O^{(i)}
$$

那么模型等于被强迫认为每个尺度同等重要。这对真实任务通常过于粗糙。更合理的做法是加门控权重：

$$
O=\sum_i \alpha_i O^{(i)}
$$

这样模型可以按位置、按层、按任务动态选择更重要的尺度。

### 7. 训练时稳定，推理时未必省

训练中的全序列并行，和推理中的增量解码，不是同一个问题。  
某些 Dilated Attention 结构在训练中能有效降低长序列成本，但在自回归推理场景下，KV cache 的组织、增量更新方式、采样索引维护方式可能会带来额外复杂度。因此它不是“训练省了，推理自然也省”。

下面把常见问题压缩成一张表：

| 问题 | 现象 | 原因 | 处理方式 |
|---|---|---|---|
| 只用单一 dilation | 局部或长距能力明显偏弱 | 只覆盖一种尺度 | 使用 $r=1,2,4,\dots$ 的多模式组合 |
| segment 太小 | 感受野扩不起来 | 每段可用 token 太少 | 增大基础 $w$，再逐层或逐模式扩张 |
| segment 太大 | 局部变粗、吞吐下降 | 分辨率下降，访存不友好 | 回退到更小基础窗口 |
| 没有 offset | 某些位置长期覆盖不足 | 所有头采样点重合 | 为不同头设置不同偏移 |
| 没有门控混合 | 不同尺度互相冲淡 | 简单平均过于粗糙 | 引入 softmax 权重 $\alpha_i$ |
| 理论线性，实际不快 | GPU 利用率低 | gather/scatter 与小 kernel 过碎 | 做块化、张量化和 kernel 融合 |
| 训练有效，推理麻烦 | 增量解码实现复杂 | KV cache 布局不友好 | 单独设计推理态缓存策略 |

如果要用一句话概括工程权衡，那就是：Dilated Attention 把“完全密集的表达力”换成“多尺度稀疏图的可扩展性”，前提是这个稀疏图必须覆盖均衡、尺度合理，而且实现足够硬件友好。

---

## 替代方案与适用边界

Dilated Attention 只是长序列建模方案中的一类，不是唯一答案。它的优势在于同时保留局部细节和远距跳跃连接，但代价是参数设计和实现复杂度更高。因此做方案选择时，最好放在更大的方法谱系中看。

下面先做横向对比：

| 方法 | 核心思想 | 复杂度特征 | 优点 | 局限 | 适合场景 |
|---|---|---|---|---|---|
| 标准 Attention | 全连接两两交互 | $O(N^2)$ | 表达力最直接，定义最标准 | 长序列成本极高 | 短到中等上下文 |
| Sliding Window | 只看邻近窗口 | $O(N\cdot w)$ | 简单稳定，局部建模强 | 远距传播路径长 | 语言建模、局部依赖强任务 |
| Global Token / Longformer 类 | 局部窗口 + 少量全局点 | 近线性 | 关键位置可做全局汇聚 | 依赖全局点设计 | 文档分类、结构化长文本 |
| Linformer / Performer 类 | 用低秩或核近似压缩注意力 | 近线性 | 理论上更省，近似形式明确 | 近似误差与实现复杂度明显 | 对近似误差容忍较高的任务 |
| Dilated Attention | 多尺度空洞稀疏连接 | 近似 $O(N\cdot d)$ | 同时保留局部与远距能力 | 参数设计和工程门槛高 | 超长上下文、跨尺度依赖任务 |

从“依赖传播机制”角度看，这几类方法的差异更容易理解：

- 标准 Attention：任何两点单层直连，表达力最强，但成本最高
- 滑动窗口：只保留近邻边，远距离靠多层慢慢传
- 全局 token 类：在局部边之外，再放少量“中转站”
- 低秩/核近似类：不改“全连接”这个形式，而是近似其计算过程
- Dilated Attention：直接把连接图改成多尺度稀疏结构

Dilated Attention 最适合的，是“超长上下文且确实存在跨尺度依赖”的任务。例如：

- 长文档摘要：既要看句内与段内细节，也要跨章节追主线
- 仓库级代码理解：既要看当前函数，也要追远处接口、配置与测试
- 长日志分析：既要识别局部异常，也要拼接远距因果链
- 长轨迹建模：既要看短时状态变化，也要看长期策略演化

但在下面这些情况下，它未必值得上：

- 上下文只有几千 token，标准 Attention 已经足够
- 任务主要依赖局部关系，窗口注意力更简单直接
- 推理栈或训练栈不支持高效稀疏 kernel
- 任务天然存在少数关键锚点，用全局 token 方法更直接
- 你更关心工程稳定性，而不是把上下文极限拉到十万甚至更高

可以用下面这个选型逻辑快速判断：

1. 如果上下文不长，优先标准 Attention。因为它定义最直接、实现最成熟、行为也最稳定。
2. 如果依赖主要是局部的，优先滑动窗口。它的收益通常最确定。
3. 如果文本中存在明显的“标题、摘要、特殊标记、CLS 位”这类锚点，可以优先考虑全局 token 类方法。
4. 如果目标是十万、百万甚至更长上下文，而且任务确实需要从局部到远距的跨尺度依赖，Dilated Attention 才会体现出明显优势。

简单说，Dilated Attention 不是“默认替代标准 Attention”的通用答案，而是超长上下文场景下，一种非常有针对性的结构化稀疏解法。

---

## 参考资料

- Jiayu Ding, Shuming Ma, Li Dong, et al. *LongNet: Scaling Transformers to 1,000,000,000 Tokens*. arXiv:2307.02486. 这是 Dilated Attention 的核心一手资料，给出定义、复杂度讨论、多尺度设计与超长序列实验。
- Microsoft Research, *LongNet: Scaling Transformers to 1,000,000,000 Tokens*. 官方研究介绍页，适合先把握方案目标、实验定位和工程动机，再回头读论文。
- TorchScale 文档与实现。LongNet 相关代码实现可帮助理解 segment、dilation、并行训练和实际工程接口应如何组织。
- Tri Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. 这不是 Dilated Attention 论文，但对理解“为什么理论复杂度之外还要关心 kernel 与 IO 成本”非常重要。
- 结构化稀疏注意力相关工作，如 Sparse Transformer、Longformer、BigBird。它们不是同一种方法，但适合作为横向参照，帮助理解 Dilated Attention 在长序列建模中的位置。
- 二手概览材料可用于建立直觉，但公式、复杂度和边界判断应以 LongNet 原论文与实现为准。
