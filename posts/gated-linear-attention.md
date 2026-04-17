## 核心结论

Gated Linear Attention，简称 GLA，可以理解为“带门控的线性注意力”。门控就是一个可学习的开关，决定信息该保留多少、输出多少。它解决的问题很直接：普通 linear attention 速度快、状态固定，但“看什么都差不多”，缺少 softmax attention 那种按内容选择重点的能力；GLA 在这个固定状态上再加两道门，让模型能有选择地记忆和读出。

它的核心不是把注意力重新做成二次复杂度，而是在保持递推状态大小固定的前提下，引入两类内容感知的 gate：

$$
S_t = G_t \odot S_{t-1} + k_t v_t^\top
$$

$$
o_t = \sigma_{\text{out}}(W_{\text{out}} S_t + b_{\text{out}}) \odot (W_{\text{read}} S_t)
$$

这里的 $S_t$ 是状态矩阵，也就是“压缩后的上下文记忆”；$G_t$ 是 forget gate，也就是“遗忘门”，控制旧记忆留下多少；$\sigma_{\text{out}}$ 是 output gate，也就是“输出门”，控制当前读出多少。结果是：GLA 仍然保留线性注意力的高效递推，但比普通 linear attention 更像一个会筛选信息的序列模型，在长文本和高分辨率视觉任务里尤其有价值。

一句直观的话概括：GLA 就像一个带 LSTM 风格调节器的 linear attention。forget gate 决定旧信息删掉多少，output gate 决定当前说出来多少，因此它能在很长的上下文里只保留重要 token，而不是把所有历史都一视同仁地累加。

---

## 问题定义与边界

先定义问题。attention 的本质是“当前 token 去读取历史 token 的信息”。softmax attention 的优点是选择性强，谁相关就给谁更高权重；缺点是代价高，序列长度为 $n$ 时，完整两两交互通常带来 $O(n^2)$ 级别的注意力矩阵开销。上下文从 2k、8k 拉到 32k 时，显存和吞吐都会迅速恶化。

linear attention 的思路是把“逐对比较”改写成“先累积状态，再从状态中读出”。这样每来一个 token，只需更新一个固定大小的状态矩阵，不再保存完整注意力图。它解决了扩展性问题，但新的问题也很明显：普通线性累积缺少精细选择性，旧信息很容易被机械叠加，模型不容易学会“哪些该忘、哪些该保留”。

这就是 GLA 的边界条件：它不是为了在所有场景替代 softmax attention，而是为了在“上下文极长、内存受限、需要递推推理”的情况下，补上线性注意力最缺的那块能力，也就是内容感知的选择性记忆。

下面是一个工程上足够实用的对比：

| 机制 | 时间/状态特征 | 选择性 | 长序列伸缩性 | 硬件适配 |
|---|---|---:|---:|---:|
| Softmax Attention | 需要显式处理 token 两两关系 | 强 | 弱 | 很成熟，FlashAttention 优化充分 |
| Linear Attention | 固定状态递推，读写成本稳定 | 弱 | 强 | 好，适合流式和长序列 |
| GLA | 固定状态递推 + 数据感知 gate | 中到强 | 强 | 好，但实现比普通 linear 更复杂 |
| Mamba/类似 gated RNN | 递推状态更新，门控更强 | 强 | 强 | 很强，但与 attention 家族接口不同 |

玩具例子先看一个。假设模型在读一篇很长的错误日志。普通 linear attention 可能把“时间戳、路径、重复警告”不断混进状态里；GLA 会更倾向于让 forget gate 压低这些重复背景信息，把真正异常的 token，比如 `OOM`、`permission denied`、`segfault`，保留得更久。这个能力不是来自完整 softmax，而是来自 gate 对状态更新的调节。

真实工程例子是 diffusion 模型。DiG 把 GLA 用到 diffusion transformer 中，目标不是“理论更优美”，而是解决高分辨率图像下 token 数暴涨导致的效率问题。分辨率上去以后，softmax attention 的代价会很重，而 GLA 可以在保持较低内存占用的同时提供比普通线性注意力更强的内容选择性，因此在大分辨率生成时更有工程吸引力。

---

## 核心机制与推导

先从普通 linear attention 的递推形式出发。它通常可以写成：

$$
S_t = S_{t-1} + k_t v_t^\top
$$

这里 $k_t v_t^\top$ 是外积，也就是把当前 key 和 value 组合成一个矩阵增量。直观上，这一步是在往“记忆表”里写入当前 token 的信息。问题在于，每一步都只会加，不会有针对性地删，久而久之状态里会塞进很多不重要的信息。

GLA 的第一步改动就是把“只加不忘”改成“先忘后加”：

$$
S_t = G_t \odot S_{t-1} + k_t v_t^\top
$$

这里的 $\odot$ 是逐元素乘法。也就是说，$G_t$ 的每个位置都在决定旧状态对应位置保留多少。如果某个维度对应的信息已经不重要，$G_t$ 就可以把它压低；如果仍然重要，就允许它继续存在。这就是 selective memory，中文可以叫“选择性记忆”，也就是模型不是把所有历史一锅端，而是按内容筛选。

第二步是读出。即便状态已经更新完，模型也不必把全部状态直接暴露给后续层，而是再过一道输出门：

$$
o_t = \sigma_{\text{out}}(W_{\text{out}} S_t + b_{\text{out}}) \odot (W_{\text{read}} S_t)
$$

这一步的含义是：状态里“有”什么是一回事，当前时刻“要不要说出来”是另一回事。forget gate 解决“记不记”，output gate 解决“说不说”。这两个动作分开之后，模型的表达力会明显高于普通 linear attention。

看一个数值玩具例子。设

$$
S_{t-1}=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix},
\quad
G_t \approx
\begin{bmatrix}
0.6 & 0 \\
0 & 0.23
\end{bmatrix},
\quad
k_t v_t^\top=
\begin{bmatrix}
0.2 & 0 \\
0 & 0.2
\end{bmatrix}
$$

则有

$$
S_t = G_t \odot S_{t-1} + k_t v_t^\top
\approx
\begin{bmatrix}
0.8 & 0 \\
0 & 0.43
\end{bmatrix}
$$

如果 output gate 此时近似是 $0.7$，那么读出就会变成

$$
o_t \approx 0.7 \cdot S_t
=
\begin{bmatrix}
0.56 & 0 \\
0 & 0.301
\end{bmatrix}
$$

这个例子说明两件事。第一，旧记忆并不是整块保留，而是按位置衰减。第二，即使状态里保留了信息，最终输出仍然可以再筛一次。

为什么这种门控能提升 in-context learning，也就是“在上下文里临时学会当前任务”的能力？Google Research 对这一点的解释是：多层 GLA 可以实现一类带数据依赖权重的 Weighted Projected Gradient Descent，简称 WPGD。白话解释是，模型像在上下文里做一个“按样本重要性加权”的小优化过程，而 gate 就是这些权重的来源。可把它理解成：不是每个历史 token 都等价地参与“当前任务拟合”，门控会给更相关的 token 更高贡献。

文中常见的一类表达写法可以抽象成：

$$
x^\top P X^\top (y \odot \omega)
$$

这里 $\omega$ 就是由 gate 诱导出来的权重。这个式子不必死记，重要的是理解其含义：gating 本质上在做 weighting。也就是说，门控不是额外花哨装饰，而是在上下文学习中引入“谁更重要”的数据依赖重加权。

---

## 代码实现

实现 GLA 的关键点只有两个：

1. 维护一个固定大小的状态矩阵 `S`。
2. 每读入一个 token，先算 gate，再更新状态，再读出输出。

下面给出一个可运行的 Python 玩具实现。它不是训练代码，而是最小机制演示，重点是把 forget gate、output gate 和外积更新写清楚。

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def outer(u, v):
    return [[ui * vj for vj in v] for ui in u]

def elem_mul(a, b):
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def scalar_mul(s, a):
    return [[s * a[i][j] for j in range(len(a[0]))] for i in range(len(a))]

# 上一步状态
S_prev = [
    [1.0, 0.0],
    [0.0, 1.0],
]

# forget gate 的 logits，经过 sigmoid 后约为 [0.6, 0.23]
g1 = sigmoid(0.4054651081)   # ~= 0.6
g2 = sigmoid(-1.2083112059)  # ~= 0.23
G = [
    [g1, 0.0],
    [0.0, g2],
]

# 当前 token 的 key/value
k = [1.0, 0.0]
v = [0.2, 0.0]
kv = outer(k, v)
kv[1][1] = 0.2  # 构造与文章一致的对角增量

# 状态更新：S_t = G_t ⊙ S_{t-1} + k_t v_t^T
S_t = add(elem_mul(G, S_prev), kv)

# output gate
out_gate = 0.7
O_t = scalar_mul(out_gate, S_t)

assert round(S_t[0][0], 2) == 0.80
assert round(S_t[1][1], 2) == 0.43
assert round(O_t[0][0], 2) == 0.56
assert round(O_t[1][1], 2) == 0.30

print("S_t =", S_t)
print("O_t =", O_t)
```

如果用深度学习框架，一步伪代码通常就是这样：

```python
gate_f = sigmoid(W_f @ h_t + b_f)         # forget gate
S = gate_f * S + torch.outer(k_t, v_t)    # 固定 k x k 状态更新
read = W_read(S)
gate_o = sigmoid(W_o @ read + b_o)        # output gate
o_t = gate_o * read
```

这里 `torch.outer(k_t, v_t)` 是外积，输出形状是固定的，所以每一步更新状态的核心成本稳定在 $O(k^2)$。注意，这里的 $k$ 指特征维度，不是序列长度。序列变长时，状态大小不跟着线性增长，这是 GLA 能做长序列递推的根本原因。

真实工程例子可以看高分辨率图像生成。假设把一张超高分辨率图像切成大量 token，再送入生成模型。softmax attention 要处理大规模 token 两两关系，显存和带宽压力很大；GLA 只维护固定状态，并用 gate 控制哪些视觉上下文继续留在状态里，哪些应该被快速衰减。这样做的目标不是复刻完整全局注意力，而是用更低成本保留关键结构，比如大轮廓、局部纹理依赖、跨块一致性。

---

## 工程权衡与常见坑

GLA 的工程难点不在“能不能写出来”，而在“门控会不会退化”。最常见的问题是 sigmoid 饱和。饱和就是输出长期接近 0 或 1，梯度会变弱，门控就不再敏感。

下面是常见问题表：

| 问题 | 典型原因 | 缓解方式 |
|---|---|---|
| forget gate 接近 0，旧状态被清空过快 | gate 偏置过小或训练早期不稳定 | 调整偏置初始化，配合 norm |
| forget gate 接近 1，状态几乎不忘 | 偏置过大，模型学成“全保留” | 降低初始偏置，引入 gate dropout |
| output gate 全开，信息泄漏过多 | 输出门缺少约束 | 输出前做归一化，控制读出尺度 |
| 状态数值爆炸 | key/value 尺度失衡 | 对输入做 LayerNorm，控制 feature map 范围 |
| 退化成普通 linear attention | gate 学不到内容差异 | 检查 gate 输入是否足够依赖内容 |

一个很典型的调参例子是 gate bias。若 forget gate 初始偏置设成 0，sigmoid 后大约是 0.5，看似中性，但实际训练里可能很快两极化。把偏置往负方向拉一点，比如从 `0.0` 调到 `-1.0`，相当于让模型更容易先学会“谨慎保留”，再逐步打开；再加上 `gate dropout=0.1`，常常比完全不约束稳定得多。

还要注意归一化。很多人第一次实现时只关注 gate，忘了 key/value 的尺度管理。结果是外积项 `k_t v_t^T` 数值太大，把遗忘机制直接淹没。这样门控即使存在，也很难真正主导状态更新。工程上通常需要配合 LayerNorm、RMSNorm 或受控 feature map，让“旧记忆衰减”和“新信息写入”处于相近数值范围。

另一个坑是并行训练实现。GLA 的递推定义很自然，但训练时如果完全按 token 串行计算，吞吐会差。原始工作的重要贡献之一，就是给出并行且硬件友好的训练方式，使它不只是理论上递推高效，而是实际在 GPU 上也能跑得快。这也是为什么 GLA 不是“再加一个门就行”，而是需要算法和内核实现一起设计。

---

## 替代方案与适用边界

如果上下文很短，比如 1k token 左右，而且你最关心绝对表达力，softmax attention 仍然通常是最稳妥的选择。因为它直接显式建模任意 token 对之间的关系，不需要把信息压缩进固定状态。

如果主要目标是极长序列、流式推理、低内存占用，那么普通 linear attention 已经有价值；但只要任务开始依赖“谁重要、谁不重要”的内容选择性，GLA 往往比纯线性版本更合适。

如果你更偏向序列模型或状态空间模型一侧，Mamba、RWKV 这类 gated RNN/SSM 也属于同一大方向：都在强调递推状态和内容感知更新。区别在于，GLA 仍保留 attention 风格的 key/value 读写形式，更容易作为 attention 的替代层插进现有 Transformer 结构里；Mamba 则更像一套不同的骨架。

可以把适用边界总结成下面这张表：

| 方案 | 最适合的场景 | 优点 | 局限 |
|---|---|---|---|
| Softmax Attention | 短到中等上下文，高精度建模 | 表达力最强，生态成熟 | 长序列成本高 |
| Linear Attention | 超长序列，强吞吐，流式推理 | 状态固定，扩展性好 | 选择性不足 |
| GLA | 长序列且需要选择性记忆 | 兼顾扩展性与内容感知 | 实现与训练稳定性更复杂 |
| Mamba/RWKV | 递推建模、低延迟推理 | 选择性强，长序列友好 | 与标准 attention 接口差异更大 |

因此，一个简单判断标准是：

- 只需 1k 左右上下文：优先 softmax attention。
- 16k 以上长文本、长日志、长代码、长视频 token：GLA 比普通 linear attention 更值得考虑。
- 高分辨率视觉生成、需要大规模 token 推理：GLA 是很现实的折中方案。
- 若任务极度依赖复杂非线性状态更新：可以考虑 Mamba 或 GLA 与局部 softmax/SSM 结合。

GLA 不等于“全面替代 Transformer”。它更准确的定位是：在线性注意力这条路线里，用门控把“固定状态的高效性”和“按内容选择记忆”的能力重新接起来。

---

## 参考资料

1. Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim. *Gated Linear Attention Transformers with Hardware-Efficient Training*. ICML 2024. https://huggingface.co/papers/2312.06635  
2. Yingcong Li, Maryam Fazel, Ankit Singh Rawat, Samet Oymak, Davoud Ataee Tarzanagh. *Gating is Weighting: Understanding Gated Linear Attention through In-context Learning*. COLM 2025. https://research.google/pubs/gating-is-weighting-understanding-gated-linear-attention-through-in-context-learning/  
3. Lianghui Zhu, Zilong Huang, Bencheng Liao, Jun Hao Liew, Hanshu Yan, Jiashi Feng, Xinggang Wang. *DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention*. arXiv:2405.18428. https://papers.cool/arxiv/2405.18428  
4. Emergent Mind. *Gated Linear Attention (GLA)* 概览页。https://www.emergentmind.com/papers/2312.06635
