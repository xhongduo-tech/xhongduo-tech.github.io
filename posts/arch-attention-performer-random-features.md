## 核心结论

Performer 的核心价值，不是“把注意力写得更花”，而是把标准 Transformer 中最贵的那一步，从“所有 token 两两计算”改写成“先压缩，再读取”的线性流程。

标准 softmax 注意力的核心项是 $e^{q^\top k}$。Performer 用 FAVOR+（Fast Attention Via positive Orthogonal Random features，意思是“用正交且为正的随机特征快速近似注意力”）把它写成随机特征内积的期望：

$$
\mathrm{softmax}(q,k)\approx \mathbb{E}[\phi(q)^\top \phi(k)]
$$

这里 $\phi(\cdot)$ 是随机特征映射，白话说，就是把原来在 $d$ 维空间里的向量，映射到一个新的 $m$ 维空间里，让原本难算的核函数变成点积近似。

于是整层注意力可以写成：

$$
\mathrm{Attention}(Q,K,V)\approx D^{-1}\big(\phi(Q)(\phi(K)^\top V)\big)
$$

其中 $D$ 是归一化项。这个公式最重要的意义是：你不再需要显式构造 $L\times L$ 的注意力矩阵。计算顺序变成“先算 $(\phi(K)^\top V)$，再乘 $\phi(Q)$”，复杂度从 $O(L^2d)$ 下降到 $O(Lmd)$。

当序列长度 $L$ 很大，且 $m\ll L$ 时，这个改写是决定性的。比如 $L=8192, d=512$ 的蛋白质序列任务里，Performer 可以支持 batch=8 训练，而传统全注意力模型往往 batch=1 都会 OOM。这不是“快一点”，而是“能不能训起来”的区别。

另外，Performer 不是随便采样随机向量，而是优先使用正交随机特征。正交的意思是“这些随机方向尽量彼此不重复”，白话说就是减少重复采样带来的浪费。相比独立高斯随机特征，它能显著降低估计方差，在相同 $m$ 下得到更稳定的近似。

---

## 问题定义与边界

Performer 要解决的问题非常明确：标准 Transformer 的 softmax 注意力在长序列上代价太高。

对长度为 $L$、隐藏维度为 $d$ 的输入，标准注意力要先算 $QK^\top$，这会生成一个 $L\times L$ 的矩阵。矩阵里的每个元素都表示一个 token 对另一个 token 的相关性。白话说，模型会把每个位置和所有位置都比较一遍。于是：

- 计算复杂度约为 $O(L^2d)$
- 显存复杂度约为 $O(L^2)$

当 $L$ 从 512 增长到 8192 时，二次复杂度会迅速失控。

Performer 的思路不是删掉注意力，而是近似 softmax 核本身。它假设：如果能找到映射 $\phi$，使得

$$
e^{q^\top k}\approx \phi(q)^\top \phi(k)
$$

那么就能把原来需要两两比较的过程，改写成先聚合 key/value，再由 query 去读取聚合结果。

新手可以把它理解成下面这件事：

1. 先把每个 query 和 key 投影成一个 $m$ 维向量 $\phi(q), \phi(k)$  
2. 把所有 key-value 信息压缩成一个 $m$ 维“摘要”  
3. 每个 query 再去读取这个摘要，而不是逐个和所有 key 配对

这就是“先算 $(\phi(K)^\top V)$，再乘 $\phi(Q)$”的直观含义。

下面先对比标准 attention 和 Performer：

| 方法 | 时间复杂度 | 显存复杂度 | 是否近似 | 长序列能力 |
|---|---:|---:|---|---|
| 标准 Softmax Attention | $O(L^2d)$ | $O(L^2)$ | 否 | 差，$L$ 大时很容易 OOM |
| Performer | $O(Lmd)$ | $O(md)$ 或 $O(Lm)$ | 是 | 强，适合 $L\gg m$ |
| 关键条件 | 当 $m\ll L$ 时收益明显 | 不需存完整注意力图 | 有方差 | 更适合 8k、16k 以上 |

它的适用边界也很清楚：

- 适合超长序列，尤其是 $L\gg m$ 的场景
- 适合内存受限但必须保留全局依赖的任务
- 不适合要求“完全精确 softmax”且显存足够的场景
- 多层堆叠后近似误差会累积，必须控制方差

所以 Performer 不是“永远优于标准注意力”，而是“在长序列约束下更可训练”。

---

## 核心机制与推导

先看标准注意力：

$$
\mathrm{Att}(Q,K,V)=\mathrm{diag}(A\mathbf{1})^{-1}AV,\quad A_{ij}=e^{q_i^\top k_j}
$$

这里 $A$ 是未归一化注意力核矩阵。问题在于，直接构造 $A$ 需要 $L^2$ 个元素。

Performer 的关键观察是：softmax 核 $e^{q^\top k}$ 可以写成一个随机特征期望。对某个随机向量 $\omega$，定义特征映射：

$$
\phi_\omega(x)=\exp\left(\omega^\top x-\frac{\|x\|^2}{2}\right)
$$

如果对多个随机向量采样并拼接，就得到 $\phi(x)\in\mathbb{R}^m$。此时可以用 Monte Carlo（蒙特卡洛，意思是“用随机采样逼近期望”）估计：

$$
e^{q^\top k}\approx \phi(q)^\top \phi(k)
$$

于是：

$$
A \approx \phi(Q)\phi(K)^\top
$$

再把它代回注意力公式：

$$
\mathrm{Att}(Q,K,V)\approx D^{-1}\phi(Q)\phi(K)^\top V
$$

利用矩阵乘法结合律：

$$
\mathrm{Att}(Q,K,V)\approx D^{-1}\big(\phi(Q)(\phi(K)^\top V)\big)
$$

其中归一化项为：

$$
D=\mathrm{diag}\big(\phi(Q)(\phi(K)^\top \mathbf{1})\big)
$$

这一步就是 Performer 真正省钱的原因。因为：

- $\phi(K)^\top V$ 的形状是 $m\times d$
- $\phi(Q)$ 的形状是 $L\times m$
- 不需要构造 $L\times L$ 的矩阵

### 玩具例子

设 $n=4,d=8,m=2$。那么：

- $\phi(Q)$ 是 $4\times 2$
- $\phi(K)$ 是 $4\times 2$
- $V$ 是 $4\times 8$

先算：

$$
\phi(K)^\top V \in \mathbb{R}^{2\times 8}
$$

成本大约是 $4\times 2\times 8=64$ 次乘加。

再算：

$$
\phi(Q)(\phi(K)^\top V)\in \mathbb{R}^{4\times 8}
$$

而标准 attention 需要先构造 $4\times 4$ 的注意力图，再乘 $V$，核心量级是 $4^2\times 8=128$。这里差距还不大，因为 $n$ 很小；但当 $n=8192,m=256$ 时，差距会被放大成数量级差异。

### 为什么要用正交随机特征

随机特征近似的问题，不在于“会不会偏”，而在于“方差有多大”。方差可以理解成“同样的方法重复多次，结果抖动有多大”。

如果 $\omega_1,\dots,\omega_m$ 彼此独立，那么不同采样方向可能重复覆盖同一块区域，信息利用率不高。正交随机特征要求这些方向近似正交，效果是：

- 降低估计方差
- 在相同 $m$ 下提高近似质量
- 多层网络中更稳定

常见经验是，理论上只要

$$
m=\Theta(d\log d)
$$

就能把误差控制在合理范围内。它不是说“必须精确等于这个值”，而是说特征数应随着维度增长，但不需要像序列长度那样暴涨。

| 参数 | 含义 | 增大后的影响 |
|---|---|---|
| $d$ | 原始特征维度 | 表达能力增强，但映射更复杂 |
| $m$ | 随机特征数 | 近似更准，但线性项成本更高 |
| 正交化 | 约束采样方向互不重复 | 方差下降，更稳定 |
| redraw | 训练中重采样特征 | 减少固定采样误差积累 |

### 真实工程例子

蛋白质序列建模是 Performer 很典型的应用。蛋白质链很长，长度上千甚至上万，而且远距离位置之间可能存在功能相关性，不能简单截断上下文。

假设输入长度 $L=8192$，隐藏维度 $d=512$。标准 Transformer 要构造 $8192\times8192$ 注意力图，单层就已经非常吃显存；实际训练里常常只能把 batch 压到 1，甚至还要减层数、减维度。Performer 则把核心代价改成 $O(Lmd)$，如果取 $m=256$，就能在同样硬件上保住更大的 batch，例如 batch=8。这里的收益不是理论上的，而是直接决定吞吐、稳定性和是否能完成训练。

---

## 代码实现

下面给一个最小可运行的 Python 示例。它不追求数值最优，只演示 Performer 的顺序化计算逻辑：先算 `kv = phi_k.T @ V`，再算 `phi_q @ kv`，最后用归一化项 `z` 做缩放。

```python
import numpy as np

def orthogonal_gaussian_matrix(m, d, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    while len(blocks) * d < m:
        g = rng.normal(size=(d, d))
        q, _ = np.linalg.qr(g)
        blocks.append(q.T)
    w = np.concatenate(blocks, axis=0)[:m]
    return w

def phi(x, w):
    # x: [n, d], w: [m, d]
    x_norm_sq = np.sum(x * x, axis=1, keepdims=True) / 2.0
    proj = x @ w.T
    features = np.exp(proj - x_norm_sq) / np.sqrt(w.shape[0])
    return features

def performer_attention(Q, K, V, w, eps=1e-9):
    phi_q = phi(Q, w)              # [n, m]
    phi_k = phi(K, w)              # [n, m]

    kv = phi_k.T @ V               # [m, dv]
    k_sum = phi_k.sum(axis=0)      # [m]
    z = 1.0 / (phi_q @ k_sum + eps)  # [n]

    out = (phi_q @ kv) * z[:, None]
    return out

# toy example
n, d, dv, m = 4, 8, 3, 6
rng = np.random.default_rng(42)
Q = rng.normal(size=(n, d))
K = rng.normal(size=(n, d))
V = rng.normal(size=(n, dv))
w = orthogonal_gaussian_matrix(m, d, seed=123)

out = performer_attention(Q, K, V, w)

assert out.shape == (n, dv)
assert np.isfinite(out).all()
assert np.all(np.abs(out) < 1e6)
```

这个实现对应的核心步骤就是：

```python
phi_q = phi(Q)
phi_k = phi(K)
kv = phi_k.T @ V
z = 1 / (phi_q @ phi_k.sum(axis=0))
out = (phi_q @ kv) * z[:, None]
```

其中：

- `phi_q` 和 `phi_k` 是随机特征映射后的 query/key
- `kv` 是把所有 key-value 信息先压缩到 $m$ 维
- `z` 是归一化因子，对应公式里的 $D^{-1}$

如果把这个流程和标准 attention 对比，可以把它理解成：

- 标准 attention：先生成完整注意力图，再拿图去加权 `V`
- Performer：先把 `K,V` 压成一个低维摘要，再由 `Q` 去读取摘要

实际工程里还会加更多细节：

- 按头分别采样或共享随机特征
- 训练时使用 redraw 周期性重采样
- 对 causal mask 做前缀和式线性实现
- 做数值稳定处理，避免指数爆炸

---

## 工程权衡与常见坑

Performer 最大的工程代价，不是实现复杂，而是“近似带来的方差管理”。

它对 softmax 核的估计通常是无偏的。无偏的意思是“平均起来不偏离真值”。但单次估计仍然会抖动，这个抖动就是方差。单层误差可能很小，但模型一深、序列一长、训练一久，误差就可能层层累积，最后反映在困惑度、生成质量或检索精度上。

常见坑主要有下面几类。

| 问题 | 表现 | 原因 | 常见处理 |
|---|---|---|---|
| 随机特征数太小 | 精度明显掉 | Monte Carlo 方差过大 | 提高 $m$ |
| 不做正交化 | 结果波动大 | 采样方向重复 | 用 orthogonal features |
| 不做 positive feature | 数值不稳定 | 核近似失去正值结构 | 使用 FAVOR+ 正值构造 |
| 长层堆叠后变差 | perplexity 上升 | 误差逐层累积 | redraw、增大 $m$、减深度 |
| 和 exact baseline 差距大 | 怀疑实现或近似失控 | 方差或归一化错误 | 对比 FlashAttention |

### 真实工程例子：PG-19 长文本训练

PG-19 是长文本语言建模数据集，序列很长，特别适合暴露近似误差累积的问题。一个常见现象是：Performer 在显存上明显更省，但如果关闭正交化、positive feature 或 redraw，困惑度会逐渐落后于精确注意力。

这时很实用的排查方法是拿 FlashAttention 做 baseline。FlashAttention 的关键点是 exact，也就是“结果仍然是精确 softmax，只是通过更好的 IO 调度减少显存访问”。所以：

- 如果 FlashAttention 明显更好，说明 Performer 的近似误差在作祟
- 如果两者接近，说明当前 $m$ 和随机特征设置基本够用

| 方法 | 是否精确 | 内存优化来源 | 误差来源 | 适合拿来做基线吗 |
|---|---|---|---|---|
| FAVOR+ Performer | 否 | 核近似 + 线性顺序计算 | Monte Carlo 方差 | 是 |
| FlashAttention | 是 | 分块计算 + 更少 HBM 访问 | 无近似误差 | 非常适合 |

工程上一个务实结论是：Performer 解决的是“长序列可训练性”，不是“在所有指标上替代 exact attention”。如果任务对 perplexity 很敏感，或者你发现长上下文质量持续劣化，就应该认真评估近似误差是否值得。

---

## 替代方案与适用边界

Performer 不是唯一的长序列方案，它只是“用随机特征近似 softmax”这一条路线的代表。

最常被拿来比较的是 FlashAttention。FlashAttention 不改模型定义，只改计算过程，所以输出仍然是 exact softmax attention。它适合 GPU 内存尚可、但希望更高吞吐和更稳定精度的场景。简单说：

- 要 exact，优先看 FlashAttention
- 要极长上下文且显存吃紧，优先看 Performer

另一些常见替代方案包括 Reformer、Linformer 等。它们的思路不同：

- Reformer 通过局部敏感哈希近似“谁和谁应该互相看”
- Linformer 假设注意力矩阵本身低秩，用低秩投影压缩
- Performer 则是直接近似 softmax 核，理论结构更统一

| 方法 | 时间复杂度 | 是否近似 | 主要思想 | 适用边界 |
|---|---:|---|---|---|
| Performer | $O(Lmd)$ | 是 | 随机特征近似 softmax 核 | 超长序列、显存紧张 |
| FlashAttention | 结果仍接近 $O(L^2)$ 计算，但 IO 更优 | 否 | 重排计算与分块 | 追求 exact、硬件较强 |
| Reformer | 通常低于全注意力 | 是 | LSH 稀疏路由 | 稀疏依赖明显的任务 |
| Linformer | 近似线性 | 是 | 低秩投影 | 注意力矩阵低秩假设成立时 |

因此选择标准可以很直接：

- 如果 GPU 内存允许，而且你要最稳妥的困惑度或精确对齐，选 FlashAttention
- 如果序列长到 8k、16k 甚至更高，且训练已经被 $L^2$ 卡住，Performer 更现实
- 如果任务结构本身有稀疏模式或低秩结构，也可以评估 Reformer、Linformer 之类方法

Performer 的优势不是“绝对最好”，而是“在长序列与资源受限之间给出一个可证明、可实现、可训练的折中”。

---

## 参考资料

1. Choromanski 等，*Rethinking Attention with Performers*，ICLR 2021。  
   这是 FAVOR+ 的原始论文，给出了 softmax 核的随机特征推导、无偏性证明和长序列实验。新手如果要补全理论细节，优先读这篇。

2. Tri Dao 等，FlashAttention 相关文章与官方说明。  
   这组工作强调“不做近似”的 exact attention 优化，适合用来理解为什么 FlashAttention 常被当作 Performer 的工程基线。

3. Felix X. Yu 等，*Orthogonal Random Features*，NeurIPS 2016。  
   这篇工作解释了为什么正交随机特征比独立随机特征方差更低，对理解 Performer 中的 orthogonal features 很关键。

4. 阅读顺序建议。  
   如果你是初学者，先读 Performer 论文中的问题定义和主公式，再看 FlashAttention 的 exact 思路，最后补 Orthogonal Random Features。这样更容易分清“近似 attention”和“精确 attention 优化”到底差在哪里。
