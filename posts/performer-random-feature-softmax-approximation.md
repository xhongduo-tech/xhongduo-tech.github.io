## 核心结论

Performer 的关键不是“改掉注意力”，而是把 Softmax Attention 里的指数核写成可分离的形式，再利用矩阵乘法结合律把二次复杂度改成线性复杂度。

设 softmax 核为

$$
\kappa(q,k)=\exp(q^\top k)
$$

Performer 用随机特征映射 $\phi(x)$ 去近似这个核，使得

$$
\mathbb{E}[\phi(q)^\top \phi(k)] = \exp(q^\top k)
$$

一种常见写法是正随机特征

$$
\phi(x)=\frac{1}{\sqrt m}\exp\!\left(-\frac{\|x\|^2}{2}\right)
\left[\exp(\omega_1^\top x),\dots,\exp(\omega_m^\top x)\right]
$$

其中 $\omega_i \sim \mathcal{N}(0,I)$。这里“随机特征”可以先理解成：把一个向量投影到若干随机方向，再把这些投影值变成新的特征坐标。

注意一个精度上必须说清的点：真正的 Attention 不是只算核矩阵，还要做行归一化。因此准确写法不是直接

$$
\mathrm{Attention}(Q,K,V)\approx \phi(Q)\phi(K)^\top V
$$

而是

$$
\mathrm{Attention}(Q,K,V)
\approx
D^{-1}\phi(Q)\bigl(\phi(K)^\top V\bigr)
$$

其中

$$
D=\mathrm{diag}\bigl(\phi(Q)(\phi(K)^\top \mathbf{1})\bigr)
$$

也就是每个 query 的分母要单独算出来。很多简介只写“$\phi(Q)\phi(K)^\top$ 近似 softmax”，这在讲核近似时没问题，但放到完整 Attention 实现里会漏掉归一化。

复杂度上，标准注意力需要显式形成 $N\times N$ 分数矩阵，计算量约为 $O(N^2 d)$；Performer 先算 $\phi(K)^\top V$ 和 $\phi(K)^\top \mathbf{1}$，再左乘 $\phi(Q)$，复杂度变成 $O(Nmd)$，其中 $m$ 是随机特征维度，通常取到足够大以控制误差。

---

## 问题定义与边界

标准自注意力对每个 token 都要和所有 token 计算匹配分数。这里“token”就是序列里的一个离散位置，可以是一词、一子词，或图像中的一个 patch。

对长度为 $N$ 的序列，分数矩阵大小是 $N\times N$。如果隐藏维度是 $d$，则核心瓶颈是：

| 方法 | 计算主项 | 空间主项 | 是否保留 full attention | 主要假设 |
|---|---:|---:|---|---|
| 标准 Softmax Attention | $O(N^2 d)$ | $O(N^2)$ | 是 | 无 |
| Performer | $O(Nmd)$ | $O(Nm)$ | 是近似 | softmax 核可用随机特征近似 |
| Linformer | 约 $O(Nkd)$ | 约 $O(Nk)$ | 否，低秩近似 | 注意力矩阵近似低秩 |
| Reformer | 依实现而定，常低于二次 | 更低 | 否，局部/分桶近似 | 相似 token 可聚类或哈希 |

边界也要明确：

1. Performer 近似的是 softmax 核，不是精确等价。
2. 它更适合长序列，因为只有在 $N$ 很大时，省掉 $N^2$ 才明显。
3. 当 $m$ 不够大时，误差会直接反映到注意力分布上。
4. 如果序列本来很短，标准 softmax 往往更简单，且在现代 GPU 上不一定更慢。

一个面向新手的玩具理解是：原本每个 token 要和另外 $N$ 个 token 单独打分；Performer 改成“先把所有 key 压成 $m$ 个随机特征上的统计量”，然后每个 query 只和这 $m$ 个统计量交互。

---

## 核心机制与推导

先看标准注意力。忽略多头和 batch 维度后，单层自注意力可写成

$$
\mathrm{Att}(Q,K,V)=\mathrm{softmax}(QK^\top)V
$$

把第 $i$ 个 query 的输出写开：

$$
y_i=
\frac{\sum_{j=1}^{N}\exp(q_i^\top k_j)v_j}
{\sum_{j=1}^{N}\exp(q_i^\top k_j)}
$$

难点在于，分子和分母都包含所有 $j$，所以每个 $q_i$ 都要与全部 key 成对交互。

Performer 的核心是用可分离核近似：

$$
\exp(q^\top k)\approx \phi(q)^\top \phi(k)
$$

于是分子变成

$$
\sum_{j=1}^{N}\phi(q_i)^\top \phi(k_j) v_j
=
\phi(q_i)^\top \left(\sum_{j=1}^{N}\phi(k_j)v_j^\top\right)
$$

分母变成

$$
\sum_{j=1}^{N}\phi(q_i)^\top \phi(k_j)
=
\phi(q_i)^\top \left(\sum_{j=1}^{N}\phi(k_j)\right)
$$

所以第 $i$ 个输出可写成

$$
y_i \approx
\frac{
\phi(q_i)^\top \left(\sum_{j=1}^{N}\phi(k_j)v_j^\top\right)
}{
\phi(q_i)^\top \left(\sum_{j=1}^{N}\phi(k_j)\right)
}
$$

矩阵形式就是

$$
Y \approx D^{-1}\phi(Q)\bigl(\phi(K)^\top V\bigr)
$$

其中

$$
D=\mathrm{diag}\bigl(\phi(Q)(\phi(K)^\top \mathbf{1})\bigr)
$$

这一步的本质是“把所有 pairwise 交互，改写成对全局统计量的查询”。

### 为什么这个 $\phi(x)$ 能近似指数核

利用高斯随机向量的矩母函数。若 $\omega\sim\mathcal{N}(0,I)$，则有

$$
\mathbb{E}[\exp(\omega^\top x)] = \exp\!\left(\frac{\|x\|^2}{2}\right)
$$

因此

$$
\mathbb{E}\left[
\exp\!\left(\omega^\top q-\frac{\|q\|^2}{2}\right)
\exp\!\left(\omega^\top k-\frac{\|k\|^2}{2}\right)
\right]
=
\exp(q^\top k)
$$

这正是上面特征映射成立的原因。白话说，随机投影后的指数特征，平均起来恰好“拼回”原来的 softmax 核。

### 玩具例子

取二维向量：

$$
q=(1,0),\quad k=(0,1)
$$

它们的点积是 $q^\top k=0$，所以真实核值为

$$
\exp(q^\top k)=1
$$

若取两个随机方向的一个极简示意版：

$$
\omega_1=(1,0),\quad \omega_2=(0,1)
$$

则

$$
\phi(q)=\frac{e^{-1/2}}{\sqrt 2}[e^1,e^0],\quad
\phi(k)=\frac{e^{-1/2}}{\sqrt 2}[e^0,e^1]
$$

数值近似为：

| 向量 | 第一维 | 第二维 |
|---|---:|---:|
| $\phi(q)$ | 1.166/1.414 | 0.607/1.414 |
| $\phi(k)$ | 0.607/1.414 | 1.166/1.414 |

两者内积约为 $0.89$，已经接近 $1$。这个例子不精确，因为 $m=2$ 太小，而且这里取的是手工方向，不是真正的高斯随机样本。但它足够说明：随机特征展开后，点积可以逼近指数核。

### 真实工程例子

假设做 16k token 的长文档问答，单头维度 $d=64$。

标准注意力的打分矩阵大小是：

$$
16000 \times 16000 = 2.56\times 10^8
$$

仅分数矩阵就已经非常大，还没算多头、batch、梯度缓存。

如果 Performer 取 $m=256$，则主计算从“每个 token 对所有 token”变成“每个 token 对 256 个特征统计量”，主项近似从 $O(N^2d)$ 改为 $O(Nmd)$。这里虽然常数项和实现细节仍重要，但量级已经从平方变线性，更适合超长文本、长语音、蛋白序列这类场景。

---

## 代码实现

下面给出一个可运行的简化 Python 实现。它演示三件事：

1. 如何生成高斯随机特征；
2. 如何计算 Performer 近似注意力；
3. 如何验证输出形状与归一化分母为正。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def exact_attention(Q, K, V):
    scores = Q @ K.T
    probs = softmax(scores, axis=-1)
    return probs @ V

def positive_random_features(X, omega):
    # X: [N, d], omega: [m, d]
    # output: [N, m]
    sq_norm = np.sum(X * X, axis=-1, keepdims=True) / 2.0
    return np.exp(X @ omega.T - sq_norm) / np.sqrt(omega.shape[0])

def performer_attention(Q, K, V, omega, eps=1e-8):
    phi_q = positive_random_features(Q, omega)   # [N, m]
    phi_k = positive_random_features(K, omega)   # [N, m]

    kv = phi_k.T @ V                             # [m, dv]
    k1 = phi_k.T @ np.ones((K.shape[0], 1))      # [m, 1]

    numerator = phi_q @ kv                       # [N, dv]
    denominator = phi_q @ k1                     # [N, 1]

    return numerator / (denominator + eps), denominator

def make_orthogonal_rows(m, d, seed=0):
    # 简化版 ORF：按块生成正交行，再拼接到 m 行
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < m:
        A = rng.normal(size=(d, d))
        Qm, _ = np.linalg.qr(A)
        rows.extend(list(Qm))
    return np.array(rows[:m])

def main():
    rng = np.random.default_rng(42)
    N, d, dv, m = 6, 4, 3, 128

    Q = rng.normal(size=(N, d))
    K = rng.normal(size=(N, d))
    V = rng.normal(size=(N, dv))

    omega = rng.normal(size=(m, d))
    approx, denom = performer_attention(Q, K, V, omega)
    exact = exact_attention(Q, K, V)

    assert approx.shape == exact.shape == (N, dv)
    assert np.all(denom > 0), "正特征下分母应为正"
    
    # 误差不会为 0，但在 m 足够大时通常可接受
    mae = np.mean(np.abs(approx - exact))
    assert mae < 0.5, mae

if __name__ == "__main__":
    main()
```

这段代码是教学版，不是生产版。真实实现还会处理：

| 步骤 | 输入形状 | 输出形状 | 作用 |
|---|---|---|---|
| 生成 $\omega$ | $(m,d)$ | $(m,d)$ | 采样随机方向 |
| 计算 $\phi(Q)$ | $(N,d)$ | $(N,m)$ | 把 query 映射到特征空间 |
| 计算 $\phi(K)$ | $(N,d)$ | $(N,m)$ | 把 key 映射到特征空间 |
| 计算 $\phi(K)^\top V$ | $(N,m),(N,d_v)$ | $(m,d_v)$ | 聚合 value 的全局统计量 |
| 计算 $\phi(K)^\top \mathbf{1}$ | $(N,m)$ | $(m,1)$ | 得到归一化分母需要的统计量 |
| 左乘 $\phi(Q)$ 并归一化 | $(N,m)$ | $(N,d_v)$ | 得到近似注意力输出 |

如果是自回归场景，还不能直接“看见未来”。这时要把上面的全局求和改成 prefix-sum，也就是前缀累计和，保证第 $t$ 个位置只使用 $1\sim t$ 的 key/value 统计量。

---

## 工程权衡与常见坑

Performer 不是“免费加速”。它是用随机近似换复杂度，所以工程效果高度依赖实现细节。

| 常见坑 | 现象 | 原因 | 处理方式 |
|---|---|---|---|
| $m$ 太小 | 输出抖动大，逼近 softmax 很差 | 特征估计方差高 | 增大 $m$，通常先从 $d$ 的数倍开始试 |
| 忘记分母归一化 | 输出尺度异常 | 只算了核矩阵近似，没算 attention 归一化 | 必须同时计算 $\phi(K)^\top \mathbf{1}$ |
| 直接用非正特征 | 训练不稳，甚至分母异常 | softmax 权重本应非负 | 使用正随机特征 |
| 指数溢出 | `exp` 变成 `inf` | 内积过大或范数太大 | 做数值平移、混合精度保护、分块计算 |
| 随机方向相关性强 | 误差偏大 | 特征彼此冗余 | 使用 ORF，减少方差 |
| 短序列强行替换 | 速度没提升，精度还下降 | 近似误差抵不过二次项开销 | 短序列保留标准 attention |

这里的 ORF 指正交随机特征。白话说，就是不要让不同随机方向太像，否则等于重复采样同一类信息。Performer 里的 FAVOR+ 在“正特征 + 正交化”这两个点上都做了针对 softmax 的稳定化设计。

一个真实工程坑是长文本训练中出现“注意力变平均”。常见原因不是模型不会学，而是 $m$ 取太小、随机方向没处理好，导致不同 query 的分子分母都被压成相似值，最后每个 token 看起来都“差不多重要”。

---

## 替代方案与适用边界

Performer 并不是所有长序列任务的统一答案。它适合“仍想保留 softmax 风格的全局注意力”，但又承受不起 $N^2$ 成本的场景。

| 方法 | 核心思路 | 复杂度特征 | 优势 | 局限 |
|---|---|---|---|---|
| 标准 Attention | 精确 softmax | 二次 | 最稳、最直接 | 长序列成本高 |
| Performer | 随机特征近似 softmax 核 | 线性于 $N$ | 保留全局交互，理论保证较强 | 有近似误差，要调 $m$ |
| Linformer | 低秩投影 | 线性或近线性 | 实现相对直接 | 依赖低秩假设 |
| Reformer | 哈希/分桶相似 token | 低于二次 | 超长序列显存友好 | 近邻结构可能漏掉全局关系 |
| FlashAttention | IO 优化精确 attention | 仍是二次，但更快 | 不引入近似误差 | 序列极长时仍受 $N^2$ 限制 |

适用边界可以直接概括为：

1. 序列极长，优先考虑 Performer。
2. 序列中等长度，先看 FlashAttention 一类精确优化。
3. 任务天然局部，如语音帧局部建模，也许滑窗或稀疏注意力更合适。
4. 如果模型对注意力分布极其敏感，且你无法接受近似误差，仍应保留标准 softmax。

一个新手容易混淆的点是：FlashAttention 和 Performer 解决的问题不同。前者主要是“把精确 attention 做得更省显存、更高效”，后者主要是“把 attention 的数学复杂度从二次改成线性”。

---

## 参考资料

1. Krzysztof Choromanski, et al. *Rethinking Attention with Performers*. ICLR 2021.  
用于 FAVOR+、正随机特征、ORF、复杂度和理论保证的原始来源。  
https://research.google/pubs/rethinking-attention-with-performers/

2. OpenReview 论文页面：*Rethinking Attention with Performers*.  
用于核对论文版本与摘要描述。  
https://openreview.net/forum?id=Ua6zuk0WRH

3. Michael Brenndoerfer. *Linear Attention: Breaking the Quadratic Bottleneck with Kernel Feature Maps*.  
用于补充线性注意力的结合律直觉和核特征映射解释。  
https://mbrenndoerfer.com/writing/linear-attention-kernel-feature-maps-efficient-transformers

4. Choromanski et al. arXiv HTML 版本。  
用于核对 softmax 核、正随机特征与 FAVOR+ 的公式表述。  
https://ar5iv.labs.arxiv.org/html/2009.14794
