## 核心结论

线性注意力的因果约束处理，本质上是把“当前 Query 和全部历史 Key 逐个做打分”改写成“先把历史压缩成固定大小状态，再用当前 Query 去读这个状态”。

传统因果 softmax 注意力对第 $i$ 个 token 的输出可写成：

$$
y_i=\sum_{j \le i}\alpha_{ij}V_j,\quad
\alpha_{ij}=\frac{\exp(Q_i^\top K_j)}{\sum_{t\le i}\exp(Q_i^\top K_t)}
$$

线性注意力用可分解核，也就是“能拆成两边特征映射乘积的相似度函数”，把 $\exp(Q_i^\top K_j)$ 近似或替换成：

$$
\mathrm{sim}(Q_i,K_j)=\phi(Q_i)^\top \phi(K_j)
$$

于是有：

$$
y_i=
\frac{\sum_{j\le i}\phi(Q_i)^\top \phi(K_j)V_j}
{\sum_{j\le i}\phi(Q_i)^\top \phi(K_j)}
=
\frac{\phi(Q_i)^\top \left(\sum_{j\le i}\phi(K_j)\otimes V_j\right)}
{\phi(Q_i)^\top \left(\sum_{j\le i}\phi(K_j)\right)}
$$

这一步依赖乘法结合律。它把原本随序列长度增长的历史访问，改成对两个状态的读取：

- $S_i=\sum_{j\le i}\phi(K_j)\otimes V_j$
- $Z_i=\sum_{j\le i}\phi(K_j)$

因果约束不再需要额外的上三角 mask 去“屏蔽未来”，而是直接体现在状态只累加到 $i$ 为止。换句话说，系统从一开始就没有看到未来 token。

可以把它理解成一句话：先复制历史，再查询当前。所谓“复制历史”，就是把所有过去的 $\phi(K_j)V_j$ 累加进状态；所谓“查询当前”，就是用当前的 $\phi(Q_i)$ 对这个状态做一次投影，得到输出。

下面这张表能直接看出差异：

| 方案 | 历史表示方式 | 第 $i$ 步怎么得到输出 | 因果约束怎么实现 |
|---|---|---|---|
| 传统 softmax 注意力 | 保存全部 $K,V$ | 当前 $Q_i$ 与每个历史 $K_j$ 分别打分，再加权求和 | 对 $j>i$ 做 mask |
| 因果线性注意力 | 保存固定状态 $S_i,Z_i$ | 当前 $\phi(Q_i)$ 一次读取累计状态 | 状态只累计到当前步 |

如果特征维度记为 $d$，则单步更新状态的核心代价是 $O(d^2)$，不再随上下文长度 $N$ 增长；整段序列的计算是 $O(Nd^2)$。因此它尤其适合流式推理、长上下文增量生成和显存受限部署。

---

## 问题定义与边界

问题很具体：自回归生成时，模型只能看见过去，不能看见未来；同时，推理服务又希望显存开销尽量稳定，不要随着上下文长度线性增长。

这里“因果”指的是“第 $i$ 个位置只能依赖 $1\sim i$ 的信息”。“KV cache”指的是“为了后续 token 复用历史注意力，需要把所有过去的 Key 和 Value 存起来”。在传统 Transformer 推理里，序列越长，KV cache 越大，这就是长上下文推理越来越吃显存的直接原因。

真实工程例子：一个 GPT 风格模型在服务端做流式问答。假设用户会话可能达到几万 token，传统注意力每层都要保留完整 KV cache，显存随着长度持续增长。如果服务部署在单卡显存紧张的环境，瓶颈通常不是矩阵乘法速度，而是 KV cache 撑满显存。因果线性注意力的目标，就是把“长度相关存储”换成“固定大小状态存储”。

它解决的是以下需求：

| 维度 | 传统 softmax KV 缓存 | 因果线性注意力状态 |
|---|---|---|
| 存储 | 随序列长度线性增长，典型是 $O(Nd)$ 或更高常数 | 固定为状态矩阵 $S$ 和向量 $Z$，典型是 $O(d^2)$ |
| 推理复杂度 | 每来一个 token 都要读全部历史 KV | 每来一个 token 只更新并读取状态 |
| 因果控制 | 依赖 mask 保证看不到未来 | 状态递推天然只包含过去 |
| 训练并行性 | 强，可对整段序列并行 | 仍可并行实现，但递推解释更强 |

边界也要讲清楚。

第一，它不是“完全免费替代 softmax”。softmax 的归一化和竞争机制很强，线性注意力用的是可分解核，表达性质不同，效果要看任务和映射 $\phi$ 的设计。

第二，它在推理时天然适合顺序更新，但训练时如果真的一 token 一 token 跑，会浪费并行硬件。实际工程通常仍然用并行前缀和、chunk 分块等办法训练，而不是把 GPU 当 RNN 逐步器来用。

第三，它更适合“在线、流式、边缘部署、长序列推理”这些场景；如果你的目标是尽量逼近标准 softmax 的精确行为，或者训练时更看重成熟生态，那么需要和其他方案一起比较，而不是默认它全面更优。

---

## 核心机制与推导

核心机制只有三步：定义核、重排求和、递推更新。

“核”可以理解成“衡量 Query 和 Key 相似度的函数”。线性注意力要求它能写成两边映射的内积：

$$
\mathrm{sim}(Q_i,K_j)=\phi(Q_i)^\top \phi(K_j)
$$

于是因果形式的输出变成：

$$
y_i=
\frac{\sum_{j\le i}\phi(Q_i)^\top \phi(K_j)V_j}
{\sum_{j\le i}\phi(Q_i)^\top \phi(K_j)+\varepsilon}
$$

把与 $j$ 无关的 $\phi(Q_i)$ 提到外面：

$$
y_i=
\frac{\phi(Q_i)^\top \sum_{j\le i}\left(\phi(K_j)\otimes V_j\right)}
{\phi(Q_i)^\top \sum_{j\le i}\phi(K_j)+\varepsilon}
$$

定义两个状态：

$$
S_i=S_{i-1}+\phi(K_i)\otimes V_i,\quad S_0=0
$$

$$
Z_i=Z_{i-1}+\phi(K_i),\quad Z_0=0
$$

则：

$$
y_i=\frac{\phi(Q_i)^\top S_i}{\phi(Q_i)^\top Z_i+\varepsilon}
$$

这里的 $\otimes$ 是外积，也就是“把一个向量和另一个向量拼成矩阵”。如果 $\phi(K_i)\in\mathbb{R}^{d}$，$V_i\in\mathbb{R}^{d_v}$，那么 $S_i\in\mathbb{R}^{d\times d_v}$。它保存的不是单个历史 token，而是“历史被压缩后的统计量”。

这就是它等价 RNN 的原因。RNN 的核心特征是“有一个固定大小隐状态，每步更新一次，再读出输出”。因果线性注意力里，状态就是 $(S_i,Z_i)$，更新规则由当前 token 决定，读出规则由当前 Query 决定，所以它在推理视角上就是一个矩阵状态 RNN。

可以用一个文本图示理解：

| 时刻 | 输入 | 状态更新 | 输出读取 |
|---|---|---|---|
| $i-1$ | 已处理历史 | 持有 $S_{i-1}, Z_{i-1}$ | 已得到 $y_{i-1}$ |
| $i$ | 新来的 $(Q_i,K_i,V_i)$ | 加上 $\phi(K_i)\otimes V_i$ 与 $\phi(K_i)$ | 用 $\phi(Q_i)$ 读取 $S_i,Z_i$ 得到 $y_i$ |

下面给一个玩具例子，维度取 $d=2$，$\phi$ 取恒等映射，也就是 $\phi(x)=x$。

第一个 token：

- $K_1=[1,0]$
- $V_1=[1,2]$
- $Q_1=[0,1]$

先更新状态：

$$
S_1=K_1\otimes V_1=
\begin{bmatrix}
1\\
0
\end{bmatrix}
\otimes
\begin{bmatrix}
1\\
2
\end{bmatrix}
=
\begin{bmatrix}
1 & 2\\
0 & 0
\end{bmatrix}
$$

$$
Z_1=K_1=[1,0]
$$

再读取输出：

$$
Q_1^\top S_1=[0,1]
\begin{bmatrix}
1 & 2\\
0 & 0
\end{bmatrix}
=
[0,0]
$$

$$
Q_1^\top Z_1=[0,1]\cdot[1,0]=0
$$

所以如果不加 $\varepsilon$，这里分母会变成 0；如果只看未归一化读出，则输出是 $[0,0]$。这个例子说明两件事：

- 因果性是天然的，因为状态里只有第一个 token。
- 数值稳定性不能忽略，因为某些 $Q_i$ 可能和累计的 $Z_i$ 几乎正交。

继续加第二个 token：

- $K_2=[0,1]$
- $V_2=[3,4]$
- $Q_2=[1,1]$

则：

$$
S_2=S_1+K_2\otimes V_2=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},\quad
Z_2=[1,1]
$$

读出：

$$
Q_2^\top S_2=[1,1]
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
=[4,6]
$$

$$
Q_2^\top Z_2=2
$$

于是：

$$
y_2=[2,3]
$$

这就是“先把前序 $\phi(K)V$ 累加，再对当前 $\phi(Q)$ 做一次投影”的完整过程。

---

## 代码实现

实现时最关键的数据结构只有两个：

- `S`：形状为 `d_phi x d_value` 的状态矩阵
- `Z`：形状为 `d_phi` 的状态向量

推理循环的伪代码如下：

```pseudo
initialize S = zeros(d_phi, d_value)
initialize Z = zeros(d_phi)

for each token i:
    q = phi(Q[i])              # 当前查询特征
    k = phi(K[i])              # 当前键特征
    v = V[i]                   # 当前值向量

    S = S + outer(k, v)        # 把当前 token 写入历史状态
    Z = Z + k                  # 更新归一化统计量

    numerator = q^T * S        # 用当前查询读取累计内容
    denominator = q^T * Z + eps
    y[i] = numerator / denominator
```

下面给一个可运行的 Python 版本。它演示单层、单头、因果顺序更新，并包含断言。

```python
import numpy as np

def phi(x):
    # 一个常见选择是 ELU(x) + 1，保证非负
    return np.where(x > 0, x + 1.0, np.exp(x))

def causal_linear_attention(Q, K, V, eps=1e-8):
    n, d = Q.shape
    dv = V.shape[1]
    S = np.zeros((d, dv), dtype=np.float64)
    Z = np.zeros(d, dtype=np.float64)
    Y = np.zeros((n, dv), dtype=np.float64)

    for i in range(n):
        q = phi(Q[i])
        k = phi(K[i])
        v = V[i]

        S += np.outer(k, v)
        Z += k

        numerator = q @ S
        denominator = q @ Z + eps
        Y[i] = numerator / denominator

    return Y

# 玩具例子
Q = np.array([[0.0, 1.0],
              [1.0, 1.0]])
K = np.array([[1.0, 0.0],
              [0.0, 1.0]])
V = np.array([[1.0, 2.0],
              [3.0, 4.0]])

Y = causal_linear_attention(Q, K, V)

assert Y.shape == (2, 2)
assert np.all(np.isfinite(Y))
assert Y[1, 0] > Y[0, 0]
assert Y[1, 1] > Y[0, 1]
print(Y)
```

这段代码里，`phi` 使用了 `ELU + 1` 形式。它的作用是把特征映射到非负区域，避免归一化项频繁出现负值抵消。首次出现的“非负”可以理解成“映射后的每个分量不小于 0，这样累计和更稳定”。

如果要集成到现有 Transformer 推理里，思路也很直接：

- 训练阶段仍然保留 `Q,K,V` 的批处理计算。
- 推理阶段每个注意力头维护自己的 `S,Z`。
- 新 token 到来时，先算当前头的 `q,k,v`，更新状态，再读出结果。
- 如果需要长流式吞吐，可以把多个 token 组成 chunk，在 chunk 内做小批量前缀累加，在 chunk 间传递最终状态。

真实工程例子：做语音流式识别或边缘侧文本生成时，服务端往往更关心“每来一个 token 的延迟是否稳定”。传统 KV cache 在会话越来越长时，单步访存成本和内存占用都会变重；线性注意力则把每层每头的历史压成固定状态，部署者更容易提前估算峰值显存。

---

## 工程权衡与常见坑

线性注意力的优势很明确，但工程上并不是只把公式抄下来就结束。最常见的问题集中在训练并行、特征映射和归一化稳定性上。

先看常见坑：

| 问题 | 现象 | 原因 | 建议 |
|---|---|---|---|
| 推理能递推，训练却变慢 | 直接逐 token 训练吞吐很差 | GPU 擅长批量并行，不擅长纯顺序循环 | 训练用并行前缀和或 chunkwise 实现 |
| $\phi$ 取值有负数 | 输出震荡、归一化异常 | $Z_i$ 会发生抵消，分母可能接近 0 | 优先用非负映射，如 `ELU+1`、正特征映射 |
| $q^\top Z_i$ 很小 | 数值爆炸或 NaN | 分母接近 0 | 加 $\varepsilon`，必要时做裁剪或缩放 |
| 状态太小导致精度损失 | 长程依赖效果不稳定 | 固定状态压缩历史会丢信息 | 调整特征维度、做多头分摊、做任务验证 |
| chunk 切分不当 | 吞吐和延迟都一般 | chunk 太小开销大，太大又像回到全序列 | 用服务负载测试选 chunk 大小 |

其中最容易被忽视的是：训练和推理的最优实现不一样。

推理时顺序递推非常自然，因为系统本来就是 token-by-token 生成。训练时如果照抄这个顺序，会浪费大量并行能力。所以很多工程实现采用 chunked 推理或 chunkwise training。所谓 “chunk” 就是“把连续的一小段 token 作为一个块处理”。块内可以并行计算一部分前缀统计，块间只传递最终状态。这样做的目的是在“状态递推”与“硬件并行”之间找平衡。

另一个关键权衡是 $\phi$ 的选择。

- `ReLU` 简单，但大量 0 会让有效特征过稀，某些位置的分母容易偏小。
- `ELU + 1` 更平滑，且天然非负，通常比裸 `ReLU` 更稳定。
- 更复杂的随机特征映射或专门设计的核近似，可能更接近 softmax，但实现和调参复杂度更高。

一个经验判断是：如果你发现模型经常需要靠很大的 `eps` 才稳定，通常不是 `eps` 本身的问题，而是 $\phi$、缩放方式或状态统计范围出了问题。

---

## 替代方案与适用边界

线性注意力不是唯一的因果加速方案。要判断它是否适合，最好和另外几类方案放在一起看。

“FlashAttention”可以理解成“重写 softmax 注意力的计算顺序和显存访问方式，让同样的数学结果跑得更快、更省显存”。它并没有把历史压成固定大小状态，推理时仍然需要 KV buffer。也就是说，它优化的是实现效率，不是把注意力结构改成 RNN 状态。

“局部注意力 / 分块注意力”可以理解成“每个 token 只看附近窗口，不看全局”。它直接减少访问范围，因此能降低成本，但代价是远距离依赖被硬截断。

下面做一个简洁对比：

| 方案 | 优势 | 适用边界 |
|---|---|---|
| 线性注意力 | 固定状态递推，长序列推理内存稳定 | 适合流式生成、边缘部署；不保证完全复现 softmax 行为 |
| FlashAttention | 对标准 softmax 几乎无损，加速成熟 | 适合训练和常规推理；仍需保存 KV cache |
| 局部/滑窗注意力 | 实现直接，复杂度随窗口而非全长增长 | 适合局部上下文主导任务；远程依赖受限 |
| Reformer/稀疏注意力 | 用稀疏或哈希减少全对全计算 | 适合特定长序列任务；实现复杂，效果依赖模式设计 |

真实判断场景可以这样理解：

- 如果你做的是大模型离线训练，希望最大程度保持 softmax 语义，同时充分利用现成高性能内核，FlashAttention 往往更现实。
- 如果你做的是在线生成，用户上下文不断增长，设备显存严格受限，而且系统必须流式稳定运行，线性注意力更有结构优势。
- 如果任务本身只需要最近一段上下文，比如部分语音、日志流或局部时序建模，局部注意力可能更简单直接。

所以它的适用边界不是“全面替代 Transformer 注意力”，而是“当固定状态递推比完整历史缓存更重要时，给出一种严格因果、可流式部署的方案”。

---

## 参考资料

1. Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*  
   链接：https://arxiv.org/abs/2006.16236  
   贡献：线性注意力的代表性论文，给出因果形式、状态递推和 RNN 等价视角。建议最先读。

2. LIONS EPFL, *LION Part I: Linear Attention*  
   链接：https://lions-epfl.github.io/2025/lion-part1-model/  
   贡献：对 $S_i,Z_i$ 递推、归一化形式和工程实现讲得更直白，适合读完论文后补机制细节。

3. AJKdrag, *Causal Linear Attention as an RNN*  
   链接：https://ajkdrag.in/2-Zettels/causal-linear-attention-as-an-RNN  
   贡献：突出“推理时固定状态”的视角，适合从部署和增量生成角度理解。

4. Emergent Mind, *Linear Transformers*  
   链接：https://www.emergentmind.com/topics/linear-transformers  
   贡献：整理线性 Transformer 的核心公式、复杂度和变体，适合快速复习。

5. Emergent Mind, *Linear Attention Architectures*  
   链接：https://www.emergentmind.com/topics/linear-attention-architectures  
   贡献：从架构层面对线性注意力家族做总结，方便横向比较不同设计。
