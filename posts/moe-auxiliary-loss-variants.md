## 核心结论

MoE，Mixture of Experts，指“只激活少数几个专家网络的稀疏模型”。它的核心难点不是把专家堆多，而是让路由器（router，负责给 token 分配专家的小网络）既学会分工，又不要把所有 token 都塞给少数热门专家。

Importance Loss、Load Loss 和 router z-loss 可以看成三类互补的辅助约束：

| 损失项 | 直接约束对象 | 主要目标 | 典型作用 |
| --- | --- | --- | --- |
| Importance Loss | 平均路由概率 $\hat p_j$ | 让“想去哪里”的分布更均衡 | 防止 router 的软概率长期偏向少数专家 |
| Load Loss | 实际负载 $f_j$ 与平均概率 $\hat p_j$ | 让“真的去了哪里”的分配更均衡 | 直接抑制热门专家吞掉大部分 token |
| z-loss | router logits $z_{ij}$ | 控制 logits 幅度，避免 softmax 过热 | 降低数值爆炸、梯度饱和、NaN 风险 |

三者分工不同。Importance Loss 管“倾向”，Load Loss 管“结果”，z-loss 管“数值温度”。如果只用前两者，专家负载可能好一些，但路由 logits 仍可能变得极大，softmax 接近 one-hot，训练会越来越硬；如果只用 z-loss，数值稳定了，但专家仍可能严重失衡。

玩具例子可以直接说明差异。假设 4 个专家，100 个 token，最终分配是 $[60,30,10,0]$。这说明第 1 个专家已经过热，第 4 个专家完全闲置。Importance Loss 会看到平均概率不均，推动 router 降低对热门专家的偏爱；Load Loss 会进一步利用“真的被选中了多少次”这个信息，把梯度更集中地打到过热专家上；z-loss 则不关心选了谁，而是关心 router 给分是不是越来越夸张。

真实工程里，这三项约束尤其重要。ST-MoE 这类超大稀疏模型在训练时常把总损失写成
$$
L = L_{\text{task}} + c_B L_{\text{balance}} + c_z L_z,
$$
其中 $L_{\text{balance}}$ 可以包含 Importance Loss、Load Loss 或它们的变体，$c_z$ 常取较小值如 $0.001$。原因很直接：大模型里只要 router 稍微失控，局部专家会爆满，激活值和梯度也会一起失控。

再把三者的职责压缩成一句工程判断：

| 训练现象 | 优先怀疑什么 | 常见对应项 |
| --- | --- | --- |
| 平均概率明显偏向少数专家 | router 软分布塌缩 | Importance Loss |
| 每步实际 token 计数严重失衡 | 热门专家拥堵 | Load Loss |
| logits / LogSumExp 持续抬升 | softmax 过热 | z-loss |

---

## 问题定义与边界

先明确问题。设一个 batch 里有 $T$ 个 token，$N$ 个专家。对于第 $i$ 个 token，router 输出一组 logits
$$
z_i = (z_{i1}, z_{i2}, \dots, z_{iN}),
$$
再经过 softmax 变成概率
$$
g_{ij} = \frac{e^{z_{ij}}}{\sum_{k=1}^N e^{z_{ik}}}.
$$
这里的 $g_{ij}$ 可以理解为“第 $i$ 个 token 被送去第 $j$ 个专家的倾向强度”。

如果采用 top-1 路由，则第 $i$ 个 token 的实际专家编号为
$$
a_i = \arg\max_{j} g_{ij}.
$$
如果采用 top-k 路由，则会选出概率最大的 $k$ 个专家，并按实现决定是否重新归一化权重。本文先以 top-1 为主，因为它更容易把“软倾向”和“硬结果”的区别看清楚。

问题在于，主任务损失只关心最终预测对不对，并不天然关心专家是否均衡使用。于是训练会出现一个常见路径：少数专家先学到更有用的模式，router 就更爱把 token 送给它们；这些专家因为拿到更多 token，又更新得更快；其他专家越来越少被选中，几乎拿不到梯度。这个正反馈会把系统推向“少数专家工作，多数专家围观”。

可以把边界说得更具体：

| 现象 | 直接后果 | 为什么主任务损失不够 |
| --- | --- | --- |
| 少数专家占据大部分 token | 吞吐受限，容量溢出，部分 token 被丢弃或降级 | 主任务只看预测误差，不看专家利用率 |
| 大量专家长期闲置 | 参数浪费，训练不充分 | 未被选中的专家几乎没有梯度 |
| logits 绝对值越来越大 | softmax 饱和，数值不稳定 | 主任务会奖励“非常确定”的路由，直到过头 |

新手版玩具例子：4 个专家、100 个请求，如果 100 个都被送到专家 1，那么专家 2、3、4 基本没机会学东西。这不是“模型自动学会了分工”，而是“系统提前塌缩成单专家偏置”。

再把“为什么会塌缩”写成最短因果链：

$$
\text{更多 token} \Rightarrow \text{更多梯度更新} \Rightarrow \text{专家更强} \Rightarrow \text{router 更偏爱} \Rightarrow \text{更多 token}
$$

这就是辅助损失存在的必要性。它们不是在替代主任务，而是在打断这个正反馈。

这里还要说明边界。本文讨论的是稀疏 MoE 中最常见的 router 训练设置：softmax gating、top-1 或 top-k 选择、辅助损失与主任务联合优化。对于完全不同的路由方式，比如树状路由、检索式路由、强化学习式离散决策，公式形式可能不同，但“均衡负载”和“数值稳定”仍然是同一类问题。

还要补一个常被忽略的术语表，便于新手对照：

| 术语 | 直译 | 在本文里的含义 |
| --- | --- | --- |
| router | 路由器 | 产生每个专家分数的小网络 |
| logits | 未归一化分数 | softmax 之前的原始输出 |
| gating probabilities | 路由概率 | softmax 后的 $g_{ij}$ |
| top-1 / top-k | 选 1 个 / 选 k 个 | 每个 token 最终激活多少专家 |
| load balance | 负载均衡 | 不让少数专家长期过热 |

---

## 核心机制与推导

先定义两个最常用的统计量。

平均路由概率：
$$
\hat p_j = \frac{1}{T}\sum_{i=1}^T g_{ij}
$$
它表示“从软概率角度看，第 $j$ 个专家平均分到了多少注意力”。

实际负载：
$$
f_j = \frac{1}{T}\sum_{i=1}^T \mathbf{1}\{\operatorname{argmax}(g_i)=j\}
$$
其中 $\mathbf{1}$ 是指示函数，意思是“如果第 $i$ 个 token 最终选中了专家 $j$，就记 1，否则记 0”。$f_j$ 表示“从硬路由结果看，第 $j$ 个专家真的接到了多少 token”。

两者的区别必须单独强调：

| 量 | 看的是 | 连续/离散 | 典型用途 |
| --- | --- | --- | --- |
| $\hat p_j$ | 倾向 | 连续 | 判断 router 是否长期偏爱某些专家 |
| $f_j$ | 结果 | 离散 | 判断实际拥堵是否已经发生 |

### 1. Importance Loss：约束软概率分布

一种常见写法是最小化 $\hat p$ 的平方和，或者等价地最小化变异系数 $CV^2$。直觉是一样的：如果所有专家平均概率都接近 $\frac{1}{N}$，那么分布更均匀。

写成简单形式：
$$
L_{\text{imp}} = \sum_{j=1}^N \hat p_j^2
$$

因为 $\sum_j \hat p_j = 1$，这个量在均匀分布时最小。可以直接用拉格朗日乘子验证：
$$
\mathcal{L}(\hat p,\lambda)=\sum_{j=1}^N \hat p_j^2 + \lambda\left(\sum_{j=1}^N \hat p_j - 1\right)
$$
对每个 $\hat p_j$ 求导得
$$
2\hat p_j + \lambda = 0.
$$
所有 $\hat p_j$ 必须相等，再结合总和为 1，得到
$$
\hat p_j = \frac{1}{N}, \qquad
L_{\text{imp,min}} = \sum_{j=1}^N \left(\frac{1}{N}\right)^2 = \frac{1}{N}.
$$

它的梯度方向很容易看：
$$
\frac{\partial L_{\text{imp}}}{\partial \hat p_j} = 2\hat p_j
$$
哪一项 $\hat p_j$ 大，哪一项就会收到更强的下降压力。再通过 softmax 链式传导回 logits，router 会被鼓励降低热门专家的平均概率，提高冷门专家的平均概率。

如果把梯度继续展开到 $g_{ij}$，还有一个直观结论：
$$
\frac{\partial L_{\text{imp}}}{\partial g_{ij}}
=
\frac{\partial L_{\text{imp}}}{\partial \hat p_j}
\cdot
\frac{\partial \hat p_j}{\partial g_{ij}}
=
2\hat p_j \cdot \frac{1}{T}
=
\frac{2\hat p_j}{T}.
$$
也就是说，一个专家越热门，所有流向它的 token 概率都会收到更大“降温”信号。

但它有一个局限：它看的是“软倾向”，不一定等于“真实分配”。如果某些 token 的 top-1 选择长期固定，而非选中专家只分到一些很小的概率尾巴，Importance Loss 看到的偏差可能没有实际负载偏差那么严重。

下面给一个极简例子。假设只有两个 token、四个专家：

$$
g_1 = [0.51, 0.49, 0, 0], \qquad
g_2 = [0.51, 0.49, 0, 0].
$$

此时
$$
\hat p = [0.51, 0.49, 0, 0]
$$
看起来专家 1 和专家 2 很接近；但 top-1 下两个 token 都会被分给专家 1，于是
$$
f = [1,0,0,0].
$$
这就是“软概率看起来还行，硬负载已经塌了”的最小例子。

### 2. Load Loss：约束硬分配结果

Load Loss 把真实负载 $f_j$ 引入进来。一个常见写法是
$$
L_{\text{load}} = \alpha N \sum_{j=1}^N f_j \hat p_j
$$
其中 $\alpha$ 是权重系数，$N$ 是专家数，用来把量级调到合适范围。

先解释这个形式为什么有意义。若某个专家同时满足两件事：

1. 实际上已经接了很多 token，即 $f_j$ 大；
2. router 平均上仍给它很高概率，即 $\hat p_j$ 也大；

那么乘积 $f_j\hat p_j$ 就会很大，这表示“它既拥堵，又继续被偏爱”，应当被重点压制。

这项损失的关键点不是形式本身，而是梯度行为。通常实现里 $f_j$ 来自离散 top-1 或 top-k 选择，不对它反向传播，真正有梯度的是 $\hat p_j$。于是：
$$
\frac{\partial L_{\text{load}}}{\partial \hat p_j} = \alpha N f_j
$$
这意味着谁当前真的接了更多 token，谁对应的梯度权重就更大。它不像 Importance Loss 那样只看“平均概率偏没偏”，而是直接根据“实际拥堵程度”给热门专家更强惩罚。

继续看前面的玩具例子。4 个专家、100 个 token，真实负载比例
$$
f = [0.6,0.3,0.1,0.0]
$$
平均概率
$$
\hat p = [0.5,0.3,0.15,0.05]
$$
则
$$
L_{\text{load}}
= \alpha \cdot 4 \cdot (0.6\times0.5 + 0.3\times0.3 + 0.1\times0.15 + 0\times0.05)
= 1.62\alpha
$$
如果恢复成完全均匀：
$$
f = \hat p = [0.25,0.25,0.25,0.25]
$$
那么
$$
L_{\text{load}} = \alpha \cdot 4 \cdot 4 \times 0.25^2 = \alpha
$$
损失显著下降。

还可以把这个公式再读成一句工程语言：

| 情况 | $f_j$ | $\hat p_j$ | 结果 |
| --- | --- | --- | --- |
| 已拥堵且仍被偏爱 | 大 | 大 | 惩罚最大 |
| 暂时拥堵但概率已回落 | 大 | 小 | 惩罚中等 |
| 暂时空闲但概率在上升 | 小 | 大 | 惩罚较小 |
| 空闲且概率也低 | 小 | 小 | 惩罚最小 |

因此，Load Loss 比 Importance Loss 更像“拥堵控制器”，不是单纯的“分布整形器”。

### 3. z-loss：约束 logits 的绝对幅度

z-loss 出发点不同。它不直接看专家是否均衡，而是看每个 token 的 logits 是否越来越大。定义为
$$
L_z = \frac{1}{T}\sum_{i=1}^T \left(\log\sum_{j=1}^N e^{z_{ij}}\right)^2
$$
括号里的 $\log\sum e^z$ 叫 LogSumExp，简称 LSE，可以理解为“softmax 归一化常数的对数形式”。

先看两个性质。

第一，LSE 会随着 logits 整体平移而线性增长。若对某个 token 的所有专家都加上常数 $c$，则
$$
\operatorname{LSE}(z_i + c\mathbf{1}) = c + \operatorname{LSE}(z_i).
$$
因此 z-loss 会直接惩罚“整体分数越打越高”的现象。

第二，LSE 与最大 logit 紧密相关：
$$
\max_j z_{ij} \le \operatorname{LSE}(z_i) \le \max_j z_{ij} + \log N.
$$
所以它虽然不是直接取最大值，但可以稳定地跟踪“这一行 logits 是否已经很大”。

它为什么有效？因为对第 $i$ 个 token 的第 $j$ 个 logit 求导：
$$
\frac{\partial L_z}{\partial z_{ij}}
= \frac{2}{T}\,\operatorname{LSE}(z_i)\cdot \operatorname{softmax}(z_i)_j
$$
这里能看到两个关键信号：

1. 只要某个 token 的整体 logits 变大，$\operatorname{LSE}(z_i)$ 就变大，惩罚整体增强。
2. 某个专家的 softmax 概率越大，它收到的阻尼也越大。

所以 z-loss 的效果不是“强行均匀”，而是“别把分数打得太离谱”。这能缓解 softmax 饱和，因为 softmax 一旦过热，最大 logit 对应概率接近 1，其它接近 0，梯度会非常尖锐，数值也更容易不稳定。

再给一个数值感受。设某个 token 的 logits 为：

$$
z^{(a)} = [2,1,0,0], \qquad
z^{(b)} = [20,10,0,0].
$$

则二者 softmax 都会偏向第 1 个专家，但第二组更极端。对应的 LSE 近似为
$$
\operatorname{LSE}(z^{(a)}) \approx 2.49,\qquad
\operatorname{LSE}(z^{(b)}) \approx 20.00.
$$
平方后差距更大，因此 z-loss 会明显更强地压制第二种过热状态。

### 4. 三类梯度流向的区别

| 损失项 | 主要看什么 | 梯度更偏向哪里 |
| --- | --- | --- |
| Importance Loss | 平均软概率 $\hat p_j$ | 概率长期偏高的专家 |
| Load Loss | 实际负载 $f_j$ 与平均概率 $\hat p_j$ | 真实拥堵的热门专家 |
| z-loss | 每个 token 的 logits 幅度 | logits 过大的 token 和其高概率专家 |

一句话总结推导结果：Importance Loss 让“概率分布别歪”，Load Loss 让“拥堵专家赶紧降温”，z-loss 让“所有 logits 都别飙太高”。

如果把三者放在同一张表里，新手最容易建立整体图景：

| 问题层级 | 问题是什么 | 对应损失 |
| --- | --- | --- |
| 分布层 | router 长期偏爱少数专家 | Importance Loss |
| 调度层 | 实际 token 计数已经失衡 | Load Loss |
| 数值层 | logits 太大，softmax 太硬 | z-loss |

真实工程例子是大规模 ST-MoE 训练。模型参数上百亿、专家数很多、batch 很大时，router 的小偏差会被放大成系统性失衡。此时只靠均衡负载还不够，因为高 logits 会让路由变得极端，最终拖垮稳定性。于是工程上常把负载平衡项和 z-loss 一起加，前者稳分工，后者稳数值。

---

## 代码实现

下面用一个可运行的 Python 例子，把三个量都算出来。这个实现不是完整训练代码，但逻辑和真实训练里的 batch tensor 计算是一致的。代码包含：

1. `numpy` 版前向计算，便于看清公式；
2. `torch` 版自动求梯度，便于确认梯度方向确实存在；
3. 两组对照实验：负载失衡对照、logits 放大对照。

### 1. NumPy 版：计算三个辅助项

```python
import numpy as np

def softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def logsumexp(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)

def cv_squared(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    var = np.mean((x - mean) ** 2)
    return var / (mean ** 2 + eps)

def moe_aux_losses_numpy(logits, alpha=0.1, z_weight=0.001):
    """
    logits: [T, N]
    alpha: load loss 系数
    z_weight: z-loss 权重
    """
    logits = np.asarray(logits, dtype=np.float64)
    assert logits.ndim == 2, "logits must be a 2D array of shape [T, N]"

    probs = softmax(logits, axis=1)                # g_ij
    T, N = probs.shape

    p_hat = probs.mean(axis=0)                     # \hat p_j
    hard_assign = np.argmax(probs, axis=1)
    counts = np.bincount(hard_assign, minlength=N)
    f = counts / T                                 # f_j

    importance_loss_l2 = np.sum(p_hat ** 2)
    importance_loss_cv2 = cv_squared(p_hat)
    load_loss = alpha * N * np.sum(f * p_hat)

    lse = logsumexp(logits, axis=1)
    z_loss = np.mean(lse ** 2)

    total_aux = importance_loss_l2 + load_loss + z_weight * z_loss
    return {
        "probs": probs,
        "hard_assign": hard_assign,
        "counts": counts,
        "p_hat": p_hat,
        "f": f,
        "importance_loss_l2": importance_loss_l2,
        "importance_loss_cv2": importance_loss_cv2,
        "load_loss": load_loss,
        "z_loss": z_loss,
        "total_aux": total_aux,
    }

def print_report(name, out):
    print(f"=== {name} ===")
    print("counts               :", out["counts"])
    print("p_hat                :", np.round(out["p_hat"], 4))
    print("f                    :", np.round(out["f"], 4))
    print("importance_loss_l2   :", round(out["importance_loss_l2"], 6))
    print("importance_loss_cv2  :", round(out["importance_loss_cv2"], 6))
    print("load_loss            :", round(out["load_loss"], 6))
    print("z_loss               :", round(out["z_loss"], 6))
    print("total_aux            :", round(out["total_aux"], 6))
    print()

# 例子 A：明显偏向前两个专家
logits_a = np.array([
    [4.0, 1.0, 0.5, -1.0],
    [3.5, 1.2, 0.3, -0.5],
    [3.2, 1.0, 0.1, -0.7],
    [2.8, 1.5, 0.2, -0.3],
    [1.0, 2.5, 0.4, -0.2],
    [0.8, 2.7, 0.6, -0.4],
    [0.5, 2.4, 1.2, -0.8],
    [2.9, 1.4, 0.0, -0.6],
], dtype=np.float64)

# 例子 B：更接近均衡
logits_b = np.array([
    [2.0, 1.8, 1.7, 1.6],
    [1.9, 2.1, 1.8, 1.7],
    [1.7, 1.8, 2.2, 1.9],
    [1.8, 1.7, 1.9, 2.1],
    [2.1, 1.9, 1.8, 1.7],
    [1.8, 2.2, 1.7, 1.9],
    [1.7, 1.8, 2.1, 1.9],
    [1.9, 1.7, 1.8, 2.2],
], dtype=np.float64)

out_a = moe_aux_losses_numpy(logits_a, alpha=0.1, z_weight=0.001)
out_b = moe_aux_losses_numpy(logits_b, alpha=0.1, z_weight=0.001)
out_a_scaled = moe_aux_losses_numpy(logits_a * 3.0, alpha=0.1, z_weight=0.001)

assert out_a["probs"].shape == (8, 4)
assert np.allclose(out_a["probs"].sum(axis=1), 1.0)
assert np.isclose(out_a["p_hat"].sum(), 1.0)
assert np.isclose(out_a["f"].sum(), 1.0)
assert out_a["importance_loss_l2"] >= 0
assert out_a["load_loss"] >= 0
assert out_a["z_loss"] >= 0

assert out_b["importance_loss_l2"] < out_a["importance_loss_l2"]
assert out_b["load_loss"] < out_a["load_loss"]
assert out_a_scaled["z_loss"] > out_a["z_loss"]

print_report("imbalanced logits", out_a)
print_report("more balanced logits", out_b)
print_report("imbalanced logits x3", out_a_scaled)
```

这段代码可以直接运行。你会观察到两个稳定现象：

| 对照 | 会发生什么 | 原因 |
| --- | --- | --- |
| `logits_b` 对比 `logits_a` | `importance_loss` 和 `load_loss` 更小 | 分布更均衡 |
| `logits_a * 3.0` 对比 `logits_a` | `z_loss` 明显增大 | logits 幅度整体变大 |

### 2. PyTorch 版：确认梯度确实能回传

上面的 NumPy 版只能算值，不能看自动求导。下面给出一个最小 PyTorch 版本，用来确认辅助项会对 logits 产生梯度。

```python
import torch

def moe_aux_losses_torch(logits, alpha=0.1, z_weight=0.001):
    """
    logits: [T, N], requires_grad=True
    """
    probs = torch.softmax(logits, dim=1)
    T, N = probs.shape

    p_hat = probs.mean(dim=0)
    hard_assign = probs.argmax(dim=1)
    counts = torch.bincount(hard_assign, minlength=N).to(logits.dtype)
    f = counts / T

    importance_loss = torch.sum(p_hat ** 2)
    load_loss = alpha * N * torch.sum(f.detach() * p_hat)

    lse = torch.logsumexp(logits, dim=1)
    z_loss = torch.mean(lse ** 2)

    total_aux = importance_loss + load_loss + z_weight * z_loss
    return total_aux, {
        "p_hat": p_hat.detach(),
        "f": f.detach(),
        "importance_loss": importance_loss.detach(),
        "load_loss": load_loss.detach(),
        "z_loss": z_loss.detach(),
    }

logits = torch.tensor([
    [4.0, 1.0, 0.5, -1.0],
    [3.5, 1.2, 0.3, -0.5],
    [3.2, 1.0, 0.1, -0.7],
    [2.8, 1.5, 0.2, -0.3],
    [1.0, 2.5, 0.4, -0.2],
    [0.8, 2.7, 0.6, -0.4],
    [0.5, 2.4, 1.2, -0.8],
    [2.9, 1.4, 0.0, -0.6],
], dtype=torch.float64, requires_grad=True)

loss, info = moe_aux_losses_torch(logits, alpha=0.1, z_weight=0.001)
loss.backward()

assert logits.grad is not None
assert logits.grad.shape == logits.shape
assert torch.isfinite(logits.grad).all()

print("aux loss:", float(loss))
print("p_hat   :", info["p_hat"].numpy())
print("f       :", info["f"].numpy())
print("grad L2 :", float(torch.norm(logits.grad)))
```

这里有一个实现细节必须明确：`f` 是离散分配统计，通常不参与反向传播，所以例子里用了 `f.detach()`。这与前面推导的“真正有梯度的是 $\hat p_j$”一致。

### 3. 代码和公式如何一一对应

| 公式符号 | 代码变量 | 含义 |
| --- | --- | --- |
| $z_{ij}$ | `logits[i, j]` | router 原始分数 |
| $g_{ij}$ | `probs[i, j]` | softmax 后路由概率 |
| $\hat p_j$ | `p_hat[j]` | 平均路由概率 |
| $f_j$ | `f[j]` | 实际负载比例 |
| $\operatorname{LSE}(z_i)$ | `logsumexp(logits, axis=1)[i]` 或 `torch.logsumexp` | 每个 token 的 LogSumExp |
| $L_{\text{imp}}$ | `importance_loss` | 软分布均衡项 |
| $L_{\text{load}}$ | `load_loss` | 硬负载均衡项 |
| $L_z$ | `z_loss` | logits 温度约束项 |

如果换成真实训练代码，结构几乎不变，只是会再包上一层主任务损失，例如交叉熵：

$$
L = L_{\text{CE}} + c_{\text{imp}}L_{\text{imp}} + c_{\text{load}}L_{\text{load}} + c_zL_z
$$

### 4. 真实工程中还会再加什么

| 工程步骤 | 作用 |
| --- | --- |
| capacity 限制 | 每个专家最多接收一定数量 token，防止单专家爆满 |
| top-k 路由 | 每个 token 发送到多个专家，提升表达能力，但实现更复杂 |
| token dropping / rerouting | 专家满载时丢弃或改派 token |
| router 指标监控 | 记录 $\hat p_j$、$f_j$、overflow、LSE 统计 |

有些实现会把前两项合并成一个 balance loss，再统一乘一个系数。数值上没有唯一标准，重点是让辅助项足够影响 router，又不能压过主任务。

---

## 工程权衡与常见坑

最常见的误区，是把“有辅助损失”误认为“负载就一定均衡”。实际上，loss 的存在不等于优化一定有效，router 仍可能找到绕开的办法。

### 1. Sigmoid gating 或极小概率逃逸

有些路由实现不是标准 softmax，而是 sigmoid 或其它独立打分方式。此时 router 可能把很多概率压到极小，比如 $10^{-20}$ 甚至更小。这样做会带来一个问题：某些依赖概率乘积或归一化的负载损失会被“压扁”，看起来 loss 很小，但实际路由已经失衡。

新手版理解：系统表面上说“我每个专家都给了一点点概率”，实际上那点概率小到近乎不存在，辅助项几乎失效。

可观察的症状通常是：

| 症状 | 监控上看到什么 |
| --- | --- |
| 少数专家持续被选中 | `counts` 长期偏斜 |
| 平均概率看似不极端 | $\hat p_j$ 没有想象中偏 |
| loss 下降但路由没改善 | 辅助损失数值不高 |

### 2. z-loss 权重过大或过小

z-loss 太小，softmax 仍会过热；z-loss 太大，router 会被迫把 logits 都压扁，导致路由区分度不足，主任务学习变慢。它不是越大越稳，而是存在一个“够用但不过度”的区间。

一个常见判断办法是同时看两组统计：

| 指标 | 过小 z-loss | 过大 z-loss |
| --- | --- | --- |
| LogSumExp 均值/最大值 | 持续上升 | 过低且变化小 |
| 路由熵 | 过低，接近 one-hot | 过高，专家区分不足 |
| 主任务 loss | 可能后期波动或 NaN | 前期下降变慢 |

### 3. 只看 $\hat p$，不看 $f$

如果只记录平均概率，可能误以为路由已经接近均衡。但 top-1 选择是离散的，真实分配可能仍严重偏斜。工程监控时应同时看软统计和硬统计。

前面已经给过一个最小反例，这里再用一句话重述：**“概率接近”不等于“argmax 结果接近”。**

### 4. capacity overflow 被忽略

即使 balance loss 看起来不错，也可能出现单步 batch 中个别专家瞬时爆满。若没有容量限制，吞吐和显存都会受影响；若有容量限制但未监控，则 token 可能被丢弃或走降级路径，影响任务质量。

这里要补一个常见公式。若每个专家容量设为
$$
C = \left\lceil \frac{\text{capacity factor} \times T}{N} \right\rceil,
$$
那么任一专家在单步中可接收的 token 数都不会超过 $C$。capacity factor 越大，越不容易丢 token，但内存和计算也会增加。

### 5. 辅助损失量级失衡

另一个实际问题是三项损失的数值尺度可能不同。比如：

- `importance_loss` 可能天然在 $[1/N,1]$ 附近；
- `load_loss` 会受 $\alpha$ 和 $N$ 影响；
- `z_loss` 因为是平方项，数值可能更大。

如果不做权重校准，就可能出现“某个辅助项名义上存在，实际上不起作用”或“某项过强，主任务被压住”。

下面把常见坑和规避方式放在一起：

| 常见坑 | 现象 | 规避措施 |
| --- | --- | --- |
| 概率被压到极小值 | load loss 几乎不工作 | 调整归一化方式、检查 gating 设计、重新校准 balance 项 |
| z-loss 太弱 | logits 越训越大，softmax 饱和 | 监控 LogSumExp 均值和最大值，逐步上调 $c_z$ |
| z-loss 太强 | 路由分不出专家，主任务下降 | 从小权重起步，观察路由熵和任务 loss |
| 只监控平均概率 | 误判为“已经均衡” | 同时监控 $\hat p_j$、$f_j$、每专家 token 数 |
| 忽略 capacity | 热门专家溢出，token 丢失 | 设置容量上限并记录 overflow 比例 |
| 辅助项尺度不匹配 | 调参无效、训练行为反常 | 单独记录各项绝对值和梯度范数 |

真实工程里，一个很实用的排查顺序是：先看每专家 token 计数，再看平均路由概率，再看 logits 或 LogSumExp 的统计分布。因为“分配失衡”和“数值过热”往往是连锁出现的，不应分开盲调。

---

## 替代方案与适用边界

Importance Loss、Load Loss、z-loss 不是唯一选择。还可以用熵约束、KL 散度等方式让路由分布更平滑。

熵约束，entropy regularization，意思是“鼓励概率分布不要过于尖锐”。例如最大化
$$
H(g_i) = -\sum_{j=1}^N g_{ij}\log g_{ij}
$$
它能让单个 token 的路由概率更分散。若写成加入总损失的正则项，通常是最小化
$$
L_{\text{ent}} = -\frac{1}{T}\sum_{i=1}^T H(g_i)
= \frac{1}{T}\sum_{i=1}^T \sum_{j=1}^N g_{ij}\log g_{ij}.
$$

KL 约束则是让平均路由分布靠近某个目标分布，比如均匀分布：
$$
D_{\text{KL}}(\hat p \,\|\, u), \quad u_j = \frac{1}{N}
$$
展开后就是
$$
D_{\text{KL}}(\hat p \,\|\, u)
=
\sum_{j=1}^N \hat p_j \log \frac{\hat p_j}{1/N}
=
\sum_{j=1}^N \hat p_j \log (N\hat p_j).
$$
它和 Importance Loss 有相似目的，都是拉平均分布回到均衡状态。

还可以考虑下面这些变体：

| 方法 | 思路 | 与本文三项的关系 |
| --- | --- | --- |
| 对 $\hat p$ 做 KL 到均匀分布 | 明确指定目标分布 | 与 Importance Loss 同类，都是管软分布 |
| 对单 token 概率做熵正则 | 防止单步过尖 | 更接近“局部平滑”，但不直接看负载 |
| 对每专家计数做方差惩罚 | 直接拉平 token 数 | 与 Load Loss 同类，但形式不同 |
| 温度调节（softmax temperature） | 改变概率尖锐度 | 可缓解过热，但不等于 z-loss |

但这些替代方案和 z-loss 的职责不同。熵和 KL 更多是在“分布形状”层面施压，不能直接替代 z-loss 对 logits 幅度的约束。也就是说，哪怕平均分布看起来合理，logits 仍可能整体过大，softmax 仍可能数值不稳。

| 方法 | 主要优点 | 主要缺点 | 适用边界 |
| --- | --- | --- | --- |
| Importance Loss | 简单直接，易实现 | 只看软概率，不直接看真实拥堵 | 中小规模 MoE，作为基础平衡项 |
| Load Loss | 更贴近真实负载 | 依赖离散路由统计，调参更敏感 | top-1/top-k 稀疏路由的主力方案 |
| z-loss | 专门抑制 logits 过热 | 不解决负载均衡本身 | 大模型、长训练、高风险数值场景 |
| 熵约束 | 让单 token 路由更平滑 | 容易削弱稀疏性 | 早期训练稳定化、小模型 |
| KL 到均匀分布 | 目标清晰，可解释 | 仍主要约束平均分布 | 专家数较少、希望明确均衡目标时 |

适用边界可以这样记：

1. 小模型、少量专家、训练相对稳定时，Importance Loss 或 KL/entropy 往往就够用。
2. 一旦进入明显稀疏、大 batch、超大参数规模场景，Load Loss 几乎是刚需，因为它更关心真实拥堵。
3. 当发现 logits 统计值持续升高、softmax 接近 one-hot、出现 NaN 或梯度异常时，z-loss 往往不是可选优化，而是稳定训练的必要组件。

还可以再补一句判断标准：如果你的问题是“专家没分开”，先看 Importance / KL；如果你的问题是“几个专家全挤爆了”，先看 Load；如果你的问题是“训练开始发散或 router 过热”，先看 z-loss。

---

## 参考资料

下面把参考资料从“能看什么”改成“为什么值得看”。

| 资料 | 主要贡献 | 可复现内容 |
| --- | --- | --- |
| Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* | 早期系统化提出稀疏 MoE 与负载均衡辅助项 | 可对照 Importance / Load 类平衡思路 |
| Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* | 给出 top-1 路由、大规模训练下的负载平衡实践 | 可复现最小路由与专家计数统计 |
| Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models* | 讨论大规模稀疏训练稳定性，并引入 router z-loss | 可对照总损失形式与 z-loss 经验权重 |
| GShard / Megatron-LM / DeepSpeed-MoE 相关实现与文档 | 展示 capacity、top-k、overflow、监控指标等工程细节 | 可对照实际训练代码与路由监控 |
| Emergent Mind 等综述型资料 | 汇总不同 balance loss 公式及直觉 | 适合快速核对概念与术语 |

建议阅读顺序是：先看负载均衡公式，再看 ST-MoE 对 z-loss 的设计动机，最后结合具体实现观察监控指标。这样更容易把“分布均衡”和“数值稳定”这两类问题分开理解。

如果需要一个最短阅读路径，可以按下面顺序：

1. 先读 Switch Transformer，理解 top-1 路由为什么会产生负载失衡。
2. 再读 ST-MoE，理解为什么“负载均衡了”仍不够，还需要 z-loss 稳定数值。
3. 最后看具体框架实现，重点对照 `counts`、`p_hat`、overflow 和 LogSumExp 监控。

最后把全文压缩成一句判断：**Importance Loss 解决“router 长期偏爱谁”，Load Loss 解决“实际上谁已经被挤爆”，z-loss 解决“router 打分是否已经过热到威胁训练稳定性”。**
