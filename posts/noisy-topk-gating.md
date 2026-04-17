## 核心结论

Noisy Top-k Gating 是稀疏 MoE 中最常见的一类路由函数。路由的任务很直接：对每个 token，只选少数几个专家参与计算，而不是让全部专家都执行一次前馈。

它的关键不在“随机选专家”，而在“训练时给路由分数加一层可控噪声”。这层噪声只影响训练阶段，用来打破分数接近时的长期僵局，让原本总是排在第 2、第 3 的专家也有机会进入 Top-k，拿到梯度并继续学习。标准形式可以写成：

$$
H(x)=xW_g+\epsilon \odot \operatorname{Softplus}(xW_{\text{noise}})
$$

其中：

| 符号 | 含义 |
|---|---|
| $x \in \mathbb{R}^{d}$ | 单个 token 的输入表示 |
| $W_g \in \mathbb{R}^{d \times E}$ | 干净路由打分矩阵 |
| $W_{\text{noise}} \in \mathbb{R}^{d \times E}$ | 噪声尺度矩阵 |
| $\epsilon \sim \mathcal N(0, I)$ | 独立高斯噪声 |
| $E$ | 专家总数 |
| $\odot$ | 逐元素乘法 |

然后只保留前 $k$ 个最大值：

$$
G(x)=\operatorname{Softmax}(\operatorname{KeepTopK}(H(x),k))
$$

`KeepTopK` 的含义是：只保留前 $k$ 个 logit，其余位置统一设成 $-\infty$。这样做之后，softmax 只会在入选专家之间归一化，未入选专家的权重严格为 0：

$$
\operatorname{KeepTopK}(h,k)_i=
\begin{cases}
h_i, & i\in \operatorname{TopK}(h,k) \\
-\infty, & \text{otherwise}
\end{cases}
$$

对白话理解：如果有几个“略微领先”的专家从训练一开始就总被选中，它们会持续看到更多样本，学得更快，排名也更稳，最后形成自增强循环。Noisy Top-k 的作用是，在训练时给这些分数加一点输入相关的扰动，让“差一点就入选”的专家偶尔也能上场；到了推理阶段，再把扰动关闭，保证行为稳定。

它的收益可以概括为：

| 目标 | 没有噪声时 | 有 Noisy Top-k 时 |
|---|---|---|
| 探索不同专家 | 容易早期固化 | 分数接近时可交换排名 |
| 训练覆盖率 | 部分专家长期闲置 | 更多专家能拿到梯度 |
| 推理稳定性 | 稳定 | 训练期探索，推理期同样稳定 |
| 计算量 | 由 Top-k 控制 | 同样由 Top-k 控制 |

结论可以压缩成一句话：Noisy Top-k 用训练期噪声换取更好的专家探索，但噪声必须受控，只能大到足以打破局部僵局，不能大到把路由变成近似随机。

---

## 问题定义与边界

MoE，Mixture of Experts，意思是“专家混合”。它把一个大层拆成多个子网络，每个子网络叫一个专家。对某个 token 来说，并不是所有专家都要参与；路由器会先做一次打分，再决定这个 token 送去哪些专家。

问题的根源是路由偏置会自我强化。

假设一个 batch 里，路由器在训练初期就偏爱某几个专家。那这些专家会：

1. 接收到更多 token。
2. 获得更多梯度更新。
3. 更快适应训练分布。
4. 在后续路由中继续维持高分。

这会形成明显的正反馈。最终结果不是“模型有很多专家，所以表达力更强”，而是“少数专家在工作，多数专家挂名存在”。

设总专家数为 $E$，每个 token 只激活 $k$ 个专家，且 $k \ll E$。那么路由问题可以分解成四步：

1. 给每个 token 计算 $E$ 个专家分数。
2. 只保留前 $k$ 个专家。
3. 对这 $k$ 个专家的权重做归一化。
4. 避免少数专家吞掉几乎全部流量。

这个问题本质上同时包含两类目标：

| 目标类型 | 具体要求 |
|---|---|
| 单 token 目标 | 选出当前最合适的少数专家 |
| 全局训练目标 | 不让专家使用率长期失衡 |

一个更具体的玩具例子：

- 有 4 个专家：`E0 E1 E2 E3`
- 每个 token 只能选 2 个专家
- 如果路由器几乎总把分数排成 `E0 > E1 > E2 > E3`
- 那么 `E2` 和 `E3` 会长期拿不到样本
- 长期拿不到样本，就几乎没有有效训练
- 最终这 4 个专家在参数量上存在，在功能上却接近只有 2 个专家

Noisy Top-k 想解决的是训练期探索不足，不是推理期随机化。因此它的边界很明确：

| 阶段 | 是否加噪 | 是否做 Top-k | 是否常配合负载损失 |
|---|---|---|---|
| 训练 | 是 | 是 | 是，通常需要 |
| 推理 | 否 | 是 | 不再新增随机探索 |

这里有两个容易混淆的边界。

第一，Noisy Top-k 不是为了让推理“更有随机性”。推理时通常直接关闭噪声，使用干净 logits 做 Top-k。

第二，Top-k 不是可选装饰，而是稀疏 MoE 的计算核心。若不做 Top-k，而是对全部专家做 dense softmax，那么：

$$
\text{单 token 计算成本} \propto E
$$

而稀疏 Top-k 的目标是把成本近似控制在：

$$
\text{单 token 计算成本} \propto k, \quad k \ll E
$$

所以 Noisy Top-k 的完整目标不是“加噪声”，而是“在固定稀疏计算预算下，提高训练阶段的探索能力”。

---

## 核心机制与推导

先看没有噪声时的基本门控：

$$
\text{clean}(x)=xW_g
$$

这一步只是一个线性映射。它把 token 表示投影到专家空间，得到每个专家的原始偏好分数。若输入维度为 $d$、专家数为 $E$，那么：

$$
x \in \mathbb{R}^{d}, \quad W_g \in \mathbb{R}^{d\times E}, \quad \text{clean}(x)\in \mathbb{R}^{E}
$$

可以把它理解为：路由器先给每个专家打一分，分越高，越说明这个 token 适合该专家处理。

接着加入噪声项：

$$
\text{noise}(x)=\epsilon \odot \operatorname{Softplus}(xW_{\text{noise}})
$$

其中：

$$
\operatorname{Softplus}(z)=\log(1+e^z)
$$

Softplus 在这里不是用来分类，而是用来生成正的噪声尺度。因为高斯噪声本身可正可负，真正需要被约束为正值的是“标准差”或“幅度”：

$$
\sigma(x)=\operatorname{Softplus}(xW_{\text{noise}}) > 0
$$

于是上式也可以改写成更容易理解的形式：

$$
\text{noise}(x)=\epsilon \odot \sigma(x), \qquad \epsilon \sim \mathcal N(0,I)
$$

总 logit 为：

$$
H(x)=\text{clean}(x)+\text{noise}(x)
$$

展开后就是：

$$
H(x)=xW_g+\epsilon \odot \operatorname{Softplus}(xW_{\text{noise}})
$$

然后做稀疏选择：

$$
\tilde H_i(x)=
\begin{cases}
H_i(x), & i \in \operatorname{TopK}(H(x),k) \\
-\infty, & \text{otherwise}
\end{cases}
$$

最后归一化：

$$
G_i(x)=\frac{e^{\tilde H_i(x)}}{\sum_{j=1}^{E}e^{\tilde H_j(x)}}
$$

因为未选中的位置是 $-\infty$，所以：

$$
e^{-\infty}=0
$$

这意味着只有入选的 $k$ 个专家会得到非零权重。于是：

$$
\sum_{i=1}^{E} G_i(x)=1, \qquad G_i(x)=0 \text{ for } i \notin \operatorname{TopK}
$$

### 一个可手算的玩具例子

下面用一个可以手算的例子把整个过程走完。设输入为：

$$
x=[1,2]
$$

令专家数为 2，取：

$$
W_g=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
W_{\text{noise}}=
\begin{bmatrix}
0.5&0.5\\
0.5&0.5
\end{bmatrix},
\qquad
\epsilon=[1,-1]
$$

先算干净分数：

$$
xW_g=[1,2]
$$

这表示如果完全不加噪声，专家 2 的分数更高。

再算噪声尺度：

$$
xW_{\text{noise}}=[1,2]
\begin{bmatrix}
0.5&0.5\\
0.5&0.5
\end{bmatrix}
=[1.5,1.5]
$$

然后计算：

$$
\operatorname{Softplus}(1.5)=\log(1+e^{1.5})\approx 1.70141328
$$

因此噪声尺度向量是：

$$
[1.70141328,1.70141328]
$$

再与噪声样本逐元素相乘：

$$
[1,-1]\odot[1.70141328,1.70141328]=[1.70141328,-1.70141328]
$$

于是加噪后的 logits 为：

$$
H(x)=[1,2]+[1.70141328,-1.70141328]=[2.70141328,0.29858672]
$$

可以看到，原本第 2 个专家占优；加噪之后，第 1 个专家反而翻到前面。这就是“噪声打破局部排序”的最小例子。

如果此时取 `k=1`，那么只保留最大值，结果是：

$$
\operatorname{KeepTopK}(H,1)=[2.70141328,-\infty]
$$

所以最终权重为：

$$
\operatorname{Softmax}([2.70141328,-\infty])=[1,0]
$$

如果取 `k=2`，则两个专家都保留，softmax 后：

$$
\operatorname{Softmax}([2.70141328,0.29858672]) \approx [0.91697, 0.08303]
$$

这个例子说明两件事：

1. 噪声先影响的是排序。
2. Top-k 再决定到底允许几个专家存活。

为了避免新手把两步混在一起，可以直接记住下面这张表：

| 步骤 | 作用 | 会改变什么 |
|---|---|---|
| 加噪声 | 打破接近分数的固定排序 | 哪些专家更可能进入前列 |
| Top-k | 强制稀疏 | 最终有几个专家参与计算 |
| Softmax | 对入选专家分配相对权重 | 入选专家之间的权重比例 |

### 流程图

```text
token 向量 x
   ->
线性打分 xW_g
   ->
计算噪声尺度 Softplus(xW_noise)
   ->
采样高斯噪声 epsilon
   ->
逐元素相乘，得到 noise(x)
   ->
H(x) = clean(x) + noise(x)
   ->
KeepTopK(H, k)
   ->
Softmax
   ->
选中的 k 个专家及其权重
```

### 为什么它能鼓励探索

把问题压缩成两个专家最容易看清楚。

设两个专家的干净分数分别为 $s_1,s_2$，且：

$$
s_1 > s_2, \qquad \Delta=s_1-s_2>0
$$

再设它们的噪声分别为 $\eta_1,\eta_2$。加噪后，专家 2 超过专家 1 的条件是：

$$
s_2+\eta_2 > s_1+\eta_1
$$

移项后得到：

$$
\eta_2-\eta_1 > \Delta
$$

因此翻盘概率是：

$$
P(H_2>H_1)=P(\eta_2-\eta_1>\Delta)
$$

如果把噪声近似看成独立高斯：

$$
\eta_1 \sim \mathcal N(0,\sigma_1^2), \qquad \eta_2 \sim \mathcal N(0,\sigma_2^2)
$$

那么差值仍然服从高斯分布：

$$
\eta_2-\eta_1 \sim \mathcal N(0,\sigma_1^2+\sigma_2^2)
$$

于是可写成：

$$
P(H_2>H_1)=1-\Phi\left(\frac{\Delta}{\sqrt{\sigma_1^2+\sigma_2^2}}\right)
$$

其中 $\Phi$ 是标准正态分布的累积分布函数。

这个式子直接说明了探索强度的来源：

- $\Delta$ 越大，翻盘越难。
- $\sigma$ 越大，翻盘越容易。
- 当 $\sigma \ll \Delta$ 时，排名几乎固定。
- 当 $\sigma \gg \Delta$ 时，排名会变得过于随机。

所以 Noisy Top-k 的工程核心不是“有没有噪声”，而是“噪声尺度与分数差值的相对量级是否合适”。

---

## 代码实现

下面给出一个可直接运行的极简实现，只依赖 `numpy`。它重点展示四件事：

1. 训练和推理必须分开。
2. `softplus` 负责输出正的噪声尺度。
3. 非 Top-k 位置必须设成 `-np.inf`，否则得不到真正稀疏的 softmax。
4. 代码里要明确区分“干净 logits”和“加噪 logits”，否则调试时很难判断问题出在哪一步。

```python
import numpy as np


def softplus(x):
    # 数值稳定版 softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softmax(x):
    # 支持含 -inf 的输入
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    shifted = x - x_max
    exp_x = np.exp(shifted)

    # 对 -inf 位置，exp(-inf) = 0，会自动得到真正的稀疏权重
    total = np.sum(exp_x)
    return exp_x / total


def keep_topk(logits, k):
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, {n}], got {k}")

    topk_idx = np.argpartition(logits, -k)[-k:]
    masked = np.full_like(logits, -np.inf, dtype=np.float64)
    masked[topk_idx] = logits[topk_idx]
    return masked, topk_idx


def noisy_topk_gate(x, Wg, Wnoise, k, training=True, eps=None):
    x = np.asarray(x, dtype=np.float64)
    Wg = np.asarray(Wg, dtype=np.float64)
    Wnoise = np.asarray(Wnoise, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("This demo expects a single token vector with shape [d].")
    if Wg.ndim != 2 or Wnoise.ndim != 2:
        raise ValueError("Wg and Wnoise must be 2D matrices.")
    if Wg.shape != Wnoise.shape:
        raise ValueError("Wg and Wnoise must have the same shape [d, E].")
    if x.shape[0] != Wg.shape[0]:
        raise ValueError("Input dimension does not match weight matrices.")

    clean_logits = x @ Wg

    if training:
        if eps is None:
            eps = np.random.randn(*clean_logits.shape)
        else:
            eps = np.asarray(eps, dtype=np.float64)
            if eps.shape != clean_logits.shape:
                raise ValueError("eps must have the same shape as logits.")

        noise_scale = softplus(x @ Wnoise) + 1e-9
        noisy_logits = clean_logits + eps * noise_scale
    else:
        noise_scale = np.zeros_like(clean_logits)
        noisy_logits = clean_logits.copy()

    masked_logits, topk_idx = keep_topk(noisy_logits, k)
    probs = softmax(masked_logits)

    return {
        "clean_logits": clean_logits,
        "noise_scale": noise_scale,
        "noisy_logits": noisy_logits,
        "topk_idx": np.sort(topk_idx),
        "masked_logits": masked_logits,
        "probs": probs,
    }


def main():
    x = np.array([1.0, 2.0])
    Wg = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    Wnoise = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    eps = np.array([1.0, -1.0])

    out_train_k2 = noisy_topk_gate(x, Wg, Wnoise, k=2, training=True, eps=eps)
    out_train_k1 = noisy_topk_gate(x, Wg, Wnoise, k=1, training=True, eps=eps)
    out_infer_k1 = noisy_topk_gate(x, Wg, Wnoise, k=1, training=False)

    assert np.allclose(out_train_k2["clean_logits"], np.array([1.0, 2.0]))
    assert np.allclose(out_train_k2["noise_scale"], np.array([1.70141328, 1.70141328]), atol=1e-8)
    assert np.allclose(out_train_k2["noisy_logits"], np.array([2.70141328, 0.29858672]), atol=1e-8)
    assert np.allclose(out_train_k2["probs"], np.array([0.91696974, 0.08303026]), atol=1e-8)

    assert np.allclose(out_train_k1["probs"], np.array([1.0, 0.0]), atol=1e-8)
    assert np.allclose(out_infer_k1["clean_logits"], np.array([1.0, 2.0]), atol=1e-8)
    assert np.allclose(out_infer_k1["noisy_logits"], np.array([1.0, 2.0]), atol=1e-8)
    assert np.allclose(out_infer_k1["probs"], np.array([0.0, 1.0]), atol=1e-8)

    np.set_printoptions(precision=8, suppress=True)
    print("training, k=2")
    print("clean logits:", out_train_k2["clean_logits"])
    print("noise scale :", out_train_k2["noise_scale"])
    print("noisy logits:", out_train_k2["noisy_logits"])
    print("topk idx    :", out_train_k2["topk_idx"])
    print("probs       :", out_train_k2["probs"])
    print()

    print("training, k=1")
    print("probs       :", out_train_k1["probs"])
    print()

    print("inference, k=1")
    print("probs       :", out_infer_k1["probs"])


if __name__ == "__main__":
    main()
```

这段代码运行后，输出会接近：

```text
training, k=2
clean logits: [1. 2.]
noise scale : [1.70141328 1.70141328]
noisy logits: [2.70141328 0.29858672]
topk idx    : [0 1]
probs       : [0.91696974 0.08303026]

training, k=1
probs       : [1. 0.]

inference, k=1
probs       : [0. 1.]
```

如果你第一次接触 MoE，可以把这段代码按下面的方式理解：

| 变量 | 角色 |
|---|---|
| `clean_logits` | 不加噪声时的原始专家分数 |
| `noise_scale` | 每个专家噪声的幅度 |
| `noisy_logits` | 训练时真正用于排序的分数 |
| `topk_idx` | 被选中的专家下标 |
| `masked_logits` | 未入选位置被置为 `-inf` 的 logits |
| `probs` | 最终给入选专家的归一化权重 |

真实工程里，这段逻辑通常嵌在 Transformer 的 FFN 替换层中。流程大致是：

1. token 进入 router。
2. router 产生专家分数和 Top-k 选择。
3. token 被分发到选中的专家。
4. 专家计算结果按门控权重加权汇总。
5. 再返回主干网络。

因此，MoE 能做到“参数规模很大，但单 token 实际只计算少量专家”。这是它区别于普通 dense FFN 的核心价值。

---

## 工程权衡与常见坑

Noisy Top-k 最难的部分不是公式，而是噪声强度的工程控制。如果噪声尺度设置不当，整个路由系统会立刻表现出两种极端之一：

- 探索太弱，专家分布迅速固化。
- 探索太强，路由接近随机，训练难以收敛。

可以先看一个典型现象。

在多专家路由网络中，如果 `W_noise` 初始化过大，那么：

$$
\sigma(x)=\operatorname{Softplus}(xW_{\text{noise}})
$$

会从训练一开始就给出很高的噪声尺度。结果是同一个 token 在相邻 step 中频繁切换专家，专家负载剧烈波动，loss 曲线抖动明显，验证指标下降很慢。相反，如果噪声几乎为 0，那么训练会很快固定在少数热门专家上，后续很难再把冷门专家拉回来。

常见问题和对应缓解策略可以整理为下表：

| 问题 | 典型症状 | 原因 | 缓解策略 |
|---|---|---|---|
| 噪声太大 | 路由近似随机，loss 抖动大，收敛慢 | $\sigma(x)$ 远大于专家分数差 | 降低 `W_noise` 初始化；训练后期衰减噪声 |
| 噪声太小 | 少数专家长期热门，其他专家闲置 | 无法打破固定排序 | 增强初期探索；提高负载均衡损失权重 |
| 只加噪不做负载正则 | 局部能翻盘，但全局仍失衡 | 噪声只处理局部排序，不约束整体流量 | 加 importance/load balance loss |
| 推理忘记关噪声 | 同输入多次推理结果不同 | 训练逻辑误带入推理 | 推理阶段只用 clean logits |
| Top-k 实现错误 | 未选专家仍有极小概率 | 没有把非 Top-k 位置设成 `-inf` | 显式 mask 为 `-np.inf` |
| 容量约束太紧 | 热门专家 token 被丢弃 | 每个专家能接收的 token 上限过低 | 调整 capacity factor、batch 组织方式 |
| 专家过多但 batch 太小 | 专家利用率统计噪声大 | 单步样本不够覆盖全部专家 | 增大 batch、做跨设备聚合 |

### 为什么常配合负载均衡损失

Noisy Top-k 只能解决“分数接近时，次优专家有机会翻盘”的问题，但它不能单独保证全局专家流量均匀。

这可以用一句话概括：

- Noisy Top-k 处理的是局部探索。
- Load balance loss 处理的是全局分布。

常见做法是同时监控两类指标：

| 指标 | 观察目标 |
|---|---|
| 每个专家被选中的 token 数 | 看流量是否极端偏斜 |
| 每个专家获得的总门控权重 | 看重要性是否失衡 |
| 专家丢 token 比例 | 看容量是否过紧 |
| 路由熵 | 看路由是否过早塌缩 |

很多论文或工程实现里会加入 importance loss、load loss 或变体正则，其直觉都类似：不要让某几个专家长期吞掉大部分 batch。

### 一个常见误解

Noisy Top-k 的噪声不是像 dropout 那样“随机丢神经元”。

两者作用位置不同：

| 方法 | 作用位置 | 改变的对象 |
|---|---|---|
| Dropout | 隐层激活 | 哪些神经元参与内部计算 |
| Noisy Top-k | 路由分数 | 哪些专家被选中 |

所以它不是专家内部的正则化，而是专家选择机制上的探索控制。

### 一个实用的调参顺序

如果你需要从零搭一个可训练的 MoE 路由器，通常可以按下面顺序调：

1. 先确认无噪声版本的 Top-k 和 dispatch 逻辑正确。
2. 再加入很小的噪声，只观察路由统计是否变得更均匀。
3. 然后加入负载均衡损失，防止专家长期失衡。
4. 最后再调容量因子和 batch 规模。

这个顺序有意义，因为如果基础的 Top-k 或 dispatch 实现就错了，后面所有“探索”现象都不可信。

---

## 替代方案与适用边界

Noisy Top-k 不是唯一的探索方法。它只是当前稀疏 MoE 中最自然的一类方案，因为它同时满足三个条件：

1. 天然适合稀疏选择。
2. 训练阶段能引入探索。
3. 仍可嵌入连续优化流程。

常见替代方法至少有三类。

第一类是 $\varepsilon$-greedy。它来自多臂老虎机问题，规则很简单：

$$
\text{以概率 } 1-\varepsilon \text{ 选择当前最优动作，以概率 } \varepsilon \text{ 随机探索}
$$

它的优点是实现极简单，缺点也明显：探索是离散硬切换，不够平滑，和神经网络中的连续打分机制衔接得不自然。

第二类是 Gumbel-Top-k / Gumbel-Softmax。它通过加入 Gumbel 噪声近似离散采样，常用于“想保留采样语义，同时又希望保留一定可微性”的场景。

第三类是 dense softmax 加温度。它不做真正稀疏，而是调节分布尖锐度：

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}
$$

其中温度 $\tau$ 越小，分布越尖锐；$\tau$ 越大，分布越平滑。这个方法最稳定，但通常不满足稀疏 MoE 的计算预算目标。

对比如下：

| 方法 | 是否天然稀疏 | 是否便于反传 | 探索方式 | 主要问题 |
|---|---|---|---|---|
| Noisy Top-k | 是 | 是，工程上成熟 | 对 logits 加输入相关高斯噪声 | 对噪声尺度敏感 |
| $\varepsilon$-greedy | 是，但实现偏硬 | 较差 | 直接随机替换动作 | 不平滑，训练统计抖动大 |
| Gumbel-Top-k | 可做稀疏 | 较好 | 基于 Gumbel 采样 | 温度与估计器选择敏感 |
| Dense Softmax + temperature | 否，通常稠密 | 是 | 调温度平滑分布 | 算力和通信成本高 |

适用边界可以这样判断：

| 需求 | 更合适的方法 |
|---|---|
| 稀疏激活 + 固定计算预算 + 训练期探索 | Noisy Top-k |
| 需要明确的采样建模语义 | Gumbel 系方法 |
| 不在意稀疏，只想稳定训练 | Dense softmax + temperature |
| 做概念验证、快速实验 | $\varepsilon$-greedy |

本质区别可以压缩成一句话：

- Noisy Top-k 是在连续分数空间里做受控探索。
- $\varepsilon$-greedy 是在离散动作空间里插入随机试错。
- Dense softmax 是通过连续分布平滑来避免硬选择。

如果你的目标是“在稀疏 MoE 中保持固定预算，同时又尽量不让专家塌缩”，Noisy Top-k 往往仍是默认起点。

---

## 参考资料

1. Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. ICLR 2017. 这篇论文是稀疏 MoE 和 Noisy Top-k gating 的经典来源，给出了路由公式、稀疏激活方式，以及负载均衡辅助损失的基本形式。  
   https://research.google/pubs/outrageously-large-neural-networks-the-sparsely-gated-mixture-of-experts-layer/

2. Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen. *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*. ICLR 2021. 这篇工作展示了大规模稀疏 MoE 在真实分布式训练中的落地方式，适合补充理解路由、容量约束和工程扩展。  
   https://openreview.net/forum?id=qrwe7XHTmYb

3. Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. JMLR 2022. 这篇论文使用更简单的 Top-1 路由作为对照，有助于理解“更强稀疏性、训练稳定性和负载均衡”之间的取舍。  
   https://www.jmlr.org/papers/v23/21-0998.html

4. Sauerwald. *Randomised Algorithms, Lecture 15: Bandit Algorithms*. 这份讲义适合用来理解 exploration vs exploitation 的基本框架，以及为什么 $\varepsilon$-greedy 常被当作对照思路。  
   https://www.cl.cam.ac.uk/teaching/2122/RandAlgthm/lec15_bandits.pdf

5. Sarwar et al. *StructMoE: Structured Mixture of Experts Using Low Rank Experts*. PMLR 2024. 这篇文章展示了现代 MoE 结构设计的一种延展方向，适合作为进一步阅读材料，而不是 Noisy Top-k 本身的入门材料。  
   https://proceedings.mlr.press/v262/sarwar24a.html

6. 如果只需要抓住本文重点，优先读第 1 篇和第 3 篇：第 1 篇负责定义 Noisy Top-k 的原始形式，第 3 篇负责帮助建立“稀疏路由如何在大模型中变成可训练工程系统”的直觉。
