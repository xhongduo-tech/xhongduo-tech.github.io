## 核心结论

Sophia 优化器的核心价值，是把“梯度告诉你往哪走”和“Hessian 告诉你这一维有多陡”合并起来做更新。这里的 Hessian 可以理解成“损失函数在某个参数附近的弯曲程度”；弯得越厉害，说明这一步更容易走过头。Sophia 不再像 Adam 那样用梯度平方的指数滑动平均值来估计尺度，而是直接维护对角 Hessian 的估计，再把梯度动量按这个曲率做逐坐标缩放。

它的更新规则可以写成：

$$
\theta_{t+1}
=
\theta_t
-
\eta_t\cdot
\operatorname{clip}\left(
\frac{m_t}{\max(\gamma h_t,\epsilon)},
1
\right)
$$

其中 $m_t$ 是梯度的指数滑动平均，$h_t$ 是对角 Hessian 的指数滑动平均，$\gamma$ 是缩放系数，$\epsilon$ 是数值稳定项。所有运算都按坐标逐元素进行，`clip(x, 1)` 表示把每一维限制在 $[-1,1]$。这意味着无论某个维度的 Hessian 估计多不稳定，单步更新都不会无限放大。

对零基础读者，一个可用的直觉是：每个参数都有自己的“油门限制器”。梯度告诉你该往前还是往后，Hessian 告诉你这条路是平地还是陡坡，clip 再保证你不会一脚油门踩到底。论文给出的结果是，在 GPT-2 Medium（355M）这类语言模型预训练里，Sophia 达到相同验证损失所需的 wall-clock 时间大约是 Adam 的一半量级，同时内存占用仍与 AdamW 同级。

一个玩具例子：某个参数维度上，梯度动量 $m=3$，Hessian EMA $h=4$，$\gamma=0.01$，$\epsilon=10^{-6}$，学习率 $\eta=10^{-3}$。此时分母是 $\max(0.01\times 4,10^{-6})=0.04$，比例为 $3/0.04=75$，再经过裁剪变成 $1$，最终更新量是 $10^{-3}$。这说明 Sophia 不是简单地“二阶更快”，而是“用二阶信息约束每维步长上限”。

下表先给出一个定位：

| 优化器 | 预调节量 | 是否显式用曲率 | 单维更新是否硬裁剪 | 额外代价 |
|---|---|---:|---:|---:|
| SGD | 无 | 否 | 否 | 最低 |
| AdamW | 梯度平方 EMA | 否 | 否 | 低 |
| Sophia | 对角 Hessian EMA | 是 | 是 | 每隔 $k$ 步多一次二阶估计 |
| SignSGD | 仅符号 | 否 | 天然固定幅度 | 低 |

---

## 问题定义与边界

Sophia 要解决的问题，不是“让每一步都更复杂”，而是“在大模型训练里更快达到同样质量”。语言模型训练的瓶颈通常是总训练时间和总算力消耗，而不是单纯的优化器公式是否优雅。AdamW 的优点是稳定、成熟、易调，但它的二阶近似本质上来自梯度平方，这个量并不直接等于曲率，尤其在 Transformer 这种不同层、不同参数块曲率差异很大的模型里，可能会错过更有效的步长分配方式。

这里需要先明确几个术语。

“Hessian”是损失函数对参数的二阶导数矩阵，直白地说，它描述“这个方向到底有多弯”。  
“对角 Hessian”是只看这个矩阵主对角线的版本，直白地说，就是只估计“每个参数自己这一维的弯曲程度”，不管参数之间的相互耦合。  
“EMA”是指数滑动平均，直白地说，就是“给最近值更大权重的平滑统计量”。  
“Hutchinson 估计”是一种随机方法，直白地说，就是不显式构造整个 Hessian，而是用随机向量和 Hessian-向量积去近似其对角信息。

Sophia 的边界也很明确：

| 维度 | 适用情况 | 不适用或需谨慎 |
|---|---|---|
| 模型类型 | Transformer、LLM 预训练 | 小模型上收益可能不明显 |
| 曲率异质性 | 层间差异大时更有价值 | 曲率变化很平滑时优势缩小 |
| 计算资源 | 能接受每隔 $k$ 步额外一次 backward | 极端卡通信、卡同步的分布式环境 |
| 调参成本 | 愿意调 $k,\rho,\gamma$ | 希望完全沿用 AdamW 默认超参 |
| 规模外推 | 论文覆盖到中大规模 LM | 70B+ 级别通信代价仍需额外验证 |

可以用“山路开车”做新手比喻，但要把边界说清楚：AdamW 更像按车速历史估计该不该减速，Sophia 更像额外看了前方路面的坡度。若你的路况本来就很平，额外看坡度的收益不一定明显；若你在多车并行、通信昂贵的场景下行驶，读这份“坡度信息”本身也有代价。

真实工程例子是 GPT-2 Medium 355M 在 OpenWebText 一类数据上的预训练。论文报告中，Sophia 在大约 50k 步附近可以达到 AdamW 约 100k 步的验证损失水平。这个结果重要，不是因为“步数少”，而是因为对训练团队来说，节省的是实际 wall-clock 和 GPU 时间。

---

## 核心机制与推导

Sophia 的状态主要有两份账：

1. 梯度账：$m_t$，即梯度的 EMA。
2. 曲率账：$h_t$，即对角 Hessian 的 EMA。

第一份账回答“往哪走”；第二份账回答“这一维能走多快”。

### 1. 梯度 EMA

设当前梯度为 $g_t$，则

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

这一步和 Adam 系很接近，目的是降噪并保留方向稳定性。

### 2. 周期性更新对角 Hessian

Sophia 不会每步都估计 Hessian，因为那样太贵。它每隔 $k$ 步更新一次 $h_t$。若当前步不是 Hessian 刷新步，就直接沿用上一份估计；若是刷新步，则用 Hutchinson 估计：

$$
\operatorname{diag}(H)
\approx
\mathbb{E}[z \odot (Hz)]
$$

其中 $z$ 是随机向量，常见取法是 Rademacher 向量，即每个元素独立取 $\pm 1$。  
$Hz$ 不是显式构造 Hessian，而是通过 Hessian-vector product 得到。直白地说，你不用真的把一个巨大二阶矩阵算出来，只要问“这个矩阵乘上某个向量会得到什么”，这在自动求导框架里通常只需额外一次 backward 级别的代价。

然后再对这个二阶估计做平滑：

$$
h_t = \beta_2 h_{t-k} + (1-\beta_2)\hat h_t
$$

其中 $\hat h_t$ 是当前采样得到的对角 Hessian 近似。

### 3. 曲率缩放与裁剪

如果只做 $m_t / h_t$，在非凸区域或噪声很大的地方可能会出问题，因为 Hessian 估计可能过小、为负、或严重波动。所以 Sophia 不直接信任二阶量，而是先做保护：

$$
d_t = \max(\gamma h_t, \epsilon)
$$

再用

$$
u_t = \operatorname{clip}\left(\frac{m_t}{d_t},1\right)
$$

最后更新：

$$
\theta_{t+1} = \theta_t - \eta_t u_t
$$

这里的 clip 非常关键。它不是可有可无的小技巧，而是 Sophia 稳定性的核心。因为一旦某一维的 $\gamma h_t$ 很小，分数会很大；没有裁剪，这种“大步长”会直接造成爆炸。加了裁剪后，单维步长上限固定为学习率 $\eta_t$。

这时会出现一个重要退化现象：如果很多维度都被裁到 $\pm 1$，Sophia 在这些维度上就接近 SignSGD。也就是说，当曲率信息不可靠时，它不会继续“强行二阶”，而是自动退回到更保守的符号更新。

参数关系可以整理为：

| 参数 | 作用 | 太小会怎样 | 太大会怎样 |
|---|---|---|---|
| $\beta_1$ | 平滑梯度方向 | 方向抖动大 | 响应变慢 |
| $\beta_2$ | 平滑 Hessian 估计 | 曲率噪声大 | 跟不上曲率变化 |
| $k$ | Hessian 更新周期 | 过于频繁，额外代价高 | 曲率信息过旧 |
| $\gamma$ 或 $\rho$ | 曲率缩放强度 | 分母过小，容易大量 clip | 曲率作用被削弱 |
| $\epsilon$ | 数值稳定 | 可能除零或过大步长 | 过度抹平小曲率差异 |

如果画成流程，逻辑就是：

梯度 $g_t$ → 更新 $m_t$ → 判断是否到第 $k$ 步 → 若是则做 Hutchinson 估计并更新 $h_t$ → 计算 $\max(\gamma h_t,\epsilon)$ → 做逐坐标 clip → 更新参数。

---

## 代码实现

下面给一个可运行的玩具实现。它不依赖深度学习框架，只演示 Sophia 的核心更新逻辑：梯度 EMA、周期性 Hessian EMA、分母保护和逐坐标裁剪。

```python
from dataclasses import dataclass
from typing import List

def clip_scalar(x: float, bound: float = 1.0) -> float:
    return max(-bound, min(bound, x))

@dataclass
class SophiaState:
    m: List[float]
    h: List[float]
    step: int = 0

def sophia_step(
    theta: List[float],
    grad: List[float],
    hessian_diag_estimate: List[float] | None,
    state: SophiaState,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.99,
    gamma: float = 0.01,
    eps: float = 1e-8,
    k: int = 10,
):
    assert len(theta) == len(grad) == len(state.m) == len(state.h)

    state.step += 1

    # 1) 梯度 EMA
    for i, g in enumerate(grad):
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * g

    # 2) 每隔 k 步刷新一次 Hessian EMA
    if state.step % k == 0:
        assert hessian_diag_estimate is not None
        assert len(hessian_diag_estimate) == len(theta)
        for i, h_hat in enumerate(hessian_diag_estimate):
            state.h[i] = beta2 * state.h[i] + (1 - beta2) * h_hat

    # 3) 曲率缩放 + clip
    new_theta = theta[:]
    for i in range(len(theta)):
        denom = max(gamma * state.h[i], eps)
        ratio = state.m[i] / denom
        update = lr * clip_scalar(ratio, 1.0)
        new_theta[i] -= update

    return new_theta, state

# 玩具例子：与文中数值一致
theta = [1.0]
state = SophiaState(m=[0.0], h=[4.0], step=0)

# 令 beta1=0，使 m 直接等于当前梯度 3
theta2, state = sophia_step(
    theta=theta,
    grad=[3.0],
    hessian_diag_estimate=None,
    state=state,
    lr=1e-3,
    beta1=0.0,
    beta2=0.99,
    gamma=0.01,
    eps=1e-6,
    k=10,
)

# m=3, h=4, denom=0.04, ratio=75, clip 后更新 0.001
assert abs(theta2[0] - 0.999) < 1e-12

# 再验证：若 h 很大，则更新不会触发 clip
theta = [1.0]
state = SophiaState(m=[0.0], h=[1000.0], step=0)
theta2, state = sophia_step(
    theta=theta,
    grad=[1.0],
    hessian_diag_estimate=None,
    state=state,
    lr=1e-3,
    beta1=0.0,
    gamma=0.01,
    eps=1e-6,
    k=10,
)
# denom=10, ratio=0.1, 更新 0.0001
assert abs(theta2[0] - 0.9999) < 1e-12
```

如果迁移到真实训练框架，核心差别主要在 Hessian 估计部分。伪代码如下：

```python
for step, batch in enumerate(loader, start=1):
    loss = model(batch)
    loss.backward(create_graph=(step % k == 0))

    # 1. 读出梯度，更新 m
    update_m()

    # 2. 每隔 k 步，用随机向量 z 计算 HVP，得到 z * (H z)
    if step % k == 0:
        sample_z()
        hvp = hessian_vector_product(loss, params, z)
        h_hat = z * hvp
        update_h(h_hat)

    # 3. 参数更新：clip(m / max(gamma*h, eps), 1)
    update_params()

    # 4. weight decay / zero_grad
```

在真实工程里，最值得注意的不是公式，而是调度关系：

| 组件 | 每步执行 | 每隔 $k$ 步执行 |
|---|---:|---:|
| 前向传播 | 是 | 是 |
| 一阶反向传播 | 是 | 是 |
| 更新梯度 EMA | 是 | 是 |
| Hessian 向量积 | 否 | 是 |
| 更新 Hessian EMA | 否 | 是 |
| 参数 clip 更新 | 是 | 是 |

一个真实工程例子：在 GPT 类模型预训练中，优化器通常已经是吞吐敏感路径。Sophia 的做法不是每步都引入昂贵二阶，而是把代价摊到周期中。若 $k=10$，你可以粗略理解为“每 10 步里有 1 步更重一些”，而不是“所有步都更重”。

---

## 工程权衡与常见坑

Sophia 最容易被误解的地方，是“用了 Hessian，就一定更聪明”。实际工程里，二阶信息并不天然可靠，尤其在随机小批量训练中。Hutchinson 估计本身带噪声，EMA 只能缓解，不能消除。

第一个常见坑是 $h_t$ 估计不稳定，甚至出现负值。负的对角 Hessian 在非凸优化里并不奇怪，因为局部区域可能是鞍点或向下弯。如果你直接拿这个值做除法，更新会被反向放大。Sophia 的设计是通过 $\max(\gamma h_t,\epsilon)$ 和 clip 双重限制避免失控。代价是，一旦很多维度都被裁剪，它的行为会越来越接近 SignSGD，收敛性质也会随之变化。

第二个坑是 $k$ 不存在通用最优值。论文里常用 $k=10$，但这只是经验值，不是定理。模型更大、batch 更大、并行更强时，曲率变化节奏会变，最优刷新频率也会变。$k$ 太小，通信和额外 backward 代价会上升；$k$ 太大，Hessian EMA 会过时，失去“按曲率调速”的意义。

第三个坑是不要把 $\gamma$ 当成普通缩放常数随便改。它直接决定有多少维度会进入 clip 区域。若 $\gamma$ 太小，分母普遍过小，训练大部分时间都在做符号更新；若太大，Sophia 又会越来越像“弱化版 AdamW”。

有些实现会用所谓 `win_rate` 监控有多少维度被裁剪，可以把它理解成“裁剪命中率”：

$$
\text{win\_rate}
=
\frac{1}{d}
\sum_{i=1}^{d}
\mathbf{1}
\left(
\left|
\frac{m_{t,i}}{\max(\gamma h_{t,i},\epsilon)}
\right| \ge 1
\right)
$$

其中 $d$ 是参数维度数。直白地说，这个指标衡量“当前有多少坐标的原始更新已经超过了 clip 上限”。经验上若这个比例极低，说明 Hessian 缩放太保守；若极高，说明你几乎一直在做 SignSGD。

风险与缓解可以整理为：

| 风险 | 现象 | 原因 | 缓解 |
|---|---|---|---|
| Hessian 噪声大 | loss 抖动、更新忽大忽小 | Hutchinson 采样方差高 | 增大 $\beta_2$、调大 $k$、检查 batch |
| 过度 clip | 大量维度更新恒为 $\pm \eta$ | $\gamma$ 太小或 $h_t$ 太小 | 调大 $\gamma/\rho$ |
| 曲率信息过旧 | 收敛变慢 | $k$ 太大 | 提高刷新频率 |
| 通信成本高 | 多卡效率下降 | HVP 同步开销明显 | 减少刷新次数，评估是否值得 |
| 直接照搬 AdamW 超参 | 效果不稳定 | 两者缩放机制不同 | 单独调 $\gamma,\rho,k$ |

对初级工程师最实用的判断标准是：先看收益来自哪里。如果你训练的是几百万参数的小模型，或者本来训练已经高度稳定，Sophia 的额外复杂度可能换不回明显收益。如果你训练的是曲率差异很大的 Transformer，而且 wall-clock 很贵，Sophia 才更像一把有效工具。

---

## 替代方案与适用边界

Sophia 不是“全面替代 AdamW”的结论，更准确的说法是：它在“高维、曲率异质、训练很贵”的场景里，比 AdamW 更值得尝试。

先看横向比较：

| 方案 | 优点 | 缺点 | 更适合什么场景 |
|---|---|---|---|
| AdamW | 稳定、成熟、生态完整 | 对曲率适应有限 | 默认首选、大多数训练任务 |
| Sophia | 曲率感知、收敛速度潜力高、内存不增 | 需调 $k,\gamma$，分布式二阶代价更敏感 | LLM 预训练、长周期大训练 |
| SignSGD | 实现简单、步长固定 | 信息利用少，精细收敛能力弱 | 极端鲁棒需求、通信压缩研究 |
| SGD/Momentum | 简洁、可解释 | 对超参和调度更敏感 | CV 等已有成熟 recipe 的任务 |

与 AdamW 的本质差别，是 AdamW 用梯度历史的平方估计尺度，Sophia 用曲率历史估计尺度。前者更像“根据过去加速度猜路况”，后者更像“直接估计路面坡度”。如果模型的层间曲率差异很明显，Sophia 更可能把学习率资源分配到真正需要的地方。

但它也有明确禁忌边界。

第一，超大规模分布式训练下，70B+ 级别模型的 Hessian-vector 通信仍然是现实问题。二阶估计即便只在每 $k$ 步做一次，也可能把同步热点放大。  
第二，若任务本身不是长时间预训练，而是中小规模微调，优化器收益往往不如数据、正则化、学习率计划的重要性高。  
第三，若团队需要极强的可复现性和成熟工具链，AdamW 依然是更稳妥的默认选项。

因此，最合理的采用策略通常不是“立刻全量切换”，而是：

1. 在已有 AdamW 基线上对比相同步数和相同 wall-clock。
2. 重点监控验证损失下降速度、clip 命中率和多卡吞吐。
3. 只在收益覆盖额外复杂度时推广。

---

## 参考资料

| 标题 | 链接 | 说明 |
|---|---|---|
| Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training | https://ar5iv.org/pdf/2305.14342 | 原论文，定义、公式、实验结果的主来源 |
| Liuhong99/Sophia | https://github.com/Liuhong99/Sophia | 官方实现与 README，包含默认超参与训练建议 |
| Sophia Optimizer Overview | https://www.emergentmind.com/topics/sophia-optimizer | 对更新公式、clip 机制和示例的整理性说明 |
