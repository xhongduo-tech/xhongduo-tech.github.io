## 核心结论

LAMB 是在 AdamW 上增加“逐层步长缩放”的优化器。所谓“逐层”，不是所有参数共享同一个有效学习率，而是每一层或每个参数张量单独决定自己的步长尺度；所谓“步长缩放”，就是先按 AdamW 算出更新方向，再用该层参数范数与更新范数的比值去放大或缩小这一步。

它的核心公式可以写成：

$$
\theta_{t+1}=\theta_t-\eta \cdot r_t \cdot u_t
$$

其中，$u_t$ 是 AdamW 的基础更新，$r_t=\frac{\|\theta_t\|}{\|u_t\|}$ 是 trust ratio，中文常写作“信任比”。它的含义很直接：某一层当前参数本身有多大，这一层单步更新就允许大致保持同一量级，而不是被同一个全局学习率硬性驱动。

如果把更新写成“相对步长”的形式，会更容易理解。设某层参数为 $\theta_l$，对应更新为 $\Delta \theta_l=-\eta r_l u_l$，那么它的相对更新量近似满足：

$$
\frac{\|\Delta \theta_l\|}{\|\theta_l\|}
\approx
\eta \cdot \frac{\|r_l u_l\|}{\|\theta_l\|}
=
\eta \cdot \frac{\left\|\frac{\|\theta_l\|}{\|u_l\|}u_l\right\|}{\|\theta_l\|}
\approx \eta
$$

这就是 LAMB 的设计直觉：不同层的绝对参数规模可以差很多，但相对更新幅度尽量落在接近的量级上。

它解决的问题不是“AdamW 不会收敛”，而是“AdamW 在超大 batch 下难调”。当 batch 扩大到 $8\text{K}$、$16\text{K}$、$32\text{K}$ 甚至更高时，不同层的参数范数差异会让统一学习率变得很别扭。有些层更新太猛，有些层更新太弱。LAMB 用逐层 trust ratio 把这个节奏拉回到更一致的范围。

先看一个最小例子。假设两层的 AdamW 基础更新范数都等于 2，但参数范数不同：

| 层 | 参数范数 $\|\theta_l\|$ | AdamW 更新范数 $\|u_l\|$ | trust ratio $r_l$ | 有效步长倍率 |
|---|---:|---:|---:|---:|
| A | 8 | 2 | 4 | 4 倍 |
| B | 0.5 | 2 | 0.25 | 0.25 倍 |

同一个全局学习率下，LAMB 会让 A 层走得更大，让 B 层走得更小，因为它们当前参数自身的量级不同。

原文常用“按身高调步长”的比喻，但更准确的说法是：LAMB 不是修改方向，而是给每一层加一个“相对尺度校准器”。方向仍由 AdamW 决定，LAMB 只决定这一层该走多大。

真实工程里，LAMB 最著名的结果来自 BERT 预训练。You 等人在 2019 年的论文《Large Batch Optimization for Deep Learning: Training BERT in 76 minutes》中报告，LAMB 支持把 BERT 预训练 batch 扩展到约 32K 乃至更高的量级，并将训练时间压缩到 76 分钟。这说明它的主要价值不是单步更“聪明”，而是让超大批量训练真正可用。

---

## 问题定义与边界

先定义边界。LAMB 讨论的是“参数分层明显、模型很大、批量很大、分布式训练”的场景，典型例子是 Transformer 预训练。它不是给所有任务都免费提速的通用答案。

AdamW 的问题在于：它对每个参数坐标做自适应缩放，但没有直接约束“整层参数的更新幅度应该和该层参数量级相匹配”。如果某一层 $\|\theta_l\|$ 很大，另一层很小，统一全局学习率下，两层的相对更新比例就可能失衡。大 batch 会进一步放大这个问题，因为梯度噪声变小后，训练更依赖优化器本身的步长设计，而不是依赖随机噪声“帮你刹车”。

这个说法对新手容易抽象，可以拆成两层含义：

| 层面 | AdamW 已经做了什么 | AdamW 没有直接做什么 |
|---|---|---|
| 坐标级 | 对每个参数坐标按历史梯度波动做缩放 | 不关心整层参数整体有多大 |
| 层级 | 无明确层级约束 | 不保证每层相对更新幅度一致 |

白话看，可以把不同层想成不同重量的门。你用同样的推门规则去推所有门，轻门容易推过头，重门又推不动。LAMB 做的事不是换推门方向，而是根据门本身的“体量”调整这一步到底用多大力。

下面这个表可以看清它和 AdamW 的边界差异：

| 模型/场景 | 批次大小 | 训练难点 | AdamW 表现 | LAMB 表现 |
|---|---:|---|---|---|
| 中小型 CNN / 小 Transformer | 32-1024 | 层间范数差异影响较小 | 通常够用 | 收益有限 |
| 大型 Transformer 预训练 | 8K-32K | 不同层相对步长难统一 | 学习率难调，易不稳 | 更容易维持统一训练节奏 |
| 多机分布式超大 batch | 8K-32K+ | 还要考虑额外范数同步 | 通信较简单 | 有额外同步和实现复杂度 |

所以问题定义必须收紧为两句：

1. LAMB 主要解决“超大 batch 下，AdamW 的逐参数自适应不足以保证逐层更新尺度合理”的问题。  
2. 它的收益前提是：模型层级结构明显，层与层参数范数差异大，而且训练确实受限于大 batch 稳定性，而不是别的瓶颈。

如果你训练的是小模型、小 batch、单机任务，LAMB 往往不是第一选择。因为它引入了额外复杂度，但不一定带来对应收益。很多任务里，先把 AdamW、warm-up、学习率调度、梯度裁剪、输入管线和混合精度配好，收益通常比贸然换 LAMB 更稳定。

---

## 核心机制与推导

LAMB 继承了 AdamW 的一阶矩和二阶矩。这里“一阶矩”可以理解为梯度的滑动平均，用来稳定更新方向；“二阶矩”可以理解为梯度平方的滑动平均，用来估计每个坐标的波动大小，从而在波动大时自动缩小步子。

AdamW 的标准写法是：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$

偏差校正后：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

基础更新量写成：

$$
u_t=\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda \theta_t
$$

这里 $\lambda \theta_t$ 是权重衰减项，也就是 AdamW 把正则从梯度里解耦后的写法。注意它不是“梯度的一部分”，而是单独加到更新里。

LAMB 在这一步之后，不立即执行 $\theta_t-\eta u_t$，而是先计算逐层 trust ratio：

$$
r_t=\frac{\|\theta_t\|_2}{\|u_t\|_2}
$$

最终更新变成：

$$
\theta_{t+1}=\theta_t-\eta \cdot r_t \cdot u_t
$$

如果考虑工程稳定性，通常还会写成带保护项和裁剪的形式：

$$
r_t=\mathrm{clip}\left(\frac{\|\theta_t\|_2}{\|u_t\|_2+\delta}, r_{\min}, r_{\max}\right)
$$

这里：

| 符号 | 含义 | 工程作用 |
|---|---|---|
| $\delta$ | 很小的正数 | 防止分母接近 0 时数值不稳定 |
| $r_{\min}$ | trust ratio 下界 | 防止某层几乎不更新 |
| $r_{\max}$ | trust ratio 上界 | 防止某层被放大过头 |

这个推导的关键点是：AdamW 负责“方向”和“坐标级别的自适应”，LAMB 再补一个“层级别的尺度校准”。前者回答“往哪走”，后者回答“这一层一步走多大”。

看一个更完整的最小数值例子：

| 层 | $\|\theta_l\|$ | $\|u_l\|$ | 裁剪前 $r_l$ | 若裁剪区间为 [0.01, 10]，裁剪后 |
|---|---:|---:|---:|---:|
| layer A | 8 | 2 | 4 | 4 |
| layer B | 0.5 | 2 | 0.25 | 0.25 |
| layer C | 100 | 2 | 50 | 10 |

这张表说明三件事：

1. 同样的 AdamW 更新范数，不同层会得到不同的有效学习率。  
2. 参数范数大的层会倾向于拿到更大的更新。  
3. 必须裁剪，否则个别层会因为范数异常大而把步长放大过头。

把其中一层展开成公式更直观。若某层 $\|\theta\|=8,\|u\|=2$，学习率 $\eta=10^{-3}$，则：

$$
r=\frac{8}{2}=4,\qquad
\Delta\theta=-10^{-3}\cdot 4 \cdot u=-0.004u
$$

相比 AdamW 的 $-0.001u$，这层实际走了 4 倍步长。这里的重点不是“更大就更好”，而是“参数本身更大时，这一层允许更大的绝对更新，以保持相近的相对更新尺度”。

还可以再看一个反例。若某层参数很小，例如 $\|\theta\|=0.2$、$\|u\|=2$，那么：

$$
r=\frac{0.2}{2}=0.1
$$

这层会被显著减速。这样做的目的是避免小参数层被一个对大层合适的全局学习率直接冲坏。

真实工程里，这对 Transformer 很重要。Embedding 层、注意力层、前馈层、LayerNorm 相关参数，它们的参数规模、梯度分布、更新敏感性都不一致。超大 batch 训练时，如果所有层被一个全局学习率硬绑定，很容易出现有的层已经基本不动，有的层还在剧烈摆动。LAMB 的逐层尺度调整，修的正是这个失衡。

---

## 代码实现

下面给一个最小可运行的 Python 实现。它不依赖 PyTorch，只用标准库，演示单层参数的一步 LAMB 更新，并打印关键中间量。代码可以直接保存为 `lamb_demo.py` 后运行。

```python
import math


def l2_norm(xs):
    return math.sqrt(sum(x * x for x in xs))


def clip(x, low, high):
    return max(low, min(high, x))


def lamb_step(
    param,
    grad,
    m,
    v,
    t,
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-6,
    weight_decay=0.01,
    clip_min=0.01,
    clip_max=10.0,
):
    assert len(param) == len(grad) == len(m) == len(v)

    new_m = [beta1 * mi + (1.0 - beta1) * gi for mi, gi in zip(m, grad)]
    new_v = [beta2 * vi + (1.0 - beta2) * (gi * gi) for vi, gi in zip(v, grad)]

    m_hat = [mi / (1.0 - beta1 ** t) for mi in new_m]
    v_hat = [vi / (1.0 - beta2 ** t) for vi in new_v]

    update = []
    for p, mh, vh in zip(param, m_hat, v_hat):
        adam_part = mh / (math.sqrt(vh) + eps)
        decay_part = weight_decay * p
        update.append(adam_part + decay_part)

    w_norm = l2_norm(param)
    u_norm = l2_norm(update)

    if w_norm == 0.0 or u_norm == 0.0:
        trust_ratio = 1.0
    else:
        trust_ratio = w_norm / (u_norm + 1e-12)

    trust_ratio = clip(trust_ratio, clip_min, clip_max)

    new_param = [p - lr * trust_ratio * u for p, u in zip(param, update)]
    return {
        "new_param": new_param,
        "new_m": new_m,
        "new_v": new_v,
        "update": update,
        "w_norm": w_norm,
        "u_norm": u_norm,
        "trust_ratio": trust_ratio,
    }


def main():
    param = [3.0, 4.0]   # L2 范数 = 5
    grad = [0.1, -0.2]
    m = [0.0, 0.0]
    v = [0.0, 0.0]

    result = lamb_step(param, grad, m, v, t=1, lr=1e-3)

    print("param_before :", [round(x, 6) for x in param])
    print("grad         :", [round(x, 6) for x in grad])
    print("update       :", [round(x, 6) for x in result["update"]])
    print("w_norm       :", round(result["w_norm"], 6))
    print("u_norm       :", round(result["u_norm"], 6))
    print("trust_ratio  :", round(result["trust_ratio"], 6))
    print("param_after  :", [round(x, 6) for x in result["new_param"]])

    assert len(result["new_param"]) == len(param)
    assert result["trust_ratio"] >= 0.01
    assert result["trust_ratio"] <= 10.0
    assert result["new_param"] != param


if __name__ == "__main__":
    main()
```

这段代码的执行逻辑和公式是一一对应的：

| 代码步骤 | 数学含义 |
|---|---|
| `new_m`, `new_v` | 更新一阶矩、二阶矩 |
| `m_hat`, `v_hat` | 做偏差校正 |
| `update` | 计算 AdamW 基础更新 $u_t$ |
| `w_norm`, `u_norm` | 计算 $\|\theta_t\|$ 与 $\|u_t\|$ |
| `trust_ratio` | 计算并裁剪 trust ratio |
| `new_param` | 执行 $\theta_{t+1}=\theta_t-\eta r_t u_t$ |

如果你运行它，会看到一个现象：`trust_ratio` 不一定接近 1。它可能明显大于 1，也可能明显小于 1。新手最容易误解这里，以为 trust ratio 应该总是一个“微调系数”。其实不是，它在很多层上可以是主导步长大小的核心因素。

在 PyTorch 这类框架里，核心逻辑通常放在 `optimizer.step()` 中，流程如下：

1. 对每个参数张量维护 `exp_avg` 和 `exp_avg_sq`。  
2. 用偏差校正算出 AdamW 基础更新 $u_l$。  
3. 对每个参数张量分别计算 $\|\theta_l\|$ 与 $\|u_l\|$。  
4. 算出 trust ratio，并执行裁剪。  
5. 用 `p -= lr * trust_ratio * u` 更新。  
6. 若训练早期不稳，再叠加 warm-up 学习率调度。

如果想把上面的纯 Python 实现映射到 PyTorch，可以把“一个列表”理解成“一个参数张量”。LAMB 并不要求一整层必须是一个 Python 对象；工程实现里，很多时候是按张量做 trust ratio，而不是严格按神经网络模块做聚合。

分布式训练里还要注意一个额外点：如果“层”的定义跨设备切分，或者某个参数张量被分片保存，那么 $\|\theta_l\|$ 和 $\|u_l\|$ 的计算就不是本地完整值，必须做 AllReduce 才能得到全局范数。这一步就是 LAMB 的额外通信成本来源之一。

真实工程例子里，BERT Large 的大 batch 预训练通常会配合：

- 线性 warm-up  
- bias correction  
- decoupled weight decay  
- trust ratio clip  
- 混合精度训练

原因很简单：LAMB 让步长更适合逐层尺度，但它不会自动解决训练早期的高学习率爆炸问题。warm-up 仍然是必要配套。

---

## 工程权衡与常见坑

LAMB 不是“开关一开就更快”的优化器，它拿稳定的大 batch 收益，换来了实现、调参和通信复杂度。

最常见的坑如下：

| 问题 | 现象 | 原因 | 常见规避方式 |
|---|---|---|---|
| 额外通信 | 多机训练吞吐下降 | 需要同步参数范数和更新范数 | 尽量按本地完整层划分；融合归约；减少碎片化参数组 |
| warm-up 不足 | 前几百步或前几千步发散 | trust ratio 放大后，早期更新过猛 | 使用线性 warm-up，再进入主学习率计划 |
| 不做 bias correction | 前期数值偏移大 | 一阶/二阶矩初值为零，估计有偏 | 保留 Adam 标准偏差校正 |
| clip 范围不合理 | 某些层几乎不动或更新爆炸 | trust ratio 过小或过大 | 常用区间如 `[0.01, 10]`，再按任务调 |
| 把所有参数都一视同仁 | LayerNorm/bias 表现异常 | 这类参数范数和训练行为特殊 | 常按 AdamW 习惯对 bias 和 norm 参数去掉 weight decay |
| 误以为大 batch 一定更快 | 训练总时长反而不降 | 通信和输入管线成为瓶颈 | 先确认瓶颈在优化器稳定性，而不是 IO 或网络 |

这里最值得单独强调的是 warm-up。很多人第一次实现 LAMB，公式本身没错，但一上来就用目标大学习率训练，结果几百步内损失爆炸。原因不是 LAMB 理论错了，而是训练初期 $m_t,v_t$ 还没稳定，trust ratio 又可能把某些层放大，组合起来非常容易冲过头。

这个过程可以分成三步理解：

1. 训练刚开始时，一阶矩和二阶矩都还在从零起步。  
2. 这时基础更新 $u_t$ 的统计量并不稳定。  
3. 如果再叠加较大的 trust ratio 和较高学习率，单步更新很容易过大。

另一个工程点是“层”的粒度。论文里的 layer-wise 是按层做，但实际代码里，很多实现是按参数张量做 trust ratio。这通常是可行的，但要知道它和“严格按模块层聚合”不完全等价。张量切得越碎，额外归约和数值抖动越明显。

还要注意两个容易忽略的细节：

| 细节 | 为什么重要 |
|---|---|
| `weight decay` 是否参与 trust ratio 对应的更新范数计算 | 不同实现略有差异，会影响有效步长 |
| `bias`、`LayerNorm` 参数是否单独分组 | 这类参数通常不做权重衰减，否则容易影响收敛 |

工程上常见的稳妥做法是：先以成熟的 AdamW 参数分组逻辑为基线，只把主权重张量换成 LAMB 的更新规则，而不是把所有参数都无差别塞进同一个层级缩放机制里。

---

## 替代方案与适用边界

如果你的训练不是超大 batch，大多数时候先用 AdamW。它实现成熟、通信简单、调参经验丰富，在中小规模训练里通常更划算。

可以用一句更准确的话总结适用边界：小批量训练时，优化器主要解决“单步更新是否稳定”；超大批量训练时，优化器还要解决“不同层能否在相近的相对尺度下同步前进”。LAMB 明显偏后者。

下面是一个简表：

| 优化器 | 主要优势 | 主要代价 | 适用场景 |
|---|---|---|---|
| SGD / Momentum | 简单、泛化常好 | 对学习率更敏感 | 经典视觉任务、大量成熟配方 |
| AdamW | 稳定、易用、生态成熟 | 超大 batch 下未必最好调 | 小到中等 batch，通用 Transformer 训练 |
| LAMB | 逐层自适应步长，适合超大 batch | 额外范数计算与通信，调参更复杂 | 大模型预训练，8K-32K batch |
| NVLAMB | 面向大规模混合精度和多卡经验优化 | 更依赖具体实现细节 | NVIDIA 大规模训练栈、工业预训练 |

和相近方案相比：

1. AdamW 的优势是简单。若 batch 不大，LAMB 的额外复杂度未必值得。  
2. LARS 也是逐层缩放，但它更偏向 SGD 系路线，在注意力模型上通常不如 LAMB 稳定。  
3. NVLAMB 可以看作工程化更强的一支，重点补了 warm-up、bias correction、混合精度、梯度预归一化等大规模训练细节。  
4. 一些基于全局梯度裁剪、学习率缩放规则、吞吐自适应的方案，也能改善大 batch，但它们解决的问题不完全相同。LAMB 的独特点在于“逐层相对尺度校准”。

因此，是否使用 LAMB，判断标准不是“它是不是更先进”，而是下面三个条件是否同时成立：

1. 你确实想把 batch 推到很大。  
2. 你当前瓶颈确实是大 batch 下训练不稳或难调。  
3. 你愿意承担额外实现和通信成本。

如果这三个条件不满足，AdamW 往往是更稳的基线。如果满足，LAMB 的价值就很明确：它不是替代学习率调度，而是让“大 batch + 深层模型 + 分层参数尺度差异”这件事变得可控。

---

## 参考资料

| 资料 | 年份 | 重点 |
|---|---:|---|
| [You et al., *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes*](https://openreview.net/forum?id=Syx4wnEtvH) | 2019 | LAMB 原始论文，给出 trust ratio 的基本思想、收敛分析，以及 BERT 大 batch 训练结果 |
| [UC Berkeley Technical Report 版本](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2019/EECS-2019-103.html) | 2019 | 更容易查到论文中的训练配置、批量规模、迭代步数等细节 |
| [NVIDIA, *Pretraining BERT with Layer-wise Adaptive Learning Rates*](https://developer.nvidia.com/blog/pretraining-bert-with-layer-wise-adaptive-learning-rates/) | 2019 | 工程化版本 NVLAMB，总结了 bias correction、梯度预归一化、混合精度等稳定性经验 |
| [TensorFlow Addons LAMB 实现入口](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py) | 近年实现 | 便于对照工程代码理解 LAMB 在框架中的写法 |

You 等人 2019 年的论文最值得记住，因为它回答了“为什么要折腾大 batch 优化器”。核心不是公式更复杂，而是它证明了：当 BERT 预训练目标是把训练时间从“天”压缩到“小时”时，优化器必须支持超大批量，而 LAMB 正是为这个目标设计的。

同时也要注意一个工程事实：论文和博客中常见的“32K batch”往往是近似说法，原始论文常给出更具体的批量数字与步数配置。写文章时保留“32K 级别”这个表述最稳妥，因为它强调的是数量级，而不是某个单一整数本身。
