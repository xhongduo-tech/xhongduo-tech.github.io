## 核心结论

权重衰减是给参数大小加惩罚，白话说就是“不让某些权重无限长大”。在最常见的形式里，它对应在目标函数中加入 $\frac{\lambda}{2}\|w\|_2^2$，使优化器在拟合数据之外，还要为“大参数”付出代价。

真正要区分的是两件事。

第一，`L2 正则化` 和 `AdamW` 里的 `weight decay` 在自适应优化器上不完全等价。对 SGD，它们常常可以看成一回事；但对 Adam 这类会按坐标缩放梯度的优化器，如果把惩罚项直接混进梯度，就会让正则项也被自适应机制“改形”。AdamW 的做法是把衰减项从梯度更新里拆开，单独对参数做一次收缩，因此每个参数都按统一比例向 0 缩。

第二，实践中真正稳定的量通常不是单独的学习率 $\eta$，也不是单独的衰减率 $\lambda$，而是它们的乘积 $\eta\lambda$。把 AdamW 看成参数的指数滑动平均后，可得到时间尺度
$$
\tau_{\text{iter}}=\frac{1}{\eta\lambda}
$$
它描述“旧参数要保留多久”。如果批量大小、数据集规模或训练步数变化，但你希望正则强度保持接近，就应该优先保持 $\eta\lambda$ 对应的时间尺度不乱跳。

一个对新手最有用的操作结论是：如果你在 PyTorch 的 `AdamW` 里把基础学习率翻倍，常见的补偿做法是把 `weight_decay` 减半，让 $\eta\lambda$ 近似不变。否则你以为自己只改了学习率，实际连正则强度也一起改了。

---

## 问题定义与边界

本文讨论的对象是“模型训练中的参数正则化”，重点放在使用 AdamW 时权重衰减如何影响泛化。泛化，白话说就是模型在没见过的新样本上也能表现稳定，而不是只会背训练集。

问题的核心不是“权重越小越好”，而是“模型是否因为参数过大、路径过强或对噪声过拟合，导致验证集性能下降”。权重衰减只是解决这类问题的一种手段，不是唯一手段，也不是越大越安全。

下表先把三个常见正则化手段放在一起看：

| 方法 | 机制 | 白话解释 | 常见设置 | 主要副作用 |
| --- | --- | --- | --- | --- |
| 权重衰减 | 惩罚参数范数 | 每一步都把参数轻微往 0 拉回去 | `1e-5` 到 `1e-1`，依任务而变 | 过大时会把模型压得学不动 |
| Dropout | 训练时随机屏蔽神经元 | 不让网络长期依赖固定通路 | 全连接层常见 `0.1` 到 `0.5` | 收敛变慢，和 BN 组合要谨慎 |
| Label smoothing | 把硬标签变软 | 不让模型对正确类过度自信 | `0.05` 到 `0.2` | 过大时会削弱判别边界 |

玩具例子：训练一个两层 MLP 做 MNIST 分类。若不做正则，模型可能训练集 99% 准确率、验证集开始回落。加入 `dropout=0.5` 后，相当于每步都在训练一个随机子网络；再加适度权重衰减，则这些子网络共享的参数不会膨胀得太快，验证损失通常更平稳。

边界也要说清楚。本文不讨论：
- 稀疏正则化如 L1
- 早停、数据增强等非参数正则
- 每层独立设计复杂衰减规则的高级优化器细节

本文只回答一个主问题：在 AdamW 体系下，如何理解权重衰减、如何调它、以及它如何与 dropout、label smoothing 配合。

---

## 核心机制与推导

先从最基础的目标函数开始：
$$
L'(w)=L(w)+\frac{\lambda}{2}\|w\|_2^2
$$
其中 $\lambda$ 是正则强度，白话说就是“你对大权重有多反感”。

如果直接对这个目标求梯度，得到
$$
\nabla L'(w)=\nabla L(w)+\lambda w
$$
在普通 SGD 中，更新为
$$
w_{t+1}=w_t-\eta(\nabla L(w_t)+\lambda w_t)
=(1-\eta\lambda)w_t-\eta\nabla L(w_t)
$$
所以 SGD 下，L2 正则与权重衰减常常能写成同一个形式。

但在 Adam 中，梯度会先经过一阶矩、二阶矩估计，再除以 $\sqrt{\hat v_t}+\epsilon$。如果把 $\lambda w$ 直接并进梯度，那么这个正则项也会被自适应缩放，不同参数受到的“衰减力”就不再统一。

AdamW 的关键改动是把两部分拆开：
$$
w_{t+1}=w_t-\eta_t\left[\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda w_t\right]
$$
这里 $\hat m_t,\hat v_t$ 是 Adam 的偏差修正动量统计，白话说就是“过去梯度的平滑估计”。式子里前半部分负责沿梯度方向学习，后半部分负责把参数向 0 收缩。两者职责分开，这就是 decoupled weight decay。

再看一个最小数值例子。设当前
- $w=1.0$
- 自适应梯度项为 $0.5$
- $\eta=0.01$
- $\lambda=0.001$

则一步 AdamW 后：
$$
w' = 1.0 - 0.01(0.5 + 0.001\times 1.0)=0.99499
$$
如果没有衰减，则是 $0.995$。差值很小，但它每一步都会发生，训练几万步后累计影响就明显。

更重要的是时间尺度。把更新写成
$$
w_{t+1}=(1-\eta\lambda)w_t-\eta g_t^{\text{adapt}}
$$
其中 $g_t^{\text{adapt}}$ 表示 Adam 处理后的梯度方向。这个式子和指数滑动平均很像，因此可得到有效记忆长度
$$
\tau_{\text{iter}}=\frac{1}{\eta\lambda}
$$
它表示参数大约记住多少步历史。若按 epoch 表示，设数据集大小为 $D$，批量大小为 $B$，则每个 epoch 的步数约为 $D/B$，于是
$$
\tau_{\text{epoch}}=\frac{B}{\eta\lambda D}
$$

这直接给出缩放规则：
- 若 $B$ 变大，而希望每个 epoch 的正则强度不变，则应相应增大 $\lambda$
- 若 $D$ 变大，而希望时间尺度不变，则应相应减小 $\lambda$
- 若学习率 $\eta$ 调大，则通常要减小 $\lambda$，以维持 $\eta\lambda$ 稳定

可把 coupled 与 decoupled 的差别压缩成一张表：

| 方案 | 更新特征 | 调参含义 | 风险 |
| --- | --- | --- | --- |
| Adam + L2 | 正则项进入梯度，再被自适应缩放 | $\lambda$ 会受梯度尺度影响 | 不同参数衰减不均匀 |
| AdamW | 参数单独乘以收缩系数 | $\lambda$ 更像“纯粹的收缩强度” | 仍需和学习率联动调节 |
| SGD + L2 / weight decay | 两者近似等价 | 理解最直接 | 对复杂尺度变化不如 AdamW 灵活 |

真实工程例子：预训练或微调 ViT、GPT 一类模型时，训练配置常会改批量大小、token 数或总步数。如果只是照抄旧实验的 `weight_decay=0.01`，但学习率和批量都变了，实际正则强度往往已经不是原来那套。经验上，很多“同模型换规模后突然更难训”的问题，本质不是模型坏了，而是 $\eta\lambda$ 这个有效时间尺度被改掉了。

---

## 代码实现

下面先用一个最小可运行的 Python 例子，把“一步 AdamW 更新”和“保持 $\eta\lambda$ 不变”的调参逻辑写清楚。

```python
def adamw_step(w, adapted_grad, lr, weight_decay):
    # 简化版：adapted_grad 已经表示 Adam 处理后的梯度项
    return w - lr * (adapted_grad + weight_decay * w)

def keep_product_constant(base_lr, base_wd, new_lr):
    target = base_lr * base_wd
    return target / new_lr

# 玩具例子：验证一步更新
w0 = 1.0
g = 0.5
lr = 0.01
wd = 0.001

w1 = adamw_step(w0, g, lr, wd)
assert abs(w1 - 0.99499) < 1e-12

# 验证：学习率翻倍时，weight decay 减半，可保持 lr * wd 不变
base_lr = 1e-3
base_wd = 0.1
new_lr = 2e-3
new_wd = keep_product_constant(base_lr, base_wd, new_lr)

assert abs(base_lr * base_wd - new_lr * new_wd) < 1e-15
assert abs(new_wd - 0.05) < 1e-12
```

如果你自己实现 AdamW，核心思想不是把 `lambda * w` 加进原始梯度再交给自适应预条件器，而是分两步：
1. 用 Adam 部分计算自适应梯度更新
2. 单独做参数收缩

可写成伪代码：

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * (grad * grad)
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)

param = param - lr * m_hat / (sqrt(v_hat) + eps)   # 梯度更新
param = param - lr * weight_decay * param          # 解耦的权重衰减
```

工程上还有一个很容易忽略的点：PyTorch/Optax 常用实现里，衰减系数通常直接和当前学习率相乘，所以调学习率时，`weight_decay` 不能总当成独立旋钮。一个实用写法是给调度器挂钩，固定目标乘积：

```python
target = base_lr * base_weight_decay

def update_weight_decay(optimizer, current_lr):
    current_wd = target / current_lr
    for group in optimizer.param_groups:
        group["weight_decay"] = current_wd
```

这样做的含义很明确：你不是固定 `weight_decay` 数值本身，而是固定有效正则强度 $\eta\lambda$。对做学习率 sweep 的实验尤其有用，因为它把二维搜索近似压成一条“对角线搜索”。

---

## 工程权衡与常见坑

第一个坑是把 AdamW 当成“随便调学习率、衰减照抄旧值”的优化器。很多训练崩掉不是因为模型太深，而是学习率调大后，$\eta\lambda$ 同时变大，参数被收缩得过猛，模型表现为前期 loss 降得快、后面上不去，或者验证集明显欠拟合。

第二个坑是把权重衰减加在所有参数上。偏置、LayerNorm、BatchNorm 的缩放参数通常不希望被同样衰减，因为这些参数不直接承担“表示容量过大”的问题。工程上常见做法是只对权重矩阵施加衰减，对 bias 和 norm 参数设 `weight_decay=0`。

第三个坑是只靠权重衰减解决泛化。若模型主要问题是特征共适应，dropout 往往更直接；若模型过度自信，label smoothing 更有效。只把 $\lambda$ 一路加大，常见结果不是“更稳”，而是“表达能力被压瘪”。

第四个坑是忽略实现差异。论文里的 AdamW 思想是解耦的，但具体框架实现仍可能把衰减和当前学习率绑定。调参时应先确认你使用的库到底按什么公式更新。

可以用下面这个检查表快速决策：

| 场景 | 建议 | 原因 |
| --- | --- | --- |
| 批量大小增大 | 优先检查是否需要增大 $\lambda$ | 保持按 epoch 计的正则时间尺度稳定 |
| 学习率翻倍 | 在 PyTorch 常见实现里尝试将 $\lambda$ 减半 | 让 $\eta\lambda$ 近似不变 |
| 数据集显著增大 | 考虑减小 $\lambda$ | 数据更多时，对强正则的依赖通常下降 |
| Dropout 已很强 | 不要再盲目加大 $\lambda$ | 双重正则可能导致欠拟合 |
| 微调预训练模型 | 先从较小 $\lambda$ 起步 | 预训练权重已有结构，过强收缩会破坏已有表示 |

真实工程例子：做 ViT 图像分类微调时，常见配置会同时用 AdamW、轻度 label smoothing、适量数据增强。若你把基础学习率从 `5e-4` 提到 `1e-3`，却还保留 `weight_decay=0.05`，有效收缩强度就翻倍了。结果常见为训练精度不低，但验证精度反而不稳定，误以为是增强过强，实际上先该检查的是学习率和权重衰减是否还匹配。

---

## 替代方案与适用边界

权重衰减不是唯一正则化方法，更不是所有任务的主角。

Dropout 适合处理“网络路径过度绑定”的问题。白话说，它通过随机断开神经元，让模型别老走同一条捷径。对小型全连接网络、容易过拟合的中小数据集任务，它常与权重衰减互补。

Label smoothing 适合处理“模型置信度过高”的问题。白话说，它告诉模型“正确类通常最可能，但不要把概率打满到 100%”。在分类任务，尤其是大类数或存在标注噪声时，它常能让输出分布更平滑。

当基础 AdamW 已经调得比较稳，但仍然在不同模型宽度、不同数据规模间迁移困难时，才有必要考虑更高级的衰减规则，例如按层自适应衰减、按范数重参数化的变体。它们更细，但也带来更多旋钮，不适合一上来就用。

下表给出一个实用对比：

| 方案 | 优势 | 弱点 | 适用边界 |
| --- | --- | --- | --- |
| AdamW | 默认强、实现普遍、易于迁移 | 仍需和学习率联调 | 大多数深度学习训练先从这里开始 |
| Dropout + Label smoothing | 对抗共适应和过度自信很有效 | 不能替代参数范数控制 | 分类任务、数据较小、网络易过拟合 |
| 自适应衰减变体 | 可针对层、宽度、统计量精细设计 | 调参复杂、复现门槛高 | 当基础 AdamW 在规模变化下明显失效时再考虑 |

给初学者的落地顺序可以非常简单：
1. 先用 AdamW，确认 `lr` 和 `weight_decay` 联动合理
2. 如果验证集抖动且模型容易记住训练集，再加 dropout
3. 如果分类输出过于自信，再加 label smoothing
4. 只有在跨规模迁移反复失败时，才研究更复杂的 decay 变体

玩具例子可以这样理解：dropout 是“训练时随机关掉一些神经元”，权重衰减是“每一步都把参数往回拽一点”。前者减少路径依赖，后者限制参数膨胀，两者作用点不同，所以可以一起用，而不是二选一。

---

## 参考资料

- [AdamW: Weight-Decay Scaling Rule, Emergent Mind](https://www.emergentmind.com/topics/weight-decay-scaling-rule-for-adamw)
- [How to set AdamW’s weight decay as you scale model and dataset size, arXiv:2405.13698](https://www.emergentmind.com/articles/2405.13698)
- [AdamW Timescale Framework, Emergent Mind](https://www.emergentmind.com/topics/adamw-timescale)
- [Dropout Regularization in Deep Learning, DigitalOcean](https://www.digitalocean.com/community/tutorials/droput-regularization-deep-learning)
- [How to jointly tune learning rate and weight decay for AdamW, Fabian Schaipp](https://fabian-sp.github.io/posts/2024/02/decoupling/)
