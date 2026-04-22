## 核心结论

SWA（Stochastic Weight Averaging，随机权重平均）是在训练后期对多个模型权重做等权平均的方法。它不改变模型结构，不新增推理分支，只改变最终使用的参数取值。

核心公式是：

$$
\bar w_K = \frac{1}{K}\sum_{k=1}^{K} w_k
$$

其中，$w_k$ 表示训练后期第 $k$ 次保存下来的模型权重快照，$\bar w_K$ 表示平均后的 SWA 权重。

普通训练通常使用最后一个 checkpoint，等价于“最后停在哪个点，就用哪个点”。SWA 的做法是：训练已经接近收敛后，不急着只相信最后一次参数，而是收集多个后期参数点，再取平均。新手可以先把它理解成：普通训练选最后一个稳定位置，SWA 选多个稳定位置的中心。

SWA 的目标不是让训练集 loss 必然更低，而是让最终参数落在更平坦的低损失区域。平坦区域指的是：参数有小幅扰动时，损失变化不剧烈。经验上，这类解通常有更好的泛化能力，也就是在未见过的数据上表现更稳。

在工程实践中，SWA 通常用于常规训练之后的最后 10-20 个 epoch。典型流程是：先正常训练到一个较好模型，再用较高恒定学习率或周期学习率继续探索低损失区域，周期性收集权重，最后做平均，并重新计算 BatchNorm 统计量。

---

## 问题定义与边界

SWA 解决的问题不是“模型能不能训练起来”，而是“模型已经基本收敛后，如何得到泛化更稳定的最终参数”。

收敛是指训练过程中的 loss、验证指标或参数更新已经进入相对稳定阶段。泛化是指模型在训练集之外的数据上仍能保持性能。SWA 关注的是后者：当普通训练已经得到可用模型后，能否用低成本方式进一步改善最终 checkpoint。

一个新手版例子：你训练了一个图像分类器，训练集准确率已经很高，验证集准确率在几个 epoch 之间轻微波动，例如 91.2%、91.5%、91.1%、91.6%。这时最后一个 epoch 未必就是最好的泛化点。SWA 会把后期多个波动不大的模型参数平均，最终模型往往比单个 checkpoint 更稳。

| 项目 | SWA 的做法 | 不适用/容易误解 |
|---|---|---|
| 使用阶段 | 训练后期 | 从头全程平均 |
| 平均对象 | 后期快照权重 | 每一步梯度或损失 |
| 目标 | 更平坦的解、更好泛化 | 直接替代所有优化策略 |
| 推理前处理 | 需重算 BN | 直接拿平均权重就推理 |

这里要明确几个边界。

第一，SWA 不是优化器本身。它通常搭配 SGD、Adam、AdamW 等优化器使用。优化器负责每一步如何更新参数，SWA 负责在后期如何维护一个平均参数副本。

第二，SWA 不是训练全程平均。训练早期参数还在快速学习基础表示，直接平均早期参数可能把未成熟的模型状态混入最终结果。更常见的做法是先完成常规训练，再在后期启用 SWA。

第三，SWA 不是模型集成。模型集成是多个模型分别推理，再融合预测结果；SWA 是把多个权重快照合成一个模型，推理成本仍然是一份模型。

第四，SWA 需要注意 BatchNorm。BatchNorm 是一种归一化层，会维护训练数据的 running mean 和 running variance。权重平均后，这些统计量和新权重可能不匹配，所以通常要重新跑一遍训练集或校准集来更新 BN 统计。

---

## 核心机制与推导

SWA 的关键不是“平均”这个动作本身，而是“在训练后期继续探索一片低损失区域，然后对多个点等权平均”。

设训练后期收集到 $K$ 个权重快照：

$$
w_1, w_2, \dots, w_K
$$

SWA 的最终权重为：

$$
\bar w_K = \frac{1}{K}\sum_{k=1}^{K} w_k
$$

实际实现时，不需要保存所有历史快照，可以用在线更新公式：

$$
\bar w_k = \bar w_{k-1} + \frac{1}{k}(w_k - \bar w_{k-1})
$$

在线更新的意思是：每来一个新快照，就把当前平均值向新快照移动一小步。第 $k$ 个快照的权重占比是 $\frac{1}{k}$，最后结果与一次性求平均相同。

玩具例子：假设模型只有一个参数，后期保存了三个权重：

$$
w_1=1.0,\quad w_2=3.0,\quad w_3=2.0
$$

则 SWA 权重为：

$$
\bar w = \frac{1.0+3.0+2.0}{3}=2.0
$$

它不是选择 1.0 或 3.0，而是选择中间的 2.0。如果 1.0 和 3.0 都在低损失区域边缘附近，2.0 可能更接近低损失区域中心，对参数扰动更稳。

更完整的机制可以分成五步：

1. 前段正常训练，得到一个已经可用的模型。
2. 后段保持较高恒定学习率，或使用周期学习率，让参数继续在低损失区域移动。
3. 每个 epoch 或固定 step 收集一次权重快照。
4. 对这些后期快照做等权平均。
5. 在平均权重上重新计算 BatchNorm 统计量。

| 阶段 | 目的 | 学习率策略 | 输出 |
|---|---|---|---|
| 前期 | 学到可用表示 | 常规衰减 | 普通训练权重 |
| 后期 | 在低损失区探索 | 较高恒定 / 周期学习率 | 多个快照 |
| 收尾 | 得到稳定模型 | 平均快照 + 更新 BN | SWA 模型 |

真实工程例子：训练 ResNet-50 做工业缺陷分类。前 80 个 epoch 使用常规 cosine decay，把模型训练到稳定验证精度；最后 15 个 epoch 切换到 `swa_lr=0.05`，每个 epoch 结束调用一次 `update_parameters()`；训练结束后用训练集或校准集调用 `update_bn()`。这种设置通常不会增加推理成本，训练成本只增加最后一小段管理流程。在 ImageNet 等视觉任务中，SWA 常见收益大约是 0.5%-1% 的精度提升，但具体收益依赖基础训练配方、模型结构和数据质量。

---

## 代码实现

SWA 的代码实现核心是三件事：保存平均模型、定期更新平均权重、训练后重算 BatchNorm。

在 PyTorch 中，常用工具是 `torch.optim.swa_utils` 里的 `AveragedModel`、`SWALR` 和 `update_bn`。`AveragedModel` 用来维护平均后的模型副本，`SWALR` 用来控制 SWA 阶段的学习率，`update_bn` 用来重新计算 BatchNorm 统计。

最小训练骨架如下：

```python
import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

model = ...
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

swa_model = AveragedModel(model)
swa_start = 160
scheduler = SWALR(optimizer, swa_lr=0.05)

for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, optimizer)

    if epoch >= swa_start:
        swa_model.update_parameters(model)
        scheduler.step()
    else:
        standard_scheduler.step()

update_bn(train_loader, swa_model)
```

下面是一个不依赖 PyTorch 的可运行玩具实现，用来验证 SWA 的在线平均公式。它只模拟一维权重，不代表完整神经网络训练，但能说明平均逻辑。

```python
def swa_average(weights):
    avg = 0.0
    for k, w in enumerate(weights, start=1):
        avg = avg + (w - avg) / k
    return avg

snapshots = [1.0, 3.0, 2.0]
result = swa_average(snapshots)

assert result == 2.0
assert abs(result - (sum(snapshots) / len(snapshots))) < 1e-12

def loss(w):
    return (w - 2.0) ** 2

assert loss(result) <= loss(1.0)
assert loss(result) <= loss(3.0)
```

这个例子中的 `loss(w) = (w - 2.0)^2` 是一个玩具损失函数。损失函数是衡量模型预测错误程度的函数，值越小表示当前参数越适合这个简化任务。平均后的权重正好落在 2.0，因此损失最小。

| 组件 | 作用 | 常见误区 |
|---|---|---|
| `AveragedModel` | 保存平均后的模型 | 当成 EMA 用 |
| `update_parameters()` | 更新权重平均 | 只调用一次 |
| `SWALR` | 后期学习率策略 | 后期学习率降太低 |
| `update_bn()` | 重算 BN 统计 | 忽略导致掉点 |

实现时还要区分参数和缓冲区。参数是通过反向传播学习到的权重，例如卷积核和线性层矩阵。缓冲区 buffers 是模型保存但不直接由梯度更新的状态，例如 BatchNorm 的 running mean 和 running variance。SWA 主要平均参数；BN 统计则通常在训练后通过数据重新估计。

---

## 工程权衡与常见坑

SWA 的优点是成本低、改动小、推理阶段没有额外模型结构。它通常不需要重写网络，只需要在训练脚本后期增加平均模型和 BN 重算步骤。

但 SWA 不是无条件提升。它的收益依赖一个前提：基础训练配方已经足够稳定。如果原始模型还没学到有效表示，或者训练过程本身不稳定，SWA 很难通过平均修复根本问题。

| 常见坑 | 后果 | 规避方法 |
|---|---|---|
| 把 SWA 用成全程平均 | 容易破坏优化过程 | 只在后期启用 |
| 后期学习率太低 | 探索不足，平均无效 | 保持较高或周期学习率 |
| 不重算 BN | 推理性能掉点 | 必跑 `update_bn()` |
| 快照太少 | 平均不稳定 | 收集多个后期点 |
| 与 EMA 混淆 | 误用策略 | 明确 SWA 是等权平均 |
| 训练未稳定就切换 | 结果变差 | 先完成基础收敛 |

第一个常见问题是过早启用。训练早期的参数变化很大，模型还没有形成稳定特征。此时做平均，可能会把质量很差的权重混进来。更稳妥的方式是先按照原训练计划跑到接近收敛，再启用 SWA。

第二个问题是后期学习率太低。如果学习率已经衰减到接近 0，参数几乎不再移动，多个快照之间差异很小。此时平均多个点，和使用最后一个点差别不大。SWA 需要后期仍有一定探索能力，所以常配合较高恒定学习率或周期学习率。

第三个问题是忽略 BatchNorm。平均后的权重和原模型的 BN 统计不一定匹配。直接拿平均权重推理，可能导致精度下降。正确做法是在 SWA 模型上重新跑一遍训练集或校准集，只更新 BN 统计，不更新梯度。

第四个问题是把 SWA 和 EMA 混淆。EMA 是 Exponential Moving Average，指数滑动平均，越新的参数权重越高；SWA 是后期快照等权平均，每个快照权重相同。二者都能平滑参数轨迹，但理论动机和使用习惯不同。

工程结论是：SWA 的成本主要来自训练后期的调度管理和 BN 重算，而不是额外模型结构。对于已经稳定的视觉分类、部分 NLP 微调、表格模型或推荐模型实验，它可以作为低成本的最终 checkpoint 优化手段。

---

## 替代方案与适用边界

SWA 适合“已有可用模型，想进一步提升泛化与稳定性”的场景。它不是所有任务的首选，也不应替代基础训练调参。

| 方法 | 平均方式 | 成本 | 优点 | 适用场景 |
|---|---|---|---|---|
| SWA | 等权平均后期快照 | 低 | 简单、稳定、常提升泛化 | 单模型训练后优化 |
| EMA | 指数滑动平均 | 低 | 更偏向最近参数，平滑噪声 | 在线训练、持续学习 |
| Ensemble | 多模型预测融合 | 高 | 通常效果最好 | 高成本推理可接受 |
| 选最佳 checkpoint | 单点选择 | 低 | 实现最简单 | 只求稳定基线 |

EMA 更重视最近参数。公式通常类似：

$$
\theta_{\text{ema}} \leftarrow \alpha \theta_{\text{ema}} + (1-\alpha)\theta
$$

其中，$\alpha$ 是衰减系数，越接近 1，历史参数保留越多。EMA 适合训练过程中持续维护一个平滑模型，常见于目标检测、半监督学习和大模型训练流程。

Ensemble 的思路是保留多个模型，推理时让它们投票或平均预测。它通常效果更强，但推理成本也更高。SWA 只输出一个模型，所以更适合推理成本敏感的场景。

最佳 checkpoint 选择是最简单的基线：保存验证集表现最好的单个模型。它不需要额外训练策略，但容易受验证集噪声影响。如果验证集较小，不同 checkpoint 之间的排名可能不稳定，SWA 往往能提供更稳的最终点。

SWA 的适用边界包括：

| 场景 | 是否适合 SWA | 原因 |
|---|---|---|
| 基础训练已稳定，验证集轻微波动 | 适合 | 后期快照有平均价值 |
| 训练预算允许多跑 10-20 epoch | 适合 | 有足够后期探索空间 |
| 模型含大量 BatchNorm | 适合但要谨慎 | 必须重算 BN 统计 |
| 训练刚开始就不稳定 | 不优先 | 应先修学习率、数据、损失函数 |
| 验证集几乎无波动且模型已很强 | 收益可能有限 | 平均空间不大 |
| 推理成本可接受多个模型 | 可考虑 ensemble | 集成可能比 SWA 更强 |

直接落地时，建议先建立一个稳定的普通训练基线，再加 SWA。不要同时大幅改动数据增强、优化器、学习率、损失函数和 SWA，否则很难判断收益来自哪里。

---

## 参考资料

1. [Averaging Weights Leads to Wider Optima and Better Generalization](https://bayesgroup.org/publications/2018-averaging-weights-leads-to-wider-optima-and-better-generalization/)
2. [PyTorch AveragedModel Documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)
3. [PyTorch Optimizer Documentation: SWA, SWALR and update_bn](https://docs.pytorch.org/docs/stable/optim.html)
4. [Stochastic Weight Averaging in PyTorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
5. [timgaripov/swa Official Implementation](https://github.com/timgaripov/swa)
