## 核心结论

训练监控的本质，是把“模型有没有学到东西”“训练是不是在浪费算力”“问题出在模型、数据还是硬件”这三件事，从猜测变成可观察。指标是训练过程的可观测信号，意思是能被记录和画图的数值；工具是把这些数值持续收集、展示、归档的系统。

对初学者最重要的结论有三条。

1. 先看趋势，再看单点。单个 step 的 loss 抖动很常见，真正有意义的是若干 epoch 上的整体走势。
2. 至少同时监控训练集与验证集。只有训练指标，通常只能看出“模型会不会背题”，看不出“模型能不能泛化”。
3. 训练监控不只是看 loss。学习率、梯度范数、吞吐、显存、GPU 利用率，常常比准确率更早暴露问题。

一个新手最常见的场景是训练图像分类模型。TensorBoard 同时画出训练 loss 和验证 loss：前者持续下降，后者先下降后上升，这就是过拟合的典型信号。此时继续训练通常只会让模型更会记住训练数据，不会让线上效果更好，应该优先尝试早停、数据增强、正则化或减小模型容量。

| 指标 | 白话解释 | 主要判断意义 | 常见工具 |
|---|---|---|---|
| 损失 `loss` | 模型当前错得有多严重 | 是否在收敛，是否震荡 | TensorBoard / W&B / MLflow |
| 准确率 `accuracy``/F1` | 预测对了多少 | 任务效果是否提升 | TensorBoard / W&B / MLflow |
| 学习率 `lr` | 每次参数更新迈多大步 | 是否过大导致震荡，过小导致学不动 | TensorBoard / W&B |
| 梯度范数 `grad norm` | 梯度整体大小 | 是否梯度爆炸或消失 | TensorBoard / W&B |
| GPU/CPU 利用率 | 硬件忙不忙 | 是否算力浪费、是否有 I/O 瓶颈 | W&B / Prometheus / Grafana |
| 显存/内存占用 | 存储压力多大 | 是否会 OOM，batch 是否过大 | W&B / 系统监控 |
| 吞吐 `samples/s` | 每秒处理多少样本 | 是否存在通信或数据加载瓶颈 | W&B / Prometheus |

---

## 问题定义与边界

训练监控要解决的问题，不是“自动帮你把模型训好”，而是“尽快知道训练有没有朝正确方向前进”。这里的“正确方向”至少包括三层含义：

1. 优化层面：loss 是否下降，梯度是否稳定。
2. 泛化层面：验证集指标是否同步改善。
3. 系统层面：算力是否被有效利用，训练是否卡在 I/O、通信或某个异常节点上。

边界也要说清楚。监控系统只能告诉你“发生了什么”，不能直接替你决定“模型结构该怎么改”。例如验证 loss 上升，监控面板能指出过拟合趋势，但它不能自动判断应该加 dropout、换数据增强，还是减少网络深度。诊断仍然需要工程判断。

单机训练和分布式训练的监控重点并不相同。单机更关注模型是否学对；多节点除了模型本身，还要关注节点之间是否跑得一致。

| 训练场景 | 重点指标 | 典型问题 |
|---|---|---|
| 单机训练 | train/val loss、accuracy、lr、grad norm、显存 | 过拟合、欠拟合、OOM、学习率不合适 |
| 多节点训练 | 吞吐、通信延迟、各节点 loss、GPU 利用率、数据加载延迟 | 某节点拖慢整体、通信开销过大、负载不均、数据分片异常 |

真实工程例子：8 张卡做分布式训练，整体吞吐比预期低 30%。监控显示 7 张卡 GPU 利用率接近 95%，第 8 张卡只有 55%，但显存占用正常、loss 也没有明显异常。这通常说明不是模型本身的问题，而是该节点的数据加载、网络通信或本地磁盘 I/O 较慢。没有这类监控，工程师很容易误以为是 batch size、学习率或模型结构有问题，结果优化方向完全错。

---

## 核心机制与推导

训练监控之所以必要，来自深度学习优化本身的不稳定性。模型更新依赖梯度，梯度通过链式法则从输出层一路传回前面各层：

$$
\frac{\partial L}{\partial w_i}=\frac{\partial L}{\partial a_n}\prod_{j=i}^{n}\frac{\partial a_j}{\partial a_{j-1}}
$$

这里 $L$ 是损失函数，表示模型当前误差；$w_i$ 是第 $i$ 层参数；连乘项表示后面所有层的导数会不断相乘。若这些导数大多小于 1，乘很多次后就会非常接近 0，形成梯度消失；若大多大于 1，乘很多次后就会迅速变大，形成梯度爆炸。

梯度消失的白话解释是：前面层几乎收不到训练信号，参数更新很慢，模型像“学不进去”。梯度爆炸的白话解释是：更新步子过大，loss 剧烈震荡甚至变成 `NaN`。所以监控梯度范数，不是附加项，而是直接对应优化稳定性的核心指标。

玩具例子：假设一个 4 层网络的局部导数分别是 `0.2, 0.3, 0.5, 0.4`，那么连乘为：

$$
0.2 \times 0.3 \times 0.5 \times 0.4 = 0.012
$$

这意味着传回前面层的梯度已经衰减到原来的 1.2%。如果网络更深，这个值会更小。相反，如果局部导数是 `2, 1.5, 1.8, 2.2`，连乘约为 `11.88`，梯度会被迅速放大。

学习曲线也是诊断的理论基础。训练 loss 与验证 loss 的关系，通常比单独看准确率更可靠。常见情况可以压缩成下面这个判断表：

| 训练 loss | 验证 loss | 典型含义 |
|---|---|---|
| 都高，且下降缓慢 | 都高 | 欠拟合，模型表达能力不够或训练不充分 |
| 训练低，验证高 | 差距持续扩大 | 过拟合，模型记住了训练集 |
| 都下降且接近 | 都较低 | 训练健康，泛化正常 |
| 训练和验证都剧烈震荡 | 不稳定 | 学习率过大、梯度异常、数据分布波动 |

一个简单判断式是泛化间隙，也就是训练与验证的差值：

$$
\text{gap} = L_{val} - L_{train}
$$

若 `gap` 持续扩大，通常说明泛化恶化。比如第 5 个 epoch 时，训练 loss 从 `0.45` 降到 `0.30`，验证 loss 却从 `0.80` 升到 `0.95`，这不是“训练越来越好”，而是“训练集越来越熟，验证集越来越差”。

还要注意学习率和损失曲线是联动的。学习率太大，loss 可能上下乱跳；学习率太小，loss 会下降但极慢，吞吐正常却几乎没有有效进展。所以很多团队会把 `lr` 与 `loss` 放在同一面板看，判断调度器是否按预期工作。

---

## 代码实现

最小可用实现分两层：第一层是训练循环中记录指标；第二层是把指标送到可视化平台。初学者不必一开始追求复杂系统，先做到“每个 epoch 都有完整记录”已经能解决大部分问题。

下面先给一个可运行的 Python 玩具例子，不依赖深度学习框架，只模拟训练日志，并用 `assert` 验证诊断逻辑。

```python
from math import isnan

def diagnose(train_losses, val_losses, grad_norms, lrs):
    assert len(train_losses) == len(val_losses) == len(grad_norms) == len(lrs)
    assert len(train_losses) >= 3

    if any(isnan(x) for x in train_losses + val_losses + grad_norms + lrs):
        return "数值异常"

    if max(grad_norms) > 100:
        return "梯度爆炸"

    if max(grad_norms) < 1e-6:
        return "梯度消失"

    if train_losses[-1] < train_losses[0] and val_losses[-1] > min(val_losses):
        return "过拟合"

    if train_losses[-1] > 0.8 * train_losses[0] and val_losses[-1] > 0.8 * val_losses[0]:
        return "欠拟合"

    return "基本正常"

# 玩具例子：过拟合
train_losses = [1.2, 0.8, 0.5, 0.3]
val_losses = [1.1, 0.9, 0.95, 1.02]
grad_norms = [3.0, 2.5, 2.0, 1.8]
lrs = [1e-3, 1e-3, 1e-3, 1e-3]

result = diagnose(train_losses, val_losses, grad_norms, lrs)
assert result == "过拟合"

# 玩具例子：梯度爆炸
train_losses2 = [1.5, 1.7, 5.0]
val_losses2 = [1.6, 1.8, 5.5]
grad_norms2 = [2.0, 15.0, 300.0]
lrs2 = [1e-2, 1e-2, 1e-2]

result2 = diagnose(train_losses2, val_losses2, grad_norms2, lrs2)
assert result2 == "梯度爆炸"

print("all checks passed")
```

这个例子虽然简单，但它表达了一个工程原则：监控不是“看图说话”，而是可以沉淀成规则。很多团队会把类似规则接入告警系统，例如“连续 3 个 epoch 验证 loss 上升则提示早停候选”“任意 step 梯度范数超过阈值则标记异常 run”。

如果使用 W&B，训练循环通常只需要下面这种量级的代码：

```python
import wandb

wandb.init(project="train-monitor-demo", config={
    "model": "resnet18",
    "batch_size": 128,
    "lr": 1e-3,
})

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    val_acc = evaluate_accuracy(...)
    grad_norm = compute_grad_norm(...)
    lr = optimizer.param_groups[0]["lr"]

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/acc": val_acc,
        "optim/lr": lr,
        "optim/grad_norm": grad_norm,
    })
```

这段代码的重点不在语法，而在记录习惯。指标命名最好有层次，例如 `train/loss`、`val/loss`、`system/gpu_mem`，这样面板更清晰，也方便后续筛选。

真实工程例子：训练一个推荐模型，团队常常会同时记录四类信息。

| 类别 | 例子 |
|---|---|
| 模型效果 | `train_loss`、`val_auc`、`val_logloss` |
| 优化状态 | `lr`、`grad_norm`、`weight_norm` |
| 资源状态 | `gpu_util`、`gpu_mem_mb`、`cpu_percent` |
| 训练效率 | `samples_per_sec`、`step_time_ms`、`dataloader_time_ms` |

TensorBoard 更适合快速接入、查看标量曲线和图结构；MLflow 更强调实验追踪和版本管理；W&B 在协作、面板能力、系统指标自动采集方面通常更完整。工具不同，但接入原则相同：固定记录频率、固定命名方式、固定关键指标集合。

---

## 工程权衡与常见坑

训练监控最常见的误区，是把它当成“最后看一眼结果”的工具。实际上它应该参与整个训练生命周期，从首个 step 就开始记录。因为很多问题在早期就已经出现，只是最终效果差时才被注意到。

下面是最常见的问题、对应指标和处理方向。

| 问题 | 典型监控信号 | 常见原因 | 典型解决方案 |
|---|---|---|---|
| 梯度爆炸 | `grad norm` 急升，loss 震荡或 `NaN` | 学习率过大、初始化差、网络太深 | 梯度裁剪、减小学习率、改初始化 |
| 梯度消失 | `grad norm` 长期极小，loss 降得很慢 | 深层网络导数连乘过小、激活函数饱和 | ReLU 类激活、残差连接、归一化 |
| 过拟合 | 训练好、验证差，gap 扩大 | 模型太大、数据太少、正则不足 | 早停、dropout、权重衰减、数据增强 |
| 欠拟合 | 训练和验证都差 | 模型太弱、训练不够、特征不足 | 更大模型、更久训练、改特征或数据 |
| 模式崩溃 | GAN 输出样本单一，多样性下降 | 生成器和判别器失衡 | 调学习率、谱归一化、改训练节奏 |
| 算力浪费 | GPU 利用率低，step time 高 | 数据加载慢、通信阻塞 | 增加预取、优化 I/O、排查网络 |

几个工程上很实际的坑需要单独指出。

第一，只看平均值，不看分布。比如平均 GPU 利用率 90% 看起来不错，但如果 8 卡里 7 张 99%、1 张 30%，整体训练仍然会被最慢节点拖住。分布式场景一定要看每节点明细。

第二，只看最终最优指标，不看过程。两个实验最终验证准确率都到 88%，但一个 2 小时收敛，另一个 10 小时还伴随多次异常震荡，它们的工程价值完全不同。监控曲线能揭示“到达结果的成本”。

第三，日志粒度不合适。每个 step 全量上报会带来额外开销，特别是在大规模训练下；但只在每个 epoch 记录，又会错过短时异常。常见折中是：step 级记录优化与系统指标，epoch 级记录验证指标。

第四，把相关当因果。验证 loss 上升不一定只代表过拟合，也可能是验证集采样方式变了、数据预处理有 bug、标签错位，甚至评估代码更新了。监控帮你缩小排查范围，但不能替代数据与代码核验。

第五，GAN 等对抗训练要额外关注模式崩溃。模式崩溃的白话解释是：生成器只会产出少数几种样本，看起来 loss 可能还不算离谱，但结果缺乏多样性。新手容易只盯着生成器 loss，却忽略样本可视化和多样性指标。若判别器 loss 很快接近 0，且生成样本越来越像，通常意味着判别器过强，应考虑调低判别器学习率、使用谱归一化，或调整更新频率。

---

## 替代方案与适用边界

不是所有项目都需要一开始就接完整实验平台。监控方案的选择，应该取决于团队规模、训练复杂度和协作需求。

轻量方案是本地日志加 CSV，再用 `matplotlib` 或 `pandas` 画图。这种方式的优点是零平台依赖、成本低；缺点是没有实时看板、缺少协作和实验版本管理。对个人学习、小型课程作业、离线对比实验，它已经够用。

中等规模团队常用 TensorBoard 或 MLflow。TensorBoard 上手快，适合训练过程可视化；MLflow 对实验记录、参数、模型产物管理更友好。若团队要多人共享实验结果、远程查看和报表整理，W&B 的效率通常更高。

基础设施导向的团队会把 Prometheus + Grafana 接进来。它们并不是专门为模型训练设计的，但很适合看系统层指标，比如节点 CPU、网络、磁盘、容器状态。对于大规模分布式训练，通常会形成“双层监控”：训练平台看模型指标，基础设施平台看机器与网络指标。

| 方案 | 适合场景 | 优点 | 限制 |
|---|---|---|---|
| 本地 CSV + matplotlib | 个人学习、小实验 | 简单、无平台依赖 | 不实时、协作差、实验管理弱 |
| TensorBoard | 单机或中小规模训练 | 接入快、曲线直观 | 实验管理能力一般 |
| MLflow | 需要实验版本追踪 | 参数、指标、模型统一管理 | 面板交互能力通常不如 W&B |
| W&B | 团队协作、远程监控、系统指标联动 | 可视化强、对比方便、自动采集多 | 需要接入平台与网络环境 |
| Prometheus + Grafana | 大规模分布式基础设施监控 | 系统指标强、告警体系成熟 | 不直接解决模型效果监控 |
| 自研 dashboard | 有定制流程和合规要求的团队 | 高度可控 | 开发和维护成本高 |

新手可行路径通常是这样的：先把每个 epoch 的 `train_loss`、`val_loss`、`lr` 写入 CSV，训练后自己画图；当你开始频繁比较不同参数组合、需要保存 run 历史、需要多人看同一组实验时，再迁移到 TensorBoard、MLflow 或 W&B。这个迁移顺序更符合工程成本，而不是一开始就引入过多系统复杂度。

---

## 参考资料

- TensorBoard 官方概览: https://www.tensorflow.org/tensorboard
- MLflow Tracking 文档: https://mlflow.org/docs/latest/ml/tracking/
- Weights & Biases 文档与系统指标说明: https://docs.wandb.ai/
- GeeksforGeeks, Vanishing and Exploding Gradients: https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/
- Artificial Intelligence Wiki, Overfitting and Underfitting Detection: https://artificial-intelligence-wiki.com/machine-learning/model-evaluation-and-validation/overfitting-and-underfitting-detection/
- MathWorks, Monitor GAN Training Progress: https://www.mathworks.com/help/deeplearning/ug/monitor-gan-training-progress.html
- 分布式系统监控实践参考: https://the-pi-guy.com/blog/monitoring_distributed_systems_best_practices_and_tools/
