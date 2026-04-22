## 核心结论

EMA（Exponential Moving Average，指数移动平均）是给模型参数维护一份指数加权平滑的“影子权重”：训练时优化器继续更新原始权重，评估和推理时通常切换到 EMA 权重。

直观理解：原始权重像实时跳动的温度计，EMA 像做了平滑后的读数。这个说法只用于建立直觉，严格来说 EMA 不是第二个优化器，也不参与反向传播，它只是按固定规则保存一份更平滑的参数副本。

训练中的参数会受 mini-batch 采样、学习率、梯度噪声、正则化和数据增强影响而波动。EMA 的价值在于降低这种短期抖动对评估结果的影响。它通常不会让训练 loss 本身神奇下降，但常能让验证集指标更稳，在生成模型、半监督学习和长时间训练任务中尤其常见。

原始权重与 EMA 权重的区别如下：

| 对象 | 记号 | 谁更新 | 训练时使用 | 评估/推理时使用 | 主要作用 |
|---|---:|---|---|---|---|
| 原始权重 | $\theta_t$ | 优化器 | 是 | 可用，但通常不如 EMA 稳 | 接收梯度更新 |
| EMA 权重 | $\bar{\theta}_t$ | EMA 规则 | 通常不参与前向训练 | 常优先使用 | 平滑参数轨迹 |

常见流程是：

```text
forward
  -> backward
  -> optimizer.step()
  -> ema.update()
  -> optimizer.zero_grad()
```

核心结论可以压缩成三句话：

1. EMA 是参数层面的平滑，不是 loss 平滑，也不是梯度平均。
2. EMA 通常放在 `optimizer.step()` 之后更新，因为它要平均“本步优化后的参数”。
3. `decay` 通常取 `0.999` 到 `0.9999`，训练越长、参数越抖，EMA 的收益通常越明显。

---

## 问题定义与边界

本文讨论的是“模型权重的 EMA”，也就是对模型参数本身做移动平均。这里的参数指神经网络中会被优化器更新的张量，例如线性层的 `weight` 和 `bias`、卷积层的卷积核、Transformer 中注意力层和 MLP 层的权重。

它不是下面几件事：

| 容易混淆的对象 | 是否是本文讨论对象 | 说明 |
|---|---:|---|
| loss 的移动平均 | 否 | 只用于日志展示或训练监控，不改变模型参数 |
| 梯度平均 | 否 | 例如梯度累积、分布式 all-reduce，作用在梯度上 |
| 优化器动量 | 否 | momentum/Adam 的一阶矩在优化器内部影响下一步更新 |
| 模型权重 EMA | 是 | 维护一份可用于评估/推理的影子参数 |

统一记号如下：

| 符号 | 含义 |
|---|---|
| $t$ | 第 $t$ 次优化器更新之后 |
| $\theta_t$ | 第 $t$ 步后的原始模型参数 |
| $\bar{\theta}_t$ | 第 $t$ 步后的 EMA 参数 |
| $\beta$ | EMA 的 decay，表示历史参数保留比例 |
| $1-\beta$ | 当前参数注入 EMA 的比例 |

玩具例子：假设一个模型只有一个参数。第 0 步参数是 `1.0`，第 1 步被梯度推到 `3.0`，第 2 步又到 `0.8`，第 3 步再到 `2.5`。原始权重的变化很剧烈。如果每次评估刚好落在某个抖动点上，验证指标也可能跟着波动。EMA 会把这些点按时间加权混合，得到更平滑的参数轨迹。

EMA 主要解决的是这类问题：

| 场景 | EMA 是否适合 | 原因 |
|---|---:|---|
| 扩散模型训练 | 适合 | 训练长、参数轨迹波动，采样常依赖 EMA 权重 |
| GAN 训练 | 适合 | 生成器和判别器相互影响，参数不稳定较常见 |
| 半监督学习 | 适合 | teacher 模型可用 student 权重 EMA 构造 |
| 小模型短训练 | 不一定 | 训练步数少时，EMA 可能还没跟上 |
| loss 曲线很噪，想让日志好看 | 不适合 | 这应使用日志平滑，不是权重 EMA |
| 优化器或学习率配置明显错误 | 不适合 | EMA 不能替代正确的优化设置 |

真实工程例子：扩散模型训练中经常同时保存原始 checkpoint 和 EMA checkpoint。OpenAI 的 `improved-diffusion` 训练流程会保存类似 `ema_0.9999_200000.pt` 的权重文件，采样时通常使用 EMA 权重。这不是因为 EMA 改变了扩散模型的目标函数，而是因为长时间训练后的 EMA 参数更适合作为采样模型。

---

## 核心机制与推导

EMA 的更新公式是：

$$
\bar{\theta}_t = \beta \bar{\theta}_{t-1} + (1-\beta)\theta_t
$$

其中 $\beta$ 是历史记忆程度。$\beta$ 越接近 1，表示越相信过去的 EMA，当前参数进入 EMA 的比例越小，曲线越平滑，但响应越慢。

等价写法是：

$$
\bar{\theta}_t \leftarrow \bar{\theta}_{t-1} + (1-\beta)(\theta_t - \bar{\theta}_{t-1})
$$

这个写法更适合新手理解：每一步 EMA 都从旧值出发，朝当前权重 $\theta_t$ 迈一小步。步长比例是 $1-\beta$。

常见 decay 取值如下：

| $\beta$ | 当前参数权重 $1-\beta$ | 常见场景 | 特点 |
|---:|---:|---|---|
| 0.9 | 0.1 | 玩具实验、短序列平滑 | 跟得快，平滑弱 |
| 0.99 | 0.01 | 中小训练任务 | 平滑较明显 |
| 0.999 | 0.001 | 深度学习常见设置 | 跟得慢，稳定性更强 |
| 0.9999 | 0.0001 | 扩散模型、长训练 | 非常平滑，需要足够训练步数 |

手算一个最小例子。设 $\beta=0.9$，初始 $\theta_0=\bar{\theta}_0=1$。

第 1 步，原始权重变成 $\theta_1=3$：

$$
\bar{\theta}_1 = 0.9 \times 1 + 0.1 \times 3 = 1.2
$$

第 2 步，原始权重变成 $\theta_2=5$：

$$
\bar{\theta}_2 = 0.9 \times 1.2 + 0.1 \times 5 = 1.58
$$

可以看到，原始权重从 `1` 到 `3` 再到 `5`，EMA 只从 `1` 到 `1.2` 再到 `1.58`。这就是“平滑”的本质：它不是预测更好的参数，而是把近期参数轨迹压成一个更稳定的估计。

如果展开递推式，可以看到 EMA 是历史权重的指数加权和：

$$
\bar{\theta}_t
= (1-\beta)\theta_t
+ \beta(1-\beta)\theta_{t-1}
+ \beta^2(1-\beta)\theta_{t-2}
+ \cdots
$$

离当前越近的参数权重越大，越早的参数权重按 $\beta^k$ 衰减。这也是“指数移动平均”这个名字的来源。

---

## 代码实现

下面是一个可运行的 Python 玩具实现，用标量模拟模型权重的 EMA：

```python
def ema_update(ema_value, current_value, decay):
    return decay * ema_value + (1 - decay) * current_value


decay = 0.9
theta_0 = 1.0
ema = theta_0

theta_1 = 3.0
ema = ema_update(ema, theta_1, decay)
assert abs(ema - 1.2) < 1e-12

theta_2 = 5.0
ema = ema_update(ema, theta_2, decay)
assert abs(ema - 1.58) < 1e-12

print(ema)
```

真实训练中，EMA 类通常需要做四件事：

| 接口 | 作用 |
|---|---|
| `update(model)` | 在 `optimizer.step()` 后，用当前模型参数更新 EMA |
| `copy_to(model)` | 把 EMA 权重复制进模型，用于评估或推理 |
| `store(model)` | 暂存当前原始权重，方便评估后恢复 |
| `restore(model)` | 恢复原始权重，继续训练 |

最小 PyTorch 风格伪代码如下：

```python
import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(),
                alpha=1.0 - self.decay,
            )

    @torch.no_grad()
    def store(self, model):
        self.backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(self.backup[name])
        self.backup = {}
```

训练流程：

```python
ema = ModelEMA(model, decay=0.9999)

for batch in dataloader:
    loss = model(batch)
    loss.backward()

    optimizer.step()
    ema.update(model)
    optimizer.zero_grad(set_to_none=True)
```

评估流程：

```python
model.eval()

ema.store(model)
ema.copy_to(model)

with torch.no_grad():
    metrics = evaluate(model, valid_loader)

ema.restore(model)
model.train()
```

保存与恢复时，建议把原始权重、EMA 权重、优化器状态和步数一起存：

| 内容 | 是否建议保存 | 原因 |
|---|---:|---|
| `model.state_dict()` | 是 | 恢复训练需要原始权重 |
| `ema.shadow` | 是 | 评估/推理需要 EMA 权重 |
| `optimizer.state_dict()` | 是 | Adam 动量等状态影响后续训练 |
| `global_step` | 是 | decay warmup、日志、学习率调度可能依赖步数 |
| 只保存 EMA 权重 | 视情况 | 只做推理可以，继续训练不够完整 |

注意：实际工程里还要处理 buffer。buffer 是模型中不由优化器更新、但会随训练变化的状态，例如 BatchNorm 的 `running_mean` 和 `running_var`。有些 EMA 实现只平均参数，不处理 buffer；有些实现会把 buffer 直接复制或一起平均。使用前必须确认框架行为。

---

## 工程权衡与常见坑

EMA 的收益和代价都很直接。

收益是评估更稳，推理效果经常更好。代价是额外保存一份参数，显存或内存占用约增加一份模型权重大小；训练代码也要多维护保存、加载、切换和恢复逻辑。

常见错误比公式本身更容易造成问题：

| 错误做法 | 后果 | 正确做法 |
|---|---|---|
| 在 `optimizer.step()` 前更新 EMA | EMA 平均的是旧参数 | 先 `optimizer.step()`，再 `ema.update()` |
| 只保存原始权重 | 推理时拿不到 EMA 效果 | checkpoint 同时保存 `model` 和 `ema` |
| 评估后不恢复原始权重 | 后续训练从 EMA 权重继续，状态混乱 | `store -> copy_to -> evaluate -> restore` |
| decay 过大且训练很短 | EMA 长时间跟不上当前模型 | 短训练用较小 decay 或 warmup |
| 不确认 BN buffer 行为 | 训练和评估统计不一致 | 明确 buffer 是复制、平均还是重新估计 |
| 把 EMA 当成正则化万能药 | 掩盖优化问题 | 先检查学习率、数据、loss 和优化器 |
| 分布式训练中各卡各自更新 EMA | 不同进程 EMA 不一致 | 在同步后的模型参数上更新，或只在主进程保存一致状态 |

一个典型工程错误是：训练脚本保存了 `model.pt` 和 `ema_0.9999.pt`，但评估脚本默认加载 `model.pt`。训练日志显示使用 EMA 后验证指标提升，离线评估却复现不出来。根因不是模型退化，而是评估入口加载了错误权重。

正确流程应明确区分：

```text
继续训练：加载 model + optimizer + ema + global_step
只做评估：加载 ema 权重，或加载 model 后 copy EMA 到 model
只做部署：导出 EMA 权重对应的模型
```

真实工程中还要注意混合精度。训练时原始参数可能有 FP32 master weights、FP16/BF16 前向权重、优化器内部状态。EMA 最稳妥的做法通常是维护 FP32 版本的 shadow weights，因为低精度 EMA 长时间累积可能带来额外数值误差。

---

## 替代方案与适用边界

EMA 不是唯一的参数平均方法。它的特点是“在线维护”：训练每走一步，就更新一次影子权重。

几种方案的区别如下：

| 方法 | 核心做法 | 适用边界 | 与 EMA 的区别 |
|---|---|---|---|
| EMA | 每步对当前参数做指数加权平均 | 训练过程持续波动、希望评估更稳 | 在线更新，越近权重越重要 |
| SWA | 对训练后期多个权重做等权平均 | 学习率周期或训练后期平坦区域探索 | 通常偏训练后期，不是每步指数衰减 |
| checkpoint averaging | 选多个 checkpoint 离线平均 | 已有多个稳定 checkpoint | 离线处理，不影响训练过程 |
| 普通 checkpoint 选优 | 选验证集最好的单个 checkpoint | 指标可靠、评估成本可接受 | 不平均参数，只做模型选择 |
| 教师-学生方法 | teacher 提供训练信号，student 学习 | 半监督、一致性训练 | EMA 可用于构造 teacher，但目标函数也会变化 |

SWA（Stochastic Weight Averaging，随机权重平均）是一种把多个训练后期模型权重做平均的方法。它和 EMA 都在参数空间做平均，但使用方式不同：EMA 是训练中持续维护影子权重，SWA 更常见于训练后期或特定学习率策略下的权重平均。

普通 checkpoint 选优也很常见：每隔一段时间保存模型，在验证集上选择指标最好的那个。它的优点是简单，缺点是容易受单次验证波动影响，而且需要频繁评估。EMA 则是把很多训练步的参数轨迹压进一份权重里，减少对单个时间点的依赖。

半监督学习里的 Mean Teacher 是另一个真实工程例子。student 模型正常训练，teacher 模型不是直接反向传播更新，而是用 student 权重的 EMA 更新。teacher 输出更稳定的预测，student 通过一致性损失向 teacher 靠近。这里 EMA 不只是评估技巧，而是训练算法的一部分。

EMA 的适用边界可以这样判断：

| 条件 | 建议 |
|---|---|
| 训练步数很长，验证指标波动明显 | 优先尝试 EMA |
| 生成模型采样质量不稳定 | 优先尝试 EMA checkpoint |
| 半监督 teacher 需要稳定预测 | 可以用 EMA teacher |
| 训练只有几百步 | 谨慎使用，decay 不宜过大 |
| 模型已经稳定且 checkpoint 选优足够好 | EMA 边际收益可能有限 |
| 当前主要问题是欠拟合 | EMA 通常不是关键解法 |
| 当前主要问题是数据错误或标签噪声极大 | 先处理数据，不要指望 EMA 修复 |

实践上，EMA 是一种低成本、高概率有收益的技巧，但它不是训练系统的核心矛盾。先保证数据、目标函数、优化器、学习率和评估流程正确，再加入 EMA，收益才容易被稳定复现。

---

## 参考资料

如果要实现 EMA，优先看框架 API；如果想理解为什么有效，再看论文；如果想看真实训练代码，再看开源项目。

1. [TensorFlow tf.train.ExponentialMovingAverage](https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage)
2. [PyTorch AveragedModel and get_ema_multi_avg_fn](https://docs.pytorch.org/docs/stable/optim.html)
3. [Mean teachers are better role models](https://arxiv.org/abs/1703.01780)
4. [openai/improved-diffusion](https://github.com/openai/improved-diffusion)
5. [Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits](https://openreview.net/forum?id=2M9CUnYnBA)
