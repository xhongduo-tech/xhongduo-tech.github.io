## 核心结论

训练中出现梯度 `NaN`，本质上不是“模型突然坏了”，而是反向传播图里的某一步已经不再产生有限数。`NaN` 是 “Not a Number”，白话说就是这个结果在当前数学定义下已经说不清了，比如 `0/0`、`log(负数)`、`inf - inf`。一旦某层梯度变成 `NaN`，参数更新公式

$$
w \leftarrow w - \eta \nabla_w L
$$

里的 $\nabla_w L$ 就失去意义，后续权重、激活、损失都会被污染，训练通常会在几步内全面崩掉。

梯度 `NaN` 最常见的三个来源是：

| 来源 | 典型触发 | 直接后果 |
|---|---|---|
| 梯度爆炸 | 深层链式相乘、学习率过大 | 梯度先变 `inf`，再在后续计算中变 `NaN` |
| 非法数学操作 | 除零、`log(0)`、`log(负数)`、`sqrt(负数)` | 前向或后向直接产生 `NaN` |
| 非有限输入被带入训练 | 数据、标签、参数、学习率本身含 `NaN/Inf` | 整条计算链被污染 |

诊断的正确思路不是“先调参碰碰运气”，而是两步并行：

1. 找到第一处从有限数变成 `NaN` 或 `inf` 的张量。
2. 判断这是增长失控，还是值域非法。

如果是增长失控，就要限制更新链，比如梯度裁剪

$$
\mathrm{clip}(\nabla), \quad \|\nabla\| \le c
$$

如果是值域非法，就要在算子之前做保护，比如先 `mask` 再除法、用 `log_softmax` 代替 `softmax` 后再取对数、对分母和概率加 `epsilon`。

一个新手最容易理解的玩具例子是：两层网络把学习率设到 `1.0` 甚至更高，训练几步后梯度迅速放大，先出现极大值，再出现 `inf`，最后损失和梯度都变成 `NaN`。这说明“梯度 `NaN`”经常不是单点故障，而是整条更新链在数值上已经失控。

---

## 问题定义与边界

梯度 `NaN` 的定义很明确：反向传播得到的某个梯度元素不是有限浮点数。有限浮点数，白话说就是计算机还能正常表示和继续参与运算的数；`NaN` 和 `inf` 都不属于这个范围。

这里要先划清边界。

第一，梯度 `NaN` 不等于框架 bug。PyTorch、TensorFlow 这类框架大多数时候只是忠实执行你给定的计算图。如果图里存在非法运算，框架只是把结果算出来。

第二，梯度 `NaN` 不一定在出问题那一层才第一次被你看到。常见情况是前向某层已经产生 `inf`，但损失暂时还没炸，真正到反向传播时才出现 `NaN`。

第三，不是所有“最终没用到的非法路径”都会自动消失。自动求导会沿原始计算图传播，而不是按人脑理解“这个值最后反正被 mask 掉了就没事”。

一个必须掌握的最小例子是除零。设：

- `x = [1, 1]`
- `div = [0, 1]`

如果直接做 `x / div`，第一个位置已经非法。即使你后面只把第二个位置用于 loss，计算图里仍然记录了第一个位置的除零路径，反向传播时仍可能得到 `x.grad = [NaN, 1]`。这就是“非法路径没有提前切掉”的典型问题。

所以，问题边界可以总结成一句话：只要非法值进入计算图，哪怕暂时没显式炸出来，后续也可能在反向阶段集中暴露。

下面这张表可以作为排查起点：

| 可能触发 `NaN` 的操作 | 典型场景 | 预防策略 |
|---|---|---|
| 除法 | 归一化、比例缩放、IoU、方差标准化 | 除前 `mask`，或分母加 `epsilon` |
| 对数 | 交叉熵、KL、概率建模 | 用 `log_softmax`，对输入 `clamp` |
| 指数 | `softmax`、能量模型 | 先减最大值，或用稳定实现 |
| 平方根 | 方差、距离 | 先保证非负，必要时加 `epsilon` |
| 大学习率更新 | 优化器步长过大 | 降低 `lr`，增加 warmup，做梯度裁剪 |
| 非有限输入 | 脏数据、坏标签、AMP 溢出 | 训练前检查 `isfinite` |

这里的核心不是背清单，而是理解一条原则：所有算子都必须工作在自己的合法值域里。值域，白话说就是这个运算允许输入落入的范围。只要你把输入送到值域之外，`NaN` 迟早出现。

---

## 核心机制与推导

梯度为什么会从“很大”进一步变成 `NaN`？关键在链式法则。

设某一层参数为 $w$，中间激活为 $a$，损失为 $L$，则

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial w}
$$

如果网络有很多层，梯度会变成多项连乘：

$$
\frac{\partial L}{\partial w_k}
=
\frac{\partial L}{\partial a_n}
\cdot
\frac{\partial a_n}{\partial a_{n-1}}
\cdot
\frac{\partial a_{n-1}}{\partial a_{n-2}}
\cdots
\frac{\partial a_{k+1}}{\partial w_k}
$$

这里每一项都可以看成局部放大器。如果这些雅可比矩阵的范数长期大于 1，梯度就会指数增长。雅可比矩阵，白话说就是“这一层输出对输入变化有多敏感”的线性近似。

可以用一个极简符号图理解：

$$
g_k
\leftarrow
g_n \cdot J_n \cdot J_{n-1} \cdots J_{k+1}
$$

若很多层满足 $\|J_i\| > 1$，则近似有

$$
\|g_k\| \approx \|g_n\| \prod_i \|J_i\|
$$

于是当层数增加时，$\|g_k\|$ 可能迅速超过浮点数可表示范围，先变 `inf`。一旦后面再遇到减法、归一化、除法、乘 `0` 等操作，就会进一步变成 `NaN`。

这时再看参数更新公式：

$$
w \leftarrow w - \eta \nabla_w L
$$

如果学习率 $\eta$ 本身很大，或者梯度 $\nabla_w L$ 已经接近上限，那么更新步长会突然失控。很多初学者看到的是“loss 下一步直接 NaN 了”，但真正的机制通常是：

1. 某层梯度过大。
2. 参数一次更新过头。
3. 下一次前向进入非法值域。
4. 再下一次后向出现 `NaN`。

### 玩具例子：为什么大学习率会把整条链打坏

假设一维参数 $w$，损失为

$$
L(w)=w^2
$$

则梯度是

$$
\frac{dL}{dw}=2w
$$

更新后：

$$
w_{t+1}=w_t-2\eta w_t=(1-2\eta)w_t
$$

若 $\eta = 0.1$，则 $|1-2\eta|=0.8$，参数会收缩。  
若 $\eta = 1.5$，则 $|1-2\eta|=2$，参数每一步翻倍，数值迅速爆炸。真实神经网络当然远比这个复杂，但“步长过大导致状态离开稳定区域”的机制是一致的。

另一类机制不是连乘爆炸，而是算子本身不稳定。最常见的是交叉熵手写错误。很多人会写：

1. `outputs = softmax(logits)`
2. `loss = -targets * log(outputs)`

问题在于如果某个 `logits` 特别大，`softmax` 里先做 `exp(logits)`，可能上溢；另一些位置则可能在归一化后非常接近 `0`，再取 `log` 时就得到极大的负值，甚至 `-inf`。接着一参与加法或乘法，整个 loss 就可能变成 `NaN`。

稳定写法是直接用 `log_softmax`，因为它把“指数 + 归一化 + 对数”合并成了数值更稳定的形式。必要时再对概率或分母做 `clamp`，也就是把值强行截断到安全区间。

梯度裁剪的作用，就是在增长链已经开始失控、但还没彻底炸穿时，强制把梯度范数限制住：

$$
g' = g \cdot \min\left(1, \frac{c}{\|g\|}\right)
$$

这里的 $c$ 是阈值。白话说，如果梯度还正常，就原样通过；如果太大，就按比例缩小。它不能修复非法数学操作，但能显著减少“梯度爆炸把参数推入非法区域”的概率。

---

## 代码实现

实际排查时，不要一上来就改很多超参数。最有效的方法是把诊断流程写进训练代码。

先看一个纯 `python` 的最小可运行例子，演示“过大学习率导致发散”。它不是深度学习框架代码，但能直接说明更新机制：

```python
def train_scalar(w, lr, steps):
    history = []
    for _ in range(steps):
        grad = 2.0 * w          # L = w^2
        w = w - lr * grad
        history.append(w)
    return history

stable = train_scalar(1.0, lr=0.1, steps=5)
unstable = train_scalar(1.0, lr=1.5, steps=5)

assert abs(stable[-1]) < abs(stable[0])
assert abs(unstable[-1]) > abs(unstable[0])
```

这个例子说明：即使最简单的二次函数，只要学习率过大，更新也会发散。真实网络里一旦叠加深层链式法则、激活函数和归一化，发散后的结果很容易从“大数”升级成 `inf/NaN`。

下面是新手最该记住的 PyTorch 除零防护例子：

```python
import torch

x = torch.tensor([1.0, 1.0], requires_grad=True)
div = torch.tensor([0.0, 1.0])

mask = div != 0
safe = torch.zeros_like(x)
safe[mask] = x[mask] / div[mask]

loss = safe.sum()
loss.backward()

assert torch.equal(x.grad, torch.tensor([0.0, 1.0]))
```

这里的关键不是“把 `NaN` 算出来后再处理”，而是“不要让非法除法进入图”。

如果你在真实训练脚本里做诊断，建议按下面顺序加检查：

```python
import torch
from torch import nn

def assert_finite_tensor(name, t):
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} contains NaN or Inf")

def train_step(model, batch, optimizer):
    x, y = batch
    assert_finite_tensor("inputs", x)
    assert_finite_tensor("targets", y)

    optimizer.zero_grad(set_to_none=True)

    with torch.autograd.detect_anomaly():
        logits = model(x)
        assert_finite_tensor("logits", logits)

        loss = nn.functional.cross_entropy(logits, y)
        assert_finite_tensor("loss", loss)

        loss.backward()

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise ValueError(f"Non-finite gradient in {name}")

    optimizer.step()
    return loss.item(), float(total_norm)
```

这段流程里有四个关键点：

| 位置 | 要做什么 | 目的 |
|---|---|---|
| 数据进入训练前 | 检查 `inputs/targets` 是否 finite | 排除脏数据 |
| 前向后 | 检查 `logits/loss` | 区分前向炸还是后向炸 |
| `backward` 时 | `detect_anomaly()` | 报出第一处异常算子 |
| `step` 前 | 梯度裁剪 + 检查 `grad` | 防止爆炸继续扩散 |

如果你用 PyTorch Lightning，`finite_checks` 这类工具能更快定位具体参数名。它的价值不在于“自动修复”，而在于把排查粒度从“整个模型坏了”缩小到“哪一层的哪个参数先坏了”。

### 真实工程例子

真实工程里，`NaN` 往往不是第一轮就出现，而是训练了几百轮后才爆。比如生成模型、检测模型、大 batch 混合精度训练里，经常会出现“前 700 多轮都正常，第 780 轮突然 `NaN`”的现象。这通常说明：

1. 输入数据大体没问题，否则更早就会炸。
2. 某层参数在长期迭代中逐渐被推到危险区间。
3. 学习率调度、loss 权重、AMP 缩放或某个自定义算子在这个阶段触发了失稳。

这种场景下最有效的办法不是盲目重跑，而是保存中间 checkpoint，在接近出错轮次时开启更密集的 finite 检查和梯度日志，缩小问题窗口。

---

## 工程权衡与常见坑

工程上最难的地方，不是知道 `NaN` 可能由什么引起，而是很多原因会互相叠加。下面这张表是更实用的排查视角。

| 典型触发源 | 表现 | 常见误判 | 工程应对 |
|---|---|---|---|
| 学习率过大 | loss 剧烈震荡，随后 `NaN` | 误以为是数据集有毒 | 降低基础 `lr`，加 warmup，观察梯度范数 |
| scheduler 异常 | 某轮后突然全局崩溃 | 只盯模型结构 | 每次 `step` 后打印当前 `lr` |
| 输入含 `NaN/Inf` | 第一轮就不稳定 | 误以为是初始化差 | 数据加载后立刻 `isfinite` 检查 |
| 未 guard 的除法 / `log` | 某些 batch 才炸 | 误以为是随机波动 | 对分母加 `epsilon`，或先 `mask` |
| AMP 溢出 | FP16 下更常见 | 误以为 float32 也会同样炸 | 用 `GradScaler`，必要时退回 float32 |
| 自定义 loss 权重过大 | 单一分支梯度压倒其他分支 | 误以为主干网络不稳定 | 分别记录每个 loss 分量的量级 |

这里有几个常见坑需要单独说明。

第一，梯度裁剪不是万能药。它只能处理“梯度太大”，不能处理“前向本身非法”。比如 `log(-1)`，你后面再怎么裁剪梯度也没用，因为错误在前向就已经发生。

第二，`epsilon` 不是越大越安全。很多新手为了防止除零，把分母写成 `x / (d + 1e-2)`。这虽然不容易炸，但可能明显改变原问题的数值意义。正确做法是根据量纲和数据范围选择足够小、但不会被浮点吞掉的 `epsilon`。

第三，`NaN` 可能来自优化器状态，而不只是模型参数。像 Adam 这类优化器会维护动量和二阶矩估计。如果这些状态被污染，哪怕你把参数临时修正回来，后续 `step()` 仍可能继续产出异常。

第四，混合精度训练要特别注意“看上去只在 GPU 上炸”。这通常不是 GPU 特有 bug，而是 FP16 动态范围更小，更容易上溢或下溢。AMP 的白话理解是“用更小的数值格式换速度”，代价就是稳定区间更窄，所以需要 `GradScaler` 帮你缩放。

第五，很多人喜欢在 loss 变 `NaN` 后直接 `continue` 跳过这个 batch。这个策略只能作为临时容错，不能当根治手段。因为如果模型参数已经被污染，跳过一个 batch 没有意义。

---

## 替代方案与适用边界

处理梯度 `NaN` 并不只有“找到 bug 然后改掉”这一条路。更准确地说，有三类策略。

第一类是稳定替代实现，也就是换一个数值更稳的等价写法。典型例子：

- `softmax + log` 改为 `log_softmax`
- 手写交叉熵改为框架内置 `cross_entropy`
- 直接除法改为先 `mask` 再除
- 概率、方差、分母在进入危险算子前先 `clamp`

这类方案适合“公式没错，但实现不稳”。

第二类是增长约束，也就是接受梯度可能偏大，但不让它大到破坏训练。常见手段包括梯度裁剪、权重衰减、残差结构、归一化层、学习率 warmup、合理初始化。这类方案适合“训练过程可能跨过危险区，但还没彻底非法”。

第三类是路径屏蔽，也就是彻底不让非法分支进入计算图。比如使用布尔 `mask`，或者在更复杂场景里使用 `MaskedTensor`。它适合“部分样本或部分位置天然无效”的问题，比如缺失值、变长序列 padding、稀疏标注、局部除零。

下面这张表更适合做方案选型：

| 替代方案 | 适用边界 | 优点 | 局限 |
|---|---|---|---|
| `log_softmax` / 内置稳定 loss | 分类、概率建模 | 改动小，收益直接 | 只解决特定算子不稳 |
| `clamp` / `epsilon` | 分母接近 0、概率接近边界 | 简单有效 | 可能引入偏差 |
| 梯度裁剪 | 梯度爆炸、深层网络、大模型训练 | 抑制更新失控 | 不修复非法前向 |
| 降低学习率 / warmup | 训练初期不稳、后期震荡 | 原理直接 | 收敛可能变慢 |
| AMP + `GradScaler` | 需要 GPU 加速、显存紧张 | 兼顾速度和稳定性 | 仍需调试 scale |
| `mask` / `MaskedTensor` | 部分位置无效、存在除零路径 | 从图上删掉非法分支 | 代码复杂度更高 |

`MaskedTensor` 的思想可以白话理解成：不是“这个位置算出来了但别看它”，而是“这个位置从一开始就不参与有效梯度传播”。这比事后 `isfinite` 检查更彻底，因为它直接改变了计算图。

但要注意适用边界。如果你的学习率已经跑到 `inf`，或者参数已被连续几轮 `NaN` 污染，再好的 mask 和 clamp 也救不回来，必须回退到早期 checkpoint 重新训练。替代方案的前提是：系统还处在可恢复的数值区域内。

---

## 参考资料

- Codegenes，《Common Causes of NANs During Neural Network Training: Why They Happen & How to Fix Them》，2026-01-16，https://www.codegenes.net/blog/common-causes-of-nans-during-training-of-neural-networks/
- Baeldung，《Common Causes of NaNs During Training》，2025-02-28，https://www.baeldung.com/cs/ml-training-nan-errors-fix
- PyTorch 官方文档，《Autograd mechanics》，包含除零导致 `NaN` 梯度示例，https://docs.pytorch.org/docs/stable/notes/autograd.html
- PyTorch MaskedTensor 文档，《Distinguishing between 0 and NaN gradient》，https://docs.pytorch.org/maskedtensor/main/notebooks/nan_grad.html
- PyTorch Lightning 文档，`finite_checks` 相关 API，https://lightning.ai/docs/pytorch/LTS/api/pytorch_lightning.utilities.finite_checks.html
- Stack Overflow，PRO-GAN 训练后期某卷积层产生 `NaN` 梯度案例，https://stackoverflow.com/questions/79406638/pytorch-conv-layer-produces-nan-gradients-regardless-of-the-input
