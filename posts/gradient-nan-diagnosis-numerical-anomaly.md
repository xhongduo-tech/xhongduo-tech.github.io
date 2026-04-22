## 核心结论

梯度 NaN 是反向传播中某个局部导数先变成 `NaN/Inf`，再沿链式法则传播到参数梯度的数值故障。

这里的 `NaN` 是 Not a Number，表示计算结果不是一个合法数字；`Inf` 是 infinity，表示正无穷或负无穷。训练中看到参数梯度里有 `NaN`，通常不是“这个参数自己坏了”，而是前面某个算子已经产生了非法值，后续梯度只是被污染。

诊断优先级应当是：

1. 找第一处异常。
2. 判断异常来自前向、loss、局部导数还是参数梯度。
3. 用稳定算子或训练策略消除根因。
4. 再观察最终梯度是否恢复有限值。

一个常见因果链如下：

```text
极端 logits / 全零分母 / log(0)
        ↓
中间算子产生 NaN 或 Inf
        ↓
局部导数变成 NaN 或 Inf
        ↓
链式法则传播异常
        ↓
多个参数的 grad 同时变成 NaN
```

玩具例子：前向 loss 可能暂时看起来正常，但中间某一步出现 `0/0`，反向时对应局部导数变成 `NaN`，后几层参数梯度全部异常。

真实工程例子：Transformer 的 attention mask 如果把某个样本的所有 token 都遮掉，attention scores 没有任何合法位置可归一化，softmax 可能产生 `NaN`，之后这个异常会传到 attention projection、FFN，甚至 embedding 层的梯度。

---

## 问题定义与边界

“梯度 NaN”需要先拆清楚观察对象。参数梯度、激活梯度、loss、前向激活都可能异常，但它们的定位方式不同。

| 异常类型 | 常见触发点 | 观察位置 | 是否一定导致梯度 NaN |
|---|---|---|---|
| 前向 NaN | `0/0`、`log(0)`、非法归一化 | activation、logits | 通常会 |
| loss NaN | 手写交叉熵、概率为 0 后取 log | loss 标量 | 通常会 |
| 局部导数 NaN | `sqrt(0)` 附近、除零、`0 * Inf` | backward anomaly | 通常会 |
| 参数梯度 NaN | 异常沿链式法则传播后 | `p.grad` | 已经是结果 |
| 梯度 Inf | `exp` 溢出、梯度爆炸、FP16 溢出 | `p.grad` 或中间梯度 | 很可能会进一步变 NaN |

边界条件也要明确。本文讨论的是神经网络训练中的数值异常，重点包括 softmax、log、exp、mask、归一化、初始化、学习率、混合精度等场景。数据标签错误、显存损坏、分布式通信错误也可能导致异常，但不属于本文主线。

典型触发源包括：

| 触发源 | 例子 | 直接风险 |
|---|---|---|
| 除零 | `x / x.sum()` 且 `x.sum() == 0` | `NaN/Inf` |
| `log(0)` | `torch.log(prob)` 且 `prob == 0` | `-Inf`，后续可能变 `NaN` |
| `exp` 溢出 | `exp(1000)` | `Inf` |
| softmax 全遮蔽 | attention mask 全为无效位 | 分母无意义 |
| 初始化过大 | 多层线性层输出方差膨胀 | 梯度爆炸 |
| FP16 溢出 | 半精度范围较小 | 更早出现 `Inf` |

softmax 输入极端 logits 时，前向不一定立刻在最终 loss 暴露问题。比如某些中间实现先算 `exp(logits)`，再做归一化。若 logits 太大，`exp` 先变成 `Inf`；若 logits 太小，`exp` 可能下溢到 0。之后再归一化，就可能出现 `Inf/Inf` 或 `0/0`。

---

## 核心机制与推导

反向传播依赖链式法则。链式法则是“总梯度等于沿路径各个局部导数相乘”的规则。设三层计算为：

$$
y = f_3(f_2(f_1(x)))
$$

参数梯度可以写成：

$$
\frac{\partial y}{\partial x}
=
\frac{\partial f_3}{\partial f_2}
\cdot
\frac{\partial f_2}{\partial f_1}
\cdot
\frac{\partial f_1}{\partial x}
$$

只要其中一个局部导数是 `NaN`，整个乘积通常都会变成 `NaN`。所以最后某个参数的 `grad` 是 `NaN`，只能说明异常传播到了这里，不能说明这里就是根因。

softmax 是高频根因之一。它的定义是：

$$
softmax(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

这个公式数学上正确，但直接实现不稳定。若 `z = [-1000, -1000]`，在浮点计算中 `exp(-1000)` 很可能下溢为 0，于是得到：

$$
\frac{0}{0 + 0}
$$

结果就是 `NaN`。稳定写法是先减最大值：

$$
softmax(z)_i =
\frac{\exp(z_i - m)}{\sum_j \exp(z_j - m)},\quad m = \max_j z_j
$$

因为对所有元素同时减去同一个常数，不改变 softmax 的结果，但能把最大指数项变成 `exp(0)=1`。对 `z = [-1000, -1000]`，`m = -1000`，于是变成：

$$
\frac{\exp(0)}{\exp(0)+\exp(0)} = 0.5
$$

交叉熵中也不要手写 `log(softmax(z))`。稳定形式是：

$$
log\_softmax(z)_i = z_i - logsumexp(z)
$$

`logsumexp` 是一种稳定计算 $\log\sum_j \exp(z_j)$ 的方法，避免先算出极大或极小的指数值。

梯度爆炸时，梯度裁剪可以限制梯度范数。梯度范数是把所有梯度看作一个向量后得到的长度。常见形式是：

$$
g \leftarrow g \cdot \min\left(\frac{max\_norm}{\lVert g\rVert_2 + 1e^{-6}}, 1\right)
$$

它不能修复所有 NaN，但能降低学习率过大、初始化过大、长序列反向传播导致的梯度爆炸风险。

---

## 代码实现

诊断分两条线：一条线定位异常，另一条线替换不稳定实现。

下面代码给出最小训练循环。它包含前向检查、反向异常检测、逐层梯度 hook、稳定 `log_softmax` 和梯度裁剪。`assert` 用来保证玩具例子中的稳定 softmax 结果是有限的。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

def assert_finite(name, t):
    assert torch.isfinite(t).all(), f"{name} has NaN or Inf"

def stable_softmax(z):
    m = z.max(dim=-1, keepdim=True).values
    return torch.exp(z - m) / torch.exp(z - m).sum(dim=-1, keepdim=True)

# 玩具例子：不稳定写法可能得到 0/0，稳定写法正常。
z = torch.tensor([[-1000.0, -1000.0]])
p = stable_softmax(z)
assert torch.isfinite(p).all()
assert torch.allclose(p, torch.tensor([[0.5, 0.5]]))

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)

model = TinyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

def attach_grad_hooks(model):
    hooks = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            def make_hook(param_name):
                def hook(grad):
                    if not torch.isfinite(grad).all():
                        print("first suspicious grad:", param_name)
                    return grad
                return hook
            hooks.append(p.register_hook(make_hook(name)))
    return hooks

hooks = attach_grad_hooks(model)

x = torch.randn(8, 4)
y = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0])

optimizer.zero_grad(set_to_none=True)

with torch.autograd.detect_anomaly():
    logits = model(x)
    assert_finite("logits", logits)

    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(log_probs, y)
    assert_finite("loss", loss)

    loss.backward()

for name, p in model.named_parameters():
    if p.grad is not None:
        if torch.isnan(p.grad).any():
            print("NaN in grad:", name)
            break
        assert torch.isfinite(p.grad).all(), f"bad grad in {name}"

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

for h in hooks:
    h.remove()
```

如果只想快速扫一遍参数梯度，可以用更短的检查：

```python
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print("NaN in grad:", name)
        break
```

如果要定位前向第一处异常，可以给模块注册 forward hook。forward hook 是模块前向执行后触发的回调函数，适合检查每一层输出是否有限。

```python
def add_forward_checks(model):
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            tensors = output if isinstance(output, tuple) else (output,)
            for t in tensors:
                if torch.is_tensor(t) and not torch.isfinite(t).all():
                    print("bad forward output:", name)
        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))
    return hooks
```

实际排查顺序建议固定为：检查输入数据，检查 logits，检查 loss，开启 `detect_anomaly()`，再逐层检查梯度。这样可以避免在最终梯度里反复搜索传播结果。

---

## 工程权衡与常见坑

不要把 softmax 后的概率再喂给 `CrossEntropyLoss`。PyTorch 的 `CrossEntropyLoss` 期望输入是 logits，会在内部做稳定的 `log_softmax` 和负对数似然。如果先手动 softmax，再传进去，相当于把概率当作 logits 处理，不仅语义错误，也会增加数值风险。

| 常见坑 | 表现 | 定位手段 | 修复策略 |
|---|---|---|---|
| `softmax` 后接 `CrossEntropyLoss` | loss 不稳定，收敛慢 | 检查 loss 输入 | 直接传 logits |
| 手写 `log(softmax(x))` | `log(0)`、`-Inf` | 检查 log 输入 | 用 `F.log_softmax` |
| attention mask 全遮蔽 | 某批次突然 NaN | 检查 mask 每行有效数 | 保证至少一个有效位置 |
| 学习率过大 | loss 先震荡再 NaN | 降低 lr 对比实验 | 降 lr、warmup、裁剪 |
| 初始化不当 | 训练一开始就爆 | 检查激活方差 | 用 Xavier 或 Kaiming 初始化 |
| FP16 溢出 | 只在混合精度出现 | 关闭 AMP 对比 | loss scaling、局部 FP32 |
| 分母未加 epsilon | 归一化偶发 NaN | 检查分母最小值 | 加 `eps` 并处理全零输入 |

真实工程中的问题经常不是单点触发，而是叠加触发。比如 Transformer 训练中，学习率偏大使 attention logits 变得极端；FP16 又缩小了可表示范围；某些样本被 mask 后有效 token 很少。三个因素叠加后，softmax 或 loss 处第一次出现异常，最后表现为整网梯度 NaN。

梯度裁剪有成本。它会改变真实梯度方向的尺度，过强裁剪可能让模型学得慢。它适合处理梯度爆炸，不适合掩盖 `log(0)`、非法 mask、手写不稳定 softmax 这类确定性错误。

`detect_anomaly()` 也有成本。它会让反向传播变慢，并保留更多调试信息。适合定位问题，不适合长期打开训练。

初始化同样需要匹配激活函数。Xavier 初始化通常适合 tanh、sigmoid 或线性层的方差平衡；Kaiming 初始化通常适合 ReLU 系列激活。初始化过大时，深层网络的激活和梯度方差会层层放大，更容易把 `exp`、softmax、归一化推到极端区域。

---

## 替代方案与适用边界

不同 NaN 根因对应不同修复方式。不要用一种手段处理所有情况。

| 方案 | 解决的问题 | 副作用 | 推荐场景 |
|---|---|---|---|
| 稳定 softmax / `log_softmax` | `exp` 溢出、`log(0)` | 基本无 | 分类、attention、对比学习 |
| `torch.logsumexp` | 手写 log-sum-exp 不稳定 | 基本无 | 概率模型、能量模型 |
| 梯度裁剪 | 梯度爆炸 | 可能减慢训练 | RNN、Transformer、大 lr |
| 降低学习率 | 更新步过大 | 收敛可能变慢 | loss 震荡后 NaN |
| warmup | 初期更新过猛 | 训练策略更复杂 | 大模型、Transformer |
| 损失缩放 | FP16 梯度下溢 | 需要 AMP 管理 | 混合精度训练 |
| 局部 FP32 | FP16 溢出 | 显存和速度成本 | softmax、归一化、loss |
| 改初始化 | 激活/梯度方差异常 | 需匹配网络结构 | 深层 MLP、CNN |
| 数据清洗 | 输入本身非法 | 需要额外校验 | 空样本、异常标签、全零特征 |

如果 NaN 主要来自 FP16 下的 `exp` 溢出，优先使用稳定算子，并让 softmax、归一化、loss 等敏感部分在 FP32 中计算。如果 NaN 来自梯度爆炸，优先降低学习率、加 warmup、做梯度裁剪。如果 NaN 来自全零分母或全遮蔽 mask，梯度裁剪没有根治效果，必须修正数据或 mask 逻辑。

玩具例子适合验证公式和算子稳定性，真实工程例子要关注批次、样本、mask、精度和超参的组合。定位时要保存触发 NaN 的 batch，因为同一段代码可能只在少数极端样本上失败。

最小实用流程是：

| 步骤 | 目标 | 工具 |
|---|---|---|
| 1 | 判断输入是否合法 | `torch.isfinite(x).all()` |
| 2 | 判断前向是否异常 | forward hook |
| 3 | 判断 loss 是否异常 | `torch.isfinite(loss)` |
| 4 | 定位反向异常算子 | `torch.autograd.detect_anomaly()` |
| 5 | 找第一层异常梯度 | parameter hook |
| 6 | 修复根因 | 稳定 API、mask 修正、裁剪、降 lr |

---

## 参考资料

1. [torch.autograd.detect_anomaly](https://docs.pytorch.org/docs/stable/autograd.html#debugging-and-anomaly-detection)
2. [torch.isnan](https://docs.pytorch.org/docs/stable/generated/torch.isnan.html)
3. [torch.nn.functional.log_softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html)
4. [torch.logsumexp](https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html)
5. [torch.nn.utils.clip_grad_norm_](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html)
6. [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
7. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://www.microsoft.com/en-us/research/publication/delving-deep-into-rectifiers-surpassing-human-level-performance-on-imagenet-classification/)

调试时优先查这些 API：`torch.autograd.detect_anomaly()`、`torch.isnan()`、`torch.isfinite()`、`F.log_softmax()`、`torch.logsumexp()`、`torch.nn.utils.clip_grad_norm_()`。
