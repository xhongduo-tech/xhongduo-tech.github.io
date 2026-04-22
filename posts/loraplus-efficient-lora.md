## 核心结论

LoRA+ 是 LoRA 的训练策略改进：它不改变模型结构，不增加可训练参数，只是把 LoRA 里的两个低秩矩阵 `A` 和 `B` 放到不同学习率的参数组里更新。

标准 LoRA 的核心公式是：

$$
W = W_0 + \frac{\alpha}{r}BA
$$

其中，`W0` 是冻结的原始权重，`BA` 是新学出来的低秩增量，`r` 是低秩秩数，`alpha` 是缩放系数。低秩是指用更小的矩阵乘积近似一个大矩阵更新，从而减少训练参数。

LoRA+ 的核心公式是：

$$
\eta_A = \eta,\quad \eta_B = \lambda \cdot \eta,\quad \lambda \ge 1
$$

其中，`ηA` 是 `A` 的学习率，`ηB` 是 `B` 的学习率，`λ` 是 LoRA+ 的学习率倍率。学习率是每一步参数更新的步长，步长越大，参数变化越快。

| 对比项 | 标准 LoRA | LoRA+ |
|---|---:|---:|
| 是否增加可训练参数 | 否 | 否 |
| 是否修改前向结构 | 否 | 否 |
| `A` 和 `B` 是否同学习率 | 是 | 否 |
| 常见设置 | `lr_A = lr_B` | `lr_B = λ * lr_A` |
| 核心收益 | 参数高效 | 训练节奏更合理 |

新手可以这样理解：标准 LoRA 让 `A` 和 `B` 两组参数使用同一个步长，但它们在训练早期承担的作用并不一样。LoRA+ 的改动是让更需要快速形成有效输出的一侧，通常是 `B`，走得更快。这个解释只是帮助理解更新方向，准确说法仍然是：LoRA+ 通过分离 `A/B` 学习率来改善低秩适应的优化动态。

---

## 问题定义与边界

LoRA 的问题不是简单的“参数太少”，而是 `A` 和 `B` 在低秩分解里角色不对称，却经常被同一个学习率更新。

设一个线性层原本是：

$$
h = W_0x
$$

LoRA 加入低秩增量后变成：

$$
h = W_0x + \frac{\alpha}{r}BAx
$$

这里 `A` 先把输入从高维压到低秩空间，`B` 再把低秩表示映射回输出维度。压缩和还原是两个不同角色，因此它们的梯度尺度、初始化状态和早期更新效果都可能不同。

标准 LoRA 常见初始化是：`A` 随机初始化，`B` 初始化为零。这样做的好处是训练刚开始时 `BA = 0`，模型输出不被 LoRA 分支突然扰动。但副作用是：早期真正能让 LoRA 分支产生非零输出的关键步骤，主要依赖 `B` 从零开始长出来。

问题可以定义为：同学习率下，`A` 和 `B` 的优化节奏不匹配，导致低秩增量的有效学习速度低于理想状态。

| 方法 | 是否增参 | 是否改结构 | 是否改优化策略 | 适合场景 |
|---|---:|---:|---:|---|
| 标准 LoRA | 否 | 是，加入低秩分支 | 否 | 预算有限、任务中等难度 |
| LoRA+ | 否 | 与 LoRA 相同 | 是，分离学习率 | 已用 LoRA 但收敛慢 |
| 提高 rank | 是 | 否，仍是 LoRA 结构 | 否 | 容量不足而不是训练慢 |
| 全参数微调 | 是 | 否 | 可选 | 预算充足、任务要求高 |

分类任务里的边界很清楚：如果模型已经能学到大致方向，但前几百步验证集指标上升很慢，可能是 LoRA 更新节奏不理想；如果模型无论训练多久都无法拟合训练集，问题更可能是 rank 太低、数据质量差、任务格式错误或基础模型能力不足。

LoRA+ 解决的是“能学，但学得慢或不够稳”的问题。它不是所有参数高效微调方法的替代品，也不能把容量不足的 LoRA 自动变成高容量方法。

---

## 核心机制与推导

LoRA 把一个大矩阵更新 `ΔW` 写成两个小矩阵的乘积：

$$
\Delta W = \frac{\alpha}{r}BA
$$

其中：

| 矩阵 | 形状示例 | 白话解释 | 常见初始化 |
|---|---|---|---|
| `A` | `r × d_in` | 把输入压到低秩空间 | 随机初始化 |
| `B` | `d_out × r` | 把低秩结果还原到输出空间 | 零初始化 |
| `BA` | `d_out × d_in` | 实际加到原权重上的更新 | 初始为零 |

玩具例子：设只有一个输入和一个输出，LoRA 分支是：

$$
y = BAx
$$

取 `x = 1`，目标 `y^* = 1`，初始化 `A = 1`，`B = 0`。平方损失为：

$$
L = \frac{1}{2}(y - y^*)^2
$$

此时 `y = 0`，损失存在。对 `A` 和 `B` 求导：

$$
\frac{\partial L}{\partial B} = (BAx - y^*)Ax
$$

$$
\frac{\partial L}{\partial A} = (BAx - y^*)Bx
$$

代入 `A = 1`、`B = 0`、`x = 1`、`y* = 1`：

$$
\frac{\partial L}{\partial B} = -1
$$

$$
\frac{\partial L}{\partial A} = 0
$$

结论是：第一步 `A` 不动，`B` 先动。这个极简例子不能代表全部深度网络，但它说明了一个关键事实：在常见初始化下，`B` 早期更直接决定 LoRA 分支是否能产生有效输出。

标准 LoRA 与 LoRA+ 的机制可以写成：

```text
标准 LoRA:
x -> A -> B -> Δh
     lr    lr

LoRA+:
x -> A -> B -> Δh
     η    λη
```

更新节奏示意如下：

| 阶段 | 标准 LoRA | LoRA+ |
|---|---|---|
| 初始化 | `B = 0`，LoRA 分支输出为零 | 相同 |
| 早期 | `B` 从零开始，但步长与 `A` 相同 | `B` 用更大学习率，更快形成有效增量 |
| 中期 | `A/B` 共同调整低秩空间 | `A/B` 仍共同调整，但节奏分离 |
| 后期 | 可能需要更多步数收敛 | 通常更快接近可用解 |

下面是一个可运行的纯 Python 玩具例子，比较同学习率和 LoRA+ 的第一步更新差异：

```python
def one_step(A, B, x, target, lr_A, lr_B):
    y = B * A * x
    grad_common = y - target
    grad_B = grad_common * A * x
    grad_A = grad_common * B * x

    new_A = A - lr_A * grad_A
    new_B = B - lr_B * grad_B
    new_y = new_B * new_A * x
    return new_A, new_B, new_y

A0, B0, x, target = 1.0, 0.0, 1.0, 1.0
base_lr = 1e-4
ratio = 16

_, B_lora, y_lora = one_step(A0, B0, x, target, base_lr, base_lr)
_, B_plus, y_plus = one_step(A0, B0, x, target, base_lr, base_lr * ratio)

assert abs(B_lora - 1e-4) < 1e-12
assert abs(B_plus - 1.6e-3) < 1e-12
assert y_plus > y_lora
```

真实工程例子：在 RoBERTa 或 Llama 系列模型上做 MNLI 分类、指令微调、长文本生成时，常见做法是冻结主干权重，只在注意力投影层挂 LoRA。此时参数量已经很低，继续省参数不是主要目标。更重要的是让有限的低秩参数更快进入有效训练区间。LoRA+ 的价值就在这里：在参数量不变的前提下，通过优化器参数分组改善训练动力学。

---

## 代码实现

LoRA+ 的实现重点不是改前向传播，而是改优化器参数组。前向仍然是：

$$
W = W_0 + \frac{\alpha}{r}BA
$$

训练时需要把参数拆开：

| 参数组 | 是否训练 | 学习率建议 | 说明 |
|---|---:|---:|---|
| 主干参数 | 否 | 无 | 冻结，不进 optimizer |
| LoRA `A` | 是 | `base_lr` | 低秩输入投影 |
| LoRA `B` | 是 | `base_lr * loraplus_lr_ratio` | 低秩输出投影 |
| embedding LoRA | 可选 | `loraplus_lr_embedding` | 如果 embedding 上也挂 LoRA，单独处理 |

PyTorch 风格的最小写法如下：

```python
import torch

base_lr = 1e-4
loraplus_lr_ratio = 16
weight_decay = 0.01

lora_A_params = []
lora_B_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "lora_A" in name:
        lora_A_params.append(param)
    elif "lora_B" in name:
        lora_B_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": lora_A_params, "lr": base_lr},
        {"params": lora_B_params, "lr": base_lr * loraplus_lr_ratio},
    ],
    weight_decay=weight_decay,
)
```

`A` 和 `B` 不是两个模型，而是同一个 LoRA 模块里的两组参数。LoRA+ 只是让这两组参数用不同步长更新。

如果训练脚本已经支持 LoRA，接入 LoRA+ 通常只需要改三处：

| 配置项 | 含义 | 常见起点 |
|---|---|---:|
| `base_lr` | `A` 的基础学习率 | `1e-4` 或按原 LoRA 配置 |
| `loraplus_lr_ratio` | `B` 相对 `A` 的倍率 | `4, 8, 16` |
| `loraplus_lr_embedding` | embedding LoRA 的单独学习率 | 按任务小网格搜索 |

更完整的接入流程是：

1. 冻结主干模型参数。
2. 给目标层注入 LoRA 模块。
3. 根据参数名或模块类型收集 `lora_A` 和 `lora_B`。
4. 构造 optimizer 参数组。
5. 保持 scheduler、loss、forward 逻辑不变。
6. 用验证集曲线比较标准 LoRA 和 LoRA+。

关键检查项是：不要让 `lora_A` 或 `lora_B` 被遗漏，也不要把冻结的主干参数误放进 optimizer。很多“LoRA+ 无效”的问题，实际来自参数分组错误。

---

## 工程权衡与常见坑

`loraplus_lr_ratio` 不是越大越好。它必须和 `base_lr` 一起看，因为真正作用在 `B` 上的是：

$$
\eta_B = \lambda \eta
$$

如果 `base_lr = 2e-4`，`ratio = 16`，那么 `B` 的学习率就是 `3.2e-3`。这可能有效，也可能过大。判断标准不是默认值，而是训练损失、验证指标和梯度稳定性。

| 常见坑 | 表现 | 处理方式 |
|---|---|---|
| 只改 `ratio`，不改 `base_lr` | 前期快，后期波动 | 降低 `base_lr` 或降低 `ratio` |
| 把 `16` 当硬规则 | 不同任务效果不一致 | 搜索 `4, 8, 16` |
| 在过容易任务上盲目增大 `ratio` | 指标收益小，可能过拟合 | 先用标准 LoRA 做基线 |
| embedding LoRA 混用同一配置 | embedding 更新不稳定 | 单独设置 embedding 学习率 |
| 参数名匹配错误 | 实际仍是标准 LoRA | 打印 optimizer 参数组检查 |

调参建议如下：

| 步骤 | 固定什么 | 搜索什么 | 判断标准 |
|---|---|---|---|
| 第一步 | LoRA rank、batch size、训练步数 | 标准 LoRA 的 `base_lr` | 找到可靠基线 |
| 第二步 | `base_lr` | `ratio = 4, 8, 16` | 看验证集收敛速度 |
| 第三步 | 最优 `ratio` | 小幅调整 `base_lr` | 控制后期波动 |
| 第四步 | 主干 LoRA 配置 | embedding 学习率 | 避免 embedding 过冲 |

真实工程例子：在 MNLI 或长指令微调中，可以先固定 rank、batch size 和训练步数，用标准 LoRA 找到可工作的 `base_lr`。然后试 `ratio = 4, 8, 16`。如果 `ratio = 16` 让验证集更快上升，但后期曲线抖动明显，优先把 `base_lr` 从 `1e-4` 降到 `5e-5`，而不是只把 `ratio` 从 `16` 改到 `8`。

训练曲线通常可以这样读：

| 曲线现象 | 可能原因 | 建议 |
|---|---|---|
| LoRA+ 前期明显更快 | `B` 学习率带来有效更新 | 保留设置，观察后期 |
| 前期快但验证集回落 | `ηB` 过大或过拟合 | 降低 `base_lr` 或加正则 |
| 与标准 LoRA 几乎重合 | 任务太简单或瓶颈不在优化 | 不必强行使用 LoRA+ |
| 训练损失震荡 | 学习率组合过激 | 同时检查 scheduler 和梯度裁剪 |

LoRA+ 的工程价值是低成本尝试。它的风险也集中在学习率配置上，不需要改模型结构，所以回滚成本很低。

---

## 替代方案与适用边界

LoRA+ 适合“已经决定使用 LoRA，但训练节奏不理想”的场景。它不是所有微调问题的默认首选。

如果你在一个很小的分类数据集上微调，标准 LoRA 已经很快收敛，验证集稳定，LoRA+ 的收益可能不明显。相反，在更难的生成任务、复杂指令跟随、多领域迁移任务中，低秩分支需要更充分地学习新行为，LoRA+ 更容易体现优势。

| 方案 | 改什么 | 优点 | 适用边界 |
|---|---|---|---|
| 标准 LoRA | 增加低秩适配分支 | 简单、稳定、参数少 | 首选基线 |
| 增大 rank 的 LoRA | 增加低秩容量 | 表达能力更强 | 显存和训练成本上升 |
| LoRA+ | 分离 `A/B` 学习率 | 不增参，改善收敛 | 主要解决优化节奏 |
| AdaLoRA | 动态分配 rank | 更灵活利用参数预算 | 实现和调参更复杂 |
| DoRA | 分解方向和幅度 | 可能提升表达质量 | 成本和兼容性需评估 |
| 全参数微调 | 更新全部权重 | 能力上限高 | 显存、算力、灾难性遗忘风险高 |

选择建议可以按问题类型判断：

| 场景 | 优先选择 |
|---|---|
| 预算很低，先要可跑通 | 标准 LoRA |
| 标准 LoRA 收敛慢，但能学 | LoRA+ |
| 训练集拟合不上 | 提高 rank 或换更强方法 |
| 数据很少，验证集容易波动 | 保守学习率，标准 LoRA 或小 ratio |
| 任务复杂且预算足够 | LoRA+、DoRA、部分全参微调都可比较 |
| 需要最高效果且资源充足 | 全参数微调或混合策略 |

核心判断是：先区分瓶颈来自容量，还是来自优化。容量不足时，LoRA+ 不能替代增大 rank；优化节奏不合理时，盲目增大 rank 也可能浪费参数。

LoRA+ 的实际定位应该是：标准 LoRA 之后的低成本优化增强。它的最大优点不是概念复杂，而是改动小、兼容性强、容易做消融实验。对初级工程师来说，最推荐的实验方式是先建立标准 LoRA 基线，再只改变 optimizer 参数组，比较同样训练步数下的验证指标。

---

## 参考资料

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354)
3. [LoRA+ ICML 2024 / PMLR 论文页](https://proceedings.mlr.press/v235/hayou24a.html)
4. [LoRA+ 官方代码仓库](https://github.com/nikhil-ghosh-berkeley/loraplus)
5. [Microsoft LoRA 官方实现](https://github.com/microsoft/LoRA)
