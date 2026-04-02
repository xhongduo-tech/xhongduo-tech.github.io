## 核心结论

批量大小的缩放规则，指的是当你把一次参数更新所使用的样本数变大时，学习率也按某个规则一起调整，目的是让训练动力学尽量保持不变。对白话解释就是：批量变大后，梯度平均得更稳了，所以单步可以走得更大一些，否则训练会“太谨慎”，吞吐上去了，但收敛速度没跟上。

最常用的是线性缩放规则：

$$
\eta_{\text{new}} = k \cdot \eta_{\text{base}}, \quad k=\frac{B_{\text{new}}}{B_{\text{base}}}
$$

其中 $B$ 是批量大小，$\eta$ 是学习率。比如基准配置是 $B=32,\ \eta=0.1$，当批量扩到 $B=256$ 时，$k=8$，目标学习率就变成：

$$
\eta_{\text{new}} = 8 \times 0.1 = 0.8
$$

但这条规则不能裸用。原因是训练初期参数还处在高曲率区域，高曲率的白话解释是“损失面很陡、方向很敏感”，此时直接上大步长很容易让 loss 突然飙升。工程上几乎总要配合 warmup，即预热，让学习率从小值逐步升到目标值：

$$
\eta_t = \eta_{\max}\cdot \frac{t}{T_{\text{warmup}}}
$$

例如把 $0.1 \rightarrow 0.8$ 的放大放在前 1000 步线性完成，而不是第 1 步就直接用 0.8。结论可以压缩成两句：

1. 批量放大时，学习率通常也应按比例放大，否则训练会变慢。
2. 学习率放大时，warmup 通常不是可选项，而是稳定训练的必要配套。

---

## 问题定义与边界

这里先把“批量”说清楚。批量大小是一次梯度估计中参与计算的样本数。对白话解释就是：模型每更新一次参数，要先看多少条数据。看的数据越多，这次梯度估计通常越稳定。

但工程里至少有三个不同概念，不能混为一谈：

| 概念 | 定义 | 是否改变单次梯度估计噪声 | 是否改变参数更新频率 |
|---|---|---:|---:|
| 单卡 micro batch | 单张卡一次前向/反向实际处理的样本数 | 是 | 否 |
| 全局真实批量 | 所有设备一次同步反向合计的样本数 | 是 | 否 |
| 等效批量 | 全局真实批量再乘梯度累积步数 | 部分是 | 是 |

梯度累积的白话解释是：先做多次前向和反向，把梯度攒起来，再统一更新一次参数。它让“每次更新看到的样本数”变大，但不会让每个 micro batch 的瞬时显存压力变小以外的训练动力学完全等价。因为它同时降低了更新频率。

看一个真实配置例子：

- 8 卡训练
- 每卡 `batch=256`
- 梯度累积 `accum_steps=4`

那么每次 `optimizer.step()` 对应的等效批量是：

$$
B_{\text{eff}} = 8 \times 256 \times 4 = 8192
$$

如果基准配置是单卡或多卡合计 `B_base=256`，那放大倍数就是：

$$
k = \frac{8192}{256} = 32
$$

这时要考虑的不是“我只开了 8 卡”，而是“我每次更新实际用了 32 倍样本”。很多新手在这里犯错：只看每卡 batch，忽略数据并行和梯度累积的乘积，最后 learning rate 缩放完全不对。

下面这个表更贴近调参现场：

| 真实批量 | 累积后等效批量 | LR 缩放比 | 是否通常需要更长 warmup |
|---|---:|---:|---:|
| 32 | 32 | 1x | 否 |
| 256 | 256 | 8x | 是 |
| 256 | 1024 | 32x 相对 32 基准 | 是 |
| 2048 | 8192 | 256x 相对 32 基准 | 强烈需要 |

边界也要说清楚。缩放规则不是物理定律，它依赖几个前提：

1. 大批量确实让梯度噪声按预期下降。
2. 优化器仍处在稳定区间，没有被高曲率破坏。
3. 模型、归一化方式、正则项和数据分布没有让“大 batch + 大 lr”的组合失效。
4. 训练目标关注的是吞吐与稳定性，不是盲目追求极大批量。

所以，批量大小缩放规则回答的问题是：“在扩大并行度时，如何尽量保留原来的优化行为？”它不是回答“批量越大是不是一定越好”。

---

## 核心机制与推导

先看为什么线性缩放在很多场景下成立。

设单样本梯度为 $g_i$，一个 batch 的平均梯度是：

$$
\bar g_B = \frac{1}{B}\sum_{i=1}^{B} g_i
$$

如果样本之间近似独立，平均梯度的方差会随 batch 增大而下降，大致满足：

$$
\mathrm{Var}(\bar g_B) \propto \frac{1}{B}
$$

这句话的工程含义是：batch 越大，梯度噪声越小。既然噪声更小，每一步就可以走得更大一些，否则参数更新会过于保守。

对 SGD 而言，一步更新近似是：

$$
\Delta \theta = -\eta \bar g_B
$$

当 $B$ 变成原来的 $k$ 倍时，$\bar g_B$ 的期望基本不变，但噪声更小。为了让“每步更新的有效尺度”维持相近，经验上就使用：

$$
\eta_{\text{new}} = k \eta_{\text{base}}
$$

这不是严格数学等价，而是一个优化近似：用更大的学习率，补偿更低的随机噪声，让训练速度不要因为 batch 变大而意外变慢。

### 玩具例子

假设你在训练一个二维线性模型，基准配置是：

- `batch=32`
- `lr=0.1`

现在把 batch 提到 `256`。如果你仍然用 `lr=0.1`，平均梯度虽然更稳定，但每次更新并没有更大胆，结果往往是每秒处理样本更多了，可达到同样 loss 所需的参数更新次数没有明显减少，甚至总训练 token 不变时还可能收敛更慢。

如果改成线性缩放：

- `batch=256`
- `lr=0.8`

那么单步会明显更大，更接近原先小 batch 的优化节奏。但如果一开始直接用 0.8，训练前几十步很容易出现 loss spike。于是加入 1000 步 warmup：

$$
\eta_t = 0.8 \cdot \frac{t}{1000}, \quad t < 1000
$$

这时模型先用小步探路，再逐步进入大步长区间，稳定性通常会好很多。

### 为什么 warmup 不是装饰，而是稳定机制

训练初期参数离稳定盆地还远，损失面常常具有高曲率。曲率的白话解释是：你沿某个方向稍微走一点，loss 就变化很快。对局部二次近似：

$$
L(\theta) \approx L(\theta_0) + g^\top(\theta-\theta_0) + \frac{1}{2}(\theta-\theta_0)^\top H(\theta-\theta_0)
$$

其中 $H$ 是 Hessian，表示局部曲率。若最大特征值为 $\lambda_{\max}$，梯度下降在该方向上的稳定步长通常要满足近似条件：

$$
\eta < \frac{2}{\lambda_{\max}}
$$

如果学习率一开始就太大，超过稳定上界，就会在高曲率方向上“弹飞”。这类现象常被称为 catapult。catapult 的白话解释是：参数不是缓慢下坡，而是因为步子太大被局部曲率弹出去，loss 先暴涨、再剧烈震荡，严重时直接数值爆炸。

warmup 的作用就是把训练早期的大学习率暂时压住。它并不改变最终目标学习率，而是延迟你到达那个大步长的时间，让模型先进入更平滑、更可控的区域。

### 真实工程例子

考虑一个多机多卡预训练任务，目标是把吞吐做高：

- 16 台机器
- 每台 8 卡
- 每卡 micro batch = 8
- 梯度累积 = 16

那么等效批量为：

$$
B_{\text{eff}} = 16 \times 8 \times 8 \times 16 = 16384
$$

如果基准实验是 `B=512, lr=3e-4`，按线性缩放，目标学习率会变成：

$$
\eta_{\text{new}} = \frac{16384}{512}\times 3\times 10^{-4} = 9.6\times 10^{-3}
$$

这个数值本身不一定能直接用。原因不是公式错，而是系统已经进入“大 batch 高风险区”：更新频率变低、优化器状态变化变快、早期曲率尖锐。此时工程上往往需要同时做四件事：

1. 拉长 warmup。
2. 监控梯度范数和 loss spike。
3. 必要时从线性缩放退回到次线性缩放。
4. 结合优化器特性做额外稳定化，例如 LARS 或 LAMB。

所以，“线性缩放”更像第一近似，而不是最终配置。

---

## 代码实现

下面给出一个可运行的 Python 示例，演示如何根据基准 batch 计算目标学习率，并在梯度累积场景下只在真正更新参数时推进 warmup。这里用一个玩具优化器模拟训练过程。

```python
from dataclasses import dataclass

@dataclass
class Config:
    base_batch: int = 32
    new_batch: int = 256
    base_lr: float = 0.1
    warmup_steps: int = 1000
    accum_steps: int = 4
    total_micro_steps: int = 20

class ToyOptimizer:
    def __init__(self, lr: float):
        self.lr = lr
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def zero_grad(self):
        pass

def scaled_lr(base_lr: float, base_batch: int, new_batch: int) -> float:
    scale = new_batch / base_batch
    return base_lr * scale

def warmup_lr(target_lr: float, update_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return target_lr
    if update_step >= warmup_steps:
        return target_lr
    return target_lr * (update_step / warmup_steps)

cfg = Config()
target_lr = scaled_lr(cfg.base_lr, cfg.base_batch, cfg.new_batch)
opt = ToyOptimizer(lr=0.0)

update_steps = 0
lr_history = []

for micro_step in range(1, cfg.total_micro_steps + 1):
    # 假装这里完成了一次 forward/backward
    should_update = (micro_step % cfg.accum_steps == 0)

    if should_update:
        lr = warmup_lr(target_lr, update_steps, cfg.warmup_steps)
        opt.lr = lr
        opt.step()
        opt.zero_grad()
        lr_history.append(lr)
        update_steps += 1

assert target_lr == 0.8
assert len(lr_history) == cfg.total_micro_steps // cfg.accum_steps
assert lr_history[0] == 0.0
assert lr_history[1] == 0.8 / 1000
assert opt.step_count == 5

print("target_lr =", target_lr)
print("lr_history =", lr_history)
```

这段代码体现了两个关键点：

1. 学习率缩放看的是“新批量 / 基准批量”的比例。
2. 如果使用梯度累积，应当以“参数更新步”为单位推进 warmup，而不是每个 micro step 都推进一次，否则 warmup 会比预期结束得更早。

如果把它翻成更接近训练框架的伪代码，结构通常是这样：

```python
base_lr = 0.1
base_batch = 32
global_batch = world_size * per_device_batch * accum_steps
target_lr = base_lr * (global_batch / base_batch)

update_step = 0
for micro_step, batch in enumerate(loader, start=1):
    loss = model(batch) / accum_steps
    loss.backward()

    if micro_step % accum_steps == 0:
        if update_step < warmup_steps:
            lr = target_lr * (update_step / warmup_steps)
        else:
            lr = target_lr

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.step()
        optimizer.zero_grad()
        update_step += 1
```

这里最容易写错的是 `global_batch` 的定义。一定要确认它是否已经包含 `world_size` 和 `accum_steps`，否则缩放倍数会被算错一层或两层。

---

## 工程权衡与常见坑

批量缩放不是单参数调优，而是一个联动系统。至少要同时看 `batch size`、`learning rate`、`warmup`、`gradient accumulation` 和 `optimizer`。

常见坑可以先看这个表：

| 场景 | 表面现象 | 根因 | 常见修正 |
|---|---|---|---|
| 只放大 batch，不放大 lr | 吞吐提升但 loss 降得慢 | 更新过于保守 | 按比例放大学习率 |
| 放大 lr，不做 warmup | 早期 loss spike，甚至 NaN | 高曲率区不稳定 | 增加 warmup |
| 用梯度累积但不按等效批量算 lr | 训练很稳但很慢 | 忽略更新频率变化 | 按等效批量估算缩放比 |
| 累积很多步后直接超大 lr | 后期抖动明显 | 已超稳定区 | 延长 warmup 或减小缩放强度 |
| 套用线性缩放到所有模型 | 某些模型退化 | 噪声结构或优化器不同 | 改用 sqrt 缩放或自适应方案 |

### 坑 1：把“稳定”误判成“合理”

例如 4 卡训练，每卡 `batch=512`，梯度累积 `4` 步，则：

$$
B_{\text{eff}} = 4 \times 512 \times 4 = 8192
$$

如果基准配置是 `B=512, lr=0.1`，那按线性缩放，目标学习率应是：

$$
0.1 \times \frac{8192}{512} = 1.6
$$

但有人为了求稳，仍然坚持用 `0.1`。结果通常不是“更好”，而是“更慢”。因为你每次更新已经用到了 16 倍样本，但更新幅度没有同步增加，优化器会显得非常保守。

### 坑 2：warmup 太短

把学习率从 0.1 拉到 0.8，只给 50 步 warmup，形式上“有 warmup”，实际上等于没有。warmup 的长度通常应跟以下因素一起看：

- 放大倍数有多大
- 模型是否深
- 初始化是否激进
- 优化器是否对大 lr 敏感
- 训练前期 loss 是否已经有明显尖峰

经验上，放大倍数越大，warmup 越不能省。

### 坑 3：把梯度累积当成完全等价的大 batch

梯度累积可以得到相同的等效样本数，但它降低了参数更新频率。对白话解释就是：模型更久才动一次参数。因此它在 wall-clock 时间、优化路径、归一化统计、调度器步数定义上，往往不与“真大 batch”完全一致。你不能只照抄 batch 缩放公式，还要检查 scheduler 是按 micro step 还是 update step 工作。

### 坑 4：只看 loss，不看梯度范数和稳定边界

大 batch 配大 lr 时，loss 未必立刻炸，有时先表现为：

- 梯度范数持续增大
- 激活值范围漂移
- validation 指标波动变大
- 后期泛化变差

所以工程上要一起监控：

- `train loss`
- `grad norm`
- `learning rate`
- `update step`
- `overflow / NaN`
- `validation metric`

---

## 替代方案与适用边界

线性缩放不是唯一选择。它适合“批量放大后噪声明显下降，而且模型在更大学习率下仍能保持稳定”的情况。一旦这个前提不成立，就要换策略。

| 策略 | 公式或做法 | 适用情境 | 主要代价 |
|---|---|---|---|
| 线性缩放 | $\eta_{\text{new}}=k\eta_{\text{base}}$ | 中等到较大 batch 扩展，系统较稳定 | 依赖 warmup，易踩稳定边界 |
| 平方根缩放 | $\eta_{\text{new}}=\sqrt{k}\eta_{\text{base}}$ | 扩 batch 后稳定性差、曲率高 | 提速不如线性明显 |
| LARS | 按层自适应缩放 | 超大 batch 的视觉训练常见 | 额外超参，行为更复杂 |
| LAMB | 层级自适应 + Adam 类方法 | 超大 batch 的 Transformer 训练 | 实现和调参复杂 |
| 不激进扩 lr | 保持较小 lr，延长训练 | 稳定优先、资源允许 | 总训练时长更长 |

一个典型边界例子是：batch 从 32 扩到 64，只有 2 倍。如果监控发现 Hessian 最大特征值对应的稳定上界已经接近当前学习率，那么直接把 lr 从 0.1 拉到 0.2 可能并不划算。这时更稳妥的方案反而是：

$$
\eta_{\text{new}} = \sqrt{2}\times 0.1 \approx 0.141
$$

再配一个较短 warmup。它牺牲一点理论吞吐收益，换来更稳定的优化过程。

还有一种常见边界是“内存允许扩 batch，但泛化开始变差”。大 batch 会降低梯度噪声，而梯度噪声有时本身就带一点正则化效果。此时不应只靠继续放大学习率硬顶，可以考虑：

- 增强数据扰动，如 mixup
- 提高权重衰减或其他正则
- 延长训练周期
- 调整学习率衰减策略
- 控制批量不要无限放大

所以最终判断标准不是“规则有没有套上”，而是“吞吐、稳定性、收敛速度、最终指标”是否同时成立。

---

## 参考资料

- Michael Brenndoerfer, *Learning Rate Warmup for Large Batch Training*：讨论线性缩放、平方根缩放与 warmup 的配套关系。  
- Emergent Mind, *Warmup Strategy: Accelerating Convergence*：讨论 warmup、曲率、catapult 与稳定边界。  
- Continuum Labs, *Gradient Accumulation*：解释梯度累积、等效批量与更新频率的区别。
