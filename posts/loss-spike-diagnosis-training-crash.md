## 核心结论

Loss spike 指训练过程中 loss 在单个 step 或少数几个 step 内突然增大到历史稳定区间的数十倍到数个数量级。它不是“正常波动”，而是训练过程在数值、优化器状态或数据输入上出现异常的信号。

一个直接判定式是：

$$
spike_t = \frac{L_t}{median(L_{t-n:t-1})}
$$

其中 $L_t$ 是第 $t$ 个 step 的 loss，$median(L_{t-n:t-1})$ 是前 $n$ 个 step 的 loss 中位数。工程上如果这个比值远大于 1，例如从 `0.8` 跳到 `96.0`，也就是 `120x`，就应该按 spike 处理，而不是把它当作训练噪声。

新手版理解：前面 100 个 step 的 loss 一直在 `0.8` 左右，第 101 个 step 突然跳到 `96.0`，这不是“模型学得快”，而是训练过程某个环节失稳了。

诊断顺序不要只看 loss。优先检查 `grad_norm`、`inf` / `nan`、attention score、logit 分布、异常 batch 和样本 id。常见原因包括学习率过大、梯度爆炸、数据异常、混合精度溢出。恢复策略通常是回滚到最近稳定 checkpoint，降低学习率后重启，并继续监控异常指标。

| 症状 | 可能原因 | 优先排查项 |
|---|---|---|
| loss 突然放大几十倍以上 | 学习率过大或梯度爆炸 | `grad_norm`、学习率曲线、optimizer 状态 |
| loss 变成 `nan` 或 `inf` | 混合精度溢出或非法数值 | `found_inf`、loss scale、AMP 日志 |
| 某个 batch 后 loss 突然异常 | 数据污染或极端样本 | `batch_id`、`sample_hash`、输入长度、标签 |
| logit 极大或 attention 极端集中 | Transformer 内部数值异常 | `max|logit|`、attention score 分布 |
| 回滚后仍频繁 spike | 系统性训练配置问题 | 学习率、精度策略、数据管线、裁剪阈值 |

---

## 问题定义与边界

Loss spike 的定义是：loss 在单个 step 或少数几个 step 内突然放大到稳定区间的数十倍到数个数量级。它强调“突然变化”和“偏离历史稳定区间”。

它和“高 loss”不同。高 loss 可能只是模型尚未收敛；spike 则表示训练曾经处于相对稳定状态，随后突然失控。

新手版理解：如果训练一开始 loss 就一直是 `10`，这叫“模型还没学会”；如果一直是 `0.8`，突然某一步变成 `120`，这才是 spike。

边界也要分清。验证集 loss 缓慢上升，不一定是 spike，可能是过拟合。训练集 loss 在一个 batch 上瞬间暴涨，更符合 spike。

| 现象 | 典型表现 | 是否属于 loss spike |
|---|---|---|
| 正常波动 | `0.78 -> 0.82 -> 0.76` | 通常不是 |
| 训练未收敛 | 从一开始就长期维持在高位 | 通常不是 |
| loss spike | `0.8 -> 96.0 -> 120.0` | 是 |
| 过拟合 | 训练 loss 降低，验证 loss 缓慢上升 | 通常不是 |
| 数据异常触发 | 某个 batch 后 loss 突然暴涨 | 通常是 |

一个简单的时间序列示意：

```text
loss
120 |                         *
100 |                       *
 80 |
 60 |
 40 |
 20 |
  1 | *  *  *  *  *  *  *
    +----------------------------
      1  2  3  4  5  6  7  8 step
```

前面多个 step 在稳定区间，第 8 个 step 突然跳高，这才是诊断重点。

| 术语 | 白话解释 | 工程含义 |
|---|---|---|
| $L_t$ | 第 $t$ 步的损失值 | 当前 batch 的训练目标误差 |
| `grad_norm` | 梯度整体大小 | 参数即将被更新的强度 |
| `found_inf` | 是否发现无穷大或非法数值 | AMP 用来判断是否跳过更新 |
| `loss scale` | 把 loss 放大的比例 | 混合精度中用于减少 FP16 下溢 |
| `checkpoint` | 训练快照 | 恢复模型、优化器和调度器状态的依据 |

---

## 核心机制与推导

Loss spike 往往来自四类机制：学习率过大、梯度爆炸、异常数据、混合精度数值溢出。

学习率是每次参数更新的步长。学习率过大时，同样的梯度会造成过大的参数移动。参数一旦被推离原来的稳定区域，模型输出会突然偏移，loss 立刻升高。

梯度是 loss 对参数的导数，表示“为了降低 loss，参数应该往哪个方向移动”。梯度爆炸指梯度范数突然变得极大，导致一次更新幅度过大。

新手版理解：某个 batch 把参数更新“推歪”了，下一步模型输出就会严重偏离，loss 立刻升高。

Transformer 场景中，还要看 attention。attention 是模型在不同 token 之间分配关注权重的机制。它的核心计算可以写成：

$$
A = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

其中 $Q$、$K$、$V$ 分别是 query、key、value，$d_k$ 是 key 的维度。超长样本或异常输入可能让 $\frac{QK^T}{\sqrt{d_k}}$ 数值过大，softmax 后 attention 分布接近极端 one-hot，也就是几乎只看某一个 token。随后 logit 可能异常放大，loss 冲高。

梯度裁剪是限制梯度范数的保护机制。设全局梯度范数为 $g$，裁剪阈值为 $\tau$，常见形式是：

$$
g \leftarrow g \cdot min(\frac{\tau}{g}, 1)
$$

当 $g \le \tau$ 时不改变梯度；当 $g > \tau$ 时把梯度缩小到阈值附近。

混合精度训练通常使用 FP16 或 BF16 加速。AMP 是 automatic mixed precision，意思是自动混合精度。FP16 的数值范围更小，所以容易出现 `inf` 或 `nan`。为了避免梯度太小被舍入为 0，AMP 会使用 loss scale：

$$
\tilde{L} = sL
$$

其中 $s$ 是 loss scale。如果 AMP 检测到梯度里有 `inf` / `nan`，`scaler.step()` 会跳过 `optimizer.step()`，避免用坏梯度更新参数。

| 机制 | 推导链路 | 典型观测 |
|---|---|---|
| 学习率过大 | 学习率过高 -> 参数更新幅度过大 -> 输出偏移 -> loss spike | `lr` 高，`grad_norm` 未必极端 |
| 梯度爆炸 | 梯度范数飙升 -> 更新不稳定 -> loss spike | `grad_norm` 同步飙升 |
| 脏样本 | 输入分布异常 -> logit/attention 异常 -> loss spike | 特定 `batch_id` 反复触发 |
| FP16 溢出 | 数值超出范围 -> `inf` / `nan` -> 跳步或坏更新风险 | `found_inf=True`，loss scale 下降 |

玩具例子：前 100 步 loss 大约是 `0.8`，第 101 步变成 `96.0`，同时 `grad_norm` 从 `0.7` 变成 `54.0`。如果裁剪阈值 $\tau=1$，裁剪系数约为：

$$
\frac{1}{54} \approx 0.0185
$$

这说明该步梯度需要被强烈压缩，否则一次更新就可能破坏模型状态。

真实工程例子：训练一个 Transformer 语言模型时，某个超长样本进入 batch，attention score 的最大值显著高于平时，attention 分布接近 one-hot，`max|logit|` 也同步升高。随后 loss 从 `1.x` 冲到 `1e2` 以上。此时排查重点不是“模型突然不会了”，而是定位该 batch 的样本 hash、输入长度、attention score、`grad_norm` 和 AMP 状态。

---

## 代码实现

代码层面要实现三件事：监控、告警、恢复。不是“看到 loss 大了就手动重启”，而是自动记录 `loss`、`grad_norm`、`found_inf`、`batch_id`、`sample_hash` 和 checkpoint 信息。出问题后能回滚到最近稳定点，而不是从头猜。

下面是一个可运行的 Python 玩具例子，用历史中位数检测 spike：

```python
from statistics import median

def detect_loss_spike(losses, window=5, threshold=10.0):
    alerts = []
    for i in range(window, len(losses)):
        baseline = median(losses[i-window:i])
        ratio = losses[i] / max(baseline, 1e-12)
        if ratio >= threshold:
            alerts.append((i, losses[i], baseline, ratio))
    return alerts

losses = [0.81, 0.79, 0.83, 0.80, 0.82, 0.78, 96.0]
alerts = detect_loss_spike(losses, window=5, threshold=10.0)

assert len(alerts) == 1
assert alerts[0][0] == 6
assert alerts[0][3] > 100
```

真实训练代码里，需要正确处理 AMP、梯度裁剪和 checkpoint 保存顺序。核心伪代码如下：

```python
loss = model(batch)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=τ)
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

注意顺序：先 `scaler.unscale_(optimizer)`，再裁剪梯度。否则裁剪的是被 loss scale 放大后的梯度，数值含义不可靠。

更完整的训练步骤可以写成：

```python
import torch

def train_step(model, batch, optimizer, scheduler, scaler, max_norm):
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(batch["input_ids"])
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
    )

    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    new_scale = scaler.get_scale()

    found_inf = new_scale < old_scale
    if not found_inf:
        scheduler.step()

    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm.detach().cpu()),
        "found_inf": bool(found_inf),
        "lr": optimizer.param_groups[0]["lr"],
        "batch_id": batch.get("batch_id"),
        "sample_hash": batch.get("sample_hash"),
    }
```

checkpoint 不能只保存模型权重。恢复训练时，optimizer、scheduler、AMP scaler、step、样本元数据都要保存，否则恢复后的训练状态不一致。

```python
def save_checkpoint(path, model, optimizer, scheduler, scaler, step, metadata):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "metadata": metadata,
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt.get("metadata", {})
```

| 日志字段 | 作用 |
|---|---|
| `step` | 定位异常发生在哪一步 |
| `loss` | 判断是否出现 spike |
| `grad_norm` | 判断是否伴随梯度爆炸 |
| `max|logit|` | 判断输出是否异常放大 |
| `found_inf` | 判断 AMP 是否发现非法数值 |
| `batch_id` / `sample_hash` | 定位异常数据 |
| `lr` | 判断学习率是否过高或调度异常 |
| `checkpoint_id` | 支持回滚到稳定点 |

---

## 工程权衡与常见坑

最常见的误区是把问题直接归咎于数据，忽略学习率、精度设置和优化器状态。数据确实可能触发 spike，但学习率过大、梯度裁剪缺失、AMP 配置不当，也同样常见。

另一个高频问题是恢复不完整。新手版理解：模型参数像“位置”，optimizer 像“惯性”，scheduler 像“速度表”，AMP scaler 像“数值保护开关”。只恢复位置，不恢复其他状态，训练会重新失衡。

| 常见坑 | 问题后果 | 正确做法 |
|---|---|---|
| 只看 loss，不看 `grad_norm` | 看不到梯度爆炸前兆 | 同步记录 loss 和全局梯度范数 |
| 只存模型，不存 optimizer/scheduler/scaler | 恢复后训练状态不一致 | checkpoint 保存完整训练状态 |
| 在 `unscale` 前做梯度裁剪 | 裁剪的是放大梯度 | 先 `unscale_`，再 `clip_grad_norm_` |
| 没有样本级日志 | 无法定位坏样本 | 记录 `batch_id` 和 `sample_hash` |
| 过度怀疑脏数据，忽略学习率 | 反复过滤数据但问题仍在 | 同时检查 lr、AMP、梯度范数 |
| 每次 spike 都继续训练 | 可能污染后续 checkpoint | 回滚到最近稳定 checkpoint |

恢复流程可以固定成清单：

1. 回滚到最近一个 loss、`grad_norm`、AMP 状态都正常的 checkpoint。
2. 定位异常 batch 和样本 hash，检查输入长度、标签、token 分布和数据格式。
3. 如果确认是坏样本，过滤或修复样本。
4. 如果 `grad_norm` 同步飙升，启用或收紧梯度裁剪。
5. 如果伴随 `found_inf=True`，检查 AMP、loss scale、FP16/BF16 策略。
6. 将学习率下调 `2x` 到 `10x` 后重启训练。
7. 重启后继续监控 loss、`grad_norm`、`max|logit|`、attention 统计量和 checkpoint 状态。

工程上要接受一个事实：loss spike 的根因可能不是单点问题，而是多个因素叠加。例如学习率偏高时，正常 batch 还能勉强训练；一旦遇到超长样本或异常标签，就触发梯度爆炸。只修数据或只降学习率，都可能不彻底。

---

## 替代方案与适用边界

不是所有 loss 波动都要用同一种处理方式。必须区分“短期数值异常”和“长期优化困难”。训练稳定性方案通常是组合使用，而不是单独依赖某一种。

新手版理解：如果只是偶发的单 batch 异常，回滚和过滤坏样本可能够用；如果频繁出现 spike，说明训练配置本身有系统性问题，需要调学习率或精度策略。

| 方案 | 适合场景 | 不能解决的问题 |
|---|---|---|
| 梯度裁剪 | 梯度爆炸、偶发大梯度 | 不能修复错误标签或坏数据根因 |
| 降低学习率 | 参数更新过猛、频繁不稳定 | 可能放慢收敛 |
| 过滤异常样本 | 数据污染、极端输入、错误标签 | 需要样本级日志支持 |
| 关闭或调整 AMP | FP16 溢出、频繁 `found_inf` | 会影响速度和显存 |
| 更密集 checkpoint | 需要快速回滚 | 不能阻止 spike 发生 |
| attention/logit 监控 | Transformer 内部异常 | 需要额外日志和存储开销 |

决策可以按症状分流：

| 观测结果 | 优先动作 |
|---|---|
| 单次 spike | 先回滚到最近稳定 checkpoint |
| 频繁 spike | 查学习率、精度策略、数据分布 |
| 伴随 `inf` / `nan` | 查 AMP、loss scale、非法数值来源 |
| 伴随 `grad_norm` 飙升 | 查梯度爆炸，启用或调整梯度裁剪 |
| 只在特定 batch 出现 | 查样本 hash、输入长度、标签和数据格式 |
| 回滚后同一步复现 | 优先怀疑数据或确定性数值问题 |
| 不固定 step 随机出现 | 优先怀疑学习率、AMP 或分布尾部样本 |

适用边界要明确。梯度裁剪能缓解爆炸，但不能修复错误标签、极端异常样本或长期不合理的学习率设置。降低学习率能减少更新幅度，但如果数据管线会产生非法 token 或空样本，学习率再低也不能从根上解决。关闭 AMP 能减少 FP16 溢出，但会增加显存和训练时间。

因此，稳定训练的实践通常是组合策略：学习率 warmup、合理 scheduler、梯度裁剪、AMP 状态监控、样本级日志、周期性 checkpoint、异常数据审查。单个工具只能覆盖一类风险，不能替代完整诊断链路。

---

## 参考资料

| 资料 | 用途 |
|---|---|
| PyTorch `clip_grad_norm_` | 用于说明梯度裁剪的正确位置和用法 |
| PyTorch AMP examples | 用于说明 AMP 的 `scaler.scale()`、`scaler.step()`、`scaler.update()` 机制 |
| PyTorch Saving and Loading Models | 用于说明恢复训练时需要保存状态 |
| Attention Is All You Need | 用于说明 Transformer attention 公式来源 |
| PyTorch Lightning checkpointing | 用于参考工程框架中的 checkpoint 状态保存 |

1. [PyTorch: torch.nn.utils.clip_grad_norm_](https://docs.pytorch.org/docs/2.9/generated/torch.nn.utils.clip_grad.clip_grad_norm_.html)  
用于说明梯度裁剪 API 的行为、参数和返回值。

2. [PyTorch: Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)  
用于说明 AMP 中 `scaler.scale()`、`unscale_()`、`step()` 和 `update()` 的推荐顺序。

3. [PyTorch: Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)  
用于说明模型状态保存与加载的基本方式。

4. [Vaswani et al., Attention Is All You Need](https://arxiv.org/pdf/1706.03762)  
用于说明 scaled dot-product attention 的公式来源。

5. [PyTorch Lightning: Saving and loading checkpoints](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)  
用于参考工程训练框架如何保存模型、优化器、scheduler 和训练状态。
