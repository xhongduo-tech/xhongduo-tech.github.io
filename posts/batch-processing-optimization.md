## 核心结论

批处理优化的目标，不是单纯把 `batch size` 调大，而是让一次前向与反向里，尽可能多的计算都落在“有效 token”上。动态批处理和混合精度正好分别解决两个不同问题：

1. 动态批处理解决长度不齐造成的 padding 浪费。它按 token 预算组批，而不是按样本数死板组批。
2. 混合精度解决显存与算力利用率问题。它让大部分矩阵乘法用 FP16 或 BF16 跑，同时把容易数值不稳定的部分留在 FP32。
3. AMP 是自动混合精度，白话讲就是框架帮你决定哪些算子降精度、何时做 loss scaling，减少手写精度控制代码。

最实用的组合是：

$$
B_i \times L_i \approx T_{\text{target}}
$$

其中 $B_i$ 是第 $i$ 个微批的样本数，$L_i$ 是该批对齐后的序列长度，$T_{\text{target}}$ 是目标 token 预算。先用这个规则把每个微批装满，再用 AMP 节省显存，通常就能把吞吐抬高一截。

一个新手版玩具例子：

- 固定批处理：每批 8 条，最长 128 token，那么 8 条都要 pad 到 128。
- 动态批处理：短样本用 `8 x 64`，长样本用 `4 x 128`，两批都接近 512 token。
- 结果：短样本不再被长样本拖着补无效 padding。

| 方案 | 组批规则 | padding 浪费 | 显存利用 | 吞吐潜力 |
| --- | --- | --- | --- | --- |
| 固定样本数 batch | 每批固定 8 条 | 高 | 不稳定 | 中 |
| 动态批处理 | 每批固定 token 预算 | 低 | 更稳定 | 高 |
| 动态批处理 + AMP | token 预算 + 低精度算子 | 更低 | 更高 | 最高 |

---

## 问题定义与边界

先定义三个核心问题。

“padding”是把短序列补齐到同一长度的占位 token。它不提供真实训练信号，但会消耗算力和显存。

“混合精度”是同时使用 16 位与 32 位浮点数训练。白话讲，就是把适合快算的地方交给低精度，把容易不稳定的地方留给高精度。

“loss scaling”是先把 loss 放大再反传，避免 FP16 梯度太小直接变成 0。白话讲，就是给很小的梯度先“放大再计算”。

固定 batch 的问题通常有三类：

| 问题 | 现象 | 根因 | 后果 |
| --- | --- | --- | --- |
| padding waste | 短序列被补到很长 | 按样本数组批 | 有效 token 比例下降 |
| memory headroom | 显存明明没满却放不进更多样本 | 最长样本决定整批形状 | 吞吐上不去 |
| precision 风险 | loss NaN、梯度为 0 | FP16 动态范围窄 | 不收敛或精度掉点 |

边界也要说清楚。

动态批处理不是 sequence packing。packing 是把多条序列拼成一条长序列，padding 更少，但索引、mask、attention 处理更复杂。动态批处理只是把长度接近的样本放一起，工程改造小很多。

混合精度也不是所有设备都一样：

- GPU 常见是 FP16 或 BF16，重点看 Tensor Core 对齐。
- TPU 更偏向 BF16，很多转换由 XLA 自动完成。
- NPU 通常也支持 AMP，但算子白名单、loss scale、精度回退策略与 GPU 不完全相同。

---

## 核心机制与推导

动态批处理的核心不是“动态改 batch size”这句话，而是“固定 token 预算”。

设目标 token 数为 $T_{\text{target}}$，某一批对齐后长度为 $L_i$，则该批样本数近似取为：

$$
B_i = \left\lfloor \frac{T_{\text{target}}}{L_i} \right\rfloor
$$

如果硬件要求长度对齐，实际参与计算的长度应写成：

$$
L_i' = \left\lceil \frac{L_i}{a} \right\rceil \times a
$$

其中 $a$ 是对齐粒度，例如 GPU 上常见取 8，TPU 常见关注 8 或 128 的可整除性，某些 NPU 任务会偏好 power-of-two 形状。于是更实际的公式是：

$$
B_i \times L_i' \approx T_{\text{target}}
$$

这套机制为什么有效？因为同一批里长度更接近，`max_length - mean_length` 变小，padding 比例自然下降。算子处理的是更密的张量，反向传播也少做了无效 token 的计算。

AMP 的核心则是两步：

1. 前向和大部分线性代数算子在低精度下执行。
2. 参数主副本、梯度累加、部分归约操作保留在 FP32。

在 FP16 训练里，loss scaling 常写成：

$$
\tilde{L} = s \cdot L
$$

先对放大后的 $\tilde{L}$ 反向传播，得到放大后的梯度 $\tilde{g}$，再恢复：

$$
g = \frac{\tilde{g}}{s}
$$

如果检测到 overflow，说明梯度太大，缩放因子 $s$ 要减小；如果长期稳定，就可以逐步增大 $s$。这就是动态 loss scaling 的基本逻辑。

玩具例子如下。设 $T_{\text{target}}=512$，原始长度有 48、60、100 三类：

- 对齐到 8 后，48 仍是 48，60 会变成 64，100 会变成 104。
- 短序列可以组成 `8 x 64 = 512`。
- 长序列可以组成 `4 x 104 = 416`，再混入相近长度样本，接近预算上限。

真实工程例子可以看 NeMo-RL 的做法：先在 chunk 内按长度排序，再按 `max_tokens_per_microbatch` 分组，并对长度做硬件对齐。这样即使不用 packing，也能显著降低 padding 开销，在多卡场景里也更容易把不同设备的负载压到接近水平。

流程可以理解为：

`排序 -> 分桶 -> 对齐 -> 按 token 预算切微批 -> autocast 前向 -> GradScaler 反传`

---

## 代码实现

下面先给一个可运行的 Python 玩具实现，只演示“按 token 预算动态组批”和“padding 浪费统计”。

```python
from math import ceil

def round_up(x, multiple):
    return ceil(x / multiple) * multiple

def dynamic_batches(lengths, token_budget, align=8):
    lengths = sorted(lengths)
    batches = []
    i = 0
    while i < len(lengths):
        current = []
        max_len = 0
        while i < len(lengths):
            candidate_max = round_up(max(max_len, lengths[i]), align)
            if (len(current) + 1) * candidate_max <= token_budget:
                current.append(lengths[i])
                max_len = candidate_max
                i += 1
            else:
                break
        batches.append((current, max_len))
    return batches

def padding_ratio(batch):
    seqs, padded_len = batch
    used = sum(seqs)
    allocated = len(seqs) * padded_len
    return 1 - used / allocated

lengths = [48, 50, 60, 61, 62, 100, 101, 110, 32, 33, 34, 35]
batches = dynamic_batches(lengths, token_budget=512, align=8)

assert len(batches) >= 2
assert all(len(seqs) * padded_len <= 512 for seqs, padded_len in batches)
assert round_up(61, 8) == 64

avg_padding = sum(padding_ratio(b) for b in batches) / len(batches)
assert avg_padding < 0.35
print(batches)
print("avg_padding=", round(avg_padding, 4))
```

训练循环里，动态批处理通常放在 `DataLoader` 或自定义 `collate_fn` 一侧；AMP 放在训练步内部。PyTorch 结构可以写成：

```python
# pseudo code
scaler = torch.amp.GradScaler("cuda")

for microbatch in dynamic_batch_iterator:
    input_ids, labels = microbatch
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids)
        loss = loss_fn(logits, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

不同平台关注点如下：

| 平台 | 常见低精度 | API/机制 | 重点对齐 |
| --- | --- | --- | --- |
| NVIDIA GPU | FP16 / BF16 | `torch.autocast` + `GradScaler` | 常看 8 的倍数，匹配 Tensor Core |
| TPU | BF16 | XLA/框架自动转换 + mixed precision policy | 常看 batch 与特征维对 8/128 |
| Ascend NPU | FP16 / BF16 | PyTorch AMP 或 Ascend 适配方案 | 常看规则化 shape 与 loss scale 稳定性 |

如果你在 TensorFlow/Keras 中做同样的事，思路不变：数据侧维护 `microbatch_token_count`，框架侧设置 mixed precision policy。区别只在 API，不在原理。

---

## 工程权衡与常见坑

批处理优化不是“越大越好”，而是三方平衡：显存限制、梯度噪声、计算效率。

| 坑 | 影响 | 解决方式 |
| --- | --- | --- |
| 只调大样本数，不看 token 数 | 长序列批次直接 OOM | 改成 token budget 驱动 |
| 长度未按硬件对齐 | 吞吐没有预期提升 | GPU 常按 8，对 TPU/NPU 看官方建议 |
| FP16 直接全量启用 | 梯度下溢或 loss NaN | 用 AMP，不手动全局 `.half()` |
| 动态批次波动太大 | 学习率与梯度统计不稳定 | 控制 `T_target` 波动，必要时做梯度累积 |
| 把 dynamic batching 和 packing 同时开 | 逻辑冲突，调试困难 | 二选一 |
| 监控只看 samples/s | 被短序列“虚高”吞吐误导 | 重点看 tokens/s 与有效 token 比例 |

一个常见误区是：AMP 省下显存后，立刻把 batch size 翻 4 倍，但学习率、梯度裁剪、梯度累积策略全不改。这样虽然吞吐上去了，优化器统计分布却变了，训练可能更不稳。更稳妥的做法是逐步扩大 `T_target`，每次观察：

- `tokens/s`
- 显存峰值
- overflow 次数
- 验证集 loss 是否异常抖动

真实工程里，NeMo-RL 文档明确指出动态批处理和 packing 是互斥的；同时它也强调动态批处理通过按长度分组来减少 padding。对很多已经上线的 SFT 流程，这种方式改造更小。另一方面，GPU 上若长度、隐藏维、词表维完全不考虑 Tensor Core 对齐，理论上的 AMP 收益往往拿不满。

---

## 替代方案与适用边界

三种常见路线可以直接比较：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 固定 batch + FP32 | 最稳、最简单 | 慢、显存占用高 | 基线验证、小模型 |
| 动态批处理 + AMP | 改造成本适中、收益高 | 需要处理动态 shape 与监控 | 通用 SFT、微调、长度分布不均的数据 |
| sequence packing + AMP | 有效 token 比例最高 | 实现复杂，attention/mask 改动大 | 极长序列、padding 特别严重的场景 |

可以用一个简单决策逻辑：

1. 数据长度差异大，但现有训练代码不想大改，先上动态批处理。
2. 显存已经吃紧，再开 AMP 或 BF16。
3. 如果你的模型和 attention 内核已经支持 packing，再考虑从动态批处理升级到 packing。
4. 如果混合精度反复 overflow，且调 loss scale、算子白名单后仍不稳定，退回全 FP32 或优先使用 BF16。

新手版类比可以这样记：

- 动态批处理像“自动调袋子大小”。
- packing 像“把多件商品塞进同一个大箱子”。
- AMP 像“把重货用叉车搬，但账本仍用精确记账”。

这三个动作里，动态批处理 + AMP 通常是收益和复杂度最平衡的一组。

---

## 参考资料

1. NVIDIA, *Train With Mixed Precision*：解释混合精度、主权重、loss scaling 与 Tensor Core 原理。  
   https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html

2. PyTorch, *Automatic Mixed Precision package - torch.amp*：`autocast` 与 `GradScaler` 的官方用法。  
   https://docs.pytorch.org/docs/stable/amp

3. PyTorch, *Automatic Mixed Precision examples*：更贴近训练循环的 AMP 示例。  
   https://docs.pytorch.org/docs/stable/notes/amp_examples

4. NVIDIA NeMo-RL, *Sequence Packing and Dynamic Batching*：动态批处理与 sequence packing 的实现边界、配置和约束。  
   https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html

5. TensorFlow, *Mixed precision*：Keras mixed precision policy 的官方说明，包含 GPU 与 TPU 场景。  
   https://www.tensorflow.org/guide/mixed_precision

6. Google Cloud TPU, *Cloud TPU performance guide*：TPU 上 batch、特征维与 8/128 对齐的性能建议。  
   https://cloud.google.com/tpu/docs/performance-guide

7. Google Cloud TPU, *Improve your model's performance with bfloat16*：TPU 上 BF16 与 FP32 累加的实践说明。  
   https://cloud.google.com/tpu/docs/bfloat16

8. 昇腾社区, *混合精度适配简介*：Ascend PyTorch 场景下 AMP 与 APEX 的适配方式。  
   https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/PT_LMTMOG_0026.html

9. 昇腾社区, *混合精度配置选择*：哪些算子更适合保留 FP32，适合做 NPU 侧稳定性排查。  
   https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/trainingmigrguide/LMaccuracy_0018.html
