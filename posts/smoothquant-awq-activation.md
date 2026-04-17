## 核心结论

SmoothQuant 与 AWQ 都在解决同一个核心问题：大语言模型的激活值里存在少量异常大的通道，直接做低比特量化时，这些通道会把整体量化范围拉得很大，导致大部分正常数值被粗糙表示，精度明显下降。

它们的共同思路可以写成一个等价变换：

$$
Y = XW = (X \operatorname{diag}(s)^{-1})(\operatorname{diag}(s)W)
$$

这里的 `scale`（缩放系数）就是一组按通道定义的数，用来把某些通道缩小、再在权重侧补回来。白话说，就是先把“不好量化的激活”压平，再把这部分难度转移给“更容易按通道量化的权重”。

两者的区别在于目标不同：

| 方法 | 主要处理对象 | 关键思想 | 常见比特 |
|---|---|---|---|
| SmoothQuant | 激活 outlier | 全通道平滑，把激活难度迁移到权重 | INT8 |
| AWQ | 显著权重通道 | 找出对输出最重要的少量通道，优先保护它们 | INT4 / INT3 / INT8 |

玩具例子先看最简单版本：某层两个输入通道的激活最大值分别是 `100` 和 `1`。如果直接统一做 INT8 量化，量化范围必须覆盖 `[-100, 100]`，那么那个幅度只有 `1` 的通道会只占到很少几个量化格子，误差非常大。SmoothQuant 会先把 `100` 对应通道缩小，比如缩到 `10`，再把对应权重放大 10 倍，最终矩阵乘法结果不变，但量化更容易。AWQ 则更进一步，它不平均处理所有通道，而是重点保护那些“对输出贡献特别大”的显著通道。

结论可以压缩成一句话：SmoothQuant 更像“整体平衡激活与权重的动态范围”，AWQ 更像“把有限的量化精度优先分给关键通道”。在无需重训的 PTQ（Post-Training Quantization，训练后量化，指模型训练完成后再做压缩）场景里，两者都是大模型部署的主流方案。

---

## 问题定义与边界

先定义问题。量化的目标是把浮点数映射到低比特整数，比如 INT8 或 INT4。比特越低，存储越省，访存越少，推理越快，但表示精度也越差。真正困难的地方不在“权重能不能量化”，而在“激活是不是有极端值”。

激活是模型在前向计算时中间产生的张量，可以理解为“这一层当前输入的数值表示”。在 LLM 中，很多层的激活分布并不均匀，少数通道会出现非常大的值，也就是 outlier。只要有少数 outlier 存在，统一量化区间就会被它们主导。

例如某一层两个通道的最大绝对值如下：

| 通道 | 激活最大值 | 直接 INT8 的影响 |
|---|---:|---|
| channel 0 | 100 | 决定整体范围必须覆盖 100 |
| channel 1 | 1 | 大部分有效位被浪费，分辨率很差 |

如果使用对称量化，INT8 可表示约 256 个离散等级，那么步长大致是：

$$
\Delta \approx \frac{100}{127}
$$

这意味着第二个通道里大量接近 `1` 或 `0.5` 的值都会被映射到很粗糙的整数格点上，信息损失严重。

这个问题的边界也要说清楚：

| 现象 | 本质 | 影响 |
|---|---|---|
| 激活 outlier | 少数激活通道幅度异常大 | 激活量化误差变大 |
| 显著通道 | 少数通道对输出贡献特别大 | 这些通道一旦量化失真，整体精度掉得快 |
| 权重量化 | 权重通常更稳定、可按通道处理 | 比激活更适合承接量化难度 |

SmoothQuant 和 AWQ 主要适用于以下边界：

1. 目标是推理部署，不是训练。
2. 希望做 PTQ，而不是重新训练或长时间微调。
3. 能接受离线标定，即拿少量样本跑一遍统计信息。
4. 不希望运行时引入新算子，最好把缩放预先融合到权重里。
5. 主要面向线性层、注意力投影层、MLP 层这类以矩阵乘法为主的模块。

不适合的情况也很明确：如果你可以重新训练模型，量化感知训练通常会更稳；如果硬件根本不支持低比特算子，仅仅压缩权重未必能换来端到端加速。

---

## 核心机制与推导

先从 SmoothQuant 讲起。设输入激活为 $X \in \mathbb{R}^{n \times d}$，权重为 $W \in \mathbb{R}^{d \times m}$，输出为：

$$
Y = XW
$$

如果定义一个按输入通道的对角缩放矩阵 $\operatorname{diag}(s)$，则有：

$$
Y = (X \operatorname{diag}(s)^{-1})(\operatorname{diag}(s) W)
$$

这是严格等价的，因为中间乘了一个单位变换。关键在于，选择合适的 $s$ 之后，新的激活 $X' = X\operatorname{diag}(s)^{-1}$ 会更平滑，更适合量化。

SmoothQuant 给出的典型选择是：

$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}
$$

其中：

- $X_j$ 表示第 `j` 个输入通道上的激活
- $W_j$ 表示与该通道对应的权重列或组
- $\alpha \in [0,1]$ 是平衡参数

白话解释：如果某个通道激活很大、权重不大，那么 $s_j$ 会变大，于是这个通道的激活会被除以更大的数，异常值被压下去；同时对应权重会被乘大，保证乘法结果不变。

### SmoothQuant 的玩具例子

假设两个通道：

- 激活最大值：`[100, 1]`
- 权重最大值：`[1, 1]`
- 取 $\alpha = 0.5$

则：

$$
s_1 = \frac{100^{0.5}}{1^{0.5}} = 10,\quad s_2 = \frac{1^{0.5}}{1^{0.5}} = 1
$$

变换后：

- 新激活最大值约为 `[10, 1]`
- 新权重最大值约为 `[10, 1]`

如果原来激活量化必须覆盖 `100`，现在只需重点覆盖 `10`，激活侧量化误差会显著下降。代价是第一列权重被放大到 10，但权重通常更容易做 per-channel quantization（按通道量化，指每个通道用自己的 scale），所以这笔账通常划算。

### AWQ 的核心思路

AWQ 的全称是 Activation-aware Weight Quantization，也就是“激活感知的权重量化”。它的观察更细：不是所有通道都同样重要，只有一小部分显著通道决定了大部分输出质量。

“显著通道”可以白话理解为：只要这些通道失真，输出就会明显坏掉；不显著通道粗一点问题没那么大。

AWQ 的目标不是单纯平滑全部激活，而是寻找一个缩放，使量化后的权重在真实激活输入下的误差最小。可写成类似下面的目标：

$$
\arg\min_s \left\|Q(\operatorname{diag}(s)W)\operatorname{diag}(s)^{-1}X - WX \right\|
$$

这里 $Q(\cdot)$ 表示量化算子。实际工程实现通常会进一步简化成对每个通道的格点搜索。核心步骤是：

1. 用少量标定数据收集激活统计。
2. 根据激活幅度识别最重要的少量通道，常见比例是 `0.1%` 到 `1%`。
3. 对这些通道尝试不同的 scale。
4. 选择让输出误差最小的缩放方案。
5. 把缩放融合进权重表示，运行时仍然保持低比特整数计算。

可以把 AWQ 理解成“有偏分配误差预算”。不是让所有通道都平均好，而是优先让关键通道尽量好。

### 为什么两者都有效

量化误差本质上来自有限离散格点。若某通道真实值范围太大，步长就会变粗。把异常大的激活缩回去，本质上是在减小该通道的量化步长：

$$
\Delta_j \propto \frac{\max(|X_j|)}{2^b - 1}
$$

其中 $b$ 是量化比特数。SmoothQuant 通过减小 $\max(|X_j|)$ 降低 $\Delta_j$；AWQ 通过优先保护关键通道，降低关键误差对最终输出的放大效应。

真实工程例子是 TinyChat 系列工作。它把 AWQ 与硬件友好的低比特 kernel 结合，在 Jetson Orin 64GB 这类移动 GPU 上运行 LLaMA-2-70B 的低比特版本。这说明 PTQ 不只是论文里的误差优化，而是真能决定“一个模型能不能塞进设备、能不能达到可用吞吐”的部署问题。

---

## 代码实现

下面先给一个最小可运行的 SmoothQuant 玩具实现。它展示三件事：

1. 等价变换前后输出近似不变。
2. 缩放后激活动态范围被压平。
3. 可以直接据此做离线标定。

```python
import numpy as np

def smoothquant_scale(x_max, w_max, alpha=0.5, eps=1e-8):
    x_max = np.asarray(x_max, dtype=np.float64)
    w_max = np.asarray(w_max, dtype=np.float64)
    return (x_max ** alpha) / (w_max ** (1.0 - alpha) + eps)

def symmetric_quantize(x, num_bits=8):
    qmax = 2 ** (num_bits - 1) - 1
    scale = np.max(np.abs(x)) / max(qmax, 1)
    scale = max(scale, 1e-8)
    q = np.round(x / scale).clip(-qmax - 1, qmax)
    return q.astype(np.int32), scale

def dequantize(q, scale):
    return q.astype(np.float64) * scale

# 玩具输入：2个输入通道，第一通道有明显 outlier
X = np.array([
    [100.0, 1.0],
    [80.0, -1.0],
    [50.0, 0.5],
])

W = np.array([
    [0.02, -0.03],
    [1.20,  0.80],
])

# 原始输出
Y_ref = X @ W

# 统计每个输入通道的最大绝对值
x_max = np.max(np.abs(X), axis=0)
w_max = np.max(np.abs(W), axis=1)

s = smoothquant_scale(x_max, w_max, alpha=0.5)

# 等价变换
S_inv = np.diag(1.0 / s)
S = np.diag(s)
X_s = X @ S_inv
W_s = S @ W

Y_new = X_s @ W_s

# 量化前后应该严格接近
assert np.allclose(Y_ref, Y_new, atol=1e-6)

# 缩放后第一通道动态范围下降
assert np.max(np.abs(X_s[:, 0])) < np.max(np.abs(X[:, 0]))

# 简单模拟激活 INT8 量化
qX, x_scale = symmetric_quantize(X, num_bits=8)
qXs, xs_scale = symmetric_quantize(X_s, num_bits=8)

X_dq = dequantize(qX, x_scale)
Xs_dq = dequantize(qXs, xs_scale)

err_before = np.mean(np.abs((X_dq @ W) - Y_ref))
err_after = np.mean(np.abs((Xs_dq @ W_s) - Y_ref))

print("scale s =", s)
print("activation max before =", x_max)
print("activation max after =", np.max(np.abs(X_s), axis=0))
print("mean abs error before =", err_before)
print("mean abs error after  =", err_after)

assert err_after <= err_before + 1e-6
```

上面代码虽然是玩具版本，但核心流程和真实实现一致：先统计，再求 `s`，再把 `s` 融入权重侧。

### SmoothQuant 的标定伪代码

```python
# calibration stage
for layer in model.linear_layers:
    x_max = collect_activation_max(layer, calib_data)   # per-channel
    w_max = collect_weight_max(layer.weight)            # per-channel
    s = (x_max ** alpha) / (w_max ** (1 - alpha) + eps)

    layer.weight = diag(s) @ layer.weight
    fuse_inverse_scale_to_prev_layer_output(layer.prev, s)
```

这里“融合”很关键。融合的意思是把缩放提前吸收到前一层或当前层参数中，避免运行时真的插入一个 `diag(s)` 矩阵乘法，否则会多出额外算子，抵消量化带来的收益。

### AWQ 的简化伪代码

```python
# awq calibration stage
for layer in model.linear_layers:
    act_stat = collect_activation_stat(layer, calib_data)
    salient_idx = topk_channels(act_stat, ratio=0.01)

    best_scale = ones(num_channels)
    best_error = inf

    for candidate in grid_search(1.0, 1.3, step=0.05):
        scale = ones(num_channels)
        scale[salient_idx] = candidate

        w_scaled = diag(scale) @ layer.weight
        w_q = quantize_int4_per_channel(w_scaled)
        y_err = eval_output_error(w_q, scale, calib_data)

        if y_err < best_error:
            best_error = y_err
            best_scale = scale

    layer.weight = fuse_scale_and_quantize(layer.weight, best_scale)
```

AWQ 与 SmoothQuant 的实现差异可以总结如下：

| 维度 | SmoothQuant | AWQ |
|---|---|---|
| 统计对象 | 激活极值 + 权重极值 | 激活重要性 + 输出误差 |
| scale 计算 | 公式直接给出 | 通过搜索确定 |
| 优化重点 | 平滑所有通道 | 保护显著通道 |
| 常见用途 | W8A8 | W4/W3 更常见 |

真实工程中，代码不会真的构造大对角矩阵，而是直接对通道维做逐元素乘除。原因很简单：对角矩阵只是一种数学表达，真正实现时直接 broadcast 更高效。

---

## 工程权衡与常见坑

第一类权衡是 SmoothQuant 的 $\alpha$。它控制量化难度在激活和权重之间如何分配。

| 参数 | 倾向 | 风险 |
|---|---|---|
| $\alpha \to 0$ | 更少缩放激活 | 激活 outlier 仍然严重 |
| $\alpha \approx 0.5$ | 较均衡 | 通常是安全起点 |
| $\alpha \to 1$ | 大量难度推给权重 | 权重范围膨胀，权重量化变难 |

举个直观例子：

- 若 $\alpha = 0.2$，激活缩放不够，激活 INT8 误差可能仍然偏大。
- 若 $\alpha = 0.8$，激活很好量化，但某些权重通道会被放大很多，权重侧误差反而上升。

第二类权衡是 AWQ 的显著通道 scale。scale 越大，被保护的通道越不容易丢精度，但其余通道能分到的量化范围越少。工程上常见经验是只轻度放大，通常不会无约束地把 scale 拉得很高。

例如当显著通道 scale 拉到 `1.5x` 甚至更高时，经常会出现两个副作用：

1. 非显著通道的误差明显增大。
2. 权重分布被拉长，低比特分桶更粗糙。

因此 AWQ 常配合格点搜索、裁剪和 per-channel quantization 使用，而不是凭经验拍一个 scale。

常见坑主要有这些：

| 常见误区 | 后果 | 规避方式 |
|---|---|---|
| 只看权重分布，不看激活 | 找不到真正误差来源 | 必须做标定采样 |
| 标定数据太少或分布偏 | scale 不稳定 | 选覆盖常见输入的样本 |
| 运行时保留额外缩放算子 | 推理变慢 | 提前融合参数 |
| AWQ 保护通道过多 | 收益被稀释 | 只保留少量高价值通道 |
| 忽略硬件 kernel 支持 | 理论压缩，实际不加速 | 先确认平台支持 W4/W8 路径 |

还有一个经常被忽视的点：量化不是只看 perplexity 或离线指标。真正上线时还要看吞吐、显存占用、prefill 与 decode 的行为差异，以及内核是否支持 fused dequant。否则你可能得到一个“精度还行但跑不快”的方案。

---

## 替代方案与适用边界

SmoothQuant 和 AWQ 都属于 PTQ 路线，优势是部署成本低。你不需要重新训练模型，也不需要保留完整训练链路，只要拿到模型权重和少量标定数据即可。

和其他常见方案对比：

| 方案 | 是否需要重训/优化 | 主要优化对象 | 适合场景 |
|---|---|---|---|
| SmoothQuant | 不需要重训 | 激活与权重联合平滑 | W8A8 推理部署 |
| AWQ | 不需要重训 | 激活感知的权重量化 | W4/W3 边缘部署 |
| GPTQ | 不需要完整重训，但需逐层误差补偿 | 权重 | 离线压缩，常见于 W4 |
| QLoRA | 需要微调 | 训练参数高效更新 | 低成本微调，不是纯部署方案 |
| 量化感知训练 | 需要训练 | 模型整体 | 可重训且追求极限精度 |

这里要强调边界：QLoRA 的目标更多是“低成本微调大模型”，不是“只做部署压缩”；GPTQ 虽然也常用于 PTQ，但它的思路是对量化误差做二阶近似补偿，而不是利用激活缩放去搬运难度。它们都重要，但解决的问题不完全一样。

真实工程例子仍然是 TinyChat。它把 AWQ 与平台相关优化结合，在 Jetson Orin 上推动了超大模型的本地运行。这个例子说明了 AWQ 的典型适用边界：

1. 显存和带宽很紧。
2. 目标硬件对低比特算子较友好。
3. 不能接受重训成本。
4. 更在意“能不能部署起来”而不是只看最优理论精度。

而 SmoothQuant 更常见于服务器侧的 W8A8 部署，尤其适合那些已经有成熟 INT8 kernel、希望把激活也压低的场景。AWQ 则在更低比特上更有吸引力，因为 INT4 下“保护关键通道”带来的收益通常更明显。

所以可以这样记：

- 想做稳定的 INT8 全栈推理，优先看 SmoothQuant。
- 想做 INT4 甚至 INT3 的极限压缩，优先看 AWQ。
- 如果可以接受训练或蒸馏成本，再考虑量化感知训练等更重的方法。

---

## 参考资料

1. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*  
   ICML 2023。给出了激活平滑的等价变换、$\alpha$ 的定义方式，以及多种 LLM 上的 W8A8 结果。

2. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*  
   MLSys 2024。核心贡献是用激活信息识别显著通道，在低比特权重量化下优先保护关键部分，并强调硬件友好实现。

3. TinyChat / AWQ 相关项目与工程说明  
   展示了 AWQ 与 kernel fusion、platform-aware packing 等工程优化结合后的真实部署效果，特别适合理解“论文方法如何落地到边缘设备”。

4. 论文项目页与开源实现  
   适合直接查看标定流程、量化脚本和实际支持的模型列表，用来复现会比只读论文更直接。
