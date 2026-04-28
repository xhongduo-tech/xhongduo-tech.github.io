## 核心结论

后训练量化校准，指的是在不重新训练模型的前提下，用一批代表性样本估计激活张量的取值范围，再据此生成 `scale` 和 `zero point`，把浮点数稳定映射到 INT8 或 INT4。这里的“激活”可以先理解成模型中间层吐出的连续数值，不是最终分类结果，而是后续层继续计算要用到的中间信号。

新手最容易误解的一点是：量化不是“把数变小”，而是“先画格子再装数”。校准做的事，就是决定这些格子覆盖多宽、每格多密。格子画错了，后面再怎么严格执行整数推理，误差也已经被写进参数里了。

下面这条因果链最重要：

| 校准集质量 | `scale` 估计 | 直接误差 | 最终表现 |
|---|---|---|---|
| 覆盖真实分布 | 范围贴近线上输入 | 舍入误差可控 | 精度下降通常较小 |
| 被异常值主导 | 范围过大，步长过粗 | 普通值分辨率下降 | 细节丢失，层输出变钝 |
| 覆盖不足 | 范围过小 | 线上值被裁剪 `clip` | 少数任务或长序列明显退化 |

还可以把“校准充分”和“校准不足”直接对比：

| 情况 | 量化前看起来 | 量化后实际效果 |
|---|---|---|
| 校准充分 | 激活范围稳定，长短输入都覆盖 | 平均指标和长尾场景都较稳 |
| 校准不足 | 离线验证样本表现还行 | 一上线遇到代码、长上下文、多轮对话就掉点 |

玩具例子很直观。假设只有 8 个格子，本来 $[-1,1]$ 里的普通值可以分得比较细；如果突然混进一个 `8.0`，整个格子范围会被迫拉宽，原本 `0.12`、`0.15`、`0.18` 这些值可能都落到同一个整数档位里。量化器没有“坏”，坏的是校准时对范围的估计。

真实工程里更常见的情况是：校准集只含短文本，但线上有大量长上下文、代码补全和多轮对话。某些 attention 或 MLP 层在长序列上的激活尾部会变重，结果往往不是“平均精度小降”，而是少数 token 位置、少数任务类型突然退化得很明显。

---

## 问题定义与边界

本文讨论的是 PTQ，也就是 `Post-Training Quantization`，白话说就是“模型训练已经结束，再做量化压缩”。本文不讨论 QAT，也就是 `Quantization Aware Training`，它是在训练过程中显式模拟量化误差；也不重点讨论只量化权重的 `weight-only quantization`，因为后者对激活校准的依赖明显更弱。

先把核心术语摆清楚：

| 术语 | 定义 | 白话解释 |
|---|---|---|
| `PTQ` | 后训练量化 | 训练完再压缩模型 |
| `calibration` | 校准 | 用样本估计激活正常范围 |
| `observer` | 统计器 | 运行样本时偷偷记下 min/max 或直方图 |
| `scale` | 缩放因子 | 浮点步长映射到整数格子的比例 |
| `zero point` | 零点 | 让整数某个值对应真实的 0 |
| `saturation` | 饱和 | 值超出范围后被硬截断 |

本文使用的基本符号如下：

| 符号 | 含义 |
|---|---|
| $x$ | 原始浮点激活 |
| $q$ | 量化后的整数值 |
| $s$ | `scale`，缩放因子 |
| $z$ | `zero point`，零点 |
| $[Q_n, Q_p]$ | 整数可表示范围，比如 INT8 常见为 $[-127,127]$ 或 $[-128,127]$ |

量化与反量化的基本形式是：

$$
q=\mathrm{clip}(\mathrm{round}(x/s)+z,\;Q_n,\;Q_p),\qquad \hat x=s(q-z)
$$

其中 $\hat x$ 是反量化后的近似值。`clip` 的意思是“超出范围就直接截断”，这是很多线上退化的直接来源。

本文讨论什么、不讨论什么，可以再收紧一层：

| 本文讨论 | 本文不展开 |
|---|---|
| 激活校准范围怎么估计 | 训练期 QAT 的反向传播细节 |
| `scale` / `zero point` 怎么来 | 量化编译器底层 kernel 优化 |
| MinMax、Percentile、Histogram 的差异 | 稀疏化、蒸馏等其他压缩路线 |
| INT8/INT4 对校准的敏感性 | 所有硬件后端的指令级实现 |

边界要讲清楚，是因为很多工程问题并不是“量化不行”，而是“拿错了工具”。如果你的场景主要是权重占内存、激活不占瓶颈，那么 weight-only 量化可能更合适；如果你的模型对长上下文异常敏感，直接做激进的 activation INT4，本来就是高风险决策。

---

## 核心机制与推导

最常见的是对称量化和非对称量化。

| 方案 | 常见公式 | 适用特点 |
|---|---|---|
| 对称量化 | $s=T/Q_p,\;z=0,\;T=\max |x|$ | 实现简单，硬件友好，常用于权重或近似对称分布激活 |
| 非对称量化 | $s=(x_{\max}-x_{\min})/(Q_p-Q_n)$，$z\approx Q_n-x_{\min}/s$ | 能更贴合偏移分布，但实现更复杂 |

误差主要来自两类。

第一类是步长误差。量化本质上是在做离散化，步长就是 $s$。当异常值把最大阈值 $T$ 拉大时，$s$ 会一起变大，普通值之间原本可分辨的细节就会被合并。

第二类是裁剪误差。如果校准时没见过足够大的值，阈值估小了，那么线上出现更大激活时就会被 `clip`。这不是“近似”，而是“直接截断”。

看一个最小数值例子。做 INT8 对称量化，取整数范围 $[-127,127]$。

- 如果校准集中出现异常值 `8.0`，则
  $$
  s=\frac{8}{127}\approx0.063
  $$
  对普通值 $x=0.20$，
  $$
  q=\mathrm{round}(0.20/0.063)=3,\qquad \hat x=3\times0.063\approx0.189
  $$
  误差不算灾难，但分辨率已经明显变粗。

- 如果没有异常值，范围只到 `1.0`，则
  $$
  s=\frac{1}{127}\approx0.0079
  $$
  同样的 `0.20` 会被映射得更细。

- 反过来，如果校准只见过 $\pm1.0$，线上来了 `1.5`，它会被截到 `1.0` 附近，单点误差可直接到 `0.5`。

误差来源、现象和后果可以整理成表：

| 误差来源 | 现象 | 后果 |
|---|---|---|
| 少量异常值抬高范围 | `scale` 变大，步长变粗 | 普通值分辨率下降 |
| 校准覆盖不足 | 线上值越界被 `clip` | 局部层输出饱和，长尾任务退化 |
| 分布偏移 | 离线统计和线上不同 | 某些输入类型掉点明显 |
| 比特数过低 | 可用整数格子太少 | INT4 对范围误差更敏感 |

INT8 和 INT4 的差异，不只是“少了 4 bit”，而是格子数量差很多。若采用对称量化，INT8 常见正侧有 127 个档位，INT4 常见只有 7 个。

| 位宽 | 常见整数范围 | 正侧档位数 | 同样阈值下的步长 |
|---|---|---|---|
| INT8 | $[-127,127]$ | 127 | $T/127$ |
| INT4 | $[-7,7]$ | 7 | $T/7$ |

因此同一阈值 $T$ 下，INT4 的步长大约比 INT8 粗

$$
\frac{T/7}{T/127}=\frac{127}{7}\approx18.1
$$

这就是为什么 INT4 对校准偏差、异常值和长尾分布更敏感。INT8 很多时候还能“带伤运行”，INT4 往往会把问题直接放大到任务级可见。

---

## 代码实现

工程实现里，校准通常不是“直接把模型转成整数”，而是分成 `collect -> observe -> compute qparams -> fake quant -> validate` 这几步。`observer` 可以先理解成一个挂在层上的小统计器，负责记录激活的 `min/max`、直方图或分位数信息。

最小流程如下：

| 步骤 | 做什么 | 产物 |
|---|---|---|
| 1 | 跑校准集 | 激活统计量 |
| 2 | 计算 `scale` / `zero point` | 量化参数 |
| 3 | 对关键层做 fake quant | 近似模拟量化误差 |
| 4 | 跑验证集 | 看平均指标和长尾样本 |
| 5 | 调 observer 或样本分布 | 重新校准 |

下面是一个可运行的简化 Python 版本，用最小 `MinMaxObserver` 展示校准和 fake quant 的关系：

```python
import math

class MinMaxObserver:
    def __init__(self):
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def observe(self, values):
        for v in values:
            self.min_val = min(self.min_val, v)
            self.max_val = max(self.max_val, v)

    def calculate_qparams_symmetric_int8(self):
        T = max(abs(self.min_val), abs(self.max_val))
        scale = T / 127.0 if T > 0 else 1.0
        zero_point = 0
        return scale, zero_point

def fake_quant_symmetric_int8(x, scale):
    q = round(x / scale) if scale != 0 else 0
    q = max(-127, min(127, q))
    return q, q * scale

# 玩具校准集：被异常值 8.0 主导
observer = MinMaxObserver()
observer.observe([0.1, 0.2, -0.3, 8.0])
scale_bad, zp_bad = observer.calculate_qparams_symmetric_int8()

q1, xhat1 = fake_quant_symmetric_int8(0.2, scale_bad)

# 更合理的校准集：范围只到 1.0
observer2 = MinMaxObserver()
observer2.observe([0.1, 0.2, -0.3, 1.0])
scale_good, zp_good = observer2.calculate_qparams_symmetric_int8()

q2, xhat2 = fake_quant_symmetric_int8(0.2, scale_good)

# 覆盖不足时的饱和
q3, xhat3 = fake_quant_symmetric_int8(1.5, scale_good)

assert scale_bad > scale_good
assert abs(xhat2 - 0.2) < abs(xhat1 - 0.2)
assert xhat3 <= 1.0 + 1e-6

print("bad scale:", scale_bad, "reconstructed:", xhat1)
print("good scale:", scale_good, "reconstructed:", xhat2)
print("clipped example:", xhat3)
```

这段代码说明三件事：

1. 校准集里有异常值时，`scale` 会变大。
2. `scale` 变大后，普通值的重建精度会变差。
3. 如果校准范围只到 `1.0`，那么 `1.5` 会被直接裁剪。

PyTorch 风格的真实流程通常更像下面这样：

```python
# 伪代码
model.eval()
attach_observers(model)

for x in calibration_loader:
    _ = model(x)  # observer 记录各层激活统计

for obs in all_observers(model):
    scale, zero_point = obs.calculate_qparams()
    save_qparams(obs.name, scale, zero_point)

run_fake_quant_validation(model, val_loader)
```

一个最小配置表如下：

| 配置项 | 常见选择 | 说明 |
|---|---|---|
| 校准数据 | 500 到数千条代表性样本 | 关键不是数量绝对大，而是分布像线上 |
| `observer` 类型 | `MinMax` / `Histogram` / `Percentile` | 分别对应简单、稳健、抗异常值 |
| 量化粒度 | per-tensor / per-channel | per-channel 更细，但实现更复杂 |
| bit width | INT8 / INT4 | INT4 压得更狠，也更脆弱 |

真实工程例子：LLM 做 PTQ 时，如果校准集只来自短问答，而线上包含代码补全和长上下文阅读，attention 输出和部分 MLP 激活常会在长序列位置出现更重尾分布。离线平均分可能只降一点，但用户会在特定 token 位置看到突然重复、漏词或逻辑跳变。这类退化不是“平均噪声变大”，而是“局部层在特定输入上被饱和或过度粗化”。

---

## 工程权衡与常见坑

校准最核心的工程原则不是“样本越多越好”，而是“样本必须像真实流量”。长度分布、任务类型、语言、模板、系统提示词、异常请求，都可能改变激活分布。

常见坑可以直接列成表：

| 问题 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 只拿训练集子集做校准 | 离线指标好看，线上掉点 | 训练分布不等于推理分布 | 按任务、长度、语言分层抽样 |
| 只看平均指标 | 长尾任务突然失败 | 平均值掩盖局部退化 | 加失败样本回放和 token 级分析 |
| 直接用 MinMax 吃掉所有异常值 | 普通值分辨率变差 | 单个 outlier 拉大范围 | 试 Percentile 或 Histogram |
| 样本太少、batch 太小 | `scale` 不稳定 | 统计量方差大 | 逐步加样本，观察收敛 |
| INT4 直接照搬 INT8 流程 | 精度断崖式下降 | 格子太少，更怕偏差 | 优先平滑、per-channel 或 weight-only |

“校准样本数 vs `scale` 收敛”虽然不同模型数值不同，但趋势通常类似：

| 校准样本数 | `scale` 稳定性 | 常见风险 |
|---|---|---|
| 10 到 50 | 很不稳定 | 极易被偶然样本主导 |
| 100 到 500 | 开始稳定 | 长尾场景可能仍未覆盖 |
| 500 到 2000 | 多数任务可用 | 需看是否覆盖真实流量 |
| 更大 | 增益递减 | 采样质量比盲目加量更重要 |

代表性采样策略建议如下：

| 维度 | 为什么要覆盖 |
|---|---|
| 输入长度 | 长上下文会改变尾部分布 |
| 任务类型 | 问答、摘要、代码、对话激活模式不同 |
| 语言 | 不同语言 token 统计特征不同 |
| 模板/系统提示 | Prompt 结构会影响中间层激活 |
| 异常样本 | 线上极端请求常触发饱和问题 |

新手要特别注意：不要只看最终任务分数。量化问题经常是层级局部失真，先在少量样本上做 layer-wise 输出比对，再决定是不是校准问题、是不是 observer 问题、是不是 bit width 太激进。

---

## 替代方案与适用边界

校准不是只有一种算法。不同方法本质上是在回答同一个问题：哪些值值得被保留得更细，哪些值可以被牺牲。

| 方案 | 优点 | 代价 | 适用场景 |
|---|---|---|---|
| MinMax | 简单直接 | 极易受异常值影响 | 分布干净、追求实现简单 |
| Percentile | 能忽略极少数 outlier | 可能裁掉真实大值 | 长尾较重但可容忍少量裁剪 |
| Histogram / Entropy | 更细致地找阈值 | 统计和实现更复杂 | 追求更稳的 PTQ |
| SmoothQuant | 把激活 outlier 压力转移到权重 | 需要额外变换流程 | LLM 激活重尾明显 |
| Weight-only quantization | 风险较低，部署简单 | 激活内存和带宽收益有限 | 激活量化不稳时的保守路线 |
| QAT | 精度上限更高 | 需要训练资源 | 高价值场景、PTQ 不够稳时 |

可以把决策逻辑粗略写成一棵文字版决策树：

1. 激活 outlier 重不重？
2. 如果不重，先试 INT8 + MinMax 或 Histogram。
3. 如果较重，先试 Percentile 或 Histogram。
4. 如果是 LLM 且长上下文明显敏感，优先考虑 SmoothQuant。
5. 如果 activation INT4 误差不可控，先退回 weight-only 或 INT8。
6. 如果业务必须极限压缩且精度要求高，再考虑 QAT。

“何时不用 INT4”也应该明确：

- 线上输入长度变化很大，但你拿不到代表性校准流量。
- 模型对长上下文、代码、多轮推理特别敏感。
- 没有足够的验证集去看长尾任务和失败样本。
- 你只能跑最简单的 MinMax 校准，没有能力做误差定位。
- 当前瓶颈主要是权重存储，不是激活带宽或显存。

一句话概括适用边界：不是所有场景都该硬上 activation INT4。很多时候，先做更稳的 INT8、per-channel、SmoothQuant 或 weight-only，工程收益反而更好。

---

## 参考资料

资料用途先给一个总表：

| 资料名 | 适用内容 | 读者收益 |
|---|---|---|
| TensorRT Accuracy 文档 | 代表性校准数据与精度权衡 | 理解工业部署为什么强调 representative data |
| PyTorch Quantization 文档 | observer、fake quant、PTQ 工作流 | 对照主流框架实现 |
| Intel Neural Compressor 文档 | MinMax / Entropy / Percentile 差异 | 理解不同校准算法怎么选 |
| SmoothQuant 论文 | LLM outlier 与 PTQ | 理解为什么长尾激活会成为瓶颈 |

1. [Improving Model Accuracy - NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/accuracy-builder.html)  
2. [Quantization API Reference - PyTorch](https://docs.pytorch.org/docs/stable/quantization-support.html)  
3. [FakeQuantize - PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.fake_quantize.FakeQuantize.html)  
4. [Calibration Algorithms in Quantization - Intel Neural Compressor](https://intel.github.io/neural-compressor/latest/docs/source/calibration.html)  
5. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html)  
6. [Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs](https://openreview.net/forum?id=G01UBemh55)
