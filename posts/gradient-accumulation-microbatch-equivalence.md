## 核心结论

梯度累积是把一次大批量更新拆成多次小批量反向传播，再合并成一次参数更新。对零基础读者，可以把它理解成“分期付款版的大 batch”：每次只拿一小份数据进显存，但最后按大 batch 的方式更新一次参数。

最重要的结论有三条：

1. 对普通 SGD，梯度累积 $K$ 步与直接使用 $K$ 倍 batch size 做一次更新是严格等价的，前提是参数在这 $K$ 个 micro-batch 期间不更新，且梯度按均值而不是按和来对齐。
2. 对 Adam、AdamW 这类自适应优化器，自适应的意思是“优化器会根据历史梯度大小自动调步长”，梯度累积通常只能近似等价，因为一阶矩 $m_t$ 和二阶矩 $v_t$ 的时间粒度变了。
3. 在流水线并行里，micro-batch 数不仅影响显存，还影响 bubble。bubble 可以理解成“流水线里 GPU 闲着没活干的比例”。常见近似公式是
   $$
   \text{bubble}=\frac{S-1}{M+S-1}
   $$
   其中 $S$ 是 pipeline stage 数，$M$ 是一次全局 batch 被切成的 micro-batch 数。工程上通常希望 bubble 低于 10%，即大致满足 $M \gtrsim 9(S-1)$。

玩具例子：每个 micro-batch 是 16 个样本，累积 4 步，则有效批量是
$$
B_{\text{eff}} = 16 \times 4 = 64
$$
这在 SGD 下等价于一次直接处理 64 个样本，但显存占用接近 batch size 为 16 的情况。

---

## 问题定义与边界

本文讨论的问题是：当显存放不下目标 batch size 时，如何用梯度累积在不改变训练目标的前提下，尽量复现大 batch 训练的更新效果。

先统一几个变量：

| 变量 | 含义 | 备注 |
| --- | --- | --- |
| $m$ | 每个 micro-batch 的样本数 | 单次前向/反向真正进显存的 batch |
| $K$ | gradient accumulation 步数 | 每 $K$ 个 micro-batch 更新一次 |
| $B_{\text{eff}}$ | 有效批量 | $B_{\text{eff}} = m \times K$，多卡时还要再乘数据并行卡数 |
| $S$ | pipeline stage 数 | 模型被切成几段 |
| $M$ | pipeline 中的 micro-batch 数 | 影响 bubble |
| bubble | 流水线空泡比例 | 近似为 $\frac{S-1}{M+S-1}$ |

边界也要说清楚：

1. 本文说的“等价”，默认是指一次参数更新前累计得到的梯度向量相同，或者足够接近，不是指 wall-clock 时间、吞吐、收敛曲线都相同。
2. 严格等价主要成立在 SGD 及其“先聚合梯度、后做一步更新”的情形。
3. 只要训练过程中有依赖“步数”的状态，等价性就会被打破或变弱，例如 Adam 的动量缓存、某些按 step 更新的学习率调度器、BatchNorm 的运行统计。
4. 在 pipeline 并行里，micro-batch 数 $M$ 和梯度累积步数经常耦合，但不是同一个概念。前者主要决定流水线填充程度，后者主要决定多久执行一次 `optimizer.step()`。

一个新手最容易理解的场景是：你本来想用 batch size 64，但显存只能放 16。于是把 64 拆成 4 份，每份 16，做 4 次 forward/backward，只累加梯度，不更新参数。第 4 次结束后再统一更新一次。这样做的目标不是提速，而是在显存受限时保住大 batch 的梯度统计特性。

---

## 核心机制与推导

核心原因只有一句话：梯度对样本均值损失是线性可加的。线性可加的意思是“分开算再求平均”和“合起来直接算平均”结果一样。

设单个样本损失为 $\ell(x_i,\theta)$，一个大 batch 含 $K$ 个 micro-batch，每个 micro-batch 大小相同为 $m$。总 batch 的平均损失是
$$
L(\theta)=\frac{1}{Km}\sum_{k=1}^{K}\sum_{j=1}^{m}\ell(x_{k,j},\theta)
$$

对参数求梯度：
$$
\nabla L(\theta)=\frac{1}{K}\sum_{k=1}^{K}\left(\frac{1}{m}\sum_{j=1}^{m}\nabla \ell(x_{k,j},\theta)\right)
$$

把第 $k$ 个 micro-batch 的平均梯度记成 $g_k$，则
$$
\nabla L(\theta)=\frac{1}{K}\sum_{k=1}^{K} g_k
$$

这就是梯度累积的数学基础。只要你在每个 micro-batch 上做的是“平均梯度”，并且参数 $\theta$ 在这 $K$ 次反向传播期间保持不变，那么先分别求 $g_1,\dots,g_K$ 再求平均，和直接对大 batch 求一次平均梯度完全一致。

因此 SGD 的一步更新
$$
\theta_{t+1}=\theta_t-\eta \cdot \nabla L(\theta_t)
$$
可以被拆成：

1. 取第 1 个 micro-batch，算梯度，加到 `.grad`
2. 取第 2 个 micro-batch，继续加
3. ...
4. 第 $K$ 个 micro-batch 算完后，执行一次 `optimizer.step()`

这里有一个最常见的实现细节：到底该在每次 `loss.backward()` 前除以 $K$，还是最后把梯度整体除以 $K$？两者在数学上等价，工程上前者更常见，因为不容易忘。

玩具例子：假设四个 micro-batch 的梯度分别是 $2,4,6,8$，那么大 batch 平均梯度是
$$
\frac{2+4+6+8}{4}=5
$$
如果你每次直接反向传播原始 loss，不做 `/4`，累计后得到的是 20，不是 5，更新步幅会放大 4 倍。

为什么 Adam 只能近似等价？因为 Adam 不只是看当前梯度 $g_t$，还会维护历史状态：
$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
$$
$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$
然后再用偏置修正后的 $\hat m_t,\hat v_t$ 更新参数。这里的关键问题是：Adam 把“第几次更新”当成时间轴。梯度累积把原本多个小步合并成一个大步，改变了 $g_t$ 进入优化器状态的节奏。即使最终累计梯度均值相同，$m_t$、$v_t$ 的演化路径也不完全一样，所以只能说近似复现。

真实工程例子：MLSys Book 给出的 GPT-2 训练案例中，单卡能放下的 micro-batch 是 16，希望全局有效 batch 达到 512。做法是 8 张 GPU、每卡 micro-batch 16、累积 4 步，于是每卡有效 batch 为 64，全局是 $8 \times 16 \times 4 = 512$。这样做通信次数更少，成本更低，而收敛质量与直接大 batch 很接近。这就是梯度累积在大模型训练中的典型用途。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，用线性回归验证“整批 SGD 更新”和“micro-batch 累积更新”在数学上相同。代码里的 `assert` 会检查两种方式更新后的参数是否一致。

```python
import numpy as np

# 一个非常小的线性模型: y = w * x
x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 4.0, 6.0, 8.0])

lr = 0.1
w0 = 0.5

def mse_grad(w, xb, yb):
    # 平均损失 L = mean((w*x - y)^2)
    # 对 w 的梯度: dL/dw = mean(2*(w*x - y)*x)
    return np.mean(2.0 * (w * xb - yb) * xb)

# 方案 A: 一次用 batch size = 4
g_full = mse_grad(w0, x, y)
w_full = w0 - lr * g_full

# 方案 B: micro-batch size = 2, 累积 2 步
grad_acc_steps = 2
micro_size = 2
grad_sum = 0.0

for i in range(0, len(x), micro_size):
    xb = x[i:i + micro_size]
    yb = y[i:i + micro_size]
    # 关键点: 每个 micro-batch 先求平均梯度，再除以累积步数
    grad_sum += mse_grad(w0, xb, yb) / grad_acc_steps

w_acc = w0 - lr * grad_sum

assert np.allclose(g_full, grad_sum)
assert np.allclose(w_full, w_acc)

print("full batch grad =", g_full)
print("accumulated grad =", grad_sum)
print("updated weight =", w_acc)
```

在 PyTorch 训练循环里，标准写法通常是这样：

```python
grad_acc_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(dataloader, 1):
    outputs = model(**batch)
    loss = outputs.loss

    # 除以累积步数，确保最终梯度是“均值”而不是“总和”
    loss = loss / grad_acc_steps
    loss.backward()

    if step % grad_acc_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

这段代码里有两个关键动作：

1. `loss / grad_acc_steps`：保证累计后的梯度和大 batch 平均梯度一致。
2. 只在整除时 `optimizer.step()`：保证参数在累积窗口内不变化，否则就不再等价。

如果是分布式数据并行，还要注意“每卡 micro-batch”与“全局有效 batch”的关系：
$$
B_{\text{global}} = m \times K \times N_{\text{data-parallel}}
$$
其中 $N_{\text{data-parallel}}$ 是数据并行卡数。

---

## 工程权衡与常见坑

梯度累积解决的是显存问题，不是免费的性能优化。它的代价通常是更少的参数更新频率，以及更长的单次优化步时间。

常见坑可以直接列成表：

| 问题 | 现象 | 原因 | 处理方式 |
| --- | --- | --- | --- |
| 忘记除以 `grad_acc_steps` | loss 看着正常，但训练发散或步子过大 | 累积的是梯度和，不是梯度均值 | 每次 `backward` 前先 `/K` |
| 提前 `optimizer.step()` | 与大 batch 不等价 | 参数在累积窗口内变化了 | 只在第 $K$ 步更新 |
| 学习率照搬小 batch 配置 | 不稳定或收敛慢 | 有效 batch 变了，噪声规模也变了 | 按有效 batch 重新调学习率 |
| Adam 结果与预期不完全一致 | 同样 `B_eff` 但曲线不同 | 动量和二阶矩状态的时间粒度改变 | 接受近似性，重新调 warmup 和 betas |
| BatchNorm 表现异常 | 小 micro-batch 下统计不稳 | BN 依赖当前 batch 统计量 | 大模型训练里更常用 LayerNorm |
| pipeline 中 micro-batch 太少 | GPU 利用率低 | bubble 太大 | 增加 $M$，让流水线更满 |

bubble 问题值得单独展开。设 pipeline 有 $S=4$ 个 stage，如果一次只切成 $M=8$ 个 micro-batch，则
$$
\text{bubble}=\frac{4-1}{8+4-1}=\frac{3}{11}\approx 27.3\%
$$
也就是说，平均有超过四分之一的时间设备在等待。若希望 bubble 低于 10%，需要
$$
\frac{S-1}{M+S-1}<0.1
$$
代入 $S=4$，得到大致 $M \ge 27$。这就是“micro-batch 太少，流水线并行不划算”的原因。

但 micro-batch 也不是越多越好。因为 $M$ 变大后，往往意味着你在追求更大的有效 batch，进而减少单位样本上的参数更新次数。对 SGD，大 batch 常用线性学习率缩放，但这个规律有适用范围。Goyal 等人在 ImageNet/ResNet-50 上展示了 batch size 到 8192 仍可工作，但再继续放大通常就需要更精细的调参，且不同任务上限差异很大。换句话说，“有效 batch 越大越好”不是通用规律。

工程上更实用的判断方式是同时满足两个条件：

1. pipeline bubble 足够小，通常希望低于 10% 到 15%。
2. 有效 batch 仍处在当前任务和优化器可稳定训练的区间内。

---

## 替代方案与适用边界

如果显存够，直接增大真实 batch size 往往比梯度累积更简单，因为它不会引入“更新频率下降”和“优化器状态粒度变化”这两个副作用。梯度累积本质上是内存换时间。

几种常见方案可以对比：

| 方法 | 等效批量 | 内存 | 等价性 | 备注 |
| --- | --- | --- | --- | --- |
| 梯度累积 | 可增大 | 低于直接大 batch | SGD 严格，Adam 近似 | 最常见，简单直接 |
| 直接大 batch | 真实增大 | 最高 | 完全一致 | 前提是显存足够 |
| ZeRO/FSDP | 可支持更大 batch 或更大模型 | 通过切分状态降内存 | 与原优化器逻辑基本一致 | 复杂度更高 |
| Pipeline 并行 | 主要解决模型放不下 | 单卡激活更可控 | 与调度有关 | 需要同时优化 bubble |
| 混合精度 | 不改变数学目标 | 显著降显存 | 近似一致 | 通常应优先开启 |

适用边界可以概括为：

1. 如果模型只是“稍微放不下”，优先试混合精度、激活检查点、再考虑梯度累积。
2. 如果模型参数本身放不下，单靠梯度累积没用，因为它只减少激活带来的瞬时显存，不解决模型和优化器状态的常驻显存，此时更需要 ZeRO、FSDP、张量并行或 pipeline 并行。
3. 如果你使用 AdamW 训练大语言模型，梯度累积通常是标准配置，但不要把“有效 batch 相同”直接理解成“训练行为完全相同”。学习率、warmup、梯度裁剪阈值都可能要跟着调整。
4. 如果已经能稳定放下目标 batch，就没必要为了“理论上的大 batch 等价”再加梯度累积，因为它通常会降低训练吞吐。

一个真实工程判断准则是：先定模型和并行策略，再从显存上限推出单卡 micro-batch，接着根据目标有效 batch 反推累积步数，最后检查 bubble 和收敛曲线是否在可接受范围内。不要先拍脑袋设一个很大的 `grad_acc_steps`，再希望训练自己变好。

---

## 参考资料

1. MLSys Book, AI Training 章节，关于梯度累积的数学等价、GPT-2 案例与工程权衡。  
   https://mlsysbook.ai/contents/core/training/training.html

2. Sebastian Raschka, *Finetuning LLMs With Gradient Accumulation*，适合理解“为什么要在 loss 上除以累积步数”。  
   https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html

3. CMU 课程材料，关于 pipeline parallelism 中 bubble 与 micro-batch 数的关系，适合理解流水线调度。  
   https://cmu-l3.github.io/anlp-fall2025/static_files/anlp-f2025-18-scaling-parallelism.pdf

4. Emergent Mind, *Pipeline Parallelism*，整理了 1F1B 调度下的 bubble ratio 公式与不同调度变体。  
   https://www.emergentmind.com/topics/pipeline-parallelism

5. Priya Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*，大 batch 线性缩放规则的经典论文。  
   https://arxiv.org/abs/1706.02677

6. Uplatz, *Gradient Accumulation: A Comprehensive Technical Guide*，偏工程实现，适合核对代码模式与常见坑。  
   https://uplatz.com/blog/gradient-accumulation-a-comprehensive-technical-guide-to-training-large-scale-models-on-memory-constrained-hardware/
