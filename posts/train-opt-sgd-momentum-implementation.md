## 核心结论

SGD with Momentum 的核心不是“把学习率调大”，而是给参数更新加一个**速度项**。速度项可以理解成“过去几步梯度的累积趋势”，白话说就是：这一步不只看眼前的梯度，还会参考之前一直往哪个方向在走。它的标准形式是

$$
v_t = \beta v_{t-1} + \eta \nabla L(\theta_{t-1}), \qquad
\theta_t = \theta_{t-1} - v_t
$$

其中 $\eta$ 是学习率，$\beta$ 是动量系数，常用默认值是 $\beta=0.9$。这个机制的直接作用有两个：

1. 在“目标方向基本一致”的维度上持续加速。
2. 在“梯度来回变号”的高曲率方向上减少左右抖动。

Nesterov Momentum，也常写成 NAG，是动量法的前瞻版本。它不是在当前位置算梯度，而是在“按当前速度先看一小步”之后再算梯度，等价形式可写成

$$
\theta_t = \theta_{t-1} - \eta \nabla L(\theta_{t-1} - \beta v_{t-1})
$$

白话说，普通 Momentum 是“边跑边修正”，Nesterov 是“先看前面地形再修正”。在经典的光滑凸优化设定下，这种前瞻可把收敛率从 $O(1/t)$ 提升到 $O(1/t^2)$。这里的“收敛率”就是误差随迭代次数下降的速度。

一个适合新手的玩具理解是“小球下坡”。纯 SGD 每一步只看脚下坡度，遇到狭长山谷会左右撞墙；Momentum 像小球有了惯性，会沿着总体向下的方向保持速度；Nesterov 则像先探一眼前方坡度，再决定脚下这一步怎么迈。

真实工程里，结论更具体：

| 场景 | 更常见选择 | 原因 |
|---|---|---|
| 大模型预训练 | AdamW | 自适应学习率更稳，热启动和大规模训练更省心 |
| 视觉分类训练，如 ResNet | SGD + Momentum | 训练后期常有更好的泛化性能 |
| LLM 中部分新方案 | Muon/μNSGD + AdamW 混合 | 对矩阵参数做更强的方向更新，同时保留非矩阵参数稳定性 |

---

## 问题定义与边界

问题很简单：给定损失函数 $L(\theta)$，我们要找到让它尽量小的参数 $\theta$。但“简单”只发生在公式里，实际训练中最常见的问题是纯 SGD 走得不稳。

SGD，随机梯度下降，白话就是“每次看一个小批次样本，沿着当前梯度的反方向走一步”。它的问题主要出现在两类地形：

1. **高曲率方向**。曲率可以理解成“地形弯得有多急”。在很陡又很窄的谷底，梯度方向会频繁左右变化。
2. **低曲率方向**。沿着真正该前进的长轴方向，纯 SGD 每步推进又往往偏慢。

典型例子是 ridge 或狭长谷底损失面。纯 SGD 会在左右方向不断反复，像在窄走廊里撞墙；Momentum 则把过去方向积累成速度，让前进方向更稳定。

这里要先明确边界。Momentum 解决的是**更新轨迹的稳定性和加速问题**，不是万能优化器。它不自动按参数尺度调学习率，也不替代正则化，不保证在所有非凸问题上都比 AdamW 好。

下面这张表把边界列清楚：

| 问题 | 纯 SGD | SGD + Momentum | Nesterov |
|---|---|---|---|
| 狭长谷底左右震荡 | 明显 | 明显改善 | 通常进一步改善 |
| 沿主方向推进速度慢 | 慢 | 更快 | 更快 |
| 是否需要额外状态 | 不需要 | 需要 `momentum_buffer` | 需要 `momentum_buffer` |
| 参数内存开销 | 最低 | 约再加一份参数状态 | 约再加一份参数状态 |
| 理论加速结论 | 基线 | 多数是经验改善 | 光滑凸问题下有经典加速理论 |
| 非凸训练稳定性 | 依赖调参 | 一般较稳 | 对学习率更敏感 |

额外状态是什么意思？就是优化器要为每个参数保存一份速度缓冲。PyTorch 里通常叫 `momentum_buffer`。如果模型是 7B 参数、FP32 存储，那么参数本体大约需要 28GB，而再加一份同形状的动量缓冲会额外占约 28GB 中的一部分状态空间。按很多工程资料的常见粗略口径，单看一份动量状态可以近似理解为“再加一份参数量级的显存/内存成本”；在一些混合精度和分片场景里，统计口径会不同，但结论不变：动量不是免费的。

再看 Muon。它可以理解成“在 Nesterov 风格更新上，再对矩阵型动量做归一化和正交化处理”的一类方法。它的边界更严格：通常只处理 $\ge 2D$ 的矩阵参数，像偏置、Embedding、LayerNorm 标量/向量参数，仍然需要 AdamW 或别的优化器配合。

---

## 核心机制与推导

先看最基础的推导。设第 $t$ 步梯度为 $g_t=\nabla L(\theta_{t-1})$，Momentum 更新为

$$
v_t = \beta v_{t-1} + \eta g_t
$$

$$
\theta_t = \theta_{t-1} - v_t
$$

这里 $v_t$ 是速度，白话就是“历史梯度的指数衰减累积”。所谓指数衰减，就是越新的梯度权重越大，越旧的梯度权重按 $\beta, \beta^2, \beta^3$ 递减。把式子展开可以看到：

$$
v_t = \eta g_t + \beta \eta g_{t-1} + \beta^2 \eta g_{t-2} + \cdots
$$

这说明速度并不是某一步的瞬时反应，而是过去一串梯度方向的加权平均。因此：

1. 如果多个 step 的梯度方向大体一致，速度会越来越大，形成加速。
2. 如果某个方向的梯度频繁正负来回，历史项会互相抵消，震荡被削弱。

### 玩具例子：手算两步速度

设学习率 $\eta=0.01$，动量系数 $\beta=0.9$，初始速度 $v_0=0$。

第一步梯度 $g_1=2$：

$$
v_1 = 0.9 \times 0 + 0.01 \times 2 = 0.02
$$

$$
\theta_1 = \theta_0 - 0.02
$$

第二步梯度变成 $g_2=1$：

$$
v_2 = 0.9 \times 0.02 + 0.01 \times 1 = 0.028
$$

$$
\theta_2 = \theta_1 - 0.028
$$

注意这里第二步的瞬时梯度其实更小了，从 2 变成 1，但更新步长反而从 0.02 增到 0.028。原因不是“学得更激进了”，而是历史方向一致，速度项在继续累积。

### 为什么能抑制震荡

看一个二维二次函数：

$$
L(x,y) = 100x^2 + y^2
$$

这里 $x$ 方向非常陡，$y$ 方向很平。纯 SGD 会在 $x$ 方向因为梯度大而来回过冲，在 $y$ 方向又推进很慢。Momentum 通过历史平均压制 $x$ 方向反复变号的影响，同时沿 $y$ 方向持续积累速度。这就是它在“狭长谷底”里通常更好用的原因。

### Nesterov 为什么更前瞻

Nesterov 的关键不是另造一个速度公式，而是改了“梯度在哪儿算”。普通 Momentum 在当前位置算梯度；Nesterov 先看预测点

$$
\theta_{t-1} - \beta v_{t-1}
$$

再在这个点上计算梯度。于是更新可写成

$$
v_t = \beta v_{t-1} + \eta \nabla L(\theta_{t-1} - \beta v_{t-1})
$$

$$
\theta_t = \theta_{t-1} - v_t
$$

白话说，如果当前速度已经把你推向某个方向，那就别等真正走过去才发现前面坡变陡或变缓，而是提前在预测位置看梯度。这样做在经典光滑凸问题中形成了更强的误差压缩，所以有 $O(1/t^2)$ 的加速结论。这里要强调边界：这个理论结论依赖凸性与光滑性，不应直接照搬到深度网络所有非凸训练场景。

### Muon 的额外一步：对矩阵动量做正交化

Muon 可以看成“带矩阵归一化/正交化的 Nesterov 风格更新”。它的目标不是只追求更大步长，而是让更新方向在矩阵空间里更稳定。常见做法是对二维动量矩阵做 Newton-Schulz 迭代近似正交化。Newton-Schulz 可以理解成“用迭代方法把矩阵拉回更接近正交的形状”。一些实现会使用固定迭代步数，例如 5 步，并配合经验系数如 $(3.4445,\,-4.775,\,2.0315)$。

一个抽象写法是：

$$
B_{k+1} = aB_k + bB_k(B_k^\top B_k) + cB_k(B_k^\top B_k)^2
$$

其中 $B_k$ 是归一化后的矩阵动量迭代值，系数 $(a,b,c)$ 由实现设定。它的作用不是替代梯度，而是对“要更新的方向结构”做矩阵级修正。这也是为什么 Muon 通常只适用于权重矩阵，而不适合偏置或一维参数。

### 真实工程例子：ResNet 和大模型

在 CV 任务里，比如 ResNet 训练，工程上经常看到这样一条经验路线：前期如果用 Adam，损失下降可能很快，但后期切换到 SGD + Momentum 常常带来更好的验证集表现。这背后不是 SGD 绝对更强，而是它的更新噪声结构和非自适应特性有时更利于泛化。

在大模型预训练里，主流仍然是 AdamW。原因很现实：AdamW 有一阶矩和二阶矩，二阶矩可以理解成“梯度大小波动的历史估计”，白话说就是它会根据每个参数过去梯度幅度自动调节步长，通常更稳。但它状态更多，内存也更贵。Muon 这类方法尝试在矩阵权重上把内存和性能重新平衡。

---

## 代码实现

下面先用一个最小可运行 Python 例子实现 SGD with Momentum 和 Nesterov，并用 `assert` 检查数值。

```python
import math

def sgd_momentum_step(theta, grad, velocity, lr=0.01, beta=0.9):
    velocity = beta * velocity + lr * grad
    theta = theta - velocity
    return theta, velocity

def nesterov_grad(theta, velocity, grad_fn, beta=0.9):
    lookahead = theta - beta * velocity
    return grad_fn(lookahead)

def quadratic_grad(theta):
    # L(theta) = theta^2, grad = 2*theta
    return 2.0 * theta

# 玩具例子：验证题目里的两步动量
theta = 1.0
v = 0.0

theta, v = sgd_momentum_step(theta, grad=2.0, velocity=v, lr=0.01, beta=0.9)
assert abs(v - 0.02) < 1e-12
assert abs(theta - 0.98) < 1e-12

theta, v = sgd_momentum_step(theta, grad=1.0, velocity=v, lr=0.01, beta=0.9)
assert abs(v - 0.028) < 1e-12
assert abs(theta - 0.952) < 1e-12

# Nesterov：在 lookahead 点算梯度
theta = 1.0
v = 0.0
g = nesterov_grad(theta, v, quadratic_grad, beta=0.9)
theta, v = sgd_momentum_step(theta, grad=g, velocity=v, lr=0.1, beta=0.9)

assert theta < 1.0
assert v > 0.0
```

如果把它映射到 PyTorch `torch.optim.SGD` 的思路，核心伪代码大致是这样：

```python
for p in params:
    grad = p.grad
    buf = state[p].get("momentum_buffer")

    if momentum != 0:
        if buf is None:
            buf = grad.clone()
        else:
            buf = momentum * buf + (1 - dampening) * grad

        if nesterov:
            d_p = grad + momentum * buf
        else:
            d_p = buf
    else:
        d_p = grad

    p = p - lr * d_p
    state[p]["momentum_buffer"] = buf
```

这里最关键的工程结构有两个：

1. `state[p]`  
   这是每个参数自己的优化器状态字典。白话说，就是“每个参数名下单独存一份历史信息”。

2. `momentum_buffer` 或 `momentum_buffer_list`  
   这是动量缓存。PyTorch 在实际实现里会把一组参数对应的 buffer 组织起来处理。

保存和恢复训练时，这些状态会进入 `state_dict`。因此如果你中途保存 checkpoint，再恢复训练，Momentum 不会从零开始，而是继续沿着之前积累的速度走。对长训练来说，这很重要。

Muon/μNSGD 的伪代码会多一步矩阵正交化：

```python
for p in params:
    grad = p.grad
    buf = state[p].get("momentum_buffer", zeros_like(p))

    buf = momentum * buf + lr * grad

    if p.ndim >= 2:
        buf = orthogonalize(buf)   # Newton-Schulz 近似正交化
        update = buf
    else:
        # 偏置、Embedding、Norm 等不走 Muon，通常转交给 AdamW
        continue

    if nesterov:
        update = grad + momentum * update

    p = p - update
    state[p]["momentum_buffer"] = buf
```

真实工程里不会像上面这么简化，因为还要处理 weight decay、mixed precision、参数分组、fused kernel 等问题，但核心结构就是：  
“先维护动量状态，再决定是否使用 Nesterov，再更新参数。”

---

## 工程权衡与常见坑

Momentum 的优点明显，但它的工程代价和坑也很明确。

### 1. 状态内存不是小事

每个参数一份 `momentum_buffer`，意味着状态规模和参数规模同量级。模型越大，这个成本越难忽略。对 7B 级参数来说，这不是“多一点缓存”，而是训练能不能放下的问题。

### 2. Nesterov 不是无脑开关

在很多任务里打开 `nesterov=True` 有收益，但不是默认一定更好。它更前瞻，也更依赖学习率和动量匹配。如果学习率已经偏大，Nesterov 可能把过冲放大。

### 3. Muon 不能一把梭到所有参数

Muon 这类矩阵优化器通常只适合二维及以上张量。Embedding、bias、LayerNorm 权重如果强行套上矩阵正交更新，常见后果是训练不稳甚至 NaN。

下面用表格列出常见坑：

| 坑 | 原因 | 缓解方案 |
|---|---|---|
| 显存突然爆掉 | 每个参数都多一份 `momentum_buffer` | 降低 batch size，启用混合精度，使用 ZeRO/分片，重新评估优化器状态开销 |
| 损失震荡变大 | 学习率过高，动量又大 | 先降学习率，再决定是否保留 `beta=0.9` |
| 开启 Nesterov 后更不稳 | lookahead 放大了步长敏感性 | 配合 warmup 或更保守的初始学习率 |
| 恢复训练后效果异常 | 没有正确恢复 `state_dict` | 同时保存并加载模型参数和优化器状态 |
| Muon 跑到 Embedding 上 NaN | 非矩阵参数不适合矩阵正交更新 | 将 Embedding、bias、Norm 参数交给 AdamW |
| 以为 Momentum 等于更快收敛 | 只看训练损失，不看泛化 | 同时观察验证集，不要只盯训练曲线 |

### 真实工程例子：混合优化器分组

在 LLM 或多模块网络里，常见做法不是“全模型统一一个优化器”，而是按参数类型分组：

- 线性层权重矩阵：Muon 或 Nesterov/SGD-M
- Embedding、bias、LayerNorm：AdamW
- 特殊缩放参数：单独更保守的学习率

这样做的原因很直接：不同参数的几何结构不同，更新规则也不该完全一样。矩阵参数更适合谈“方向结构”；偏置和归一化参数更适合谈“数值稳定”。

---

## 替代方案与适用边界

如果只问“现在工程上最稳妥的默认值是什么”，答案通常还是 AdamW。因为它对大多数现代深度学习任务更省调参成本，尤其是预训练前期、warmup 阶段、稀疏梯度或尺度差异大的参数集合。

但这不意味着 SGD + Momentum 过时。它在 CV 场景里仍然是强基线，特别是你关心最终泛化而不只是训练集下降速度时。Muon 则处在更偏前沿的位置，适合愿意接受更复杂参数分组与实现约束、换取矩阵参数内存/收敛收益的场景。

| 方案 | 收敛体验 | 泛化表现 | 内存 | 适用场景 |
|---|---|---|---|---|
| AdamW | 通常最稳，前期很友好 | 不一定最好 | 高，需要更多状态 | 大模型预训练、复杂任务默认选择 |
| SGD + Momentum | 前期可能慢，后期稳定 | CV 中常较强 | 中，需一份动量状态 | ResNet、分类、检测等视觉任务 |
| Nesterov | 比普通动量更前瞻 | 依任务而定 | 中 | 光滑问题、希望更激进加速时 |
| Muon/μNSGD | 对矩阵权重有潜力 | 仍在快速演进 | 相比 AdamW 可更省 | LLM 训练/微调中的矩阵参数更新 |

一个实用流程可以这样理解：

1. 预训练前期或新任务上手：先用 AdamW，保证稳定。
2. 如果是 CV 任务并追求最终泛化：后期尝试切到 SGD + Momentum。
3. 如果是大矩阵权重占主导的模型，并且愿意维护参数分组：尝试 Muon 处理 $\ge 2D$ 参数，其余参数继续 AdamW。
4. 如果你对理论加速有兴趣，且问题更接近光滑凸设定：再考虑 Nesterov 的优势是否能兑现到实际任务。

换句话说，SGD with Momentum 不是 AdamW 的“低配版”，而是另一种优化偏好：它少做自适应，多依赖方向累积；因此在一些任务里更容易得到干净的泛化行为，在另一些任务里则不如 AdamW 省心。

---

## 参考资料

- PyTorch `torch.optim.SGD` 文档  
  https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD  
  用途：说明 `momentum`、`nesterov`、`state_dict` 和优化器状态组织方式。

- Nesterov 加速与收敛率综述  
  https://ideas.repec.org/a/spr/coopap/v83y2022i2d10.1007_s10589-022-00401-y.html  
  用途：给出光滑凸优化中 $O(1/t)$ 到 $O(1/t^2)$ 的理论背景与推广讨论。

- Muon Optimizer 介绍  
  https://huggingface.co/blog/onekq/muon-optimizer  
  用途：解释 Muon/μNSGD 的设计思想、矩阵正交化更新和使用边界。

- SWATS 相关介绍  
  https://phas-ml.github.io/paper/2022/05/03/switching-adam-to-sgd.html  
  用途：说明从 Adam 切换到 SGD 在部分视觉任务中带来更好泛化的经验与研究背景。

- PyTorch 优化器状态与内存成本分析  
  https://www.yuhan.one/blog/no-out-of-memory  
  用途：帮助理解优化器状态为什么会显著增加显存/内存占用。

- Intel Extension for PyTorch 优化器技术说明  
  https://intel.github.io/intel-extension-for-pytorch/xpu/2.0.110%2Bxpu/tutorials/technical_details/optimizer_fusion_gpu.html  
  用途：补充工程实现层面对优化器状态、融合和性能细节的理解。
