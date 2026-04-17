## 核心结论

L1/L2 正则化的目标都不是“让损失更小”，而是限制模型参数不要无约束增长，从而降低过拟合。过拟合可以白话理解为：模型把训练集里的偶然噪声也当成了规律。

L2 正则化在损失里加入参数平方和，常写成 $\lambda_2 \|w\|_2^2$。它的效果是持续压小权重，但通常不会把权重直接压成 0，所以更像“整体收缩”。在贝叶斯视角里，它对应高斯先验，也就是“事先认为参数大概率靠近 0，且偏离越大越不合理”。

L1 正则化在损失里加入参数绝对值和，常写成 $\lambda_1 \|w\|_1$。它的效果是让一部分权重直接变成 0，所以会产生稀疏性。稀疏性可以白话理解为：很多参数被删掉，只保留少数真正起作用的参数。贝叶斯视角下，它对应拉普拉斯先验。

Elastic Net 同时使用 L1 和 L2：

$$
L_{\text{total}} = L_{\text{data}} + \lambda_2 \|w\|_2^2 + \lambda_1 \|w\|_1
$$

它不是“更高级的默认答案”，而是一个折中：既想要部分稀疏，又不希望优化过程太不稳定时使用。

在深度学习工程里，L2 常常以 weight decay 的形式出现。weight decay 可以白话理解为：每次更新后都顺手把参数按比例缩一点。对 SGD 来说，L2 与 weight decay 基本等价；但对 Adam 这类自适应优化器，直接把 $\lambda w$ 塞进梯度并不等同于真正的权重衰减，因此更推荐 AdamW 这类 decoupled weight decay，也就是“把梯度更新和权重收缩拆开做”。

---

## 问题定义与边界

先把问题限定清楚：本文讨论的是参数正则化，即直接约束模型权重 $w$ 的方法，不讨论 Dropout、数据增强、早停这类“间接正则化”。

它主要解决两个问题：

| 问题 | 典型表现 | 正则化作用 |
|---|---|---|
| 过拟合 | 训练误差低，验证误差高 | 限制参数复杂度 |
| 参数不稳定 | 特征相关、噪声大、权重爆涨 | 让解更平滑或更稀疏 |

L2 更适合“希望所有参数都保留，但别太大”的场景。比如图像分类、语言模型、推荐模型中的大部分深度网络训练。原因很直接：L2 是光滑的，光滑可以白话理解为“梯度变化连续，优化器容易走”。因此它和 SGD、Momentum、Adam 这类一阶方法兼容性都比较好。

L1 更适合“希望自动删特征”的场景，尤其是高维表格数据，例如广告点击率预估、风控、医疗统计建模。因为它会把一部分系数压成 0，相当于顺手做了特征选择。

但边界也要说清楚。L1 在 0 点不可导。不可导可以白话理解为：正好走到 0 时，普通梯度没有唯一斜率，所以标准梯度下降法在这里不够自然。理论上通常用子梯度或软阈值处理，工程上常见专门求解器如 coordinate descent。coordinate descent 可以白话理解为：每次只更新一个参数，轮流把所有参数扫一遍。

因此，不能把“L1 能产生稀疏”简化成“任何优化器加个 $\|w\|_1$ 都能自然变稀疏”。在很多深度网络里，直接用 SGD 跑 L1，经常只能看到参数变小，却看不到明显、稳定、可复现的稀疏结构。

玩具例子：假设只有两个参数 $w_1,w_2$。如果数据损失的等高线是椭圆，L2 的约束边界是圆，L1 的约束边界是菱形。椭圆先碰到菱形尖角的概率更高，而尖角恰好落在坐标轴上，所以更容易出现 $w_1=0$ 或 $w_2=0$。这就是“L1 更容易做稀疏”的几何来源，不是魔法。

---

## 核心机制与推导

设数据损失为 $L_{\text{data}}(w)$，总损失为：

$$
L_{\text{total}}(w)=L_{\text{data}}(w)+\lambda_2\|w\|_2^2+\lambda_1\|w\|_1
$$

### 1. L2 为什么像“整体收缩”

L2 项对单个参数 $w_i$ 的梯度是 $2\lambda_2 w_i$。若把常数并进超参数，也常写成 $\lambda w_i$。用最简单的梯度下降更新：

$$
w \leftarrow w - \eta (\nabla L_{\text{data}}(w) + \lambda w)
$$

整理得：

$$
w \leftarrow (1-\eta\lambda)w - \eta \nabla L_{\text{data}}(w)
$$

这说明即便数据梯度为 0，参数也会按比例缩小。比例项 $(1-\eta\lambda)$ 就是“乘法收缩”。

玩具例子：若当前 $w=[2,-3]$，学习率 $\eta=0.01$，$\lambda=0.1$，且此时数据梯度为 0，则

$$
w' = 0.999 \cdot [2,-3] = [1.998,-2.997]
$$

它们都变小了，但都没有变成 0。这正是 L2 的典型行为。

### 2. L1 为什么会产生稀疏

L1 项对 $w_i \neq 0$ 的梯度是 $\lambda_1 \operatorname{sign}(w_i)$，也就是只看正负，不看大小。$\operatorname{sign}$ 可以白话理解为：正数取 $+1$，负数取 $-1$，0 单独处理。

如果把一个梯度步和 L1 的 proximal 操作合在一起，单个参数更新可写成软阈值：

$$
w_i \leftarrow \operatorname{sign}(z_i)\max(|z_i|-\eta\lambda_1,0)
$$

其中 $z_i$ 是先按数据损失更新后的中间值。

这条公式非常关键。它不是“把参数简单减一个常数”，而是先减，再检查是否穿过 0。一旦穿过，就直接钉在 0。于是小权重会被持续清除，大权重则只被轻微裁掉。

继续上面的玩具例子。若 $z=[0.0008,-0.003]$，$\eta\lambda_1=0.001$，则：

- 第一个参数：$\max(0.0008-0.001,0)=0$，直接归零
- 第二个参数：$\max(0.003-0.001,0)=0.002$，保留负号后为 $-0.002$

这就是 L1 的稀疏来源：小参数会被硬性截断。

### 3. 贝叶斯视角为什么常被提到

贝叶斯先验可以白话理解为：在看到数据之前，先对参数可能长什么样做一个概率假设。

- L2 对应高斯先验：大参数出现概率快速下降，但整体是平滑的
- L1 对应拉普拉斯先验：在 0 附近更尖，因此更偏好大量参数贴近 0

这解释了它们为何都“偏好小权重”，但偏好的方式不同：L2 偏好“均匀小”，L1 偏好“多数为 0，少数非 0”。

### 4. AdamW 为什么重要

对 SGD，L2 和 weight decay 接近等价；但对 Adam，不应把两者混为一谈。Adam 会对不同参数使用不同的自适应步长，自适应步长可以白话理解为：梯度历史不同，更新幅度也不同。

如果把 L2 项直接加到梯度里：

$$
g_t = \nabla L_{\text{data}}(w_t) + \lambda w_t
$$

那么这部分“正则梯度”也会进入 Adam 的一阶、二阶矩估计，被自适应缩放，结果是不同参数受到的衰减强度不再只由 $\lambda$ 决定。

AdamW 采用解耦更新：

$$
w_{t+\frac{1}{2}} = \text{AdamStep}(w_t, \nabla L_{\text{data}})
$$

$$
w_{t+1} = (1-\eta\lambda) w_{t+\frac{1}{2}}
$$

也就是先做 Adam 的梯度步，再单独做权重衰减。这样 weight decay 的语义才稳定，接近 SGD 下的“统一收缩”。

真实工程例子：训练 Transformer 或 BERT 类模型时，常见配置是 `AdamW + weight_decay=0.01`，同时对 bias 和 LayerNorm 参数关闭衰减。原因是这两类参数对模型表达和数值稳定性更敏感，盲目衰减经常收益不高，甚至伤害收敛。

---

## 代码实现

下面给一个最小可运行的 Python 例子，分别展示 L2 收缩、L1 软阈值，以及一个简化版的 AdamW 更新。代码不依赖第三方库。

```python
import math

def l2_step(w, lr, lam):
    # 当数据梯度为 0 时，L2 等价于乘法收缩
    return [(1 - lr * lam) * x for x in w]

def soft_threshold(x, threshold):
    if x > threshold:
        return x - threshold
    if x < -threshold:
        return x + threshold
    return 0.0

def l1_step(w, lr, lam):
    th = lr * lam
    return [soft_threshold(x, th) for x in w]

def adamw_step(w, grad, m, v, t, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1):
    new_w, new_m, new_v = [], [], []
    for wi, gi, mi, vi in zip(w, grad, m, v):
        mi = beta1 * mi + (1 - beta1) * gi
        vi = beta2 * vi + (1 - beta2) * (gi ** 2)

        m_hat = mi / (1 - beta1 ** t)
        v_hat = vi / (1 - beta2 ** t)

        # 先做 Adam 梯度步
        wi = wi - lr * m_hat / (math.sqrt(v_hat) + eps)
        # 再做 decoupled weight decay
        wi = wi * (1 - lr * weight_decay)

        new_w.append(wi)
        new_m.append(mi)
        new_v.append(vi)
    return new_w, new_m, new_v

# L2: 整体缩小，不会直接归零
w = [2.0, -3.0]
w2 = l2_step(w, lr=0.01, lam=0.1)
assert abs(w2[0] - 1.998) < 1e-12
assert abs(w2[1] + 2.997) < 1e-12

# L1: 小参数会被软阈值截断为 0
w = [0.0008, -0.003]
w1 = l1_step(w, lr=0.01, lam=0.1)
assert w1[0] == 0.0
assert abs(w1[1] + 0.002) < 1e-12

# AdamW: 梯度步和衰减分离
w = [1.0, -2.0]
grad = [0.1, -0.2]
m = [0.0, 0.0]
v = [0.0, 0.0]
new_w, new_m, new_v = adamw_step(w, grad, m, v, t=1)
assert len(new_w) == 2 and len(new_m) == 2 and len(new_v) == 2
assert new_w[0] < 1.0
assert new_w[1] > -2.0
```

如果把它映射到真实框架，思路通常是：

| 场景 | 推荐做法 |
|---|---|
| 线性模型、稀疏需求强 | L1 或 Elastic Net，优先用专门求解器 |
| 常规深度网络 | SGD/Momentum + weight decay，或 AdamW |
| 自适应优化器下想要稳定衰减 | 用 AdamW，不要手搓 Adam+L2 |
| 想显式删特征 | 用 L1/Elastic Net，不要只靠 L2 |

真实工程里，PyTorch 一般直接用 `torch.optim.AdamW`。同时要把参数分组：卷积核、线性层权重使用 decay；bias、LayerNorm、Embedding 的某些参数通常不做 decay。这一步经常比“把 `weight_decay` 从 0.01 调到 0.02”更影响结果。

---

## 工程权衡与常见坑

第一，L2 不是越大越好。$\lambda$ 太小时没有约束效果，太大时模型会欠拟合。欠拟合可以白话理解为：模型连训练集都学不好，因为参数被压得太死。

第二，L1 的“稀疏”不等于“性能一定更好”。在强相关特征很多时，L1 往往会随机保留其中一个、压掉另一些，导致结果不稳定。Elastic Net 在这里通常更稳，因为 L2 会让相关特征的系数变化更平滑。

第三，Adam + L2 不等于 AdamW。这是深度学习里最常见的误解之一。前者是“把正则项并进梯度”，后者是“更新后再单独衰减权重”。如果你用的是自适应优化器，又想要可控的 weight decay，优先检查自己到底调用的是不是 AdamW。

第四，不是所有参数都该做 weight decay。常见经验是跳过：

| 参数类型 | 常见处理 |
|---|---|
| bias | 通常不衰减 |
| LayerNorm / BatchNorm 的缩放与偏移 | 通常不衰减 |
| 大多数线性层、卷积层权重 | 通常衰减 |

第五，L1 在深度网络里经常不如想象中好用。原因不是理论错，而是优化难度、硬件效率、实现细节都会影响结果。很多时候你以为自己在“做稀疏训练”，实际上只是给优化器增加了噪声。

工程检查可以直接按下面做：

| 检查项 | 建议 |
|---|---|
| 优化器是否为 Adam 类 | 是的话优先 AdamW |
| 是否需要显式特征选择 | 需要时考虑 L1 或 Elastic Net |
| 是否存在大量相关特征 | 优先考虑 Elastic Net 而不是纯 L1 |
| 是否区分参数组 | bias / norm 参数通常排除 decay |
| 是否做超参数搜索 | 对 $\lambda_1,\lambda_2$ 必做交叉验证或验证集搜索 |

---

## 替代方案与适用边界

Elastic Net 是最直接的替代方案。它通常写成“整体强度 $\lambda$ + 混合比例 $\alpha$”：

$$
L = L_{\text{data}} + \lambda \left(\alpha \|w\|_1 + (1-\alpha)\|w\|_2^2 \right)
$$

其中 $\alpha=1$ 接近 Lasso，也就是纯 L1；$\alpha=0$ 接近 Ridge，也就是纯 L2。它特别适合“高维、相关特征多、既想要部分稀疏又不想太跳”的问题。

对深度学习来说，很多时候更实用的替代路线不是“把 L1 用到底”，而是：

- 大模型训练：AdamW 或 SGD + weight decay
- 泛化增强：Dropout、数据增强、标签平滑
- 模型压缩：训练后剪枝，而不是训练时强压 L1

真实工程例子：一个小样本宽表任务，样本 1 万、特征 5 万。此时可先用 Elastic Net 做基线，借助交叉验证挑选 $\alpha$ 和 $\lambda$，获得一组稀疏且稳定的特征子集。若后续改用神经网络，通常不会直接照搬高强度 L1，而是更多依赖 L2、Dropout 和早停。

相反，在超大 Transformer 任务里，目标通常不是“筛掉输入特征”，而是“保持训练稳定和泛化能力”。这时主流方案仍然是 AdamW，配合较小但稳定的 `weight_decay`，例如 0.01 左右，再用学习率预热和调度器控制收敛过程。

最终选择可以归纳为：

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| L2 / weight decay | 平滑、稳定、易优化 | 不产生稀疏 | 深度网络默认首选 |
| L1 | 可做特征选择 | 非光滑、优化难 | 线性模型、高维表格 |
| Elastic Net | 稀疏与稳定折中 | 超参数更多 | 相关特征多的表格任务 |
| AdamW | 自适应优化下衰减语义清晰 | 仍需调 decay 和参数组 | Transformer、BERT、现代 DL |
| Dropout / 数据增强 | 不直接约束参数，泛化常有效 | 不替代显式稀疏 | 神经网络辅助正则 |

---

## 参考资料

- Artificial-Intelligence-Wiki, Regularization Techniques: L1 L2 - Complete Guide  
- DL Notes, AdamW  
- Cornell Optimization Wiki, AdamW  
- GRAUSOFT, Understanding L1 Regularization  
- BytePlus, Challenges with L1 regularization  
- PyTorch Docs, `torch.optim.AdamW`  
- R Documentation / ADMM Elastic Net  
- Auroria, From Bayesian Priors to Weight Decay via MAP  
- Tensortonic, Regularization 与优化更新说明
