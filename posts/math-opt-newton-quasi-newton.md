## 核心结论

牛顿法、BFGS、L-BFGS 解决的是同一类问题：最小化一个可微目标函数。区别不在“能不能优化”，而在“用多少曲率信息”。

“曲率”可以先理解成函数表面的弯曲程度。梯度只告诉你当前往哪边下降，曲率还告诉你这个方向到底陡不陡、该走大步还是小步。牛顿法直接使用 Hessian，也就是二阶导数组成的矩阵；BFGS 不直接算 Hessian，而是用相邻两步的位置变化和梯度变化去逼近逆 Hessian；L-BFGS 再进一步，只保留最近几步历史，用更小内存近似同样的信息。

在强凸且足够光滑的局部区域，牛顿法的收敛速度最好。若记最优点为 $x^\*$，误差常写作 $e_k=\|x_k-x^\*\|$，那么局部可达到近似
$$
e_{k+1}\le C e_k^2
$$
这叫“二阶收敛”或“局部超线性中的更强形式”，直观上是误差接近“平方缩小”。

但代价同样直接。若参数维度为 $n$，显式构造 Hessian 通常要 $O(n^2)$ 存储，解线性系统或求逆常见代价是 $O(n^3)$。这决定了牛顿法更适合中小规模、结构清晰、Hessian 稳定可求的问题。BFGS 把每步代价降到 $O(n^2)$，L-BFGS 进一步把内存降到 $O(mn)$，其中 $m$ 常取 10 到 20。

| 方法 | 用到的信息 | 单步时间量级 | 内存量级 | 典型收敛特性 | 适用规模 |
|---|---:|---:|---:|---|---|
| Newton | 梯度 + Hessian | $O(n^3)$ | $O(n^2)$ | 局部二阶收敛 | 中小规模 |
| BFGS | 梯度 + $(s_k,y_k)$ 历史 | $O(n^2)$ | $O(n^2)$ | 局部超线性 | 中等规模 |
| L-BFGS | 梯度 + 最近 $m$ 对 $(s_k,y_k)$ | $O(mn)$ 到 $O(n^2)$ 间，常视实现而定 | $O(mn)$ | 局部超线性，依赖线搜索 | 大规模 |

一个最小玩具例子可以直接看出三者关系。对
$$
f(x)=(x-2)^2
$$
有 $\nabla f(x)=2(x-2)$，$\nabla^2 f(x)=2$。牛顿更新为
$$
x_{k+1}=x_k-\frac{1}{2}\cdot 2(x_k-2)=2
$$
所以一步到最优点。这个例子说明：当目标函数是简单二次型时，曲率信息足够精确，Newton、BFGS、L-BFGS 可以表现得几乎一致。

---

## 问题定义与边界

本文讨论的是**无约束优化**，也就是直接在整个参数空间里找最小值，不带显式等式或不等式约束。目标函数记为
$$
\min_x f(x)
$$

为了让结论成立，通常至少要求：

| 条件 | 白话解释 | 为什么重要 |
|---|---|---|
| 可微 | 梯度存在，能知道下降方向 | 一阶信息是所有方法的基础 |
| 二阶可微 | Hessian 存在 | 牛顿法需要，BFGS 的理论也常依赖 |
| 强凸 | 函数整体像“碗”，不会有多个局部谷底乱跳 | 保证唯一极小点，便于讨论收敛 |
| Lipschitz 梯度 | 梯度变化不会无限剧烈 | 用于控制近似误差 |

这里的“强凸”可以先理解成“碗底足够结实，不会出现平坦到没有方向感的区域”。更正式地说，若存在 $\mu>0$ 使得
$$
\nabla^2 f(x)\succeq \mu I
$$
则函数在该区域内有统一的下界曲率。

BFGS 与 L-BFGS 的核心边界条件是曲率条件
$$
y_k^\top s_k>0
$$
其中
$$
s_k=x_{k+1}-x_k,\qquad y_k=\nabla f(x_{k+1})-\nabla f(x_k)
$$
$s_k$ 是“这一步走了多远”，$y_k$ 是“梯度因此变化了多少”。若 $y_k^\top s_k>0$，就说明沿这一步观测到了正曲率，逆 Hessian 近似才容易保持正定。正定可以先理解成“这个矩阵给出的方向仍然是下降方向，而不是上升方向”。

真实工程里，逻辑回归就是一个典型例子。若使用 $L_2$ 正则化，目标函数通常是凸的，梯度和 Hessian 都能写出来。此时：

- Newton 可以直接用解析 Hessian。
- BFGS 可以只依赖梯度，避免手写 Hessian。
- L-BFGS 在特征维度很高时更实用，因为不需要存整张 $n\times n$ 矩阵。

所以，这三种方法主要属于“光滑优化”工具箱，而不是所有机器学习问题的默认解。

---

## 核心机制与推导

牛顿法来自二阶泰勒展开。对当前点 $x_k$ 附近，有
$$
f(x_k+p)\approx f(x_k)+\nabla f(x_k)^\top p+\frac12 p^\top \nabla^2 f(x_k)p
$$
把这个二次近似对 $p$ 求极小，得到
$$
\nabla^2 f(x_k)p=-\nabla f(x_k)
$$
所以
$$
p_k=-[\nabla^2 f(x_k)]^{-1}\nabla f(x_k),\qquad x_{k+1}=x_k+p_k
$$
这就是牛顿方向。它不是“顺着梯度走”，而是“先看地形弯曲，再校正梯度”。

BFGS 的思路是：既然完整 Hessian 太贵，那就只要求一个近似逆矩阵 $H_k$，让它满足最新观测到的曲率信息
$$
H_{k+1}y_k=s_k
$$
这叫“割线条件”，白话就是：新矩阵至少要解释刚刚这一步观察到的曲率关系。满足这个条件、又尽量少改动旧矩阵 $H_k$，可得到经典更新：
$$
H_{k+1}=(I-\rho_k s_k y_k^\top)H_k(I-\rho_k y_k s_k^\top)+\rho_k s_k s_k^\top
$$
其中
$$
\rho_k=\frac{1}{y_k^\top s_k}
$$

这是一种秩二更新。“秩二”可以理解成：每次只做一个很低成本的结构修正，而不是重建整张矩阵。

二维玩具例子更直观。假设某一步得到
$$
s_k=(1,0)^\top,\qquad y_k=(0.5,0)^\top
$$
说明第一维上“走了 1，梯度变化了 0.5”，即第一维曲率被观测到。BFGS 更新后，会把逆 Hessian 在第一维的尺度调到更贴近真实曲率，而第二维基本延续旧估计。这就是“边走边修地图”。

L-BFGS 再进一步，不保存完整 $H_k$，只保存最近 $m$ 对 $(s_i,y_i)$。计算方向时用“两环递推”：

1. 反向遍历历史，逐步从梯度中扣除已知曲率分量。
2. 用一个标量初始矩阵 $H_0$ 近似缩放。
3. 正向遍历历史，把各步信息补回来。

核心形式是
$$
q\leftarrow g_k,\qquad \alpha_i=\rho_i s_i^\top q,\qquad q\leftarrow q-\alpha_i y_i
$$
然后
$$
r\leftarrow H_0 q,\qquad \beta_i=\rho_i y_i^\top r,\qquad r\leftarrow r+s_i(\alpha_i-\beta_i)
$$
最终搜索方向为
$$
d_k=-r
$$

它的关键价值不在公式更漂亮，而在于**不显式存矩阵**。这使得高维问题仍能使用近似二阶信息。

---

## 代码实现

下面先给一个最小可运行例子，分别验证一维二次函数上牛顿法与简化 L-BFGS 的方向结果。

```python
import numpy as np

def f(x):
    return (x - 2.0) ** 2

def grad(x):
    return 2.0 * (x - 2.0)

def newton_step(x):
    hessian = 2.0
    return x - grad(x) / hessian

x0 = 10.0
x1 = newton_step(x0)
assert abs(x1 - 2.0) < 1e-12

# 一步梯度下降，构造 L-BFGS 的第一对历史
alpha = 0.5
xg = x0 - alpha * grad(x0)
s = np.array([xg - x0])
y = np.array([grad(xg) - grad(x0)])
g = np.array([grad(xg)])

# m=1 的两环递推
rho = 1.0 / float(y @ s)
q = g.copy()
a = rho * float(s @ q)
q = q - a * y

gamma = float((s @ y) / (y @ y))  # 常见标度
r = gamma * q

b = rho * float(y @ r)
r = r + s * (a - b)
direction = -r

# 对一维二次函数，方向应等于到最优点的牛顿方向
assert direction.shape == (1,)
assert direction[0] > 0
assert abs((xg + direction[0]) - 2.0) < 1e-8
```

这个例子说明：对简单二次函数，L-BFGS 只要获得一对有效的 $(s,y)$，就能重建出正确曲率方向。

若在工程中直接使用库，常见写法更短。比如 SciPy 中：

- `method="BFGS"`：提供目标函数和梯度即可。
- `method="L-BFGS-B"`：额外支持边界约束，并用 `maxcor` 控制历史长度。
- `trust-ncg` 或同类 trust-region Newton 变体：适合能提供 Hessian 或 Hessian 向量积的场景。

一个真实工程例子是高维逻辑回归训练。假设特征维度十万级，显式 Hessian 基本不可接受，这时 L-BFGS 常作为强基线：它比纯 SGD 更善于处理病态问题，也就是不同方向曲率差异很大的问题；但它又不需要存十万乘十万的矩阵。

大模型里更现实的情况是：全量预训练通常不会直接用标准 L-BFGS，因为梯度噪声大、模型非凸、批量分布变化快；但在小规模 fine-tuning、低秩适配、少量参数优化、二阶方法研究基线中，L-BFGS 仍然经常出现。K-FAC、Shampoo 这类方法也属于“利用曲率信息但避免完整 Hessian”的路线，只是结构化假设更强。

---

## 工程权衡与常见坑

最常见的误判是“牛顿法迭代少，所以一定更快”。这是错的。优化总耗时等于“每步代价 × 步数”，而不是只看步数。Newton 常输在单步成本，L-BFGS 常赢在整体 wall-clock。

| 方法 | 主要优势 | 主要风险 | 常见补救 |
|---|---|---|---|
| Newton | 步数少，局部收敛快 | Hessian 贵，不正定时会出错 | 阻尼、trust region、共轭梯度近似解 |
| BFGS | 不用显式 Hessian，方向质量高 | 仍需 $O(n^2)$ 内存 | 强 Wolfe 线搜索、跳过坏更新 |
| L-BFGS | 内存低，适合高维 | 噪声大时历史信息失真 | 限制历史长度、阻尼、重启 |
| mL-BFGS | 对噪声更稳 | 实现更复杂，调参更多 | 动量与阻尼联合调节 |

第一个坑是 **$y_k^\top s_k\le 0$**。这意味着新观测到的曲率不满足正定要求。如果还强行更新，BFGS 近似矩阵可能变坏，下一步方向甚至不再下降。工程上常见做法有两种：

- `skip_update`：直接跳过这次更新。
- `damp_update`：做阻尼，把坏曲率修正成较温和的版本。

第二个坑是 **非凸区域**。牛顿法在非凸问题里可能碰到不正定 Hessian，此时解出的方向不一定是下降方向。很多二阶方法因此使用 trust region，也就是限制“这一步最多只相信局部模型多大范围”。

第三个坑是 **小批量噪声**。L-BFGS 假设历史梯度能代表真实曲率，但深度学习的小 batch 梯度方差很大，$(s,y)$ 容易被噪声污染。历史一旦失真，两环递推输出的方向也会失真。这也是为什么 vanilla L-BFGS 很少直接用于大规模随机训练，而带动量或阻尼的变体更常见。

---

## 替代方案与适用边界

如果问题规模很大、非凸强、梯度噪声高，第一选择通常仍是 SGD 或 Adam。它们只用一阶信息，虽然单步方向不如二阶法精细，但便宜、稳、对噪声更宽容。

可以把常见方案理解成下面这个分层：

| 场景 | 更合适的方法 | 原因 |
|---|---|---|
| 小到中规模、强凸、Hessian 可得 | Newton | 曲率最准确，局部收敛最快 |
| 中规模、想要二阶效果但不写 Hessian | BFGS | 实现直接，理论成熟 |
| 高维光滑问题 | L-BFGS | 内存显著更低 |
| 高噪声深度学习训练 | SGD / Adam / mL-BFGS / K-FAC / Shampoo | 需要更稳的随机优化与结构近似 |

真实工程中，一个务实策略是“分阶段优化”：

1. 前期用 Adam 快速进入可用区域，因为它对初始化和噪声更不敏感。
2. 后期切到 L-BFGS 或其他 quasi-Newton 方法，利用更好的局部曲率加速收敛。
3. 若模型规模继续扩大，再考虑 K-FAC、Shampoo 这类结构化二阶近似。

这里要强调边界：L-BFGS 不是“Adam 的上位替代”。它更像在**局部精修阶段**有价值的工具，尤其当目标函数已经相对平滑、梯度噪声可控时才更能发挥优势。若你还处在损失剧烈震荡、batch 很小、非凸结构复杂的早期训练阶段，L-BFGS 的历史信息很容易过期，效果反而不如简单的一阶法。

---

## 参考资料

- StronglyConvex, Big Table of Convergence Rates: https://www.stronglyconvex.com/blog/big-table-of-convergence-rates.html
- Cornell SciML, Advanced Optimization: https://cvw.cac.cornell.edu/SciML/diffsim/advanced-optimization
- APMXL, BFGS Algorithm Details: https://apxml.com/courses/optimization-techniques-ml/chapter-2-second-order-optimization-methods/bfgs-algorithm-details
- OptimLib, L-BFGS API Notes: https://optimlib.readthedocs.io/en/latest/api/lbfgs.html
- SciPy, BFGS Documentation: https://docs.scipy.org/doc/scipy/reference/
- mL-BFGS distributed DNN experiments: https://pmc.ncbi.nlm.nih.gov/articles/PMC12393816/
- Springer overview on Newton and quasi-Newton convergence: https://link.springer.com/article/10.1007/s10107-022-01913-5
