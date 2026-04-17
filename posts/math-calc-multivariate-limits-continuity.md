## 核心结论

多元函数的极限，比一元函数多了一个本质难点：**路径依赖**。路径依赖的意思是，同一个点可以从很多不同路线靠近，如果不同路线看到的函数值趋向不同结果，那么极限就不存在。

一元函数讨论 $\lim_{x\to a}f(x)$ 时，通常只需要比较左侧和右侧；多元函数讨论 $\lim_{\mathbf{x}\to \mathbf{a}}f(\mathbf{x})$ 时，没有“左”和“右”这么简单，因为 $\mathbf{x}$ 可以从平面、空间中的任意方向靠近 $\mathbf{a}$。因此，多元极限存在的要求更强：

$$
\lim_{\mathbf{x}\to \mathbf{a}} f(\mathbf{x}) = L
\iff
\forall \epsilon>0,\ \exists \delta>0,\ 
0<\|\mathbf{x}-\mathbf{a}\|<\delta \Rightarrow |f(\mathbf{x})-L|<\epsilon
$$

这里的“范数” $\|\mathbf{x}-\mathbf{a}\|$ 可以理解为“点 $\mathbf{x}$ 到目标点 $\mathbf{a}$ 的距离”。它不是单方向差值，而是把所有方向统一装进一个球形邻域里。这个定义的意思很直接：只要离目标点足够近，不管从哪个方向靠近，函数值都必须离 $L$ 足够近。

连续性则是在极限基础上再加一层要求。若函数在点 $\mathbf{a}$ 处有定义，且

$$
\lim_{\mathbf{x}\to \mathbf{a}}f(\mathbf{x}) = f(\mathbf{a}),
$$

就说 $f$ 在 $\mathbf{a}$ 处连续。连续的白话意思是：输入做很小变化，输出也只做很小变化，没有突然跳变。

玩具例子最能说明问题。考虑

$$
f(x,y)=\frac{xy}{x^2+y^2},\quad (x,y)\neq(0,0)
$$

问它在 $(0,0)$ 附近是否有极限。若沿路径 $y=0$ 靠近原点，

$$
f(x,0)=0
$$

所以沿这条路极限是 $0$。但若沿路径 $y=x$ 靠近原点，

$$
f(x,x)=\frac{x^2}{2x^2}=\frac12
$$

所以沿这条路极限是 $\frac12$。两条路径给出两个结果，因此整体极限不存在。

这个结论不是纯数学形式主义。深度学习里，损失函数是“参数到误差”的多元函数，参数向量往往有上百万维。优化器之所以能沿梯度前进，隐含前提就是局部变化足够平滑；如果局部像“路径一换就跳值”，训练就会不稳定，动量、学习率调度和梯度估计都会失真。

---

## 问题定义与边界

多元极限讨论的是：当输入点接近某个目标点时，函数输出是否稳定趋近某个固定值。这里“输入点”通常写作 $\mathbf{x}\in\mathbb{R}^n$，“目标点”写作 $\mathbf{a}\in\mathbb{R}^n$。$\mathbb{R}^n$ 的意思是 $n$ 维实数空间，二维平面和三维空间只是它的低维特例。

与一元函数相比，多元函数的判断规则可以先用一张表看清：

| 维度 | 输入靠近方式 | 极限存在的要求 | 常见误判 |
|---|---|---|---|
| 一元函数 $f(x)$ | 左侧、右侧 | 左极限 = 右极限 | 只看单边 |
| 二元函数 $f(x,y)$ | 任意直线、曲线、分段路径 | 所有路径同趋于一个值 | 只测几条直线 |
| 更高维函数 $f(\mathbf{x})$ | 任意方向、任意轨迹 | 所有局部靠近方式都一致 | 把数值抽样当严格证明 |

这里要明确两个边界。

第一，**“沿所有路径相同”是极限存在的必要条件，但在教学中通常用它来证伪，不常直接用它来证真。**  
原因很简单：路径太多，不可能真的逐条检查。找到两条路径给出不同结果，可以立刻证明极限不存在；但只检查了十条、百条路径都相同，仍然不能推出极限一定存在。要证明存在，通常需要不等式夹逼、极坐标变换、范数估计等整体方法。

第二，**连续不等于可导。**  
可导的白话意思是：函数不仅没有跳变，还能在局部被一个线性函数很好近似。连续只要求“不跳”，可导要求“局部像直线或平面”。例如绝对值函数在一元下是连续但在尖点不可导；多元里也有类似现象。对深度学习而言，连续性是底线，可导性决定梯度是否定义良好，二阶连续性还会影响 Hessian、牛顿法、曲率分析等更高阶方法。

一个新手容易混淆的问题是：偏导数存在，是不是就连续？答案是否定的。偏导数只是在坐标轴方向上看变化率，相当于只检查少数特殊路径；而连续性要求控制整个邻域。只会沿坐标轴看，不足以保证全方向都正常。

再看一个简单例子：

$$
g(x,y)=
\begin{cases}
\frac{x^2y}{x^4+y^2}, & (x,y)\neq(0,0) \\
0, & (x,y)=(0,0)
\end{cases}
$$

若沿 $y=0$，有 $g(x,0)=0$；若沿 $y=x^2$，有

$$
g(x,x^2)=\frac{x^4}{x^4+x^4}=\frac12
$$

所以它在原点也不连续。这个例子比 $\frac{xy}{x^2+y^2}$ 更有代表性，因为它告诉你：只测直线还不够，抛物线这类非线性路径也可能暴露问题。

---

## 核心机制与推导

多元极限的机制，本质上是“**用一个统一的局部半径，控制所有方向的输出误差**”。这就是 $\epsilon$-$\delta$ 定义的核心。

$\epsilon$ 可以理解为你允许输出误差有多大，$\delta$ 可以理解为你愿意把输入限制得多近。若极限存在，则对任意小的 $\epsilon$，总能找到一个统一的 $\delta$，使得只要点落进以 $\mathbf{a}$ 为中心、半径为 $\delta$ 的小球里，函数值就必然落进以 $L$ 为中心、半径为 $\epsilon$ 的区间里。

关键在“统一”二字。若某条路径需要 $\delta_1$，另一条路径需要完全不同的控制尺度，甚至根本无法控制到同一输出值，那就说明不存在一个统一邻域能把所有方向都管住，极限于是不存在。

先看可证明存在的例子：

$$
f(x,y)=x^2+y^2
$$

证明它在 $(0,0)$ 处极限为 $0$。设 $r=\sqrt{x^2+y^2}=\|(x,y)\|$，则

$$
|f(x,y)-0| = x^2+y^2 = r^2
$$

只要令 $\delta=\sqrt{\epsilon}$，当 $\|(x,y)\|<\delta$ 时，就有

$$
|f(x,y)| = r^2 < \delta^2 = \epsilon
$$

这说明无论从哪条路径进入原点，只要距离够小，输出一定够小，因此极限存在且为 $0$。

再看经典反例：

$$
f(x,y)=\frac{xy}{x^2+y^2}
$$

若尝试把它改写成极坐标，令 $x=r\cos\theta,\ y=r\sin\theta$，则

$$
f(x,y)=\frac{r^2\cos\theta\sin\theta}{r^2(\cos^2\theta+\sin^2\theta)}
=\cos\theta\sin\theta
=\frac12\sin 2\theta
$$

这一步非常关键。极坐标里的 $r$ 表示离原点的距离，$\theta$ 表示方向角。现在你会发现，$r\to0$ 时，表达式里根本没有 $r$ 了，只剩方向 $\theta$。这说明输出值不由“离得多近”决定，而由“从哪个方向靠近”决定。方向一变，函数值就变，所以极限不可能存在。

这就是路径依赖的代数机制：如果把函数写成“靠近尺度 $r$”和“方向变量 $\theta$”后，仍然保留明显的方向项，就高度可疑；若能化成只依赖 $r$ 且当 $r\to0$ 时收敛到固定值，则往往可证明极限存在。

连续性的定义只是把这个机制再推进一步：

$$
f \text{ 在 } \mathbf{a} \text{ 处连续}
\iff
\lim_{\mathbf{x}\to\mathbf{a}}f(\mathbf{x})=f(\mathbf{a})
$$

也就是说，函数在点上的“实际取值”和周围靠近它时“大家共同趋向的值”必须一致。若周围趋向 $L$，但函数硬把点值定义成别的数，那也是不连续。

真实工程例子可以放到神经网络训练中理解。设参数向量为 $\theta$，损失函数为 $\mathcal{L}(\theta)$。梯度下降每一步都假设：当参数从 $\theta$ 变到 $\theta+\Delta\theta$ 时，损失变化大致满足

$$
\mathcal{L}(\theta+\Delta\theta)
\approx
\mathcal{L}(\theta)+\nabla \mathcal{L}(\theta)^\top \Delta\theta
$$

这里 $\nabla \mathcal{L}$ 叫梯度，白话就是“各个参数方向上的最陡上升率”。这个近似成立，要求损失在局部至少连续、通常还要可微。若损失表面在很小尺度上频繁跳变，那么梯度就不再稳定代表局部趋势，动量会积累错误方向，学习率再怎么调也难救。

BatchNorm 在小 batch 下就是一个典型工程类比。它每个 batch 都用当前样本的均值和方差做归一化。若 batch 太小，这两个统计量会大幅波动，相当于网络内部某层的函数形式在相邻步之间发生跳动。数学上它未必真的成为不连续函数，但从数值优化角度看，轨迹会表现出“近似不连续”的现象，训练容易抖动甚至发散。

---

## 代码实现

多元极限的严格证明靠数学推导，但工程上可以先做数值检查。数值检查不能替代证明，却能快速发现明显的路径依赖。

下面给出一个可运行的 Python 例子。它做两件事：

1. 沿多条路径采样，观察函数值是否趋向一致。
2. 用断言验证经典反例确实会因为路径不同而给出不同极限。

```python
import math

def f_bad(x, y):
    if x == 0 and y == 0:
        return 0.0
    return (x * y) / (x * x + y * y)

def f_good(x, y):
    return x * x + y * y

def check_limit(f, paths, radii, tol=1e-3):
    records = []
    for r in radii:
        values = []
        for path in paths:
            x, y = path(r)
            values.append(f(x, y))
        spread = max(values) - min(values)
        records.append((r, values, spread))
    return records

paths = [
    lambda r: (r, 0.0),          # y = 0
    lambda r: (r, r),            # y = x
    lambda r: (r, r * r),        # y = x^2
    lambda r: (r * math.cos(r), r * math.sin(r)),  # 弯曲路径
]

radii = [1e-1, 1e-2, 1e-3, 1e-4]

bad_records = check_limit(f_bad, paths, radii)
good_records = check_limit(f_good, paths, radii)

# 反例：至少有某个半径层面，不同路径值明显不同
assert any(spread > 0.1 for _, _, spread in bad_records)

# 正例：随着半径减小，所有路径上的函数值都靠近 0
assert all(max(abs(v) for v in values) < 1e-2 for r, values, _ in good_records if r <= 1e-3)

for name, records in [("bad", bad_records), ("good", good_records)]:
    print(f"\nFunction: {name}")
    for r, values, spread in records:
        rounded = [round(v, 6) for v in values]
        print(f"r={r:.0e}, values={rounded}, spread={spread:.6f}")
```

这个例子里，`spread` 表示同一半径下不同路径取值的最大差。若一个函数在目标点附近真的有极限，那么当半径越来越小，`spread` 应该趋向 $0$。对 `f_bad`，沿 $y=0$ 与 $y=x$ 的取值会稳定分离；对 `f_good`，所有路径都会压向 $0$。

如果想写得更接近实际工程中的“loss continuity guard”，可以加入随机方向采样。随机方向不是证明工具，但能扩大覆盖面。

```python
import math
import random

def sample_random_paths(num_paths=20):
    paths = []
    for _ in range(num_paths):
        a = random.uniform(-2.0, 2.0)
        b = random.uniform(0.5, 2.0)
        paths.append(lambda r, a=a, b=b: (r, a * (r ** b)))
    return paths

def numeric_limit_guard(f, radii, tol):
    base_paths = [
        lambda r: (r, 0.0),
        lambda r: (0.0, r),
        lambda r: (r, r),
        lambda r: (r, r * r),
    ]
    paths = base_paths + sample_random_paths()
    for r in radii:
        values = [f(*path(r)) for path in paths]
        if max(values) - min(values) > tol:
            return False
    return True

assert numeric_limit_guard(f_good, [1e-2, 1e-3, 1e-4], tol=1e-2) is True
assert numeric_limit_guard(f_bad, [1e-2, 1e-3, 1e-4], tol=1e-2) is False
```

真实工程例子可以对应到训练前检查。比如你实现了自定义损失函数或归一化层，可以对一组很接近的参数扰动 $\Delta\theta$ 做采样，检查 $\mathcal{L}(\theta+\Delta\theta)$ 是否出现异常跳变。如果小扰动导致损失大幅离散，很可能有数值不稳定、除零、精度下溢、条件分支不平滑等问题。

---

## 工程权衡与常见坑

最常见的误区，是把“看起来差不多”当成“极限存在”。

第一坑：**只测直线。**  
很多反例在所有直线路径上都表现正常，但沿某条曲线路径会暴露问题。前面的 $g(x,y)=\frac{x^2y}{x^4+y^2}$ 就是典型。只检查 $y=mx$ 往往不够，至少还要考虑 $y=x^2$、$y=x^3$、极坐标弯曲路径等。

第二坑：**把数值实验当证明。**  
数值检查只能说明“目前没看到问题”，不能推出“严格成立”。原因有两个：路径无限多，浮点数精度有限。因此文章或代码评审里要把“数值证据”和“数学证明”分开写。

第三坑：**连续、可导、梯度稳定混为一谈。**  
连续只保证不跳；可导保证有局部线性近似；梯度稳定还涉及条件数、数值精度、批统计波动等实现层面问题。工程中很多训练问题不是理论上不连续，而是数值上表现得像不连续。

下面这张表把 BatchNorm 与 GroupNorm 放在“连续优化友好性”上比较一下：

| 方案 | 统计量来源 | 小 batch 表现 | 对优化轨迹的影响 | 适用场景 |
|---|---|---|---|---|
| BatchNorm | 当前 batch 的均值方差 | 容易波动 | 相邻 step 归一化基准跳动，梯度噪声大 | 大 batch CNN 常见 |
| GroupNorm | 单样本内分组统计 | 更稳定 | 对 batch 大小不敏感，轨迹更平滑 | 小 batch、检测分割 |
| LayerNorm | 单样本特征维统计 | 稳定 | 常用于序列模型 | Transformer |
| InstanceNorm | 单样本单通道统计 | 稳定但表达偏强约束 | 常用于风格迁移 | 图像生成 |

把它翻成直白语言：BatchNorm 在小 batch 下像“每次称重都换一台没校准好的秤”，输入明明差不多，输出却因为统计基准变化而抖动。GroupNorm 更像“每件物品都用自己内部统一标准衡量”，对 batch 大小不敏感。

第四坑：**混合精度训练中的精度截断。**  
FP16 的可表示范围和精度都比 FP32 小。很小的梯度可能直接下溢为 $0$，一些相近但不相等的值会被量化成同一个数。数学上连续的函数，落到有限精度表示后，数值轨迹可能变成一段一段的台阶。优化器看到的就不再是平滑表面，而是近似离散表面。

常见规避方案如下：

1. `GroupNorm` 或其他不依赖 batch 统计的归一化，替代小 batch 下的 `BatchNorm`。
2. 增大有效 batch size，可通过梯度累加实现，而不必真的把显存翻倍。
3. 使用 `FP32` 主权重，让参数更新保留高精度。
4. 使用 `loss scaling`，把很小的梯度先放大，再回传，减少 FP16 下溢。
5. 对自定义损失与算子做数值梯度检查，排查除零、log 负数、sqrt 负数等不稳定点。

这些方案的共同目标，不是把训练变成严格数学上的处处光滑，而是尽可能恢复“局部小变化对应小输出变化”的优化条件。

---

## 替代方案与适用边界

并不是所有工程问题都能严格证明极限和连续性，尤其在高维非凸优化里，严格证明往往成本极高。这时可以使用一些“近似连续化”的替代办法，但必须知道边界在哪里。

第一类替代方案是**数值平滑**。  
例如动量、梯度累积、指数滑动平均等，都是把相邻步骤的信息做平均，让优化路径不至于对单步噪声过度敏感。它们的作用更像“滤波器”，不是从理论上修复函数本身，而是减弱数值层面的跳动。适合工程训练稳定化，不适合代替极限证明。

第二类替代方案是**用光滑函数替换非光滑函数**。  
最常见例子是 ReLU。ReLU 定义为

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

它在 $x=0$ 处连续，但不可导。不可导的白话意思是：拐角太尖，无法用唯一斜率描述。如果你需要更平滑的梯度，可用 softplus：

$$
\mathrm{softplus}(x)=\log(1+e^x)
$$

softplus 可以看成 ReLU 的平滑近似。它在全域连续且可导，且当 $x$ 很大时接近 $x$，当 $x$ 很小时接近 $0$。

下面给出一个简单代码：

```python
import math

def relu(x):
    return max(0.0, x)

def softplus(x):
    return math.log1p(math.exp(x))

xs = [-10, -1, 0, 1, 10]
relu_vals = [relu(x) for x in xs]
softplus_vals = [round(softplus(x), 6) for x in xs]

assert relu(-1) == 0.0
assert relu(2) == 2.0
assert softplus(10) > 9.9
assert softplus(-10) < 1e-3

print("relu:", relu_vals)
print("softplus:", softplus_vals)
```

玩具层面上，这说明一个尖角函数可以被一个圆滑函数逼近。真实工程里，若某个自定义激活、门控函数、裁剪操作在关键区间引入过于剧烈的折点，可以考虑用 softplus、sigmoid 近似、Huber loss 等平滑版本替代。

但边界必须说清楚：

| 方法 | 能解决什么 | 不能解决什么 | 适用边界 |
|---|---|---|---|
| 数值路径采样 | 快速发现明显路径依赖 | 不能严格证明极限存在 | 调试、单测、训练前检查 |
| 动量/梯度累积 | 降低步间抖动 | 不改变函数本身的理论性质 | 工程稳定化 |
| softplus 替代 ReLU | 提供更平滑梯度 | 可能改变稀疏性与收敛行为 | 对平滑性敏感的模型 |
| 极坐标/范数估计 | 严格证明极限存在或不存在 | 推导可能复杂 | 理论分析、教材问题 |

因此，替代方案只能作为“工程近似修复”。如果你在做数学证明题，必须回到定义、不等式和路径构造；如果你在做系统调参，数值守卫和光滑近似往往更实用。

---

## 参考资料

- LibreTexts, “Limits and Continuity of Multivariable Functions”  
  用途：多元极限、连续性的正式定义与基础例题。

- APEX Calculus / PCC, “Limits of Multivariable Functions”  
  用途：路径依赖的直观解释，帮助理解“任意路径”这一点。

- Vaia 对经典题 $\frac{xy}{x^2+y^2}$ 的解析  
  用途：展示沿不同路径趋近原点导致极限不存在的标准反例。

- ScienceDirect 上关于 Batch Normalization 的资料  
  用途：理解 batch 统计量波动如何影响训练稳定性。

- NVIDIA, “Train With Mixed Precision”  
  用途：解释 FP16、loss scaling、FP32 主权重等混合精度训练实践。

- 深度学习框架官方文档中关于 `BatchNorm`、`GroupNorm`、`LayerNorm` 的说明  
  用途：比较不同归一化方案在小 batch 场景下的行为差异。
