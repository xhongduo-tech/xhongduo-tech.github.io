## 核心结论

数值稳定性是指：算法在浮点数舍入误差存在时，输出误差不会被算法结构不必要地放大。

这里的重点不是“浮点计算必须完全等于实数计算”。这在大多数情况下做不到。重点是：同一个数学公式，换一种等价写法，可能让浮点计算从可用变成不可用。

玩具例子是：

$$
\sqrt{1+x}-1 = \frac{x}{\sqrt{1+x}+1}
$$

这两个表达式在实数数学里完全等价。但当 `x = 1e-16` 时，直接计算 `sqrt(1 + x) - 1` 在 Python 的 binary64 浮点数里可能得到 `0.0`；而计算 `x / (sqrt(1 + x) + 1)` 可以得到接近 `5e-17` 的结果。不是公式错了，而是浮点数只能保留有限位，写法不同会导致舍入路径不同。

| 表达式 | 数学等价 | 浮点行为可能不同 | 是否更稳定 |
|---|---:|---:|---:|
| `sqrt(1 + x) - 1` | 是 | 是，两个接近的数相减 | 否 |
| `x / (sqrt(1 + x) + 1)` | 是 | 是，避免接近数相减 | 是 |
| `log(sum(exp(xs)))` | 是 | 是，`exp` 可能溢出 | 否 |
| `m + log(sum(exp(x - m)))` | 是 | 是，平移后更安全 | 是 |

浮点误差常用下面的模型描述：

$$
fl(a \oplus b) = (a \oplus b)(1+\delta),\quad |\delta| \le u
$$

其中 `fl` 表示浮点计算结果，$\oplus$ 表示一次基础运算，$u$ 是 unit roundoff，也就是一次正确舍入的相对误差上界。数值稳定算法的目标，就是让这些很小的 $\delta$ 不被减法、指数、长序列求和等结构放大。

---

## 问题定义与边界

实数是数学里的连续对象。浮点数是计算机里用有限二进制位表示的近似值。算法输出误差是指程序算出的结果和理想数学结果之间的差异。

数值稳定性讨论的是“算法在有限精度下是否可靠”，不是“输入数据本身是否有噪声”。如果传感器数据本来就错了，那是数据质量问题；如果数学公式正确但程序写法把舍入误差放大了，那才是数值稳定性问题。

binary64，也就是多数语言里的双精度浮点数，基本形式可以写成：

$$
x = (-1)^s \times (1.f)_2 \times 2^e
$$

其中 `s` 是符号位，`f` 是尾数部分，`e` 是指数。binary64 的有效精度是 $p=53$ 位。常用约定是：

$$
eps = 2^{-52},\quad u = eps / 2 = 2^{-53}
$$

`eps` 是 `1.0` 到下一个更大可表示浮点数之间的间隔。`u` 是一次正确舍入时常用的相对误差上界。

| 术语 | 白话解释 | 关注点 |
|---|---|---|
| 实数 | 数学中无限精度的数 | 理想结果 |
| 浮点数 | 计算机中有限位近似表示的数 | 可表示范围和精度 |
| 绝对误差 | `|computed - exact|` | 差了多少 |
| 相对误差 | `|computed - exact| / |exact|` | 相对真实值差了多少 |
| 舍入误差 | 结果无法精确表示时被迫取近似 | 单次运算误差 |
| 条件数放大 | 问题本身会放大输入扰动 | 输入轻微变化导致输出剧烈变化 |

一个新手容易忽略的例子是：

```text
1.000000000000001 - 1.000000000000000
```

数学上这是一个很小但明确的差。浮点计算中，两个数的前面有效位几乎完全相同，相减后大量有效位抵消，剩下的低位更容易被舍入误差影响。这类现象叫 catastrophic cancellation，中文常译为灾难性消去，意思是“接近的数相减后，有效数字大量丢失”。

边界也要说清楚：`exp(1000)` 得到溢出，不是普通意义上的“精度不够”。溢出是指真实结果超过了浮点数可表示范围。精度损失是“能表示但不够准”，溢出是“已经表示不了”。

---

## 核心机制与推导

浮点计算可以理解为：每一步基础运算先按实数规则得到理想结果，再舍入成最近的可表示浮点数。对加减乘除这类基础运算，常用模型是：

$$
fl(a \oplus b) = (a \oplus b)(1+\delta),\quad |\delta| \le u
$$

这个模型说明：单次运算的相对误差通常很小。但算法由很多步组成，小误差可能被结构放大。

减法是最常见的放大器。设要计算 `x - y`，当 $x \approx y$ 时，结果 $x-y$ 很小，而输入规模 $|x|+|y|$ 仍然很大。它的误差放大可以用近似条件数表示：

$$
\kappa_{sub} \approx \frac{|x| + |y|}{|x-y|}
$$

条件数是一个衡量“输入小扰动会被问题本身放大多少”的数。当 $x$ 和 $y$ 非常接近时，分母 $|x-y|$ 很小，$\kappa_{sub}$ 会变得很大。此时哪怕输入只有很小舍入误差，输出也可能明显不可靠。

第一个玩具例子是 `sqrt(1 + x) - 1`。当 `x` 很小时，`sqrt(1 + x)` 非常接近 `1`。直接减去 `1` 会触发接近数相减。稳定写法通过有理化变形：

$$
\sqrt{1+x}-1
= \frac{(\sqrt{1+x}-1)(\sqrt{1+x}+1)}{\sqrt{1+x}+1}
= \frac{x}{\sqrt{1+x}+1}
$$

这个改写避免了两个接近的数相减。

第二个例子是 log-sum-exp。它经常出现在分类模型、概率模型和深度学习中。朴素写法是：

$$
\log\left(\sum_i \exp(x_i)\right)
$$

如果某个 $x_i$ 很大，`exp(x_i)` 会溢出。稳定写法是先取最大值 $m = \max_i x_i$，再平移：

$$
LSE(x) = m + \log\left(\sum_i \exp(x_i - m)\right)
$$

因为所有 $x_i - m \le 0$，所以 `exp(x_i - m)` 最大不会超过 `1`，直接避免了正向溢出。

softmax 也用同样的思路：

$$
softmax_i(x) = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}
$$

推导流程可以概括为：

```text
朴素写法
  -> 接近数相减 / exp 溢出 / 长序列误差累积
  -> 找到数学等价变形
  -> 平移、因式分解、对数域或稳定求和
  -> 保留相同数学含义，改变浮点舍入路径
```

真实工程例子是 Transformer 或推荐系统里的 softmax。模型输出的 logits 可能跨度很大，例如 `[1000, 1001, 1002]`。直接 `exp(1002)` 在普通双精度里就会溢出。工程实现不会直接对原始 logits 做指数，而是先减最大值，把它变成 `[-2, -1, 0]`，再指数化和归一化。

---

## 代码实现

代码里的数值稳定性重点不是“把公式照抄进程序”，而是“把公式写成适合浮点计算的路径”。

先看 `sqrt(1 + x) - 1` 的不稳定写法和稳定写法：

```python
import math

def naive_sqrt_minus_one(x: float) -> float:
    return math.sqrt(1.0 + x) - 1.0

def stable_sqrt_minus_one(x: float) -> float:
    return x / (math.sqrt(1.0 + x) + 1.0)

x = 1e-16
naive = naive_sqrt_minus_one(x)
stable = stable_sqrt_minus_one(x)

assert naive == 0.0
assert abs(stable - 5e-17) < 1e-30
```

这里 `assert naive == 0.0` 不是说真实答案为 0，而是展示 binary64 的舍入路径已经把差值抹掉了。稳定写法保留了接近真实值的结果。

再看 log-sum-exp 和 softmax：

```python
import math

def naive_logsumexp(xs):
    return math.log(sum(math.exp(x) for x in xs))

def stable_logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

def stable_softmax(xs):
    m = max(xs)
    shifted = [math.exp(x - m) for x in xs]
    total = sum(shifted)
    return [v / total for v in shifted]

xs = [1000.0, 1001.0, 1002.0]

try:
    naive_logsumexp(xs)
    assert False, "naive version should overflow for this input"
except OverflowError:
    pass

lse = stable_logsumexp(xs)
probs = stable_softmax(xs)

assert math.isfinite(lse)
assert all(math.isfinite(p) for p in probs)
assert abs(sum(probs) - 1.0) < 1e-15
assert probs[-1] > probs[0]
```

这个例子对应真实工程中的分类概率计算。softmax 的输出需要归一化为概率，概率和应该接近 `1.0`。先减最大值不会改变 softmax 结果，因为分子和分母同时乘上了同一个比例因子。

长序列求和也需要注意。直接从左到右累加时，大数可能吞掉小数。Kahan summation 是一种补偿求和方法，白话解释是：额外记录每次被舍入丢掉的一小部分，后续再补回来。

```python
def naive_sum(xs):
    total = 0.0
    for x in xs:
        total += x
    return total

def kahan_sum(xs):
    total = 0.0
    c = 0.0
    for x in xs:
        y = x - c
        t = total + y
        c = (t - total) - y
        total = t
    return total

xs = [1e16, 1.0, -1e16]
assert naive_sum(xs) == 0.0

ys = [1.0] * 100000
assert kahan_sum(ys) == 100000.0
```

| 输入规模 | 朴素实现风险 | 稳定实现策略 | 适用场景 |
|---|---|---|---|
| 很小 `x` 的 `sqrt(1+x)-1` | 接近数相减 | 有理化改写 | 数学函数近似 |
| 大 logits | `exp` 溢出 | 先减最大值 | softmax、分类模型 |
| 很小概率连乘 | 下溢到 0 | 改到对数域 | HMM、CRF、序列概率 |
| 长数组求和 | 误差累积 | pairwise sum 或 Kahan | 统计、指标聚合 |

常见稳定原语包括 `logsumexp`、`logaddexp`、`log1p`、`expm1`、pairwise sum 和 Kahan summation。`log1p(x)` 表示稳定计算 `log(1+x)`；`expm1(x)` 表示稳定计算 `exp(x)-1`。它们专门处理“小量加 1”或“小量从 1 附近减掉”的场景。

---

## 工程权衡与常见坑

数值稳定性不是免费午餐。更稳定的写法可能增加分支、增加运算次数，也可能让代码不如原始公式直观。工程判断要看输入范围、误差要求和性能预算。

最常见的坑是把“数学等价”当成“浮点等价”。例如：

$$
(a+b)-a
$$

在实数里等于 $b$。但如果 $a$ 非常大，$b$ 非常小，`a+b` 可能直接舍入回 `a`，最后结果变成 `0`。

长序列求和是另一个常见坑。比如统计系统中累计几千万个金额、点击率、损失值，直接顺序累加会让误差随着长度扩大。pairwise sum 是把数组分治求和，先求局部和，再合并。它通常比从左到右累加更稳定，因为它减少了“大数吞小数”的机会。

真实工程中的 softmax 更典型。分类模型、推荐系统和 Transformer 都会产生 logits。如果 logits 跨度过大，直接 `exp(logit)` 会 overflow；如果概率很小，连乘还可能 underflow 到 `0`。先减最大值是工程里最常见、成本最低、效果最明确的修复方式。

| 常见坑 | 具体表现 | 规避策略 |
|---|---|---|
| 数学等价不等于浮点等价 | 等价公式算出不同结果 | 选择误差不放大的写法 |
| 先大数运算再减回去 | 小差值被舍入抹掉 | 因式分解或有理化 |
| 顺序累加长数组 | 小数被大累计值吞掉 | pairwise sum、Kahan |
| 忽略边界输入测试 | 正常样例通过，线上极端值失败 | 专门测极大、极小、近似相等 |
| 直接概率连乘 | 很快下溢到 0 | 改用对数域 |

规避策略可以按问题类型选择：

| 问题类型 | 策略 | 例子 |
|---|---|---|
| 指数可能溢出 | 平移 | `x - max(x)` |
| 接近数相减 | 因式分解或有理化 | `sqrt(1+x)-1` 改写 |
| 概率极小 | 对数域表达 | 乘法变加法 |
| 长序列累积 | 稳定求和 | pairwise sum、Kahan |
| 特殊小量函数 | 稳定原语 | `log1p`、`expm1` |

测试也要覆盖数值边界：

| 测试输入 | 目的 | 示例 |
|---|---|---|
| 近似相等输入 | 检查灾难性消去 | `1.000000000000001 - 1.0` |
| 极大值输入 | 检查溢出 | `exp(1000)`、大 logits |
| 极小值输入 | 检查下溢 | 很小概率连乘 |
| 长序列输入 | 检查误差累积 | 百万级浮点求和 |

---

## 替代方案与适用边界

没有单一“永远最稳定”的写法。数值稳定性要结合输入范围、性能成本、可读性和精度收益判断。

| 方案 | 输入范围 | 性能成本 | 可读性 | 精度收益 |
|---|---|---:|---:|---:|
| 朴素算术 | 范围窄、维度低 | 低 | 高 | 低到中 |
| 平移重写 | 有统一偏移不改变结果 | 低 | 中 | 高 |
| 对数域 | 概率很小、乘法链很长 | 中 | 中 | 高 |
| 专用稳定原语 | 常见数学函数场景 | 低到中 | 高 | 高 |

对数域计算是重要替代方案。对数域是指不直接保存概率 $p$，而是保存 $\log p$。这样多个概率相乘：

$$
p_1p_2p_3
$$

可以改成：

$$
\log p_1 + \log p_2 + \log p_3
$$

新手例子是连续乘很多小概率。假设每一步概率都是 `1e-10`，乘 100 次就是 `1e-1000`，这远小于普通浮点数能表示的正常范围，容易下溢到 `0`。在对数域里，它只是 `100 * log(1e-10)`，仍然是一个有限数。

真实工程中，HMM、CRF、序列标注、语音识别和搜索排序里的路径概率，经常需要对很长链路的概率做乘法。直接在概率域算会很脆弱，对数域更合适。

`logsumexp` 和直接 `log(sum(exp(x)))` 的边界也很明确。输入范围小、维度低时，朴素写法可能看起来没问题，例如 `x = [1, 2, 3]`。但只要输入跨度变大，或者维度变高，稳定原语更安全。工程代码里，如果这个函数处在模型训练、推理、指标统计等核心路径，优先使用稳定实现。

也不必过度优化。输入小、范围窄、误差不敏感时，可以优先保证代码清晰度。例如 UI 展示里的简单百分比计算，通常不需要引入复杂补偿求和。但边界测试仍然要保留，因为今天范围窄，不代表未来调用方不会传入极端值。

实践判断可以用三条规则：

1. 看到接近数相减，先怀疑灾难性消去。
2. 看到 `exp`、概率连乘、softmax，优先考虑平移或对数域。
3. 看到长序列求和，至少考虑 pairwise sum；精度要求高时考虑 Kahan 或库函数。

---

## 参考资料

基础理论：

1. David Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*  
   用来理解浮点表示、舍入误差和经典误差模型。

2. Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*  
   用来系统学习条件数、稳定性和误差分析方法。

语言文档：

1. Python 官方教程：*Floating-Point Arithmetic: Issues and Limitations*  
   用来理解 Python 中 `0.1 + 0.2`、舍入显示和 binary64 行为。

2. Python `math` 模块文档  
   用来确认 `log1p`、`expm1`、`fsum` 等稳定接口。

数值库文档：

1. NumPy 官方文档：`numpy.finfo`  
   用来确认 `eps`、最小正数、最大浮点数等机器参数。

2. SciPy 官方文档：`scipy.special.logsumexp`  
   用来学习工程里如何使用稳定的 log-sum-exp。

3. NumPy 官方文档：`numpy.logaddexp`  
   用来处理两个对数值相加的稳定计算。

论文：

1. Pierre Blanchard, Desmond J. Higham, Nicholas J. Higham, *Accurately Computing the Log-Sum-Exp and Softmax Functions*  
   用来深入理解 log-sum-exp 和 softmax 的稳定计算边界。

建议阅读路径：

1. 先读 Goldberg，建立浮点误差的基本模型。
2. 再读 Python 官方浮点说明，理解语言层表现。
3. 然后看 NumPy 和 SciPy 文档，掌握工程接口。
4. 最后读 Higham 相关材料，系统理解稳定算法设计。
