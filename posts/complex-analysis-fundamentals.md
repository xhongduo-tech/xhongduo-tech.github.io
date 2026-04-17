## 核心结论

复分析研究的是复变函数，也就是自变量和函数值都允许是复数的函数。这里最重要的对象是解析函数。解析函数指在某个开区域内处处复可导的函数；“复可导”不是沿某一条方向可导，而是沿任意方向逼近时导数都一致。

对初学者，最该先记住三件事：

1. 解析函数的局部线性近似一定是“缩放 + 旋转”。这就是它保角，也就是局部保持角度的原因。
2. 在单连通区域内，如果函数处处解析，那么沿任意正向闭合曲线的积分都为零：
   $$
   \oint_C f(z)\,dz=0
   $$
3. 一旦区域内部出现孤立极点，闭路积分就不再由“函数整体”决定，而是只由这些极点的留数决定：
   $$
   \oint_C f(z)\,dz = 2\pi i \sum_k \operatorname{Res}(f,z_k)
   $$

这两条从“积分为零”到“由少数极点贡献全部积分”的过渡，就是复分析最强的计算能力来源。

先看一个最小例子。取
$$
f(z)=z^2
$$
它是整函数，也就是整个复平面都解析，所以对任意不穿过无穷远点的闭合曲线都有
$$
\oint_{|z|=1} z^2\,dz = 0
$$
这里的结论不依赖单位圆这一条特殊路径，换成任意简单闭合曲线也成立，只要曲线围成的区域里没有奇点。

再看
$$
f(z)=\frac1z
$$
它在 $z=0$ 有一个孤立极点。若单位圆包住原点，则
$$
\oint_{|z|=1}\frac1z\,dz = 2\pi i
$$
同样是单位圆、同样是闭路积分，结果从 $0$ 变成 $2\pi i$，差别只来自内部是否含有奇点。这就是留数定理最核心的判断方式：

| 被积函数 | 路径内部是否有奇点 | 闭路积分结果 |
|---|---:|---:|
| $z^2$ | 否 | $0$ |
| $\frac1z$ | 是，原点 | $2\pi i$ |
| $\frac1{z-2}$，路径仍为 $|z|=1$ | 否 | $0$ |

公式上，柯西积分公式是整个体系的中心：
$$
f(z_0)=\frac{1}{2\pi i}\oint_C \frac{f(z)}{z-z_0}\,dz
$$
它说明区域内部一点的函数值，可以完全由边界上的积分恢复。对工程直觉来说，这意味着“内部信息可由边界编码”。

还可以顺手记住它的高阶版本：
$$
f^{(n)}(z_0)=\frac{n!}{2\pi i}\oint_C \frac{f(z)}{(z-z_0)^{n+1}}\,dz
$$
这说明解析函数不只是可导一次，而是可以无限次求导，并且每一阶导数都能由边界积分表示。这种“边界控制内部”的强约束，是复分析比普通实变积分强得多的原因。

---

## 问题定义与边界

先把讨论边界收紧，否则公式很容易误用。

解析函数的标准写法是
$$
f(z)=u(x,y)+iv(x,y),\quad z=x+iy
$$
其中 $u,v$ 是实函数。若 $f$ 在某点复可导，并且偏导足够好，则需满足柯西-黎曼方程：
$$
\frac{\partial u}{\partial x}=\frac{\partial v}{\partial y},\qquad
\frac{\partial u}{\partial y}=-\frac{\partial v}{\partial x}
$$
这组方程的含义不是“形式上长得对就行”，而是要求函数在二维平面上的横向变化和纵向变化彼此兼容，这样从不同方向逼近时才会得到同一个复导数。

但只写方程还不够，实际使用时要同时看“区域”和“奇点类型”。

| 条件 | 典型情况 | 能否直接用柯西积分定理 | 能否直接用留数定理 |
|---|---|---:|---:|
| 区域是开集，函数处处解析 | 整函数、圆盘内无奇点 | 可以 | 不需要 |
| 区域是开集、单连通，内部有孤立极点 | 有限个极点、无分支点 | 不可以 | 可以 |
| 区域非单连通 | 挖去一个点或一条裂缝 | 需额外检查路径同伦关系 | 视奇点和路径而定 |
| 含分支点 | $\sqrt z,\log z$ | 一般不可以直接套 | 一般不可以直接套 |
| 奇点在边界上 | 极点落在积分曲线上 | 不可以 | 不可以 |

这里有三个常见术语：

- 开集：每个点附近都还能取一个小圆盘而不跑出区域。
- 单连通：区域里没有“洞”，闭合曲线都能连续缩成一点。
- 孤立极点：函数只在某一点坏掉，而且附近其他点都还正常。

再补一个术语表，避免新手一开始就被“奇点”几个词混住：

| 术语 | 直观含义 | 典型例子 | 能否直接用留数 |
|---|---|---|---:|
| 可去奇点 | 这个点暂时没定义，但补上后可解析 | $\frac{\sin z}{z}$ 在 $z=0$ | 通常不必单独用 |
| 极点 | 函数在该点发散，像 $\frac1{(z-z_0)^m}$ | $\frac1z,\frac1{(z-1)^2}$ | 可以 |
| 本性奇点 | Laurent 负幂项无限多 | $e^{1/z}$ 在 $z=0$ | 可以算留数，但局部行为复杂 |
| 分支点 | 绕一圈后函数值换层 | $\sqrt z,\log z$ 在 $z=0$ | 不能直接按普通留数套 |

初学版例子是
$$
f(z)=\frac1z
$$
若路径是单位圆 $|z|=1$，原点在内部，所以可以用留数定理，结果是 $2\pi i$。若路径换成以 $2$ 为圆心、半径 $1/2$ 的小圆，原点不在内部，则积分为 $0$。这说明留数法不是“看到极点就算”，而是“只算路径内部的极点”。

把这个判断写成一句更可执行的话：

$$
\oint_C f(z)\,dz
\quad\text{只取决于 } C \text{ 内部的奇点，不取决于外部奇点。}
$$

还要强调：留数定理常写成“区域内部仅有孤立极点”。这是为了排除分支点这类更麻烦的奇性。比如 $\sqrt z$ 在 $z=0$ 不是极点，而是分支点。此时函数值绕原点一圈后会切换分支，不能直接按普通单值解析函数处理。

最后给一个初学者最容易混淆的反例：
$$
f(z)=\overline z
$$
它在实分析意义下当然“可导得动”，但不是解析函数。因为写成
$$
\overline z=x-iy
$$
对应
$$
u(x,y)=x,\qquad v(x,y)=-y
$$
于是
$$
u_x=1,\quad v_y=-1
$$
不满足柯西-黎曼方程，所以它不是解析函数。这个例子说明：复可导比实可导严格得多。

---

## 核心机制与推导

复导数为什么比实导数更强？因为它要求极限
$$
f'(z_0)=\lim_{h\to 0}\frac{f(z_0+h)-f(z_0)}{h}
$$
与 $h$ 的逼近方向无关。实分析里通常只沿数轴左右逼近；复分析里 $h$ 来自整个二维平面，方向无限多，因此约束更强。

设
$$
f(z)=u(x,y)+iv(x,y)
$$
沿实轴方向取 $h=\Delta x$，得到
$$
f'(z)=u_x+iv_x
$$
沿虚轴方向取 $h=i\Delta y$，得到
$$
f'(z)=v_y-iu_y
$$
两者相等，就得到
$$
u_x=v_y,\qquad u_y=-v_x
$$
这就是柯西-黎曼方程。

### 1. 从复导数到“旋转 + 缩放”

解析函数在局部的一阶近似可以写成
$$
f(z_0+\Delta z)\approx f(z_0)+f'(z_0)\Delta z
$$
把复数 $f'(z_0)=a+ib$ 看成一个线性变换，它对应的实二维矩阵是
$$
J=
\begin{pmatrix}
u_x & u_y\\
v_x & v_y
\end{pmatrix}
=
\begin{pmatrix}
a & -b\\
b & a
\end{pmatrix}
$$
这个矩阵的作用等价于：

1. 把长度乘上 $\sqrt{a^2+b^2}=|f'(z_0)|$
2. 再旋转一个角度 $\arg f'(z_0)$

所以在 $f'(z_0)\neq 0$ 的点，解析函数局部保持角度。这就是“保角”的来源。它不是额外性质，而是复可导本身带出来的线性结构。

### 2. 从“积分为零”到柯西积分公式

若 $f$ 在单连通区域内解析，则柯西积分定理给出
$$
\oint_C f(z)\,dz=0
$$
这条结论很强，因为它意味着同一类路径上的积分只与端点有关。进一步，考虑
$$
g(z)=\frac{f(z)-f(z_0)}{z-z_0}
$$
虽然分母在 $z=z_0$ 看起来有问题，但由于分子也同时为零，实际上这个点是可去奇点。于是 $g$ 仍解析，可以对它使用柯西积分定理。整理后得到
$$
f(z_0)=\frac{1}{2\pi i}\oint_C\frac{f(z)}{z-z_0}\,dz
$$
这就是柯西积分公式。

它的意义可以拆成三层：

| 层次 | 结论 | 含义 |
|---|---|---|
| 函数值 | $f(z_0)$ 由边界积分决定 | 内部信息可由边界恢复 |
| 导数 | $f^{(n)}(z_0)$ 也有边界积分公式 | 解析函数自动无限可导 |
| 估计 | 可以推出最大模原理等结论 | 局部行为受到全局约束 |

### 3. 从 Laurent 展开到留数

再向前一步，如果函数在某点不解析，但这种“不解析”可以用 Laurent 展开描述：
$$
f(z)=\sum_{n=-\infty}^{\infty} a_n (z-z_0)^n
$$
那么其中 $a_{-1}$ 就叫留数。为什么偏偏是这一项重要？因为对闭路积分而言，几乎所有幂次项都没有净贡献：

$$
\oint_C (z-z_0)^n\,dz = 0 \quad (n\neq -1)
$$

只有
$$
\oint_C \frac{1}{z-z_0}\,dz = 2\pi i
$$
因此
$$
\oint_C f(z)\,dz = 2\pi i\cdot a_{-1}
$$
若内部有多个孤立极点，就把每个极点的 $a_{-1}$ 相加，这就是留数定理：
$$
\oint_C f(z)\,dz = 2\pi i\sum_k \operatorname{Res}(f,z_k)
$$

这里可以把留数理解成一句更朴素的话：

$$
\text{留数}=\text{奇点附近 Laurent 展开里 } \frac1{z-z_0} \text{ 的系数}
$$

这也是为什么很多计算最后都在找“哪一项会留下净积分”。

### 4. 常见留数公式

对简单极点，如果
$$
f(z)=\frac{\phi(z)}{\psi(z)},\qquad \psi(z_0)=0,\ \psi'(z_0)\neq 0
$$
那么
$$
\operatorname{Res}(f,z_0)=\frac{\phi(z_0)}{\psi'(z_0)}
$$

对 $m$ 阶极点，留数计算公式是
$$
\operatorname{Res}(f,z_k)=\frac{1}{(m-1)!}\lim_{z\to z_k}\frac{d^{m-1}}{dz^{m-1}}\left((z-z_k)^m f(z)\right)
$$

两种公式适用边界如下：

| 极点类型 | 最方便的公式 |
|---|---|
| 简单极点 | $\operatorname{Res}(f,z_0)=\lim_{z\to z_0}(z-z_0)f(z)$ |
| $\frac{\phi}{\psi}$ 型简单极点 | $\operatorname{Res}(f,z_0)=\frac{\phi(z_0)}{\psi'(z_0)}$ |
| 高阶极点 | 导数公式 |
| 已有 Laurent 展开 | 直接读 $a_{-1}$ |

玩具例子看
$$
f(z)=\frac{z+1}{(z-i)(z+2)}
$$
在 $z=i$ 附近，把 $(z-i)$ 提出来：
$$
(z-i)f(z)=\frac{z+1}{z+2}
$$
所以
$$
\operatorname{Res}(f,i)=\frac{i+1}{i+2}
$$
把分母有理化一下更直观：
$$
\frac{i+1}{i+2}
=\frac{(1+i)(2-i)}{(2+i)(2-i)}
=\frac{3+i}{5}
$$
若积分路径只包住 $z=i$ 而不包住 $z=-2$，那么
$$
\oint_C f(z)\,dz=2\pi i\cdot \frac{3+i}{5}
$$
进一步化简可得
$$
\oint_C f(z)\,dz
= \frac{2\pi}{5}(-1+3i)
$$
这比直接参数化路径积分要简单得多，因为我们只需要极点附近的一阶信息。

---

## 代码实现

用 `sympy` 可以直接做留数计算，也可以自己写“圆形路径内极点求和”的最小实现。下面给出一个可运行的脚本，覆盖三件事：

1. 自动找出有理函数的极点
2. 判断哪些极点在圆路径内部
3. 用留数定理计算闭路积分，并和参数化积分做符号验证

```python
import sympy as sp

z, t = sp.symbols("z t", complex=True)
I = sp.I
pi = sp.pi


def poles_of_rational(expr):
    """Return poles of a rational function as a list of roots."""
    _, den = sp.fraction(sp.together(expr))
    den = sp.expand(den)
    return sp.solve(sp.Eq(den, 0), z)


def is_inside_circle(point, center, radius, tol=1e-10):
    """Strictly inside the circle; boundary points are treated as outside."""
    point_c = complex(sp.N(point))
    center_c = complex(sp.N(center))
    return abs(point_c - center_c) < float(radius) - tol


def residues_inside_circle(expr, center=0, radius=1):
    """Return (pole, residue) pairs for poles strictly inside |z-center|<radius."""
    poles = poles_of_rational(expr)
    result = []
    for p in poles:
        if is_inside_circle(p, center, radius):
            result.append((sp.simplify(p), sp.simplify(sp.residue(expr, z, p))))
    return result


def contour_integral_by_residue(expr, center=0, radius=1):
    """Compute ∮ f(z) dz on the positively oriented circle |z-center|=radius."""
    pairs = residues_inside_circle(expr, center, radius)
    residue_sum = sp.simplify(sum(res for _, res in pairs))
    return sp.simplify(2 * pi * I * residue_sum), pairs


def contour_integral_by_param(expr, center=0, radius=1):
    """Direct parameterization check: z(t)=center+radius*e^{it}, t in [0,2π]."""
    z_t = center + radius * sp.exp(I * t)
    dz_dt = sp.diff(z_t, t)
    integrand = sp.simplify(expr.subs(z, z_t) * dz_dt)
    return sp.simplify(sp.integrate(integrand, (t, 0, 2 * pi)))


if __name__ == "__main__":
    # 例1：整函数 z^2，在任何闭合圆上积分都应为 0
    expr1 = z**2
    val1, poles1 = contour_integral_by_residue(expr1, center=0, radius=1)
    direct1 = contour_integral_by_param(expr1, center=0, radius=1)

    assert poles1 == []
    assert sp.simplify(val1) == 0
    assert sp.simplify(direct1) == 0

    # 例2：1/z 在单位圆上的积分为 2πi
    expr2 = 1 / z
    val2, poles2 = contour_integral_by_residue(expr2, center=0, radius=1)
    direct2 = contour_integral_by_param(expr2, center=0, radius=1)

    assert poles2 == [(0, 1)]
    assert sp.simplify(val2 - 2 * pi * I) == 0
    assert sp.simplify(direct2 - 2 * pi * I) == 0

    # 例3：(z+1)/(z^2+1) 在半径 1.1 的圆内包含 i 和 -i
    expr3 = (z + 1) / (z**2 + 1)
    val3, poles3 = contour_integral_by_residue(expr3, center=0, radius=1.1)
    direct3 = contour_integral_by_param(expr3, center=0, radius=1.1)

    res_i = sp.residue(expr3, z, I)
    res_minus_i = sp.residue(expr3, z, -I)

    assert set(p for p, _ in poles3) == {I, -I}
    assert sp.simplify(val3 - 2 * pi * I * (res_i + res_minus_i)) == 0
    assert sp.simplify(val3 - direct3) == 0

    print("Example 1 integral:", sp.simplify(val1))
    print("Example 2 integral:", sp.simplify(val2))
    print("Example 3 poles and residues:", poles3)
    print("Example 3 integral:", sp.simplify(val3))
    print("all assertions passed")
```

运行这段脚本时，三组结果分别对应：

| 例子 | 路径内部极点 | 积分结果 |
|---|---|---|
| $z^2$ | 无 | $0$ |
| $\frac1z$ | $0$ | $2\pi i$ |
| $\frac{z+1}{z^2+1}$，半径 $1.1$ | $\pm i$ | $2\pi i\big(\operatorname{Res}(f,i)+\operatorname{Res}(f,-i)\big)$ |

这段代码的实现逻辑很直接：

1. 把有理函数整理成分子分母形式。
2. 解出分母为零的位置，得到候选极点。
3. 判断极点是否在圆路径内部。
4. 对内部极点求留数并乘上 $2\pi i$。
5. 再用路径参数化积分做一次核对，避免“代码写对了公式、路径却写错了方向”。

这里顺手解释两个初学者容易混淆的点。

第一，`sp.residue(expr3, z, I)` 只是在 $z=i$ 处求局部留数，不代表整条路径积分。路径若把 $i$ 和 $-i$ 都包进去，就必须把两个留数都加起来。

第二，路径几何决定“算哪些极点”，函数表达式本身不决定。代码里“找极点”只是第一步，“判断是否在路径内部”同样关键。

再补一个更贴近工程的例子。设控制系统传递函数
$$
F(s)=\frac{2}{(s+1)(s+2)}
$$
做部分分式分解：
$$
\frac{2}{(s+1)(s+2)}=\frac{2}{s+1}-\frac{2}{s+2}
$$
所以逆拉普拉斯变换得到
$$
f(t)=2e^{-t}-2e^{-2t},\qquad t>0
$$
留数法给出的视角是：$s=-1,-2$ 是两个极点，它们分别对应两个指数模态，而系数就是对应的留数权重。对工程实现者，这意味着“求响应”可以转化为“找极点 + 算权重”。

---

## 工程权衡与常见坑

留数法在工程里最有价值的场景，不是课堂上的圆积分，而是频域分析、逆拉普拉斯变换、稳定性分析和信号响应计算。因为很多问题原本需要解微分方程，换到复平面后，变成极点分类和留数求和。

以控制系统为例：
$$
F(s)=\frac{2}{(s+1)(s+2)}
$$
若求逆拉普拉斯，常见做法是在复平面上选取合适半平面闭合路径。对 $t>0$，通常向左半平面闭合，使大圆弧贡献消失，再用留数定理把积分变成极点求和。这里每个极点都对应一个系统模态，极点实部决定衰减或发散速度，虚部决定振荡频率。

把这件事和工程解释对应起来：

| 极点位置 | 时域含义 | 稳定性含义 |
|---|---|---|
| 实部 $<0$ | 指数衰减 | 稳定 |
| 实部 $>0$ | 指数增长 | 不稳定 |
| 纯虚极点 | 持续振荡 | 临界稳定附近需单独分析 |
| 共轭复极点 | 衰减或增长振荡 | 由实部控制包络，由虚部控制频率 |

这类方法的优势是统一，但坑也集中。

| 常见失误 | 后果 | 规避方法 |
|---|---|---|
| 路径方向写反 | 积分整体差一个负号 | 默认使用正向，即逆时针 |
| 极点落在边界上 | 公式直接失效 | 改路径、做主值积分，或单独处理 |
| 把分支点当极点 | 结果完全错误 | 先判断是极点、可去奇点还是分支点 |
| 忽略区域不是单连通 | 错误套用“闭路积分为零” | 先看区域是否有洞 |
| 大圆弧贡献未验证就丢弃 | 漏项 | 检查 Jordan 引理或做衰减估计 |
| 只算极点，不判断是否在内部 | 多算或漏算 | 先确定路径包围关系 |
| 数值实现只解分母零点 | 对非有理函数失效 | 明确函数类型，必要时改用数值积分 |

再给一个初学版坑：很多人看到
$$
\oint_C \frac{1}{z}\,dz
$$
就直接写成 $2\pi i$。这只有在路径正向包住原点时才成立。若路径不包住原点，结果是 $0$；若顺时针走，结果是 $-2\pi i$；若原点在边界上，普通留数公式不能直接用。

可以把这三种情况写成一张最小判断表：

| 路径情况 | $\displaystyle \oint_C \frac1z\,dz$ |
|---|---:|
| 逆时针包住原点一次 | $2\pi i$ |
| 不包住原点 | $0$ |
| 顺时针包住原点一次 | $-2\pi i$ |

另一个常见坑是 Jordan 引理。它可以粗略理解为“某些指数项乘上有理函数时，大半圆那一段会自动衰减到零”，但这不是默认成立的。比如闭合方向选错，指数因子可能在大圆上爆炸，那就不能把弧段积分直接丢掉。

还有一个更细的坑：有些教材例子只处理简单极点，容易让人误以为“留数就是把 $(z-z_0)$ 乘回去再代入”。这只对简单极点成立。若是二阶或更高阶极点，就必须用导数公式或 Laurent 展开，直接代入会少掉关键项。

---

## 替代方案与适用边界

留数法很强，但不是“遇到积分就上”。判断标准可以先压缩成一张表。

| 场景 | 优先方法 |
|---|---|
| 有理函数，闭合路径明确，只有孤立极点 | 留数法 |
| 逆拉普拉斯、频域响应、稳定性分析 | 留数法 |
| 含分支点，如 $\sqrt z,\log z$ | 先选分支，再做参数化或割线轮廓 |
| 路径穿过奇点 | 改路径或做主值积分 |
| 非解析区域、边界不规则且无明显极点结构 | 实轴参数化或数值积分 |
| 只需数值结果，不追求解析式 | 数值积分更直接 |

一个典型反例是
$$
f(z)=\sqrt{z}
$$
这里 $z=0$ 是分支点，不是极点。分支点可以理解为“绕一圈后函数值会跳到另一层”，所以函数不是普通单值解析函数。若你直接把它套进留数定理，逻辑前提就已经坏了。

这时更稳妥的做法是先选分支，比如主值分支，再把路径参数化。若在极坐标里写
$$
z=re^{i\theta}
$$
则
$$
\sqrt z=\sqrt r\,e^{i\theta/2}
$$
但这里的 $\theta$ 不能随便绕一整圈，否则函数值会切换分支。也就是说，分支切线不是技术细节，而是定义的一部分。

再看另一个常见对象：
$$
\log z=\ln|z|+i\arg z
$$
问题不在公式本身，而在 $\arg z$ 不是单值的。你必须先规定
$$
-\pi<\arg z<\pi
$$
或者别的角度区间，函数才有明确意义。一旦选了主值分支，通常还要沿负实轴或别的射线切开平面。此时路径如果跨过分支切线，积分规则又会变化，所以不能把它和“普通极点”混为一类。

所以“何时用留数、何时不用”可以简单记成两句话：

1. 只要函数在路径内外是单值解析，且奇点是孤立极点，优先考虑留数法。
2. 只要遇到分支点、边界奇点、非开区域或路径穿奇点，先停下来检查定义域，再决定改路径、改变量还是改成数值方法。

对初学者，更实用的操作顺序是：

1. 先问函数是否单值。
2. 再问奇点是不是孤立极点。
3. 再问奇点是否在路径内部而不是边界上。
4. 最后才开始套公式。

这个顺序看起来慢，实际比“先算再说”更快，因为多数错误都出在前提没检查。

---

## 参考资料

- [complex-analysis.com: Cauchy Integral Formula](https://complex-analysis.com/content/cauchy_integral_formula.html)  
  讲清楚柯西积分公式、导数表示和其几何含义，适合理解“边界决定内部”这条主线。

- [MathWorld: Cauchy-Riemann Equations](https://mathworld.wolfram.com/Cauchy-RiemannEquations.html)  
  定义紧凑，符号统一，适合快速核对柯西-黎曼方程和常见记号。

- [Wikipedia: Cauchy–Riemann equations](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations)  
  适合查定义、条件、几何解释以及与保角映射的关系。

- [Wikipedia: Cauchy's integral formula](https://en.wikipedia.org/wiki/Cauchy%27s_integral_formula)  
  适合补足高阶导数公式、最大模原理等后续联系。

- [Wikipedia: Residue theorem](https://en.wikipedia.org/wiki/Residue_theorem)  
  系统整理留数定理、Laurent 展开、极点分类与典型计算模板。

- [Wikipedia: Laurent series](https://en.wikipedia.org/wiki/Laurent_series)  
  若想真正理解“为什么是 $a_{-1}$ 留下净贡献”，这一条很关键。

- [GeeksforGeeks: Residue Theorem](https://www.geeksforgeeks.org/engineering-mathematics/residue-theorem/)  
  例子较直观，适合快速复习“如何从极点算闭路积分”。

- Churchill, Brown, Verhey, *Complex Variables and Applications*  
  经典入门教材，适合把解析函数、柯西理论、留数法连成一个完整体系。

- Ahlfors, *Complex Analysis*  
  更偏数学结构，适合在入门后继续深入保角映射、解析延拓与 Riemann 曲面。

- Ablowitz, Fokas, *Complex Variables: Introduction and Applications*  
  更强调与微分方程、物理和工程问题的连接，适合从“为什么这些公式有用”这个角度继续读。
