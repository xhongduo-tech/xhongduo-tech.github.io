---
title: 柯西-黎曼条件（Cauchy-Riemann 方程）
date: 2026-08-07
---

# 柯西-黎曼条件（Cauchy-Riemann 方程）

<div class="epigraph">
<p>解析函数的实部与虚部不是两个独立的函数，而是一枚硬币的两面。</p>
<footer>—— 复分析常识</footer>
</div>

<div class="article-byline">
<p>第二级 · 复变函数与积分变换 ｜ 西交《复变函数》§2.2 ｜ 2026-08-07</p>
</div>

## 为什么柯西与黎曼同时出现

上一节我们得到可导的必要条件——**柯西-黎曼方程**：$u_x = v_y$、$u_y = -v_x$。这一节要把这对方程讲透：它们的历史、它们的几何含义、以及如何用它快速判别一个函数是否可导。柯西（Cauchy）与黎曼（Riemann）相隔几十年，却先后独立发现这对方程对复分析的核心地位——正如黎曼本人所说，这个条件把复变函数理论从「形式运算」变成了「真正的分析」。<span class="marginnote">柯西在 1821 年就写出这对方程，黎曼则在 1851 年的博士论文《单复变函数一般理论的基础》里把它作为解析函数的<strong>定义核心</strong>，并由此发展出黎曼曲面与共形映射。物理学家达朗贝尔在更早的 1752 年已在流体力学里用过它们，所以有时也叫达朗贝尔-欧拉方程。</span>

## 1 柯西-黎曼方程的推导

设 $f(z) = u(x,y) + iv(x,y)$ 在 $z_0 = x_0 + iy_0$ 可导。上一节的公式解析已经通过「沿实轴」「沿虚轴」两条路径分别计算差商极限，并让它们相等，得到

$$\boxed{\ u_x = v_y, \qquad u_y = -v_x\ }$$

这对方程就是**柯西-黎曼方程（Cauchy-Riemann equations）**，简称 C-R 方程，通常在点 $z_0$ 处成立。把它写开：

$$\frac{\partial u}{\partial x} = \frac{\partial v}{\partial y}, \qquad \frac{\partial u}{\partial y} = -\frac{\partial v}{\partial x}$$

**重点：柯西-黎曼方程是复函数可导的必要条件。** 若 $f$ 在 $z_0$ 可导，则 C-R 方程在 $z_0$ 成立。反过来说，C-R 方程成立并不保证可导——还必须额外要求 $u,v$ 在 $z_0$ 可微（即全微分存在），这留待下一节《函数可导与解析的充要条件》补全。今天先学会用它做**排除法**。

**辨析｜易错点：C-R 方程的两个符号别记反。** 许多学生把 $u_y = -v_x$ 错记成 $u_y = v_x$。一个可靠的记忆法：把两个方程交叉相乘，应有 $u_x u_y = -v_y v_x$，符号靠「虚部对 $x$ 的偏导前面带负号」来锚定。另一个验证法：对 $f(z)=z=x+iy$，$u_x=1,\ u_y=0,\ v_x=0,\ v_y=1$，满足 $u_x=v_y=1$ 且 $u_y=-v_x=0$——若记错符号，连 $z$ 的导数都会算错。

## 2 用 C-R 方程淘汰「假可导」的函数

C-R 方程的第一个用途是快速判断一个函数**不**可导。来看三个经典例子。

**例 1：$f(z) = \bar{z} = x - iy$。** 此时 $u = x$，$v = -y$，于是 $u_x = 1$，$v_y = -1$。第一条方程 $u_x = v_y$ 变成 $1 = -1$，不成立。故 $f(z)=\bar{z}$ **处处不可导**——尽管它在整个平面上连续。

**例 2：$f(z) = |z|^2 = x^2 + y^2$。** 此时 $u = x^2+y^2$，$v = 0$。于是 $u_x = 2x$，$v_y = 0$，第一条方程要求 $2x = 0$，即 $x=0$；又 $u_y = 2y$，$v_x = 0$，第二条方程要求 $2y = 0$，即 $y=0$。两条同时成立只在原点 $z=0$。所以 $f(z)=|z|^2$ **只在 $z=0$ 可能可导**（还需可微性），其余处处不可导。<span class="marginnote">这个例子很有启发性：$|z|^2$ 在实轴上等于 $x^2$，是实函数里最光滑的函数之一，但作为复函数它几乎处处不可导。可见「复可导」与「实可导」是两个完全不同的强度。</span>

**例 3：$f(z) = z^2 = (x^2-y^2) + i(2xy)$。** 此时 $u = x^2-y^2$，$v=2xy$。算得 $u_x = 2x$，$v_y = 2x$；$u_y = -2y$，$v_x = 2y$。两条方程处处成立，加上 $u,v$ 都是多项式（自动可微），于是 $f(z)=z^2$ **在全平面解析**，且 $f'(z)=2z$。C-R 方程在多项式身上总是「处处通过」。

这三个例子浓缩了 C-R 方程的判别价值：**它把「路径无关」这条看不见的要求，变成了一道可以机械执行的偏导数检查。**<span class="marginnote">练习手感：$f(z)=z^n$ 对任意正整数 $n$ 都满足 C-R 方程，全平面解析，$f'(z)=nz^{n-1}$——与实数情形的幂函数求导公式完全吻合。下一节我们会看到，只要 C-R 方程加可微性成立，求导公式就是 $f'(z)=u_x+iv_x$。</span>

## 3 公式解析：C-R 方程的几何含义

C-R 方程不只是代数检查，它有深刻的几何意义。设 $f=u+iv$ 可导且 $f'(z_0)\ne 0$。将 $f'$ 写成极坐标：

$$f'(z_0) = u_x + iv_x = r_0 e^{i\theta_0}$$

回忆第一章：乘以 $e^{i\theta_0}$ 是旋转 $\theta_0$ 角，乘以 $r_0$ 是缩放 $r_0$ 倍。于是「在 $z_0$ 附近，$f$ 把一小片平面近似地旋转 $\theta_0$ 并均匀缩放 $r_0$ 倍」——这是复可导函数最本质的局部行为。我们分两步看 C-R 方程如何保证这一点：

- **第一步，认识雅可比矩阵。** $f$ 作为 $\mathbb{R}^2 \to \mathbb{R}^2$ 的映射，其雅可比矩阵是
$$J_f = \begin{pmatrix} u_x & u_y \\ v_x & v_y \end{pmatrix}$$
它描述 $f$ 在 $z_0$ 附近如何把微小向量 $\begin{pmatrix} dx \\ dy \end{pmatrix}$ 映射到 $\begin{pmatrix} du \\ dv \end{pmatrix}$。

- **第二步，代入 C-R 方程。** 由 $u_y = -v_x$、$v_y = u_x$，雅可比矩阵化为
$$J_f = \begin{pmatrix} u_x & -v_x \\ v_x & u_x \end{pmatrix} = \begin{pmatrix} a & -b \\ b & a \end{pmatrix}, \qquad a = u_x,\ b = v_x$$
而 $\begin{pmatrix} a & -b \\ b & a \end{pmatrix}$ 正是「旋转 + 均匀缩放」的矩阵形式（旋转矩阵乘上缩放因子 $\sqrt{a^2+b^2}$）。

- **第三步，读出结论。** C-R 方程迫使雅可比矩阵具有「旋转-缩放」的特殊形状：**它没有各向异性的拉伸**。在实映射里，雅可比矩阵可以是任意 $2\times 2$ 矩阵——一个微小圆盘会被压成椭圆；而在复可导映射里，微小圆盘只能被旋转 + 均匀放大，**仍然保持为圆盘**。这正是第六章《共形映射》中「保角」性质的雏形。<span class="marginnote">这一点也预告了复分析与调和函数的关系：$u$ 与 $v$ 都被 C-R 方程牵制，各自都满足拉普拉斯方程 $\Delta u = 0$、$\Delta v = 0$（在二阶可导时），所以解析函数的实部虚部都是<strong>调和函数</strong>——这是第二章末《调和函数》的主题。</span>

**一句话直觉：解析函数在局部只能「转」与「缩放」，不能「拉成椭圆」。** 这就是为什么复可导如此苛刻，也是为什么它如此美丽。

## 4 极坐标下的柯西-黎曼方程

许多问题在极坐标下更自然（比如求 $z^n$、$\log z$ 的可导性）。若 $z = re^{i\theta}$，$f = u(r,\theta) + iv(r,\theta)$，则 C-R 方程化为

$$u_r = \frac{1}{r}\, v_\theta, \qquad v_r = -\frac{1}{r}\, u_\theta \qquad (r \ne 0)$$

作为检验，看 $f(z) = \frac{1}{z} = \frac{1}{r} e^{-i\theta}$。实部 $u = \frac{1}{r}\cos\theta$，虚部 $v = -\frac{1}{r}\sin\theta$。算 $u_r = -\frac{1}{r^2}\cos\theta$，$\frac{1}{r}v_\theta = \frac{1}{r}(-\frac{1}{r}\cos\theta) = -\frac{1}{r^2}\cos\theta$，一致；再算 $v_r = \frac{1}{r^2}\sin\theta$，$-\frac{1}{r}u_\theta = -\frac{1}{r}(-\frac{1}{r}\sin\theta) = \frac{1}{r^2}\sin\theta$，也一致。故 $\frac{1}{z}$ 在 $z\ne 0$ 处解析，且由 $f'(z)=u_r+iv_r$ 的方向导公式可推出 $f'(z)=-\frac{1}{z^2}$。

极坐标形式的推导只需对 $u_r, u_\theta$ 用链式法则展开 $u_x, u_y$ 再代回直角 C-R 方程即可，值得在练习本上推一遍。<span class="marginnote">极坐标 C-R 方程的一个著名推论：$f(z)=z^\alpha$（$\alpha$ 为实数）在 $z\ne 0$ 处是否解析，取决于 $\alpha$ 取什么值。到第二章讲幂函数时，我们会用这套方程逐一验明。</span>

## 5 补充：C-R 方程的极坐标记忆与物理图景

极坐标形式的 C-R 方程容易记混，这里给一个可靠的推导线索与物理对照。

**推导线索（自己推一遍胜过背）：** 直角坐标 C-R $u_x=v_y,\ u_y=-v_x$，极坐标里 $x=r\cos\theta,\ y=r\sin\theta$。由链式法则 $u_r=u_x\cos\theta+u_y\sin\theta$，$v_\theta=v_x(-r\sin\theta)+v_y(r\cos\theta)$。代入直角 C-R 得 $u_r=\frac1r v_\theta$；同理 $v_r=-\frac1r u_\theta$。**两对方程同一回事，只是坐标系不同。**

**物理图景：解析函数 = 无旋无源场。** 设 $f=u+iv$ 是某平面流动的复势，$u$ 是速度势、$v$ 是流函数。C-R 方程 $u_x=v_y$、$u_y=-v_x$ 翻译成流体语言：

- $u_x=v_y$：速度场的「散度为零」条件——流体不可压缩；
- $u_y=-v_x$：速度场的「旋度为零」条件——流动无旋。

**两条 C-R 方程 = 不可压 + 无旋 = 理想流体的全部假设。** 这就是为什么复势方法能统治空气动力学：**解析函数天然对应「物理上允许的流动」。**

**例（极坐标判别）：** 判定 $f(z)=\sqrt{r}e^{i\theta/2}$（$r>0$，主分支）是否解析。$u=\sqrt r\cos\frac\theta2$，$v=\sqrt r\sin\frac\theta2$。$u_r=\frac1{2\sqrt r}\cos\frac\theta2$，$\frac1r v_\theta=\frac1r\cdot\sqrt r\cdot\frac12\cos\frac\theta2=\frac1{2\sqrt r}\cos\frac\theta2$ ✓；$v_r=\frac1{2\sqrt r}\sin\frac\theta2$，$-\frac1r u_\theta=-\frac1r\cdot\sqrt r\cdot(-\frac12\sin\frac\theta2)=\frac1{2\sqrt r}\sin\frac\theta2$ ✓。C-R 在 $r>0$ 成立，$f=\sqrt z$ 在割开平面上解析——**极坐标形式让根式函数的验证直接可做。**

**辨析｜易错点：极坐标 C-R 在 $r=0$ 处失效。** 公式含 $\frac1r$ 因子，$r=0$ 时无意义——原点要单独处理。**$\sqrt z$ 在 $z=0$ 是支点（第二章），C-R 判据本就不覆盖支点。**

## 6 补充：C-R 方程的等价表达

C-R 方程 $u_x=v_y,\ u_y=-v_x$ 还有两种等价写法，不同场合各有便利。

**等价形式一：$\frac{\partial f}{\partial \bar z}=0$。** 把复函数看成两个独立变量 $z$ 与 $\bar z$ 的函数，形式偏导：

$$\frac{\partial f}{\partial \bar z}=\frac12\left(\frac{\partial f}{\partial x}+i\frac{\partial f}{\partial y}\right)$$

若 $\frac{\partial f}{\partial\bar z}=0$，则 $u_x+iv_x+i(u_y+iv_y)=0$，拆实虚即得 $u_x=v_y,\ u_y=-v_x$——**正是 C-R 方程**。**「$f$ 解析 ⟺ $f$ 不显含 $\bar z$」**——这是「解析函数只依赖 $z$」的严格化。

**例（用 $\frac{\partial}{\partial\bar z}$ 判定）：** $f(z)=\bar z$，$\frac{\partial f}{\partial\bar z}=1\ne0$，处处不可导 ✓。$f(z)=z^2$，$\frac{\partial f}{\partial\bar z}=0$，解析 ✓。

**等价形式二：$f'(z)$ 的三个公式一致。** 可导时 $f'=u_x+iv_x=v_y-iu_y$——两个方向算的导数相等，这正是 C-R 的内容。**「导数存在且唯一」与「C-R 成立」同义。**

**重点：$\frac{\partial}{\partial\bar z}$ 记号在复几何与多复变里是标准语言。** 现在记住「解析 ⟺ $\bar z$ 导数消失」，日后读更深的书时无缝衔接。

**综合例：** 求使 $f(z)=az^2+b\bar z+c$ 解析的条件。$\frac{\partial f}{\partial\bar z}=b$，故 $f$ 解析 ⟺ $b=0$——**含 $\bar z$ 的项系数必须为零**。验证：$b=0$ 时 $f=az^2+c$ 是多项式，全平面解析 ✓。

**辨析｜易错点：$\frac{\partial f}{\partial\bar z}=0$ 只在「形式导数」意义下理解。** $z$ 与 $\bar z$ 不是独立实变量，这个「偏导」是形式记号，不是通常偏导。**它用于「快捷判定」，不用于「求值」。**

## 7 小结

- **柯西-黎曼方程**：$u_x = v_y$，$u_y = -v_x$；极坐标形式为 $u_r = \frac{1}{r}v_\theta$，$v_r = -\frac{1}{r}u_\theta$。
- **必要不充分**：可导 ⟹ C-R 成立；C-R 成立 + $u,v$ 可微 ⟹ 可导（下一节补充分性）。
- **快速排除**：$f(z)=\bar{z}$ 处处不可导、$f(z)=|z|^2$ 仅在原点可能可导，都可用 C-R 方程一眼看穿。
- **几何含义**：C-R 方程迫使雅可比矩阵成为「旋转-缩放」形，局部微小圆盘保持为圆盘——这就是共形映射的雏形。
- **求导捷径**：可导时 $f'(z) = u_x + iv_x$（或极坐标下的 $e^{-i\theta}(u_r + iv_r)$）。

在下一节，我们将补齐「C-R 方程 + 可微性 ⟹ 可导」这条正向方向，给出**函数可导与解析的充要条件**，并用它系统验证指数、对数、三角等初等函数的解析性。
