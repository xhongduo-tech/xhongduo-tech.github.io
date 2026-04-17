## 核心结论

矩阵函数的任务，是把原本定义在标量上的函数 $f(x)$，推广为定义在方阵上的 $f(A)$。目标不是“发明一套新记号”，而是让矩阵也能继承标量函数的代数和几何意义。

最关键的结论有两条。

1. 如果矩阵 $A$ 可以对角化，即存在可逆矩阵 $V$ 使得
   $$
   A = V \operatorname{diag}(\lambda_1,\dots,\lambda_n)V^{-1},
   $$
   那么大量常见矩阵函数都满足
   $$
   f(A)=V\operatorname{diag}(f(\lambda_1),\dots,f(\lambda_n))V^{-1}.
   $$
   这句话的含义是：先把矩阵变到特征坐标系，再把函数逐个作用到特征值上，最后再变回原坐标。

2. 矩阵指数
   $$
   e^A=\sum_{k=0}^{\infty}\frac{A^k}{k!}
   $$
   是最常用的矩阵函数。它同时承担两件事：
   一是在微分方程
   $$
   \dot x = Ax
   $$
   中给出状态随时间的传播规律；
   二是在 Lie 理论里，把“无穷小生成元”映射成“有限变换”。

对新手最直接的例子是对角矩阵。若
$$
A=\operatorname{diag}(1,2),
$$
则
$$
e^A=\operatorname{diag}(e,e^2).
$$
这里没有把矩阵“压缩”为一个数，也没有把每个元素单独取指数，而是保留了矩阵结构，只对特征值层面的信息做指数变换。

先看最核心的对应关系：

| 标量函数 | 矩阵版本 | 可对角化时的形式 | 常见用途 |
|---|---|---|---|
| $f(x)$ | $f(A)$ | $V\operatorname{diag}(f(\lambda_i))V^{-1}$ | 统一描述对线性变换施加函数 |
| $e^x$ | $e^A$ | $V\operatorname{diag}(e^{\lambda_i})V^{-1}$ | 线性系统传播、旋转、稳定性分析 |
| $\log x$ | $\log A$ | $V\operatorname{diag}(\log \lambda_i)V^{-1}$ | 从有限变换恢复生成元、模型离散化/连续化 |

再补一个容易混淆的点：

| 对象 | 运算方式 | 结果 |
|---|---|---|
| 标量 $x$ | $e^x$ | 一个数 |
| 向量 $v$ | 一般不直接写 $e^v$ | 通常无统一线性代数定义 |
| 矩阵 $A$ | $e^A=\sum_{k\ge 0}A^k/k!$ | 一个同维度方阵 |

---

## 问题定义与边界

问题可以精确写成：

给定一个方阵 $A\in\mathbb{C}^{n\times n}$ 和一个标量函数 $f$，怎样定义 $f(A)$，使它既符合数学结构，又适合数值计算？

这里有两个要求：

1. 数学上要一致  
   至少要和标量情况兼容，比如对角矩阵时应退化为“逐个对角元素做函数”。

2. 数值上要可实现  
   不能只给出形式定义，还要能在计算机里稳定求值。

常见定义路径有三类。

### 1. 幂级数定义

如果函数在某个区域里有幂级数展开
$$
f(x)=\sum_{k=0}^\infty c_k x^k,
$$
那么可以定义
$$
f(A)=\sum_{k=0}^\infty c_k A^k.
$$
这是最自然的一条路，因为矩阵乘法本来就定义良好。

例如指数函数满足
$$
e^x=\sum_{k=0}^{\infty}\frac{x^k}{k!},
$$
所以自然得到
$$
e^A=\sum_{k=0}^{\infty}\frac{A^k}{k!}.
$$

如果你刚接触这一点，可以把它理解为：标量幂级数里的 $x^k$ 被整体替换成矩阵幂 $A^k$。

### 2. 特征值分解定义

若矩阵可以对角化，
$$
A=V\Lambda V^{-1},\qquad \Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n),
$$
则直接定义
$$
f(A)=V\operatorname{diag}(f(\lambda_1),\dots,f(\lambda_n))V^{-1}.
$$

这条定义最直观，因为对角矩阵的函数最容易理解。它也是很多推导的起点。

### 3. Jordan 形或 Schur 分解定义

如果矩阵不能对角化，特征向量不够，就不能直接套上面的公式。这时仍然可以定义矩阵函数，但需要更一般的结构工具。

- Jordan 形适合理论推导
- Schur 分解适合数值计算

两者都在处理同一个事实：即使矩阵没有足够多的特征向量，函数 $f(A)$ 仍然可以定义，只是公式不再只是“对特征值逐点作用”那么简单。

先把可用边界列清楚：

| 输入矩阵条件 | 可用方法 | 说明 |
|---|---|---|
| 可对角化 | 特征值分解 | 最直观，适合教学和解析推导 |
| 不可对角化 | Jordan 形、Schur 分解 | 仍可定义，但实现更复杂 |
| 一般方阵 | 幂级数法（若级数在谱附近收敛） | 指数函数是最典型的全局可用情形 |
| 可逆矩阵 | 可能存在矩阵对数 | 不可逆矩阵一定不是任何矩阵指数 |
| 谱避开负实轴且满足主值条件 | 主对数存在且唯一 | 这是数值库最常采用的分支 |

“谱”就是矩阵特征值的集合，记作
$$
\sigma(A)=\{\lambda:\det(\lambda I-A)=0\}.
$$
在矩阵函数里，谱的重要性很高，因为标量函数 $f$ 最终是作用在这些特征值附近的。

矩阵对数 $\log B$ 是矩阵指数的逆问题，即寻找矩阵 $X$ 使得
$$
e^X=B.
$$
它和标量对数相比更复杂，原因在于矩阵指数并不是一一对应。

矩阵对数有三个关键边界：

1. $B$ 必须可逆  
   因为对任意矩阵 $X$ 都有
   $$
   \det(e^X)=e^{\operatorname{tr}(X)}\neq 0.
   $$
   所以矩阵指数永远可逆。

2. 矩阵对数通常不唯一  
   标量复对数本来就是多值的：
   $$
   \log z = \ln|z| + i(\arg z + 2k\pi).
   $$
   矩阵对数会继承这种多值性。

3. 主对数只在特定谱条件下定义良好  
   常见要求是谱不碰到非正实轴，或者更准确地说，不穿过主支切口。

一个局部近似例子是
$$
B=I+\varepsilon M,
$$
当 $\varepsilon$ 很小时，若
$$
\|B-I\|<1,
$$
则可用级数
$$
\log B=\sum_{k=1}^{\infty}\frac{(-1)^{k+1}}{k}(B-I)^k.
$$
这和标量公式
$$
\log(1+x)=x-\frac{x^2}{2}+\frac{x^3}{3}-\cdots
$$
是完全平行的。

这里的“范数” $\|\cdot\|$ 可以先粗略理解为矩阵大小的度量。常见选择包括谱范数、Frobenius 范数等。对新手来说，此处只需要记住：$\|B-I\|<1$ 是保证级数收敛的一个充分条件。

---

## 核心机制与推导

矩阵函数成立的根本原因，是相似变换不改变线性变换的本质，只改变坐标表示。

如果
$$
A=V\Lambda V^{-1},\qquad \Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n),
$$
那么
$$
A^k=(V\Lambda V^{-1})^k=V\Lambda^kV^{-1}.
$$
这一步很关键。因为每个中间的 $V^{-1}V$ 都会抵消，只剩下首尾两个变换矩阵。

于是对幂级数函数
$$
f(x)=\sum_{k=0}^{\infty}c_kx^k,
$$
有
$$
f(A)=\sum_{k=0}^{\infty}c_kA^k
=\sum_{k=0}^{\infty}c_kV\Lambda^kV^{-1}
=V\left(\sum_{k=0}^{\infty}c_k\Lambda^k\right)V^{-1}.
$$
而对角矩阵的幂非常简单：
$$
\Lambda^k=\operatorname{diag}(\lambda_1^k,\dots,\lambda_n^k),
$$
所以
$$
\sum_{k=0}^{\infty}c_k\Lambda^k
=
\operatorname{diag}\!\left(
\sum_{k=0}^{\infty}c_k\lambda_1^k,\dots,
\sum_{k=0}^{\infty}c_k\lambda_n^k
\right)
=
\operatorname{diag}(f(\lambda_1),\dots,f(\lambda_n)).
$$
最终得到
$$
f(A)=V\operatorname{diag}(f(\lambda_1),\dots,f(\lambda_n))V^{-1}.
$$

这条公式可以拆成三步理解：

1. 进入特征坐标系
2. 对每个特征值做标量函数
3. 返回原坐标系

这就是“矩阵函数保留特征值意义”的具体来源。

### 1. 矩阵指数的幂级数机制

矩阵指数定义为
$$
e^A=I+A+\frac{A^2}{2!}+\frac{A^3}{3!}+\cdots
$$
由于指数函数的幂级数在整个复平面都收敛，所以对任意方阵 $A$，$e^A$ 都存在。

如果 $A$ 可对角化，则
$$
e^A=V\operatorname{diag}(e^{\lambda_1},\dots,e^{\lambda_n})V^{-1}.
$$

这条式子可以直接解释动力学行为：

- 若 $\operatorname{Re}(\lambda_i)>0$，对应方向会指数增长
- 若 $\operatorname{Re}(\lambda_i)<0$，对应方向会指数衰减
- 若 $\operatorname{Im}(\lambda_i)\neq 0$，对应方向会伴随振荡或旋转

例如
$$
\lambda = a+ib
$$
对应的标量指数是
$$
e^{\lambda t}=e^{at}e^{ibt}=e^{at}(\cos bt+i\sin bt),
$$
其中 $a$ 控制放大或衰减，$b$ 控制角频率。这也是为什么在线性系统里，特征值的实部和虚部分别对应稳定性和振荡频率。

再看一个最简单的上三角矩阵：
$$
A=
\begin{bmatrix}
\lambda & 1\\
0 & \lambda
\end{bmatrix}
=
\lambda I + N,\qquad
N=
\begin{bmatrix}
0 & 1\\
0 & 0
\end{bmatrix},
\quad N^2=0.
$$
因为 $\lambda I$ 与 $N$ 可交换，
$$
e^A=e^{\lambda I+N}=e^\lambda e^N.
$$
而
$$
e^N=I+N=
\begin{bmatrix}
1 & 1\\
0 & 1
\end{bmatrix},
$$
故
$$
e^A=e^\lambda
\begin{bmatrix}
1 & 1\\
0 & 1
\end{bmatrix}.
$$
这个例子说明：不可对角化时，矩阵函数除了作用在特征值上，还会受到 Jordan 块结构影响。

### 2. 旋转生成元的玩具例子

考虑
$$
G=\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix}.
$$
这个矩阵表示平面中的无穷小旋转生成元。它本身不是“旋转了一个有限角度”，而是描述“瞬时旋转方向”的线性规则。

先计算它的幂：
$$
G^2=-I,\qquad G^3=-G,\qquad G^4=I.
$$
于是指数展开
$$
e^{\theta G}
=I+\theta G+\frac{\theta^2G^2}{2!}+\frac{\theta^3G^3}{3!}+\cdots
$$
可以按偶数项与奇数项分组：
$$
e^{\theta G}
=
\left(I-\frac{\theta^2}{2!}I+\frac{\theta^4}{4!}I-\cdots\right)
+
\left(\theta G-\frac{\theta^3}{3!}G+\frac{\theta^5}{5!}G-\cdots\right).
$$
第一部分是 $\cos\theta\,I$，第二部分是 $\sin\theta\,G$，所以
$$
e^{\theta G}=\cos\theta\, I+\sin\theta\, G
=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}.
$$

这一步非常重要，因为它把三个概念连起来了：

| 对象 | 角色 |
|---|---|
| $G$ | 无穷小旋转生成元 |
| $e^{\theta G}$ | 有限角度旋转 |
| $\sin,\cos$ | 指数级数按奇偶项拆分后的自然结果 |

当
$$
\theta=\frac{\pi}{2},
$$
有
$$
e^{\theta G}
=
\begin{bmatrix}
0 & -1\\
1 & 0
\end{bmatrix},
$$
表示逆时针旋转 $90^\circ$。

如果作用在向量
$$
x=\begin{bmatrix}1\\0\end{bmatrix}
$$
上，则
$$
e^{\frac{\pi}{2}G}x=
\begin{bmatrix}0\\1\end{bmatrix}.
$$
这比只看矩阵公式更直观：矩阵指数在这里确实产生了一个有限旋转。

### 3. 矩阵对数的逆向机制

矩阵对数的目标是寻找 $X$ 使
$$
e^X=B.
$$
如果 $B$ 足够接近单位矩阵，就可以用局部幂级数
$$
\log B=\sum_{k=1}^{\infty}\frac{(-1)^{k+1}}{k}(B-I)^k,
\qquad \|B-I\|<1.
$$

这一定义好理解，但它只适用于“离 $I$ 不太远”的局部区域。远离单位矩阵时，不能直接依赖这条级数。

若 $B$ 可对角化，形式上可以写成
$$
\log B=V\operatorname{diag}(\log\lambda_1,\dots,\log\lambda_n)V^{-1}.
$$
但这里立刻会遇到一个新问题：复对数有分支。

对复数
$$
\lambda = re^{i\theta},
$$
它的对数不是唯一的，而是
$$
\log \lambda = \ln r + i(\theta + 2k\pi),\qquad k\in\mathbb Z.
$$
因此矩阵对数通常也不唯一。

“主对数”的意思，是固定采用主值分支
$$
\operatorname{Arg}(\lambda)\in(-\pi,\pi),
$$
从而得到一个优先选取的对数值。对应到矩阵上，就要求谱不要落到主分支切口上，通常表述为：特征值不在非正实轴上。

一个简单标量类比是：

| 标量情形 | 矩阵情形 |
|---|---|
| $e^x>0$ 时有实对数 | 矩阵可逆时才可能有矩阵对数 |
| $\log z$ 多值 | $\log B$ 也可能多值 |
| 主值切口通常取负实轴 | 主矩阵对数也依赖谱避开这条切口 |

再给一个可直接验证的例子。设
$$
B=
\begin{bmatrix}
e & 0\\
0 & e^2
\end{bmatrix}.
$$
则一个自然的矩阵对数是
$$
\log B=
\begin{bmatrix}
1 & 0\\
0 & 2
\end{bmatrix}.
$$
因为
$$
e^{\log B}=B.
$$
这和对角矩阵的“逐点作用”完全一致。

### 4. 真实工程例子：状态迁移矩阵

连续时间线性系统
$$
\dot x(t)=Ax(t)
$$
的解为
$$
x(t)=e^{At}x(0).
$$
这里
$$
\Phi(t)=e^{At}
$$
称为状态迁移矩阵。

这个名字的物理含义很直接：它把初始状态 $x(0)$ 映射到时刻 $t$ 的状态 $x(t)$。

为什么它一定满足微分方程？因为
$$
\frac{d}{dt}e^{At}
=
A e^{At}
=
e^{At}A,
$$
所以
$$
\frac{d}{dt}\bigl(e^{At}x(0)\bigr)=Ae^{At}x(0)=Ax(t).
$$

再看一个具体系统：
$$
A=\begin{bmatrix}
0 & 1\\
-2 & -3
\end{bmatrix}.
$$
这类矩阵常见于二阶弹簧-阻尼系统的一阶化表示。若定义状态
$$
x(t)=\begin{bmatrix}
q(t)\\
\dot q(t)
\end{bmatrix},
$$
则二阶方程
$$
\ddot q + 3\dot q + 2q = 0
$$
可以改写成
$$
\dot x = Ax.
$$

工程上，$e^{At}$ 的用途包括：

| 场景 | 用法 |
|---|---|
| 数值仿真 | 直接推进状态 $x(t+\Delta t)=e^{A\Delta t}x(t)$ |
| 离散化建模 | 从连续模型得到离散状态矩阵 |
| 卡尔曼滤波 | 构造状态传播矩阵 |
| 控制设计 | 分析模态衰减、闭环稳定性 |
| 振动系统 | 从特征值读出阻尼和振荡频率 |

如果 $A$ 的特征值为 $-1$ 和 $-2$，那么系统各模态都衰减，因此系统是稳定的。这就是“特征值信息通过指数映射变成时间传播”的一个标准例子。

---

## 代码实现

下面给出一个完整可运行的 Python 示例，覆盖三个目标：

1. 用幂级数近似计算旋转生成元的矩阵指数
2. 用 `scipy.linalg.expm` 和 `logm` 计算指数与对数
3. 用断言验证公式和数值结果

运行环境：

```bash
python -m pip install numpy scipy
```

示例代码：

```python
import numpy as np
from math import factorial, pi
from scipy.linalg import expm, logm, norm


def exp_series(A, terms=40):
    """使用幂级数近似 exp(A)。适合教学，不适合大规模工程。"""
    n = A.shape[0]
    result = np.eye(n, dtype=float)
    power = np.eye(n, dtype=float)

    for k in range(1, terms):
        power = power @ A
        result = result + power / factorial(k)

    return result


def log_series_near_I(B, terms=40):
    """当 ||B - I|| < 1 时，用局部级数近似 log(B)。"""
    n = B.shape[0]
    I = np.eye(n, dtype=float)
    X = B - I

    result = np.zeros_like(B, dtype=float)
    power = X.copy()

    for k in range(1, terms + 1):
        result = result + ((-1) ** (k + 1)) * power / k
        power = power @ X

    return result


def main():
    # 1) 旋转生成元示例
    G = np.array([
        [0.0, -1.0],
        [1.0,  0.0],
    ])

    theta = pi / 2
    A = theta * G

    E_series = exp_series(A, terms=40)
    E_lib = expm(A)

    expected_rotation = np.array([
        [0.0, -1.0],
        [1.0,  0.0],
    ])

    print("series exp(theta G) =\n", E_series)
    print("scipy  exp(theta G) =\n", E_lib)

    assert np.allclose(E_series, expected_rotation, atol=1e-10)
    assert np.allclose(E_lib, expected_rotation, atol=1e-10)

    # 2) 对角矩阵示例
    D = np.diag([1.0, 2.0])
    ED = expm(D)
    expected_diag = np.diag([np.e, np.e ** 2])

    print("exp(diag(1,2)) =\n", ED)
    assert np.allclose(ED, expected_diag, atol=1e-12)

    # 3) 主值范围内的对数恢复
    B = expm(0.3 * G)
    L = logm(B)

    print("log(exp(0.3 G)) =\n", L)
    assert norm(L - 0.3 * G) < 1e-10

    # 4) 接近单位矩阵时，用级数近似矩阵对数
    M = np.array([
        [0.2, 0.1],
        [0.0, -0.1],
    ])
    eps = 0.1
    B_near_I = np.eye(2) + eps * M

    L_series = log_series_near_I(B_near_I, terms=50)
    L_lib = logm(B_near_I)

    print("series log(B_near_I) =\n", L_series)
    print("scipy  log(B_near_I) =\n", L_lib)

    assert norm(L_series - L_lib) < 1e-8

    # 5) 状态迁移矩阵示例
    A_sys = np.array([
        [0.0, 1.0],
        [-2.0, -3.0],
    ])
    dt = 0.2
    F = expm(A_sys * dt)

    x0 = np.array([1.0, 0.0])
    x1 = F @ x0

    print("state transition matrix F = exp(A_sys * dt) =\n", F)
    print("x(dt) = F x0 =\n", x1)

    # 回代验证：若 L = logm(F)，则 expm(L) 应恢复 F
    A_recovered = logm(F) / dt
    assert norm(expm(A_recovered * dt) - F) < 1e-10


if __name__ == "__main__":
    main()
```

这段代码里有几个点值得单独说明。

### 1. `exp_series` 为什么能工作

因为它直接按定义累加
$$
e^A=\sum_{k=0}^{\infty}\frac{A^k}{k!}.
$$
对小矩阵和教学示例，这很直观。但它在工程里通常不是首选，因为：

- 收敛虽然成立，但不一定高效
- 高阶幂可能带来舍入误差
- 对大矩阵成本很高

### 2. `log_series_near_I` 为什么只适合局部

它实现的是
$$
\log(I+X)=\sum_{k=1}^{\infty}\frac{(-1)^{k+1}}{k}X^k,
$$
因此要保证
$$
\|X\|<1.
$$
若矩阵离单位阵较远，直接使用这条级数可能收敛很慢，甚至不收敛。

### 3. 为什么工程里更常用 `expm`

`scipy.linalg.expm` 的内部算法不是直接傻算幂级数，而通常基于更稳定的数值路线，例如：

- 缩放与平方
- Padé 逼近
- 与 Schur 分解相关的稳定实现

所以教学代码和生产代码的定位要分开：

| 方法 | 适合场景 |
|---|---|
| 手写幂级数 | 理解定义、验证小例子 |
| `scipy.linalg.expm` | 一般工程计算 |
| `scipy.sparse.linalg.expm_multiply` | 大型稀疏矩阵，只关心 $e^Av$ |

真实工程的典型流程通常是：

1. 建立连续系统矩阵 $A$
2. 计算
   $$
   F=e^{A\Delta t}
   $$
   得到一步离散传播矩阵
3. 若要反推连续生成元，再尝试
   $$
   A=\frac{1}{\Delta t}\log F
   $$
4. 最后用
   $$
   e^{(\log F)}\approx F
   $$
   做回代验证

---

## 工程权衡与常见坑

矩阵函数的数学可定义，不等于工程上总是安全可算。尤其是矩阵对数，很多问题都出在“形式上能写，数值上不稳”。

最常见的坑如下：

| 问题情形 | 风险 | 应对措施 |
|---|---|---|
| 矩阵不可逆 | 对数不存在 | 先检查行列式、秩或最小奇异值 |
| 存在负实特征值 | 主对数可能不存在 | 先做谱分析，确认是否接受非主值 |
| 特征值靠近分支切口 | 数值结果对扰动敏感 | 尽量采用 Schur 路线，避免直接特征分解 |
| 矩阵不可对角化 | 直接套 $V\diag(f(\lambda))V^{-1}$ 会失效 | 改用 Jordan/Schur 方法 |
| 大规模稀疏矩阵 | 显式求 $e^A$ 成本过高 | 改算 $e^Av$，使用 Krylov 方法 |
| 非正规矩阵 | 特征值看起来安全，但瞬态可能很大 | 结合条件数、Schur 形和残差一起判断 |

这里补充一个新手容易忽略的事实：矩阵元素逐个做函数，通常不是矩阵函数。

例如
$$
A=
\begin{bmatrix}
0 & 1\\
0 & 0
\end{bmatrix}.
$$
若逐元素取指数，会得到
$$
\begin{bmatrix}
e^0 & e^1\\
e^0 & e^0
\end{bmatrix}
=
\begin{bmatrix}
1 & e\\
1 & 1
\end{bmatrix},
$$
但真正的矩阵指数是
$$
e^A=I+A=
\begin{bmatrix}
1 & 1\\
0 & 1
\end{bmatrix},
$$
因为这里 $A^2=0$。两者完全不同。

再看矩阵对数里的一个高频误区。很多人看到离散系统矩阵 $F$ 后，会直接写
$$
A=\frac{1}{\Delta t}\log F.
$$
这个公式在形式上对，但有几个隐藏前提：

1. $F$ 必须可逆
2. 需要明确选取哪个对数分支
3. 采样频率必须足够高，否则会出现频率别名
4. 结果是否符合物理模型，还要额外验证

例如若离散特征值位于复平面负实轴附近，那么主对数可能不存在，或者数值结果非常敏感。这不是软件错误，而是问题本身就处在分支边界附近。

在实际软件中，`logm` 返回“某个矩阵对数”，不保证一定是你想要的那个物理生成元。一个常见现象是：

- 输入矩阵是实矩阵
- 输出 `logm` 却出现复数项

这通常说明谱跨越了对数分支切口，或者非常接近切口。

因此更稳妥的检查顺序是：

1. 先看矩阵是否可逆
2. 再看特征值是否落在负实轴或靠近它
3. 明确需要的是主对数，还是任意一个对数
4. 调用库函数后做回代验证：
   $$
   e^{\logm(F)}\approx F
   $$
5. 若场景有物理含义，再检查恢复出的生成元是否满足实系数、稳定性、频率范围等先验约束

---

## 替代方案与适用边界

并不是每个问题都应该直接显式构造 $f(A)$。在大矩阵、稀疏矩阵或非正规矩阵场景下，直接求完整矩阵函数往往代价过高，甚至不是正确目标。

常见替代路线如下：

| 替代方法 | 适用矩阵特性 | 核心思路 |
|---|---|---|
| Jordan 形 | 理论推导、小规模教材例子 | 显式处理不可对角化结构 |
| Schur 分解 + 递推 | 一般稠密矩阵 | 数值稳定性通常优于直接特征分解 |
| 缩放与平方 | 一般矩阵指数 | 先把矩阵缩小，再重复平方恢复 |
| Padé 逼近 | 稠密矩阵指数 | 用有理函数近似指数函数 |
| Krylov 子空间 | 大型稀疏矩阵 | 直接近似 $e^Av$，不显式形成 $e^A$ |
| 多项式近似 | 重复作用同一函数时 | 用 Chebyshev 等基做快速逼近 |

先说结论性的选择原则：

1. 小规模、教学推导  
   优先特征值分解、幂级数、Jordan 形

2. 一般稠密矩阵  
   优先 Schur 分解、Padé、缩放与平方

3. 大规模稀疏矩阵  
   优先计算 $f(A)v$，避免显式构造 $f(A)$

4. 涉及矩阵对数  
   先验证谱条件，再谈算法实现

这里最值得单独说明的是 Krylov 方法。它的目标通常不是求整个
$$
e^A,
$$
而是求
$$
e^Av.
$$
这在工程里经常已经足够，因为很多仿真、传播、扩散问题真正关心的是“作用后的结果向量”，而不是整张矩阵本身。

Krylov 方法的基本想法是：在由
$$
\{v,Av,A^2v,\dots\}
$$
张成的低维子空间里逼近矩阵函数的作用。这样就把原本高维的大问题压缩成一个小得多的问题。

一个标准流程是：

1. 从向量 $v$ 出发生成 Krylov 子空间
   $$
   \mathcal K_m(A,v)=\operatorname{span}\{v,Av,\dots,A^{m-1}v\}
   $$
2. 在这个子空间里构造小矩阵 $H_m$
3. 计算小矩阵的指数 $e^{H_m}$
4. 再映射回原空间，得到 $e^Av$ 的近似

它的优势在于：

- 不需要显式形成 $e^A$
- 能利用稀疏矩阵乘向量的结构
- 适合 $10^5\times 10^5$ 这类高维问题

真实例子包括：

| 应用 | 典型矩阵来源 |
|---|---|
| 电路网络 | 稀疏耦合系统 |
| PDE 离散化 | 有限差分/有限元得到的大稀疏矩阵 |
| 图扩散模型 | 图 Laplacian 或邻接矩阵 |
| 马尔可夫过程 | 生成矩阵或转移矩阵 |

直观上可以这样理解：

- 显式求 `expm(A)`：先把整张地图完整画出来
- 只求 `expm(A) @ v`：只计算当前路径附近真正要经过的部分

因此在大规模问题里，真正可行的不是“求全矩阵”，而是“求矩阵函数对向量的作用”。

---

## 参考资料

- Nicholas J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008.  
  链接：https://epubs.siam.org/doi/book/10.1137/1.9780898717778  
  用途：矩阵函数的标准参考书，系统覆盖定义、Schur 方法、Padé 逼近、矩阵对数与数值稳定性。

- Nicholas J. Higham, “What Is a Matrix Function?”, *SIAM News*.  
  链接：https://www.siam.org/publications/siam-news/articles/what-is-a-matrix-function/  
  用途：用较短篇幅解释矩阵函数的核心概念，适合从标量函数过渡到矩阵函数。

- Moler, C. and Van Loan, C., “Nineteen Dubious Ways to Compute the Exponential of a Matrix, Twenty-Five Years Later”, *SIAM Review*, 2003.  
  链接：https://doi.org/10.1137/S00361445024180  
  用途：解释为什么矩阵指数的工程计算不能只靠朴素幂级数，也说明了常见数值算法的优缺点。

- SciPy Documentation, `scipy.linalg.expm`  
  链接：https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html  
  用途：查看 Python 中矩阵指数的实际接口与数值实现说明。

- SciPy Documentation, `scipy.linalg.logm`  
  链接：https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.logm.html  
  用途：查看矩阵对数的接口、返回行为和精度说明。

- SciPy Documentation, `scipy.sparse.linalg.expm_multiply`  
  链接：https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html  
  用途：说明大规模稀疏矩阵场景下为什么更应计算 $e^Av$ 而不是显式构造 $e^A$。

- Wikipedia, “Matrix exponential”  
  链接：https://en.wikipedia.org/wiki/Matrix_exponential  
  用途：查对矩阵指数的基本定义、性质以及旋转矩阵示例。

- Wikipedia, “Logarithm of a matrix”  
  链接：https://en.wikipedia.org/wiki/Logarithm_of_a_matrix  
  用途：查对矩阵对数的存在性、非唯一性、主对数与分支切口条件。

- Wikipedia, “State-transition matrix”  
  链接：https://en.wikipedia.org/wiki/State-transition_matrix  
  用途：查对连续时间线性系统解与状态迁移矩阵的标准写法。

- MathWorks, `logm` documentation  
  链接：https://www.mathworks.com/help/matlab/ref/logm.html  
  用途：对照工程软件在矩阵对数分支问题上的警告与行为描述。
