## 核心结论

矩阵运算的数值稳定性，指算法在浮点数环境中不会把舍入误差过度放大。更实用的目标是**后向稳定**：计算出来的结果，可以解释为对原始输入做了一个很小扰动之后的精确解。

用公式表示，如果算法对输入 $A,b$ 算出 $\hat{x}$，并且存在很小的 $\Delta A,\Delta b$，使得

$$(A+\Delta A)\hat{x}=b+\Delta b$$

同时 $\|\Delta A\|/\|A\|$ 和 $\|\Delta b\|/\|b\|$ 都很小，就说这个求解过程具有后向稳定性。它不要求每一步都没有误差，而是要求最终误差能被解释成“输入轻微变动”。

矩阵计算里要优先记住四条结论：

| 场景 | 不推荐写法 | 推荐写法 | 原因 |
|---|---|---|---|
| 解 $Ax=b$ | `inv(A) @ b` | LU 分解后回代 | 避免显式求逆带来的误差放大 |
| 一般稠密方阵 | 手写无主元消元 | 部分主元 LU：$PA=LU$ | 控制消元乘子，减少误差增长 |
| 最小二乘 | 解 $A^TAx=A^Tb$ | QR 或 SVD | 因为 $\kappa(A^TA)=\kappa(A)^2$ |
| 病态问题 | 盲目相信稳定算法 | 先看 $\kappa(A)$ | 稳定算法不能改变问题本身难度 |

**条件数** $\kappa(A)$ 是衡量问题对输入扰动敏感程度的量，常见定义是

$$\kappa(A)=\|A\|\cdot\|A^{-1}\|$$

它描述的是问题本身有多难，不是某个算法写得好不好。稳定算法只能避免额外把问题算坏，不能把病态问题变成良态问题。

一个核心工程原则是：不要把数学公式直接翻译成代码。解线性方程时，数学上 $x=A^{-1}b$ 成立，但工程代码不应该先算 $A^{-1}$。更稳的路线是先分解：

$$PA=LU$$

然后解两个三角方程：

$$Ly=Pb,\quad Ux=y$$

这相当于走安全路线，而不是抄近路。

---

## 问题定义与边界

讨论数值稳定性时，要分清三个层次：

| 层次 | 解释 | 例子 | 能否靠算法消除 |
|---|---|---|---|
| 算法稳定性 | 浮点计算过程是否额外放大误差 | 无主元消元可能产生巨大乘子 | 可以改善 |
| 问题条件数 | 输入稍微变化时，精确解是否大幅变化 | $\kappa(A)$ 很大时解很敏感 | 不能消除 |
| 结果可解释性 | 结果是否等价于小扰动输入的精确解 | 后向稳定的输出 | 可以作为目标 |

**浮点数**是计算机对实数的有限精度近似表示。多数十进制小数在机器里不能被精确保存，所以每次加减乘除都可能产生舍入误差。

稳定性讨论的对象，是这些舍入误差在算法内部如何传播。它的边界也很明确：算法不能改变输入矩阵的条件数。若一个矩阵本来就接近奇异，稳定算法最多保证“没有额外恶化太多”，不能保证答案有很多有效数字。

玩具例子：考虑两个很接近的方程组。矩阵

$$A=\begin{bmatrix}1&1\\1&1.000001\end{bmatrix}$$

两行几乎线性相关。右端 $b$ 只要有很小变化，解就可能明显变化。这不是 LU、QR、SVD 的锅，而是问题本身接近不可区分。

真实工程例子：在结构力学和有限元分析中，刚度矩阵常常很大、稀疏，并且可能因为网格质量、边界条件、材料参数差异而病态。此时直接求逆会把载荷误差、边界条件误差和舍入误差一起放大。工程上通常使用带主元策略的分解、预条件迭代法，或者在需要诊断病态性时使用 SVD。

因此，数值稳定性不是“保证结果一定准”，而是“在给定问题条件下，尽量不引入额外灾难”。

---

## 核心机制与推导

矩阵稳定计算的主线有三条：部分主元 LU、Householder 正交变换、Golub-Kahan SVD 路线。

第一条是 LU 分解的部分主元选取。LU 分解把矩阵拆成下三角矩阵 $L$ 和上三角矩阵 $U$。**主元**是消元时用来做除法的那个元素。如果主元非常小，消元乘子会非常大，误差容易被放大。

例子：

$$A=\begin{bmatrix}10^{-20}&1\\1&1\end{bmatrix}$$

如果第一步不换行，主元是 $10^{-20}$，乘子是 $10^{20}$，这会制造巨大中间量。部分主元选取会在当前列选绝对值最大的元素作为主元，相当于先交换两行：

$$P=\begin{bmatrix}0&1\\1&0\end{bmatrix}$$

于是

$$PA=\begin{bmatrix}1&1\\10^{-20}&1\end{bmatrix}=LU$$

其中

$$L=\begin{bmatrix}1&0\\10^{-20}&1\end{bmatrix},\quad
U=\begin{bmatrix}1&1\\0&1-10^{-20}\end{bmatrix}$$

此时消元乘子是 $10^{-20}$，而不是 $10^{20}$。部分主元的意义不是让误差消失，而是避免算法主动制造巨大误差放大器。

第二条是 Householder 变换。**正交变换**是保持向量长度不变的线性变换。Householder 反射常写成

$$H=I-\tau vv^T$$

并满足

$$H^TH=I$$

这意味着它不会改变二范数。用它做 QR 分解时，可以稳定地把矩阵变成上三角形式。

玩具例子：令

$$x=\begin{bmatrix}4\\3\end{bmatrix},\quad \|x\|_2=5$$

可以构造一个 Householder 反射，使得

$$Hx=\begin{bmatrix}-5\\0\end{bmatrix}=y$$

即 $Hx=y$。它把第二个分量消成 0，同时长度仍然是 5。这比直接用不稳定的初等行操作更适合做最小二乘。

第三条是 SVD。**奇异值分解**把矩阵写成 $A=U\Sigma V^T$，其中奇异值反映矩阵在不同方向上的拉伸强度。Golub-Kahan 路线通常先用正交变换把矩阵双对角化：

$$Q^TAP=B$$

其中 $B$ 是双对角矩阵。然后对 $B$ 做隐式 QR 迭代，得到高质量奇异值。整体流程可以概括为：

```text
原矩阵
  -> 主元选取或正交变换
  -> 三角形式 / 双对角形式
  -> 回代 / 隐式 QR
  -> 解、最小二乘解或奇异值
```

这里还要特别强调正常方程。最小二乘问题

$$\min_x \|Ax-b\|_2$$

可以推导出正常方程：

$$A^TAx=A^Tb$$

数学上它成立，但数值上常常不该直接这么算，因为

$$\kappa(A^TA)=\kappa(A)^2$$

条件数被平方后，本来还能接受的问题可能变得非常敏感。工程上更稳的路线是 QR；如果矩阵明显病态，或者需要伪逆和秩判断，则用 SVD。

---

## 代码实现

实现层面的原则是：调用稳定分解，而不是手写消元和求逆。接口也应该优先暴露 `solve`，而不是暴露 `inverse`。

下面代码只依赖 NumPy，可直接运行。它展示同一个线性方程组上，“直接求逆”和“直接求解”的接口差异。NumPy 的 `solve` 会调用底层 LAPACK 求解例程，实际路线是分解后回代，而不是先显式形成逆矩阵。

```python
import numpy as np

A = np.array([[1e-20, 1.0],
              [1.0,   1.0]])
b = np.array([1.0, 2.0])

# 不推荐：数学上成立，但工程上不应优先使用
x_inv = np.linalg.inv(A) @ b

# 推荐：直接求解 Ax=b，底层使用分解与回代
x_solve = np.linalg.solve(A, b)

assert np.allclose(A @ x_solve, b)
assert np.allclose(x_inv, x_solve)

# 条件数只是问题敏感性的指标，不是算法稳定性的证明
cond = np.linalg.cond(A)
assert cond > 1
```

对于最小二乘，优先使用 QR 或库提供的 `lstsq`。`lstsq` 会根据实现选择稳定路线，通常比手写正常方程更可靠。

```python
import numpy as np

A = np.array([[1.0, 1.0],
              [1.0, 1.000001],
              [1.0, 0.999999]])
b = np.array([2.0, 2.000001, 1.999999])

# 不推荐：显式形成 A.T @ A，条件数会平方
x_normal = np.linalg.solve(A.T @ A, A.T @ b)

# 推荐：最小二乘接口，避免手写正常方程
x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

assert np.allclose(A @ x_lstsq, b, atol=1e-8)
assert np.linalg.cond(A.T @ A) > np.linalg.cond(A)
```

对于 SVD 伪逆，适合处理病态矩阵、低秩矩阵或需要截断小奇异值的场景。**伪逆**是逆矩阵的推广；当矩阵不可逆或不是方阵时，它仍能给出最小二乘意义下的解。

```python
import numpy as np

def svd_pinv_solve(A, b, rcond=1e-12):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = rcond * s[0]
    s_inv = np.array([1.0 / value if value > cutoff else 0.0 for value in s])
    return Vt.T @ (s_inv * (U.T @ b))

A = np.array([[1.0, 1.0],
              [1.0, 1.0 + 1e-10],
              [1.0, 1.0 - 1e-10]])
b = np.array([2.0, 2.0, 2.0])

x = svd_pinv_solve(A, b)

assert np.allclose(A @ x, b, atol=1e-8)
```

Householder 在高性能库里通常不显式构造完整的 $H$。实际实现只存反射向量 $v$ 和标量 $\tau$，需要作用到矩阵时再计算

$$A \leftarrow A-\tau v(v^TA)$$

这样可以减少存储和计算，也避免把原本结构化的反射器展开成稠密矩阵。

---

## 工程权衡与常见坑

工程中最常见的错误，是把纸面公式直接翻译成代码。数学推导关注等价关系，数值计算还要关注中间量、舍入误差、条件数和存储格式。

| 常见坑 | 问题 | 替代方案 |
|---|---|---|
| 显式求 $A^{-1}$ | 误差传播路径更长，计算量也更高 | 用 `solve`、LU 分解和回代 |
| 使用正常方程 | $\kappa(A^TA)=\kappa(A)^2$ | 用 QR 或 SVD |
| 无主元高斯消元 | 小主元导致巨大乘子 | 至少使用部分主元 LU |
| 显式展开 Householder | 浪费存储，破坏实现效率 | 只存 $v,\tau$ 并按反射器应用 |
| 忽略迭代中的正交性丢失 | Lanczos、Golub-Kahan 迭代会积累误差 | 必要时重正交化 |
| 把稳定等同于高精度 | 病态问题仍然敏感 | 同时检查条件数、残差和物理意义 |

真实工程例子：有限元刚度矩阵 $K$ 对应方程

$$Ku=f$$

其中 $u$ 是位移，$f$ 是载荷。若直接计算 $K^{-1}f$，边界条件误差、载荷噪声、网格带来的病态性都会被逆矩阵放大。更常见的工程路线是：对稀疏刚度矩阵使用稀疏直接法、带主元策略的分解，或配合预条件器的迭代法。若系统接近奇异，还需要检查约束是否不足，而不是只换一个求解函数。

稳定性优先于少量额外计算。多做一次分解通常是可接受成本；一次不稳定求解带来的错误结论，可能会让后续仿真、控制、优化全部偏离。

还要注意残差和误差不是一回事。残差是

$$r=b-A\hat{x}$$

残差小说明 $\hat{x}$ 代回方程后看起来满足方程，但若 $A$ 的条件数很大，解本身仍可能离真实解很远。因此工程检查应至少包括残差、条件数估计、输入尺度和领域约束。

---

## 替代方案与适用边界

不同矩阵问题不应该固定使用一种算法。应根据矩阵结构、规模、稀疏性、病态程度和目标精度选择路线。

| 问题类型 | 推荐方法 | 不推荐方法 | 适用原因 |
|---|---|---|---|
| 一般稠密方阵 $Ax=b$ | 部分主元 LU | 显式求逆 | 成本适中，稳定性通常足够 |
| 对称正定方阵 | Cholesky | 一般求逆 | 更快、更省存储，但要求正定 |
| 超定最小二乘 | QR | 正常方程 | 避免条件数平方 |
| 病态最小二乘 | SVD | 盲目 QR 后直接相信结果 | 可观察奇异值并截断 |
| 大规模稀疏系统 | 稀疏 LU、迭代法、预条件法 | 稠密分解 | 利用稀疏结构 |
| 只需少量奇异值 | Lanczos / randomized SVD | 完整 SVD | 降低成本，但要管理正交性 |

正常方程的适用边界很窄。若 $A$ 条件数小、问题规模适中、精度要求不高，正常方程可能够用。但只要进入工程默认设置，QR 通常是更稳的选择。核心原因仍然是：

$$\kappa(A^TA)=\kappa(A)^2$$

例如 $\kappa(A)=10^6$ 时，正常方程对应的条件数可能达到 $10^{12}$。在双精度浮点数中，这会明显吞掉有效数字。

SVD 不是永远最优。它通常更稳，也能提供秩和奇异值信息，但成本更高。若只是解一个良态的方阵系统，部分主元 LU 已经足够。若是超定最小二乘，QR 往往是稳定性和性能之间的平衡点。若要分析病态性、构造伪逆、做低秩近似，SVD 才是更合适的工具。

算法选择的原则可以压缩成一句话：先判断问题类型，再判断条件数和结构，最后选择能控制误差传播的分解方法。

---

## 参考资料

| 来源 | 能支撑的结论 |
|---|---|
| LAPACK `DGETRF` 文档 | LU 分解与部分主元选取，对应 $PA=LU$ |
| LAPACK `DGETRS` / `DGESV` 文档 | 使用 LU 因子解线性方程，避免显式求逆 |
| LAPACK `DGEQRF` 文档 | QR 分解的工程实现路线 |
| LAPACK `DLARFG` / `DLARF` 文档 | Householder 反射器以 $v,\tau$ 形式生成和应用 |
| LAPACK `DGEBD2` / `DBDSQR` 文档 | SVD 中双对角化和双对角矩阵 QR 迭代 |
| Golub, G. H. & Kahan, W. (1965), *Calculating the Singular Values and Pseudo-Inverse of a Matrix* | Golub-Kahan SVD 路线的理论来源 |
| Demmel, J. & Kahan, W. (1990), *Accurate Singular Values of Bidiagonal Matrices* | 双对角矩阵奇异值高精度计算的理论来源 |

实现级资料：

- LAPACK `DGETRF`: https://netlib.org/lapack/explore-html/db/d04/group__getrf_gaea332d65e208d833716b405ea2a1ab69.html
- LAPACK `DGETRS`: https://www.netlib.org/lapack/explore-html/df/d36/group__getrs_gaacd7a8465c8cc0e4e8b88ba4b453630c.html
- LAPACK `DGESV`: https://netlib.org/lapack/explore-html/d8/da6/group__gesv_ga831ce6a40e7fd16295752d18aed2d541.html
- LAPACK `DGEQRF`: https://netlib.org/lapack/explore-html/d0/da1/group__geqrf_gade26961283814bb4e62183d9133d8bf5.html
- LAPACK `DLARFG`: https://www.netlib.org/lapack/explore-html/d8/d0d/group__larfg_gadc154fac2a92ae4c7405169a9d1f5ae9.html
- LAPACK `DLARF`: https://www.netlib.org/lapack/explore-html/d2/d97/group__larf_gade3d1409d3046a8b5b39bb500456b349.html
- LAPACK `DGEBD2`: https://netlib.org/lapack/explore-html/d9/d03/group__gebd2_gabbe4b3257196bd5d7355d8840c69a2cf.html
- LAPACK `DBDSQR`: https://netlib.org/lapack/explore-html/d6/d51/group__bdsqr_gade20fbf9c91aa7de0c3d565b39588dc5.html

理论级资料：

- Golub, G. H. & Kahan, W. (1965): *Calculating the Singular Values and Pseudo-Inverse of a Matrix*. https://doi.org/10.1137/0702016
- Demmel, J. & Kahan, W. (1990): *Accurate Singular Values of Bidiagonal Matrices*. https://www.netlib.org/lapack/lawnspdf/lawn03.pdf
