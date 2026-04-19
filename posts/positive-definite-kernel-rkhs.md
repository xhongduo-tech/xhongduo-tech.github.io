## 一句话核心定义

正定核 $k(x,x')$ 是一个把“输入对”映射成内积的函数；再生核希尔伯特空间（RKHS）是与它对应的函数空间，其中每个点值都能写成内积：

$$
f(x)=\langle f, k(x,\cdot)\rangle_{\mathcal H}
$$

严格地说，机器学习里更常写“正半定核（PSD kernel）”。只要任意有限样本下的 Gram 矩阵都半正定，这个核就可用。

---

## 面向新手的直观解释

普通线性模型只会在原始特征上做点积。核方法把“先映射到高维特征空间再做点积”这一步折叠成一个核函数 $k$，所以不用显式构造高维特征。

RKHS 的关键不是“空间很抽象”，而是“取值可以被表示成内积”。这意味着求 $f(x)$ 不需要额外写一个评价规则，只要拿 $f$ 去和代表元 $k(x,\cdot)$ 做内积即可。

同一个 $k$ 在不同方法里扮演的角色一致：在核 SVM 里它控制间隔和边界形状，在核岭回归里它控制平滑方式，在高斯过程中它就是协方差函数。

---

## 关键公式与机制

统一记号如下：

| 记号 | 含义 |
|---|---|
| $k(x,x')$ | 核函数 |
| $K_{ij}=k(x_i,x_j)$ | Gram 矩阵 |
| $\mathcal H$ | 与 $k$ 对应的 RKHS |
| $f\in\mathcal H$ | RKHS 中的函数 |
| $\alpha$ | 核方法的系数向量 |

正定核的判据是：对任意样本 $\{x_i\}_{i=1}^n$，

$$
K_{ij}=k(x_i,x_j),\qquad
\forall c\in\mathbb R^n,\; c^\top K c\ge 0
$$

RKHS 的再生性质是：

$$
f(x)=\langle f, k(x,\cdot)\rangle_{\mathcal H}
$$

这条式子直接解释了“点值评价变成内积运算”。

核岭回归的闭式解写成：

$$
\hat f(x)=\sum_{i=1}^n \alpha_i k(x_i,x),\qquad
\alpha=(K+\lambda I)^{-1}y
$$

这里 $\lambda$ 是正则项。核 SVM 的对偶问题也只依赖 $K_{ij}$，而不依赖显式特征维度。

高斯过程与 RKHS 的关系可以记成一句话：同一个核 $k$，在 GP 里是协方差，在 RKHS 里是内积核。前者给概率不确定性，后者给函数几何结构。

---

## 一个最小数值例子

取一维输入 $x_1=1,x_2=2$，核函数用线性核：

$$
k(x,x')=xx'
$$

对应 Gram 矩阵是：

$$
K=
\begin{bmatrix}
1 & 2\\
2 & 4
\end{bmatrix}
$$

它是半正定的，因为对任意 $c=(c_1,c_2)^\top$，

$$
c^\top K c=(c_1+2c_2)^2\ge 0
$$

这说明 $k$ 合法。

再取 RKHS 里的函数 $f(x)=3x$。在线性核对应的空间里，可以把它看成 $\mathcal H=\{ax\}$，并定义

$$
\langle ax,bx\rangle_{\mathcal H}=ab
$$

则 $k(2,\cdot)=2x$，所以

$$
\langle f, k(2,\cdot)\rangle_{\mathcal H}
=\langle 3x,2x\rangle_{\mathcal H}
=6
=f(2)
$$

这就是再生性质的最小例子：先算内积，再得到点值。

---

## 一个真实工程场景

小样本回归是核方法最典型的工程场景之一，比如材料配方预测、传感器标定、质量分数估计。

当样本量只有几百到几千，但特征之间存在明显非线性关系时，核岭回归常比手工多项式特征更省事。流程通常是：选一个 $k$（常见是 RBF），调 $\lambda$ 和核宽度，直接解 $(K+\lambda I)\alpha=y$。

好处是实现简单，收敛稳定，和高斯过程共享同一类核设计。代价也明确：训练阶段通常要存整个 $n\times n$ Gram 矩阵，样本再大就会碰到 $O(n^2)$ 内存和 $O(n^3)$ 求解成本。

---

## 常见坑与规避

| 坑 | 直接后果 | 规避方式 |
|---|---|---|
| 把“核函数”当成任意相似度 | 可能不是合法核 | 检查 Gram 矩阵是否半正定 |
| 只看函数形式，不看样本规模 | 大样本时算不动 | 先估算 $n^2$ 内存和求解代价 |
| 把 RKHS 和高维显式特征混为一谈 | 容易误判计算成本 | 记住“核技巧”只算 $k(x,x')$ |
| 误以为核越复杂越好 | 过拟合或数值不稳 | 用验证集调核宽度和正则项 |
| 忽略近奇异 Gram 矩阵 | 解不稳定 | 加 $\lambda I$ 或 jitter |
| 把 GP 和 RKHS 当成同一件事 | 解释会混乱 | GP 关注不确定性，RKHS 关注函数空间几何 |

一个判断标准很实用：如果你的方法最后必须显式处理 $n\times n$ 矩阵，那它更适合中小规模；如果目标是百万级样本，通常要换近似核、Nyström、随机特征或直接改用其他模型。

---

## 参考来源

1. N. Aronszajn, *Theory of Reproducing Kernels*, Transactions of the AMS, 1950. [DOI / AMS PDF](https://doi.org/10.1090/S0002-9947-1950-0051437-7)
2. Bernhard Schölkopf, Alexander J. Smola, *Learning with Kernels*. [MIT Press](https://mitpress.mit.edu/9780262536578/learning-with-kernels/)
3. Carl Edward Rasmussen, Christopher K. I. Williams, *Gaussian Processes for Machine Learning*. [MIT Press](https://mitpress.mit.edu/9780262182539/gaussian-processes-for-machine-learning/)
4. scikit-learn 文档，*Kernel ridge regression*. [官方文档](https://scikit-learn.org/stable/modules/kernel_ridge.html)
5. scikit-learn 文档，*Gaussian Processes*. [官方文档](https://scikit-learn.org/stable/modules/gaussian_process.html)

{"summary":"正定核把非线性学习写成可计算的内积。"}
