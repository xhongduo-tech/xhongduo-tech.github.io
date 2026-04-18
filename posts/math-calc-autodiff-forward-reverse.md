## 1. 一句话核心定义

自动微分是沿计算图机械执行链式法则的方法。前向模式传播切向量，计算 Jacobian-Vector Product，简称 JVP；反向模式传播伴随向量，计算 Vector-Jacobian Product，简称 VJP。对函数 $f:\mathbb{R}^m\to\mathbb{R}^n$，前向模式适合输入维度小，反向模式适合输出维度小。

---

## 2. 面向新手的直观解释

把函数看成一串基本算子。前向模式像“带着一个输入方向一起往前算”，得到这个方向在输出端造成的变化。反向模式像“先完成前向，再把输出端的梯度往回传”，所以训练里常用它直接拿到标量损失对全部参数的梯度。

计算图不是预先写死的公式表，而是在前向执行时动态构建出来的。反向传播时，系统从输出节点往输入节点回走，按依赖顺序把局部导数乘起来。

---

## 3. 关键公式/机制

设

$$
f(x)=y,\quad x\in\mathbb{R}^m,\ y\in\mathbb{R}^n,\quad J=\frac{\partial f}{\partial x}\in\mathbb{R}^{n\times m}.
$$

前向模式计算

$$
Jv \in \mathbb{R}^n,
$$

其中 $v\in\mathbb{R}^m$ 是输入方向。反向模式计算

$$
u^\mathsf{T}J \in \mathbb{R}^m,
$$

其中 $u\in\mathbb{R}^n$ 是输出端的伴随向量。若要显式拼出完整 Jacobian，前向模式通常按“列”推进，约需 $m$ 次方向传播；反向模式通常按“行”推进，约需 $n$ 次方向传播。

常用经验是：

| 任务 | 方向 | 典型产物 | 更划算的维度 |
| --- | --- | --- | --- |
| 输入扰动传到输出 | 前向模式 | $Jv$ | $m$ 小、$n$ 大 |
| 输出梯度传回输入 | 反向模式 | $u^\mathsf{T}J$ | $n$ 小、$m$ 大 |

---

## 4. 一个最小数值例子

取

$$
f(x_1,x_2)=
\begin{bmatrix}
x_1x_2 \\
x_1+x_2
\end{bmatrix}.
$$

在点 $(2,3)$ 处，

$$
J=
\begin{bmatrix}
3 & 2 \\
1 & 1
\end{bmatrix}.
$$

前向模式，取输入方向 $v=(1,4)^\mathsf{T}$：

$$
Jv=
\begin{bmatrix}
3 & 2 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
1\\4
\end{bmatrix}
=
\begin{bmatrix}
11\\5
\end{bmatrix}.
$$

反向模式，取输出权重 $u=(5,6)^\mathsf{T}$：

$$
u^\mathsf{T}J=
\begin{bmatrix}
5 & 6
\end{bmatrix}
\begin{bmatrix}
3 & 2 \\
1 & 1
\end{bmatrix}
=
\begin{bmatrix}
21 & 16
\end{bmatrix}.
$$

这个例子能直接检查两件事：JVP 输出维度跟着 $n$ 走，VJP 输出维度跟着 $m$ 走。

---

## 5. 一个真实工程场景

训练神经网络时，损失通常是标量，参数却是百万到百亿级。此时反向模式只要一次 backward，就能得到损失对全部参数的梯度，所以几乎所有主流深度学习框架都把它作为默认训练路径。

相反，在科学计算里常见另一类任务：输入参数很少，输出观测很多，比如对仿真模型做参数敏感度分析。这时前向模式往往更合适，因为只需要跟随少量输入方向传播，不必为每个输出单独回传一次。

---

## 6. 常见坑与规避

| 坑 | 表现 | 规避 |
| --- | --- | --- |
| 把 JVP 和 VJP 搞反 | 方向维度对不上 | 记住 JVP 是“输入方向到输出”，VJP 是“输出权重回输入” |
| 显式构造大 Jacobian | 时间和内存都爆 | 优先用 `jvp` / `vjp`，不要直接展开矩阵 |
| 反向模式保存太多激活 | 长序列训练 OOM | 用 activation checkpointing，只存检查点，反向时重算 |
| checkpoint 里有随机算子 | 前后向不一致 | 保留 RNG 状态，或确认算法允许非确定性 |
| 原地改写张量 | 梯度错误或报错 | 避免破坏计算图版本计数 |

---

## 7. 参考来源

1. [JAX: Forward- and reverse-mode autodiff in JAX](https://docs.jax.dev/en/latest/jacobian-vector-products.html)
2. [JAX: `jax.jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)
3. [JAX: `jax.vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html)
4. [PyTorch: Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd)
5. [PyTorch: `torch.autograd.functional.jvp`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jvp.html)
6. [PyTorch: `torch.autograd.functional.vjp`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.vjp.html)
7. [PyTorch: `torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint)
8. [Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (JMLR, 2018)](https://www.jmlr.org/papers/v18/17-468.html)

{"summary":"前向模式算JVP，反向模式算VJP；计算图前向构建、反向遍历。"}
