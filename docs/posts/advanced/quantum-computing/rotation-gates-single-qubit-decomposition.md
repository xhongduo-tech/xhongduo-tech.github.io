---
title: 旋转门：Rx、Ry、Rz 与任意单比特门分解
date: 2026-08-07
---

# 旋转门：Rx、Ry、Rz 与任意单比特门分解

<div class="epigraph">
<p>先生们，这毫无疑问是对的，但它绝对是悖论：我们无法理解它，也不知道它是什么意思。然而我们已经证明了它，所以我们知道它必然为真。</p>
<footer>—— 本杰明 · 皮尔斯（Benjamin Peirce），论欧拉恒等式</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§4.2 单比特旋转门 ｜ 2026-08-07</p>
</div>

## 为什么从旋转门开始

前面三篇里，你见到的单比特门要么是「转半圈」（Pauli 门、H），要么是「转八分之一圈」（S、T）。但量子演化是连续的：绕 $z$ 轴转 $30°$ 与转 $31°$ 是两个不同的门，而真实的量子比特（比如超导芯片里被微波脉冲驱动）每一次操作本质上就是「绕某个轴转某个角度」。**旋转门 Rx、Ry、Rz 把「转半圈/转八分之一圈」推广成「任意角度」**，是连接「抽象的离散门」与「物理的连续脉冲」之间的桥。

更重要的是，旋转门引出一个漂亮的数学结论：**任意单比特幺正门都可以写成三次绕坐标轴旋转的乘积**——这是单比特门版本的「欧拉角分解」。学完本篇，你就有了一张「任意 $2\times 2$ 幺正矩阵都能被造出来」的保证书，这正是后面讨论「通用量子门集」的起点：只要会造任意单比特门，再配一个 CNOT，就能造出所有量子线路。

## 1 三个坐标轴旋转

**核心概念：** 对三个 Pauli 矩阵 $X, Y, Z$，定义**旋转门**为绕对应轴转 $\theta$ 弧度的幺正算符：

$$
R_x(\theta) = e^{-i\theta X/2}, \qquad
R_y(\theta) = e^{-i\theta Y/2}, \qquad
R_z(\theta) = e^{-i\theta Z/2}
$$

把它们展开成显式矩阵。因为 $X^2 = I$，指数可以奇偶劈开（和上一篇推导 $R_{\hat n}(\theta)$ 母公式完全相同的技巧），得到

$$
R_x(\theta) = \begin{pmatrix}
\cos\frac{\theta}{2} & -i\sin\frac{\theta}{2}\\
-i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
\end{pmatrix}, \qquad
R_y(\theta) = \begin{pmatrix}
\cos\frac{\theta}{2} & -\sin\frac{\theta}{2}\\
\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
\end{pmatrix}, \qquad
R_z(\theta) = \begin{pmatrix}
e^{-i\theta/2} & 0\\
0 & e^{i\theta/2}
\end{pmatrix}
$$

**重点：所有角度的分母都有个「2」**——转 $\theta$ 弧度的旋转门，矩阵里出现的是 $\theta/2$。这个「半角」是本篇最容易出错的地方，也是几何直觉与矩阵代数之间的第一道裂缝：布洛赫球上转 $\theta$，对应的量子门是 $e^{-i\theta X/2}$，不是 $e^{-i\theta X}$。原因在第二节讲。

验证几个熟悉的特例：$R_x(\pi) = \begin{pmatrix}0&-i\\-i&0\end{pmatrix} = -iX$，忽略全局相位 $-i$ 就是 Pauli 门 $X$；$R_z(\pi/2) = e^{-i\pi/4}\begin{pmatrix}1&0\\0&i\end{pmatrix} \equiv S$；$R_z(\pi/4) = e^{-i\pi/8}\begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix} \equiv T$。**旋转门把前面所有「离散门」都统一成「绕轴转任意角」的语言**：X 是 $R_x(\pi)$，S 是 $R_z(\pi/2)$，T 是 $R_z(\pi/4)$，H 是 $R_{\hat n}(\pi)$（绕对角轴）。

## 2 公式解析：为什么是 $\theta/2$

这一节回答两个问题：指数为什么写成 $-i\theta X/2$，以及 $\theta/2$ 从哪来。

**第一步，从旋转矩阵的指数定义出发。** 绕 $x$ 轴转 $\theta$ 的门定义为 $R_x(\theta) = e^{-i\theta X/2}$。用指数级数展开：

$$
e^{-i\theta X/2} = I - i\frac{\theta}{2}X + \frac{1}{2!}\left(-i\frac{\theta}{2}X\right)^2 + \frac{1}{3!}\left(-i\frac{\theta}{2}X\right)^3 + \cdots
$$

**第二步，利用 $X^2 = I$ 奇偶劈开。** 偶数幂 $X^{2k} = I$、奇数幂 $X^{2k+1} = X$，于是偶数项收集余弦、奇数项收集正弦：

$$
e^{-i\theta X/2} = \underbrace{\sum_{k}\frac{(-1)^k}{(2k)!}\left(\frac{\theta}{2}\right)^{2k}}_{\cos\frac{\theta}{2}}\, I
\; -\; i X \underbrace{\sum_{k}\frac{(-1)^k}{(2k+1)!}\left(\frac{\theta}{2}\right)^{2k+1}}_{\sin\frac{\theta}{2}}
= \cos\frac{\theta}{2}\, I - i\sin\frac{\theta}{2}\, X
$$

这正是第一节的 $R_x(\theta)$ 矩阵。**半角的来源在指数里就已经注定**：矩阵元里凑出 $\cos\frac{\theta}{2}$ 是因为展开参数是 $\theta/2$。

**第三步，几何直觉：为什么定义里要塞一个「2」？** 对 $R_z(\theta)$ 作用在布洛赫球北极 $\lvert 0\rangle$ 上，得到 $R_z(\theta)\lvert 0\rangle = e^{-i\theta/2}\lvert 0\rangle$——只差一个全局相位，态没动；而作用在赤道态 $\lvert +\rangle = \frac{\lvert 0\rangle+\lvert 1\rangle}{\sqrt2}$ 上：

$$
R_z(\theta)\lvert +\rangle = \frac{e^{-i\theta/2}\lvert 0\rangle + e^{i\theta/2}\lvert 1\rangle}{\sqrt{2}}
= e^{-i\theta/2}\,\frac{\lvert 0\rangle + e^{i\theta}\lvert 1\rangle}{\sqrt{2}}
$$

两个分量的**相对相位**是 $e^{i\theta}$。也就是说：量子门里的参数 $\theta$ 直接等于布洛赫球上转过的角度。若定义时少了那个「2」，相对相位就会变成 $e^{i\theta/2}$、旋转慢一半——物理上转 $90°$ 的脉冲，矩阵里却像是只转了 $45°$。所以「$2$ 在指数上、不在球面上」是理解旋转门的关键。<span class="marginnote">不同教科书对 $R_z$ 的约定略有出入：有的写成 $\mathrm{diag}(1, e^{i\theta})$，去掉了整体的 $e^{-i\theta/2}$ 相位。两者只差全局相位、物理等价，但<strong>相对相位</strong>与布洛赫球的对应关系不变。做题、写代码前先确认约定，避免正负号对不上。</span>

## 3 布洛赫球上的旋转与特殊角度

旋转门的几何图像极其简单：**$R_x(\theta)$ 让布洛赫球上的点绕 $x$ 轴转 $\theta$，$R_y$、$R_z$ 同理**。北极 $\lvert 0\rangle$ 被 $R_z$ 转不动（它在轴上），被 $R_y$ 转到赤道以下、再转回来。

把常见角度列成一张速查表：

| 门 | 角度 | 作用效果 |
| --- | :---: | --- |
| $R_x(\pi)$ | $\pi$ | $\equiv X$：比特翻转 |
| $R_y(\pi)$ | $\pi$ | $\equiv Y$：翻转 + 相位 |
| $R_z(\pi)$ | $\pi$ | $\equiv Z$：相位翻转 |
| $R_z(\pi/2)$ | $\pi/2$ | $\equiv S$ |
| $R_z(\pi/4)$ | $\pi/4$ | $\equiv T$ |
| $R_x(\pi/2)$ | $\pi/2$ | 把 $\lvert 0\rangle$ 送到 $\lvert +\rangle$ 与 $\lvert -i\rangle$ 之间的「赤道四分之一」 |

**重点：旋转门是单参数连续族，Pauli 门、S、T 都是它在特殊角度的采样点。** 这条「连续族离散采样」的思路正是通用量子门集的哲学——我们无法在硬件上精确实现任意连续角度，但可以用离散的 H、S、T 任意逼近，误差可控（Solovay-Kitaev 定理，见第三篇）。

**辨析｜易错点：** 旋转门之间**不对易**。$R_x(\theta_1)$ 与 $R_z(\theta_2)$ 一般不能交换次序——绕不同的轴旋转，谁先谁后结果不同（这就是为什么任意单比特门分解需要「三次、绕不同的轴」而非一次）。把旋转想成「标量乘法」、随意交换顺序，是初学阶段最常见的错误。

## 4 任意单比特门分解：Z-Y-Z 定理

现在来到本篇最深刻的结论。**任何 $2\times 2$ 幺正矩阵 $U$，都可以写成（忽略全局相位）三次旋转的乘积**：

$$
U = e^{i\alpha}\, R_z(\beta)\, R_y(\gamma)\, R_z(\delta)
$$

其中 $\alpha, \beta, \gamma, \delta$ 是四个实数。这就是 N&C 定理 4.1，量子版本的「欧拉角分解」。<span class="marginnote">这个定理也可以只用 $R_x$ 与 $R_z$ 写：$U = e^{i\alpha}R_z(\beta)R_x(\gamma)R_z(\delta)$。三种写法（Z-Y-Z、X-Z-X 等）本质相同——只需要「两个不同轴的旋转」就能生成所有单比特门，因为绕同轴的两个旋转可以合并。</span>

为什么一定存在这样的分解？数自由度：忽略全局相位后，任意单比特幺正门由 **3 个实参数**决定；而 $R_z(\beta) R_y(\gamma) R_z(\delta)$ 恰好提供 $\beta, \gamma, \delta$ 三个参数。参数个数对上，剩下的是构造性的证明。

一个直接的构造是「几何路线」：任意幺正门 $U$ 在布洛赫球上是一个旋转，绕某个轴 $\hat n$ 转 $\phi$ 角（第一篇的母公式）。任何一个三维旋转都可以分解成「先绕 $z$ 轴转 $\delta$、再绕 $y$ 轴转 $\gamma$、再绕 $z$ 轴转 $\beta$」——这正是经典力学里欧拉角的迁移：先对齐轴的方位，再转出所需的角，最后归位。所以 Z-Y-Z 分解的几何含义是：

- **$R_z(\delta)$**：把目标旋转的轴搬到合适的位置；
- **$R_y(\gamma)$**：绕新的 $y$ 方向转出主旋转角；
- **$R_z(\beta)$**：归位，补偿第一次 $z$ 旋转带来的多余转动。

**重点：这条定理保证「单比特门的世界是二维参数的、可被三次旋转覆盖的」**——它把「无穷多个 $2\times2$ 幺正矩阵」压缩成「四个实数」，也让硬件上实现任意单比特门变成「打三个微波脉冲」的问题。Qiskit 里的通用门 `U3(θ, φ, λ)` 正是这种分解的直接体现：`U3(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ)`（带上各自的相位约定）。

## 5 辨析：半角陷阱与旋转方向

本节把最容易翻车的几个点集中清算。

**辨析｜易错点一（半角）：** $R_z(\theta)$ 的矩阵是 $\mathrm{diag}(e^{-i\theta/2}, e^{i\theta/2})$。若你图省事写成 $\mathrm{diag}(1, e^{i\theta})$，计算叠加态干涉时相对相位会差一倍。判别方法：用 $\lvert +\rangle$ 检验——正确的约定会让 $R_z(\theta)\lvert +\rangle$ 产生相对相位 $e^{i\theta}$，错误的约定产生 $e^{i\theta/2}$。

**辨析｜易错点二（方向）：** $R_z(\theta)$ 给 $\lvert 1\rangle$ 乘的是 $e^{+i\theta/2}$、给 $\lvert 0\rangle$ 乘的是 $e^{-i\theta/2}$（取决于指数里 $\theta$ 的正负号约定）。这个符号约定在不同教材、不同数值库之间可能相反。写代码时**不要凭记忆推断正负**，先用 $\lvert +\rangle$ 或布洛赫球可视化验证一次。

**辨析｜易错点三（全局相位）：** 分解定理里那个 $e^{i\alpha}$ 是全局相位，物理上不可观测、常被省略。但如果把这个全局相位当成「相对相位」丢掉，在受控门里就会出错——受控门里全局相位会变成**条件相位**（见下一篇）。所以：「单比特门单独用时，全局相位无所谓；放进受控门，全局相位就有所谓了。」

## 6 一个可运行的示例（Qiskit）

用 Qiskit 验证旋转门，并检验「任意单比特门 = Z-Y-Z 三次旋转」：

```python
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, random_unitary

# 1) Rz(θ)|+⟩ 的相对相位应等于 e^{iθ}
theta = 0.6
qc = QuantumCircuit(1)
qc.h(0)          # 制备 |+⟩
qc.rz(theta, 0)  # 绕 z 轴转 theta
sv = Statevector(qc).data
print("相对相位 e^{iθ} =", np.exp(1j * theta))

# 2) 随机取一个单比特幺正矩阵 U，验证 Z-Y-Z 分解
U = random_unitary(2).data
# 用 transpiler 把 U 编译成 Rz/Ry/Rz 序列
qc2 = QuantumCircuit(1)
qc2.unitary(U, [0])
tqc = transpile(qc2, basis_gates=["rz", "ry"], optimization_level=3)
print(tqc.draw())
```

第一段对照第二节：$R_z(\theta)$ 作用在 $\lvert+\rangle$ 上，两个分量的相对相位正是 $e^{i\theta}$。第二段说明一个工程事实：Qiskit 的编译器把任意 `unitary` 自动翻译成旋转门序列——`basis_gates=["rz", "ry"]` 就够用，因为 Z-Y-Z 定理保证了「任意单比特门都能用两种旋转造出来」。把线路画出来，你会看到分解定理在编译器里真实地运行着。

## 7 小结

- **旋转门**：$R_x(\theta) = e^{-i\theta X/2}$、$R_y(\theta) = e^{-i\theta Y/2}$、$R_z(\theta) = e^{-i\theta Z/2}$，是绕坐标轴的连续旋转。
- **半角**：矩阵元出现 $\cos\frac{\theta}{2}$、$\sin\frac{\theta}{2}$；$\theta$ 是布洛赫球上的实际转角，指数上的「2」保证相对相位 $e^{i\theta}$ 正确。
- 特殊角度回扣离散门：$R_x(\pi) \equiv X$、$R_z(\pi/2) \equiv S$、$R_z(\pi/4) \equiv T$、$H = R_{\hat n}(\pi)$。
- **Z-Y-Z 分解定理**：任意 $2\times2$ 幺正矩阵 $U = e^{i\alpha}R_z(\beta)R_y(\gamma)R_z(\delta)$，三个参数覆盖所有单比特门。
- **辨析**：旋转门不对易、半角与符号约定易错、全局相位在受控门里会变成条件相位。
- 工程落地：Qiskit 的 `U3(θ, φ, λ)` 与 transpiler 的 `rz`/`ry` 编译都是分解定理的直接应用。

在下一节，我们走出单比特门的世界，进入真正的量子纠缠领域：**受控门 CNOT、CZ 与受控-U**——控制比特与目标比特通过受控操作建立关联，而关联正是纠缠的代数来源。
