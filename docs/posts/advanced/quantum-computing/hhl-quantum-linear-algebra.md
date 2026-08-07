---
title: HHL 算法与量子线性代数
date: 2026-08-07
---

# HHL 算法与量子线性代数

<div class="epigraph">
<p>如果线性方程组的规模大到经典无法处理，HHL 或许是量子计算的第一个「真正」应用。</p>
<footer>—— 哈罗（Aram Harrow）、哈西迪姆（Avinatan Hassidim）与劳埃德（Seth Lloyd）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Harrow, Hassidim, Lloyd 2009（PRL）｜ 2026-08-07</p>
</div>

## 为什么从 HHL 开始

到目前为止的 QML 都在「变分 + 编码」的 NISQ 框架里打转；**HHL 算法**是「门模型 + 相位估计」的**理论旗舰**——它用量子线路在**对数时间**内求解线性方程组 $A\vec x = \vec b$。如果数据能以「量子态」形式给出、且答案只需要「某些统计量」，HHL 给出对经典方法的**指数级加速**。它是「量子线性代数」的开山之作，也是所有「量子优势」宣称中最有理论分量、也最受去量化攻击的靶子。<span class="marginnote">HHL 发表于 A. Harrow, A. Hassidim, S. Lloyd, "Quantum algorithm for linear systems of equations," <i>PRL</i> 103 (2009) 150502。它的核心是「把求解线性方程组变成相位估计 + 受控旋转 + 逆相位估计」。本节讲清它「怎么做到指数加速」以及「为什么这个加速有严格的前提」。</span>

## 1 问题的量子化设定

经典线性方程组 $A\vec x = \vec b$（$A$ 是 $N\times N$ 厄米矩阵，$N = 2^n$）。**量子版本**：数据以量子态给出——

$$
A \lvert x\rangle = \lvert b\rangle, \qquad \lvert b\rangle = \sum_i b_i \lvert i\rangle
$$

目标：制备 $\lvert x\rangle \propto A^{-1}\lvert b\rangle$（归一化后）。加速声称：若 $A$ 是稀疏的、条件数 $\kappa$ 良好，则 HHL 用时

$$
O\left( \log(N)\, s^2 \kappa^2 / \epsilon \right)
$$

其中 $s$ 是稀疏度、$\epsilon$ 是精度。<span class="marginnote">对比经典：一般稠密求解是 $O(N^3)$，稀疏迭代是 $O(N \cdot s \cdot \kappa)$——都是 $N$ 的多项式。HHL 的 $\log N$ 是<strong>指数级改进</strong>。但注意前提：<strong>$\lvert b\rangle$ 要能高效制备、答案要能高效读出（只能取统计量，不能逐分量读）</strong>——这两条就是去量化攻击的落脚点。</span>

## 2 HHL 的三步骨架

HHL 由三个模块拼成（全部来自第五篇的工具箱）：

1. **相位估计**：把 $A$ 的本征相位「读」进辅助寄存器。设 $A = \sum_j \lambda_j \lvert u_j\rangle\langle u_j\rvert$，$\lvert b\rangle = \sum_j b_j \lvert u_j\rangle$。相位估计给出 $\sum_j b_j \lvert u_j\rangle \lvert \tilde\lambda_j\rangle$（本征值编码进寄存器）。
2. **受控旋转**：根据 $\lambda_j$ 旋转辅助比特，把 $\frac{1}{\lambda_j}$ 的权重「写」进振幅：$\sum_j b_j \lvert u_j\rangle \lvert\tilde\lambda_j\rangle \left(\sqrt{1-\frac{c^2}{\lambda_j^2}}\lvert0\rangle + \frac{c}{\lambda_j}\lvert1\rangle\right)$。
3. **逆相位估计**：把 $\lvert\tilde\lambda_j\rangle$ 寄存器「清空」，辅助比特测到 $\lvert1\rangle$ 时得到 $\lvert x\rangle \propto \sum_j \frac{b_j}{\lambda_j}\lvert u_j\rangle = A^{-1}\lvert b\rangle$。<span class="marginnote">每一步都呼应前面的内容：相位估计（第五篇）读出本征相位、受控旋转是「受控-$U$」的变体（第三篇）、逆变换清寄存器（QFT 的可逆性）。HHL 是「量子算法工具箱」的集大成演示——这也是它作为教材经典章节的原因。</span>

## 3 公式解析：为什么受控旋转能实现 $A^{-1}$

核心是「用相位估计把 $A^{-1}$ 变成受控旋转」。$A$ 的对角化

$$
A = \sum_j \lambda_j \lvert u_j\rangle\langle u_j\rvert \;\Longrightarrow\; A^{-1} = \sum_j \frac{1}{\lambda_j} \lvert u_j\rangle\langle u_j\rvert
$$

- **第一步，谱分解**：$A$ 厄米可对角化，本征值 $\lambda_j$（实数）、本征向量 $\lvert u_j\rangle$。
- **第二步，逆的本征分解**：$A^{-1}$ 的同一个本征向量、本征值取倒数 $\frac{1}{\lambda_j}$。
- **第三步，实现倒数**：相位估计让寄存器读出 $\lambda_j$，受控旋转把「$\frac{c}{\lambda_j}$」作为辅助比特的 $\lvert1\rangle$ 振幅——测到 $\lvert1\rangle$ 的分量携带 $\frac{1}{\lambda_j}$ 权重，即 $A^{-1}$ 作用。<span class="marginnote">要点：<strong>量子算法不「求逆矩阵」，而是「把每个本征分量按 $\frac{1}{\lambda_j}$ 缩放」</strong>——利用谱分解把矩阵运算变成「本征空间的振幅调制」。这套「对角化 + 调制」的范式，也是量子模拟（$e^{-iAt}$）与量子信号处理（QSP）的共同骨架。</span>

## 4 公式解析：加速与前提

HHL 的复杂度来源：

$$
T_{\rm HHL} = O\left( \frac{s \kappa^2 \log N}{\epsilon} \right)
$$

- **第一步，$\log N$ 来自相位估计**：相位估计用 $O(\log N)$ 个控制比特的 QFT，成本对数于矩阵维数。
- **第二步，$\kappa^2$ 来自条件数**：$\lambda_j$ 很小时 $\frac{1}{\lambda_j}$ 很大，受控旋转的振幅小、成功概率低，需要「振幅放大」补偿——补偿次数与 $\kappa$ 相关。
- **第三步，$\epsilon$ 来自精度**：相位估计的误差与旋转的近似误差都要控住，总精度预算 $\epsilon$。<span class="marginnote">读法：<strong>加速来自 $\log N$（问题维数），代价来自 $\kappa^2/\epsilon$（问题病态性）</strong>。HHL 对「好条件 + 稀疏」的矩阵是指数加速，对病态矩阵则退化为多项式甚至更差。这也是「量子线性代数的边界」的定量刻画。</span>

**辨析｜易错点：** HHL 的输出是**量子态** $\lvert x\rangle$，不是经典向量。读取答案的方式受限：只能测量「期望值」（如 $\langle x\rvert M \lvert x\rangle$）、或采样「分量分布」（需多次）。**你无法直接「看」到所有 $x_i$**——想得到完整解向量，需要 $\Omega(N)$ 次测量，指数加速瞬间蒸发。这个「制备快、读取慢」的错位，正是 HHL 去量化争论的核心。

## 5 量子线性代数的全景与去量化

HHL 开启的「量子线性代数」家族还包括：

- **量子奇异值分解（quantum SVD）**、**量子矩阵乘法**、**量子最小二乘**、**量子主成分分析（qPCA）**——把经典矩阵运算「量子化」。
- 这些算法共享 HHL 的前提：「数据量子化 + 输出统计量 + 矩阵稀疏/低秩」。

**去量化（dequantization）**：Tang 等（2019 起）证明——若输入以「采样 + 查询（sample-query）」模型给出，许多「量子线性代数」算法可以被**经典算法以多项式（而非指数）时间模拟**，量子加速消失。结论：量子优势需要「输入量子态是廉价的、输出是廉价的统计量」这类**更强的前提**。<span class="marginnote">去量化给 QML 的教训：<strong>「指数加速」的算法若输入输出都不方便，实际价值有限</strong>。量子线性代数的真正优势需要「数据天然在量子态里」——这正是「量子数据上的量子学习」的意义（第十一篇第一节），也解释了为什么 HHL 式的乐观在实用层面被大幅回调。</span>

## 6 小结

- **HHL**：$A\lvert x\rangle = \lvert b\rangle$，用「相位估计 + 受控旋转 + 逆相位估计」在对数时间求解。
- **加速**：$O(s\kappa^2\log N/\epsilon)$——对稀疏良态矩阵指数加速。
- **机制**：谱分解 + 本征空间振幅调制，不做「矩阵求逆」。
- **前提**：$\lvert b\rangle$ 可高效制备、答案只读统计量——「制备快、读取慢」的错位是软肋。
- **去量化**：sample-query 模型下许多量子线性代数可被经典多项式模拟——量子优势需要更强的输入输出前提。

在下一节，我们把整个第十一篇的乐观与怀疑摆上桌——**量子机器学习的加速争议与去量化**。
