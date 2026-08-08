---
title: Schmidt 分解与纯化
date: 2026-08-07
---

# Schmidt 分解与纯化

<div class="epigraph">
<p>在数学里，你并不是理解了一个东西，你只是对习惯了它。</p>
<footer>—— 约翰 · 冯 · 诺伊曼（John von Neumann）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Nielsen & Chuang《量子计算》§2.5 Schmidt 分解与纯化 ｜ 2026-08-07</p>
</div>

## 为什么从 Schmidt 分解开始

上一篇我们发现了「纠缠偷走纯度」：贝尔态的一半是 $I/2$，一个纯的整体对应混合的局部。今天把这件事翻过来，回答两个对称的问题：**一个任意的两体纯态，怎么看出它缠得有多紧？** 以及 **一个任意的混合态，能不能重新浸入一个纯态、只是把另一半藏起来？**

答案分别是 **Schmidt 分解**与**纯化**——它们是一对互逆的镜子。Schmidt 分解给两体纯态找「标准形」：无论态多复杂，总能写成少数几对正交基的加权和，系数（Schmidt 系数）直接读出资质的强弱；纯化则反过来，把任意混合态 $\rho_A$ 垫一个辅助系统变成纯态，且不改变 A 上的任何测量统计。这两把工具是纠缠度量、隐形传态、量子信道容量与开放系统演化的共同地基。<span class="marginnote">Schmidt 分解最早由德国数学家埃哈德 · 施密特（Erhard Schmidt）在 1907 年研究积分方程时提出，本是泛函分析里关于希尔伯特空间内积核的标准技巧；冯 · 诺伊曼把它引入量子力学，用来分析复合系统与熵。</span>

## 1 定理：两体纯态的标准形

设 $|\psi\rangle_{AB}$ 是复合系统 $AB$ 的任意纯态，$d_A, d_B$ 分别为两子系统的维数。**Schmidt 分解定理**说：存在 $A$ 的一组标准正交基 $\{|i_A\rangle\}$、$B$ 的一组标准正交基 $\{|i_B\rangle\}$，以及非负实数 $\lambda_i$，使得

$$
|\psi\rangle_{AB} = \sum_i \lambda_i\, |i_A\rangle \otimes |i_B\rangle, \qquad \sum_i \lambda_i^2 = 1
$$

其中 $\lambda_i \geq 0$ 称为 **Schmidt 系数（Schmidt coefficients）**，指标 $i$ 最多取到 $\min(d_A, d_B)$ 项。<span class="marginnote">注意三个细节：系数是非负实数（相位被吸进基矢里了）；两个子系统共享同一组指标——$|i_A\rangle$ 与 $|i_B\rangle$ 一一配对；求和项数受限于较小的那个子系统的维数。</span>

这个定理的「标准形」意义在于：**再乱的两体纯态，换一组聪明的基后，也只是「配好对」的对角叠加。** 交叉项、纠缠的形态全部被坐标变换吸收，剩下的只有一行系数 $\{\lambda_i\}$——纠缠的全部信息就浓缩在这行数里。

## 2 证明思路：一切归结为奇异值分解

定理为什么成立？把态写成系数矩阵再奇异值分解（SVD）即可。具体走四步：

- **第一步，选两组任意的正交基。** 在 $A$ 取基 $\{|j\rangle\}$，在 $B$ 取基 $\{|k\rangle\}$，把态展开成
$$
|\psi\rangle = \sum_{j,k} a_{jk}\, |j\rangle|k\rangle
$$
系数构成一个 $d_A \times d_B$ 矩阵 $A = (a_{jk})$。

- **第二步，对 $A$ 做 SVD。** 任意矩阵都可分解为 $A = U \Sigma V^\dagger$，其中 $U$、$V$ 是酉矩阵，$\Sigma$ 是「非负对角阵」$\mathrm{diag}(\sigma_1, \sigma_2, \dots)$，$\sigma_i$ 叫奇异值。展开写：

$$
a_{jk} = \sum_i U_{ji}\,\sigma_i\, V_{ki}^*
$$

- **第三步，定义 Schmidt 基。** 令
$$
|i_A\rangle = \sum_j U_{ji}|j\rangle, \qquad |i_B\rangle = \sum_k V_{ki}^*|k\rangle
$$
$U, V$ 酉 ⇒ 两组新基各自标准正交。

- **第四步，代回重组。** $\sum_{jk} a_{jk}|j\rangle|k\rangle = \sum_i \sigma_i\big(\sum_j U_{ji}|j\rangle\big)\big(\sum_k V_{ki}^*|k\rangle\big) = \sum_i \sigma_i |i_A\rangle|i_B\rangle$，取 $\lambda_i = \sigma_i$ 即得定理。归一化 $\sum_i\lambda_i^2 = \|\psi\|^2 = 1$ 自动成立。

**重点：** 所以 Schmidt 分解不是「新的深奥定理」，而是**奇异值分解在量子态上的翻译**。SVD 是你可能已经认识的老朋友——它是线性代数里矩阵分解的瑞士军刀，也是机器学习的推荐系统、主成分分析（PCA）与隐语义模型的数学内核；今天我们给同一个工具换了一身量子的衣服。<span class="marginnote">跨界一点看：SVD 在数据科学里被用来「找出矩阵最要紧的几个方向」，Schmidt 分解在量子信息里被用来「找出两体态最要紧的几对关联」——同一个数学，两个舞台。本博客第一级《线性代数》与第三级《矩阵分解》还会系统展开 SVD。</span>

## 3 Schmidt 秩与纠缠

Schmidt 系数里有几个非零，记为 **Schmidt 秩（Schmidt rank）** $r$。它几乎是「纠缠度」的代名词：

- $r = 1$：只有一个非零系数，$|\psi\rangle = |i_A\rangle|i_B\rangle$ 是**乘积态**——子系统各自是纯态，无纠缠。
- $r \geq 2$：态**纠缠**。$r$ 越大，参与纠缠的「正交方向」越多。
- **最大纠缠态**：$r$ 个系数全部相等，$\lambda_i = 1/\sqrt{r}$。贝尔态（$r=2$，$\lambda_1=\lambda_2=1/\sqrt2$）就是两比特系统的最大纠缠态。

**辨析｜易错点：** Schmidt 秩与「态含几个基向量」无关，与「展开后有几项」也无关——一个 $|00\rangle + |01\rangle + |10\rangle + |11\rangle$ 四项俱全的态（$= |+\rangle|+\rangle$）Schmidt 秩是 **1**，因为它能拆成乘积态；而只有两项的 $|00\rangle + |11\rangle$ 秩是 **2**。判断纠缠要看**能否拆成乘积**，不是看项数多少——这与张量积篇「叠加 ≠ 纠缠」的结论一脉相承。

Schmidt 分解还立刻给出两个约化密度算符的**谱**。把 $|\psi\rangle = \sum_i\lambda_i|i_A\rangle|i_B\rangle$ 代入，非对角交叉项因 $\langle i_B|i'_B\rangle = 0$ 全部消失：

$$
\rho_A = \mathrm{tr}_B|\psi\rangle\langle\psi| = \sum_i \lambda_i^2 |i_A\rangle\langle i_A|, \qquad
\rho_B = \mathrm{tr}_A|\psi\rangle\langle\psi| = \sum_i \lambda_i^2 |i_B\rangle\langle i_B|
$$

**重点：** $A$ 与 $B$ 的约化密度算符**有相同的非零本征值** $\{\lambda_i^2\}$。所以两体纯态的纠缠熵是良定义的：$S(\rho_A) = S(\rho_B) = -\sum_i \lambda_i^2\log\lambda_i^2$。纯态时 $S=0$，最大纠缠时 $S = \log r$。<span class="marginnote">这就是第四篇《纠缠的度量》里「纠缠熵」的来历：子系统的熵既度量「我们丢掉 B 后损失了多少信息」，又度量「A 与 B 缠得有多紧」。两体纯态的熵作为纠缠度量，本质上就是在数 Schmidt 系数的「不均匀程度」。</span>

## 4 纯化：把混合态浸回纯态大海

现在反着来。给定系统 A 的任意混合态 $\rho_A$，我们想构造一个两体纯态 $|\psi\rangle_{AR}$（$R$ 是虚构的辅助系统，叫**参考系统（reference system）**），使得

$$
\mathrm{tr}_R\big(|\psi\rangle\langle\psi|_{AR}\big) = \rho_A
$$

构造只有三步：

- **第一步，把 $\rho_A$ 对角化**：$\rho_A = \sum_i p_i |i_A\rangle\langle i_A|$，$p_i \geq 0$、$\sum_i p_i = 1$（$\rho_A$ 厄米，总有本征分解）。
- **第二步，在 $R$ 里准备一组标准正交基** $\{|i_R\rangle\}$，维度取 $R$ 的维数 $\geq$ 非零 $p_i$ 的个数。
- **第三步，定义纯态**：

$$
|\psi\rangle_{AR} = \sum_i \sqrt{p_i}\, |i_A\rangle |i_R\rangle
$$

验证只需一行：$\mathrm{tr}_R|\psi\rangle\langle\psi| = \sum_{i,i'}\sqrt{p_ip_{i'}}|i_A\rangle\langle i'_A|\,\langle i'_R|i_R\rangle = \sum_i p_i|i_A\rangle\langle i_A| = \rho_A$。<span class="marginnote">这正是一个「信息论等式」：混合态 $\rho_A$ 携带的不确定性，可以被看作「A 与外界 R 的纠缠」——把不知道的事归因于一个看不见的纠缠对象。量子纠错与退相干理论里，环境就扮演这个 $R$，而「系统变混合」被理解为「系统与环境的纠缠在增长」。</span>

**辨析｜易错点：** 纯化**不唯一**——对 $R$ 施加任意酉变换 $U_R$，$|\psi'\rangle = (I_A \otimes U_R)|\psi\rangle$ 同样是 $\rho_A$ 的纯化。这个自由度不是缺陷，而是工具：在证明量子信道容量、推导 Holevo 界时，我们可以挑选最顺手的那个纯化。反过来，一旦选定纯化，**A 的所有测量统计与 $\rho_A$ 完全相同**，因为约化密度算符相同——「A 与一个更大的纯态世界关联」这一图像，完全不会影响 A 的可见行为。

## 5 公式解析：SVD 与纯化的两个算例

**算例一：一个「看起来乱」的态，其实是最大纠缠。** 设

$$
|\psi\rangle = \frac12\big(|00\rangle + |01\rangle + |10\rangle - |11\rangle\big)
$$

- **第一步，写成系数矩阵**：按 $\{|j\rangle\}\otimes\{|k\rangle\}$ 排布，$M = \frac12\begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix}$。
- **第二步，做 SVD**：$MM^\dagger = \frac14\begin{pmatrix}1&1\\1&-1\end{pmatrix}\begin{pmatrix}1&1\\1&-1\end{pmatrix} = \frac12 I$，所以奇异值都是 $1/\sqrt2$，Schmidt 系数 $\lambda_1 = \lambda_2 = 1/\sqrt2$。
- **第三步，读出 Schmidt 基**：取 $U = I$，则 $V^\dagger = \sqrt2\,M = \frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$，$V$ 的两列正是 $|+\rangle, |-\rangle$。于是

$$
|\psi\rangle = \frac1{\sqrt2}\Big(|0_A\rangle|+_B\rangle + |1_A\rangle|-_B\rangle\Big)
$$

一个四项俱全、看似平凡叠加的态，换基之后竟是**最大纠缠态**（秩 2、系数相等），约化密度算符 $\rho_A = \rho_B = I/2$。这就是 Schmidt 分解的威力：**它把纠缠的本相从「表象」里剥离出来**。

**算例二：混合态的纯化。** 设 $\rho_A = \frac34|0\rangle\langle0| + \frac14|1\rangle\langle1|$。引入参考系统 $R$（两维），令

$$
|\psi\rangle_{AR} = \sqrt{\tfrac34}\,|0_A\rangle|0_R\rangle + \sqrt{\tfrac14}\,|1_A\rangle|1_R\rangle
$$

对 $R$ 求部分迹：对角项贡献 $\frac34|0\rangle\langle0|$ 与 $\frac14|1\rangle\langle1|$，交叉项被 $\langle 0_R|1_R\rangle = 0$ 杀掉，恰好还原 $\rho_A$。**注意** $|\psi\rangle_{AR}$ 的 Schmidt 秩是 2——一个秩 2 的混合态，其纯化必是秩 2 的纠缠态；一般地，**$\rho_A$ 的秩就是其纯化态的 Schmidt 秩**。用 NumPy 可以复现两个算例：

```python
import numpy as np

# 算例一：Schmidt 分解 = 系数矩阵的 SVD
M = np.array([[1, 1], [1, -1]]) / 2          # |ψ⟩ = (|00⟩+|01⟩+|10⟩-|11⟩)/2
U, lam, Vh = np.linalg.svd(M)
print(lam)                                   # [0.7071, 0.7071] —— λ₁=λ₂=1/√2
print(U @ np.diag(lam) @ Vh)                 # 还原系数矩阵

# 算例二：混合态的纯化
psi_AR = np.sqrt([0.75, 0.25])[:, None] * np.eye(2)   # √¾|0R⟩+√¼|1R⟩
rho_A = psi_AR @ psi_AR.T                    # 对 R 求部分迹
print(rho_A)                                 # diag(0.75, 0.25) 还原 ρ_A
```

## 6 小结

- **Schmidt 分解**：任意两体纯态可写为 $|\psi\rangle = \sum_i \lambda_i |i_A\rangle|i_B\rangle$，$\lambda_i \geq 0$、$\sum_i\lambda_i^2 = 1$；本质是系数矩阵的 **SVD**。
- **Schmidt 秩**决定纠缠：秩 1 ⇔ 乘积态，秩 ≥ 2 ⇔ 纠缠，系数全相等 ⇔ 最大纠缠；判断纠缠看「能否拆成乘积」，不是看项数。
- **约化谱相等**：$\rho_A, \rho_B$ 有相同的非零本征值 $\{\lambda_i^2\}$，纠缠熵 $S = -\sum\lambda_i^2\log\lambda_i^2$ 良定义。
- **纯化**：$\rho_A = \sum_i p_i|i_A\rangle\langle i_A|$ 的纯化是 $|\psi\rangle_{AR} = \sum_i\sqrt{p_i}|i_A\rangle|i_R\rangle$；不唯一（$R$ 上任意酉），但所有可见统计不变。
- 密度算符、部分迹、Schmidt 分解、纯化——四件工具合起来，构成理解「纠缠与子系统」的完整工具箱。

在下一节，我们把视角从「多体系统的数学」收回到**最朴素的两个字母**：$|0\rangle$ 与 $|1\rangle$。一个量子比特凭什么承载这些精致的结构？它的叠加、相位与测量，将带我们进入布洛赫球的几何世界。
