---
title: Kronecker 积与张量积
date: 2026-08-11
---

# Kronecker 积与张量积

<div class="epigraph">
<p>张量积是一种"组合"：两个系统的状态空间相乘，两个变换的作用逐项复制——它是多体世界在线性代数里的语法。</p>
<footer>—— 化用自威廉·克朗内克（Leopold Kronecker）</footer>
</div>

<div class="article-byline">
<p>第二级 · 矩阵论 ｜ Horn & Johnson《Matrix Analysis》 ｜ 2026-08-11</p>
</div>

## 为什么从 Kronecker 积开始

我们已经处理过矩阵的各种运算：加减、乘、伴随、逆。这一节引入一种全新的乘法——**Kronecker
积（张量积）**：它把「两个系统」拼成「一个更大的系统」。统计学的重参数化、信号处理的张量化、量子力学的多粒子态、
以及下一节线性矩阵方程，全都依赖这个构造。直观上，$A \otimes B$ 的意思是「$A$ 管第一个因子、$B$
管第二个因子」，两者**并行、互不干扰**——这正是张量积的「直积」语义<span class="marginnote">从极限到大模型的连接：大模型的张量并行、LoRA 里的低秩分解、
Transformer 多头注意力的「头维度重组」都可看作 Kronecker/张量结构的运用；量子计算的纠缠态则是
$|\psi\rangle\otimes|\phi\rangle$ 的线性组合。张量积是"复合系统"的标准语言。
</span>。

## 1 定义与 vec 算子

**Kronecker 积（Kronecker product）**：设 $A$ 是 $m \times n$ 矩阵，
$B$ 是 $p \times q$ 矩阵，则 $A \otimes B$ 是 $mp \times nq$ 矩阵，
按块定义为

$$A \otimes B = \begin{pmatrix} a_{11}B & a_{12}B & \cdots & a_{1n}B \\ a_{21}B & a_{22}B & \cdots & a_{2n}B \\ \vdots & & \ddots & \vdots \\ a_{m1}B & a_{m2}B & \cdots & a_{mn}B \end{pmatrix}$$

每个元素 $a_{ij}$ 都被整块 $B$ 替换。<span class="marginnote">例：$\begin{pmatrix}1&2\\3&4\end{pmatrix} \otimes \begin{pmatrix}0&5\\6&7\end{pmatrix}$
是一个 $4\times4$ 矩阵，由四个 $2\times2$ 块拼成。注意块内是 $B$ 的<strong>完整拷贝</strong>，
不是逐元素乘。</span>

**vec
算子（vectorization）**：把矩阵按**列堆叠**成一个列向量：$\operatorname{vec}(A) = (a_{11}, \dots, a_{m1}, a_{12}, \dots, a_{mn})^{T}$。
vec 算子把「矩阵方程」翻译成「向量方程」，是连接 Kronecker 积与线性方程组的桥。最核心的恒等式：

$$\operatorname{vec}(AXB) = (B^{T} \otimes A)\,\operatorname{vec}(X)$$

**这条恒等式是整个线性矩阵方程理论的支点**：左边是「$X$ 被左右两边同时作用」，
右边变成「$\operatorname{vec}(X)$ 被一个大矩阵乘」——把「矩阵方程」降维成「普通线性方程组」。

## 2 基本性质：混合积与谱

Kronecker 积满足一批漂亮的性质。最重要的**混合积性质（mixed-product property）**：

$$(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$$

只要各乘积的维度相容。它意味着「两个大积的普通乘法 =
各自小矩阵普通乘法的张量积」——**张量积把普通乘法"保结构"地放大到复合空间**。<span class="marginnote">证明要点：$A\otimes B$ 与 $C\otimes D$ 都是块结构，
相乘时块内 $B$ 与 $D$ 对齐相乘、块间 $A$ 与 $C$ 对齐相乘，恰好得到 $(AC)\otimes(BD)$。
这是"交换直积顺序"的代数根基。</span>

**谱与行列式**：

- $(A \otimes B)^{*} = A^{*} \otimes B^{*}$，$(A \otimes B)^{T} = A^{T} \otimes B^{T}$；
- 若 $A, B$ 可逆，则 $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$；
- $\operatorname{tr}(A \otimes B) = \operatorname{tr}(A)\,\operatorname{tr}(B)$；
- $\det(A \otimes B) = \det(A)^{p}\det(B)^{m}$（$A$ 为 $m\times m$、$B$ 为 $p\times p$）。

**特征值结构**：若 $A$ 有特征值 $\lambda_1, \dots, \lambda_m$，$B$ 有特征值
$\mu_1, \dots, \mu_p$，则 $A \otimes B$ 的全部特征值为**全部两两乘积**
$\{\lambda_i \mu_j\}$，$A \otimes I + I \otimes B$ 的全部特征值为
$\{\lambda_i + \mu_j\}$。<span class="marginnote">这个谱规律是
Kronecker 积最深刻的财产：复合系统的谱 = 分量的谱的「张量组合」。量子力学里两个独立系统的总能量
$E = E_1 + E_2$（求和）正是 $\lambda_i + \mu_j$ 的形式；这也预告了下一节
Lyapunov 方程谱的来历。</span>

**辨析｜易错点：** Kronecker 积**不满足交换律**
$A \otimes B \neq B \otimes A$（只差一个置换，即存在置换矩阵 $P$ 使
$B \otimes A = P(A\otimes B)P^{T}$）。
初学者常误以为张量积可交换——**两个因子的次序由"谁管第一空间、谁管第二空间"决定，换序即换角色**。
另一高频错误：$\operatorname{vec}(AXB)$ 中矩阵是 $B^{T}\otimes A$ 而非
$A\otimes B^{T}$，次序与转置必须与「左右乘」对应。

## 3 张量积的向量视角

对向量 $u \in \mathbb{C}^{m}$、$v \in \mathbb{C}^{p}$，张量积
$u \otimes v$ 是 $\mathbb{C}^{mp}$ 中的向量，
分量取两两乘积：$(u \otimes v)_{(i-1)p + j} = u_i v_j$。<span class="marginnote">物理记号
$|\psi\rangle \otimes |\phi\rangle$：两个系统组成复合系统，基向量取「第一系统基 ×
第二系统基」的笛卡尔积。$u\otimes v$ 对应"可分离态"，而一般的
$\sum_k c_k u_k \otimes v_k$ 对应纠缠态——张量积的线性组合就是纠缠的数学表述。</span>

张量积把**线性作用的复合**正确实现：对线性映射 $A: \mathbb{C}^m \to \mathbb{C}^m$、
$B: \mathbb{C}^{p} \to \mathbb{C}^{p}$，
$(A \otimes B)(u \otimes v) = (Au) \otimes (Bv)$。
两个变换在各子空间上**独立地**作用，互不耦合——这是「张量积 = 并行系统」的语言证据。

## 4 应用：张量化、量子比特与并行结构

Kronecker 积不是孤立构造，它是「张量积」在矩阵上的坐标表达。三个应用场景最能体现它的价值。

**张量化（tensorization）**：把多维数组（张量）按某一模式展开成矩阵，再用 Kronecker 结构分析，
是信号处理、图像处理与张量网络的标准操作。例如二维离散傅里叶变换（DFT）可写成

$$F_{MN} = (F_M \otimes F_N)\, P$$

其中 $P$ 是排列矩阵——**二维变换是分块 Kronecker 结构**。这使二维 FFT 可以分解为若干一维 FFT
的组合，复杂度从 $O(N^2)$ 级降到 $O(N\log N)$ 级。<span class="marginnote">直觉：二维变换「先沿行、再沿列」的可分离性，正是 Kronecker
积"各行独立作用"的结构体现。可分离核在图像卷积、小波变换里无处不在，而可分离性 = Kronecker 可分解性。
</span>

**量子比特的数学**：单量子比特态是 $\mathbb{C}^2$ 中的单位向量
$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$。
两个独立量子比特的联合态是张量积空间
$\mathbb{C}^2 \otimes \mathbb{C}^2 \cong \mathbb{C}^4$：

$$|\psi\rangle \otimes |\phi\rangle = \begin{pmatrix} \alpha \phi_1 \\ \alpha \phi_2 \\ \beta \phi_1 \\ \beta \phi_2 \end{pmatrix}$$

而两比特门（如 CNOT 门）是 $4\times4$ 矩阵，单比特门是 $U \otimes I$ 或
$I \otimes V$ 形式的 Kronecker 积。**量子线路 = 一系列 Kronecker 结构门的复合**。
<span class="marginnote">纠缠的本质：并非所有 $\mathbb{C}^4$ 态都能写成
$u\otimes v$ 的形式。不能写成张量积的态就是纠缠态（如 Bell 态
$\frac1{\sqrt2}(|00\rangle+|11\rangle)$）。可分离性测试 =
"能否写成单个张量积"——Kronecker 积给了纠缠一个精确的代数判据。</span>

**并行与分布式计算**：张量并行把大矩阵按行/列切分到多设备，其梯度与误差的聚合天然涉及 Kronecker 结构；
深度学习中的批量矩阵乘法、多头注意力的"头重组"，也都可用 Kronecker/分块语言描述。<span class="marginnote">从极限到大模型的连接：LoRA 的低秩增量 $BA$ 与 Kronecker
无关但同为"结构化低维参数"；而更大的故事是——注意力、卷积、FFT 都藏在某类 Kronecker/张量分解的框架下，
理解张量积就是理解"如何把大计算拆成可并行的子块"。</span>

**矩阵方程的回响**：下一节 Sylvester/Lyapunov 方程的 vec 展开正是 Kronecker
积最重要的应用——把"矩阵乘法方程"翻译成"大线性方程组"。这条桥在本节已经搭好。

**辨析｜易错点：**
$A \otimes (B \otimes C) = (A \otimes B) \otimes C$（结合律成立），但
$A \otimes B \neq B \otimes A$（只差置换）。在张量化应用中，**排列矩阵 $P$
的出现说明"哪个因子管哪根轴"是约定问题**——实现时必须小心轴的顺序，否则结果张量是"转置"了的。
另一个易错点：$\operatorname{vec}(AXB)$ 的公式是 $(B^{T}\otimes A)$，
若按行堆叠（$\operatorname{vec}$ 的另一种约定），转置的位置会换到 $A$
上——堆叠方向必须与公式约定一致。

## 5 公式解析：vec 恒等式 $\operatorname{vec}(AXB) = (B^{T}\otimes A)\operatorname{vec}(X)$

这是全篇最值得拆解的公式，拆四步：

- **第一步，先看 $AXB$ 的第 $j$ 列**：$(AXB)_{:,j} = AX b_{:,j}$（$B$ 的第 $j$ 列乘到右边）。于是 $\operatorname{vec}(AXB)$ 按列堆叠后，第 $j$ 块是 $A(X b_{:,j})$。
- **第二步，把 $Xb_{:,j}$ 展开**：$Xb_{:,j} = \sum_{k} b_{kj} x_{:,k}$，即 $X$ 各列的线性组合，权重是 $B$ 第 $j$ 列的元。
- **第三步，重排为块乘**：对每列 $j$ 都这么做，等价于「$A$ 作用在 $X$ 的每个列上，再按 $B$ 的转置元加权」——恰好是 $B^{T}\otimes A$ 乘 $\operatorname{vec}(X)$ 的块结构。
- **第四步，为什么转置**：$B$ 的第 $j$ 列元 $b_{kj}$ 在堆叠后出现在 $B^{T}$ 的第 $j$ 行，所以权重矩阵取 $B^{T}$ 而非 $B$。**转置来自 vec「按列堆叠」的约定**——若按行堆叠，转置会跑到 $A$ 上。

## 6 小结

- **Kronecker 积** $A\otimes B$ 是「块替换」式构造，维度 $mp \times nq$；vec 算子按列堆叠。
- **核心恒等式** $\operatorname{vec}(AXB) = (B^{T}\otimes A)\operatorname{vec}(X)$，把矩阵方程翻译成线性方程组。
- **混合积性质** $(A\otimes B)(C\otimes D) = AC\otimes BD$；不满足交换律（差一个置换）。
- **谱结构**：$A\otimes B$ 谱为 $\{\lambda_i\mu_j\}$，$A\otimes I + I\otimes B$ 谱为 $\{\lambda_i + \mu_j\}$。
- 易错点：$B^{T}\otimes A$ 的次序；张量积换序即换角色；vec 堆叠方向决定转置位置。

在下一节，我们将用刚才的 vec 恒等式一口气解决最重要的两类线性矩阵方程——Sylvester 方程与 Lyapunov
方程。
