---
title: Hartree-Fock方法
date: 2026-08-07
---

# Hartree-Fock方法

<div class="epigraph">
<p>每一个电子都感觉到其他电子的平均势场，而它自己也贡献于这个平均势场——这就是自洽场。</p>
<footer>—— 道格拉斯 · 哈特里（Douglas Hartree）与弗拉基米尔 · 福克（Vladimir Fock）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子化学 ｜ Szabo & Ostlund《Modern Quantum Chemistry》Ch.3 ｜ 2026-08-07</p>
</div>

## 为什么从 Hartree-Fock 方法开始

前面我们把多电子波函数压缩成一条 Slater 行列式，但留下一个悬而未决的问题：**行列式里的轨道到底长什么样？** 答案是 Hartree-Fock（HF）方法给出的——通过自洽场（self-consistent field, SCF）迭代，找到使总能量最低的那组轨道。HF 是量子化学的「第零级近似」：一切更精确的方法（MP、CC、CI、DFT 的 Kohn-Sham）都以它为起点或参照，它的轨道还被用作化学家理解分子图像的直觉工具。

这一篇要讲清楚 HF 的三件事：**变分原理怎么落成 Roothaan 方程**、**Fock 算符里各项的物理含义**、以及**SCF 迭代为什么是「自洽」的**。

## 1 从变分到 HF 方程

HF 的逻辑链是：在「波函数 = 单 Slater 行列式」这一约束下，对轨道作变分使能量最低。用拉格朗日乘子法加上轨道正交归一约束 $\langle \phi_i | \phi_j \rangle = \delta_{ij}$，变分条件导出**HF 方程**：

$$\hat{f} \, \phi_i = \varepsilon_i \, \phi_i$$

其中 $\hat{f}$ 是**Fock 算符**，$\varepsilon_i$ 是**轨道能量**。<span class="marginnote">HF 方程的形式与单电子薛定谔方程 $\hat{h}\phi = \varepsilon\phi$ 惊人地相似——只是把单电子哈密顿量换成了 Fock 算符。轨道能量 $\varepsilon_i$ 是「一个电子占据轨道 $i$ 时的能量」，近似地（Koopmans 定理）对应电离能/电子亲和能的负值。</span>

关键点在于：**Fock 算符依赖被它作用的轨道本身**。它不是一个「给定」的算符，而是由全体占据轨道构造出来的平均势场——这正是「自洽」二字的由来。

## 2 Fock 算符的三项结构

对一个闭壳层体系，Fock 算符可以显式写成：

$$\hat{f} = \hat{h} + \sum_{j}^{\text{occ}} \left( 2\hat{J}_j - \hat{K}_j \right)$$

- **$\hat{h}$**：**单电子算符**，包含动能与核吸引，与具体哪个电子无关。
- **$\hat{J}_j$**：**库仑算符**，描述「占据轨道 $j$ 的电子云对当前电子的静电排斥」。它的作用 $\hat{J}_j\phi_i(\mathbf{x}) = \left[ \int |\phi_j(\mathbf{x}')|^2 \frac{1}{|\mathbf{x}-\mathbf{x}'|} d\mathbf{x}' \right] \phi_i(\mathbf{x})$——一个乘法势。
- **$\hat{K}_j$**：**交换算符**，是非局域算符，作用后把轨道里的电子「交换」出去：$\hat{K}_j\phi_i(\mathbf{x}) = \left[ \int \phi_j^*(\mathbf{x}')\frac{1}{|\mathbf{x}-\mathbf{x}'|}\phi_i(\mathbf{x}')d\mathbf{x}' \right]\phi_j(\mathbf{x})$。它没有经典对应，是费米子反对称性的产物。

<span class="marginnote">库仑算符是「局部」的乘法势，交换算符是「非局部」的积分算符——这是 HF 与 DFT 的一个关键技术差异：HF 的交换项难以直接处理，而 DFT 用一个局部势 $v_{xc}(\mathbf{r})$ 近似整个交换相关效应。</span>

**辨析｜易错点：** 初学者常把「Fock 算符的本征值 $\varepsilon_i$」当作「把第 $i$ 个电子拿出来时的体系总能量差」。实际上 HF 轨道能量是「平均场框架内的单电子能量」，总能量不等于轨道能量之和（前一篇已述）；电离能的严格表述需要 Koopmans 定理（$I_i \approx -\varepsilon_i$，忽略弛豫与关联）或 ΔSCF 方法。

## 3 Roothaan-Hall 方程：HF 的线性代数形式

把 LCAO 展开 $\phi_i = \sum_\mu c_{\mu i}\chi_\mu$ 代入 HF 方程，并投影到基函数 $\{\chi_\mu\}$ 上，得到矩阵形式的 **Roothaan-Hall 方程**：

$$\mathbf{FC} = \mathbf{SC}\boldsymbol{\varepsilon}$$

其中 $\mathbf{F}$ 是**Fock 矩阵**（$F_{\mu\nu} = \langle \chi_\mu | \hat{f} | \chi_\nu \rangle$），$\mathbf{S}$ 是**重叠矩阵**（$S_{\mu\nu} = \langle \chi_\mu | \chi_\nu \rangle$，基函数一般不正交），$\mathbf{C}$ 是轨道系数矩阵，$\boldsymbol{\varepsilon}$ 是对角轨道能量矩阵。<span class="marginnote">这是一个<strong>广义本征值问题</strong>。若基函数正交（$\mathbf{S} = \mathbf{I}$），就退化为普通本征方程 $\mathbf{FC} = \mathbf{C}\boldsymbol{\varepsilon}$。现实中基函数不正交，必须先做对称正交化（如 Löwdin 正交化）再求解。</span>

Fock 矩阵的每个元素也依赖轨道系数（通过密度矩阵），因此方程是**非线性**的，只能迭代求解——这就是 SCF 迭代。

## 4 公式解析：SCF 迭代与密度矩阵

SCF 迭代的完整流程，是 HF 的灵魂：

$$
\mathbf{P} \xrightarrow{\text{构造}} \mathbf{F} \xrightarrow{\text{求解}} \mathbf{C} \xrightarrow{\text{重建}} \mathbf{P} \xrightarrow{\text{收敛?}} \text{输出}
$$

其中**密度矩阵** $\mathbf{P}$ 定义为（闭壳层）：

$$P_{\mu\nu} = 2 \sum_{i}^{\text{occ}} c_{\mu i} c_{\nu i}$$

逐步拆解：

- **初始化**：猜测初始密度矩阵 $\mathbf{P}^{(0)}$，通常用「扩展 Hückel」或原子密度近似。
- **构造 Fock 矩阵**：$F_{\mu\nu} = H^{\text{core}}_{\mu\nu} + \sum_{\lambda\sigma} P_{\lambda\sigma} \left[ (\mu\nu|\lambda\sigma) - \frac{1}{2}(\mu\lambda|\nu\sigma) \right]$，其中 $H^{\text{core}}$ 是单电子积分矩阵，$(\mu\nu|\lambda\sigma)$ 是**双电子排斥积分**。
- **求解广义本征方程** $\mathbf{FC} = \mathbf{SC}\boldsymbol{\varepsilon}$，得到新轨道系数。
- **重建密度矩阵**，计算新旧 $E$ 的差；若能量与密度矩阵的变化小于阈值（如 $10^{-8}$ Ha），则收敛，否则回到第二步。

<span class="marginnote">双电子积分 $(\mu\nu|\lambda\sigma)$ 的数目是 $O(K^4)$（$K$ 为基函数数），是 HF 计算的瓶颈。正因如此，现代程序用各种技巧（积分筛选、密度拟合/RI、线性标度方法）降低这一成本——这是后面《程序与软件》一篇要展开的工程现实。</span>

收敛失败的常见处理：**阻尼（damping）**——每次迭代不直接用新密度，而是新旧混合 $P^{\text{new}} = \alpha P^{\text{out}} + (1-\alpha)P^{\text{old}}$；或 **DIIS**（直接反演迭代子空间），用过去几步的误差信息外推，加速收敛。

## 5 HF 的成就与局限

HF 方法到底好不好？要放在**坐标系**里看：

| 维度 | HF 的表现 |
| --- | --- |
| 基态分子结构 | 很好：平衡键长、键角的 HF 预测已相当接近实验 |
| 能量学 | 一般：反应能、解离能的误差可达每化学键几十 kJ/mol |
| 化学键断裂 | 差：两电子同时断裂（如 $\ce{H2}$ 解离）HF 定性失败 |
| 激发态 | 差：需用 CIS/TD-DFT 等方法 |
| 计算成本 | 低：$O(N^4)$，可处理数百原子 |

HF 的核心缺陷是**忽略动态电子关联**：平均场让两个电子「感受不到」彼此的瞬时位置，费米穴只处理了自旋平行的电子，自旋反平行的电子之间没有任何关联。这个缺失的能量差称为**关联能**：

$$E_{\text{corr}} = E_{\text{exact}} - E_{\text{HF}}$$

关联能通常只占总能量的约 1%，但对化学问题（尤其反应能、解离能）而言，这 1% 往往正是决定性的部分。<span class="marginnote">「HF 忽略了关联能」听起来是小事，实则关系重大：化学反应的能量差往往是两个大数的差，HF 的 1% 误差在绝对尺度上比反应能本身还大。所以化学家宁愿放弃 HF 的「变分保证」，去用非变分的 MP2、CCSD(T) 等关联方法——精度远比「能量在真值上方」重要。</span>

### Koopmans 定理：轨道能量的化学意义

HF 轨道能量 $\varepsilon_i$ 除了是「方程的本征值」，还有一个漂亮的化学解释——**Koopmans 定理**：

> 在忽略轨道弛豫（离子化后其余轨道不变）与电子关联的近似下，从占据轨道 $i$ 电离一个电子所需能量 $I_i$ 等于 $-\varepsilon_i$；电子亲和能 $A_a$ 等于 $-\varepsilon_a$（虚轨道）。

$$I_i \approx -\varepsilon_i$$

<span class="marginnote">Koopmans 定理让「分子轨道能级图」有了真实的物理意义：HOMO 能量近似等于电离能的负值，LUMO 能量近似等于电子亲和能的负值，HOMO-LUMO 能隙近似对应最低激发能。这正是化学家乐于用「能级图」讨论反应性的依据——但它建立在「离子化后轨道不动」这一强近似上。</span>

**辨析｜易错点：** Koopmans 定理给出的电离能往往**偏大**（约 1–2 eV），因为忽略了离子化后电子的**弛豫**与**关联**。更准确的电离能要用 ΔSCF（分别算中性分子与离子，取能量差）或含关联的方法。所以「HOMO 能量 = 电离能」只能当作定性工具，不能用于精确的热化学。

另一个实用提示：HF 虽然对能量「不够准」，但它的**波函数与轨道**常被当作后续方法（MP2、CCSD、CASSCF）的**起点**——这些方法的精度都以「HF 参考」为参照。所以 HF 不只是「第一步近似」，更是整座方法大厦的**地基**：地基的稳定性（SCF 是否收敛、参考是否合适）决定了上层建筑的可靠性。

## 6 小结

- HF 在「波函数 = 单 Slater 行列式」约束下对轨道变分，导出 **HF 方程 $\hat{f}\phi_i = \varepsilon_i\phi_i$**。
- **Fock 算符 = 单电子算符 + 库仑势 − 交换势**；交换项是反对称性的量子效应。
- LCAO 化后得 **Roothaan-Hall 方程 $\mathbf{FC} = \mathbf{SC}\boldsymbol{\varepsilon}$**，是非线性的广义本征问题。
- **SCF 迭代**：猜测密度 → 构造 Fock → 求解 → 重建密度 → 判敛；阻尼与 DIIS 帮助收敛。
- HF 对**结构预测很好，对能量预测不够**；缺失的关联能由后续电子相关方法补上。

在下一节，我们将回答一个实际操作问题：轨道在什么**基函数**上展开、基函数怎么选？——这就是**基组理论与选择**，它决定了 HF 计算能达到的精度上限。
