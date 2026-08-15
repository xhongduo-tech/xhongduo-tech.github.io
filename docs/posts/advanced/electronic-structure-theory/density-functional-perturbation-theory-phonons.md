---
title: 密度泛函微扰理论与声子
date: 2026-08-07
---

# 密度泛函微扰理论与声子

<div class="epigraph">
<p>晶体不是冻结的快照，而是一张由声子编织的活网。</p>
<footer>—— 马克斯 · 玻恩（Max Born）与 黄昆</footer>
</div>

<div class="article-byline">
<p>第四级 · 第一性原理计算与电子结构理论 ｜ R. M. Martin《Electronic Structure》第21章 ｜ 2026-08-07</p>
</div>

## 为什么从密度泛函微扰理论与声子开始

到目前为止，我们把晶体当成了「原子钉死在平衡位置」的静态体系。但真实材料永远是活的：原子在平衡位置附近热振动，这种振动以**声子（phonon）**的形式存在。声子决定热容、热导、超导（BCS 机制）、拉曼散射、相变的软模——没有声子，材料科学就失去了一半。<span class="marginnote">声子是晶格振动的量子化集体激发：就像光子的波粒二象性，晶格振动既是原子的集体位移模式，也携带能量 $\hbar\omega$ 的准粒子。声子谱就是「晶格振动的能带图」。</span>

问题是：声子来自原子受力对位移的响应，也就是**能量的二阶导数**。直接做法是有限位移法——把原子挪一点，重新自洽，用差分算二阶导数。这个方法简单但贵，且数值噪声大。**密度泛函微扰理论（Density Functional Perturbation Theory, DFPT）**提供了更优雅的路径：不挪原子，直接在线性响应理论的框架里解析地求能量二阶导。本节就来拆解这条路径。

## 1 声子问题的数学表述

晶格振动的核心是**动力学矩阵**。设体系有 $N$ 个原子，第 $\kappa$ 个原子沿 $\alpha$ 方向位移 $u_{\kappa\alpha}$，则谐波近似下的势能展开为

$$
E = E_0 + \frac12\sum_{\kappa\alpha,\kappa'\alpha'} \Phi_{\kappa\alpha,\kappa'\alpha'}\,u_{\kappa\alpha}u_{\kappa'\alpha'}
$$

其中 $\Phi$ 是**力常数矩阵（interatomic force constant matrix）**——势能对位移的二阶导数：

$$
\Phi_{\kappa\alpha,\kappa'\alpha'} = \frac{\partial^2 E}{\partial u_{\kappa\alpha}\,\partial u_{\kappa'\alpha'}}
$$

对周期体系做傅里叶变换，得到每个波矢 $\mathbf{q}$ 下的**动力学矩阵**：

$$
D_{\kappa\alpha,\kappa'\alpha'}(\mathbf{q}) = \frac{1}{\sqrt{M_\kappa M_{\kappa'}}}\sum_{\mathbf{R}} \Phi_{\kappa\alpha,\kappa'\alpha'}(\mathbf{R})\,e^{i\mathbf{q}\cdot\mathbf{R}}
$$

求解动力学矩阵的本征值问题 $\det[D(\mathbf{q}) - \omega^2 I] = 0$，本征值 $\omega^2(\mathbf{q})$ 就是声子频率的平方，本征矢是振动模式。<span class="marginnote">动力学矩阵对角化的物理意义：$3N$ 个自由度（每原子 3 个方向）在每个 $\mathbf{q}$ 点给出 $3N$ 支声子色散关系。声学支（低频，$\omega\to 0$ 当 $\mathbf{q}\to 0$）对应整体位移，光学支（高频）对应原子间相对振动。$\mathbf{q}\to 0$ 处声学支频率必须为零——这是声子的「金标准」，用于校验计算的正确性。</span>

## 2 从 Hellmann-Feynman 力到线性响应

力常数是能量二阶导，但计算它并不需要能量二阶导那么麻烦。**Hellmann-Feynman 定理**告诉我们，原子受力等于势能对核坐标的导数，可写成电子密度的积分：

$$
F_{\kappa\alpha} = -\frac{\partial E}{\partial u_{\kappa\alpha}}
= -\int n(\mathbf{r})\,\frac{\partial V_{\mathrm{ion}}}{\partial u_{\kappa\alpha}}\,\mathrm{d}\mathbf{r}
$$

关键在于：**力只依赖密度 $n$ 与离子势的一阶导数**。对力再求导，得到力常数：

$$
\Phi_{\kappa\alpha,\kappa'\alpha'} = \frac{\partial F_{\kappa\alpha}}{\partial u_{\kappa'\alpha'}}
= -\int \frac{\partial n(\mathbf{r})}{\partial u_{\kappa'\alpha'}}\,\frac{\partial V_{\mathrm{ion}}}{\partial u_{\kappa\alpha}}\,\mathrm{d}\mathbf{r} - \int n(\mathbf{r})\,\frac{\partial^2 V_{\mathrm{ion}}}{\partial u_{\kappa\alpha}\partial u_{\kappa'\alpha'}}\,\mathrm{d}\mathbf{r}
$$

第二项只含离子势，容易算；第一项需要**密度对位移的一阶响应 $\partial n/\partial u$**——这是 DFPT 的主角。<span class="marginnote">DFPT 的核心洞见：求声子不需要「挪原子再自洽」，只需要解一次<strong>线性响应方程</strong>得到 $\partial n/\partial u$。这就像微积分里用解析求导代替数值差分：既准确（无差分噪声），又高效（一次求解得到任意扰动的响应）。</span>

**辨析｜易错点：** Hellmann-Feynman 力只有在**自洽收敛的密度**下才成立。若密度未收敛，波函数对核坐标的导数不再为零，会引入非物理的 Pulay 力。因此 DFPT 要求高质量的自洽解作起点——这是所有响应计算的前提。

## 3 DFPT 的线性响应方程

密度对位移的响应 $\partial n/\partial u$ 由 Kohn-Sham 轨道的响应构成：

$$
\frac{\partial n(\mathbf{r})}{\partial u_{\kappa\alpha}} = \sum_i f_i\,\left[ \psi_i^*(\mathbf{r})\,\frac{\partial \psi_i(\mathbf{r})}{\partial u_{\kappa\alpha}} + \mathrm{c.c.} \right]
$$

而轨道响应 $\partial \psi_i/\partial u$ 满足**线性响应 Sternheimer 方程**（一阶微扰的 Kohn-Sham 方程）：

$$
\big(\hat{H}_{\mathrm{KS}} - \varepsilon_i\big)\frac{\partial \psi_i}{\partial u_{\kappa\alpha}}
= -\left( \frac{\partial V_{\mathrm{eff}}}{\partial u_{\kappa\alpha}} - \frac{\partial \varepsilon_i}{\partial u_{\kappa\alpha}} \right)\psi_i
$$

注意到方程左边有 $\hat{H}_{\mathrm{KS}}-\varepsilon_i$，这是奇异的（对占据态为零）。解决之道：只求 $\partial \psi_i$ 在空态（未占据）子空间的分量，占据-占据部分的贡献由正交约束吸收掉。<span class="marginnote">Sternheimer 方程这个名字是致敬原始论文；它的优雅在于<strong>不需要显式求和空态</strong>——大多数微扰方法（如直接微扰论）要遍历所有虚态，而 DFPT 通过解线性方程组直接得到响应，计算量只与基组大小成正比。</span>

而扰动 $\partial V_{\mathrm{eff}}/\partial u$ 本身又依赖密度响应（因为 Hartree 势与交换关联势依赖密度）：

$$
\frac{\partial V_{\mathrm{eff}}}{\partial u_{\kappa\alpha}} = \frac{\partial V_{\mathrm{ion}}}{\partial u_{\kappa\alpha}} + \int \frac{\partial n(\mathbf{r}')}{\partial u_{\kappa\alpha}}\frac{1}{|\mathbf{r}-\mathbf{r}'|}\,\mathrm{d}\mathbf{r}' + \int \frac{\delta^2 E_{\mathrm{xc}}}{\delta n(\mathbf{r})\delta n(\mathbf{r}')}\,\frac{\partial n(\mathbf{r}')}{\partial u_{\kappa\alpha}}\,\mathrm{d}\mathbf{r}'
$$

**这是一个自洽方程**：扰动 → 密度响应 → 扰动更新 → 新的密度响应。与 Kohn-Sham 的 SCF 完全同构，只是把「密度」换成「密度响应」。因此 DFPT 也被称为「响应函数的自洽场」。<span class="marginnote">DFPT 与 SCF 的同构性是它的理论之美：所有你熟悉的 SCF 收敛技巧（混合、预条件、对称性约化）都能直接搬到响应方程上。一个写好 SCF 求解器的代码库，往往能很自然地扩展出 DFPT。</span>

## 4 公式解析：从响应到声子频率

把整条链压缩，声子频率计算的本质是「二阶导数」问题：

$$
\omega^2 = \frac{1}{M}\frac{\partial^2 E}{\partial u^2}
\;\xleftarrow{\text{一阶响}}\; \frac{\partial n}{\partial u}
\;\xleftarrow{\text{线性响应}}\; \frac{\partial V_{\mathrm{eff}}}{\partial u}
$$

分三步拆解这条链：

- **第一步，扰动源**：位移原子相当于给体系加了一个微扰 $\partial V_{\mathrm{ion}}/\partial u$。这个扰动通过 Sternheimer 方程驱动密度产生响应 $\partial n/\partial u$。
- **第二步，自洽循环**：密度响应反过来通过 Hartree 与交换关联核（$\delta^2 E_{\mathrm{xc}}/\delta n\delta n$）修正有效势扰动，再驱动新一轮响应——直到自洽。
- **第三步，组装动力学矩阵**：用收敛的响应组装力常数 $\Phi$，傅里叶变换到每个 $\mathbf{q}$，对角化得 $\omega^2(\mathbf{q})$。声子频率的平方可正可负：**负 $\omega^2$ 意味着虚频，代表该振动模式不稳定**——体系会自发沿该模式畸变，这是相变软模理论的核心信号。

**辨析｜易错点：** 声子计算里最常遇到「虚频」（imaginary frequency）。新手常以为这是代码错误，其实虚频是物理信息：它表示该结构在谐波近似下是鞍点而非局域极小。要么结构需要弛豫，要么真实相是低对称相。**判读虚频、找到对应的畸变模式，是材料相变研究的日常**。

## 5 声子谱的应用与延伸

DFPT 声子谱支撑着材料科学的半壁江山：

**热力学性质**：声子态密度 $g(\omega)$ 直接进入热容、零点能、自由能的公式：

$$
F_{\mathrm{vib}} = k_BT\int_0^\infty g(\omega)\ln\Big(2\sinh\frac{\hbar\omega}{2k_BT}\Big)\,\mathrm{d}\omega
$$

由此可算自由能随温度的变化，判断高温相 vs 低温相——这是**相图计算的第一性原理路径**。<span class="marginnote">振动自由能听起来抽象，但它决定了「这个材料在 1000 K 下哪个相更稳」。地热材料、高温合金、电池正极的相稳定性研究，都建立在这条公式上。算它需要声子谱，而 DFPT 是最常用的手段。</span>

**电声耦合与超导**：声子与电子耦合强度 $\lambda$ 进入 McMillan-Allen-Dynes 公式，可预测超导临界温度 $T_c$。MgB$_2$、氢化物高温超导体的理论预言，正是建立在 DFPT 声子谱与电声耦合矩阵元之上。

**输运与热导**：声子-声子散射率（三阶力常数）结合声子谱，可解 Boltzmann 输运方程得到晶格热导率——热电材料的筛选依赖这条链。

**辨析｜易错点：** DFPT 一阶响应给出的是**谐波**声子。温度升高、振动幅度大时，谐波近似失效，需要**非谐效应**（三阶/四阶力常数）——这通常用有限位移法的超胞计算，成本陡增。DFPT 擅长谐波，非谐则是另一场硬仗。<span class="marginnote">有限位移法（frozen phonon）与 DFPT 的对比：有限位移法在超胞里挪原子、算受力、做数值差分，实现简单、可并行但噪声大、且 $\mathbf{q}$ 点受超胞尺寸限制；DFPT 解析、精确、可算任意 $\mathbf{q}$，但实现复杂。现代代码（如 Quantum ESPRESSO、VASP、Phonopy）两种都提供。</span>

## 6 小结

- **声子** = 晶格振动的量子化集体模式，由动力学矩阵对角化给出 $\omega^2(\mathbf{q})$。
- **Hellmann-Feynman 定理**：力只依赖密度与离子势导数；力常数需要密度响应 $\partial n/\partial u$