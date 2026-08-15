---
title: 热力学基本方程与麦克斯韦关系
date: 2026-08-07
---

# 热力学基本方程与麦克斯韦关系

<div class="epigraph">
<p>五个热力学函数、一条基本方程、四组麦克斯韦关系——热力学的全部解析结构，浓缩在一页纸上。</p>
<footer>—— 热力学与统计物理引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 汪志诚《热力学·统计物理》第二～三章 ｜ 2026-08-07</p>
</div>

## 为什么从热力学基本方程开始

第九章我们学习了热力学第一、第二定律，但还没有把它们的微分形式「组装」起来。**热力学基本方程（fundamental equation）**把内能写成状态函数的全微分 $\mathrm{d}U = T\mathrm{d}S - p\mathrm{d}V$——它把第一定律与第二定律合成一个微分关系。由此定义**热力学势（亥姆霍兹自由能、吉布斯自由能、焓）**，并推出**麦克斯韦关系（Maxwell relations）**——把不可测量的偏导（熵对体积）换成可测量的偏导（压强对温度）。这一节是热力学从「定律」走向「计算工具」的关键。

## 1 热力学基本方程

由第一定律 $\mathrm{d}U = \delta Q - p\mathrm{d}V$ 与第二定律（可逆过程 $\delta Q = T\mathrm{d}S$）：

$$\mathrm{d}U = T\mathrm{d}S - p\mathrm{d}V$$

**热力学基本方程**：内能是熵与体积的函数 $U(S, V)$，其全微分为上式。

由全微分条件（$\mathrm{d}U$ 是恰当微分，交叉偏导相等）：

$$T = \left(\frac{\partial U}{\partial S}\right)_V, \qquad -p = \left(\frac{\partial U}{\partial V}\right)_S$$

**重点：热力学基本方程 $\mathrm{d}U = T\mathrm{d}S - p\mathrm{d}V$——第一、第二定律的联合微分形式，$U$ 以 $(S, V)$ 为自然变量。** 它把两个不可直接测量的量（熵、内能）与可测量的量（$T$、$p$、$V$）联系起来。<span class="marginnote">「自然变量」的概念：$U$ 最自然的自变量是 $S$ 与 $V$（因为它们直接出现在全微分里）。换到别的变量（如 $T$、$p$）需要勒让德变换——这定义了焓、自由能等新热力学函数。每个热力学势有其「自然变量」，在相应条件下最方便。</span>

## 2 热力学势

由勒让德变换定义四个热力学函数：

**内能** $U(S, V)$：$\mathrm{d}U = T\mathrm{d}S - p\mathrm{d}V$；
**焓** $H = U + pV$：$\mathrm{d}H = T\mathrm{d}S + V\mathrm{d}p$（自然变量 $S, p$，等压过程热）；
**亥姆霍兹自由能** $F = U - TS$：$\mathrm{d}F = -S\mathrm{d}T - p\mathrm{d}V$（自然变量 $T, V$，等温等容）；
**吉布斯自由能** $G = U - TS + pV$：$\mathrm{d}G = -S\mathrm{d}T + V\mathrm{d}p$（自然变量 $T, p$，等温等压）。

**重点：四个热力学势——$U$、$H$、$F$、$G$——各有自然变量与全微分；$F$（等温等容）、$G$（等温等压）是「自由能」，在对应条件下取极小（平衡判据）。** 相变（下节）、化学平衡用 $G$；等温过程用 $F$；等压热用 $H$。<span class="marginnote">「自由能为什么叫自由」：$F$、$G$ 是在等温条件下「可以做功的那部分能量」——总内能减去「束缚在热能 $TS$」的部分。等温等压下系统朝 $G$ 最小的方向演化——平衡判据。相变（固→液→气）就是在 $T$、$p$ 下比较两相的 $G$，低的相稳定。</span>

**四个热力学势速查表**：

| 势 | 定义 | 自然变量 | 全微分 | 适用条件 |
| --- | --- | --- | --- | --- |
| 内能 $U$ | — | $(S, V)$ | $T\mathrm{d}S - p\mathrm{d}V$ | 孤立/绝热过程 |
| 焓 $H$ | $U + pV$ | $(S, p)$ | $T\mathrm{d}S + V\mathrm{d}p$ | 等压过程的热量 |
| 亥姆霍兹 $F$ | $U - TS$ | $(T, V)$ | $-S\mathrm{d}T - p\mathrm{d}V$ | 等温等容 |
| 吉布斯 $G$ | $U - TS + pV$ | $(T, p)$ | $-S\mathrm{d}T + V\mathrm{d}p$ | 等温等压 |

注意「自然变量」的规律：换一对变量就用一次勒让德变换，同时「翻转」一个状态函数（$U \to H$ 换 $V \leftrightarrow p$，$U \to F$ 换 $S \leftrightarrow T$）。

## 3 麦克斯韦关系

由热力学势的全微分「交叉偏导相等」（如 $\frac{\partial}{\partial V}(\frac{\partial U}{\partial S}) = \frac{\partial}{\partial S}(\frac{\partial U}{\partial V})$），得**麦克斯韦关系**：

$$\left(\frac{\partial T}{\partial V}\right)_S = -\left(\frac{\partial p}{\partial S}\right)_V, \qquad \left(\frac{\partial T}{\partial p}\right)_S = \left(\frac{\partial V}{\partial S}\right)_p$$

$$\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial p}{\partial T}\right)_V, \qquad \left(\frac{\partial S}{\partial p}\right)_T = -\left(\frac{\partial V}{\partial T}\right)_p$$

**重点：麦克斯韦关系把「含熵的偏导」（难测）换成「含 $p$、$V$、$T$ 的偏导」（可测）——是热力学计算的桥梁。** 例如第 3 式把 $(\partial S/\partial V)_T$ 换成 $(\partial p/\partial T)_V$，后者可由状态方程直接求。<span class="marginnote">「麦克斯韦关系的用途」：理想气体 $pV = nRT$ 给 $(\partial p/\partial T)_V = nR/V$，于是 $(\partial S/\partial V)_T = nR/V$——熵对体积的依赖由状态方程得到。麦克斯韦关系让「不可测的熵变」变成「可测的压强-温度关系」，是推导热力学恒等式（内能只与温度有关、Cp-Cv 关系）的核心工具。</span>

**麦克斯韦关系一览表（记忆辅助）**：

| 序号 | 含熵偏导（难测） | 可测替代 | 来源势 |
| --- | --- | --- | --- |
| 1 | $(\partial T/\partial V)_S$ | $-(\partial p/\partial S)_V$ | $U(S, V)$ |
| 2 | $(\partial T/\partial p)_S$ | $(\partial V/\partial S)_p$ | $H(S, p)$ |
| 3 | $(\partial S/\partial V)_T$ | $(\partial p/\partial T)_V$ | $F(T, V)$ |
| 4 | $(\partial S/\partial p)_T$ | $-(\partial V/\partial T)_p$ | $G(T, p)$ |

记忆要点：**前两式来自含 $S$ 的势（$U$、$H$），后两式来自含 $T$ 的势（$F$、$G$）**。最常用的是第 3 式 $(\partial S/\partial V)_T = (\partial p/\partial T)_V$——因为 $(\partial p/\partial T)_V$ 只需状态方程就能求。推导比硬记可靠：取任一热力学势的全微分，令交叉偏导相等，符号自然正确。

## 4 公式解析：用麦克斯韦关系求内能

证明理想气体内能只与温度有关（与体积无关），用麦克斯韦关系。

$$
\left(\frac{\partial U}{\partial V}\right)_T = T\left(\frac{\partial p}{\partial T}\right)_V - p = T\cdot\frac{nR}{V} - \frac{nRT}{V} = 0
$$

- **第一步，写 $\mathrm{d}U$ 在 $(T, V)$ 变量下的形式**：$\mathrm{d}U = T\mathrm{d}S - p\mathrm{d}V$，把 $\mathrm{d}S = (\frac{\partial S}{\partial T})_V\mathrm{d}T + (\frac{\partial S}{\partial V})_T\mathrm{d}V$ 代入。
- **第二步，提取 $(\partial U/\partial V)_T$**：$\left(\frac{\partial U}{\partial V}\right)_T = T\left(\frac{\partial S}{\partial V}\right)_T - p$。
- **第三步，用麦克斯韦关系**：$\left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial p}{\partial T}\right)_V$，代入。
- **第四步，理想气体**：$\left(\frac{\partial p}{\partial T}\right)_V = \frac{nR}{V}$，$p = \frac{nRT}{V}$，所以 $(\partial U/\partial V)_T = T\cdot\frac{nR}{V} - \frac{nRT}{V} = 0$——内能与体积无关，只由温度决定（验证了第 32 节的结论）。

**辨析｜易错点：**麦克斯韦关系的四式「符号与变量」容易记混。记忆法：每式把「熵对某变量」换成「另一势的交叉偏导」；含 $T$ 的两个式子是「$S$ 对 $V$ ↔ $p$ 对 $T$」（正号）、「$S$ 对 $p$ ↔ $V$ 对 $T$」（负号）。推导比记忆可靠：从热力学势的全微分 + 交叉偏导相等一步步推，符号自然正确。

**数值算例（迈耶关系 $C_p - C_V$）**：用热力学恒等式 $C_p - C_V = T\left(\frac{\partial p}{\partial T}\right)_V\left(\frac{\partial V}{\partial T}\right)_p$。对 1 mol 理想气体：$(\partial p/\partial T)_V = R/V$，$(\partial V/\partial T)_p = R/p$，于是 $C_p - C_V = T\cdot\frac{R}{V}\cdot\frac{R}{p} = \frac{RT}{V}\cdot\frac{R}{p}$。代入理想气体 $pV = RT$ 得 $C_p - C_V = p\cdot\frac{R}{p} = R$——正是熟知的**迈耶关系**。对非理想气体 $C_p - C_V \neq R$，偏差反映分子间相互作用；麦克斯韦关系把「热容差值」化为「状态方程可测的偏导」，这正是它的工程价值（查表、算热容、算熵变）。

## 5 热力学势与平衡判据

各热力学势在相应约束下给出平衡判据：

| 约束条件 | 判据势 | 平衡条件 |
| --- | --- | --- |
| 孤立系统（$U,V$ 固定） | 熵 $S$ | $S$ 最大（熵增加原理） |
| 等温等容（$T,V$ 固定） | 亥姆霍兹 $F$ | $F$ 最小 |
| 等温等压（$T,p$ 固定） | 吉布斯 $G$ | $G$ 最小 |

**重点：不同约束下系统平衡由不同势的极值判定——熵最大（孤立）、$F$ 最小（等温等容）、$G$ 最小（等温等压）。** 这是「热力学第二定律的势表述」：系统的自发演化方向是「势减小」方向。相变与化学平衡的分析都以 $G$ 最小为基础。<span class="marginnote">「判据的统一」：熵增原理（孤立系统）、自由能最小（等温系统）都是第二定律在不同约束下的表述。化学反应的进行方向（朝 $G$ 减小的方向）、两相共存（两相 $G$ 相等）都由 $G$ 判据决定——下一节相变将用吉布斯自由能分析。</span>

## 6 自然变量与勒让德变换的直觉

**辨析｜易错点：为什么 $U$ 不用 $(T, V)$ 而用 $(S, V)$ 做自变量？** 内能的自然变量是 $S$、$V$，因为基本方程 $\mathrm{d}U = T\mathrm{d}S - p\mathrm{d}V$ 中 $S$、$V$ 是「独立微分」的变量，$T$、$p$ 只是它们的系数。想换成可测的 $(T, V)$ 做自变量，就用**勒让德变换** $F = U - TS$：它把「$U$ 对 $S$ 的斜率（=$T$）」从函数里「剥出来」，变成新函数的自变量。

**几何直觉**：$U(S, V)$ 是 $(S, V, U)$ 空间中的曲面；$F(T, V)$ 是固定 $V$ 时 $U$ 曲面在斜率 $T$ 处的切平面截距——所以 $F$ 又叫「切平面截距变换」。勒让德变换在分析力学中同样出现（哈密顿量 $H$ 是拉格朗日量 $L$ 对广义速度 $\dot{q}$ 的勒让德变换）——「同一能量函数换一套自变量」，是贯穿热力学与分析力学的一把手术刀。<span class="marginnote">「四条线索」：热力学势（$U,H,F,G$）、麦克斯韦关系、勒让德变换、自然变量——这四者构成热力学「解析结构」的完整拼图。统计物理中 $F = -kT\ln Z$（配分函数 → 自由能）再次接上这条线索：$F$ 是统计与热力学的接口，也是「从极限到大模型」统计物理章节的核心桥。本专题下一章《热力学与统计物理》将系统展开。</span>

## 7 小结

- **热力学基本方程**：$\mathrm{d}U = T\mathrm{d}S - p\mathrm{d}V$——第一 + 第二定律的微分联合。
- **热力学势**：$U(S,V)$、$H(S,p)$、$F(T,V)$、$G(T,p)$；$F$、$G$ 是自由能（等温可用能）。
- **麦克斯韦关系**：四式，把熵偏导换成可测的 $p$、$V$、$T$ 偏导。
- 应用：证明理想气体内能只与温度有关、$C_p - C_V = R$（迈耶关系）、熵变计算。
- **勒让德变换**：换一套自然变量即换一个热力学势；$F$ 是 $U$ 的切平面截距。
- **平衡判据**：孤立 $S$ 最大、等温等容 $F$ 最小、等温等压 $G$ 最小。
- 相变与化学平衡都归结为 $G$ 的比较与极小化。

在下一节，我们用吉布斯自由能分析——**相平衡与相变初步**。
