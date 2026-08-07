---
title: 麦克斯韦方程组的微分形式
date: 2026-08-07
---

# 麦克斯韦方程组的微分形式

<div class="epigraph">
<p>把通量与环流的积分写成散度与旋度的微分——麦克斯韦方程组从「整体」变成「局域」，电磁场在每一点的规律就此立定。</p>
<footer>—— 电动力学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 郭硕鸿《电动力学》第一章 ｜ 2026-08-07</p>
</div>

## 为什么从微分形式开始

第 71 节我们写出了麦克斯韦方程组的积分形式——它描述「通量/环流」的整体性质。电动力学需要**微分形式**：用**散度（divergence）**与**旋度（curl）**表达场在每一点的局域规律。微分形式是电磁场理论的标准语言——它让「场如何由源激发」「场如何相互转化」变成一组偏微分方程，也是电磁波方程（第 69 节）推导的基础。这一节建立微分算子，把积分形式改写为微分形式。

## 1 散度、旋度与高斯、斯托克斯定理

**散度（divergence）**：$\nabla\cdot\boldsymbol{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}$——度量场的「源」强度（单位体积的通量）。

**旋度（curl）**：$\nabla\times\boldsymbol{F}$——度量场的「涡旋」强度（单位面积的环流）。

**高斯定理（散度定理）**：$\oint_S\boldsymbol{F}\cdot\mathrm{d}\boldsymbol{S} = \int_V\nabla\cdot\boldsymbol{F}\,\mathrm{d}V$——通量 = 散度体积分。

**斯托克斯定理**：$\oint_L\boldsymbol{F}\cdot\mathrm{d}\boldsymbol{l} = \int_S(\nabla\times\boldsymbol{F})\cdot\mathrm{d}\boldsymbol{S}$——环流 = 旋度面积分。

**重点：高斯定理与斯托克斯定理把「积分（整体）」与「微分（局域）」连接起来——积分形式的麦克斯韦方程组经它们可改写为微分形式。** 这两个定理是矢量分析的核心工具：通量对应散度、环流对应旋度。

## 2 麦克斯韦方程组的微分形式

用高斯与斯托克斯定理，把积分形式（第 71 节）改写为微分形式：

**① 电场高斯定理**（电场的源是电荷）：

$$\nabla\cdot\boldsymbol{D} = \rho_f$$

**② 磁场高斯定理**（无磁单极子）：

$$\nabla\cdot\boldsymbol{B} = 0$$

**③ 法拉第定律**（变化磁场产生电场）：

$$\nabla\times\boldsymbol{E} = -\frac{\partial\boldsymbol{B}}{\partial t}$$

**④ 安培-麦克斯韦定律**（电流与变化电场产生磁场）：

$$\nabla\times\boldsymbol{H} = \boldsymbol{j}_f + \frac{\partial\boldsymbol{D}}{\partial t}$$

其中 $\rho_f$ 是自由电荷密度、$\boldsymbol{j}_f$ 是传导电流密度。

**重点：微分形式把「积分 = 总量」变为「散度/旋度 = 每一点的源」。** ① 说「电场散度 = 电荷密度」（电荷是电场的源）、② 说「磁场散度恒为零」（无磁荷）、③ 说「电场旋度 = −磁场变化率」（变化磁场生电场）、④ 说「磁场旋度 = 电流 + 变化电场」（电流与位移电流生磁场）。<span class="marginnote">「微分形式 vs 积分形式」：积分形式看「整体通量/环流」，微分形式看「局域散度/旋度」。两者由高斯/斯托克斯定理等价——同一物理的两种表述。微分形式是偏微分方程组，更适合作为场论的出发点（边界条件、电磁波、辐射都从微分形式推导）。</span>

## 3 公式解析：由电荷分布求场（微分形式的使用）

已知电荷密度 $\rho(x, y, z)$，用微分形式说明如何求电场。

$$
\nabla\cdot\boldsymbol{E} = \frac{\rho}{\varepsilon_0} \quad\text{（真空）}, \qquad \nabla\times\boldsymbol{E} = 0 \quad\text{（静电场无旋）}
$$

- **第一步，静电场两个方程**：$\nabla\cdot\boldsymbol{E} = \rho/\varepsilon_0$（源方程）、$\nabla\times\boldsymbol{E} = 0$（无旋，静电场保守）。
- **第二步，无旋 ⟹ 有势**：$\nabla\times\boldsymbol{E} = 0$ ⟹ 存在标势 $\phi$ 使 $\boldsymbol{E} = -\nabla\phi$（静电势）。
- **第三步，代入源方程**：$\nabla\cdot(-\nabla\phi) = \rho/\varepsilon_0$ ⟹ $\nabla^2\phi = -\rho/\varepsilon_0$——**泊松方程**。
- **第四步，求解**：给定电荷分布，解泊松方程得电势，再取梯度得电场——静电学的微分形式路线。

**辨析｜易错点：**「无旋场必有势」的严格条件（单连通区域）与「静电场无旋」的适用边界（静电场、无变化磁场）要清楚。$\nabla\times\boldsymbol{E} = 0$ 只在静电场成立；有变化磁场时（方程③）电场有旋、不能再定义电势。

## 4 微分形式与边值问题

微分形式的麦克斯韦方程组配合**边界条件**（第 120 节），构成电磁场的边值问题——电动力学的核心求解框架：

- 介质分界面两侧场量的关系由边界条件给出（由积分形式在界面取薄盒/窄环推出）；
- 典型边值问题：导体腔、波导、谐振腔——解微分方程 + 边界条件。

**重点：微分形式 + 边界条件 = 电磁场边值问题的标准框架。** 解方程 $\nabla^2\phi = -\rho/\varepsilon_0$（泊松）或波动方程，配合界面条件（$\boldsymbol{D}$、$\boldsymbol{B}$ 法向分量与 $\boldsymbol{E}$、$\boldsymbol{H}$ 切向分量的连续/跳变），得到唯一解（唯一性定理，第 121 节）。

<span class="marginnote">「从积分到微分再到求解」是电动力学的标准路线：先写微分方程组（局域规律），再补边界条件（界面行为），最后解偏微分方程（具体问题）。第 121 节唯一性定理保证「解是唯一的」——边值问题的理论根基。这条路线也呼应第二十二章统计物理中「微分方程 + 边界条件」的求解模式。</span>

## 5 微分形式的意义

- **局域性**：方程在每一点成立，描述场与源的局域关系——比积分形式更「精细」；
- **偏微分方程组**：是电磁波的波动方程、静电场泊松方程、磁场的矢量方程的统一出发点；
- **规范场论的起点**：麦克斯韦方程组的微分形式（协变形式）是狭义相对论与量子电动力学的基础；
- **数值计算**：FDTD（时域有限差分）、有限元等电磁场数值方法都从微分形式离散化。

**重点：微分形式是电磁场理论的标准语言——它让麦克斯韦方程组成为一组局域的偏微分方程，是电磁波、辐射、边值问题的共同出发点。** 从第 71 节的「四个积分方程」到本节的「四个微分方程」，电磁学的数学语言完成了从「整体」到「局域」的升级。<span class="marginnote">「协变形式」预告：麦克斯韦方程组可以用四维时空语言写成高度对称的形式（$\partial_\mu F^{\mu\nu} = \mu_0J^\nu$）——在狭义相对论（第 17 章）下电磁场是「四维张量」，电场与磁场是同一张量的不同分量（随参考系互相转化）。这是电动力学的最高观点，也是电弱统一（第 112 节）的出发点。</span>

## 6 小结

- **微分算子**：散度 $\nabla\cdot$（源）、旋度 $\nabla\times$（涡旋）；高斯/斯托克斯定理连接积分与微分。
- **麦克斯韦方程组（微分形式）**：$\nabla\cdot\boldsymbol{D} = \rho_f$、$\nabla\cdot\boldsymbol{B} = 0$、$\nabla\times\boldsymbol{E} = -\partial\boldsymbol{B}/\partial t$、$\nabla\times\boldsymbol{H} = \boldsymbol{j}_f + \partial\boldsymbol{D}/\partial t$。
- 静电场：$\nabla\times\boldsymbol{E} = 0$ ⟹ $\boldsymbol{E} = -\nabla\phi$ ⟹ 泊松方程 $\nabla^2\phi = -\rho/\varepsilon_0$。
- **微分形式 + 边界条件 = 边值问题**（唯一性定理保证解唯一）。
- 微分形式是电磁波、辐射、数值方法、相对论协变形式的出发点。

在下一节，我们研究界面上场的连接——**电磁场的边值关系**。
