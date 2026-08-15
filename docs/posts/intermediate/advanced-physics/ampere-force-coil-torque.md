---
title: 安培力与载流线圈在磁场中的力矩
date: 2026-08-07
---

# 安培力与载流线圈在磁场中的力矩

<div class="epigraph">
<p>导线在磁场中受力，线圈在磁场中转起来——安培力把电与磁之间的对话，变成了实实在在的力与运动。</p>
<footer>—— 电磁学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 程守洙《普通物理学》第十二章 §12-5 ｜ 2026-08-07</p>
</div>

## 为什么从安培力开始

磁场对运动电荷施洛伦兹力，而对导线里的无数运动电荷合力，就是**安培力（Ampere force）**——磁场对电流（载流导线）的作用力。安培力是电动机、扬声器、电流计、电磁弹射的原动力；载流线圈在磁场中受到的**磁力矩**则让电表指针转动、让电机转子旋转。这一节从安培力公式讲起，重点推导载流线圈的磁力矩——它是电学与力学交汇的经典，也是「磁矩」概念的舞台。

## 1 安培力

电流元 $I\,\mathrm{d}\boldsymbol{l}$ 在磁场 $\boldsymbol{B}$ 中受安培力：

$$\mathrm{d}\boldsymbol{F} = I\,\mathrm{d}\boldsymbol{l}\times\boldsymbol{B}$$

有限长直导线在匀强磁场中受力：

$$\boldsymbol{F} = I\boldsymbol{l}\times\boldsymbol{B}, \qquad F = BIl\sin\theta$$

$\theta$ 是导线与磁场的夹角。方向由左手定则（或叉积）判断。<span class="marginnote">安培力是洛伦兹力的宏观表现：导线中每个运动电荷受 $q\boldsymbol{v}\times\boldsymbol{B}$，乘上单位体积的电荷数、再沿导线积分，得到 $I\mathrm{d}\boldsymbol{l}\times\boldsymbol{B}$。微观与宏观两条路殊途同归，是「运动电荷受力」与「电流受力」的统一。</span>

**重点：安培力与磁场、电流互相垂直。** $F = BIl\sin\theta$：导线与磁场平行（$\theta = 0$）时受力为零，垂直（$\theta = 90°$）时受力最大。这是判断「什么位置导线受力」的关键。

**辨析｜易错点：**安培力公式中 $l$ 是「在磁场中的有效长度」。导线在磁场外的部分不受力。弯曲导线在匀强磁场中受力，等效于「两端点连线」的长度——例如半圆导线的安培力等于直径直导线的受力。

## 2 匀强磁场中的矩形线圈

矩形线圈边长 $a$（宽）、$b$（高），电流 $I$，置于匀强磁场 $\boldsymbol{B}$ 中。设线圈法线 $\hat{\boldsymbol{n}}$ 与 $\boldsymbol{B}$ 夹角为 $\theta$。

- 竖直边（长 $b$）受力：大小 $F = BIb$，方向相反、共线（抵消），不产生力矩；
- 水平边（长 $a$）受力：大小 $F = BIa$，方向相反、不共线，形成力偶，产生力矩。

磁力矩：

$$M = BIa\cdot b\sin\theta = BIS\sin\theta$$

其中 $S = ab$ 是线圈面积。写成矢量形式，引入**磁矩（magnetic moment）** $\boldsymbol{\mu}_m = IS\hat{\boldsymbol{n}}$：

$$\boldsymbol{M} = \boldsymbol{\mu}_m\times\boldsymbol{B}, \qquad M = \mu_mB\sin\theta$$

<span class="marginnote">磁矩 $\boldsymbol{\mu}_m = IS\hat{\boldsymbol{n}}$ 描述线圈的「磁性强度」，方向由右手定则（四指沿电流、拇指指法线）。磁力矩的公式与电偶极子在电场中的力矩（$\boldsymbol{p}\times\boldsymbol{E}$）完全同构——「偶极矩 × 场」的普适结构，机械上对应「力 × 力臂」。</span>

## 3 磁力矩的应用：电动机与电流计

### 电动机

载流线圈在磁场中受磁力矩而转动。磁力矩总是驱使线圈法线转向磁场方向（$M = \mu_mB\sin\theta$，$\theta \to 0$ 时力矩为零）。**要让线圈持续转动，需要换向器（commutator）在转过平衡位置时改变电流方向**——直流电动机的核心。

**重点：磁力矩使线圈转向磁场方向（$M \propto \sin\theta$，稳定平衡在 $\theta = 0$）。** 这就像指南针（磁偶极子）指向地磁场方向——磁矩与场对齐时能量最低。

### 电流计（电表）

电流计把电流转换为指针偏转：线圈在磁场中受磁力矩 $M = NBIS\sin\theta$（$N$ 匝），游丝提供反抗力矩，指针偏转角度正比于电流。为使偏转与角度线性，电流计用**辐射状磁场**（磁力矩恒为 $NBIS$，与角度无关，仅由 $I$ 决定）。

## 4 公式解析：磁力矩的数值计算

一个 100 匝矩形线圈，面积 $S = 20\ \text{cm}^2$，电流 $I = 0.5\ \text{A}$，置于 $B = 0.2\ \text{T}$ 的匀强磁场中。求线圈法线与磁场成 $30°$ 时的磁力矩。

$$
M = NBIS\sin\theta = 100\times0.2\times0.5\times20\times10^{-4}\times\sin30°
$$

- **第一步，写磁力矩公式**：$M = NBIS\sin\theta$（$N$ 匝）。
- **第二步，单位换算**：$S = 20\ \text{cm}^2 = 20\times10^{-4}\ \text{m}^2$。
- **第三步，代入**：$M = 100\times0.2\times0.5\times20\times10^{-4}\times0.5 = 0.01\ \text{N·m}$。
- **第四步，体会**：磁力矩 = 磁矩（$NIS$）× 磁感应强度 × $\sin\theta$。磁矩 $NIS = 100\times0.5\times20\times10^{-4} = 0.1\ \text{A·m}^2$，$M = \mu_mB\sin\theta$。

## 5 磁力矩与磁矩的应用

- **电流计/电表**：磁力矩驱动指针，偏转正比于电流；
- **电动机**：磁力矩驱动转子，换向器保持持续转动；
- **扬声器**：音圈在磁隙中受力振动，电流变声音；
- **磁矩与微观**：电子轨道运动与自旋都有磁矩，原子磁矩决定材料的磁性（磁介质的磁化，下节）；
- **电磁弹射**：载流导体在磁场中受安培力加速（航母弹射器、磁悬浮列车的直线电机）。

<span class="marginnote">磁力矩公式 $\boldsymbol{M} = \boldsymbol{\mu}_m\times\boldsymbol{B}$ 是微观磁性与宏观磁化的桥梁：原子磁矩在磁场中转向排列，是铁磁材料被磁化的根源（下节《磁介质的磁化与磁场强度》）。在量子力学里，电子自旋磁矩与磁场的作用（自旋-轨道耦合、塞曼效应）更是十八、二十三章的核心。</span>

## 6 对照表：三种「×」运算与常见混淆

磁场问题里最容易混的是三种力/力矩公式的适用对象。把它们并排：

| 量 | 公式 | 适用对象 | 方向/轴线 |
| --- | --- | --- | --- |
| 安培力（直导线） | $F = BIl\sin\theta$ | 载流直导线 | 垂直于 $\boldsymbol{l}$ 与 $\boldsymbol{B}$ 所在平面 |
| 安培力（电流元） | $\mathrm{d}\boldsymbol{F} = I\,\mathrm{d}\boldsymbol{l}\times\boldsymbol{B}$ | 任意电流元 | 叉积右手定则 |
| 磁力矩（线圈） | $M = NBIS\sin\theta$ | 平面载流线圈 | 使 $\hat{\boldsymbol{n}}$ 转向 $\boldsymbol{B}$ |
| 洛伦兹力 | $F = qvB\sin\theta$ | 运动点电荷 | 垂直于 $\boldsymbol{v}$ 与 $\boldsymbol{B}$ |

**辨析｜易错点：**「安培力使导线平动、磁力矩使线圈转动」是同一物理的两面。矩形线圈竖直边所受的力虽相互抵消，但这两条边构成**力偶**——注意不要把「力偶」与「平衡力」混为一谈：力偶合力为零却产生净力矩，平衡力则既无力也无力矩。还要区分：磁力矩与线圈面积有关（$M \propto S$），安培力与导线长度有关（$F \propto l$）；对同一线圈，$N$ 匝时磁矩与力矩都放大 $N$ 倍。

**数值手感：**把上节的 100 匝线圈放进 $B = 0.5\ \text{T}$ 的磁场、电流减为 $I = 0.2\ \text{A}$、法线与磁场成 $90°$，则

$$M = NBIS\sin\theta = 100\times0.5\times0.2\times20\times10^{-4} = 0.02\ \text{N·m}$$

与上例同量级——「增大匝数」与「增大电流」在产生磁力矩上等价，这正是电流计标度的原理：把微小电流放大成可读的机械偏转。

<span class="marginnote">安培力、磁力矩、洛伦兹力同源于 $\boldsymbol{F} = q\boldsymbol{v}\times\boldsymbol{B}$：对单个电荷、对电流元积分、对线圈取力偶，只是同一叉积的三次展开。抓住源头，公式再多也不会散。</span>

## 7 术语速查

- **磁矩（magnetic moment）** $\boldsymbol{\mu}_m = NIS\hat{\boldsymbol{n}}$：线圈磁性「强度」的度量，方向由电流的右手定则确定。
- **换向器（commutator）**：直流电机在平衡位置翻转电流方向、保持单向转动的装置。
- **辐射状磁场**：电流计为使偏转与电流成正比而采用的磁场构型。
- **力偶**：大小相等、方向相反、作用线不重合的一对力，合力为零而力矩不为零。

## 7 小结

- **安培力**：$\mathrm{d}\boldsymbol{F} = I\,\mathrm{d}\boldsymbol{l}\times\boldsymbol{B}$，直导线 $F = BIl\sin\theta$；与磁场、电流垂直。
- 弯曲导线有效长度 = 两端点连线；磁场外部分不受力。
- **磁力矩**：$M = NBIS\sin\theta$，矢量形式 $\boldsymbol{M} = \boldsymbol{\mu}_m\times\boldsymbol{B}$。
- **磁矩**：$\boldsymbol{\mu}_m = NIS\hat{\boldsymbol{n}}$，方向右手定则。
- 应用：电动机（换向器）、电流计（辐射状磁场）、扬声器、电磁弹射。
- 磁力矩与电偶极矩力矩同构：$\boldsymbol{p}\times\boldsymbol{E}$ ↔ $\boldsymbol{\mu}_m\times\boldsymbol{B}$。

在下一节，我们研究磁介质——**磁介质的磁化与磁场强度**，看物质如何在磁场中响应。
