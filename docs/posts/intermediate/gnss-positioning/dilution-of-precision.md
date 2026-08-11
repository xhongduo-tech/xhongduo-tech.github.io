---
title: 精度几何因子（DOP）与定位精度评估
date: 2026-08-11
---

# 精度几何因子（DOP）与定位精度评估

<div class="epigraph">
<p>上帝恒以几何化事。</p>
<footer>—— 柏拉图（Plato），语出普鲁塔克《宴会丛谈》</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · GNSS 定位与导航 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 DOP 开始

第 4 篇的最小二乘解里有一行字值得停下来：$\hat{\mathbf{x}} = (\mathbf{H}^{\mathrm{T}}\mathbf{H})^{-1}\mathbf{H}^{\mathrm{T}}\Delta\boldsymbol{\rho}$。解算软件明明给出的是「一个点」，但用户真正关心的是「这个点到底准不准」。答案藏在 $(\mathbf{H}^{\mathrm{T}}\mathbf{H})^{-1}$ 里——它描述的是**卫星在天上的几何排布如何把观测误差放大成位置误差**。

**精度几何因子（Dilution of Precision，DOP）**就是把这层放大关系提炼成几个可读的标量：PDOP、HDOP、VDOP、TDOP、GDOP。这一篇从协方差传播推导 DOP 的定义，解释它为什么是「几何决定的上限」，并给出工程上的使用规范——任何 GNSS 接收机都会在屏幕上显示 HDOP/PDOP，看懂它是看懂定位质量的第一步。<span class="marginnote">DOP 的数学是第二级《概率论与数理统计》中协方差传播律的直接应用：观测向量的方差经过线性变换 $\mathbf{A} = (\mathbf{H}^{\mathrm{T}}\mathbf{H})^{-1}\mathbf{H}^{\mathrm{T}}$ 变成状态估计的方差——线性代数与统计在这里握手。</span>

## 1 从观测误差到位置误差

第 4 篇的最小二乘解可写成 $\hat{\mathbf{x}} = \mathbf{A}\,\Delta\boldsymbol{\rho}$，其中 $\mathbf{A} = (\mathbf{H}^{\mathrm{T}}\mathbf{H})^{-1}\mathbf{H}^{\mathrm{T}}$。若观测误差的方差为 $\sigma_{UERE}^2$（第 5 篇的 UERE，即等效距离误差），由协方差传播律，解的状态协方差为

$$\mathbf{Q}_{\hat{x}} = \sigma_{UERE}^2 \, (\mathbf{H}^{\mathrm{T}}\mathbf{H})^{-1}$$

**重点：定位误差 = 观测质量（UERE）× 几何放大（$(\mathbf{H}^{\mathrm{T}}\mathbf{H})^{-1}$）。** 前者是接收机与环境的属性，后者纯粹是几何属性——DOP 就是几何放大量的标量化。

## 2 公式解析：DOP 家族的定义

设协方差矩阵 $(\mathbf{H}^{\mathrm{T}}\mathbf{H})^{-1}$ 的四个对角元分别对应 $x, y, z$（位置）与 $c\,\delta t_u$（时间），记为 $q_{11}, q_{22}, q_{33}, q_{44}$，则

$$\text{GDOP} = \sqrt{q_{11} + q_{22} + q_{33} + q_{44}}$$

$$\text{PDOP} = \sqrt{q_{11} + q_{22} + q_{33}}, \qquad \text{TDOP} = \sqrt{q_{44}}$$

$$\text{HDOP} = \sqrt{q_{11} + q_{22}}, \qquad \text{VDOP} = \sqrt{q_{33}}$$

逐步拆解：

- **第一步，看 $\mathbf{H}$ 的构成**：$\mathbf{H}$ 的每一行是 $(-\mathbf{u}_i^{\mathrm{T}},\ 1)$，$\mathbf{u}_i$ 是视线单位向量。$\mathbf{H}^{\mathrm{T}}\mathbf{H}$ 的几何信息全部来自这些视线向量的**方向分布**。
- **第二步，看对角元**：$\mathbf{H}^{\mathrm{T}}\mathbf{H}$ 可逆时，其逆的对角元越大，对应方向的位置方差越大。卫星挤在一起（视线近平行）时，$\mathbf{H}^{\mathrm{T}}\mathbf{H}$ 近奇异，逆矩阵元素爆炸——DOP 变大。
- **第三步，读家族关系**：GDOP 是全部四个分量的总放大；PDOP 只管三维位置；HDOP/VDOP 把位置拆成水平与垂直；TDOP 只管钟差。自然有

$$\text{GDOP}^2 = \text{PDOP}^2 + \text{TDOP}^2, \qquad \text{PDOP}^2 = \text{HDOP}^2 + \text{VDOP}^2$$

- **第四步，乘上 UERE 得绝对误差**：$\sigma_{pos} = \text{PDOP} \times \sigma_{UERE}$。UERE 是 5 m、PDOP 是 2，那么三维位置误差约 10 m——DOP 把「精度」从「测量精度」翻译成「定位精度」。

**易错点：** DOP 是**纯几何量**，不包含任何观测噪声信息。两台接收机在同一时刻同一地点，DOP 完全相同——即使一台是廉价手机、一台是大地测量接收机。DOP 告诉你的是「若观测误差为 1 m，位置误差会放大几倍」，而不是「现在误差是几米」。

## 3 几何直觉：四面体与天空分布

为什么分散的卫星更好？想象每颗卫星的视线是一条从接收机射向天空的射线。当射线**向四周散得很开**时，任意方向的位置变化都会被某颗卫星「看见」，解在三维空间被充分约束——相当于一个体积大的**四面体**框住了接收机。

当卫星**挤在天区的一小块**时，所有视线近乎平行，沿视线方向的约束强，但**垂直方向几乎没有约束**——解被「拍扁」成一个薄片，不确定性沿某个方向被拉得很长。

![好几何与差几何的 DOP 对比](/images/gnss-positioning/dilution-of-precision-1.svg)

图里左图四颗卫星均匀分布（几何好、PDOP 小），右图四颗卫星挤在一角（几何差、PDOP 大）。这就是「卫星分布越散、DOP 越小」的直观来源。<span class="marginnote">顺带一个高度相关的细节：<strong>VDOP 几乎总大于 HDOP</strong>。因为所有卫星都在接收机上方，垂直方向永远少一维约束——这是 GNSS 高程精度天然比平面差的几何根源。</span>

## 4 工程上的 DOP 使用规范

- **阈值**：航空与测绘规范通常要求 **PDOP < 6**；更严格场景（如精密进近）要求 HDOP < 2、VDOP < 3。PDOP > 10 通常视为几何不可用。
- **随时间的演变**：卫星在天空移动，DOP 随时间起伏。接收机与软件会做「星座选择」，只采用使 DOP 最小的卫星子集——这一优化正是第 10 篇多星座融合的价值所在。
- **加权修正**：真实解算常对低仰角卫星降权（第 4 篇），此时「有效 DOP」不是从 $\mathbf{H}$ 而是从加权 $\mathbf{H}^{\mathrm{T}}\mathbf{W}\mathbf{H}$ 计算。高度角截断既能降噪声，也可能牺牲几何——两者要权衡。
- **skyplot**：接收机软件里的「天空图」把各卫星的仰角/方位角画在圆盘上——一眼就能看出此刻的几何好坏，是野外作业判断定位质量的标准工具。

## 5 DOP 与误差预算的合流

把第 5 篇的 UERE 与这一篇的 DOP 合起来，就得到定位精度的完整表达式：

$$\sigma_{pos} = \text{DOP} \times \sigma_{UERE}$$

- 误差预算回答了「观测本身多准」（UERE）；
- DOP 回答了「观测的准被几何放大几倍」。

两条腿缺一不可：一颗星的观测再准，几何差时照样定位不可靠；几何再好，UERE 差时精度也上不去。这正是所有差分/PPP 技术（第 7、8 篇）与所有多星座方案（第 10 篇）最终都要回到的这个公式。

## 6 小结

- **DOP = 几何对观测误差的放大倍数**，来自 $\mathbf{Q}_{\hat{x}} = \sigma_{UERE}^2(\mathbf{H}^{\mathrm{T}}\mathbf{H})^{-1}$。
- DOP 家族：GDOP / PDOP / HDOP / VDOP / TDOP，满足 $\text{GDOP}^2 = \text{PDOP}^2 + \text{TDOP}^2$。
- 定位误差 = **DOP × UERE**：一条腿是观测质量，一条腿是几何质量。
- DOP 是**纯几何量**，与接收机质量无关；卫星越分散、DOP 越小。
- **VDOP 通常大于 HDOP**（卫星都在上方），工程规范常要求 PDOP < 6。
- 加权与星座选择会改变有效 DOP，需在「几何收益」与「噪声代价」间权衡。

单系统能提供 4–10 颗可见卫星，DOP 常常是瓶颈。下一节，我们把四个星座的卫星全部请进方程：**多星座与多频定位**。
