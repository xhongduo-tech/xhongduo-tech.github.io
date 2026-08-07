---
title: 平面向量的数量积
date: 2026-08-07
---

# 平面向量的数量积

<div class="epigraph">
<p>做功不是看你用了多大力，而是看力在运动方向上起了多大作用。</p>
<footer>—— 物理直觉（功的向量定义）</footer>
</div>

<div class="article-byline">
<p>第一级 · 基础数学 ｜ 人教A版 必修第二册 §6.2.4 ｜ 2026-08-07</p>
</div>

## 为什么从数量积开始

前面学的加法、数乘，两个向量运算的结果还是向量。本节引入的**数量积（dot product）**第一次改变了这个局面：两个向量相乘，结果是一个**数**。为什么需要这种运算？看物理里的做功：用力 $F$ 推箱子沿水平方向走 $s$，做功不是 $|F||s|$，而是**只把力在运动方向上的分量算进去**——$W=|F||s|\cos\theta$，其中 $\theta$ 是力与位移的夹角。斜着拉，只有「往前的那部分力」在做功。<span class="marginnote">「投影到方向上来度量贡献」是数量积的灵魂。这个词会一路延伸：在几何里是点到直线距离、在机器学习里是相似度与注意力分数。今天理解 $|F||s|\cos\theta$，等于提前解锁了大模型里「两个向量有多像」的算法。</span>

## 1 向量的夹角与数量积的定义

**向量的夹角**：已知非零向量 $\vec{a},\vec{b}$，作 $\overrightarrow{OA}=\vec{a}$、$\overrightarrow{OB}=\vec{b}$，则 $\angle AOB=\theta$（$0\le\theta\le\pi$）叫作向量 $\vec{a}$ 与 $\vec{b}$ 的夹角。<span class="marginnote">夹角范围固定在 $[0,\pi]$，这是约定：方向相反夹角是 $\pi$，不是 $-\pi$。角的概念一旦规定范围，后续「垂直」「共线」等判定才不会出现两义性。</span> 当 $\theta=\frac{\pi}{2}$ 时，称 $\vec{a}$ 与 $\vec{b}$ **垂直**，记作 $\vec{a}\perp\vec{b}$。

**数量积（内积）**：已知两个非零向量 $\vec{a},\vec{b}$，它们的夹角为 $\theta$，则

$$
\vec{a}\cdot\vec{b}=|\vec{a}|\,|\vec{b}|\,\cos\theta
$$

叫做 $\vec{a}$ 与 $\vec{b}$ 的**数量积**（或内积），读作「$\vec{a}$ 点乘 $\vec{b}$」。结果是一个实数。规定零向量与任何向量的数量积为 0。

**重点：$\vec{a}\cdot\vec{b}$ 是一个数，不是向量。** 它的符号由 $\cos\theta$ 决定：$\theta$ 为锐角时为正，为钝角时为负，垂直时为零。两个向量「相乘」得到的是标量——这正是它与数乘的根本区别。

## 2 投影：数量积的几何意义

向量 $\vec{b}$ 在向量 $\vec{a}$ 方向上的投影数量为 $|\vec{b}|\cos\theta$，于是数量积可写成：

$$
\vec{a}\cdot\vec{b}=|\vec{a}|\cdot\left(|\vec{b}|\cos\theta\right)
$$

即**「$\vec{a}$ 的长度 × $\vec{b}$ 在 $\vec{a}$ 方向上的投影」**。反过来也成立：$\vec{a}\cdot\vec{b}=|\vec{b}|\cdot(|\vec{a}|\cos\theta)$。<span class="marginnote">投影（projection）的原义是「投下的影子」：把 $\vec{b}$ 垂直投射到 $\vec{a}$ 所在直线上，影子长就是 $|\vec{b}|\cos\theta$。影子可正可负——投影落在 $\vec{a}$ 的反方向时取负值。</span> 这个几何意义让数量积的「正负」变得直观：两个向量方向接近（夹角小），投影大，点乘为正；方向相斥（夹角大），点乘为负；方向垂直，投影为零。

## 3 数量积的性质与运算律

由定义可以直接推出四条**重要性质**（设 $\vec{a},\vec{b}$ 为非零向量）：

1. **垂直判定**：$\vec{a}\perp\vec{b} \iff \vec{a}\cdot\vec{b}=0$。
2. **同向与反向**：$\vec{a}$ 与 $\vec{b}$ 同向时 $\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|$，反向时 $\vec{a}\cdot\vec{b}=-|\vec{a}||\vec{b}|$。
3. **模长公式**：$\vec{a}\cdot\vec{a}=|\vec{a}|^2$，即 $|\vec{a}|=\sqrt{\vec{a}\cdot\vec{a}}$。
4. **夹角的计算**：$\cos\theta=\dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}|\,|\vec{b}|}$。

运算律方面，数量积满足**交换律**、**数乘结合律**与**分配律**：

$$
\vec{a}\cdot\vec{b}=\vec{b}\cdot\vec{a}, \qquad
(\lambda\vec{a})\cdot\vec{b}=\lambda(\vec{a}\cdot\vec{b}), \qquad
(\vec{a}+\vec{b})\cdot\vec{c}=\vec{a}\cdot\vec{c}+\vec{b}\cdot\vec{c}
$$

**辨析｜易错点：** 数量积虽满足分配律，却**不满足消去律**，也**没有结合律**。由 $\vec{a}\cdot\vec{b}=\vec{a}\cdot\vec{c}$ 推不出 $\vec{b}=\vec{c}$——因为 $\vec{a}\cdot(\vec{b}-\vec{c})=0$ 只能说明 $\vec{a}$ 与 $\vec{b}-\vec{c}$ 垂直。而 $\vec{a}\cdot\vec{b}\cdot\vec{c}$ 根本无意义：$\vec{a}\cdot\vec{b}$ 是数，数再点乘向量没有定义。**凡是套用「乘法可消去、可结合」习惯的人，在这里都会栽跟头**——分配律是它唯一的「加法乘法混合」通道。<span class="marginnote">为什么？因为数量积把一个向量「压」成了一个数，丢失了方向信息——两个不同向量在同一方向上的投影可能相等，消去律自然失效。信息丢失是「降维」运算的共同代价。</span>

## 4 公式解析：$\cos\theta=\dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}|\,|\vec{b}|}$

这是数量积最值钱的一条应用：**用坐标与运算反求夹角**。拆三步理解：

- **第一步，从定义反解**：定义式 $\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|\cos\theta$ 两边除以 $|\vec{a}||\vec{b}|$，得 $\cos\theta=\dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$。它只是定义的变形，却把「几何角」换成了「代数式」。
- **第二步，用坐标落地**：若 $\vec{a}=(x_1,y_1)$、$\vec{b}=(x_2,y_2)$，则 $\vec{a}\cdot\vec{b}=x_1x_2+y_1y_2$（这个坐标公式我们将在坐标表示一节严格推导）。代入得 $\cos\theta=\dfrac{x_1x_2+y_1y_2}{\sqrt{x_1^2+y_1^2}\sqrt{x_2^2+y_2^2}}$——**夹角从此可以纯代数计算**。
- **第三步，特殊的取值**：$\theta=0$ 时 $\cos\theta=1$，即两向量同向且成比例；$\theta=\pi$ 时 $\cos\theta=-1$，反向；$\theta=\frac{\pi}{2}$ 时分子为 0。这个公式把几何里的「角」与代数里的「坐标」焊在一起——解析几何的求角问题从此都归它管。

## 5 数量积与物理：做功

回到开头：恒力做功

$$
W=\vec{F}\cdot\vec{s}=|\vec{F}|\,|\vec{s}|\,\cos\theta
$$

力的方向与位移方向一致时，$\cos\theta=1$，做功最大；力与位移垂直时做功为零——比如手提箱子水平行走，向上的支持力并不做功。<span class="marginnote">这个物理例子点出数量积的一个深刻侧面：它度量的是「一个向量在另一个向量方向上的有效贡献」。这个概念在机器学习里被称作「相似度」：两个词向量点乘越大，语义越接近；注意力机制里的打分，本质上也是点乘。一门数学，从做功到 AI，共用同一个运算。</span> 高中阶段，数量积是解三角形、判定垂直、求长度与夹角的核心工具；到了大学与 AI 时代，它是线性代数的内积、是神经网络的前向传播里最基本的计算单元。

## 6 小结

- **数量积** $\vec{a}\cdot\vec{b}=|\vec{a}||\vec{b}|\cos\theta$，结果是一个**数**；夹角范围 $[0,\pi]$。
- 几何意义：$|\vec{a}|$ × $\vec{b}$ 在 $\vec{a}$ 方向上的投影；符号由夹角的锐钝决定。
- 性质：**垂直 $\Leftrightarrow$ 点乘为 0**；$\vec{a}\cdot\vec{a}=|\vec{a}|^2$；$\cos\theta=\dfrac{\vec{a}\cdot\vec{b}}{|\vec{a}||\vec{b}|}$。
- 满足交换律、分配律；**不满足消去律，也没有结合律**——点乘降维丢失方向信息。
- 物理意义：做功 $W=\vec{F}\cdot\vec{s}$；推广到高维即「相似度」与注意力打分。

在下一节，我们将回答一个根本问题：平面里的任意向量，能不能只用两个固定的向量拼出来？这就是**平面向量基本定理**——向量世界的「坐标系」。
