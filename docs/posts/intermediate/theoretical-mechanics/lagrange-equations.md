---
title: 拉格朗日方程的建立
date: 2026-08-07
---

# 拉格朗日方程的建立

<div class="epigraph">
<p>给我一个拉格朗日函数，我就能写出整个宇宙的运动方程。</p>
<footer>—— 约瑟夫-路易 · 拉格朗日（Joseph-Louis Lagrange）</footer>
</div>

<div class="article-byline">
<p>第二级 · 理论力学 ｜ 周衍柏《理论力学》第五章 ｜ 2026-08-07</p>
</div>

## 为什么从拉格朗日方程开始

到达朗贝尔原理，动力学已能「以静化动」，但方程还是写在真实坐标上的——每条约束都要被虚位移原理悄悄处理，推导仍然繁琐。拉格朗日 1788 年迈出了决定性一步：**把动力学普遍方程改写进广义坐标**，得到一组对任何完整约束系统都普遍成立的方程。这组方程不问力、不问约束，只需要一个标量函数——**拉格朗日函数**。<span class="marginnote">拉格朗日方程是分析力学的皇冠：<strong>整个系统的动力学被压缩进一个函数 $L = T - V$</strong>，运动方程由 $L$ 自动「生成」。它对任何广义坐标都成立，形式与坐标选择无关——这种「一个函数生成全部方程」的范式，后来被量子力学（作用量）、场论（拉格朗日密度）全面继承。</span>

从「从极限到大模型」主线看，拉格朗日方程是「能量语言」对「力语言」的彻底胜利：不再需要画受力图、不必理会约束反力，只写动能与势能。

## 1 从动力学普遍方程到拉格朗日方程

设系统有 $s$ 个自由度，广义坐标 $q_1, \ldots, q_s$，动能为 $T$，广义力为 $Q_k$。从动力学普遍方程出发，把虚功项改写到广义坐标：

$$
\sum_{k=1}^{s} \left(Q_k - \frac{d}{dt}\frac{\partial T}{\partial \dot{q}_k} + \frac{\partial T}{\partial q_k}\right)\delta q_k = 0
$$

由于 $\delta q_k$ 相互独立，括号内每一项都必须为零：

$$
\frac{d}{dt}\frac{\partial T}{\partial \dot{q}_k} - \frac{\partial T}{\partial q_k} = Q_k, \qquad k = 1, 2, \ldots, s
$$

这就是**第二类拉格朗日方程**（含广义力的形式）。推导的关键在于把 $\frac{\partial T}{\partial \dot{q}_k}$ 与 $\frac{\partial T}{\partial q_k}$ 通过链式法则与变分换序巧妙地组织起来。<span class="marginnote">推导中有一个漂亮的恒等式：$\frac{\partial\vec{r}_i}{\partial q_k} = \frac{\partial\dot{\vec{r}}_i}{\partial\dot{q}_k}$——「位矢对广义坐标的偏导」等于「速度对广义速度的偏导」。它把动能对广义速度的偏导与虚功的系数联系起来，是整个推导的钥匙。</span>

## 2 保守系统：拉格朗日函数

当所有主动力都是保守力时，广义力 $Q_k = -\partial V/\partial q_k$，代入拉格朗日方程：

$$
\frac{d}{dt}\frac{\partial T}{\partial \dot{q}_k} - \frac{\partial T}{\partial q_k} = -\frac{\partial V}{\partial q_k}
$$

定义**拉格朗日函数（Lagrangian）**：

$$
L = T - V
$$

则方程统一写成**标准拉格朗日方程**：

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_k} - \frac{\partial L}{\partial q_k} = 0, \qquad k = 1, 2, \ldots, s
$$

**整个动力学系统只需一个标量函数 $L = T - V$ 就能完整描述。**<span class="marginnote">为什么是 $T - V$ 而不是 $T + V$？从推导看，势能项进入方程时带负号（$Q_k = -\partial V/\partial q_k$），于是 $L = T - V$ 让「$T$ 对广义坐标的偏导」与「$V$ 对广义坐标的偏导」在方程里恰好同号。这个「$-V$」约定不是拍脑袋，而是从保守力定义里长出来的。</span>

## 3 拉格朗日方程的应用步骤

用拉格朗日方程解题有一套固定流程：

1. **确定自由度** $s$，选取广义坐标 $q_1, \ldots, q_s$；
2. **写出动能** $T(q, \dot{q})$ 与势能 $V(q)$，构造 $L = T - V$；
3. **逐坐标代入**标准方程 $\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_k} - \frac{\partial L}{\partial q_k} = 0$；
4. 解出运动方程，必要时加初始条件求解。

<span class="marginnote">流程的每一步都是「照方抓药」，不画受力图、不列约束方程——这正是分析力学的解放之处。任何复杂机构，只要写出 $L$，方程自动出来。工程上的多体动力学软件（ADAMS、RecurDyn）内部跑的就是自动生成拉格朗日方程。</span>

## 4 公式解析：标准拉格朗日方程

$$

\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_k} - \frac{\partial L}{\partial q_k} = 0

$$

对这条公式做三步拆解：

- **第一步，$\frac{\partial L}{\partial \dot{q}_k}$ 是什么**：$L$ 对广义速度求偏导。它是「广义动量」$p_k = \frac{\partial L}{\partial \dot{q}_k}$——对 $L = \frac{1}{2}m\dot{q}^2$ 的简单情形，$p = m\dot{q}$ 正是普通动量。对 $\dot{q}$ 求导是动力学方程的「加速度侧」。
- **第二步，$\frac{d}{dt}$ 与 $\frac{\partial L}{\partial q_k}$ 的平衡**：第一项是广义动量的时间变化率（惯性项），第二项是 $L$ 对广义坐标的偏导（广义力侧）。**「动量的时间变化率 = 广义力」**——这是牛顿第二定律在广义坐标下的翻译。
- **第三步，为什么只有 $L$ 出现**：整个方程只含一个函数 $L$。$V$ 不依赖 $\dot{q}$ 时 $\frac{\partial V}{\partial \dot{q}_k} = 0$，$T$ 中的广义坐标依赖由 $\frac{\partial T}{\partial q_k}$ 捕捉。**一个标量函数编码全部动力学，这是分析力学最深刻的抽象**。

## 5 辨析：拉格朗日方程与牛顿方程

| 特征 | 牛顿方程 | 拉格朗日方程 |
| --- | --- | --- |
| 语言 | 力与加速度 | 能量（$L = T - V$） |
| 约束反力 | 需显式处理 | 自动消去 |
| 坐标系 | 只能惯性系直角/自然坐标 | 任意广义坐标 |
| 方程数 | 通常 3 个/质点 | $s$ 个（自由度） |
| 适用 | 简单系统 | 复杂多体、约束系统 |

**辨析｜易错点：** 标准拉格朗日方程只适用于**完整约束**系统（$s$ 个独立广义坐标）。非完整约束（如冰刀、滚动轮）需要拉格朗日乘子法或 Routh 方程。直接把标准方程套到非完整系统上会得到错误结果。

**辨析｜易错点：** 拉格朗日方程中的偏导是「固定其它 $q$、$\dot{q}$ 不变」意义下的偏导——$q_k$ 与 $\dot{q}_k$ 被视为**独立变量**。初学者常困惑「$\dot{q}$ 明明是 $q$ 的导数，为何能独立取偏导」——这是拉格朗日力学的核心约定：在变分空间里 $q$ 与 $\dot{q}$ 独立。把这个约定记牢，计算就不会错。

## 6 应用示例：单摆的拉格朗日方程

用拉格朗日方程推导单摆的运动方程。摆长 $l$，摆锤质量 $m$，取摆角 $\theta$ 为广义坐标。

**解**：

**（1）自由度与广义坐标**：单摆在铅垂平面内运动，1 个自由度，取 $\theta$。

**（2）写动能与势能**：

$$
T = \frac{1}{2}ml^2\dot\theta^2, \qquad V = -mgl\cos\theta
$$

**（3）构造拉格朗日函数并代入方程**：

$$
L = T - V = \frac{1}{2}ml^2\dot\theta^2 + mgl\cos\theta
$$

计算各偏导：

$$
\frac{\partial L}{\partial\dot\theta} = ml^2\dot\theta, \quad
\frac{d}{dt}\frac{\partial L}{\partial\dot\theta} = ml^2\ddot\theta, \quad
\frac{\partial L}{\partial\theta} = -mgl\sin\theta
$$

代入 $\frac{d}{dt}\frac{\partial L}{\partial\dot\theta} - \frac{\partial L}{\partial\theta} = 0$：

$$
ml^2\ddot\theta + mgl\sin\theta = 0 \quad\Longrightarrow\quad \ddot\theta + \frac{g}{l}\sin\theta = 0
$$

**讨论**：全程没有画受力图、没有处理绳中张力（它是约束力，被理想约束自动消去），直接从 $L$ 得到运动方程。小角度时 $\sin\theta \approx \theta$，方程线性化为 $\ddot\theta + \frac{g}{l}\theta = 0$——简谐振动，频率 $\sqrt{g/l}$。这正是下一节《小振动理论》的雏形。

> 若改用直角坐标 $(x, y)$ 推导，必须先引入摆长约束、再解约束反力，繁琐得多——拉格朗日方程「选对坐标即成功一半」的优势在此毕现。

## 7 小结

- **第二类拉格朗日方程**：$\frac{d}{dt}\frac{\partial T}{\partial \dot{q}_k} - \frac{\partial T}{\partial q_k} = Q_k$，从动力学普遍方程经广义坐标改写而来。
- **拉格朗日函数** $L = T - V$：一个标量函数编码整个系统的动力学。
- **标准形式**：$\frac{d}{dt}\frac{\partial L}{\partial \dot{q}_k} - \frac{\partial L}{\partial q_k} = 0$，对保守完整系统普遍成立。
- 广义动量 $p_k = \partial L/\partial \dot{q}_k$：方程是「广义动量的变化率 = 广义力」。
- 应用四步法：定自由度、选坐标、写 $L$、代方程。
- **只适用完整约束**；$q$ 与 $\dot{q}$ 在变分意义下独立。

在下一节，拉格朗日方程的对称性将结出守恒律——**循环坐标与守恒量**，看「不出现某坐标」如何自动给出守恒定律。
