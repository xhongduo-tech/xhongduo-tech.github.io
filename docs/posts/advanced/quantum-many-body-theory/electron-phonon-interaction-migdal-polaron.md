---
title: 电子-声子相互作用（米氏理论、极化子、超导机制）
date: 2026-08-07
---

# 电子-声子相互作用（米氏理论、极化子、超导机制）

<div class="epigraph">
<p>晶格振动是电子的一条隐秘通道：它们靠声子传递着一种温柔的吸引，足以把库仑斥力掀翻——这就是超导。</p>
<footer>—— 赫伯特 · 弗勒利希（Herbert Fröhlich）论电声耦合（转述）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics*, Ch. 7 ｜ 2026-08-07</p>
</div>

## 为什么从电子-声子相互作用开始

固体里的电子不是孤军奋战：离子实的振动（声子）会改变电位，反过来电子的运动会极化晶格。这个**电子-声子相互作用（electron-phonon interaction）**是凝聚态物理里最古老也最深刻的相互作用之一。<span class="marginnote"><strong>为什么重要</strong>：它驱动了金属电阻（声子散射）、质量增强、极化子，以及 1957 年 BCS 超导的配对机制——几乎所有传统超导体的「胶水」都是声子。</span>本篇沿着**极化子**（一个电子拖着一团晶格形变）、**米氏理论（Migdal 近似）**（多体理论对电声系统最重要的技术结论）与**超导机制**（声子如何把费米子配成对）三条线，把这一相互作用的物理与图论完整梳理。

## 1 电声耦合哈密顿量

离子实位移 $u_i$ 通过形变势 $V_{\mathrm{ep}}$ 与电子密度耦合。在二次量子化里，电子-声子顶点耦合电子与声子场：

$$\hat{H}_{\mathrm{ep}} = \sum_{\mathbf{k},\mathbf{q},\sigma} g(\mathbf{q})\,c_{\mathbf{k}+\mathbf{q},\sigma}^\dagger c_{\mathbf{k},\sigma}(a_{\mathbf{q}} + a_{-\mathbf{q}}^\dagger)$$

$g(\mathbf{q})$ 是动量依赖的耦合强度（形变势或 Fröhlich 型）。算符结构一目了然：电子从 $\mathbf{k}$ 散射到 $\mathbf{k}+\mathbf{q}$，同时吸收（$a_{\mathbf{q}}$）或发射（$a_{-\mathbf{q}}^\dagger$）一个声子。<span class="marginnote"><strong>两种典型顶点</strong>：形变势耦合 $g\sim q$（长波时弱，纵声子主导）；极性晶体里 Fröhlich 耦合 $g\sim 1/q$（长波最强，纵光学声子主导）。后者是极化子物理的基础，也是离子晶体与金属的差异所在。</span>这一项正是金属电阻与超导配对共同的微观来源。

## 2 极化子：电子 + 晶格极化云

当电子在极性晶格里运动，它吸引正离子、排斥负离子，拖着一团晶格形变一起走——这个复合体叫**极化子（polaron）**。

- **大极化子（large polaron）**：电子与形变云半径远大于晶格常数，电子几乎自由但有效质量增强、能量下移。用微扰计算，自能实部给出 $m^*/m\approx 1+\alpha/6$（$\alpha$ 是 Fröhlich 耦合常数）。<span class="marginnote"><strong>耦合常数 $\alpha$</strong>：$0<\alpha<6$ 左右为大极化子微扰区，$\alpha$ 很大时微扰失效。中间态与自陷（self-trapping）是经典难题，至今仍有活跃讨论。</span>
- **小极化子（small polaron）**：形变半径小于晶格常数，电子被局域在单个格点上「自陷」，只能靠热激活或隧穿跳跃移动——迁移率随温度升高（区别于金属中电子随温度升高而减慢）。

**重点：极化子是「电子 + 玻色云」的最简单量子液体。** 它展示了费米子与玻色场的纠缠如何改变粒子的质量、局域化与输运——这套图像也是理解重费米子、以及声子参与超导的前奏。

## 3 米氏理论：Migdal 定理与电子-声子图

多体理论对电声系统最重要的技术成果是 **Migdal 定理（Migdal's theorem, 1958）**：因为声子能量 $\hbar\omega_D$ 远小于电子能量尺度（费米能 $\varepsilon_F$），电声自能的**顶点修正**与**交叉图**的贡献被小参数 $\hbar\omega_D/\varepsilon_F$ 压制，可以忽略。于是自能只保留「单圈电子-声子图」（电子线夹一条声子线），称为 **Migdal 近似**。

$$\Sigma(\mathbf{k},i\omega_n) \approx -\frac{1}{\beta}\sum_{\mathbf{q},m} |g(\mathbf{q})|^2\, D_0(\mathbf{q},i\omega_m)\, G_0(\mathbf{k}-\mathbf{q},i\omega_n-i\omega_m)$$

**重点：Migdal 定理把电声问题从「任意高阶图」压缩成「可解的少图问题」。** 对金属（$\hbar\omega_D/\varepsilon_F\sim10^{-2}$）这个近似极好，金属超导的理论几乎全部建立其上。<span class="marginnote"><strong>何时失效</strong>：当 $\hbar\omega_D$ 不再远小于 $\varepsilon_F$（如某些半金属、重费米子、或接近零能隙情形），顶点修正不可忽略。Migdal 近似的边界本身就是一个研究课题。</span>

## 4 公式解析：从 Migdal 自能到质量增强

把 Migdal 自能算到解析结果，能直接看到「声子如何让电子变重」：

$$
m^* = m\left(1 + \lambda\right), \qquad \lambda = 2\int_0^{\infty}\frac{d\omega}{\omega}\,\alpha^2F(\omega)
$$

- **$\lambda$ 是电声耦合常数**：$\alpha^2F(\omega)$ 是 Eliashberg 谱函数（电声耦合强度按声子频率的分布，可由隧穿实验测得），积分权重 $1/\omega$ 使低频声子贡献更大。
- **有效质量增强**：$m^*/m=1+\lambda$。对铝 $\lambda\approx0.4$，铅 $\lambda\approx1.5$，铌 $\lambda\approx1.0$ 量级——金属比热测量的质量增强直接给出 $\lambda$。
- **对超导的意义**：$\lambda$ 越大，声子配对越强，超导临界温度越高。Eliashberg 理论（Mahan 第10章）用 $\alpha^2F(\omega)$ 与库仑赝势 $\mu^*$ 定量预测 $T_c$——传统超导体里 $T_c$ 的最高纪录（MgB$_2$，39 K）就来自强电声耦合。

**重点：一个积分 $\lambda$ 同时控制质量增强与超导强度。** 声子既增加电子惯性，又提供配对吸引——两个效应共用同一个谱函数 $\alpha^2F(\omega)$，这是电声物理最漂亮的统一。

## 5 声子如何促成超导：动态吸引

声子吸引电子配对的机理值得单独说清。频率依赖的有效电子-电子相互作用是

$$V_{\mathrm{eff}}(\omega) = V_c + \frac{|g|^2\,\omega_q}{\omega^2 - \omega_q^2}$$

在低频区（$\omega<\omega_q$），第二项为负——声子提供的**吸引**超过了库仑排斥的屏蔽部分。两个电子交换一个虚声子，净效果是相互吸引，这就是 BCS 配对的微观根源。<span class="marginnote"><strong>关键在动态性</strong>：库仑相互作用近乎瞬时，声子媒介的相互作用却带延迟（$\sim1/\omega_D$）。延迟让两个电子在交换虚声子的瞬间不需要同时同地靠近——泡利原理的压力因此被绕开。这正是「声子胶水」能克服库仑斥力的核心机制。</span>超导转变温度由 BCS 公式 $T_c \sim \omega_D e^{-1/\lambda_{\mathrm{eff}}}$ 给出，指数因子解释了为什么 $T_c$ 普遍远低于声子频率。

## 6 极化子、Migdal 与超导的总览

| 现象 | 尺度/对象 | 关键量 | 物理要点 |
| --- | --- | --- | --- |
| 大极化子 | 半径 $\gg a$ | $\alpha$ | 质量增强 $1+\alpha/6$ |
| 小极化子 | 半径 $\sim a$ | 跳跃积分 | 自陷、热激活迁移 |
| Migdal 近似 | 金属电声 | $\hbar\omega_D/\varepsilon_F$ | 顶点修正可忽略 |
| 质量增强 | 比热 | $\lambda$ | $m^*/m=1+\lambda$ |
| BCS 配对 | 超导 | $\omega_D$,$\lambda$ | 虚声子交换吸引 |
| Eliashberg | 强耦合超导 | $\alpha^2F(\omega)$,$\mu^*$ | $T_c$ 定量预测 |

## 7 具体例子：用 λ 估算铅的临界温度

把电声理论接上实验数字，用著名的 **McMillan 公式**（强耦合 BCS 的实用版本）估算铅的 $T_c$。铅的参数：$\lambda\approx1.55$，德拜温度约 $\Theta_D\approx96\,\mathrm{K}$，库仑赝势 $\mu^*\approx0.1$。

$$T_c \approx \frac{\Theta_D}{1.45}\exp\left[-\frac{1.04(1+\lambda)}{\lambda-\mu^*(1+0.62\lambda)}\right] \approx \frac{96}{1.45}\,e^{-2.63} \approx 7\,\mathrm{K}$$

几个要点：

- **实验对照**：铅的实测 $T_c=7.2\,\mathrm{K}$，与估算几乎完全一致。强耦合金属（铅、铌）必须用 Eliashberg/McMillan 而非常用弱耦合 BCS（后者给出的 $T_c$ 会系统性偏低）。
- **指数因子的作用**：$\lambda-\mu^*(1+0.62\lambda)\approx0.9$ 很小的分母让指数很大，$T_c$ 被压到声子频率的百分之几——这解释了为什么传统超导 $T_c$ 很难超过几十 K。
- **为什么是铅**：铅的电声耦合强（软声子、高 $\lambda$），是 McMillan 公式的标准案例；对比铝（$\lambda\approx0.4$，$T_c\approx1.2\,\mathrm{K}$），$\lambda$ 差四倍，$T_c$ 差六倍——超导对耦合常数高度敏感。

**重点：一个公式、两组材料参数，就能把超导临界温度从「现象」变成「可算的量」。** 这正是电声超导理论作为凝聚态物理最成功定量理论的底气所在。

## 8 小结

- 电声耦合哈密顿量含形变势与 Fröhlich 两种顶点，是电阻与超导的共同来源。
- **大极化子**质量增强 $1+\alpha/6$，**小极化子**自陷跳跃，是电子-玻色云纠缠的两极。
- **Migdal 定理**用 $\hbar\omega_D/\varepsilon_F\ll1$ 压制顶点修正，电声自能简化为单圈图。
- **电声耦合常数 $\lambda$** 同时控制质量增强 $m^*/m=1+\lambda$ 与超导强度。
- 声子媒介的动态吸引（延迟、可绕开泡利压力）是 BCS 配对的微观根源，$T_c\sim\omega_D e^{-1/\lambda}$。
- Eliashberg 函数 $\alpha^2F(\omega)$ 可由隧穿实验测得，是强耦合超导定量理论的核心输入。
- McMillan 公式用 $\lambda$、$\Theta_D$、$\mu^*$ 定量预测 $T_c$，铅、铝等金属的估算与实验高度一致。
- 电声耦合同时是电阻、质量增强与超导的来源，统一性强是本主题的最大特点。
- 对强耦合超导（铅、铌、MgB$_2$），必须用 Eliashberg/McMillan 框架而非常用弱耦合 BCS。

## 9 公式速查：一页纸复习

| 对象 | 表达式 | 一句话要点 |
| --- | --- | --- |
| 电声顶点 | $g(\mathbf{q})c^\dagger c(a+a^\dagger)$ | 吸收/发射声子 |
| 大极化子 | $m^*/m=1+\alpha/6$ | 微扰区 |
| Migdal 自能 | $\Sigma\approx-G_0D_0$ 单圈 | 顶点修正被压制 |
| 耦合常数 | $\lambda=2\int\frac{d\omega}{\omega}\alpha^2F$ | 质量增强 + 超导 |
| 动态吸引 | $V_{\mathrm{eff}}=V_c+\frac{\|g\|^2\omega_q}{\omega^2-\omega_q^2}$ | 低频为负 |
| BCS $T_c$ | $T_c\sim\omega_D e^{-1/\lambda}$ | 指数压低 |

**易错复盘**：Fröhlich 耦合 $g\sim1/q$ 与形变势 $g\sim q$ 的动量依赖别混；Migdal 近似只在 $\hbar\omega_D\ll\varepsilon_F$ 时成立；$\lambda$ 的 $1/\omega$ 权重使低频声子更重要；超导的吸引是「动态的」，静态极限下不存在。另注意 $\lambda$ 既进质量增强又进 $T_c$ 指数，同一量不能重复计数。

**知识连线**：本篇承接第 3 篇超导专题的 BCS 内容，是 Eliashberg 强耦合理论（已有博文）的直接入口；极化子概念延伸到重费米子、磁性半导体与钙钛矿光伏。「电子借声子结成对」这种「借中间媒介实现合作」的结构，与「从极限到大模型」里模型通过隐变量形成关联表示的思想形成有趣类比。

在下一节，我们将进入一个完全不同的破坏性效应：无序系统与安德森局域化——杂质平均、弱局域化与标度理论。
