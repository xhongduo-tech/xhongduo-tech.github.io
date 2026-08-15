---
title: Cooper不稳定性与BCS超导
date: 2026-08-07
---

# Cooper不稳定性与BCS超导

<div class="epigraph">
<p>费米面是一个「不设防」的表面：哪怕吸引力再弱，只要它是净吸引的，两个电子就会在费米面外结成束缚对。费米面的失稳是普遍的——问题从来不是「会不会」，而是「靠什么」。</p>
<footer>—— L. N. Cooper（*Physical Review\*, 1956）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子多体理论 ｜ G. D. Mahan, *Many-Particle Physics\*, Ch. 10 ｜ 2026-08-07</p>
</div>

## 为什么需要 BCS 理论

超导电性（电阻精确为零 + Meissner 效应）1911 年被 Onnes 发现，却困扰理论界近半个世纪：没有散射怎么会零电阻？1950 年 Fröhlich 提示声子中介吸引；1956 年 **Cooper** 用一条简洁的论证击穿障碍——费米面在**任意弱的净吸引**下都不稳定；1957 年 **Bardeen、Cooper、Schrieffer**（BCS）给出完整理论，用一个「配对的凝聚」同时解释零电阻与 Meissner 效应。1972 年诺贝尔奖，公认是凝聚态物理最伟大的理论之一。

BCS 理论的现代意义远超超导本身：它是「**对称破缺的量子化范例**」——Cooper 对凝聚破缺 $U(1)$ 粒子数守恒，配对的相位成为超导序参量；它也是理解强关联体系（铜氧化物、铁基超导）的基准。这一篇把 Cooper 不稳定性、BCS 波函数、能隙方程与 $T_c$ 公式讲透。<span class="marginnote">Cooper 论证为何如此震撼：教科书级的「束缚态」需要强相互作用（氢原子、核力），而 Cooper 证明在费米面附近<strong>任意弱的吸引</strong>都能形成束缚对——条件是「其他电子把费米面占满」提供的相空间保护。费米面不是「帮忙」，而是「必须」：正因泡利原理禁止散射进已占态，一对电子反而被锁在费米面外的小壳层里，微弱吸引足以束缚。</span>

## 1 Cooper 问题

**Cooper 问题**：在充满费米海（$T=0$）的体系里，在费米面外加入**两个电子**，它们之间有吸引相互作用（由声子中介的净吸引）。设两电子自旋相反、总动量 $\mathbf{K}=0$（静止对），试探波函数：

$$|\Psi\rangle = \sum_{\mathbf{k}>k_F} g_{\mathbf{k}}\, c^\dagger_{\mathbf{k}\uparrow}c^\dagger_{-\mathbf{k}\downarrow}|F\rangle$$

用薛定谔方程求束缚能。相互作用取 BCS 简化形式（在能量壳层 $|\xi|\lt \hbar\omega_D$ 内取常数 $-V$）：

$$(2\xi_{\mathbf{k}} - E)\,g_{\mathbf{k}} = -V\sum_{\mathbf{k}'>k_F} g_{\mathbf{k}'}$$

设 $g$ 在壳层内为常数，得到束缚能：

$$E = -2\hbar\omega_D\,e^{-2/N(0)V}$$

**重点：只要 $V>0$（净吸引），Cooper 对必然束缚，束缚能随 $V$ 指数小但**恒不为零**。** 两个「多余」的电子在费米海顶上结成对——这不是「弱的物理」，这是「费米面的结构使然」。Cooper 对的有效半径 $\xi_0 = \hbar v_F/\pi\Delta \sim 10^3\,\text{Å}$，远大于晶格常数——配对是「动量空间」的事件而非「实空间」的事件。<span class="marginnote">Cooper 对半径 $\sim 1000\text{Å}$ 意味着配对的两个电子在实空间里距离极远、中间隔着上千个原子——所以 BCS 超导的「对」不是束缚分子，而是动量空间的相干态。这也解释了为什么 BCS 平均场如此精确：每个对与上万个其他对重叠，涨落被平均掉。</span>

## 2 BCS 波函数

Cooper 处理的是「两个电子」；BCS 的飞跃是把所有电子**同时**配成对。**BCS 基态波函数**是一个配对的相干态：

$$|\text{BCS}\rangle = \prod_{\mathbf{k}}\big(u_{\mathbf{k}} + v_{\mathbf{k}}c^\dagger_{\mathbf{k}\uparrow}c^\dagger_{-\mathbf{k}\downarrow}\big)|0\rangle$$

其中 $u_{\mathbf{k}}^2 + v_{\mathbf{k}}^2 = 1$，$v_{\mathbf{k}}^2$ 是态 $\mathbf{k}$ 被对占据的概率。**重点：BCS 基态不是「费米海 + 一些对」，而是每个动量态都处于「空/占」的叠加**——粒子数不确定（$U(1)$ 破缺），相位确定。这正是凝聚体序参量的来源：

$$\Delta_{\mathbf{k}} = V\sum_{\mathbf{k}'}\langle c_{-\mathbf{k}'\downarrow}c_{\mathbf{k}'\uparrow}\rangle = V\sum_{\mathbf{k}'}u_{\mathbf{k}'}v_{\mathbf{k}'}$$

$\Delta_{\mathbf{k}}$ 就是**超导能隙（gap）**，同时也是超导序参量。基态能量相对费米海降低，凝聚能在 $T=0$ 时为：

$$E_{\text{cond}} = -\frac{1}{2}N(0)\Delta^2$$

<span class="marginnote">BCS 基态是「配对凝聚」的量子版本：每个 $(u,v)$ 混合都是一个 Bogoliubov 变换——与上一节玻色凝聚的 Bogoliubov 变换完全同构！区别只在「玻色凝聚是粒子凝聚，超导凝聚是配对凝聚」——两个费米子绑成一个「复合玻色子」再凝聚。这个「配对 → 玻色化 → 凝聚」的三部曲是 BCS 与 BEC 统一的桥梁。</span>

## 3 平均场自洽与能隙方程

用平均场把相互作用哈密顿量分解，保留「配对场」通道：

$$-V\,c^\dagger_{\mathbf{k}\uparrow}c^\dagger_{-\mathbf{k}\downarrow}c_{-\mathbf{k}'\downarrow}c_{\mathbf{k}'\uparrow} \;\to\; \Delta_{\mathbf{k}}c^\dagger_{\mathbf{k}\uparrow}c^\dagger_{-\mathbf{k}\downarrow} + \Delta_{\mathbf{k}}^* c_{-\mathbf{k}\downarrow}c_{\mathbf{k}\uparrow}$$

其中 $\Delta_{\mathbf{k}}$ 自洽确定（$\Delta = V\sum u v$）。对角化后的**准粒子谱**：

$$E_{\mathbf{k}} = \sqrt{\xi_{\mathbf{k}}^2 + |\Delta_{\mathbf{k}}|^2}$$

**重点：准粒子谱出现能隙 $|\Delta|$——费米面处的激发能量不再从零开始。** 这个能隙是超导最深刻的物理：它「锁住」费米面，使任何单粒子激发都要付出至少 $|\Delta|$ 的能量代价，从而抵抗散射（零电阻）与磁场穿透（Meissner）。能隙方程在 $T=0$ 时：

$$\frac{1}{N(0)V} = \int_0^{\hbar\omega_D}\frac{d\xi}{\sqrt{\xi^2+\Delta^2}} = \sinh^{-1}\frac{\hbar\omega_D}{\Delta}$$

弱耦合极限解出：

$$\Delta_0 = 2\hbar\omega_D\,e^{-1/N(0)V}$$

**辨析｜易错点：** 初学者常混淆三个「能隙」：**准粒子能隙 $E_{\mathbf{k}}=\sqrt{\xi^2+\Delta^2}$**（单粒子激发的最低能量）、**配对振幅 $\Delta$**（序参量/能隙参数）、**超导能隙 $2\Delta$**（光吸收、隧穿实验测到的双粒子阈值）。三者关系：单粒子激发最小能量是 $\Delta$，而拆开一个 Cooper 对（产生两个准粒子）需要 $2\Delta$。实验里隧穿谱测到的是 $2\Delta$。

## 4 公式解析：临界温度 $T_c$

把 $T_c$ 从有限温度能隙方程解出来，是 BCS 最著名的成果：

- **第一步，写有限温能隙方程**：有限温度下 $\Delta(T) = V\sum_{\mathbf{k}}\frac{\Delta}{2E_{\mathbf{k}}}\tanh\frac{E_{\mathbf{k}}}{2k_BT}$。
- **第二步，取 $T\to T_c$ 极限**：$\Delta\to0$，$\tanh$ 展开为 $E/2k_BT_c$，方程线性化为 $\frac{1}{N(0)V} = \int_0^{\hbar\omega_D}\frac{d\xi}{2\xi}\tanh\frac{\xi}{2k_BT_c}$。
- **第三步，积分**：弱耦合下积分给出 $\ln\frac{2\hbar\omega_D}{1.13k_BT_c} \approx 1/N(0)V$，从而：
  $$k_BT_c = 1.13\,\hbar\omega_D\,e^{-1/N(0)V}$$
- **第四步，与 $\Delta_0$ 联系**：$\frac{2\Delta_0}{k_BT_c} \approx 3.52$（普适比）。

**重点：$T_c \propto \omega_D e^{-1/N(0)V}$，且 $2\Delta_0/k_BT_c = 3.52$ 是普适常数。** 这两个结果给出两个著名预言：**同位素效应**（$T_c\propto M^{-1/2}$，因为 $\omega_D\propto M^{-1/2}$——用不同同位素替换离子时 $T_c$ 移动，证明声子参与配对）与**普适比**（与材料无关，是 BCS 弱耦合的指纹）。BCS 用这两个可测预言把「配对机制」从抽象变成可检验。

## 5 BCS 与「从极限到大模型」

BCS 理论是「从极限到大模型」里「**微扰失效处的胜利**」：Cooper 对束缚能是指数小的（$e^{-1/\lambda}$），**任何有限阶微扰论都看不到它**——只有非微扰的「配对重排」才能抓住。这给机器学习一个深刻提醒：**某些涌现效应（涌现能力、相变式的行为跃迁）可能本质上是非微扰的**——用「参数再多加一点」的微扰式推演可能永远抓不住它们，需要「重新配对」式的理论（相变理论、RG 流）来理解。<span class="marginnote">更具体的类比：BCS 的「配对通道」对应机器学习里「通道/电路」的选择——模型在训练中「自发选择」某个表征通道（如某种特征检测器），就像 BCS 体系「自发选择」配对通道。理解「哪些通道会失稳」也许是理解模型涌现能力的钥匙。可参考第四级《大模型原理》。</span>

对多体理论自身，BCS 是通往强耦合超导（Eliashberg）、隧穿（Josephson）与高温超导（t-J 模型）的基准——下一节我们看看声子耦合不再弱时会发生什么：**Eliashberg 强耦合超导理论**。

## 6 小结

- **Cooper 不稳定性**：费米面外一对电子在任意弱净吸引下形成束缚对，束缚能 $E=-2\hbar\omega_De^{-2/N(0)V}$ 指数小但不为零。
- **BCS 基态**是配对相干态 $\prod(u+v\,c^\dagger c^\dagger)|0\rangle$：粒子数破缺、相位确定，$\Delta$ 是序参量。
- 平均场准粒子谱 $E_{\mathbf{k}}=\sqrt{\xi_{\mathbf{k}}^2+\Delta^2}$ 出现能隙 $\Delta$——超导抗散射与抗磁场的根源。
- $T=0$ 能隙 $\Delta_0 = 2\hbar\omega_De^{-1/N(0)V}$；$T_c = 1.13\hbar\omega_De^{-1/N(0)V}$；普适比 $2\Delta_0/k_BT_c = 3.52$。
- **同位素效应**（$T_c\propto M^{-1/2}$）证明声子配对；能隙方程是「配对 → 凝聚 → 相变」的完整闭环。
- 区分三个「能隙」：准粒子能隙 $\Delta$、配对振幅、隧穿阈值 $2\Delta$。

在下一节，我们放宽 BCS 的弱耦合假设：**Eliashberg 强耦合超导理论**——当声子谱有结构、耦合不再小时，如何用 Green 函数自洽地处理配对，以及强耦合如何修正 $T_c$ 与普适比。


## 公式速查：一页纸复习

| 对象 | 公式 | 一句话要点 |
| --- | --- | --- |
| Cooper 束缚能 | $E = -2\hbar\omega_D\,e^{-2/N(0)V}$ | 任意弱净吸引都束缚，指数小不为零 |
| BCS 基态 | $|\text{BCS}\rangle = \prod(u+v\,c^\dagger c^\dagger)|0\rangle$ | 配对相干态，粒子数破缺 |
| 准粒子谱 | $E_{\mathbf{k}} = \sqrt{\xi_{\mathbf{k}}^2+\Delta^2}$ | 费米面处有能隙 $\Delta$ |
| 零温能隙 | $\Delta_0 = 2\hbar\omega_D\,e^{-1/N(0)V}$ | 弱耦合指数公式 |
| 临界温度 | $k_BT_c = 1.13\hbar\omega_D\,e^{-1/N(0)V}$ | $2\Delta_0/k_BT_c = 3.52$ |
| 同位素效应 | $T_c \propto M^{-1/2}$ | 证明声子参与配对 |

**易错复盘**：三点要盯住。其一，区分三个「能隙」：准粒子能隙 $\Delta$、配对振幅 $\Delta$、隧穿阈值 $2\Delta$——拆开一个 Cooper 对需要 $2\Delta$；其二，BCS 基态不是「费米海 + 一些对」，而是每个动量态都处于空/占叠加的相干态；其三，Cooper 对半径约 1000 Å——配对是动量空间事件，不是实空间的束缚分子。

**知识连线**：Cooper 不稳定性来自第 3 篇电子-声子相互作用（声子中介吸引）；BCS 基态是第 3 篇 Bogoliubov 变换（玻色凝聚）的费米子配对版。「费米面在任意弱吸引下失稳」是「从极限到大模型」里「非微扰效应不能被有限阶微扰捕捉」的物理实例——指数小的束缚能，任何微扰展开都看不到。

**实践与辨析**：为什么 Cooper 对的束缚能在弱耦合极限下指数小但仍确定存在？提示：费米面的泡利保护提供了相空间，$e^{-1/\lambda}$ 来自积分 $\int d\xi/\xi$ 的对数。为什么必须自旋相反？提示：自旋相反的电子可以占据相同的动量空间壳层，泡利不禁止。易错提醒：BCS 的 $e^{-1/\lambda}$ 不是微扰结果——它是求和无穷级数后的非微扰指数，任何有限阶微扰都得不到它。

**延伸思考**：若把 $V$ 换成纯库仑排斥（$V<0$），费米面还失稳吗？提示：裸排斥不能配对；但库仑排斥被推迟效应压低（Eliashberg 的 $\mu^*$）后，声子中介的净吸引可以胜出——这正是声子超导在强库仑背景下仍能存在的原因。