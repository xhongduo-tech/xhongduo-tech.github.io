---
title: BCS 基态与 Bogoliubov 正则变换
date: 2026-08-07
---

# BCS 基态与 Bogoliubov 正则变换

<div class="epigraph">
<p>超导基态不是一堆独立的库珀对，而是全体电子共享同一个相位的相干凝聚体。</p>
<footer>—— 约翰 · 巴丁、莱昂 · 库珀、罗伯特 · 施里弗（Bardeen, Cooper & Schrieffer，1957）</footer>
</div>

<div class="article-byline">
<p>第四级 · 超导物理 ｜ Tinkham《Introduction to Superconductivity》第3章；Ketterson & Song 第26、27章 ｜ 2026-08-07</p>
</div>

## 为什么从 BCS 波函数开始

库珀问题证明了「一对电子能束缚」，但超导是 $N$ 个电子（$N/2$ 个对）的集体行为。施里弗的突破在于猜出**整个系统的基态波函数**：不是「一堆库珀对的乘积」，而是一个所有对共享配分权的**相干态**。BCS 波函数有两个等价但互补的处理方法：变分法直接猜波函数，或 Bogoliubov 正则变换把哈密顿量对角化。后者是现代超导理论的标准工具——能隙、准粒子、热力学全部从它导出。本篇把 BCS 波函数与 Bogoliubov 变换讲透，下一篇再讨论能隙方程与 $T_c$。

## 1 BCS 约化哈密顿量

BCS 只保留对配对的相互作用项（约化哈密顿量）：

$$
H_{\text{red}} = \sum_{\boldsymbol{k}} 2\epsilon_{\boldsymbol{k}}\, b_{\boldsymbol{k}}^{\dagger}b_{\boldsymbol{k}} - V\sum_{\boldsymbol{k}\boldsymbol{k}'} b_{\boldsymbol{k}'}^{\dagger}b_{\boldsymbol{k}}
$$

其中 $b_{\boldsymbol{k}} = c_{-\boldsymbol{k}\downarrow}c_{\boldsymbol{k}\uparrow}$ 湮灭一个「$\pm\boldsymbol{k}$ 自旋反平行」的电子对，$b_{\boldsymbol{k}}^{\dagger}$ 是产生。第一项是对的能量（$2\epsilon_{\boldsymbol{k}}$），第二项是配对的散射：一对 $\boldsymbol{k}$ 被打散、一对 $\boldsymbol{k}'$ 被造出。<span class="marginnote">约化哈密顿量丢掉了很多相互作用细节，但抓住了超导的精髓：只允许「$( \boldsymbol{k}\uparrow, -\boldsymbol{k}\downarrow)$ 对 ⇄ 另一个对」的散射。BCS 证明：这类散射足以给出正确的基态与能隙；其余相互作用（自旋涨落、磁杂质等）作为微扰再叠加——这正是后来「约化模型 + 修正」路线的模板。</span>

## 2 BCS 基态波函数

BCS 基态用「每个配对态 $\boldsymbol{k}$ 被占据或空」的叠加来构造：

$$
|\text{BCS}\rangle = \prod_{\boldsymbol{k}} \left(u_{\boldsymbol{k}} + v_{\boldsymbol{k}}\, b_{\boldsymbol{k}}^{\dagger}\right) |0\rangle
$$

其中 $|u_{\boldsymbol{k}}|^2 + |v_{\boldsymbol{k}}|^2 = 1$。物理意义：$|v_{\boldsymbol{k}}|^2$ 是态 $\boldsymbol{k}$ 被一对占据的概率，$|u_{\boldsymbol{k}}|^2$ 是空置概率。<span class="marginnote">这个乘积波函数是「相干态」而非「Fock 数态」：每个 $\boldsymbol{k}$ 处于占据与空置的叠加。粒子数有涨落（$\Delta N \sim N^{1/2}$ 量级），但相对涨落 $\Delta N/N \to 0$——所以宏观上粒子数仍然是好量子数。这正是「自发对称破缺」的量子版本：相位确定，粒子数不确定。</span>

变分参数 $v_{\boldsymbol{k}}$ 由最小化平均能量确定。令 $\xi_{\boldsymbol{k}} = \epsilon_{\boldsymbol{k}} - E_F$，解出：

$$
v_{\boldsymbol{k}}^2 = \frac{1}{2}\left(1 - \frac{\xi_{\boldsymbol{k}}}{E_{\boldsymbol{k}}}\right), \qquad u_{\boldsymbol{k}}^2 = \frac{1}{2}\left(1 + \frac{\xi_{\boldsymbol{k}}}{E_{\boldsymbol{k}}}\right)
$$

其中 $E_{\boldsymbol{k}} = \sqrt{\xi_{\boldsymbol{k}}^2 + \Delta^2}$ 是**准粒子激发能**，$\Delta$ 是**能隙**。在费米面（$\xi=0$），$u=v=1/\sqrt2$：配对概率最高；远离费米面，一个概率趋向 1、另一个趋向 0——配对只发生在费米面附近的薄壳内，与库珀问题的结论自洽。

## 3 Bogoliubov 正则变换

Bogoliubov（以及 Valatin）提出用正交变换引入**准粒子算符**：

$$
\gamma_{\boldsymbol{k}\uparrow} = u_{\boldsymbol{k}} c_{\boldsymbol{k}\uparrow} - v_{\boldsymbol{k}} c_{-\boldsymbol{k}\downarrow}^{\dagger}, \qquad
\gamma_{\boldsymbol{k}\downarrow} = u_{\boldsymbol{k}} c_{\boldsymbol{k}\downarrow} + v_{\boldsymbol{k}} c_{-\boldsymbol{k}\uparrow}^{\dagger}
$$

$\gamma$ 是费米子（满足反对易关系），代表「Bogoliubov 准粒子」——真实超导体中电子的基本激发单元。<span class="marginnote">$\gamma_{\boldsymbol{k}\uparrow}$ 混合了「一个 $\boldsymbol{k}\uparrow$ 电子」与「一个 $-\boldsymbol{k}\downarrow$ 空穴」：准粒子是电子与空穴的叠加。这正是能隙物理的来源——准粒子「想当电子又当不了」，被配对能量绑住，激发需要付出至少 $\Delta$ 的能量。</span>

用 $\gamma$ 重写约化哈密顿量，选取 $u_{\boldsymbol{k}}, v_{\boldsymbol{k}}$ 使得交叉项（$\gamma\gamma$、$\gamma^{\dagger}\gamma^{\dagger}$）消失，得到**对角化**形式：

$$
H = \sum_{\boldsymbol{k}} E_{\boldsymbol{k}}\left(\gamma_{\boldsymbol{k}\uparrow}^{\dagger}\gamma_{\boldsymbol{k}\uparrow} + \gamma_{\boldsymbol{k}\downarrow}^{\dagger}\gamma_{\boldsymbol{k}\downarrow}\right) + E_0
$$

基态（所有 $\gamma$ 真空）能量 $E_0 \lt  E_{\text{normal}}$，能量差就是**凝聚能**。

## 4 公式解析：能隙从变分涌现

把变分条件写成自洽方程，能隙 $\Delta$ 的出现是 BCS 最美妙的一步：

$$
\frac{1}{N(0)V} = \int_0^{\hbar\omega_D} \frac{d\xi}{\sqrt{\xi^2 + \Delta^2}}
$$

- **第一步，写平均能量**：$\langle H_{\text{red}}\rangle = \sum 2\xi_{\boldsymbol{k}} v_{\boldsymbol{k}}^2 - V\sum_{\boldsymbol{k}\boldsymbol{k}'} u_{\boldsymbol{k}'}v_{\boldsymbol{k}'}u_{\boldsymbol{k}}v_{\boldsymbol{k}}$。
- **第二步，定义能隙**：令 $\Delta = V\sum_{\boldsymbol{k}} u_{\boldsymbol{k}} v_{\boldsymbol{k}}$（配对的「凝聚序参量」），能量变成 $E_0 = \sum_{\boldsymbol{k}}[\xi_{\boldsymbol{k}} - E_{\boldsymbol{k}}] + \Delta^2/V$。
- **第三步，变分求极值**：对 $u_{\boldsymbol{k}},v_{\boldsymbol{k}}$ 变分，得到 $E_{\boldsymbol{k}} = \sqrt{\xi_{\boldsymbol{k}}^2+\Delta^2}$ 与上面的自洽方程。
- **第四步，看出能隙**：准粒子最小激发能是 $\min E_{\boldsymbol{k}} = \Delta$（在 $\xi=0$ 处）。$\Delta$ 不是输入参数，而是**自洽解**——它由 $N(0)V$ 通过指数关系决定，$\Delta_0 \approx 2\hbar\omega_D e^{-1/N(0)V}$。

解出零温能隙与临界温度之比：

$$
\frac{2\Delta_0}{k_BT_c} \approx 3.53
$$

这是一个纯常数，与材料无关——BCS 最著名的普适预言。实验上铅的比值 $\approx 4.5$、锡 $\approx 3.6$、铝 $\approx 3.4$，偏离来自强耦合修正。<span class="marginnote">$2\Delta_0/k_BT_c \approx 3.53$ 常被当作「验证 BCS」的第一检验。偏离太多（如 $>4.5$）通常提示强耦合或非声子机制——铜氧化物的比值可达 5–8，是最早暗示「BCS 弱耦合不够」的证据之一。</span>

## 5 凝聚能与热力学的一致性

把 $E_0$ 与正常态能量相减，得到**凝聚能密度**：

$$
E_{\text{cond}} = \frac{1}{2}N(0)\Delta_0^2
$$

这必须与唯象热力学的 $H_c^2/8\pi$ 一致（热力学篇的记账本）。代入 $\Delta_0$ 的表达式，确实导出 $H_c \propto N(0)^{1/2}\Delta_0$，两个理论在「凝聚能」这个公共点完美对接——这是 BCS 理论正确性的一个内部一致性检验。<span class="marginnote">这条一致性把「微观 BCS」与「宏观 GL」缝合：GL 序参量 $\psi$ 的模正比于 $\Delta$（配对振幅），GL 自由能的 $\alpha$、$\beta$ 系数可以从 BCS 严格导出（GL 是 BCS 在 $T_c$ 附近的极限）。两大理论体系就此统一。</span>

## 6 BCS 波函数的两个深层后果

BCS 波函数不只是「算出了能隙」——它携带了两个影响深远的物理后果，理解它们才能把握 BCS 的真正分量：

**后果一：自发对称破缺（U(1) 破缺）**。BCS 基态是 $N$ 与 $N+2$ 粒子态的叠加，粒子数不确定而**相位确定**。这打破了「电子数守恒」的规范对称性——超导态选择了一个特定的相位 $\varphi$，正如铁磁体选择了一个特定的磁化方向。这个对称破缺是「宏观量子相位」的本源，也是约瑟夫森效应（相位差 → 电流）的根。它提示：**超导不是「很多库珀对的简单堆积」，而是一个对称性自发破缺的新物态**——这在概念上远比「电阻为零」深刻。

**后果二：BCS 与玻色-爱因斯坦凝聚（BEC）的联系**。库珀对是复合玻色子，BCS 基态是它们的凝聚。当配对从「弱耦合」（库珀对尺寸 $\xi_0$ 远大于对间距）连续过渡到「强耦合」（对紧密结合、间距 $\gg$ 尺寸），就发生 **BCS-BEC 过渡**：BCS 端是「重叠的大对」，BEC 端是「独立的分子玻色子」。冷原子实验用 Feshbach 共振完美实现了这个过渡——超导与冷原子凝聚在这条路上汇合。<span class="marginnote">把 BCS-BEC 过渡放在更大的图景：它把「超导」（费米子配对）与「超流」（玻色子凝聚）统一成一个连续谱系。铜氧化物、铁基超导在相图的某些区域可能处于「中间耦合」（BCS 与 BEC 之间），这为高温超导提供了一种不同于声子机制的思路——「预形成对 + 弱相干」（见《赝能隙》一篇）。BCS 波函数是理解这一切的起点。</span>

**Bogoliubov 变换的普适性再强调**：它不仅是超导的工具——超流氦、冷原子 BEC、拓扑绝缘体表面态（Majorana 费米子）都用同样的「准粒子混合」结构。Bogoliubov 准粒子的「电子-空穴叠加」是凝聚态「准粒子」概念的典范：**激发的实体不是裸粒子，而是裸粒子的相干混合**——这是理解现代凝聚态物理的钥匙之一。

## 7 历史注脚：BCS 的诞生与诺奖

BCS 理论是凝聚态物理最著名的成功故事之一，它的历史能给我们几点启发：

**合作的形式**：巴丁（Bardeen）是半导体/超导的资深权威，库珀（Cooper）提供了两体问题的数学突破，施里弗（Schrieffer）最终猜出波函数——**三个人的互补（理论直觉 + 数学技巧 + 大胆猜测）是 BCS 成功的关键**。1957 年论文以三人姓氏联名发表，1972 年获诺贝尔奖。

**波函数是怎么「猜」出来的**：施里弗花了近一年尝试各种多体波函数，最后在旅途中「灵光一现」写下乘积波函数——它的正确性立刻被验证（能隙、凝聚能、热力学全部对）。**「一个大胆的猜测 + 立即验证」是理论物理最戏剧性的时刻之一**——BCS 波函数从「猜测」到「教科书」只用了一年。

**BCS 的持久影响**：BCS 理论不仅解释了超导，还确立了「自发对称破缺 + 准粒子 + 能隙」的范式——它影响了一整代凝聚态理论（核物理的对关联、粒子物理的电弱统一、冷原子凝聚）。**BCS 是「范式转移」的教科书案例**：它把超导从「材料之谜」变成了「对称性破缺物理」的一个实例。<span class="marginnote">BCS 的教训放在今天格外有启发：当常规方法（微扰论、变分）都失败时，<strong>换个基础（相干态、对称性破缺）往往比加大计算量更有效</strong>。施里弗没有「算得更久」，而是「想得更基础」——这提醒我们，科学突破常常来自「换坐标系」而非「增加精度」。</span>

**BCS 之后的路**：BCS 留下两大未竟事业——(1) 强耦合修正（Eliashberg，已解决）；(2) 高温超导机制（铜氧化物 d 波、铁基 s$_\pm$，未解决）。**BCS 的框架被证明是「可推广」的，但推广的极限——磁性机制、强关联——至今仍是前沿**。理解 BCS 基态，就是站在这条未竟之路的起点。

## 8 小结

- **BCS 基态** $|\text{BCS}\rangle = \prod_{\boldsymbol{k}}(u_{\boldsymbol{k}} + v_{\boldsymbol{k}}b_{\boldsymbol{k}}^{\dagger})|0\rangle$：全体配对态共享相位的相干态，粒子数有涨落但相对涨落消失。
- 变分解 $v_{\boldsymbol{k}}^2 = (1 - \xi_{\boldsymbol{k}}/E_{\boldsymbol{k}})/2$：配对集中在费米面附近薄壳。
- **Bogoliubov 变换**引入准粒子 $\gamma_{\boldsymbol{k}\sigma}$（电子-空穴叠加），使哈密顿量对角化，激发能 $E_{\boldsymbol{k}} = \sqrt{\xi_{\boldsymbol{k}}^2 + \Delta^2}$。
- 能隙 $\Delta$ 是自洽解，$\Delta_0 \approx 2\hbar\omega_D e^{-1/N(0)V}$；普适比值 $2\Delta_0/k_BT_c \approx 3.53$。
- 凝聚能 $N(0)\Delta_0^2/2$ 与唯象 $H_c^2/8\pi$ 一致，BCS 与 GL 就此统一。

在下一节，我们深入研究能隙方程本身：$\Delta(T)$ 如何随温度演化、如何解出临界温度 $T_c$、以及 $T_c$