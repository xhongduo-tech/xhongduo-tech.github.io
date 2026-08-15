---
title: 过渡态理论（Eyring 方程）
date: 2026-08-07
---

# 过渡态理论（Eyring 方程）

<div class="epigraph">
<p>新的科学真理往往不是靠说服反对者来获胜，而是靠反对者终将死去、新一代熟悉它的学者成长起来。</p>
<footer>—— 马克斯 · 普朗克（Max Planck）</footer>
</div>

<div class="article-byline">
<p>第二级 · 化学动力学（深化） ｜ Atkins & de Paula《Physical Chemistry》Focus 18 ｜ 2026-08-07</p>
</div>

## 为什么从过渡态理论开始

Arrhenius 方程用 $A$ 与 $E_{\mathrm{a}}$ 两个经验参数装下了全部温度依赖，但留下两个悬案：指前因子 $A$ 的分子本质是什么？活化能垒上的分子到底长什么样？1935 年，艾林（Henry Eyring）与埃文斯-波兰尼（Evans & Polanyi）几乎同时给出了答案——**过渡态理论（transition state theory，TST）**。<span class="marginnote">TST 是现代化学动力学的理论中枢：它把反应速率归结为「活化络合物的平衡浓度 × 它穿过分隔面的频率」，而这两者都可用统计力学与量子化学计算。今天 AI 驱动的化学，很大一部分就是在用机器学习势能面喂给 TST 及其量子化版本。</span>这一篇我们从势能面出发，走到 Eyring 方程，并厘清它与 Arrhenius 方程的亲缘与差别。

## 1 势能面与鞍点

一个双分子反应的体系能量是全部核坐标的函数——**势能面（potential energy surface，PES）**。反应物谷与产物谷之间隔着一道「山脊」，最低处的一个山口就是**鞍点（saddle point）**，对应**过渡态（transition state）**的构型。<span class="marginnote">「鞍点」这个词来自马鞍：沿一个方向它是峰（沿反应坐标），沿垂直方向它是谷（振动坐标）。这正是「活化络合物在反应坐标方向不稳定、在其他方向稳定」的几何语言。</span>

把势能面沿反应坐标切开，就得到能量剖面图——一条从反应物谷爬上鞍点再滑向产物谷的曲线，峰值即过渡态。下图左是二维势能面的等高线图，反应路径（虚线）恰好穿过鞍点：

![二维势能面与鞍点](/images/chemical-kinetics/pes-saddle-point.svg)

**过渡态理论的三大基本假设**：

1. 存在一个分隔面（dividing surface）穿过鞍点，分子一旦穿越即注定生成产物（**不回头假设**，除非发生再碰撞）。
2. 反应物与活化络合物之间保持（准）平衡。
3. 活化络合物沿反应坐标的振动可分离，且其穿过分隔面的速率可当作经典运动处理。

三大假设里最脆弱的是「不回头」：真实反应中分子穿越分隔面后可能被弹回（recrossing），这会系统性高估速率——这正是后文「变分 TST」要修正的缺口。

## 2 Eyring 方程：速率 = 平衡浓度 × 穿越频率

设反应 $\ce{A + B <=> X^{\ddagger} -> 产物}$，其中 $\ce{X^{\ddagger}}$ 是活化络合物。若它与反应物处于平衡，平衡常数 $K^{\ddagger} = \dfrac{[\ce{X^{\ddagger}}]}{[\ce{A}][\ce{B}]}$；活化络合物沿反应坐标以特征频率 $\nu$ 振动，平均而言每次振动有概率穿过分隔面。由此：

$$
k = \frac{k_{\mathrm{B}}T}{h}\,K^{\ddagger}
$$

这就是**Eyring 方程**的基础形态，其中 $\dfrac{k_{\mathrm{B}}T}{h}$ 是普适频率因子，25 °C 时约 $6.2\times10^{12}\,\mathrm{s^{-1}}$。用统计力学把 $K^{\ddagger}$ 换成配分函数之比，得到完整形式：

$$
k = \frac{k_{\mathrm{B}}T}{h}\,\frac{q_{\ddagger}}{q_{\ce{A}}q_{\ce{B}}}\,e^{-E_0/RT}
$$

或用量热力学量写成更便于实验对照的形式：

$$
k = \frac{k_{\mathrm{B}}T}{h}\,e^{\Delta S^{\ddagger}/R}\,e^{-\Delta H^{\ddagger}/RT}
$$

其中 $\Delta H^{\ddagger}$ 是**活化焓**，$\Delta S^{\ddagger}$ 是**活化熵**。<span class="marginnote">活化熵是 TST 独有的「新情报」：它度量活化络合物比反应物更「有序」还是更「无序」。双分子缔合为活化络合物通常 $\Delta S^{\ddagger} < 0$（自由度数变成内振动，熵减）；解离则 $\Delta S^{\ddagger} > 0$。实验上从 Arrhenius 图斜率得 $E_{\mathrm{a}}$，截距再扣除 $T$ 因子即可剥离出 $\Delta S^{\ddagger}$。</span>

## 3 与 Arrhenius 方程的握手

把 Eyring 方程与 Arrhenius 方程 $k = Ae^{-E_{\mathrm{a}}/RT}$ 对照，可建立两套参数的关系：

$$
E_{\mathrm{a}} = \Delta H^{\ddagger} + RT
$$

$$
A = \frac{k_{\mathrm{B}}T}{h}\,e\,e^{\Delta S^{\ddagger}/R}
$$

- **活化能 $E_{\mathrm{a}}$ 与活化焓 $\Delta H^{\ddagger}$ 差一个 $RT$**（约 2.5 kJ/mol 于室温），对绝大多数定量工作可忽略。
- **指前因子 $A$ 的本质是活化熵**：$A$ 远大于「标准频率」$k_{\mathrm{B}}T/h \approx 10^{13}$ 意味着活化络合物比反应物无序（$\Delta S^{\ddagger} > 0$），远小于则意味着过渡态「紧绷」（高度有序）。**这是 TST 对 Arrhenius 方程的最深贡献：给神秘的 $A$ 一个分子解释。**<span class="marginnote">一个经典对比：双分子重排（$\ce{A + B}$）的 $A$ 常在 $10^{10}$ 上下，对应负活化熵；单分子解离的 $A$ 可高达 $10^{15}$，对应正活化熵。从 $A$ 的数量级就能猜出反应的分子性——这正是实验动力学用来「反推机理」的快速判据。</span>

**辨析｜易错点：** TST 的平衡假设与真实快反应之间总有偏差，当反应穿越分隔面后又被弹回（recrossing），真实速率会低于 Eyring 预言——这就是「变分过渡态理论」要修正的缺口。另一个易错点：$\Delta H^{\ddagger}$ 不是反应焓变 $\Delta H^\circ$，前者是「到鞍点的焓差」，后者是「到产物谷的焓差」。

## 4 公式解析：$k = \dfrac{k_{\mathrm{B}}T}{h} K^{\ddagger}$

把 Eyring 方程拆成四步理解：

- **第一步，$\dfrac{k_{\mathrm{B}}T}{h}$ 是穿越频率**。活化络合物沿反应坐标的振动可看作「试探穿越」：每振动一次就以一定概率跨过分隔面。频率的量级由热运动给出（$kT/h$），它不依赖具体分子——这是「普适频率因子」名字的由来。
- **第二步，$K^{\ddagger}$ 是「成事概率」**。它衡量体系中「处于鞍点、准备穿越」的分子比例。$K^{\ddagger}$ 越小，能到达鞍点的分子越少，速率越慢。
- **第三步，把两者相乘**。速率 = 尝试穿越的频率 × 到达鞍点的比例，即得 $k$。直观上：**Eyring 方程把「反应速率」写成「振动闹钟」与「爬坡成功率」的乘积**。
- **第四步，联系 Arrhenius**。$K^{\ddagger}$ 携带指数因子 $e^{-\Delta G^{\ddagger}/RT}$，展开为焓项×熵项即得到含 $\Delta S^{\ddagger}$ 的形式；指数因子对应 Arrhenius 的 $E_{\mathrm{a}}$，熵项对应 $A$。

## 5 数值算例：算一笔「普适频率」

- **第一步，代入常数**：$k_{\mathrm{B}} = 1.381\times10^{-23}\,\mathrm{J\,K^{-1}}$、$T = 298\,\mathrm{K}$、$h = 6.626\times10^{-34}\,\mathrm{J\,s}$。
- **第二步，相除**：$\dfrac{k_{\mathrm{B}}T}{h} = \dfrac{1.381\times10^{-23}\times298}{6.626\times10^{-34}} \approx \dfrac{4.116\times10^{-21}}{6.626\times10^{-34}} \approx 6.2\times10^{12}\,\mathrm{s^{-1}}$。
- **第三步，读含义**：这是「每秒振动约 $6\times10^{12}$ 次」——若活化络合物每一振动都成功穿越，速率常数就是这个量级的上限。<span class="marginnote">把这个数与实验测得的 $A$ 对照：若 $A \approx 10^{13}\,\mathrm{s^{-1}}$，说明活化熵接近零、过渡态与反应物无序度相当；若 $A$ 小两个数量级，多半是负活化熵在「惩罚」高有序的过渡态。这条量级直觉，是读动力学文献最快的判据。</span>

## 6 Arrhenius 与 Eyring 参数对照

| 特征 | Arrhenius | Eyring (TST) |
| --- | --- | --- |
| 参数 | $A$、$E_{\mathrm{a}}$ | $\Delta H^{\ddagger}$、$\Delta S^{\ddagger}$ |
| 关系 | $k = Ae^{-E_{\mathrm{a}}/RT}$ | $k = \frac{k_{\mathrm{B}}T}{h}e^{\Delta S^{\ddagger}/R}e^{-\Delta H^{\ddagger}/RT}$ |
| 活化能 | $E_{\mathrm{a}} = \Delta H^{\ddagger} + RT$ | — |
| 指前因子 | 经验常数 | $A = \frac{k_{\mathrm{B}}T}{h}e\,e^{\Delta S^{\ddagger}/R}$ |
| 物理意义 | 拟合参数 | 活化熵、活化焓的统计力学解释 |

## 7 常见误区与辨析

**辨析｜易错点（汇总）：**

- 「TST 假设反应物真的到达过渡态」是简化：TST 假设的是「反应物与活化络合物处于准平衡」，不是「所有反应物都爬到鞍点」。
- 「$\Delta H^{\ddagger}$ = 反应焓变」是错的：前者到鞍点，后者到产物谷，两个不同的能量差。
- 「$A$ 永远约 $10^{13}$」是错的：活化熵可以把 $A$ 拉开好几个数量级，$A$ 是「机理指纹」。
- 「穿越分隔面就注定生成产物」在 recrossing 存在时失效，真实速率低于 Eyring 预言。
- 「TST 完全量子正确」不成立：量子隧穿、零点能修正都会偏离经典 TST，低温下尤其明显。
- 活化熵的符号判据：缔合型 $\Delta S^{\ddagger} < 0$、解离型 $> 0$——从 $A$ 的数量级可反推分子性。

## 8 小结

- **势能面**上的鞍点即**过渡态**；TST 三假设：不回头、准平衡、经典穿越。
- **Eyring 方程** $k = \dfrac{k_{\mathrm{B}}T}{h}K^{\ddagger}$：普适频率因子 × 活化络合物平衡常数。
- **活化熵 $\Delta S^{\ddagger}$** 赋予 Arrhenius 指前因子 $A$ 分子意义；$E_{\mathrm{a}} = \Delta H^{\ddagger} + RT$。
- TST 的局限：穿越后弹回、量子隧穿等，催生了变分 TST 与量子化修正。
- 数值直觉：$\dfrac{k_{\mathrm{B}}T}{h} \approx 6\times10^{12}\,\mathrm{s^{-1}}$ 是速率常数的「振动闹钟」上限。
- 关联阅读：TST 的活化络合物与《反应机理与中间体》的过渡态概念同源；统计力学的配分函数视角在第三级《统计力学》中展开。

在下一节，我们把 TST 的「平衡假设」拆掉一根柱子：单分子反应明明只有一个分子，它的「碰撞」从何而来？这是 **单分子反应理论（Lindemann/RRKM）**。
