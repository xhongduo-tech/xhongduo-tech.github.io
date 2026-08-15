---
title: 诱导表示与 Frobenius 互反律
date: 2026-08-07
---

# 诱导表示与 Frobenius 互反律

<div class="epigraph">
<p>诱导表示是表示论的「升降机」：从子群升到大群，从大群降回子群。</p>
<footer>—— 威廉 · 富尔顿 与 乔 · 哈里斯（Fulton & Harris）</footer>
</div>

<div class="article-byline">
<p>第二级 · 群表示论 ｜ Serre《有限群的线性表示》§7.1–7.2 ｜ 2026-08-07</p>
</div>

## 为什么从诱导表示开始

到目前为止，我们研究的是一个固定群 $G$ 的内部结构。但表示论最有力的工具之一，恰恰是**跨群**的：给定子群 $H \le G$ 的一个表示，怎么「升级」成 $G$ 的表示？反过来，$G$ 的表示怎么「降级」回 $H$？前者叫**诱导（induction）**，后者叫**限制（restriction）**。二者之间由一条漂亮的对偶关系——**Frobenius 互反律**——连接。它是构造对称群表示、紧群表示、乃至模表示论的基础设施。

从「从极限到大模型」主线看，诱导与限制是「改变分辨率」的数学原型。<span class="marginnote">把「在子群上的表示」升级成「在群上的表示」，就像把「细粒度的局部信息」整合成「全局信息」；限制则相反，是「只看局部」。多尺度分解（如图像金字塔、小波）的思路与诱导/限制同构：从细到粗、从粗到细。</span>而在物理中，诱导表示描述「粒子如何从保持对称性的子群升到整个对称群」——量子场论里从 $SO(3)$ 的子群表示构造整体表示，用的正是诱导。<span class="marginnote">经典例子：从旋转群子群 $SO(2)$ 的「一维相位表示」诱导出 $SO(3)$ 的球谐函数表示。诱导表示 = 从局部对称性「推广」到全局对称性。</span>

## 1 限制：先学会「降维」

**限制（restriction）**：设 $\rho : G \to \mathrm{GL}(V)$ 是 $G$ 的表示，$H \le G$ 是子群。把 $\rho$ 只看在 $H$ 上，得到 $H$ 的表示

$$\mathrm{Res}_H^G \rho : H \to \mathrm{GL}(V), \qquad g \mapsto \rho(g) \ (g \in H)$$

特征标也相应限制：$\chi_{\mathrm{Res}}(h) = \chi(h)$ 对 $h \in H$。<span class="marginnote">限制是「傻瓜操作」：什么都不变，只是把定义域从 $G$ 缩到 $H$。一个 $G$ 的不可约表示限制到 $H$ 上通常<strong>不再不可约</strong>，而是分解成若干 $H$-不可约表示——这种分解在物理中叫「能级在对称性破缺下的分裂」，是晶体场理论的核心。</span>

限制的特征标内积满足自然性质：对 $H$ 上的类函数 $\varphi$，$\langle \varphi, \chi_{\mathrm{Res}} \rangle_H$ 有直接意义，但更重要的关系在下一节。

## 2 诱导：用张量积「升级」

**诱导表示（induced representation）**：设 $H \le G$，$(\sigma, W)$ 是 $H$ 的表示。定义 $G$ 的表示

$$\mathrm{Ind}_H^G W = \mathbb{C}[G] \otimes_{\mathbb{C}[H]} W$$

其中 $\mathbb{C}[G]$ 视作右 $\mathbb{C}[H]$-模（通过右乘），$W$ 视作左 $\mathbb{C}[H]$-模（通过 $\sigma$）；$G$ 通过**左乘**作用在 $\mathbb{C}[G]$ 上（保张量积关系）。<span class="marginnote">「用张量积定义」一句话就能说清：把子群的表示「用群代数撑大」。若 $G = \bigsqcup_{i=1}^{[G:H]} t_i H$ 是陪集分解，则 $\mathrm{Ind}_H^G W = \bigoplus_i t_i \otimes W$，维数乘以陪集数：$\dim \mathrm{Ind} = [G:H] \cdot \dim W$。每个陪集「搬」一份 $W$。</span>

等价地，诱导表示可以这样理解：取陪集代表元 $t_1, \dots, t_r$（$r = [G:H]$），则

$$\mathrm{Ind}_H^G W = t_1 W \oplus t_2 W \oplus \cdots \oplus t_r W$$

群元 $g$ 通过「把 $t_i W$ 搬到 $t_j W$（$g t_i = t_j h$，其中 $h \in H$，然后 $h$ 按 $\sigma$ 作用）」实现它的作用。<span class="marginnote">这个「陪集搬家」的直观版本更容易用：$G$ 作用在每个陪集块上，就像置换作用，只是块内还要再按 $\sigma(h)$ 转动。$H = \{e\}$ 时，诱导表示就是「把平凡表示升级」——恰好得到置换表示。</span>

**诱导特征标的 Frobenius 公式**：对 $g \in G$，

$$\chi_{\mathrm{Ind}}(g) = \frac{1}{|H|} \sum_{\substack{t \in G \\ t^{-1} g t \in H}} \chi(t^{-1} g t)$$

即把 $g$ 的所有「共轭到 $H$ 内」的形变 $t^{-1}gt$ 的特征标求和再平均。<span class="marginnote">这条公式是计算诱导特征标的实用工具：只需要在子群 $H$ 上算特征标，再按「$g$ 的哪些共轭落回 $H$」加权求和。它把「大群上的计算」归约为「子群上的计算」。</span>

## 3 Frobenius 互反律

**Frobenius 互反律（Frobenius reciprocity）**：设 $\sigma$ 是 $H$ 的表示、$\rho$ 是 $G$ 的表示，则内积满足

$$\langle \mathrm{Ind}_H^G \chi_\sigma, \ \chi_\rho \rangle_G = \langle \chi_\sigma, \ \mathrm{Res}_H^G \chi_\rho \rangle_H$$

即「诱导的特征标在 $G$ 上与 $\rho$ 的内积」等于「$\sigma$ 在 $H$ 上与 $\rho$ 之限制的内积」。<span class="marginnote">一句话：<strong>诱导是限制的伴随函子</strong>（adjoint functor）：$\mathrm{Hom}_G(\mathrm{Ind}, \rho) \cong \mathrm{Hom}_H(\sigma, \mathrm{Res})$。作为线性算子的「伴随」，把重数从一边搬到另一边——这是范畴论「伴随性」在表示论里的第一次自然出场。</span>

互反律的用法：想知道「$\mathrm{Ind}_H^G \sigma$ 里含 $\rho$ 几块」，只需算「$\sigma$ 在 $\rho|_H$ 里含几块」——后者是小子群上的计算，通常简单得多。<span class="marginnote">例：$S_3$ 中取 $H = \langle (12) \rangle \cong \mathbb{Z}/2$。把 $H$ 的符号表示 $\varepsilon_H$ 诱导到 $S_3$，用互反律与 $S_3$ 的特征标表可得 $\mathrm{Ind} \cong \mathbf{1} \oplus \varepsilon$。整个计算只用到两行特征标值。</span>

**辨析｜易错点：** 诱导与限制的方向极易记反。**诱导 $\mathrm{Ind}_H^G$ 是把 $H$ 的表示变成 $G$ 的表示（下标是子群、上标是大群）；限制 $\mathrm{Res}_H^G$ 是反方向。** 口诀：「$\mathrm{Ind}$ 往上走、$\mathrm{Res}$ 往下走」。互反律里，$G$ 一侧的表示出现在诱导里、$H$ 一侧的表示出现在限制里，两边角色不能换位。

## 4 公式解析：Frobenius 互反律的展开

把互反律展开成重数语言。设 $\sigma$ 不可约（$H$ 上）、$\rho$ 不可约（$G$ 上），$m_{\mathrm{Ind}}(\rho)$ 表示「$\rho$ 在 $\mathrm{Ind}_H^G \sigma$ 中的重数」，$n_{\mathrm{Res}}(\sigma)$ 表示「$\sigma$ 在 $\mathrm{Res}_H^G \rho$ 中的重数」，则

$$m_{\mathrm{Ind}}(\rho) = \langle \chi_{\mathrm{Ind}}, \chi_\rho \rangle_G = \langle \chi_\sigma, \chi_{\mathrm{Res}} \rangle_H = n_{\mathrm{Res}}(\sigma)$$

- **第一步，读左边**：$m_{\mathrm{Ind}}(\rho)$ 用第一正交关系写成内积 $\langle \chi_{\mathrm{Ind}}, \chi_\rho \rangle_G$——这是「重数 = 投影系数」的标准用法。
- **第二步，读桥**：Frobenius 互反律把 $G$ 上的内积换成 $H$ 上的内积。求和范围从「整个 $G$」缩到「子群 $H$」，$|G|$ 换成 $|H|$ 做归一化——这是「换一个空间算同一个投影」。
- **第三步，读右边**：$\langle \chi_\sigma, \chi_{\mathrm{Res}} \rangle_H$ 恰好是「$\sigma$ 在 $\rho$ 限制中的重数」。所以互反律断言：**「往上诱导后再看 $\rho$ 含几块」=「先往下限制再看 $\sigma$ 含几块」**。
- **第四步，读伴随性**：重数的两个方向相等，说明诱导与限制互为伴随。这一视角让互反律的证明只需一行（张量积的 Hom-张量伴随），也让它从「一条公式」升华为「一种结构」。

**辨析｜易错点：** 重数相等不是「表示相等」。$\mathrm{Ind}_H^G \sigma$ 与 $\rho$ 之间的关系是「含几块」，不是「等于」；同样 $\mathrm{Res}_H^G \rho$ 与 $\sigma$ 是「含几块」。互反律说的是两个**数字**相等。若写成「$\mathrm{Ind}$ 与 $\mathrm{Res}$ 相等」则是范畴错误——它们是群不同、空间不同的两个对象。

## 5 诱导表示的两个经典应用

**应用一（构造群的全部表示）**：对许多群，不可约表示可以由「小子群的表示诱导」组合出来。对超可解群、$p$-群及很多可解群，存在**Mackey 定理**式的「诱导–限制」对，说明如何从「子群的不可约表示」出发拼出大群的不可约表示。<span class="marginnote">对称群 $S_n$ 的研究（下一篇）很大程度上建立在这种「从 $S_{n-1}$ 到 $S_n$ 逐层诱导」的框架上：杨图的分支规则正是限制/诱导在 $S_{n-1} \hookrightarrow S_n$ 上的分解规律。</span>

**应用二（Burnside 引理与轨道计数的表示论版本）**：若 $H = \{e\}$，则 $\mathrm{Ind}_{\{e\}}^G \mathbf{1}$ 是置换表示（$G$ 作用于自身右乘），它的不变子空间维数等于轨道数。把「共轭类的计数」写成诱导表示的特征标值，可以统一 Burnside 引理与类方程。<span class="marginnote">这条线通向群作用理论的表示论再表述：不动点计数、轨道计数都可以用「诱导特征标在某元素处的值」表达，从而被内积工具统一处理。</span>

## 6 例：从 $\mathbb{Z}/2$ 诱导出 $S_3$ 的表示

把 Frobenius 公式与互反律合起来算一个完整例子。取 $H = \{e, s\}$，$s = (12)$，$H \cong \mathbb{Z}/2$。$H$ 有两个一维不可约表示：平凡 $\mathbf{1}_H$ 与符号 $\varepsilon_H$（$s \mapsto -1$）。

**诱导平凡表示。** $\mathrm{Ind}_H^G \mathbf{1}_H$ 的维数 $= [G:H] \cdot 1 = 3$。用 Frobenius 公式算特征标：$g = e$ 时，所有 $t$ 都使 $t^{-1}et = e \in H$，贡献 $6$ 项各 $1$，除以 $|H| = 2$ 得 $\chi(e) = 3$；$g = s$ 时，$t^{-1}st \in H$ 当且仅当 $t^{-1}st = s$，即 $t$ 属于中心化子 $\{e, s\}$，两项各 $1$，得 $\chi(s) = 1$；$g = t$（三轮换）时无共轭落入 $H$，得 $\chi(t) = 0$。于是 $\chi = (3, 1, 0)$——正是置换表示的特征标。<span class="marginnote">这印证了几何直观：「从平凡表示诱导」=「$G$ 在陪集 $G/H$ 上的置换表示」。$[G:H] = 3$ 个陪集，$G$ 置换它们，特征标自然等于置换表示。</span>

**诱导符号表示。** $\varepsilon_H(s) = -1$，同样的公式给出 $\chi_{\mathrm{Ind}} = (3, -1, 0)$。用互反律（或直接内积）分解：

$$\langle \chi_{\mathrm{Ind}}, \chi_{\mathbf 1} \rangle = \tfrac16(3 - 3) = 0, \qquad \langle \chi_{\mathrm{Ind}}, \chi_{\varepsilon} \rangle = \tfrac16(3 + 3) = 1, \qquad \langle \chi_{\mathrm{Ind}}, \chi_{\mathrm{std}} \rangle = \tfrac16(6 + 0) = 1$$

故 $\mathrm{Ind}_H^G \varepsilon_H \cong \varepsilon \oplus \mathrm{std}$。两条诱导的分解与 $S_3$ 的特征标表完全吻合，Frobenius 互反律在这类一维子群情形成了「核对重数」的利器。

## 7 Mackey 分解定理：诱导的再分解

诱导与限制反复使用时会遇到一个问题：$\mathrm{Res}$ 之后再 $\mathrm{Ind}$（或反过来）如何分解？**Mackey 分解定理**给出精确答案：对子群 $K \le G$，

$$\mathrm{Res}_K^G\, \mathrm{Ind}_H^G W \cong \bigoplus_{KgH} \mathrm{Ind}_{K \cap gHg^{-1}}^K\, \big({}^g W\big)$$

其中求和跑遍双陪集 $KgH$，${}^g W$ 是 $W$ 经 $g$ 共轭后的表示。<span class="marginnote">双陪集 $KgH = \{kgh\}$ 是「同时按左右两个子群划分」的块，个数一般少于陪集数。Mackey 公式说：限制-诱导的复合，等于「沿双陪集逐块再诱导」。</span>

当 $K = H$ 且双陪集只有一个时，$\mathrm{Res}\circ\mathrm{Ind}$ 保持简单；多双陪集时则出现「共轭子群表示」的求和。Mackey 定理是诱导表示可计算性的基石，下一篇对称群的分支规则正是它在 $S_{n-1} \hookrightarrow S_n$ 上的具体形态——每个「删角格」对应一个双陪集块。

**诱导特征标公式的自检。** Frobenius 公式求和的条件「$t^{-1}gt \in H$」实质上是在数「$g$ 的哪些共轭落入 $H$」。对 $g = e$，条件恒成立，共 $|G|$ 项各贡献 $\chi(e)$，除以 $|H|$ 得 $\chi_{\mathrm{Ind}}(e) = [G:H]\,\chi(e)$——正是「维数乘以陪集数」，与张量积定义一致。任何诱导计算都应从这里自查：单位元处的值必须等于 $[G:H] \cdot \dim \sigma$。

**一处思想注记。** 诱导表示是表示论中「由局部到整体」的典型操作，其思想广泛见于：群作用理论（轨道–稳定子定理）、纤维丛的截面构造、以及物理学里「从子群对称性构造整体对称性」的所谓「诱导表示法」。Frobenius 互反律的伴随性视角，更让它在范畴论中获得自然位置——伴随对几乎总能翻译成「限制—诱导」。

诱导表示在模表示论中同样关键：当 $[G:H]$ 与域特征互素时，诱导–限制在模表示中保持良好性质，这是 Brauer 理论的一条支柱。

## 8 小结

- **限制** $\mathrm{Res}_H^G$：把 $G$ 的表示「只看在 $H$ 上」，特征标直接取值；$G$-不可约通常裂成 $H$-不可约的组合。
- **诱导** $\mathrm{Ind}_H^G$：$\mathbb{C}[G] \otimes_{\mathbb{C}[H]} W$，维数乘以陪集数 $[G:H]$；陪集块「搬家」给出显式作用。
- **Frobenius 公式**：$\chi_{\mathrm{Ind}}(g) = \frac{1}{|H|}\sum_{t^{-1}gt \in H} \chi(t^{-1}gt)$，把大群计算归约为子群计算。
- **Frobenius 互反律**：$\langle \mathrm{Ind}, \rho \rangle_G = \langle \sigma, \mathrm{Res} \rangle_H$；诱导与限制互为伴随，重数可双向计算。
- 方向口诀：**$\mathrm{Ind}$ 向上、$\mathrm{Res}$ 向下**；互反律是「重数相等」，不是「表示相等」。

在下一节，我们把表示论的全部工具用到最经典也最深刻的群上：**对称群 $S_n$**。它的不可约表示由杨图（Young diagrams）与对称化子（Specht modules）刻画，而诱导表示正是构造它们的引擎——从平凡表示一步步诱导出 $S_n$ 的全部不可约表示，并借 Frobenius 互反律与 Burnside 的轨道计数接通。
