---
title: 环同态基本定理与同构定理
date: 2026-08-07
---

# 环同态基本定理与同构定理

<div class="epigraph">
<p>群论的同构定理在环论里重演——只把「正规子群」换成「理想」，「商群」换成「商环」。</p>
<footer>—— 自 题（环同构定理笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§8.4 ｜ 2026-08-07</p>
</div>

## 为什么从环同态基本定理开始

第四篇我们证明了群同态基本定理 $G/\ker f \cong \operatorname{Im} f$，随后配齐了三个同构定理。环论完全复刻这套结构——**环同态基本定理** $R/\ker \varphi \cong \operatorname{Im} \varphi$ 与三个环同构定理，只需把「正规子群」换成「理想」、「商群」换成「商环」。这不是巧合，而是「同态-核-商」三件套在所有代数结构上重演的证据。

环同态基本定理是商环与同态的汇合点：它说明「每个环同态都是一个商环的嵌入」，也给出「证明 $R/I \cong S$」的标准流程。本节把环同态基本定理与三个同构定理讲透，并展示它们在「$\mathbb{Z}[x]/\langle x^2+1\rangle \cong \mathbb{Z}[i]$」这类构造中的实际用法。

## 1 环同态基本定理

**定理（环同态基本定理 / First Isomorphism Theorem for Rings）：** 设 $\varphi : R \to S$ 是环同态，则

1. $\ker \varphi$ 是 $R$ 的理想（第七篇与理想篇已证）；
2. $\operatorname{Im} \varphi$ 是 $S$ 的子环；
3. **$R / \ker \varphi \cong \operatorname{Im} \varphi$**，诱导同构为

$$
\bar{\varphi}(a + \ker \varphi) = \varphi(a)
$$

**证明**：加法的群同态基本定理给出 $R/\ker\varphi \cong \operatorname{Im}\varphi$（加法群的同构）；剩下的工作是验证这个同构**还保持乘法**：

$$
\bar\varphi\big((a + I)(b + I)\big) = \bar\varphi(ab + I) = \varphi(ab) = \varphi(a)\varphi(b) = \bar\varphi(a+I)\bar\varphi(b+I)
$$

每一步都是定义代入——商环乘法、诱导映射、$\varphi$ 保乘、再拆回。$\blacksquare$<span class="marginnote">环同态基本定理的证明几乎「免费」：加法层面的群同态基本定理已经把同构造好，只需再检查它保持乘法（一步代入）。「核是理想」是全部前提——若 $\ker\varphi$ 只是子环，商环根本不存在。理想与正规子群的对偶，在这里完成闭环。</span>

**例：**
- $\varphi : \mathbb{Z} \to \mathbb{Z}_n$，$\varphi(k) = k \bmod n$：$\ker \varphi = n\mathbb{Z}$，$\mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}_n$；
- $\varphi_x : \mathbb{R}[x] \to \mathbb{R}$，$\varphi_x(f) = f(x)$（代入）：$\ker \varphi_x = \langle x - a \rangle$（$f(a) = 0 \iff x - a \mid f$），$\mathbb{R}[x]/\langle x - a\rangle \cong \mathbb{R}$；
- $\varphi : \mathbb{Z}[x] \to \mathbb{Z}[i]$，$\varphi(f) = f(i)$（代入 $i$）：$\ker \varphi = \langle x^2 + 1 \rangle$，$\mathbb{Z}[x]/\langle x^2 + 1\rangle \cong \mathbb{Z}[i]$——高斯整数从商环长出来。

## 2 环的第二同构定理

**定理（第二同构定理）：** 设 $S \le R$（子环），$I \trianglelefteq R$（理想），则

$$
(S + I) / I \ \cong \ S / (S \cap I)
$$

**证明**：取 $\varphi : S \to (S+I)/I$，$\varphi(s) = s + I$。满射（$s + i + I = s + I$，$i \in I$ 被吸收），核为 $S \cap I$，套第一同构定理。$\blacksquare$<span class="marginnote">第二同构定理在环里与群里完全平行：$(S + I)/I \cong S/(S \cap I)$。直觉是「$(S + I)$ 里把 $I$ 压掉，剩下的就是 $S$ 压掉它与 $I$ 的交」。$S + I$ 是「$S$ 与 $I$ 的并生成的子环」——注意 $S + I$ 是子环，因为 $I$ 是理想（吸收性保证 $S I \subseteq I$）。</span>

**例：** $R = \mathbb{Z}$，$S = m\mathbb{Z}$，$I = n\mathbb{Z}$。则 $S + I = \gcd(m,n)\mathbb{Z}$，$S \cap I = \mathrm{lcm}(m,n)\mathbb{Z}$，第二同构定理给出

$$
\frac{\gcd(m,n)\mathbb{Z}}{n\mathbb{Z}} \cong \frac{m\mathbb{Z}}{\mathrm{lcm}(m,n)\mathbb{Z}}
$$

两边都是循环群（环）——一个 gcd/lcm 恒等式，被同构定理「免费」复刻。

## 3 环的第三同构定理与对应定理

**定理（第三同构定理）：** 设 $I \trianglelefteq R$，$J$ 是含 $I$ 的理想（$I \le J \trianglelefteq R$），则

$$
(R / I) \big/ (J / I) \ \cong \ R / J
$$

**证明**：$\varphi : R/I \to R/J$，$\varphi(r + I) = r + J$。良定义（$r' - r \in I \subseteq J$），核为 $J/I$，套第一同构定理。$\blacksquare$——「先商 $I$ 再商 $J/I$」等于「一步商 $J$」，与群论完全一致。<span class="marginnote">第三同构定理的直觉是「连续取模 = 一次取模」：$\mathbb{Z}/6\mathbb{Z}$ 再商掉 $2\mathbb{Z}/6\mathbb{Z}$，等于 $\mathbb{Z}/2\mathbb{Z}$。「$(R/I)/(J/I) = R/J$」像分数相消——同构定理的「分数算术」味道在环里同样浓郁。</span>

**对应定理（Correspondence Theorem for Rings）：** 映射 $J \mapsto J/I$ 建立了「$R$ 中含 $I$ 的理想」与「$R/I$ 的全部理想」之间的一一对应，且保持包含与「理想性」：

$$
J \text{ 是 } R \text{ 的理想（含 } I \text{）} \iff J/I \text{ 是 } R/I \text{ 的理想}
$$

**例：** $\mathbb{Z}$ 中含 $n\mathbb{Z}$ 的理想是 $d\mathbb{Z}$（$d \mid n$），对应 $\mathbb{Z}_n$ 的理想 $\langle \bar d \rangle$——「$\mathbb{Z}_n$ 的理想按 $n$ 的因子排列」从对应定理长出来。<span class="marginnote">对应定理是「看商环的内部结构」的工具：$R/I$ 的理想恰是「$R$ 里夹在 $I$ 与 $R$ 之间的理想」。它让「$R/I$ 是域吗」变成「$I$ 是极大理想吗」（第六节）——极大理想与素理想的全部理论都建立在对应定理上。</span>

## 4 公式解析：判定 R/I ≅ S 的标准流程

环同态基本定理最常见的用途是「证明商环同构于某个环」。给出标准四步流程。

**第一步，构造满同态。** 找一个环同态 $\varphi : R \to S$，目标是 $\operatorname{Im}\varphi = S$（满射）且 $R/\ker\varphi$ 恰好是想要的商环。

**第二步，算核。** 确定 $\ker\varphi = \{ r \mid \varphi(r) = 0 \}$，通常写成主理想 $\langle \text{某关系} \rangle$。

**第三步，套定理。** $R/\ker\varphi \cong \operatorname{Im}\varphi$，即「商环 $\cong$ 像」。

**第四步，翻译成目标。** 若 $\ker\varphi = I$ 且 $\operatorname{Im}\varphi = S$，则 $R/I \cong S$。$\blacksquare$

**例：$\mathbb{R}[x]/\langle x^2 + 1 \rangle \cong \mathbb{C}$ 用流程重证。** ① 取 $\varphi : \mathbb{R}[x] \to \mathbb{C}$，$\varphi(f) = f(i)$（代入 $i$），满射（$a + bi = \varphi(a + bx)$）；② 核：$f(i) = 0 \iff x^2 + 1 \mid f$（$x^2 + 1$ 是 $f$ 的因子，因为 $f$ 有根 $i$ 且有实系数则也有根 $-i$），$\ker\varphi = \langle x^2 + 1\rangle$；③ 定理给出 $\mathbb{R}[x]/\langle x^2 + 1\rangle \cong \mathbb{C}$。$\blacksquare$<span class="marginnote">这套「构造同态 → 算核 → 套定理」的流程与群论完全同型，是环论里「证明同构」的标准武器。做题时的瓶颈通常在第二步（算核）——它常常归结为「求值同态的核是某个主理想」，需要用到整除性（$f(a) = 0 \iff x - a \mid f$，第九篇带余除法）。</span>

## 5 例子：商环的「同构谱系」

用同态基本定理把几个关键商环的同构一次清点。

| 商环 | 同构于 | 同态（核） |
| --- | --- | --- |
| $\mathbb{Z}/n\mathbb{Z}$ | $\mathbb{Z}_n$ | $k \mapsto k \bmod n$（$n\mathbb{Z}$） |
| $\mathbb{R}[x]/\langle x - a\rangle$ | $\mathbb{R}$ | 代入 $a$（$\langle x-a\rangle$） |
| $\mathbb{R}[x]/\langle x^2 + 1\rangle$ | $\mathbb{C}$ | 代入 $i$（$\langle x^2+1\rangle$） |
| $\mathbb{Z}[x]/\langle x^2 + 1\rangle$ | $\mathbb{Z}[i]$ | 代入 $i$（$\langle x^2+1\rangle$） |
| $\mathbb{F}_2[x]/\langle x^2 + x + 1\rangle$ | $\mathbb{F}_4$ | —（不可约多项式） |
| $\mathbb{R}[x]/\langle x^2 \rangle$ | 对偶数环 | —（$x^2 = 0$ 的关系） |

**观察**：同样是「代入」，不同环里核的形态不同；同样是「商掉二次多项式」，$\mathbb{R}$ 上造出 $\mathbb{C}$、$\mathbb{F}_2$ 上造出 $\mathbb{F}_4$。<span class="marginnote">表格的最后一列揭示「商环的配方」：商掉 $\langle f \rangle$ 等于「强制 $f = 0$」。$\langle x^2 \rangle$ 强制 $x^2 = 0$，得到对偶数（形式导数的舞台）；$\langle x^2 + 1\rangle$ 强制 $x^2 = -1$，得到复数。<strong>商环是「按方程取模」的机器，同态基本定理是这台机器的使用手册。</strong></span>

## 6 小结

- **环同态基本定理**：$R/\ker\varphi \cong \operatorname{Im}\varphi$，诱导同构 $\bar\varphi(a + I) = \varphi(a)$。
- **第二同构定理**：$(S + I)/I \cong S/(S\cap I)$。
- **第三同构定理**：$(R/I)/(J/I) \cong R/J$。
- **对应定理**：含 $I$ 的理想 ↔ $R/I$ 的理想，保持理想性。
- **标准流程**：构造满同态 → 算核 → 套定理 → 翻译；$\mathbb{R}[x]/\langle x^2+1\rangle \cong \mathbb{C}$ 是旗舰例子。

在下一节，我们追问「商环何时是域/整环」：**极大理想与素理想**。对应定理把这个问题翻译成「理想何时是极大/素」——域与整环由此从理想世界重新长出来。
