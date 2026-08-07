---
title: 逻辑函数的公式化简法
date: 2026-08-07
---

# 逻辑函数的公式化简法

<div class="epigraph">
<p>削繁就简，乃数学之灵魂。</p>
<footer>—— 亨利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数字逻辑 ｜ 阎石《数字电子技术基础》 第 2 章 §2.7 ｜ 2026-08-07</p>
</div>

## 为什么从公式化简开始

一个逻辑式写出来不等于设计完成。同样实现一个功能，`AB + A\overline{B}` 要两个与门加一个或门，而化简后的 `A` 只需要一根导线——**化简就是省钱**：元器件少、连线少、功耗低、故障点少。<span class="marginnote">工业上，化简的收益直接反映在成本上：一块芯片上少用一千个门，晶圆面积就小一圈，良率与单价双双改善。历史上从 SSI 时代起，「化简逻辑式」就是数字设计师的每日功课。</span>公式化简法是第一件武器，它靠上一节的常用公式对逻辑式做等价变形，直到无法再简。

## 1 化简的目标与判据

数字电路里最常用的逻辑式是**与-或式**（先与后或），化简的目标是得到**最简与-或式**。什么叫最简？两条标准缺一不可：

1. **乘积项最少**（与门数量最少）；
2. **每个乘积项的因子最少**（每个与门的输入端最少）。

例如 $Y = AB + A\overline{B}$，两个乘积项（对应两个与门），每项两个因子；并项后 $Y = A$，一个乘积项、一个因子，显然更省。<span class="marginnote">注意「与-或最简」不是唯一的——有时存在多个同样简练的等价式（如 $AB+\overline{A}C$ 与 $AC+\overline{A}B$ 在某些条件下等价）。教材以「与-或最简」为基准，卡诺图化简也是围绕这个目标。</span>

**辨析｜易错点：** 「最简与-或式」与「标准与-或式（最小项之和）」是两回事。前者是最简的代数形式，后者是含全部变量的规范形式。化简的目的正是**离开**臃肿的标准形式，走向节省的实现。

## 2 四种基本化简手法

公式化简法说穿了只有四种手法，全部由常用公式支撑：

**① 并项法**——利用 $AB + A\overline{B} = A$ 把两项并为一项：

$$Y = A\overline{B}C + ABC = AC(\overline{B}+B) = AC$$

**② 吸收法**——利用 $A + AB = A$ 消去多余项：

$$Y = \overline{A}B + \overline{A}B\overline{C} = \overline{A}B$$

**③ 消因子法**——利用 $A + \overline{A}B = A + B$ 消去多余因子：

$$Y = A + \overline{A}B + \overline{A}C = A + B + C$$

**④ 配项法**——利用互补律 $A + \overline{A} = 1$ 或 $A + A = A$ 先添加冗余项，再配合其他手法化简。这一手最需技巧：

$$Y = AB + \overline{A}C + BC = AB + \overline{A}C + BC(A + \overline{A}) = AB + \overline{A}C$$

注意最后一步：$BC(A+\overline{A}) = ABC + \overline{A}BC$，而 $ABC$ 被 $AB$ 吸收、$\overline{A}BC$ 被 $\overline{A}C$ 吸收——**冗余项先加后消，为吸收创造条件**。<span class="marginnote">配项法是公式化简里唯一「先变复杂再变简单」的手法，也是最难的一步。它没有固定套路，但有一条经验：当一项分别与另一项的部分因子形成互补时，试着把缺的变量补全。</span>

## 3 公式解析：一道综合例题

用一道综合题把四种手法串起来。化简

$$Y = \overline{A}B + \overline{A}\overline{B} + AB + \overline{A}C$$

**第一步，并项前两项**：$\overline{A}B + \overline{A}\overline{B} = \overline{A}(B + \overline{B}) = \overline{A}$。此时

$$Y = \overline{A} + AB + \overline{A}C$$

**第二步，吸收 $\overline{A}$**：$\overline{A} + \overline{A}C = \overline{A}$，得到

$$Y = \overline{A} + AB$$

**第三步，消因子**：$\overline{A} + AB = \overline{A} + B$（用 $A+\overline{A}B=A+B$，令 $A\to\overline{A}$）。

**第四步，验证**：列真值表或代入四组输入，原式与新式逐格一致。最简结果 $Y = \overline{A} + B$，从三项五项简化到两项。<span class="marginnote">最后一步验证绝不能省——公式化简高度依赖技巧，一旦配项或吸收出错，化简结果可能不再与原函数等价。真值表是永不失效的裁判。</span>

## 4 公式化简的局限

公式化简法强大，但有两个致命缺点：

- **不系统**：什么时候用哪种手法、配哪一项，依赖人的经验和灵感，没有机械规则。同一个式子，高手三步化简，新手可能绕半天。
- **不判定最优**：化简到底了没有？有没有更省的写法？公式法无法给出「已到最简」的证明，只能靠感觉。

正是为了克服这两个缺点，下一节引入**卡诺图化简法**——把逻辑式映射到一张方块图上，让「找可并的项」变成「圈相邻的格」，机械、直观、而且一眼看出是否最简。<span class="marginnote">这也是教材的顺序安排：先讲公式法建立代数直觉，再讲卡诺图获得系统化工具。对四变量以内的小函数，卡诺图几乎完胜公式法；对五变量以上，则交给计算机化的 Quine-McCluskey 算法。</span>

## 5 化简的实战：一个完整综合题

把四种手法综合运用于一个五变量题目，检验熟练度。化简：

$$Y = \overline{A}B + AB\overline{C} + \overline{A}BC + A\overline{B}\,\overline{C} + B\overline{C}$$

**第一步，观察公共因子**：前两项 $\overline{A}B + AB\overline{C}$ 无直接公共因子，先放下；后两项 $A\overline{B}\,\overline{C} + B\overline{C}$ 提取 $\overline{C}$：$\overline{C}(A\overline{B} + B)$。

**第二步，化简括号内**：$A\overline{B} + B = A + B$（用消因子法，令 $A \to A\overline{B}$？不——直接观察：$A\overline{B} + B = A\overline{B} + B(A + \overline{A}) = A\overline{B} + AB + \overline{A}B = A + \overline{A}B = A + B$）。

**第三步，代入回原式**：原式变为 $\overline{A}B + AB\overline{C} + \overline{A}BC + \overline{C}(A+B)$。

**第四步，逐项吸收**：$\overline{A}B + \overline{A}BC = \overline{A}B$（吸收）；$AB\overline{C} + \overline{C}A = A\overline{C}$（并项，注意 $\overline{C}A$ 与 $AB\overline{C}$ 合并后 $B$ 被吸收）；整理得 $Y = \overline{A}B + A\overline{C} + B\overline{C}$。

**第五步，再消冗余**：$B\overline{C}$ 是 $\overline{A}B$ 与 $A\overline{C}$ 的冗余项（用公式四），可去掉：

$$Y = \overline{A}B + A\overline{C}$$

**第六步，验证**：代入任意一组输入（如 $A=0,B=1,C=1$）：原式各项与化简式都应得 1——手工列真值表确认。化简从五项三因子压到两项两因子，节省了两个门。

这道题示范了「并项 → 吸收 → 消冗余」的组合拳：公式化简不是逐项瞎试，而是**先找公共因子、再逐层化简、最后检查冗余**。

## 6 化简结果的形式选择

化简的最终结果不一定是「与-或式」——按实现电路选形式，是工程上的进阶考量。

**四种常用形式**：

| 形式 | 表达 | 实现门 |
| --- | --- | --- |
| 与-或式 | $Y = AB + \overline{A}C$ | 与门 + 或门 |
| 或-与式 | $Y = (A+B)(\overline{A}+C)$ | 或门 + 与门 |
| 与非-与非式 | $\overline{\overline{AB}\cdot\overline{\overline{A}C}}$ | 只用与非门 |
| 或非-或非式 | $\overline{\overline{A+B}+\overline{\overline{A}+C}}$ | 只用或非门 |

**为什么要有多种形式**：不同逻辑族擅长不同门——TTL 家族以与非门为基本单元，用与非-与非式最省；有些场合只有或非门，就用或非-或非式。**化简的目标是「适配目标门的函数形式」，不只是「数学最简」**。

**形式转换的方法**：

- 与-或式 → 或-与式：取反演（或对偶）。
- 与-或式 → 与非-与非式：对整体取两次反（德·摩根展开）。
- 或-与式 → 或非-或非式：同理。

**工程判断**：现代 EDA 综合器自动做形式选择与门映射——设计师写「行为」，工具选「形式」。但理解形式转换，能让你：读懂综合报告、优化关键路径、在无工具时手算。

**辨析｜易错点：** 「最简与-或式」与「最简与非-与非式」的化简目标不同——前者减少与项与因子，后者还要考虑「取反后的形式」。做题时先看清要求「化简成什么形式」，再选化简策略。

## 7 小结

- 化简目标是**最简与-或式**：乘积项最少、每项因子最少。
- 四种手法：**并项**（$AB+A\overline{B}=A$）、**吸收**（$A+AB=A$）、**消因子**（$A+\overline{A}B=A+B$）、**配项**（先加冗余再吸收）。
- 化简结果必须**用真值表验证**，防止技巧性失误。
- 公式法不系统、难判最优；下节卡诺图将弥补这两点。

在下一节，我们将学会把函数搬进卡诺图——用「圈 1」的几何操作完成化简，直观、快速、而且可判定最简。
