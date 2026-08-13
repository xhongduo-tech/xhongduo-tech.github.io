---
title: 交叉偶联反应：Heck、Suzuki、Negishi 等
date: 2026-08-07
---

# 交叉偶联反应：Heck、Suzuki、Negishi 等

<div class="epigraph">
<p>把两个本来互不相干的碳原子焊在一起——这是钯在合成化学里最伟大的神迹。</p>
<footer>—— 2010 年诺贝尔化学奖评论（Heck、Negishi、Suzuki）</footer>
</div>

<div class="article-byline">
<p>第二级 · 金属有机化学 ｜ Crabtree《The Organometallic Chemistry of the Transition Metals》第14章 ｜ 2026-08-07</p>
</div>

## 为什么从交叉偶联开始

有机合成的核心问题是**构建 C–C 键**。传统方法（格氏试剂、锂试剂）怕水怕氧、选择性与官能团兼容性差。**交叉偶联（cross-coupling）**用钯催化把两个「准备好」的碳片段焊在一起，条件温和、官能团兼容、选择性惊人——2010 年诺贝尔化学奖因此授予 Heck、Negishi、Suzuki。<span class="marginnote">交叉偶联让制药工业发生了革命：今天的抗肿瘤药、抗生素、液晶材料、OLED 分子，大量依赖 Suzuki、Heck、Negishi 偶联构建联芳基骨架。有人说「没有钯偶联，现代药物化学会塌掉一半」。</span>

这一节先讲交叉偶联的统一循环（Pd(0)/Pd(II)），再逐个看 Heck、Suzuki、Negishi、Stille 等家族成员。

## 1 统一循环：所有偶联的公共骨架

所有交叉偶联（除 Heck 外）共享同一个四步循环，通式：

$$
\ce{R-X + R'-M + Pd(0) -> R-R' + M-X + Pd(0)}
$$

1. **氧化加成**：Pd(0)（16e）插入 C–X 键 → Pd(II)(R)(X)（18e）。X 是卤素或类卤素（OTf、OTs）。
2. **转金属**：R′ 从有机金属试剂（R′–M）转移到 Pd 上 → Pd(II)(R)(R′)。
3. **还原消除**：R 与 R′ 偶联放出 R–R′，Pd 回 Pd(0)。
4. **配体交换/再生**：Pd(0) 被配体稳定，等待下一轮。

**重点：交叉偶联 = OA + 转金属 + RE 三积木**。<span class="marginnote">循环里 Pd 在 0 与 +II 之间穿梭（OA 升、RE 降），氧化态账本与加氢完全同构——只是把「烯烃插入」换成了「转金属」。理解这个统一循环，就等于理解了 Suzuki/Negishi/Stille/Sonogashira 的全部：它们只差「R′–M 是什么」。</span>

各家族的区别只在**有机金属试剂 R′–M**：

| 反应 | R′–M | 发现者 |
| --- | --- | --- |
| Kumada | $\ce{R'MgX}$（格氏） | 1972 |
| Negishi | $\ce{R'ZnX}$ | 1977 |
| Stille | $\ce{R'SnR''3}$ | 1978 |
| Suzuki | $\ce{R'B(OR)2}$ | 1979 |
| Sonogashira | 炔烃 + Cu 助催化 | 1975 |

## 2 公式解析：Suzuki 偶联的完整循环

以最常用的 **Suzuki–Miyaura 偶联**为例，拆解每一步：

$$
\ce{Ar-X + Ar'-B(OH)2 ->[\ce{Pd(PPh3)4}, \ce{Base}] Ar-Ar' + X-B(OH)2}
$$

- **第一步，氧化加成**：$\ce{Pd(0)}$ 插入 $\ce{Ar-X}$，得 $\ce{Pd(II)(Ar)(X)}$。电子账：Pd 0→+II，d¹⁰→d⁸，配位数 +1。
- **第二步，碱的作用**：碱（$\ce{Na2CO3}$、$\ce{K3PO4}$）先与硼酸配位/活化，把 Ar′ 从硼「活化」成更易转移的形式——**碱是 Suzuki 的灵魂**，没有碱硼酸不转移。<span class="marginnote">Suzuki 里碱的必要性常被低估：碱与 $\ce{B(OH)2}$ 形成硼酸盐阴离子，使 C–B 键极性反转、Ar′ 更容易作为碳负离子转移给 Pd。这就是为什么 Suzuki 比 Kumada 温和——不需要强碱性金属试剂，弱碱就能启动。</span>
**第三步，转金属**：Ar′ 从硼搬到 Pd 上 → $\ce{Pd(II)(Ar)(Ar')}$，硼带着 X 离开。
**第四步，还原消除**：Ar 与 Ar′ 偶联 → 联芳烃 Ar–Ar′，Pd 回 0 价，循环完成。<span class="marginnote">Suzuki 偶联的工业价值在于「官能团兼容性极好」：醛、酮、酯、酰胺在偶联条件下都能活着。制药里构建联芳基药物骨架（抗高血压药、抗真菌药）几乎非它莫属。</span>

**易错点｜辨析：** 转金属的方向要搞对——是「有机金属试剂的碳搬家到 Pd」上，不是 Pd 的碳搬过去。R′–M 里 M 是 Mg/Zn/Sn/B，**Pd 抢走 R′**，M 与 X 结合离开。转金属的驱动力是「新 M–X 键比旧 M–R′ 键更强」。

## 3 Heck 反应：不用有机金属的偶联

**Heck 反应（Mizoroki–Heck）**是特殊的偶联：它不需要有机金属试剂，只靠烯烃与卤代芳烃：

$$
\ce{Ar-X + CH2=CHR ->[Pd] Ar-CH=CHR + H-X}
$$

循环：OA 得 $\ce{Pd(II)(Ar)(X)}$ → 烯烃配位并插入 Pd–Ar → **β-氢消除**放出产物烯烃与 Pd–H → 碱把 H–X 带走，Pd 回 0 价。<span class="marginnote">Heck 的巧妙在于：<strong>用 β-氢消除代替还原消除</strong>收尾——产物是取代烯烃，H 留给金属、再被碱洗掉。这使它不需要预先制备任何有机金属试剂，直接偶联烯烃与芳卤。</span>

**重点：Heck 是「插入 + β-消除」型偶联**，其余家族是「转金属 + RE」型——两类偶联的收尾完全不同，但都围着 Pd(0)/Pd(II) 转。Heck 也是合成取代烯烃（肉桂酸类、苯乙烯类）的主力。

## 4 家族成员的个性与选择

**Negishi 偶联（$\ce{R'ZnX}$）**：锌试剂活性高、官能团兼容好，能偶联芳基、烯基、烷基（包括 sp³–sp³）。活性与兼容的平衡最好。<span class="marginnote">Negishi 偶联是「最难做但最全能」的成员：Zn 试剂可以偶联三个 sp³ 碳（烷基-烷基偶联），而 Suzuki 做 sp³ 偶联常被 β-消除拖累。工业上复杂分子晚期偶联常选 Negishi。</span>

**Stille 偶联（$\ce{R'SnR''3}$）**：锡试剂稳定、对水氧不敏感，但锡毒性大——现代合成逐渐用 Suzuki 替代。

**Sonogashira 偶联（末端炔 + 芳卤）**：用 Cu(I) 共催化，把炔基接到芳环上，是合成芳基炔（药物、材料）的主力。

**Kumada 偶联（$\ce{R'MgX}$）**：最老的偶联，格氏试剂太活泼，官能团兼容差，但便宜——工业简单底物仍用。

**选型口诀**：要官能团兼容选 Suzuki/Sonogashira，要偶联 sp³ 碳选 Negishi，要便宜选 Kumada，要温和选 Suzuki。<span class="marginnote">现代合成里 Suzuki 与 Sonogashira 是绝对主流，Negishi 用于难底物，Stille 与 Kumada 各有拥趸。选择偶联方法的本质，是在「活性、兼容性、毒性、成本」四维里取平衡——这跟选配体是一个道理。</span>

## 5 催化剂的现实：配体与 Pd 源

交叉偶联的催化剂不只「Pd」，而是「Pd + 配体」的组合：

**Pd 源**：$\ce{Pd(PPh3)4}$、$\ce{Pd2(dba)3}$、$\ce{Pd(OAc)2}$