---
title: 小非编码 RNA 的生物发生与出核
date: 2026-08-07
---

# 小非编码 RNA 的生物发生与出核

<div class="epigraph">
<p>小 RNA 是细胞里最精悍的调控员——二三十个核苷酸，就能让一整类基因噤声。</p>
<footer>—— 对 microRNA 与 RNAi 研究的概括</footer>
</div>

<div class="article-byline">
<p>第四级 · RNA 生物学 ｜ Elliott & Ladomery《Molecular Biology of RNA》第14章 ｜ 2026-08-07</p>
</div>

## 为什么从小非编码 RNA 的生物发生开始

前两篇的主角是 mRNA——携带编码信息的信使。但真核转录组里真正的多数派是**不编码蛋白质的 RNA**，其中一小群特别精悍：**只有 20–30 个核苷酸的小非编码 RNA**。它们包括 **microRNA（miRNA）、小干扰 RNA（siRNA）、piRNA**，是细胞最强大的「基因静音」工具，参与发育、免疫、癌症几乎一切过程。

在《mRNA 出核与核质转运》里我们见过 pre-miRNA 走 Exportin-5；这一篇把镜头对准这些小 RNA 的**出生全过程**：它们大多不是直接合成好就工作的，而是**在核内加工成前体、出核、再到胞质成熟**——一个比 mRNA 更讲究「两段式」的旅程。下一篇再讲它们如何执行 RNA 干扰。

## 1 小非编码 RNA 的家族

小非编码 RNA 至少分三大类，来源与功能各异：

**microRNA（miRNA）**：约 22 nt，由基因组编码的内源发夹转录本加工而来，主要通过**翻译抑制与 mRNA 降解**调控数千个靶基因。人类已注释上千条。

**小干扰 RNA（siRNA）**：约 21 nt，由长双链 RNA（病毒、转座子）加工而来，主要执行**序列特异性的转录后沉默**，也是人工 RNAi 技术的核心。

**piRNA（PIWI-interacting RNA）**：24–31 nt，主要存在于生殖细胞，负责**沉默转座子**、保护基因组完整性；与 PIWI 蛋白家族结合，生物发生途径完全不同（不依赖 Dicer）。<span class="marginnote">piRNA 是「生殖细胞的基因组卫兵」：它们识别并沉默转座子，防止自私基因在生殖系里爆发——这也是基因组完整性维护的一线机制。</span>

此外，**snRNA**（剪接用）与 **snoRNA**（rRNA/tRNA 修饰用）也属于小 RNA，但各有独立的生物发生与运输体系。

## 2 miRNA 生物发生：从 pri-miRNA 到成熟 miRNA

miRNA 的经典生物发生是一条清晰的「两切三地」流水线：

**核内第一步——pri-miRNA**：RNA 聚合酶 II 转录出一条长长的 **pri-miRNA**，含一个或多个发夹结构（茎环）。pri-miRNA 被加帽、加 poly(A)，但功能关键的是折叠出的茎环。

**核内第二步——Drosha 切割**：**Microprocessor 复合物**（RNase III 酶 **Drosha** + 双链 RNA 结合蛋白 **DGCR8**）识别发夹，在茎环两侧各切一刀，释放出约 **60–70 nt** 的 **pre-miRNA**——一个带 2 nt 3' 悬突的发夹。<span class="marginnote">Drosha 像一个「模板尺」，DGCR8 负责测量距离、确定切割位点——RNA 结构在这里被当成「加工尺寸的标尺」，切割误差直接影响最终产物的 5' 端（进而影响靶向）。</span>

**出核**：pre-miRNA 被 **Exportin-5**（XPO5）识别，依赖 **RanGTP** 转运出核。

**胞质第三步——Dicer 切割**：胞质里 **Dicer**（另一种 RNase III）从发夹的环端再切一刀，去除环与 3' 悬突，产生约 **22 nt** 的**双链 miRNA 双体**（miRNA/miRNA\* 或 5p/3p）。

**装载与链选择**：双体被装载进 **Ago 蛋白**（RNA 诱导沉默复合物 RISC 的核心），其中一条链作为**向导链（guide strand）**保留，另一条**过客链（passenger strand）**被丢弃。

## 3 公式解析：链选择的「热力学不对称规则」

双链 miRNA 的两条链，哪条当向导？规则是**不对称性**：Ago 蛋白倾向装载 5' 端配对较不稳定的那条链。

$$
\underbrace{\Delta G_{5'\text{-guide}} \lt  \Delta G_{5'\text{-passenger}}}_{\text{向导链 5' 端更「松」}} \Rightarrow \text{Ago 装载向导链}
$$

- **第一步，比较两端稳定性**：双体两端的碱基配对强度不同。一端配对紧、一端配对松。
- **第二步，Ago 的偏爱**：Ago 的 MID 结构域优先捕获 5' 端「松散」的链——因为松开的一端更容易被 Ago 的疏水口袋接纳。
- **第三步，后果**：向导链进入 Ago 后，其 5' 端的第 2–8 位（种子区）负责识别靶 mRNA 的 3'UTR；过客链被解旋或切割降解。

这条规则解释了为什么有些 miRNA 的 5p 链占主导、有些 3p 链占主导——**结局由双体的热力学不对称决定**。理解它，也是设计人工小 RNA（siRNA/shRNA）时选择向导链的依据。

## 4 出核：Exportin-5 与 RanGTP

小 RNA 的出核沿用核转运的通用框架：**货物 + 出核受体 + RanGTP**。

**Exportin-5** 识别 pre-miRNA 的**双链结构**（约 60–70 nt 发夹 + 3' 悬突），核内高浓度的 **RanGTP** 与之结合，三体复合物穿过核孔；到胞质后 RanGTP 水解为 RanGDP，复合物解离，pre-miRNA 被释放到胞质。

出核的**特异性来自结构而非序列**：Exportin-5 认的是「双链发夹 + 短悬突」这个形状，所以人工 shRNA 也能被它运输——这也是 RNAi 工具能工作的原因。<span class="marginnote">同理，其他小 RNA 有各自的受体：tRNA 走 Exportin-t（识别成熟三叶草结构）、snRNA 在超甲基化帽修饰后走 Exportin-1、snoRNA 走 Exportin-1 加适配蛋白——「一货一车」的模块化在出核环节再次体现。</span>

## 5 胞质成熟：Dicer 与链选择

出核后的胞质加工是统一的**Dicer 站台**：Dicer 是一个「分子卡尺」，其 PAZ 结构域结合底物 3' 端 2 nt 悬突，两个 RNase III 结构域在特定距离处各切一刀，产生 21–23 nt 的双体。这个「卡尺」功能让 Dicer 能精确控制产物长度。

随后是 **RISC 装载**：Ago2 接受双体，按上节的热力学规则选链，形成**成熟的 RISC**——一个以单链小 RNA 为向导、以 Ago 为催化核心的沉默机器。至此，miRNA 才算「上岗」。

### 非经典生物发生：mirtrons 与 Drosha 不依赖路径

miRNA 的经典通路（Drosha → Exportin-5 → Dicer）不是唯一的。**非经典（non-canonical）生物发生**展示了「模块化加工」的另一种可能：

**Mirtrons**：某些 miRNA 的前体**直接来自剪接释放的内含子**。内含子被剪接体切下、套索解开后，恰好折叠成发夹——无需 Drosha/DGCR8，直接以 pre-miRNA 形态出核、被 Dicer 加工成成熟 miRNA。**「剪接的废料」变成了「调控的小 RNA」**，是分子经济学的典范。<span class="marginnote">Mirtrons 让我们重新审视「内含子 = 垃圾」的直觉：在植物与动物中都发现的内含子来源 miRNA，说明剪接与 miRNA 两条通路能共享底物——一个内含子既可以被丢弃，也可以被回收成调控 RNA。</span>

**其他非经典路径**：某些 tRNA 或 snoRNA 来源的小 RNA（tRF、sdRNA）也能进入 Ago 蛋白；部分 miRNA 跳过 Dicer（如 pre-miR-451 直接由 Ago2 剪切）。

**与经典通路的共同终点**：无论前体从哪来，最终都汇入「Ago 装载 → 向导链选择 → 靶向沉默」这条统一出口——**通路多样化、出口统一化**，是非编码 RNA 生物发生的普遍设计。

**一句话**：非经典生物发生说明「加工机器不是一成不变的流水线」，而是可以换前体、换酶的模块系统——理解它们，对注释 miRNA 基因、解释非编码 RNA 的演化都至关重要。

## 6 核心对比表：三类小 RNA 的生物发生

| 特征 | miRNA | siRNA | piRNA |
| --- | --- | --- | --- |
| 前体 | 内源 pri-miRNA（发夹） | 长双链 RNA | 单链长转录本 |
| 核内加工 | Drosha/DGCR8 | 部分经核内酶 | 不依赖 Drosha/Dicer |
| 出核 | Exportin-5 | 出核不统一 | 主要在胞质/生殖细胞 |
| 胞质切割 | Dicer | Dicer | PIWI 加工 + 乒乓循环 |
| 长度 | ~22 nt | ~21 nt | 24–31 nt |
| 结合蛋白 | Ago | Ago | PIWI |
| 主要功能 | 转录后调控 | 抗病毒/沉默 | 沉默转座子 |

**核心要点**：三类小 RNA 共享「**加工成短链 + 蛋白装载**」的框架，但前体来源、加工酶与结合蛋白各不相同——**生物发生的多样性决定了功能的多样性**。

### 小 RNA 作为药物与诊断工具

小 RNA 的生物发生知识直接转化为临床应用：

**miRNA 模拟物与抑制剂**：用「miRNA mimic」（双链模拟物，走 Dicer/Ago 通路）补充抑癌 miRNA；用「antagomir / anti-miR」（反义寡核苷酸）抑制致癌 miRNA。它们都借用了内源生物发生与装载机器——**理解通路，才能设计能走通通路的分子**。

**siRNA 药物**：Patisiran、Inclisiran 等 siRNA 药物直接利用 RISC 通路敲低致病基因——合成 siRNA 的设计（链选择、修饰）完全基于我们对 Dicer/Ago 装载规则的理解（见《microRNA 与 RNA 干扰》）。

**循环 miRNA 作为液体活检标志物**：miRNA 在血液中稳定存在（被包裹在囊泡或与蛋白结合），特定 miRNA 谱可作为癌症、心血管病的诊断标志物。

**一句话**：小 RNA 的「生物发生规则」决定了「药物如何设计」——从合成序列、选择向导链到递送途径，每一步都是对生物发生通路的工程化复用。

## 7 小结

- 小非编码 RNA 三大类：**miRNA（22 nt）、siRNA（21 nt）、piRNA（24–31 nt）**，另有 snRNA/snoRNA。
- miRNA 生物发生两切三地：核内 **Drosha/DGCR8** 切出 pre-miRNA → **Exportin-5/RanGTP** 出核 → 胞质 **Dicer** 切出双体 → **Ago** 装载。
- 链选择遵循**热力学不对称规则**：5' 端配对更松的链成为向导链。
- 出核是「**一货一车**」：pre-miRNA 走 Exportin-5、tRNA 走 Exportin-t、snRNA 走 Exportin-1。
- **Dicer 是分子卡尺**，控制产物长度；成熟 RISC 是执行沉默的最终形态。
- piRNA 走完全不同的 **PIWI + 乒乓循环** 途径，专职沉默转座子。

在下一节，我们跟随成熟 miRNA 进入工作岗位：**microRNA 与 RNA 干扰**——RISC 如何在序列配对后沉默靶 mRNA，这套机制又如何被人类改造成基因敲低工具。
