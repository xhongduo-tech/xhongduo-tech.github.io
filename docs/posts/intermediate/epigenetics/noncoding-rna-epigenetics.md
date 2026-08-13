---
title: 非编码 RNA 介导的表观遗传调控
date: 2026-08-07
---

# 非编码 RNA 介导的表观遗传调控

<div class="epigraph">
<p>基因组不只是蛋白质的蓝图，更是 RNA 的舞台。</p>
<footer>—— 约翰 · 马蒂克（John Mattick）</footer>
</div>

<div class="article-byline">
<p>第二级 · 表观遗传学 ｜ 薛京伦主编，《表观遗传学》第6章 ｜ 2026-08-07</p>
</div>

## 为什么从非编码 RNA 开始

人类基因组约 98% 不编码蛋白质，很长一段时间被称为「垃圾 DNA」。
但越来越多的证据表明，
这些区域**大量转录成非编码 RNA（ncRNA）**，
而其中相当一部分直接参与表观调控——引导甲基化到转座子、
招募多梳到发育基因、
沉默整条 X 染色体。
非编码 RNA 是表观遗传调控里「看不见的指挥者」。

在上一节的异染色质里我们已经见过一个例子：
着丝粒的重复序列转录出 RNA，
被加工成 siRNA 后引导 `SUV39H`。
本节把「RNA 引导表观修饰」这条主线展开：
从短的 siRNA/piRNA 到长的 lncRNA（`Xist`、
`HOTAIR`），
它们共同构成 RNA 层面的表观调控网络。
<span class="marginnote">非编码 RNA 与表观遗传的关系常被一句话概括：
「RNA 指路，
蛋白执行」。
RNA 提供序列特异的「地址」，蛋白复合物提供「施工队」。
</span>

## 1 RNAi 通路与异染色质：siRNA 引导甲基化

**RNAi（RNA interference）** 通路是 RNA 引导表观修饰的最古老模式，在裂殖酵母、植物、果蝇与哺乳动物中保守：

**裂殖酵母的经典回路**：
着丝粒重复序列转录出双链 RNA → **Dicer** 切成 ~23 nt 的 **siRNA** → 装载进 **RITS**（RNA-induced transcriptional silencing）复合物 → RITS 借 siRNA 的碱基互补「指认」着丝粒转录本 → 招募 `CLRC`（含 Clr4/SUV39H）在着丝粒写 `H3K9me3` → 异染色质形成。
同时异染色质又抑制新的转录，形成**「转录→siRNA→沉默→少转录」的闭环**。
<span class="marginnote">这个回路在植物里同样经典：
启动子区 dsRNA 能诱导 DNA 甲基化（RdDM，
RNA-directed DNA methylation），
`DRM2` 等甲基转移酶被 24 nt siRNA 引导到同源位点——「RNA 指路、
DNMT 施工」。
</span>

**重点：** RNAi 引导的表观沉默是**序列特异性**的——siRNA 与靶序列互补，所以理论上可以靶向任何同源位点。
这使 RNAi 成了天然的「表观编辑器」：
在植物中，
RNA 病毒入侵时，
siRNA 能把病毒序列对应的基因组位点甲基化关闭。

## 2 piRNA：转座子的「哨兵」

**piRNA（PIWI-interacting RNA）** 是动物生殖系里专门对付转座子的一类小 RNA：

**特征**：
长 24–32 nt（比 siRNA 长）、
由 **PIWI 蛋白**（`Miwi`/`Mili` 等）结合、
由单链前体经「乒乓循环」（ping-pong cycle）扩增产生。
**功能**：piRNA 引导 PIWI 蛋白识别转座子转录本，通过两个层面抑制：一是**切割转座子 mRNA**（转录后抑制）；
二是**引导 DNA 甲基化与 `H3K9me3`** 到转座子位点（转录抑制）。
<span class="marginnote">生殖系必须守住转座子——因为转座子一旦在生殖细胞里活跃，
突变会传给下一代。
所以 piRNA 是「基因组的防火墙」，专门保护遗传信息免遭内源"入侵者"的破坏。
</span>

**辨析｜易错点：** siRNA、miRNA、piRNA 三者别混。
siRNA 由长双链 RNA 切出、作用同源位点；
miRNA 由发夹前体切出、主要抑制 mRNA 翻译/稳定性；
piRNA 由单链转录本经乒乓扩增、主要在生殖系沉默转座子。
**作用机制、产生途径、功能场景各不相同。**

## 3 长非编码 RNA：Xist、HOTAIR 与「顺式招募」

长非编码 RNA（**lncRNA**，>200 nt）是当代表观研究的热点。
它们像「分子胶水」，把染色质修饰复合物招募到特定基因组位置。
两个代表性案例：

**`Xist`：整条 X 染色体的沉默**。
`Xist` 从失活 X 染色体（Xi）上的基因座转录，
随后**顺式覆盖整条 Xi**（cis-acting），
招募 `PRC2`（写 `H3K27me3`）、
`SMRT/HDAC`、
以及 `SMCHD1` 等，
最终把整条 X 压实成异染色质（第18篇详谈）。
<span class="marginnote">`Xist` 的顺式作用方式值得强调：
它只沉默「生产它的那条 X」，
不沉默另一条——因为 lncRNA 大多<strong>留在转录位点附近</strong>，
不像 mRNA 那样被运输到胞质翻译。
这条「顺式限制」是 X 失活特异性的核心。
</span>

**`HOTAIR`：跨染色质招募的反式例子**。
`HOTAIR` 从 `HOXC` 基因座转录，
但能**反式**（trans）结合到远端的 `HOXD` 基因座上，
招募 `PRC2` 与 `LSD1` 复合物，
介导 `HOXD` 的沉默。
`HOTAIR` 的高表达与乳腺癌转移相关。
<span class="marginnote">HOTAIR 证明 lncRNA 不一定要顺式作用——它可以像「长了地址的胶水」一样跑到别的染色质上去招募抑制复合物。
不过，
HOTAIR 的反式功能在近年的敲除小鼠中受到部分质疑，
学界仍在讨论其生理重要性——这提醒我们：
lncRNA 功能的研究既激动人心，
也充满争议。
</span>

## 4 lncRNA 的表观机制：四种「指路」方式

lncRNA 参与表观调控的机制可以归纳为四种：

**招募（recruit）**：作为支架把修饰复合物带到特定位点（`Xist` → PRC2；
`Kcnq1ot1` → 印迹沉默）。
**变构（allosteric）**：结合蛋白复合物后改变其构象/活性（如某些 lncRNA 激活或抑制 EZH2 的酶活）。
**遮蔽（decoy/sequester）**：把蛋白「吸走」，使其无法到达靶位点（lncRNA 与转录因子或修饰酶结合，阻止它们去该去的地方）。
**作为结构骨架（scaffold）**：
把多个蛋白复合物「搭」在一起，
形成核内小体或染色质结构（如 `NEAT1` 搭建 paraspeckle 核体）。
<span class="marginnote">最后一种尤其值得注意：
lncRNA 能像建筑钢筋一样把蛋白「浇铸」成核体——`NEAT1` 是核旁斑（paraspeckle）的骨架，
没有它核体就散架。
RNA 不只是「信使」，还是「建材」。
</span>

## 5 公式解析：RNA 引导的序列特异性

RNA 引导表观修饰的核心优势是**序列特异**：RNA 的碱基序列与靶 DNA/RNA 互补，提供「地址编码」。
可以用一条简单的配对示意表达特异性：

$$
\text{lncRNA/siRNA 序列} \; \xrightarrow{\text{碱基互补}} \; \text{靶位点} \; \xrightarrow{\text{招募修饰复合物}} \; \text{表观修饰}
$$

三步拆解：

- **第一步，看编码容量**：一个 23 nt 的 siRNA，理论上能区分 $4^{23} \approx 7\times 10^{13}$