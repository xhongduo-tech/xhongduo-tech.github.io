---
title: ARIES 恢复算法：日志记录、分析与重做阶段
date: 2026-08-07
---

# ARIES 恢复算法：日志记录、分析与重做阶段

<div class="epigraph">
<p>ARIES 用 LSN 给日志钉上顺序，让重做精确到每一次修改。</p>
<footer>—— 莫哈纳钱德拉 · 莫汉（C. Mohan，ARIES 作者）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ 《数据库系统概念》 第15章 恢复系统 ｜ 2026-08-07</p>
</div>

## 为什么从 ARIES 开始

前面讲的恢复算法是「概念版」，工程上需要一个能处理真实复杂性的工业级算法。**ARIES（Algorithm for Recovery and Isolation Exploiting Semantics）**是 IBM 提出的恢复算法，也是 PostgreSQL（WAL）、MySQL InnoDB、SQL Server 等主流数据库恢复系统的思想基础。它用**日志序列号（LSN）**给每条日志定序，用「脏页表 + 活跃事务表」精确控制重做范围。这一节讲 ARIES 的日志记录结构、**分析（Analysis）阶段**与**重做（Redo）阶段**；下一节讲撤销阶段与补偿日志。

## 1 ARIES 的基石：LSN

**日志序列号（Log Sequence Number, LSN）**：每条日志记录的唯一、递增编号，也是该日志在日志文件中的地址。

**核心要点：LSN 让日志「可寻址、可比序」。** 每条数据页记录**最后一个修改它的日志的 LSN**（**pageLSN**），每个缓冲页记录 **recLSN**（自上次刷盘以来第一次修改的 LSN）——这些字段让恢复能判断「某页是否需要重做」：**若页的 pageLSN ≥ 日志记录的 LSN，说明该修改已在该页上（无需重做）**。

## 2 ARIES 的日志记录类型

ARIES 在基本日志之上扩展：

- **更新记录（update record）**：带 LSN 的修改记录（Undo 用 old、Redo 用 new）。
- **仅撤销记录（undo-only record）**：**仅 Undo** 的记录（如逻辑操作记录，重做可由其他记录完成）。
- **Begin / Commit / Abort / End**：事务生命周期。
- **CLR（补偿日志记录）**：撤销阶段产生，记录「已撤销某修改」——下一节专讲。
- **Checkpoint（检查点记录）**：检查点记录，含活跃事务表与脏页表。

**核心要点：ARIES 日志携带「恢复所需的一切元数据」。** 每条修改记录带旧值、新值、LSN；页面维护 pageLSN——恢复时可以精确判断「这条修改要不要应用到某页」。

## 3 检查点中的两个表

ARIES 检查点记录包含**两张表**：

- **活跃事务表（Active Transaction Table, ATT）**：未提交事务及其状态、lastLSN。
- **脏页表（Dirty Page Table, DPT）**：缓冲池中**未落盘**的脏页及其 recLSN（页第一次变脏时的 LSN）。

**这两张表是分析阶段的输入**——它们让恢复知道「从哪开始重做、哪些事务要撤销」。

## 4 恢复三阶段概览

ARIES 恢复分三个阶段：

1. **分析（Analysis）**：从最近检查点开始扫描日志，重建 ATT 与 DPT，确定 **RedoLSN**（需要重做的起点）。
2. **重做（Redo）**：从 RedoLSN 起重放修改，恢复崩溃前的内存状态。
3. **撤销（Undo）**：按 LSN 逆序回滚未提交事务（下一节）。

**公式解析：RedoLSN 的确定**

设检查点的脏页表为 DPT，则需要重做的日志起点：

$$
\text{RedoLSN} = \min \{ \text{recLSN} \mid \text{页} \in \text{DPT} \}
$$

- **第一步，DPT 里最早的 recLSN**：最早的可能未落盘的修改——早于它的修改都已刷盘。
- **第二步，跳过安全区间**：RedoLSN 之前的日志无需重做（其修改已在磁盘）。
- **第三步，重做范围**：从 RedoLSN 到日志末尾。
- **第四步，分析阶段的产出**：确定 RedoLSN + 重建 ATT/DPT——**分析是「定位战场」**。

## 5 重做阶段

**Redo 阶段**：从 RedoLSN 开始**顺序**扫描日志，对每条修改记录：

- 若目标页不在 DPT 中：跳过（该页已落盘，修改无需重做）。
- 若记录 LSN < 页的 recLSN：跳过（该页在检查点后已刷过盘）。
- 否则：**重做**（写新值），更新页的 LSN。

**核心要点：Redo 用 LSN 精确跳过「无需重做」的修改。** 这就是 ARIES 比朴素「全量重放」高效的地方——**只重做真正没落盘的修改**。<span class="marginnote">Redo 是<strong>幂等且可重复</strong>的：崩溃可能发生在重做中途，重启后再次重做——同一修改重放多次结果不变。这个「恢复本身可崩溃」的性质，是 ARIES 稳健性的关键。</span>

## 6 小结

- ARIES 用 **LSN** 给日志定序，页面记 pageLSN/recLSN——精确判断是否需要重做。
- 日志类型：修改（含 CLR）、begin/commit/abort/end、checkpoint。
- 检查点含 **ATT（活跃事务表）与 DPT（脏页表）**。
- 恢复三阶段：**分析（定位）→ 重做（重建）→ 撤销（回滚）**。
- RedoLSN = DPT 中最小的 recLSN；Redo 用 LSN 跳过无需重做的修改。

在下一节，我们看 ARIES 的第三阶段——**撤销阶段与补偿日志记录（CLR）**。
