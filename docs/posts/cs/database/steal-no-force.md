---
title: 缓冲管理策略：STEAL 与 NO-FORCE
date: 2026-08-07
---

# 缓冲管理策略：STEAL 与 NO-FORCE

<div class="epigraph">
<p>脏页什么时候刷盘，决定了恢复算法要 undo 还是要 redo——四个象限，四个代价。</p>
<footer>—— 吉姆 · 格雷（Jim Gray，图灵奖得主）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ 《数据库系统概念》 第15章 恢复系统 ｜ 2026-08-07</p>
</div>

## 为什么从 STEAL 与 NO-FORCE 开始

恢复算法不是孤立的——它必须与**缓冲池策略**配合。两个关键决策决定恢复的形态：

- **STEAL vs NO-STEAL**：未提交事务的脏页**能否提前刷盘**？
- **FORCE vs NO-FORCE**：提交时，脏页**是否必须立即刷盘**？

这两个二选一组合成**四个象限**，每个象限对应不同的恢复需求。现代数据库的标准组合是 **STEAL + NO-FORCE**（缓冲池灵活、提交快），代价是恢复要做 Undo + Redo 两件事。这一节把四象限与恢复需求的关系讲透——这是理解 ARIES 为什么「分析-重做-撤销」三阶段的钥匙。

## 1 两个决策维度

**STEAL（偷）**：缓冲池满时，能否把**未提交事务**的脏页写回磁盘？

- **STEAL**：可以——缓冲池压力小，但磁盘可能含未提交修改（崩溃需 Undo）。
- **NO-STEAL**：不可以——未提交修改只在内存，崩溃无需 Undo，但缓冲池可能被未提交页占满。

**FORCE（强制）**：事务提交时，其脏页**是否必须落盘**？

- **FORCE**：必须——提交慢（要等刷盘），但已提交修改必在磁盘（崩溃可能无需 Redo）。
- **NO-FORCE**：不必——提交快（只刷日志），已提交修改可能在内存（崩溃需 Redo）。

**核心要点：STEAL 决定要不要 Undo，NO-FORCE 决定要不要 Redo。**

## 2 四个象限

| 组合 | Undo 需要？ | Redo 需要？ | 特点 |
| --- | --- | --- | --- |
| STEAL + FORCE | 是 | 否 | 提交慢、恢复只有 Undo |
| NO-STEAL + FORCE | 否 | 否 | 恢复最简，但缓冲/提交都受限 |
| STEAL + NO-FORCE | 是 | 是 | **现代标准**，恢复最复杂 |
| NO-STEAL + NO-FORCE | 否 | 是 | 无 Undo，但缓冲受限 |

**公式解析：恢复的复杂度由象限决定**

$$
\text{恢复工作量} = \mathbb{1}[\text{STEAL}] \cdot \text{Undo 量} + \mathbb{1}[\text{NO-FORCE}] \cdot \text{Redo 量}
$$

- **第一步，STEAL → Undo**：未提交脏页可能在磁盘，需撤销。
- **第二步，NO-FORCE → Redo**：已提交修改可能在内存，需重放。
- **第三步，最简象限**：NO-STEAL + FORCE 恢复工作量最小——但运行时限制大。
- **第四步，为什么选 STEAL + NO-FORCE**：缓冲池不会被未提交页堵死（STEAL），提交不用等刷盘（NO-FORCE）——**用「恢复更复杂」换「运行时更灵活」**。

## 3 为什么现代数据库选 STEAL + NO-FORCE

**缓冲池的现实**：缓冲池容量有限，不能为「未提交事务」预留全部空间——事务可能修改海量页。**NO-STEAL 在长事务面前必然失败**（缓冲池被未提交页占满，其他事务无法运行）。

**提交的现实**：`COMMIT` 若要求全部脏页落盘，一次大事务提交要刷几百 MB——**延迟不可接受**。NO-FORCE 只需日志落盘（顺序小写），提交毫秒级。

**核心要点：STEAL + NO-FORCE 是「现代数据库的必然选择」。** 它让缓冲池高效运转、提交快速——代价是恢复必须做完整的 Undo + Redo。**ARIES 就是为这个象限设计的工业级恢复算法**。

<span class="marginnote">对比第 9 章「缓冲池脏页写回」：那里讲「NO-FORCE 延迟写回配合 WAL」——这里把它严格化为四象限之一。<strong>STEAL 与 NO-FORCE 不是可选项，而是大数据库的生存需求</strong>——恢复算法必须能处理「最难的象限」。</span>

## 4 恢复算法如何匹配策略

**不同缓冲策略需要不同的恢复算法**：

| 缓冲策略 | 恢复算法 | 说明 |
| --- | --- | --- |
| NO-STEAL + FORCE | 无需 Undo/Redo | 恢复只需丢弃日志 |
| STEAL + FORCE | 仅 Undo | 已提交修改必在磁盘 |
| NO-STEAL + NO-FORCE | 仅 Redo | 未提交修改必在内存 |
| **STEAL + NO-FORCE** | **Undo + Redo** | **ARIES 三阶段** |

**辨析｜易错点：** 恢复算法与缓冲策略**必须匹配**。若实现为 STEAL 却用「仅 Redo」算法，未提交修改残留磁盘——数据错误。**设计恢复系统 = 先定缓冲策略，再定恢复算法**。

## 5 小结

- 两个维度：**STEAL**（未提交脏页可否提前刷盘）与 **FORCE**（提交时是否必须刷盘）。
- 四象限：STEAL/FORCE、NO-STEAL/FORCE、STEAL/NO-FORCE、NO-STEAL/NO-FORCE。
- **STEAL → 要 Undo；NO-FORCE → 要 Redo**。
- 现代数据库标准 = **STEAL + NO-FORCE**：缓冲灵活、提交快，恢复最复杂。
- 恢复算法必须与缓冲策略匹配；ARIES 为 STEAL + NO-FORCE 而生。

在下一节，我们进入工业级恢复的巅峰——**ARIES 恢复算法：日志记录、分析与重做阶段**。
