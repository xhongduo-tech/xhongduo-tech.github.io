---
title: 并发与增量 GC
date: 2026-09-08
section: cs
---

# 并发与增量 GC

<div class="epigraph">
<p>增量把标记切成小片与突变者交错；并发让标记与用户线程真正并行。三色不变式靠读/写屏障维持，否则漏标或悬浮。</p>
<footer>—— 据 Dijkstra et al., On-the-Fly Garbage Collection；Steele；Jones 手册；G1/CMS 实践对照整理</footer>
</div>

上一课[写屏障与卡表](/cs/write-barrier-card-table) 给了记录写入的钩子。缺口是**缩短停顿**：增量/并发标记。本课钉三色（白灰黑）与屏障选择，分代晋升下一课。不把某个 JVM 收集器当唯一理论。

## 问题

Stop-the-world 复制简单，延迟差。并发：突变者同时改图。Dijkstra：写屏障维持不变式。现代：SATB（snapshot-at-the-beginning）或增量更新。缺口是**与突变者的合同**，不是卡表字节。

增量：STW 但切片，仍要世界一致的根。

### 并发 GC 不是「无暂停」

通常有短 STW（根扫描、再标记）。无暂停要更强（Azul、Shenandoah 的进步点名），代价更高。

<span class="marginnote">Dijkstra, Lamport, Martin, Scholten。Yuasa SATB。CMS/G1/Shenandoah。Jones 手册并发章。</span>

## 方法

三色：黑=扫完，灰=待扫，白=未访问。结束时白是垃圾。<span class="marginnote">「三色」翻译成白话：黑是查完账的，灰是排着队待查的，白是还没排到的。收尾时还白着的就当垃圾收——前提是整个标记过程中没有活对象悄悄从队伍里溜走，屏障就是防溜队的。</span>屏障：防止黑指向白而不灰。实现选 SATB 或增量更新。<span class="marginnote">两者的差别可以类比拍照：SATB 在标记开始时拍一张全图快照，之后被摘掉的引用一律按「当时还活着」保留（宁可错放）；增量更新不拍快照，只盯「黑指向白」的新引用，发现就把白重新染灰（宁可错收再修）。前者浮动垃圾多，后者重标记成本高。</span>

```mermaid
flowchart TD
  MUT["突变者"] --> BAR["读/写屏障"]
  BAR --> TRI["三色不变式"]
  TRI --> REC["回收白对象"]
```

与 JIT：安全点、load 屏障（Brooks/LVB）影响每一条指针读。

## 机制

浮动垃圾：SATB 可能把本轮已死当活，下轮收。吞吐换延迟。不要在屏障里分配而无 TLAB 小心再入。

<span class="marginnote">初学者容易以为并发 GC 跑完就该把垃圾收干净，实际上 SATB 这类方案会保住「标记开始那一刻还活着」的全部对象——哪怕下一纳秒它就死了，也得活到下一轮才被收，这就是「浮动垃圾」。它是用多占一轮内存，换用户线程不用停下来等标记。</span>

验证：漏屏障是最毒的 bug。

```mermaid
flowchart TD
  B["对象 B：黑，已扫完"] --> W["对象 W：白"]
  G["对象 G：灰，待扫"] --> W
  MUT["突变者同时运行"] --> OP1["把 B 的字段指向 W"]
  MUT --> OP2["删掉 G 到 W 的引用"]
  OP1 --> DAN["黑直接指向白：不变式被破坏"]
  OP2 --> CUT["W 失去唯一的灰色路径"]
  DAN --> LOST["没有任何收集器线程会再扫到 W"]
  CUT --> LOST
  LOST --> FIX["屏障出手：新引用记灰，或 SATB 保旧快照"]
```

## 边界

本课不写 G1 region 全文。后课默认：低延迟靠屏障+三色。下一课分代假设与晋升。

也不把并发 GC 当数据库 MVCC 课。

## 小结

- 增量/并发用三色+屏障与突变者共存。
- 仍常有短暂停；浮动垃圾换延迟。
- 与卡表分代正交，可组合。
- 出处：Dijkstra et al.；Jones 手册；Yuasa SATB。
