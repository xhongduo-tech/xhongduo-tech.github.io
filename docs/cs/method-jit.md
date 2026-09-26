---
title: 方法 JIT 与热点
date: 2026-09-08
section: cs
---

# 方法 JIT 与热点

<div class="epigraph">
<p>计数调用或回边，过阈值则把整方法编译成 native，替换入口。冷方法留在解释器，摊还编译成本。</p>
<footer>—— 据 Deutsch and Schiffman；Aycock, A Brief History of Just-In-Time；HotSpot 方法编译对照整理</footer>
</div>

上一课[线程化分派](/cs/threaded-dispatch) 仍逐条解释。缺口是 **JIT**：热点方法一次编译，多次执行。本课钉方法级、计数、替换，追踪 JIT 下一课。不写去优化。

## 问题

AOT 编译全部，启动慢、体积大。纯解释启动快、峰值慢。JIT：用轮廓（解释时计数）决定。缺口是**编译单位 = 方法**，不是 ELF 全程序。

<span class="marginnote">术语翻译：JIT（just-in-time，即时编译）就是「程序跑起来之后才把字节码翻译成本机指令」，与 AOT（ahead-of-time，提前编译）相对。它赌的是热点集中——少数方法吃掉绝大多数执行时间，只编译它们就摊薄了编译成本。</span>

入口：调用点从解释器桩改成 compiled 入口（或栈上 on-stack replacement 后课）。

### 热点不是 PGO 文件

PGO 是另一次运行的离线轮廓；JIT 轮廓是本次进程的。可结合：AOT 用 PGO，JVM 用在线。

<span class="marginnote">Deutsch–Schiffman 已把 JIT 当 Smalltalk 路径。Aycock 综述。HotSpot。本课方法 JIT；TraceMonkey 下一课。</span>

## 方法

每方法 invocation/backedge 计数。阈值到：编译队列（可后台线程）。安装：更新调用点、vtable。失败：保留解释器。

```mermaid
flowchart TD
  INT["解释 / 计数"] --> HOT["过阈值"]
  HOT --> COMP["方法编译"]
  COMP --> ENT["替换入口"]
```

与[内联](/cs/inlining-heuristics)：JIT 内联用在线热度更准。与 LTO：JIT 看见的是已加载类，开世界仍在（新类）。

<span class="marginnote">数字实例：编译一个方法花 5 ms，它随后被调用 100 万次，摊到每次只多 5 ns；而解释执行同一段每次可能多付几百 ns——编译开销一次付清、后面全赚。冷方法从不编译，一点不亏；只热一阵的方法则可能回不了本。</span>

## 机制

编译线程与应用线程：队列、代码缓存上限、回收冷码。不要在持锁时同步编译导致卡顿——启发。

ISA：JIT 是[交叉](/cs/cross-compile-triple) 的特例，目标=本机。

<span class="marginnote">常见误区：以为「编译过的方法一定更快、永远用」。后台编译本身占 CPU，代码缓存过大还会挤占指令缓存；方法冷下去后产物还要回收，甚至可能去优化回退到解释器——所以阈值与队列要克制，别在持锁路径上同步编译。</span>

```mermaid
flowchart TD
  HOT["方法过阈值"] --> Q["进编译队列"]
  Q --> BG["后台编译线程翻译成 native"]
  BG --> CACHE["产物放进代码缓存"]
  CACHE --> SWAP["调用点改跳 compiled 入口"]
  CACHE --> FULL{"代码缓存满了?"}
  FULL -->|"是"| EVICT["回收最冷方法腾位"]
```

## 边界

本课不写追踪。后课默认：热点方法可编译。下一课追踪 JIT：以路径为单位而非方法。

也不把 JIT 当恶意运行时注入教程。

## 小结

- 方法 JIT：计数 → 编译整方法 → 换入口。
- 摊还启动与峰值。
- 代码缓存与后台编译是工程。
- 出处：Deutsch and Schiffman；Aycock；HotSpot 文献。
