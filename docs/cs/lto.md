---
title: LTO
date: 2026-09-08
section: cs
---

# LTO

<div class="epigraph">
<p>链接期优化把多个编译单元的 IR 放到一起，得到全程序调用图与内联，而不要求源文件合成一份翻译单元。</p>
<footer>—— 据 Glek and Hubička, Optimizing real-world applications with GCC LTO；LLVM LTO/ThinLTO；龙书过程间优化整理</footer>
</div>

上一课[PGO](/cs/pgo)给热度，但单 TU 看不见其它 `.c` 的函数体。缺口是 **LTO**：编译器吐 IR 位码，链接器调优化器再代码生成。ThinLTO：摘要先行、并行后端，减内存。本课钉工作流，不写 ELF 节布局——链接课序。

## 问题

C 的分别编译使 `static` 小函数跨文件无法内联，虚调用目标不全。LTO：全程序（或近全）IR。缺口是**何时优化、谁当链接器插件**，不是 PGO 插桩。

代价：内存、时间、增量编译失效。ThinLTO 用函数摘要+导入热 callee 近似全 LTO。

<span class="marginnote">直觉类比：LTO 像交稿前把全班各章汇成一册通读润色，普通链接只是把各章钉在一起。ThinLTO 则是每人先交一页「章节摘要」，主编按摘要决定谁要参考谁的哪几节，再各自回去改——不必把整本书摊在同一张桌子上。</span>

### LTO 不是「把 .c 都 `#include` 在一起」

语言规则仍按 TU 看名字内部链接；LTO 在 IR 级合并。宏与 `static` 的身份按符号，不是文本粘贴。

<span class="marginnote">GCC LTO。LLVM gold/lld 插件。ThinLTO（Johnson et al.）。本课不写 fat LTO 对象的全部格式。</span>

## 方法

`clang -flto`：`.o` 里是 bitcode。`lld` 调 `lto`：合并、内联、IPSCCP、DCE 无用全局、再分发后端。与 PGO：先 LTO 插桩或 IR 级轮廓。

```mermaid
flowchart TD
  TU["各 TU → IR"] --> LINK["链接器插件"]
  LINK --> IPO["全程序优化"]
  IPO --> CG["代码生成"]
```

与[调用图](/cs/interprocedural-callgraph)：LTO 才有闭世界（静态链接）。动态库边界仍开。

## 机制

符号可见性：被其它 `.so` 用的符号不能删、不能改约定。`hidden`/`internal` 使 DCE 更狠。不要 LTO 进不兼容的 IR 版本。

<span class="marginnote">常见误区：初学者以为加上 `-flto` 就自动全程序优化。只要符号还被其它动态库引用，就不能删不能内联，闭世界假设名存实亡——LTO 的收益大头来自可见性收窄成 `hidden` 的「内部」符号，导出越多收益越薄。</span>

```mermaid
flowchart TD
  FULL["全 LTO: 全部 IR 一次性合并"] --> PRO["内联与 DCE 视野最全"]
  FULL --> COST["单进程内存与链接时间大"]
  THIN["ThinLTO 两阶段"] --> PH1["阶段一: 并行生成函数摘要与索引"]
  PH1 --> RES["链接时合并索引, 决定跨模块导入"]
  RES --> PH2["阶段二: 各模块并行优化与代码生成"]
  PH2 --> THER["收益近似全 LTO, 开销近普通链接"]
```

并行：ThinLTO 两阶段，摘要全局，优化仍分片。

## 边界

本课不写链接脚本。后课默认：跨 TU 内联与 DCE 靠 LTO。下一课 UB：全程序分析更敢删「不可能」路径，风险也更大。

<span class="marginnote">为什么重要：LTO 让全程序分析「更有底气」，一旦源码藏着 UB（有符号溢出、未初始化读），被删的可能正是你以为在工作的代码。开 LTO 前先清掉 UB 警告、跑一遍 sanitizer，收益才不会被调试灾难吃掉。</span>

也不把 LTO 当 Java 的 JIT 全程序（那是运行时）。

## 小结

- LTO：链接时优化 IR，补全程序视野。
- ThinLTO 用摘要换可扩展性。
- 可见性与 `.so` 边界限制闭世界。
- 出处：Glek and Hubička；LLVM LTO/ThinLTO；对照龙书 IPO。
