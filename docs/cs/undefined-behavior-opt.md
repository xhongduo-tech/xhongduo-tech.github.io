---
title: 未定义行为与优化
date: 2026-09-08
section: cs
---

# 未定义行为与优化

<div class="epigraph">
<p>语言把某些状态标成未定义，优化器可假设它们不发生。于是「程序员觉得会做什么」与「生成代码做什么」在 UB 处分裂。</p>
<footer>—— 据 ISO C/C++ 未定义行为条款；Lattner, What Every C Programmer Should Know About Undefined Behavior；CompCert 对定义行为子集的对照整理</footer>
</div>

上一课[LTO](/cs/lto)让分析看见更多路径。缺口是**合法变换的上限**：有符号溢出、野指针、数据竞争（C）是 UB，优化可删检查、把循环当无限、根据「不能越界」收紧范围。本课钉「假设不发生」如何变成变换，不把全部 UB 列表当百科。中端课序在此收紧契约。

## 问题

[健全性](/cs/type-soundness) 相对「出错状态」才有意义。C 的出错是 UB，语义允许任何目标码。优化：`if (p) load p; load p` 第二次可提前，因空指针 UB。缺口是**把标准当许可**，不是区间格细节。

<span class="marginnote">「UB」可以理解成编译器的免检通行证：语言把某些状态（有符号溢出、解引用野指针）标成「不允许发生」，优化器于是放心假设它们不存在，顺手删掉你写的防御性检查。</span>

程序员用溢出当绕回，优化可删 `if (x+1<x)`。这是误解契约，不是编译器随机坏。

### UB 不是未指定或实现定义

未指定：选一种；实现定义：必须文档。UB：无要求。不要三词混用。

<span class="marginnote">ISO C 3.4.3。Lattner 的系列短文。Regehr 的博客与 cse473 课常见材料。CompCert 选可定义子集。本课不发明标准条款编号之外的「秘密 UB」。</span>

## 方法

IR 带 `nsw`/`nuw`/`inbounds` 等毒标志，把源语言 UB 显式化。变换：利用毒值传播、立即 UB 则后继不可达。诊断：sanitize（ASan/UBSan）在另一配置关闭这些假设。

```mermaid
flowchart TD
  SRC["源程序"] --> UB["语言 UB"]
  UB --> ASSUME["优化假设不发生"]
  ASSUME --> XFORM["激进变换"]
```

与值范围、LICM、别名（TBAA）都吃同一口粮。fast-math 是浮点侧的类似开关，下一课。

## 机制

poison 与立即 UB 在 IR 里走的是两条不同的触发线：

```mermaid
flowchart TD
  NSW["带 nsw 的加法溢出"] --> POISON["产生毒值"]
  POISON --> PROP["毒值沿数据流传播"]
  PROP --> USE["被分支/存储使用时才成 UB"]
  WILD["解引用野指针"] --> IMM["立即 UB"]
  IMM --> DEAD["所在块后继不可达"]
  USE --> OPT["优化据此删检查或重排"]
  DEAD --> OPT
```

毒值 vs 立即 UB：LLVM 区分 poison 与 UB，避免「一条毒指令核掉整个函数」过激。语言律师与 IR 律师要对齐。不要在安全语言 IR 里默许 C 的 `nsw`。

<span class="marginnote">数字实例：`if (x+1 \lt x) …` 想检测有符号回绕。x+1 溢出是 UB，优化器假设「溢出不发生」，条件恒假，整个判断被直接删除；要回绕语义得改无符号类型或加 `-fwrapv`。</span>

LTO 使跨函数的 UB 假设传播更远，bug 表面更离奇。

## 边界

本课不写全部 sanitizer。后课默认：中端合法性 ⊂ 语言未定义之外的行为。下一课 fast-math：浮点契约的显式放松。

也不把 UB 当「可以生成病毒」的许可证来讲利用；本课只讲优化契约。

<span class="marginnote">常见误区：把「未指定」「实现定义」「未定义」当一个意思。未指定是编译器任选一种；实现定义是必须写进文档；UB 是无任何要求——宽容度依次放开，混用这三个词会完全错估程序行为。</span>

## 小结

- UB 让优化假设某些状态不可达。
- 源级「绕回」直觉与 C 有符号溢出冲突。
- sanitizer 与 `-fwrapv` 等改变契约。
- 出处：ISO C/C++；Lattner；对照 Leroy CompCert 子集。
