---
title: SSA 构造 Cytron
date: 2026-09-08
section: cs
---

# SSA 构造 Cytron

<div class="epigraph">
<p>在支配边界插入 φ，再按支配树重命名，得到静态单赋值。Cytron 等人给出与支配树相关的高效放置。</p>
<footer>—— 据 Cytron et al., Efficiently Computing Static Single Assignment Form, 1991；Lengauer and Tarjan 支配；Appel, SSA is Functional Programming 整理</footer>
</div>

上一课[值范围](/cs/value-range-analysis)已在 SSA 上说话。主干[SSA](/cs/ssa-form)给直觉。缺口是**构造算法**：支配树、支配边界、φ 迭代插入、重命名栈。本课钉 Cytron 步骤，不写析构——下一课。

## 问题

朴素在每个汇合点对每个变量插 φ 太多。只需：变量在块定值，则在其支配边界插 φ，再迭代。缺口是**支配边界 DF**，不是值编号。

<span class="marginnote">支配边界 DF 翻译成大白话：从你统治的块「绕出去」第一站遇到的那些块——变量在这条路能流到、但你的块管不住它的位置。汇合正是发生在这种「绕出去」的地方，所以 φ 恰好只需要插在 DF 上：这是「插得准」的根据，也是 Cytron 算法比朴素全插少一大截的原因。</span>

Lengauer–Tarjan 算支配。剪枝 SSA：无 use 的名不插 φ。semi-pruned 用 liveness 粗筛。

### φ 的操作数对应前驱边

顺序与 CFG 前驱列表对齐。错位则语义错。这不是「任意交换」。

<span class="marginnote">Cytron et al. 1991 TOPLAS。Appel 1998。主干 SSA 课已禁止把 φ 当机器指令。本课补算法。</span>

## 方法

1. 算支配树与 DF。2. 对每个变量的定值块工作表插 φ。3. 支配树 DFS，栈重命名，φ 操作数填入到达名。

```mermaid
flowchart TD
  CFG["CFG"] --> DOM["支配树"]
  DOM --> DF["支配边界"]
  DF --> PHI["插 φ"]
  PHI --> RN["重命名"]
```

与 mem2reg：alloca 的 store/load 先当定值/使用再走同一算法——接 SROA。

<span class="marginnote">初学者最容易在这里犯的错：把 φ 当成一条会在运行时执行的机器指令。实际上 φ 是编译器内部的「多选一」记号——控制流从哪条前驱边进来，φ 的值就取哪条边对应的操作数；析构阶段它会被翻译成普通拷贝。这也是为什么 φ 的操作数顺序必须和 CFG 前驱列表严格对齐，错一位整个语义就换了。</span>

## 机制

不可约 CFG 仍可 SSA，φ 更多。关键边有时先拆，方便后边析构。不要在构造时做 GVN；顺序是先 SSA 再优化。

复杂度：近线性支配 + 与定值数相关的 φ 插入。

```mermaid
flowchart TD
  ENTER["进入块 B"] --> PUSH["每个在 B 定值的变量压入新名"]
  PUSH --> REW["B 内定值与使用改写为栈顶名"]
  REW --> SUCC["对每个后继：往其 φ 的对应前驱槽填栈顶名"]
  SUCC --> KIDS["递归处理支配树的子块"]
  KIDS --> POP["离开 B：弹掉本块压入的名字"]
```

<span class="marginnote">重命名栈的直觉类比：像编辑器的撤销栈——进入一个块就给用到的变量各「存档」一次，块内所有出现都读栈顶；离开块时把这一层的存档弹掉，外面看到的还是进块前的名字。支配序遍历保证「活着的影响范围」恰好等于「弹掉之前走过的子树」，所以栈不会乱。</span>

## 边界

本课不写 memory SSA 的全部 chi/mu。后课默认：标量可按 Cytron 构造 SSA。下一课析构：离开 SSA，φ 变传送。

也不把 SSA 当源语言语法。

## 小结

- Cytron：DF 上插 φ，支配序重命名。
- 剪枝减少无用 φ。
- 支配算法用 Lengauer–Tarjan 等。
- 出处：Cytron et al., 1991；Lengauer–Tarjan；Appel, 1998。
