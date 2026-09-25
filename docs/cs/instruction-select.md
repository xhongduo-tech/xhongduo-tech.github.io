---
title: 指令选择
date: 2026-09-08
section: cs
---

# 指令选择

<div class="epigraph">
<p>把 IR 运算树（或 DAG）盖上目标机的指令模板；同一树可以有多种覆盖，代价不同。</p>
<footer>—— 据龙书第 8 章；Aho, Ganapathi and Tjiang, Code Generation Using Tree Matching, 1989 整理</footer>
</div>

上一课[可用表达式](/cs/available-expr)仍在虚拟名上。本课不着色。缺口是：三地址的 `+`、`load` 要变成[RISC-V 整数指令](/cs/riscv-int-isa)或另一 ISA 的具体 opcode。选择可以局部树匹配，也可以 tiling。寄存器还当无限，名字仍是虚拟的。

## 问题

IR 是目标无关的。`t3 <- t1 + t2` 在 RISC-V 是 `add`，在 CISC 可能是带访存的一条。缺口是模板：树模式 $\leftrightarrow$ 指令 + 代价。最大吞或动态规划选最小代价覆盖。本课用树模式直觉，不写完整 BURG 生成器。

<span class="marginnote">直觉类比：指令选择像拿一套现成的贴纸去盖一棵运算树——每张贴纸是一个指令模板，可以只盖一个节点，也可以连枝带叶盖住一小撮；盖法不止一种，目标是找出总代价最小的完整覆盖。</span>

组成课已有指令格式与寄存器堆；本课只问「哪条指令实现这个 IR 节点」。调用序列的参数寄存器留给 ABI。

### 选择不是分配

选出 `add rd, rs1, rs2` 仍带虚拟寄存器。分配把虚拟映到 `x10` 或溢出。顺序通常先选后分配，或二者交错；主干先选。窥孔还可以在选完后改写。

<span class="marginnote">常见误区：初学者容易以为这一步结束寄存器就叫 `x10` 之类的真名了；此时输出的仍是 `t3` 这类虚拟名，要等寄存器分配才落位。另外选错模板通常只是多几条指令、慢一点，并不会改变语义。</span>

<span class="marginnote">龙书 8.9 节树重写。Aho–Ganapathi–Tjiang 的树匹配。RISC 上选择往往接近 1:1，CISC 上 tiling 收益大。本课两边都承认。</span>

## 方法

把基本块内 IR 看成森林/DAG。对每个根，用动态规划或贪婪匹配盖模板。访存指令带宽度，来自类型课的 size。非法组合（未对齐）按 ABI 与 ISA 拒绝或拆成多条。

```mermaid
flowchart TD
  IR["IR 树 / DAG"] --> PAT["ISA 模板"]
  PAT --> TILE["覆盖"]
  TILE --> MI["机器指令 + 虚拟寄存器"]
```

[指令格式](/cs/instruction-format)决定立即数能否塞进一条；放不下则先 `lui`/`addi` 一类，仍是选择。

## 机制

代价：周期或长度。错误选择会多指令，不一定错语义。副作用（标志位）在 CISC 上要进模板。RISC-V 整数子集本课够用；浮点、向量不在本课。

一个具体问题：立即数放不进单条模板时，选择阶段怎么拆。

```mermaid
flowchart TD
  E["IR：t3 ← t1 + 100000"] --> P1["模板 A：addi rd, rs1, imm"]
  P1 --> C1{"12 位立即数放得下吗"}
  C1 -->|"放不下"| P2["模板 B：lui 高 20 位 + addi 低 12 位"]
  P2 --> OUT["两条指令，结果仍叫 t3"]
```

<span class="marginnote">数字实例：RISC-V 的 I 型立即数只有 12 位，能表达 -2048 到 2047。要给 `t1` 加 100000，一条 `addi` 装不下，选择阶段就得换成 `lui` 装高 20 位、`addi` 补低 12 位的组合模板。</span>

与超标量、流水线课：选择可以避开已知冒险对，但不代替分配与调度。本课不排流水线调度。

## 边界

本课不填着色、不写调用惯例细节、不做全局指令调度。窥孔是更窄窗口的本地改写，后两课才到。后课默认：虚拟寄存器机器指令已选出。冲突图着色把虚拟收成物理寄存器或溢出。

## 小结

- 指令选择 = ISA 模板覆盖 IR；代价可选优。
- 输出仍含虚拟寄存器。
- RISC 近 1:1；复杂 ISA 才显 tiling。
- 出处：Aho et al., 龙书第 8 章；Aho, Ganapathi and Tjiang, 1989。
