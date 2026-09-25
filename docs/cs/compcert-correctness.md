---
title: CompCert 与编译器正确性
date: 2026-09-08
section: cs
---

# CompCert 与编译器正确性

<div class="epigraph">
<p>对 Clight 到汇编的每一遍给出证明：若源有定义行为，目标的观察与源相关。信任基是证明助手核与公理，而不是优化直觉。</p>
<footer>—— 据 Leroy, Formal Verification of a Realistic Compiler, 2009；对照[证明助手](/cs/proof-assistants) 与[操作语义](/cs/operational-semantics) 整理</footer>
</div>

上一课[DSL](/cs/dsl-embedding) 仍可能错降。缺口是**经证明的编译**：CompCert。本课钉模拟关系与「不证明 UB 源」，模糊测试下一课是另一条质量路。不重写 CIC。

## 问题

优化编译器大，测试覆盖不到全部变换。[健全性](/cs/type-soundness) 是语言的；这里是**编译器**的：$\mathrm{beh}(S) \supseteq \mathrm{beh}(C(S))$ 一类（具体陈述看论文：未定义可更任意）。缺口是**逐遍证明**，不是 lex。

CompCert 不覆盖全部 GCC 优化，换保证。

### 正确性不是「生成代码很快」

性能另测。证明保证语义，不保证最优寄存器分配。

<span class="marginnote">术语翻译：向前/向后模拟是一种「影子戏」论证——要求源语言每走一步，目标语言都能对应走出（若干）步，且可观察行为保持一致。每一遍都对得上影子，逐遍串联，整个编译的语义保持就成立了。</span>

<span class="marginnote">Leroy 2009 CACM/POPL 系列。Coq。本课不列全部中间语言名。与 UB 课：源 UB 则定理不约束。</span>

## 方法

每遍：小步或大步语义 + 向前/向后模拟。组合。抽取 OCaml 编译器。信任：Coq 核、反汇编器、ABI 公理。

```mermaid
flowchart TD
  SRC["Clight"] --> P1["已证遍"]
  P1 --> ASM["汇编"]
  SRC --> SIM["模拟关系"]
  ASM --> SIM
```

与[SSA](/cs/ssa-form)：CompCert 有自己的 SSA 与证明负担。工业 LLVM 用测试+fuzz，少全证。

<span class="marginnote">常见误区：以为「经过证明的编译器」能让任何 C 程序都正确。定理只承诺：源程序有定义行为时，目标行为与源一致；源一旦触发未定义行为（UB），定理完全不约束——优化器做什么都不算错。保证的边界由源程序的规矩决定，不由编译器决定。</span>

## 机制

未模型化的 IO、内联汇编、并发（早期）在保证外。不要把 CompCert 当 C++ 全语言。链接：分离编译有后续工作，点名。

```mermaid
flowchart TD
  THM["逐遍模拟定理"] --> PROV["被证明的编译遍"]
  PROV --> EXT["抽取为 OCaml 代码"]
  EXT --> RUN["运行时依赖 OCaml 环境"]
  T1["信任基: Coq 核心核对器"] -.-> PROV
  T2["信任基: 反汇编器与 ABI 公理"] -.-> EXT
```

抽取代码仍要 OCaml 运行时——信任基。

<span class="marginnote">直觉类比：信任基就像地基的面积。普通编译器的「可信」建立在几十万行代码加海量测试上，地基巨大；CompCert 把地基缩到一个小的证明核对器加少数公理——需要信任的东西越少，「编译器藏着 bug」的面就越小。这就是「换保证」的实质。</span>

## 边界

本课不写 Csmith。后课默认：现实编译器可形式化子集。下一课编译器模糊测试 Csmith。

也不把 CompCert 当电子设计自动化。

## 小结

- CompCert：逐遍模拟，定义行为被保留。
- UB 源不在定理内；优化范围小于 GCC。
- 信任基是助手核与公理。
- 出处：Leroy, 2009；对照 Wright–Felleisen、Coq。
