---
title: 线程化分派
date: 2026-09-08
section: cs
---

# 线程化分派

<div class="epigraph">
<p>把下一条字节码的处理地址直接写进流：间接跳转到该地址，而不是每次回到中央 `switch`。减少分派的间接预测失败。</p>
<footer>—— 据 Ertl and Gregg, The Structure and Performance of Efficient Interpreters；Bell, Threaded Code, 1973 整理</footer>
</div>

上一课[栈式对寄存器式](/cs/stack-vs-register-vm) 选了编码。缺口是**分派**：`switch` 每个 case 后跳回循环头，分支预测惨。线程化：computed goto（GCC 标签作值），每条 handler 末尾 `goto *pc++`。本课钉 indirect threading / direct / context threading，不写 JIT。

## 问题

解释器热路径是分派，不是加法本身。线程化把「取下一条入口」变成间接跳。缺口是**控制流形状**，不是操作数编码。

直接线程：指令流就是地址数组。间接：操作码仍是字节，经表变地址。Token threaded：折中。

### 线程化不是操作系统线程

名字来自「用地址串起来」。与 pthread 无关。不要混。

<span class="marginnote">Bell 1973。Ertl–Gregg。Forth 传统。C 的 `goto *` 是 GNU 扩展，可移植要用跳表。</span>

## 方法

实现：`void* table[256]`；handler 末 `goto *table[*pc++]`。超级指令：把高频对（load+add）合成一 handler，减分派次数——与[内联](/cs/inlining-heuristics) 同思想在解释器里。

```mermaid
flowchart TD
  OP["操作码"] --> TBL["入口表"]
  TBL --> H["handler"]
  H --> NEXT["间接跳下一条"]
```

与 PGO：按轮廓选超级指令。与 JIT：热点仍编译，冷路径留线程化解释。

## 机制

间接跳仍难预测，但比「都跳回同一 switch」好。代码体积：handler 复制。调试：PC 仍要可映射源。

两种控制流形状的差别：

```mermaid
flowchart LR
  subgraph SW["中央 switch"]
    A1["取指令"] --> S1["跳回 switch 判断"]
    S1 --> H1["执行 handler"]
    H1 --> S1
  end
  subgraph TH["线程化分派"]
    A2["取指令"] --> H2["执行 handler"]
    H2 --> A2
  end
```

<span class="marginnote">直觉类比：中央 switch 像所有快递都要回总仓分拣一次再发出；线程化像每件包裹分拣完就写着下一站地址直发。省下的正是每条指令一次的「绕回路口」——它是最频繁的控制流，预测失败的代价也乘上了这条频率。</span>

不要在 handler 里调用会改变栈对齐的未知函数而不保存 VM 状态。

## 边界

本课不写方法 JIT。后课默认：高效解释器用线程化分派。下一课方法 JIT 与热点。

<span class="marginnote">术语翻译：「线程化分派」里的线程不是操作系统线程——是说指令流像用「下一跳地址」一根线串起来：每条 handler 干完活直接跳去下一条，不再绕回中央路口。名字来自 Bell 1973 的 threaded code。</span>

<span class="marginnote">为什么重要：分支预测器对两种形状的待遇不同。switch 里每个 case 都回到同一个地址，预测器难猜该出哪个方向；handler 末尾的间接跳目标虽各不相同，但与操作码强相关，训练之后命中率明显更高——这就是同一解释器换个分派方式就能快百分之几十的原因。</span>

也不把线程化当 GPU warp。

## 小结

- 线程化：handler 末间接跳，去掉中央 switch。
- 超级指令减分派。
- 仍是解释器，不是 native 编译。
- 出处：Bell, 1973；Ertl and Gregg。
