---
title: SSA 析构与 φ 消除
date: 2026-09-08
section: cs
---

# SSA 析构与 φ 消除

<div class="epigraph">
<p>机器没有 φ。析构在每条前驱边末尾插入传送，把 SSA 名合并回可着色的存储。错序会引入交换问题。</p>
<footer>—— 据 Briggs, Harvey and Simpson, Practical Considerations for SSA Destruction；Cytron et al.；Appel 整理</footer>
</div>

上一课[Cytron 构造](/cs/ssa-construction-cytron)插入 φ。后端[寄存器分配](/cs/regalloc-color)前通常离开 SSA。缺口是**析构**：φ 变成 `mov`。经典坑：并行 φ 的交换（`x,y = y,x`）需要临时。本课钉边拷贝与关键边，不写着色。

## 问题

φ 语义是「沿进入边同时选择」。实现：在前驱块末 `x3 = x1` 或 `x3 = x2`。若同一块两个 φ 交叉赋值，串行 `mov` 会覆盖。缺口是**并行拷贝的串行化**，不是 DF。

<span class="marginnote">「并行 φ」的坑可以类比成换两杯水：左手橙汁、右手清水，要在不洒的前提下交换——直接倒会混，必须先腾一个空杯。机器指令也是一次只能动一杯，所以 `x,y = y,x` 必须引入临时 `t`：`t = x; x = y; y = t`。编译器漏了这一步，两个变量的值就悄悄变成同一个。</span>

关键边：无唯一插入点，先拆块。

### 析构不是「删掉 φ 不管」

随便删 φ 会丢失汇合。必须有拷贝或让分配器同时分配 φ 与操作数到同一寄存器（合并）。

<span class="marginnote">Briggs 等 SSA destruction。Sreedhar 的方法减少拷贝。LLVM 在分配中处理 PHI。本课常规：先拆 φ 再着色，或 SSA 上着色（Chordal）点名。</span>

## 方法

拆关键边。为每个 φ 与每个前驱生成拷贝。对块末并行拷贝图：找交换环，用临时或 swap。再跑拷贝传播。

```mermaid
flowchart TD
  PHI["φ"] --> CRIT["拆关键边"]
  CRIT --> MOV["前驱末传送"]
  MOV --> SER["并行拷贝串行化"]
```

与[合并](/cs/coalescing-splitting) 后课：析构产生的 `mov` 正是合并的输入。

<span class="marginnote">关键边翻译一下：一条「上游分岔、下游汇合」的边——上游还有别的出口、下游还有别的入口，所以这条边自己没有「末尾」可放指令。解决办法是把它劈成两段，中间凭空造一个空块，指令就放进那个空块里。不先拆边，φ 的拷贝要么放不进去，要么会被上游的其他后继错误地执行到。</span>

## 机制

未定义的 φ 操作数（未初始化变量）须按语言填毒值或拒绝。不要在析构后还当 SSA 做 SCCP。

异常边：前驱是 invoke 的 unwind，插入点在专用边块。

```mermaid
flowchart TD
  PAIR["并行 φ：x,y = y,x"] --> NAIVE["直接串行：先 x = y"]
  NAIVE --> WRONG["y 的旧值已被覆盖，语义错"]
  PAIR --> TEMP["引入临时 t 保存一个源"]
  TEMP --> ORDER["x = y 再 y = t，各读各的源"]
  ORDER --> RIGHT["结果等于同时赋值，语义正确"]
```

<span class="marginnote">初学者容易以为析构产生的 `mov` 会原样留在最终机器码里、白白变慢。实际上这些拷贝正是下一课「合并」的原料：分配器把源与目标分到同一寄存器，`mov` 就消失了。析构的职责只是「语义正确地降级」，优化交给后面的阶段——这也是「先析构再着色」这条流水线的分工。</span>

## 边界

本课不写 Briggs 着色。后课默认：φ 在后端前变成传送。下一课自然循环：许多循环优化要在 SSA 上认循环，构造/析构顺序是「优化在 SSA，分配前析构」。

也不把析构当 DCE。

## 小结

- 析构：φ → 前驱边拷贝；关键边先拆。
- 并行赋值要处理交换。
- 与随后的合并、着色衔接。
- 出处：Cytron et al.；Briggs, Harvey and Simpson；Appel。
