---
title: 最弱前置条件
date: 2026-09-08
section: cs
---

# 最弱前置条件

<div class="epigraph">
<p>$\mathrm{wp}(S,Q)$ 是使 $S$ 执行后 $Q$ 成立的最弱前置。Dijkstra 用它定义语义：从后往前算，循环变成不动点。</p>
<footer>—— 据 Dijkstra, A Discipline of Programming, 1976；Winskel 整理</footer>
</div>

上一课[Hoare 逻辑](/cs/hoare-logic) 的三元组要人提供 $P$ 与不变式。缺口是**最弱前置** $\mathrm{wp}(S,Q)$：$\{P\}S\{Q\}$ 部分正确当且仅当 $P\Rightarrow\mathrm{wp}(S,Q)$（在终止版本下还有「必停」）。本课给赋值、顺序、if 的等式；while 用 $\mu$。

## 问题

$\mathrm{wp}(x:=e,Q)=Q[e/x]$。$\mathrm{wp}(S_1;S_2,Q)=\mathrm{wp}(S_1,\mathrm{wp}(S_2,Q))$。$\mathrm{wp}(\mathrm{if}\ b\ S_1\ S_2,Q)=(b\land\mathrm{wp}(S_1,Q))\lor(\neg b\land\mathrm{wp}(S_2,Q))$。循环：$\mathrm{wp}(\mathrm{while}\ b\ S,Q)$ 是最弱的 $I$，满足 $I\iff(\neg b\land Q)\lor(b\land\mathrm{wp}(S,I))$——即某个单调变换的最弱不动点 / 最大前不动点，说法随部分/完全正确而变。本课要「循环 = 不动点方程」，计算下一课 Knaster–Tarski。

$\mathrm{sp}$（最强后置）从前往后，对符号执行更亲。点名。

<span class="marginnote">数字实例：$Q$ 是 $x\gt0$，语句 $x:=x+1$，代入得 $\mathrm{wp}=x+1\gt0$，即 $x\gt-1$——这正是能让后条件成立的最大起始集合，比人手写的 $x\ge0$ 更弱。</span>

### 最弱不是「最容易证」

最弱 $P$ 允许最多起始状态，公式可能更复杂。人写的不变式往往强于 $\mathrm{wp}$，但更好证。

<span class="marginnote">Dijkstra 1976；谓词变换器。ESC/Java、Boogie 把 VC 生成建成工具，本课不写前端。完全正确的 wp 要求体降低变式。</span>

## 方法

对无循环程序从后往前算一遍 $\mathrm{wp}$。指出循环必须近似不动点或给不变式当归纳假设。对照赋值公理：它正是 $\mathrm{wp}$ 的特例。

<span class="marginnote">常见误区：「最弱」不是「最好证」——$\mathrm{wp}$ 给出的是能跑通的最大状态集，公式可能又长又怪；工程师给个更强的直觉不变式（如 $x\ge0$），证明往往反而更短。</span>

```mermaid
flowchart TD
  Q["后条件 Q"] --> WP["wp(S,Q)"]
  WP --> PRE["合法前置 P ⇒ wp"]
  WP --> FIX["while：不动点"]
```

## 机制

验证条件生成：用户给不变式，工具检查体保持、进入、退出，即若干 $\mathrm{wp}$ 蕴含。SMT 后课吃这些公式。没有不变式则要对循环求不动点——有限域上即模型检验的前/后像，再后课。

<span class="marginnote">术语翻译：验证条件（VC）就是工具替你做的几次代入检查——它不真正「跑」程序，而是把「执行后 $Q$ 成立」翻译成几条一阶公式，丢给 SMT 求解器判真假。</span>

```mermaid
flowchart TD
  P["前置 P 与不变式 I"] --> C1{"进入：P ⇒ I？"}
  C1 -->|"通过"| C2{"保持：I∧b ⇒ wp(S,I)？"}
  C2 -->|"通过"| C3{"退出：I∧¬b ⇒ Q？"}
  C3 -->|"三条全过"| OK["循环部分正确"]
  C1 -->|"失败"| FIX["加强不变式或改 P"]
  C2 -->|"失败"| FIX
```

可靠：谓词变换器与操作语义一致。

Dijkstra 的卫式命令：$\mathrm{wp}(S_0\square S_1,Q)=\mathrm{wp}(S_0,Q)\land\mathrm{wp}(S_1,Q)$ 当两条都可执行。魔命令 $\mathbf{abort}$ 的 wp 是假。循环的近似：从假迭代 $\Phi$ 得到最弱不变式的不足逼近，有限抽象域上会停。VC 生成把用户不变式当归纳假设，检查几条蕴含。

## 边界

本课不定义卫式命令全集，不引入概率 wp。不把抽象解释写完。后课默认：无环程序的 $P$ 可算；循环要不变式或不动点。下一课格上的 Knaster–Tarski。

无环程序的 $P$ 可从后往前算；循环变成谓词变换器的不动点。用户不变式是归纳假设，工具只检查蕴含。下一课用格定理保证最小/最大不动点存在。

## 小结

- $\mathrm{wp}(S,Q)$ 是使后条件成立的最弱前置。
- 赋值、顺序、if 有等式；while 是不动点。
- Hoare 三元组 $=P\Rightarrow\mathrm{wp}$。
- 出处：Dijkstra, 1976；Winskel。
