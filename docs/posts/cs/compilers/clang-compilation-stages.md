---
title: 用 Clang 观察编译流程的各个阶段
date: 2026-08-07
---

# 用 Clang 观察编译流程的各个阶段

<div class="epigraph">
<p>把编译器的每道工序停下来看一遍，理论的每一层就都有了形状。</p>
<footer>—— 仿自 LLVM 文档对 Clang 阶段的描述</footer>
</div>

<div class="article-byline">
<p>第三级 · 编译原理 ｜ 《编译原理》（龙书）外延 · LLVM 专题 ｜ 2026-08-07</p>
</div>

## 为什么从 Clang 观察编译流程开始

前 74 篇的理论把编译分成了词法、语法、语义、IR、优化、代码生成——但纸上得来终觉浅。**Clang**（LLVM 的 C/C++ 前端）把每一道工序都做成了可观察的中间产物：词法分析出记号、语法分析出 AST、生成出 LLVM IR、优化后 IR 变好、后端出汇编。用几个命令，就能把前面所有章节「亲眼看一遍」。<span class="marginnote">「观察编译流程」是学习编译原理最有效的一步：`clang -E`（预处理）、`-fsyntax-only`（语法）、`-emit-llvm`（IR）、`-S`（汇编）——每换一个 flag，就看到编译的一道工序。把教材里的抽象阶段映射到具体命令，理论就落地了。</span>

## 1 Clang 观察命令全景

给定 C 程序 `demo.c`，Clang 的各个阶段对应命令：

| 阶段 | 命令 | 产物 |
| --- | --- | --- |
| 预处理 | `clang -E demo.c` | 展开宏后的源文本 |
| 词法/语法分析 | `clang -fsyntax-only demo.c` | 检查并报错（不产出） |
| 转储 AST | `clang -Xclang -ast-dump demo.c` | 语法树/语义树 |
| 生成 LLVM IR | `clang -S -emit-llvm demo.c` | `demo.ll`（文本 IR） |
| 优化 IR | `clang -O2 -S -emit-llvm demo.c` | 优化后的 `demo.ll` |
| 生成汇编 | `clang -S demo.c` | `demo.s` |
| 生成目标文件 | `clang -c demo.c` | `demo.o` |
| 链接 | `clang demo.o -o demo` | 可执行文件 |

**逐步推进**：每一条命令都是「走到某一步停下」，让观察者看清这一步的产物。<span class="marginnote">「-Xclang -ast-dump」是 Clang 的隐藏彩蛋：它把完整 AST（含类型、作用域、语义信息）转储成树形文本——语法分析（第三篇）与语义分析（第五篇）的产物一次性可视化。对着一个小程序跑它，比读十页教材更能理解「语法树长什么样」。</span>

## 2 一个观察实例：从源到 IR

用一段小程序观察整个流程：

```c
int square(int x) {
    return x * x;
}

int main() {
    int a = square(3);
    return a + 1;
}
```

**生成 IR**（`clang -S -emit-llvm demo.c`）：

```llvm
define i32 @square(i32 %x) {
  %1 = mul i32 %x, %x
  ret i32 %1
}

define i32 @main() {
  %1 = call i32 @square(i32 3)
  %2 = add i32 %1, 1
  ret i32 %2
}
```

观察点：

- **SSA**：`%1`、`%2` 每个只赋值一次。
- **强类型**：`i32` 显式标注。
- **函数**：`define i32 @square(i32 %x)`——返回类型、参数类型都在 IR 里。

`square(3)` 是常量参数——优化器（-O2）会把它折叠吗？下面看。<span class="marginnote">「IR 是 SSA 的活教材」：`%1 = mul i32 %x, %x` 之后，所有用 `%1` 的地方都是「x 的平方」——定义-使用关系写死在名字里。对照第七十三节讲的 SSA 规则，这段 IR 就是最直观的示例。</span>

## 3 优化前后的对比

**优化**（`clang -O2 -S -emit-llvm demo.c`）：

```llvm
define i32 @square(i32 %x) {
  %1 = mul i32 %x, %x
  ret i32 %1
}

define i32 @main() {
  ret i32 10          # square(3) 被折叠成 9，9+1 = 10
}
```

观察点：

- **常量传播 + 折叠**：`call @square(i32 3)` 被内联，`3*3=9` 折叠，`9+1=10` 再折叠——`main` 直接返回 10。
- **过程间 + 内联**：`square` 被内联进 `main`，调用开销消除。
- **死代码**：无用的指令被清掉。

这一行 `ret i32 10`，是内联、常量传播、常量折叠、死代码消除四道优化合力的产物——前几篇的优化理论在这里「眼见为实」。<span class="marginnote">「ret i32 10」的震撼：源程序里 `main` 调用了 `square`、算了 `a+1`，优化后直接返回常量——编译器把所有运行期计算都提前到了编译期。这就是「优化是把运行期工作搬到编译期」的最直观演示。</span>

## 4 公式解析：观察命令与阶段的一一对应

每条 Clang 命令对应「编译流程的一个前缀」：

$$\text{compile}(src) = \text{link} \circ \text{assemble} \circ \text{codegen} \circ \text{opt} \circ \text{frontend}(src)$$

$$\text{flag 决定在哪个 } \circ \text{ 处停止: } -E \text{ 停在 frontend 前},\ -emit-llvm \text{ 停在 opt 后},\ -S \text{ 停在 codegen 后}$$

- **第一步，函数复合**：整个编译是「前端 → 优化 → 代码生成 → 汇编 → 链接」的函数复合。
- **第二步，flag 即断点**：`-E` 在预处理后停、`-fsyntax-only` 在语义检查后停、`-emit-llvm` 在优化后停、`-S` 在汇编生成后停、`-c` 在目标文件后停。
- **第三步，观察即验证**：每个断点看一眼产物，就等于「验证」这一阶段的正确输出——教学与调试同法。

**「flag = 编译流程的断点」让每一道工序都可以被单独观察与调试**——这是 LLVM 工具链设计者给学习者的礼物。

## 5 进一步观察：pass 与后端

- **观察单个 pass**：`opt -passes=gvn demo.ll -S`——只跑一个 pass，看 IR 变化。
- **观察 pass 流水线**：`opt -O2 demo.ll -S`——跑整套优化，看最终 IR。
- **观察后端**：`llc demo.ll`——把 IR 变成汇编（指令选择、寄存器分配的可视化）。

用 `opt` 把第八篇的每个优化（`-gvn`、`-licm`、`-sccp`）单独跑一遍，就能亲眼看到「CSE 把重复计算去掉」「外提把循环不变计算搬出去」——理论全部可视化。<span class="marginnote">「opt 是优化器的示波器」：`opt -passes=gvn demo.ll -S` 单独观察值编号的效果，`opt -passes=licm` 单独观察循环外提——每个 pass 的输入输出对比，就是最生动的优化教学。LLVM 团队调试优化器也靠这套「单 pass 观察」。</span>

## 6 小结

- **Clang 观察命令**：`-E`（预处理）、`-fsyntax-only`（语法）、`-ast-dump`（AST）、`-emit-llvm`（IR）、`-S`（汇编）、`-c`（目标文件）、链接。
- 一段 C 程序通过 Clang 逐阶段可见：**预处理 → 语法 → IR → 优化 IR → 汇编**。
- 优化前后对比：内联 + 常量传播 + 折叠 + 死代码让 `main` 直接 `ret 10`——理论眼见为实。
- **flag = 编译流程的断点**：每个 flag 停在某个阶段，让每道工序可单独观察与验证。
- `opt -passes=X` 可单独观察单个优化 pass；`llc` 观察后端——整套流水线全可视化。

在下一节，我们用 LLVM 亲手实现一门语言的前端：**Kaleidoscope 教程解读**。
