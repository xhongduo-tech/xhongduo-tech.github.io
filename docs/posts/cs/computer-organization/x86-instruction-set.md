---
title: x86 指令体系结构：数据传送、算术逻辑与控制转移指令
date: 2026-08-08
---

# x86 指令体系结构：数据传送、算术逻辑与控制转移指令

<div class="epigraph">
<p>x86 是一部用兼容性写成的活化石——每一代 CPU 都在给这座古老的教堂添一座新塔。</p>
<footer>—— 佚名，逆向工程研究者</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机组成原理 ｜ 唐朔飞《计算机组成原理》第四章 · CS:APP §3.2–3.6 ｜ 2026-08-08</p>
</div>

## 为什么拿 x86 当「指令解剖标本」

前面学的指令格式、寻址方式，都是抽象骨架；本课用 **x86-64** 把骨架填上血肉。x86 是统治桌面与服务器 40 年的指令集，理解它的三类核心指令——**数据传送、算术逻辑、控制转移**——你就能读懂几乎所有 C 代码编译出的汇编，也能理解 RISC 与 CISC 之争在真实世界的落点。这是 CS:APP 第三章的精髓，也是逆向工程与性能分析的基本功。<span class="marginnote">本节用的 x86-64 是 AT&T 语法（`movq %rax, %rbx`）还是 Intel 语法（`mov rbx, rax`）取决于工具链。CS:APP 用 AT&T 语法（源在前，与直觉相反），我们这里标注两种避免混淆。</span>

## 1 寄存器与数据类型x86-64 的通用寄存器 16 个，每个都有「历史别名」：低 32 位叫 `eax`，低 16 位 `ax`，低 8 位 `al`——这是兼容 8086 的遗产。

| 寄存器 | 特殊用途 |
| --- | --- |
| `%rax` | 返回值、累加 |
| `%rbx` | 被调用者保存 |
| `%rcx` / `%rdx` | 第 4、3 个参数 |
| `%rsi` / `%rdi` | 第 2、1 个参数 |
| `%rsp` | 栈指针 |
| `%rbp` | 帧指针（可选） |

**数据类型宽度**：`b`（8 位字节）、`w`（16 位字）、`l`（32 位双字）、`q`（64 位四字）——指令助记符后缀对应宽度，如 `movb/movw/movl/movq`。<span class="marginnote">函数调用的前 6 个整数参数依次放在 `%rdi, %rsi, %rdx, %rcx, %r8, %r9`（System V ABI）——这是 x86-64 的调用约定。理解它，才能读懂任何函数的开头几条指令。</span>

## 2 数据传送指令**`mov` 家族**是数据传送的主力：

```asm
movq %rax, %rbx    ; rbx = rax（寄存器间）
movq $0x3f, %rax   ; rax = 0x3f（立即数）
movq (%rdi), %rax  ; rax = M[rdi]（访存）
movq %rax, (%rdi)  ; M[rdi] = rax（写内存）
```

**重点：x86 不允许「内存到内存」一条 mov**——必须经过寄存器中转。这与 RISC 的 Load/Store 精神一致。

**符号扩展与零扩展**：`movslq`（有符号扩展到 q）、`movzbl`（零扩展到 l）——见《符号扩展》一课。<span class="marginnote">x86 有一条奇怪的指令 `leaq`（load effective address）：它不访存，只计算地址。`leaq (%rdi,%rsi,4), %rax` 让 `rax = rdi + rsi*4`——编译器常拿它做「不相关的算术」（乘常数、加偏移），因为它能在一个周期算完。</span>

## 3 算术与逻辑指令**算术**：`addq/subq/imulq`（有符号乘）、`divq/idivq`（除法，商在 rax、余数在 rdx）。

```asm
addq %rax, %rbx    ; rbx += rax
subq $8, %rsp      ; rsp -= 8（压栈预留）
```

**逻辑与移位**：`andq/orq/xorq/notq`、`salq`（左移）、`sarq`（算术右移）、`shrq`（逻辑右移）。

**重点：`imulq` 是补码乘法，`mulq` 是无符号乘法**——同一串位按哪种解读由指令决定（见《操作数类型》）。<span class="marginnote">编译器把 `x * 8` 优化成 `leaq (%rax,%rax,4), %rax; salq $1` 之类——用移位和加法拼出乘法。而 `x / 8` 若 x 有符号，需要 `sarq` + 加偏移修正（因为算术右移向下取整，C 除法向零取整），这个「修正序列」是逆向工程里辨认除法的特征码。</span>

## 4 控制转移指令**无条件跳转**：`jmp label` 或 `jmp *%rax`（寄存器间接跳转，函数指针）。

**条件跳转**：读标志寄存器，`je/jne`（相等/不等）、`jg/jl`（有符号大于/小于）、`ja/jb`（无符号大于/小于）、`js/js`（符号）。

**比较与测试**：`cmpq a, b`（算 b-a 更新标志不写结果）、`testq a, b`（算 a&b 更新标志）。

```asm
cmpq %rsi, %rdi    ; 比较 rdi 与 rsi
jle  .L2           ; 若有符号 rdi <= rsi 则跳转
```

**过程调用**：`call`（压返回地址 + 跳转）、`ret`（弹返回地址 + 跳回）。

<span class="marginnote">条件跳转的助记符语义与比较方向要小心：`cmp a, b` 实际算 `b - a`，所以 `jle` 跳转条件是「b <= a」（注意 AT&T 语法操作数顺序相反）。Intel 语法下 `cmp a, b` 是 `a - b`——两种语法方向相反，是初学汇编最大的坑。</span>

## 5 公式解析：一个 C 表达式的汇编翻译以 `x = a + b*c;` 为例，观察「运算符 → 指令」的翻译：

```asm
movq  %rcx, %rax    ; rax = c（c 在 rcx）
imulq %rdx, %rax    ; rax = c * b（b 在 rdx）
addq  %rsi, %rax    ; rax = c*b + a（a 在 rsi）
movq  %rax, x(%rip) ; x = rax（写全局）
```

逐项拆解：

- **第一步，操作数就位**：三个变量分别已在参数寄存器 `rsi/rdx/rcx`。
- **第二步，优先级**：先算 `b*c`（乘法），再算加法——编译顺序由运算优先级决定。
- **第三步，访存**：结果写回全局变量 `x`，用 RIP 相对寻址（`x(%rip)`，与位置无关）。
- **第四步，观察**：每条指令只做一件基本事——x86 的算术也可以在内存上（`imulq (%rdx), %rax`），编译器选寄存器版更快。

**核心结论：读懂汇编 = 把「寄存器 + 指令 + 寻址」翻译回高级语言表达式**；优先级、类型、指针都由指令选择体现。<span class="marginnote">用 `objdump -d` 或 `gcc -S` 看任意 C 函数的汇编，是训练这个技能的最好方式。CS:APP 的配套实验 bomb lab 就是「给你汇编，叫你反推 C 程序」——本课的知识足够你入门了。</span>

## 6 拓展：x86 汇编速记**五条必记**：
- 参数寄存器顺序：rdi→rsi→rdx→rcx→r8→r9，返回值 rax。
- mov 不允许内存到内存，必须经寄存器中转。
- leaq 是「算地址不当访存」——编译器常用它做乘法。
- imulq 有符号、mulq 无符号——同串位不同解读。
- cmp 算 b-a 更新标志，jle 条件 b<=a（AT&T 方向相反）。

**辨析｜易错点**：AT&T 语法源在前（`movq %rax, %rbx` 是 rbx=rax），与 Intel 相反——方向反是初学最大坑。

- 自测 1：前 6 个参数寄存器顺序？
- 自测 2：leaq 为什么能做算术？
- 自测 3：AT&T 与 Intel 语法差异？

## 7 一句话回顾- 参数 rdi→rsi→rdx→rcx→r8→r9，返回 rax。
- mov 内存到内存不允许。
- AT&T 源在前，与 Intel 方向相反。

## 8 小结- x86-64 寄存器：16 个通用寄存器，`%rax` 返回值、`%rdi/%rsi...` 参数、`%rsp` 栈指针。
- **数据传送**：`mov` 家族，不允许内存到内存；`leaq` 是「算地址不当访存」的怪胎。
- **算术逻辑**：`add/sub/imul`、`and/or/xor`、移位；有符号/无符号用不同指令。
- **控制转移**：`jmp/je/jg/ja` 读标志；`cmp/test` 更新标志；`call/ret` 做过程调用。
- 汇编翻译 C：优先级定顺序、类型定指令、指针定寻址——多读反汇编是唯一捷径。

- rax 返回值、rdi/rsi/rdx/rcx/r8/r9 参数
- mov 内存到内存不允许
- leaq 算地址不当访存
- imul 有符号、mul 无符号
- cmp 算 b-a 更新标志
- jle 条件 b<=a（AT&T）
- call 压返回地址、ret 弹回
- AT&T 源在前、Intel 目的在前
在下一节，我们处理数据在内存里的排列——**数据的对齐存放与大端、小端存储模式**。
