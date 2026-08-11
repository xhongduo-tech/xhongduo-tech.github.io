---
title: x86-64 调用约定与 ABI
date: 2026-08-11
---

# x86-64 调用约定与 ABI

<div class="epigraph">
<p>标准的好处在于：可供选择的实在太多了。</p>
<footer>—— 安德鲁 · 塔能鲍姆（Andrew S. Tanenbaum）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 汇编语言 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么需要 ABI

上一节的栈帧揭示了「谁压栈、谁弹栈」的机械过程，但还留着一个悬案：**函数的参数到底从哪来？返回值放哪？** 你可以把参数压进栈，调用方被调方商量好顺序就行——但 C 里 `f(1,2)` 与另一个 C 文件里的 `f` 函数、与 C 标准库、与第三方库，凭什么能互相调用？答案不是「商量」，而是一份**书面契约**。

这份契约叫 **ABI（Application Binary Interface，应用二进制接口）**：它规定参数放哪些寄存器、返回值在哪、哪些寄存器谁负责保存、栈怎么对齐。ABI 是编译器的约定俗成，没有硬件强制——违反它的后果是函数互相调用时数据错乱、栈崩溃，且不报任何编译错误。<span class="marginnote">ABI 的存在让「不同语言、不同编译器、不同库之间互相调用」成为可能：Python 的 C 扩展、Rust 的 C FFI、深度学习框架的 C++ 核心都靠它。读懂了 ABI，就理解了「链接」这层工程魔法的核心。</span>本专题以 System V AMD64（Linux/macOS 默认）为主，对照 Windows x64。

## 1 参数与返回值放在哪

System V AMD64 约定，整数与指针参数按顺序放进**参数寄存器**，第 7 个起才上栈：

$$
\text{arg}_1,\dots,\text{arg}_6 \mapsto \%\text{rdi}, \%\text{rsi}, \%\text{rdx}, \%\text{rcx}, \%\text{r8}, \%\text{r9}
$$

返回值放 `%rax`（浮点返回放 `%xmm0`）。一张完整的对应表：

| C 调用 | 寄存器 |
| --- | --- |
| `f(a, b, c, d, e, g)` | `%rdi %rsi %rdx %rcx %r8 %r9` |
| 第 7 个及以后 | 按右到左压入参数构造区（栈） |
| 返回值 | `%rax` |
| 浮点参数 | `%xmm0 ~ %xmm7`（按出现顺序） |
| 可变参数（varargs） | `%al` 存浮点参数的个数 |

于是 `f(1,2,3,4,5,6,7)` 会先把 7 压栈，再依次把 1~6 放进 `%rdi,%rsi,%rdx,%rcx,%r8,%r9`。<span class="marginnote">栈上参数的位置有讲究：调用方把第 7 个参数压入自己的栈帧（参数构造区），被调方用 `16(%rbp)`、`24(%rbp)` 访问——因为 `%rbp` 上方依次是返回地址（8 字节）和保存的 `%rbp`，正偏移 16 才是第一个栈参数。</span>

## 2 谁负责保存谁：callee-saved 与 caller-saved

函数 A 调用 B，A 正用着的寄存器，B 也会用。规则按「保存责任」分两类：

**被调用方保存（callee-saved）**：`%rbx`、`%rbp`、`%r12~%r15`。B 若想用它们，**必须先压栈保存、退出前恢复原值**——A 可以放心地假设这些寄存器在 B 返回后「原封不动」。
- **调用方保存（caller-saved）**：`%rax`、`%rcx`、`%rdx`、`%rsi`、`%rdi`、`%r8~%r11`。A 若在调用 B 之后还要用它们的值，**A 自己负责在 `call` 前保存**（压栈或挪到别处）。

<span class="marginnote">为什么两分？被调用方保存的寄存器对调用者最友好（A 不用动手备份），但 B 要为每个用到的被调用方保存寄存器付「压栈+弹栈」的开销。编译器会聪明地选择：短暂临时量放 caller-saved，长期存活的变量放 callee-saved。</span>

这个约定解释了上一节入口序里 `pushq %rbp` 与函数末尾 `popq %rbp` 的全部意义。

## 3 公式解析：寄存器保存权衡

设一个函数在任意时刻存活的局部变量集合为 $L$，可用的 callee-saved 寄存器为 $C$，则调用点上的额外成本为：

$$
\text{保存开销} = 2 \times |C \cap \text{使用}|
$$

（每次 `call` 前存、返回后取，各自一次内存访问）。拆解：

- **第一步**：$|C| = 6$（`%rbx,%rbp,%r12~%r15`），即编译器最多可免费保留 6 个「跨调用存活」的值；
- **第二步**：若存活的局部变量多于 6，超出的部分必须压栈或挪进 caller-saved 并绕过调用；
- **第三步**：于是「一个函数调用了别的函数」与「一个函数从不调用别的函数」在寄存器使用策略上完全不同——后者甚至可以放心用满 caller-saved，因为反正没人打断它。

这就是为什么编译器对「叶子函数」（不再调用其他函数）的优化特别激进：零保存成本，还能独占全部寄存器。

## 4 System V 的附加细则

- **红区（red zone）**：`%rsp` 以下 128 字节归当前函数「免费使用」，不必显式调整 `%rsp`——叶子函数可以直接 `movq %rax, -8(%rsp)` 而不做 `subq`。中断/信号处理会保护这块区域。<span class="marginnote">红区是把「少一条 `subq`」压榨到极致的产物；但内联汇编里若自己写了 `push`，会破坏对红区的假设，所以编译器生成内联汇编时总是先 `subq` 建帧。</span>
**栈对齐**：如前一篇所述，`call` 前 `%rsp` 16 字节对齐，栈帧取 16 的倍数。
- **`%rax` 的隐藏用途**：对可变参数函数（如 `printf`），调用方需在 `%al` 中写明向量寄存器的使用个数。

## 5 一个完整的编译实例

把「callee-saved 寄存器」用在真实编译里看一遍。C 代码：

```c
long f(long x) {
    long y = x * 3;        // y 要跨 helper() 调用存活
    return y + helper(2);
}
```

`-O1` 生成的典型骨架：

```
f:
    pushq %rbx                # 保存调用者的 %rbx
    movq  %rdi, %rbx          # y 挪进 %rbx
    leaq  (%rbx,%rbx,2), %rbx # %rbx = 3x
    movq  $2, %rdi
    call  helper              # helper 随便踩 %rax/%rcx/... 都没关系
    addq  %rbx, %rax          # y 安然无恙
    popq  %rbx                # 恢复调用者的 %rbx
    ret
```

`y` 被放进 callee-saved 的 `%rbx` 而非 caller-saved 的 `%rax`——正是因为在 `call helper` 期间 `%rax` 会被返回值覆盖。**callee-saved 的价值就是「跨调用免费存活」**：若改用 caller-saved，编译器得在 `call` 前压栈、返回后弹栈，多两次内存访问。这条取舍，是理解所有编译输出的钥匙。

## 6 Windows x64：另一个世界的契约

System V 只在 Linux/macOS/BSD 阵营使用；Windows 用自己的 ABI：

| 项目 | System V AMD64 | Windows x64 |
| --- | --- | --- |
| 参数寄存器 | `%rdi %rsi %rdx %rcx %r8 %r9` | `%rcx %rdx %r8 %r9` |
| 阴影区（shadow space） | 无 | 调用方预留 32 字节栈空间 |
| callee-saved | `%rbx %rbp %r12~%r15` | `%rbx %rbp %rdi %rsi %r12~%r15` |
| 参数不足 4 个时 | 不上栈 | 也仍保留 32 字节 |
| 栈对齐 | 16 字节 | 16 字节 |

Windows 要求调用方在栈上预留给被调方 **32 字节阴影区**（存放寄存器参数的可能溢出），即便参数少于 4 个也不能省。<span class="marginnote">跨平台库（如 Python 的 C 扩展）要为两套 ABI 各编译一份；Rust、Go 等语言若想用 C FFI 也得按宿主平台的 ABI 走。ABI 就是「二进制兼容」这层胶水本身。</span>手写汇编最怕的灾难之一，就是把两个 ABI 的调用约定混在一起——参数全错位，程序行为随机。

## 7 小结

- **ABI** 是二进制层的调用契约：参数寄存器、返回值、寄存器保存责任、栈对齐一网打尽。
- System V AMD64：**6 个整数参数寄存器**（`%rdi %rsi %rdx %rcx %r8 %r9`），第 7 个起压栈，返回在 `%rax`。
- **callee-saved**（`%rbx %rbp %r12~%r15`）由被调方保存，**caller-saved** 由调用方负责。
- 浮点参数走 `%xmm0~%xmm7`，varargs 用 `%al` 记个数。
- 红区 128 字节是叶子函数的免费区；`call` 前 `%rsp` 16 字节对齐。
- **易错点｜Windows x64 参数用 `%rcx %rdx %r8 %r9` 且要 32 字节阴影区，两套 ABI 不可混用**。

在下一节，我们将看到 ABI 里那几个条件码标志如何被真正消费——**控制流与跳转**：cmp、test、跳转指令家族，以及 if/while/switch 是如何被编译的。
