---
title: 浮点与 SIMD 指令
date: 2026-08-11
---

# 浮点与 SIMD 指令（SSE/AVX/NEON）

<div class="epigraph">
<p>大致正确，好过精确的错误。</p>
<footer>—— 约翰 · 梅纳德 · 凯恩斯（John Maynard Keynes）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 汇编语言 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从浮点与 SIMD 开始

到目前为止，我们看到的全是整数算术——但科学计算、机器学习、图形学的世界由浮点数统治。训练一个大模型，`y = x · w` 里每一次乘加都是浮点运算；而 `double` 的表示、精度、指令形态，与 `int` 完全是另一套体系。这一节把 CS:APP §3.11 的 SSE/AVX 浮点指令讲透：从 IEEE 754 的位表示，到 `%xmm`/`%ymm` 寄存器，到 AVX 的「一条指令算多个数」。

**SIMD（Single Instruction, Multiple Data，单指令多数据）** 是浮点世界的效率引擎：一条 `vaddpd %ymm1, %ymm2, %ymm0` 同时把 4 个 double 加起来。大模型推理里的矩阵乘法加速、FlashAttention 的算子，底层全靠 AVX/NEON 这类向量指令撑起每秒几万亿次的运算——这是「从极限到大模型」里，汇编这层离大模型最近的地方。<span class="marginnote">x86 的浮点历史是一段「三套并存」的演进：x87 栈式 FPU（`%st(0)`~`%st(7)`，80 位扩展精度）→ SSE2（128 位 XMM，标量与打包）→ AVX（256 位 YMM，三操作数）。如今 x87 只作历史遗留，编译器默认走 SSE2/AVX。</span>

## 1 浮点数的位表示：IEEE 754

在讲指令之前，必须先回答「浮点数在内存里长什么样」。IEEE 754 的 `double` 用 64 位：

$$
x = (-1)^{s} \times (1 + f) \times 2^{e - 1023}
$$

- **s（1 位）**：符号位；
- **e（11 位）**：阶码，无符号整数，真实指数是 $e - 1023$（偏移 1023）；
- **f（52 位）**：尾数的小数部分，规格化数隐含首位 1。

`float` 是它的 32 位版：1+8+23 位，偏移 127。`double` 的**机器精度**（machine epsilon）$\epsilon = 2^{-52} \approx 2.22\times 10^{-16}$，即最多可靠约 15~16 位十进制数字。

<div class="marginnote">规格化数把首位 1「隐含」了：任何非零规格化 double 的尾数都写成 $1.f\ldots$，那个 1 不用存，白赚 1 位精度。代价是 0、次正规数、$\infty$、NaN 要单列一套编码——这是 IEEE 754 里专门的分支，也是 `x != x` 能检测 NaN 的原因。</div>

## 2 公式解析：一个 double 的精度到底能存多大

把「多少位尾数」翻译成「多少位十进制数字」：

$$
\text{有效十进制位数} \approx \log_{10}(2^{52}) = 52 \times \log_{10} 2 \approx 15.65
$$

拆三步：

- **第一步**：52 位尾数能区分 $2^{52}$ 个等间距的量级步长；
- **第二步**：$2^{52} \approx 4.5\times 10^{15}$，即约 15.65 位十进制；
- **第三步**：所以 `double` 保证约 15~16 位十进制有效数字，`float` 只有约 $23\log_{10}2 \approx 6.9$ 位（约 7 位）。

这就是为什么累计误差敏感的科学计算用 `double`、而大模型参数权重可以用 `float16`/`bfloat16` 压缩——**位数越少，省的内存越多，代价是有效数字变短**。`bfloat16` 甚至砍到尾数只有 7 位，只保指数范围（≈float32），靠的就是「大模型梯度幅度跨度大、尾数精度相对不敏感」这一经验事实。

## 3 float 与 double 的取舍

`float` 省一半内存与带宽，但只有约 7 位有效数字；`double` 16 位，是 C 里浮点运算的默认宽度（`float` 参与表达式时通常被提升为 `double` 计算）。工程经验：

- **图像、音频、游戏中间量**：`float` 足够，省缓存、省带宽；
- **科学计算、累加求和**：用 `double`，否则误差随项数累积；
- **大模型权重**：用 `float16` / `bfloat16`，以「指数范围优先、尾数让位」换 2 倍内存与带宽——梯度跨越多个数量级时，指数范围比尾数精度更值钱。

同一条运算，`double` 走 64 位、`float` 走 32 位，寄存器与指令都不同（`movsd` 对 `movss`），混用必须显式转换：`cvtsd2ss`（double→float）、`cvtss2sd`（float→double）。<span class="marginnote">编译器对 `float` 变量仍可能偷偷用 `double` 中间量计算，直到存回才截断——这是 C 语言「FLT_EVAL_METHOD」规定的自由度。要严格按 float 算，得靠 `-ffloat-store` 或内联汇编。</span>

## 4 SSE2 标量浮点：%xmm 寄存器

x86-64 提供 16 个 **128 位 XMM 寄存器** `%xmm0~%xmm15`，SSE2 的**标量浮点指令**只用它的低 64/32 位：

```
movsd %xmm1, %xmm0     # 双精度：%xmm0 = %xmm1
addsd %xmm1, %xmm0     # %xmm0 += %xmm1（标量 double 加法）
mulsd %xmm1, %xmm0     # 标量乘法
cvtsi2sdq %rdi, %xmm0  # 整数 %rdi → double，放入 %xmm0
cvtsd2si %xmm0, %rax   # double → 整数（截断）
ucomisd %xmm1, %xmm0   # 比较：设条件码 CF/ZF/PF
```

浮点运算**不进通用寄存器**：参数用 `%xmm0~%xmm7` 传（System V），返回值在 `%xmm0`。比较指令 `ucomisd` 把结果写进**条件码**（CF/ZF）——于是上一节的 `j*` 跳转可以接着用，只是浮点没有「无符号」的概念，NaN 还要额外用 PF 位判。

## 5 编译器眼中的 double：一个完整例子

考虑 C 函数

```c
double f(double a, double b) {
    return a * 2.0 + b;
}
```

`-O2` 下的典型汇编：

```
  mulsd  %xmm1, %xmm0     # a * 2.0（%xmm1 里预存 2.0）
  addsd  %xmm1, %xmm0     # 2a + b
  ret
```

乘常数 2.0 甚至被优化成 `addsd %xmm0, %xmm0`（乘 2 = 自加）——**浮点指令的形态和整数如出一辙，只是换了寄存器家族**。<span class="marginnote">注意浮点加法的结合律不成立：$(a+b)+c \neq a+(b+c)$ 在舍入下可能不同，所以编译器不敢随便重排浮点运算——除非你开 `-ffast-math` 让它放弃 IEEE 语义。这是浮点优化与整数优化本质不同的一处。</span>

## 6 AVX：256 位与三操作数

**AVX** 把 XMM 扩展到 256 位 **YMM 寄存器**（`%ymm0~%ymm15`），一条指令同时处理 4 个 double 或 8 个 float：

```
vaddpd %ymm1, %ymm2, %ymm0   # %ymm0 = %ymm2 + %ymm1（4 个 double 打包加）
vmulpd %ymm1, %ymm2, %ymm0   # 打包乘
```

三操作数形式（`v` 前缀）是 AVX 的招牌：**源操作数不被覆盖**，`%ymm0 = %ymm2 + %ymm1`，这消除了 SSE 双操作数的「先搬再算」，流水线友好得多。ARM 世界的对应物是 **NEON**（`vaddq_f64` 等），语义类似、编码不同。<span class="marginnote">AVX-512 把向量再翻倍到 512 位（`%zmm`，一次 8 个 double），但更高的频率/功耗、以及「降频」问题让它在部分场景争议不断。深度学习算子库会做<strong>运行时 CPU 分派</strong>：检测机器支持 AVX2 还是 AVX-512，选对应的 kernel 版本。</span>矩阵乘法加速器的核心思路，就是把乘加循环**分块**并尽量用向量指令一次算多组——这就是「SIMD 优化」的全部直觉。

## 7 小结

- IEEE 754：`double` 是 1+11+52 位，$\epsilon=2^{-52}$，约 **15~16 位十进制有效数字**；`float` 约 7 位。
- SSE2 提供 16 个 **`%xmm` 128 位寄存器**，标量浮点用 `movsd/addsd/mulsd/divsd/cvtsd2si`，比较走 `ucomisd` → 条件码。
- 浮点参数/返回值走 `%xmm0~%xmm7` / `%xmm0`，不进通用寄存器。
- **AVX**：256 位 `%ymm`，`v` 前缀三操作数，一条指令算 4 个 double；ARM 对应 NEON。
- **易错点｜浮点不满足结合律，编译器默认不敢重排浮点运算**。
- 大模型推理的矩阵乘法，机器层就是层层展开的 SIMD 打包乘加。

在下一节，我们离开「指令」进入「系统」——**中断与系统调用机制**：用户程序如何把控制权交给操作系统，RISC-V 的 ecall 与 x86-64 的 syscall 如何工作。
