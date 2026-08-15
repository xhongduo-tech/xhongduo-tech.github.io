---
title: PTX 指令集与 Hopper 新指令
date: 2026-08-07
---

# PTX 指令集与 Hopper 新指令

<div class="epigraph">
<p>汇编是人与硬件之间最后的诚实对话。</p>
<footer>—— 汇编语言学习者的共识</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 硬件：NVIDIA H100/Hopper ｜ Hopper 论文 §5 ｜ 2026-08-07</p>
</div>

## 为什么从 PTX 讲起

前两节我们站在「CUDA C 程序员」的高度看线程与内存。但高级语言会被编译成什么？中间经过什么？**PTX** 是这一切的中转站：CUDA C 先被编译成 PTX（虚拟指令集），再由驱动编译成 SASS（硬件机器码）。理解 PTX，你才能看懂「为什么 Hopper 的新硬件能力（TMA、wgmma）必须配套新指令」，也才能在需要手写底层优化（写 inline PTX、看汇编）时有的放矢。<span class="marginnote">「PTX 是虚拟指令集」这一点很关键：它不绑定具体硬件，而是定义了一套稳定的指令抽象。同一份 PTX 可以运行在 Ampere、Hopper、甚至未来的 GPU 上——驱动会把它翻译成对应硬件的机器码。这让 CUDA 程序天然具备「一次编译、多代运行」的前向兼容性。</span>

## 1 编译链：CUDA C → PTX → SASS

一段 CUDA 代码的完整旅程是两层编译：

**第一层：CUDA C → PTX。** 由 `nvcc` 完成。CUDA C 的 kernel 函数被翻译成 PTX 指令（虚拟 ISA）。这一层负责：语法解析、类型检查、循环展开、部分优化。

**第二层：PTX → SASS。** 由 GPU 驱动中的编译器完成。PTX 被翻译成当前硬件（如 GH100）的机器码 SASS。这一层做寄存器分配、指令调度、针对具体 SM 微架构的优化。

为什么要分两层？三个好处：

**前向兼容**：应用发布 PTX，新 GPU 上市后无需重新编译即可运行（驱动自动翻译）。
**硬件独立**：开发者不用为每代 GPU 重写汇编。
- **可检查性**：PTX 是人类可读的中间层，便于性能分析与调试。

你可以用 `nvcc -ptx` 生成 PTX，用 `cuobjdump` 反汇编 SASS，对比两层看编译器做了什么。**看 SASS 是「看看硬件到底怎么跑的」，看 PTX 是「看看我的代码的逻辑形态」**——两者用途不同。<span class="marginnote">对比 CPU 世界：LLVM 的 IR（中间表示）角色与 PTX 类似——高级语言先编译成 IR，再针对不同 CPU 架构生成机器码。区别是 PTX 还承担了「运行时前向兼容」的职责，而 LLVM IR 主要在离线编译时使用。</span>

## 2 PTX 指令的基本形态

PTX 指令的语法接近传统汇编，核心形态是「目标 = 操作」：

```
操作码.类型  目标寄存器, 源寄存器, 源寄存器;
```

几个代表性指令：

| 指令 | 含义 | 示例 |
| --- | --- | --- |
| `ld.global` / `st.global` | 读写全局内存 | `ld.global.f32 %f1, [%rd1];` |
| `add.f32` / `mul.f32` | 浮点加减乘 | `add.f32 %f1, %f2, %f3;` |
| `fma.rn.f32` | 融合乘加 | `fma.rn.f32 %f1, %f2, %f3, %f4;` |
| `ldmatrix` | 从共享内存加载矩阵到寄存器 | Ampere 引入 |
| `cp.async` | 异步拷贝全局→共享 | Ampere 引入 |
| `wgmma` | warpgroup 级矩阵乘 | Hopper 引入 |
| `cp.async.bulk` | TMA 大块异步拷贝 | Hopper 引入 |
| `mbarrier.arrive` / `mbarrier.try_wait` | 异步屏障 | Hopper 引入 |
| `mapa` / `mapc` | 映射分布式共享内存 | Hopper 引入 |

注意指令后缀里的 `.f32`、`.global` 等修饰符：它们指定数据类型与内存空间。PTX 的类型修饰符（`.f32`、`.f16`、`.s32`、`.b32` 等）是「数据带类型」的关键——编译器与硬件据此选择正确的执行单元。<span class="marginnote">「类型修饰符决定执行单元」在 GPU 上尤其重要：`add.f32` 走 FP32 流水线，`add.s32` 走 INT32 流水线，`fma` 走融合单元，`wgmma` 走 Tensor Core。在 SASS 里看指令类型，就能推断你的代码用上了哪条计算流水线——这是性能分析的第一步。</span>

## 3 Hopper 的新指令：硬件能力如何暴露给软件

Hopper 论文 §5 专门讲了新架构的指令支持。核心洞察是：**新硬件能力必须通过新指令暴露**，而新指令往往以「组合拳」形式出现，共同实现某一类异步流水线。

**组合拳之一：异步数据搬运。** `cp.async.bulk` 系列指令承载 TMA：一条指令完成「全局 → 共享」「共享 → 共享（DSMEM）」的多维张量搬运，由单线程触发，硬件异步执行。相比 Ampere 的 `cp.async`（每线程 16 字节、需整 warp 协作），它是「一键搬一块」。

**组合拳之二：warpgroup 矩阵乘。** `wgmma` 指令承载第四代 Tensor Core：由 128 线程协作执行 $64 \times 8 \times 16$ 以上的大矩阵块，支持异步执行，是第四代 Tensor Core 性能的「软件接口」。它把「矩阵乘」从「一条同步的小指令」升级为「一条异步的大指令」——配合 mbarrier，整个 GEMM 可以流水化。

**组合拳之三：异步同步。** `mbarrier.arrive` / `mbarrier.try_wait` 提供细粒度的完成信号。搬运完成、矩阵乘提交，都用 mbarrier 通知消费者——这是「异步化」能否成立的同步基石。

**组合拳之四：分布式共享内存。** `mapa` / `mapc` 把簇内其他块的共享内存映射进本地地址空间，让跨块通信不再绕道全局内存——这是《线程块簇与分布式共享内存》一节的指令层实现。

四套组合拳合在一起，就是 Hopper 异步执行模型的完整指令图景：**TMA 搬（cp.async.bulk）→ mbarrier 等（arrive/try_wait）→ wgmma 算 → mapa/mapc 跨块协作**。

## 4 实战：怎么写与怎么读 PTX

用 `nvcc -ptx` 可以把 CUDA C 编译成 PTX 文件，用 `cuobjdump --dump-sass` 可以得到 SASS 汇编。对普通开发者，至少掌握两个技能：

**写 inline PTX**：当 CUDA C 表达不了某个硬件能力时，用 `asm()` 内嵌。例如 TMA 搬运在早期 CUDA 版本没有 C++ 封装，需要手写 `cp.async.bulk`。写法：

```cuda
asm volatile(
    "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
    "[%0], [%1], %2, [%3];"
    :: "r"(smem_addr), "l"(gmem_addr), "r"(bytes), "r"(bar_addr)
    : "memory");
```

**读 SASS 定位性能**：性能分析时，打开 SASS 看每个 kernel 的指令组成——有没有 `wgmma`（用上了 Tensor Core）、有没有 `cp.async.bulk`（用上了 TMA）、`fma` 比例高不高。**一条 SASS 指令的类型，就是「硬件执行路径」的指纹**。

**辨析｜易错点：** 不是所有 PTX 指令都能在所有 GPU 上跑。PTX 有**版本与架构要求**（`-arch` 指定），且某些新指令（如 `wgmma`）需要对应算力（sm_90a）。**「PTX 前向兼容」不等于「每代都能用最新指令」**——旧 GPU 上驱动会把新指令「翻译」成旧实现或报错。写 inline PTX 时，务必确认目标算力。

## 5 公式解析：一层编译如何换来前向兼容

把「两层编译」的收益量化。设 GPU 硬件每 $G$ 代一换，每代新增 $K$ 条指令，应用开发者有 $A$ 个应用。若只做一层编译（应用绑定机器码），每代要重编译：

$$
N_{\text{recompile}} = A \times \frac{T}{G}
$$

- $A$：应用数量（成千上万）。
- $T$：应用的生命周期（年）。
- $G$：硬件换代周期（约 1–2 年）。

两步拆解：

- **第一步，看重编译成本**：每代 GPU 换代，全部应用都要重写/重编译——对大生态（CUDA 有百万级开发者）是灾难。
- **第二步，看 PTX 的缓冲**：应用只编译到 PTX（虚拟 ISA），驱动运行时翻译到新硬件——**应用永远不用重编译，新增指令由新驱动暴露**。

这条式子解释了 PTX 的战略地位：**它是 CUDA 生态「一次编写、多代运行」承诺的技术载体**。Hopper 的新指令之所以能迅速被 PyTorch/cuBLAS 用上，正是因为它们在 PTX 层就绪，驱动一升级就全部生效。

## 6 小结

- 编译链是**两层**：CUDA C → PTX（虚拟 ISA）→ SASS（硬件机器码），前向兼容由驱动保证。
- PTX 指令形态「操作码.类型 目标, 源…」，**类型修饰符决定走哪条执行流水线**。
- Hopper 新指令是「组合拳」：**cp.async.bulk（搬）+ mbarrier（等）+ wgmma（算）+ mapa/mapc（跨块）**。
- 实战技能：**写 inline PTX** 表达硬件能力、**读 SASS** 判断执行路径（有没有 wgmma/cp.async.bulk）。
- **辨析**：PTX 前向兼容 ≠ 每代都能用最新指令；新指令有算力要求（如 wgmma 需 sm_90a）。
- 两层编译把「每代重编译」变成「驱动运行时翻译」，是 CUDA 生态「一次编写、多代运行」的技术基石。

在下一节，我们从「指令层」升到「执行层」——**CUDA Graphs 与程序化依赖**，看怎么把一串 kernel 启动组织成一张一次启动的图。