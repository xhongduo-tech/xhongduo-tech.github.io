---
title: Occupancy（占用率）的计算与调优
date: 2026-08-07
---

# Occupancy（占用率）的计算与调优

<div class="epigraph">
<p>过早优化是一切罪恶之源。</p>
<footer>—— 唐纳德 · 克努特（Donald Knuth）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 基础设施 ｜ AI基础设施技术栈 第一篇 ｜ 2026-08-07</p>
</div>

## 为什么从 Occupancy 开始

前三课我们回答了「数据怎么读得快」：内存层次是地图，合并访存是规则，Shared Memory 是可精确控制的仓库。这一课回答最后一个问题：**有多少活可以同时干？** GPU 靠并发隐藏延迟——一个 warp 在等 600 周期的访存时，调度器换另一个 warp 上。这个「并发储备」的定量名字叫 **Occupancy（占用率）**。它是连接前三课的枢纽：你每多用一点寄存器、每多用一块 Shared Memory、每把 block 设得大一点，Occupancy 都可能掉一截。<span class="marginnote">克努特那句「过早优化」在这里是清醒剂：<strong>Occupancy 是手段，不是目标</strong>。调优的第一步永远是先测量，确认瓶颈在「并发不够」还是「数据没搬好」——否则就会对着 100% 占用率空欢喜。</span>

## 1 用并发掩盖延迟：occupancy 是什么

回顾执行模型：每个 SM 最多驻留 64 个 warp（2048 线程），4 个调度器每周期挑就绪的 warp 发射。当一个 warp 发起访存后要等几百周期，这段时间里调度器靠**其他 warp** 继续发射指令，延迟就被「藏」起来了。

**Occupancy** 的定义：当前**活跃 warp 数 / SM 最大 warp 数**。

$$\text{Occupancy} = \frac{\text{活跃 warp}}{\text{最大 warp}（每 SM 64）}$$

它衡量的是**隐藏延迟的并发储备**：Occupancy 越高，SM 手里可换的 warp 越多，越不容易出现「所有 warp 都在等内存、执行单元空转」的局面。但它只是**能力（capability）**，不是吞吐本身——**Occupancy 100% 的 kernel 不一定比 50% 的快**，这正是本课要反复强调的。<span class="marginnote">把 Occupancy 类比成餐厅的「翻台储备」：桌位多，客人来了不排队，但<strong>如果每桌的菜本身做得很差，桌位再多也快不起来</strong>。你的「菜」就是访存效率与指令级并行。</span>

## 2 四个硬资源：occupancy 由谁决定

Occupancy 不是自由的——每个 SM 有**四个硬上限**，任何一个卡住，能驻留的 block 就少：

| 资源 | A100 / H100 每 SM 上限 | kernel 里的对应项 |
| --- | --- | --- |
| 线程数 | 2048 | block 大小 × block 数 |
| 块数 | 32 | block 数 |
| 寄存器文件 | 65536 个 32 位寄存器 | 每线程寄存器数 × 线程数 |
| Shared Memory | 164 / 228 KB | 每 block Shared 用量 |

kernel 每线程多要 8 个寄存器、每 block 多用 16KB Shared，都可能让「能驻留的 block 数」降下来，进而压低 Occupancy。寄存器还有个**分配粒度**的细节：现代架构按 warp 分配寄存器，粒度是 **256 个寄存器 / warp（即每线程 8 个）**——编译器报了 33 个，实际占用按 40 算。这意味着你的「精打细算」经常要按 8 的倍数取整。

## 3 调节旋钮：把 occupancy 拧到想要的位置

四个主要的调优旋钮：

**旋钮一：block 大小。** 线程数必须是 32 的倍数（否则产生不满的尾部 warp）。128 / 256 是常见选择。block 太小（如 32）会让「每 SM 最大块数 32」先卡住；太大（如 1024）则一个 SM 只能放下 2 个 block，调度灵活性下降。

**旋钮二：`__launch_bounds__`。** 告诉编译器「最多多少线程 / 最少多少 block 每 SM」，编译器据此压缩每线程寄存器预算：

```cpp
__global__ __launch_bounds__(256, 4) void kernel(const float* x,
                                                 float* y, int n) {
    // 编译器按「每 SM 至少 4 个 block」压缩每线程寄存器预算
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 2.0f;
}
```

**旋钮三：`-maxrregcount`。** 编译期全局限制每线程寄存器数。与 `__launch_bounds__` 二选一用，别混用打架。

**旋钮四：Shared Memory 用量。** 每 block 用多少 Shared 直接进「块数」公式（见下节）。很多 kernel 在这里被卡住——不是寄存器，而是 Shared。

还有两个**实测**工具，别靠拍脑袋：

```cpp
// CUDA Occupancy API：实测指定 launch 配置能达到的 block 数
int numBlocks = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel,
                                              256 /*blockDim*/, 0);
```

以及 NVIDIA 的 Occupancy Calculator 表格、Nsight Compute 里的 occupancy 面板——它们把「四个上限谁先卡住」算得清清楚楚，是调优的第一现场。<span class="marginnote">Nsight Compute 的 occupancy 页面会直接告诉你当前 kernel 的<strong>限制因素是 registers 还是 shared 还是 threads</strong>——这比在公式里手动代入快得多。第十篇《监控与性能剖析》会完整带你走一遍这个工作流。</span>

## 4 公式解析：把四个上限叠进一个公式

每 SM 能驻留的 block 数，是四个上限同时作用的结果，取最小值：

$$
\text{blocks/SM} = \min\left(
\left\lfloor \frac{T_{\max}}{B_t} \right\rfloor,\quad
\left\lfloor \frac{S_{\max}}{S_b} \right\rfloor,\quad
\left\lfloor \frac{R_{\max}}{R_t \times B_t} \right\rfloor,\quad
B_{\max}
\right)
$$

然后：

$$
\text{Occupancy} = \frac{\text{blocks/SM} \times B_t}{T_{\max}}
$$

对这条式子做三步拆解：

- **第一步，认符号**：$T_{\max}=2048$（每 SM 最大线程），$B_t$（每 block 线程数），$S_{\max}$（每 SM 最大 Shared 字节），$S_b$（每 block Shared 字节），$R_{\max}=65536$（寄存器文件），$R_t$（每线程寄存器数），$B_{\max}=32$（每 SM 最大 block 数）。
- **第二步，看第一、三项**：第一项是「按线程数最多能装几个 block」，第三项是「按寄存器总量最多能装几个 block」——寄存器项分母是 $R_t \times B_t$（一个 block 总共吃掉多少寄存器）。
- **第三步，代入实例**：设 $B_t = 256$、$R_t = 40$、$S_b = 32$KB，A100 上 $S_{\max} = 164$KB：

$$
\min(\lfloor 2048/256 \rfloor,\ \lfloor 164/32 \rfloor,\ \lfloor 65536/(40\times 256) \rfloor,\ 32)
= \min(8,\ 5,\ 6,\ 32) = 5
$$

Occupancy $= 5 \times 256 / 2048 = 62.5\%$。**卡住它的是 Shared Memory（第二项 5）**。把 $S_b$ 降到 24KB，第二项变 $\lfloor 164/24 \rfloor = 6$，min 变 6，Occupancy 升到 75%；再把 $R_t$ 压到 32（用 `-maxrregcount=32`），第三项变 $65536/8192 = 8$，min 变 6（仍被 Shared 卡），Occupancy 75%——**可见，不找到「最短的板」，乱调是白调**。

## 5 辨析｜易错点

- **「Occupancy 越高越快」**——不一定。低 Occupancy + 高指令级并行（ILP）或更低缓存抖动，常常更快。**目标是吞吐，不是占用率**。内存瓶颈的 kernel 尤其如此：若访存已有足够的内存级并行（MLP），多几组 warp 反而挤占 L2/带宽。
- **「把寄存器数压到最低就对了」**——可能引入**寄存器溢出到 local memory**，让局部变量访问从 1 周期掉到几百周期。压寄存器前先看 `nvcc -Xptxas -v` 的 spill 计数。
- **「block 大小取 1024 最划算」**——一个 SM 只能放下 2 个 1024 线程的 block，调度灵活性差；256 / 512 往往更优。小 grid 时还要警惕**尾效应**：不满的最后一个 block 浪费尾部线程。
- **「__syncthreads 是免费的」**——每次屏障都有停顿 + 同步开销，过度使用（例如每层归约都同步）会压低有效吞吐；在资源受限时它还挤占本可用于并行的空间。
- **「Shared Memory 不影响 Occupancy」**——错。$S_b$ 直接进 blocks/SM 公式，是四大资源之一。前面课里「Shared 越大越好」的错误印象，在这里被正式修正。

## 6 小结

- **Occupancy** = 活跃 warp / 最大 warp（每 SM 64），是隐藏延迟的**并发储备**，是能力而非目标。
- 四个硬上限：**线程数（2048）、块数（32）、寄存器文件（65536）、Shared Memory（164/228KB）**；blocks/SM 取四者最小值。
- 调优旋钮：**block 大小**、**`__launch_bounds__`**、**`-maxrregcount`**、**Shared 用量**；寄存器按每线程 8 个的粒度取整分配。
- 用 CUDA Occupancy API 与 Nsight Compute 实测，先找「最短的板」再调，别乱拧。
- **高 Occupancy 不等于快**：内存瓶颈下堆 warp 可能反而挤占带宽；压寄存器要防溢出到 local memory。

在下一节，我们将从「一个 SM 里同时有多少活」升级到「整张卡上多个任务如何交替」——**CUDA Stream 与异步执行、事件计时**：当一个 kernel 在等内存时，另一个 kernel 可以在别的 stream 上算起来。这其实是用**时间上的重叠**继续隐藏延迟，与 Occupancy 用**空间上的并发**隐藏延迟殊途同归。
