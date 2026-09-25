---
title: warp 级原语
date: 2026-09-18
section: cs
---

# warp 级原语

<div class="epigraph">
<p>warp 内部有一条不经过任何存储的邮路：shuffle 把一个寄存器直接送到另一个 lane，一拍即达——代价是邮路半径只有 32。</p>
<footer>—— 据 NVIDIA CUDA C++ Programming Guide 的 Warp Shuffle 与 Vote 函数、PTX ISA 文档整理</footer>
</div>

[上一课](/cs/gpu-memory-hierarchy-practice)把字节放到了对层，但层次解决的是「到哪取数」；warp 内部的交换还有一条更窄的路。缺口在归约这类模式上：32 个 lane 各持一个部分和，要互相传递——走 smem 就要算地址、过屏障、挨 bank 的账，而交换半径只有 32。本栏的 [SIMT](/cs/gpu-simt) 已经说了 lane 共享一个 warp 的执行；大模型栏的 [Warp specialization](/llm/warp-specialization) 讲的是 warp 之间的角色分工。本课在 warp 内部：shuffle 与 vote 两个家族，以及它们把哪些存储流量整个消掉。

## 问题

warp 归约的朴素写法把 smem 当公告板：每步一半 lane 写自己的部分和、另一半读邻居的，步与步之间插 `__syncthreads`。每一步交换都付三重税：地址计算与 bank 规划、屏障的等待、以及访存指令本身。可交换的范围其实只有一个 warp——lane 之间的寄存器本来就近在咫尺。缺口是：哪些通信能沉到 warp 内、用什么原语沉、沉下去之后省掉的是哪几重税。判断反了（该走 smem 的跨 warp 通信硬塞进 shuffle）会在编译期就报错；判断漏了（warp 内通信仍走 smem）则白付三重税。

## 方法

shuffle 家族按「谁拿到谁的值」分类。`__shfl_sync(mask, v, srcLane)` 直接指定源 lane；`__shfl_up_sync` / `__shfl_down_sync` 做扫描式的邻居交换；`__shfl_xor_sync(v, k)` 与 lane $i\oplus k$ 交换——butterfly 模式，第 $0,1,2,3,4$ 步取 $k=1,2,4,8,16$，五步之后 32 个 lane 全部持有总和，一条 warp 全归约就此完成，全程没有一条存储指令。<span class="marginnote">「xor 交换」翻译成大白话：每个 lane 找与自己编号二进制恰好差一位的伙伴互换数据——第 1 步差最后一位（0↔1、2↔3，相邻成对），第 2 步差倒数第二位（0↔2，跨 2）……像锦标赛：32 人五轮交换后，人人都知道了总分，且没碰过一次内存。</span>`width` 参数把交换限制在宽度为 $2^j$ 的子组内，分段归约不用再写循环。

vote 家族把判断折叠成位图。`__ballot_sync(mask, p)` 返回一个 32 位整数，第 $i$ 位是 lane $i$ 谓词的值；`__any_sync` / `__all_sync` 直接归约成布尔；`__match_any_sync` 找出同值的 lane 组。典型用法：先 ballot 拿到「谁需要走慢路径」，再决定是分裂还是统一走——把「两条路径各跑一遍」的发散成本（第一课的相加账）压成一次位测试加一条路径。<span class="marginnote">数字实例：lane 0、5、9 的谓词为真时，ballot 返回 $2^0+2^5+2^9=1+32+512=545$。一条 `v == 0` 判断「全假」，`v & (v-1)` 循环数「有几个真」——原本要靠 warp 发散两轮才能摸清的局面，变成对一个整数的算术。</span>

```mermaid
flowchart TD
  P["warp 内条件判断 p"] --> DIV["朴素写法：两条路径各跑一遍"]
  DIV --> TAIL["长尾 warp 拖全场"]
  P --> BAL["ballot：一条指令拿到 32 位位图"]
  BAL --> T{"位图读数：全一致吗？"}
  T -->|"全 0 或全 1"| UNI["整 warp 统一走一条路径"]
  T -->|"混合"| SPLIT["按位分组，或统一走慢路径"]
  UNI --> DONE["发散成本压成一次位测试"]
  SPLIT --> DONE
````mask` 参数是 Volta 独立线程调度在 API 上的落点：参与交换的 lane 必须以同一 mask 执行同一语句，mask 不满是未定义行为，不再是老卡上的「碰巧能跑」。

```mermaid
flowchart TD
  IN["32 个 lane 各持一个值"] --> X1["xor 1：与相邻 lane 交换，16 对"]
  X1 --> X2["xor 2：跨 2 交换，8 对"]
  X2 --> X4["xor 4 与 xor 8 与 xor 16"]
  X4 --> OUT["五步后 32 个 lane 全持总和"]
  ALT["朴素方案：smem 公告板"] --> TAX["地址计算 + 屏障 + bank 账"]
```

## 机制

快的原因在硬件位置：shuffle 走寄存器堆的 lane 间交换通路，不占访存单元的端口、不进 smem、不需要屏障——上一课说「寄存器零等待」，shuffle 是把零等待用在了通信上。步数是 $\log_2 32 = 5$，通信量逐半收缩：$16+8+4+2+1$ 共 31 次成对交换；smem 方案同样五步，但每步多一重屏障税，block 只有一个 warp 时屏障也省不掉（块级屏障要凑齐所有 warp）。vote 的价值同理：ballot 一个指令拿到全部 lane 的谓词，替代「先发散两轮、各自退出」的写法，长尾 warp 不再拖全场。

迁移到新卡时的注意点在 mask 语义：Ampere 之前 mask 不满常被容忍，之后是未定义——把「全 warp 在场」写成字面量 `0xffffffff` 而非 `__activemask()`，因为后者的值取决于执行时刻谁在场，拿它当参与集合是把自己交给调度顺序。<span class="marginnote">常见误区：把 shuffle 当免费且无条件的操作。它是全 warp 的同步语句，mask 里列的每个 lane 都必须执行同一句；写在只有部分 lane 进入的 if 里而不改 mask，结果不是「其他 lane 等着」，是整条语句未定义——新卡上换驱动就可能换答案。</span>

## 边界

半径 32 是硬边界：跨 warp 的归约必须回到 smem 加 `__syncthreads`，那里等着的是下一课的 bank 账。cooperative groups 提供了 `reduce` 之类的封装，展开后仍是这套原语，语义不新增。scan 的正确性证明（排他性与结合律）是算法层的合同，本课只给机制。发散路径里调 shuffle 要格外小心：不在场的 lane 不会参与，部分活跃 warp 的交换结果按 mask 定义——先想清楚参与集合，再写表达式。

<span class="marginnote">`__activemask()` 的易错点：它返回的是「执行到这一句时谁在场」，不是「谁应该参与」。用它当 shuffle 的 mask，同一份代码会随调度顺序给出不同结果——正确写法是显式传参与集合，或用 `__syncwarp` 先对齐。</span>

## 小结

- warp 内通信不必经过存储：shuffle 走寄存器堆的 lane 交换通路，不占访存端口、免屏障。
- butterfly 归约取 $k=1,2,4,8,16$ 五步，32 lane 全持总和；`width` 参数给出分段子组。
- vote 家族把谓词折叠成位图，ballot 加路径判断替代发散两轮。
- mask 是 Volta 独立线程调度的 API 落点：参与集合必须显式，不满是未定义行为。
- 半径 32 是硬边界；跨 warp 回 smem 与屏障，那边的账在下一课。
- 出处：NVIDIA CUDA C++ Programming Guide 的 `__shfl_*_sync` / `__ballot_sync` 等内建函数章；NVIDIA PTX ISA 文档。
