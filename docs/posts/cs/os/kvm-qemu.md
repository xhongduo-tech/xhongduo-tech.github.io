---
title: KVM 架构：内核模块与 QEMU 的协作
date: 2026-08-07
---

# KVM 架构：内核模块与 QEMU 的协作

<div class="epigraph">
<p>KVM 是内核里的「虚拟化引擎」，QEMU 是用户态的「模拟大师」——一个管 CPU 加速，一个管设备模拟，珠联璧合。</p>
<footer>—— 佚名，Linux 内核课堂</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ Linux KVM 与 QEMU 架构 ｜ 2026-08-07</p>
</div>

## 为什么从 KVM 架构开始

上一节讲了硬件辅助虚拟化，这一节看它在 Linux 上的落地——**KVM + QEMU 架构**。KVM（Kernel-based Virtual Machine）是**内核模块**（负责 CPU/内存虚拟化），QEMU 是**用户态进程**（负责设备模拟与管理）。两者分工协作，构成现代 Linux 虚拟化的主力方案（OpenStack、Kubernetes 底层）。<span class="marginnote">回顾《虚拟化基础》：KVM 是硬件辅助虚拟化（VT-x/SVM）。它的架构特色是「<strong>内核做核心、用户态做外围</strong>」——<strong>KVM 模块把「虚拟机就是进程」变成现实，QEMU 进程把「虚拟机像普通进程一样管理」</strong>。</span>

## 1 KVM 的核心思想：虚拟机就是进程

**KVM（Kernel-based Virtual Machine）**：Linux 内核的虚拟化模块（`kvm.ko` + `kvm_intel.ko`/`kvm_amd.ko`），利用 VT-x/SVM 提供 CPU 与内存虚拟化。

**KVM 的颠覆性设计**：**虚拟机 = 一个普通进程**。

- 每个虚拟 CPU（vCPU）= 一个**线程**（qemu 进程的线程）。
- 虚拟机 = 一个 **QEMU 进程**（有进程号、可调度、可 kill）。
- **内核已有的机制（调度、内存管理）直接复用**——虚拟机不特殊。

**为什么这是天才设计**：

- **调度**：vCPU 线程走 Linux 调度器——多虚拟机自动公平分 CPU。
- **内存**：虚拟机的内存 = 进程的地址空间——用页表/大页管理。
- **管理**：`virsh`/`virt-manager`/`virt-install` 直接管虚拟机——无需专门的虚拟化管理器。

**vCPU 的执行**：用户态进程执行 `KVM_RUN` ioctl → 内核进入**非根模式**跑 vCPU → 特权操作/中断触发 **VMEXIT** 回到内核处理 → 需要设备模拟时返回用户态交 QEMU。

## 2 QEMU：用户态的设备模拟

**QEMU**：用户态程序，负责：

- **设备模拟**：为客户机模拟磁盘、网卡、显卡、USB 等设备。
- **虚拟机生命周期**：创建/配置/启动/关闭虚拟机。
- **管理接口**：通过 ioctl 与 KVM 模块交互。

**QEMU 的两种模式**：

- **纯软件模拟**（TCG）：完全用 CPU 模拟——慢，用于无 KVM 时或跨架构。
- **硬件加速**（KVM）：CPU/内存交给 KVM 加速，QEMU 只管设备——快，现代默认。

**QEMU + KVM 的分工**：

| 组件 | 层次 | 职责 |
| --- | --- | --- |
| KVM 模块 | 内核 | CPU 虚拟化（vCPU 执行、VMEXIT）、内存虚拟化 |
| QEMU | 用户态 | 设备模拟、虚拟机管理、与 KVM 交互 |

**协作流程**：

1. QEMU 为每个 vCPU 创建一条用户态线程；
2. vCPU 线程执行 `KVM_RUN` ioctl 进入内核；
3. KVM 用 `vmlaunch`（VM Entry）切换到非根模式，直接运行客户机代码（不经内核）；
4. 客户机执行特权操作、访问设备或发生中断 → **VMEXIT** 回到 KVM；
5. KVM 处理内存/中断类事件，若是设备访问则返回用户态，由 QEMU 模拟设备；
6. 处理完毕，QEMU 再次 `KVM_RUN`，vCPU 继续跑。

**设备模拟的两种方式**：

- **纯软件模拟**：QEMU 完全模拟设备（兼容好、慢）。
- **virtio**：半虚拟化设备（客户配合、快）——见下篇。

## 3 公式解析：VMEXIT 的成本与频率

vCPU 执行中每次特权操作/中断触发 **VMEXIT**（从非根模式回到根模式）。设 VMEXIT 处理耗时 $T_{exit}$：

$$\text{vCPU 有效执行率} \approx \frac{T_{run}}{T_{run} + f_{exit} \cdot T_{exit}}$$

- **$T_{run}$**：vCPU 连续执行时间（非根模式）。
- **$f_{exit}$**：VMEXIT 频率（每秒触发次数）。
- **$T_{exit}$**：每次 VMEXIT 处理成本。

**直觉**：**VMEXIT 越频繁，虚拟化开销越大**——这是虚拟化性能的关键指标。现代优化（**KVM 的 PV 特性**：`kvm-clock` 虚拟时钟、`virtio` 半虚拟化设备）都是为了**减少 VMEXIT**（把「要进 hypervisor 的操作」变成「客户自己处理」）。**「减少进出」是虚拟化性能优化的主线。**

**数值算例：VMEXIT 有多贵**。一个普通虚拟机每秒可能触发几十万次 VMEXIT（时钟中断、特权指令、页表访问、设备访问都会触发）。若每次 VMEXIT 处理约 1–3 µs，每秒 30 万次就吃掉约 0.3–0.9 秒 CPU——**占满一个核的 30–90%**。因此每一项 KVM 优化都在「砍掉一种 VMEXIT 来源」：

- **kvm-clock**：半虚拟化时钟取代物理时钟中断，减少定时 VMEXIT；
- **virtio**：共享内存环 + 通知，避免每次 I/O 都 VMEXIT；
- **EPT（扩展页表）**：客户页表翻译不再触发 VMEXIT。

**同一主线贯穿到底：虚拟化性能 = 让客户机「少进进出出」**。

**辨析｜易错点：** 「QEMU 是 KVM 的组成部分」是混淆。**QEMU 是独立的用户态程序，KVM 是独立的内核模块**——QEMU 可以不用 KVM（纯模拟），KVM 也离不开 QEMU（内核不管设备模拟）。**「QEMU+KVM」是一个组合**（`qemu-kvm`），不是同一个东西。另一个易错点：**「KVM 是 Type-2 hypervisor」**——严格说 KVM 模块是内核的一部分（Type-1 风格），QEMU 是管理进程；整体被归为「Type-1.5」。**别用 Type-1/Type-2 的简单二分硬套 KVM。**

## 4 核心对比表：KVM vs QEMU 纯模拟

| 维度 | QEMU 纯软件模拟 | QEMU + KVM |
| --- | --- | --- |
| CPU 虚拟化 | TCG 二进制翻译 | **VT-x 硬件加速** |
| 性能 | 差（慢 10~100 倍） | **接近原生** |
| 设备模拟 | QEMU 软件 | QEMU 软件（+virtio） |
| 客户 OS | 可跨架构（ARM 模拟 x86） | 同架构为主 |
| 用途 | 教学、跨架构 | **生产虚拟化标准** |

**术语速查表**：

| 术语 | 含义 |
| --- | --- |
| KVM | 内核虚拟化模块（`kvm.ko` + 架构模块） |
| vCPU | 虚拟 CPU，一个用户态线程 |
| QEMU | 用户态设备模拟与管理进程 |
| `KVM_RUN` | 让 vCPU 进入非根模式执行的 ioctl |
| VMEXIT | 从非根模式回到根模式的事件 |
| 非根模式 | 客户机代码运行的受限 CPU 模式 |
| virtio | 半虚拟化设备（共享内存环） |
| TCG | QEMU 纯软件模拟引擎 |

## 5 小结

- **KVM**：内核虚拟化模块，核心思想是「**虚拟机 = 进程**」——复用内核调度/内存机制。
- **QEMU**：用户态设备模拟 + 虚拟机管理，通过 ioctl 与 KVM 交互。
- vCPU 执行：`KVM_RUN` → 非根模式 → **VMEXIT** → 内核/QEMU 处理。
- **VMEXIT 频率决定虚拟化开销**——减少进出是性能优化主线。
- KVM + QEMU 是现代 Linux 虚拟化标准，OpenStack/K8s 底层。

在下一节，我们看虚拟化的内存与 I/O——**内存与 I/O 虚拟化：EPT、影子页表、virtio 与设备直通**。
