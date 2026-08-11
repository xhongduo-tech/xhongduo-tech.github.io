---
title: 固件与内核逆向
date: 2026-08-11
---

# 固件与内核逆向

<div class="epigraph">
<p>你无法信任一份不是你亲手完全创造的代码。</p>
<footer>—— 肯 · 汤普森（Ken Thompson，《论对信任的信任》，1984）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 逆向工程与二进制分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从内核与固件开始

到目前为止，逆向的对象都是「应用程序」——运行在操作系统之上的用户态代码。但真正的深水区在下面两层：**内核（kernel）**——操作系统自己，以及**固件（firmware）**——连操作系统之前的那层软件。<span class="marginnote">汤普森的《论对信任的信任》是整个计算机安全史的基石：他证明了一个编译器可以植入后门，而这个后门会随着编译器自我复制传播到所有用它编译的程序——「你信任的每一层，都可能欺骗你」。逆向工程往下走到内核与固件，正是在验证这个论断：最底层恰恰是最需要被审视的。</span>

为什么分析者要下到内核？两个动机：**恶意代码下沉**（rootkit、驱动后门利用内核特权做用户态做不到的事），以及**漏洞挖掘**（内核漏洞的威力远高于用户态漏洞——提权、沙箱逃逸都发生在内核边界）。

## 1 内核视角：一个不同的世界

用户态逆向的很多假设在内核里失效：

**权限与上下文**：内核代码运行在 CPU 特权态（ring 0），任何指针错误都是系统级崩溃（蓝屏 / panic）——**分析内核样本的调试事故代价是「整机重启」**；
- **无进程边界**：内核的代码运行在所有进程的上下文里，没有「我是哪个进程」的概念，也不能像用户态那样用进程隔离来保护自己；
- **调度与中断**：内核可以阻塞、睡眠，但更多时候运行在中断上下文（不可睡眠）、DPC/IRQL 等级上——很多用户态逻辑（文件 I/O）在中断上下文里根本不能做。<span class="marginnote">Windows 的 IRQL（中断请求级别）是内核逆向的一把尺：`<= DISPATCH_LEVEL` 才能拿自旋锁、`<= APC_LEVEL` 才能碰分页内存。逆向驱动时看到对 `KeAcquireSpinLock` 的调用，就能推断它运行在什么 IRQL 下、能做什么操作。</span>

内核逆向的工具也与用户态不同：Windows 用 **WinDbg** 做内核调试（双机调试：被调试机 + 调试机）、内核符号 `ntoskrnl.exe` 的 `kd>` 命令；Linux 用 **kgdb / ftrace / kprobes**。

## 2 系统调用：内核的用户态接口

内核逆向最常见的入口是**系统调用（syscall）**——用户态程序请求内核服务的唯一通道。Windows 上用户态调用 `NtCreateFile` 之类的 **NT API**，它们转成系统调用序号，经 `KiSystemService` 分发到内核的 `NtCreateFile` 实现；Linux 上通过 `int 0x80`/`syscall` 指令直接进内核。

逆向系统调用的意义在于：**系统调用表是内核的「功能地图」**。逆向一个 rootkit 时，你先看它 hook 了哪些系统调用，就立刻知道它想隐藏什么——文件（`NtQueryDirectoryFile`）、进程（`NtQuerySystemInformation`）、注册表键，一一对应。<span class="marginnote">在《API Hook 与钩子注入》里我们见识过 SSDT hook：rootkit 替换系统调用表项后，`NtQuerySystemInformation` 被换成了隐藏进程的过滤器。内核逆向与 hook 技术在这一点上完全汇合——hook 是手段，读系统调用表是理解手段。</span>

### 2.1 系统调用序号与内核 API 命名

理解系统调用要从两个「数字与名字」的细节入手。**第一，系统调用序号**：Windows 的 `NtCreateFile` 在 32 位系统上对应序号 `0x25`、64 位上是 `0x55`——序号因架构而异，恶意代码做 syscall 直调（inline syscall）时硬编码的正是序号；Linux 的 x86-64 系统调用表序号同理（`openat=257`、`execve=59`）。**内核 API 命名**则是另一把尺：`Zw*` 前缀（`ZwCreateFile`）是内核里「规范入口」，`Nt*` 是它的实现；`Mm*` 管内存（`MmProbeAndLockPages`）、`Ke*` 管调度与同步（`KeAcquireSpinLock`）、`Ex*` 管资源分配（`ExAllocatePool`）、`Ob*` 管对象管理（`ObReferenceObject`）。<span class="marginnote">这串前缀就是内核逆向的「识字表」：看到一个 `MmMapLockedPagesSpecifyCache` 就知道驱动在把物理内存映射进用户态——结合《API Hook 与钩子注入》的知识，这正是进程注入类恶意驱动的地基。前缀 + 功能语义，构成内核代码分析的第一层解译。</span>

## 3 驱动逆向：Windows 与 Linux 的解剖

内核代码大多以**驱动（driver）**形式存在。Windows 驱动用 `DriverEntry` 作为入口，注册 `IRP` 派遣函数（`IRP_MJ_*` 一类）；Linux 驱动用 `module_init` 入口与 `file_operations` 结构体注册回调。<span class="marginnote">逆向驱动先找「回调表」：Windows 的 IRP MajorFunction 表、Linux 的 file_operations/ioctl——它们列出驱动能处理的每一种请求。读懂回调表，就等于拿到了驱动功能的目录。</span>

**驱动逆向的完整链条**：

1. 从入口函数开始，找到它注册的**设备对象与回调**；
2. 逐个读回调：`IOCTL` 处理函数最值得读——驱动与用户态通信的「命令通道」就在 `DeviceIoControl` 的 code 分发逻辑里；
3. 关注驱动与用户态交换的缓冲区与数据结构（`METHOD_BUFFERED` / `METHOD_IN_DIRECT` 决定数据怎么传）；
4. 对照内核 API（`Zw*` / `Ex*` / `Mm*` 前缀的函数）理解每个操作在做什么——这一步需要系统性的内核 API 知识，也是 Dang 书里花了大篇幅铺垫「内核 API 速览」的原因。

恶意驱动的常见把戏也在这套流程里现形：隐藏进程/文件（hook 系统调用）、**直接内核对象操纵（DKOM）**（把进程从链表中摘除）、窃取令牌（`DuplicateToken` 提权）。

## 4 固件逆向：从镜像到代码

固件（firmware）是「最底层」——BIOS/UEFI、路由器、嵌入式设备的启动代码。逆向固件比内核多一道关卡：**拿到的是裸镜像，连「这是什么格式」都要自己猜**。标准流程：

1. **识别镜像**：`binwalk` 扫描固件镜像里的签名，识别出压缩内核、文件系统、引导加载器各在哪个偏移——现代路由器固件常是「U-Boot + 压缩内核 + squashfs 文件系统」的组合；
2. **提取文件系统**：`binwalk -e` 或手动按偏移切割，把 squashfs/cpio 解开，得到其中的二进制程序（用户态）与内核模块；
3. **逆向固件里的代码**：用 IDA 的嵌入式处理器支持（ARM、MIPS、PowerPC 都有）加载提取出的程序——嵌入式逆向的难点从「读懂」转移到「读对架构」，ARM/Thumb 切换、MIPS 延迟槽都是经典陷阱；<span class="marginnote">MIPS 的「延迟槽」就是一条控制流谜题：跳转指令的下一条指令无论如何都会先执行——反汇编时若不知晓这点，会把这条指令误判到错误的分支里。架构差异是固件逆向比内核逆向更「地广人稀」的原因之一。</span>
4. **分析与验证**：用 QEMU 的 `user` 模式直接运行提取出的用户态程序（不需要完整硬件）做动态验证——固件逆向的「沙箱」由此而来。

**固件漏洞挖掘**则复用前面全部方法：提取二进制 → 在 QEMU 里跑 → 模糊测试 → 崩溃回溯，一条流水线从「一个路由器固件」直达「一个 RCE 漏洞」——这正是《漏洞挖掘与模糊测试》的嵌入式版本。

### 4.1 UEFI 固件：现代启动链的逆向

现代 x86 机器的「第一行代码」已经不在 BIOS 里，而在 **UEFI 固件**中。UEFI 是一个小型的运行时环境，用自己的可执行格式（`PE32+` 变体）、自己的 ABI（EFI ABI，`EFI_MAIN`/`EFI_PHYSICAL_ADDRESS` 是它的坐标），以及一套「引导服务」（`Boot Services`）与「运行时服务」（`Runtime Services`）的接口。<span class="marginnote">UEFI 逆向的入口与 Windows 驱动惊人地相似：找入口点（`_ModuleEntryPoint`）、读它的 EFI 协议表、找 `StartImage` 与 `LoadImage` 的调用链。而安全研究对它格外上心——bootkit 能抢在操作系统之前运行，就是利用 UEFI 代码里的漏洞（如被广泛研究的「LogoFAIL」「Spectre 的固件应用」类问题）。</span>

UEFI 固件逆向的实用路径：用 UEFITool 把固件镜像解开成一个个 `DXE driver`（PE 文件）→ 用 IDA 按架构加载 → 在 OVMF（开源 UEFI 固件）里做 QEMU 模拟验证。这条「解镜像 + 反汇编 + 模拟验证」的链条，与路由器固件逆向（`binwalk` + QEMU）是同一套方法论在两种架构上的复现——**固件逆向的核心方法论始终是：解开、反汇编、模拟跑起来**。

## 5 小结

- 内核逆向的前提认知：**ring 0、无进程边界、中断上下文限制**；调试事故代价是整机崩溃。
- **系统调用表是内核的功能地图**：先看 rootkit hook 了哪些 syscall，就知道它想隐藏什么。
- 驱动逆向的链条：**入口 → 设备对象与回调 → IOCTL 分发 → 内核 API 对照**；DKOM、令牌窃取是恶意驱动经典把戏。
- 固件逆向四步：**识别镜像 → 提取文件系统 → 按架构反汇编 → QEMU 动态验证**；嵌入式架构（ARM/MIPS）陷阱比 x86 更多。
- 工具分工：Windows 用 **WinDbg** 内核调试，Linux 用 **kgdb/ftrace/kprobes**，固件用 **binwalk + QEMU**。

在下一节，我们把整套方法论收尾在最前沿的工具上——不读指令、直接让程序替我们回答「哪条路径可达」，这就是**二进制插桩与符号执行**。
