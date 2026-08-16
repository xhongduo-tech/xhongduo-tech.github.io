---
title: UEFI 体系（驱动模型、Protocol、UEFI Shell 与变量服务）
date: 2026-08-07
---

# UEFI 体系（驱动模型、Protocol、UEFI Shell 与变量服务）

<div class="epigraph">
<p>标准的好处在于，可供选择的标准总是那么多。</p>
<footer>—— 安德鲁 · 塔能鲍姆（Andrew S. Tanenbaum）</footer>
</div>

<div class="article-byline">
<p>第三级 · 固件与启动链（BIOS/UEFI/嵌入式引导） ｜ UEFI Forum《UEFI Specification》2.10 第2/3/7章 ｜ 2026-08-07</p>
</div>

## 为什么从 UEFI 开始

前两节走完了 x86 的物理启动链，但那还是「为了跑起来而存在」的底层组织。从本节起进入 **UEFI（Unified Extensible Firmware Interface，统一可扩展固件接口）** 的逻辑世界：它是一套被规范化的接口，让固件不再是一坨约定俗成的汇编代码，而是一个有驱动模型、有服务表、有 shell 的微型软件系统。<span class="marginnote">UEFI 规范由 UEFI Forum 维护，当前主版本 2.x，公开可读。它的 C 语言数据类型定义几乎全部映射到 EDK II 的 `MdePkg` 头文件里，学规范时对照 `uefi.h` 看最省力。</span>

对《Embedded Firmware Solutions》的读者而言，这一节对应书里对 EFI 的介绍与 EDK II 构建体系（第6、7章）。理解 UEFI 的钥匙是一句话：**它不是「BIOS 的别名」，而是「一个运行在 OS 之前的微系统」**——有自己的协议总线、服务表与命令行，OS 只是它的一个「客人」。

## 1 UEFI 的来路：EFI 1.10 与 Intel 的平台规划

UEFI 的前身是 **EFI（Extensible Firmware Interface）**，1990 年代末由 Intel 为 Itanium 平台提出。Itanium 想摆脱 x86 BIOS 的历史包袱，于是从零设计一套干净的引导接口，1999 年发布 EFI 1.02，2002 年发布 EFI 1.10。2005 年 Intel 把这套工作移交给新成立的 UEFI Forum，规范改名 UEFI 2.0，并在 2011 年后随 Windows 8 的 Secure Boot 要求全面铺开。

历史包袱正是 UEFI 要解决的：传统 BIOS 用**实模式 + 中断表 + 约定地址**工作，引导扇区只有 512 字节，代码必须塞进 1 MB 地址空间，且没有统一的驱动模型。<span class="marginnote">为什么 2011 年 Windows 8 成为转折点？微软要求获得 Windows 认证的 PC 默认开启 Secure Boot，而 Secure Boot 的签名验证基础设施（PK/KEK/db）只有在 UEFI 规范里才有定义。于是 OEM 一夜之间从 BIOS 迁到 UEFI。</span>

UEFI 时代还有一个细节值得记住：**固件里的可执行程序叫 `.efi`，遵循 PE/COFF 镜像格式**。这意味着 UEFI 驱动、Shell 工具、OS Loader 在格式上是同一种「PE 程序」，只是加载环境不同——这让「用同一个工具链开发固件与驱动」成为可能，也直接让 OS 厂商能写出统一的加载器（如 Windows 的 `bootmgfw.efi`）。

## 2 系统表：UEFI 世界的入口

UEFI 固件的运行时入口是一张 **EFI System Table**。所有 UEFI 代码（驱动、shell、OS loader）都通过它的两个指针获得能力：

- **Boot Services（启动服务）**：内存管理、事件、协议句柄、镜像加载等，仅在 OS 调用 `ExitBootServices()` 之前可用；
- **Runtime Services（运行时服务）**：变量、实时钟、重置、`UpdateCapsule` 等，`ExitBootServices()` 之后依然驻留。

EFI 的**句柄（Handle）** 与 **Protocol** 构成它的核心抽象：一个句柄代表「系统里的某个东西」（一块磁盘、一块网卡、一段内存），而 Protocol 是「这个东西上挂着的接口」。要操作设备，就「打开它句柄上的某个 Protocol」。

这张系统表还有一个容易被忽略的成员——**EFI Configuration Table**：它是一组「名字 + 全局指针」，让固件把额外的全局信息（如 ACPI 表指针、SMBIOS 指针）交给 OS。OS 的 `EFI_SYSTEM_TABLE` 里经常塞着 FADT、RSDT 等表的地址，这正是《外设枚举》一节里 ACPI 交接的入口之一。可以说，**系统表 + 配置表 = 固件给 OS 的「总目录」**。

## 3 公式解析：协议定位——从 GUID 到接口

UEFI 没有 C++ 的对象与虚表，它用 **GUID + 结构体指针** 模拟面向对象。一个 Protocol 的「类型」就是一个 GUID：

$$
\text{Protocol} = \big( \text{GUID}, \; \text{函数指针集合} \big)
$$

拆开看三层：

- **第一步，GUID 是什么**：一个 128 位全局唯一标识符。例如 PCI 总线协议是 `gEfiPciIoProtocolGuid = 4cf5b200-68b8-4ca5-9eec-b23e3f50029a`。GUID 保证「不同厂商不会撞名」。
- **第二步，怎么用**：代码调用 `LocateHandleBuffer` 或 `OpenProtocol`，传入想要的那个 GUID，Boot Services 在句柄数据库里搜索匹配项，返回接口指针。接口本质是一张函数指针表（struct）。
- **第三步，为什么这么设计**：它让「发现能力」变成运行时查询。传统 BIOS 里「这块网卡有没有 PXE」是编译期写死的中断调用；UEFI 里则是运行期遍历句柄、比对 GUID——这就是**协议总线**。

<span class="marginnote">类比操作系统课程里的虚拟文件系统：VFS 用 `struct file_operations` 挂接不同文件系统，UEFI 用「GUID + 函数指针表」挂接不同设备驱动。理解了 VFS，就理解了 UEFI 的驱动模型。</span>

## 4 启动服务、变量服务与驱动模型

**启动服务**是 Boot Services 时期的核心工具箱：内存分配（`AllocatePages`）、事件与定时器、`StartImage`/`Exit`、协议管理。OS Loader 依赖它们完成从固件到内核的过渡。

**变量服务（Variable Services）** 是 UEFI 最被低估的持久化设施。它在 Flash 里的实现同样讲究：EDK II 用**「备份 + 事务」**方式管理变量区——写入时先写副本、再原子切换，防止「写一半掉电」把变量区写坏；大变量（如安全启动证书库）与普通变量分块存放，避免相互踩踏。变量形如「名字 + GUID + 值 + 属性」，存于 SPI Flash 的非易失区，可通过 `GetVariable`/`SetVariable` 读写。<span class="marginnote">`BootOrder`、`Boot0000`…`Boot0008` 这些变量决定启动顺序；`SecureBoot`、`PK`、`KEK`、`db`、`dbx` 这些变量支撑安全启动（见本专题《安全启动》）。变量是固件与 OS 之间「跨重启的备忘录」。</span>

**UEFI 驱动模型**则把设备驱动分成「驱动（Driver）」与「I/O 句柄」两层：驱动通过 `DriverBinding` 协议绑定控制器句柄，必要时实例化子句柄。这种「绑定—解绑」机制让热插拔（USB、PCIe）在固件层面就有了表达。

Boot Services 里还有一套**事件（Event）**模型：固件可以创建定时事件、I/O 事件，并在事件触发时回调注册的函数。这套机制让固件能实现异步逻辑——等待某设备就绪、轮询某端口、超时处理——而不必死循环占住 CPU。对从 OS 编程转过来的人，这像是「固件里的 select/epoll」：**事件循环是 UEFI 的非阻塞编程基础**。

## 5 UEFI Shell：固件的命令行

**UEFI Shell** 是运行在固件之上的命令行环境，相当于「OS 之前的 Linux shell」。它提供 `map`、`ls`、`reset`、`bcfg`（配置启动项）、`ver` 等命令，也能执行 `.efi` 程序。固件工程师最常用的场景是：<span class="marginnote">EDK II 里写一个 20 行的 `.efi` 工具，丢进 FAT 格式的启动 U 盘，就能在 shell 里直接跑——这比反复烧 Flash 快一个数量级。这也是 UEFI 对比 BIOS 的开发体验优势。</span>

- 在 shell 下用 `bcfg boot add` 手工加启动项；
- 用内存读写命令 dump 某地址的寄存器；
- 加载厂商提供的诊断 `.efi` 工具，不动 Flash 就完成现场排障。

Shell 还支持**脚本**：根目录的 `startup.nsh` 会在 Shell 启动时自动执行，可以写循环、条件与命令序列——现场排障时可以放一个「一键 dump 关键寄存器」的脚本进去。这套「固件里的 shell」是 UEFI 对传统 BIOS「无法交互」问题最直接的改善。

## 6 核心对比表：UEFI 与传统 BIOS

纯概念主题，本节用核心对比表代替公式解析。**把握 UEFI 的关键是把它和 BIOS 放在一起对照**：

| 维度 | 传统 BIOS | UEFI |
| --- | --- | --- |
| 运行模式 | 16 位实模式 | 32/64 位保护模式、长模式 |
| 代码规模 | 引导扇区 512 B + ROM 程序 | 完整 C 驱动，Flash 通常 8–32 MB |
| 接口方式 | INT 10h/13h 中断 | Boot/Runtime Services、Protocol |
| 设备发现 | 写死的中断调用 | 句柄 + Protocol 运行时枚举 |
| 持久化配置 | CMOS + RTC | 非易失变量 |
| 图形/鼠标 | 文本屏为主 | GOP 图形输出协议 |
| 扩展性 | 厂商私有 | 规范定义 + 公开 GUID |

如果只记一句话，记住这个：**UEFI 是一种「接口的哲学」——把一切能力都变成「通过服务/协议查询获得」**。内存要问、设备要查、启动项要翻变量表，没有任何「魔法地址」。这种显式化让固件可组合、可审计、可移植，也正因如此，它才配得上承载 Secure Boot、capsule 更新这类需要精确语义的安全机制。

**易错点｜辨析：** 「UEFI 模式」「Legacy 模式」「CSM」三词常被混用。UEFI 模式指走完整的 UEFI 引导链；Legacy 模式指走传统 BIOS 路径；CSM 是 UEFI 固件里「向后兼容」的模拟层。多数主板的启动菜单里三者并存，但同一块磁盘通常只能用其中一种方式引导——这一点在下一节《传统 BIOS 与 legacy 兼容》展开。

## 7 EDK II：UEFI 的开源实现

谈 UEFI 就绕不开它的参考实现 **EDK II（EFI Development Kit II）**——由 TianoCore 项目维护的开源代码，是《Embedded Firmware Solutions》第7章实操的地基。理解 EDK II 的构建体系，就能读懂「UEFI 的代码是怎么组织出来的」：

- **包（Package）**：按职能划分的模块集合，核心是 `MdePkg`（UEFI/PI 规范的 C 定义）与 `MdeModulePkg`（标准实现：DXE 核心、变量驱动、串口驱动等）；
- **模块（Module）**：一个可执行的驱动或库，用 `.inf` 文件描述自身依赖与来源；
- **平台配置数据库（PCD）**：平台级可调参数，如 `PcdMaxVariableSize`、`PcdDebugPrintErrorLevel`——UEFI 世界的「编译期配置」；
- **构建描述**：`.dsc`（平台描述）、`.fdf`（Flash 布局）驱动 `build` 命令，把数百个模块链接成最终的固件卷。

<span class="marginnote">一个常被误读的点：<strong>UEFI 规范与 EDK II 不是一回事</strong>。规范是纸面上的协议，EDK II 是参考实现；厂商可以完全不碰 EDK II 而另写一套合规实现（事实上确有独立实现）。学 UEFI 时「规范 + EDK II 源码对照看」，是最高效的路径。</span>

这套体系让「固件」从「一个 ROM 程序」变成「一批可裁剪、可配置、可移植的模块」——这就是为什么同一套 EDK II 能支撑 PC、服务器、ARM 设备三种平台。

工程上的意义更直接：**调固件不再是「改一个大程序」而是「改一个模块再重建」**。开发者改某个驱动后，`build` 只重编受影响的模块，再经 `GenFv` 生成新的固件卷，配合 `flash` 工具刷进去。这种「模块级迭代」把固件开发从「黑盒整体替换」推进到「可增量迭代」——对比传统 BIOS 的「改一行重刷整个 ROM」，是开发效率的质变。

## 8 小结

- UEFI 是一个**运行在 OS 之前的微系统**，前身是 Intel 为 Itanium 设计的 EFI，2011 年后随 Secure Boot 全面普及。
- 它的入口是 **EFI System Table**，内含 Boot Services 与 Runtime Services 两张能力表。
- 设备抽象用 **句柄 + Protocol**：类型是 128 位 GUID，接口是函数指针表，靠运行时查询发现。
- **变量服务**把配置持久化到 Flash，支撑启动顺序与安全启动；**UEFI Shell** 提供固件级命令行。
- 与 BIOS 相比，UEFI 在运行模式、驱动模型、扩展性与安全基础设施上全面换代。
- **EDK II 是 UEFI 的开源参考实现**：`MdePkg`/`MdeModulePkg` + PCD + INF/DEC/DSC/FDF 构成了「模块化固件」的工程模板，也是《Embedded Firmware Solutions》实操章的落点。

在下一节，我们将反方向看 UEFI 的阴影面：传统 BIOS 的实模式遗产与 CSM 兼容层——为什么 2020 年了系统还要保留一段 16 位的历史。
