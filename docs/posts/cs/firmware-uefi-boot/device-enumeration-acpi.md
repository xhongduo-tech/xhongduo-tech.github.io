---
title: 外设枚举（PCIe 枚举、ACPI 表的生成）
date: 2026-08-07
---

# 外设枚举（PCIe 枚举、ACPI 表的生成）

<div class="epigraph">
<p>一切应当尽量简单，但不能更简单。</p>
<footer>—— 阿尔伯特 · 爱因斯坦（Albert Einstein）</footer>
</div>

<div class="article-byline">
<p>第三级 · 固件与启动链（BIOS/UEFI/嵌入式引导） ｜ UEFI Forum《UEFI Specification》与《ACPI Specification》相关章节 ｜ 2026-08-07</p>
</div>

## 为什么从外设枚举开始

内存点亮之后，固件面前是一个「还不知道自己长什么样」的机器：板卡插槽里有没有显卡？M.2 上挂着什么 SSD？板载网卡占了哪个地址？固件必须像考古学家一样**逐层扫描总线**，给每个设备一个身份（BDF）、一套资源（MMIO/中断），最后把这些事实**翻译成操作系统认识的表格（ACPI）**。这就是**外设枚举（Device Enumeration）**。

对《Embedded Firmware Solutions》的读者，这一节是把「芯片组配置」从手册术语拉回体系结构直觉的关键：**PCIe 枚举决定「设备怎么被找到」，ACPI 决定「设备怎么被 OS 消费」**。<span class="marginnote">两条总线体系在这里交汇：PCIe 是高速串行互连（挂显卡、SSD、网卡），而 ACPI 表是固件与 OS 之间的「硬件说明书」——OS 不直接探测一切硬件，而是读固件留下的表。</span>

## 1 PCIe 拓扑与 BDF 命名

PCI/PCIe 的拓扑是一棵**总线树**：从**根复合体（Root Complex）** 出发，根端口（Root Port）连接**总线（Bus）**，总线下挂**设备（Device）**，每个设备可有若干**功能（Function）**。每个功能用三元组唯一编号：

$$
\text{BDF} = (\text{Bus}, \text{Device}, \text{Function}), \qquad 0 \le B \le 255, \; 0 \le D \le 31, \; 0 \le F \le 7
$$

BDF 的读法：`00:1f.0` 意思是「Bus 0、Device 31、Function 0」——这正是很多 x86 主板南桥（PCH）的默认位置。功能号只有 3 位（0–7），因此「一块物理卡上装两个网卡」在 BDF 上就是 `x:y.0` 与 `x:y.1`——**功能号把「单插槽多设备」这件事直接写进了地址空间**。Linux 里 `lspci`、Windows 设备管理器列出的正是这套 BDF。<span class="marginnote">PCIe 还允许通过 Switch 扩展总线，形成多级总线树；每棵子树占一个 Bus 号，固件要按深度优先逐层分配 Bus 号。枚举的本质就是「给这棵树上的每个节点发身份证」。</span>

## 2 枚举流程：扫描、分配、配置

固件的 PCIe 枚举通常分三步：

1. **扫描（Discovery）**：从 Bus 0 出发，对每个「疑似设备槽位」写配置空间，读厂商/设备 ID。读到 `0xFFFFFFFF` 表示空槽，读不到则跳过。遇 Bridge（PCIe 上有下游总线）就递归进入下一层。
2. **分配（Allocation）**：为每个设备分配 MMIO 窗口、I/O 端口、中断号。设备通过配置空间的 `BAR`（Base Address Register）声明自己「想要多大窗口」，固件把系统里空闲的地址空间切给它。

一个现代细节：**大显存大 NVMe 需要 64 位 BAR**。显存动辄 8–24 GB，若只能落在 32 位地址空间（4 GB 以下），很快就耗尽。于是 PCIe 支持 64 位 BAR，把窗口放到 4 GB 之上——而现代 PC 的 MMIO 高位窗口（Above 4G Decoding）正是为此而生。**「能不能打开 Above 4G」直接决定大显存能否被完整识别**，这是装机党最熟悉的开关之一。
3. **配置（Configuration）**：写入分配好的资源、使能 Bus Master、设置 Max Payload/Read Request 大小，最后启动设备的电源管理状态。

<span class="marginnote">想亲眼看到这套枚举的产物？Linux 里 `lspci -v` 展示 BDF、BAR、链路能力，`lspci -tv` 画出总线树，`dmesg | grep pci` 能看到枚举顺序——它们都是固件在 DXE/BDS 阶段做过的同一件事的「事后回放」。</span>

BAR 的「尺寸探测」是个巧妙机制：固件把 `0xFFFFFFFF` 写进 BAR，再读回来——低位里「有多少位被清零」就说明窗口大小是多大（比如 16 MB 窗口对应地址对齐到 16 MB）。**不用读任何文档，固件就能知道每个设备想要多大地址窗口**，这是 PCI 枚举里最优雅的约定之一。

**易错点｜辨析：** 很多调试新手看到「设备不见了」就以为是硬件坏。其实**枚举失败的多数原因是资源不够**——特别是 32 位 MMIO 空间被大显存/大 NVMe 占满，32 位设备（老显卡、部分桥接芯片）拿不到窗口。固件日志里 `BAR` 分配失败、设备落到无资源状态，都是这一类问题。

## 3 公式解析：PCIe 链路带宽

设备枚举时，链路能力决定设备能跑多快。PCIe 链路带宽公式是：

$$
\text{BW} = \text{lanes} \times \text{data rate} \times \text{encoding}
$$

拆三步看：

- **第一步，`lanes`（通道数）**：一条链路由 1/2/4/8/16 对差分线组成，标作 x1/x2/x4/x8/x16。x16 显卡槽就是 16 对收发线。
- **第二步，`data rate`**：每通道每方向的数据速率。PCIe 3.0 是 8 GT/s（Giga-Transfers/s），PCIe 4.0 是 16 GT/s，PCIe 5.0 是 32 GT/s。
- **第三步，`encoding`（编码效率）**：不是全部数据位都是有效载荷。PCIe 1.0/2.0 用 8b/10b 编码，效率 80%；PCIe 3.0+ 改用 128b/130b 编码，效率约 98.5%。

编码史其实是一部「带宽挤牙膏史」：8b/10b 每发 8 位要付 2 位开销（保证 DC 平衡与时钟恢复），到了 PCIe 3.0 靠更聪明的扰码器把开销压到 2/130，带宽立刻上涨近 20%——**「同样线速下能挤出的有效带宽」是每代 PCIe 的核心卖点**。

代入 PCIe 3.0 x16：`16 × 8 × 0.985 ≈ 126 GB/s`（双向各算一份），这与显卡接口的 32 GB/s（单向 x16 每向 `16 × 8 × 0.985 / 4`）口径一致——厂商标注时务必先问清楚是单向还是双向。

### 中断：从共享 INTx 到 MSI

枚举的另一个常被忽略的环节是**中断分配**。传统 PCI 用 **INTx 共享中断**：多个设备共用一个中断线，靠驱动「猜测是不是自己」来响应——效率低、易冲突。PCIe 时代主推 **MSI/MSI-X（消息信号中断）**：设备直接向 CPU 写一条「中断消息」，不再共享引脚。

对固件而言，MSI 是「给设备分配一段能写消息的 MMIO 地址 + 向量号」，且必须在设备使能前完成。**MSI-X 把中断从「共享引脚」变成「私有消息」**，是现代高性能网卡与 NVMe 能跑到百万级 IOPS 的前提之一——也是枚举时「中断号总是不对」的常见故障源。

## 4 配置空间与 ECAM

每个 PCIe 设备有 **4 KB 配置空间**（PCIe 从传统 PCI 的 256 字节扩展而来），内含厂商 ID、设备 ID、BAR、能力链表等。

配置空间里最有信息量的是 **能力链表（Capability List）**：从 `0x34` 的 Capability Pointer 出发，一条条链着 PM（电源管理）、MSI/MSI-X、Express（PCIe 专属能力）等。读这份链表，固件就知道「这设备支持什么、能跑多快链路、能否热插拔」——**枚举的本质之一，就是把这 4 KB 配置空间读透**。访问方式历史上是 I/O 端口 `0xCF8/0xCFC` 的「配置读/写」，PCIe 时代则优先用 **ECAM（Enhanced Configuration Access Mechanism）**：把整个配置空间**内存映射**到一段固定地址（通常是 `0xE0000000` 附近），OS 可以像读内存一样直接访问任意 BDF 的配置。

这段 ECAM 基址记录在哪？在 **ACPI 的 MCFG 表**里。**这就是固件与 OS 交接的关键点**：<span class="marginnote">OS（Linux 的 `pci_ecam`、Windows 的 PCI 总线驱动）启动后并不重新枚举，而是直接读固件留下的 MCFG、MPTable 或 ACPI 设备信息，继续 OS 世界的驱动工作。固件的枚举结论，通过 ACPI 表「交棒」给 OS。</span>这也是为什么修改 PCIe 拓扑后必须重刷固件：**树的结构是固件第一次建的，OS 只是它的租客**。

## 5 ACPI：把硬件事实写成表格

**ACPI（Advanced Configuration and Power Interface）** 是固件与 OS 之间的「硬件合同」。它的物理入口是 **RSDP（Root System Description Pointer）**，位于 EBDA 或 `0xE0000–0xFFFFF` 区域，RSDP 指向 XSDT/FADT，FADT 又指向 **DSDT**——一份用 **AML（ACPI Machine Language）** 编写的解释执行字节码，描述平台的所有设备与电源状态。

固件在 DXE/BDS 阶段生成并填充这些表，并通过 **EFI Configuration Table** 把表指针挂进 UEFI 系统表——这就是「ACPI 与 UEFI 的接口」：OS 从 UEFI 系统表里读到 RSDP 的地址，再顺着表链走完。**ACPI 表是「固件生成、UEFI 托付、OS 消费」的三方接力**。<span class="marginnote">ACPI 不是纯数据，还带一套解释器。AML 里的方法（如 `_OSI`、`_PTS`、`_WAK`）会在 OS 里被执行，让「关屏幕」「休眠」「唤醒」这类行为由固件定义的代码驱动。DSDT 有 bug 时，Linux 社区甚至会提供 DSDT override——因为 OS 里跑的解释器可以加载补丁后的表。</span>

- **MADT**：中断控制器（Local APIC/IO-APIC）拓扑；
- **MCFG**：PCIe ECAM 基址；
- **HPET/DSDT**：定时器与平台设备；
- **FADT**：电源管理寄存器、引导标志。

对操作系统开发者，这张表链的「消费方式」也很统一：**内核先找到 RSDP，校验签名 `RSD PTR`，再跟随 XSDT 里的指针逐张加载**。哪张表缺失或校验失败，对应功能（如 HPET 定时器、PCIe ECAM、APIC 中断）就整体退化为「检测不到」——**ACPI 表的完好性，是硬件功能在 OS 里可见的前提**。

操作系统读取这些表的方式也值得记住：Linux 内核在启动早期解析 RSDP，逐张校验 **ACPI 表头（签名、长度、校验和）**，再按需调用 AML 方法。

**电源状态**是这套解释器最活跃的领域：**S 状态**（S0 运行、S3 睡眠、S5 关机）与 **D/C 状态**（设备/处理器运行深度）都通过 AML 方法协调。OS 要睡 S3 时，先调 `_PTS` 让固件保存关键寄存器，唤醒时调 `_WAK` 恢复现场——**每一次合盖睡眠都是一次固件与 OS 的握手**，DSDT 里 `_PTS/_WAK` 有 bug 就表现为「睡下去醒不来」或「唤醒后设备失灵」。**固件与 OS 对 ACPI 表的「信任」是单向的**：OS 无条件接收固件给的地址，只做结构校验——这意味着表一旦有误，OS 只能「将错就错」或直接拒绝启动，这正是 DSDT 覆盖（override）成为 Linux 社区刚需的原因。

操作系统把 ACPI 表当作「固件写好的硬件事实」，据此构建 `acpi` 总线、驱动绑定与电源策略。

另一个现代话题是 **ACPI 热插拔**：PCIe 的插拔事件（如 NVMe 从 M.2 拔出）经中断上报后，固件运行 **AML 的 `_EJR/_PS0` 等方法**通知 OS，OS 再走 uevent 卸载/加载驱动。也就是说「拔插设备时系统怎么响应」，一部分由固件的 AML 脚本决定——**热插拔是固件与 OS 深度协作的现场**，也是 DSDT 写得不好时「插上没反应」的原因。**固件表写错 = OS 里的设备失踪、休眠失效或性能被锁**——ACPI 是 firmware 影响 OS 体验最直接的一层。

## 6 小结

- PCIe 是总线树，设备用 **BDF =（Bus, Device, Function）** 唯一命名，枚举分扫描、分配、配置三步。
- 与 ARM 世界的**设备树（DTB）**相比，ACPI 是「固件生成 + OS 解释执行」的动态模型，而 DTB 是「板级静态快照」——两者的取舍我们在嵌入式引导一节会继续对照。
- 常见「设备失踪」实为 **32 位 MMIO 资源耗尽**，不是硬件坏。
- PCIe 带宽 `BW = lanes × data rate × encoding`，注意单向/双向口径与 8b/10b 编码效率。
- PCIe 配置空间经 **ECAM 内存映射**，基址记录在 **MCFG 表**里。
- **ACPI**（RSDP→XSDT→FADT→DSDT/AML）是固件写给 OS 的硬件合同，表写错会直接影响 OS 行为。
- ACPI 表经 **EFI Configuration Table** 挂进 UEFI 系统表交给 OS；表缺失/校验失败会让对应硬件功能在 OS 里「消失」。
- 中断分配从共享 **INTx** 演进到 **MSI/MSI-X**；电源状态靠 AML 的 `_PTS/_WAK` 与 OS 握手；热插拔同样经 AML 方法协作。
- BAR 用「写 `0xFFFFFFFF` 再读回」自报窗口大小；能力链表记录设备支持的全部能力。
- 动手验证枚举结果的快捷路径：`lspci -v`、`lspci -tv`、`dmesg | grep pci`。
- 枚举「树的结构」是固件第一次建、OS 只是租客；改 PCIe 拓扑后必须重刷固件才能让 OS 看到新结构。
- PCIe 3.0 x16 理论带宽 ≈ 126 GB/s（双向），厂商标注先问清单向还是双向、编码效率是否已扣除。
- 现代 PCIe 的 **64 位 BAR / Above 4G** 是「大显存大 NVMe 被完整识别」的前提开关。
- ACPI 表缺失或校验失败，对应硬件功能在 OS 里「消失」——表链的完好性是硬件功能可见的前提。
- 中断分配从共享 INTx 走向 MSI/MSI-X，是现代高速网卡与 NVMe 高 IOPS 的前提之一。

在下一节，我们把镜头从 x86 切换到 ARM：嵌入式系统的引导链 BootROM→TF-A→U-Boot→内核，看同一套「从零到 OS」的问题，在精简指令集的世界里如何被重新回答。
