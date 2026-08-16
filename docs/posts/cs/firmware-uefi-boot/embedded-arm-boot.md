---
title: 嵌入式引导（ARM 的 BootROM→TF-A→U-Boot→内核链）
date: 2026-08-07
---

# 嵌入式引导（ARM 的 BootROM→TF-A→U-Boot→内核链）

<div class="epigraph">
<p>预测未来最好的方式，就是去发明它。</p>
<footer>—— 艾伦 · 凯（Alan Kay）</footer>
</div>

<div class="article-byline">
<p>第三级 · 固件与启动链（BIOS/UEFI/嵌入式引导） ｜ Zimmer, Sun, Jones &amp; Reinauer《Embedded Firmware Solutions》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从嵌入式引导开始

x86 的启动链依赖「复位向量 + 统一固件规范」，而 ARM 世界是另一套哲学：**每颗 SoC 的 BootROM 各不相同，引导链由「芯片厂商 BootROM + 公开的 TF-A + 通用 U-Boot」三段拼装**。理解 ARM 引导，就理解了嵌入式系统「厂商私有与开源通用并存」的生态底色。

对《Embedded Firmware Solutions》而言，这正是第8章「Putting It All Together」的主题：把前面各章的 FSP、coreboot、U-Boot 思想收束到一块真实的 ARM SoC 上。<span class="marginnote">ARM 生态的经典笑话是「同一个 Linux 内核，在一块板子上能引导，在另一块上就可能不行」——差异的根源往往就在 BootROM/引导链与设备树（DTB）上，而不是内核本身。</span>

## 1 ARM 引导全景：权限分级是钥匙

ARM 引导链的真正驱动力是 **异常级别（Exception Level）**。ARMv8 把处理器特权分成四档：

- **EL3**：最高权限，运行 **TF-A（Trusted Firmware-A）** 的 BL31，负责安全世界切换与电源管理；
- **EL2**：虚拟化层，运行 Hypervisor；
- **EL1**：操作系统内核所在；
- **EL0**：用户态应用。

引导的本质，就是**从 EL3 一路降到 EL1 的过程**：越早的阶段权限越高，越晚的阶段越接近应用。安全世界（TrustZone）则通过 EL3 的监控（Monitor）模式在「安全/非安全」两个世界之间切换。

## 2 BootROM 与 SPL：芯片的第一口气

每颗 ARM SoC 出厂时都烧了一段 **BootROM（固化只读引导）**，由芯片厂商在流片时焊死，无法修改。它的任务极其有限：<span class="marginnote">BootROM 的「微小」是安全与成本的折中：它不可变，所以可以当信任锚；它越小，芯片面积与验证成本越低。攻击 BootROM 的唯一途径是芯片级漏洞利用——这正是「第一段代码必须不可改」的原因。</span>

1. 上电后初始化极少量硬件（时钟、RAM 控制器的一小部分、UART），尽快点亮一个能输出日志的串口——**「先让板子开口说话」在嵌入式引导的第一毫秒就要做**；
2. 从固定位置寻找引导源（eMMC/SD/NOR/QSPI/USB），读取第一级引导程序；
3. 把它加载进 SRAM，跳转执行。

BootROM 从哪里找引导源？答案是**启动引脚（boot strap）**：SoC 上几根被拉高/拉低的引脚编码了「从 eMMC / SD / SPI NOR / USB / UART 启动」的选择，BootROM 读引脚、按优先级依次尝试。开发板上的「拨码开关切启动模式」操作，改的正是这几根引脚的电平——**换启动介质，本质是改 BootROM 的寻源顺序**。

### 为什么 BootROM 要在 SRAM 里跑

BootROM 的宿主内存是芯片内置 **SRAM**（通常只有几十到几百 KB），而不是 DRAM——因为 DRAM 还没初始化。于是「第一级代码必须足够小，能塞进 SRAM 且不自举 DDR」就成了硬约束。U-Boot 的 **SPL** 正是被这个约束逼出来的设计：SPL 初始化 DDR，再把「不满足小约束」的完整 U-Boot 请进 DRAM。

由于 SRAM 通常只有几十 KB，第一级程序往往只能再加载一个更完整的第二级——这就是 **SPL（Secondary Program Loader）** 的由来。U-Boot 的 SPL 正是这样工作的：SPL 负责初始化 DDR，再加载完整的 U-Boot 主程序。

## 3 TF-A：EL3 世界的信任固件

**TF-A（Trusted Firmware-A）** 是 ARM 官方维护的开源安全固件，按 BL（Boot Loader）分级组织：

| 阶段 | 运行级别 | 职责 |
| --- | --- | --- |
| BL1 | EL3（BootROM 内） | 信任根，加载 BL2 |
| BL2 | EL3 | 校验并加载 BL31/BL32/BL33 |
| BL31 | EL3 | 常驻运行时服务（SMC、电源管理） |
| BL32 | S-EL1 | 可信 OS（如 OP-TEE） |
| BL33 | EL2/EL1 | 非安全世界引导程序（U-Boot/EDK II） |

**核心对比表**（纯概念主题，以表代替公式）：把 ARM 与 x86 的启动链并列，差异一目了然：

| 维度 | x86（PI） | ARM（TF-A） |
| --- | --- | --- |
| 第一段代码 | 复位向量 → SEC | BootROM → BL1 |
| 信任根 | CRTM（可更新区域） | BootROM（焊死） |
| 权限分级 | 实模式→保护模式→长模式 | EL3→EL2→EL1 |
| 临时内存 | Cache-as-RAM | SoC 内 SRAM |
| 安全世界 | SMX/TXT、SMM | TrustZone/EL3 |
| 启动加载器 | UEFI OS Loader | U-Boot 等 |

BL1 从 BootROM 接棒后，校验 BL2；BL2 负责把 BL31（常驻）、BL32（TEE）与 BL33（非安全世界）逐一加载进内存，完成 **EL3 信任链的建立**。

BL31 常驻 EL3 后，普通世界（EL1 内核 / EL2 虚拟化）与安全世界（TEE）之间的一切特权操作都要经过它：非安全世界调用 **SMC（Secure Monitor Call）** 指令，把请求交给 BL31 的运行时服务（如电源管理、可信服务）。**SMC 是 ARM 双世界唯一的「门」**——这也是为什么「CPU 睡眠/唤醒」「密钥查询」最终都会走到 EL3。之后 BL31 常驻 EL3，处理安全调用（SMC）与电源状态切换，系统进入「非安全世界跑 Linux，安全世界跑 OP-TEE」的双世界格局。

## 4 U-Boot 与内核交接

进入非安全世界后，常见引导程序是 **U-Boot（Das U-Boot）**。它的工作比 PC 上的引导加载器更「全能」：

- 从存储/NFS/串口/网络（TFTP、PXE）加载内核镜像；
- 解析 **设备树（Device Tree Blob, DTB）**，把「这块板子上有哪些外设、地址在哪、中断怎么接」交给内核；

设备树本身值得多说一句：它是一棵描述「硬件事实」的树形数据（node + property），用 **DTS 源文件**编写、经编译器生成 **DTB 二进制**。它「固定」在板级，与内核镜像解耦——**换一块外设变体板卡，往往只需换 DTB，不必换内核**，这正是嵌入式「硬件可插拔」的软件基础。
- 设置启动参数，`booti`/`bootm` 跳到内核入口。

U-Boot 的强大来自它自带一个**命令行与脚本环境**：`env` 环境变量、`bootcmd` 启动命令、`tftp`/`ext4load` 等加载命令一应俱全。开发者的日常是「`setenv bootargs console=ttyS0 ...` 改启动参数 → `saveenv` 存进存储 → `boot` 重新引导」——**改一行启动参数不用重编固件**，这是嵌入式迭代速度的重要来源。<span class="marginnote">对比 x86 的 UEFI Shell（本专题《UEFI 体系》），U-Boot 命令与 UEFI Shell 命令在哲学上同构：都是「OS 之前可交互的命令行」——只是 U-Boot 更深地绑定了具体板卡的硬件语义。</span>

**设备树是嵌入式引导与 PC 引导最大的不同**：PC 内核靠 ACPI 表与 PCI 枚举认识硬件，ARM 内核靠固件传来的 DTB 认识硬件。

两者的取舍也值得对比：ACPI 是「固件生成、OS 解释执行」的动态模型，能表达运行期状态与电源行为；DTB 是「板级静态快照」，简单直接、几乎没有运行时开销。ARM 生态同时存在两种选择——服务器用 ACPI（为了与数据中心 OS 对齐），嵌入式用 DTB（为了轻量）——**「选 ACPI 还是 DTB」本质是「选生态还是选轻量」的权衡**。DTB 是「板级硬件事实的快照」，U-Boot 可以就地打补丁（`fdt set`），内核则只读它。<span class="marginnote">如果 DTB 与真实硬件不符，症状很典型：内核起来了但某个驱动失联、中断号错乱、内存大小不对。排查的第一步永远是「先确认 U-Boot 传给内核的 DTB 与实际板卡一致」。</span>

## 5 嵌入式引导的现代变体

嵌入式引导没有「标准答案」，只有「生态惯例」，值得列几条现役路线：

- **主流 Linux SoC**：BootROM → SPL → U-Boot → DTB → 内核，TF-A 按需插入 EL3；
- **UEFI on ARM**：ARM 服务器用 EDK II + ACPI，为的是与 x86 生态对齐（企业级 OS 直接支持）；
- **Android 设备**：BootROM → xBL/ABL → 各自的 bootloader → Android boot image；
- **极简系统**：BootROM 直接引导裸机程序或 Zephyr 等 RTOS，跳过 U-Boot。

值得注意的是，**TF-A 并非 ARM 的专利**：业界还有 vendor 私有变体（如高通的 xBL、NVIDIA 的 BootROM 链），都遵循「BL1→BL2→BL31→BL33」的逻辑骨架，只是实现与签名流程私有。**看懂 TF-A 的标准骨架，就能快速读懂任何私有 ARM 引导链**——它提供了通用语言。

**易错点｜辨析：** 不要以为「嵌入式 = 一定没有 UEFI」。ARM 服务器几乎全部用 UEFI 引导链；而手机 SoC 则是「私有 BootROM + 私有链」。**判断依据不是 ISA（指令集），而是你要对齐的软件生态**——要对齐数据中心生态就用 UEFI+ACPI，要对齐裸机/RTOS 生态就保留 U-Boot。

## 6 小结

- ARM 引导由 **BootROM（不可变）→ TF-A（EL3）→ U-Boot → 内核** 三级拼装，每颗 SoC 的 BootROM 私有。
- 权限模型是 **EL3/EL2/EL1/EL0** 四级，引导是「从最高权限一路降级」的过程。
- **TF-A** 按 BL1–BL33 分工：BL1/BL2 建信任链，BL31 常驻 EL3，BL32 跑 TEE。
- **SMC 指令是双世界唯一的门**：普通世界与 TEE 的一切特权交互都经 BL31 转发。
- 与 x86 相比，ARM 的临时内存是 SoC 内 SRAM、信任根焊死在 BootROM、硬件事实靠 **DTB** 而非 ACPI 传递。
- **DTB 是板级静态快照**，换外设变体板卡常只需换 DTB 不换内核；ACPI 与 DTB 的选择本质是「生态 vs 轻量」的权衡。
- BootROM 按**启动引脚**选引导源，在 SRAM 里跑第一级代码——「先让板子开口说话」是嵌入式引导从第一毫秒就要做对的事。
- 现代嵌入式引导还有 UEFI-on-ARM、Android 私有链等多条路线，选择取决于目标生态。
- **U-Boot 自带命令行与环境变量**，改启动参数无需重编固件；其脚本哲学与 UEFI Shell 同构，但更贴近具体板卡硬件。
- 看懂 **TF-A 标准骨架（BL1→BL2→BL31→BL33）** 就能快速读懂任何私有 ARM 引导链——它提供了通用语言。
- 启动引脚（boot strap）决定 BootROM 从哪找引导源；换启动介质本质是改这几根引脚的电平。

在下一节，我们回到「固件怎么做出来的」话题：开源固件 coreboot 与 LinuxBoot，看社区如何把闭源黑盒固件推向透明化。
