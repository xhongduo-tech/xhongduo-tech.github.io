---
title: 开源固件（coreboot/LinuxBoot、固件供应链透明化）
date: 2026-08-07
---

# 开源固件（coreboot/LinuxBoot、固件供应链透明化）

<div class="epigraph">
<p>只要眼球足够多，所有 bug 都会变浅。</p>
<footer>—— 埃里克 · 雷蒙（Eric S. Raymond），引自林纳斯定律（Linus's Law）</footer>
</div>

<div class="article-byline">
<p>第三级 · 固件与启动链（BIOS/UEFI/嵌入式引导） ｜ Zimmer, Sun, Jones &amp; Reinauer《Embedded Firmware Solutions》第4/5章 ｜ 2026-08-07</p>
</div>

## 为什么从开源固件开始

主流 PC 的固件是闭源黑盒：OEM 从 AMI、Insyde、Phoenix 买来 BIOS，厂商签 NDA 才能看代码。但固件恰恰是**整个平台最大的攻击面之一**，黑盒意味着「你不信任的东西在初始化你信任的一切」。开源固件运动（coreboot、LinuxBoot）要做的，正是把这段关键代码从黑盒变成白盒。

《Embedded Firmware Solutions》第4章（coreboot 构建）与第5章（Chromebook 固件内部）是这条线的完整教材：coreboot 的历史、构建体系、启动阶段与设备树。<span class="marginnote">coreboot 前身叫 LinuxBIOS，2003 年由 Ron Minnich 在洛斯阿拉莫斯实验室发起——最初想法很激进：干脆用一个简化 Linux 内核直接初始化硬件。后来演进为「极简引导层 + 可插拔 payload」的形态。</span>

这段历史里有一个转折点值得记住：**2010 年前后项目从 LinuxBIOS 改名 coreboot**，原因是「我们根本不再依赖 Linux」——引导代码是裸的 C 程序，Linux 只是 payload 的一种选择。改名的同时也确立了核心纪律：**bootblock 要尽可能小、romstage 只碰必要硬件、把「用户可见的功能」全部外置给 payload**。这条纪律至今仍是 coreboot 设计的北极星。

## 1 coreboot 的设计哲学

coreboot 的目标是 **最小化引导代码**：它只做「把硬件初始化到能运行 payload 的程度」，其余一切交给 payload 去完成。这个哲学体现在它的启动阶段划分上：

- **bootblock**：从 Flash 顶部入口执行，初始化最低限度的 CPU/内存访问路径，然后加载 romstage；
- **romstage**：初始化 Cache-as-RAM 与 DRAM（通常调用 Intel FSP 的 `TempRamInit`/`FspInitEntry`），把内存参数固化下来；
- **ramstage**：初始化芯片组与设备，构建 **coreboot 设备树**；
- **payload**：交给下一个程序——SeaBIOS（传统 BIOS 模拟）、TianoCore/EDK II（UEFI）、U-Boot 或 LinuxBoot。

coreboot 的镜像也不是「一个扁平的二进制」，而是按 **CBFS（Coreboot File System）** 组织成文件系统形态的容器：bootblock 在 Flash 顶部，romstage/ramstage/payload 作为「文件」各占一段，文件系统元数据记录各自的位置与压缩状态。**「固件镜像 = 一个可检索的文件系统」**，这是 coreboot 与「一整块 ROM 程序」的传统 BIOS 最大的结构差异。

<span class="marginnote">对比 UEFI 的六阶段，coreboot 更「薄」：UEFI 自带协议总线、驱动模型与 shell，coreboot 只保留「点亮内存、配好芯片组」这一核心，把用户交互与策略全部外置到 payload。代价是生态没有 UEFI 统一，好处是启动更快、代码更少、审查更容易。</span>

**核心对比表**（纯概念主题，以表代替公式）：coreboot、UEFI 与 LinuxBoot 的分工常被混淆：

| 维度 | coreboot | UEFI（EDK II） | LinuxBoot |
| --- | --- | --- | --- |
| 定位 | 最小引导层 | 完整固件框架 | coreboot payload |
| 初始化范围 | 内存 + 芯片组 | 全部硬件 + 驱动模型 | 复用 Linux 驱动 |
| 用户接口 | 无（交给 payload） | UEFI Shell/Setup | Linux 用户态 |
| 驱动来源 | 手工 + FSP | MdeModulePkg 等 | Linux 内核驱动 |
| 典型启动时间 | 数百毫秒级 | 秒级 | 秒级 |
| 可审查性 | 高（开源） | 部分开源 | 高 |

## 2 coreboot 的实际收益

为什么数据中心与 Chromebook 愿意用 coreboot？几个可量化的理由：

- **启动速度**：Chromebook 用 coreboot 把「按电源键到浏览器」压到数秒；服务器无图形化固件界面也能省掉秒级的初始化开销；
- **供应链透明**：代码可审计，减少「固件里藏后门」的信任风险（见本专题《固件安全》）；

**Chromebook 是 coreboot 最大规模的商业验证**（也是《Embedded Firmware Solutions》第5章的主题）。Google 从 2011 年起让 Chromebook 全面采用 coreboot，并把「快速启动 + 固件升级 + 恢复模式」做成了产品体验的一部分：电源键到浏览器通常只要数秒，固件更新随 Chrome OS 一起交付，坏了可进 recovery 重装系统与固件。**「开源固件不是玩具，而是量产级产品」**——Chromebook 的出货量是这句话的注脚。
- **可裁剪**：按板卡裁剪，只保留需要的组件，Flash 占用比通用 UEFI 小得多；

coreboot 对「板卡」的支持方式是**设备树（coreboot device tree）**：每个主板的 `.c` 文件里描述「这颗 SoC 挂哪些设备、资源怎么分配」，ramstage 据此执行硬件初始化。它与 ARM 的 DTB 同名不同物：coreboot 的设备树是**编译期代码**（不是运行期数据），但理念相通——**把「板级差异」从代码里抽出来描述**。
- **社区协作**：核心板卡维护在 open 的 Git/Gerrit 流程里，补丁经 sign-off 提交，质量问题公开可查。

coreboot 社区还有一套「**board 维护者 + sign-off**」的文化：每个主板的代码由认领它的维护者负责，提交须带 Signed-off-by 并经过邮件列表评审。这套流程保证了「哪怕核心开发者离开，板卡支持也不会失联」——**开源固件的可持续性，靠的是流程而非个人**。

<span class="marginnote">想动手试试？装个 `qemu` + coreboot 的 QEMU 板卡，或买一块 MinnowBoard MAX / UP² 这类社区支持的主板，按文档 `menuconfig` 一跑，就能在真实硬件上感受「编译一次固件」的全流程——比纯读文档有效得多。</span>

**易错点｜辨析：** 有人说「coreboot 就是没有图形界面的 BIOS」。更准确的说法是：**coreboot 把「图形界面、启动菜单、UEFI 协议」这些本该属于上层的东西，交给了可替换的 payload**。同一块主板上，你可以换不同的 payload 得到不同的体验——这是「固件可组合」思想的极致体现。

### 一次 coreboot 构建长什么样

`_Embedded Firmware Solutions_` 第4章给了完整的实操路径，把它压缩成三步：

1. **配置**：`make menuconfig` 选择主板的 mainboard（如 `minnowboard_max`）、选择要加载的 payload 与 FSP 路径；
2. **编译**：`make` 依次构建工具链、romstage/ramstage、payload，最后把各段**打包进 CBFS**，生成可烧写的 `coreboot.rom`；
3. **烧录**：用 `flashrom` 把镜像写进 SPI Flash——`flashrom` 本身是开源固件生态最重要的工具之一，支持数百种芯片。

这套流程最鲜明的特征是**可重复、可审计**：同一份源码 + 同一份配置，任何人能在任何机器上重建出字节级一致的固件——这正是「供应链透明」在工程上的落点。

## 3 LinuxBoot：让 Linux 当引导程序

**LinuxBoot**（曾名 NERF/NetBoot）是 coreboot 生态里的激进派 payload：**直接把一个精简 Linux 内核当引导程序用**。它不再写私有引导代码，而是利用 Linux 内核里已经成熟、且被大规模审查过的驱动。

LinuxBoot 的典型价值在**数据中心**：

- 网络启动（PXE/HTTP）不再依赖固件实现，而是用内核的网络栈；
- 现场诊断（内存、CPU、NIC 测试）用 Linux 用户态工具完成，而非固件私有脚本；

LinuxBoot 的实现细节也体现了「复用内核」的极致：它把内核启动参数（`init` 指向一个 busybox）编进镜像，引导时内核以**极小的 initramfs** 起来，跑几行脚本加载真正要启动的 OS——相当于**「用 Linux 内核当引导程序，用 shell 脚本当引导逻辑」**，把整个 Boot 路径都变成可审计、可调试的 Linux 环境。
- 固件供应链透明化：整个 boot 路径都是开源代码。

<span class="marginnote">LinuxBoot 的成名战役是 Facebook 在 Open Compute Project 里的实践：服务器启动时先用一个极小的 Linux 做硬件初始化与诊断，再把真正服务的 OS 拉起来。启动时间与故障诊断效率都明显改善——「固件」在这里彻底回归了「软件」。</span>

## 4 开源固件的边界与未来

开源固件不是万能的，边界同样清晰：

- **CPU/内存初始化仍是闭源**：Intel FSP、AMD AGESA 以二进制形式嵌入，coreboot 调用它们。这是开源固件绕不开的「黑盒内核」——也是固件供应链透明化至今未竟全功的根本原因；

这个「黑盒内核」不是 coreboot 的懈怠，而是**知识产权与验证成本的现实**：内存训练参数、微码补丁、CPU 私有状态机属于芯片厂商的核心资产，开源等于把这些「芯片的软性实现」公开。**开源固件的边界，最终划在「芯片厂商愿意公开什么」这条线上**——这也解释了为什么 AMD 在部分平台公开更多 AGESA 细节后，社区支持立刻跃升。
- **指纹与密钥**：部分平台的密钥管理、Intel ME/PSP 的私有固件仍不可见（见《固件安全》）；
- **新平台支持滞后**：闭源固件随芯片首发，coreboot 支持往往要等社区适配。

即便如此，趋势明确：**固件正在向「开源 + 少量必要二进制」演进**。

几个值得关注的落点：Intel 的 **Slim Bootloader**（面向精简平台的轻量 UEFI 实现，代码开源）、coreboot 社区不断扩大的**板卡支持矩阵**、以及 **UEFI-on-coreboot**（把 EDK II 当 payload，同时拿到 coreboot 的启动速度与 UEFI 的生态）这条「鱼与熊掌兼得」的路线。**开源固件不再是与 UEFI 对立的另类选择，而正成为「多快好省」的主流选项之一**。AMD 在部分平台开源了 AGESA 的更多部分，Intel 的 Slim Bootloader 与 FSP 也在推动「标准接口 + 模块化」；而 LinuxBoot 已经把「引导代码可审计」从口号变成了数据中心里每天运行的事实。

## 5 小结

- coreboot 是**最小引导层**：只初始化到能运行 payload，把接口与策略外置。
- 启动流程分 **bootblock → romstage → ramstage → payload**，内存初始化常依赖 Intel FSP。
- 对比 UEFI，coreboot 更薄更快更可审查，但生态分散、无统一驱动模型。
- 镜像用 **CBFS** 组织成「固件文件系统」：bootblock 在 Flash 顶部，romstage/ramstage/payload 作为可检索文件各占一段。
- **LinuxBoot** 用精简 Linux 内核当 payload，把固件供应链透明化落到实处。
- Chromebook 的**量产验证**证明开源固件可支撑「秒级启动 + 在线更新 + 恢复模式」的产品体验；`flashrom` 是开源烧录生态的地基。
- 边界仍在：**CPU 初始化、密钥管理、独立安全处理器固件**是开源固件暂时碰不到的黑盒。
- 边界的原因不是社区不努力，而是**芯片厂商的知识产权与现实验证成本**——「芯片厂商愿意公开什么」决定了开源固件的上限。
- coreboot 的**启动阶段（bootblock→romstage→ramstage→payload）**与 x86 启动链的 SEC/PEI/DXE 遥相呼应，是「同一问题、两套组织」的绝佳对照。
- **Slim Bootloader、UEFI-on-coreboot** 等路线正让「开源 + UEFI 生态」从二选一变成可兼得。
- coreboot 社区靠「**board 维护者 + sign-off**」流程保证板卡支持的可持续性——开源固件的可持续性靠流程而非个人。

在下一节，我们讨论固件如何被更新：capsules 更新、防回滚与 A/B 分区——把「固件可修复」和「固件可信任」接在一起。顺便说一句，coreboot 与 Chromebook 的「云端恢复」正是这个机制在开源世界的现成范例。
