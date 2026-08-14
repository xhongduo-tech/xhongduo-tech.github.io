---
title: Fastboot 模式与线刷命令集
date: 2026-08-07
---

# Fastboot 模式与线刷命令集

<div class="epigraph">
<p>Fastboot 是刷机的「底层终端」——在系统还没起来之前，Bootloader 已经在这里等着接受你的命令。</p>
<footer>—— 刷机工具手册（Recovery 与固件刷写教程）</footer>
</div>

<div class="article-byline">
<p>生活技能树 · 移动设备系统定制与刷机 ｜ Recovery 与固件刷写教程 ｜ 2026-08-07</p>
</div>

## 为什么 Fastboot 是线刷的核心

上一节的 ADB 是在系统运行时「隔着窗户说话」，而 Fastboot 是**系统还没起来、Bootloader 已在运行时的「直通通道」**。解锁、刷 Bootloader、写 boot/recovery/system 分区、切槽位、回锁——这些操作都只能在 Fastboot 里做。它是**救砖与刷底层的第一选择**：即使系统全毁、Recovery 进不去，只要 Bootloader 还活着，Fastboot 就还能把设备救回来。这一篇讲透 Fastboot 模式与线刷命令集。Fastboot 背后的 Bootloader 概念，是第三级《操作系统》里「引导程序」的具体实例——理解了它，Linux 的 GRUB、Windows 的启动管理器都能触类旁通。

## 1 Fastboot 是什么：Bootloader 上的线刷终端

**Fastboot** 既是 Bootloader 的一个模式，也是与这个模式通信的电脑端命令行工具。它运行在设备侧的 Bootloader 环境里——此时内核与系统都还没加载，Fastboot 程序直接从 Bootloader 读取命令。

Fastboot 与 ADB 的分工必须分清：<span class="marginnote">ADB 与 Fastboot 是两套独立协议：<strong>ADB 需要系统里的 adbd 运行（Recovery 里也有），Fastboot 则由 Bootloader 直接提供，不需要系统</strong>。所以「系统崩了 → adb 用不了 → 但 fastboot 还能用」——这正是救砖的路径依赖。误把 fastboot 命令当 adb 敲，或反过来，是新手最常见的手误。</span>

Fastboot 的能力核心就两件：**写分区**（`flash`）与**查状态/切状态**（`getvar`、`oem`、锁相关命令）。它比 ADB 更底层，也因此更危险——命令没有「确认」二次拦截，敲下去就执行。

## 2 进入 Fastboot 模式与设备识别

**进入 Fastboot 模式**有三种常见方式：

**命令进入**：系统正常时，`adb reboot bootloader`。
**按键进入**：关机状态下按特定组合键。不同品牌不同：多数小米/一加是「音量下 + 电源」长按，三星有专门的 Download 模式（音量下 + 音量上 + 插线），Pixel 是「音量下 + 电源」。
**故障进入**：系统崩溃时部分机型自动掉进 Fastboot，此时是救砖的窗口。

**电脑端识别设备**：在 Fastboot 模式下敲

```
fastboot devices
```

能看到设备即成功。**看不到设备是线刷第一大卡点**，常见原因：没装 Fastboot 的 USB 驱动（Windows）、数据线问题、设备没真正进入 Fastboot（还在系统里）。<span class="marginnote">区分「设备还在系统里」还是「真的进了 Fastboot」：<strong>敲 `adb devices` 能看到的是系统态，`fastboot devices` 能看到的是 Fastboot 态</strong>。用错命令看设备，会误以为设备坏了。判断设备当前处于哪个模式，先看屏幕画面：Fastboot 模式一般有机器人图标或命令行界面。</span>

## 3 线刷命令集：flash、erase、boot 与槽位

Fastboot 的核心命令围绕「写、擦、读、切」四个动作展开：

**写分区（flash）**：
```
fastboot flash boot boot.img
fastboot flash recovery recovery.img
fastboot flash system system.img
```
格式是 `fastboot flash <分区> <镜像>`，一次一条命令。

**擦除分区（erase）**：把分区清空，如 `fastboot erase cache`。擦 `data` 分区等价于恢复出厂的数据部分。

**临时启动（boot）**：`fastboot boot <镜像>` **不写入分区**，仅从内存临时启动一个内核/Recovery。这是「先试试，不落盘」的安全操作——试 TWRP 常用 `fastboot boot twrp.img`。

**查询状态（getvar / oem）**：
```
fastboot getvar all
fastboot oem device-info
```
`getvar` 读设备变量（当前槽位、版本等），`oem device-info` 查锁状态。<span class="marginnote">`getvar all` 是最值得先跑的一条命令：<strong>它一次性暴露当前槽位、分区方案、解锁状态、回滚指数等关键信息</strong>。动手刷之前先跑一遍，等于给设备做一次「体检」，能避免很多「刷错槽位」「机型不匹配」的事故。</span>

**解锁/回锁**：
```
fastboot flashing unlock
fastboot flashing lock
```
Pixel 系用 `flashing` 动词，部分厂商用 `fastboot oem unlock`。回锁前务必刷回官方原版系统。

**槽位操作（A/B 设备）**：
```
fastboot set_active b
fastboot --slot b flash boot boot_b.img
fastboot --slot all flash boot boot.img
```
A/B 设备要指明槽位，`--slot all` 表示「两个槽都刷」——**切槽与刷错槽是 A/B 设备最常见的翻车点**。

## 4 公式解析：一次完整线刷的命令编排

线刷不是一条命令，而是一组命令按顺序执行。以刷入一套官方线刷包为例，标准编排是：

$$
\text{fastboot devices} \rightarrow \text{flash boot} \rightarrow \text{flash system} \rightarrow \text{flash vendor} \rightarrow \text{erase data} \rightarrow \text{reboot}
$$

逐步拆解：

- **`fastboot devices`**：确认设备在线。**缺了这一步，后面所有命令都发给「空气」**，报 `< waiting for any device >`。
- **`flash boot/system/vendor`**：把线刷包里的镜像逐个写入对应分区。顺序一般按镜像依赖排，先底层后上层。
- **`erase data`**：清空用户数据。官方刷机包的脚本通常默认执行，所以「线刷官方包会清数据」——除非你自己删掉这行。
- **`reboot`**：刷完重启。若刷的是带两个槽的包，重启前可能还有 `set_active` 切槽。

厂商的线刷包通常附带了现成脚本（如 Pixel 的 `flash-all.sh`/`.bat`），内容就是这组命令的自动执行。**读懂脚本里的命令序列，你就知道工具在背后干了什么**——这也是为什么 Fastboot 命令是「理解一切线刷工具」的钥匙。

## 5 核心要点：Fastboot 命令速查表

| 命令 | 作用 | 典型场景 |
| --- | --- | --- |
| `fastboot devices` | 列出 Fastboot 设备 | 线刷前确认 |
| `fastboot flash <分区> <镜像>` | 写镜像到分区 | 刷 boot/recovery/system |
| `fastboot erase <分区>` | 清空分区 | 清 cache/data |
| `fastboot boot <镜像>` | 临时启动不写入 | 试 TWRP/内核 |
| `fastboot getvar all` | 读设备变量 | 体检/查槽位 |
| `fastboot oem device-info` | 查锁状态 | 解锁前确认 |
| `fastboot flashing unlock/lock` | 解锁/回锁 | Pixel 系 |
| `fastboot set_active <槽>` | 切换 A/B 槽位 | A/B 设备 |
| `fastboot --slot all flash <分区> <镜像>` | 双槽都刷 | A/B 设备 |
| `fastboot reboot` | 重启设备 | 刷写完成 |

## 6 术语速查表

| 术语 | 含义 | 关键点 |
| --- | --- | --- |
| Fastboot 模式 | Bootloader 的线刷环境 | 不需要系统 |
| flash | 写镜像到分区 | 核心动作 |
| erase | 擦除分区 | 清数据 |
| boot（fastboot） | 临时启动镜像 | 不落盘 |
| getvar | 读设备变量 | 刷前体检 |
| slot | A/B 槽位 | 切错即翻车 |
| `--slot all` | 双槽操作 | 一次性刷两槽 |
| flashing unlock | 解锁命令 | Pixel 系 |
| oem device-info | 查锁状态 | 解锁前确认 |
| 线刷包 | 含镜像+脚本的压缩包 | 官方刷机资源 |

## 7 快速自查清单

线刷前最后一遍检查：

- 是否已确认设备在 **Fastboot 模式**（`fastboot devices` 有响应）？
- 线刷包的镜像**分区名**与设备分区方案是否匹配？
- A/B 设备是否确认了**当前槽位与目标槽位**？
- 是否知道这次线刷会不会**清空 data**（脚本里有没有 erase data）？
- 解锁/回锁命令是否**与厂商匹配**（flashing vs oem）？
- 是否已备好**官方线刷包**作为救砖底牌？

## 8 小结

- Fastboot 是 **Bootloader 提供的线刷终端**，与 ADB 是两套独立协议，系统崩溃后仍可用。
- 核心动作四类：**flash（写）、erase（擦）、boot（临时启动）、getvar/oem（读状态）**。
- A/B 设备必须管理**槽位**，`--slot all` 双槽全刷，切错槽是常见翻车点。
- 一次线刷是一组**按序编排的命令**：devices → flash 各分区 → erase data → reboot。
- 解锁/回锁命令**因厂商而异**：Pixel 用 `flashing`，部分厂商用 `oem`。

在下一节，我们把镜头转向手机端的另一个恢复入口：**Recovery 模式与官方恢复功能详解**——卡刷与恢复出厂的现场。
