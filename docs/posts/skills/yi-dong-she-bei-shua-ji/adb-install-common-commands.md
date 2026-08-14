---
title: ADB 工具安装与常用调试命令
date: 2026-08-07
---

# ADB 工具安装与常用调试命令

<div class="epigraph">
<p>ADB 是一把瑞士军刀——传文件、进模式、装应用、读日志，刷机的每一步都从它开始。</p>
<footer>—— Android 开发工具手册（Recovery 与固件刷写教程）</footer>
</div>

<div class="article-byline">
<p>生活技能树 · 移动设备系统定制与刷机 ｜ Recovery 与固件刷写教程 ｜ 2026-08-07</p>
</div>

## 为什么刷机从 ADB 开始

ADB（Android Debug Bridge，安卓调试桥）是电脑与手机之间的**官方调试通道**。刷机的三个常用入口——**Fastboot 模式、Recovery 模式、EDL/深刷模式**——绝大多数都要先通过 ADB 让手机「跳」进去；传刷机包、备份数据、读错误日志也全靠它。可以说 ADB 是刷机者每天接触最多的工具。这一篇把它的安装、连接与常用命令讲透，你之后的每一次刷机都会用到这里的命令。而 ADB 的「客户端-服务器-守护进程」结构，本质上是一个典型的分布式调试模型，可与第三级《操作系统》与《计算机网络》中的进程通信概念相互印证。

## 1 ADB 是什么：电脑与手机的调试桥梁

**ADB（Android Debug Bridge）**：Google 官方提供的命令行工具，采用 **C/S（客户端/服务器）架构**，包含三个部分：运行在电脑上的**客户端**（你敲命令的地方）、后台的 **ADB 服务器**（管理连接）、运行在手机上的 **ADB 守护进程 adbd**（执行命令）。<span class="marginnote">理解 ADB 的三段式结构，能帮你排掉很多「连不上」的错：<strong>电脑客户端 → 电脑本地 5037 端口上的 ADB 服务器 → 手机上的 adbd</strong>。命令敲下后，客户端把请求交给服务器，服务器再转发给手机。三个环节任一处断（服务器没起、端口占用、手机 adbd 没开），命令就失败。</span>

ADB 能做什么，决定了它为什么是刷机基石：

**传文件**：`push`/`pull` 在电脑与手机之间搬文件。
**执行命令**：`shell` 进入手机 Linux 命令行，读写文件、改设置。
**控制设备**：`reboot` 让手机重启进各种模式。
- **装应用**：`install` 直接安装 APK。
- **读日志**：`logcat` 抓系统日志，排查问题。
- **卡刷**：`sideload` 把系统包从电脑直接喂给 Recovery。

一句话：**ADB 是刷机操作里「电脑控制手机」的万能通道**。

## 2 安装与连接：从装工具到「设备授权」

**安装 ADB** 分三步：

**第一步，获取 platform-tools**。Google 官方提供 **platform-tools** 压缩包（含 `adb`、`fastboot` 等命令），解压即用，无需安装。Windows/macOS/Linux 都有对应版本。<span class="marginnote">为什么不推荐从非官方渠道下载「整合版刷机包」？<strong>因为 platform-tools 本身是开源的官方工具，但被第三方二次打包后可能夹带私货</strong>。官方途径是 Android 开发者网站下载压缩包，或通过包管理器（macOS 的 Homebrew 装 `android-platform-tools`）安装。</span>

**第二步，配置环境变量（可选但推荐）**。把解压目录加入 PATH，这样在任何路径都能直接敲 `adb`。Windows 是「系统属性 → 环境变量」，macOS/Linux 是往 `~/.zshrc` 里加 `export PATH=...`。

**第三步，装好 USB 驱动（Windows 重点）**。Windows 需要厂商 USB 驱动才能识别手机；macOS/Linux 一般免驱。

**连接手机**，需要手机端配合：

- 打开「开发者选项」（连续点击版本号 7 次）。
- 开启 **USB 调试**。
- 数据线连接电脑，手机上弹出「允许 USB 调试吗」，勾选「一律允许」并确认。
- 电脑上敲 `adb devices`，能看到设备即连接成功。

**连接成功的标志**：`adb devices` 列出设备并标注 `device` 状态。若显示 `unauthorized`，说明手机端没点允许；显示 `offline`，多半是驱动或线材问题。这些状态词是刷机排错的第一线索。

## 3 常用命令分组：设备、文件、安装、重启

把高频命令按用途分组记忆，效率最高：

**设备与连接**：
- `adb devices`：列出已连接设备与状态。
- `adb kill-server` / `adb start-server`：重启 ADB 服务器（连接异常时的第一招）。
- `adb pair <ip>:<port>`：无线调试配对，之后 `adb connect <ip>:<port>` 即可无线连接。

**文件传输**：
- `adb push <本地> <手机路径>`：把文件从电脑传到手机。
- `adb pull <手机路径> <本地>`：把文件从手机拉到电脑。
- `adb install <apk>`：安装应用；加 `-r` 覆盖安装、`-d` 允许降级。

**Shell 与权限**：
- `adb shell`：进入手机命令行。
- `adb shell <命令>`：直接执行单条命令，如 `adb shell getprop ro.build.version.release` 查看系统版本。
- `adb root`：重启 adbd 为 root 权限（需系统支持，Root 后常用）。
- `adb remount`：把系统分区改为可写（Root 后修改系统文件常用）。

**重启与模式切换**：
- `adb reboot`：正常重启。
- `adb reboot bootloader`：重启进 **Fastboot 模式**（刷机线刷入口）。
- `adb reboot recovery`：重启进 **Recovery 模式**（卡刷入口）。
- `adb reboot edl`：重启进高通 **EDL/9008 深刷模式**（救砖入口，部分机型支持）。

**系统与卡刷**：
- `adb sideload <卡刷包.zip>`：在 Recovery 里从电脑直接刷入卡刷包。
- `adb logcat`：实时抓取系统日志。
- `adb backup` / `adb restore`：备份/恢复应用数据（部分 Android 版本）。

这几个分组覆盖了刷机的日常：**连设备、传文件、进模式、装应用、刷系统**。

## 4 公式解析：adb 命令的通用语法

所有 ADB 命令都遵循同一套语法，读懂了它，任何陌生命令都能自己推导：

$$
\text{adb} \quad \underbrace{\text{[<全局选项>]}}_{\text{如 -s 指定设备}} \quad \underbrace{\text{<动作>}}_{\text{如 push / shell / reboot}} \quad \underbrace{\text{[<参数>]}}_{\text{如路径、文件名}}
$$

逐步拆解：

- **`adb`**：固定前缀，调用 ADB 客户端。
- **`[<全局选项>]`**：影响「对哪台设备操作」。多设备时用 `-s <序列号>` 指定，例如 `adb -s emulator-5554 shell`。不带则默认唯一设备，多设备时会报「多个设备」错误。
- **`<动作>`**：要执行的命令动词——`devices`、`push`、`pull`、`install`、`shell`、`reboot`、`sideload` 等。
- **`[<参数>]`**：动作的补充，路径、文件名、开关（`-r`、`-d`）。

一个实例：`adb -s 0123456789ABCDEF push rom.zip /sdcard/`——「对序列号 0123456789ABCDEF 的设备，把电脑上的 rom.zip 传到手机存储根目录」。**套上这个语法框，读任何 ADB 教程里的命令都不再费劲**。

## 5 核心要点：高频 ADB 命令速查表

| 场景 | 命令 | 备注 |
| --- | --- | --- |
| 查看设备 | `adb devices` | 状态含 device/unauthorized/offline |
| 传文件到手机 | `adb push <文件> <手机路径>` | 刷机包常用 |
| 从手机拉文件 | `adb pull <手机路径> <本地>` | 备份导出 |
| 进入 Fastboot | `adb reboot bootloader` | 线刷入口 |
| 进入 Recovery | `adb reboot recovery` | 卡刷入口 |
| 进入 EDL | `adb reboot edl` | 救砖入口 |
| 卡刷系统包 | `adb sideload <包.zip>` | 需在 Recovery 内 |
| 安装 APK | `adb install <apk>` | `-r` 覆盖 |
| 查看版本 | `adb shell getprop ro.build.version.release` | 确认系统版本 |
| 抓日志 | `adb logcat` | 排错利器 |
| 重启服务器 | `adb kill-server && adb start-server` | 连接异常第一招 |

## 6 术语速查表

| 术语 | 含义 | 关键点 |
| --- | --- | --- |
| ADB | 安卓调试桥 | 电脑↔手机通道 |
| adbd | 手机端调试守护进程 | USB 调试开关控制 |
| platform-tools | 官方工具包 | 含 adb/fastboot |
| USB 调试 | 开发者选项开关 | 连接前提 |
| unauthorized | 未授权状态 | 手机端没点允许 |
| offline | 离线状态 | 驱动/线材问题 |
| push/pull | 双向文件传输 | 传刷机包/备份 |
| sideload | Recovery 内直刷 | 卡刷的一种 |
| reboot bootloader | 进 Fastboot | 线刷入口 |
| EDL | 高通深刷模式 | 救砖入口 |

## 7 快速自查清单

遇到 ADB 问题，按这个顺序排错：

- `adb devices` 是否**列出了设备且状态为 `device`**？
- 状态是 `unauthorized`？→ 手机端**重新点允许**。
- 状态是 `offline`？→ 换**原装数据线**、重装驱动、重启 `adb kill-server`。
- 设备列表为空？→ 确认**已开 USB 调试**、驱动是否装好。
- 命令报「multiple devices」？→ 用 `-s <序列号>` 指定设备。
- 无线调试连不上？→ 用 `adb pair` 重新配对。

## 8 小结

- ADB 是**电脑↔手机调试通道**，采用客户端/服务器/adbd 三段式结构。
- 连接三部曲：**开 USB 调试 → 连数据线 → 手机上点允许**，`adb devices` 验证。
- 命令按用途分组：**设备、文件、Shell、重启进模式、系统卡刷**，进 Fastboot/Recovery/EDL 都靠 `adb reboot`。
- 所有命令套统一语法：`adb [全局选项] <动作> [参数]`，读陌生命令即套此框架。
- 排错顺序：**设备状态 → 授权 → 线材/驱动 → 服务器重启**。

在下一节，我们进入线刷的主战场：**Fastboot 模式与线刷命令集**——用 `fastboot flash` 直接写分区的完整语法。
