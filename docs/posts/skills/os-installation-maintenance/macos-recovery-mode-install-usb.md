---
title: macOS 恢复模式与安装U盘制作
date: 2026-08-07
---

# macOS 恢复模式与安装U盘制作

<div class="epigraph">
<p>磨刀不误砍柴工。</p>
<footer>——中国谚语</footer>
</div>

<div class="article-byline">
<p>技能 · 操作系统安装与日常维护 ｜ macOS 支持文档（Apple Support）「创建可引导安装器」 ｜ 2026-08-07</p>
</div>

## 为什么从 macOS 恢复模式开始

Windows 和 Linux 装系统都靠「外部启动 U 盘」，macOS 却有一条更省事的路径——**恢复模式（Recovery Mode）**：按住组合键开机，Mac 会直接进入一个隐藏的恢复分区，从那里就能重装系统、修复磁盘、重置密码，全程不需要 U 盘。这是 Apple「封闭生态」的体现：系统、恢复工具、固件升级都自己包圆。

但恢复模式依赖网络或本地恢复分区，装全新机器、做批量部署时，一个**可引导的 macOS 安装 U 盘**仍不可替代。本节讲清楚恢复模式是什么、什么时候用，以及如何用官方 `createinstallmedia` 命令做安装 U 盘——它对应 Windows 的「媒体创建工具」。

## 1 恢复模式：Mac 的「隐身后备车间」

按住指定组合键开机，会进入恢复模式，它实际上是一个独立的恢复环境（RecoveryOS），包含「磁盘工具」「重新安装 macOS」「系统迁移助理」「终端」等工具。常用组合键：

| 组合键 | 作用 |
| --- | --- |
| `Command-R` | 进入本地恢复模式（优先） |
| `Option-Command-R` | 网络恢复，下载最新兼容版本 |
| `Shift-Command-R` | 启动磁盘自带的原始 macOS 版本 |
| `Option`（启动时按住） | 进入启动管理器，选择启动磁盘/启动 U 盘 |

<span class="marginnote">`Option-Command-R` 会从 Apple 服务器下载恢复映像，只要网络够快，即使本机恢复分区损坏也能救回系统——这是恢复模式的「最后保险」。Apple Silicon 芯片的 Mac 稍有不同：开机后<strong>按住电源键</strong>直到「正在载入启动选项」出现。</span>

## 2 createinstallmedia：官方制作安装 U 盘

要制作可引导安装器，Apple 官方支持文档给的标准做法是用 `createinstallmedia` 命令。先决条件：

- 一只 **≥16 GB** 的 U 盘（建议 USB 3.0），会**被清空**。
- 从 App Store 下载对应版本的「安装 macOS xxx.app」应用（放在 `/Applications/` 下）。
- 磁盘为 **APFS 或 macOS 扩展（日志式）**，命名为「MyVolume」（官方文档以此为例）。

在「终端」里执行：

```
sudo /Applications/Install\ macOS\ Sonoma.app/Contents/Resources/createinstallmedia --volume /Volumes/MyVolume --nointeraction
```

`--volume` 指定目标 U 盘，`--nointeraction` 跳过交互确认。命令会先抹掉 U 盘，再写入引导文件与安装器，耗时通常 20–40 分钟。<span class="marginnote">命令里的路径带空格，所以写成 `Install\ macOS`（反斜杠转义）。如果你下的是 Ventura，就把 `Sonoma` 换成 `Ventura`。`createinstallmedia` 是 Apple 提供的命令行工具，功能等价于图形化的「制作安装器」流程。</span>

## 3 启动安装 U 盘：T2 芯片与 Apple Silicon 的差异

做好的 U 盘要能启动，还得看 Mac 的硬件世代，这里是最容易困惑的部分：

- **Intel Mac 且无 T2**：开机按住 `Option` 进入启动管理器，选择黄色图标的安装 U 盘即可。
- **T2 安全芯片的 Intel Mac**：除 `Option` 外，还受「启动安全性实用工具」约束——外部启动需要在恢复模式里把启动安全策略调低。
- **Apple Silicon Mac**：按住电源键进「启动选项」，插上 U 盘后通常能直接识别；但签名校验更严，系统与固件版本不匹配可能拒绝启动。<span class="marginnote">T2 芯片把固件密钥、Secure Boot 的类机制都收进了专用芯片，是 Apple 版「TPM + Secure Boot」。它既是安全增强，也是修机门槛：外部介质启动不再是「插上就能用」。</span>

## 4 核心对比表：恢复模式 vs 安装 U 盘 vs 网络恢复

| 对比项 | 恢复模式（本地） | 安装 U 盘 | 网络恢复 |
| --- | --- | --- | --- |
| 需要介质 | 否（用本机恢复分区） | 是（≥16 GB U 盘） | 否（依赖网络） |
| 安装版本 | 本机自带/最新 | 你指定的版本 | 最新兼容版本 |
| 适用场景 | 日常重装、修复 | 全新机、批量部署、指定版本 | 本机恢复分区损坏 |
| 依赖 | 恢复分区完好 | 制作时用的 .app | 网络连通 |

日常维护用恢复模式最省事；做安装 U 盘更像「准备一份标准工具」，一次做好，随处可用。

## 5 动手：制作并验证安装 U 盘

**第一步：下载安装器。** 在 App Store 搜索「macOS Sonoma」并下载，得到 `/Applications/Install macOS Sonoma.app`。

**第二步：格式化 U 盘。** 用「磁盘工具」把 U 盘抹成「Mac OS 扩展（日志式）」，名称设为 `MyVolume`。

**第三步：跑命令。** 打开「终端」，执行上面的 `createinstallmedia` 命令，输入管理员密码，等待进度完成。

**第四步：验证。** 终端显示 `Install media now ready` 表示成功。重启，`Option` 进启动管理器，能看到「Install macOS Sonoma」黄色图标即验证通过。

## 6 易错点：macOS 介质制作的三个坑

- **U 盘命名带空格或中文**：`--volume /Volumes/My` 这种写法会因空格被终端拆成两个参数而报错，统一用简短英文名最稳。
- **下载的是「安装程序」还是「系统镜像」**：`createinstallmedia` 只认 `/Applications` 下的「安装 macOS xxx.app」，从第三方下的 `.dmg` 系统镜像没法直接用它制作。
- **以为恢复模式必须联网**：本地恢复（`Command-R`）用的是本机隐藏分区，不联网也能进；网络恢复才需要网络。两者混用是常见误区。

## 7 补充速查：macOS 启动组合键与安装器要点

把 macOS 的「组合键全家」与安装器关键点汇总：

| 组合键/命令 | 作用 |
| --- | --- |
| `Command-R` | 本地恢复模式 |
| `Option-Command-R` | 网络恢复（最新版） |
| `Shift-Command-R` | 出厂自带 macOS 版本 |
| `Option` | 启动管理器，选启动盘 |
| Apple Silicon 电源键 | 进启动选项 |
| `createinstallmedia` | 制作安装 U 盘命令 |
| `--volume /Volumes/MyVolume` | 指定目标 U 盘 |
| `--nointeraction` | 跳过交互确认 |
| U 盘要求 | ≥16 GB、USB 3.0、会被清空 |
| 安装器来源 | `/Applications` 下的 .app |
| U 盘命名 | 简短英文名，别带空格 |
| 制作耗时 | 20–40 分钟 |

## 8 补充速查：三种恢复路径一句话分清

| 恢复路径 | 一句话 |
| --- | --- |
| 本地恢复（`Command-R`） | 用本机恢复分区，最快 |
| 网络恢复（`Option-Command-R`） | 联网下载恢复映像，最稳 |
| 安装 U 盘 | 可指定版本，适合全新机与批量部署 |

选择建议：日常维护先用本地恢复；本机恢复分区损坏用网络恢复；要指定版本或批量装机，做安装 U 盘。

## 9 小结

- **恢复模式**是 Mac 的隐身后备车间：`Command-R` 本地恢复、`Option-Command-R` 网络恢复、`Shift-Command-R` 出厂版本。
- 官方制作安装 U 盘的命令是 **`createinstallmedia --volume /Volumes/MyVolume --nointeraction`**，U 盘需 **≥16 GB**。
- 启动 U 盘受硬件约束：**T2 芯片**需放宽外部启动安全策略，**Apple Silicon** 按电源键进启动选项。
- 日常重装优先用恢复模式；**指定版本、全新机、批量部署**才需要安装 U 盘。
- U 盘用简短英文名；安装器必须来自 `/Applications` 下的 .app。
- 三平台介质至此齐了：Windows 用媒体创建工具，Linux 用 `dd`，macOS 用 `createinstallmedia`。

在下一节，介质备齐、准备完毕，我们正式进入第2篇——从 **Windows 全新安装与磁盘分区选择** 开始，亲手把系统装进磁盘。
