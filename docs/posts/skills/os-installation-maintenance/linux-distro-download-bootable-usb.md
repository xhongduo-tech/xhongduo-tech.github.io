---
title: Linux 发行版镜像下载与启动盘制作
date: 2026-08-07
---

# Linux 发行版镜像下载与启动盘制作

<div class="epigraph">
<p>授人以鱼，不如授人以渔。</p>
<footer>——中国谚语</footer>
</div>

<div class="article-byline">
<p>技能 · 操作系统安装与日常维护 ｜ 《鸟哥的Linux私房菜：基础学习篇》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从发行版镜像开始

Windows 的介质制作是「官方工具一条龙」，Linux 则相反——它把每一环节都摊开给你看：先选**发行版（distribution）**，再手动**校验哈希**，最后用 `dd` 之类的命令**整盘写入**。过程「麻烦」，但每步都透明、可复现。理解这套流程，你就真正理解了启动介质是怎么工作的，而不是只会点「下一步」。

鸟哥在第3章安装 CentOS 时走的正是这条手动路径。本节以 Ubuntu 为例，顺带讲 Fedora、CentOS 系（Rocky Linux）的差异，把「下载 → 校验 → 写盘」三步打通。

## 1 选发行版：先决定你要哪个「Linux」

「Linux」指内核；「发行版」是内核 + 软件包 + 安装器 + 桌面的组合。常见三类：

- **Ubuntu 系**（Debian 家族）：安装器图形化最友好，驱动支持全面，社区资料最多，新手首选。桌面环境默认 GNOME。
- **Fedora / CentOS / Rocky 系**（Red Hat 家族）：包管理用 `dnf`，企业环境标准。CentOS 已转为流式发布，生产环境多用 Rocky Linux 或 AlmaLinux 接替。
- **Debian**：以稳定著称，是 Ubuntu 的底座，适合追求极致稳定的人。<span class="marginnote">鸟哥的书全程基于 CentOS/Rocky 讲解，但「下载镜像 → 校验 → 写盘 → 安装」这套动作在 Ubuntu 上完全一样，只是包管理从 `yum/dnf` 换成 `apt`。学方法，不必拘泥发行版。</span>

对新手，Ubuntu 桌面版（Desktop ISO）最省心；它同时提供 Server 版（无图形界面，用于服务器）。选发行版时还要考虑一件事：**你想用它的软件生态**——Ubuntu 系 `apt` 源最全，Red Hat 系 `dnf` 更偏企业运维，两者命令不同，别混着记。

## 2 下载镜像：认准官网与哈希

镜像只能从**发行版官网或官方镜像站**下载，别用第三方「XX 系统之家」——那类站点篡改镜像植入后门的事时有发生。Ubuntu 官方下载页会列出多个镜像站（mirror），自动按地域选快的。

下载后**必须校验完整性**：官网在每个 `.iso` 旁公布一串 64 位十六进制数，即 **SHA-256 哈希**。计算你下载文件的哈希，与官网公布值比对，一致才说明文件没在传输中损坏或被篡改：

```
sha256sum ubuntu-24.04.1-desktop-amd64.iso
# 输出：3fce4a1d…9e51e2ab  ubuntu-24.04.1-desktop-amd64.iso
```

把输出与官网的 64 位值逐字比对（或 `grep` 匹配），这就是「下载 → 校验」的标准闭环。<span class="marginnote">SHA-256 是安全哈希函数：任意长度输入都得到固定 256 位（32 字节，即 64 个十六进制字符）输出，输入差一个比特，输出就完全不同。它把「验证文件没坏」变成「比对两个 64 位字符串」，成本极低。</span>

**辨析｜易错点：** 别跳过校验。镜像下载中断、镜像站文件不完整、硬盘坏道，都可能让 ISO「看起来能打开但装到一半报错」。花十秒做一次 `sha256sum`，能省下一整晚的排查。

## 3 写盘：dd 与图形工具

拿到经过校验的 ISO，就要「整盘写入」U 盘。Linux 下最硬核的方式是 `dd` 命令：

```
sudo dd if=ubuntu-24.04.1-desktop-amd64.iso of=/dev/sdb bs=4M status=progress
```

其中 `if=` 是输入文件（镜像），`of=` 是目标设备（整块 U 盘，**不是**分区 `sdb1`），`bs=4M` 是每次读写块大小，`status=progress` 显示进度。<span class="marginnote">`of=` 必须写整盘 `/dev/sdb` 而非 `/dev/sdb1`——写入目标是「设备的引导结构 + 数据区」，分区只是它内部的一部分。写错盘符会摧毁目标磁盘全部数据，操作前用 `lsblk` 确认设备号。</span>

不想敲命令，可用图形工具：

| 工具 | 平台 | 特点 |
| --- | --- | --- |
| Rufus | Windows | 上一节介绍过，也支持写 Linux ISO |
| balenaEtcher | Win/mac/Linux | 界面最友好，双击即用 |
| Ventoy | 跨平台 | 装好后把 ISO 拖进 U 盘即可启动 |
| `dd` | Linux 自带 | 最透明，一条命令搞定 |

写完后 `sync` 确保缓存落盘，再弹出 U 盘。

## 4 公式解析：写盘时间怎么估算

写盘耗时可按下式估算，它是「镜像容量 ÷ 写入速度」：

$$
t = \frac{C}{v} = \frac{8\,\text{GB}}{100\,\text{MB/s}} = \frac{8192\,\text{MB}}{100\,\text{MB/s}} \approx 82\,\text{s}
$$

分三步拆解：

- **第一步，统一单位**：镜像 8 GB，USB 3.0 的典型持续写入约 100 MB/s。先换算：8 GB = 8192 MB。
- **第二步，相除**：时间 = 容量 ÷ 速度 = 8192 ÷ 100 ≈ 82 秒。
- **第三步，理解差异**：实际会比这慢——U 盘主控写缓存耗尽后掉速、`dd` 逐块复制、`status=progress` 会显示实时速率。若速度只有 20 MB/s（劣质 U 盘），同样 8 GB 要近 7 分钟。这就是为什么 USB 3.0 大容量 U 盘值得投资。

这条公式对 Windows/macOS 介质同样成立，它是「等待要等多久」的第一个直觉来源。

## 5 动手：Ubuntu 启动盘完整流程

把上面串成一条路径：

**第一步：下载。** 官网下载 Ubuntu 桌面版 ISO（约 5–6 GB）。

**第二步：校验。** `sha256sum` 比对官网哈希，不一致就重新下载。

**第三步：写盘。** 用 `dd`（Linux）或 balenaEtcher/Rufus（Windows）写入 ≥8 GB 的 U 盘。

**第四步：启动。** 重启进启动菜单，选 U 盘。Ubuntu 的 GRUB 菜单出现即成功；选「Try or Install Ubuntu」可先体验后安装。

**第五步：装完取出 U 盘。** 安装完成后重启，别让 U 盘继续作为第一启动项，否则每次开机都进安装器。

## 6 发行版速查：三大家族一句话分清

| 家族 | 包管理 | 安装器 | 典型发行版 | 适合谁 |
| --- | --- | --- | --- | --- |
| Debian 系 | `apt` | Ubiquity（图形化） | Ubuntu、Debian、Linux Mint | 新手、桌面 |
| Red Hat 系 | `dnf`/`yum` | Anaconda（图形化） | Fedora、Rocky、AlmaLinux | 企业、服务器 |
| Arch 系 | `pacman` | 无图形安装器 | Arch、Manjaro | 进阶、爱折腾 |

## 7 补充速查：镜像下载与写盘速记

把下载校验与写盘要点收敛成速查表：

| 环节 | 动作 | 命令/工具 |
| --- | --- | --- |
| 选发行版 | 官网下载 ISO | Ubuntu 桌面版为例 |
| 校验哈希 | 比对官网 SHA-256 | `sha256sum 镜像.iso` |
| 图形写盘 | Etcher/Rufus | 跨平台 |
| 命令写盘 | `dd` 整盘写入 | `sudo dd if=... of=/dev/sdX bs=4M` |
| 确认设备 | `lsblk` | 别写错盘 |
| 落盘同步 | `sync` | 写完必做 |
| 启动测试 | 启动菜单选 U 盘 | 见 GRUB 菜单 |
| 装完收尾 | 弹出 U 盘 | 改回硬盘启动 |

## 8 小结

- 发行版分 **Ubuntu/Debian 系（apt）**、**Red Hat 系（dnf/yum）**、**Arch 系（pacman）** 三大阵营，新手从 Ubuntu 桌面版起步最顺。
- 镜像**只从官网或官方镜像站下载**，下载后必做 **SHA-256 校验**（`sha256sum`）。
- `dd` 整盘写入：`sudo dd if=镜像 of=/dev/sdX bs=4M status=progress`，`of=` 写整盘而非分区。
- 图形工具按需选：Etcher 最友好、Rufus 跨平台、Ventoy 一 U 盘多系统。
- 写盘时间 ≈ 容量 ÷ 速度；USB 3.0 大容量盘能省下大量等待时间。

在下一节，我们切换到 Apple 世界——**macOS 恢复模式与安装U盘制作**，看看 macOS 的介质制作与 Windows/Linux 有什么同与不同。
