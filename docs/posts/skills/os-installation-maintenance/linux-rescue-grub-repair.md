---
title: Linux 救援模式与 GRUB 引导修复
date: 2026-08-07
---

# Linux 救援模式与 GRUB 引导修复

<div class="epigraph">
<p>山重水复疑无路，柳暗花明又一村。</p>
<footer>——陆游《游山西村》</footer>
</div>

<div class="article-byline">
<p>技能 · 操作系统安装与日常维护 ｜ 《鸟哥的Linux私房菜：基础学习篇》（开机流程、模块管理与 Loader） ｜ 2026-08-07</p>
</div>

## 为什么从救援模式开始

Linux 用户最常遇到的「死亡时刻」：开机看到 `GRUB> ` 提示符、或直接黑屏、或 `error: no such partition`——**引导坏了**。好消息是 Linux 引导故障几乎都有解，因为 Linux 世界有一条通用路径：**用启动 U 盘（Live 环境）进场 → chroot 进入故障系统 → 重建引导**。坏消息是这条路径命令多、概念多，新手容易在 `chroot` 前就卡住。

本节把这条路径拆开讲：引导失败怎么分类、GRUB 的引导链是什么、救援模式怎么进、`chroot` 为什么是「钥匙」，最后给出一条可照抄的修复流程。它承接鸟哥《基础学习篇》「开机流程、模块管理与 Loader」章节。

## 1 引导失败：先看清死在哪

Linux 开机引导链：**固件（UEFI/BIOS）→ GRUB → 内核 → initramfs → 根文件系统 → 服务**。失败点不同，症状就不同：

| 症状 | 故障点 | 说明 |
| --- | --- | --- |
| 黑屏只显示 `GRUB> ` | GRUB 配置文件损坏 | 需要重建 GRUB 配置 |
| `error: no such partition` | 分区表/磁盘变化 | GRUB 找不到分区 |
| 能见菜单但选 Linux 就重启 | 内核/initramfs 问题 | 重装内核或 initramfs |
| 卡在 `Waiting for device` | 根文件系统挂载失败 | 检查 `fstab` 与根分区 |

<span class="marginnote">最经典的引导事故：重装 Windows 后 GRUB 被覆盖（双系统场景，见《Windows 与 Linux 双系统安装与引导》）；或调整分区后 `/etc/fstab` 里的 UUID 对不上，根文件系统挂不上。记住「引导坏 = 重建 GRUB，根挂不上 = 修 fstab」。</span>

## 2 认识引导链：修复前先懂结构

修复 GRUB 前，先理解两个关键概念：

- **GRUB 本体**：`/boot/grub/grub.cfg` 是菜单配置，`/boot/grub` 里的模块文件是 GRUB 的零件。若磁盘未变只是配置坏了，`update-grub` 重生成配置即可。
- **GRUB 安装位置**：UEFI 模式下 GRUB 的 `.efi` 文件在 ESP 分区（`/boot/efi/EFI/grub/grubx64.efi`），固件从那里执行它。若这个文件被覆盖/删除，要 `grub-install` 重新写入。

两条核心命令：

```
sudo update-grub                 # 重新生成 grub.cfg（Debian/Ubuntu 系）
sudo grub2-mkconfig -o /boot/grub2/grub.cfg   # Red Hat 系
```

**重点：修复 GRUB 的两件事——「重装 GRUB 本体（grub-install）」与「重建菜单（update-grub）」。** 大多数「菜单坏了」只需第二件；「GRUB 完全没了」才需要第一件。

## 3 救援模式：用 Live 环境进场

「救援模式」在 Linux 语境里通常指用**启动 U 盘进入 Live 桌面环境**（就是当初装系统用的那个 U 盘）。Live 环境是一个「临时 Linux」，它独立于故障系统运行，让你能访问故障系统的磁盘：

**第一步：用启动 U 盘启动。** 插入 Linux 安装/启动 U 盘，进启动菜单选 U 盘。

**第二步：选「Try Ubuntu」（试用）**。不安装，直接进入 Live 桌面。

**第三步：打开终端。** 此时你在一个「洁净的外部世界」，可以挂载并修复故障系统。<span class="marginnote">鸟哥书里的「rescue」参数、以及 systemd 的 `systemd.unit=rescue.target`（救援目标）是另一条路：从 GRUB 菜单给内核加参数直接进单用户/救援模式，常用于「能进 GRUB 但系统起不来」。两者都叫救援模式，场景不同：Live U 盘修引导，内核参数救援修系统配置。</span>

## 4 chroot：进入故障系统的钥匙

**`chroot`（change root）** 让你把「根目录」临时切换到故障系统的磁盘，仿佛「灵魂出窍进了故障系统的身体」——之后执行的命令都以故障系统为根。它是救援的核心：

```
sudo mount /dev/sda2 /mnt            # 挂载故障系统的根分区（按实际情况）
sudo mount /dev/sda1 /mnt/boot/efi   # 挂载 ESP（UEFI 需要）
sudo mount --bind /dev /mnt/dev && sudo mount --bind /proc /mnt/proc && sudo mount --bind /sys /mnt/sys
sudo chroot /mnt                     # 切换根
```

进入 `chroot` 后，你就「住进了」故障系统：

```
update-grub
grub-install /dev/sda
```

`grub-install /dev/sda` 把 GRUB 写到磁盘的引导位置（或 UEFI 的 ESP），`update-grub` 重建菜单——这两条是修复引导的「王炸组合」。<span class="marginnote">`--bind` 挂载 `/dev`、`/proc`、`/sys` 是因为 `chroot` 后没有这些虚拟文件系统，很多命令（如 `grub-install` 探测硬件）会失败。这是新手最容易漏的一步，漏了就会出现「chroot 进去了但命令报错」。</span>

## 5 核心对比表：常见引导故障与解药

| 故障 | 症状 | 解药 |
| --- | --- | --- |
| GRUB 菜单坏了 | 菜单不全/错误 | `update-grub` |
| GRUB 丢失 | `GRUB> ` 或黑屏 | `grub-install` + `update-grub` |
| 双系统 Windows 覆盖 GRUB | 直接进 Windows | 重装 GRUB |
| 根挂不上 | 卡在开机 | 修 `/etc/fstab` 的 UUID |
| 内核损坏 | 选内核就重启 | 重装内核包 |

## 6 动手：一次完整的 GRUB 修复

**第一步：U 盘启动进 Live。** 选「Try Ubuntu」。

**第二步：挂载根分区。** `lsblk` 确认分区号，`sudo mount /dev/sda2 /mnt`。

**第三步：挂载 ESP。** `sudo mount /dev/sda1 /mnt/boot/efi`。

**第四步：挂虚拟文件系统。** 三条 `mount --bind` 命令。

**第五步：chroot 进场。** `sudo chroot /mnt`。

**第六步：重建引导。** `grub-install /dev/sda` + `update-grub`。

**第七步：退出重启。** `exit` 退出 chroot，`reboot` 重启，拔掉 U 盘，看到 GRUB 菜单即成功。

## 7 速查表：救援模式命令速记

| 命令 | 作用 |
| --- | --- |
| `lsblk` | 查看磁盘与分区 |
| `mount /dev/sda2 /mnt` | 挂载根分区 |
| `mount /dev/sda1 /mnt/boot/efi` | 挂载 ESP |
| `chroot /mnt` | 进入故障系统 |
| `grub-install /dev/sda` | 重装 GRUB |
| `update-grub` | 重建 GRUB 菜单 |
| `exit` / `reboot` | 退出 / 重启 |

## 8 小结

- 引导链：**固件 → GRUB → 内核 → 根文件系统 → 服务**，不同失败点症状不同。
- 修复 GRUB 两件事：**`grub-install`（重装本体）+ `update-grub`（重建菜单）**。
- 救援路径：**Live U 盘进场 → 挂载 → chroot → 重建引导**。
- **chroot** 是进入故障系统的钥匙，别忘了 `--bind` 挂载 `/dev`、`/proc`、`/sys`。
- 双系统被 Windows 覆盖引导 → 重装 GRUB；根挂不上 → 修 `/etc/fstab`。
- Linux 引导故障几乎都有解，别急着重装系统。

在下一节，我们处理 Apple 世界的开机故障——**macOS 启动故障与磁盘工具急救**。
