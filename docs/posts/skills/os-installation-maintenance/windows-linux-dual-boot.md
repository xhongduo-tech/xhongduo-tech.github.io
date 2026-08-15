---
title: Windows 与 Linux 双系统安装与引导
date: 2026-08-07
---

# Windows 与 Linux 双系统安装与引导

<div class="epigraph">
<p>一山二虎，各据一方。</p>
<footer>——改写自中国谚语</footer>
</div>

<div class="article-byline">
<p>技能 · 操作系统安装与日常维护 ｜ 《鸟哥的Linux私房菜：基础学习篇》第3章 · Microsoft Windows 官方支持文档 ｜ 2026-08-07</p>
</div>

## 为什么从双系统开始

很多人既离不开 Windows（办公、游戏、某些专业软件），又想体验 Linux（开发、学习、隐私）。**双系统（dual boot）**就是在同一台电脑上装两个系统，开机时用引导菜单选择进哪个。它不需要虚拟机那样分内存，性能是满血的，是「两个世界兼得」的主流方案。

但双系统也是本专题最容易翻车的场景——两大引导系统（Windows Boot Manager 与 GRUB）在同一块磁盘上争夺控制权，时间设置、分区顺序、安全启动稍有差错，就会「进不了其中一个系统」。鸟哥在第3章用「多重引导」专节讲了这套规则，本节把它讲透：顺序怎么安排、引导谁说了算、坑在哪。

## 1 双系统的铁律：先 Windows，后 Linux

双系统第一条铁律是**安装顺序不能乱**：必须先装 Windows，再装 Linux。原因很简单——**Windows 的引导器（Boot Manager）不认识也不尊重其他系统的引导记录**，安装 Windows 时它会默认接管整块磁盘的引导；而后装 Linux 时，GRUB 会主动检测到已有的 Windows 并把它加入菜单。<span class="marginnote">反过来的后果很经典：如果先装 Linux 再装 Windows，Windows 安装器会直接覆盖掉 GRUB，Linux 从此「隐身」；修复它需要进 Linux 救援模式重装 GRUB，绕一大圈。先 Windows 后 Linux，就是为了避开这条弯路。</span>

**重点：先装 Windows，后装 Linux。** 这句话是本节的灵魂，也是「Windows 总是活得比 Linux 久」的哲学在引导层的体现。

## 2 引导权之争：谁掌管开机菜单

双系统装完后，开机时看到的那个「选系统」菜单，绝大多数情况下是 **GRUB**（Linux 的引导器），而不是 Windows 的。原因：GRUB 后安装，它会把自己写进 ESP 引导入口，并在自己的菜单里收录 Windows。<span class="marginnote">GRUB 全称 GNU GRand Unified Bootloader，是 Linux 世界的事实标准引导器。它的配置文件在 `/boot/grub/grub.cfg`，发行版提供 `update-grub`（Debian 系）或 `grub2-mkconfig`（Red Hat 系）自动重建菜单。</span>

引导流程：

```
固件（UEFI） → GRUB（在 ESP 里） → 选 Windows → Windows Boot Manager → Windows
                                 └→ 选 Linux → 内核 → Linux
```

当你的电脑开机直接进 Windows、看不到 GRUB 菜单时，多半是固件把引导顺序设成了「先 Windows Boot Manager」，或 `BootOrder` 里 GRUB 排在后面——进固件启动菜单手动选一次 GRUB 即可。

## 3 安装顺序与分区安排

完整流程可以这样拆：

| 步骤 | 操作 | 说明 |
| --- | --- | --- |
| 1 | 备份数据 | 双系统折腾磁盘，备份是底线 |
| 2 | 全新安装 Windows | 走上一节的「自定义」安装 |
| 3 | 压缩出 Linux 空间 | 磁盘管理里压缩 C: 卷，留出未分配空间 |
| 4 | 从 U 盘启动 Linux | 用《Linux 发行版镜像下载与启动盘制作》做的 U 盘 |
| 5 | 在未分配空间分区 | 让 Linux 安装器接管未分配区域 |
| 6 | 把 GRUB 装到 ESP | 安装器自动完成，收录 Windows 进菜单 |
| 7 | 完成重启 | 出现 GRUB 双系统菜单即成功 |

其中第 3 步「压缩卷」很关键：在 Windows 的「磁盘管理」里对 C: 右键 →「压缩卷」，输入要腾出的空间大小（如 100 GB）。压缩会得到「未分配空间」，Linux 就装在这里——两个系统各自占块地方，互不越界。<span class="marginnote">更稳妥的做法是给 Linux 单独一块物理硬盘（SSD），Windows 和 Linux 各占一块盘，互不干扰，任何一块盘坏了另一块照常启动。预算允许时优先考虑这个方案。</span>

## 4 核心对比表：Windows Boot Manager vs GRUB

| 对比项 | Windows Boot Manager | GRUB |
| --- | --- | --- |
| 所属 | 微软 | GNU 项目（开源） |
| 配置位置 | ESP 的 `\EFI\Microsoft\Boot\` | ESP 的 `\EFI\grub\` + `/boot/grub/` |
| 菜单重建 | 系统自动 | `update-grub` / `grub2-mkconfig` |
| 能否引导对方 | 不认识 Linux | 主动收录 Windows |
| 双系统首选 | 否 | 是（后装者接管） |

**判断要点：** 双系统的引导菜单交给 GRUB 管。Windows 更新偶尔会「抢回」引导权（重写 Boot Manager），此时只需进 Linux 重跑一次 `update-grub` 即可收复失地。

## 5 常见问题：时间偏差与菜单消失

双系统有两个经典坑，几乎每个人都会遇到一次：

**时间偏差 8 小时。** Windows 把硬件时钟（CMOS）当作本地时间，Linux 默认把硬件时钟当作 UTC，两者差了整 8 小时（UTC+8）。表现为「进 Windows 时间对，进 Linux 时间差 8 小时」或反之。修复二选一：

- 让 Linux 把硬件时钟当本地时间：`timedatectl set-local-rtc 1`。
- 让 Windows 用 UTC：注册表加 `RealTimeIsUniversal` 键（较麻烦，不推荐新手）。

<span class="marginnote">推荐第一种：Linux 执行 `timedatectl set-local-rtc 1` 一条命令，Windows 那边就正常了。这是双系统新手遇到的第一道「玄学题」，其实只是时区约定不同。</span>

**GRUB 菜单消失。** 装完 Windows 更新后开机直接进 Windows，或反过来。前者用 Linux 修复 `update-grub`；后者在 Windows 里用 `bcdedit` 重建。Windows 的修复命令是 `bcdedit /set {bootmgr} path \EFI\grub\grubx64.efi`——让固件把引导指向 GRUB。

## 6 动手：双系统安装的完整流程

**第一步：备份。** 整个磁盘按 3-2-1 备份。

**第二步：装 Windows。** 全新安装，确保 Windows 先独占磁盘。

**第三步：压缩卷。** 磁盘管理里压缩 C:，腾出 ≥100 GB 未分配空间。

**第四步：装 Linux。** U 盘启动 Linux 安装器，选择「与 Windows 共存」或手动在未分配空间分区。

**第五步：装 GRUB。** 安装器默认把 GRUB 写入 ESP 并收录 Windows，保持默认即可。

**第六步：验证双菜单。** 重启出现 GRUB 菜单，Windows 与 Linux 都能进入。

**第七步：修时间。** Linux 里执行 `timedatectl set-local-rtc 1`，解决 8 小时偏差。

## 7 易错点：双系统的三个坑

- **顺序反了**：先 Linux 后 Windows，GRUB 被覆盖，Linux 进不去。记住「先 Windows 后 Linux」。
- **Linux 装了但开机没菜单**：多半是固件 `BootOrder` 里 GRUB 被排到后面，进固件把 GRUB 设为第一启动项。
- **分区互相越界**：压缩出来的空间要「未分配」，别在 Windows 的磁盘管理里把它格式化成 NTFS，否则 Linux 安装器找不到干净的空间。

## 8 补充速查：双系统安装顺序速记

把双系统要点收敛成一张对照表：

| 要点 | 内容 |
| --- | --- |
| 安装顺序 | 先 Windows，后 Linux |
| 引导控制 | GRUB 后装接管菜单 |
| 分区安排 | Windows 先占，Linux 用未分配空间 |
| 腾空间 | 磁盘管理 → 压缩卷 |
| 时间偏差 | `timedatectl set-local-rtc 1` |
| 菜单消失 | `update-grub` 重建 |
| 双盘方案 | 各占一块盘最省心 |
| 安全启动 | Linux 引导受阻时检查 |

## 9 小结

- 双系统铁律：**先 Windows，后 Linux**，让 GRUB 后装接管引导菜单。
- 引导菜单交给 **GRUB** 管理，它会主动收录 Windows；Windows 更新可能抢回引导权，用 `update-grub` 收复。
- 分区安排：Windows 先装，磁盘管理「压缩卷」腾出未分配空间给 Linux。
- **时间偏差 8 小时**：Linux 执行 `timedatectl set-local-rtc 1` 修复。
- GRUB 菜单消失时，按「固件 BootOrder → update-grub → bcdedit」顺序排查。
- 预算允许时用两块物理硬盘装双系统，最省心。

在下一节，我们深入 Linux 的安装核心——**Linux 自定义分区与挂载点规划**，学会把磁盘按需求切成「根、home、swap」等挂载点。
