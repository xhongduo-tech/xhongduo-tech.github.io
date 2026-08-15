---
title: Linux 使用 LVM 分区安装与扩容
date: 2026-08-07
---

# Linux 使用 LVM 分区安装与扩容

<div class="epigraph">
<p>海纳百川，有容乃大。</p>
<footer>——林则徐</footer>
</div>

<div class="article-byline">
<p>技能 · 操作系统安装与日常维护 ｜ 《鸟哥的Linux私房菜：基础学习篇》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从 LVM 开始

上一节的「固定分区」有个天生的痛点：**空间分错了很难改**。根分区只给了 40 GB，用了半年满了，想扩容就要搬数据、改分区，折腾半宿。**LVM（Logical Volume Manager，逻辑卷管理）** 就是为了解决「空间不够用」而生——它把「物理磁盘分区」和「逻辑存储空间」解耦，让磁盘空间像内存条一样可以「动态调整」。

鸟哥在第7章把 LVM 作为「进阶磁盘管理」专门介绍。本节从「普通分区的局限」出发，讲清 LVM 的三层架构（PV/VG/LV）和它的核心价值——**在线扩容**，并给出一套「装了 LVM 之后空间满了怎么办」的完整解法。

## 1 普通分区的局限

先看清要解决的问题。传统的「一个分区一个文件系统」模式下，磁盘管理有三个死穴：

- **扩容难**：根分区满了，不能就地扩展，得用 `gparted` 之类的工具搬数据、改分区表，而且改动有风险。
- **跨盘难**：一个分区只能在一块物理磁盘上，两块 500 GB 的盘拼不成一个 1 TB 的「大空间」。
- **快照难**：想给数据做「时间点副本」，普通文件系统没有好用的机制。

这些局限在日常桌面很少暴露，但在「空间会增长、数据要保护」的服务器与长期使用的个人机上，会越来越痛。LVM 正是冲着这三个痛点来的。<span class="marginnote">其实普通分区也有简单解法：`/home` 独立分区 + 定期清理。但 LVM 解决的是「结构性问题」——它让「空间管理」从「物理层」上移到「逻辑层」，从此分区大小不是写死的。</span>

## 2 LVM 三层架构：PV、VG、LV

LVM 把「磁盘」抽象成三层，从下往上分别是：

- **PV（Physical Volume，物理卷）**：把物理分区（如 `/dev/sda1`）标记为「可被 LVM 管理」的卷。
- **VG（Volume Group，卷组）**：把多个 PV 合并成一个「大池子」，相当于「内存条总和」。
- **LV（Logical Volume，逻辑卷）**：从 VG 这个大池子里划出的逻辑分区，才是你真正格式化、挂载、使用的「盘」。

对应关系：

```
/dev/sda1 ──pvcreate──> PV1 ─┐
/dev/sdb1 ──pvcreate──> PV2 ─┼──vgcreate──> VG ──lvcreate──> LV1 ──mkfs+挂载
/dev/sdc1 ──pvcreate──> PV3 ─┘                          └──> LV2
```

**重点：用户用的是 LV，管理者管的是 PV/VG。** 装系统时选的「LVM 分区」就是「先把磁盘建成 VG，再从 VG 里划 LV」。LV 满了，只需要「往 VG 里加 PV」或「压缩别的 LV」，再 `lvextend` 扩大目标 LV——全程在线，不用重启。<span class="marginnote">这套分层让「扩容」从「改分区表」变成「改参数」：加一块硬盘 → `pvcreate` 它 → `vgextend` 并入 VG → `lvextend` 扩大 LV → `resize2fs` 扩大文件系统。四步全是命令，数据零风险。</span>

## 3 LVM 的命令体系

LVM 的命令按「对哪层操作」分组，规律很清晰：

| 层级 | 查看 | 创建 | 删除 |
| --- | --- | --- | --- |
| 物理卷 PV | `pvdisplay` / `pvs` | `pvcreate` | `pvremove` |
| 卷组 VG | `vgdisplay` / `vgs` | `vgcreate` | `vgremove` |
| 逻辑卷 LV | `lvdisplay` / `lvs` | `lvcreate` | `lvremove` |

命名规律：「`pv`/`vg`/`lv` + 动作」——`pvs` 看物理卷、`vgs` 看卷组、`lvs` 看逻辑卷。这六个前缀词是 LVM 的全部词汇表，记住它们，命令手到擒来。

## 4 核心对比表：普通分区 vs LVM

| 对比项 | 普通分区 | LVM |
| --- | --- | --- |
| 空间上限 | 单盘上限 | 可跨多盘合并 |
| 扩容 | 搬数据改分区表，风险高 | 在线 `lvextend`，零风险 |
| 缩容 | 同样麻烦 | 支持（较谨慎） |
| 快照 | 无 | `lvcreate --snapshot` 支持 |
| 复杂度 | 低 | 中（多一层概念） |
| 适用 | 桌面简单场景 | 服务器、长期使用、爱折腾 |

**判断要点：** 桌面用户嫌麻烦可以不用 LVM（上一节的 `/home` 独立分区够用）；服务器、打算长期使用、数据敏感的用户，值得在装系统时勾选「使用 LVM」。Ubuntu 安装器甚至默认提供「LVM」选项——勾上即可，安装器自动完成全部 PV/VG/LV 创建。

## 5 公式解析：扩容时该给 LV 加多少

扩容最实用的公式是「目标容量 = 当前用量 × 冗余系数」。设 LV 当前用了 $U$，想留出 $\alpha$ 倍的余量：

$$
\text{新 LV 容量} = U \times (1 + \alpha)
$$

分三步拆解：

- **第一步，查当前用量**：`df -h` 看目标挂载点用了多少，记为 $U$。比如 `/` 用了 55 GB。
- **第二步，定冗余系数**：想留 50% 余量，$\alpha = 0.5$。
- **第三步，算目标并执行**：新容量 = 55 × 1.5 ≈ 82.5 GB。先 `lvextend -L 82G /dev/vg/root`，再 `resize2fs /dev/vg/root` 扩大文件系统，`df -h` 验证。

**辨析｜易错点：** `lvextend` 只扩大「逻辑卷」，**不会**自动扩大「文件系统」——ext4 必须再跑一次 `resize2fs`，xfs 用 `xfs_growfs`。漏了第二步，扩容后 `df -h` 看到的大小没变，这是新手最常见的「扩容失败」假象。

## 6 动手：创建 LVM 并在线扩容

**创建阶段（装系统时）：**

1. 安装器选「手动分区」，磁盘分区方式选「LVM」或「作为物理卷使用」。
2. 建好 VG（如取名 `vg0`），从 VG 里建 LV 挂载到 `/` 与 `/home`。
3. 完成安装后，`pvs`、`vgs`、`lvs` 三条命令确认三层结构都在。

**扩容阶段（空间满了）：**

```
sudo pvcreate /dev/sdb1          # 1. 把新分区标记为物理卷
sudo vgextend vg0 /dev/sdb1      # 2. 并入卷组，VG 变大
sudo lvextend -L +50G /dev/vg0/root   # 3. 给 root 的 LV 增加 50G
sudo resize2fs /dev/vg0/root     # 4. 扩大文件系统（ext4）
df -h                            # 5. 验证
```

全程不需要重启、不中断服务，这就是「在线扩容」。<span class="marginnote">第 4 步 `resize2fs` 也支持「只扩文件系统到 LV 实际大小」，所以 `lvextend -l +100%FREE` 配 `resize2fs` 可以一次扩满剩余空间——一句 `lvextend -l +100%FREE` 就能把 VG 里所有余量都给某个 LV。</span>

## 7 速查表：LVM 常用命令速记

| 场景 | 命令 |
| --- | --- |
| 看物理卷 | `pvs` / `pvdisplay` |
| 看卷组 | `vgs` / `vgdisplay` |
| 看逻辑卷 | `lvs` / `lvdisplay` |
| 新建物理卷 | `pvcreate /dev/sdX1` |
| 新建卷组 | `vgcreate vg0 /dev/sdX1` |
| 新建逻辑卷 | `lvcreate -L 20G -n lvname vg0` |
| 扩大逻辑卷 | `lvextend -L +10G /dev/vg0/lvname` |
| 扩大文件系统（ext4） | `resize2fs /dev/vg0/lvname` |
| 扩大文件系统（xfs） | `xfs_growfs /mount/point` |
| 创建快照 | `lvcreate --snapshot -L 1G -n snap /dev/vg0/lvname` |

## 8 小结

- 普通分区三大死穴：**扩容难、跨盘难、快照难**，LVM 正是为此而生。
- LVM 三层架构：**PV（物理卷）→ VG（卷组）→ LV（逻辑卷）**，用户用的是 LV，管理者管 PV/VG。
- 命令按层分组：`pv*`/`vg*`/`lv*` 三大前缀，`pvs`/`vgs`/`lvs` 查看，`create`/`extend` 操作。
- **在线扩容四步**：`pvcreate` → `vgextend` → `lvextend` → `resize2fs`，全程不重启。
- **易错点**：`lvextend` 不会自动扩文件系统，ext4 要补 `resize2fs`。
- Ubuntu 安装器默认提供 LVM 选项，勾上即可自动完成全部结构创建。

在下一节，我们回到 Apple 世界——**macOS 安装与「磁盘工具」分区**，看看 APFS 怎么把「分区」玩出新的花样。
