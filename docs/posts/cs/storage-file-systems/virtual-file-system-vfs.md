---
title: 虚拟文件系统 VFS
date: 2026-08-11
---

# 虚拟文件系统 VFS

<div class="epigraph">
<p>真正的软件工程不是消灭复杂性，而是把它藏到接口后面。</p>
<footer>—— 根据大卫 · 惠勒（David Wheeler）「计算机科学中的任何问题都可以通过增加一层间接层解决」改写</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 存储与文件系统 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 VFS 开始

前几篇我们见识了文件系统的多样性：ext4 是就地更新 + journaling，XFS 以 extent 闻名，tmpfs 根本不碰磁盘，NFS 的文件躺在另一台机器上。如果没有统一层，内核就得为每种文件系统写一套系统调用——`open_ext4()`、`read_nfs()`……而应用也只认 `open()`、`read()` 这几个名字。**虚拟文件系统（Virtual File System，VFS）**就是这层统一：它向上提供一套标准接口，向下定义文件系统必须实现的操作集合，让上百种存储实现共存于同一棵目录树。<span class="marginnote">Tanenbaum §4.5 用「两个接口」概括 VFS：VFS 接口（应用 ↔ VFS）与 VFS 文件系统接口（VFS ↔ 具体文件系统）。Linux 的 VFS 是这套思想的工业实现。</span>

VFS 也是理解分布式文件系统的关键：NFS 客户端就是「一个把 RPC 当作磁盘 IO 的 VFS 文件系统」，下一节讲 NFS 时你已经在心里有了它的骨架。

## 1 分层模型：中间隔着一层「虚」的

![VFS 分层：应用程序通过统一系统调用访问 VFS，VFS 把请求分发到 ext4、XFS、tmpfs、procfs、NFS 等具体文件系统，再由它们落到磁盘、内存或网络。](/images/storage-file-systems/virtual-file-system-layers.svg)

VFS 处在应用与具体文件系统之间，扮演**总调度**。它的核心承诺是：**应用看到的只有「文件」「目录」「打开的文件」这三个概念，至于数据在哪块盘、哪个网络端口、哪块内存，VFS 负责翻译**。<span class="marginnote">「虚」在哪？VFS 不存储任何真实数据，它只提供通用对象模型与分发机制——数据永远在具体文件系统里。procfs 甚至没有「数据」，它只是内核状态的视图。</span>

Linux VFS 用四个对象落实这套模型：

**超级块对象（superblock object）**：整个已挂载文件系统的抽象。
- **索引节点对象（inode object）**：单个文件/目录的元数据抽象。
- **目录项对象（dentry object）**：路径名解析的缓存单元——「路径分量 → inode」的对应关系。
- **文件对象（file object）**：一次打开的文件视图，持有当前偏移量、打开模式。

**重点：** dentry 是理解 VFS 性能的门把手。`/usr/lib/libc.so` 的路径解析每次要逐分量查目录，dentry cache 把「分量名 → inode」的解析结果缓存下来，第二次访问同一个路径就是一次内存哈希查找。<span class="marginnote">dentry cache 与 inode cache、page cache 一起构成 VFS 的三级缓存，它们都是「少碰磁盘」的产物——与第 2 篇 inode 的内存镜像一脉相承。</span>

## 2 分发机制：file_operations 与函数指针

VFS 的关键设计是**操作表（operations table）**：每个具体文件系统都提交一组函数指针，实现 VFS 规定的操作集。<span class="marginnote">Linux 里是 `struct file_operations`、`struct inode_operations`、`struct super_operations`——一套「接口 = 函数指针数组」的面向对象手法，C 语言的多态。你在第 1 篇看到的 `open/read/write` 就是通过这张表被分发到 ext4 或 NFS 的实现。</span>

一次 `read()` 的旅程：

1. 应用调用 `read(fd, buf, n)`。
2. 内核根据 fd 找到对应的**文件对象**。
3. 文件对象里的 `f_op->read` 指向某个具体文件系统的读函数。
4. 分发到 `ext4_read()` 或 `nfs_read()`，各回各家。

**辨析｜易错点：** VFS 不是「又一种文件系统」，而是「文件系统的框架」。没有 VFS，ext4 也能工作；有了 VFS，ext4、NFS、tmpfs 才能同时在 `/`、`/home`、`/tmp` 下共存，并且让 `cp` 不知道自己在跨文件系统拷贝。你把「接口层」与「实现层」搞混，后面读分布式文件系统就会处处别扭。

## 3 挂载的抽象：mount 表与跨越边界

VFS 维护一张**挂载表（mount table）**，记录「哪个文件系统挂到了哪个目录」。路径解析时，每经过一个目录都要问一遍：这里是不是挂载点？是，就切换到底下那个文件系统的根。<span class="marginnote">读 `/mnt/nfs/home/x` 的路径，可能在 `/mnt/nfs` 处从 ext4 的目录树跳进 NFS 的目录树——名字连续，物理跳转。这就是 VFS「一棵树、多个世界」的真相。</span> 挂载机制我们在第 1 篇见过，VFS 让它可以跨文件系统类型——`mount -t nfs`、`mount -t tmpfs` 由此而来。

## 4 内存型文件系统：没有磁盘的「文件」

VFS 的抽象能力最精彩的证据，是**非磁盘文件系统**：

- **tmpfs**：数据全在内存，掉电即失，`/tmp` 与 `/dev/shm` 用它——速度是磁盘的几十倍。
- **procfs**：`/proc/cpuinfo` 这类「文件」没有持久数据，读它等于读内核变量，写它等于改内核参数。
- **sysfs、cgroupfs**：设备模型与控制组的视图，同样「无盘」。

**重点：** procfs 证明 VFS 的接口抽象力——只要实现 `read` 操作，让「内容」在读取时即时生成，内核状态就能伪装成一个文件系统。这正应了麦克罗伊的那句「一切皆文件」：**抽象到极致，接口就是一切**。

## 5 核心要点：VFS 与相关抽象的关系

| 抽象层 | 统一的对象 | 类比 |
| --- | --- | --- |
| VFS | 文件 / 目录 / 打开文件 | 硬件「设备驱动」接口 |
| 块设备层 | 块（block） | 数据库 JDBC/ODBC |
| 系统调用层 | fd / 路径 | POSIX 标准 |
| 分布式 RPC | 过程调用 | NFS 客户端的内核实现 |

这三层各管一段：VFS 管「文件语义」、块设备层管「扇区读写」、系统调用管「进程怎么触达内核」。NFS 把 VFS 的「file_operations」实现成网络 RPC，下一节将沿着这条线把文件系统搬上网络。

## 6 小结

- VFS 是**接口层**：向上统一系统调用，向下要求文件系统实现操作集，让多种文件系统共存一棵树。
- 四大对象：**superblock、inode、dentry、file**；dentry cache 缓存路径解析结果。
- 分发靠**操作表（file_operations 函数指针）**，`read()` 一路查表落到具体实现。
- **挂载表**让路径解析能跨文件系统边界无缝跳转。
- tmpfs/procfs 证明 VFS 抽象力：**接口即一切**，数据来源可以完全虚拟化。

在下一节，我们沿着 VFS 的接口把文件系统伸向网络——**分布式文件系统 NFS**：客户端那头的 `open/read/write`，到底怎样变成网络那头的一串 RPC。
