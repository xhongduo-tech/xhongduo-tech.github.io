---
title: overlayfs
date: 2026-09-08
section: cs
---

# overlayfs

<div class="epigraph">
<p>联合挂载把只读 lower 与可写 upper 叠成一棵树：查找先 upper 后 lower，删除用 whiteout，拷贝上写把修改留在 upper。</p>
<footer>—— 据 Linux overlayfs 文档；OCI 镜像层实践对联合挂载的用法</footer>
</div>

[上一课](/cs/fs-quota)收口了单层布局与记账。容器要在同一只读镜像上开许多可写视图，不能每实例拷一份 [ext4](/cs/ext2-block-groups)。缺口是 **overlayfs**：VFS 里的联合挂载，不是新的盘格式。

## 问题

lower 只读，upper 可写，merged 是用户看见的根。同名文件：upper 遮住 lower。删除 lower 里有的名字：upper 放 whiteout 字符设备（或 xattr 标记），查找时当作不存在。改 lower 文件：copy-up 到 upper 再改，lower 不变。缺口不是再讲 [inode](/cs/inode-dir)，而是：两个目录树如何合成 lookup，以及 inode 号在 merged 视图里是否稳定（容器与 `stat` 的痛点）。

<span class="marginnote">直觉类比：overlay 像在书页上放一张透明描图纸——读的时候透到下一页，写之前先把原页描到纸上再改，原书永远干净；撕掉描图纸（删容器）就回到原样，whiteout 则是纸上的「此页作废」戳。</span>

<span class="marginnote">多层 lower 是「从左到右」的栈。workdir 是 overlay 内部 scratch，必须与 upper 同文件系统。本课不把存储驱动写成 Docker 百科。</span>

## 方法

`lookup`：在 upper 找，命中则停；若是 whiteout 则负缓存；否则去 lower。`mkdir`/`creat` 落在 upper。对 lower 文件第一次写：从 lower 读进 [页缓存](/cs/page-cache)，在 upper 建文件，拷数据与部分 xattr，再切到 upper inode。目录 copy-up 只造空目录骨架，子项仍可从 lower 透出。

```mermaid
flowchart TD
  LOOK["lookup"] --> U["upper"]
  U -->|"whiteout"| NEG["当作不存在"]
  U -->|"没有"| L["lower"]
  WRITE["写 lower 文件"] --> CU["copy-up 到 upper"]
```

## 机制

overlay 把 [快照](/cs/fs-snapshots) 那种「共享未改块」提升到目录树接口：共享的是 lower 文件，而不是 btrfs 结点。镜像层于是可以是 tar + overlay，而不强迫宿主机用 COW FS。不要把本课写成量化栏的分层订单簿。硬链接跨层、renames 与 whiteout 的边角是实现雷区，课序只要求机制：遮罩、白出、上写。

一份只读镜像为什么能喂几十个容器还互不干扰：lower 全局只挂一份，每个容器只各有一层薄薄的 upper；读穿透到 lower，写先 copy-up 到自己的 upper，谁也不碰公共的原始层。删掉容器只是丢掉它的 upper，镜像原样无损。

```mermaid
flowchart TD
  IMG["只读镜像 lower 层：全局一份"] --> A["容器 A 的 upper 可写层"]
  IMG --> B["容器 B 的 upper 可写层"]
  A --> W1["改配置文件：copy-up 后写进 A 的 upper"]
  B --> W2["没改过的文件仍透出 lower 原样"]
  W1 --> MA["A 的 merged 视图：看到 A 的修改"]
  W2 --> MB["B 的 merged 视图：看到原始内容"]
```

与配额：upper 上的分配才进可写层配额；lower 只读不记账到容器 uid，除非另做项目配额。


实现上：copy-up 会丢掉 lower 上某些 xattr 或打开的租约语义，数据库文件放 lower 再在容器里写是经典坑。inode 号在 merged 视图里可能随 copy-up 改变，stat 不稳定。 读法上只引用[上一课](/cs/fs-quota)的结论，不把对象换成训练推理或限价簿。

<span class="marginnote">常见误区：初学者容易以为目录 copy-up 会复制整棵子树。实际它只造一个空目录骨架让改名、权限这类元数据操作生效，子文件仍从 lower 透出；真正的整文件拷贝只发生在第一次写某个文件时。</span>

<span class="marginnote">为什么重要：同一文件 copy-up 前后 inode 号会变，`stat` 出来的结果不稳定——拿 inode 当缓存键或做监控聚合的程序，在 overlay 上会莫名其妙失效，这是容器环境里排查最久的一类坑。</span>

本课在操作系统进阶的「文件系统实现 / 接口进阶」课序里，对象是 **overlayfs**。

- 先修只引用，不重导：上一课的结论当公理，本课只补差。
- 五栏不吞并：不把本课写成大模型训练/推理，也不写成限价簿或权重量化。
- 文献用 OSTEP、McKusick、内核文档与具名会议论文；不发明 arXiv 编号。

## 边界

本课不引入 FUSE 版 unionfs 的全部历史。不保证 NFS 作 lower 时所有缓存语义。下一课把「文件系统不必在内核里」做成协议：FUSE。


版本字段会变，课序钉的是机制对象「overlayfs」，不是某一主线内核的结构体名。
后课默认：联合挂载用 upper/lower/whiteout/copy-up。用户态实现 VFS 操作，下一课 FUSE。

## 小结

- overlay 叠树：遮罩、whiteout、copy-up。
- 它是 VFS 联合，不是新的盘布局。
- 用户态文件系统协议是下一课。
- 出处：Linux `Documentation/filesystems/overlayfs.rst`；OCI 层实践。
