---
title: 路径查找
date: 2026-09-08
section: cs
---

# 路径查找

<div class="epigraph">
<p>把路径字符串拆成分量，从根或工作目录起逐级 lookup；权限在每一层目录的搜索位上检查。</p>
<footer>—— 据 Bach 对 namei 的叙述；POSIX 对路径名解析的整理</footer>
</div>

[上一课](/cs/inode-dir)规定目录是名字到 inode 号的文件，并给出 `/a/b` 的直观走法。[dcache](/cs/dcache) 能加速每一分量。缺口是**完整算法**：绝对/相对、`.` `..`、工作目录、根（chroot）、末尾斜线、以及何时跟随符号链接——链接本身下一课才展开，本课留下「遇链接则再解析」的接口。

## 问题

只说「从根 inode 读目录」不够：进程有 cwd 与 root，线程可以有不同 cwd。查找必须在每一层验证执行（搜索）权限，否则用户能靠猜 inode 号越权。缺口不是再定义 inode 字段，而是 namei 状态机：当前目录 inode、剩余分量、标志（要目录还是要文件、是否跟链接）。最后一分量对 `unlink` 与 `open` 处理不同。

本课不把 Linux 的 `nameidata` 每个标志位背完。

<span class="marginnote">「namei」就是 name to inode：把名字翻译成 inode 号的内核过程。直觉类比：像送快递——路径 `/a/b/c` 是三级门牌，每一级大门（目录）都得先有钥匙（搜索权限）才进得去，最后那间房才是真正的收件人（目标 inode）。</span>

<span class="marginnote">「。」是当前，「..」是父。越过挂载点要换超级块，那是 VFS/挂载课；本课承认查找会碰到挂载点接口。</span>

## 方法

绝对路径从进程 root 开始，相对路径从 cwd。循环：取下一分量，dcache 或目录文件 lookup，检查当前目录的搜索位。末分量按调用意图处理（必须存在 / 必须不存在 / 跟随链接）。深度与符号链接跳数有上限，防环。结果是目标 inode（或 dentry），交给 `open` 填[文件表](/cs/file-table)。

<span class="marginnote">「防环」的数字实例：符号链接 A→B→A 这种死循环，靠「最多跟随 40 跳」截断，超限返回 ELOOP 错误；路径分量总数也有上限（如每段 255 字节、全路径 4096 字节），防止查找无限消耗内核时间。</span>

```mermaid
flowchart TD
  PATH["路径字符串"] --> SPLIT["拆分量"]
  SPLIT --> CUR["从 root 或 cwd"]
  CUR --> LOOK["lookup + 搜权"]
  LOOK --> NEXT["下一分量"]
```

## 机制

路径查找把「用户看见的树」变成 inode 号序列，是一切按名系统调用的共同前缀。它不搬文件数据页——那是页 Cache。权限沿路径每一目录检查，最后才看目标的读/写位，这解释了为何对目录没有 x 就不能打开其下文件。不要把查找写成安全课的提权步骤；只讲检查点。

<span class="marginnote">常见误区：以为权限只看目标文件自己的 rwx。实际上沿路每一层目录都要通过 x（搜索位）检查——即使你拥有 `report.txt` 的读写权，只要 `docs` 目录没给你 x，照样打不开。数字实例：`chmod 666 report.txt` 后别的用户仍进不来，卡的不是文件，是中间那层目录的门。</span>

```mermaid
flowchart TD
  P["open /home/alice/docs/report.txt"] --> C1["检查 / 的 x 权限"]
  C1 --> C2["检查 home 的 x 权限"]
  C2 --> C3["检查 alice 的 x 权限"]
  C3 --> C4{"docs 有 x 权限？"}
  C4 -- "有" --> C5["lookup 得 report.txt 的 inode"]
  C4 -- "无" --> DENY["整个查找在此失败：EACCES"]
  C5 --> LAST["最后才检查文件自身的读/写位"]
```

## 边界

本课不引入 `openat` 相对任意 fd 的全部细节，只承认「起点可以不是 cwd」。不把 Unicode 正规化当内核默认。硬链接与符号链接在下一课改变「名与对象」；本课的 lookup 假定分量要么是目录项要么失败。

后课默认：分量级 lookup 已有。两个名字对着同一 inode，或一个名字存着另一条路径，下一课硬链接与符号链接。

## 小结

- namei 从 root/cwd 逐分量 lookup，并检查目录搜索位。
- 末分量语义随系统调用而变。
- 链接如何改写解析，是下一课。
- 出处：Bach, *UNIX*；POSIX pathname resolution；Tanenbaum *MOS*。
