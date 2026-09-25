---
title: 文件模式位
date: 2026-09-08
section: cs
---

# 文件模式位

<div class="epigraph">
<p>每个 inode 带有类型与 rwx 三位一组：属主、同组、其他人；内核在 open 与执行时按有效身份核对。</p>
<footer>—— 据 Ritchie and Thompson；Bach；POSIX 对 mode 的整理</footer>
</div>

[上一课](/cs/crash-consistency)保证恢复后 inode 还在。[inode 与目录](/cs/inode-dir) 已列权限字段，尚未当课。[路径查找](/cs/path-lookup) 用了目录搜索位。缺口是把 **st_mode** 收成可执行的规则：类型比特、九位 rwx、以及与后课 setuid 的接口。不是 ACL 百科——那是安全课更细的模型。

## 问题

若只有「能 open 就能读写」，多用户机器无法共享 `/tmp` 又保住私有文件。Unix：uid/gid 对 inode 的 uid/gid，匹配属主则看属主三位，否则看组，再否则看 other。目录的 r=读目录项，w=增删项，x=把该目录当查找路径。文件的 x=允许 exec。缺口不是新的磁盘块，而是 `open`/`execve` 里的这次比较。

本课不把 POSIX ACL 的每一项写成默认。

<span class="marginnote">`umask` 在创建时关掉一些位。`chmod` 改模式。类型（普通、目录、链接、设备）占 mode 的高位，与 rwx 共存一个字。</span>

```mermaid
flowchart TD
  M["mode 0o750 = rwxr-x---"] --> T["高位类型: 普通文件"]
  M --> O["属主 rwx = 111: 读+写+执行"]
  M --> G["组 r-x = 101: 读+执行, 不可写"]
  M --> X["其他 --- = 000: 一律拒绝"]
  O --> C["内核按身份挑一组核对"]
  G --> C
  X --> C
```

<span class="marginnote">数字实例：`chmod 750 f` 里的 750 是八进制：7 = 二进制 111（rwx），5 = 101（r-x），0 = ---。合起来就是「属主全能、同组可读可进入、其他人一律拒绝」。每三位一组正好对应一位八进制数。</span>

## 方法

创建：申请 inode，写入创建者 uid/gid 与 `mode & ~umask`。访问：沿路径已查过各目录 x；对目标比较 r/w。root（或有能力者）通常绕过，除执行位等少数例外——细节留给安全课。与 [VFS](/cs/vfs) 接头：具体 FS 把 mode 存在 inode，VFS 做通用检查。

```mermaid
flowchart TD
  ID["有效 uid/gid"] --> MATCH{"属主/组/其他"}
  MATCH --> RWX["对应三位"]
  RWX --> DEC["允许或 EACCES"]
```

## 机制

模式位把隔离从页表接到文件对象：同一内核，不同用户看见不同可打开集合。它不代替用户/内核分裂。管道与套接字也有 inode 模式，但常由创建者限定。不要把模式位写成提权 cookbook；只讲核对发生在 VFS。

下一课的 setuid 会在 exec 时暂时改有效 uid，本课的比较规则仍然适用，只是「有效身份」变了。

<span class="marginnote">直觉类比：目录的 x 位像走廊的门禁——没有 x，你连走进去翻目录项（按名字找文件）的资格都没有；r 只是让你站在门口抄目录名单。所以「能列出目录内容」不等于「能打开里面的文件」，后者还要看文件自身的九位。</span>

## 边界

本课不引入 SELinux 标签。不把 Windows ACL 当对照考试。粘滞位、setgid 目录继承在下一课与 setuid 一起，以免本课吞并。安全栏的最小特权会引用本课九位，不在这里重写。

后课默认：open/exec 按九位核对。exec 时身份提升与 `/tmp` 删除规则，下一课 setuid 与粘滞位。

<span class="marginnote">常见误区：初学者遇到「打不开文件」就 `chmod 777`。正确做法是先问「我是谁（id）」「文件归谁（ls -l）」，再看该走属主、组还是其他那一组；滥开 777 会把写权限暴露给机器上所有用户，root 绕过检查也是少数例外而非常规。</span>

## 小结

- mode 含类型与三组 rwx；目录 x 是搜索。
- 核对用有效 uid/gid；umask 限制新建。
- setuid/粘滞位是下一课。
- 出处：Ritchie and Thompson, 1974；Bach, *UNIX*；POSIX `<sys/stat.h>`。
