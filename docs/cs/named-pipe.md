---
title: 命名管道
date: 2026-09-08
section: cs
---

# 命名管道

<div class="epigraph">
<p>FIFO 把管道这种内核缓冲挂进目录：无亲缘进程也可以用路径打开同一条字节流。</p>
<footer>—— 据 POSIX FIFO；Stevens, UNIX Network Programming；Bach 整理</footer>
</div>

[上一课](/cs/ipc-pipe)的匿名管道只靠继承描述符。[inode 与目录](/cs/inode-dir) 已能表示特殊文件。[文件表](/cs/file-table) 对 FIFO 与普通文件共用 fd。缺口是 **命名**：`mkfifo` 创建目录项，两端 `open` 会合，缓冲仍在内核，不经磁盘文件内容。

## 问题

无亲缘的两个作业要对齐字节流，匿名 `pipe()` 做不到。若改用真正文件，要自己做轮询与截断。FIFO：inode 类型是管道，没有数据块指针；`open` 读端/写端在对方出现前可阻塞（除非 `O_NONBLOCK`）。缺口不是新的调度睡眠，而是路径与会合语义。权限用上一单元的模式位。

<span class="marginnote">直觉类比：匿名管道像两家合住的室内传菜口，必须先是一家人（有亲缘、继承描述符）才用得上；FIFO 像小区门口的公共收发室——门牌号（路径）写在目录里，任何住户都能凭地址找到它，不必互相认识。</span>

本课不把 System V `msgget` 当 FIFO 的另一种名字讲完。

<span class="marginnote">两端都关闭后缓冲丢弃。名字仍在目录里，可再打开。这与普通文件「内容还在盘上」不同。</span>

## 方法

`mkfifo(path, mode)`：VFS 创建 FIFO inode。进程 A `open` 读、B `open` 写，内核把它们接到同一环形缓冲，此后 `read`/`write` 与匿名管道相同，包括 `SIGPIPE`。[路径查找](/cs/path-lookup) 与 dcache 照常。不要把 FIFO 写到交换区当文件页——数据只在内存缓冲。

```mermaid
flowchart TD
  NAME["目录中的 FIFO"] --> OPEN["两端 open 会合"]
  OPEN --> BUF["内核环形缓冲"]
  BUF --> RW["read/write 同匿名管道"]
```

## 机制

命名管道把 IPC 接到文件系统命名空间，让 shell 重定向与无关进程共用同一抽象。它仍是单向字节流，无消息边界（除非用户自己定）。下一课 Unix 域套接字会补双向、数据报与传描述符。本课只补「有路径的管道」。

<span class="marginnote">常见误区：初学者以为 `mkfifo` 会在磁盘上划出一块空间存数据。实际上 FIFO 的 inode 没有数据块指针——名字在目录里，数据只在内核内存缓冲；两端都关闭后缓冲就丢弃，盘上从不落数据。</span>

<span class="marginnote">数字实例：POSIX 规定不超过 `PIPE_BUF`（Linux 上为 4096 字节）的一次 `write` 是原子的，不会与其他写者的字节交错；Linux 上 FIFO 的缓冲容量约 64 KB，写满后 `write` 阻塞——这正是下游卡住时上游自动「背压」的来源。</span>

阻塞会合把调度再请进来：写者未出现则读者睡。

```mermaid
flowchart TD
  R1["读者先 open"] --> B1["阻塞等写端出现"]
  W1["写者后 open"] --> MEET["内核接到同一缓冲"]
  B1 --> MEET
  W2["写者先 open"] --> B2["阻塞等读端出现"]
  R2["读者后 open"] --> MEET
  B2 --> MEET
  NB["带 O_NONBLOCK 打开"] --> NOWAIT["立即返回，不等待会合"]
```

## 边界

本课不引入 Linux 特有的 `O_DIRECT` 对 FIFO 的限制表。不保证原子写长度与匿名管道完全相同的每一个边界情况，教学上同一 POSIX 规则。下一课需要记录边界、双向、以及凭据。

后课默认：无亲缘进程可用路径会合字节流。更丰富的本地套接字，下一课 Unix 域。

## 小结

- FIFO 是挂在目录上的管道，open 会合。
- 数据在内核缓冲，不在文件块。
- 双向与传 fd 是 Unix 域套接字的缺口。
- 出处：POSIX FIFO；Stevens, *UNP*；Bach, *UNIX*。
