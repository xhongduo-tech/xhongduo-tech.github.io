---
title: ELF 加载
date: 2026-09-08
section: cs
---

# ELF 加载

<div class="epigraph">
<p>内核读程序头，按段 `mmap` 进地址空间，设入口；动态则把解释器 `ld.so` 一并映射，由它完成重定位与构造器。</p>
<footer>—— 据 System V ABI；Linux `fs/binfmt_elf.c` 传统行为；主干[加载与动态链接](/cs/load-dynlink)；Levine 整理</footer>
</div>

上一课[ELF 格式](/cs/elf-format) 给出表。缺口是**执行**：`execve` 之后谁映射谁跳。主干加载课有直觉。本课钉内核 vs `ld.so` 分工、ASLR、辅助向量 `auxv`。交叉编译下一课收束链接课序。

## 问题

`ET_EXEC` 固定地址（现代少）；`ET_DYN` PIE 随机基址。内核：映射 `PT_LOAD`，处理 `PT_INTERP`、栈上放 `argv`/`envp`/`auxv`。`ld.so`：找 `.so`、写 GOT、跑 `DT_INIT`。缺口是**这条启动链**，不是节头字段表。

栈权限、NX、RELRO 在加载时落实。

### 加载不是解释字节码

映射后跳到机器入口。字节码 VM 是后一单元。不要把 `ld.so` 当 JVM。

<span class="marginnote">binfmt_elf。SysV。主干 load-dynlink。本课补内核视角与 auxv（AT_PHDR、AT_ENTRY、AT_RANDOM）。</span>

## 方法

对照 `readelf -l` 与 `/proc/self/maps`。动态：`LD_DEBUG=files` 看搜索。静态：无 interp，内核跳 `e_entry`。

```mermaid
flowchart TD
  EXEC["execve"] --> KERN["内核映射 PT_LOAD"]
  KERN --> INT{"有 PT_INTERP?"}
  INT -->|是| LDSO["ld.so 绑定"]
  INT -->|否| ENT["跳 e_entry"]
  LDSO --> ENT
```

<span class="marginnote">术语翻译：auxv（辅助向量）就是内核塞给新程序的一叠「便签条」：除了 argv 参数与 envp 环境变量，再附送几条关键情报——程序头表在哪（AT_PHDR）、入口在哪（AT_ENTRY）、一小撮随机字节（AT_RANDOM）供栈保护用。`ld.so` 靠这叠便签开工，不用自己再摸内存。</span>

与[unwind](/cs/stack-unwinding)：加载器注册的对象列表供展开器找 FDE。

## 机制

失败：找不到 `.so`、重定位溢出、栈执行被拒。setuid 忽略部分环境（`LD_PRELOAD`）——安全点名。不要在 setuid 路径依赖预加载。

```mermaid
flowchart TD
  PH["程序头 PT_LOAD 逐项处理"] --> MAP["按 p_vaddr mmap 进地址空间"]
  MAP --> TXT["代码段：R+X，文件页映射"]
  MAP --> DAT["数据段：RW，文件页映射"]
  DAT --> BSS[".bss 无文件内容，用匿名页清零"]
  PH --> STK["另建栈，压入 argc/argv/envp/auxv"]
  STK --> JMP["就绪，跳到入口"]
```

<span class="marginnote">数字实例：段按页（通常 4 KB）对齐映射，要求 `p_offset` 与 `p_vaddr` 对 0x1000 取模相等——所以常看到代码段文件偏移 0x1000、虚拟地址 0x401000。一个几百 KB 的可执行文件，映射只建立页表项，代码页等真正访问时缺页才从磁盘调入。</span>

构造器顺序：与跨 `.so` 依赖有关，陷阱。

## 边界

本课不写 `mmap` 系统课全文。后课默认：ELF 由内核+ld.so 加载。下一课交叉编译与三元组：生成哪一种 ELF。

也不把加载当浏览器加载 URL。

<span class="marginnote">常见误区：初学者以为 `execve` 之后整个可执行文件已被「读进内存」——其实内核只建了映射，代码页要等第一次访问、缺页才从文件读入。也别把 `ld.so` 当 JVM 那样的解释器：它做的是找库、修 GOT、跑重定位，最终跳进去执行的是本机机器码。</span>

## 小结

- 内核映射段并设栈；动态把控制交给 ld.so。
- PIE/ASLR 改基址，重定位必须 PIC。
- auxv 把phdr 与随机数传给用户态。
- 出处：ELF/SysV；Linux binfmt_elf；Levine；主干 load-dynlink。
