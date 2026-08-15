---
title: CTF Pwn：栈溢出、堆利用与 ROP
date: 2026-08-07
---

# CTF Pwn：栈溢出、堆利用与 ROP

<div class="epigraph">
<p>程序员的 bug 是黑客的礼物——尤其是那些关乎内存的。</p>
<footer>—— 佚名（二进制安全社区俗谚）</footer>
</div>

<div class="article-byline">
<p>第三级 · 网络攻防技术（渗透测试/CTF/红蓝对抗/应急响应） ｜ CTF 竞赛资料 ｜ 2026-08-07</p>
</div>

## 为什么从 CTF Pwn 继续

CTF 六大题型里，**Pwn** 被公认为最硬核的一类——它要攻击的是**程序的内存**。Pwn（源自「own」的谐音，意为「拿下」）题给一个二进制程序，要求通过**漏洞利用（exploitation）**控制其执行流，最终拿到 shell 读取 flag。Pwn 与系统漏洞利用（Metasploit 篇）同源，但更偏「微观」：你要理解栈、堆、寄存器、二进制保护机制，手工构造出能让程序执行攻击者代码的输入。为什么学 Pwn？因为它是**对「程序如何运行」理解最深的一条路**——溢出、ROP、堆利用，每一个概念都直指操作系统与编译器的底层机制。本节拆解 Pwn 三大主线：**栈溢出**、**ROP**、**堆利用**，并以「公式解析」的方式拆透一条栈溢出 payload 的构造。<span class="marginnote">Pwn 与《逆向工程与二进制分析》（cs-reverse-engineering）专题深度互补：Pwn 是「利用」，逆向是「理解」——逆向看程序的「怎么写的」，Pwn 想「怎么打」。掌握汇编、栈帧、调用约定（第三级《汇编语言》《计算机组成原理》的基础），是 Pwn 的地基。</span>

## 1 程序的防线：二进制保护机制

动手攻击之前，先认识程序的「防御工事」——现代编译器默认开启多种安全机制，Pwn 题正是「在防御之下突破」的博弈。用 `checksec` 查看：

**NX（No-eXecute，不可执行栈）**：栈和堆**不可执行**——攻击者注入的 shellcode 放在栈上也无法执行。→ 攻击转向 ROP（代码重用）。

**ASLR（Address Space Layout Randomization，地址空间随机化）**：每次运行，栈/堆/库的地址随机化——攻击者无法硬编码地址。→ 需要**泄露地址**（信息泄露漏洞）或**部分覆盖**。

**PIE（Position Independent Executable）**：程序自身代码也被随机化，与 ASLR 叠加。

**Canary（栈金丝雀）**：函数入口在栈上放一个随机值，返回前校验——溢出若覆盖了它，程序直接崩溃退出。→ 需要**泄露 canary** 或**溢出不触及它**。

**RELRO（重定位只读）**：保护 GOT（全局偏移表），防止篡改函数地址。

**核心要点｜保护机制与对应攻击思路：**

| 保护 | 防御内容 | 攻击思路 |
| --- | --- | --- |
| NX | 栈不可执行 | ROP |
| ASLR | 地址随机化 | 泄露地址 |
| PIE | 代码随机化 | 泄露基址 |
| Canary | 栈破坏检测 | 泄露/绕过 |
| RELRO | GOT 只读 | 改其他目标 |

<span class="marginnote">`checksec` 是 Pwn 题的第一条命令：`checksec ./pwn` 一眼列出所有保护。保护机制决定攻击路线——全开（Full RELRO + PIE + NX + Canary）的题与全关的题，打法完全不同。<strong>先 checksec，再想打法</strong>，是 Pwn 的铁律。</span>

## 2 栈溢出：最经典的内存漏洞

**栈溢出（stack buffer overflow）**：程序向栈上的固定长度缓冲区写入超过其容量的数据，**溢出部分覆盖了栈上相邻的数据**——包括**返回地址（return address）**。函数返回时，CPU 跳转到返回地址指向的代码——如果攻击者控制了这个地址，就控制了程序的执行流。

经典的脆弱代码：

```c
void vulnerable() {
    char buf[64];
    gets(buf);   // 不限制长度，可无限写入
    return;
}
```

**公式解析：一条栈溢出 payload 的构成。**

一次典型的栈溢出利用，payload 按栈布局从低地址到高地址排列为：

$$
\underbrace{\text{A}\times64}_{\text{填充 buf}} \quad \underbrace{\text{EBP 4字节}}_{\text{覆盖旧帧指针}} \quad \underbrace{\text{跳转地址}}_{\text{覆盖返回地址}}
$$

拆开看每一段：

- **填充段（$\text{A}\times64$）**：填满缓冲区 `buf`，让后续数据开始「越界」。
- **EBP 覆盖（4 字节）**：旧帧指针，可填任意值（溢出链里常无所谓）。
- **返回地址（跳转地址）**：**最关键的一环**——函数 `return` 时 CPU 会跳到这里。填成攻击者想要的地址（如 `system("/bin/sh")` 的地址），程序返回后即执行攻击者代码。

**为什么能控制？** `gets(buf)` 不限制长度，`strcpy`/`sprintf` 等不安全的字符串函数同理——它们不检查目标缓冲区边界。**编译器本可阻止，但「不安全函数 + 无边界检查」是 C 语言的历史遗留**，大量老代码仍在使用。

**核心要点｜栈溢出的三要素：**

| 要素 | 说明 |
| --- | --- |
| 不安全函数 | `gets`、`strcpy`、`sprintf` 等 |
| 越界写入 | 填充长度超过缓冲区 |
| 返回地址控制 | 覆盖 EIP/RIP 跳转 |

## 3 ROP：在没有可执行栈时接管控制

NX 开启后，栈上的 shellcode 无法执行——攻击者改用**代码重用（code reuse）**：**借用程序自身已有的代码片段**。**ROP（Return-Oriented Programming，返回导向编程）**的核心思想：程序里存在大量以 `ret` 结尾的指令序列，称为 **gadget**（如 `pop rdi; ret`、`system@plt`）。攻击者把这些 gadget 的地址**串成一条链**放在栈上，每次 `ret` 都跳到下一个 gadget，最终拼出完整的攻击逻辑（如「把 `/bin/sh` 的地址放进 `rdi`，再调用 `system`」）。

**公式解析：一条 ROP 链的布局。**

在 x64 上调用 `system("/bin/sh")` 的 ROP 链，栈从低到高为：

$$
\underbrace{\text{填充}}_{\text{到达返回地址}} \quad \underbrace{\text{pop\_rdi; ret}}_{\text{gadget}} \quad \underbrace{\text{"/bin/sh"地址}}_{\text{参数}} \quad \underbrace{\text{system 地址}}_{\text{目标函数}}
$$

拆开看这条链的每一环：

- **填充段**：覆盖到返回地址，让函数返回时 `ret` 弹出第一个 gadget 的地址。
- **`pop rdi; ret`（gadget）**：x64 调用约定里，`system` 的第一个参数放在 `rdi` 寄存器——gadget 把栈顶的值弹进 `rdi`，再 `ret` 到下一个地址。
- **`"/bin/sh"` 地址（参数）**：被 `pop rdi` 弹出的值——`system("/bin/sh")` 的参数。
- **`system` 地址（目标函数）**：程序里 `system@plt` 的地址（NX 下可执行的是程序自身代码，`system` 来自 libc 但通过 PLT 可达）。

**为什么 ROP 能绕过 NX？** 因为 ROP **不注入新代码**，而是复用程序已有的代码——可执行栈不存在了，但「执行 `ret` 链」完全合法。**每一条 gadget 都是现成的、可执行的代码**，攻击者只是把它们的地址按序摆好。找到足够多的 gadget（`ROPgadget` 工具自动搜）、按调用约定排好参数，就能拼出任意逻辑——这就是「代码重用攻击」的本质。<span class="marginnote">ROP 的进阶变体：<strong>SROP</strong>（Sigreturn-Oriented Programming，用 `sigreturn` 系统调用伪造信号帧）、<strong>JOP</strong>（Jump-Oriented Programming）、<strong>BROP</strong>（Blind ROP，无源码无二进制时盲打）。CTF 里 90% 的 ROP 题只需要 `pop rdi; ret` + 一个函数地址，但理解「gadget = 复用已有指令」这一思想，是应对一切变体的钥匙。</span>

## 4 堆利用：攻击动态内存的分配器

**堆（heap）**是程序**动态申请内存**的区域——`malloc`/`free`（C）、`new`/`delete`（C++）。**堆利用（heap exploitation）**攻击**堆分配器（allocator）**的元数据结构，把「动态内存」变成「任意读写」。堆利用比栈溢出复杂，因为堆不像栈那样有固定的返回地址可覆盖——它靠**元数据篡改 + 释放后引用**达成攻击。

**三大经典堆攻击**：

**UAF（Use-After-Free，释放后使用）**。程序**释放内存后仍保留指针**并继续使用——攻击者抢先在该位置分配受控数据，程序「使用」时读到的就是攻击者构造的内容。UAF 是堆题最常见的前置漏洞。

**Double Free（双重释放）**。同一块内存被 `free` 两次——分配器把同一块空闲块放进链表两次，后续两次 `malloc` 可能返回同一地址，造成重叠。

**tcache poisoning / fastbin attack**。现代 glibc 的 tcache（线程缓存）用单向链表管理空闲块，链表头存着「下一个空闲块的地址」——攻击者**篡改链表指针**，让下一次 `malloc` 返回**任意目标地址**，从而对任意地址读写（写 GOT 改函数地址、写 `__free_hook` 劫持执行流）。

**核心要点｜堆攻击的思维框架：**

| 思路 | 原理 | 落点 |
| --- | --- | --- |
| UAF | 释放后使用悬垂指针 | 构造受控数据 |
| Double Free | 同一块入链两次 | 分配重叠块 |
| 链指针篡改 | 伪造空闲链表 | 任意地址分配 |

**辨析｜易错点：** 栈溢出与堆利用的分工要分清——**栈溢出的目标常是「改返回地址」**（直接控制执行流），**堆利用的目标常是「获得任意读/写」**（再借写入去改 GOT、改函数指针，间接控制执行流）。堆题比栈题多一层「先控制分配器，再控制程序」的间接性，所以上手顺序一定是**先栈后堆**。<span class="marginnote">Pwn 的配套工具链：<strong>pwntools</strong>（Python 库，`p64()` 打包地址、`sendline()` 发包、`ELF()` 解析、`ROP()` 自动找 gadget——Pwn 题几乎必用）、<strong>gdb 插件 pwndbg/gef</strong>（调试）、`checksec`（看保护）、`one_gadget`（找通用 ROP 链）。CTF 入门 Pwn 的标准流程：`checksec` 看保护 → `file` 看架构 → 逆向/分析漏洞 → `pwntools` 写 exploit → 调试排错。</span>

## 5 小结

- **Pwn** 攻击程序的内存，与 Metasploit（第15篇）同源但更「微观」：理解栈、堆、寄存器、保护机制。
- **保护机制**（NX/ASLR/PIE/Canary/RELRO）决定攻击路线，`checksec` 是 Pwn 的第一条命令。
- **栈溢出**覆盖返回地址控制执行流；payload = 填充 + EBP + 跳转地址；`gets`/`strcpy` 等是漏洞源头。
- **ROP** 在 NX 下复用程序自身 gadget 拼出攻击逻辑——不注入新代码，靠 `ret` 链执行已有指令。
- **堆利用**攻击分配器：UAF、Double Free、tcache poisoning，目标是先获任意读写再劫持执行流。
- 工具链：**pwntools + pwndbg + checksec** 是 Pwn 标配；先栈后堆，由简到繁。

在下一节，我们从「利用」转到「理解」——**CTF Reverse：逆向工程与算法还原**，把程序翻个底朝天，读懂它藏起来的逻辑。