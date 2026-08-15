---
title: 后渗透：权限提升（Windows/Linux 提权）
date: 2026-08-07
---

# 后渗透：权限提升（Windows/Linux 提权）

<div class="epigraph">
<p>拿到 shell 只是比赛的开始，提权才是真正的战场。</p>
<footer>—— 佚名（渗透测试社区俗谚）</footer>
</div>

<div class="article-byline">
<p>第三级 · 网络攻防技术（渗透测试/CTF/红蓝对抗/应急响应） ｜ 渗透测试实践指南 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从权限提升开始

上一节我们用 Metasploit 拿到了 shell——但**这个 shell 往往不是最高权限**。Web 服务的漏洞通常让你以 `www-data`（Linux）或 IIS 应用池账户（Windows）运行，权限极低：读不了关键文件、装不了持久化后门、横向移动处处受阻。

**权限提升（privilege escalation）**就是后渗透阶段的第一仗：把「低权限的立足点」升级为「root（Linux）/SYSTEM（Windows）的完全控制」。《渗透测试实践指南》第 7 章把提权列为后渗透的核心动作。为什么提权如此重要？

因为**权限决定你「能看什么、能做什么、能待多久」**——低权限 shell 是过客，高权限 shell 才是主人。本节拆解 Linux 与 Windows 两套提权方法论。

<span class="marginnote">提权分两条路：<strong>垂直提权（escalation）</strong>——从低权限角色爬到高权限（本节主题）；<strong>水平提权</strong>——同级用户之间横向扩展（web 里叫水平越权）。操作系统层面，提权的终极目标 Linux 是 `root`、Windows 是 `SYSTEM` 或管理员——<strong>拿到最高权限，才能读写一切、隐藏自己、维持访问</strong>。</span>

## 1 提权的核心思路：找「过度授权」的入口

提权方法千变万化，但底层逻辑高度统一——**找到系统里「给了你比你应得更多权限」的缺口**。这些缺口来自三类：

**配置错误**。文件权限过宽、sudo 配置不当、服务以高权限运行、可写目录被 PATH 引用——「管理员懒了一下」就是提权入口。

**内核/服务漏洞**。操作系统或高权限服务存在可利用的 CVE——低权限用户触发，以内核/服务权限执行代码（如经典的 Dirty COW、EternalBlue 提权版）。

**凭证泄露**。系统里藏着的口令、哈希、密钥——`/etc/shadow`、备份文件、进程参数、历史命令里的明文口令。

**核心要点｜提权思维三问：**

1. 我是谁？——`whoami`/`id`，确认当前权限。

2. 我能跑什么？——sudo 列表、SUID 文件、可写脚本。

3. 系统里藏着什么？——口令、密钥、可写的关键文件。

**公式解析｜提权的通用流程：**
$$
\text{信息收集（当前权限/系统状态）} \rightarrow \text{找过度授权入口} \rightarrow \text{触发利用} \rightarrow \text{验证高权限}
$$
信息收集决定一切——提权的「智慧」不在利用本身，而在「找到那个入口」。这也是为什么提权文章反复强调「枚举先行」。

<span class="marginnote">自动化工具能帮你快速枚举提权入口：<strong>LinPEAS / WinPEAS</strong> 一键扫描「SUID、sudo、可写文件、内网凭证、服务漏洞」，输出按可疑度排序的清单；`linux-exploit-suggester`（LES）根据内核版本列出可能的内核利用。但要记住：<strong>工具给的是候选，最终要人工判断哪个入口真实可行</strong>——这正是渗透测试「自动化 + 人工」的协作模式。</span>

## 2 Linux 提权：sudo、SUID 与内核漏洞

Linux 提权有三大经典入口：

**sudo 配置不当**。`sudo -l` 查看当前用户能免密执行哪些命令。如果 `sudo` 允许你以 root 运行某个命令，而这个命令又支持「逃逸到 shell」（如 `vim`、`find`、`python`），就能 `sudo vim` 后 `:!bash` 拿到 root。

GTFOBins 网站专门收录「能被 sudo 逃逸成 shell 的命令」，是查这类入口的速查表。

**SUID 文件**。`find / -perm -4000 2>/dev/null` 找带 SUID 位（setuid）的可执行文件。

SUID 程序以**文件所有者身份**运行——如果 root 的 SUID 程序存在可利用的漏洞（如旧版本 `pkexec` 的 CVE-2021-4034 PwnKit），低权限用户就能借它执行 root 命令。

**内核漏洞**。查内核版本（`uname -a`），用 exploit-suggester 匹配已知内核 CVE。经典如 **Dirty COW（CVE-2016-5195）**——写时复制竞态，允许任意用户改写只读文件（一度可覆盖 `/etc/passwd` 添加 root 用户）。

<span class="marginnote">GTFOBins 的用法是查「某命令被 sudo/SUID 授予特权后如何逃逸」：`sudo` + `python3` → 直接 `sudo python3 -c 'import os; os.system("/bin/bash")'`。这些「特权命令逃逸」是 CTF 与真实渗透的常考入口，<strong>背熟常见逃逸比盲扫效率高得多</strong>。</span>

**核心要点｜Linux 提权检查清单：**

| 检查项 | 命令 | 发现什么 |
| --- | --- | --- |
| 当前用户 | `id` | 组、权限 |
| sudo 权限 | `sudo -l` | 免密命令 |
| SUID 文件 | `find / -perm -4000` | 特权二进制 |
| 内核版本 | `uname -a` | 匹配 CVE |
| 可写文件 | `find / -writable` | 覆盖提权 |
| 明文凭证 | `/etc/shadow`、历史命令 | 口令 |

## 3 Windows 提权：服务、令牌与哈希

Windows 提权的入口与 Linux 不同，四大经典：

**服务权限错误（服务以高权限运行）**。`service` 以 SYSTEM 运行，但服务可执行文件的路径可写、或服务配置可被普通用户修改——替换可执行文件或修改 `binPath`，服务重启后即以 SYSTEM 执行攻击者代码。

**未引用的服务路径（unquoted service path）**。

服务路径含空格且未加引号（如 `C:\Program Files\My App\app.exe`），Windows 按空格拆分解析路径——在 `C:\` 或 `C:\Program Files\` 下放置恶意 `My App.exe`，服务启动时被优先加载（经典提权手法）。

**令牌提权（token impersonation）**。`SeImpersonatePrivilege` 权限（如 IIS 用户常带）允许冒充其他进程令牌——用 `JuicyPotato`/`PrintSpoofer` 等工具冒充 SYSTEM 令牌执行命令。

**哈希传递（pass-the-hash，PtH）**。Windows 认证可用口令的 **NTLM 哈希**直接进行——抓到的管理员哈希不需要破解，直接 `psexec`/`wmiexec` 用它登录别的机器（呼应下一节横向移动）。

<span class="marginnote">哈希传递是 Windows 内网渗透的招牌技术：攻击者用 `mimikatz`/`secretsdump` 从内存或 `SAM` 抓出 NTLM 哈希，然后用 `crackmapexec`/`impacket` 的 psexec 直接以哈希登录目标——<strong>不需要知道明文口令</strong>。这解释了为什么 Windows 域环境里「内存中明文口令 + 哈希」要严防，也呼应第18篇的横向移动。</span>

**核心对比表｜Linux vs Windows 提权：**

| 维度 | Linux | Windows |
| --- | --- | --- |
| 目标权限 | root | SYSTEM/管理员 |
| 配置类入口 | sudo、SUID | 服务路径、权限 |
| 凭证类入口 | /etc/shadow、密钥 | SAM、内存哈希 |
| 经典工具 | LinPEAS、LES | WinPEAS、mimikatz |
| 招牌技术 | sudo 逃逸 | 哈希传递、令牌冒充 |

## 4 提权之后：为什么权限决定一切

提权不只是「更有面子」，它带来三个实际能力：**读关键数据**（数据库文件、配置里的口令、域控哈希）；**写系统任意位置**（安装持久化后门）；**完全隐藏**（清日志、关监控）。这正是后渗透后续动作（横向移动、权限维持）的前提——**没有 root/SYSTEM，横向移动与持久化都无从谈起**。

**辨析｜易错点：** 初学者拿到低权限 shell 就急着「打内核漏洞」，其实 70% 的提权靠**配置错误**而非内核 CVE。先把 sudo、SUID、可写文件、凭证翻一遍，再考虑内核漏洞——因为内核利用风险高（可能崩溃目标机）、成功依赖版本精确匹配。

**由简到繁**是提权的铁律，也符合「先枚举、后利用」的渗透哲学。

<span class="marginnote">提权与「从极限到大模型」知识树的连接：它的理论基础是第二级《操作系统》的「进程权限、UID/GID、特权分离」与第三级《计算机组成原理》的「内存管理」；它与本专题第24篇 CTF Pwn 的「劫持执行流」同源——<strong>一个是「借系统漏洞提权」，一个是「借程序漏洞控流」，底层都是「让代码以更高权限运行」</strong>。</span>

## 5 小结

- **提权**把低权限 shell 升级为 root/SYSTEM，是后渗透的第一仗，决定「能看什么、能做什么」。
- 提权逻辑 = 找**过度授权入口**（配置错误、服务漏洞、凭证泄露），先枚举再利用。
- **Linux** 三大入口：sudo 配置不当（GTFOBins 逃逸）、SUID 文件、内核 CVE（Dirty COW 等）。
- **Windows** 四大入口：服务权限错误、未引用服务路径、令牌冒充（SeImpersonate）、哈希传递。
- 经典工具：**LinPEAS/WinPEAS**（枚举）、**LES**（内核漏洞）、**mimikatz/impacket**（凭证与哈希）。
- 铁律：**由简到繁**——先配置错误，再内核漏洞；提权成功才有横向与持久化。

在下一节，我们带着提权后的权限走出单机——**横向移动与内网穿透**，看如何从一台机器跳到整个内网。