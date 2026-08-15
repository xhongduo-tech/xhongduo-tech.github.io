---
title: 认识系统服务与 systemd 管理
date: 2026-08-07
---

# 认识系统服务与 systemd 管理

<div class="epigraph">
<p>init 是 Unix 的第一个进程，systemd 是 Linux 的新管家。</p>
<footer>—— 系统启动与服务管理的时代更替</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ 鸟哥《Linux私房菜》 第17章 ｜ 2026-08-07</p>
</div>

## 为什么从 systemd 开始

每一台 Linux 开机后都有一个 PID 为 1 的进程——它是所有进程的祖先，负责拉起系统、启动服务、监控进程生死。过去它叫 `init`，靠一串串 shell 脚本按顺序启动；现在几乎每个主流发行版都用 **systemd** 接手了这份工作。`systemctl` 与 `journalctl` 这两个命令，从此成为「管理服务器上一切服务」的日常入口。上一节我们学会了让任务按计划执行，而任务要执行、服务要常驻，前提是它们被 systemd 管好。本章把 systemd 的核心模型讲透：**unit**（服务的最小单位）、**target**（启动目标）、**依赖**（谁先谁后），并用对比表看清它与老 init 的差别。<span class="marginnote">systemd 由 Lennart Poettering 主导开发，2010 年起逐渐成为主流发行版默认 init。它同时是 init 系统的替代者、日志系统（journald）与定时器（systemd-timer）——名字里的「d」就是 daemon。</span>

## 1 systemd：一套管全部

**systemd** 不仅是进程管理器，还接管了日志（journald）、定时任务（systemd-timer）、网络、挂载、socket 等一整套系统服务——它的设计目标是用**统一声明式的方式**描述「系统里每个东西该长什么样」。这带来一个核心概念：

**unit（单元）** 是 systemd 管理的最小单位，每种 unit 对应一种系统资源：

| unit 类型 | 后缀 | 管什么 |
| --- | --- | --- |
| service | `.service` | 服务（守护进程） |
| target | `.target` | 一组 unit 的集合（启动目标） |
| socket | `.socket` | 网络或本地 socket |
| timer | `.timer` | 定时任务（cron 的现代替代） |
| mount | `.mount` | 挂载点 |
| slice | `.slice` | 资源分组 |

<span class="marginnote">unit 文件默认放在 `/etc/systemd/system/`（管理员自定义）与 `/lib/systemd/system/`（发行版自带）。`systemctl list-units` 列出当前加载的全部 unit，`--type=service` 只看服务。</span>

**`systemctl`** 是操作 unit 的主命令，最常用的六个动作：

```bash
systemctl start nginx        # 立即启动
systemctl stop nginx         # 立即停止
systemctl restart nginx      # 重启
systemctl reload nginx       # 重新加载配置（不断服务）
systemctl status nginx       # 查看状态与最近日志
systemctl is-active nginx    # 只看是否运行
```

## 2 start 与 enable：一次与永久

新手最容易混的是 `start` 与 `enable` 的差别：

**核心对比表：start/enable/status 三兄弟**

| 命令 | 作用 | 时机 | 反操作 |
| --- | --- | --- | --- |
| `systemctl start` | 立刻运行 | 本次会话 | `stop` |
| `systemctl enable` | 开机自动启动 | 下次开机起 | `disable` |
| `systemctl status` | 查看状态 | 任意时刻 | — |

**公式解析：服务的「当前运行」与「开机自启」是两个独立开关**

$$
\text{服务状态} = \underbrace{\text{是否正在运行}}_{\text{start / stop}} \quad \times \quad \underbrace{\text{是否开机自启}}_{\text{enable / disable}}
$$

拆解这条公式：

- **第一步**：两个开关互不干扰——一个服务可以「正在运行」但没 enable（重启后消失），也可以 enable 但当前没在跑（重启后才会拉起）。
- **第二步**：生产上改配置后的标准动作是 `systemctl restart nginx` 或 `reload`；装机后的标准动作是 `systemctl enable --now nginx`——`--now` 表示「立即启动且开机自启」，一次搞定两件事。

## 3 unit 文件：服务的说明书

systemd 的「统一声明式」落到具体，就是**unit 文件**——一个描述「这个服务怎么跑」的文本文件。一个精简的 nginx.service 长这样：

```
[Unit]
Description=The nginx HTTP and reverse proxy server
After=network.target

[Service]
Type=forking
ExecStart=/usr/sbin/nginx
ExecReload=/usr/sbin/nginx -s reload
ExecStop=/usr/sbin/nginx -s quit

[Install]
WantedBy=multi-user.target
```

三个小节：

- `[Unit]`：元数据与依赖。`After=network.target` 表示**等到网络就绪后再启动**——注意它是「次序」而非「必须」。
- `[Service]`：怎么跑。`Type=forking` 表示主进程会派生（fork）出子进程后自身退出，nginx 这类传统守护进程都这么写；`ExecStart` 是启动命令，`ExecReload` 是热重载。
- `[Install]`：什么时候开机启动。`WantedBy=multi-user.target` 表示「当系统进入多用户目标时启动我」。

<span class="marginnote">`systemctl cat nginx` 直接打印该服务当前的 unit 文件全文（含 override 合并后的结果），是「这个服务到底怎么配的」的最快答案。</span>

**重点**：改完 unit 文件，必须执行 `systemctl daemon-reload` 让 systemd 重新读取，否则它还在用旧配置。这是「改了配置却不生效」的第一排查点。

## 4 target：运行级别的新说法

SysV 时代的 **runlevel（运行级别）** 用数字 0–6 描述系统状态：0 关机、1 单用户、3 多用户文本、5 图形。systemd 用 **target** 取代了这套数字：

**核心对比表：runlevel 与 target**

| runlevel | 对应 target | 含义 |
| --- | --- | --- |
| 0 | poweroff.target | 关机 |
| 1 | rescue.target | 单用户救援 |
| 3 | multi-user.target | 多用户文本 |
| 5 | graphical.target | 图形界面 |

**`systemctl get-default`** 查看默认 target，`systemctl set-default multi-user.target` 把开机目标改成文本模式——服务器上去掉图形界面就是这一行。临时切换用 `systemctl isolate multi-user.target`。

## 5 journalctl：systemd 的日志

systemd 把日志也一并接管了：**`journald`** 收集所有服务的标准输出与标准错误，统一写进二进制日志（默认 `/var/log/journal/`），用 `journalctl` 查询：

```bash
journalctl -u nginx             # 只看 nginx 服务的日志
journalctl -u nginx -f          # 跟踪（tail -f 的日志版）
journalctl -p err               # 只看 err 及以上级别
journalctl --since "1 hour ago" # 时间窗过滤
journalctl -b                   # 本次开机的日志
```

<span class="marginnote">`journalctl -k` 看内核日志，等价于老的 `dmesg`；`journalctl --list-boots` 列出历次开机会话，配合 `-b -1` 可回看「上一次开机发生了什么」——排查重启前故障的利器。</span>

**易错点**：journald 是「程序写了什么就收什么」，日志总上限由 `SystemMaxUse` 控制（默认约为所在分区大小的 10%）。老派应用还会自己写 `/var/log/nginx/access.log` 这类文件日志——**两套日志并行存在**，排查时别只盯一头。

## 6 小结

- **systemd 是一套**：进程管理 + 日志（journald）+ 定时（timer）+ 挂载 + socket，统一声明式。
- **unit 是基本单位**：`.service`/`.target`/`.timer` 等后缀表示类型；`systemctl` 是操作入口。
- **start 与 enable 两码事**：start 只管本次运行，enable 管开机自启，`enable --now` 一次到位。
- **target 取代 runlevel**：multi-user.target 即传统「3」，get-default/set-default 管理默认启动目标。
- **日志交给 journalctl**：`-u` 看服务、`-p` 过滤级别、`-f` 跟踪；二进制日志默认在 /var/log/journal。

在下一节，我们把日志的话题接过去：/var/log 里各文件记什么、日志级别怎么读、logrotate 如何防止日志写爆磁盘——这就是**日志文件管理与日志轮替**。
