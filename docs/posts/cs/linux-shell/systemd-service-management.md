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

每一台 Linux 开机后都有一个 PID 为 1 的进程——它是所有进程的祖先，负责拉起系统、启动服务、监控进程生死。过去它叫 `init`，靠一串串 shell 脚本按顺序启动；现在几乎每个主流发行版都用 **systemd** 接手了这份工作。`systemctl` 与 `journalctl` 这两个命令，从此成为「管理服务器上一切服务」的日常入口。本章把 systemd 的核心模型讲透：**unit**（服务的最小单位）、**target**（启动目标）、**依赖**（谁先谁后），并用对比表看清它与老 init 的差别。

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