---
title: 日志文件管理与日志轮替
date: 2026-08-07
---

# 日志文件管理与日志轮替

<div class="epigraph">
<p>日志是服务器留给未来的信——前提是你记得去读，并且它还没被写爆。</p>
<footer>—— 运维经验之谈（Logging wisdom）</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ 鸟哥《Linux私房菜》 第18章 ｜ 2026-08-07</p>
</div>

## 为什么从日志管理开始

上一节我们用 `journalctl` 看日志，但日志不只是「出问题时翻一翻」——它是一台服务器的**病历本**：谁登录过、哪个服务报过错、磁盘告警何时开始，全在里面。而日志管理有两个永恒矛盾：**它必须被记录**，又**不能无限增长**。于是有了 `logrotate`：一套「按时按量把日志切分成旧文件、压缩、清理、给服务发信号」的轮替机制。本章先认识日志住哪、怎么分级，再拆解 logrotate 的配置与公式，最后对比「文件日志」与「journald」两条路。

## 1 日志住哪里：/var/log

Linux 日志约定俗成地住在 **`/var/log`**。几个最常见的文件：

| 文件 | 记录什么 |
| --- | --- |
| `/var/log/messages` | 系统级通用消息（RHEL 系） |
| `/var/log/syslog` | 系统通用消息（Debian 系） |
| `/var/log/auth.log` | 认证与登录记录 |
| `/var/log/secure` | RHEL 系的认证记录 |
| `/var/log/dmesg` | 内核环形缓冲（开机硬件检测） |
| `/var/log/cron` | cron 任务执行记录 |
| `/var/log/nginx/access.log` | 具体服务的访问日志 |

<span class="marginnote">`/var/log` 的命名并非巧合：`/var` 是「会变化的数据」（variable data），专门放日志、缓存、运行时文件。根分区只放系统本体，日志与用户数据隔离在 `/var` 或独立分区，防止日志写爆根分区拖垮系统。</span>

**日志级别**是读懂日志的刻度，从高到低：

| 级别 | 含义 |
| --- | --- |
| `emerg`（0） | 系统不可用 |
| `alert`（1） | 必须立即处理 |
| `crit`（2） | 严重错误 |
| `err`（3） | 错误 |
| `warning`（4） | 警告 |
| `notice`（5） | 正常但重要 |
| `info`（6） | 信息 |
| `debug`（7） | 调试 |

排查时先按级别过滤：`journalctl -p err` 只看错误与更严重，能立刻过滤掉九成噪音。

## 2 日志轮替：logrotate

**日志轮替（log rotation）** 解决「日志无限增长」的问题：达到阈值时，把当前日志改名为带序号的历史文件，并新建一个空日志继续写；历史文件按策略压缩、保留若干份、最后删除最老的。主力工具是 **`logrotate`**：

```bash
logrotate /etc/logrotate.conf     # 按主配置执行轮替
logrotate -d /etc/logrotate.conf  # 试运行（debug），不真正改动
logrotate -f /etc/logrotate.conf  # 强制执行（忽略时间判断）
```

`-d` 试运行是黄金习惯——任何轮替策略改动，先 dry-run 看它「打算做什么」再正式执行。

**典型配置**（放在 `/etc/logrotate.d/` 下一个文件一段）：

```
/var/log/myapp/*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    create 0640 root adm
    postrotate
        systemctl reload myapp > /dev/null
    endscript
}
```

每行的含义：

- `weekly`：每周轮替一次；可换成 `daily`、`monthly` 或按 `size 100M`。
- `rotate 4`：最多保留 4 份历史，之后删最老的。
- `compress`：历史文件压缩成 `.gz`。
- `delaycompress`：刚轮替的那份先不压（进程可能还在写）。
- `create`：轮替后新建日志文件的权限与属主。
- `postrotate/endscript`：轮替完成后执行的命令——服务需要**发信号重开日志文件**时在这里做。

## 3 公式解析：轮替能留多久

**`rotate N` 到底能让日志存活多久**，是一个可计算的量：

$$
\text{可回溯总时长} \approx \text{轮替周期} \times (N + 1)
$$

以 `weekly` + `rotate 4` 为例：

- **第一步**：`weekly` 表示每 7 天把当前日志切成一份历史。
- **第二步**：`rotate 4` 保留 4 份历史 + 1 份当前，共 5 个文件。
- **第三步**：总跨度约 $7 \times 5 = 35$