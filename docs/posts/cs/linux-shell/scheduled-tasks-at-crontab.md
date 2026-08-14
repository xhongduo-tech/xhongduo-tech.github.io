---
title: 例行性工作排程：at 与 crontab
date: 2026-08-07
---

# 例行性工作排程：at 与 crontab

<div class="epigraph">
<p>最好的运维，是把「以后要做的」变成「系统自己会做的」。</p>
<footer>—— 定时任务（scheduling）的工程智慧</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ 鸟哥《Linux私房菜》 第15章 ｜ 2026-08-07</p>
</div>

## 为什么从例行工作排程开始

数据库每天凌晨要备份、日志每周要归档、证书每月要检查到期——这些「到了点就该干」的事，不该靠人记，而该交给系统的**排程（scheduling）**机制。Linux 给出两个工具：**`at`** 处理「一次性、未来某个时刻」的任务，**`crontab`** 处理「周期性、反复发生」的任务。crontab 是每台服务器上「自动化」的起点：几乎每个生产系统都挂着一堆 cron 任务，而 `cron` 的五个时间字段，也是理解排程语法的一把万能钥匙——连 Kubernetes 的 CronJob 用的都是同一套字段。

## 1 一次性任务：at

**`at`** 让任务在指定的未来时间执行一次，之后不再出现。基本用法：

```bash
at now + 5 minutes          # 5 分钟后执行（然后输入命令，ctrl-d 结束）
at 14:30                    # 今天 14:30
at 23:00 08/20/2026         # 指定日期
atq                         # 列出待执行的任务队列
atrm 3                      # 删除编号为 3 的待执行任务
```

`at` 的守护进程 `atd` 每分钟检查一次任务队列。任务执行时的工作目录、环境变量由创建时的快照决定。<span class="marginnote">`at` 适合「晚点提醒」「错峰执行」这类一次性场景。交互输入命令后必须 `Ctrl-D` 结束输入，`atq` 可随时查看排队中的任务。</span>

**易错点**：`at` 任务执行时的输出默认会**以邮件形式发给用户**——很多服务器没配邮件服务，输出就悄悄丢了。想让结果可见，在任务里重定向到文件：`echo "备份完成" > /tmp/at-result.log`。

## 2 周期任务：crontab

**`crontab`** 是「cron table」——一张周期性任务表。每个用户一份，用 `crontab -e` 编辑、`crontab -l` 查看、`crontab -r` 清空：

```bash
crontab -e            # 编辑当前用户的 crontab（默认 vim）
crontab -l            # 列出当前用户的全部任务
crontab -r            # 删除当前用户全部任务
```

每一行一个任务，格式是：

```
分  时  日  月  周    命令
```

**公式解析：cron 的五个时间字段**

$$
\underbrace{\min}_{0{-}59} \quad \underbrace{\text{hour}}_{0{-}23} \quad \underbrace{\text{day-of-month}}_{1{-}31} \quad \underbrace{\text{month}}_{1{-}12} \quad \underbrace{\text{day-of-week}}_{0{-}7}
$$