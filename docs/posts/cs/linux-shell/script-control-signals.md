---
title: 脚本控制：信号捕捉、后台运行与运行控制
date: 2026-08-07
---

# 脚本控制：信号捕捉、后台运行与运行控制

<div class="epigraph">
<p>脚本不只是「从头跑到尾」——它要能优雅地回应中断，也能安静地在后台坚持。</p>
<footer>—— Shell 脚本的运行控制哲学</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ Blum《Shell 脚本编程大全》 第16章 ｜ 2026-08-07</p>
</div>

## 为什么从脚本控制开始

普通脚本被 Ctrl-C 一按就当场暴毙——中间改了一半的配置文件怎么办？临时文件谁清理？而长任务一关终端就死，又该怎么办？本章给脚本装上「自我管理」的能力：**`trap`** 让脚本捕获信号、在退出前清理现场；**后台运行**与 **`nohup`** 让长任务脱离终端独立存活；**`sleep`**、**`wait`**、**`jobs`** 则组成一套进程调度的工具箱。学完这些，你的脚本不再是一碰就碎的脆壳，而是「中断了会善后、离开了还能跑」的成熟程序。

## 1 信号：脚本的外来事件

进程随时可能收到**信号（signal）**——内核或另一个进程发给它的异步通知。对脚本而言，最常见的几个：

| 信号 | 编号 | 触发 | 默认行为 |
| --- | --- | --- | --- |
| INT | 2 | 按 `Ctrl-C` | 终止 |
| TERM | 15 | `kill PID` | 终止 |
| HUP | 1 | 关闭终端 | 终止 |
| QUIT | 3 | 按 `Ctrl-\` | 终止并生成核心转储 |
| KILL | 9 | `kill -9` | 终止，**不可捕获** |

**关键**：INT、TERM、HUP 这类信号**可以被脚本捕获**，让脚本在退出前做清理；KILL 不可捕获，只能被强杀。这就是「为什么 kill 之前先 kill -15」的脚本侧原因——给脚本一个善后的机会。

## 2 公式解析：trap 的用法

**`trap`** 让脚本在收到指定信号时执行一段命令，而不是直接死掉：

$$
\text{trap} \; \underbrace{\text{"命令"}}_{\text{收到信号时执行}} \; \underbrace{\text{信号名或编号}}_{\text{监听哪些信号}}
$$

一个「退出前清理临时文件」的经典脚本：

```bash
#!/bin/bash
tmpfile="/tmp/process.$$"

cleanup() {
    echo "清理临时文件..."
    rm -f "$tmpfile"
}

trap cleanup EXIT        # 无论脚本怎么退出，都执行 cleanup
trap 'echo "收到 Ctrl-C"; exit 1' INT   # 捕获中断

echo "写临时数据" > "$tmpfile"
sleep 30
```

拆解这条 trap：

- **第一步**：`trap cleanup EXIT`——注册一个「退出钩子」：脚本**无论正常结束还是中途退出**，都会先执行 `cleanup` 函数。
- **第二步**：`trap '...' INT`——按 Ctrl-C 时不再默认终止，而是先打印提示、再显式 `exit 1`（于是又触发 EXIT 钩子完成清理）。
- **第三步**：`$$` 是脚本的 PID，用它拼进临时文件名，避免多个实例互踩。
- **第四步**：结果是——即使被中断，临时文件也会被删掉。

**易错点**：`trap` 的字符串参数**由当前 shell 展开**，里面的变量要用单引号包住避免过早展开。`trap '' INT` 用空命令「忽略」信号；`trap - INT` 恢复默认行为。而 **KILL 无法被 trap**——这是它的设计本意。<span class="marginnote">`trap ... EXIT` 的 EXIT 不是信号而是「退出事件」，专门用于清理钩子。多条 trap 注册同一信号时<strong>后者覆盖前者</strong>。`trap 'echo bye' DEBUG` 还能在每个命令执行前触发——调试脚本时偶尔有用。</span>

## 3 后台运行：让脚本脱离终端

长任务最怕「关终端就死」。上一章我们认识了 `&`、`nohup`，这里把它们与脚本组合完整：

```bash
./long_task.sh &                # 后台运行
nohup ./long_task.sh &          # 忽略 HUP，关终端也继续
nohup ./long_task.sh > out.log 2>&1 &   # 标准姿势：输出也重定向
```

`nohup` 只忽略 HUP 信号——终端关闭时不会再「顺手」杀掉脚本。但注意：脚本自己 `exit`、被 `kill`、被 KILL，都照样结束；nohup 不是「永不结束」，只是「不受终端关闭影响」。<span class="marginnote">`&` 把任务放进当前 shell 的后台作业表（`jobs` 可看），作业与终端仍关联；`nohup` 切断 HUP 关联。想要真正「脱离终端、独立会话」，更彻底的是 `setsid` 或交给 systemd——生产级守护进程的归宿是 systemd，见第 2 篇《systemd 管理》。</span>

**易错点**：后台任务的输出**仍写到终端**——两个后台任务一起跑，输出会互相穿插。规范做法是把每个后台任务的输出重定向到独立文件。`nohup` 不重定向时默认写 `nohup.out`。

## 4 运行控制：jobs、wait 与 sleep

三个命令组成后台脚本的调度骨架：

| 命令 | 作用 |
| --- | --- |
| `jobs` | 列出当前 shell 的后台作业 |
| `jobs -l` | 连 PID 一起显示 |
| `wait` | 等待某个后台作业完成（可带 PID） |
| `sleep N` | 暂停 N 秒 |

**`wait`** 让脚本「等后台任务干完再继续」——否则脚本自己跑完退出，后台任务成了孤儿：

```bash
#!/bin/bash
./backup_a.sh &
pid_a=$!
./backup_b.sh &
pid_b=$!

wait "$pid_a" && echo "备份 A 完成"
wait "$pid_b" && echo "备份 B 完成"
```

`$!` 是**最近一个后台作业的 PID**，`wait "$pid"` 阻塞直到该作业结束，并把它的退出状态作为 `wait` 的退出状态——于是 `&&` 能判断成功与否。<span class="marginnote">`wait` 不加参数等<strong>所有</strong>后台作业。`sleep` 在脚本里最常见的用途是「等待服务就绪」与「限流降频」：`until curl -sf localhost:8080; do sleep 2; done` 是等服务的标准循环。</span>

**易错点**：`wait` 只等**当前 shell 的**后台作业。脚本里 `cmd &` 后立刻 `wait`，能并行执行多条任务；但如果脚本**本身**已经是在后台运行（被 `&`），它的子后台任务关系要理清——必要时用 PID 精确等待。

## 5 实战：一个带清理的守护脚本骨架

把信号、后台、控制组合起来，写一个「日志轮询脚本」的骨架：

```bash
#!/bin/bash
# 每 10 秒把新日志行复制到归档，Ctrl-C 时正常收尾
logfile="/var/log/app.log"
archive="/tmp/app.archive"
running=1

cleanup() {
    running=0
    echo "收到退出信号，正在收尾..."
}
trap cleanup INT TERM

while [ "$running" -eq 1 ]; do
    tail -n 1 "$logfile" >> "$archive"
    sleep 10
done
echo "已停止，归档保存在 $archive"
```

`INT`/`TERM` 信号把 `running` 置 0，循环自然结束、脚本打印收尾信息——**不靠硬杀，靠协作退出**。这是所有「能安全停止」的守护脚本的共同结构：一个可被信号改写的标志位 + 一个检查它的主循环。<span class="marginnote">真正的系统守护进程不用这个手写骨架，而是交给 systemd 的 `Restart=` 与 `ExecStop=`。但理解「信号 → 标志位 → 循环退出」的协作模型，是看懂一切服务框架的第一步。</span>

## 6 小结

- **信号是外来事件**：INT/TERM/HUP 可捕获，KILL 不可；`kill -15` 给脚本善后的机会。
- **trap 注册钩子**：`trap cleanup EXIT` 无论怎么退出都清理，`trap '...' INT` 拦截中断。
- **后台三件套**：`&` 后台、`nohup` 忽略 HUP、输出重定向到文件——标准姿势一条链。
- **jobs/wait/sleep**：`$!` 拿后台 PID、`wait "$