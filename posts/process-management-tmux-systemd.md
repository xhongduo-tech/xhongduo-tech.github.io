## 核心结论

`tmux`、`screen` 和 `systemd service` 解决的不是同一个问题。

`tmux` 是终端复用器，白话说，就是把多个命令行界面装进一个 SSH 连接里，并且允许你断开后再接回去。它最适合“人要回来继续看”的任务，比如模型训练、长时间编译、数据库迁移排查。你关掉 SSH 客户端后，`tmux` server 还在，进程通常会继续跑，重新登录后再 `tmux attach` 就能回到原来的界面。

`systemd service` 是服务管理器，白话说，就是让系统按规则启动、停止、重启某个长期进程。它最适合“人不需要盯着，但服务必须稳定活着”的任务，比如推理接口、同步守护进程、定时消费程序。它的核心价值不是“后台运行”，而是“自动拉起 + 生命周期可控 + 日志统一收集”。

`journalctl` 是 systemd 日志查询工具，白话说，就是专门查看 systemd 管理的服务日志。把服务注册成 unit 之后，标准输出和标准错误通常会自动进入 journal，你就不需要再手工找零散日志文件。

一个直接判断表：

| 方案 | 核心用途 | SSH 断开后进程是否继续 | 系统重启后是否自动恢复 | 是否适合交互排查 | 日志是否统一 |
| --- | --- | --- | --- | --- | --- |
| `tmux` | 保持交互式终端会话 | 是 | 否 | 是 | 否，默认仍看程序自身输出 |
| `screen` | 老牌会话保持 | 是 | 否 | 是 | 否 |
| `systemd service` | 守护长期服务 | 是 | 是，可 `enable` | 弱 | 是，配合 `journalctl` |
| `nohup` | 一次性后台命令 | 通常是 | 否 | 否 | 弱，默认写 `nohup.out` |

所以结论很简单：训练类、排查类任务先用 `tmux`；线上推理、常驻 worker、需要自动恢复的进程用 `systemd service`；`journalctl` 负责把服务日志查清楚。

---

## 问题定义与边界

先定义问题。这里讨论的是“长时间运行的 Linux 进程如何在会话断开、进程异常、机器重启后仍可控”。

边界主要有三层：

1. 会话层。
   你通过 SSH 登录后，拿到的是一个 shell。很多命令默认挂在这个 shell 或其控制终端上。登录断开后，shell 退出，相关作业可能收到 `SIGHUP`。`SIGHUP` 可以理解为“终端挂断通知”。

2. 进程层。
   进程可能因为代码异常、资源不足、依赖不可用而退出。即使 SSH 没断，它也可能自己崩。此时“后台运行”不等于“可恢复”。

3. 系统层。
   机器重启、服务开机自启、日志持久化，这些都不属于 `tmux` 的能力范围，而是 `systemd` 的职责范围。

一个玩具例子最容易看清问题：

你 SSH 到一台云主机，直接运行：

```bash
python train.py
```

如果你的网络抖动、笔记本合盖、VPN 断开，SSH 会话中断。此时训练是否继续，取决于 shell、作业控制和信号处理方式，不应该靠“碰运气”判断。对新手来说，最稳妥的做法是先把训练放进 `tmux`：

```bash
tmux new -s training
python train.py
# 按 Ctrl+B，然后按 d
```

这里的 `session` 是会话，白话说，就是一整套可恢复的终端工作区。`tmux` 不是简单把命令放后台，而是把命令放进自己的 server 管理的伪终端里。

再看真实工程例子：

你有一个 `infer.py`，对外提供模型推理 HTTP 服务。它不是“跑完就结束”的脚本，而是要长期监听端口。这个场景如果还靠 `tmux attach` 去盯，就已经偏离目标了。线上服务更需要：

- 崩了自动拉起
- 开机自动启动
- 标准输出被统一收集
- 能按服务名过滤日志
- 有启动频率限制，避免重启风暴

这就是 `systemd service` 的边界。

---

## 核心机制与推导

### 1. `tmux` 为什么能在 SSH 断开后继续运行

`tmux` 采用的是 server/client 模型。白话说，真正管理窗口和进程的是 `tmux server`，你眼前这个 SSH 里的终端只是一个 client。

它的层次通常写成：

$$
\text{Server} \rightarrow \text{Session} \rightarrow \text{Window} \rightarrow \text{Pane}
$$

- `session`：一组工作区
- `window`：一个标签页
- `pane`：一个分屏里的单独终端

你连上 SSH 后执行 `tmux attach`，本质上是让当前终端连接到已经存在的 `tmux server`。当 SSH 断开时，client 没了，但 server 还在，所以里面的 shell、训练进程、监控命令还能继续跑。

常用按键的意义可以这样记：

| 快捷键 | 含义 | 白话解释 |
| --- | --- | --- |
| `Ctrl+B d` | detach | 暂时离开当前 tmux，会话继续活着 |
| `Ctrl+B c` | new window | 新开一个标签页 |
| `Ctrl+B %` | split left/right | 左右分屏 |
| `Ctrl+B "` | split up/down | 上下分屏 |
| `Ctrl+B o` | next pane | 在分屏间切换 |
| `tmux ls` | list sessions | 看有哪些会话 |
| `tmux attach -t training` | reattach | 接回指定会话 |

这里要纠正一个常见误解：`tmux` 并不是“必须手动 `detach` 才能在 SSH 断开后保活”。即使是突然断网，`tmux` server 一般也会继续存在。`detach` 的主要作用是主动离开，而不是保活的唯一条件。

### 2. `systemd` 为什么能自动守护

`systemd` 管理的是 unit。`service` 是其中一种 unit 类型，白话说，就是“如何把一个长期进程注册成系统服务的配置文件”。

一个服务至少要回答四个问题：

- 什么时候启动
- 启动什么命令
- 失败后是否重启
- 日志去哪儿

最核心的重启参数是：

- `Restart=`：什么情况下重启
- `RestartSec=`：重启前等多久

官方文档明确建议，长时间运行的服务优先考虑 `Restart=on-failure`。这意味着：正常退出不重启，异常退出、信号杀死、超时等失败情况才重启。

但这还不够。因为如果程序启动后 1 秒就崩，单靠 `Restart=on-failure` 会无限循环重启。于是 systemd 又提供启动速率限制：

- `StartLimitBurst=`：时间窗口内允许多少次启动
- `StartLimitIntervalSec=`：这个窗口多长

可以写成一个直观的约束：

$$
\text{最大允许启动频率} = \frac{\text{StartLimitBurst}}{\text{StartLimitIntervalSec}}
$$

例如：

- `RestartSec=5s`
- `StartLimitBurst=3`
- `StartLimitIntervalSec=60s`

表示在 60 秒窗口内，最多允许 3 次启动尝试。若服务每次都很快失败，超过这个次数后，systemd 会停止继续自动拉起，并把 unit 置入失败状态，避免重启风暴拖垮系统。

### 3. `journalctl` 为什么适合配合 service

`journalctl` 查询的是 systemd journal。白话说，就是结构化日志数据库，而不是一个单独文本文件。

它的好处有三个：

- 直接按 unit 查：`journalctl -u foo.service`
- 直接追尾：`journalctl -f -u foo.service`
- 直接看最近 N 行：`journalctl -u foo.service -n 50`

这意味着你不再需要记“日志到底写在 `/var/log/foo.log` 还是某个自定义目录”。

不过日志是否能跨重启保留，要看 journald 的存储模式。`Storage=auto` 时，如果 `/var/log/journal` 存在，就会持久化到磁盘；否则常见行为是落到 `/run/log/journal`，重启后丢失。

---

## 代码实现

先给一个新手最常用的 `tmux` 流程。

```bash
tmux new -s training
python3 train.py
# 按 Ctrl+B，然后按 d，返回普通 shell

tmux ls
tmux attach -t training
```

如果你不想先进会话再输命令，可以一步创建并后台启动：

```bash
tmux new-session -d -s training 'python3 train.py'
tmux ls
tmux attach -t training
```

一个更像真实工作的做法是把监控和训练分开：

```bash
tmux new -s training
# 窗口 0：跑训练
python3 train.py

# Ctrl+B c，新窗口 1：看 GPU
watch -n 1 nvidia-smi

# Ctrl+B c，新窗口 2：看磁盘/日志
tail -f train.log
```

下面是一个最小可用的 `systemd` service。假设脚本路径是 `/home/user/infer.py`。

```ini
[Unit]
Description=Model inference daemon
After=network.target

[Service]
Type=simple
User=user
WorkingDirectory=/home/user
ExecStart=/usr/bin/python3 /home/user/infer.py
Restart=on-failure
RestartSec=5s
StartLimitIntervalSec=60
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
```

保存到：

```bash
/etc/systemd/system/infer.service
```

然后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable infer.service
sudo systemctl start infer.service
sudo systemctl status infer.service
```

日志查看：

```bash
sudo journalctl -u infer.service -n 50
sudo journalctl -u infer.service -f
sudo journalctl -u infer.service --since "2026-03-20 10:00:00"
```

如果你希望日志跨重启保留，先启用持久化目录：

```bash
sudo mkdir -p /var/log/journal
sudo systemd-tmpfiles --create --prefix /var/log/journal
sudo journalctl --flush
```

下面给一个可运行的 Python 玩具代码，用来模拟“前几次失败，之后成功”的守护场景。它不是调用 systemd，而是演示“失败重试 + 速率限制”这个核心思想。

```python
from collections import deque

def should_block_restart(failure_times, interval_sec, burst):
    """
    failure_times: 升序时间戳列表，单位秒
    返回每次失败后是否触发“超过启动限制”
    """
    window = deque()
    blocked = []

    for t in failure_times:
        while window and t - window[0] > interval_sec:
            window.popleft()
        window.append(t)
        blocked.append(len(window) > burst)
    return blocked

# 玩具例子：60 秒内允许 3 次，第四次开始被挡
times = [0, 5, 10, 15, 80]
result = should_block_restart(times, interval_sec=60, burst=3)

assert result == [False, False, False, True, False]

def next_restart_time(last_failure_time, restart_sec):
    return last_failure_time + restart_sec

assert next_restart_time(10, 5) == 15
print("ok")
```

这个例子对应的直觉是：如果服务在 `0, 5, 10, 15` 秒连续失败，那么第 4 次已经超出“60 秒内最多 3 次”的限制；等到了 `80` 秒，旧窗口滑出，新一轮计数才重新开始。

真实工程例子可以这样落地：

- 模型训练：放进 `tmux`，因为你需要随时 `attach` 看 loss、显存、调试输出。
- 模型推理 API：注册成 `infer.service`，因为你要的是开机自启、异常重启、日志统一查。
- 数据迁移脚本：如果需要手工确认中间状态，用 `tmux` 或 `screen`；如果已经做成幂等化批处理，再考虑 `systemd-run` 或正式 service。

---

## 工程权衡与常见坑

最常见的错误不是命令不会写，而是把工具用错层次。

| 问题 | 表现 | 解决 |
| --- | --- | --- |
| 用 `tmux` 承担线上守护职责 | 服务挂了没人知道，机器重启后也没了 | 把常驻服务迁移到 `systemd service` |
| 用 `systemd` 管理交互式训练 | 想看中间输出、手工中断、切分屏都很别扭 | 训练和排查优先用 `tmux` |
| `Restart=always` 配得过猛 | 代码一崩就无限拉起，CPU 飙高 | 优先 `Restart=on-failure`，并设置启动限速 |
| 忘了 `daemon-reload` | 改完 unit 文件但配置不生效 | 每次改 unit 后执行 `systemctl daemon-reload` |
| `ExecStart` 路径写错 | `status` 显示启动即失败 | 用绝对路径，先手工运行验证命令 |
| 只会 `journalctl -f` | 只能看当前滚动日志，排查历史困难 | 加 `-u`、`-n`、`--since`、`-b` 组合查询 |
| 日志不持久 | 重启后历史日志不见 | 创建 `/var/log/journal` 并 `journalctl --flush` |

再说几个容易被忽略的细节。

第一，`Type=simple` 通常已经够用。很多 Python、Node、Go 的前台常驻进程都应该以前台方式运行，让 systemd 直接盯住主进程。不要再在脚本里自己 `daemonize`，否则 systemd 反而难跟踪。

第二，`ExecStart` 不会像你交互式 shell 那样自动加载完整环境。`.bashrc` 里有的变量、conda 激活脚本、`PATH` 修改，service 未必能看到。解决办法是写绝对路径，必要时显式写 `Environment=`、`WorkingDirectory=`，或者让 `ExecStart` 指向一个你自己控制的启动脚本。

第三，`tmux` 保的是终端状态，不是业务可用性。进程在 `tmux` 里照样可能因为 OOM、代码异常、端口占用而退出。`tmux` 解决的是“会话不断”，不是“服务不挂”。

第四，`screen` 和 `tmux` 很像，但行为细节不完全一样。`screen` 默认支持自动 detach，适合老系统；`tmux` 在窗口、pane、脚本化管理上通常更现代。

---

## 替代方案与适用边界

`screen` 是 `tmux` 最直接的替代品。它也是终端复用器，核心命令流非常像：

```bash
screen -S migration
python3 migrate.py
# 按 Ctrl+A，然后按 d
screen -ls
screen -r migration
```

如果你的机器是老发行版、默认没装 `tmux`，或者团队习惯了 `Ctrl+A d`，那就直接用 `screen`。它不是“过时到不能用”，只是扩展性和窗口管理体验通常不如 `tmux`。

`nohup` 则更轻。它的定义就是忽略挂断信号并把输出从终端重定向出去。适合一次性脚本，不适合复杂交互。例如：

```bash
nohup python3 once_job.py > once_job.log 2>&1 &
```

它的问题也很明显：没有 session、没有分屏、没有统一重启策略、没有服务级生命周期。

如果你只是临时想让某个命令作为 unit 跑一下，但又不想手写 service 文件，可以考虑 `systemd-run`。它适合短期实验或一次性验证，等需求稳定后再落成正式 unit。

可以用下面这个表快速选型：

| 工具 | 特长 | 适用边界 |
| --- | --- | --- |
| `tmux` | 会话、窗口、分屏管理强，可重新 attach | 训练、调试、长时间交互任务 |
| `screen` | 兼容性强，很多老系统直接可用 | 需要 detach/reattach，但不追求复杂界面 |
| `nohup` | 最简单的后台化 | 一次性脚本、对交互和守护要求都低 |
| `systemd service` | 自动重启、开机启动、统一日志 | 常驻服务、线上进程、需要可运维性 |
| `systemd-run` | 临时 unit | 短期实验、快速验证服务化行为 |

最终的适用边界可以压缩成一句话：

- 人要回来继续操作，用 `tmux` 或 `screen`
- 系统要自己持续托管，用 `systemd service`
- 只想把一次命令丢后台，用 `nohup`

---

## 参考资料

- tmux Wiki: https://github.com/tmux/tmux/wiki
- OpenBSD `tmux(1)` 手册: https://man.openbsd.org/tmux.1
- GNU Screen Manual: https://www.gnu.org/software/screen/manual/screen.html
- GNU Screen Detach 章节: https://www.gnu.org/software/screen/manual/html_node/Detach
- `systemd.service` 手册: https://manpages.ubuntu.com/manpages/resolute/man5/systemd.service.5.html
- `systemd.unit` 启动限速说明: https://manpages.ubuntu.com/manpages/trusty/man5/systemd.service.5.html
- `journalctl(1)` 手册: https://manpages.ubuntu.com/manpages/resolute/en/man1/journalctl.1.html
- `journald.conf` 手册: https://www.freedesktop.org/software/systemd/man/journald.conf.html
- `systemd-journald.service` 手册: https://www.freedesktop.org/software/systemd/man/systemd-journald.service.html
- `nohup(1)` 手册: https://man7.org/linux/man-pages/man1/nohup.1.html
- Bash Signals 说明: https://www.gnu.org/s/bash/manual/html_node/Signals.html
