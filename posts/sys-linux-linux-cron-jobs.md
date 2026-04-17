## 核心结论

Linux 定时任务最常见的两种方案是 `cron` 和 `systemd timer`。`cron` 是“按时间表达式触发命令的老式调度器”，白话说，就是系统每分钟看一次时间表，到了就执行；`systemd timer` 是“由 systemd 管理的定时触发器”，白话说，就是把“什么时候触发”和“触发后怎么运行、怎么记日志、失败后怎么处理”拆成标准化单元交给系统统一管理。

两者都能“定时跑起来”，但生产环境真正难的部分不是触发，而是四件事：失败能否被发现、任务能否避免并发重入、机器重启后是否要补跑、日志是否可追踪。`cron` 在这四件事上原生能力弱，往往需要你在脚本外再包一层日志、锁、告警和重试；`systemd timer` 则把这些能力放进了 unit 模型里，更适合长期、关键、需要稳定运维的任务。

可以先记住一个最简判断：

| 方案 | 擅长的事 | 天然短板 |
| --- | --- | --- |
| `cron` | 简单表达式调度，配置快，几乎所有 Linux 都有 | 重试、互斥、持久化补跑、统一日志都弱 |
| `systemd timer` | 依赖管理、日志归集、随机延迟、补跑、服务级重试 | 配置比 cron 多，依赖 systemd 环境 |

新手版理解可以压缩成一句话：`cron` 更像“在 `/etc/crontab` 上写一行命令”；`systemd timer` 更像“定义一个时间规则，再绑定一个受 systemd 管理的服务”。

---

## 问题定义与边界

定时任务的问题，不是“能不能每天 2 点执行”，而是“每天 2 点执行这件事，在系统波动、脚本报错、机器重启、夏令时切换、任务跑很久时，还是否可信”。

这里的目标边界有四个：

| 问题边界 | 含义 | `cron` 的天然表现 | `systemd timer` 的天然表现 |
| --- | --- | --- | --- |
| 静默失败 | 任务失败了但没人知道 | 常见，需要手工重定向日志和接告警 | 日志默认进 `journald`，更容易查 |
| 重入保护 | 上一次没跑完，下一次又来了 | 默认会重入，需要 `flock` 等外部锁 | 可借助 service 语义和锁控制得更清晰 |
| 持久化触发 | 机器关机时错过的任务，重启后要不要补 | 默认不补跑 | `Persistent=true` 可补跑 |
| 时区/DST | 时区或夏令时变化时会不会重复或漏跑 | 风险明显 | 仍要理解时间语义，但管理能力更强 |

所谓“静默失败”，白话说就是脚本其实已经报错了，但调度器没有把错误变成一个容易被发现的事件。很多新手以为“没有报警就是成功”，这在 `cron` 场景里经常不成立，因为一条 crontab 只负责发起命令，不负责告诉你业务结果是否可信。

一个玩具例子：你写了 `0 2 * * * python cleanup.py`，脚本里数据库密码过期了。`cron` 仍然会照常在第二天、第三天继续启动这条命令，但如果你没把标准输出和标准错误写入日志，也没接告警，那你看到的只是“任务还在跑”，不是“任务还成功”。

对应地，`systemd timer` 的边界也不是“全自动无坑”。例如 `RandomizedDelaySec` 是“随机延迟窗口”，白话说就是为了避免很多机器同一秒一起启动任务而故意加一点随机抖动；但这个延迟不是你想象中的“永久记住”，如果机器短期开关机，某次窗口可能直接错过。因此，`systemd timer` 解决的是“可管理性”，不是“从此不需要理解时间和状态”。

---

## 核心机制与推导

先看 `cron`。它的基本机制可以概括为：

1. `crond` 常驻运行。
2. 每分钟检查一次当前时间。
3. 把当前时间和 crontab 表达式做匹配。
4. 匹配成功就启动对应命令，通常交给 shell 执行。

这意味着 `cron` 的调度粒度天然是“分钟级轮询”。表达式本身很直观，但所有“执行后的事情”几乎都不在调度器内部。比如失败重试、并发互斥、依赖数据库先启动、统一收集日志，通常都要你自己在命令或脚本里额外实现。

再看 `systemd timer`。它由两个 unit 配合完成：

- `xxx.timer`：描述“什么时候触发”
- `xxx.service`：描述“触发后执行什么”

其中 `[Timer]` 里的核心字段通常是：

- `OnCalendar`：日历式时间规则，白话说就是“像日历一样定义触发点”
- `RandomizedDelaySec`：随机延迟窗口，避免同秒打爆资源
- `AccuracySec`：精度合并窗口，允许 systemd 把相近的触发合并，减少唤醒次数
- `Persistent=true`：持久化补跑，机器停机错过触发点后，重启时补一次

可以用一个近似公式理解下一次执行时刻：

$$
T_{next} = OnCalendar + RandomizedDelaySec + AccuracySec_{coalesce} + persistence\_adjust
$$

这里的含义不是“严格逐项相加的源码实现”，而是帮助理解四种因素共同决定最终触发点：

- `OnCalendar` 先给出一个基准时间，例如每天 `02:00:00`
- `RandomizedDelaySec` 在这个基准之后加一个随机偏移
- `AccuracySec` 允许 systemd 为了系统整体效率，把相近时刻做合并
- `Persistent=true` 会根据上一次实际触发记录，判断重启后是否需要补跑

新手版例子：假设你定义每天 2 点执行备份，`RandomizedDelaySec=900`，那么实际执行可能是 `02:03:12`，也可能是 `02:14:51`。如果凌晨那段时间机器是关机状态，且 timer 配了 `Persistent=true`，系统重启后会检查这次 2 点任务是否错过，若错过则补跑一次。

这里有一个重要推导：`cron` 的“时间匹配”和“服务运行”是松耦合的，而 `systemd timer` 的“时间触发”和“服务生命周期”是同一套管理体系的一部分。所以 `systemd timer` 更适合描述“这个任务依赖网络、失败要重试、日志要归档、错过要补跑”的真实工程语义。

真实工程例子：一组 200 台机器每天凌晨归档日志。如果都在 `02:00:00` 同时压缩并上传对象存储，存储网关和出口带宽会瞬间抖动。此时 `cron` 常见写法是每台机器自己 `sleep $((RANDOM%900))`，把随机逻辑塞进脚本里；`systemd timer` 则可以把抖动直接写成 `RandomizedDelaySec=900`，调度规则和执行逻辑分离，运维更容易审计。

---

## 代码实现

先看 `cron` 的基本语法。经典格式是：

| 字段 | 含义 | 取值示例 |
| --- | --- | --- |
| 分钟 | 第几分钟执行 | `0`, `*/5` |
| 小时 | 第几小时执行 | `2`, `*/6` |
| 日 | 每月第几天 | `1`, `15` |
| 月 | 第几月 | `1`, `12`, `*` |
| 星期 | 周几 | `0-7`, `1-5` |
| 命令 | 实际执行内容 | `/usr/local/bin/backup.sh` |

例如每天凌晨 2 点执行：

```cron
0 2 * * * /usr/local/bin/backup.sh
```

这行配置简单，但还不够工程化。更稳妥的 `cron` 写法至少要补两件事：日志和锁。

```cron
0 2 * * * flock -n /var/run/backup.lock /usr/local/bin/backup.sh >> /var/log/backup.log 2>&1
```

这里 `flock` 是“文件锁工具”，白话说就是先抢一个锁，抢不到就不执行，避免上一次没跑完这一次又进来。

再看 `systemd timer`。最小可用版本通常是两个文件。

`/etc/systemd/system/backup.service`：

```ini
[Unit]
Description=Nightly backup job
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/backup.sh
User=root
Restart=on-failure
RestartSec=60
```

`/etc/systemd/system/backup.timer`：

```ini
[Unit]
Description=Run nightly backup at 2 AM

[Timer]
OnCalendar=*-*-* 02:00:00
RandomizedDelaySec=900
AccuracySec=1min
Persistent=true

[Install]
WantedBy=timers.target
```

启用方式：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now backup.timer
sudo systemctl list-timers --all
sudo systemctl status backup.timer
sudo journalctl -u backup.service
```

如果只想做一个新手版可运行示例，可以把 `ExecStart` 指向一个简单脚本，例如写文件确认是否执行过。

为了把“触发时间计算”讲清楚，下面给一个可运行的 Python 玩具程序。它不是 systemd 源码实现，而是一个教学模型，用来说明“基准时间 + 随机延迟 + 是否补跑”的核心思路。

```python
from datetime import datetime, timedelta

def next_run(base_time, randomized_delay_sec=0, missed=False, now=None):
    """
    教学版模型：
    - base_time: 日历规则给出的基准触发时间
    - randomized_delay_sec: 随机延迟秒数，这里为可重复演示直接取固定值
    - missed: 是否因为关机等原因错过
    - now: 当前时间，用于模拟补跑
    """
    scheduled = base_time + timedelta(seconds=randomized_delay_sec)
    if missed:
        assert now is not None
        return now
    return scheduled

base = datetime(2026, 3, 12, 2, 0, 0)

# 正常执行：2点后加 5 分钟随机延迟
t1 = next_run(base, randomized_delay_sec=300, missed=False)
assert t1 == datetime(2026, 3, 12, 2, 5, 0)

# 错过触发后补跑：机器 8 点重启，则在重启后补一次
t2 = next_run(base, randomized_delay_sec=300, missed=True, now=datetime(2026, 3, 12, 8, 0, 0))
assert t2 == datetime(2026, 3, 12, 8, 0, 0)

print("demo ok")
```

这个例子要表达的不是“systemd 就这么算”，而是两个工程事实：

1. 触发点和实际运行点不一定相同。
2. “补跑”是调度器状态的一部分，不只是脚本多执行一次。

如果你只是想快速验证某条定时配置，`cron` 的上手门槛确实更低；但如果你要把任务交给团队长期维护，`timer + service` 的结构更清晰，因为时间、依赖、日志、重试都不必混在一个 shell 命令里。

---

## 工程权衡与常见坑

`cron` 最大的优点是简单，最大的问题也是简单。它简单到只负责“发起”，很多新手会把“命令被发起了”误认为“任务系统是可靠的”。在真实环境里，可靠调度至少还包含幂等、互斥、可观测和失败恢复。

常见坑可以直接列出来：

| 坑 | 发生在哪 | 原因 | 规避策略 |
| --- | --- | --- | --- |
| 静默失败 | `cron` | 输出没落日志，失败没人看见 | 重定向日志，接监控或邮件告警 |
| 重入冲突 | `cron` / `systemd` 都可能 | 上次未结束，下次已到点 | `flock`、数据库锁、幂等设计 |
| DST/时区跳变 | `cron` 更明显 | 本地时间前跳或回拨 | 尽量用 UTC，关键任务避免脆弱时点 |
| 随机延迟窗口被错过 | `systemd timer` | 机器不在可运行窗口内 | 延迟窗口不要超过实际在线窗口 |
| 把重试写成无限循环 | 两者都可能 | 失败控制放进脚本后失控 | 明确重试次数和退避策略 |
| 日志不可追踪 | `cron` 更常见 | 日志散落在脚本和重定向里 | `journalctl` 或统一日志路径 |

看一个典型误区。有人写：

```cron
0 2 * * * /usr/local/bin/backup.sh
```

如果 `backup.sh` 需要 3 小时，而第二天凌晨 2 点前还没结束，那么第二天新实例会直接启动，可能导致两份备份同时写同一目录，最终把结果弄坏。解决方案不是“期待脚本跑快一点”，而是显式加入互斥，例如 `flock`，并让脚本本身具备幂等性。幂等性是“同一操作执行多次，结果仍然一致”，白话说就是补跑或重复跑不会把系统搞乱。

再看 `systemd timer` 的常见误判。很多人以为配了：

```ini
[Timer]
OnCalendar=*-*-* 02:00:00
RandomizedDelaySec=900
Persistent=true
```

就一定会在 2:00 到 2:15 之间某个时刻执行一次。实际上，如果机器在那段窗口内根本没运行，或者 timer 本身未激活，那么这一轮可能不会按你直觉发生。`Persistent=true` 解决的是“错过基准触发后是否补跑”的问题，不是“任何情况下都保证在原窗口内完成”。

真实工程里，对“每日备份、归档、账单汇总”这类关键任务，建议把设计拆成三层：

1. 调度层：负责什么时候触发，优先用 `systemd timer`
2. 执行层：负责真正做事，脚本必须幂等
3. 观测层：负责日志、退出码、告警和任务时长监控

如果缺少第三层，再好的调度器也只能做到“按时失败”。

---

## 替代方案与适用边界

`cron` 不是过时工具，它只是更适合简单边界。比如一台临时机器上每 5 分钟清理一次 `/tmp/test-*`，或者容器镜像里根本没有 systemd，此时 `cron` 仍然是合理选择。它的优势在于部署成本低、理解成本低、验证速度快。

`systemd timer` 更适合这些场景：

| 场景 | 更推荐的方案 | 原因 |
| --- | --- | --- |
| 临时脚本、开发机验证 | `cron` | 写一行即可，改动快 |
| 容器内无 systemd | `cron` 或外部调度器 | 运行环境不支持 timer |
| 每日备份、归档、同步 | `systemd timer` | 需要补跑、日志、依赖和重试 |
| 多机批量任务 | `systemd timer` | 可直接加随机延迟控制抖动 |
| 高关键度任务 | `systemd timer` | 更容易统一运维和审计 |

还要看到第三类替代方案：应用级调度器和平台级调度器。例如 Kubernetes 里的 `CronJob`、Airflow、CI/CD 平台定时任务、云厂商的事件调度器。它们不是本文重点，但边界很明确：

- 如果任务跟单机文件系统、单机服务生命周期强绑定，`systemd timer` 很合适。
- 如果任务跨机器编排、依赖 DAG、要统一页面化管理，单机 `cron/systemd` 就不一定够。
- 如果运行环境是 Docker 单容器且没有 systemd，继续纠结 timer 没意义，应该换到容器编排层或外部调度器。

一个新手常见选择题可以这样判断：在 Docker 容器里跑每小时清理缓存，容器本身没有 systemd，那么别强行引入完整 init 系统，通常用 `cron` 或把调度交给宿主机/平台更实际；在 RHEL 9 主机上做每日数据库备份，则 `systemd timer + service + Persistent=true + Restart=on-failure` 往往比单条 crontab 更稳。

结论不是“systemd timer 全面替代 cron”，而是：当任务从“能跑”升级为“必须稳定可运维”，调度器就不能只看表达式简洁度。

---

## 参考资料

1. [systemd.timer 官方手册](https://www.freedesktop.org/software/systemd/man/devel/systemd.timer.html)  
说明 `OnCalendar`、`RandomizedDelaySec`、`AccuracySec`、`Persistent` 的具体语义，是理解 timer 行为的第一手资料。

2. [Cronradar: Why Cron Jobs Fail Silently](https://cronradar.com/blog/why-cron-jobs-fail-silently)  
从实战角度分析 cron 的静默失败、日志缺失和监控不足问题，适合理解为什么“能触发”不等于“可运维”。

3. [OneUptime: systemd timers as an alternative to cron on RHEL 9](https://oneuptime.com/blog/post/2026-03-04-systemd-timers-alternative-cron-rhel-9/view)  
给出用 systemd timer 替代 cron 的工程视角，重点讨论持久化补跑、随机延迟和关键任务场景。
