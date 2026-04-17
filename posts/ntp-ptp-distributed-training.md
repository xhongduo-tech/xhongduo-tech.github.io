## 核心结论

NTP 和 PTP 都是在多台机器之间“对齐时间”的协议。白话说，它们都回答同一个问题：这台机器现在到底几点了，但精度等级不同。

对分布式训练，先记住三个结论：

| 方案 | 常见部署形态 | 典型精度级别 | 适合场景 | 代价 |
|---|---|---:|---|---|
| NTP + Chrony | 普通服务器、交换机无硬件时间戳 | 互联网常见为几十毫秒；机房内好网络通常是几十到数百微秒 | 大多数训练集群、日志对齐、常规超时控制 | 部署简单 |
| PTP / IEEE 1588 | NIC 与交换机支持硬件时间戳 | 子微秒级，常见到微秒内，硬件好时可逼近 100 ns 量级 | 高频采样、严格时序对齐、精密测控、超大规模训练 | 需要硬件支持 |
| 不做统一时间治理 | 各节点各自跑 | 漂移不可控 | 不建议用于多机训练 | 排障成本极高 |

最直观的比较是：如果一组机器用 NTP，在机房内做到 $100\ \mu s$ 量级已经算不错；如果换成 PTP，并且网卡和交换机都支持硬件时间戳，误差可以压到 $0.1\ \mu s$ 到几微秒，差距是 $10^2 \sim 10^3$ 倍。对“只要能跑就行”的训练任务，NTP 往往够用；对“跨节点启动、采样、打点必须几乎同时发生”的场景，PTP 才是正解。

---

## 问题定义与边界

时钟漂移是“机器自己的钟越走越偏”。白话说，两台服务器刚开机时一样准，过一段时间就会慢慢出现差值。网络抖动是“同一条链路每次传输耗时不一样”。这两个因素叠加后，节点之间看到的“现在”就不再一致。

分布式训练里，时间不同步通常不是直接把模型训练错，而是先在三个位置出问题：

1. 日志时间线错位，排障时以为 rank 3 先挂，实际上 rank 7 更早卡住。
2. 超时判断不一致，有的 rank 认为已经等很久了，另一些 rank 认为还没到。
3. 监控与调度系统错误关联事件，形成“假超时”或“假死锁”的排查噪音。

这里需要定义一条可信时间链路，也就是每个节点都满足：

- `timedatectl status` 中 `System clock synchronized: yes`
- `NTP service: active`
- `chronyc tracking` 能看到稳定的 `Reference ID`
- `chronyc sources -v` 中被选中的上游时间源一致，且 Stratum 没有异常分叉

Stratum 可以理解为“离权威时间源有几跳”。数字越小，通常离根时间越近。Reference ID 可以理解为“你到底跟的是哪一个上游时钟”。如果 8 台训练机里有 6 台跟 `ntp-a`，2 台跟 `ntp-b`，即使都显示“已同步”，它们也可能不在同一条时间链路上。

下面这张表可以帮助新手先建立误差边界：

| 概念 | 白话解释 | 它如何扩大误差 |
|---|---|---|
| 时钟漂移 | 本地晶振不准，时间会自己慢慢跑偏 | 节点长期运行后偏差累积 |
| 网络抖动 | 同一网络包每次耗时不同 | NTP 估计往返延迟时更不稳定 |
| Stratum | 离根时间源的层级 | 层级越深，累计不确定性通常越大 |
| Reference ID | 当前跟随的时间源标识 | 不同节点跟不同源时，整体一致性变差 |

玩具例子：两台机器 A、B 都在同一交换机下。A 比真实时间快 $0.6\ ms$，B 比真实时间慢 $0.4\ ms$。单机看都“不算离谱”，但 A 和 B 之间已经差了 $1.0\ ms$。如果你的训练脚本把 1 ms 以内事件当成“几乎同时”，这个假设已经失效。

---

## 核心机制与推导

Chrony 是 Linux 上常见的 NTP 实现。它不会简单地“看到时间不准就立即拨过去”，而是持续估计两件事：

1. 网络带来的测量误差
2. 本地时钟自身的漂移速度

Chrony FAQ 给出的关键量是：

$$
root\_distance = \frac{root\_delay}{2} + root\_dispersion
$$

其中：

- `root_delay` 是到上游时间源的累计往返延迟，除以 2 后可近似理解为单向传播误差上界
- `root_dispersion` 是累计离散度，白话说是“因为时钟不稳定和历史测量误差带来的不确定性”

所以 `root_distance` 可以理解为“当前这台机器离真实时间最多可能偏多少”。

玩具例子：

假设某节点上看到：

- `root_delay = 0.20 ms`
- `root_dispersion = 0.05 ms`

那么：

$$
root\_distance = 0.20/2 + 0.05 = 0.15\ ms
$$

这表示仅从 NTP 估计链路来看，这台机器当前就可能带着 $0.15\ ms$ 的最大不确定区间在跑。若另一台机器的 `root_distance` 也是 $0.15\ ms$，两台节点之间的相对误差可能接近 $0.30\ ms$。这还没算上应用层排队和调度抖动。

NTP 与 PTP 的差异，本质上不是“谁更先进”，而是“谁更依赖软件估计，谁更依赖硬件打点”。

- NTP：通过周期性请求多个时间服务器，结合网络延迟估计，在内核里缓慢修正系统时钟。
- PTP：采用主从时钟模型，网卡在收发报文的瞬间打硬件时间戳；如果交换机支持 Transparent Clock，还会把交换机内部转发延迟从误差里扣掉。

可以把两者简化为两条链路：

| 协议 | 误差主要来源 | 消减方法 |
|---|---|---|
| NTP | 网络 RTT 抖动、软件调度延迟、本地晶振漂移 | 多次采样、统计滤波、频率平滑 |
| PTP | 硬件振荡器误差、链路非对称、设备时间戳能力 | NIC 硬件时间戳、Transparent Clock、Boundary Clock |

因此，NTP 更像“不断估算并纠偏”，PTP 更像“尽量在物理层附近直接测准”。

---

## 代码实现

先给出最实用的 Ubuntu 排查顺序。目标不是“看命令有没有输出”，而是确认整组机器是否在同一时间链路上。

```bash
timedatectl status
chronyc tracking
chronyc sources -v
chronyc sourcestats -v
```

典型关注字段如下：

```text
timedatectl status
  System clock synchronized: yes
  NTP service: active

chronyc tracking
  Reference ID    : C0A80101
  Stratum         : 3
  System time     : 0.000123456 seconds slow of NTP time
  Last offset     : -0.000031234 seconds
  RMS offset      : 0.000045678 seconds
  Residual freq   : +2.345 ppm
  Root delay      : 0.000200000 seconds
  Root dispersion : 0.000050000 seconds

chronyc sources -v
  ^* ntp-a.example.com
  ^+ ntp-b.example.com
```

字段解释：

- `System clock synchronized`：系统时钟是否已经被同步服务接管
- `Reference ID`：当前时间最终跟随的上游来源
- `Last offset`：最近一次校正时，本机与参考时间的偏移
- `Residual freq`：剩余频率误差，白话说是“这块表还在以多快的速度继续跑偏”
- `^*`：当前被选中的主时间源

如果节点刚启动，偏差很大，可以手动触发一次快速校正：

```bash
sudo chronyc makestep
chronyc tracking
```

下面给一个可运行的 Python 小程序，用于把 `chronyc tracking` 里的核心字段转成误差估计，并对阈值做断言：

```python
import re

sample = """
Reference ID    : C0A80101
Stratum         : 3
Last offset     : -0.000031234 seconds
Residual freq   : +2.345 ppm
Root delay      : 0.000200000 seconds
Root dispersion : 0.000050000 seconds
"""

def parse_seconds(field: str, text: str) -> float:
    m = re.search(rf"{re.escape(field)}\s*:\s*([+-]?\d+\.\d+)", text)
    assert m, f"missing field: {field}"
    return float(m.group(1))

root_delay = parse_seconds("Root delay", sample)
root_dispersion = parse_seconds("Root dispersion", sample)
last_offset = parse_seconds("Last offset", sample)

root_distance = root_delay / 2 + root_dispersion
max_uncertainty = abs(last_offset) + root_distance

assert abs(root_distance - 0.00015) < 1e-12
assert max_uncertainty < 0.001  # 小于 1 ms
print(f"root_distance={root_distance*1e6:.1f} us")
print(f"max_uncertainty={max_uncertainty*1e6:.1f} us")
```

真实工程例子：8 节点 PyTorch 训练，启动前先在所有节点执行一次一致性巡检。

```bash
for host in node1 node2 node3 node4 node5 node6 node7 node8; do
  echo "== $host =="
  ssh $host "timedatectl status | egrep 'System clock synchronized|NTP service'"
  ssh $host "chronyc tracking | egrep 'Reference ID|Stratum|Last offset|Residual freq|Root delay|Root dispersion'"
  ssh $host "chronyc sources -v | egrep '^\^\*|^\^\+'"
done
```

如果发现某台机器 `Reference ID` 不同，或者 `Residual freq` 明显偏大，就不要急着改训练脚本，先修时间链路。

---

## 工程权衡与常见坑

分布式训练的 watchdog 是“超时看门狗”，白话说就是后台线程盯着某次 collective 是否卡太久。这里最容易踩的坑，是把“通信超时”与“时间链路有问题”混为一谈。

先澄清一个容易过时的点：旧文章常写 NCCL 默认超时是 1800 秒，但截至 2026-03-21 核对的 PyTorch 2.10 官方文档，`init_process_group(..., timeout=...)` 对 NCCL 的默认值是 10 分钟，其他后端是 30 分钟。排查时必须先确认你依赖的是哪一层超时设置，而不是照搬旧参数。

常见坑可以直接按表排：

| 坑 | 症状 | 排查命令 | 缓解措施 |
|---|---|---|---|
| 节点未真正同步 | 某些 rank 日志时间线明显错位 | `timedatectl status` | 确认 `System clock synchronized: yes` |
| 跟随不同时间源 | 所有机器都显示已同步，但偏差仍大 | `chronyc sources -v` | 统一上游 NTP 源 |
| 本地晶振或 RTC 异常 | `Residual freq` 长期偏高 | `chronyc tracking` | 排查主板 RTC、虚拟化宿主机时间策略 |
| 误把死锁当漂移 | timeout 调大后只是更晚失败 | 训练日志 + NCCL debug | 先确认 collective 顺序、网络与 OOM |
| 跨地域训练链路长 | 偏移与 RTT 同时波动 | `chronyc tracking` + 网络监控 | 提高超时前先缩短链路或分层同步 |

真实工程例子：一组跨机训练任务在凌晨反复报 `watchdog timeout`。最初大家以为是 RoCE 网络抖动，于是把超时从 10 分钟调到 30 分钟，结果只是更晚报错。后面对 12 台机器做时间巡检，发现其中 1 台 `Reference ID` 与其他节点不同，且 `Last offset` 一直在毫秒级摆动。修复 NTP 上游配置后，训练恢复正常。这类问题里，“超时”是表现，不是根因。

一个实用经验是：如果你怀疑时钟问题，不要只看“当前 offset 很小”，还要看 `Residual freq`。因为 offset 小只代表“现在碰巧对齐”，频率误差大则意味着它很快又会偏掉。

---

## 替代方案与适用边界

如果硬件不支持 PTP，不代表没法把事情做好。关键不是追求绝对最小误差，而是先定义“我的训练系统能接受多大时间不确定性”。

| 方案 | 适用条件 | 缺点 |
|---|---|---|
| NTP + Chrony | 普通 CPU/GPU 服务器、无 PTP 交换机 | 精度受网络和软件抖动影响更大 |
| PTP + 硬件时间戳 | NIC、交换机、驱动都支持 | 成本高，部署复杂 |
| NTP + 更严格巡检 | 已有集群不方便改硬件 | 只能把风险压低，不能达到子微秒 |
| 适度延长训练超时 | 已确认只是链路慢，不是时钟漂移或死锁 | 可能掩盖真实问题 |

没有 Transparent Clock 的环境里，可以继续用 Chrony，但要把策略收紧。例如限制可接受延迟源、持续观察频率稳定性。配置思想可以类似：

```conf
server ntp-a.example.com iburst maxdelay 0.1
server ntp-b.example.com iburst maxdelay 0.1
makestep 1.0 3
rtcsync
```

这里 `maxdelay 0.1` 的意思是“往返延迟太大的源就别信”。白话说，不让高抖动上游进入候选集。它不是万能药，但能减少“坏时间源偶尔被选中”的概率。

什么时候必须上 PTP？

- 你需要子微秒级对齐
- 你已经证明 NTP 链路是健康的，但应用仍对时序极度敏感
- 你能控制 NIC、交换机和驱动版本

什么时候 NTP 足够？

- 普通多机训练
- 主要需求是日志、监控、任务调度一致
- 你能接受微秒到毫秒级的不确定性，并愿意做定期巡检

---

## 参考资料

1. Chrony FAQ：`root_distance = root_delay / 2 + root_dispersion` 的公式来源，也解释了 root dispersion 与系统最大误差估计的意义。https://chrony-project.org/faq
2. Ubuntu Server 文档 `Synchronize time using chrony`：`timedatectl status`、`chronyc tracking`、`chronyc sources` 的检查方法与典型输出。https://ubuntu.com/server/docs/how-to/networking/chrony-client/
3. Ubuntu Server 文档 `About time synchronisation`：说明 `chrony` 与 `timedatectl/timesyncd` 的角色边界，适合理解 Ubuntu 上谁在真正管时间。https://documentation.ubuntu.com/server/explanation/networking/about-time-synchronisation/
4. NTP.org `RFC 1129`：给出互联网环境下 NTP 常见可达到“几十毫秒”量级的历史工程结论，用来支撑公网 NTP 的误差边界。https://www.ntp.org/reflib/rfc/rfc1129/
5. NTP.org `Executive Summary: Computer Network Time Synchronization`：说明 NTP 精度依赖环境，普通条件下是毫秒级，配合 PPS 可到微秒级。https://www.ntp.org/reflib/exec/
6. NTP.org `IEEE 1588 Precision Time Protocol (PTP)`：给出 PTP 在专用硬件条件下约 100 ns 量级的精度预期，对比 NTP 的环境依赖性。https://www.ntp.org/reflib/ptp/
7. Red Hat 文档 `PTP`：明确指出 PTP 在硬件支持下可达到 sub-microsecond，支撑“PTP 适合更高精度同步”的工程判断。https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/6/html/migration_planning_guide/sect-migration_guide-networking-ptp
8. PyTorch 官方 `torch.distributed` 文档：`init_process_group(timeout=...)` 当前默认值与 NCCL 后端超时行为说明。https://docs.pytorch.org/docs/stable/distributed.html
9. NVIDIA OSMO `NCCL Errors`：watchdog timeout 的定义、常见诱因以及调试思路，用于训练侧排障。https://nvidia.github.io/OSMO/release/6.0/user_guide/troubleshooting/nccl.html
