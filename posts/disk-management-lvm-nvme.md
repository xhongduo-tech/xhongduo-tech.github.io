## 核心结论

磁盘管理不是单独选一个工具，而是把三层问题拆开处理：LVM 负责“怎么灵活分配容量”，RAID 负责“怎么在性能和冗余之间取舍”，NVMe 调优负责“怎么把硬件并行能力真正用出来”。

对大多数 Linux 服务器，稳定的实践顺序是：先确定 RAID 级别，再把 RAID 设备交给 LVM 管理，最后对文件系统和 NVMe 队列做调优。这样做的原因很直接：RAID 先决定底层可靠性和可用容量，LVM 再在这个基础上提供在线扩容，NVMe 调优则解决高并发下吞吐和延迟不稳定的问题。

一个实用结论是：写密集、低延迟、业务不能停的场景，优先考虑 `RAID 10 + LVM + NVMe + 定期 fstrim`。其中 RAID 10 提供镜像冗余和较稳定写性能，LVM 提供在线扩容，NVMe 通过 PCIe 多通道和深队列提高并行，`fstrim` 则把“哪些块已经不用了”告诉 SSD 控制器，减少写放大。写放大就是 SSD 为了完成一次逻辑写入，实际在内部做了更多物理写入，这会拖慢性能并消耗寿命。

必须先记住两个数量级公式：

$$
\text{RAID 5 可用容量} \approx (N - 1) \times \text{单盘容量}
$$

$$
\text{RAID 10 可用容量} \approx 50\% \times \text{原始总容量}
$$

NVMe 带宽也有硬上限，它首先受 PCIe 通道数约束。常见近似值可以直接背下来：

| 接口 | 理论带宽近似 |
|---|---:|
| PCIe Gen3 x4 | 3.9 GB/s |
| PCIe Gen4 x4 | 7.9 GB/s |

这意味着，如果硬件只有 Gen3 x4，再怎么调队列，单盘顺序吞吐也不可能突破这一级别。

---

## 问题定义与边界

磁盘管理要解决的不是“磁盘能不能用”，而是三个更具体的问题：

| 目标 | 典型问题 | 主要工具 |
|---|---|---|
| 容量扩展 | 分区不够大，业务不能停机 | LVM |
| 性能与冗余 | 单盘太慢，或坏盘会丢数据 | RAID |
| SSD 持续性能 | 刚上线很快，跑久后写入变慢 | NVMe 调优 + `fstrim` |

LVM，全称 Logical Volume Manager，中文通常叫逻辑卷管理。白话解释：它把“硬盘空间”从“固定分区”变成“可再分配的资源池”。  
其中三层结构必须区分清楚：

| 缩写 | 全称 | 白话解释 |
|---|---|---|
| PV | Physical Volume | 物理卷，能加入 LVM 的底层磁盘或分区 |
| VG | Volume Group | 卷组，把多个 PV 聚成一个空间池 |
| LV | Logical Volume | 逻辑卷，从 VG 里切出来给文件系统使用 |

可以把它理解成下面这个流向：

```text
/dev/sdb   /dev/sdc
   |          |
   v          v
  PV1        PV2
     \      /
      \    /
        VG
      /    \
     v      v
   LV_root  LV_data
```

玩具例子：你有两块各 100GB 的磁盘。传统分区下，`/` 分区如果建成 80GB，后来不够了，另一块磁盘即使空着也不能直接借给它。LVM 下，这两块盘先变成 PV，再合进同一个 VG，最后 `/` 对应的 LV 可以在线吃掉 VG 里的空闲空间。

边界也必须明确：

| 方案 | 解决什么 | 不解决什么 |
|---|---|---|
| LVM | 在线扩容、灵活分配 | 不自带数据冗余 |
| RAID 1/5/10 | 冗余与吞吐 | 不能替代备份 |
| NVMe 调优 | 提升并发 I/O 利用率 | 不能突破 PCIe 上限 |
| `fstrim` | 回收空闲块信息 | 不能修复坏块或替代磨损均衡算法 |

“RAID 不是备份”必须单独强调。RAID 保证的是硬盘损坏时阵列还能继续工作，但误删文件、逻辑损坏、勒索软件加密，RAID 都会把错误同步到所有成员盘。

---

## 核心机制与推导

先看 LVM 的机制。LVM 把空间按 PE 分配，PE 是 Physical Extent，白话解释就是“LVM 内部最小分配块”。当新磁盘加入卷组时，本质上是给 VG 增加更多可用 PE；当 `lvextend` 扩容时，本质上是把这些空闲 PE 分给某个 LV。

控制链路很简单：

```text
新磁盘 -> pvcreate -> 成为 PV
PV -> vgextend -> 加入 VG
VG 空闲 PE -> lvextend -> 分给 LV
LV 变大 -> resize2fs/xfs_growfs -> 文件系统可见
```

假设原来 `centos/root` 的 LV 是 13.39 GiB，VG 中新增了一块盘，空闲 PE 总量对应 16 GiB。执行：

```bash
lvextend -l +100%FREE /dev/centos/root
```

意思是把 VG 里所有剩余 PE 全分配给这个 LV。于是容量变成：

$$
13.39 + 16 = 29.39\ \text{GiB}
$$

这就是“在线扩容”的核心推导，关键不在文件系统，而在 VG 中是否还有空闲 PE。

再看 RAID。RAID 是 Redundant Array of Independent Disks，白话解释是“把多块盘按某种规则组合成一个逻辑磁盘”。不同级别的差异，本质是条带、镜像、校验三种机制怎么组合。

| RAID 级别 | 机制 | 最少磁盘数 | 可用容量 | 读性能 | 写性能 | 容错 |
|---|---|---:|---:|---|---|---|
| RAID 0 | 条带 | 2 | $N \times S$ | 高 | 高 | 0 盘 |
| RAID 1 | 镜像 | 2 | $\frac{N \times S}{2}$ | 较高 | 中等 | 1 盘/镜像组 |
| RAID 5 | 条带 + 分布式校验 | 3 | $(N-1)\times S$ | 高 | 一般到较差 | 1 盘 |
| RAID 10 | 镜像 + 条带 | 4 | $\frac{N \times S}{2}$ | 高 | 高且稳定 | 每组可坏 1 盘 |

其中 RAID 5 的写入慢，根本原因是“读-改-写”校验罚款。一次小写入往往不是直接落盘，而是：

1. 先读旧数据块
2. 再读旧校验块
3. 计算新校验
4. 写新数据块
5. 写新校验块

所以它适合读多写少，不适合高频随机写。新手可理解的版本是：RAID 5 每次写入都像要顺手补一张校验单，写得越碎，额外手续越多。

NVMe 的核心机制不同。NVMe 是给闪存设计的协议，白话解释是“它允许主机和 SSD 之间同时挂很多读写请求，而不是像老协议那样排很长单队列”。它依赖 PCIe 通道和多队列模型：

```text
应用线程
   |
   v
io_uring 提交队列 SQ
   |
   v
内核 NVMe 驱动
   |
   v
硬件提交到控制器队列
   |
   v
SSD 完成后写入 CQ
   |
   v
应用从完成队列 CQ 收割结果
```

`io_uring` 可以理解成“用户态和内核之间一套共享队列接口”，目标是减少系统调用和上下文切换。它和 NVMe 配合的重点不是“能不能异步”，而是“高并发下能不能持续把硬件队列喂满”。

经验上，`io_uring` 的 `entries` 可以先设为预计并发请求数的 2 倍，原因是应用层提交速度和硬件完成速度不一致，需要有缓冲。若提交队列空间不足，就可能出现 `-EBUSY`，因此要关注类似 `io_uring_sq_space_left` 这种可用空间指标。

---

## 代码实现

先给出一个可运行的 Python 玩具例子，用来验证常见 RAID 容量公式。这里不依赖真实磁盘，重点是把计算规则写清楚。

```python
def raid_usable_tb(level: str, disk_count: int, disk_size_tb: float) -> float:
    assert disk_count > 0
    assert disk_size_tb > 0

    if level == "raid0":
        assert disk_count >= 2
        return disk_count * disk_size_tb
    if level == "raid1":
        assert disk_count >= 2 and disk_count % 2 == 0
        return (disk_count * disk_size_tb) / 2
    if level == "raid5":
        assert disk_count >= 3
        return (disk_count - 1) * disk_size_tb
    if level == "raid10":
        assert disk_count >= 4 and disk_count % 2 == 0
        return (disk_count * disk_size_tb) / 2
    raise ValueError("unsupported raid level")


assert raid_usable_tb("raid0", 4, 2) == 8
assert raid_usable_tb("raid1", 2, 2) == 2
assert raid_usable_tb("raid5", 4, 2) == 6
assert raid_usable_tb("raid10", 4, 2) == 4

print("all assertions passed")
```

LVM 在线扩容的最小命令链如下。前提是 `/dev/sdb` 是新盘，`centos` 是卷组名，`/dev/centos/root` 是逻辑卷名，文件系统是 ext4。

```bash
pvcreate /dev/sdb
vgextend centos /dev/sdb
lvextend -l +100%FREE /dev/centos/root
resize2fs /dev/centos/root
```

如果文件系统是 XFS，最后一步不是 `resize2fs`，而是：

```bash
xfs_growfs /
```

RAID 初始化示例，使用 `mdadm` 创建 RAID 10：

```bash
mdadm --create /dev/md0 \
  --level=10 \
  --raid-devices=4 \
  /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1
```

然后把 RAID 设备交给 LVM：

```bash
pvcreate /dev/md0
vgcreate vgdata /dev/md0
lvcreate -n lvdb -L 500G vgdata
mkfs.xfs /dev/vgdata/lvdb
mount /dev/vgdata/lvdb /data
```

真实工程例子可以这样落地：一个 OLTP 数据库主分区部署在 `RAID 10 + LVM` 上。数据库日志增长快且不可预测，因此初始只给 `lvdb` 分配 500G，保留部分 VG 空闲空间。业务增长时，再接入新盘扩 VG，然后在线扩容 LV，不中断数据库实例。

NVMe 调优常见观察点和参数如下：

| 参数 | 位置 | 作用 |
|---|---|---|
| scheduler | `/sys/block/nvme0n1/queue/scheduler` | I/O 调度器 |
| nr_requests | `/sys/block/nvme0n1/queue/nr_requests` | 软件层排队深度 |
| queue_depth | `/sys/block/nvme0n1/device/queue_depth` | 设备队列深度上限 |
| discard_max_bytes | `/sys/block/nvme0n1/queue/discard_max_bytes` | trim/discard 单次上限 |

查看命令：

```bash
cat /sys/block/nvme0n1/queue/scheduler
cat /sys/block/nvme0n1/queue/nr_requests
cat /sys/block/nvme0n1/device/queue_depth
```

常见做法是让 NVMe 使用更轻的调度器：

```bash
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler
echo 1023 | sudo tee /sys/block/nvme0n1/queue/nr_requests
```

`fstrim` 建议直接启用 systemd 定时器，而不是手写 cron：

```bash
systemctl enable fstrim.timer
systemctl start fstrim.timer
systemctl status fstrim.timer
```

立即手动执行一次：

```bash
fstrim -av
```

---

## 工程权衡与常见坑

最常见的错误，是把 LVM、RAID、文件系统三个层次混在一起。正确顺序通常是：先 RAID，再 LVM，再文件系统。因为如果先在多块裸盘上分别做 LVM，后面再想切换 RAID，会非常难迁移。

第二个坑，是误以为 RAID 5 很“省空间”所以适合所有场景。它的确容量利用率更高，但对随机写和重建期负载不友好。重建就是坏盘替换后重新恢复数据，白话解释是“阵列把缺失的数据重新算一遍并写回新盘”。这个过程会持续占用磁盘带宽和 CPU，业务高峰时延迟可能明显上升。

| 常见坑 | 现象 | 规避策略 |
|---|---|---|
| RAID 5 写罚款 | 小随机写延迟高 | 写密集业务改用 RAID 10 |
| RAID 重建冲击 | 重建时吞吐下降 | 预留性能余量，避开高峰换盘 |
| LVM 扩容后文件系统没扩 | `lsblk` 变大，`df -h` 不变 | 追加 `resize2fs` 或 `xfs_growfs` |
| NVMe 队列太浅 | IOPS 上不去 | 调整 `nr_requests` 和应用并发 |
| 长期不 trim | SSD 越跑越慢 | 启用 `fstrim.timer` |
| 把 RAID 当备份 | 误删后无法恢复 | 仍需快照或异地备份 |

第三个坑，是以为 SSD 会“自动处理一切”，所以不需要 `fstrim`。实际上文件删除时，文件系统只是把逻辑块标成可用，不一定立刻告诉 SSD 控制器这些块已经废弃。没有 trim，控制器会以为很多页仍然有效，垃圾回收更难做，写放大也更高。对长期运行的数据库、日志盘、镜像仓库，这个影响会累积出来。

第四个坑，是盲目把队列深度调得很大。队列深度不是越大越好。过高会增加排队等待时间，反而拉高尾延迟。尾延迟就是最慢那部分请求的延迟，通常比平均延迟更影响线上体验。调优必须结合业务模式：顺序大块 I/O、随机小块 I/O、读多写少、写多读少，最佳点都不同。

---

## 替代方案与适用边界

如果预算有限、磁盘数量不多，替代方案可以按业务特征选，而不是默认追求“最强配置”。

| 场景 | 推荐方案 | 原因 |
|---|---|---|
| 单机实验环境 | 单盘 + LVM + 外部备份 | 简单、成本低 |
| 两盘服务器 | RAID 1 + LVM | 有冗余，结构简单 |
| 读多写少文件服务 | RAID 5 + LVM | 容量利用率较高 |
| 写密集数据库 | RAID 10 + LVM | 写延迟更稳定 |
| 老内核环境 | `libaio` + 合理队列深度 | 无法充分使用 `io_uring` |
| 调度器不支持 `none` | `mq-deadline` | 多队列块层的保守选择 |

如果只有单盘，LVM 仍然有价值，因为它解决的是“空间管理”，不是“冗余”。这时可以用 LVM 快照配合外部备份。快照就是“某一时刻的数据视图”，白话解释是“先记住旧数据，后续改动另存”，适合备份窗口或短期回滚。但快照不是长期备份，更不是高性能常态方案。

`io_uring` 也不是绝对必须。如果内核较老，或者应用栈已经深度依赖 `libaio`，继续使用成熟方案也完全合理。它们的差别主要在系统调用开销、提交完成路径和高并发下的 CPU 效率，而不是“有无异步能力”。

可以用一个简单决策原则收尾：

1. 先问是否要抗坏盘。
2. 再问是读多还是写多。
3. 再问容量是否会在线增长。
4. 最后才调 NVMe 队列和 trim 周期。

顺序不能反。因为底层冗余模型一旦选错，后面再做调优，收益很有限。

---

## 参考资料

1. LVM 在线扩容与 PV/VG/LV 关系说明：提供逻辑卷管理层次和常见命令。
2. RAID 0/1/5/10 性能与容量综述：用于对比条带、镜像、校验的差异。
3. NVMe 与 PCIe 通道带宽说明：用于说明 Gen3 x4、Gen4 x4 的带宽上限。
4. Linux `fstrim` 与 SSD 优化资料：用于说明 trim 对写放大和寿命管理的意义。
5. `mdadm`、LVM、`xfs_growfs`、`resize2fs` 官方文档：用于核对实际运维命令语义。
