## 核心结论

AI 训练集群的存储架构，通常不是“选一个存储系统”，而是按访问模式分层组合。结论先给出来：

1. `NFS` 适合放个人配置、少量共享脚本、训练日志这类“需要共享，但吞吐不大”的目录。
2. `Ceph` 适合做可横向扩展的数据湖、checkpoint、镜像仓库，以及需要块存储、对象存储、文件系统三种接口并存的场景。
3. `Lustre` 或 `GPFS` 这类并行文件系统，适合数百到数千 GPU 同时读写训练数据、预处理结果和中间文件。
4. `MinIO/S3` 这类对象存储，适合海量数据集管理、异步上传下载、跨集群共享，以及云原生训练流程。

对新手可以这样理解：

| 存储类型 | 白话解释 | 最适合放什么 | 典型瓶颈 |
|---|---|---|---|
| NFS | 一台共享文件柜，大家都去同一个管理员那里取文件 | 配置、脚本、少量日志 | 单机带宽、单机元数据压力 |
| Ceph | 多台机器组成的仓库，系统自动决定数据放哪 | checkpoint、数据湖、通用共享存储 | 网络、CRUSH/副本开销、元数据规划 |
| Lustre/GPFS | 把一个大文件切成很多条，很多存储节点一起供数千客户端并行读 | 大规模训练数据、高并发中间文件 | 条带参数、元数据服务、网络拓扑 |
| MinIO/S3 | 每个对象单独编号放入桶里，天然适合并发拿取 | 数据集分发、归档、模型产物 | 小文件过多、单对象重命名不友好 |

如果必须用一句话概括：AI 训练集群要同时满足“共享性、扩展性、并行度、成本”四个目标，单一存储系统通常做不到，所以要分层。

---

## 问题定义与边界

“AI 训练集群的存储架构”，指的不只是磁盘，而是从训练节点看到的数据访问路径，包括：

- 用户目录、配置目录
- 训练数据集
- 预处理后的缓存数据
- 模型 checkpoint
- 日志、监控、实验结果
- 长期归档的数据湖

这里先把边界讲清楚。不是所有数据都应该放在同一个地方。

一个典型的 1024 GPU 集群，可以这样划分：

- `/home`、`/config`、启动脚本、环境文件放 `NFS`
- 原始数据集、长期保存的 checkpoint 放 `Ceph` 或 `MinIO/S3`
- 高并发训练时直接读取的样本、预处理结果、临时中间文件放 `Lustre/GPFS`
- 训练完成后的模型包、评测结果、跨团队共享产物再回写到对象存储

可以把访问链路理解成下面这个简化流程：

`NFS(配置入口) -> Ceph/MinIO(长期数据湖与checkpoint) -> Lustre/GPFS(训练期高并发数据面)`

这不是“谁先进谁落后”的关系，而是职责分层。

一个玩具例子：

- 4 张 GPU 做图像分类
- 配置文件总共 10 MB
- 数据集 50 GB
- 每小时保存一次 2 GB checkpoint

此时可以把配置文件放在 NFS，把数据集和 checkpoint 放在 Ceph，甚至直接全放 Ceph 也能跑，因为并发规模不大。

一个真实工程例子：

- 1024 张 GPU 训练多模态大模型
- 原始数据数 PB 级
- 训练阶段每秒数十 GB 到数百 GB 聚合读取
- 数千个进程同时访问样本和写日志

这时如果仍然让所有训练进程直接打到 NFS，元数据和带宽都会先崩。通常做法是：

- NFS 只保留小规模共享目录
- Ceph/对象存储做长期存储和冷数据层
- Lustre/GPFS 承担训练窗口内的热数据高并发访问

所以本文边界很明确：讨论的是 AI 训练集群里的共享存储分层，不讨论单机 NVMe 本地缓存、数据库系统、也不展开云厂商托管 NAS 产品细节。

---

## 核心机制与推导

先看为什么 NFS 很容易成为瓶颈。

`NFS` 是网络文件系统，白话讲就是“由一台或少数几台服务器把目录导出给很多客户端”。客户端看起来像本地目录，但绝大多数请求最终都会汇聚到服务端。于是聚合吞吐可以近似写成：

$$
B_{total} \approx \min(B_s,\sum_{i=1}^{n} B_{client_i})
$$

其中：

- $B_s$ 是 NFS 服务端的可提供总带宽或总 IOPS
- $B_{client_i}$ 是第 $i$ 个客户端能发起的吞吐

这条式子的意思很直接：客户端再多，最终也不可能超过服务端上限。

如果有 100 个客户端，每个都想读 500 MB/s，那么理论需求是：

$$
\sum_{i=1}^{100} B_{client_i} = 50\ \text{GB/s}
$$

但如果 NFS 服务器只有 25 GbE 网络和有限磁盘阵列，真实 $B_s$ 可能只有 2 到 3 GB/s，那总吞吐还是卡在 2 到 3 GB/s。客户端越多，单个客户端分到的份额越小，而且小文件会在元数据操作上排队。

对新手可以这样理解：NFS 像一根总水管，大家都从这根管子接水；客户端再多，不会凭空长出更多总供水能力。

再看 `Ceph`。Ceph 的底层核心叫 `RADOS`，可以把它理解成“负责把对象分散到很多磁盘节点上的分布式对象层”。它的关键不是中央目录服务器决定每个对象在哪，而是用 `CRUSH` 算法算出放置位置：

$$
placement = CRUSH(object\_id,\ pool\_map)
$$

白话解释：给定对象编号和存储池规则，客户端和集群都能算出对象应该落在哪些 OSD 上。`OSD` 是对象存储守护进程，简单说就是“真正保存数据的存储节点进程”。

如果副本数设为 3，那么一个对象会被放到 3 个 OSD 上，而且通常分散在不同主机或不同机架，以抗单点故障。好处是：

- 数据路径分散，不必所有读写都先经过同一台存储头节点
- 扩容时加 OSD 就能提高容量与并行度
- 文件接口 `CephFS` 还可以把元数据和数据分开管理

`CephFS` 之所以能提供类 POSIX 文件系统语义，是因为引入了 `MDS`。`MDS` 是元数据服务器，白话讲就是“专门管目录树、文件名、权限、路径映射”。数据块仍然直接存到 OSD。于是形成了分工：

| 组件 | 作用 | 性能影响点 |
|---|---|---|
| MDS | 管文件名、目录、权限、inode | 小文件、目录遍历、rename 很敏感 |
| OSD | 管对象数据读写与副本 | 吞吐、恢复、重平衡、网络流量 |
| MON | 管集群状态与成员信息 | 一般不是数据面瓶颈 |

再看 `Lustre/GPFS`。这类并行文件系统的核心思路是“条带化”。条带化就是把一个文件切成多个连续片段，分布到多个存储目标并行读写。简化表示为：

$$
file \rightarrow \{stripe_1, stripe_2, ..., stripe_k\}
$$

如果条带大小为 $L$，条带数为 $k$，则文件会按长度 $L$ 为单位依次落到 $k$ 个目标上循环写入。可近似理解为：

$$
B_{file} \approx \sum_{j=1}^{k} B_{OST_j}
$$

这里 `OST` 是对象存储目标，白话讲就是“并行文件系统里真正装文件数据的存储终点”。

这意味着：

- 大文件顺序读写时，可以把吞吐摊到多个 OST
- 很多 GPU 同时读取同一批大文件时，并行度更高
- 合理设置条带大小和条带数很关键

新手版理解：Lustre/GPFS 不是让 1000 个 GPU 去挤一扇门，而是把文件拆成很多段，让它们同时走 20 扇门。

最后看对象存储 `MinIO/S3`。对象存储不强调目录层级语义，而强调“对象 + 键”。每个对象可独立上传下载，请求天然容易分散。对于大对象，还能用 multipart upload，即把一个对象拆成多个 part 并发上传。于是对于大模型 checkpoint 或海量数据包，对象存储往往比传统文件系统更适合做长期保存和跨集群共享。

但对象存储不擅长强 POSIX 语义，比如频繁 append、细粒度随机覆盖、目录级 rename。这也是为什么训练集群里它通常和并行文件系统互补，而不是完全替代。

---

## 代码实现

先看一组最小命令例子，目的是说明“为什么这个目录挂到这个存储上”。

```bash
# 1) NFS：给所有训练节点共享配置与脚本
mount -t nfs nfs-server:/export/config /mnt/config

# 2) Ceph：创建训练数据池，副本和PG需要按集群规模规划
ceph osd pool create training-data 128

# 3) CephFS：把统一文件视图挂到客户端
mount -t ceph mon1,mon2,mon3:/ /mnt/cephfs -o name=client.train,secretfile=/etc/ceph/keyring

# 4) Lustre：为训练目录设置条带参数，适合大文件并发读取
lfs setstripe -c 8 -S 16M /mnt/lustre/datasets

# 5) MinIO/S3：把checkpoint上传到对象存储做长期保留
mc cp /mnt/lustre/checkpoints/epoch10.pt minio/ml-checkpoints/run-2026-03-20/
```

这些命令背后的选择逻辑是：

- `NFS` 放 `/mnt/config`，因为配置目录小、共享方便、不会长期顶满吞吐
- `CephFS` 提供统一共享视图，适合多业务共用的可扩展目录
- `Lustre` 用 `setstripe` 调优，是因为训练时要追求高并发数据面
- `MinIO` 负责产物归档，是因为对象存储天然适合跨集群和长期保存

下面给一个可运行的 Python 玩具程序，模拟“根据数据大小和并发数选择存储路径”。这不是生产调度器，但能体现基本思路。

```python
from dataclasses import dataclass

@dataclass
class JobProfile:
    dataset_size_gb: int
    num_gpus: int
    small_files: bool
    need_s3_compat: bool

def choose_storage(job: JobProfile) -> dict:
    if job.need_s3_compat:
        return {
            "dataset": "s3://training-data/",
            "checkpoint": "s3://checkpoints/",
            "config": "/mnt/nfs/config"
        }

    if job.num_gpus <= 8 and job.dataset_size_gb < 100 and not job.small_files:
        return {
            "dataset": "/mnt/cephfs/dataset",
            "checkpoint": "/mnt/cephfs/checkpoints",
            "config": "/mnt/nfs/config"
        }

    if job.num_gpus >= 64 or job.small_files:
        return {
            "dataset": "/mnt/lustre/dataset",
            "checkpoint": "/mnt/cephfs/checkpoints",
            "config": "/mnt/nfs/config"
        }

    return {
        "dataset": "/mnt/cephfs/dataset",
        "checkpoint": "/mnt/cephfs/checkpoints",
        "config": "/mnt/nfs/config"
    }

toy = JobProfile(dataset_size_gb=50, num_gpus=4, small_files=False, need_s3_compat=False)
plan = choose_storage(toy)
assert plan["dataset"] == "/mnt/cephfs/dataset"
assert plan["config"] == "/mnt/nfs/config"

real = JobProfile(dataset_size_gb=2000, num_gpus=1024, small_files=True, need_s3_compat=False)
plan = choose_storage(real)
assert plan["dataset"] == "/mnt/lustre/dataset"
assert plan["checkpoint"] == "/mnt/cephfs/checkpoints"

cloud_native = JobProfile(dataset_size_gb=500, num_gpus=128, small_files=False, need_s3_compat=True)
plan = choose_storage(cloud_native)
assert plan["dataset"].startswith("s3://")
assert plan["config"] == "/mnt/nfs/config"
```

如果把它放到训练脚本里，结构通常类似下面这样：

```python
def load_dataset(path: str):
    if path.startswith("s3://"):
        return f"load by s3 client: {path}"
    if path.startswith("/mnt/lustre"):
        return f"load by parallel fs: {path}"
    if path.startswith("/mnt/cephfs"):
        return f"load by cephfs: {path}"
    return f"load by local fs: {path}"

storage = choose_storage(
    JobProfile(dataset_size_gb=2000, num_gpus=256, small_files=False, need_s3_compat=False)
)
result = load_dataset(storage["dataset"])
assert "lustre" in result.lower() or "cephfs" in result.lower()
```

一个真实工程例子可以这样理解：

- 预训练前，把原始样本包放在 `MinIO/S3`
- 预处理作业从对象存储批量拉取，生成按 shard 切分的训练样本
- 这些训练样本落到 `Lustre/GPFS`
- 训练过程从并行文件系统读取
- checkpoint 周期性写回 `Ceph` 或 `S3`
- 配置、tokenizer、小型词表仍然放在 `NFS`

这个流程的关键，不是“统一接口”，而是“热数据和冷数据分开”。

---

## 工程权衡与常见坑

最常见的错误，是把“能挂载”误认为“适合承载训练主流量”。

下面直接列坑。

| 坑 | 影响 | 规避办法 |
|---|---|---|
| 数千进程同时往 NFS 同一目录写小文件 | 元数据排队，吞吐暴跌，训练抖动 | NFS 只放配置和小规模共享目录；日志聚合写入；减少文件数量 |
| Ceph 只扩 OSD 不管 MDS | 小文件、目录遍历、rename 变慢 | CephFS 要分开看数据面和元数据面，独立扩容 |
| CRUSH 规则不合理，热点集中在少数 OSD | 局部盘满、局部网络打爆 | 按机架、主机做 failure domain；观察 PG 和 OSD 分布 |
| Lustre 条带数设太小 | 大文件吞吐不够 | 大文件训练集提高 `stripe_count` |
| Lustre 条带数设太大 | 小文件反而增加调度和元数据成本 | 小文件目录不要盲目大条带 |
| MinIO/S3 单连接上传超大 checkpoint | 上传延迟高，失败重试成本大 | 开启 multipart upload，并发 part 上传 |
| 训练样本是千万级小文件 | 任意后端都会被 metadata 拖累 | 预先打包成 shard，例如 tar、webdataset、record 格式 |

给一个简单伪配置，表达这些调优方向：

```ini
# NFS: 不把高并发样本目录放这里
[nfs]
use_for = config, scripts, small_logs
avoid_for = massive_small_files, hot_training_data

# CephFS: 元数据与数据独立规划
[cephfs]
mds_count = 2
data_pool = train-data
metadata_pool = train-metadata

# Lustre: 大文件目录开启更高条带
[lustre]
dataset_stripe_count = 8
dataset_stripe_size = 16M

# MinIO: 大对象分片上传
[minio]
multipart = true
part_size = 128M
parallel_upload = 8
```

这里再强调一个容易被忽略的点：AI 训练集群里“文件数量”经常比“总容量”更先出问题。

比如 100 TB 数据，如果是 100 个 1 TB 文件，元数据压力不大；但如果是 100 亿个 10 KB 文件，任何共享存储都会被目录扫描、stat、open、close 拖慢。解决方案通常不是换一种存储，而是改变数据组织方式，把小文件打包成大 shard。

所以存储架构和数据格式是耦合的。只谈后端，不谈样本组织，结论通常不完整。

---

## 替代方案与适用边界

`CephFS`、`Lustre/GPFS`、`MinIO/S3` 有重叠，但重点不同。

| 方案 | 优势 | 弱点 | 更适合的场景 |
|---|---|---|---|
| CephFS | 通用、可扩展、和块/对象接口统一 | 极高并发训练数据面通常不如专用并行文件系统 | 企业统一存储平台、中大型训练集群 |
| Lustre/GPFS | 面向 HPC 并行 I/O，训练数据吞吐强 | 运维和调优门槛高，云原生接口弱 | 万卡级训练、HPC 中心 |
| MinIO/S3 | 对象接口标准化，跨集群共享方便 | 不适合强 POSIX 语义和大量小随机写 | 数据湖、归档、checkpoint、分发 |

可以按条件直接决策：

1. 如果你已经有 HPC 集群，且训练数据读取是主瓶颈，优先 `Lustre/GPFS` 承担热数据面。
2. 如果你希望一套系统同时提供对象、块、文件三种能力，优先考虑 `Ceph`。
3. 如果你的流程偏云原生、Kubernetes、异步任务编排，`MinIO/S3` 会更自然。
4. 如果只是 4 到 16 卡的小集群，`NFS + Ceph` 通常就够了，不必一开始就上并行文件系统。
5. 如果已有 `Lustre`，`Ceph` 可以作为长期数据湖和 checkpoint 层，而不是硬替代。
6. 如果已有 `Ceph`，也不意味着它一定能完全替代 `Lustre`，特别是在万卡并行训练的数据面上。

一个常见互补关系是：

- `Ceph RGW` 提供 S3 兼容接口，满足对象存储访问
- `CephFS` 提供文件系统视图，满足通用共享需求
- `Lustre` 提供训练期的高并发热路径

这说明真正的工程目标不是“统一成一个系统”，而是“把最贵的并发路径交给最合适的系统”。

---

## 参考资料

- Ceph 官方文档: https://docs.ceph.com/en/latest/cephfs/index.html  
  说明 CephFS、MDS、RADOS 的基本机制，适合理解 Ceph 文件系统是如何建立在对象层之上的。

- Ceph CRUSH 相关文档: https://docs.ceph.com/en/latest/rados/operations/crush-map/  
  说明对象如何根据规则映射到 OSD，适合理解为什么 Ceph 能横向扩展且避免中心放置表。

- LLNL 并行文件系统说明: https://hpc.llnl.gov/hardware/file-systems/parallel-file-systems  
  展示大型 HPC 环境中的 Lustre 容量与用途，适合理解并行文件系统在真实超算中心中的角色。

- LLNL 文件系统使用指南: https://hpc.llnl.gov/documentation/user-guides/using-lc-file-systems  
  说明 NFS 目录与并行文件系统目录如何在实际集群中分工使用。

- Weka 关于 AI/ML 客户端访问模式的文章: https://www.weka.io/blog/ai-ml/fit-for-purpose-part-two-client-access/  
  适合理解 NFS、并行文件系统、对象存储在 AI/ML 负载下的访问差异与瓶颈位置。

- MinIO 面向 AI 工作负载的资料: https://min.io/solutions/ai-data-infrastructure  
  说明对象存储为何适合数据湖、模型产物、并发分发与 S3 兼容生态。
