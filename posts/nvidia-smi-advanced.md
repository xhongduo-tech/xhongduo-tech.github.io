## 核心结论

`nvidia-smi` 是 NVIDIA System Management Interface，白话说就是 GPU 的“状态检查台”。它底层调用 NVML，也就是 NVIDIA 的管理库，所以适合做单机实时诊断；到了多卡集群、Kubernetes、MIG 多租户场景，应该把它当成入口工具，而不是唯一监控系统。

最重要的两个字段是 `GPU-Util` 和 `Memory-Util`。前者表示采样窗口内“至少有一个 kernel 在执行”的时间占比，后者表示同一窗口内“显存有读写”的时间占比。它们不是瞬时值，而是窗口平均值，因此判断瓶颈时要先看采样窗口，再看比例关系。可写成：

$$
Util = 100\% \times \frac{\text{活跃时间}}{\text{采样窗口}}
$$

更具体地：

$$
GPU\text{-}Util = 100\% \times \frac{\text{ActiveKernelTime}}{\text{SamplingWindow}}
$$

$$
Memory\text{-}Util = 100\% \times \frac{\text{ActiveMemoryOpsTime}}{\text{SamplingWindow}}
$$

玩具例子：如果采样窗口是 1 秒，kernel 执行了 0.6 秒，显存读写只发生了 0.15 秒，那么 `GPU-Util=60%`，`Memory-Util=15%`。这说明负载更像“计算活跃、显存压力较低”，而不是显存带宽打满。

MIG 是 Multi-Instance GPU，白话说就是把一张大 GPU 切成几张逻辑上独立的小 GPU。启用 MIG 后，观测粒度改变了：你不再只关心整卡，还要关心每个 GPU Instance 或 Compute Instance。对 A100/A30 这类 Ampere MIG，官方文档明确说明 `nvidia-smi` 不能直接给出 MIG 设备的 `GPU-Util`/`Memory-Util`，这时应切到 DCGM 或 `dcgm-exporter`。

---

## 问题定义与边界

这篇文章解决的是三个常见问题：

1. `nvidia-smi` 屏幕上那么多字段，哪些是真正能指导排障的。
2. 单机上如何持续采样，判断是算力瓶颈、显存瓶颈，还是调度问题。
3. A100 开启 MIG 后，为什么原来的观察方法失效，以及应该换什么工具。

先把边界讲清楚：

| 场景 | 观测对象 | 首选工具 | 能看到什么 | 关键限制 |
|---|---|---|---|---|
| 单卡/普通多卡 | 整张 GPU | `nvidia-smi` / `nvidia-smi dmon` | 温度、功耗、显存、GPU/Memory 利用率 | 适合单机，不适合统一集群视图 |
| MIG 启用 | GPU 实例/计算实例 | `dcgmi dmon` / `dcgm-exporter` | MIG 粒度的活动度、DRAM 活动、实例标签 | A100/A30 上 `nvidia-smi` 不能直接报 MIG 利用率 |
| 集群级监控 | 节点 + GPU + MIG 实例 | DCGM + Prometheus | 长期时序、告警、K8s 关联 | 需要部署 `nv-hostengine` 或 exporter |

一个新手常见误判是：A100 开了 MIG，`nvidia-smi` 看到 `GPU-Util` 变成 `N/A` 或看不到实例级利用率，就以为作业没跑。这个结论通常是错的，正确理解是“默认观测口失效了”。

可以用一个简单流程判断：

`普通 GPU` -> `nvidia-smi / dmon` 即可  
`启用 MIG` -> 先看是否只需要显存占用  
`需要实例级利用率` -> 切到 `dcgmi dmon` 或 `dcgm-exporter`  
`需要跨节点汇总` -> 直接上 DCGM 体系

字段里还有两个常被误解的点：

| 字段 | 正确定义 | 常见误解 |
|---|---|---|
| `Fan` | 目标风扇转速百分比，不一定等于物理实际转速 | 误以为它一定是真实 RPM 比例 |
| `Volatile Uncorr. ECC` | 自驱动加载以来的不可纠正 ECC 错误计数 | 误以为它是设备全生命周期累计值 |

`volatile` 的意思是“易失计数”，白话说就是会重置的计数；`aggregate` 才更接近生命周期累计。

---

## 核心机制与推导

先看 `GPU-Util` 与 `Memory-Util` 的差异：

| 指标 | 分子 | 分母 | 典型高值含义 |
|---|---|---|---|
| `GPU-Util` | 至少一个 kernel 活跃的时间 | 采样窗口长度 | 计算管线忙 |
| `Memory-Util` | 显存读写活跃的时间 | 采样窗口长度 | 显存访问忙 |
| `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | 图形/计算引擎活跃比例 | DCGM 采样窗口 | 更适合集群和 MIG |
| `DCGM_FI_PROF_DRAM_ACTIVE` | DRAM 活跃周期比例 | DCGM 采样窗口 | 更适合带宽诊断 |

这里“采样窗口”是关键。官方文档说明，`nvidia-smi` 的利用率窗口通常是 1 秒，部分产品可到 1/6 秒。窗口越短，数值越抖；窗口越长，越平滑，但越容易掩盖尖峰。

继续用玩具例子推导：

- 采样窗口：1s
- kernel 活跃：0.6s
- 显存读写：0.15s

所以：

$$
GPU\text{-}Util = 100\% \times \frac{0.6}{1.0} = 60\%
$$

$$
Memory\text{-}Util = 100\% \times \frac{0.15}{1.0} = 15\%
$$

可以把这一秒想成时间轴：

`0.00s - 0.60s`：有 kernel 执行  
`0.60s - 0.75s`：其中一部分时间发生显存读写  
`0.75s - 1.00s`：GPU 空闲

因此你看到 60/15，不表示每一刻都这样，而是这一秒的平均占比。

命令行上，`-l` 或 `-lms` 控制你“多久打印一次”；它不改变硬件计数定义，但会改变你观察到的波动程度。例如：

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -lms 1000
```

如果输出一行是：

```text
2026/03/20 10:00:01.123, 0, 60 %, 15 %, 12000 MiB, 248.31 W
```

可以直接把 `60%` 理解为上一采样周期里“kernel 活跃 0.6 秒”，把 `15%` 理解为“显存读写活跃 0.15 秒”。

真实工程例子：训练集群里某张卡 `memory.used` 长期 70GB，但 `GPU-Util` 只有 18%，`Memory-Util` 也只有 10% 左右。这通常不是“显存带宽打满”，而是“模型占满显存但计算流不连续”，常见原因是数据加载慢、CPU 预处理卡住、通信等待或 micro-batch 太小。只看“显存用了很多”会误判。

MIG 下的机制又多一层。DCGM 会同时给两种视图：

- GPU device-level metrics：整卡视角
- MIG device metrics：实例视角

例如一张 A100 被切成两个 `3g.20gb` 实例。官方示例里，单个实例的 `GRACT` 接近 `0.986`，整卡只有 `0.845`。原因不是整卡变慢，而是实例只占整卡部分 SM。DCGM 的整卡值本质上是“实例活跃度 × 资源占比”的聚合结果。

---

## 代码实现

单机持续采样最直接的方法是 `dmon`。`dmon` 是 device monitor，白话说就是滚动采样模式。

```bash
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/gpu-dmon}"
INTERVAL="${2:-1}"

mkdir -p "$OUT_DIR"

while true; do
  ts="$(date +%Y%m%d-%H%M%S)"
  nvidia-smi dmon -s u -d "$INTERVAL" -c 5 --format csv,nounit,noheader > "$OUT_DIR/$ts.csv"
  sleep 1
done
```

这个脚本每次切一份小日志，适合事后把故障时间段单独回放。`-s u` 表示采样利用率组，`-d` 是间隔秒数，`-c` 是采样次数。

参数含义如下：

| 参数 | 含义 | 常用值 |
|---|---|---|
| `-s u` | 采样利用率组 | `u` / `p` / `c` / `m` |
| `-d` | 采样间隔，秒 | `1` 或 `2` |
| `-c` | 本轮采样次数 | `5`、`60` |
| `-o D` / `-o T` | 追加日期/时间 | 诊断时建议开启 |
| `--format csv,nounit,noheader` | 便于脚本解析 | 推荐 |

如果你只想查某几列，查询模式更稳：

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw \
  --format=csv,noheader,nounits -lms 1000
```

下面给一个可运行的 Python 小工具，输入几行 CSV，自动判断更像计算瓶颈还是显存瓶颈：

```python
from io import StringIO
import csv

sample = """2026/03/20 10:00:01,0,60,15,12000
2026/03/20 10:00:02,0,58,14,12020
2026/03/20 10:00:03,0,62,16,12010
"""

def classify(rows):
    gpu = sum(float(r["util_gpu"]) for r in rows) / len(rows)
    mem = sum(float(r["util_mem"]) for r in rows) / len(rows)
    if gpu >= 50 and mem <= 25:
        return "compute-heavy"
    if mem >= 50 and gpu <= 40:
        return "memory-heavy"
    return "mixed-or-waiting"

reader = csv.DictReader(
    StringIO(sample),
    fieldnames=["ts", "gpu_id", "util_gpu", "util_mem", "mem_used_mb"]
)
rows = list(reader)

result = classify(rows)
assert result == "compute-heavy"
print(result)
```

MIG 场景不要继续迷信 `nvidia-smi` 首页。直接切到 DCGM：

```bash
dcgmi discovery -c
dcgmi group -c mig-watch -a 0,i:0,i:1
dcgmi dmon -g 8 -e 1001,1004,1005
```

这里 `1001` 是图形/计算引擎活动，`1004` 是 Tensor 活动，`1005` 是 DRAM 活动。它们比 `nvidia-smi` 更适合 MIG 和集群。

集群接 Prometheus 时，建议让 `dcgm-exporter` 和 `dcgmi` 连接同一个 `nv-hostengine`，这样单机排障和时序监控口径一致：

```yaml
scrape_configs:
  - job_name: dcgm-exporter
    scrape_interval: 15s
    static_configs:
      - targets: ['gpu-node-1:9400', 'gpu-node-2:9400']
```

```bash
docker run -d --rm \
  --gpus all \
  --net host \
  --cap-add SYS_ADMIN \
  nvcr.io/nvidia/k8s/dcgm-exporter:<version> \
  -c 1000 \
  -f /etc/dcgm-exporter/dcp-metrics-included.csv
```

---

## 工程权衡与常见坑

| 坑 | 现象 | 正确处理 |
|---|---|---|
| MIG 开启后看不到实例级 `GPU-Util` | 误以为实例没跑 | 用 `dcgmi dmon` 或 `dcgm-exporter` |
| 只看 `memory.used` | 误判成显存瓶颈 | 联合看 `GPU-Util`、`Memory-Util`、功耗 |
| `volatile` ECC 当成生命周期累计 | 故障判断失真 | 同时看 `aggregate` |
| `Fan` 当真实风扇转速 | 温控判断失真 | 把它当“目标风扇占比” |
| `dmon` 设备数过多 | 部分卡没数据 | 按卡分组采样，或转 DCGM |
| DCGM 与 Nsight 同时开 Profiling | 报 resource in use | 先暂停 DCGM profiling，再跑 Nsight |

MIG 的排障流程可以固定下来：

`发现 nvidia-smi 没有实例利用率`  
-> `确认是否启用了 MIG`  
-> `停止依赖 dmon 的 util 字段`  
-> `用 dcgmi discovery -c 看实例枚举`  
-> `用 dcgmi group + dcgmi dmon 看实例级指标`

还要特别纠正一个容易被旧资料误导的点：`nvidia-smi vgpu --gpm-metrics` 不是所有 MIG 场景都能救场。官方文档说明，在 Ampere 架构的 MIG-backed vGPU 上，这个能力因为硬件限制并不支持。所以对 A100/A30，实例级利用率监控的稳定做法仍然是 DCGM，而不是赌 `vgpu` 子命令。

---

## 替代方案与适用边界

| 方案 | 适用场景 | 优点 | 限制 |
|---|---|---|---|
| `nvidia-smi` | 单机快速排障 | 零门槛、系统自带 | 不适合集群汇总 |
| `nvidia-smi dmon` | 单机持续采样 | 输出规整，便于日志切片 | MIG 利用率能力有限 |
| DCGM / `dcgmi dmon` | 集群、MIG、多卡 | 支持实例级与统一时序 | 需要部署组件 |
| `dcgm-exporter` + Prometheus | 长期监控与告警 | 适合 K8s 和多节点 | 需要指标治理 |
| Nsight Systems / Compute | 细粒度性能剖析 | 能看到 kernel、通信、时间线 | 开销更高，不适合常驻 |
| PyNVML 自定义采集 | 定制报表 | 灵活 | 维护成本高，容易口径漂移 |

实际选型很简单：

- 你在单机上追一次偶发抖动：先用 `nvidia-smi` 和 `dmon`。
- 你在 A100 MIG 环境下做多租户观测：直接用 DCGM。
- 你需要告警、历史趋势、Kubernetes 关联：用 `dcgm-exporter`。
- 你已经知道“利用率低”，但不知道具体卡在哪个 kernel 或哪个通信阶段：再上 Nsight。

也就是说，`nvidia-smi` 负责“发现异常”，DCGM 负责“持续监控”，Nsight 负责“深入剖析”。三者不是替代关系，而是层级关系。

---

## 参考资料

- NVIDIA `nvidia-smi` 官方手册：字段定义、`dmon` 参数、`GPU-Util`/`Memory-Util` 采样说明。  
  https://docs.nvidia.com/deploy/nvidia-smi/index.html

- NVIDIA DCGM User Guide / Feature Overview：`dcgmi dmon`、profiling 指标、MIG 指标归因、与 Nsight 的冲突说明。  
  https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html

- NVIDIA DCGM Exporter 文档：`dcgm-exporter` 部署方式、采样间隔、MIG 标签与 Prometheus 集成。  
  https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html

- NVIDIA MIG User Guide：MIG 模式启用、实例配置、`nvidia-smi` 在 MIG 模式下的展示边界。  
  https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html

- NVIDIA NVML API 文档：如果要自己写采集器，应直接参考 NVML，而不是依赖 `nvidia-smi` 文本输出。  
  https://docs.nvidia.com/deploy/nvml-api/index.html

- NVIDIA vGPU / AI Enterprise 文档：`nvidia-smi vgpu --gpm-metrics` 的适用条件，以及 Ampere MIG-backed vGPU 的限制。  
  https://docs.nvidia.com/ai-enterprise/release-7/7.4/infra-software/vgpu/configuration.html
