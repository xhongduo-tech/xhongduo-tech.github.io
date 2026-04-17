## 核心结论

SP、TP、PP、ZeRO 的通信重叠，本质是把“必须跨卡交换的数据”尽量提前发起，并放到与当前计算不同的执行通道里运行。这里的“重叠”可以先理解为：GPU 一边算，一边传，不再把通信和计算硬排成串行队列。

如果完全串行，一个阶段耗时近似是：

$$
t_{total} = t_{comm} + t_{compute}
$$

如果通信和计算能在不同 CUDA Stream 上并发，且硬件复制引擎没有被阻塞，阶段耗时会接近：

$$
t_{total} \approx \max(t_{comm}, t_{compute})
$$

这就是通信重叠的价值。它不会消灭通信，只是把通信“藏”到原本就要发生的计算时间里。

对训练大模型最重要的三类重叠分别是：

| 并行方式 | 主要通信 | 常见重叠手段 | 目标 |
| --- | --- | --- | --- |
| TP（Tensor Parallel，张量并行，指一层内部按张量维度拆到多卡） | AllReduce / Reduce-Scatter / AllGather | 把 AllReduce 拆成 RS + AG，并与后续 GEMM 重叠 | 隐藏层内同步开销 |
| SP（Sequence Parallel，序列并行，指按序列维度切分激活） | Reduce-Scatter / AllGather | 与 LayerNorm、Dropout、逐元素算子交错 | 降激活内存并减少等待 |
| PP（Pipeline Parallel，流水线并行，指不同层段放在不同卡组） | P2P Send / Recv | 与非依赖层计算或 1F1B 调度重叠 | 减少 stage 间空泡 |
| ZeRO/FSDP | 参数 AllGather、梯度 Reduce-Scatter | forward/backward prefetch | 把参数拉取隐藏到层执行前 |

玩具例子可以直接看数字：如果一次 TP AllGather 需要 100ms，下一层 GEMM 需要 120ms，串行是 220ms；如果通信在另一个流上先发起并与 GEMM 重叠，实际更接近 120ms。新手最该记住的不是 API 名字，而是这个时间模型。

---

## 问题定义与边界

问题不是“如何让通信更快”，而是“哪些通信其实不用阻塞主计算流”。这两件事不同。前者偏网络和硬件带宽，后者偏调度和依赖分析。

先看边界。只有满足“数据依赖还没真正发生”的通信，才有资格做 overlap。也就是说，通信可以早发起，但不能早消费。比如某层输出张量需要先 Reduce-Scatter，下一步只要使用的是已经到达的 chunk，就能边收边算；但如果后续算子要等完整张量，重叠空间就小很多。

可以把依赖关系整理成下面这张表：

| 通信阶段 | 依赖什么 | 可以和谁重叠 |
| --- | --- | --- |
| TP 的 Reduce-Scatter | 当前层局部输出已算完 | 下一层只依赖局部 shard 的计算 |
| TP/SP 的 AllGather | 某个 shard 已就绪 | 后续 chunk 级别 GEMM 或逐元素算子 |
| PP 的 Send/Recv | 上一 stage 的 micro-batch 输出 | 当前 stage 中不依赖该消息的其他 micro-batch 计算 |
| ZeRO/FSDP 的参数 AllGather | 已知后续要用哪层参数 | 当前层或前几层的计算 |
| 梯度 Reduce-Scatter | 当前层反向梯度已生成 | 更早层的反向计算 |

这里还要明确一个常见误区：不是所有通信都值得重叠。小模型、单机 8 卡、TP 很小的时候，通信本身可能不是瓶颈；这时强行引入复杂调度，收益不一定覆盖工程复杂度。

真实工程例子是 100B 以上模型跨多机训练。此时 TP 组和 PP stage 往往横跨多个节点，网络往返延迟和带宽消耗都显著上升。如果继续用“算完再统一 AllReduce、收完再统一计算”的同步方式，GPU 会出现明显空转，吞吐下降非常快。通信重叠就是用来修复这种缩放失效。

---

## 核心机制与推导

第一层机制是 TP 的 AllReduce 拆分。AllReduce 可以看成“先分片归约，再分片收集”：

$$
\text{AllReduce} = \text{Reduce-Scatter} + \text{AllGather}
$$

Reduce-Scatter 的意思是：每张卡把全局结果里“属于自己那一片”的归约值先算出来。AllGather 的意思是：再把各卡手里的那一片互相交换，拼回完整结果。

拆分的意义在于，完整张量不必一次性等齐。只要后续算子支持 chunk 化消费，就可以形成下面这种时序：

```text
时间 -->
Stream 0（计算）:  GEMM(chunk0)   GEMM(chunk1)   GEMM(chunk2)
Stream 1（通信）:  AG(chunk0)     AG(chunk1)     AG(chunk2)
```

这要求两个条件：

1. 通信按 chunk 切分，而不是一个超大 tensor 一次性传完。
2. 计算能按 chunk 消费，而不是非要等全量输入就绪。

这也是 userbuffer 的用途。可以把它理解成“预先分好的固定通信缓冲区”，让通信库和计算库都知道每个 chunk 放在哪里。没有固定 buffer，通信与计算之间就很难稳定地双缓冲。双缓冲的意思是：当前块在算时，下一块已经在传。

SP 的重叠通常跟 TP 一起出现。SP 不是独立替代 TP，而是把部分激活按 sequence 维度切开，让原本必须保留完整激活的层改为只保留局部片段。这样做经常伴随 Reduce-Scatter 和 AllGather，也就自然带来 chunk 级 overlap 的机会。

PP 的核心机制不同。PP 主要通信不是 collective，而是 stage 间点对点发送。它最常见的优化是 1F1B，也就是“一个前向、一个反向”交替推进不同 micro-batch。这里的关键不是把一条消息发得更快，而是让 Send/Recv 插入到不相关 micro-batch 的计算间隙中。

玩具例子：4 个 stage，micro-batch 数量足够多。stage 2 正在做 micro-batch 7 的前向时，可以同时接收 micro-batch 6 的反向梯度，也可以把 micro-batch 8 的输入激活准备好。只要这些动作在不同流中调度，且不会争抢同一个数据依赖，空泡就会显著变小。

ZeRO/FSDP 的机制则是“参数预取”。参数分片平时不完整保留在本卡上，执行某层前才 AllGather 回来。如果等到层真正开始执行时才拉参数，计算必然阻塞。更合理的做法是 forward prefetch：当前层还在算，下一层参数已经在后台 AllGather；反向阶段也类似。

因此整体推导逻辑很简单：

1. 找到通信真正的消费点，而不是发起点。
2. 把通信发起时间尽量前移。
3. 用 chunk、双缓冲、多流把通信与计算交错。
4. 让关键路径从“求和”变成“取最大值”。

---

## 代码实现

下面先用一个可运行的 Python 玩具程序，模拟“串行”和“理想重叠”的耗时差异。它不依赖 GPU，只验证时间模型是否成立。

```python
def serial_time(comm_ms, compute_ms, chunks=1):
    return chunks * (comm_ms + compute_ms)

def overlap_time(comm_ms, compute_ms, chunks=1):
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    # 第一块必须先等一段启动时间，之后进入流水
    return comm_ms + compute_ms + (chunks - 1) * max(comm_ms, compute_ms)

# 单块没有流水，等价于串行启动后结束
assert overlap_time(100, 120, chunks=1) == 220

# 三块进入稳定流水后，总时间明显小于纯串行
serial = serial_time(100, 120, chunks=3)
pipelined = overlap_time(100, 120, chunks=3)

assert serial == 660
assert pipelined == 460
assert pipelined < serial

# 当通信比计算慢时，总耗时由通信主导
assert overlap_time(150, 80, chunks=4) == 150 + 80 + 3 * 150
```

这个程序表达的不是精确 GPU 时间，而是 pipeline overlap 的基本结构：第一块要付出暖启动成本，后续块逐步进入“谁更慢谁决定节拍”的状态。

实际工程里，Megatron 和 DeepSpeed 的配置思路大致如下：

```python
# 伪代码：Megatron / Bridge 风格
class Args:
    tp_comm_overlap = True
    overlap_p2p_comm = True
    sequence_parallel = True

args = Args()

bridge_config = {
    "tp_comm_overlap": args.tp_comm_overlap,
    "overlap_p2p_comm": args.overlap_p2p_comm,
    "sequence_parallel": args.sequence_parallel,
}

assert bridge_config["tp_comm_overlap"] is True
assert bridge_config["overlap_p2p_comm"] is True
```

```python
# 伪代码：DeepSpeed ZeRO-3 预取配置
deepspeed_config = {
    "zero_optimization": {
        "stage": 3,
        "prefetch_bucket_size": 500_000_000,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
    },
    "aio": {
        "block_size": 1048576
    }
}

assert deepspeed_config["zero_optimization"]["stage"] == 3
assert deepspeed_config["zero_optimization"]["overlap_comm"] is True
```

如果是训练脚本，常见启用点通常包括：

| 场景 | 常见开关 | 作用 |
| --- | --- | --- |
| TP overlap | `--tp-comm-overlap` | 让 TP 的 RS/AG 与 GEMM 交错 |
| PP overlap | `--overlap-p2p-comm` | 让 stage 间 Send/Recv 不阻塞主流 |
| SP | `--sequence-parallel` | 按序列维度分片激活，配合 TP 降内存 |
| ZeRO overlap | `overlap_comm=true` | 让梯度归约、参数准备尽量提前 |
| ZeRO prefetch | `prefetch_bucket_size` | 控制参数预取 bucket 的大小 |

真实工程例子：8 机 64 卡训练一个百亿到千亿参数模型，TP=8、PP=8、ZeRO-3 开启。若不开 overlap，某些 stage 会在每个 micro-batch 的边界处等待 P2P，TP 层内还要额外等待 AllGather，ZeRO 再等参数收齐，三个等待叠加后 GPU 利用率会明显掉到低位。开启 TP/PP/ZeRO 的分层重叠后，虽然每类通信总字节数几乎没变，但大部分通信被塞进了已有计算窗口，step time 会更平滑。

---

## 工程权衡与常见坑

通信重叠不是“开一个参数就结束”。它非常依赖运行时环境、形状稳定性和缓冲区策略。

最常见的问题如下：

| 问题 | 原因 | 规避 |
| --- | --- | --- |
| overlap 开了但没提速 | 通信和计算落在同一执行通道，实际上仍串行 | 检查多流、NCCL 异步、设备连接配置 |
| H100 上几乎不重叠 | `CUDA_DEVICE_MAX_CONNECTIONS` 太小，复制引擎并发不足 | 显式设置较大的连接数并验证 profile |
| 动态 shape 导致性能抖动 | userbuffer 通常依赖固定大小 chunk | 用 padding、bucketing 或回退到非 pipeline overlap |
| bucket 过大 | 第一块启动太慢，重叠窗口被压缩 | 调整 chunk / bucket 大小，避免单次通信过粗 |
| bucket 过小 | 启动开销和调度开销过高 | 通过 profile 找到带宽与 launch 次数的平衡点 |
| ZeRO 预取过猛 | 参数太早拉取，导致显存峰值抬高 | 缩小 `prefetch_bucket_size`，控制 prefetch 深度 |
| PP micro-batch 太少 | 流水线填不满，空泡无法隐藏 | 增加 micro-batch 数或降低 PP 深度 |

一个特别容易踩的坑是把“异步调用”误认为“已经重叠”。API 返回得快，不代表设备真的在并发执行。必须用 profiler 看时间线，确认通信 kernel 和计算 kernel 是否真实交错，而不是只是在主机侧异步。

另一个坑是调试成本。全重叠后，问题往往从“功能错误”变成“时序错误”。比如某个 chunk 还没 AllGather 完就被消费，或者某个 Send/Recv 配对不一致。这类问题在小规模上可能偶现，在大规模上才稳定复现，所以需要严格的 shape、stream、event 管理。

---

## 替代方案与适用边界

如果你是刚开始搭训练系统，不必一上来就追求 SP+TP+PP+ZeRO 全重叠。更合理的做法是按复杂度递进。

可以按下面的边界判断：

| 方案 | 适用场景 | 优点 | 代价 |
| --- | --- | --- | --- |
| 纯同步 | 小模型、单机、小规模验证 | 行为稳定、最易排障 | 吞吐最低 |
| 只做 PP overlap | 先解决 stage 间空泡 | 改动集中、收益直观 | TP/ZeRO 等待仍在 |
| 做 TP/SP overlap | TP 较大、层内通信重 | 对大矩阵层收益明显 | 依赖 chunk、buffer、流调度 |
| 做 ZeRO prefetch | 参数分片严重、显存紧张 | 能隐藏参数拉取延迟 | 需要精细控制显存峰值 |
| 全重叠 | 多机大模型、吞吐优先 | 性能上限最高 | 实现和排障最复杂 |

适用边界可以概括成三条：

1. 小规模训练优先稳定，不必强求全重叠。
2. 跨机、多维并行、通信占比高时，重叠通常是必选项。
3. 如果 shape 频繁变化、模型结构动态、调试需求强，部分重叠往往比全重叠更合适。

判断标准很简单：先 profile，再决定。不要因为“文档里推荐”就默认全部打开。工程上真正重要的是关键路径是否真的被缩短，而不是配置文件里是否出现了 `overlap`。

---

## 参考资料

1. NVIDIA Megatron Core / Megatron Bridge 文档，关于 TP 通信重叠、PP P2P overlap 与相关运行时要求。  
2. NVIDIA NeMo Performance Guide，关于 `CUDA_DEVICE_MAX_CONNECTIONS`、并行训练性能调优。  
3. Leeroopedia 对 TransformerEngine 通信与 GEMM overlap、chunk、userbuffer、双缓冲机制的说明。
