---
title: DeepEP
date: 2026-09-07
section: llm
---

# DeepEP

<div class="epigraph">
    <p>专家并行的 dispatch 与 combine 必须把 InfiniBand 与 NVLink 吃满，又不能把计算用的 SM 长期占死；通信核要按拓扑转发，而不是调用一次通用 All-to-All 了事。</p>
<footer>—— DeepSeek-AI，DeepEP 库与 DeepSeek-V3 报告中的跨节点 All-to-All 核</footer>
</div>

MoE 把 token 按路由器送到不同专家，再把专家输出加权收回。语义是两次不规则 All-to-All：dispatch 与 combine。NCCL 的通用集合通信能跑通，但 DeepSeek-V3 的训练与服务把专家铺到多节点（训练 EP64 跨 8 机，解码甚至 EP320），通信体积与计算相当，还要给注意力留下 SM。DeepEP（DeepEveryParallel）是他们开源的通信库：面向专家并行的高吞吐 / 低延迟 GPU 核，支持 FP8 dispatch，运行时 JIT，机内走 NVLink、机间走 RDMA。本篇写这套核要解决什么；更一般的 EP 语义见 [EP All-to-All](/llm/ep-all2all)，分组 GEMM 见 [DeepGEMM](/llm/deepgemm)。

## 问题

V3 报告写清拓扑：节点内 8 卡 NVLink/NVSwitch，节点间 InfiniBand 全互联。NVLink 约 160 GB/s，IB 约 50 GB/s，大约 3.2 倍差距。每个 token 若任意打到 8 个专家、专家又散落各机，IB 会先于 NVLink 饱和。他们用**节点数限制路由**：一个 token 最多派到 4 个节点，先 IB 到目标节点上「同卡号」的 GPU，再 NVLink 转发到真正持有专家的卡。Warp 特化把 20 个 SM 分成通道，分别做 IB 发送、IB→NVLink 转发、NVLink 接收；combine 反向并带累加。核与门控、拓扑共设计，不是一张 `alltoall` 调用。

服务侧形态更极端。预填充 EP32、解码 EP320 时每卡一个专家，解码的 All-to-All 改成 IB 上的点对点，并用 IBGDA 降延迟。同一套「token 找专家」语义，预填充要吞吐、解码要尾延迟，通用库很难两头都到屋顶线。DeepEP 因此提供高吞吐与低延迟两套路径（V2 收到 `ElasticBuffer` 一套接口里），并支持把 DeepEP 的低延迟输出直接喂给 DeepGEMM 的 grouped FP8 GEMM。

### SM 占用是一等约束

通信核跑在 SM 上，占着的 SM 就不能给 MLA 或专家 GEMM。DualPipe 要把 dispatch/combine 藏进另一段计算，必须能指定通信用多少 SM。V3 训练叙述里定制核「节省专用于通信的 SM」；DeepEP V2 宣称相对 V1 峰值约 1.3×、SM 用量可到约 1/4，V3 类训练从约 24 个 SM 降到 4–6 个仍能维持带宽。这些是仓库基准与发布说明里的量级，换网卡、换 EP 宽度要重测。

<span class="marginnote">DeepEP 不是新的 MoE 算法。它不改 top-$k$、不代替负载均衡偏置。它实现的是 V3 报告里「高效跨节点 All-to-All」那一层。Switch / GShard 的容量因子、V3 的无辅助损失偏置，都在这层之上。</span>

## 方法

### dispatch、combine、句柄

调用方提供 token 激活、`topk_idx`、`topk_weights`。`dispatch` 返回重排后的激活、各 rank 收到的 token 元数据、以及一个 `EPHandle`，供对称的 `combine` 按原顺序加权还原。FP8 路径上，激活以 `(data, scale)` 元组传入，与 V3「上投影前 FP8、combine 保持 BF16」的精度策略一致。预填充/训练可用高吞吐接口吃满带宽；解码用低延迟接口、更小的 `num_max_tokens_per_rank`，避免按训练 batch 去注册巨大缓冲。

V2 用 `ElasticBuffer` 按 MoE 配置解析出缓冲字节、理论 SM 数与 QP 数，不再靠一维自动调参瞎搜。后端从 NVSHMEM 迁到更轻的 NCCL Gin，并可复用已有 NCCL communicator。遗留 V1 路径仍依赖 NVSHMEM。安装与 JIT 发生在运行时，不要求提交仓库时带预编译 CUDA。

### 与计算重叠

正确用法是：通信流与计算流分 stream，`EventOverlap` 一类事件让注意力或 GEMM 与 dispatch 重叠。V3 预填充甚至双微批：一批的注意力/MoE 与另一批的 dispatch/combine 交叉。解码则把一批的注意力与另一批的 dispatch+MoE+combine 重叠，因为解码注意力更重、专家侧更偏访存，通信核只配少量 SM。DeepEP 提供的是可重叠的原语，双微批编排仍在引擎里。

```mermaid
flowchart LR
  T["token + topk"] --> IB["IB 送到目标节点同号 GPU"]
  IB --> NV["NVLink 转发到专家所在卡"]
  NV --> E["本地专家 GEMM"]
  E --> NV2["NVLink 回收集加"]
  NV2 --> IB2["IB 回源并 combine"]
```

## 机制

### 为什么要「转发」而不是端到端 IB

若每个源-专家对都走一条 IB，流量无法在节点入口聚合，也用不上 NVLink 的 3 倍带宽。先 IB 到节点代表卡、再 NVLink 扇出，是在用高带宽域做最后一跳。限制每 token 触及的节点数，直接砍 IB 消息条数。Warp 间按实际负载动态分配发送/转发/接收，避免静态三分之一在某一跳空转。combine 的累加放在 NVLink→IB 或 IB 接收侧，减少把未化简的向量在慢网上搬两次。

低延迟路径牺牲的是带宽利用率换建立与同步开销：解码每步 token 少（每专家常 <256），消息小，吞吐核的启动开销会变成主导。两套核对应两条屋顶线，不是 API 命名游戏。

<span class="marginnote">仓库给出的逻辑带宽（如 SM90、CX7、EP 8×2 时 dispatch 约 90 GB/s）含本 rank 本地流量，且是特定 hidden=7168、top-8、FP8 dispatch / BF16 combine、8K token/batch 的配置。抄到合同里当「网卡保证 90 GB/s」是错的。</span>

### 和通用 NCCL All-to-All 的差别

通用 All-to-All 假设均匀、规则的分区。MoE 的目的地由路由动态决定，各专家 token 数不同，需要 permute + 变长缓冲 + 可能的 FP8 缩放因子同行。DeepEP 把门控下标当成一等输入。仍用 NCCL 当 RDMA 后端（V2 Gin）不等于「调用 `ncclAllToAll` 就等于 DeepEP」。

## 边界与工程取舍

DeepEP 绑定 Hopper 一类 SM90 PTX、较新的 CUDA/PyTorch/NCCL，以及机内 NVLink + 机间 RDMA。没有 NVLink 的 PCIe 八卡机不是 V3 报告里的域：这时应缩小 EP、把专家留在节点内，而不是指望同一套转发核在跨根复合物上跑出报告数字。V2 缓冲比 V1 大，解码卡上要把它算进 HBM 账，不能只看专家权重。0 SM 的 RDMA 低延迟 EP 在 V2 不再支持；实验性的 Engram、流水线与上下文并行原语也不要当成生产 EP 的默认开关。

负载极度不均时，再快的核也在搬空气或堵热专家。V3 预填充用冗余专家并在节点内重排、解码用统计周期更新热点副本，都是在通信层之外改图。DeepEP 不会替你做负载均衡。QP 与 SM 数若设得比理论值更激进，可能挤掉 MLA 的吞吐，端到端反而变慢——要以整层 Transformer 的墙钟为准，而不是只看 dispatch microbenchmark。

出处：DeepSeek-AI，*DeepEP*，https://github.com/deepseek-ai/DeepEP；算法与拓扑约束见 *DeepSeek-V3 Technical Report*，arXiv:2412.19437，§3.2 与部署节。不要给 DeepEP 伪造独立 arXiv 号。

<span class="marginnote">SGLang / vLLM 集成会随版本变。写部署清单应以当时引擎文档为准，并写明 EP 宽度、是否 FP8 dispatch、以及通信 SM 数，而不是只写「enable DeepEP」。</span>

## 小结

- DeepEP 是 MoE 专家并行的 dispatch/combine 通信库，针对 IB+NVLink 拓扑与 FP8。
- 节点限制路由、IB 再 NVLink 转发、可配置 SM，是为了带宽与把计算 SM 留给 MLA/GEMM。
- 高吞吐与低延迟对应预填充/训练与解码两条屋顶线。
- 输出布局应对齐 DeepGEMM 的 grouped GEMM。
- 出处：deepseek-ai/DeepEP；DeepSeek-V3 报告。
