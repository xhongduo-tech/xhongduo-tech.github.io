---
title: Huawei CloudMatrix384 超节点
date: 2026-09-07
section: llm
---

# Huawei CloudMatrix384 超节点

<div class="epigraph">
    <p>超节点论文要回答的不是「384 张卡怎么装箱」，而是：在统一总线上，如何把 prefill、decode 与缓存收成可独立伸缩的服务，并在宽专家并行下把 DeepSeek-R1 跑出可引用的吞吐。</p>
    <footer>—— Zuo et al., Serving Large Language Models on Huawei CloudMatrix384, arXiv:2506.12708</footer>
</div>

[CloudMatrix 384](/llm/cloudmatrix-384) 已经钉过生产 SKU：384 颗昇腾 910C、192 颗鲲鹏、16 柜、三平面。本篇读 **arXiv:2506.12708** 作为一篇**服务系统论文**：硬件只提供一块紧耦合的 Scale-Up 域；真正把 LLM 请求变成 tokens/s 的，是 CloudMatrix-Infer。论文把贡献收成三刀——对等的 Prefill–Decode–Caching（PDC）池、依托 UB 的大规模专家并行（文中示例 EP320）、以及面向 910C 的融合算子与 INT8。SKU 组成、柜数与单卡 TFLOPS 不在此重复；算子重叠见 [CloudMatrix-Infer](/llm/cloudmatrix-infer)，池化编址见 [资源池](/llm/cloudmatrix-resource-pool)，宽 EP 与 KV 见 [MoE KV](/llm/cloudmatrix-moe-kv)。未在论文出现的光模块只数、未发布的 UB 包格式，不写。

## 问题

MoE 变大、上下文变长之后，传统「一台 8 卡服务器 + 节点间 RDMA」会在两处同时破：张量并行与 token dispatch 一跨节点就掉到以太网档；KV 复用若绑在请求落点，缓存命中变成调度约束。论文把需求写成：计算强度、内存带宽、片间通信与延迟，再加上负载波动和硬 SLO。只加机器台数填不满 Scale-Up 这一档；只把超节点当更大的盒子、软件仍按 hostname 调度，UB 买了等于没买。

对照物是 KV-centric 的分离式服务：prefill 与 decode 拆开之后，KV 仍往往要考虑「缓存在哪台机器上」，调度带着局部性。CloudMatrix384 的主张是：域内 UB 提供高带宽、近均匀的访问，于是可以把缓存做成**对等池**，而不是每条请求先问数据住在哪。

### 论文三刀：PDC、宽 EP、硬件相关核

第一刀，PDC。Prefill 池建初始 KV 并出首 token；Decode 池自回归续写；Caching 池用超节点内 DRAM（论文还提到与 SSD 配合的分布式 KV）承接历史缓存与模型块。三者经显式 KV 传递接口通信，按负载独立加卡。论文对比：KV-centric 受数据局部性约束；peer-to-peer 降低「请求必须钉在缓存所在节点」的假设，调度更接近无状态。

第二刀，宽 EP。DeepSeek-R1 一类稀疏 MoE 的 decode 延迟，对「每 Die 一个专家、token 在超节点内 All-to-All」敏感。论文给出 EP320 量级：让 decode 集群里每颗 NPU Die 托管恰好一个专家，换更瘦的本地 GEMM 与更低的逐步延迟。910C 是双 Die 封装，计数时必须声明是封装还是 Die——384 是封装数，EP320 走的是 Die 账。

第三刀，硬件相关优化：融合 dispatch/combine、微批流水、INT8。没有这三刀，超节点只是密度更高的 8 卡集群。

<span class="marginnote">EP320 是服务策略，不是超节点铭牌。不要把 320 写进「CloudMatrix384 = 320 卡」。论文用它说明 UB 上可以铺开到「每 Die 一专家」；换模型、换专家数，度数要重算。</span>

## 方法

把一篇超节点服务论文当成实验报告来读：模型、精度、SLO、分母。评测主模型是 DeepSeek-R1。公开结果：prefill **6688 tokens/s/NPU**（4K 输入一类设定）；decode **1943 tokens/s/NPU**，TPOT 低于 50 ms；在更严的 **15 ms** 延迟约束下仍维持 **538 tokens/s/NPU**。INT8 量化被写成在一组基准上保持精度，并把关键矩阵（FFN、稠密、注意力）收到 INT8，论文称相对全精度有数倍吞吐增益——具体倍数属于该文实验，不要抄成任意模型的 SLA。

软件栈落在 CANN / HCCL / MindIE 或 [vLLM-Ascend](/llm/vllm-ascend) 一侧。并行组应落在 UB 域内：宽 TP、宽 EP 不要在超节点边界上再拼一台。域间 KV 与副本走 RDMA（RoCE）；管控与对象存储走 VPC。这与 [Scale-Up 对 Scale-Out](/llm/scale-up-vs-scale-out) 对得上：集合通信属于超节点内，跨可用区属于超节点外。

```mermaid
flowchart LR
  REQ["请求"] --> P["Prefill 池"]
  P --> C["Caching 池 · UB 对等访问"]
  C --> D["Decode 池 · 宽 EP"]
  D --> TOK["token"]
  P -.-> |"显式 KV"| C
  C -.-> |"显式 KV"| D
```

### DeepSeek-R1 数字钉在哪一组条件

分母是每 NPU（封装）吞吐，不是每 Die、不是每超节点合计后再除以营销 PFLOPS。输入长度、batch、是否 INT8、TPOT 上限，四者缺一不可比。1943 与 538 是同一套系统在不同延迟约束下的两个工作点：松约束吃吞吐，紧约束吃 TPOT。把 6688 写成「任意 prefill 的保证」，等于丢掉 4K 设定。论文还给出 tokens/s/TFLOPS 一类归一化，用来和别的加速器比效率；那是作者的对照框架，不是本博客的测量。

<span class="marginnote">INT8 自适应尺度搜索、per-token 激活与 per-channel 权重，是论文量化节的方法，不是「昇腾只支持 INT8」。精度表覆盖哪些基准，以原文表格为准，不要外推成所有安全关键任务都已过关。</span>

## 机制

PDC 能对等，靠的是 UB 同时提供内存语义与消息语义，见 [灵衢](/llm/ub-lingqu)：KV 页、专家缓冲可以按全局可寻址对象登记，而不先经过本机内核两次拷贝。论文写缓存实例相对本地复制，DRAM 开销可以从「每实例一份完整副本」降到池化后的更低倍数——那是该文缓存设计的账，用来理解「池化改的是访问语义」，不是把 384 张卡融进一个 `cudaMalloc`。

宽 EP 的机制是：decode 一步工作集小，专家若挤在少数卡上，GEMM 太肥、通信突发太尖；铺到每 Die 一专家，本地计算变瘦，dispatch 体积按 token 走 UB。FusedDispatch 在发送前量化以缩小消息，AIV-Direct 经 UB 直写对端预分配缓冲，躲开 SDMA 启动税。这与 GPU 上 DeepSeek 双微批思路同族，但是 910C 的 AIC/AIV/SDMA 异构核上的改写。

### 对等访问如何改调度假设

KV-centric 调度必须问：这条前缀的缓存在哪，跨节点搬不划算就粘在原节点，于是出现热点与空洞。UB 域内搬 KV 的成本被论文写成可接受之后，调度可以按计算空闲度派 prefill/decode，缓存命中变成池的命中率，而不是节点的命中率。代价是故障域变成超节点：通信柜或一条 L2 子平面影响整块逻辑节点，维护窗口按 16 柜设计，集群应能把该超节点从副本集摘掉。

## 边界与工程取舍

不要把 NVL72 的托盘抄成 12+4 柜。不要在没有 UB 的普通 8 卡集群上用 64 路 TP「模拟 384」。不要把 R1 的 6688 / 1943 / 538 写成任意模型、任意精度的 SLA。不要把论文里规划的更大超节点、CPU 物理分解，当成 384 这一代已经交付的形态——这一代仍是节点内 8 NPU + 4 CPU 的固定配比，逻辑上池化。

软件生态与硬件是否已经是一块逻辑节点，是两笔账。论文是华为作者在自有超节点上的系统报告，对照 GPU 数字来自作者设定的基准，读的时候分开「超节点内相对自己基线的增益」和「跨厂商屋顶线」。

<span class="marginnote">出处：arXiv:2506.12708。UB-Mesh 拓扑见 arXiv:2503.20377；本篇只把 384 的 UB 当作其递归落地来引用，不把 4D-Pod 的 1024 NPU 设计与 384 SKU 画等号。</span>

## 小结

- CloudMatrix384 论文的主体是服务系统 CloudMatrix-Infer，不是装箱清单。
- 三刀：PDC 对等池、UB 上宽 EP（示例 EP320）、910C 相关融合与 INT8。
- DeepSeek-R1：prefill 6688、decode 1943（TPOT 低于 50 ms）、15 ms 约束下 538 tokens/s/NPU，均钉原文设定。
- 出处：Zuo et al.，arXiv:2506.12708。
