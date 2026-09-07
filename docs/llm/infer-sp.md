---
title: 序列并行推理
date: 2026-09-07
section: llm
---

# 序列并行推理

<div class="epigraph">
<p>序列并行把非张量并行的算子沿序列维切开，用 Reduce-Scatter 与 All-Gather 替换 All-Reduce，不增加通信体积，只把 Norm 与 Dropout 的激活降下来。</p>
<footer>—— Korthikanti et al.，Reducing Activation Recomputation in Large Transformer Models，arXiv:2205.05198，MLSys 2023</footer>
</div>

Megatron 的序列并行（sequence parallelism, SP）诞生在训练里：张量并行已经把 GEMM 切宽，LayerNorm / Dropout 却仍拿着完整的 $b\times s\times d$，激活内存按 $s$ 涨。Korthikanti 等人把进入这些算子之前的 All-Reduce 拆成 Reduce-Scatter，让每卡只留 $s/T$ 的序列，算完再 All-Gather 进下一截 GEMM。通信体积与原来的 All-Reduce 同阶，Norm 激活降为 $1/T$。推理不是没有这条轴。长预填、大并发、TP 已经打开时，激活与注意力工作区同样按 $s$ 涨；DeepSeek-V3 的服务部署在注意力上写的是 **TP4 加 SP**。本篇只写推理侧：它省什么、和上下文并行有何不同、decode 逐步还值不值得开。

## 问题

推理的显存账分三块：权重、KV 缓存、激活 / 工作区。权重靠量化与 EP/TP 切；KV 靠 MLA、GQA、分页。激活常被忽略，直到预填 $s=128\mathrm{k}$、再叠 TP 时，LayerNorm、QKV 投影前后的临时张量把 HBM 打满，表现为 OOM 而不是「KV 不够」。训练的 SP 正是冲着这块来的。推理预填的前向只有一份激活，没有反向存档，体积比训练小，但 $s$ 可以更大，且往往不能靠重计算把中间结果丢掉——预填要尽快出第一个 token，重算拉高 TTFT。

SP 不试图让单卡看不见其他位置的注意力。注意力仍在 TP 组内按头做完整序列（Gather 之后）。切开的是注意力 **之外** 的那些按序列独立的算子。若把「推理序列并行」理解成 Ring Attention 那种切 $s$ 再传 KV，会把一次廉价的 Reduce-Scatter 估成一次环形 KV 传递。

### 没有 TP 就没有这条 SP

Korthikanti 的 SP 绑在张量并行上：组大小 $T$ 即切分份数。Megatron 配置里 `sequence_parallel=True` 且 `tensor_model_parallel_size<=1` 会直接报错。推理框架若只开 SP、不开 TP，没有对应的通信组，只是把名字写进日志。DeepSeek-V3 预填与解码的注意力都是 TP4+SP，TP 度故意保持较小，以免 decode 的 TP 通信压过收益。

<span class="marginnote">社区里「sequence parallel」还被用来称呼 Ulysses / USP 一类切序列的注意力并行。读配置时先问：注意力内部有没有跨卡 KV？没有，才是 Megatron SP。有，那是上下文并行，见下一篇。</span>

## 方法

前向在列并行线性之前，对激活沿序列做 Reduce-Scatter，每卡持有 $s/T$；LayerNorm、Dropout（若推理还留着）、以及某些按位置的缩放在分片上算；进入注意力或 MLP 的 GEMM 前 All-Gather 回完整 $s$。行并行线性之后的 Reduce 同样改写成 Reduce-Scatter，使下一层 Norm 继续看到分片。推理没有 Dropout 的反向 RNG 问题，但若训练用了 SP，推理的 LayerNorm 统计必须与分片约定一致：通常每卡在自己的 $s/T$ 上算，不在全 $s$ 上同步均值方差——这与训练前向一致，换约定会移位。

预填 $s$ 长，Gather 后的注意力仍可能是单卡内存墙。SP **不能** 把 FlashAttention 的工作区按 $T$ 缩小到 $1/T$，因为注意力看见的仍是满序列。这时要叠上下文并行，或把预填切成块（chunked prefill）。SP 的收益是 Gather 前后那些 $b\times s\times d$ 的临时张量，以及 Norm 的激活；对注意力核本身是中性的。

```mermaid
flowchart LR
  X["完整 s 激活"] --> RS["Reduce-Scatter 沿 s"]
  RS --> LN["分片 LayerNorm"]
  LN --> AG["All-Gather"]
  AG --> ATT["TP 组内满序列注意力"]
```

### Decode 逐步：$s_q=1$ 时 SP 几乎无物可切

生成一步时，新 token 的序列维是 1。沿 $s$ 切 $1/T$ 没有意义。Decode 的 SP 若仍打开，实际切的是 **已缓存上下文里那些仍要过 Norm 的路径**，或只作用于预填留下的分片布局。V3 解码仍写 TP4+SP，更多是与预填共用一套并行包装、以及 MLP/注意力残差上的布局约定，而不是每步把一个 token 再切成四分之一。逐步的主导优化是 KV 字节、专家 EP、连续批，不是 SP。

Chunked prefill（一次提示切成多块顺序进模型）可以让每一块的 $s_{\mathrm{chunk}}$ 重新享受 SP 的激活下降，代价是 TTFT 变成多段前向之和。SP 与分块预填正交：分块减的是单次注意力工作区，SP 减的是 Norm 类张量。

## 机制

通信体积：一次 All-Reduce 等于 Reduce-Scatter 加 All-Gather，元素数同阶。SP 的「免费」指的是 **不比已经在付的 TP 通信多付一笔**，不是零通信。推理预填若本来开着 TP，打开 SP 几乎不增加集合次数，只改布局。若推理本来用纯 DP 复制权重，为了开 SP 而强行上 TP，会凭空增加 All-Gather，对 decode 尤其不划算。

激活节省发生在 LayerNorm、Dropout 以及部分残差分支。Korthikanti 在训练里与选择性重计算叠在一起，报过约 $5\times$ 的激活下降；推理没有反向，倍数更小，但仍与 $s$ 和 $T$ 成正比。长上下文预填、$T=4$ 或 $8$ 时，省下的是若干个 $b\times s\times d$ 的 BF16 缓冲，对 128k 并不小。

<span class="marginnote">RoPE 与因果掩码用全局位置。SP 只切 Norm 两侧的布局，位置编码仍按全局下标。不要在分片上用局部 $0\ldots s/T$ 去算 $\theta$，那会让各卡的旋转错位，Gather 之后注意力乱掉。</span>

### 和 EP、DP 叠在一起

DeepSeek-V3 预填：注意力 TP4+SP+DP8，MoE 走 EP32。SP 只活在注意力的 TP 组里。MoE 的 token 维是「本步该卡上的 token」，与 SP 的 $s/T$ 分片不是同一条轴——dispatch 之前通常要先把序列分片 Gather 回每个 token 的完整隐向量，否则专家看到的是半截 $d$ 或半截 $s$。实现若把 SP 分片直接送进 MoE 路由，专家 id 会对错行。配置上应写清：SP 组 = 注意力 TP 组，EP 组另画。

解码 DP80 把并发请求摊到许多副本上，每个副本内部仍是 TP4+SP。SP 不分担 batch，只分担该副本内部预填（若还在预填）或布局。不要指望 SP 替代连续批。

## 边界与工程取舍

短上下文、小 TP 的 decode 服务不要为 SP 改通信图。收益接近零，实现却要维护两套布局。长预填、已经必须 TP 的设置，SP 应视为默认附件，与 Megatron 训练侧的建议一致。

GQA / MLA 不改变 SP 的语义，但改变 Gather 之后注意力的内存。MLA 的压缩 KV 让长 $s$ 的缓存可行，SP 仍只帮激活。两者一起开时，OOM 要从三张表分别看：权重、KV、激活，不要只加 SP 或只加 MLA。

框架名字混乱是主要工程风险。Megatron-LM 的 `sequence_parallel`、DeepSpeed 的 Ulysses `ds-sequence-parallel-size`、vLLM 里偶见的 sequence parallel 开关，可能指向三条完全不同的通信。上线前用一次「只开 SP、看 NCCL 是 Reduce-Scatter 还是 All-to-All 换头」的探针，比读文档标题可靠。

<span class="marginnote">出处：Korthikanti et al.，arXiv:2205.05198；Shoeybi et al. Megatron-LM 的 TP 是前置。推理部署数字见 DeepSeek-V3 报告（TP4+SP）。Ulysses（Jacobs et al.，arXiv:2309.14509）与 Ring Attention（Liu et al.，arXiv:2310.01889）不要写进本篇的公式。</span>

检查点与量化：SP 不切权重，只切激活布局，对 NVFP4 权重量化无直接关系。激活若也走窄精度，Reduce-Scatter 的元素字节下降，但数值范围要能容忍 Norm 前的集合。这是精度栈问题，不是并行策略问题。

## 小结

- 推理侧的 Megatron 式 SP 在已有 TP 组内沿 $s$ 切 Norm 类激活，注意力仍见完整序列。
- 通信体积与原 TP All-Reduce 同阶；省的是长预填的激活与工作区，不是 KV，也不是 $O(s^2)$ FLOPs。
- Decode 逐步 $s_q=1$ 时几乎无序列可切；V3 仍标 TP4+SP，主要是布局与预填共用。
- 与 EP 叠加时，进 MoE 前必须恢复 token 的完整向量；SP 组不要和 EP 组画成同一个。
- 名字易与 Ulysses / 上下文并行混淆，应用通信模式区分。
- 出处：Korthikanti et al.，MLSys 2023，arXiv:2205.05198；DeepSeek-V3，arXiv:2412.19437。
