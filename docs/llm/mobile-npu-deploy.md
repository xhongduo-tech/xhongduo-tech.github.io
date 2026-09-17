---
title: 移动端 NPU 部署
date: 2026-09-08
section: llm
---

# 移动端 NPU 部署

<div class="epigraph">
<p>手机上的 TOPS 写在发布会上；LLM decode 要的是能持续吃带宽的矩阵核、以及不会把 8GB DRAM 吃光的 KV 预算。</p>
<footer>—— 对照 ExecuTorch / Core ML / NNAPI / 厂商 QNN 的端侧部署栈；量化与内存是硬约束</footer>
</div>

[上一课](/llm/webgpu-inference)是浏览器沙箱。原生 App 可以走 NPU：Apple Neural Engine、高通 Hexagon、联发科 APU，经由 Core ML、NNAPI、QNN、ExecuTorch。本课写部署差：静态图编译、INT8/INT4、热包络、以及 decode 循环往往仍落在 CPU/GPU，因为 NPU 对动态 KV 不友好。不要把「设备有 45 TOPS」写成「70B 能跑」。

## 问题

NPU 喜欢固定形状卷积/GEMM。LLM 逐步 $n$ 变、采样不规则、投机分支，编译器要么 pad 到 $n_{\max}$（DRAM 炸），要么每步回 CPU。缺口是混合：投影与 FFN 下沉 NPU，注意力在 GPU/CPU；或整网 CPU 量化（llama.cpp 系）。切面选错的下场可算：每步把 KV 搬过一次运行时边界，带宽墙提前撞上，NPU 吃到的静态 GEMM 再快也白搭。

热功耗：持续 decode 会降频，峰值 TOPS 与持续 token/s 不是一回事——[每 token 能耗](/llm/energy-per-token)在手机上直接决定能不能聊完一条消息。差距可以拆成三个机制。利用率：TOPS 是 MAC 阵列每周期全喂满的纸面数，decode 对每个权重只做一次乘加、算术强度在 $2$ FLOP/字节量级，落在[屋顶线](/llm/arithmetic-intensity-decode)的带宽区，阵列大部分时间在等数据。带宽墙：LPDDR 的带宽还要被 ISP、显示、modem 分食，权重每步都要重读，INT4 压的就是这条流量。内存驻留：权重、KV、系统与应用争同一颗 DRAM，KV 随上下文线性涨，8GB 机器上这条账先爆。所以「TOPS ÷ 模型 FLOPs」估 TPOT 的三个前提——满阵列、独享带宽、恒定时钟——在手机上一个都不成立。

后台：OS 会杀长时间 GPU/NPU 占用。产品要把生成切成可暂停步。

<span class="marginnote">ExecuTorch 把 PyTorch 边下沉边解释；Core ML 转换对动态控制流不完整。选栈先看 generate 循环能不能留下，再看算子覆盖。</span>

## 方法

模型级：1B–8B 级量化，上下文 2K–8K 量级起步——超出这个窗口，权重加 KV 的内存驻留在 8GB 机器上就排不下，那是容量问题，不是精度问题。KV 用量化或短窗。工具链：能导出的静态子图下 NPU，循环在运行时；判据是循环每步要不要跨运行时边界——每步跨界就多一次同步与搬运，比单个算子慢更伤。与分词：移动端分词必须 C++/Rust，不能起 Python，解释器的启动与常驻内存都按百 MB 计。UI 流式走稳定前缀。电量：测持续功率与降频曲线，不是芯片海报——峰值只撑几十秒，持续功率才是「聊完一条消息」的预算。

```mermaid
flowchart TD
  PT["检查点"] --> Q["INT8/INT4 转换"]
  Q --> SPLIT["静态子图 → NPU"]
  Q --> LOOP["逐步循环在 CPU/GPU"]
  LOOP --> KV["短上下文 / 量化 KV"]
```

## 机制

屋顶线：NPU 峰值高、能用的形状窄；decode 强度低，可能根本打不满 TOPS，瓶颈回到 DRAM。这与数据中心 decode 同构，只是绝对数字更差：HBM 换 LPDDR，功耗预算从几百瓦换到几瓦，可容忍的模型体积从百 GB 换到个位 GB。混合切面由此有了依据：FFN 与投影是大而静态的 GEMM，形状固定，下沉 NPU 吃矩阵核；注意力逐步读变长的 KV，形状动态，留在 CPU/GPU 反而快。厂商 SDK 的「LLM 方案」往往是自家核 + 私有格式，可移植性低于 ONNX。选择等于绑定 SoC。

## 边界

不要在 NPU 上假设任意 [logit bias](/llm/logit-bias) 与文法掩码都能融合：采样在循环里，NPU 子图只出到 logits，采样逻辑能不能下沉先查 SDK。不要用峰值 TOPS 除以 FLOPs 估 TPOT。下一课：纯 CPU 量化路径（[CPU 推理与量化](/llm/cpu-inference-quant)），反而更可移植。

## 小结

- 手机 NPU 适合静态子图；LLM decode 循环常留在 CPU/GPU。
- 峰值 TOPS ≠ 持续 token/s：利用率、带宽墙、内存驻留、热四条账。
- 模型要小、KV 要短、权重要量化。
- 分词与 generate 必须在原生运行时。
- SoC SDK 绑定与可移植性权衡。
- 出处：ExecuTorch；Core ML；NNAPI；厂商 QNN 文档。
