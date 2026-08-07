---
title: PD 分离：Prefill 与 Decode 的解耦部署
date: 2026-08-07
---

# PD 分离：Prefill 与 Decode 的解耦部署

<div class="epigraph">
<p>把两类工作交给两拨人，各干各的，互不拖累。</p>
<footer>—— 分布式系统分工思想（源自 DistServe 论文）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ DistServe 论文（Zhong et al., 2024） ｜ 2026-08-07</p>
</div>

## 为什么从 PD 分离开始

本专题反复强调 prefill 与 decode 的计算特征天差地别：prefill 是**计算密集**（大矩阵乘、吞吐导向）、decode 是**访存密集**（小矩阵、延迟导向）。把这两类工作放在同一个推理引擎里（如 vLLM 的 Continuous Batching 混跑），会互相伤害——一个大的 prefill 挤进 decode 批，所有 decode 的延迟被拖高；反之，decode 的密集批又让 prefill 的大矩阵吞吐受限。**PD 分离（Prefill-Decode Disaggregation）**：把 prefill 和 decode 部署成两类独立的服务，各自针对自己的计算特征做极致优化。<span class="marginnote">本专题《Continuous Batching》里 prefill/decode 混批是一张「合」的牌；PD 分离是<strong>「分」的牌</strong>。合有合的道理（简单、省通信），分有分的道理（隔离、各自优化）。</span>

本篇讲 PD 分离的动机、两类节点的分工、以及它引入的新问题：KV Cache 的跨节点传输。

## 1 为什么混跑会互相伤害

一个推理引擎同时处理 prefill 与 decode 时，调度器面临两难：

- **prefill 让 decode 延迟失控**：一个 10 万 token 的 prefill 进入批，会占用整批的计算资源很久，正在 decode 的所有请求的「每 token 延迟」瞬间暴涨——对延迟敏感的在线服务是灾难。
- **decode 让 prefill 吞吐受限**：decode 批的形状「窄而深」（小矩阵、大 batch），与 prefill 的「宽而浅」（大矩阵、小 batch）不兼容，预填充的吞吐上不去。

**核心矛盾：两类工作对「批的形状」「调度粒度」「优化目标」的要求相反。** 在一个引擎里同时满足两个优化目标，只能折中——两头都不讨好。PD 分离用「物理隔离」绕过这个矛盾。

## 2 两类节点的分工

PD 分离部署中：

- **Prefill 节点**：只做 prefill。输入长 prompt，输出「第一个 token + 该 prompt 的 KV Cache」。节点内调度优化目标 = **吞吐 + 首个 token 延迟（TTFT）**；批的形状可以大而宽（一次吃很多 prompt），Chunked Prefill 在这里也自由。<span class="marginnote">Prefill 节点是<strong>「算力导向」</strong>：它要尽量把 GPU 的算力喂满，吞吐越高、TTFT 越低越好。</span>
- **Decode 节点**：只做 decode。接收「前缀 KV Cache + 已生成的 token」，逐 token 生成直到停止。节点内优化目标 = **每 token 延迟（TPOT）+ 吞吐**；批的形状是「decode 专属」的窄而深。KV Cache 长期驻留，访存优化（FlashDecoding 等）在这里收益最大。

**请求的生命周期跨两类节点**：客户端发 prompt → 到 prefill 节点 → 产出 KV Cache → 连同首 token 一起发给 decode 节点 → decode 节点持续生成 → 完成。

## 3 引入的新问题：KV Cache 传输

PD 分离把 KV Cache 从「引擎内的数据结构」变成了「跨节点的传输对象」。这个转变带来几个问题：

- **KV 量巨大**：一个长 prompt 的 KV Cache 可到数 GB。prefill 节点算完，要把它传给 decode 节点——**传输时间直接影响 TTFT 与端到端延迟**。
- **传输要快**：跨节点传输依赖高速网络（RDMA、NVLink over PCIe 等）。传输 KV 的时间必须远小于 decode 生成时间，否则「分离」得不偿失。<span class="marginnote">这就是「KV Cache 传输」成为独立研究方向的背景（见下一篇《跨节点 KV Cache 传输与 RDMA》）。<strong>KV 压缩（量化）、异步预取、多副本</strong>都是为让这条链路更快。</span>
- **一致性**：decode 节点要「接住」prefill 节点的 KV，两者必须用相同的模型、相同的并行配置，KV 的布局（TP 分片方式）必须一致——**否则 KV 无法直接拼接**。

**辨析｜易错点：PD 分离不是把「单请求」拆成两半，而是把「负载」拆成两类。** 一个服务同时有大量 prefill 请求和大量 decode 请求，把它们分流到两组节点，各自的队列独立调度。**单请求仍然要「先 prefill 后 decode」串行走**，只是不同阶段落在不同机器上。

## 4 公式解析：PD 分离的收益边界

设 prefill 工作量 $W_p$、decode 工作量 $W_d$，混跑时两类工作共享 $C$ 个 GPU。分离后 prefill 拿 $C_p$ 个 GPU、decode 拿 $C_d$ 个。

- **第一步，写混跑的总耗时**（串行资源竞争下）：prefill 总时长 $W_p / (\text{混跑吞吐})$，decode 总时长 $W_d / (\text{混跑吞吐}')$。**两类工作相互拖累**，等价于各自的「有效算力」都打了折扣。
- **第二步，写分离后的耗时**：$T_p = W_p / (\eta_p C_p)$，$T_d = W_d / (\eta_d C_d)$，其中 $\eta_p$、$\eta_d$ 是各自优化后的利用率（分离后 $\eta$ 更高，因为调度不再互相妥协）。
- **第三步，比总吞吐**：分离的整体优势条件是

$$\eta_p C_p + \eta_d C_d > \eta_{\text{mixed}} C$$

当 $\eta_p, \eta_d$ 显著高于 $\eta_{\text{mixed}}$ 时（实测 prefill 与 decode 的批形状差异越大，$\eta_{\text{mixed}}$ 越低），**即使 $C_p + C_d = C$，分离的总吞吐也更高**——加上 KV 传输的固定开销仍划算。论文报告（DistServe）：PD 分离在长序列 + 高并发下可把 SLO 达标率提升一个数量级。

## 5 小结

- **prefill 与 decode 的计算特征相反**：混跑时 prefill 拖高 decode 延迟、decode 压低 prefill 吞吐，两头折中。
- **PD 分离物理隔离两类工作**：Prefill 节点追求吞吐 + TTFT，Decode 节点追求 TPOT + 吞吐，各自极致优化。
- **请求跨节点生命周期**：prompt → prefill 节点 → KV + 首 token → decode 节点 → 持续生成。
- **新问题：KV Cache 跨节点传输**：KV 量大、要快、两侧布局必须一致，催生 KV 压缩与 RDMA 优化。
- **收益边界**：分离后各自利用率显著提升，弥补 KV 传输开销后仍有净收益，长序列高并发下尤其明显。

在下一节，我们深入 KV 传输的极致架构——**Mooncake：以 KV Cache 为中心的分离式架构**。
