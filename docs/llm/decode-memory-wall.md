---
title: Decode 的显存墙
date: 2026-09-03
section: llm
---

# Decode 的显存墙

<div class="epigraph">
<p>逐步生成时每一步的查询长度是 1，算术强度掉到带宽屋顶之下：墙钟由读出权重与不断变长的 KV 决定，而不是由峰值 FLOPS 决定。</p>
<footer>—— 把屋顶线模型接到自回归解码：compute-bound 的是 prefill，memory/bandwidth-bound 的是 decode</footer>
</div>

Decode 每步只为最新 token 做一次前向：读全部层的权重，读该请求截至目前的 KV，写回一行新的键值，发出一个 logits。与[Prefill](/llm/prefill-compute) 相反，权重无法被长序列摊销，KV 还随已生成长度 $n$ 线性膨胀。现代 GPU 的 Tensor Core 峰值远高于把这些字节从 HBM 搬到片上的能力，于是出现 **显存墙**：加更多 FLOPS 几乎不降 TPOT，加带宽、减字节、加大 batch 里的复用才降。本篇把这堵墙写成不等式与屋顶线，压缩手段见 [KV 预算](/llm/kv-as-long-context) 与 [GQA](/llm/gqa)。

## 问题

一步 decode 的主导流量大约是：参数字节（若权重不驻 L2）加上

$$
\mathrm{KV\ bytes}\propto L\cdot n\cdot h_{\mathrm{kv}}\cdot d\cdot 2\cdot b,
$$

$L$ 层数，$n$ 当前上下文，$b$ 元素宽度。算术量则是对 $n$ 个键的点积加 FFN 的一次「瘦」GEMM（激活行数为 1 或小 batch）。算术强度随 $n$ 上升（注意力部分），但权重项仍是每步扫一遍巨量参数——除非 batch 大到同一份权重服务许多请求。小 batch、长上下文的聊天，正好落在强度最低的角落：既要扫权重，又要扫很长的 KV。

产品表现是：GPU 利用率仪表显示很低，但延迟已经顶满；换一张算力翻倍、带宽差不多的卡，吐词速度几乎不动。这不是实现没写好，是工作点在带宽屋顶下。R1 类长思维链把 $n$ 再乘一截，墙从「并发稍高就 OOM」变成「单请求也搬不动」。

### 两堵墙：容量与带宽

容量墙：KV 与权重之和超过 HBM，请求进不去或被换出。带宽墙：放得下，但每步要搬的字节 / $B_{\mathrm{HBM}}$ 大于核计算时间。量化、[MLA](/llm/mla)、驱逐解决容量，也降低带宽压力；它们对 prefill 的 TTFT 帮助是另一张表。只报「能跑 128K」而 TPOT 从 20ms 变成 200ms，用户感受到的仍是墙。

<span class="marginnote">查询头数不进 KV 字节公式。GQA 减的是 $h_{\mathrm{kv}}$。把 MHA 的头数代入 decode 流量，会把带宽需求高估 $h_q/h_{\mathrm{kv}}$ 倍，规划出来的卡型是错的。</span>

## 方法

先把一步时间估成

$$
T_{\mathrm{step}}
\;\gtrsim\;
\frac{W_{\mathrm{bytes}}+K_{\mathrm{bytes}}(n)}{\eta_b\,B_{\mathrm{HBM}}}
$$

再与计算时间 $\mathrm{FLOPs}/(\eta_c\,\mathrm{peak})$ 取大者。decode 上前者几乎总更大。优化按分子、分母分类：

- 减 $W_{\mathrm{bytes}}$：权重量化、MoE 只加载被点到的专家（命中率低时反而更疼）。
- 减 $K_{\mathrm{bytes}}$：GQA/MQA/MLA、KV 量化、滑窗、驱逐、更短的思维链策略。
- 增复用：连续批处理让同一份权重乘在许多序列上，强度随 batch 升，工作点向计算屋顶挪——这是 vLLM 一类系统用分页 KV 换高并发的原因之一。
- 增 $B_{\mathrm{HBM}}$：换 HBM 规格，比换 FLOPS 更对症。

推测解码用草稿模型多吐几个 token 再验证，企图用额外计算换更少的逐步权重扫描；它只有在验证接受率高、且主模型仍受带宽限制时才划算。接受率低就变成更贵的 prefill 式核。

```mermaid
flowchart TD
  STEP["一步 decode"] --> W["读权重 W"]
  STEP --> K["读 KV(n)"]
  W --> ROOF{"屋顶线"}
  K --> ROOF
  ROOF -->|字节/带宽 > FLOPs/峰值| MEM["带宽墙：加算力无用"]
  ROOF -->|batch 很大、n 仍短"| CMP["可能靠近计算屋顶"]
  MEM --> FIX["减字节 / 加带宽 / 加 batch 复用"]
```

### 批处理不能消灭 $n$

Batch 摊的是权重。每个请求自己的 KV 仍要各自搬，总 KV 流量随并发 $\times$ 平均 $n$ 涨。高并发 + 长上下文是容量与带宽同时爆的象限。分页 KV 减少碎片，不减少渐近字节。前缀共享（同一系统提示）可以让多请求读同一段 KV，这是少数真正减 $K$ 流量的调度手段，与算法压缩正交。

## 机制

屋顶线把核标成强度（FLOPs/字节）。Prefill 强度高，走计算顶；decode 强度低，走带宽顶。注意力在 $n$ 极大时强度回升，但此前权重扫描已经把每步时间钉住，且 $n$ 极大时容量墙往往先倒。因此「长上下文 decode 是注意力二次」只说对了累积 FLOPs，没说对逐步延迟的主导项——逐步延迟经常是线性扫 KV 的带宽。二次项在 prefill 和训练上更致命。

长链 RL 模型把分布推向更大的 $n$，等于主动走进墙里。测试时缩放加 $N$ 或加树节点，是在并发维或节点维再乘流量。计算最优策略必须把带宽、而不是峰值 TFLOPS，当成稀缺资源。PRM 每步额外前向若也是瘦 GEMM，同样撞墙，不能假设验证器「很小所以免费」。

<span class="marginnote">CUDA Graph 与 kernel fusion 减的是启动开销，对已经长达数十毫秒、被 HBM 主导的一步帮助有限。先减字节，再抠核。反过来，极短上下文、极小模型上，启动开销可以主导，那时 fusion 才是一阶。</span>

## 边界与工程取舍

### 规划写清并发、长度、精度三元组

同一模型在 $n=512$、batch=1 与 $n=32\mathrm{K}$、batch=8 上不是同一个瓶颈。SLA 应分 TTFT（prefill 计算顶）与 TPOT（decode 带宽顶）。把思考模式默认打开，等于把 $n$ 的分布右移，必须单独做容量。端侧 DRAM 更窄，墙更早到来，公式相同，见[端侧 KV 预算](/llm/on-device-kv)。

不要用 prefill 的 MFU 考核 decode 服务。不要把「H100 比 A100 算力高」直接外推成吐词倍数。量化 KV 会在长 $n$ 后半段累积误差，针测与推理链要回归，不能只看短对话。分离式服务把 decode 池放到带宽型卡上是对症的，但 KV 从 prefill 池搬过来的时间要计入 TTFT，长前缀上可能得不偿失。投机解码、多 token 头同样是在用额外计算换更少的逐步扫权重；接受率一掉，每步又变回带宽墙，只是多付了草稿模型的流量。把这些技巧写进容量规划时，应同时给出接受率下限，而不是只给理想加速比。

<span class="marginnote">屋顶线见 Williams et al., 2009。推理阶段划分见 Pope et al., 2023 与 DistServe（Zhong et al., 2024）。PagedAttention 见 Kwon et al., 2023。不要给「decode 显存墙」本身编造一篇独立 arXiv。</span>

## 小结

- Decode 逐步查询长度为 1，工作点通常在带宽屋顶；加 FLOPS 不降 TPOT。
- 流量来自每步扫权重加扫 $O(n)$ 的 KV；容量墙与带宽墙要分开治。
- GQA/MLA/量化/驱逐减分子；HBM 规格与 batch 复用改分母或强度。
- Batch 摊权重不摊各请求自己的 KV；长 $n$ 高并发是最坏象限。
- 长思维链与测试时搜索是在主动增加 $n$ 或并发，必须按带宽记账。
- 考核 decode 不要用 prefill MFU；TTFT 与 TPOT 分表。
- 出处：Williams et al., *Roofline*, CACM 2009；Pope et al., 2023；Kwon et al., vLLM, 2023；Zhong et al., DistServe, 2024。
