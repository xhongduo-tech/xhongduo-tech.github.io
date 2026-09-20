---
title: Apple MLX
date: 2026-09-08
section: llm
---

# Apple MLX

<div class="epigraph">
<p>统一内存让权重、KV 与激活住在同一块 DRAM 里：没有 PCIe 拷贝，但屋顶线换成了内存控制器与 GPU/ANE 能同时咬多少带宽。</p>
<footer>—— Hannun 等，MLX 框架；Apple Silicon 统一内存是硬件前提</footer>
</div>

[上一课](/llm/openvino)在 Intel 上讲共享内存。Apple Silicon 把 CPU、GPU 做成统一内存（UMA）：MLX 用这点做数组框架，延迟求值、在 GPU 上跑 Transformer。llama.cpp 的 Metal 后端是另一条路径，见 [llama.cpp](/llm/llamacpp)。本课写 MLX 的差：Python 研究友好、统一内存上 *驻留策略简化*、以及 decode 仍被内存带宽绑住——只是没有 PCIe 那一项。不把某一 M 系列芯片的 GB/s 写成科学常数。

## 问题

CUDA 服务的加载是盘 → 主机 → HBM。[权重加载](/llm/weight-loading-streaming)的 PCIe 在 Mac 上不存在同一形态：GPU 直接看统一内存中的数组。缺口是：程序员容易以为「没有拷贝 = 没有带宽墙」。decode 每步仍要扫 $W+\mathrm{KV}$，墙是 DRAM 控制器。统一内存上 CPU 与 GPU 争同一带宽，一边 Python 预处理、一边 GPU decode，会互相伤害。

量化（4-bit）同样关键：笔记本级 DRAM 容量是容量墙。MLX 的量化类型与 GGUF k-quant 不是同一码本，质量数字不可直接比。

<span class="marginnote">统一内存可以类比办公方式：独立显卡像「车间自带仓库」，CPU 和 GPU 各管各的仓库，要共享材料就得开车搬运（PCIe 拷贝）；统一内存像「全公司共用一个大仓库」，GPU 直接在架子上取料，省掉了搬运工。但共用仓库也意味着通道只有一条——大家同时取料就会排队，这正是下文「CPU 与 GPU 争带宽」的由来。</span>

<span class="marginnote">ANE（神经引擎）与 GPU 不是自动可互换。MLX 主路径是 GPU；把算子下沉 ANE 有形状限制，LLM decode 不总适合。</span>

## 方法

用 MLX 实现或社区 LLM 包：权重以框架格式或转换脚本进统一内存，generate 循环在框架内。批处理 $B\gt 1$ 在本地聊天少见；若做，UMA 上加大 $B$ 仍摊权重，拐点逻辑同前，只是峰值不同。与 Python 互操作：大数组应保持 MLX 端，避免 `numpy()` 隐式拷到 CPU 再拷回。流式 detokenize 仍在 CPU，注意不要每 token 触发大同步。

```mermaid
flowchart TD
  W["权重在统一内存"] --> GPU["Metal GPU 扫 W+KV"]
  KV["KV 同一块 DRAM"] --> GPU
  CPU["CPU 分词 / 采样辅助"] --> BUS["争用内存带宽"]
  GPU --> BUS
```

## 机制

UMA 消灭设备拷贝，不消灭访存能量与带宽。Horowitz 的 DRAM 能量项仍在。研究迭代（改模型、立刻跑）是 MLX 的强项；多租户 SLA 不是——没有 HBM 池与成熟的连续批生态可与 vLLM 对标。选择 MLX 是为了本地与研究，不是把数据中心服务搬到 Mac Studio 就结束会计。

```mermaid
flowchart TD
  CMP["decode 每步的取数路径"] --> CUDA["CUDA 服务器：盘到主机到 HBM"]
  CMP --> MAC["Mac UMA：权重与 KV 同一块 DRAM"]
  CUDA --> C1["PCIe 拷贝是一道额外的墙"]
  MAC --> M1["无拷贝，但 DRAM 控制器是唯一的墙"]
  C1 --> SAME["共同点：每步都要扫一遍权重加 KV"]
  M1 --> SAME
  SAME --> TP["token 速度上限约等于带宽除以每 token 字节"]
```

<span class="marginnote">「没有拷贝不等于没有带宽墙」可以用数字感受：4-bit 量化的 7B 模型权重约 4 GB，decode 每生成一个 token 都要把全部权重读一遍。假设内存带宽 100 GB/s，粗略上限就是 100 ÷ 4 ≈ 25 token/s——这就是本地聊天「够用但不飞快」的物理原因，跟软件写得好不好关系不大。</span>

## 边界

不要用 MLX 的 eager 研究脚本当生产多用户引擎。不要把 Metal 与 CUDA FA 的加速比横比而不钉 $n,B$。下一课：浏览器里的 WebGPU，连 Python 都没有。

<span class="marginnote">术语翻译：eager（急切执行）就是「写一行算一行」，像做菜时切一样炒一样；MLX 默认「延迟求值」，先把整张做法记下来，到 `eval` 时一口气开火——好处是框架能合并操作、少搬数据。初学者常忘了显式求值，以为自己已经算完了，一取结果才发现还在排队。</span>

出处：MLX 项目（Hannun 等）；Apple Silicon 统一内存硬件文档。

## 小结

- MLX 利用 UMA：无 PCIe 拷贝，decode 仍绑 DRAM 带宽。
- CPU 与 GPU 争用同一总线；预处理要让路。
- 量化码本与 GGUF 不同，质量分开测。
- 适合本地与研究，不是数据中心连续批。
- 避免无意义的 `numpy()` 往返。
- 下一课：WebGPU 推理。
- 出处：MLX；Apple Silicon 统一内存。
