---
title: 如何阅读和评估一个推理引擎的源码
date: 2026-08-07
---

# 如何阅读和评估一个推理引擎的源码

<div class="epigraph">
<p>程序首先是写给人读的，只是顺便让机器执行。</p>
<footer>—— 哈罗德 · 阿贝尔森 与 杰拉尔德 · 杰伊 · 萨斯曼（Harold Abelson & Gerald Jay Sussman，《计算机程序的构造和解释》）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 第二章 ｜ 2026-08-07</p>
</div>

## 为什么读源码是评估引擎的「最后一公里」

前两节给了你坐标系：五类核心问题、训练与推理的框架差异、四家引擎的技术路线。但坐标系是静态的地图，**引擎是活的、持续演化的代码**。文档会过时、博客有滤镜、评测有批次倾向——只有源码不会骗人。

读源码不是要把每个文件都读一遍（那要数月），而是**带着问题、沿主干走一遍**，在关键节点停下来评估。这一节给出一套可复用的方法：先建心智模型，再顺着请求生命周期走，最后用四把尺子打分。这套方法同样适用于 vLLM、SGLang、TensorRT-LLM——名字换了，骨架是同一个。<span class="marginnote">动机很实际：这一专题接下来有整整两章在拆 vLLM/SGLang 的源码（PagedAttention、块表、调度器）。本文给的是「读引擎源码」的通法，后面是它的具体演练。</span>

## 1 先建心智模型：把文档读成一张图

打开一个仓库，克制住「立刻翻代码」的冲动。**先花 20 分钟把文档读成一张图。** 三样东西必看：

- **README 与 docs/**：回答「这个引擎的目标用户是谁、主打特性是什么、架构分成几块」。vLLM 的 docs 里有专门的架构图；SGLang 的 `docs/backend` 讲 Router 与 Runtime 的分工；TensorRT-LLM 的 docs 讲编译流程。
- **`examples/` 与快速上手**：看「用户怎么用」，能反推出引擎对外暴露的抽象边界。
- **GitHub 的目录结构**：读顶层目录名，通常是引擎的「骨架图」。vLLM 顶层有 `engine/`、`core/`、`worker/`、`model_executor/`、`entrypoints/`，这已经是一张模块地图。

目标不是记住细节，而是**在脑子里立起一条主线**：请求从哪进、到哪算、从哪出。第二章第一篇的「请求生命周期」就是这条主线的最佳模板。

## 2 顺着一条请求的生命周期走

主线立起来后，沿着生命周期做一次「读代码巡游」。以 vLLM 为例，一条请求的落点大致是：

| 生命周期节点 | vLLM 关键模块 | 你要看的代码 |
| --- | --- | --- |
| 入口 | `entrypoints/openai/` | HTTP 路由、OpenAI 协议解析 |
| 排队与调度 | `core/scheduler.py` | 队列策略、抢占、批的拼装 |
| 显存分配 | `core/block_manager.py` | KV 块的分配与回收 |
| 模型执行 | `worker/` + `model_executor/` | 前向、注意力后端选择 |
| 采样与停止 | `model_executor/layers/sampler.py` | 温度、top-p、停止条件 |
| 返回 | `entrypoints/` + SSE | 流式输出 |

**读法只有一个原则：只读「被调用到的路径」。** 从 `api_server.py` 进入，一路顺着调用关系往下走，遇到分支就问「这一步是哪种情况」，遇到抽象层就往下钻一层。凡是不在这条路径上的文件，第一遍全部跳过。<span class="marginnote">判断「被调用到」最快的办法是看函数签名与 grep 调用点；很多引擎的调度器都写成纯逻辑类，方便单测，读起来很顺。SGLang 的调度在 `srt/core/scheduler.py`，Router 在 `srt/router/`，结构更直白。</span>

**带着问题读，而不是为了读完而读。** 比如：当一个请求到达时，它是立刻进 batch，还是排队？batch 满了怎么办——新请求等、旧请求被抢占、还是直接拒绝？KV 块从哪来，用完怎么还？这三个问题的答案，比背下十个类名有价值得多。

## 3 评估的四把尺子：正确性、性能、内存、调度

巡游完主干，用四把尺子给引擎打分。每把尺子对应一组「要问的问题」：

**尺子一：正确性。** 采样是否实现完整（温度、top-p、logit bias、种子）？并行下是否可复现？停止条件（EOS、长度上限、停止词）处理在哪里、是否可靠？**看采样器与后处理代码**。

**尺子二：性能。** 注意力用哪个后端、为什么？有没有 CUDA Graph？kernel 是否逼近 Roofline 上限（下一节给出验证方法）？**看 `model_executor/layers/attention` 与 worker 的执行循环**。

**尺子三：内存。** KV 怎么分、怎么回收、有没有碎片？权重与激活的预留是否合理？**看 block manager 与显存统计代码**。<span class="marginnote">内存是引擎里最「脏」也最能拉开差距的部分：连续分配、页式分配、写时复制、前缀复用，四种策略的显存利用率可以差出数倍。评估时直接找「分配/回收」两个函数，看最坏情况下浪费多少。</span>

**尺子四：调度。** 调度策略是什么（FCFS？优先级？）、抢占怎么处理、长请求会不会饿死短请求、批的大小怎么决定？**看 scheduler 的核心循环**——它通常是整个引擎里最值得读的一个类。

打分不需要给数字，**用「有没有、在哪里、为什么这样设计」来回答**。回答得越具体，说明你对这个引擎的理解越深。

## 4 公式解析：用 Roofline 交叉验证内核

读内核代码时，最容易犯的错是「觉得它很复杂所以很厉害」。**复杂度不等于性能**。要验证一个 kernel 是否真的逼近硬件上限，回到第一篇的 Roofline 模型：

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}, \qquad \text{上限} = \min(P_{\text{peak}},\ \text{AI} \times \beta)$$

逐步拆解这个「验证公式」：

- **第一步，算算术强度 $\text{AI}$**：对一个 kernel，数出它的浮点运算量 FLOPs 与访存量 Bytes。注意力内核的 FLOPs 与序列长度 $S$ 的关系、访存量与权重和 KV 的关系，第一篇的表格里都算过。
- **第二步，找驻点（ridge point）**：算力峰值 $P_{\text{peak}}$ 除以带宽 $\beta$，得到「算与访的分界线」。A100 上这个驻点约在几百 FLOPs/Byte 的量级。
- **第三步，判断内核落在哪一侧**：$\text{AI}$ 远小于驻点 → 访存密集，跑不满算力是正常的，该看它有没有榨干带宽；$\text{AI}$ 远大于驻点 → 算力密集，该看它的 FLOPs 利用率（矩阵乘有没有对齐、有没有低效分支）。
- **第四步，对照实测**：用 profiler（nsys/ncu）读 kernel 实际耗时，与 Roofline 上限比。**实测与上限差在 30% 以内是好内核，差 3 倍以上就要怀疑实现问题**——这时候才值得深入 CUDA 代码找原因。

这个公式给了你一把**免于被炫技代码带偏的尺子**：读内核时先算数，再决定要不要深挖，而不是被代码的复杂度牵着走。

## 5 以 vLLM 为例的具体路径

方法讲完，给一条可以照着走的 vLLM 具体路径（SGLang 的对应物已标注）：

1. **入口**：`vllm/entrypoints/openai/api_server.py` —— 看路由与 `AsyncLLMEngine` 的启动。
2. **引擎**：`vllm/engine/async_llm_engine.py` —— 异步事件循环，把「调度」与「执行」解耦（SGLang 对应 `srt/core` 的 Engine）。
3. **调度器**：`vllm/core/scheduler.py` —— 本轮评估尺子四的主战场（SGLang 对应 `srt/core/scheduler.py`）。
4. **显存**：`vllm/core/block_manager.py` —— 尺子三的答案所在（SGLang 对应 `srt/mem_cache/` 的 RadixCache）。
5. **执行**：`vllm/worker/worker.py` + `vllm/worker/model_runner.py` —— 看执行循环、CUDA Graph、模型前向。
6. **注意力后端**：`vllm/model_executor/layers/attention/` —— 尺子二；`backends/` 目录下能一次看到 FlashAttention、FlashInfer 等多个后端的选择逻辑。
7. **采样**：`vllm/model_executor/layers/sampler.py` —— 尺子一。

**辨析｜易错点：**

- **误区一：从 CUDA kernel 开始读。** kernel 是引擎的「末端」，没有调度与显存的上下文，读它如读天书。先读调度与生命周期，最后才下沉到 kernel。
- **误区二：把 GitHub 星星当评估结果。** 星星衡量社区热度，不衡量你的负载性能。真正该做的是：读完主干后，用第十篇的压测方法在你自己的 GPU 上跑基准。
- **误区三：只读主分支不看测试。** 测试是最小的「行为文档」：`tests/` 里往往直接写明了「什么情况下会触发抢占」「块表怎么处理越界」。读一个行为不明朗的函数，先搜它的测试。

## 6 小结

- **先建心智模型**：读 README、docs、目录结构，20 分钟立起「请求从哪进从哪出」的主线。
- **沿生命周期巡游**：入口 → 调度 → 显存 → 执行 → 采样 → 返回；只读被调用到的路径。
- **四把尺子**：正确性（采样与停止）、性能（内核与后端）、内存（分配与回收）、调度（策略与抢占）。
- **Roofline 交叉验证**：算 kernel 的算术强度，与驻点对比判断它是访存密集还是算力密集，再拿实测与上限对照，免被复杂度带偏。
- **具体路径**：vLLM 从 `api_server` → `async_llm_engine` → `scheduler` → `block_manager` → `worker` → `attention` → `sampler`，一路走到底。

在下一节，我们进入 vLLM 的第一座丰碑——**PagedAttention：KV Cache 的页式内存管理**。你将看到虚拟内存的古老思想，如何在 GPU 上让 KV Cache 的利用率翻四倍。
