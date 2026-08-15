---
title: 压测工具实践：vLLM bench、genai-perf
date: 2026-08-07
---

# 压测工具实践：vLLM bench、genai-perf

<div class="epigraph">
<p>压测的意义不在数字本身，而在数字之间的规律。</p>
<footer>—— 性能测试共识（借自 Benchmarking 实践）</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ vLLM 文档与 NVIDIA genai-perf 文档 ｜ 2026-08-07</p>
</div>

## 为什么从压测工具开始

前面几篇讲了「该测什么指标」「曲线长什么样」，现在到了动手测。LLM 压测工具已相当成熟：**vLLM 自带的 benchmark 脚本**（面向引擎内部）与 **NVIDIA genai-perf**（面向服务 API，基于 Triton）是最常用的两个。会用工具不难，难的是**知道每个参数在测什么、怎么设计一次能回答问题的压测**。<span class="marginnote">本专题《并发实验》会讲怎么设计实验找拐点；本篇先讲<strong>工具怎么用</strong>——工具是锤子，设计是木工。</span>

本篇讲 vLLM bench 与 genai-perf 的用法、关键参数、以及「一次合格的压测」的流程。

## 1 vLLM benchmark 脚本

vLLM 提供一组 benchmark 脚本，最核心的是 `benchmark_serving.py`（面向在线服务）与 `benchmark_latency.py`（面向单请求延迟）。

`benchmark_serving.py` 模拟并发请求打向服务，关键参数：

```bash
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint /v1/completions \
    --request-rate 8 \
    --num-prompts 500 \
    --max-input-tokens 1024 --max-output-tokens 128
```

**`--request-rate`**：模拟「每秒来多少请求」。设成 `inf` 就是「打满」压测（压出吞吐上限）；设成固定值就是「负载测试」（在给定 QPS 下看延迟是否达标）。
**`--max-input-tokens` / `--max-output-tokens`**：固定输入输出长度，让结果可比。<span class="marginnote">固定长度是压测的基本功：<strong>长度分布不同，延迟与吞吐就没有可比性</strong>。真实负载通常是分布式的，但压测先「固定长度」找规律，再「混合长度」验真实性。</span>

输出报告：`TTFT`、`TPOT`、`E2EL` 的 P50/P95/P99，以及请求吞吐、token 吞吐、成功请求数、总耗时。

`benchmark_serving.py` 还能用 `--save-result` 把完整结果写成 JSON——里面是**逐请求的明细**（每条的 TTFT、TPOT、E2E 与 token 数）。**用明细而非控制台摘要做分析**：你可以自己算 P99、过滤掉失败请求、按输入长度分组——摘要只是统计，明细才是数据，后续做延迟归因（第十篇）也靠它。

## 2 genai-perf：服务层的标准压测

**genai-perf** 是 NVIDIA 推出的 LLM 压测工具（基于 Triton 生态），支持 OpenAI 兼容端点，参数设计更面向「服务」视角：

```bash
genai-perf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint-type openai \
    --endpoint /v1/completions \
    --concurrency 32 \
    --input-tokens 1024 --output-tokens 128
```

关键差异：

**`--concurrency`**：直接指定并发数（而不是到达率），更适合「在给定并发下测吞吐与延迟」。
**`--input-tokens`**：用合成输入（无需真实文本）控制输入长度，省去构造数据集的麻烦。
**导出 JSON**：结果可导出，供自动化与后续分析。

**vLLM bench vs genai-perf 的选型**：vLLM bench 与 vLLM 引擎深度绑定、参数细；genai-perf 面向服务 API、厂商中立、适合「对比不同引擎」。**做引擎调优用 vLLM bench，做服务 SLA 验证用 genai-perf**。

## 3 一次合格压测的流程

工具熟练后，压测的「设计」决定成败。一次合格的压测流程：

1. **明确问题**：要回答「吞吐上限是多少」「并发 32 时 P99 达标吗」「两个配置哪个好」。**没有问题的压测只是跑数字**。
2. **预热**：发一批请求让 kernel 加载、内存分配稳定，再开始正式统计（见延迟测量篇）。
3. **控制变量**：一次只改一个变量（并发、长度、引擎参数），其余固定。同时跑多组时，**组间要复位服务**（清缓存、重启）避免相互污染。
4. **多档位扫描**：并发从低到高扫一组（1、4、8、16、32、64…），画出「并发-吞吐-延迟」曲线，找拐点。<span class="marginnote">多档位扫描是压测的「重头戏」：<strong>只测一个并发点看不到曲线的形状</strong>。拐点、甜点区、过载区都需要多点描出来。</span>
5. **复现确认**：关键结论跑 2–3 遍，确认数字稳定（波动 < 5% 可接受），排除偶发噪声。

**预热与复位是实验的基本卫生**：GPU 上的 kernel 首次调用要加载或编译，显存分配也要稳定；服务端的热缓存会让第二批请求快得出奇。生产经验的底线是「预热至少一个完整 batch、组间重启服务」，否则数字之间的差异会被「冷热状态」污染，而不是被「配置差异」解释。

**辨析｜易错点：压测数字 ≠ 生产数字。** 压测用固定长度、合成输入、无缓存命中，生产有真实分布、缓存命中、网络抖动——**压测是「上界/下界参考」，不是「生产承诺」**。生产容量要以「压测 + 线上监控」双轨校准。

## 4 参数对照：两个工具的旋钮

把两个工具的关键参数放进一张表，看各自管的是「哪一层的旋钮」：

| 维度 | vLLM bench（引擎内） | genai-perf（服务外） |
| --- | --- | --- |
| 压力模型 | `--request-rate`（到达率） | `--concurrency`（并发数） |
| 长度控制 | `--max-input-tokens` / `--max-output-tokens` | `--input-tokens` / `--output-tokens` |
| 输入构造 | 真实 token 化 | 合成输入（无需语料） |
| 端点 | 引擎原生 / OpenAI | OpenAI / 自定义 Triton |
| 输出 | 控制台分位数 | JSON 可导出 |
| 最佳用途 | 引擎调优、参数扫描 | 跨引擎对比、SLA 验证 |

**读这张表的关键**：vLLM bench 的 `--request-rate` 是「负载测试」的旋钮——固定 QPS 下看延迟是否达标；genai-perf 的 `--concurrency` 是「容量测试」的旋钮——固定并发下压出最大吞吐。**选工具先选「要回答哪类问题」**：引擎内部改参数用 vLLM bench，服务对外承诺用 genai-perf。

## 5 数值算例：读懂一次压测报告

假设一次 vLLM bench 跑出：请求吞吐 8.2 req/s、token 吞吐 2460 tok/s、TTFT P95 = 1.1 s、TPOT P95 = 35 ms、E2E P95 = 12.3 s、成功请求 498/500。读这份报告的四步：

- **第一步，看成功率**：498/500 里的 2 个失败要先查——是超时、连接重置还是 `max_tokens` 截断？**失败不是噪声，是信号**，报告数字都建立在「成功的请求」之上。
- **第二步，对暗账**：$\text{TPS} = \lambda \cdot \bar{N}$。8.2 × 300（平均输出）≈ 2460，与报告的 token 吞吐吻合——**对得上说明压测设置自洽**；对不上则要查统计口径（是否混入了 prefill 的 token）。
- **第三步，看长尾**：P50 与 P95 的差距是「长尾」的度量。TTFT P95 = 1.1 s 若 P50 = 0.4 s，说明约 5% 的请求在排队——多半已接近拐点，需要回看并发曲线。
- **第四步，与基线比**：单次数字没有意义。**换参数再跑一组，看 TTFT/TPOT/吞吐往哪个方向动**——压测的价值在「数字之间的规律」，不在单个数字本身。

**一个把「读报告」变成「做决策」的诀窍**：把报告数字代入下一节的换算公式——先估拐点附近的单实例吞吐上限，再反推「目标 QPS 需要几台」。报告本身不回答「该扩几台」，但换算让它能回答。

## 6 公式解析：压测结果的换算

压测输出要换算成容量规划用的数字。设压测得 QPS $= \lambda$，平均生成长度 $\bar{N}$，则：

- **第一步，算 TPS**：$\text{TPS} = \lambda \cdot \bar{N}$。若 QPS=100、平均输出 500 token，TPS=50k——**这是「算力吞吐」，跨配置可比性更好**。
- **第二步，算单实例拐点**：从「并发扫描」曲线读拐点并发 $L_{\text{knee}}$ 与对应吞吐 $\lambda_{\text{knee}}$。
- **第三步，算实例需求**：目标 QPS $\lambda_{\text{target}}$ 下，所需实例数 $K = \lceil \lambda_{\text{target}} / \lambda_{\text{knee}} \rceil$（按拐点吞吐，而非线性外推——过拐点加并发不涨吞吐）。**容量规划的起点就是这张换算表**。

## 7 小结

- **vLLM bench** 面向引擎：`--request-rate` 模拟到达率、`--max-input-tokens`/`--max-output-tokens` 固定长度，输出 TTFT/TPOT/吞吐的分位数。
- **genai-perf** 面向服务：`--concurrency` 指定并发、合成输入、厂商中立，适合跨引擎对比与 SLA 验证。
- **选型**：引擎调优用 vLLM bench，服务验证用 genai-perf。
- **合格压测五步**：明确问题 → 预热 → 控制变量 → 多档位扫描 → 复现确认。
- **压测 ≠ 生产**：固定长度无缓存，数字是参考不是承诺，要与线上监控双轨校准。

在下一节，我们专门讲「多档位扫描」怎么读——**并发实验：找到服务的拐点与饱和点**。
