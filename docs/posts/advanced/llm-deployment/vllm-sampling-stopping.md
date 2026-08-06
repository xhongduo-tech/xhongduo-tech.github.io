---
title: vLLM 的采样、停止条件与后处理
date: 2026-08-07
---

# vLLM 的采样、停止条件与后处理

<div class="epigraph">
<p>随机数的产生太过重要，不能交给运气。</p>
<footer>—— 罗伯特 · R · 科维尤（Robert R. Coveyou），橡树岭国家实验室，1969</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型部署 ｜ LLM推理引擎体系 vLLM 采样与后处理 ｜ 2026-08-07</p>
</div>

## 为什么采样与停止值得单独一篇

前面几篇文章都在讲「怎么让模型算得快」：KV Cache 怎么省、调度器怎么排、前向怎么跑。但模型每步输出的其实只是一堆 **logits**（词汇表上的实数），从 logits 到「用户看到的文本」，中间还隔着三层工程：**采样**决定选哪个 token，**停止条件**决定生成到哪里结束，**后处理**把 token id 拼成文本、把概率汇总成 logprob。<span class="marginnote">这三个环节占了推理服务「非计算」开销的大头，也是许多「引擎输出和本地脚本不一致」的疑难杂症的源头——OpenAI 兼容 API 的多参数语义，本质上就是这三层工程的对外接口。</span>本篇文章从 `SamplingParams` 出发，把 vLLM 的采样管线、停止判定、输出组装一次讲透。

## 1 从 logits 到 token：Sampler 的职责

采样发生在**模型最后一层**之后：模型给出形状为 `[num_seqs, vocab_size]` 的 logits 张量，`Sampler` 负责把它变成 `sampled_token_ids`（每条序列一个 token id）。V0 里它在 `vllm/model_executor/layers/sampler.py`，V1 里移到 `vllm/v1/sample/sampler.py`，但管线顺序一致：

```python
# Sampler.forward 的处理管线（顺序敏感，不能乱）
# 1. 把 logits 转成 float32（数值稳定）
# 2. 应用 allowed tokens / bad words 过滤
# 3. 应用 logit bias / min-tokens 等「非 argmax 不变」处理器
# 4. 应用惩罚项：repetition / frequency / presence penalty
# 5. 温度缩放（logits / temperature）
# 6. top-k / top-p 截断
# 7. 随机采样（或贪心 argmax）
# 8. 收集 logprob 输出
```

**这个顺序不是随意的，每一步都在为下一步准备分布**：惩罚先于温度，保证惩罚的相对强度不随温度漂移；温度先于 top-p，让截断作用在真实的采样分布上。<span class="marginnote">「非 argmax 不变」处理器（如 logit bias、min-tokens 处理器）不改变 token 的排序，所以可以在采样前就地修改 logits；而惩罚项会改变排序，必须小心地按 token 出现历史计算。这里每一步的处理对象都是同一个 logits 张量，因此 logprobs 必须在破坏性修改前先备份。</span>

温度、top-p、softmax，这些操作本质上是《概率论与数理统计》里的分布变换与截断；而 logprob 用对数度量概率，正是《信息论》的语言——信息量 $-\log p$ 越大的 token 越「意外」，采样就是在信息量上做博弈。采样器不是玄学，它就是一套被工程化的概率管道。

## 2 SamplingParams：一份「生成说明书」

每次请求携带的 `SamplingParams` 就是采样与停止的全部配置。常用字段如下：

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `temperature` | 1.0 | 除以该值后做 softmax；0 表示贪心 |
| `top_k` | -1 | 保留概率最高的前 k 个 token；-1 表示不启用 |
| `top_p` | 1.0 | 保留累积概率达到 p 的最小 token 集 |
| `min_p` | 0.0 | 丢弃概率低于「最高概率 × min_p」的 token |
| `presence_penalty` | 0.0 | 对「出现过」的 token 施加线性惩罚 |
| `frequency_penalty` | 0.0 | 对「出现次数」成比例的线性惩罚 |
| `repetition_penalty` | 1.0 | 对已出现 token 按乘性惩罚 |
| `stop` / `stop_token_ids` | 无 | 停止字符串 / 停止 token id 列表 |
| `ignore_eos` | False | 为 True 时忽略 EOS，继续生成 |
| `max_tokens` | 16 | 生成的最大新 token 数 |
| `n` / `best_of` | 1 / 1 | 并行采样条数 / 候选条数 |
| `seed` | 无 | 随机种子，保证可复现 |
| `logprobs` / `prompt_logprobs` | 无 | 是否返回 top-k 概率与对数概率 |

<span class="marginnote">`seed` 是逐序列设置的随机种子：同一条请求、同一个 seed，在相同硬件与版本下可复现相同输出——这是做回归测试、对比引擎行为差异时最便宜的手段。注意跨版本、跨 CUDA 库的 RNG 实现差异仍可能导致不可复现。</span>这些参数最终都会被压进 `SamplingMetadata`，在采样器里逐序列生效。

## 3 公式解析：温度、Top-P 与惩罚如何改写分布

采样分布的计算分四步，我们全部展开。给定原始 logits $\text{logits}_i$，词汇表下标 $i$：

**第一步，惩罚项（线性 + 乘性）**：

$$
\text{logits}'_i =
\begin{cases}
\text{logits}_i \div r, & \text{logits}_i > 0 \ \text{且 token } i \text{ 已出现} \\[2pt]
\text{logits}_i \times r, & \text{logits}_i \le 0 \ \text{且 token } i \text{ 已出现}
\end{cases}
\quad\Longrightarrow\quad
\text{logits}''_i = \text{logits}'_i - \text{freq\_pen} \cdot f_i - \text{pres\_pen} \cdot \mathbb{1}[f_i > 0]
$$

**第二步，温度缩放**：$\text{logits}'''_i = \text{logits}''_i \mathbin{/} T$（$T$ 即 temperature）。

**第三步，top-k / top-p 截断**：先保留概率最高的 $k$ 个，再在这些 token 中保留累积概率刚达到 $p$ 的最小集合，其余置为 $-\infty$。

**第四步，softmax 归一化**：

$$
p_i = \frac{e^{\text{logits}'''_i}}{\sum_j e^{\text{logits}'''_j}}
$$

对这条公式链做三步拆解：

- **拆解 1（温度）**：$T$ 越小，logits 之间的差距被放大，分布越尖；$T$ 越大，分布越平。取一个三 token 的玩具分布 $\text{logits} = [2.0, 1.0, 0.0]$：$T=1$ 时概率约 $[0.67, 0.24, 0.09]$；$T=0.5$ 时先变成 $[4.0, 2.0, 0.0]$，概率约 $[0.94, 0.06, 0.00]$，几乎锁死在第一个 token；$T=2.0$ 时约 $[0.47, 0.29, 0.24]$，近乎均匀。<span class="marginnote">温度改变的是「分布的陡峭程度」，不改变 token 的排序——所以 temperature=0 时直接走贪心 argmax，而不是「除以 0」。vLLM 对 $T<10^{-5}$ 一律按贪心处理。</span>
- **拆解 2（惩罚）**：repetition penalty 是**乘性**的（已出现 token 被压低或抬高），frequency penalty 是**线性的**（每出现一次扣一份），presence penalty 是**二值**的（只要出现过就扣一份）。三者叠加：设 `frequency_penalty = 0.5`，token A 已出现 2 次，则 $\text{logits}_A$ 额外减 $0.5 \times 2 = 1.0$，可能把原本最高的 A 直接拉下马。
- **拆解 3（top-p）**：top-p 作用于**惩罚与温度之后的条件分布**。续用 $T=1$ 的例子，top_p=0.5 时，A 单独的概率 0.67 已达标，于是只保留 A——「看似随机采样，实则退化成贪心」是 top_p 设太小最常见的现象；top_p=0.9 时保留 A+B（累计 0.91）。

## 4 停止条件：长度、EOS 与 stop 字符串

生成不能无限继续。vLLM 用 `StopChecker` 在每步输出处理后检查每条序列，按固定顺序判定终止原因（`StopReason`）：

```python
def _check_stopped(seq, max_tokens, stop_token_ids, stop_strs, ignore_eos):
    # 1) 长度上限优先
    if len(seq.output_token_ids) >= max_tokens:
        return StopReason.LENGTH
    # 2) 命中 EOS（除非 ignore_eos=True）
    if not ignore_eos and seq.output_token_ids[-1] == eos_token_id:
        return StopReason.EOS
    # 3) 命中 stop token 或 stop 字符串
    if last_token in stop_token_ids or any(s in decoded_text for s in stop_strs):
        return StopReason.STOP
    return None  # 继续生成
```

三种终止原因对应不同的终结态：`LENGTH → FINISHED_LENGTH_CAPPED`，`EOS → FINISHED_STOPPED`，`STOP → FINISHED_STOPPED`（finish_reason 记为 `"stop"`）。<span class="marginnote">stop 字符串的匹配发生在<strong>解码后的文本</strong>上，而不是 token id 上——因此多字节字符（如中文、emoji）跨 token 切分时，vLLM 需要小心地把「半个字符」的边界处理好。`include_stop_str_in_output` 控制停止字符串本身是否包含在返回文本里，默认不包含。</span>此外 `max_tokens` 是硬顶，`ignore_eos=True` 会让模型在没到长度上限前「永不主动停」。

## 5 后处理：从 token id 到 RequestOutput

每步输出处理（`LLMEngine._process_model_outputs`，V1 中为 EngineCore 的 `update_from_outputs`）对每条序列做四件事：

1. **追加 token**：把采样出的 token id 追加到 `output_token_ids`。
2. **累积对数概率**：`cumulative_logprob += logprob(sampled_token)`，供 `best_of` 挑选与 `n` 多路输出排序使用。
3. **增量 detokenize**：只对新追加的 token 做解码，并通过记录偏移把跨 token 的字节正确拼回 `output_text`。**这一步比整句重新 decode 快得多**，也是流式输出的基础。<span class="marginnote">增量 detokenize 的难点在「半个字符」：一个 UTF-8 中文可能被切进两个 token，必须缓存前一个 token 的尾字节，等下一个 token 到来再合并。`skip_special_tokens` 与 `spaces_between_special_tokens` 控制特殊 token 在文本里的去留。</span>
4. **判断停止**：调用第 4 节的 `StopChecker`，得到 `finish_reason`；组内所有序列都终结后，调度器释放资源并组装 `RequestOutput`（内含每条序列的 `CompletionOutput`：index、text、token_ids、cumulative_logprob、logprobs、finish_reason）。

**重点**：`RequestOutput` 是用户拿到的最终形态，但它不是一次生成的——每步都会产出一份「当前进度」的 `RequestOutput`（配合流式输出逐 token 推送）。离线批量推理与在线流式服务，差异只在于「是攒到结束再返回，还是每步都吐」。

## 6 辨析｜易错点

**辨析｜易错点：temperature=0 不是「极低温采样」，而是贪心。** 它走 `argmax` 分支，不做随机采样，所以 `seed` 对它无意义。想要「接近贪心但仍有变化」，用 $T=0.1$ 之类的小值，而不是 0。

**辨析｜易错点：top_p 作用在「惩罚 + 温度之后」的条件分布，而不是原始 logits。** 如果先算好 top_p 掩码再改 logits，截断集可能完全错位。这也是「本地脚本和引擎输出对不上」的高频原因。

**辨析｜易错点：presence_penalty 与 frequency_penalty 不是同一件事。** presence 只问「出没出现过」（$\mathbb{1}[f_i>0]$），对重复出现的 token 不再加码；frequency 按出现次数线性叠加，专门抑制「同一句话反复说」。OpenAI API 与 vLLM 的语义一致，但实现的细节（作用于 prompt 还是仅输出）要在版本间核对。

**辨析｜易错点：logprobs 返回的是「被采样 token」的概率，不是 argmax token 的。** 当采样没有选中概率最高的 token 时，`logprobs[token_id]` 反映的是真实采样结果；想要「最高概率是谁」，要读 `logprobs` 里最大的那一项，或显式请求 top-k logprobs。

## 7 小结

- **采样管线**：logits → float32 → 过滤 → 惩罚 → 温度 → top-k/top-p → 采样 → logprob，顺序敏感。
- **分布公式**：惩罚（乘性 repetition + 线性 frequency/presence）→ 温度缩放 → top-k/top-p 截断 → softmax；温度改变陡峭度、不改变排序。
- **停止条件**：`StopChecker` 按「长度 → EOS → stop 字符串」判定，对应 `FINISHED_LENGTH_CAPPED` 与 `FINISHED_STOPPED`。
- **后处理**：追加 token、累积 logprob、增量 detokenize、组装 `RequestOutput`；流式与离线只差「是否每步吐一次」。
- 常见坑：temperature=0 是贪心、top-p 作用在最终条件分布、presence 与 frequency 语义不同、logprob 是采样结果的概率。

在下一节，我们回到调度与显存之外的主题：如何让一个引擎同时服务多种微调出来的模型——这就是 **vLLM 多 LoRA 服务原理**。
