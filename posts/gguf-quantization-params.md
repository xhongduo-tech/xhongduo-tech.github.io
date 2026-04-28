## 核心结论

GGUF 量化参数选择，核心不是“哪个文件最小”，而是同时决定三件事：权重体积、数值误差、实际推理吞吐。`bpw` 是 bits per weight，白话说就是“每个参数平均占多少比特”；它越低，模型越小，但误差通常越大。对新手最实用的默认策略是：

1. 先用 `Q4_K_M` 跑通。
2. 如果任务对质量更敏感，再升到 `Q5_K_M`。
3. 如果是高质量代码、数学或结构化输出，再考虑 `Q8_0`。
4. 长上下文场景不要只看权重，`KV cache` 很快会重新成为主导成本。

权重体积的第一条近似公式是：

$$
W_{bytes} \approx P \times \frac{bpw}{8}
$$

其中 $P$ 是参数量，白话说就是“模型里一共有多少个可学习数字”。

可以先记住一个经验判断：

| 量化类型 | 定位 | 典型有效 `bpw` | 体积 | 精度 | 常见用途 |
|---|---:|---:|---:|---:|---|
| `Q4_K_M` | 默认折中 | 约 4.9 | 小 | 中 | 本地聊天、轻量 RAG |
| `Q5_K_M` | 偏质量 | 约 5.7 | 中 | 较高 | 代码、长链路问答 |
| `Q8_0` | 偏保真 | 约 8.5 | 大 | 高 | 高质量部署、质量对比基线 |

新手可以直接把它记成一句话：先用 `Q4_K_M`，不够再升，不要一开始只盯着文件大小。

---

## 问题定义与边界

这篇文章回答的问题不是“什么是量化”，而是“在 GGUF 场景里我该选哪个量化参数”。边界必须先定清楚，否则你会算错预算。

`weights` 是权重，白话说就是“模型本身存下来的参数”；`KV cache` 是注意力缓存，白话说就是“模型为了记住上下文，给每个 token 留下的中间状态”；`runtime overhead` 是运行时额外开销，白话说就是“框架、缓冲区、页映射、调度等不属于权重和 KV 的占用”。

| 成本项 | 作用 | 是否随上下文增长 | 常见误判 |
|---|---|---|---|
| `weights` | 存模型参数 | 否 | 只看它，觉得“文件能放下就能跑” |
| `KV cache` | 存历史 token 的 K/V | 是 | 上下文一长就爆内存 |
| `runtime overhead` | 后端与缓冲区开销 | 部分相关 | 忽略后导致预算过紧 |

后文统一使用这些记号：

| 记号 | 含义 | 白话解释 |
|---|---|---|
| `P` | 参数量 | 模型总参数个数 |
| `bpw` | bits per weight | 每个参数平均占多少比特 |
| `L` | 层数 | Transformer 叠了多少层 |
| `T` | 上下文长度 | 一次保留多少 token 历史 |
| `b_kv` | KV 单元素字节数 | KV cache 里每个数占几字节 |

玩具例子：你看到一个 `7B` 模型的 `Q4_K_M` 文件只有约 4 GB，很容易得出“我 8 GB 内存肯定够”的结论。这个结论不完整，因为你还没把 `KV cache` 和运行时开销算进去。

真实工程例子：12 GB 显存单卡跑 8B 模型，权重用 `Q4_K_M` 往往能落到可运行区间；但如果你把上下文拉到 `8k`，再开更大的 batch 或并行会话，`KV cache` 很可能吃掉剩余预算，导致实际 OOM 或强制回落到更慢的路径。

---

## 核心机制与推导

第一层机制是权重体积近似由 `P × bpw / 8` 决定。这个公式不是文件格式的完整真相，但足够做一轮部署前估算。

以 7B 模型做粗算：

- `fp16`：$7 \times 10^9 \times 2 \approx 14$ GB
- `Q4_K_M`：$7 \times 10^9 \times 4.5 / 8 \approx 3.94$ GB
- `Q5_K_M`：$7 \times 10^9 \times 5.5 / 8 \approx 4.81$ GB
- `Q8_0`：$7 \times 10^9 \times 8.5 / 8 \approx 7.44$ GB

这里是“玩具估算”。真实 GGUF 文件会包含元数据、张量差异和非完全均匀量化，所以和源码文档中的实测值会有偏差。比如 `llama.cpp` 当前文档里，Llama-3.1-8B 的 `Q4_K_M / Q5_K_M / Q8_0` 约为 `4.58 / 5.33 / 7.95 GiB`，比 7B 玩具估算更大，这是正常的。

第二层机制是 `K` 系列不是“每个数简单裁成 4 位或 5 位”。源码里 `Q4_K`、`Q5_K` 使用 256 元素 super-block，再切成更小 block，配合分层 `scale/min` 存储。白话说，它不是把每个参数独立压扁，而是“按块共享统计信息”，所以有效 `bpw` 会高于名义位宽。`Q4_K` 在 `ggml-common.h` 中的典型有效位宽约为 4.5，`Q5_K` 约为 5.5，`Q6_K` 约为 6.5625。

第三层机制是 `Q4_K_M` 里的 `M`。它不是一个新的基础编码格式，而是混合量化策略，白话说就是“不同张量不一定都用同一种量化方式”。`llama.cpp` 的 `--pure` 选项可以关闭这种 mixture。工程含义很直接：`Q4_K_M` 之所以常常比“纯 4-bit”更稳，是因为它在重要张量上不完全一刀切。

`KV cache` 的近似公式可以写成：

$$
KV_{bytes} \approx T \times \sum_{l=1}^{L}(d_{k,l}+d_{v,l}) \times b_{kv}
$$

其中 $d_{k,l}, d_{v,l}$ 是第 $l$ 层 K/V 的宽度。白话说，上下文每多一个 token，你就在每一层都多存一份 K 和 V，所以它会随着 `T` 线性增长。

如果某个 32 层模型每 token 的 KV 大约是 `128 KiB`，那么：

- `4k` 上下文约 `512 MiB`
- `8k` 上下文约 `1 GiB`
- `16k` 上下文约 `2 GiB`

这就是为什么“权重已经很小”并不等于“长上下文肯定能跑”。

再看一个权重与 KV 的对比：

| 场景 | 权重量化 | 权重占用 | KV cache 占用 | 总结 |
|---|---|---:|---:|---|
| 7B，4k ctx | `Q4_K_M` | 约 3.94 GB | 约 0.5 GB | 权重主导 |
| 7B，8k ctx | `Q4_K_M` | 约 3.94 GB | 约 1.0 GB | KV 开始明显 |
| 7B，16k ctx | `Q4_K_M` | 约 3.94 GB | 约 2.0 GB | KV 接近半壁江山 |
| 7B，8k ctx | `Q8_0` | 约 7.44 GB | 约 1.0 GB | 权重重新压顶 |

结论可以压缩成一句：短上下文先看权重量化，长上下文必须把 KV 当成一等公民。

---

## 代码实现

真正落地时，建议把流程固定成三步：先估算预算，再量化，再做任务集回归。不要只跑一句“你好”就宣布部署成功。

先看一个可运行的 Python 估算器：

```python
from math import isclose

def estimate_weight_gb(params_billion: float, bpw: float) -> float:
    params = params_billion * 1e9
    return params * bpw / 8 / 1e9

def estimate_kv_gb(kv_per_token_kib: float, ctx_tokens: int) -> float:
    return kv_per_token_kib * ctx_tokens / 1024 / 1024

q4 = estimate_weight_gb(7, 4.5)
q5 = estimate_weight_gb(7, 5.5)
q8 = estimate_weight_gb(7, 8.5)

assert isclose(round(q4, 2), 3.94, rel_tol=0, abs_tol=0.01)
assert isclose(round(q5, 2), 4.81, rel_tol=0, abs_tol=0.01)
assert isclose(round(q8, 2), 7.44, rel_tol=0, abs_tol=0.01)

kv_8k = estimate_kv_gb(128, 8192)
assert isclose(round(kv_8k, 2), 1.00, rel_tol=0, abs_tol=0.01)

print({
    "Q4_K_M_weight_gb": round(q4, 2),
    "Q5_K_M_weight_gb": round(q5, 2),
    "Q8_0_weight_gb": round(q8, 2),
    "KV_8k_gb": round(kv_8k, 2),
})
```

量化命令示例：

```bash
python3 convert_hf_to_gguf.py ./models/my-model/

./llama-quantize \
  ./models/my-model/ggml-model-f16.gguf \
  ./models/my-model/ggml-model-Q4_K_M.gguf \
  Q4_K_M

./llama-quantize \
  ./models/my-model/ggml-model-f16.gguf \
  ./models/my-model/ggml-model-Q5_K_M.gguf \
  Q5_K_M
```

如果你已经有 importance matrix，优先用它，而不是只做最朴素量化：

```bash
./llama-imatrix -m ./models/my-model/ggml-model-f16.gguf -f data/calibration.txt -o imatrix.gguf

./llama-quantize \
  --imatrix imatrix.gguf \
  ./models/my-model/ggml-model-f16.gguf \
  ./models/my-model/ggml-model-Q4_K_M.gguf \
  Q4_K_M
```

推理命令示例：

```bash
./llama-cli \
  -m ./models/my-model/ggml-model-Q4_K_M.gguf \
  -c 8192 \
  -ctk f16 \
  -ctv f16 \
  -p "Write a Python function that merges two sorted arrays." \
  --no-conversation
```

如果显存或内存主要卡在长上下文，可以直接测试 KV cache 量化，而不是先继续压权重：

```bash
./llama-cli \
  -m ./models/my-model/ggml-model-Q5_K_M.gguf \
  -c 8192 \
  -ctk q8_0 \
  -ctv q8_0 \
  -p "Summarize the following 6k-token context..." \
  --no-conversation
```

对比测试不要只看“能不能启动”，要看任务稳定性。最简单的回归方式是固定一组提示词，分别跑 `Q4_K_M` 和 `Q5_K_M`：

```bash
for model in Q4_K_M Q5_K_M; do
  ./llama-cli \
    -m ./models/my-model/ggml-model-${model}.gguf \
    -f prompts/code_eval.txt \
    -n 256 \
    --temp 0 \
    --seed 42 \
    > outputs/${model}.txt
done
```

一个足够实用的伪代码流程是：

```text
输入：设备预算、目标任务、上下文长度
1. 估算 weights + KV cache + runtime overhead
2. 默认尝试 Q4_K_M
3. 若任务是代码/数学/结构化输出，优先升级到 Q5_K_M
4. 若预算不够，优先缩短上下文或量化 KV cache
5. 对固定任务集做回归，不只看启动成功率
6. 记录吞吐、失败样例、格式错误率，再定最终量化版本
```

真实工程例子：做本地代码助手时，同样是 8B 模型，`Q4_K_M` 可能已经能交互，但当你要求稳定输出 JSON、写更长函数、少出语法错时，`Q5_K_M` 常常比单纯加一些采样参数更有效，因为问题根子在数值保真度，不在提示词花样。

---

## 工程权衡与常见坑

常见错误不是“不会量化”，而是预算和评测方法错了。

| 坑点 | 后果 | 规避策略 |
|---|---|---|
| 只看模型文件大小 | 长上下文时实际 OOM | 统一估算 `weights + KV + overhead` |
| 把 `Q4_K_M` 当通用最优解 | 代码、数学、JSON 输出不稳 | 对质量敏感任务先试 `Q5_K_M` |
| 从低位量化继续 `--allow-requantize` | 质量进一步塌陷 | 尽量从 `f16/f32` 原模型重新量化 |
| 忽略后端差异 | 混合量化收益不一致 | 部署前确认目标后端支持情况 |
| 只做人工体验测试 | 难复现回退 | 固定提示集和温度做回归 |

部署前检查清单：

- 是否估算了 `KV cache`
- 是否给上下文长度留了余量
- 是否做了任务集回归
- 是否确认后端支持混合量化
- 是否避免从低位模型继续 `--allow-requantize`
- 是否记录了吞吐、错误率、格式稳定性三类指标

经验上，客服问答这类短回答、容错高、上下文在 `4k-8k` 的任务，`Q4_K_M` 通常是划算的。代码补全、RAG 汇总、数理推理则更容易暴露低位量化误差，此时“更大一点但更稳”的 `Q5_K_M` 往往更省总成本，因为你少了很多失败重试和人工兜底。

---

## 替代方案与适用边界

显存或内存不够时，思路不只有“继续把权重压得更低”。很多时候，优先级应该反过来。

| 方案 | 优点 | 代价 | 适用边界 |
|---|---|---|---|
| 缩短上下文 | 立刻省 KV | 记忆范围变短 | 多轮不深、检索片段可裁剪 |
| `KV cache` 量化 | 保住权重质量 | 可能影响长上下文质量 | 长上下文预算紧张 |
| 提升权重量化位宽 | 明显提升稳定性 | 权重更大、吞吐可能下降 | 代码、数学、结构化输出 |
| 换更大显存/内存 | 效果最直接 | 成本高 | 长期稳定生产部署 |

适用场景可以这样看：

| 场景 | 默认建议 | 升级条件 | 不建议 |
|---|---|---|---|
| 客服对话 | `Q4_K_M` | 多语言稳定性不足时升 `Q5_K_M` | 一上来就追 `Q8_0` |
| 代码补全 | `Q5_K_M` | 质量基线要求高时试 `Q8_0` | 默认用过低位宽 |
| RAG 问答 | `Q4_K_M` 或 `Q5_K_M` | 上下文长时先看 KV 预算 | 只看权重不看上下文 |
| 数学推理 | `Q5_K_M` | 仍不稳时升 `Q8_0` | 把 `Q4_K_M` 当唯一答案 |

同样是 12 GB 单卡，本地聊天优先 `Q4_K_M` 很合理；如果你做的是高质量代码助手或数理推理助手，更合理的顺序往往是：先试 `Q5_K_M`，如果放不下，再缩上下文或量化 KV；只有这些都不够，才考虑更激进地继续压权重。

换句话说，量化参数选择不是“找最小文件”，而是“把有限预算分配给最影响结果的地方”。

---

## 参考资料

量化参数

1. [llama.cpp `tools/quantize/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)  
用途：确认 `llama-quantize` 命令、`--pure`、`--allow-requantize`、`--imatrix` 和当前各量化类型的体积/吞吐对照。

2. [llama.cpp `ggml/src/ggml-common.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)  
用途：确认 `Q4_K`、`Q5_K`、`Q6_K` 的块结构、super-block 设计和典型有效 `bpw` 定义。

KV cache

3. [llama.cpp `src/llama-kv-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp)  
用途：确认 KV cache 的 K/V 类型、内存统计输出和实现层面的占用逻辑。

后端行为

4. [llama.cpp `tools/cli/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)  
用途：确认 `--cache-type-k`、`--cache-type-v` 等运行参数，判断长上下文下是否优先做 KV cache 量化。

性能评估

5. [llama.cpp `tools/perplexity/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md)  
用途：确认量化后质量评估的常见方法，理解为什么不能只看“能不能生成”。

6. [llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)  
用途：确认项目总体能力边界、支持的量化范围和部署形态，避免把某个后端经验误当成通用结论。
