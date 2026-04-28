## 核心结论

TensorRT-LLM 的推理引擎配置，本质不是“上线后慢慢再调”的运行时参数，而是在 **build-time** 先把请求形状和显存预算固定下来。`build-time` 的白话解释是：引擎构建阶段，系统要提前决定这台引擎最多接多大的 batch、最多吃多长的输入、最多保留多少 token 的中间状态，以及用什么精度存数据。

可以把它理解成“先按桌数、包间大小和仓库存量装修餐厅”。装修完以后，线上来了超过上限的客流，不是“稍微慢一点”，而是根本接不住，必须排队、截断，或者直接拒绝。

判断配置是否正确，核心看它是否匹配你的目标 workload，而不是参数是否越大越好。这里的 `workload`，白话讲就是“你的线上请求长什么样”：短问答还是长上下文聊天，是低并发还是高并发，是更看重首 token 延迟还是总吞吐。

| 参数 | 含义 | 主要影响 | 能否运行时修改 |
|---|---|---|---|
| `max_batch_size` | 最大可调度请求数 | 激活内存、并发上限、吞吐 | 否，需重建引擎 |
| `max_input_len` | 单请求最大输入长度 | 可接受 prompt 长度、prefill 成本 | 否，需重建引擎 |
| `max_seq_len` | 单请求总长度上限 | KV cache 预算、长输出能力 | 否，需重建引擎 |
| `max_num_tokens` | 单批次 token 上限 | 激活内存、调度效率 | 否，需重建引擎 |
| `precision` / 量化方式 | 权重、激活、KV cache 位宽 | 显存、吞吐、精度风险 | 大多需重建引擎 |
| `multiple_profiles` | 多组优化 profile | 混合负载性能、确定性 | 否，需重建引擎 |

---

## 问题定义与边界

先把边界讲清楚：**引擎构建时的上限**，和**推理时实际送进来的请求**，不是一回事。

如果引擎按 `max_input_len=1024` 构建，线上来了一个 3000 token 的 prompt，这不是“慢一点也能跑”，而是这个引擎的 shape 边界本来就没覆盖它。`shape` 的白话解释是张量形状，也就是一批请求在长度和批大小上的尺寸。

下面用统一记号表示配置边界：

| 记号 | 含义 |
|---|---|
| `B` | `max_batch_size` |
| `L_in` | `max_input_len` |
| `L_out` | 最大输出长度 |
| `L_seq` | 总长度，通常可理解为 `L_in + L_out`，也对应 `max_seq_len` 的约束 |
| `N` | Transformer 层数 |
| `H_kv` | KV heads 数量 |
| `D` | 每个 head 的维度 |
| `s` | 每个 KV 元素的字节数，FP16/BF16 常见为 2，INT8/FP8 常见约为 1 |

`max_input_len` 只约束输入侧，`max_seq_len` 约束的是单请求总长度，包含 prompt 和生成输出。这两个参数不能混用。

| 阶段 | 负责什么 | 不能做什么 |
|---|---|---|
| `build-time` | 决定 optimization profile、算子选择、激活内存规模、KV cache 形态、精度方案 | 不能事后临时扩容 |
| `runtime` | 在已构建 profile 范围内调度请求、分配缓存、执行推理 | 不能突破已构建的 batch/context 边界 |

玩具例子很简单。假设你构建了：

- `max_batch_size = 8`
- `max_input_len = 1024`
- `max_seq_len = 2048`

那么下面三种请求里，只有第一种一定在边界内：

| 请求 | 是否可被该引擎接受 | 原因 |
|---|---|---|
| 4 个请求，每个输入 600 token，输出 300 token | 可以 | `B=4<=8`，总长 `900<=2048` |
| 2 个请求，每个输入 1500 token，输出 100 token | 不可以 | 输入长度超过 `max_input_len` |
| 16 个请求，每个输入 100 token，输出 50 token | 不可以 | batch 超过 `max_batch_size` |

---

## 核心机制与推导

TensorRT 的底层机制是 **optimization profile**。它会为输入张量定义 `min / opt / max` 三组边界。`opt` 的白话解释是“最重点优化的那一档”，不是唯一能跑的形状；只要请求落在 `min ~ max` 范围内，引擎都能执行，但通常对 `opt` 附近最友好。

TensorRT-LLM 把这些 profile 封装得更高层了，但 `max_batch_size`、`max_num_tokens` 这类 build 参数仍然会直接影响 profile 的生成方式和内存布局。

推理时显存主要看三部分：权重、激活、KV cache。做引擎配置时，最需要盯住的是后两者。

- 激活内存：模型执行中间张量占的内存
- KV cache：自回归生成时为每层保存历史 Key/Value 的缓存

简化公式可以写成：

$$
M_{act} \propto \text{max\_num\_tokens}
$$

在去 padding 的 packed 模式下，可以把它近似理解为“单批次总 token 数越大，激活预算越高”。而在 padded 思维下，很多人会先用：

$$
\text{max\_num\_tokens} \approx B \times L_{in}
$$

做一版粗估。

连续 KV cache 下，每层大致是：

$$
M_{kv,layer} = B \times 2 \times H_{kv} \times L_{seq} \times D \times s
$$

总 KV cache 约为：

$$
M_{kv} \approx N \times M_{kv,layer}
$$

这里乘以 2，是因为要同时存 Key 和 Value。

代入一个玩具数值例子：

- `N = 32`
- `H_kv = 8`
- `D = 128`
- `B = 4`
- `L_seq = 4096`
- `s = 2`，即 FP16 KV cache

则每层 KV cache 为：

$$
4 \times 2 \times 8 \times 4096 \times 128 \times 2
= 67{,}108{,}864 \text{ bytes}
\approx 64 \text{ MiB}
$$

32 层总计约：

$$
64 \text{ MiB} \times 32 = 2048 \text{ MiB} \approx 2 \text{ GiB}
$$

这就是为什么长上下文、较大 batch、FP16 KV cache 三者叠加后，显存会非常快地涨上去。把 KV cache 从 FP16 换到 INT8 或 FP8，理论上这部分大致可减半，但代价是硬件、插件支持和精度验证成本会上来。

真实工程例子更典型。假设你做一个企业知识库问答服务：

- 白天混合流量，既有 200 token 的短问答，也有 8k token 的长文档问答
- 你既要看吞吐，也要看首 token 延迟
- 用户希望长 prompt 时不要明显卡死

这时常见做法不是构建一个“所有上限都拉满”的单 profile 引擎，而是：

- 开启 `multiple_profiles`
- 长上下文场景启用 `use_paged_context_fmha`
- 用 `max_batch_size` 与 `max_num_tokens` 做 sweep
- 再按服务 SLA 看 TTFT 和 TPOT 是否可接受

`TTFT` 是首 token 时间，白话讲就是“用户发出请求到看到第一个字”的时间。`TPOT` 是首 token 之后每个输出 token 的平均时间，不包含 `TTFT`。很多配置会同时影响两者，但方向未必一致。比如 profile 更贴近实际 workload，可能 TTFT 略降、TPOT 明显改善；而盲目拉大 `max_batch_size`，则可能让激活内存膨胀，挤压 KV cache 预算，最终让吞吐和稳定性一起变差。

一个简化流程可以记成：

`请求进入 → profile 匹配 → 预分配激活 / KV cache → context 预填充 → generation`

---

## 代码实现

先给一个最小可运行的 Python 预算脚本。它不依赖 TensorRT-LLM，只是帮助你在构建前做一版 KV cache 粗估。

```python
def kv_cache_bytes(num_layers, kv_heads, head_dim, batch_size, seq_len, bytes_per_elem):
    return num_layers * batch_size * 2 * kv_heads * seq_len * head_dim * bytes_per_elem

def mib(x):
    return x / (1024 ** 2)

def gib(x):
    return x / (1024 ** 3)

total = kv_cache_bytes(
    num_layers=32,
    kv_heads=8,
    head_dim=128,
    batch_size=4,
    seq_len=4096,
    bytes_per_elem=2,  # FP16
)

per_layer = total / 32

assert round(mib(per_layer)) == 64
assert round(gib(total)) == 2

print(f"per_layer={mib(per_layer):.2f} MiB")
print(f"total={gib(total):.2f} GiB")
```

如果你只是想先构建一个覆盖常见服务场景的基础引擎，CLI 写法可以从下面开始。注意，较新的 `trtllm-build` 文档中，KV cache 类型更常见的写法是 `--kv_cache_type paged`。

```bash
trtllm-build \
  --checkpoint_dir /path/to/checkpoint \
  --output_dir engine_outputs \
  --max_batch_size 256 \
  --max_input_len 4096 \
  --max_seq_len 8192 \
  --max_num_tokens 8192 \
  --multiple_profiles \
  --use_paged_context_fmha \
  --kv_cache_type paged
```

对应的 Python `BuildConfig` 示例：

```python
from tensorrt_llm import BuildConfig, LLM

build_config = BuildConfig(
    max_batch_size=256,
    max_input_len=4096,
    max_seq_len=8192,
    max_num_tokens=8192,
)

build_config.plugin_config.multiple_profiles = True
build_config.plugin_config.use_paged_context_fmha = True
build_config.plugin_config.paged_kv_cache = True

llm = LLM(
    model="/path/to/model",
    build_config=build_config,
)

llm.save("engine_outputs")
```

如果你的主场景是短输入、较高并发，可以把 `max_input_len` 压低一些，把 batch 和 token 调度空间留给更多请求；如果你的主场景是长上下文聊天，通常要更谨慎地平衡 `max_seq_len` 与 `max_num_tokens`，避免 prompt 直接把显存打满。

| `trtllm-build` flag | `BuildConfig` 字段 | 影响对象 |
|---|---|---|
| `--max_batch_size` | `max_batch_size` | 并发上限、激活内存 |
| `--max_input_len` | `max_input_len` | 输入长度边界 |
| `--max_seq_len` | `max_seq_len` | 单请求总长度、KV cache 预算 |
| `--max_num_tokens` | `max_num_tokens` | 单批 token 预算、激活内存 |
| `--multiple_profiles` | `plugin_config.multiple_profiles` | 混合 workload 性能 |
| `--use_paged_context_fmha` | `plugin_config.use_paged_context_fmha` | 长 prompt prefill 行为 |
| `--kv_cache_type paged` | `plugin_config.paged_kv_cache=True` 或等价 KV cache 配置 | KV cache 组织方式 |

真实工程里还要加一层运行时 shape guard，也就是“请求入口检查”。白话讲，就是在请求真正进入推理层之前，先判断它会不会越界，避免无效请求把服务线程占住。

```python
def validate_request(input_len, output_len, batch_size, *,
                     max_input_len, max_seq_len, max_batch_size):
    if batch_size > max_batch_size:
        raise ValueError("batch_size exceeds engine limit")
    if input_len > max_input_len:
        raise ValueError("input_len exceeds engine limit")
    if input_len + output_len > max_seq_len:
        raise ValueError("seq_len exceeds engine limit")
    return True

assert validate_request(
    input_len=1024,
    output_len=512,
    batch_size=4,
    max_input_len=4096,
    max_seq_len=8192,
    max_batch_size=256,
) is True
```

---

## 工程权衡与常见坑

参数不是越大越好。配置的本质是预算分配，而不是堆上限。

最常见的误区是：看见服务偶尔排队，就先把 `max_batch_size` 拉大；看见用户要长文本，就先把 `max_seq_len` 拉满；看见量化能省显存，就直接上 FP8。这样做往往会把问题从一个指标转移到另一个指标。

先看典型坑位：

| 现象 | 根因 | 规避方式 |
|---|---|---|
| 长输入请求直接失败或被截断 | `max_input_len` 没显式设够，默认常见是 1024 | 长上下文场景显式设置 `max_input_len` |
| 引擎建好后想在线改 batch 上限 | 把 build-time 参数误当 runtime 参数 | 明确边界，必要时重建引擎 |
| 吞吐没涨，显存先爆 | `max_batch_size` 盲目拉大，激活预算过高 | 按实际流量做 sweep，而不是一次拉满 |
| 长上下文下并发突然很差 | `max_num_tokens` 和 `max_seq_len` 抢显存，KV cache 被挤压 | 配合 paged context、分页 KV cache 重算预算 |
| 量化后结果漂移明显 | 精度校验不足，量化策略与模型不匹配 | 先做回归测试，再决定是否上线 |
| 打开多 profile 后结果不完全一致 | 不同 profile 可能选到不同 kernel | 对强确定性场景保守使用 |

再看精度方案。这里的“精度”不是只看位宽，而是同时影响权重存储、算子实现、硬件要求和最终质量。

| 精度方案 | 显存 | 吞吐潜力 | 精度风险 | 硬件依赖 |
|---|---|---|---|---|
| FP16 | 较高 | 高 | 低 | 通用 GPU 支持较广 |
| BF16 | 较高 | 高 | 低 | 需硬件良好支持 BF16 |
| INT8 | 更低 | 可能更高 | 中 | 依赖量化流程与插件支持 |
| FP8 | 更低 | 很高潜力 | 中到高 | 对硬件与内核支持要求高 |

几个实用判断：

- `max_batch_size` 太小：请求排队，吞吐上不去
- `max_batch_size` 太大：激活内存被预留过多，反而压缩可用 KV cache
- `max_num_tokens` 太大：尤其在长上下文场景，会直接吃掉大量可用显存
- `FP8` 不是“无代价提速”：要看 GPU 架构、内核支持和精度回归结果

工程上最稳的做法通常不是一次拍脑袋定值，而是围绕真实流量做 sweep。比如固定模型和并行策略后，测试如下组合：

- `max_batch_size`: 64 / 128 / 256
- `max_input_len`: 1024 / 2048 / 4096
- `max_num_tokens`: 4096 / 8192 / 16384
- `multiple_profiles`: 开 / 关

然后分别记录：

- 峰值显存
- TTFT
- TPOT
- 稳态吞吐
- OOM 或 shape mismatch 次数

最终选的是“满足 SLA 的最小充分配置”，不是“纸面最强配置”。

---

## 替代方案与适用边界

不是所有场景都该追求最激进配置。配置策略要跟请求分布一起看。

如果你的服务主要处理短输入、低并发、强确定性任务，比如离线评测或固定模板生成，单 profile、FP16/BF16、相对保守的 batch 上限，通常更简单也更稳。

如果你的服务主要处理长上下文聊天、RAG、多租户混合流量，那么更适合把 profile、context 和 KV cache 策略拆开考虑。

| 方案 | 适用场景 | 优点 | 代价 |
|---|---|---|---|
| 单 profile | 请求形状稳定、追求简单 | 配置直观，行为稳定 | 混合负载下性能弹性差 |
| `multiple_profiles` | 线上负载波动大 | 更适合生产，多类请求更容易命中合适 profile | 构建更慢，可能带来轻微非确定性 |
| 连续 KV cache | 场景简单、实现心智负担低 | 结构直接 | 长上下文下浪费更明显 |
| paged KV cache | 长上下文、混合并发服务 | 更利于管理缓存与复用 | 配置更复杂 |
| FP16/BF16 | 精度稳定优先 | 风险低、验证成本低 | 显存占用更高 |
| INT8/FP8 | 显存和吞吐压力大 | 有机会显著降内存或提速 | 依赖量化流程、硬件、回归测试 |
| monolithic prefill | 短 prompt 为主 | 路径简单 | 超长 prompt 时压力集中 |
| chunked / paged context | 长 prompt 为主 | 更适合长上下文 | 不是所有 workload 都一定收益 |

这里有两个边界要明确。

第一，`multiple_profiles` 更适合生产，因为线上请求长度和并发通常不是单一分布；但如果你要求完全确定性输出，就要知道不同 profile 可能选到不同 kernel，结果可能有轻微差异。

第二，`FP8` 和更激进的量化配置，适合“有明确硬件前提且做过精度回归”的场景，不适合把文档抄下来直接上生产。

对零基础到初级工程师来说，可以先记住一个决策顺序：

1. 先看请求分布，决定短上下文还是长上下文优先。
2. 再定 `max_input_len / max_seq_len / max_batch_size` 的业务上限。
3. 然后用公式粗估 KV cache 和激活预算。
4. 再选择 `multiple_profiles`、paged context、量化方案。
5. 最后用 TTFT、TPOT、吞吐和显存数据闭环验证。

---

## 参考资料

1. [Useful Build-Time Flags](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html)
2. [Memory Usage of TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/reference/memory.html)
3. [trtllm-build CLI reference](https://nvidia.github.io/TensorRT-LLM/latest/commands/trtllm-build.html)
4. [Numerical Precision](https://nvidia.github.io/TensorRT-LLM/reference/precision.html)
5. [KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)
6. [gpt-attention.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/gpt-attention.md)
7. [TTFT / TPOT definitions](https://nvidia.github.io/TensorRT-LLM/deployment-guide/quick-start-recipe-for-gpt-oss-on-trtllm.html)
