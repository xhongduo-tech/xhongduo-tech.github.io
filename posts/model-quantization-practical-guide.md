## 核心结论

模型量化的本质，是把原本用浮点数表示的权重和激活，改成更低比特的整数或低精度格式表示。白话说，就是把“占空间的大数字”压缩成“更省内存的小数字”，再在计算时按规则还原。

压缩收益首先体现在显存和内存带宽上。一个 7B 参数模型，如果按 FP16 存储，每个参数约 2 字节，总权重大约是：

$$
7 \times 10^9 \times 2 \approx 14\text{ GB}
$$

如果改成 INT8，每个参数约 1 字节，权重约 7 GB；改成 INT4，每个参数约 0.5 字节，权重约 3.5 GB。它不能把“24 GB 的卡变成 8 GB 的卡”，但能把原本只能在大显存设备上跑的模型，压缩到 8 GB 级别设备也有机会加载，尤其是在配合 KV Cache 优化、分层卸载、CPU/GPU 混合推理时更明显。

常见方案不是“谁最先进就用谁”，而是按目标选：
| 方法 | 典型精度 | 是否需要再训练 | 优点 | 代价 |
|---|---:|---|---|---|
| PTQ | INT8/INT4 | 否 | 最容易落地 | 精度可能下降 |
| QAT | INT8/INT4 | 是 | 精度通常最好 | 训练成本高 |
| GPTQ | INT4 为主 | 否，但需校准集 | 大模型权重量化效果好 | 工具链更复杂 |
| AWQ | INT4 为主 | 否，但需校准集 | 保护关键权重，精度更稳 | 部署格式受框架限制 |

一个实用判断是：先做 PTQ/INT8 看收益，再尝试 GPTQ 或 AWQ 做 INT4；只有当你能控制训练流程，且模型要长期规模化部署时，才值得上 QAT。

---

## 问题定义与边界

量化是一个映射过程：把浮点值按缩放因子映射到有限整数区间，再在计算时反量化。最常见的对称量化公式是：

$$
x_q = \operatorname{clip}(\operatorname{round}(x/s), q_{\min}, q_{\max})
$$

$$
\hat{x} = x_q \cdot s
$$

其中，$s$ 是 scale，白话说就是“把原始数值压缩到整数格子里所需的比例尺”；$x_q$ 是量化后的整数；$\hat{x}$ 是反量化后的近似值。

对新手最直观的理解是：INT8 量化就是先把权重除以一个 `scale`，把结果四舍五入到 `[-128,127]`，推理时再乘回去。原值不一定完全还原，所以量化一定会带来误差，只是误差能否接受的问题。

目标边界要说清楚。量化主要解决的是“部署约束”，不是“让模型更聪明”。

| 边界项 | 量化能解决 | 量化不能直接解决 |
|---|---|---|
| 显存不足 | 是 | - |
| 内存带宽瓶颈 | 是 | - |
| 推理吞吐不足 | 常常有帮助 | 不保证一定提升 |
| 训练数据不足 | 否 | 需要补数据或重训 |
| 模型推理逻辑错误 | 否 | 需要改模型或提示词 |

硬件边界也很重要：
| 硬件环境 | 更常见选择 |
|---|---|
| 消费级 GPU，8-16 GB 显存 | INT4 / INT8 |
| 数据中心 GPU，支持 Tensor Core | FP8 / INT8 / W4A16 |
| CPU 本地推理 | GGUF + Q4 / Q8 |
| 训练中微调 | QLoRA、4bit load、BF16 计算 |

---

## 核心机制与推导

量化最核心的矛盾是：表示范围越小，误差越大；表示位宽越高，压缩率越低。压缩率可以简单写成：

$$
\text{Compression Ratio} = \frac{\text{原始位宽}}{\text{量化后位宽}}
$$

例如 FP16 到 INT8，压缩率约为 $16/8 = 2$；FP16 到 INT4，压缩率约为 $16/4 = 4$。

实际工程里通常不是整层共用一个 scale，而是每通道量化。通道可以理解为“矩阵里每一列或每一组参数各自单独量化”，这样能减少不同分布混在一起造成的失真。对称每通道 INT8 常见 scale 为：

$$
s_{ch}=\frac{\max(|x_{\min}^{ch}|, |x_{\max}^{ch}|)}{127}
$$

### 玩具例子

假设某一通道的 4 个权重是：

$$
[-1.20,\ -0.35,\ 0.42,\ 1.00]
$$

那么该通道最大绝对值为 $1.20$，于是：

$$
s = \frac{1.20}{127} \approx 0.00945
$$

量化后：

- $-1.20 / 0.00945 \approx -127 \rightarrow -127$
- $-0.35 / 0.00945 \approx -37.0 \rightarrow -37$
- $0.42 / 0.00945 \approx 44.4 \rightarrow 44$
- $1.00 / 0.00945 \approx 105.8 \rightarrow 106$

反量化后：

- $-127 \times 0.00945 \approx -1.20$
- $-37 \times 0.00945 \approx -0.35$
- $44 \times 0.00945 \approx 0.416$
- $106 \times 0.00945 \approx 1.002$

可以看到，量化不是“整数化后直接算”，而是“整数存储，近似还原”。误差来自 rounding 和 clip。

为什么 GPTQ、AWQ 在 INT4 下通常比朴素 PTQ 更稳？因为它们不只做“统一压缩”，而是在层级上显式考虑重建误差。可以把目标粗略理解为：

$$
\min_{Q(W)} \|WX - Q(W)X\|^2
$$

这里 $W$ 是原始权重，$Q(W)$ 是量化后权重，$X$ 是校准输入。白话说，就是“希望量化后的层输出尽量接近原始层输出”。

GPTQ 的重点是逐层、按误差最小化去找更好的量化表示；AWQ 的重点是保护少量最重要权重，常见说法是保护 top-1% 重要权重不被过度压缩。它特别适合有明显 outlier 的层。outlier 指“少数数值特别大、特别重要的权重”，白话说就是“长得和其他参数不一样的大值”。

| 机制 | 关注点 | 适合场景 |
|---|---|---|
| PTQ | 快速压缩 | 先验证能否部署 |
| QAT | 训练中适配量化噪声 | 可重训模型 |
| GPTQ | 最小化层输出重建误差 | LLM INT4 部署 |
| AWQ | 保护关键权重 | 有明显 outlier 的模型 |

---

## 代码实现

下面先给一个最小可运行的 Python 量化玩具实现，再给一个真实工程例子。

```python
import math

def quantize_symmetric_int8(xs):
    max_abs = max(abs(x) for x in xs)
    scale = max_abs / 127 if max_abs != 0 else 1.0
    q = []
    for x in xs:
        v = round(x / scale)
        v = max(-127, min(127, v))
        q.append(int(v))
    deq = [v * scale for v in q]
    return scale, q, deq

xs = [-1.20, -0.35, 0.42, 1.00]
scale, q, deq = quantize_symmetric_int8(xs)

assert q == [-127, -37, 44, 106]
assert abs(deq[0] - (-1.20)) < 0.02
assert abs(deq[2] - 0.42) < 0.02
print(scale, q, deq)
```

真实工程里，开发者通常不会自己手写量化核，而是调用现成工具。下面是 `bitsandbytes` 的典型加载方式：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

prompt = "Explain quantization in one sentence."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

`NF4` 是一种为近似正态分布权重设计的 4bit 格式，白话说就是“不是均匀切 16 档，而是按更适合权重分布的方式编码”。它在 QLoRA 场景很常见。

如果要转向本地 CPU 或边缘设备部署，常见流程是把模型转换成 GGUF，再用 `llama.cpp` 推理：

| 步骤 | 工具 | 说明 |
|---|---|---|
| 加载模型 | Transformers + bitsandbytes | 先验证 4bit/8bit 是否可用 |
| 导出/转换 | 转换脚本或框架自带工具 | 生成 GGUF、GPTQ 或 AWQ 权重 |
| 推理 | llama.cpp / TensorRT-LLM | 按目标硬件选择运行时 |

一个典型的真实工程例子是：团队把 7B 聊天模型从 FP16 改成 AWQ INT4，导出 GGUF 后在 16 GB 内存的笔记本 CPU 上本地运行，用于离线问答；如果目标是机房 GPU 服务，则更常见做法是导出 GPTQ/AWQ 权重给 TensorRT-LLM，走 W4A16 或 INT8 路线。

---

## 工程权衡与常见坑

量化的第一坑，是只看“模型能不能加载”，不看端到端指标。真正该测的是 TTFT、吞吐、延迟和能耗。TTFT 指 Time To First Token，白话说就是“用户发出请求后，模型吐出第一个 token 需要多久”。

一个常见误区是：位宽越低一定越快。并不总成立。比如 4B 量级小模型在 NF4 上，有时因为反量化开销、访存模式和硬件核不匹配，整体能耗反而上升，TTFT 也未必更好。原因不是量化失效，而是“节省的显存”没有覆盖“新增的数据格式转换成本”。

| 策略 | 带宽压力 | 能耗 | 精度风险 | 常见坑 |
|---|---|---|---|---|
| INT8 PTQ | 中 | 中 | 低到中 | 激活校准不充分 |
| INT4 GPTQ | 低 | 低到中 | 中 | 某些层误差集中 |
| INT4 AWQ | 低 | 中 | 低到中 | 工具链兼容性 |
| NF4 | 低 | 不一定最低 | 中 | 小模型反量化开销明显 |

第二坑是 outlier。某些层的权重分布明显不对称，少量大值决定了输出稳定性。如果把这些值和普通权重一视同仁压成 4bit，模型可能出现“整体困惑度没太坏，但回答风格明显飘”的问题。处理方式通常有三种：

| 处理方式 | 适用情况 |
|---|---|
| 保留部分层为 FP16/BF16 | 极少数关键层失真严重 |
| 用 AWQ 保护关键权重 | 明显存在重要大值 |
| 改回 INT8 而非 INT4 | 业务更看重稳定性 |

第三坑出现在训练侧。QLoRA 虽然常用 4bit 权重加载，但并不意味着所有模块都该量化。`lm_head` 往往建议保留更高精度，否则输出词分布容易劣化，表现为重复、漏词或回答边界异常。

---

## 替代方案与适用边界

如果只是为了让模型在更小显存上先跑起来，PTQ 是最直接的起点。它不要求重训，工具也成熟。缺点是 INT4 下精度下滑可能明显，尤其在数学、代码、长上下文任务里。

如果模型是你的，且训练链路可控，QAT 往往是精度最稳的路线。因为模型在训练中已经“见过量化噪声”，会主动适应这种误差。但它需要训练预算，也要求你能改训练图。

GPTQ 和 AWQ 更适合“模型已经训练完，但我要尽量压到 INT4 且少掉点精度”的场景。两者对比可以这样看：

| 方法 | 是否需训练 | 是否需校准集 | 主要优势 | 适用边界 |
|---|---|---|---|---|
| PTQ | 否 | 通常需要 | 最简单 | 先验证部署 |
| QAT | 是 | 训练中隐含完成 | 精度最好 | 可重训 |
| GPTQ | 否 | 是 | 层级误差控制好 | 大模型离线量化 |
| AWQ | 否 | 是 | 对关键权重更友好 | outlier 明显模型 |

工具也有边界：

- `bitsandbytes` 适合快速验证、训练时 4bit/8bit 加载、QLoRA。
- `llama.cpp` 适合本地推理、CPU 部署、GGUF 生态。
- `TensorRT-LLM` 适合 NVIDIA GPU 上追求低延迟、高吞吐的正式服务。
- 如果业务追求极限吞吐，最终常常不是停留在“加载 4bit”，而是要进入特定运行时的内核优化路径，例如 W4A16、FP8、融合算子和分页 KV Cache。

选择 AWQ 的一个明确边界是：你观察到模型在 INT4 PTQ 后，少数任务退化明显，且排查发现问题集中在少量关键层或大权重上。这时 AWQ 的“保护关键权重”通常比简单换更小 scale 更有效。

---

## 参考资料

1. Reintech, *LLM Quantization Explained: INT8, INT4, GPTQ & Production Deployment*  
   核心贡献：给出从 FP16 到 INT8/INT4 的存储变化、部署流程和生产落地视角。

2. NVIDIA TensorRT Documentation, *Working with Quantized Types*  
   核心贡献：提供量化/反量化公式、量化类型和 TensorRT 中的数据类型边界。

3. Hugging Face Transformers Documentation, *Quantization*  
   核心贡献：解释 `bitsandbytes` 的 `load_in_8bit`、`load_in_4bit`、NF4、双重量化等实践接口。

4. Hugging Face Optimum Documentation, *Quantization Concept Guide*  
   核心贡献：总结 PTQ、QAT、GPTQ、AWQ 的适用场景，以及 QLoRA 中的工程注意事项。

5. NVIDIA TensorRT-LLM / NeMo Quantization 相关文档  
   核心贡献：说明服务端 W4A16、INT8、FP8 等量化路线与运行时优化的结合方式。
