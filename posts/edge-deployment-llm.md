## 核心结论

边缘部署的本质，是把原本依赖云端 GPU 的大模型，压缩到手机、工控机、Jetson、摄像头网关这类资源受限设备上运行。资源受限，指可用的 CPU、内存、存储、带宽和电池都明显少于数据中心。

结论先给出：

1. 真正决定能否落地的第一约束不是“模型参数够不够先进”，而是“模型权重、KV Cache、运行时缓存能不能装进设备内存”。
2. 真正有效的第一手段不是盲目换框架，而是先做量化、压缩、蒸馏，再配合缓存、异步推理和硬件加速。
3. 单机边缘部署通常不能同时满足“大模型、高精度、低延迟、低功耗”四个目标，工程上必须做取舍。
4. 对多数团队，最稳妥的方案不是“全本地”或“全上云”，而是边缘与云协同：简单请求本地处理，复杂请求异步回传云端。

可以先用一个估算公式判断项目是否有可行性：

$$
\text{模型存储大小} \approx \frac{\text{参数量} \times \text{位宽}}{8}
$$

如果是 7B 模型，FP16 权重大约需要：

$$
7 \times 10^9 \times 16 / 8 \approx 14 \text{ GB}
$$

这还没算 KV Cache、运行时缓冲区和操作系统占用。对 8GB 内存设备，这已经基本不可行。

---

## 问题定义与边界

边缘部署，不是“把云上的模型原样搬到本地”，而是在固定资源预算下，让模型达到“可接受效果 + 可接受延迟 + 可接受成本”的系统设计问题。

这里先明确边界：

| 维度 | 云端常见情况 | 边缘常见情况 | 工程含义 |
|---|---|---|---|
| 算力 | 高端 GPU | CPU / 小型 GPU / NPU | 吞吐低，峰值算力有限 |
| 内存 | 数十到数百 GB | 4GB-16GB 常见 | 模型大小和缓存极敏感 |
| 存储 | 宽松 | eMMC/SSD 有限 | 模型文件不能无限增大 |
| 网络 | 机房高速网络 | 4G/5G/Wi-Fi 可能抖动 | 不能依赖持续在线 |
| 功耗 | 供电稳定 | 电池或低功耗约束 | 长时推理会触发降频 |

术语先解释一次：

- 量化：把模型参数从高精度数值压缩成低精度数值，本质是“用更少比特表示同一组权重”。
- 蒸馏：用大模型教小模型，本质是“把复杂能力迁移到更小网络”。
- KV Cache：Transformer 推理时缓存历史 token 的键值对，本质是“用更多内存换更快续写速度”。
- 批处理：一次处理多条请求，本质是“让硬件少空转，提高整体吞吐”。

玩具例子可以先看一个极小设备场景。

假设你要在 4GB RAM 的旧手机上做离线问答：

- 1B 参数模型，INT4 量化后，理论权重约 $1 \times 10^9 \times 4 / 8 = 0.5$ GB。
- 再加运行时、缓存、系统占用，整体也许还能控制在 1.5GB-2.5GB。
- 如果你换成 7B FP16，权重就约 14GB，根本加载不进去。

这说明边缘部署首先是容量问题，其次才是速度问题。

---

## 核心机制与推导

边缘部署的优化链路，可以按“先缩小，再加速，最后做系统协同”理解。

### 1. 量化先解决“装不下”

量化直接降低参数位宽。常见效果如下：

| 精度 | 每参数字节数 | 相对 FP32 大小 | 典型用途 |
|---|---:|---:|---|
| FP32 | 4 | 1x | 训练、基线评测 |
| FP16 | 2 | 0.5x | GPU 推理常用 |
| INT8 | 1 | 0.25x | 边缘推理常用 |
| INT4 | 0.5 | 0.125x | 强压缩边缘场景 |

如果只看权重体积，FP32 到 INT4 理论上能缩小 8 倍。问题是，位宽降低后会带来精度损失，所以不能无限压。

更进一步，低位部署不只是“存储更小”，还意味着计算更便宜。矩阵乘法是大模型推理的主开销，低位权重让内存带宽压力下降，缓存命中率更高，硬件可以更快地喂饱计算单元。

### 2. 查表和混合精度解决“算得慢”

一些低位推理方案的核心思想，是把部分乘法改写成查表或混合精度矩阵乘法。白话说，就是把原本昂贵的数学运算，尽量改成更便宜的“提前准备好答案，然后直接取值”。

其逻辑可以抽象成：

1. 把权重压成 INT4 或更低。
2. 运行时按块解码或重解释数据布局。
3. 用查表或硬件特化指令完成近似乘加。
4. 在精度和速度之间做可控折中。

这类思路为什么有效？因为边缘设备常见瓶颈不是纯算术，而是“带宽不够 + 缓存太小 + 功耗限制”。只要减少内存搬运和乘法次数，整体吞吐就会上升。

### 3. KV Cache 解决“续写慢”，但会吃内存

生成式模型每输出一个 token，都会依赖前文历史。KV Cache 把历史中间结果保留下来，避免重复计算。其收益是速度更稳定，代价是内存增加。

简化理解：

$$
\text{总内存} \approx \text{模型权重} + \text{KV Cache} + \text{运行时缓冲}
$$

这也是很多新手容易忽略的点：模型量化后能装进去，不代表长上下文还能跑得动。上下文越长，KV Cache 越大。

### 4. 真实工程例子：Jetson 上的本地推理

真实工程里，Jetson、工控盒子、车载计算单元经常要承担视觉、语音、控制和大模型推理多个任务。此时问题不是“模型能不能启动”，而是“系统是否持续稳定”。

例如在机器人巡检场景里：

- 本地设备负责语音转文本、短指令理解、常见问答。
- 高风险或复杂请求，如长文档分析、多轮规划，异步发往云端。
- 本地保留缓存和降级策略，网络断开时仍能完成基础任务。

这类系统的关键不是单一模型 benchmark，而是端到端服务质量：首 token 延迟、平均 token/s、功耗曲线、温度触发降频后的稳定性。

---

## 代码实现

下面先给一个可运行的 Python 玩具代码，用来估算模型是否能放进目标设备内存。它不依赖第三方库，适合新手先建立容量意识。

```python
def estimate_model_gb(params_billion: float, bits: int) -> float:
    # 参数量以 billion 为单位，返回权重理论大小（GB）
    params = params_billion * 1_000_000_000
    return params * bits / 8 / (1024 ** 3)

def can_deploy(device_ram_gb: float, model_gb: float, runtime_overhead_gb: float, kv_cache_gb: float) -> bool:
    total = model_gb + runtime_overhead_gb + kv_cache_gb
    return total <= device_ram_gb

llama7b_fp16 = estimate_model_gb(7, 16)
llama7b_int4 = estimate_model_gb(7, 4)

assert llama7b_fp16 > llama7b_int4
assert round(llama7b_int4, 1) < 4.0

# 假设 8GB 设备，运行时开销 1.5GB，KV Cache 1.8GB
assert can_deploy(8.0, llama7b_int4, 1.5, 1.8) is True
assert can_deploy(8.0, llama7b_fp16, 1.5, 1.8) is False

print("7B FP16 权重大约 %.2f GB" % llama7b_fp16)
print("7B INT4 权重大约 %.2f GB" % llama7b_int4)
```

这个玩具例子说明一件事：量化不是“锦上添花”，而是边缘部署能否启动的前置条件。

再看一个更接近真实工程的 ONNX Runtime 配置示例。图优化，指在模型执行前先重写计算图，让推理路径更短、更适合目标硬件。

```python
import onnxruntime as ort

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = "optimized-model.onnx"

session = ort.InferenceSession(
    "model.onnx",
    sess_options=so,
    providers=["CPUExecutionProvider"]
)

print(session.get_providers())
```

这个配置的意义有两个：

1. 冷启动时把可优化的图提前优化掉。
2. 把优化结果落盘，下次直接加载，减少重复开销。

如果用 MLC-LLM 或类似框架，流程通常是：

```bash
mlc_llm gen_config HF://your-model
mlc_llm convert_weight ./dist/config.json
mlc_llm compile ./dist/config.json
mlc_llm chat ./dist/
```

如果用 `llama.cpp`，常见路径则是 GGUF 权重 + CPU/Metal/Vulkan 后端。它的优势是轻量、易落地，特别适合从实验快速走到设备验证。

---

## 工程权衡与常见坑

边缘部署不是单点优化，而是连续取舍。下面列工程上最常见的问题。

| 常见坑 | 现象 | 根因 | 解决思路 |
|---|---|---|---|
| 只看参数量，不看总内存 | 能加载，跑几轮就崩 | 忽略 KV Cache 和运行时缓冲 | 先做总内存预算 |
| 量化过猛 | 回复变差、事实错误增加 | 精度损失过大 | 做校准，必要时退到 INT8 或混合量化 |
| 冷启动很慢 | 首次请求卡顿明显 | 图优化和权重加载都在实时路径 | 离线优化、预热 |
| 吞吐低且抖动大 | token/s 不稳定 | CPU 降频、内存带宽不足 | 控功耗、减少并发、缩短上下文 |
| 盲目全本地 | 复杂任务质量差 | 设备算力不够支撑推理深度 | 改成边云协同 |
| 盲目全上云 | 延迟高、离线不可用 | 依赖网络 | 本地保留基础能力和缓存 |

这里给一个真实工程判断标准：

- 如果设备面向安全关键场景，如机器人、车载、工业控制，本地必须保留最低可用能力。
- 如果业务面向消费级助手，优先关注首 token 延迟、续写速度和耗电，而不是纯 benchmark 峰值。
- 如果请求分布明显长尾，应该把 80% 简单请求本地化，把 20% 复杂请求交给云。

很多团队失败，不是因为没有用最先进框架，而是没有先定义 SLA。SLA 可以理解成服务承诺，例如“首 token 小于 800ms、平均功耗小于 10W、断网时仍能执行基础命令”。

---

## 替代方案与适用边界

常见边缘部署方案不是互斥关系，而是针对不同硬件和目标做选择。

| 方案 | 优势 | 短板 | 适用场景 |
|---|---|---|---|
| `llama.cpp` | 轻量，GGUF 生态成熟，CPU 可跑 | 极致性能依赖具体平台调优 | 本地实验、CPU 设备、快速验证 |
| MLC-LLM | 多后端统一部署，覆盖移动端和 WebGPU | 编译链路相对更复杂 | 跨平台产品化 |
| ONNX Runtime | 图优化成熟，执行提供者丰富 | 对大模型特化不如专用框架 | 企业已有 ONNX 生态 |
| TensorRT Edge-LLM | Jetson/Drive 侧性能强 | 硬件绑定明显 | 机器人、自动驾驶、NVIDIA 生态 |

什么时候不适合边缘部署？

1. 模型必须保持高精度且不能蒸馏、不能量化。
2. 请求上下文极长，KV Cache 已经超过设备预算。
3. 设备供电和散热极差，持续推理会明显降频。
4. 业务要求复杂推理和高吞吐并存，但本地硬件没有加速器。

这时更合理的方案是混合部署。混合部署可以理解成“把能本地做的尽量本地做，把必须重算力处理的部分交给云”。

一个简单分工模型是：

- 边缘侧：ASR、短文本分类、意图识别、常见问答、缓存命中响应。
- 云端：长上下文推理、复杂规划、多模态融合、批量分析。
- 调度层：按网络状态、功耗、请求复杂度动态切换。

这比追求“单机跑一切”更符合工程现实。

---

## 参考资料

- Microsoft Research, Advances to low-bit quantization enable LLMs on edge devices: https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices/
- MLC LLM Documentation, Introduction: https://llm.mlc.ai/docs/get_started/introduction.html
- ONNX Runtime Documentation, Graph Optimizations: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- ScienceDirect, Empowering large language models to edge intelligence: https://www.sciencedirect.com/science/article/pii/S1574013725000310
- NVIDIA Developer Blog, Accelerating LLM and VLM inference for automotive and robotics with NVIDIA TensorRT Edge-LLM: https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/
