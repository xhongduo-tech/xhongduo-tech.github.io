## 核心结论

边缘计算中的模型部署，指把已经训练好的模型放到靠近数据源的设备上执行推理。白话说，就是让手机、摄像头、工业网关、车载设备直接“在本地想”，而不是每次都把输入发到云服务器再等结果回来。

它的核心价值有三个。第一是低延迟，也就是更短的响应时间。对语音唤醒、相机识别、机器人控制这类实时任务，几十到一百多毫秒的差异就会直接影响可用性。第二是隐私，敏感文本、语音、图像可以留在设备内，不必把原始数据上传。第三是节省带宽，尤其在视频流、连续传感器数据场景中，上传全部原始数据的成本很高。

但“把模型搬到端上”不是简单复制文件。边缘设备通常同时受三类约束：内存小、算力有限、硬件类型杂。一个在云端 GPU 跑得很顺的模型，放到手机 NPU、树莓派 CPU、Jetson GPU 上，往往会因为显存、功耗、驱动或算子支持不一致而直接失败。

因此，边缘部署的真正方法论不是“端替代云”，而是“压缩 + 编译 + 适配 + 协同”。压缩包括量化、剪枝、蒸馏；编译是把模型转换成目标硬件能高效执行的格式；适配是选择合适的运行时框架，如 `llama.cpp`、`MLC-LLM`、`ONNX Runtime`；协同则是在边和云之间明确谁做前处理、谁做主推理、谁负责长上下文和模型更新。

下表先给出最重要的决策视角：

| 部署方式 | 延迟 | 带宽占用 | 隐私风险 | 运维成本 | 适合场景 |
|---|---:|---:|---:|---:|---|
| 纯云 | 高，受网络影响 | 高 | 较高 | 中到高 | 大模型、长上下文、集中更新 |
| 纯边缘 | 低，最稳定 | 低 | 低 | 前期高、长期可控 | 实时交互、隐私敏感、本地连续运行 |
| 边云协同 | 中到低 | 中 | 中到低 | 最高 | 设备异构、任务波动大、既要实时又要吞吐 |

一个典型结论是：如果任务要求“立刻响应且原始数据不能离开设备”，优先考虑边缘部署；如果任务要求“超大模型、超长上下文、频繁更新”，云端更现实；如果两边都要，就做边云分工，而不是强行二选一。

---

## 问题定义与边界

这篇文章讨论的是推理部署，不讨论模型训练。推理，白话说，就是模型已经学完了，现在用它来回答问题、识别图片、生成文本。训练需要大量数据和算力，通常在云端完成；边缘设备主要承担推理。

边缘部署的边界要先说清楚。不是所有模型都适合端侧运行，也不是所有端侧场景都值得做本地推理。判断标准通常有四个：

| 维度 | 需要问的问题 | 对部署决策的影响 |
|---|---|---|
| 延迟 | 允许等待多久？是 50 ms、500 ms，还是 5 s？ | 延迟越严，越倾向边缘 |
| 隐私 | 原始数据能不能上传？ | 不能上传时，边缘优先 |
| 资源 | 设备有多少 RAM、存储、功耗预算？ | 资源越小，模型越要压缩 |
| 网络 | 网络是否稳定、是否持续在线？ | 网络不稳时，纯云风险大 |

边云协同常用一个总时延公式来描述：

$$
T_{\text{total}}(s,\{b\}) = T_{\text{edge}}(s,\{b\}) + T_{\text{trans}}(s) + T_{\text{cloud}}(s)
$$

其中，$s$ 表示切分点，也就是模型在第几层开始从边端切到云端；$\{b\}$ 表示各层使用的位宽集合。位宽，白话说，就是每个数用多少 bit 表示，例如 16 bit、8 bit、4 bit。位宽越小，通常越省内存、越快，但也可能损失精度。

这个公式的意义很直接：边云协同不是只看云快不快，也不是只看端快不快，而是要同时计算三段时间之和。

可行方案还必须满足资源约束，例如：

$$
M_{\text{edge}}(s,\{b\}) \le M_{\max}, \quad
B_{\text{trans}} \le B_{\max}
$$

这里的 $M_{\text{edge}}$ 是边端推理时需要的内存，$B_{\text{trans}}$ 是传输中间特征所需带宽。只要超过设备内存上限或链路带宽上限，方案就算理论上能跑，工程上也不可用。

玩具例子可以先看一个最小场景。假设你有一个 12 层的小型 Transformer：

- 如果 12 层全放云端，设备本地只负责采集输入，那么本地几乎不占算力，但网络抖动时用户会明显卡顿。
- 如果 12 层全放手机上，响应很快，但手机内存不够，模型可能根本加载不起来。
- 如果前 4 层在本地，后 8 层在云端，本地先做一轮特征提取，只把中间表示上传，那么可以减少原始数据上传量，并缩短一部分等待时间。

真实工程例子更典型。工业机器人抓取物体时，控制回路往往要求毫秒级反馈。一个可行策略是把前几层视觉特征提取和动作初判放在 Jetson 上执行，保证局部控制不被网络拖慢；更复杂的轨迹优化或跨设备调度再交给云端。这样做的本质是把“必须实时”的部分锁在边端，把“可以稍慢但更重”的部分放到云端。

---

## 核心机制与推导

边缘部署的核心机制不是某一个框架，而是三件事一起成立：模型表示更小、硬件执行更匹配、切分策略更合理。

第一件事是量化。量化，白话说，就是把原本用高精度浮点数表示的权重和激活，改成更低位宽的表示方式。比如把 FP16 或 FP32 改成 int8、int4。最直接的收益是参数体积下降，缓存占用下降，内存带宽压力下降。

如果一个模型有 $N$ 个参数，忽略元数据时，权重大致占用：

$$
\text{ModelSize} \approx N \times \frac{b}{8}\ \text{bytes}
$$

其中 $b$ 是每个参数的位宽。比如同一个 3B 参数模型：

| 位宽 | 单参数字节数 | 理论权重体积 |
|---|---:|---:|
| FP16 | 2 | 约 6 GB |
| int8 | 1 | 约 3 GB |
| int4 | 0.5 | 约 1.5 GB |

这还没算 KV Cache。KV Cache，白话说，就是大模型在生成时为了避免重复计算而保存的上下文中间结果。对长上下文任务，KV Cache 往往和模型权重一样关键，甚至更容易成为内存瓶颈。因此近年的优化不只盯着权重，也在压缩 KV Cache，例如 3 bit 缓存压缩。

第二件事是混合精度矩阵乘。矩阵乘是 Transformer 推理的核心热点。mpGEMM 可以理解为“不同输入、不同层、不同缓存用不同位宽做乘法和累加”，例如 `int8 x int2`、`FP16 x int4`。白话说，不是整个模型一刀切都变成 4 bit，而是根据每层敏感度精细分配。

一个简化理解是：

$$
Y = XW \approx \hat{X}_{b_x}\hat{W}_{b_w}
$$

其中 $\hat{X}_{b_x}$ 和 $\hat{W}_{b_w}$ 分别表示被量化到位宽 $b_x$ 和 $b_w$ 的激活与权重。位宽越低，乘法成本越低，但误差通常越大，所以要在精度与速度之间找平衡。

下面这张表可以帮助理解 trade-off：

| 配置 | 权重占用 | 激活/缓存压力 | 预期速度 | 精度风险 | 常见用途 |
|---|---|---|---|---|---|
| FP16 全量 | 高 | 高 | 基线 | 最低 | 云端基准、验证 |
| int8 权重 + FP16 激活 | 中 | 中 | 较快 | 低 | 通用边缘部署起点 |
| int4 权重 + FP16 激活 | 低 | 中 | 快 | 中 | 手机、轻量 GPU |
| int4 权重 + int8/低比特缓存 | 很低 | 低 | 很快 | 较高 | 极限内存场景 |

第三件事是蒸馏与压缩。知识蒸馏，白话说，就是让一个小模型模仿大模型的输出习惯。它解决的问题不是“让设备更快”，而是“在设备只能跑小模型时，尽量保留原有能力”。如果量化解决的是体积和带宽，蒸馏解决的是能力保留。

这里有一个容易被忽略的推导逻辑：边缘部署不是单一目标优化，而是多目标优化。你真正优化的是：

$$
\min \ \alpha T + \beta M + \gamma C - \delta A
$$

其中 $T$ 是延迟，$M$ 是内存占用，$C$ 是传输成本，$A$ 是准确率，$\alpha,\beta,\gamma,\delta$ 是业务权重。实时语音助手会把 $\alpha$ 和隐私相关权重设得更高；离线文档总结可能更重视 $A$。

玩具例子可以这样理解：一台树莓派有 8 GB RAM，你想跑一个本地问答模型。

- FP16 模型根本装不下，失败。
- int8 能装下，但首 token 很慢，用户体验一般。
- int4 模型能稳定运行，回答质量稍降，但整体可用。

这里不是“最准确的模型最好”，而是“在设备约束下，能稳定交付目标体验的模型最好”。

---

## 代码实现

工程上最实用的思路，是先把部署流程拆成 4 步：导出模型、压缩模型、选择执行提供器、验证延迟与精度。

`ONNX Runtime` 是一个跨平台推理运行时。白话说，它像一个“统一播放器”，把不同来源的模型放到 CPU、GPU、NPU 上执行。`Execution Provider` 就是它对接具体硬件后端的方式，比如 CPU、CUDA、TensorRT、CoreML、QNN。

下面先用一个可运行的 Python 玩具例子，演示“位宽变化如何影响模型体积与总时延估算”。这不是实际推理代码，但能帮助理解部署决策：

```python
def model_size_gb(params_billion: float, bits: int) -> float:
    params = params_billion * 1_000_000_000
    return params * bits / 8 / 1024 / 1024 / 1024

def total_latency_ms(edge_ms: float, trans_ms: float, cloud_ms: float) -> float:
    return edge_ms + trans_ms + cloud_ms

# 3B 模型不同位宽下的大致权重体积
fp16_size = model_size_gb(3, 16)
int8_size = model_size_gb(3, 8)
int4_size = model_size_gb(3, 4)

assert fp16_size > int8_size > int4_size
assert round(fp16_size, 1) >= 5.5  # 约 6 GB
assert round(int4_size, 1) <= 1.6   # 约 1.5 GB

# 一个简单的边云切分延迟估算
pure_cloud = total_latency_ms(edge_ms=5, trans_ms=120, cloud_ms=80)
hybrid = total_latency_ms(edge_ms=35, trans_ms=20, cloud_ms=30)
pure_edge = total_latency_ms(edge_ms=90, trans_ms=0, cloud_ms=0)

assert pure_cloud == 205
assert hybrid == 85
assert pure_edge == 90

# 这个玩具例子里，混合部署最优
assert hybrid < pure_cloud
assert hybrid < pure_edge
```

真正部署时，代码通常类似下面这样：

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)

input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
outputs = session.run(None, {"input_ids": input_ids})

assert outputs is not None
```

这几行分别对应四个动作：

| 代码动作 | 工程含义 |
|---|---|
| `InferenceSession("model.onnx")` | 加载已经导出的 ONNX 模型 |
| `providers=[...]` | 选择执行后端，如 CPU/GPU/NPU |
| `session.run(...)` | 发起一次推理 |
| `assert` | 用最小检查保证流程没有直接失败 |

如果目标是手机或边缘 SoC，实际过程一般是：

| 步骤 | 工具 | 目标 |
|---|---|---|
| PyTorch/原始模型导出 ONNX | 导出器 | 把训练框架格式转成通用推理格式 |
| 量化与图优化 | Olive / 量化工具 | 压缩体积、替换算子、减少拷贝 |
| 绑定执行后端 | ONNX Runtime EP | 让模型跑在 CPU/GPU/NPU 上 |
| 端上验证 | benchmark 脚本 | 看首 token、吞吐、内存、温度 |

如果是开源 LLM 端侧部署，`llama.cpp` 和 `MLC-LLM` 更常见。`llama.cpp` 的优势是轻、直接、社区成熟，适合 CPU 与 Apple 设备；`MLC-LLM` 的优势是统一编译到 Metal、Vulkan、CUDA、WebGPU，适合多后端统一交付。

例如 `llama.cpp` 的最小启动方式通常是：

```bash
./llama-cli -m model.gguf -c 4096
```

这背后的工程含义是：模型已经被转换成 GGUF 格式，量化配置已经写进模型文件，运行时只需要指定模型路径和上下文长度即可。

真实工程例子可以看“端上视觉编码 + 云端解码”。例如一套视觉语言系统中，摄像头所在的 Jetson 设备先本地跑视觉编码器，把图像压成中间特征；云端再接收这些特征并运行更重的语言解码器。这样做的收益是，边端负责低延迟采集和第一跳处理，云端负责大模型吞吐和集中更新。

---

## 工程权衡与常见坑

边缘部署最容易犯的错误，是把“模型能跑起来”当成“系统能上线”。能跑只是第一步，真正的工程指标至少还包括首 token 延迟、稳定吞吐、峰值内存、持续功耗、温度、断网行为、模型更新路径。

常见坑可以直接列出来：

| 常见坑 | 典型表现 | 根因 | 规避策略 |
|---|---|---|---|
| 模型过大 | 加载失败、频繁 OOM | 权重或 KV Cache 超设备内存 | 量化、减小上下文、换更小模型 |
| 执行后端不匹配 | 跑在 CPU 上异常慢 | 算子不支持或 EP 配置失败 | 提前验证 provider 和算子覆盖率 |
| 网络抖动下混合部署变慢 | 端云切分后比纯云更慢 | 传输中间特征过大、切分点错误 | 动态切分、压缩特征、预估带宽 |
| 蒸馏后效果失真 | 术语回答错误、风格漂移 | 蒸馏数据分布不对 | 用真实业务样本做持续对齐 |
| 热量和功耗失控 | 手机发热、频率下降 | 长时间高负载本地推理 | 降精度、限帧、分批推理 |
| 更新困难 | 端上模型版本碎片化 | 设备多、格式多、驱动多 | 建统一导出链路和灰度发布机制 |

有一个典型误区需要特别指出：切分点不一定越靠前越好。很多初学者会觉得“边端多算一点，就少传一点”。这只对一部分模型成立。如果中间层特征体积比原始输入还大，或者切分后引入额外格式转换，那么 `T_trans` 反而会上升，总时延变坏。

另一个常见坑是只测平均值，不测尾延迟。尾延迟，白话说，就是最慢那一小部分请求的耗时。实时系统里，用户往往是被最慢的那几次卡住，而不是被平均值卡住。边缘设备在温度升高后会降频，尾延迟会明显恶化，所以测试必须包含长时间运行。

真实工程中，边云协同常用于“端侧采集必须快，但完整推理太重”的系统。比如 NVIDIA 的一些边到云部署实践，会让边端设备先运行视觉编码和预处理，再把兼容中间表示送入云端 TensorRT-LLM 解码器。这类架构的重点不是某个单独模型快，而是整个链路在不同硬件上保持稳定和可维护。

---

## 替代方案与适用边界

不是所有场景都要把模型塞进终端。真正合理的选择，取决于业务的主约束是什么。

如果设备极弱，比如低端 MCU、超低功耗传感器，完整 LLM 部署并不现实，常见替代方案是“边缘只做触发和筛选，复杂理解全部交给云端”。例如摄像头先做运动检测，只在检测到异常时上传关键帧。

如果主要问题不是权重体积，而是长上下文导致的 KV Cache 爆炸，那么更合适的方向可能不是继续缩主模型，而是压缩缓存，例如低比特 KV Cache 技术。这类方案更适合有 GPU 的边缘服务器或近端节点，不一定适合最弱设备。

如果业务要求“始终可用，但复杂任务可以稍慢”，最实用的通常是分层策略：边缘跑一个 3B 级别的小模型做即时助手，复杂推理、长文总结、频繁热更新留给云端。网络正常时优先云增强；网络断开时边端继续执行基础功能。

最后给出一个替代方案对比表：

| 方案 | 延迟 | 模型更新频率 | 隐私风险 | 基础设施成本 | 适用边界 |
|---|---:|---:|---:|---:|---|
| Pure Edge | 最低 | 低到中 | 最低 | 端侧适配成本高 | 强实时、强隐私、弱网络 |
| Edge-Cloud Split | 低到中 | 中 | 中到低 | 最高 | 既要实时又要大模型能力 |
| Pure Cloud | 中到高 | 最高 | 较高 | 云资源成本高 | 长上下文、复杂推理、集中治理 |

可以把选择规则压缩成一句话：

- 能容忍网络且追求最大模型能力，选纯云。
- 不能容忍网络且数据敏感，选纯边缘。
- 两者都不能放弃，选边云协同，但前提是你有能力维护切分、调度和版本管理。

---

## 参考资料

- Microsoft Research, Advances to low-bit quantization enable LLMs on edge devices: https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices/
- Microsoft Community, Cross-Platform Edge AI Made Easy with ONNX Runtime: https://techcommunity.microsoft.com/blog/aiplatformblog/cross-platform-edge-ai-made-easy-with-onnx-runtime/4303521
- NVIDIA Technical Blog, Deploying Accelerated Llama 3.2 from the Edge to the Cloud: https://developer.nvidia.com/blog/deploying-accelerated-llama-3-2-from-the-edge-to-the-cloud/
- Emergent Mind, Edge-Cloud Collaborative Architecture: https://www.emergentmind.com/topics/edge-cloud-collaborative-architecture
- Emergent Mind, Model Partitioning for Edge-Cloud Collaboration: https://www.emergentmind.com/topics/model-partitioning-for-edge-cloud-collaboration
- Tom's Hardware, Google's TurboQuant compresses LLM KV caches to 3 bits: https://www.tomshardware.com/tech-industry/artificial-intelligence/googles-turboquant-compresses-llm-kv-caches-to-3-bits-with-no-accuracy-loss
- MLC-LLM Documentation, Compile Models: https://llm.mlc.ai/docs/compilation/compile_models.html
- Hugging Face Docs, llama.cpp engine overview: https://huggingface.co/docs/inference-endpoints/en/engines/llama_cpp
