## 核心结论

TGI、vLLM、TensorRT-LLM 不是同一层面的“谁替代谁”，而是三种不同优化目标下的主流模型服务框架。

- TGI，Text Generation Inference，白话讲就是“把大模型包装成可直接提供 API 的服务程序”，重点是服务化、运维友好、和 Hugging Face 生态贴合。
- vLLM，白话讲就是“用更省显存的方式管理生成过程中的中间状态”，重点是 PagedAttention 和连续批处理，适合高并发、长上下文、显存紧张的在线推理。
- TensorRT-LLM，白话讲就是“把模型编译成更贴近 NVIDIA GPU 执行方式的高性能引擎”，重点是量化、算子融合、多卡并行和极致吞吐/低延迟。

这三者分别代表三种策略：

| 框架 | 核心关注点 | 典型部署模式 | 显存策略 | 硬件依赖 |
| --- | --- | --- | --- | --- |
| TGI | 云原生服务化 | Docker、Kubernetes、云容器 | 常规 KV Cache + 批处理调优 | 通用 NVIDIA GPU |
| vLLM | 显存分页调度 | 单机服务、K8S、OpenAI 兼容 API | PagedAttention + 连续批处理 | 通用 GPU，NVIDIA 最常见 |
| TensorRT-LLM | 硬件级推理加速 | Triton、`trtllm-serve`、K8S | 编译期优化 + KV 管理 + 量化 | 强依赖 NVIDIA，A100/H100/Hopper 更优 |

如果用一句话概括选型：

- 先要“快速上线、接口稳定、易运维”，优先 TGI。
- 先要“长上下文、高并发、单位显存吞吐高”，优先 vLLM。
- 先要“把 NVIDIA GPU 压到极限”，优先 TensorRT-LLM。

一个新手也能理解的组合例子是：先用 GKE + TGI 把 `Meta-Llama-3.1-8B-Instruct` 暴露成 REST 接口；遇到长上下文容易 OOM 时，把这类请求切到 vLLM，用分页 KV Cache 降低显存峰值；最后把最密集、最稳定的生产流量模型编译成 TensorRT-LLM FP8 引擎，追求更高吞吐和更低单位 token 成本。

---

## 问题定义与边界

模型服务框架解决的不是“模型会不会回答”，而是“同一块硬件在真实请求流量下，能不能稳定、便宜、快速地回答”。

这里有三组最常见的张力。

第一组是高吞吐和低延迟的张力。

- 高吞吐，白话讲就是“单位时间内处理更多 token 或更多请求”。
- 低延迟，白话讲就是“单个用户更快看到首 token 和完整结果”。

批处理越激进，GPU 利用率通常越高，但单个请求可能要等待更久。连续批处理能缓解这个问题，但不会消除这个矛盾。

第二组是长上下文和显存容量的张力。

生成任务不是只存模型参数，还要存 KV Cache。KV Cache，白话讲就是“模型为后续 token 生成保留的历史注意力状态”。上下文越长、并发越高，KV Cache 占用越大。很多服务不是死于算力不够，而是死于显存先满。

第三组是硬件通用性和极致性能的张力。

- TGI 和 vLLM 更偏“在常见 GPU 上尽量跑得好”。
- TensorRT-LLM 更偏“在 NVIDIA GPU 上专门优化到极致”。

需求边界可以先按下面的表判断：

| 目标 | 更关注什么 | 更适合的框架 | 主要限制 |
| --- | --- | --- | --- |
| 50+ 并发对话 API | 服务稳定、扩缩容、监控 | TGI | 显存优化能力不如 vLLM 激进 |
| 长上下文问答、RAG | KV Cache 利用率、吞吐 | vLLM | 调度策略复杂，短文本场景未必最优 |
| 固定模型的高密度生产流量 | 极致性能、单位成本 | TensorRT-LLM | 编译链路复杂，强依赖 NVIDIA |
| 小团队快速 PoC | 易部署、少踩坑 | TGI | 峰值性能通常不是最高 |
| 大模型单机高负载 | 显存分页、连续批处理 | vLLM | 工程调参要求更高 |
| A100/H100 追求极限 TPS | 量化、融合、多卡 | TensorRT-LLM | 移植性和灵活性较差 |

一个直观例子：

- 你想在 GPU 集群上服务 50+ 并发对话，且要 K8S 弹性伸缩，TGI 往往更顺手。
- 你要服务超长上下文，或者在相同显存下尽量塞进更多请求，vLLM 更合适。
- 你已经确定模型版本、硬件是 A100/H100，目标是把每秒 token 数推高，TensorRT-LLM 更有优势。

所以本文边界很明确：讨论的是在线推理服务，不讨论训练框架，不讨论微调流程，也不讨论 CPU-only 的轻量本地推理。

---

## 核心机制与推导

先看为什么 vLLM 常被认为“更省显存”。

自回归生成时，每个请求都会增长自己的 KV Cache。传统实现里，KV Cache 常按连续大块显存申请。问题在于请求长度不同、结束时间不同，显存会碎片化。碎片化，白话讲就是“总剩余空间看起来够，但没有足够大的连续空间可用”。

vLLM 的核心思路是 PagedAttention。它把一个序列的 KV Cache 切成固定大小的页，再通过逻辑页到物理页的映射来组织：

$$
logical\_pages = \left\lceil \frac{sequence\_length}{page\_stride} \right\rceil
$$

这里：

- `sequence_length` 是当前序列长度。
- `page_stride` 是每页容纳的 token 数。
- `logical_pages` 是这个序列需要多少逻辑页。

玩具例子如下。假设一个请求长度为 4096 token，每页大小为 1024 token，那么：

$$
logical\_pages = \left\lceil \frac{4096}{1024} \right\rceil = 4
$$

也就是这个请求会被拆成 4 个逻辑页。每个逻辑页不要求在显存里连续，只要都能映射到某些 `physical_frame` 即可。于是即便物理显存被不同请求穿插占用，系统仍然能把这些页拼起来提供注意力计算。

可以把它理解成下面的过程：

| 步骤 | 动作 | 结果 |
| --- | --- | --- |
| 1 | 把序列按固定页长切分 | 得到逻辑页 0,1,2,3 |
| 2 | 为每个逻辑页分配物理 frame | frame 可以离散分布 |
| 3 | 注意力计算时按页查表 | 不再依赖大块连续显存 |
| 4 | 请求结束后回收页 | 减少碎片积累 |

这带来两个直接好处。

- 更少的显存碎片，能容纳更多并发请求。
- 连续批处理更容易做，因为新请求可以动态插入已有批次，而不必等待一个大批完全结束。

连续批处理，白话讲就是“不是整批请求一起进一起出，而是随时把新请求插入 GPU 正在跑的批里”。它和传统静态 batch 的差别在于，系统调度对象变成了 token 级工作流，而不是请求级工作流。

从近似角度看，吞吐可以理解为：

$$
throughput \approx \frac{\sum_i token\_count_i}{total\_time}
$$

如果系统能减少空转时间，让 GPU 更少等待，就能提高分母效率。vLLM 的优势就在这里。

再看 TensorRT-LLM。它的核心不是“分页”，而是“把模型执行图编译成更适合 NVIDIA GPU 的形式”。这里有几个关键手段：

- 量化：白话讲就是“用更低精度数字表示权重和计算结果，减少带宽和算力开销”，常见如 FP8、INT4。
- 算子融合：白话讲就是“把多步小操作合并成一步大操作，减少中间内存读写”。
- 并行优化：包括张量并行、流水并行、多卡通信优化。
- 特定解码优化：如 paged KV、speculative decoding 等。

经验上可以用一个粗略关系理解其目标：

$$
throughput \propto cores \times quant\_scale
$$

这里不是严格物理定律，而是工程近似：

- `cores` 表示可用 GPU 计算资源。
- `quant_scale` 表示量化和融合带来的有效速度增益。

例如 FP8 不是“性能自动翻倍”，而是在精度可接受的前提下，让更大一部分计算受益于 Tensor Core，高吞吐场景收益尤其明显。

真实工程例子可以这样看。某团队在 GKE 上先用 TGI 部署 `Llama 3.1 8B` 提供标准聊天 API，适合快速上线；当 RAG 请求携带超长检索上下文，显存峰值上涨，改由 vLLM 承接长上下文流量；而对日均请求量最大的标准问答模型，再单独导出到 TensorRT-LLM，换取更高 TPS 和更低单位成本。这里三者不是竞争关系，而是分工关系。

---

## 代码实现

先给一个最小可运行的玩具代码，用来说明“分页映射”这件事。它不是 vLLM 源码，而是一个可运行的思路模型。

```python
import math

def build_page_table(sequence_length: int, page_stride: int, start_frame: int = 100):
    assert sequence_length >= 0
    assert page_stride > 0

    logical_pages = math.ceil(sequence_length / page_stride) if sequence_length else 0
    page_table = {}

    for logical_page in range(logical_pages):
        physical_frame = start_frame + logical_page * 2
        token_start = logical_page * page_stride
        token_end = min(sequence_length, token_start + page_stride)
        page_table[logical_page] = {
            "physical_frame": physical_frame,
            "token_range": (token_start, token_end),
        }

    return logical_pages, page_table

pages, table = build_page_table(sequence_length=4096, page_stride=1024)
assert pages == 4
assert table[0]["token_range"] == (0, 1024)
assert table[3]["token_range"] == (3072, 4096)

# 长度不是整页倍数时，最后一页只存剩余 token
pages2, table2 = build_page_table(sequence_length=2500, page_stride=1024)
assert pages2 == 3
assert table2[2]["token_range"] == (2048, 2500)

print("page table ok")
```

上面这段代码说明了两件事：

- 序列可以拆成逻辑页。
- 逻辑页可以独立映射到物理 frame，而不要求物理连续。

接着看 TGI 的 Kubernetes 部署示例。目标是把模型直接暴露成 HTTP 服务。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tgi-llama31-8b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tgi-llama31-8b
  template:
    metadata:
      labels:
        app: tgi-llama31-8b
    spec:
      containers:
        - name: tgi
          image: ghcr.io/huggingface/text-generation-inference:latest
          args:
            - "--model-id=meta-llama/Meta-Llama-3.1-8B-Instruct"
            - "--max-input-length=4096"
            - "--max-total-tokens=8192"
            - "--max-batch-total-tokens=2048"
          ports:
            - containerPort: 80
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-secret
                  key: token
          resources:
            limits:
              nvidia.com/gpu: "1"
              cpu: "8"
              memory: "32Gi"
            requests:
              nvidia.com/gpu: "1"
              cpu: "4"
              memory: "16Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: tgi-service
spec:
  selector:
    app: tgi-llama31-8b
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 80
  type: LoadBalancer
```

这里几个关键参数要看懂：

- `--model-id`：模型仓库地址。
- `--max-input-length`：单请求最大输入长度。
- `--max-total-tokens`：输入加输出的总 token 上限。
- `--max-batch-total-tokens`：一个批次允许累计的总 token，用来控制吞吐和显存峰值。

部署后可以直接测试：

```bash
curl http://<EXTERNAL_IP>:8080/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "解释一下什么是 PagedAttention",
    "parameters": {
      "max_new_tokens": 128,
      "temperature": 0.2
    }
  }'
```

再看 vLLM 的服务示例。重点不在 K8S，而在它的分页和调度参数。

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64
```

关键参数含义：

- `--max-model-len`：最大上下文长度。
- `--gpu-memory-utilization`：允许 vLLM 使用的显存比例。
- `--max-num-seqs`：同一时刻允许调度的最大序列数。

最后是 TensorRT-LLM。典型流程是“加载 Hugging Face 权重 -> 转换检查点 -> 编译引擎 -> 启动服务”。

```bash
trtllm-build \
  --checkpoint_dir ./llama31_8b_ckpt \
  --output_dir ./llama31_8b_engine \
  --gemm_plugin fp8 \
  --max_batch_size 32 \
  --max_input_len 4096 \
  --max_output_len 1024
```

如果服务端直接启动：

```bash
trtllm-serve ./llama31_8b_engine \
  --host 0.0.0.0 \
  --port 9000
```

三者在应用层的切换逻辑通常不是“替换一整套业务代码”，而是统一成同一类推理网关：

| 请求类型 | 路由目标 | 理由 |
| --- | --- | --- |
| 标准聊天 API | TGI | 接口稳、部署快 |
| 长上下文 RAG | vLLM | 分页 KV 更省显存 |
| 大流量固定模型 | TensorRT-LLM | 单位成本最低 |
| 实验模型灰度 | TGI 或 vLLM | 改动成本小 |

这意味着业务层最好抽象成“统一推理接口”，而不是把框架细节写死在业务代码里。

---

## 工程权衡与常见坑

实际落地时，最容易出问题的不是模型本身，而是“以为换个框架就能自动变快”。

先看常见坑：

| 坑 | 表现 | 原因 | 规避策略 |
| --- | --- | --- | --- |
| TensorRT-LLM 在非 NVIDIA 环境不可用或收益很低 | 部署卡住、性能不达预期 | 强依赖 NVIDIA 编译和内核生态 | 明确硬件前提，优先在 A100/H100 验证 |
| vLLM 在短文本高并发下首 token 延迟波动 | TPS 高但用户体感不稳 | 动态批次重组引入等待 | 流式输出、前置排队缓冲、调小批次上限 |
| TGI 单卡长上下文容易 OOM | Pod 重启或请求失败 | KV Cache 峰值过高 | 限制 `max-total-tokens`，配合量化与分流 |
| 只看平均延迟，不看 P95/P99 | 线上偶发卡顿 | 批处理策略对尾延迟敏感 | 监控分位数延迟和队列长度 |
| 量化后准确率异常 | 回答变差、格式漂移 | 校准不足或量化策略不匹配 | 先离线评估任务指标再上线 |
| K8S 自动扩容太慢 | 请求尖峰时排队 | GPU 节点冷启动慢 | 保留 warm pool，区分平峰和峰值策略 |

再具体解释三个框架的真实权衡。

TGI 的优点是“服务化成本低”。你可以很快拿到标准 HTTP 接口、流式输出、基础批处理能力和云原生部署方式。问题是它不是为了把显存调度做到最极限，因此当输入特别长、并发很高、模型又很大时，瓶颈会更快暴露。

vLLM 的优点是“同样显存塞进更多活”。但它不是免费午餐。短文本高并发时，连续批处理为了提高 GPU 利用率，可能让个别请求多等几个调度周期。如果产品要求的是极低首 token 延迟，必须结合流式传输、请求缓存、分层队列，而不是只追求 token/sec。

TensorRT-LLM 的优点是“把稳定工作负载压到极限”。但它更像编译型系统，而不是即插即用的通用服务层。模型一旦频繁变更、LoRA 动态切换很多、硬件不统一，工程复杂度会明显上升。

一个真实工程例子是：某客服问答系统白天短问短答很多，夜间批量知识问答较多。白天若全部切到大 batch 的 vLLM，用户可能感觉首字出现不够快；夜间批任务若仍用 TGI 默认参数，又会浪费 GPU。更合理的做法往往是白天偏低延迟配置，夜间偏高吞吐配置，甚至分成两套服务池。

---

## 替代方案与适用边界

不是所有团队都需要 TGI、vLLM 或 TensorRT-LLM。

如果你的问题只是“尽快把一个模型跑起来”，更轻量的替代方案可能更合适。关键不是框架名气，而是你的场景是否真的需要复杂调度和硬件级优化。

| 方案 | 最适场景 | 关键优势 | 关键限制 |
| --- | --- | --- | --- |
| Hugging Face DLC / 托管推理 | 快速上线 PoC、云上试运行 | 省运维 | 灵活性有限，成本可能偏高 |
| TGI | 标准 API 服务、K8S 部署 | 上手快、生态成熟 | 极致性能不是最强 |
| vLLM | 长上下文、高并发 GPU 服务 | 显存利用率高 | 调参和调度复杂 |
| TensorRT-LLM | NVIDIA 高性能生产环境 | 峰值吞吐强 | 硬件依赖重，链路复杂 |
| Triton + TensorRT-LLM | 多模型统一推理平台 | 生产级服务治理能力更强 | 集成复杂度更高 |
| Ollama | 本地开发、演示环境 | 简单、安装快 | 性能与可控性有限 |
| llama.cpp | CPU/小显存设备、本地离线 | 资源占用低 | 大规模在线服务能力有限 |

几个典型判断标准：

- 如果不需要 GPU 强加速，只是本地调试或边缘部署，`llama.cpp` 往往更合适。
- 如果团队运维能力弱，先用托管服务或 TGI，比一上来就做 TensorRT-LLM 更稳。
- 如果需求是“高吞吐优先，且硬件是 NVIDIA”，可以考虑 Triton + TensorRT-LLM 的组合。
- 如果模型迭代频繁、实验很多，vLLM 或 TGI 通常比 TensorRT-LLM 更灵活。
- 如果请求里经常出现超长上下文，vLLM 往往比单纯堆更大 GPU 更划算。

所以适用边界可以简单总结为：

- TGI 更像“服务框架”。
- vLLM 更像“高效运行时”。
- TensorRT-LLM 更像“硬件优化编译器 + 执行引擎”。

选型的正确顺序不是“谁更先进”，而是：

1. 先明确业务目标是吞吐、延迟还是成本。
2. 再确认硬件边界是否允许 TensorRT-LLM。
3. 最后决定是否需要 vLLM 这类更强的显存调度能力。

---

## 参考资料

1. Hugging Face TGI 官方文档，主题：GKE 部署、Docker、服务接口、量化与推理参数  
   https://huggingface.co/docs/google-cloud/examples/gke-tgi-deployment

2. Hugging Face Text Generation Inference 项目与文档，主题：服务化部署、流式输出、批处理  
   https://github.com/huggingface/text-generation-inference

3. vLLM 官方文档，主题：PagedAttention、连续批处理、OpenAI 兼容服务  
   https://docs.vllm.ai/

4. NVIDIA TensorRT-LLM 官方文档，主题：FP8/INT4、引擎编译、部署与性能优化  
   https://developer.nvidia.com/tensorrt-llm

5. NVIDIA TensorRT-LLM GitHub 仓库，主题：构建命令、serve 方式、模型支持矩阵  
   https://github.com/NVIDIA/TensorRT-LLM

6. 百度开发者中心相关文章，主题：TGI、vLLM、TensorRT-LLM 架构与场景对比  
   https://developer.baidu.com/article/detail.html?id=3598085

7. 云端部署示例文章，主题：TGI 在云环境中的容器化部署与性能实践  
   https://docker.recipes/ai-ml/huggingface-tgi

8. 建议进一步查看各框架官方版本说明。命令行参数、支持模型列表、量化能力和硬件要求会随版本变化，生产部署应以当下官方文档为准。
