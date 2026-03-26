## 核心结论

Qwen3.5 可以概括为一条明确路线：不是在 Qwen3 上单纯加大参数，而是把“推理能力、多模态输入、代理执行”放到同一套架构里统一优化。对初级工程师最重要的结论有四个。

第一，官方开源旗舰是 `Qwen3.5-397B-A17B`。`397B` 是总参数，`17B` 是每个 token 真正参与计算的激活参数。激活参数可以理解为“这一步真正上工的参数数量”，它决定了单步推理成本不需要按 397B 全量计算。

第二，Qwen3.5 的多模态不是“先视觉、后语言”的旧式拼接，而是训练阶段就把文本、图像、视频、GUI 等早期融合为统一 token 流。白话说，模型不是先让一个单独视觉模型做摘要，再交给语言模型，而是从一开始就把“看”和“说”放在同一个推理链里学。

第三，它把长上下文和推理时计算扩展一起做了。开源权重原生支持 `262,144` token，上下文经 RoPE/YaRN 等方案可扩到约 `1,010,000`；托管版 `Qwen3.5-Plus` 默认就是 1M 级上下文，同时提供 `Auto / Thinking / Fast` 三种模式。`Thinking` 适合难题深推理，`Fast` 适合低延迟直答，`Auto` 则让系统自己决定何时多想、何时用工具。

第四，和 DeepSeek-R1 对比时，不能只看总参数。DeepSeek-R1 官方卡片给出的数据是 `671B / 37B / 128K`；Qwen3.5 旗舰是 `397B / 17B / 262K~1M`，并且原生支持图像、视频、GUI 理解。所以两者不是“谁参数大谁强”的关系，而是“谁在你的任务分布下更合适”的关系。

| 维度 | Qwen3.5-397B-A17B | Qwen3.5-Plus | DeepSeek-R1 |
|---|---:|---:|---:|
| 总参数 | 397B | 托管对应旗舰 | 671B |
| 激活参数 | 17B | 未单独披露，能力对应旗舰 | 37B |
| 原生上下文 | 262K | 1M 默认 | 128K |
| 模态 | 文本、图像、视频、GUI | 文本、图像、视频、GUI | 以文本为主 |
| 推理模式 | 默认 thinking，可关 | Auto / Thinking / Fast | reasoning 风格 |
| 许可 | Apache 2.0 | 服务形态 | 开源权重可用 |

---

## 问题定义与边界

Qwen3.5 解决的问题，不是普通聊天，而是“长链、多模态、带工具”的 agent 任务。agent 可以理解为“会拆步骤、会调用工具、会根据中间结果继续行动的模型系统”。

典型输入长这样：一张后台管理页面截图、一段 Python 报错日志、一个“先定位按钮再改配置并解释原因”的指令。如果还沿用传统 pipeline，往往要先做 OCR，再做截图结构分析，再让语言模型串起来，链路长、误差多、上下文还容易丢。Qwen3.5 的设计边界就是尽量把这些步骤压到同一模型里。

但它也有明确代价。虽然每步只激活 17B 参数，完整部署仍要加载 397B 权重。按 FP16 粗算，显存或统一内存需求约为
$$
397\times 10^9 \times 2 \text{ bytes} \approx 794 \text{ GB}
$$
再考虑框架开销，工程上通常会说“接近 800GB”。即便做 4bit 量化，纯权重也接近 200GB，带缓存和运行时开销后，220GB 级别内存仍然很常见。这意味着它更适合多卡集群、云端托管或推理服务框架，不适合“单张消费级显卡直接跑满旗舰”。

一个玩具例子：把“按钮在截图左上角，报错是 `ModuleNotFoundError`，请告诉我先点哪里再修什么”一起丢给模型。旧方案要 3 个模块串联；Qwen3.5 的目标是一次完成理解和推理。

一个真实工程例子：客服自动化平台要处理网页截图、用户工单文本、数据库返回值和工具调用结果。这里真正难的不是一句问答，而是 5 到 20 步的状态衔接。长上下文和原生多模态，比单轮 benchmark 分数更关键。

| 边界项 | 适合 Qwen3.5 的场景 | 不适合优先选 Qwen3.5 的场景 |
|---|---|---|
| 上下文 | 128K 以上、最好 256K 以上 | 4K 到 32K 短文本 |
| 模态 | 图像、视频、GUI、代码混合 | 纯文本 FAQ |
| 部署资源 | 多卡、云服务、推理集群 | 单机低成本本地部署 |
| 任务类型 | agent、工具调用、长链推理 | 低时延简单补全 |
| 许可 | 需要 Apache 2.0 开源权重 | 必须纯托管闭源能力 |

---

## 核心机制与推导

Qwen3.5 旗舰的核心是“Gated DeltaNet + MoE”的混合架构。`MoE` 是 Mixture of Experts，白话讲就是“把大模型拆成很多专家子模块，每次只挑少数几个上场”。`Gated DeltaNet` 可以先理解成一种更高效的线性注意力路线，用来降低超长上下文下的计算和缓存压力。

官方模型卡给出的关键结构是：60 层，总体布局为  
`15 × (3 × (Gated DeltaNet -> MoE) + 1 × (Gated Attention -> MoE))`。  
每个 token 在 512 个专家里，只激活 `10 个 routed experts + 1 个 shared expert`。

MoE 的常见输出写法是：
$$
y=\sum_{i=1}^{k} g_i(h)\cdot E_i(h), \quad \sum_{i=1}^{k} g_i(h)=1
$$
其中 $h$ 是当前 token 的隐藏状态，$E_i$ 是第 $i$ 个专家，$g_i(h)$ 是门控权重。白话讲，就是“先判断这一步最该找哪几个专家，再把它们的结果按权重混合”。

如果只看激活比例：
$$
\frac{P_{\text{active}}}{P_{\text{total}}}=\frac{17}{397}\approx 4.28\%
$$
这不表示总成本永远只有 4.28%，因为还有注意力、KV cache、视觉编码和通信开销；但它准确说明了一件事：在前馈主干上，Qwen3.5 不是每步都把 397B 全部跑一遍。这也是它吞吐率显著优于同级稠密模型的重要原因。

把它讲成更直观的玩具例子：假设 512 个专家像 512 个不同工种的工程师，当前 token 是“解释这段前端报错并根据截图找按钮”。路由器会判断这一步更需要哪些专家，比如代码理解、UI 定位、中文表达、工具调用规划，然后只叫 11 个参与，而不是把 512 个人全部拉进会议室。

Qwen3.5-VL 这个叫法在社区里会出现，但官方开源旗舰名称直接就是 `Qwen3.5-397B-A17B`，它本身已经是 vision-language foundation model。更准确的表述不是“Qwen3.5 上再外挂一个 VL 分支”，而是“Qwen3.5 旗舰从一开始就是统一视觉语言模型”。

---

## 代码实现

下面先用一个可运行的 Python 玩具实现，模拟 Top-k MoE 路由。它不是真实 Qwen3.5 权重，只是帮助理解“为什么 512 个专家里只需激活 11 个”。

```python
from math import exp

def softmax(xs):
    exps = [exp(x) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def topk_moe(hidden, expert_scores, expert_outputs, k=3):
    assert len(expert_scores) == len(expert_outputs)
    top_idx = sorted(range(len(expert_scores)), key=lambda i: expert_scores[i], reverse=True)[:k]
    top_scores = [expert_scores[i] for i in top_idx]
    weights = softmax(top_scores)
    y = 0.0
    for w, i in zip(weights, top_idx):
        y += w * expert_outputs[i](hidden)
    return y, top_idx, weights

experts = [
    lambda h: h + 1,      # 假装是“代码专家”
    lambda h: h * 2,      # 假装是“数学专家”
    lambda h: h - 3,      # 假装是“UI专家”
    lambda h: h ** 2,     # 假装是“规划专家”
]

scores = [0.2, 1.8, 0.5, 1.2]
y, idx, w = topk_moe(hidden=3.0, expert_scores=scores, expert_outputs=experts, k=2)

assert idx == [1, 3]
assert abs(sum(w) - 1.0) < 1e-9
assert y > 0

active_ratio = 17 / 397
assert round(active_ratio, 4) == 0.0428
print(y, idx, w, active_ratio)
```

真正部署时，更接近下面两种方式。

第一种是本地或自建服务，用 vLLM 直接拉起 OpenAI 兼容接口：

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

如果要启用工具调用：

```bash
vllm serve Qwen/Qwen3.5-397B-A17B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

第二种是直接走 OpenAI 兼容 SDK：

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-397B-A17B-FP8",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "阅读截图并解释报错根因，给出修复步骤。"},
                {"type": "image_url", "image_url": {"url": "https://example.com/ui.png"}}
            ],
        }
    ],
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False}
    },
    max_tokens=1024,
)

print(resp.choices[0].message.content)
```

真实工程例子：把“运维控制台截图 + nginx 配置片段 + 500 错误日志 + 修复目标”放进一次请求，让模型先看界面，再解释代码，再给出操作顺序。这正是 Qwen3.5 相比纯文本 reasoning 模型更有价值的地方。

---

## 工程权衡与常见坑

最常见的误判，是把“总参数大”直接等同于“效果更强”或“速度更慢”。对于 MoE，这种判断很容易错。因为总参数决定的是模型容量上限，激活参数才更接近每步真实算力消耗。

第二个坑，是把上下文长度只当宣传数字。1M token 并不表示任何任务都该塞到 1M。上下文越长，KV cache 压力越大，调度和时延也会上升。真正正确的做法是按任务长度分档：32K、128K、256K、1M，不同路由不同服务层。

第三个坑，是忽略多模态输入的清洗。原生多模态不等于“任何截图都能直接懂”。低分辨率、信息遮挡、重复 UI、局部裁剪错误，都会让代理任务失败。工程上仍要做截图裁切、关键区域提取、历史步骤压缩。

第四个坑，是把 DeepSeek-R1 和 Qwen3.5 当作完全同类产品。DeepSeek-R1 更像纯文本 reasoning 强项模型；Qwen3.5 则更强调 native multimodal + agent。前者在很多数学和推理 benchmark 很强，后者在“截图+文本+工具”的复合场景更自然。比较时至少要同时看四项：激活参数、上下文、模态、工具链。

| 常见误判 | 为什么错 | 更稳妥的判断方式 |
|---|---|---|
| 参数越大越快 | MoE 不按总参数全量执行 | 看激活参数与服务吞吐 |
| 上下文越长越好 | 超长上下文有明显缓存和时延代价 | 看真实平均输入长度 |
| 多模态=识图更强 | 多模态还包括 GUI、视频、工具联动 | 看任务链是否跨模态 |
| 开源权重=本地容易跑 | 397B 仍是超大模型 | 先算显存、带宽、并行策略 |
| benchmark 高=agent 更稳 | 工具调用与状态管理是另一类能力 | 做端到端 workflow 测试 |

---

## 替代方案与适用边界

如果你只做纯文本、代码补全、32K 到 64K 上下文，Qwen3 或 Qwen3-Max 往往更直接，部署链路也更成熟。对于很多中小团队，先把“文本推理”做好，比一步上原生多模态旗舰更现实。

如果你需要 1M 上下文、官方内置工具、云端即开即用，`Qwen3.5-Plus` 比自托管 `397B-A17B` 更合适。因为你买到的不只是模型本身，还包括更完整的工具集成和服务侧优化。

如果你预算更敏感，`Qwen3.5-35B-A3B`、`Qwen3.5-27B`、`Qwen3.5-122B-A10B` 这类缩放型号更像实用工程解。这里的判断标准很简单：先看你的问题是否真的需要“旗舰级多模态 agent”，再看是否值得为它付出集群成本。

| 方案 | 适合任务 | 优点 | 局限 |
|---|---|---|---|
| Qwen3.5-397B-A17B | 多模态 agent、长链推理 | 开源旗舰、原生多模态、262K+ | 部署成本极高 |
| Qwen3.5-Plus | 需要 1M 上下文和工具 | 托管省运维、Auto 模式 | 依赖云服务 |
| Qwen3-Max / Qwen3 | 纯文本 reasoning | 接口简单、文本能力稳 | 多模态链路不如 3.5 原生 |
| Qwen3.5-35B-A3B / 27B | 成本敏感部署 | 更易落地 | 上限低于旗舰 |
| DeepSeek-R1 | 纯文本推理、数学代码对比 | reasoning 强、生态成熟 | 128K，上图像和 GUI 不原生 |

---

## 参考资料

1. Alibaba Cloud Community, *Qwen3.5: Towards Native Multimodal Agents*  
   https://www.alibabacloud.com/blog/602894

2. Hugging Face, *Qwen/Qwen3.5-397B-A17B Model Card*  
   https://huggingface.co/Qwen/Qwen3.5-397B-A17B

3. Hugging Face, *Qwen/Qwen3.5-397B-A17B-FP8 Model Card*  
   https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8

4. NVIDIA NIM, *qwen / qwen3.5-397b-a17b*  
   https://docs.api.nvidia.com/nim/reference/qwen-qwen3-5-397b-a17b

5. Alibaba Cloud Model Studio, *Newly released models*  
   https://www.alibabacloud.com/help/en/model-studio/newly-released-models

6. Alibaba Cloud Model Studio, *Models*  
   https://www.alibabacloud.com/help/en/model-studio/models

7. DeepSeek 官方模型卡, *deepseek-ai/DeepSeek-R1*  
   https://huggingface.co/deepseek-ai/DeepSeek-R1

8. DataCamp, *Qwen3.5* 相关解读  
   https://www.datacamp.com/blog/qwen3-5
