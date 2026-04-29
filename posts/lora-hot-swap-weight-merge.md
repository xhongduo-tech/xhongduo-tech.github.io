## 核心结论

LoRA 的核心不是“训练出一个新模型”，而是对已有权重做一个低秩增量补丁。低秩，白话说就是“用更少的参数近似表达一块原本很大的变化”。如果基座权重记作 $W_0$，LoRA 学到的不是完整新权重，而是：

$$
\Delta W = sBA
$$

其中 $B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times k}$，$r$ 远小于 $d,k$。推理时真正参与计算的是：

$$
y = (W_0 + sBA)x
$$

“LoRA 即插即用”和“权重合并”不是两种不同算法，而是同一组参数在部署阶段的两种使用方式。

| 方式 | 本质 | 优势 | 代价 | 典型场景 |
|---|---|---|---|---|
| 运行时挂载 LoRA | 请求到来时，把 $\Delta W$ 临时叠加到基座上 | 灵活切换、多任务共享底座 | 多一层适配器管理与调度 | 多租户平台、A/B 测试 |
| 离线权重合并 | 提前把 $\Delta W$ 写回 $W_0$，得到单一模型 | 单请求路径更短、部署更简单 | 失去按请求切换能力 | 单任务、高 QPS 固定服务 |

对初级工程师最重要的判断标准只有一句：如果你需要频繁切换任务或租户，保留 LoRA 作为外挂；如果线上永远只跑一个固定适配器，就把它合并进主模型，减少请求路径长度。

---

## 问题定义与边界

本文讨论的是 LoRA 在推理和部署阶段怎么使用，不讨论 LoRA 是怎么训练出来的。推理阶段，白话说就是“模型已经训好，现在怎么把它接进服务里”。重点只有两个问题：

1. 如何在不改动基座模型文件的前提下，按请求挂载不同 LoRA。
2. 如何把某个固定 LoRA 合并进基座，导出成一个独立可部署模型。

这两个问题看起来都在“用 LoRA”，但工程目标相反。前者强调共享底座和动态切换，后者强调缩短推理链路和简化部署。

本文边界如下：

| 范围 | 是否展开 | 说明 |
|---|---|---|
| LoRA 推理时叠加机制 | 是 | 解释 $W_0 + \Delta W$ 的计算意义 |
| LoRA 离线合并 | 是 | 解释为什么合并前后数学等价 |
| 延迟、显存、部署权衡 | 是 | 面向真实服务决策 |
| 全量微调训练细节 | 否 | 不属于本文主题 |
| QLoRA 训练过程 | 否 | 只在坑点中提到量化兼容性 |
| 分布式训练和并行策略 | 否 | 与部署阶段主线无关 |

几个术语先统一：

| 术语 | 定义 | 白话解释 |
|---|---|---|
| 基座权重 $W_0$ | 预训练模型原始参数 | 没打补丁前的主模型 |
| 增量权重 $\Delta W$ | LoRA 学到的变化量 | 补丁本身 |
| rank $r$ | 低秩分解的中间维度 | 补丁压缩到多小 |
| adapter | LoRA 适配器文件 | 一份可加载的 LoRA 补丁 |
| lora_request | 请求级 LoRA 选择器 | 这次请求要挂哪个补丁 |

一个简单场景就能看出边界差异。假设同一个 7B 基座模型要同时服务“客服问答”“代码解释”“法务摘要”三个入口。如果三者风格、术语、输出格式都明显不同，那么共享一个基座并按请求挂不同 LoRA，通常比导出三个完整模型更省资源。反过来，如果线上只有一个固定的“客服问答”接口，而且长期只用同一个 LoRA，那么继续保留外挂形态只是增加系统复杂度，直接合并更合适。

---

## 核心机制与推导

LoRA 的数学结构很简单。原始线性层一般可以写成：

$$
y = W_0 x
$$

LoRA 不直接训练完整的 $W_0$，而是冻结它，只学习一项低秩修正：

$$
y = (W_0 + \Delta W)x = (W_0 + sBA)x
$$

其中缩放系数常见写法是：

$$
s = \alpha / r
$$

有些实现也支持 rsLoRA，即：

$$
s = \alpha / \sqrt{r}
$$

这里的“缩放”可以理解为控制补丁强度，避免 rank 变化时更新幅度失控。

### 玩具例子：二维矩阵直接算

设：

$$
W_0=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix},
\quad
\Delta W=
\begin{bmatrix}
0.2 & -0.1 \\
0.1 & 0
\end{bmatrix},
\quad
x=
\begin{bmatrix}
3 \\
2
\end{bmatrix}
$$

则未合并时：

$$
y=(W_0+\Delta W)x=
\begin{bmatrix}
1.2 & -0.1 \\
0.1 & 1
\end{bmatrix}
\begin{bmatrix}
3 \\
2
\end{bmatrix}
=
\begin{bmatrix}
3.4 \\
2.3
\end{bmatrix}
$$

如果提前把 LoRA 合并，得到：

$$
W_{\text{merge}} = W_0 + \Delta W =
\begin{bmatrix}
1.2 & -0.1 \\
0.1 & 1
\end{bmatrix}
$$

再推理时直接算：

$$
y=W_{\text{merge}}x=
\begin{bmatrix}
3.4 \\
2.3
\end{bmatrix}
$$

结果完全一样。这个例子说明一件事：合并不会改变数学结果，改变的是工程路径。原来路径是“先载入基座，再叠加补丁，再计算”；合并后路径是“直接用已经写好补丁的单一权重计算”。

等价性可以拆成三步看：

| 步骤 | 即插即用路径 | 合并路径 |
|---|---|---|
| 1 | 读取 $W_0$ | 读取 $W_0$ 与 $\Delta W$ |
| 2 | 推理时计算 $W_0 + \Delta W$ | 离线先算 $W_{\text{merge}}=W_0+\Delta W$ |
| 3 | 用 $(W_0+\Delta W)x$ 出结果 | 用 $W_{\text{merge}}x$ 出结果 |

真实模型里，这个过程不是只发生在一个 $2 \times 2$ 矩阵上，而是发生在注意力层或前馈层的多个线性变换上。原理没变，只是张量更大、层更多。

### 真实工程例子：多租户推理服务

假设你做一个 SaaS 文本平台，客户 A 要“法律摘要”，客户 B 要“电商客服”，客户 C 要“SQL 助手”。三者都基于同一个基座模型，但各自上传自己的 LoRA。此时最合理的架构通常不是为 A、B、C 各自常驻一整份完整模型，而是：

1. 常驻一个基座模型。
2. 按请求识别租户。
3. 根据租户选择对应的 `lora_request`。
4. 在推理阶段动态挂载。

这样做的收益是共享底座显存，降低模型副本数量；代价是服务层必须管理 adapter 生命周期、映射关系、缓存与回收策略。

如果后来客户 B 的流量占了总流量 90%，并且它的适配器半年不变，那么对 B 单独导出一份已合并模型，往往就能换到更短的请求链路和更稳定的吞吐表现。这就是“动态挂载”和“离线合并”在同一系统里并存的原因。

---

## 代码实现

代码层面要把“运行时挂载”和“离线合并”视为两条不同路径。

### 1. 运行时挂载 LoRA

这种方式下，基座模型始终保持不变，请求只携带一个“本次使用哪个 adapter”的选择器。

```python
from dataclasses import dataclass

@dataclass
class LoraRequest:
    adapter_name: str
    scale: float = 1.0

def infer(base_model, prompt: str, lora_request: LoraRequest | None = None) -> str:
    if lora_request is None:
        return f"[base] {prompt}"
    return f"[base + {lora_request.adapter_name} x {lora_request.scale}] {prompt}"

req = LoraRequest(adapter_name="legal-v1", scale=1.0)
out = infer("base-7b", "summarize contract", req)

assert "legal-v1" in out
assert "summarize contract" in out
```

在 vLLM 这类推理框架里，常见接口形态类似：

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(model="base_model_path", enable_lora=True)

lora_request = LoRARequest("legal_adapter", 1, "/path/to/legal_lora")
outputs = llm.generate(
    ["summarize this contract"],
    lora_request=lora_request,
)
```

这里 `enable_lora=True` 的意思是服务准备接受外挂适配器。`LoRARequest` 可以理解为“这次请求挂哪个补丁”。

### 2. 离线合并 LoRA

这种方式下，先把 LoRA 写回主模型，再保存成一个独立目录，之后推理时就不再需要 adapter。

```python
import numpy as np

def merge_weight(W0, B, A, alpha):
    r = A.shape[0]
    scale = alpha / r
    delta = scale * (B @ A)
    return W0 + delta

W0 = np.eye(2)
B = np.array([[1.0], [0.5]])
A = np.array([[0.2, -0.1]])
merged = merge_weight(W0, B, A, alpha=1.0)

x = np.array([[3.0], [2.0]])
y_runtime = (W0 + (1.0 / 1) * (B @ A)) @ x
y_merged = merged @ x

assert np.allclose(y_runtime, y_merged)
assert merged.shape == (2, 2)
```

使用 Hugging Face PEFT 时，合并路径通常是：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
model = PeftModel.from_pretrained(base_model, "lora_path")

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_out")
```

这里有一个很关键的工程点：`merge_and_unload()` 通常返回的是合并后的模型对象。它的语义是“生成合并结果并移除适配器包装”，不要想当然地把它当成原地修改。

### 3. 部署流程对比

| 步骤 | 运行时挂载 | 离线合并 |
|---|---|---|
| 加载基座模型 | 需要 | 需要 |
| 加载 LoRA adapter | 每次路由到对应 adapter 时需要 | 只在导出阶段需要 |
| 推理前叠加增量 | 需要 | 不需要 |
| 保存独立模型 | 不需要 | 需要 |
| 请求中传 `lora_request` | 需要 | 不需要 |

如果你把这两种路径混在一起，就容易出现“代码能跑，但部署哲学冲突”的问题。比如你已经决定合并成独立模型，却仍保留一整套 adapter 路由逻辑，那只是增加系统维护面。

---

## 工程权衡与常见坑

两种路径没有绝对优劣，只有目标函数不同。

| 维度 | 运行时挂载 LoRA | 离线合并 |
|---|---|---|
| 灵活性 | 高 | 低 |
| 单请求延迟 | 较高 | 较低 |
| 多租户支持 | 强 | 弱 |
| 底座共享能力 | 强 | 弱 |
| 部署复杂度 | 高 | 中 |
| 回滚新适配器 | 容易 | 需重新导出 |
| 适合高 QPS 固定接口 | 一般 | 强 |

### 常见坑 1：把“数学等价”误解成“系统表现完全等价”

合并前后，在理想浮点条件下线性层结果等价，但工程系统里还有量化、内存布局、内核实现、缓存策略。这意味着“理论一样”不代表“吞吐、时延、显存、数值误差”都完全一样。尤其是量化底座叠加 LoRA 再合并时，一定要做回归测试。

### 常见坑 2：多租户场景误用合并

如果平台要按用户、按任务、按版本切换 LoRA，把每个 LoRA 都合并出一份完整模型，短期看省掉了请求级调度，长期看会把模型副本、存储和发布流程全部放大。真正的瓶颈常常从“推理延迟”转移成“模型资产管理”。

### 常见坑 3：`max_lora_rank` 盲目设大

rank 是低秩分解的容量上限。容量，白话说就是“补丁最多能表达多少变化”。线上框架如果为所有请求都预留很大的 LoRA rank 空间，会浪费显存和算力。应按真实适配器的最大 rank 配置，而不是为了“以后可能会用”无限放大。

### 常见坑 4：命名和路由不统一

`adapter_name`、目录名、数据库记录、服务层 `lora_request id` 如果不统一，最常见的事故不是模型崩，而是“请求挂错 adapter”。这类问题输出也许还能看起来合理，因此排查更慢。实际工程里要把“租户 ID -> adapter 版本 -> 文件路径”做成明确映射，而不是靠手写字符串。

### 常见坑 5：忽略冷启动与缓存

运行时挂载并不只是一次矩阵加法。真实服务通常还涉及 adapter 文件加载、GPU/CPU 内存迁移、缓存命中与淘汰。如果你的租户长尾很多、切换频繁，那么理论上的“共享底座更省”不一定直接转化成线上稳定性，可能需要 adapter 预热和缓存分层。

---

## 替代方案与适用边界

LoRA 即插即用和权重合并都只是部署选择，不是唯一选择。真正的判断维度有四个：任务是否稳定、是否多租户、是否频繁切换、是否极度在意单请求延迟。

| 方案 | 适用场景 | 优点 | 代价 |
|---|---|---|---|
| 运行时挂载 LoRA | 多任务、多租户、频繁切换 | 共享底座，切换灵活 | 路由和适配器管理更复杂 |
| 离线合并 LoRA | 单任务、固定线上接口、高 QPS | 请求路径短，部署简单 | 失去动态切换能力 |
| 全量微调 | 需要大幅改变模型行为 | 表达能力强 | 训练和存储成本高 |
| 多个独立模型 | 业务强隔离、组织边界明确 | 最直观，故障域清晰 | 成本最高，资源复用最弱 |

可以用一个简单决策表判断：

| 问题 | 是 | 否 |
|---|---|---|
| 是否多租户？ | 优先考虑运行时挂载 | 继续看下一项 |
| 是否需要频繁切换任务/版本？ | 优先考虑运行时挂载 | 继续看下一项 |
| 是否对最低延迟非常敏感？ | 优先考虑离线合并 | 继续看下一项 |
| 是否长期固定一个 adapter？ | 优先考虑离线合并 | 运行时挂载通常更稳妥 |

一个实用的工程结论是：不要把系统强行设计成“全挂载”或“全合并”。真实部署常常是混合形态。核心高流量任务走合并模型，长尾任务保留挂载能力；稳定版本合并，实验版本外挂。这种折中比纯理论立场更接近生产环境。

---

## 参考资料

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. [Hugging Face PEFT LoRA Conceptual Guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
3. [Hugging Face PEFT LoRA API Reference](https://huggingface.co/docs/peft/main/package_reference/lora)
4. [vLLM Documentation: LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora.html)
5. [microsoft/LoRA GitHub Repository](https://github.com/microsoft/LoRA)
