## 核心结论

DeepSeek-VL2 是一个“视觉编码器 + 投影器 + DeepSeekMoE 语言骨干”的多模态 MoE 模型。MoE，全称 Mixture of Experts，白话说就是模型内部有多个“专家子网络”，每个 token 只调用其中一小部分，而不是每次都跑完整模型。

它解决的重点不是“模型能不能看图”，而是“在高分辨率图片、长文档、复杂版面输入下，如何控制算力、显存和上下文成本”。例如 PDF、票据、图表问答中，同一页里可能同时有正文、表格、图例、印章、OCR 字符和坐标关系。DeepSeek-VL2 不是把整张图硬塞给模型，而是把图拆成多个块，再让不同专家分工处理。

模型结构总图可以抽象为：

```text
图像 I
  |
  v
SigLIP / Vision Transformer        文本 tokens
  |                                      |
  v                                      v
视觉特征 -----------------------> 拼接后的多模态 token 序列
  |
  v
MlpProjector
  |
  v
DeepSeekMoE LLM
  |
  v
答案 / 推理结果
```

MoE 路由示意图：

```text
token h_t
  |
  v
Gate 门控器
  |
  +--> expert scores: [0.05, 0.70, 0.10, 0.20]
  |
  +--> TopK(K=2): 专家 2、专家 4
  |
  v
0.78 * E2(h_t) + 0.22 * E4(h_t) + SharedExpert(h_t)
```

最小数值例子：如果 4 个专家分数是 `[0.05, 0.70, 0.10, 0.20]`，`top_k=2`，则选中专家 2 和专家 4。若只对这两个分数归一化，权重约为 `[0.78, 0.22]`。这说明每个 token 不会把所有专家都跑一遍，只调用最相关的两个专家，同时共享专家仍然参与。

---

## 问题定义与边界

DeepSeek-VL2 面向的问题是：把高分辨率图像、长文档、复杂版面和图表推理输入统一转成语言模型能处理的 token 序列，并在推理时用稀疏专家降低每个 token 的实际计算量。

这里的“动态分辨率”指模型不会固定用一种图像尺寸处理所有输入，而是根据图像比例和细节需求切成全局图块与局部图块。简单图片只需要保留主要内容，长文档需要保留更多局部细节，所以处理方式不同。

边界也要明确：DeepSeek-VL2 不是纯视觉模型，因为它最终要进入语言模型做问答、描述和推理；也不是纯语言模型，因为输入侧有图像编码和视觉 token；它的核心是跨模态对齐与稀疏专家调度。

| 输入类型 | 难点 | 为什么需要动态分辨率 | 为什么需要 MoE |
|---|---|---|---|
| 身份证照片 | 字符小、区域固定、需要 OCR 与字段理解 | 需要保留文字区域细节，但版面较简单 | token 数中等，MoE 可降低语言推理成本 |
| 单张商品图 | 主体明显、背景干扰少 | 通常不需要过多 tile | Dense 模型也可能足够 |
| 10 页 PDF | 页面多、文字密、表格和图混合 | 需要把局部区域切细，否则小字丢失 | token 数高，稀疏激活更重要 |
| 图表问答 | 图例、坐标轴、数值关系复杂 | 需要保留坐标与标注细节 | 不同 token 可能需要不同专家处理 |
| 票据 / 发票 | 小字密集、字段位置重要 | 需要更高局部分辨率 | 结构化字段和自然语言问题可分配给不同专家 |

玩具例子：一张 2x2 的图片里，左上角是标题，右上角是表格，左下角是图例，右下角是备注。动态分辨率会把这些区域拆成不同视觉 token；MoE 会让标题 token、表格 token、图例 token 走不同专家组合。

真实工程例子：一个财务系统要回答“这张发票的付款金额是多少，税率是多少，供应商是谁”。系统先把 PDF 页渲染成图片，再切成多个 tile。金额、税率、供应商名称可能位于不同区域，模型需要同时做 OCR、字段定位和语义归纳。

---

## 核心机制与推导

视觉侧先由 `SigLIP` 提取图像特征。SigLIP 是一种视觉-语言预训练视觉编码器，白话说就是把图片切成 patch 后编码成向量。`MlpProjector` 不是视觉编码器，它的作用是把视觉特征映射到语言模型的隐藏维度，让视觉 token 能进入 LLM。

视觉侧可写成：

$$
z=\mathrm{Proj}(\mathrm{SigLIP}(I))
$$

其中 \(I\) 是输入图像，`SigLIP(I)` 是视觉特征，`Proj` 是 `MlpProjector`，输出 \(z\) 是可拼接进语言模型的视觉 token。

MoE 侧的关键是门控器。门控器，白话说就是一个给专家打分的小网络，它根据当前 token 的隐藏状态决定该 token 应该交给哪些专家处理。设输入 token 隐藏状态为 \(h_t\)，则：

$$
p_t=\text{Gate}(h_t),\quad \mathcal K_t=\mathrm{TopK}(p_t,K),\quad
y_t=\sum_{i\in \mathcal K_t}\alpha_{t,i}E_i(h_t)+\sum_{j=1}^{K_s}S_j(h_t)
$$

其中 \(E_i\) 是路由专家，\(S_j\) 是共享专家，\(\mathcal K_t\) 是选中的专家集合，\(\alpha_{t,i}\) 是归一化后的专家权重。共享专家，白话说就是每个 token 都会调用的公共专家，用来保留通用能力。

`softmax` 和 `sigmoid` 都可用于门控分数。`softmax` 会让所有专家分数相互竞争，总和为 1；`sigmoid` 更像给每个专家独立打分。`norm_topk_prob` 表示是否只对选中的 top-k 专家权重再归一化。

| 设置 | 含义 | 对路由权重的影响 | 常见风险 |
|---|---|---|---|
| `softmax` | 专家之间竞争 | 全部专家概率和为 1 | top-k 后权重可能偏尖锐 |
| `sigmoid` | 每个专家独立打分 | 分数不天然总和为 1 | 需要关注尺度 |
| `norm_topk_prob=True` | top-k 权重重新归一 | 选中专家权重和为 1 | 更易解释，需确认源码行为 |
| `norm_topk_prob=False` | 保留原始 top-k 权重 | 输出幅度受原分数影响 | 可能导致层间尺度变化 |

用给定数值推导一次。专家分数为 `[0.05, 0.70, 0.10, 0.20]`，`top_k=2`，选中专家 2 和专家 4。若对 top-k 分数归一化：

$$
\alpha_2=\frac{0.70}{0.70+0.20}=0.78,\quad
\alpha_4=\frac{0.20}{0.70+0.20}=0.22
$$

输出近似为：

$$
y=0.78E_2(h)+0.22E_4(h)+S(h)
$$

这就是 DeepSeek-VL2 在多模态场景中扩展能力的关键：图像 token、OCR token、普通文本 token 都进入同一个语言骨干，但每个 token 可以激活不同专家组合。

---

## 代码实现

源码层面可以按三个对象理解：`modeling_deepseek_vl_v2.py` 负责主模型与多模态输入组织，`siglip_vit.py` 负责视觉编码器，`MoEGate.forward()` 负责 MoE 路由逻辑。

代码流程图：

```text
images + text
  |
  +--> image processor / dynamic tiling
  |       |
  |       v
  |   SigLIP VisionTransformer
  |       |
  |       v
  |   MlpProjector
  |
  +--> tokenizer
          |
          v
  multimodal embeddings
          |
          v
  DeepSeekMoE decoder layers
          |
          +--> MoEGate.forward()
          +--> topk_idx / topk_weight
          +--> routed experts + shared experts
          |
          v
  logits / generated answer
```

关键函数列表：

| 代码对象 | 作用 | 新手解释 |
|---|---|---|
| `DeepseekVLV2ForCausalLM` | 多模态因果语言模型入口 | 接收图像和文本并生成回答 |
| `VisionTransformer` / `SigLIP` | 提取图像 patch 特征 | 把图片变成向量 |
| `MlpProjector` | 对齐视觉维度和语言维度 | 把视觉向量改成 LLM 能读的形状 |
| `MoEGate.forward()` | 计算专家路由 | 给每个 token 选择专家 |
| `topk_idx` | 被选中的专家编号 | 这个 token 要找哪些专家 |
| `topk_weight` | 被选中专家的权重 | 每个专家输出占多少比例 |

输入到输出的时序表：

| 步骤 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 1 | 图片 / PDF 页 | 动态切块 | image tiles |
| 2 | image tiles | SigLIP 编码 | vision features |
| 3 | vision features | MLP 投影 | vision tokens |
| 4 | 用户问题 | tokenizer | text tokens |
| 5 | vision + text tokens | 拼接送入 LLM | hidden states |
| 6 | hidden states | MoE gate 选专家 | `topk_idx` |
| 7 | hidden states + `topk_idx` | 专家计算并加权 | next hidden states |
| 8 | final hidden states | LM head | 答案 token |

接近源码的伪代码如下：

```python
import math

def topk_route(scores, k=2, normalize=True):
    indexed = list(enumerate(scores))
    top = sorted(indexed, key=lambda x: x[1], reverse=True)[:k]
    topk_idx = [i for i, _ in top]
    topk_scores = [s for _, s in top]

    if normalize:
        total = sum(topk_scores)
        weights = [s / total for s in topk_scores]
    else:
        weights = topk_scores

    return topk_idx, weights

scores = [0.05, 0.70, 0.10, 0.20]
topk_idx, weights = topk_route(scores, k=2)

assert topk_idx == [1, 3]          # Python 下标从 0 开始，对应专家 2 和专家 4
assert math.isclose(weights[0], 0.70 / 0.90, rel_tol=1e-6)
assert math.isclose(weights[1], 0.20 / 0.90, rel_tol=1e-6)
assert math.isclose(sum(weights), 1.0, rel_tol=1e-6)

# 接近模型主流程的抽象写法：
# vision_feat = siglip_vit(images)
# vision_token = projector(vision_feat)
# hidden_states = concat(vision_token, text_token)
# gate_prob = moe_gate(hidden_states)
# topk_idx = gate_prob.topk(k=2)
# output = routed_experts(hidden_states, topk_idx) + shared_experts(hidden_states)
```

这段代码没有实现完整神经网络，只复现 MoE 路由的最小逻辑。真实模型中，`scores` 来自 `MoEGate.forward()`，专家输出来自多个前馈网络，图像向量来自 `siglip_vit.py` 中的视觉编码器。

---

## 工程权衡与常见坑

MoE 的优势不是“总参数越大越好”，而是“每个 token 实际激活参数更少”。Dense 模型，白话说就是每个 token 都经过同一整套参数；MoE 模型总参数可以更大，但每个 token 只激活少量路由专家和共享专家。因此比较效率时不能只看总参数，要看每 token 激活参数、上下文长度、专家调度开销和显存峰值。

高分辨率 PDF 是最典型的坑。图片切得越细，模型看到的信息越多，但 token 数也会变多。tile 数暴增后，视觉 token 会挤占上下文窗口，预填充阶段显存上升，响应变慢，甚至直接 OOM。

| 问题 | 后果 | 规避 |
|---|---|---|
| tile 太多 | 显存升高 | 降低分辨率，限制 `images_spatial_crop` |
| 上下文爆 | 响应变慢，截断关键信息 | 启用增量预填充，分段处理长文档 |
| top-k 权重未归一 | 输出尺度不稳定 | 检查 `norm_topk_prob` |
| 激活分布不均 | 专家负载偏斜 | 统计 `topk_idx`，按 token 类型分桶 |
| 把 projector 当视觉编码器 | 模块理解错误 | 区分 `SigLIP` 和 `MlpProjector` |
| 图像 token 与文本 token 混看 | 难以定位路由问题 | 分别统计 image/text token 路由分布 |

检查 `MoEGate.forward()` 的 `topk_idx` 分布时，可以按图像 token 和文本 token 分桶：

```python
from collections import Counter

def count_expert_usage(topk_idx_by_token, token_types):
    """
    topk_idx_by_token: 形如 [[1, 3], [2, 5], ...]
    token_types: 形如 ["image", "image", "text", ...]
    """
    buckets = {"image": Counter(), "text": Counter()}

    for experts, token_type in zip(topk_idx_by_token, token_types):
        for expert_id in experts:
            buckets[token_type][expert_id] += 1

    return buckets

topk_idx_by_token = [[1, 3], [1, 2], [0, 3], [2, 3]]
token_types = ["image", "image", "text", "text"]
usage = count_expert_usage(topk_idx_by_token, token_types)

assert usage["image"][1] == 2
assert usage["text"][3] == 2
```

如果图像 token 长期集中到极少数专家，可能说明专家负载不均；如果文本 token 和图像 token 的路由完全没有差异，也需要检查门控输入、token 标记或统计方式是否正确。

---

## 替代方案与适用边界

DeepSeek-VL2 适合“复杂版面 + 语义理解 + 较高吞吐”的场景，例如文档问答、票据理解、图表推理、截图问答。它不一定适合极低延迟、输入非常简单、只要分类不要推理，或者部署环境不允许复杂专家调度的场景。

如果只是判断图片是猫还是狗，MoE 多模态模型可能太重；如果要问一页 PDF 里的付款金额、付款对象和条款依据，就更值得使用 DeepSeek-VL2 这类模型。

| 方案 | 优点 | 缺点 | 适用任务 | 代价 |
|---|---|---|---|---|
| Dense 多模态模型 | 实现简单，推理路径稳定 | 每 token 都跑完整前馈层 | 通用图文问答、简单文档理解 | 激活计算较高 |
| DeepSeek-VL2 MoE | 总容量大，每 token 稀疏激活 | 路由、负载、部署更复杂 | PDF 问答、票据、图表推理 | 需要管理专家调度和上下文 |
| 纯视觉模型 | 速度快，任务边界清晰 | 不擅长开放式语言推理 | 分类、检测、分割 | 需要单独接业务逻辑 |
| 纯 OCR 管线 | 可控、可解释、成本低 | 难处理跨区域语义推理 | 表单抽取、固定票据 | 模板维护成本高 |
| OCR + LLM | 工程成熟，便于插入规则 | OCR 错误会传递给 LLM | 半结构化文档问答 | 依赖 OCR 质量 |

三个任务对比：

| 任务 | 更合适的方案 | 原因 |
|---|---|---|
| 图片分类 | 纯视觉模型 | 只要类别，不需要长文本推理 |
| 文档问答 | DeepSeek-VL2 MoE 或 OCR + LLM | 需要理解版面和问题语义 |
| 表格推理 | DeepSeek-VL2 MoE | 需要同时看结构、数字和上下文 |

工程选择的核心不是“哪个模型最新”，而是输入复杂度、延迟预算、可解释性要求和部署成本。DeepSeek-VL2 的价值在于用动态分辨率保留视觉细节，再用 MoE 控制多模态 token 的计算成本。

---

## 参考资料

建议阅读顺序：先看 DeepSeek-VL2 README，理解模型能力和使用方式；再看 Hugging Face 模型卡，确认模型版本、输入格式和限制；然后看 `modeling_deepseek_vl_v2.py` 与 `siglip_vit.py`，把结构对应到代码；最后回到 DeepSeekMoE 论文和仓库理解专家路由思想。

| 资料 | 链接 | 建议关注点 |
|---|---|---|
| DeepSeek-VL2 官方仓库 README | https://github.com/deepseek-ai/DeepSeek-VL2 | 模型整体介绍、使用方式、动态分辨率说明 |
| DeepSeek-VL2 模型卡 | https://huggingface.co/deepseek-ai/deepseek-vl2 | 模型版本、能力边界、推理示例 |
| `modeling_deepseek_vl_v2.py` | https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/modeling_deepseek_vl_v2.py | 主模型、多模态 token 组织、MoE 调用 |
| `siglip_vit.py` | https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/siglip_vit.py | 视觉编码器实现 |
| DeepSeekMoE 官方仓库 README | https://github.com/deepseek-ai/DeepSeek-MoE | MoE 设计思想和训练背景 |
| DeepSeekMoE 论文页 | https://huggingface.co/papers/2401.06066 | 专家路由、共享专家、稀疏激活机制 |
