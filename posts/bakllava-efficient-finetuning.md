## 核心结论

BakLLaVA 的核心不是重新发明视觉语言模型架构，而是把 `Mistral-7B` 作为语言骨干，接上 `LLaVA-1.5` 使用的视觉编码器和 `2-layer MLP` 连接器。视觉语言模型是能同时读取图像和文本，并用自然语言回答问题的模型。

它的主要价值在三点：训练更省、推理更轻、部署更容易。公开材料显示，BakLLaVA-1 使用 Mistral 7B base 加 LLaVA 1.5 架构，并在若干基准上展示了 7B 语言底座相对 Llama 2 13B 的竞争力。这里的重点不是“图像输入变少”，而是语言底座换成了更高效的 Mistral，使同等多模态训练预算更容易得到可用效果。

对新手可以这样理解：13B 级 LLaVA 像更大的发动机，7B 级 BakLLaVA 像更强的变速箱加更省油的引擎。这个说法只是直观解释，技术上真正发生的是语言模型参数规模下降，同时 Mistral 的注意力结构提高了推理效率。

| 模型 | 语言底座 | 视觉编码器 | 连接器 | 参数规模 | 推理代价 | 效果定位 |
|---|---|---|---|---:|---|---|
| BakLLaVA-1 | Mistral-7B | LLaVA-1.5 路线，常见为 CLIP ViT-L/14 336px | 2-layer MLP | 7B 级 | 较低 | 轻量多模态对话 |
| LLaVA-1.5-7B | Vicuna-7B | CLIP ViT-L/14 336px | 2-layer MLP | 7B 级 | 中等 | 通用开源基线 |
| LLaVA-1.5-13B | Vicuna-13B | CLIP ViT-L/14 336px | 2-layer MLP | 13B 级 | 较高 | 更强但更重 |
| 更大 LLaMA/Vicuna 基线 | LLaMA/Vicuna 13B+ | 依实现而定 | 依实现而定 | 13B+ | 高 | 预算充足场景 |

---

## 问题定义与边界

BakLLaVA 要解决的问题是：输入一张图像和一段文本后，模型在统一的语言空间里完成理解、问答、推理和指令跟随。语言空间是指模型最终用文本 token 表达信息的内部表示空间；token 是模型处理文本时使用的最小片段，可以是词、子词或符号。

这不同于传统图像分类。图像分类只需要输出“猫”“狗”“发票”这类固定标签；视觉语言模型要回答开放问题。例如用户上传一张报销单截图，问“总金额是多少”，模型需要先看图，再把图中区域、文字、布局转换成可被语言模型处理的视觉 token，最后生成答案。

BakLLaVA 的边界也要说清楚：它不是更强的视觉编码器模型，而是用更合适的语言底座重做视觉语言对齐。视觉语言对齐是指让图像特征和文本特征落到可相互理解的表示空间里。本文中“更少数据达到相同性能”是基于公开材料和工程经验的归纳，不是 BakLLaVA 官方给出的严格消融实验结论。

| 任务类型 | BakLLaVA 能处理 | BakLLaVA 不擅长 | 原因 |
|---|---|---|---|
| 图文问答 | 能回答图片内容相关问题 | 不保证每个细节都准确 | 依赖视觉 token 和语言推理 |
| 截图问答 | 适合做界面、表单、报销单理解 | 不适合作为高精度 OCR 唯一来源 | 小字、密集表格容易错读 |
| 文档理解 | 可做摘要、字段解释、粗粒度提取 | 不适合强审计级字段抽取 | 缺少确定性校验 |
| 纯视觉检测 | 可描述目标 | 不擅长输出精确框坐标 | 架构目标不是检测或分割 |
| 多步业务流程 | 可辅助判断 | 不应单独执行关键决策 | 需要检索、规则和权限系统配合 |

玩具例子：图片里有一个红色按钮，旁边写着 `Submit`。用户问“这个按钮是做什么的？”BakLLaVA 的目标回答是“这是提交按钮”，而不是只输出“红色”或“按钮”。

真实工程例子：内部客服系统接入截图问答。客服上传用户后台截图，问“这个订单为什么不能退款？”模型先读界面上的状态、按钮和错误提示，再结合 prompt 输出解释。此时 BakLLaVA 负责视觉理解和语言回答，业务规则仍应由后端系统校验。

---

## 核心机制与推导

BakLLaVA 沿用 LLaVA 类模型的主干流程：图像先进入视觉编码器，得到视觉特征；视觉特征经过 projector 映射到语言模型可接收的维度；文本 token 和视觉 token 一起送入 Mistral；Mistral 按自回归方式生成答案。自回归是指模型每次根据已有 token 预测下一个 token。

公式可以写成：

$$
v = f_v(I), \quad z = g_\theta(v), \quad L = - \sum_t \log p(y_t \mid y_{<t}, x, z)
$$

其中，$I$ 是输入图像，$f_v$ 是视觉编码器，$v$ 是图像特征，$g_\theta$ 是 projector，$z$ 是投影后的视觉 token，$x$ 是用户输入文本，$y_t$ 是第 $t$ 个目标输出 token，$p(\cdot)$ 是语言模型给出的生成概率，$L$ 是语言模型训练损失。

| 步骤 | 输入 | 模块 | 输出 | 作用 |
|---|---|---|---|---|
| 1 | 图像 `I` | 图像预处理 | 标准尺寸图像 | 对齐视觉编码器输入格式 |
| 2 | 标准图像 | 视觉编码器 | 视觉特征 `v` | 提取图像语义 |
| 3 | 视觉特征 `v` | projector | 视觉 token `z` | 对齐到语言维度 |
| 4 | `z` + 文本 token `x` | Mistral | 隐状态 | 融合图文信息 |
| 5 | 隐状态 | 语言头 | 回答 token | 逐 token 生成答案 |

BakLLaVA 的结构性变化主要在语言侧：把生成概率 $p(\cdot)$ 背后的语言骨干换成 Mistral-7B。Mistral 使用 GQA 和 SWA。GQA，即 grouped-query attention，是把多个查询头共享较少的键值头，用来减少推理时的 KV cache 压力。SWA，即 sliding window attention，是让注意力主要看固定窗口内的历史 token，从而在长上下文下减少计算和缓存成本。

这解释了 BakLLaVA 的效率来源：视觉侧基本沿用 LLaVA-1.5 的成熟方案，语言侧换成更强、更省的 7B 底座。公开数据量上，BakLLaVA-1 模型卡列出 558K 图文对、158K 多模态指令数据、450K 学术 VQA 混合数据、40K ShareGPT 和额外私有数据。LLaVA-1.5 公开训练说明则包含 558K 对齐数据、150K 指令数据和约 515K VQA 数据。两者都在百万级数据量附近，因此更合理的归纳是：BakLLaVA 借助 Mistral 的语言效率，在相近数据预算下取得更好的成本效果比，而不是靠数量级更大的数据堆出来。

---

## 代码实现

实现 BakLLaVA 类模型时，可以拆成三层：加载视觉编码器、加载语言底座、插入 projector。projector 是连接器，作用是把视觉编码器输出的维度转换成语言模型的隐藏层维度。训练时通常先做特征对齐，再做视觉指令微调；参数更新范围取决于配方，可以只更新 projector，也可以进一步微调语言模型的部分或全部参数。

伪代码如下：

```python
image_features = vision_encoder(image)
image_tokens = projector(image_features)
outputs = language_model(input_ids=text_ids, visual_tokens=image_tokens)
loss = compute_lm_loss(outputs, labels)
```

下面是一个可运行的玩具实现。它不是真实 BakLLaVA，只模拟“视觉特征投影到语言维度”这个接口对齐过程：

```python
import numpy as np

class ToyProjector:
    def __init__(self, vision_dim, language_dim):
        rng = np.random.default_rng(0)
        self.w1 = rng.normal(0, 0.02, size=(vision_dim, language_dim))
        self.w2 = rng.normal(0, 0.02, size=(language_dim, language_dim))

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def __call__(self, vision_features):
        hidden = self.gelu(vision_features @ self.w1)
        return hidden @ self.w2

# 假设视觉编码器输出 4 个视觉 token，每个 token 维度为 3
vision_features = np.array([
    [1.0, 0.0, 0.5],
    [0.2, 0.8, 0.1],
    [0.0, 1.0, 0.3],
    [0.7, 0.1, 0.9],
])

projector = ToyProjector(vision_dim=3, language_dim=8)
image_tokens = projector(vision_features)

assert image_tokens.shape == (4, 8)
assert np.isfinite(image_tokens).all()
```

真实工程中更重要的不是把类写复杂，而是接口一致：tokenizer 要匹配 Mistral，chat template 要匹配训练格式，视觉特征维度要匹配 projector，训练样本里的图片占位符要和模型代码一致。chat template 是把用户、助手、系统消息拼成模型训练时见过的文本格式。

| 阶段 | 输入 | 更新参数 | 目标 | 常见问题 |
|---|---|---|---|---|
| 特征对齐 | 图文对 | projector 为主 | 让视觉 token 能被语言模型理解 | 维度、图像尺寸不匹配 |
| 指令微调 | 图文问答数据 | projector + 部分或全部 LLM | 学会按指令回答 | 模板错、过拟合 |
| 推理 | 图片 + prompt | 不更新 | 生成答案 | tokenizer 或图片占位符错 |
| 领域适配 | 内部截图/文档数据 | LoRA 或少量参数 | 贴近业务数据 | 数据许可和质量问题 |

---

## 工程权衡与常见坑

最常见的问题是 tokenizer、chat template 和底座不匹配。这类错误不一定报错，因为模型仍能生成通顺句子，但它生成的是错位格式下的句子。表现通常是：回答流畅，却总是看不懂图；或者能描述大概场景，但对用户问题答非所问。

第二个高频问题是 projector 版本错误。LLaVA-1.5 官方说明里明确不推荐使用 legacy projector，因为它可能来自不同代码版本，某些选项不一致时模型不会按预期工作。BakLLaVA 这类组合式模型更依赖接口一致性，视觉编码器、projector、语言底座三者任何一处错位都会放大成效果问题。

训练也不是越久越好。多模态微调的目标是保持语言能力，同时建立视觉语言对齐。轮数过多或学习率过高，可能让模型记住训练集的语言模式，却破坏原本的视觉对齐。对初级工程师来说，优先复现公开超参数，再小步调整，比直接加 epoch 更可靠。

| 坑位 | 症状 | 排查方法 | 修复方向 |
|---|---|---|---|
| 模板不匹配 | 回答流畅但不按角色格式输出 | 打印最终 prompt | 使用 Mistral 对应 chat template |
| projector 版本错 | 看图能力明显异常 | 检查 `mm_projector_type` 和 checkpoint 来源 | 使用同一训练配方的 projector |
| 训练过拟合对齐 | 训练集回答变好，验证图像变差 | 分开看语言损失和图文任务指标 | 降低学习率、减少 epoch |
| 许可证风险 | 技术可跑但不能商用 | 检查数据和模型卡许可 | 换可商用数据重新训练 |
| 图像预处理错 | 小字、边缘内容缺失 | 检查 resize、pad、crop 策略 | 尽量复用训练时的图像处理 |

真实工程排错例子：某团队把 BakLLaVA 接入截图问答后，发现模型总能生成很自然的客服话术，但无法回答截图里的金额。排查后发现图片占位符没有按训练格式插入，语言模型实际只看到了用户问题，没有可靠接收到视觉 token。这不是 Mistral 不会说话，也不是视觉编码器完全失效，而是多模态接口断了。

---

## 替代方案与适用边界

BakLLaVA 适合预算受限但需要图文对话能力的场景，例如客服截图问答、表单理解、内部知识助手、低频图片分析。它不等于 OCR、检测、分割或复杂规划系统。OCR 是把图片中文字识别成字符的技术；检测是找出目标类别和位置；分割是给出像素级区域。这些任务需要更确定的视觉输出时，应使用专门模型。

| 方案 | 成本 | 效果 | 部署复杂度 | 适用任务 |
|---|---|---|---|---|
| BakLLaVA | 较低 | 图文对话性价比高 | 中等 | 截图问答、表单解释、轻量助手 |
| LLaVA-1.5 | 中等 | 成熟开源基线 | 中等 | 通用多模态实验和对比 |
| 更大 LLaMA/Vicuna 基线 | 高 | 上限更高 | 较高 | 预算充足的通用助手 |
| 纯 OCR + LLM | 低到中 | 文本字段更稳定 | 中等 | 文档抽取、票据问答 |
| 检测/分割模型 + 规则 | 中到高 | 位置精度高 | 较高 | 工业质检、缺陷定位 |

场景对比很关键。客服截图问答适合 BakLLaVA，因为用户问题通常是开放自然语言，模型需要综合图像布局和文本提示生成解释。工业质检缺陷定位则更适合视觉检测模型加规则系统，因为系统要输出缺陷位置、置信度和可追溯判断，开放式回答反而不够稳定。

更稳的工程做法是组合系统：BakLLaVA 负责理解截图和生成解释，OCR 负责抽取关键文字，检索系统负责查内部文档，规则引擎负责最终业务判断。这样能把多模态模型放在它擅长的位置，而不是让它承担所有确定性工作。

---

## 参考资料

1. [BakLLaVA 官方仓库](https://github.com/SkunkworksAI/BakLLaVA)：查看项目代码、运行方式和与 LLaVA 的继承关系。
2. [BakLLaVA-1 模型卡](https://huggingface.co/SkunkworksAI/BakLLaVA-1)：查看 Mistral 7B base、LLaVA 1.5 architecture、训练数据组成和非商用语料风险说明。
3. [LLaVA 官方仓库](https://github.com/haotian-liu/LLaVA)：查看 LLaVA-1.5 的训练配方、CLIP ViT-L/14 336px、`mlp2x_gelu` projector 和训练脚本。
4. [LLaVA-1.5 论文页](https://huggingface.co/papers/2310.03744)：理解“Improved Baselines with Visual Instruction Tuning”的基线改进背景。
5. [Mistral 7B 官方博客](https://mistral.ai/news/announcing-mistral-7b/)：查看 Mistral-7B 的参数规模、GQA、SWA 和推理效率设计。

阅读顺序建议：先看 BakLLaVA 模型卡，确认它到底组合了哪些模块；再看 BakLLaVA 仓库，理解运行方式；然后回到 LLaVA-1.5 的训练说明，理解视觉编码器和 projector；最后看 Mistral 7B 博客，理解语言底座为什么能带来效率优势。
