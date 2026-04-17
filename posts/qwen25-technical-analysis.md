## 核心结论

Qwen2.5 可以概括为三件事同时推进的一代模型族：更大的高质量预训练数据、更明确的架构取舍、更工程化的对齐流程。公开技术报告给出的核心信息是，Qwen2.5 通用模型把预训练数据从上一代的 7T token 扩到 18T token，在保持 decoder-only Transformer 主干不变的前提下，继续使用 RoPE、SwiGLU、RMSNorm，并在不同参数规模上组合使用 Tie Embedding 和 GQA，以同时兼顾能力、显存和长上下文效率。

这里先解释几个术语。`decoder-only Transformer` 就是“只根据前文预测下一个 token 的语言模型结构”；`RoPE` 是“把位置信息旋转进注意力向量里的位置编码”；`SwiGLU` 是“比普通前馈层更强的一种门控激活”；`GQA` 是“多组 Query 共享较少的 Key/Value 头，以减少缓存开销”；`Tie Embedding` 是“输入词嵌入和输出词表投影共用同一套参数”。

对初学者最重要的结论不是“Qwen2.5 换了什么名词”，而是“这些改动分别解决了什么问题”：

| 维度 | Qwen2.5 的做法 | 直接作用 |
| --- | --- | --- |
| 预训练 | 18T 高质量 token | 提升常识覆盖、专业知识密度与泛化 |
| 主干结构 | RoPE + SwiGLU + decoder-only | 维持成熟训练范式，降低架构风险 |
| 词表与嵌入 | 扩展词表，部分规模采用 Tie Embedding | 降低参数量或提升多语覆盖 |
| 注意力 | GQA，如 7B 为 28 个 Q 头 / 4 个 KV 头 | 降低 KV cache 显存，利于长上下文 |
| 后训练 | SFT + 偏好优化 + 多阶段 RL | 让“能回答”进一步变成“回答更符合人类偏好” |

一个新手版直觉是：如果你希望模型在 32K 甚至更长对话里继续稳定回答，单纯把参数做大不够，还得同时控制词表、缓存、后训练偏好信号。Qwen2.5 的价值就在于它不是只堆数据，而是把“数据质量、缓存效率、偏好学习”一起做了。

---

## 问题定义与边界

Qwen2.5 试图解决的问题不是单点任务，而是通用大模型的三重目标：第一，广泛指令理解；第二，较长上下文处理；第三，输出更符合人类偏好。白话讲，就是“看得懂更多输入、记得住更长上下文、说得更像人真正想要的答案”。

这三个目标之间天然有冲突。

| 范围 | 目标 | 主要限制 |
| --- | --- | --- |
| 预训练 | 学到语言、知识、模式 | 数据质量决定上限，低质网页会稀释有效信号 |
| 长上下文 | 支持 32K 到 128K | 注意力计算和 KV cache 会快速吃掉显存 |
| 指令跟随 | 按要求输出结构化结果 | 仅做预训练时，模型未必理解“什么回答更好” |
| 偏好对齐 | 更安全、更有帮助 | 过拟合偏好数据会损伤泛化，奖励设计也可能失真 |

所以边界非常明确。Qwen2.5 不是“无限上下文、无限知识、无限稳定”的模型，它仍受三类约束：

1. 计算约束：上下文越长，缓存越大，吞吐越低。
2. 数据约束：18T token 很大，但如果筛选策略不好，大规模只会放大噪声。
3. 对齐约束：只训练语言建模损失，模型可能“会说但不懂偏好”；只做强偏好优化，模型又可能“讨好但失真”。

玩具例子很容易说明这个边界。给模型一个问题：“输出一个 JSON，字段必须是 `name` 和 `score`。”如果模型只做预训练，它可能能解释 JSON 是什么，却不一定严格输出合法 JSON；加入 SFT 后，它更容易学会格式；加入偏好优化后，它更容易学会“哪种回答更被人类接受”。这就是 `CE + 对齐损失` 的分工，而不是谁替代谁。

---

## 核心机制与推导

Qwen2.5 的底层训练仍然建立在标准自回归语言建模上。核心损失是交叉熵，意思是“让真实下一个 token 的概率尽量高”：

$$
\mathcal{L}_{\rm CE}=-\sum_{t=1}^{T}\log P(x_t\mid x_{<t};\theta)
$$

这里的 $\theta$ 是模型参数，$x_t$ 是第 $t$ 个 token。这个式子本质上只回答一个问题：在已有前文下，下一个 token 应该是什么。它保证的是连贯性、语言性和知识记忆，不直接编码“人更喜欢哪个回答”。

这就是为什么后训练要引入偏好优化。DPO 的全称是 Direct Preference Optimization，白话解释是“直接用优选答案和劣选答案做比较，不再单独训练一个显式奖励模型”。其关键量可以写成：

$$
\Delta(x,y)=\beta\log \frac{\pi_\theta(y\mid x)}{\pi_{\rm ref}(y\mid x)}
$$

它表示当前策略相对参考策略，对回答 $y$ 的偏好强度。$\beta$ 是缩放系数，用来平衡“多激进地偏离参考模型”。

对于同一个提示词 $x$，如果有优选回答 $y_+$ 和劣选回答 $y_-$，则 DPO 损失可写为：

$$
\mathcal{L}_{\rm DPO}
=
-\log \sigma\left(\Delta(x,y_+)-\Delta(x,y_-)\right)
$$

其中 $\sigma(z)=\frac{1}{1+e^{-z}}$ 是 sigmoid。这个式子很好理解：只要优选回答的相对得分高于劣选回答，loss 就下降；如果反过来，loss 就会上升并推动模型纠偏。

用一个玩具例子看数值最直观。若
$$
\Delta(x,y_+)-\Delta(x,y_-)=1.2
$$
则
$$
\mathcal{L}_{\rm DPO}\approx -\log \sigma(1.2)\approx 0.26
$$
说明排序已经比较对。若这个差值变成 $-0.5$，则
$$
\mathcal{L}_{\rm DPO}\approx -\log \sigma(-0.5)\approx 0.97
$$
说明模型更偏向劣选回答，需要继续训练。

真实工程例子可以看代码模型。Qwen2.5-Coder 的公开报告强调继续预训练约 5.5T token，并结合 SFT、DPO、GRPO 一类流程来优化代码生成、修复和推理。原因很直接：代码任务不是“像人说话”那么简单，它要求语法正确、测试通过、补全稳定。只用 CE 会学到“像代码的文本”，但不一定学到“更可能通过单元测试的代码”。偏好或强化式后训练的价值就在这里。

---

## 代码实现

下面用一个最小可运行的 Python 例子，把 DPO 数值和 GQA 的缓存收益算清楚。`KV cache` 可以理解为“生成阶段把旧 token 的 Key/Value 存起来，下个 token 直接复用”。

```python
import math

def dpo_loss(delta_pos_minus_neg: float) -> float:
    # L = -log(sigmoid(delta))
    return -math.log(1.0 / (1.0 + math.exp(-delta_pos_minus_neg)))

good = dpo_loss(1.2)
bad = dpo_loss(-0.5)

assert round(good, 2) == 0.26
assert round(bad, 2) == 0.97
assert good < bad

def kv_cache_units(seq_len: int, kv_heads: int, head_dim: int, layers: int) -> int:
    # 这里只比较相对规模，忽略 batch 和 dtype 常数
    # K 和 V 各存一份，所以乘 2
    return seq_len * kv_heads * head_dim * layers * 2

seq_len = 32768
head_dim = 128
layers = 28

# 7B 风格：28 个 Q 头，但只有 4 个 KV 头
gqa_cache = kv_cache_units(seq_len, kv_heads=4, head_dim=head_dim, layers=layers)

# 同样 28 个注意力头，如果不用 GQA，而是每个头都存 KV
mha_cache = kv_cache_units(seq_len, kv_heads=28, head_dim=head_dim, layers=layers)

assert gqa_cache * 7 == mha_cache
print("DPO loss(good) =", round(good, 4))
print("DPO loss(bad) =", round(bad, 4))
print("GQA cache ratio =", mha_cache / gqa_cache)
```

上面最后一个 `assert` 表示，在“28 个 Query 头、4 个 KV 头”的配置下，KV cache 相比传统 28 个 KV 头的 MHA 理论上缩小到原来的 $1/7$。这不是完整显存开销的全部，但它准确说明了 GQA 为什么对长上下文非常重要。

Tie Embedding 的实现也可以用极简伪代码表示：

```python
# vocab_embed: [vocab_size, hidden_dim]
# 输入时查表，输出时直接复用同一矩阵转回词表空间

def embed(input_ids, vocab_embed):
    return vocab_embed[input_ids]

def lm_head(hidden_states, vocab_embed):
    # 共享参数，避免额外再学一套输出投影
    return hidden_states @ vocab_embed.T
```

白话讲，输入层“把 token 变成向量”，输出层“把向量映回 token 概率”，如果两边共享一套参数，就能省参数，也让输入输出空间更一致。Qwen2.5 小模型公开配置中可以看到 `tie_word_embeddings=true`，而 7B 及以上公开模型则常见为 `false`。这说明 Tie Embedding 不是绝对更好，而是参数规模和工程目标下的一种取舍。

还要特别指出一个容易混淆的点：很多二手资料会把 `151643` 说成词表大小，但公开配置里它常常是 `bos_token_id`。Qwen2.5 开源配置里常见的 `vocab_size` 是 `151936` 或 `152064`，取决于具体模型版本与规模。工程上应以实际 `config.json` 为准，而不是拿单个 token id 误当词表规模。

---

## 工程权衡与常见坑

Qwen2.5 的训练难点不在“会不会写 Transformer”，而在“如何让数据、缓存、偏好三条链路同时稳定”。

| 阶段 | 主要收益 | 常见坑 |
| --- | --- | --- |
| 大规模预训练 | 获取知识和语言模式 | 低质量网页、重复语料、测试泄漏会污染能力评估 |
| SFT | 学会指令格式和任务模板 | prompt 覆盖不足，容易让模型只会少数表达套路 |
| DPO | 直接优化优选/劣选排序 | 偏好对数据分布敏感，seen prompt 复用过多会过拟合 |
| RL/GRPO | 提升任务成功率和复杂行为 | 奖励高方差导致训练不稳，可能出现 reward hacking |

第一个常见坑是把“18T token”理解成“只要量大就行”。不是。高质量语料的价值通常高于低质量语料的简单堆叠。尤其对代码和数学模型，去重、去污染、合成数据混合比例，往往比单纯多喂一些网页更重要。Qwen2.5-Coder 公开资料强调继续预训练、数据清洗和 benchmark decontamination，本质就是在避免“模型只是背过测试题”。

第二个常见坑是把 DPO 当成可以完全替代 SFT。也不是。SFT 负责建立“基础回答形状”，DPO 负责建立“偏好排序方向”。如果没有足够好的 SFT 底座，DPO 只是拿一堆相对偏好去微调概率，训练会更脆弱。

第三个常见坑是只看 PPO 或 RLHF 这个名字，不看奖励质量。传统 PPO 类 RLHF 的问题在于奖励噪声大、训练不稳定、实现成本高。DPO 的流行，本质上是把偏好学习变成更直接的监督式目标，降低了系统复杂度。再往后接 GRPO 一类方法，通常是为了在特定领域任务上继续拉升结果，例如代码测试通过率或数学答案正确率。

真实工程例子是代码评审助手。你希望它根据“指令 + 代码差异 + 单元测试结果”给出修改建议。只做 SFT，模型可能能写出像样评论，但不一定最关注真正会导致测试失败的问题。加入 DPO 后，可以用“专家更偏好的评审意见”作为训练对；再加入在线 RL 或基于结果的优化后，可以把“是否减少失败用例”纳入奖励。这样得到的模型才更接近生产环境需要的能力。

---

## 替代方案与适用边界

Qwen2.5 不是唯一方案，它是一组针对大规模通用模型较平衡的方案。资源不同，取舍也应该不同。

| 方案 | 优点 | 代价 | 适用情境 |
| --- | --- | --- | --- |
| 传统 MHA | 实现直接，概念简单 | KV cache 大，长上下文贵 | 小上下文、研究原型 |
| GQA | 显著减少缓存开销 | 头部共享带来一定表达约束 | 需要 32K/128K 上下文 |
| Tie Embedding | 降参数、输入输出空间一致 | 可能限制部分表达灵活性 | 小中型模型、参数预算紧张 |
| Untied Embedding | 输出层更自由 | 参数更多 | 大模型、追求上限 |
| 仅 SFT | 工程简单 | 偏好控制弱 | 小模型、低预算场景 |
| SFT + DPO | 稳定且实现成本较低 | 需要优劣配对数据 | 通用对话、评审、问答 |
| SFT + PPO/GRPO | 能利用结果型奖励 | 系统复杂、训练更难 | 代码、数学、工具调用等可验证任务 |

对初学者可以给出一条实用边界线：如果你在做 6B 以下的小模型，第一阶段先把 SFT 做好，比急着上复杂 RL 更重要；如果你的任务有明确正确性信号，比如代码测试、数学判题、SQL 执行结果，那么再考虑 DPO 之后接结果驱动优化，收益更明确。

还有一个边界必须说清楚：缺少 GQA 或类似缓存优化时，长上下文体验会迅速恶化；缺少足够好的词表与多语数据时，多语言指令能力会下降；缺少偏好优化时，模型可能“会续写”，但不一定“会回答”。所以 Qwen2.5 的成功不是某一个模块特别神，而是多个工程细节没有明显短板。

---

## 参考资料

| 标题 | 出处 | 重点摘要 |
| --- | --- | --- |
| Qwen2.5 Technical Report | arXiv:2412.15115, https://arxiv.org/abs/2412.15115 | 通用 Qwen2.5 系列的 18T 预训练、后训练和整体评测 |
| Qwen2.5 官方模型页 | Qwen Blog, https://qwen2.org/qwen2-5/ | 各参数规模的层数、GQA 头数、是否 Tie Embedding、上下文长度 |
| Qwen2.5-0.5B-Instruct Config | Hugging Face, https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json | 小模型公开配置，含 `tie_word_embeddings=true`、`vocab_size=151936` |
| Qwen2.5-32B Config | Hugging Face, https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json | 大模型公开配置，含 `num_attention_heads=40`、`num_key_value_heads=8`、`tie_word_embeddings=false` |
| Qwen2.5-Coder Technical Report | arXiv:2409.12186, https://arxiv.org/abs/2409.12186 | 代码专项模型的 5.5T 继续预训练、数据清洗与专项评测 |
| Qwen2.5-Math Technical Report | arXiv:2409.12122, https://arxiv.org/abs/2409.12122 | 数学专项模型的训练策略与数学推理增强 |
| Direct Alignment | RLHF Book, https://rlhfbook.com/c/08-direct-alignment | DPO 的 Bradley-Terry 推导、损失形式与直观解释 |
