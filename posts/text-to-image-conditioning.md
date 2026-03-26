## 核心结论

文生图里的“条件机制”，本质是把文本变成模型可消费的控制信号。这里的“条件”可以先白话理解为：模型生成图片时参考的说明书。现代扩散模型已经不满足于单一路说明书，而是把不同类型的文本编码器组合起来，让它们分别处理短关键词、长叙述和全局风格。

CLIP 文本编码器的强项是抓关键词、风格词和常见视觉概念，但它常见的 77 token 上限决定了它不适合承载长提示词。token 可以先白话理解为：模型内部切分文本的最小处理单元，不一定等于一个汉字或一个单词。提示词一长，CLIP 后半段信息就会被截断，结果通常是画面里局部细节存在，但整体叙事断裂。

T5 一类文本编码器的价值在于长程语义理解。长程语义可以先白话理解为：模型能同时记住并关联更长句子里前后相隔很远的条件。它更适合承载“主体是谁、动作是什么、场景怎样、前后约束如何”这类复杂描述。SD3、Flux 这类架构把 CLIP 和 T5 一起使用，就是为了把“短而强的视觉关键词”与“长而完整的语言逻辑”拆开处理。

更关键的是，文本送进模型后，不是只有一种注入路径。cross-attention 负责把序列级文本条件注入到图像特征中；adaLN 负责用一个全局向量调节网络各层的整体风格和统计分布。前者更像逐像素查词典，后者更像给整张图定基调。

| 方案 | 长度能力 | 擅长内容 | 常见注入方式 | 典型问题 |
| --- | --- | --- | --- | --- |
| 单 CLIP | 短 | 风格词、视觉关键词 | cross-attention | 长提示被截断 |
| 双 CLIP | 中等 | 扩展视觉表达 | cross-attention + pooled conditioning | 逻辑链仍有限 |
| CLIP + T5 | 长 | 关键词 + 长叙述 | cross-attention + adaLN | 算力和显存更高 |
| 长上下文 CLIP | 较长 | 尝试兼顾两者 | 多数仍以 attention 注入为主 | 生态和兼容性不如主流 |

玩具例子：prompt 写成“蒸汽朋克机械少女，铜色齿轮肩甲，夜空中漂浮发光诗句，远景有城市天际线，镜头低角度，情绪宁静但带压迫感”。如果只给 CLIP，模型更可能抓住“蒸汽朋克、机械、夜空”这类前部关键词；如果把完整句子交给 T5，再把简洁风格词留给 CLIP，画面会更容易同时保住“风格”和“故事线”。

---

## 问题定义与边界

本文讨论的是扩散模型中的文本条件机制，重点不是图像编码器，也不是训练损失，而是“文本如何被编码，并通过什么路径进入去噪网络”。去噪网络可以先白话理解为：扩散模型里逐步把噪声还原成图像的主干网络，常见实现是 U-Net 或其变体。

问题可以拆成两个子问题：

1. 文本编码器能表达多少信息。
2. 这些信息怎样注入生成网络。

第一个问题的边界，常用长度可写成：

$$
L_{clip}=77,\quad L_{t5}=512
$$

这不是说所有 77 个 token 都同样有效。实际工程中，长 prompt 经常出现“前段词更强、后段词衰减甚至丢失”的现象。因此不能把 CLIP 当成“任意长文本理解器”。它更接近一个高效但上下文短的视觉语义接口。

第二个问题的边界，是局部语义与全局调性要分开看。局部语义可以先白话理解为：某个区域该出现什么对象、姿态、材质。全局调性可以先白话理解为：整张图的风格、光照、色温、镜头气质是否统一。只解决前者，容易得到“细节对，但整体味道不对”；只解决后者，容易得到“氛围对，但对象关系乱”。

真实工程例子：一个电商海报生成系统要输出“浅色背景上的玻璃瓶护肤品，带露珠，晨光逆光，广告摄影，左侧留白用于文案，整体高级、干净、不过度饱和”。这里“玻璃瓶、露珠、留白”是局部与布局条件，“高级、干净、不过度饱和”更偏全局调性。把所有描述挤进单 CLIP，经常会出现瓶子有了，但色调不稳定，或者留白被忽略。

因此，本文的边界是：讨论 CLIP、T5、双编码器、cross-attention、adaLN 之间的协作关系，不展开 VAE、采样器、LoRA 等其他模块。

---

## 核心机制与推导

先看 cross-attention。attention 可以先白话理解为：模型在当前生成位置，去文本说明书里查“此处该参考哪些词”。在文生图里，query 来自图像侧特征，key/value 来自文本编码器输出。简化写法是：

$$
\text{attn}(q,k,v)=\text{softmax}\left(\frac{qk^\top}{\sqrt{d}}\right)v
$$

当模型使用多路文本编码器时，常见做法是把它们的序列输出拼接后作为 key/value：

$$
k,v=[e_{clip}^{(L)};e_{clip}^{(G)};e_{t5}]
$$

其中 $e_{clip}^{(L)}$、$e_{clip}^{(G)}$ 可以表示两路 CLIP 输出，$e_{t5}$ 表示 T5 输出。拼接的含义不是“谁替代谁”，而是让去噪网络在每一层、每个位置都可以同时查询多种文本表示。

如果一个 prompt 有 140 个 token，而系统采用两路 CLIP 各 77 长度、一路 T5 保留 140 长度，那么 cross-attention 可见的序列长度可近似写成：

$$
77 + 77 + 140 = 294
$$

这就是为什么双编码器或三路编码器会显著改善长提示的表达完整性。不是某个编码器突然变聪明了，而是模型可查询的条件上下文变长了，且语义来源变多了。

再看 adaLN。LayerNorm 可以先白话理解为：把一层特征先标准化，防止数值分布漂移。AdaLN 的“Adaptive”表示标准化后的缩放和偏移不再固定，而是由条件向量动态生成：

$$
\text{AdaLN}(x)=\gamma(\bar{e}_{clip}^{(G)})\cdot \text{LN}(x)+\beta(\bar{e}_{clip}^{(G)})
$$

这里 $\bar{e}_{clip}^{(G)}$ 是 pooled 向量。pooled 向量可以先白话理解为：把整段文本压成一个全局摘要。它不负责逐词定位，更适合表达“整体该是什么风格”。

为什么常见实现偏向用 CLIP pooled 向量驱动 adaLN，而不是直接用 T5 全序列？原因很实际：

| 条件类型 | 更适合的来源 | 原因 |
| --- | --- | --- |
| 局部对象关系 | T5 序列、CLIP 序列 | 需要逐 token 查询 |
| 风格、色调、镜头气质 | CLIP pooled | 需要全局摘要向量 |
| 层间调制参数 | pooled conditioning | 维度固定，易投影成 scale/shift |

玩具例子：prompt 为“红色雨伞下站着一个穿灰大衣的人，背景是模糊霓虹灯街道，整体冷色、电影感、下雨夜晚”。cross-attention 更像在问：“这一块像素要不要对应‘雨伞’、‘人’、‘霓虹灯’？”；adaLN 更像在说：“整张图都往冷色、电影感、雨夜方向调。”

如果只有 cross-attention，没有 adaLN，模型可能把“雨伞、人物、街道”都画出来，但整体不一定真的冷色、克制、电影化。反过来，如果只有 adaLN，没有足够强的 token 级条件，图可能氛围对了，但对象细节混乱。

---

## 代码实现

工程实现的核心不是“多写一个 encoder”，而是明确三步：文本分流、序列拼接、全局调制。下面给一个可运行的 Python 玩具实现，用来模拟 CLIP 截断、T5 保留和 adaLN 参数生成。

```python
from typing import List, Dict

CLIP_MAX_LEN = 77
T5_MAX_LEN = 512

def tokenize(prompt: str) -> List[str]:
    return prompt.strip().split()

def route_prompt(prompt: str) -> Dict[str, List[str]]:
    tokens = tokenize(prompt)
    clip_tokens = tokens[:CLIP_MAX_LEN]
    t5_tokens = tokens[:T5_MAX_LEN]
    return {"clip": clip_tokens, "t5": t5_tokens}

def fake_clip_pool(tokens: List[str]) -> Dict[str, float]:
    # 用长度和关键词计数模拟 pooled conditioning
    style_words = {"cinematic", "soft", "moody", "vibrant", "dramatic"}
    style_score = sum(1 for t in tokens if t.lower() in style_words)
    scale = 1.0 + 0.1 * style_score
    shift = 0.05 * len(tokens[:10])
    return {"scale": scale, "shift": shift}

def build_condition(prompt: str) -> Dict[str, object]:
    routed = route_prompt(prompt)
    clip_pool = fake_clip_pool(routed["clip"])
    cross_attn_len = len(routed["clip"]) + len(routed["t5"])
    return {
        "clip_len": len(routed["clip"]),
        "t5_len": len(routed["t5"]),
        "cross_attn_len": cross_attn_len,
        "adaln": clip_pool,
    }

prompt = " ".join(["subject"] * 100 + ["cinematic", "moody", "city", "night"])
cond = build_condition(prompt)

assert cond["clip_len"] == 77
assert cond["t5_len"] == 104
assert cond["cross_attn_len"] == 181
assert cond["adaln"]["scale"] >= 1.0
```

上面代码没有真的跑神经网络，但已经把关键结构表达清楚：

1. `clip` 端截断到 77。
2. `t5` 端保留更长上下文。
3. `clip_pool` 生成类似 adaLN 的全局调制参数。
4. cross-attention 的可见长度来自多路条件的组合。

如果写成更接近真实框架的伪代码，结构通常是这样：

```python
clip_l_emb = clip_l_encoder(prompt[:77])
clip_g_emb = clip_g_encoder(prompt[:77])
clip_g_pool = pool(clip_g_emb)

t5_emb = t5_encoder(prompt[:512])

cross_attn_inputs = concat([clip_l_emb, clip_g_emb, t5_emb], dim=1)

for block in unet_blocks:
    x = block.cross_attn(x, context=cross_attn_inputs)
    scale, shift = mlp(clip_g_pool)
    x = ada_layer_norm(x, scale=scale, shift=shift)
```

真实工程例子：在 SDXL 中，常见是双 CLIP 文本编码器协同，一路偏基础视觉语义，一路偏更强表达容量；在 Flux 或 SD3 一类设计中，还会引入 T5，形成“短上下文视觉语义 + 长上下文语言语义”的组合。实现重点不是名字，而是接口约束：

| 模块 | 输入 | 输出 | 主要用途 |
| --- | --- | --- | --- |
| CLIP-L | 短 prompt | 序列 embedding | 基础视觉关键词 |
| CLIP-G | 短 prompt | 序列 + pooled | 更强概念表达、全局摘要 |
| T5 | 长 prompt | 长序列 embedding | 复杂逻辑与长程语义 |
| U-Net attention | 图像特征 + 文本序列 | 条件化特征 | 局部内容对齐 |
| AdaLN MLP | pooled 向量 | scale/shift | 全局风格调制 |

---

## 工程权衡与常见坑

第一类坑是长度错配。很多初学者以为“prompt 写得越长越好”，但如果底层主编码器还是 CLIP 77 token，后半段根本进不去。正确做法不是盲目堆词，而是先区分：

$$
prompt_{clip}=prompt[:77], \quad prompt_{t5}=prompt[:512]
$$

也就是把强风格词、主体词、关键构图词前置给 CLIP，把完整叙述交给 T5。

第二类坑是条件失衡。负面提示词可以先白话理解为：告诉模型不要出现什么的反向条件。如果多路 encoder 都在吃 prompt，但正负条件没有同步分配，就会出现某一路过强、某一路过弱，最终表现为“局部很听话，但整体跑偏”。

第三类坑是过度依赖 cross-attention。它非常适合做局部控制，但如果想稳住整图风格，只有 token 级注入通常不够。特别是广告图、统一 IP 设定图、批量生成风格一致素材时，缺少 pooled conditioning 或 adaLN，常见问题是亮度、饱和度、镜头语言每张都飘。

| 常见坑 | 现象 | 原因 | 规避方式 |
| --- | --- | --- | --- |
| CLIP token overflow | prompt 后段失效 | 超过 77 token 截断 | 关键词前置，长叙述分流到 T5 |
| 只堆风格词 | 图有氛围但结构乱 | 缺少长程逻辑 | 用 T5 承载关系描述 |
| 只做 cross-attention | 全局色调不稳 | 没有全局调制 | 用 pooled 向量驱动 adaLN |
| 多 encoder 不平衡 | 某些条件被放大 | 正负 prompt 分配失衡 | 保持各路 prompt/negative 对称 |
| 序列过长 | 推理变慢、显存涨 | attention 成本上升 | 做长度预算和条件裁剪 |

一个工程上常见的优化策略是“条件分层”：

1. CLIP：主体、风格、镜头、材质关键词。
2. T5：完整句子、关系、场景逻辑、限制条件。
3. pooled conditioning：全局气质，如“clean studio lighting”“muted colors”。

这样做的本质，是让不同路径只负责自己最擅长的信息类型，而不是让一个编码器承担全部职责。

---

## 替代方案与适用边界

不是所有场景都必须上 CLIP + T5。方案选择取决于 prompt 长度、风格一致性要求、推理延迟预算和工程复杂度。

如果 prompt 本来就短，比如“white ceramic mug on wooden table, soft morning light”，单 CLIP 或双 CLIP 通常已经够用。这里再接 T5，收益可能小于额外的显存与延迟成本。

如果任务是小说式长描述、角色设定一致、多对象关系复杂，那么单 CLIP 几乎一定会吃力。这时 CLIP + T5 的收益明显，因为长程语义不再依赖前 77 token 的挤压表达。

如果项目最关心的是风格统一，比如品牌 KV、角色海报批量生产、游戏立绘统一画风，那么 adaLN 或类似 pooled conditioning 很重要。它不替代 cross-attention，但能显著降低“内容对了，味道不对”的波动。

| 方案 | 适用 prompt 长度 | 风格一致性需求 | 延迟/显存 | 适用场景 |
| --- | --- | --- | --- | --- |
| 单 CLIP | 短 | 低到中 | 低 | 快速原型、简单生成 |
| 双 CLIP | 短到中 | 中 | 中 | SDXL 类高质量短 prompt |
| CLIP + T5 | 中到长 | 中到高 | 高 | 复杂描述、叙事型生成 |
| 长上下文 CLIP | 中到长 | 中 | 中到高 | 想减少多编码器复杂度 |
| 强化 adaLN 的多编码器 | 任意 | 高 | 高 | 品牌风格统一、批量生产 |

可以用一个简单准则判断：

- 当 $L \leq L_{clip}$，优先考虑单 CLIP 或双 CLIP。
- 当 $L_{clip} < L \leq L_{t5}$，优先考虑 CLIP + T5。
- 当任务对全局风格一致性要求高时，即使 $L \leq L_{clip}$，也值得加入 pooled conditioning 或 adaLN。
- 当延迟是硬约束时，宁可缩短 prompt，也不要无上限堆 encoder。

玩具例子：40 token 的商品图 prompt，用双 CLIP 足够。真实工程例子：一段 200 token 的影视分镜描述，希望人物关系、动作顺序、镜头语言都保留，这时 T5 几乎是必要项。

---

## 参考资料

| 资料 | 重点内容 | 对应章节 | URL |
| --- | --- | --- | --- |
| Long-CLIP 相关论文解读 | 说明 CLIP 的 77 token 限制及其长文本短板 | 核心结论、问题定义 | https://www.emergentmind.com/papers/2403.15378 |
| SDXL pipeline 文档解读 | 双 CLIP 编码器、hidden states 拼接、pooled conditioning | 核心机制、代码实现 | https://deepwiki.com/huggingface/diffusers/3.4-sdxl-pipelines |
| Flux pipeline 文档 | CLIP 与 T5 双编码器接口设计 | 代码实现、工程权衡 | https://huggingface.co/docs/diffusers/main/api/pipelines/flux |
| SD3 / Flux 架构总结 | 多编码器策略与 token 分工 | 核心结论、替代方案 | https://blog.sotaaz.com/post/sd3-flux-architecture-en |
| Flux 技术博客解读 | CLIP 处理关键词、T5 处理长提示的直观说明 | 问题定义、核心机制 | https://fluxai.dev/blog/tutorial/2024-09-16-how-flux-ai-uses-clip-and-t5-to-parse-prompts |
| AdaLN 相关文章 | AdaLN 与 cross-attention 的角色差异 | 核心机制、工程权衡 | https://zenn.dev/fuwamoekissaten/articles/df428a9c8ac2bd |

这些资料组合起来，分别支撑三件事：一是 CLIP 的长度边界真实存在；二是 SDXL、SD3、Flux 等主流路线确实在用多编码器；三是 token 级条件注入和全局调制不是一回事，cross-attention 与 adaLN 的职责应分开理解。
