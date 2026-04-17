## 核心结论

Gemini 的多模态 tokenizer 可以理解为“把不同来源的信息先切成统一长度管理的离散片段，再送进同一套 Transformer”。这里的 token 是模型处理信息的最小记账单位，白话讲就是“模型一次次看的小片段”。文本 token 来自子词分词，图像 token 来自图像切块后再编码，音频 token 来自时间帧特征，视频 token 则是按时间抽帧后继续做视觉编码。

但有一个边界必须先说清：截至 2026 年 3 月 12 日，Google 官方公开资料并没有完整披露 Gemini 3 Pro 的内部 tokenizer 细节，比如是否确实使用某个具体的 SentencePiece 词表、视觉是否固定为 16×16 patch、音频内部是否直接以 log-mel 帧入模。这些更细的实现，多数只能依据公开 API 的计费规律、早期技术报告和通用多模态 Transformer 设计做工程推断，不能当成官方逐层公开的源码事实。

公开能确认的部分有两类。第一类是外部行为：Gemini API 会把文本、图像、音频、视频统一计入同一个上下文预算；小图常见按 258 token 计，音频按每秒 32 token 计，视频按每秒 263 token 计。第二类是架构方向：Gemini 被官方描述为 natively multimodal，也就是“原生多模态”，白话讲不是先把多模态拆成几个独立系统再在最后勉强拼接，而是从一开始就面向统一建模设计。

一个对新手最有帮助的玩具例子是：文本“ 一只猫 ”可能只占几个子词 token，一张图占几百个 token，一秒音频占 32 个 token，一段视频则按秒继续累加。它们进入模型后，本质上都变成了“序列中的位置 + 类型 + 向量表示”。模型并不需要一套单独的“图像大脑”和另一套“音频大脑”在最后做人工对齐，而是用同一套注意力机制直接在统一序列里建立关联。

---

## 问题定义与边界

多模态 tokenizer 的核心问题，不是“怎么把所有模态都压成文字”，而是“怎么把不同模态转换为可被统一序列模型处理的离散输入”。离散输入的意思是：最终都要变成一个个有顺序、有位置、可计数的 token。

如果只看公开 API，Gemini 的工程边界首先是上下文预算。设文本、图像、音频、视频分别消耗的 token 数为 $T_{\text{text}}, T_{\text{img}}, T_{\text{audio}}, T_{\text{video}}$，那么总预算满足：

$$
T_{\text{text}} + T_{\text{img}} + T_{\text{audio}} + T_{\text{video}} \leq T_{\text{context}}
$$

这里的 $T_{\text{context}}$ 不是固定 32,768。官方当前页面对 Gemini 3 Pro 标注的是 1M 输入 token；不同模型族、不同 media resolution 设置下，图像和视频每个输入对象消耗的 token 也会变化。因此，讨论“Gemini 的多模态 tokenizer 设计”时，准确边界不是某个固定数字，而是“统一预算内的跨模态序列化”。

下面这个表更适合工程预算，而不是讨论论文细枝末节：

| 模态 | 公开可观察的典型 token 成本 |
| --- | --- |
| 文本 | 动态子词分词，长度随语言和内容变化 |
| 图像 | 小图常见 258 token；更大图会按 tile 切分 |
| 音频 | 32 token/秒 |
| 视频 | 263 token/秒 |
| PDF/高分辨率媒体 | 受 `media_resolution` 影响明显 |

一个简单预算例子：10 个文本 token + 1 张小图 258 token + 1 秒音频 32 token，总计约 300 token。若改成 60 秒音频，仅音频就接近 1920 token。新手最容易忽略的点是，文本在很多任务里反而不是大头，图像和视频才是上下文预算的主要消耗者。

---

## 核心机制与推导

先看文本。子词分词的意思是“不强行按整词切，而是按常见片段切”，白话讲就是把“internationalization”拆成几个更稳定、可复用的碎片。Google 的 SentencePiece 是这类分词器的代表：它直接在原始文本上学习子词单元，不依赖英文空格式预切词。Gemini 官方没有公开说明 Gemini 3 内部一定使用了哪份 SentencePiece 模型，但从 Google 系技术栈和子词分词的工程合理性看，用 SentencePiece 风格理解文本入口是合适的。

图像侧的 patch 是“把一张大图切成小方块”。公开 API 不会直接告诉你内部 patch 大小，但它明确告诉你图像会被缩放、裁切、分块后计入统一 token 预算。工程上可以把这理解成两步：先切块，再把每块映射成向量。抽象表达是：

$$
T_{\text{img}} = \mathrm{Project}(\mathrm{Patchify}(image))
$$

音频侧的帧是“把连续声音按时间切成很多短片段”。公开文档说明音频会按每秒 32 token 表示，且会统一下采样并合并多声道。工程上常见做法是先提取时频特征，再投影成可与其他模态对齐的向量，因此可以写成：

$$
T_{\text{audio}} = \mathrm{Project}(\mathrm{FrameFeature}(audio))
$$

视频本质上是“按时间展开的图像序列”，只不过通常会伴随音轨。于是视频 token 可以写成：

$$
T_{\text{video}} = \mathrm{concat}(T_{\text{frame}_1}, T_{\text{frame}_2}, \dots)
$$

所有模态最终会形成统一输入流：

$$
T = \mathrm{concat}(T_{\text{text}}, T_{\text{img}}, T_{\text{audio}}, T_{\text{video}})
$$

然后进入同一套自注意力。自注意力的意思是“序列里每个位置都能决定自己该看谁”，白话讲就是每个 token 都在全局里找与自己最相关的信息。公式仍然是标准 Transformer 形式：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}\right)V
$$

这套公式本身不区分文本、图像、音频。真正区分来源的，通常是位置编码和模态类型标记。类型标记可以理解为“给 token 挂一个来源标签”，告诉模型“这个向量来自文本”“那个向量来自图像”。这样，当用户问“图里的人在说什么”时，文本 query token 可以通过注意力去看音频 token、图像 token 和时间相关 token，而不是只盯着文字上下文。

真实工程例子是视频摘要。假设你要总结一段产品发布会视频，常见做法不是把整段原始视频全塞进去，而是：按 1 FPS 抽帧，提取音轨，必要时附带字幕。这样进入模型的统一流大致是“帧 token 序列 + 音频 token 序列 + 文本提示词”。模型输出摘要时，就能把“画面切到哪一页 PPT”“某句话在哪个时刻说出”“这个词是不是图上也出现过”一起纳入推理。

---

## 代码实现

下面给一个可运行的玩具实现。它不是 Gemini 源码，而是用公开计费规则模拟“统一 token 预算器”。重点不是复刻内部网络，而是让新手先掌握“多模态先统一记账，再统一排队入模”的思路。

```python
from math import ceil

def estimate_text_tokens(text: str) -> int:
    # 这里只做玩具近似：中文按字、英文按空格近似都不准确，但足够演示预算。
    # 真正系统会用子词分词器。
    stripped = text.strip()
    if not stripped:
        return 0
    if " " in stripped:
        return len(stripped.split())
    return len(stripped)

def estimate_image_tokens(width: int, height: int) -> int:
    # 依据 Gemini API 公开规则做简化：
    # 小图（<=384x384）常见 258 token，大图按 tile 计。
    if width <= 384 and height <= 384:
        return 258
    crop_unit = max(1, int(min(width, height) // 1.5))
    tiles = ceil(width / crop_unit) * ceil(height / crop_unit)
    return tiles * 258

def estimate_audio_tokens(seconds: float) -> int:
    return ceil(seconds * 32)

def estimate_video_tokens(seconds: float) -> int:
    return ceil(seconds * 263)

def estimate_total_budget(
    text: str,
    images: list[tuple[int, int]],
    audio_seconds: float,
    video_seconds: float,
) -> int:
    total = estimate_text_tokens(text)
    total += sum(estimate_image_tokens(w, h) for w, h in images)
    total += estimate_audio_tokens(audio_seconds)
    total += estimate_video_tokens(video_seconds)
    return total

toy_total = estimate_total_budget(
    text="一只猫 在 沙发 上",
    images=[(384, 384)],
    audio_seconds=1,
    video_seconds=0,
)
assert toy_total == 5 + 258 + 32

real_case_total = estimate_total_budget(
    text="请总结这段发布会视频的产品功能与定价",
    images=[],
    audio_seconds=0,
    video_seconds=12,
)
assert real_case_total >= 12 * 263
print("toy_total =", toy_total)
print("real_case_total =", real_case_total)
```

如果把这段代码映射回真实系统，流程可以理解为四步。

1. 文本做子词分词，得到整数 token 序列。
2. 图像、音频、视频先做各自预处理，再转成一串“媒体 token”。
3. 给不同模态附加位置和类型信息。
4. 串成一个统一序列送进共享 Transformer。

这也是为什么工程里常见的数据结构不是“一个文本输入 + 一个图片输入 + 一个音频输入”，而是更接近下面这种统一表示：

| 字段 | 含义 |
| --- | --- |
| `token_id` | 离散编号，或媒体 token 编号 |
| `modality` | 来源标签，如 `text/image/audio/video` |
| `position` | 在统一序列中的位置 |
| `time_index` | 音频或视频的时间索引，可选 |
| `payload` | 媒体编码后的向量或引用 |

---

## 工程权衡与常见坑

最大的权衡不是“能不能统一”，而是“统一后预算和对齐是否还可控”。

第一类坑是 token 爆炸。视频每秒 263 token，音频每秒 32 token，这意味着一分钟视频大约就是 $60 \times 263 = 15780$ token，还没算提示词和其他上下文。很多新手以为多模态输入最大的成本是图片数量，实际上长视频更容易把预算直接吃满。

第二类坑是把“计费 token”误当成“内部 patch 数”。公开 API 说一张小图常见 258 token，不等于内部一定只有 258 个视觉 patch，也不等于 patch 一定固定为 16×16。API 暴露的是外部计数接口，不是完整内部表征。写技术文章时，这个边界必须说清，否则会把估算模型误写成架构事实。

第三类坑是模态类型漂移。类型 embedding 的意思是“给不同模态一个固定身份编码”。如果训练或推理阶段类型约定混乱，模型会出现跨模态对齐错误，例如把音频线索错配到不相关画面。这类问题不会以“程序报错”暴露，而是以回答逻辑矛盾的形式出现。

第四类坑是时序处理。视频不是“很多张独立图片”，而是“带时间顺序的图片流”。如果只保留帧，不保留时间关系，模型能看懂物体，但更难看懂动作、因果和转场。

下面这个表更接近真实开发问题：

| 常见坑 | 现象 | 处理方式 |
| --- | --- | --- |
| 视频 token 爆炸 | 长视频一进模型就接近上限 | 降 FPS、切片、先检索后入模 |
| 把 API 计数当内部结构 | 文章或系统设计出现伪精确结论 | 明确区分“公开行为”和“内部实现推断” |
| 类型标签不稳定 | 跨模态回答互相打架 | 固定类型 ID，保持训练推理一致 |
| 忽略时间信息 | 能识别画面，不能解释事件顺序 | 加时间位置编码或分段摘要 |
| 文本过长挤占媒体空间 | 图像和音频细节被压缩 | 先压缩文字上下文，再投喂关键媒体 |

---

## 替代方案与适用边界

统一 tokenizer 并不是唯一方案。另一条常见路线是“专用 encoder + 后融合”。encoder 是“专门把某一类输入变成向量的前置模型”，白话讲就是先让视觉模型只做视觉、语音模型只做语音，最后再把结果拼起来。

两种路线的差别可以概括如下：

| 维度 | 统一序列化方案 | 专用 encoder + fusion |
| --- | --- | --- |
| 跨模态对齐 | 直接在共享注意力里形成 | 需要额外融合层 |
| 工程复杂度 | 接口统一，预算统一 | 组件更多，调度更复杂 |
| 单模态优化 | 受共享容量约束 | 可对单模态单独极致优化 |
| 可解释性 | 更依赖整体行为分析 | 各模态中间结果更容易单独检查 |
| 适用场景 | 问答、摘要、代理推理 | 识别、检索、工业流水线 |

如果项目只做单模态 OCR 或单模态语音转写，专用 encoder 往往更直接。因为这类任务对“统一推理”要求不高，对延迟、成本和可控性要求更高。但如果任务是“看图听音再回答问题”“对视频做带语境的长摘要”“让代理同时读文档、看截图、听会议录音”，统一序列化方案更自然。

所以，Gemini 多模态 tokenizer 设计最值得记住的不是某个未经官方完整公开的内部 patch 细节，而是这一点：它把不同模态都约化为统一序列问题。统一之后，Transformer 的核心能力就不再局限于“读文字”，而是“在同一条信息流里做跨模态选择、对齐和生成”。

---

## 参考资料

- Google AI for Developers, Token counting guide: https://ai.google.dev/gemini-api/docs/tokens
- Google AI for Developers, Image understanding: https://ai.google.dev/gemini-api/docs/image-understanding
- Google AI for Developers, Audio understanding: https://ai.google.dev/gemini-api/docs/audio
- Google AI for Developers, Video understanding: https://ai.google.dev/gemini-api/docs/video-understanding
- Google AI for Developers, Media resolution: https://ai.google.dev/gemini-api/docs/media-resolution
- Google DeepMind, Gemini 1.5 Technical Report: https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf
- Google DeepMind, Gemini 3 model page: https://deepmind.google/en/models/gemini/
- Google DeepMind, Gemini 3 Pro model page: https://deepmind.google/en/models/gemini/pro/
- Google SentencePiece repository: https://github.com/google/sentencepiece
