## 核心结论

Gemini 当前公开可验证的视频理解机制，不是“先做一个专门的视频小模型，再把摘要交给大模型”，而是把视频按时间切成连续的多模态 token，直接放进长上下文窗口里做统一序列建模。这里的“长上下文”就是模型一次能同时看到的输入长度。根据 2026 年 2 月 27 日的 Gemini API 文档，默认处理方式是每秒采样 1 帧，默认分辨率下每帧约 258 个视觉 token，音频约 32 个 token/秒，再加少量元数据，总计约 300 token/秒；因此 1 小时视频大约需要 $3600 \times 300 \approx 1{,}080{,}000$ token，已经贴近 1M 上下文上限。

这件事的直接意义是：Gemini 擅长的问题不只是“这段视频讲了什么”，而是“第几分钟发生了什么”“谁先说话”“某个结果是由哪个动作触发的”。时序建模就是让模型在 token 序列里保留先后顺序，并在问答时回到对应时间点建立因果链。这里的“因果链”可以理解为“前面事件如何触发后面事件”的顺序证据。

公开资料还说明了一个重要事实：Gemini API 默认不是智能挑关键帧，而是线性 1 FPS 采样。也就是说，它更像“均匀取样的长上下文理解”，不是先做复杂镜头检测再挑帧。所谓“temporal pooling”如果严格按公开资料表述，Google 没有放出内部层级压缩细节；工程上能确认的是两类时序压缩效果：一类来自低帧率采样，另一类来自低媒体分辨率把每帧 token 从 258 降到 66。前者压缩时间密度，后者压缩每帧表征密度。

在效果上，Google 2025 年 5 月 9 日的 Gemini 2.5 视频理解博客展示了它在长视频 moment retrieval、时序计数、视频转交互应用上的能力。MINERVA 论文中的 cause and effect 子项里，Gemini 2.5 Pro 为 44.57%，VideoLLaMA3 为 40.22%。这个差距不算碾压，但说明统一长上下文建模在“跨长时间段串证据”时更稳，尤其是带音频线索的问题。

| 指标 | Gemini 公开机制 | 对工程的含义 |
|---|---|---|
| 默认采样 | 1 FPS 线性采样 | 长视频可进窗口，但快动作容易漏 |
| 默认视觉 token | 258 token/帧 | 细节更强，成本更高 |
| 低分辨率视觉 token | 66 token/帧 | 更适合超长视频 |
| 音频 token | 32 token/秒 | 说话人、语气、关键词能进入同一上下文 |
| 默认总速率 | 约 300 token/秒 | 1M 上下文约覆盖 1 小时 |
| 低分辨率总速率 | 约 100 token/秒 | 1M 上下文约覆盖 3 小时 |

一个玩具例子：把一次 20 分钟站会按 1 FPS 抽帧，模型拿到的是约 1200 张时间戳图片加整段压缩音频。问“谁最后一次说出‘发布’这个词？”时，模型要同时利用音频转写位置、说话画面和后续上下文，才能把答案定位到最后一次而不是第一次。

一个真实工程例子：Google 在 Gemini 2.5 博客里展示了对 10 分钟 keynote 做 moment retrieval，以及对 Project Astra 视频统计 17 次“拿起手机”的时序计数。这类任务不是单帧分类，而是跨多段片段聚合证据。

---

## 问题定义与边界

本文讨论的问题是：如何让一个通用多模态大模型，在“小时级视频 + 音频”的输入上完成事件定位、时序问答、因果推理和计数。这里的“小时级”不是营销词，而是输入预算问题。视频理解的第一约束不是模型聪不聪明，而是 token 能不能塞进上下文。

把问题写成公式最清楚：

$$
T_{\text{total}} = f \times C_{\text{frame}} + C_{\text{audio}} + C_{\text{meta}}
$$

其中，$f$ 是每秒采样帧数，$C_{\text{frame}}$ 是每帧视觉 token 数，$C_{\text{audio}}$ 是每秒音频 token 数，$C_{\text{meta}}$ 是时间戳等元数据开销。对 Gemini 默认设置，可近似写成：

$$
T_{\text{total}} \approx 1 \times 258 + 32 + \text{meta} \approx 300\ \text{token/s}
$$

边界同样明确。

第一，默认 1 FPS 适合会议、访谈、课堂、演讲这类低动作密度视频，不适合高速运动、手部细动作、快速切镜。动作密度可以理解为“单位时间内信息变化有多快”。变化越快，1 FPS 丢信息越严重。

第二，1M 窗口并不等于“任意 1 小时都稳”。因为文档写的是“up to 1 hour long at default media resolution”，这表示它在产品能力上支持 1 小时量级，但从粗略计算看，1 小时默认分辨率已经非常接近预算边缘。实际请求里如果再加长文本提示、工具说明或多轮历史，余量会更少。

第三，低媒体分辨率不是免费午餐。它能把每帧从 258 token 降到 66，显著扩展可处理时长，但会牺牲小字、细节物体、屏幕内容、白板内容的可读性。

下面这个表格能直接看出预算边界：

| 场景 | fps | 每帧 token | 音频 token/s | 总 token/s | 1M 可覆盖时长 |
|---|---:|---:|---:|---:|---:|
| 默认视频 | 1 | 258 | 32 | 约 300 | 约 55 分钟到 1 小时 |
| 低分辨率视频 | 1 | 66 | 32 | 约 100 | 约 2.8 到 3 小时 |
| 高动作分析 | 2 | 258 | 32 | 约 548+meta | 不到 31 分钟 |
| 静态讲座降采样 | 0.5 | 258 | 32 | 约 161+meta | 约 1.7 小时 |

对零基础读者，一个直观理解是：模型不是“看视频文件”，而是在“读一串按时间排列的视觉 token 和音频 token”。视频越长、帧率越高、分辨率越高，这串 token 越长，越容易超预算。

---

## 核心机制与推导

公开资料支持的机制可以拆成四步。

第一步，线性采样。线性采样就是按固定时间间隔取帧，而不是根据镜头变化自适应挑帧。Gemini API 默认 1 FPS，这意味着第 0 秒、第 1 秒、第 2 秒各取一帧。它的优点是简单、稳定、便于长视频预算估算；缺点是可能错过 1 秒内完成的关键动作。

第二步，视觉与音频分别 token 化。token 化就是把原始媒体压成模型可读的离散表示。公开视频文档给出默认分辨率下 258 token/帧，低分辨率 66 token/帧，音频 32 token/秒。

第三步，把带时间顺序的多模态 token 拼到同一个上下文。这里我用“统一序列建模”描述，是基于 API 侧的 token 预算与模型行为作出的工程推断：视频、音频、文本提示共享同一上下文窗口，因此问答时模型可以直接在同一注意力图里关联“某秒的画面”“那一秒的说话内容”和“用户问题”。

第四步，问答阶段进行跨时间证据聚合。注意力可以理解为“当前问题在整段上下文里查找相关证据”的机制。问“为什么观众开始鼓掌？”时，模型要把“上一段 speaker 说了什么”“屏幕切到哪页”“观众何时开始鼓掌”连成一条链，而不是只看鼓掌那一帧。

推导 token 预算时，核心公式很直接：

$$
C_{\text{sec}} = f \cdot C_{\text{frame}} + C_{\text{audio}}
$$

默认分辨率且 $f=1$ 时：

$$
C_{\text{sec}} = 1 \cdot 258 + 32 = 290
$$

加上元数据后，文档给出的工程近似是约 300 token/s。于是 1 小时预算约为：

$$
C_{\text{hour}} \approx 3600 \cdot 300 = 1{,}080{,}000
$$

低分辨率时：

$$
C_{\text{sec,low}} = 1 \cdot 66 + 32 = 98
$$

所以 1M 窗口理论上约能覆盖：

$$
1{,}000{,}000 / 98 \approx 10{,}204\ \text{s} \approx 2.83\ \text{小时}
$$

这与 API 文档里“1M 可到 3 小时低分辨率”一致。Google 博客进一步给出 2M 上下文下约 6 小时视频，这也是同一数量级的外推。

再看一个采样率变化的玩具例子。若把 fps 提到 2：

$$
C_{\text{sec}} = 2 \cdot 258 + 32 = 548
$$

这时 1M 只能覆盖：

$$
1{,}000{,}000 / 548 \approx 1824\ \text{s} \approx 30.4\ \text{分钟}
$$

这说明高 fps 不是“更准就更好”，而是“用时长换细节”。

可以把时序链条想成下面这样：

`[00:00 画面token][00:00 音频token][00:01 画面token][00:01 音频token]...`

当问题是“谁先发言？”时，模型要在这条链里找到最早出现的说话证据；当问题是“为什么 A 后面站起来？”时，模型则要把 A 之前的音频或动作作为触发条件。

---

## 代码实现

下面用一个可运行的 Python 片段演示 token 预算计算。它不是 Gemini 官方 SDK 代码，而是工程估算器，用来在接入前判断方案是否会超上下文。

```python
def estimate_video_tokens(seconds: int, fps: float = 1.0, frame_tokens: int = 258, audio_tokens_per_sec: int = 32, meta_tokens_per_sec: int = 10) -> int:
    assert seconds > 0
    assert fps > 0
    assert frame_tokens > 0
    assert audio_tokens_per_sec >= 0
    assert meta_tokens_per_sec >= 0

    tokens_per_sec = fps * frame_tokens + audio_tokens_per_sec + meta_tokens_per_sec
    total = int(seconds * tokens_per_sec)
    return total

one_hour_default = estimate_video_tokens(3600, fps=1.0, frame_tokens=258, audio_tokens_per_sec=32, meta_tokens_per_sec=10)
one_hour_low = estimate_video_tokens(3600, fps=1.0, frame_tokens=66, audio_tokens_per_sec=32, meta_tokens_per_sec=2)
half_hour_fast = estimate_video_tokens(1800, fps=2.0, frame_tokens=258, audio_tokens_per_sec=32, meta_tokens_per_sec=10)

assert 1_000_000 < one_hour_default < 1_200_000
assert 300_000 < one_hour_low < 400_000
assert half_hour_fast < 1_100_000

print(one_hour_default, one_hour_low, half_hour_fast)
```

真正的处理流程一般长这样：

```python
def build_multimodal_context(video, fps=1.0, low_resolution=False, chunk_seconds=900):
    frame_tokens = 66 if low_resolution else 258
    audio_tokens_per_sec = 32

    for chunk in split_video(video, seconds=chunk_seconds):
        frames = sample_frames(chunk, fps=fps)          # 每秒抽帧
        visual_seq = [encode_frame(f, budget=frame_tokens) for f in frames]
        audio_seq = encode_audio(chunk, rate=audio_tokens_per_sec)
        context = interleave_with_timestamps(visual_seq, audio_seq)
        yield context

def answer_long_video(video, question):
    cache = []
    for context in build_multimodal_context(video, fps=1.0, low_resolution=True):
        summary = model_summarize(context)              # 对每段先做局部总结
        cache.append(summary)

    final_context = merge(cache, question)              # 再做全局问答
    return model_answer(final_context)
```

这里有两个实现点值得注意。

第一，超长视频别直接硬塞。分段摘要再做全局问答，虽然会损失部分细节，但比一次请求爆上下文更可控。

第二，高动作区间不要全片升 fps。更合理的做法是先粗采样定位，再对可疑片段二次细采样。真实工程里，这通常比全片 2 FPS 或 5 FPS 更省钱也更稳。

一个真实工程例子是会议视频搜索。你可以先用 1 FPS + 低分辨率把 2 到 3 小时会议建立粗索引，回答“哪一段提到发布计划”；再只对该片段提高 fps 或切回默认分辨率，回答“是谁先提出延期”。

---

## 工程权衡与常见坑

最核心的权衡只有一句话：你不能同时无限要“长时长、高清细节、高动作分辨率”。

| 方案 | token 消耗 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| 1 FPS + 默认分辨率 | 中高 | 语义稳，画面细节较完整 | 1 小时已贴近上限 | 会议、访谈、课堂 |
| 1 FPS + 低分辨率 | 低 | 时长显著扩展 | 小字和小物体识别变差 | 长讲座、监控回放 |
| 2 FPS + 默认分辨率 | 很高 | 更适合快动作 | 时长减半 | 体育片段、手部操作 |
| 分段 + 缓存摘要 | 可控 | 可处理超长视频 | 摘要可能丢关键细节 | 多小时内容检索 |

常见坑有五个。

`快动作漏检`  
1 FPS 下，一个 0.4 秒动作可能一帧都采不到。规避方式是提高该片段 fps，或者先做 shot/scene 切分后对高变化片段单独重跑。

`长视频刚好卡上限`  
理论预算没超，不代表请求就稳。提示词、系统指令、输出长度都会占 token。规避方式是预留 10% 到 20% 安全余量。

`低分辨率误伤 OCR 任务`  
PPT、代码录屏、白板板书属于文字密集场景，低分辨率往往不够。此时宁可缩短时长，也不要盲目降分辨率。

`把“关键帧选择”想成黑盒魔法`  
Gemini API 对外默认是线性采样，不是自动挑最好的一小组关键帧。真正的关键帧策略通常要你自己做，比如镜头变化检测、ASR 关键词召回、动作峰值片段重采样。

`短视频盲目加帧`  
Google 资料提到的 EgoSchema 类实验现象说明，更多帧不总是更好。短视频里信息可能在 16 帧左右已接近饱和，再加帧只会增加冗余和成本。

---

## 替代方案与适用边界

Gemini 的路线可以概括成“尽量把原始多模态上下文直接放进大模型”。VideoLLaMA 家族更典型的路线是“先用视觉编码器提特征，再通过 connector 或 Q-Former 类模块把视觉信息压缩后送进语言模型”。这里的“connector”就是把视觉特征映射到语言 token 空间的桥接层。

两种路线没有绝对胜负，关键看任务。

| 维度 | Gemini | VideoLLaMA |
|---|---|---|
| 主体思路 | 长上下文统一建模 | 视觉侧先压缩，再接 LLM |
| 长视频处理 | 依赖上下文窗口和 token 预算 | 依赖 connector 压缩和最大帧数设置 |
| 时序问答 | 更适合跨长跨度串证据 | 更依赖采样质量和压缩质量 |
| 成本控制 | 直接受 token 预算约束 | 更依赖模型结构与预设帧数 |
| 典型长项 | moment retrieval、长视频 QA、时间定位 | 短中视频理解、开源可部署 |

如果任务是“统计一场 keynote 里产品演示出现了几次”“某个角色最后一次说某词是什么时候”“1 小时视频里哪一段导致后面结果”，Gemini 的统一上下文更自然，因为你不需要先决定哪 16 帧或哪 256 帧最关键。

如果任务是“几十秒短视频分类”“本地 GPU 上离线部署”“模型结构可改、可训、可微调”，VideoLLaMA 更现实。尤其在开源场景里，你能直接控制视觉编码器、最大帧数、connector、训练数据和推理路径。

下面给一个极简的 VideoLLaMA 风格流程示意：

```python
def videollama_style_infer(video, question, max_frames=180):
    frames = sample_frames(video, fps=1, max_frames=max_frames)
    vision_feats = vision_encoder(frames)          # 先抽视觉特征
    visual_queries = q_former(vision_feats)        # 用少量查询向量压缩视觉信息
    llm_inputs = connector(visual_queries, question)
    return llm_generate(llm_inputs)
```

这个流程的优点是输入长度更容易控，缺点是压缩发生得更早。一旦 connector 前就把某些时序细节折叠掉，后面的 LLM 再强也补不回来。MINERVA 的 cause and effect 子项中，Gemini 2.5 Pro 的 44.57% 高于 VideoLLaMA3 的 40.22%，可以把它理解为统一上下文在复杂因果链上更不容易断线，但这不是说 VideoLLaMA 无用，而是它更适合预算受限、可本地化部署或短视频主导的场景。

---

## 参考资料

1. [Gemini API: Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)，2026-02-27 页面版本。用于默认 1 FPS、258 token/帧、66 token/帧、32 token/秒、1M 可处理约 1 小时默认分辨率或 3 小时低分辨率等数据。
2. [Google Developers Blog: Advancing the frontier of video understanding with Gemini 2.5](https://developers.googleblog.com/gemini-2-5-video-understanding/)，2025-05-09。用于 moment retrieval、17 次手机使用计数、2M 上下文下约 6 小时低分辨率视频等工程示例。
3. [Gemini API: Media resolution](https://ai.google.dev/gemini-api/docs/media-resolution)，2026 年页面版本。用于低媒体分辨率与视频 token 成本权衡说明。
4. [google-deepmind/neptune](https://github.com/google-deepmind/neptune)，MINERVA 数据集官方仓库。用于 MINERVA 数据集定义、题型、时长范围与推理标注说明。
5. MINERVA: Evaluating Complex Video Reasoning，arXiv:2505.00681，2025-05-01。用于 cause and effect 子项中 Gemini 2.5 Pro 44.57%、VideoLLaMA3 40.22% 的 benchmark 数据。
6. [DAMO-NLP-SG/VideoLLaMA3](https://github.com/DAMO-NLP-SG/VideoLLaMA3)，2025 项目仓库。用于开源 VideoLLaMA3 的视频理解流程、`fps` 与 `max_frames` 推理接口说明。
7. [DAMO-NLP-SG/VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2)，2024 项目仓库。用于 VideoLLaMA 系列 connector 式视觉到语言桥接方案的工程背景。
8. Gemini 1.5 Technical Report，2024。用于 1H-VideoQA 长视频评测设定与“更多上下文帧提升长视频问答精度”的实验背景。
