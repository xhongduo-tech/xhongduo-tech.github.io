## 核心结论

截至 2026 年 3 月，视频理解的主线已经不是“给每一帧单独打标签”，而是“先决定看什么，再决定保留多少，再决定何时回看原视频”。这里的**时空联合建模**，白话说就是把“画面里有什么”和“这些内容在时间上怎么变化”一起处理，而不是拆成两个孤立问题。

最新一波方法大致收敛到同一条工程路线：

1. 用轻量模块挑出更可能有用的帧或片段。
2. 把保留下来的视觉信息压缩成更少的 token。
3. 让大模型只在必要时做长程推理，或者回召原始片段补证据。

对零基础读者，最直观的理解是：长视频不再按秒逐帧“全看”，而是先做关键帧截图、场景归档、片段压缩，再把这些摘要喂给大模型做问答、摘要和检索。

| 策略 | 核心做法 | 延迟 | 推理范围 | 资源占用 |
|---|---|---:|---:|---:|
| 逐帧分类 | 每帧单独编码，再汇总 | 高 | 低到中 | 很高 |
| 稀疏帧 + 压缩 | 先选关键帧，再压成少量 token | 中 | 中到高 | 中 |
| 多模态对齐 | 视频表示与文本问题联合匹配 | 中 | 高 | 中到高 |

“最新进展”的关键不是单帧识别更准，而是开始能处理三类更接近真实产品的问题：分钟级长视频摘要、跨事件段落检索、流式视频实时问答。

---

## 问题定义与边界

**视频理解**，白话说，就是让模型回答“视频里发生了什么、什么时候发生、和问题是否相关”。它至少包含三层难度：

| 层级 | 典型任务 | 难点 |
|---|---|---|
| 单帧内容 | 识别人物、物体、场景 | 静态视觉识别 |
| 短时动作 | 开门、挥手、摔倒 | 相邻帧变化 |
| 长时事件 | 会议开始、球权转换、一次事故全过程 | 跨分钟依赖、事件边界、多主体交互 |

长视频的真正瓶颈有三个。

第一，**冗余帧**很多。相邻帧常常几乎一样，全部送入模型会让计算量接近线性增加，而注意力开销又常常接近二次增长。

第二，**事件边界**难判定。所谓事件边界，白话说就是“这一段已经结束，下一段是新场景”。边界切得太频繁，系统不断建新片段，延迟上升；切得太粗，多个事件混在一起，问答时容易取错证据。

第三，**实时约束**很硬。直播、监控、会议助手这类系统不能等视频播完再统一推理，必须一边接收视频一边做压缩和缓存。

Vista 把场景切换抽象为一个阈值判定。若当前帧和场景锚点帧、相邻帧都不够相似，就触发新场景。这个想法可写成：

$$
B(F_i)=\mathbb{I}\left[S_{\text{anchor}}(F_i)<\tau \land S_{\text{adj}}(F_i)<\tau\right]
$$

其中 $B(F_i)=1$ 表示“从这一帧开始新场景”，$\tau$ 是阈值，白话说就是“相似到什么程度还算同一段”。

玩具例子：一段 2 分钟做饭视频里，前 40 秒在切菜，接着镜头转到炒锅，再转到摆盘。如果系统一直把这 2 分钟看成同一段，用户问“什么时候开始下锅”时，模型需要在很长上下文里找证据；如果系统提前按“切菜/炒菜/摆盘”三段切好，检索就会快很多。

它的适用边界也很明确。对只有 5 到 10 秒的短视频，复杂的分层压缩和回召机制往往得不偿失；对超长直播或高密度监控，单纯加采样率也不能解决问题，因为延迟和显存都会先爆掉。

---

## 核心机制与推导

这一轮进展可以用三篇代表工作来理解：Vista 负责“场景级缓存和回召”，FrameOracle 负责“预测该看哪些帧、看多少帧”，STTM 负责“把视觉 token 压缩到还能推理的规模”。

### 1. FrameOracle：先学会少看

**帧采样**，白话说，就是从很多帧里只挑少量帧送进模型。旧方法常用均匀采样，比如固定取 16 帧，但这有两个问题：内容密集的视频可能不够看，内容稀疏的视频又看太多。

FrameOracle 的改进点是同时预测两件事：

1. 哪些帧与当前问题最相关。
2. 这个问题到底需要多少帧。

已公开结果显示，它把 16 帧输入平均降到 10.4 帧而不掉精度；从 64 帧候选里平均压到 13.9 帧时，准确率还提升了 1.5%。这说明“少看”不一定更差，因为删掉的是冗余噪声，不是有效证据。

### 2. STTM：把 token 当成可合并的时空块

**token**，白话说，就是模型内部处理图像或文本的最小离散单元。视频里 token 太多，注意力代价会迅速上涨。STTM 的做法分两步：

1. 先在空间上做多粒度划分。
2. 再在时间上合并相邻帧中重叠且相似的 token。

它先用四叉树构建多粒度空间 token。四叉树，白话说，就是把一块区域不断切成四个更小区域；如果某区域变化不大，就保留大块，不必继续细分。空间合并复杂度在论文中写成：

$$
C_{STM}=\sum_{i=1}^{Lv}\frac{HW}{4^i}\times 4
$$

极限上界为：

$$
\lim_{Lv\to\infty}C_{STM}=\frac{4HW}{3}
$$

这表示空间搜索本身仍是线性量级，不会像全量 token 注意力那样快速爆炸。

时间合并时，可以把判据抽象为：

$$
M(u_t,v_{t+1})=\mathbb{I}\left[\text{overlap}(u_t,v_{t+1})>0 \land \cos(u_t,v_{t+1})>\tau_T\right]
$$

意思是：如果前后两帧里同一区域有重叠，而且特征相似度超过阈值 $\tau_T$，就把它们并到同一条时序轨迹里。

对新手的直观解释：STTM 不是每帧都把整张图切成同样多的小格子，而是先看哪里细节多、哪里细节少。细节少的地方保留大块，细节多的地方保留小块，然后再把时间上重复的块合并掉。

### 3. Vista：把压缩、缓存、回召串成完整系统

Vista 的价值不只是“压缩”，而是把流式问答系统真正需要的三件事串起来：

| 方法 | 输入 | 处理 | 输出 | 代表效果 |
|---|---|---|---|---|
| Vista | 连续视频流 | 场景切分 → 场景压缩 → 查询时回召 | 可检索的场景缓存 | 长上下文流式问答 |
| FrameOracle | 问题 + 候选帧 | 预测相关帧与所需帧数 | 更短帧序列 | 16 帧降到 10.4 帧 |
| STTM | 视频 token | 空间多粒度合并 + 时间合并 | 更少视频 token | 50% 预算下约 99.5% 相对精度 |

真实工程例子：直播电商问答。视频流持续进来，GPU 里只保留每个场景的压缩表示，CPU 保留全帧。用户突然问“主播刚才推荐的蓝色耳机是什么时候拿出来的”，系统先检索相关场景，再把必要原帧回召给大模型，而不是让大模型从头重看过去几分钟内容。

---

## 代码实现

下面给出一个可运行的 Python 玩具实现，模拟“边界检测 + 场景缓存 + 查询时回召”的骨架。它不是论文复现，而是把核心工程结构写清楚。

```python
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Frame:
    idx: int
    anchor_sim: float
    adj_sim: float
    importance: float
    payload: str


def is_new_scene(frame: Frame, tau: float) -> bool:
    return frame.anchor_sim < tau and frame.adj_sim < tau


def select_frames(frames: List[Frame], max_keep: int) -> List[Frame]:
    ranked = sorted(frames, key=lambda x: x.importance, reverse=True)
    kept = sorted(ranked[:max_keep], key=lambda x: x.idx)
    return kept


def compress_scene(frames: List[Frame]) -> Dict[str, Any]:
    kept = select_frames(frames, max_keep=max(1, min(3, len(frames))))
    return {
        "start": frames[0].idx,
        "end": frames[-1].idx,
        "summary_tokens": [f.payload for f in kept],
        "full_frames": frames,
    }


def build_scene_cache(stream: List[Frame], tau: float) -> List[Dict[str, Any]]:
    scenes = []
    current = []

    for frame in stream:
        if current and is_new_scene(frame, tau):
            scenes.append(compress_scene(current))
            current = [frame]
        else:
            current.append(frame)

    if current:
        scenes.append(compress_scene(current))
    return scenes


def recall_scene(scenes: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    # 玩具策略：如果 query 中包含 blue，就优先召回含 blue 的场景
    for scene in scenes:
        joined = " ".join(scene["summary_tokens"]).lower()
        if "blue" in query.lower() and "blue" in joined:
            return scene
    return scenes[0]


stream = [
    Frame(0, 0.95, 0.93, 0.2, "cut vegetable"),
    Frame(1, 0.91, 0.90, 0.4, "knife board"),
    Frame(2, 0.40, 0.38, 0.8, "pan fire"),
    Frame(3, 0.88, 0.87, 0.7, "oil in pan"),
    Frame(4, 0.35, 0.33, 0.9, "blue plate"),
]

scenes = build_scene_cache(stream, tau=0.5)

assert len(scenes) == 3
assert scenes[0]["start"] == 0 and scenes[0]["end"] == 1
assert scenes[1]["start"] == 2 and scenes[1]["end"] == 3
assert scenes[2]["start"] == 4 and scenes[2]["end"] == 4

hit = recall_scene(scenes, "Where is the blue object?")
assert "blue plate" in " ".join(hit["summary_tokens"])
```

如果把这个骨架映射到真实系统，通常是下面这条流水线：

1. 视频帧进入轻量编码器，计算相似度和帧重要性。
2. 边界检测决定是否关闭当前场景并写入缓存。
3. 场景压缩结果留在 GPU，原始帧或特征放在 CPU 或对象存储。
4. 问答请求到来时，先用压缩表示检索场景，再决定是否补帧。
5. 最终把多粒度 token 和文本问题一起送入大模型。

FrameOracle 风格的动态帧数预测也可以写成简化公式：

$$
k=\min\left(K_{\max}, \max\left(K_{\min}, \left\lceil \alpha \sum_i w_i \right\rceil\right)\right)
$$

其中 $w_i$ 是每帧的重要性分数，$k$ 是当前问题真正要保留的帧数。白话说，内容越复杂、越分散，系统就多看一些；内容越集中，就少看一些。

---

## 工程权衡与常见坑

视频理解系统真正难的地方不是“论文模块是否先进”，而是三组权衡是否做对。

| 配置项 | 调高后的收益 | 调高后的代价 | 常见坑 |
|---|---|---|---|
| 边界检测频率 | 更容易捕获事件切换 | 计算更重，缓存碎片化 | 场景过短，召回命中率下降 |
| 采样率 | 漏事件概率下降 | 延迟和显存上升 | 冗余帧挤占有效预算 |
| GPU 缓存比例 | 查询更快 | 显存更紧张 | 长视频会积压，后续场景被迫丢弃 |
| CPU 全帧保留 | 可回召原证据 | I/O 压力大 | 忘记做索引，回召变慢 |

最常见的坑有四个。

第一，只提高采样率，不做层级压缩。这样会把系统从“看不全”变成“看得更多但还是推不动”。

第二，只保留摘要 token，不保留原始帧。短问答可能没问题，但遇到“第 3 次传球前谁站在左侧”这类细节问题，模型没有证据可回。

第三，边界检测完全依赖视觉相似度。实际工程里很多事件边界不是画面大变，而是语义大变，例如会议中发言人切换、直播中商品切换，这时需要把音频、字幕或 OCR 一起考虑。

第四，把离线方案直接搬到实时系统。离线处理可以先看完整视频再决定采样；实时系统做不到，只能基于当前缓存和短历史做近似决策。

一个实用原则是：GPU 保目录，CPU 存全文。也就是 GPU 只保存压缩 token、场景索引和少量关键帧，CPU 保存可回召的原始内容。这样问答时先看目录，再按需翻正文。

---

## 替代方案与适用边界

不是所有场景都需要 Vista、STTM、FrameOracle 这种组合。

| 方案 | 适合视频长度 | 延迟要求 | 事件密度 | 适用场景 | 不适用场景 |
|---|---:|---:|---:|---|---|
| 完全稀疏采样 + 提示词 | 短 | 低到中 | 低 | 短视频分类、粗粒度问答 | 分钟级推理、事件检索 |
| 全量帧直接进模型 | 短到中 | 高 | 中 | 离线评测、数据量小 | 实时系统、长视频 |
| FrameOracle | 中到长 | 中 | 中 | 问题相关采样 | 无法提前定义问题的纯离线摘要 |
| STTM | 中到长 | 中 | 中到高 | token 预算紧张的 Video LLM | 视觉细节极端敏感任务 |
| Vista | 长到超长 | 低到中 | 高 | 流式问答、直播、监控 | 极短视频、一次性离线分类 |

新手可以这样判断：

- 如果你只想回答“这 8 秒短片是不是海边”，随机或均匀采样几帧通常够了。
- 如果你要回答“这 20 分钟会议里，产品经理什么时候第一次提到预算”，就必须有场景缓存、稀疏采样和按需回召。
- 如果你要在直播过程中即时回答问题，单纯全量送模基本不可行，因为延迟和显存都不稳定。

所以，最新进展不是某个单点模型突然“更聪明”，而是系统设计从“全量输入”转成“预测价值、压缩冗余、按需回看”。

---

## 参考资料

| 标题 | 年份 | 核心贡献 | 链接 |
|---|---:|---|---|
| Vista: Scene-Aware Optimization for Streaming Video Question Answering under Post-Hoc Queries | 2026 | 提出场景级切分、压缩、回召三阶段流式视频问答框架 | [arXiv:2602.08448](https://arxiv.org/abs/2602.08448) |
| FrameOracle: Learning What to See and How Much to See in Videos | 2025，2026年1月修订 | 同时预测“看哪些帧”和“看多少帧”，提升效率精度比 | [arXiv:2510.03584](https://arxiv.org/abs/2510.03584) |
| Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs | ICCV 2025 | 用四叉树空间合并加时间合并压缩视频 token，支持 KV cache 复用 | [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Hyun_Multi-Granular_Spatio-Temporal_Token_Merging_for_Training-Free_Acceleration_of_Video_LLMs_ICCV_2025_paper.html) |
