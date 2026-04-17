## 核心结论

LLaVA 的训练数据配比，核心不在“总样本数越大越强”，而在“不同训练阶段解决不同问题，所以需要不同类型的数据”。`LLaVA-Pretrain` 的 `558K` 数据主要用于视觉-语言对齐；`LLaVA-v1.5` 的 `mix665k` 主要用于指令微调，把“能把图像接到语言空间里”进一步变成“能回答、能解释、能做多轮对话”。

最值得记住的结论有三条。

第一，Pretrain 阶段的 `558K` 并不是通用意义上的“大模型预训练语料”，而是来自 LAION/CC/SBU 的过滤子集，并附带 BLIP 合成描述。它的目标很窄：让模型先学会“图像特征大致该对应哪些词、短语和句子”，即先完成跨模态对齐，而不是直接学复杂问答或长对话。这个阶段解决的是“图像表征如何落到语言模型已经理解的语义空间里”。参考数据卡可见，该数据集本身就围绕 caption-style 对齐设计，而不是围绕任务化指令设计。[Hugging Face 数据卡](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)

第二，v1.5 阶段的重点不是“再加更多图文数据”，而是“按目标把不同数据角色分层混合”。官方模型卡给出的粗粒度口径是 `158K GPT 生成指令 + 450K academic-task-oriented VQA mixture + 40K ShareGPT`。如果把 academic mixture 展开到细粒度数据源，后续复现文献给出的总量约为 `467K`，与 `158K + 40K` 相加后接近 `665K`。因此，`665K` 是整个混合指令集的总规模；`450K` 是对学术任务部分的近似写法，不是唯一精确拆分。写作或复现时，必须区分“粗粒度口径”和“细粒度口径”，否则数字看起来会对不上。[PaddleNLP 模型卡](https://paddlenlp.readthedocs.io/zh/latest/website/liuhaotian/llava-v1.5-7b/index.html)；[MouSi 附录表 10](https://xuanjing-huang.github.io/files/mousi.pdf)

第三，这个配比背后的设计思想可以概括成一句话：先用 `558K` 完成“看图能接上语言”，再用约 `665K` 的混合指令数据，把这种对齐能力改造成“按任务回答”和“像对话一样回答”的综合能力。一个足够准确的理解方式是：先让学生看大量图片和简短说明，学会“图里是什么”；再让他做混合题库，其中大头是标准问答题，小头是聊天、改写、解释类题，学会“面对不同问法该怎么答”。

下面这张表先给出最简版总览。

| 阶段 | 主要来源 | 样本量 | 主要作用 |
|---|---|---:|---|
| Pretrain | LAION/CC/SBU 过滤子集 + BLIP caption | 558K | 视觉-语言对齐 |
| V1.5 指令微调 | GPT 生成指令 | 158K | 学习多样化问法和指令格式 |
| V1.5 指令微调 | 学术任务型 VQA mixture | 官方粗粒度写法约 450K | 拉高标准视觉问答与定位能力 |
| V1.5 指令微调 | ShareGPT | 40K | 补多轮对话风格和回复自然度 |
| V1.5 全量 mix | 上述混合后的总集 | 约 665K | 平衡任务能力与对话能力 |

如果只记一句话，可以记成：

$$
\text{Pretrain 负责对齐，SFT 负责行为}
$$

这里的 `SFT` 指 supervised fine-tuning，也就是监督式指令微调。

---

## 问题定义与边界

这篇文章只讨论一个问题：`LLaVA-Pretrain` 和 `LLaVA-v1.5` 在训练数据上如何分工、如何配比、为什么要这样配。它**不讨论**模型结构、分辨率设置、LoRA、量化、推理速度，也不讨论不同底座语言模型的横向对比。

这里最容易让新手误解的是“预训练”这个词。很多人看到 `Pretrain`，会下意识认为它和语言模型的从零预训练是同一件事。不是。LLaVA 里的 `Pretrain` 更准确地说，是**视觉特征到语言空间的适配训练**。做的事情不是重建一个完整世界模型，而是训练一个多模态连接层或投影层，让视觉编码器输出的向量能被语言模型当成“有语义的前缀”来理解。

可以把两个阶段的边界先压缩成一张表。

| 维度 | LLaVA-Pretrain | LLaVA-v1.5 指令微调 |
|---|---|---|
| 目标 | 对齐 | 任务化回答 |
| 数据形态 | 图 + 简单描述或合成文本 | 图文问答、多轮对话、任务混合 |
| 关注点 | “图像语义是否能进语言空间” | “模型能否按要求输出正确答案” |
| 学习对象 | 语义映射关系 | 回答格式、任务偏好、对话风格 |
| 风险 | 对齐不充分，后续问答基础差 | 数据比例失衡，模型能力偏科 |

这个边界非常重要，因为它决定了数据该怎么选。

如果在 Pretrain 阶段混入大量复杂多轮对话，会有两个直接问题。

第一，优化目标变脏。模型还没学会稳定地把图像语义接入语言空间，就被要求学长回答、礼貌表达、风格模仿、多轮上下文延续，训练信号会互相干扰。

第二，学习顺序会反过来。正常顺序应当是先解决“看见图像后能知道在说什么”，再解决“知道之后怎么按要求说出来”。如果顺序颠倒，模型经常会出现一种表面现象：回复看起来很流畅，但一涉及图像细节就开始胡编。

因此，Pretrain 阶段更适合高覆盖、低复杂度、弱任务化的数据。这个阶段的数据价值，不是对话花样多，而是概念覆盖广、图文对应清楚、噪声可控。

v1.5 的边界则完全不同。它不是为了“继续做对齐”，而是为了把已经初步对齐的模型，塑造成一个可以完成任务的多模态助手。这时必须同时兼顾两类需求：

1. 学术型请求  
例如“图里有几个红色三角形”“这页文档左上角的字段是什么”“横轴的单位是什么”。

2. 通用型请求  
例如“帮我概括这张旅游照的氛围”“把这张图描述得更自然一点”“根据图像内容写一段说明”。

如果只堆学术问答，模型会更像判题器，答案短、硬、面向 benchmark；如果只堆通用对话，模型会更像聊天机器人，但在可判分的视觉问答任务上容易退化。`mix665k` 的配比，本质上就是在控制这个平衡点。

为了避免术语堆砌，可以把两个阶段想成两个完全不同的问题：

| 问题 | 对应阶段 | 典型错误 |
|---|---|---|
| “图像信息能不能进入语言模型？” | Pretrain | 连接不上，图像像噪声 |
| “进入之后能不能按任务要求稳定输出？” | v1.5 指令微调 | 会看不会答，或会答但答偏 |

这也是为什么文章讨论“数据配比”时，不能只盯着总样本量。真正有意义的是：**在什么阶段，用什么数据，服务什么目标**。

---

## 核心机制与推导

v1.5 的数据设计可以抽象成两大类。

第一类是学术任务型数据，记为 $A$。这类数据的目标是把模型往“短、准、可验证”的方向推，例如标准 VQA、定位、OCR、文档理解、图表理解等。

第二类是通用对话型数据，记为 $G$。这类数据的目标是把模型往“指令跟随、解释、自然交流”的方向推，例如 GPT 生成的多样指令数据和 ShareGPT 风格对话数据。

因此，通用对话数据在总混合集中的占比可以写成：

$$
r=\frac{G}{A+G}
$$

这里的 $r$ 不是一个装饰性的比例，而是训练时每一步梯度朝哪类行为倾斜的近似刻画。

按照官方常见的粗粒度口径，通用对话部分至少包括两块：

$$
G = 158K + 40K = 198K
$$

其中：

- `158K`：GPT-generated multimodal instruction-following data
- `40K`：ShareGPT data

如果把学术任务部分粗略记为：

$$
A = 450K
$$

那么通用对话占比为：

$$
r=\frac{198}{450+198}\approx 0.306
$$

也就是：

$$
r \approx 30.6\%
$$

对应地，学术任务占比约为：

$$
1-r \approx 69.4\%
$$

这个结果的解释很直接：v1.5 不是“以聊天为主、顺便做任务”，也不是“纯任务模型、顺便能聊天”，而是一个明显以任务数据为主体、同时保留约三成通用指令风格的混合方案。

这个比例为什么重要？因为训练时真正更新模型的是梯度。可以把梯度粗略理解成“模型每一步被往哪边拉”。如果 $r$ 太高，模型会被更多长回答、解释型回答、聊天式回答主导；如果 $r$ 太低，模型会更像一个标准问答器，擅长短答案和可判分任务，但对开放式请求响应会生硬。

为了让这个结论更具体，可以看一个小规模复现实验的预算分配。假设你只有 `1000` 条样本做快速原型，那么按粗粒度比例，大致应当分成：

$$
G_{1000}=1000\times 0.306 \approx 306
$$

$$
A_{1000}=1000-306=694
$$

这不表示必须机械地做成 `306` 和 `694`，而是说明如果你的混合结果偏到比如 `500:500`，那你做出来的模型行为已经不再接近 v1.5 的设计目标。

为了避免把“样本比例”和“能力比例”混为一谈，还需要补一个更工程化的公式。真实训练中，模型感受到的往往不是样本占比，而是**token 占比**。如果学术任务平均每条样本长度为 $L_A$，通用对话平均长度为 $L_G$，那么 token 级别的通用占比更接近：

$$
r_{\text{token}}=\frac{G\cdot L_G}{A\cdot L_A + G\cdot L_G}
$$

这意味着即便样本级比例是 `30.6%`，如果通用对话平均更长，那么 token 级别的训练影响力可能明显高于 `30.6%`。这也是很多复现实验“样本数看起来没问题，但模型更像聊天机器人”的原因之一。

再看一个更贴近工程的例子。假设你在做企业内部的“图像 + 文档问答”系统，用户既会问：

- “这张图里一共有几条曲线？”
- “请总结这页 PPT 的重点，用正式语气输出。”

第一类问题更偏学术任务，第二类更偏通用指令。如果全部训练数据都偏图表 QA，第二类请求的输出会偏短、偏机械；如果过量加入 ShareGPT 风格长对话，第一类请求的准确率和可判分性又容易下降。v1.5 的数据混合，本质上是在用数据分布控制这种能力交易关系。

这里还要单独说明一个经常引发混乱的点：为什么 `665K` 和 `158K + 450K + 40K = 648K` 对不上？

原因不是谁写错了，而是**统计口径不同**。官方传播中常用的是粗粒度口径，把 academic mixture 约写成 `450K`。而更细粒度的复现资料会把 academic mixture 展开成多个数据子集，例如 VQAv2、GQA、OCRVQA、VG、A-OKVQA、TextCaps、RefCOCO、DocVQA 等，总和约为 `467K`。于是：

$$
158K + 467K + 40K \approx 665K
$$

这个近似关系才与 `mix665k` 对得上。

可以把两种口径并排看：

| 统计口径 | GPT 指令 | Academic mixture | ShareGPT | 总计 |
|---|---:|---:|---:|---:|
| 粗粒度口径 | 158K | 约 450K | 40K | 约 648K |
| 细粒度口径 | 158K | 约 467K | 40K | 约 665K |

写文章、做复现、报告实验时，最好明确写一句：**本文若讨论设计思路，使用粗粒度口径；若讨论数据清单和复现实现，使用细粒度口径。** 这样读者才不会以为数字互相矛盾。[PaddleNLP](https://paddlenlp.readthedocs.io/zh/latest/website/liuhaotian/llava-v1.5-7b/index.html)；[MouSi](https://xuanjing-huang.github.io/files/mousi.pdf)

---

## 代码实现

工程里最稳妥的做法，不是把全部样本先拼成一个大 JSON 再完全随机打散，而是给每条样本保留来源标签，然后显式按目标比例组装 batch。这样做至少有三点收益：

1. 复现性更强，别人能知道你到底混了哪些来源。
2. 可观测性更强，训练中能定位是哪一类数据把分布拉偏。
3. 便于调参，可以单独提高或降低某个来源的采样权重。

下面给出一个**可直接运行**的 Python 最小示例。它不依赖 PyTorch 或深度学习框架，只演示“如何按目标比例采样多个数据源，并在一个 epoch 内检查观测分布是否接近目标分布”。

```python
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Sample:
    source: str
    sample_id: int
    text: str


def build_samples(source: str, size: int, start_id: int = 0) -> List[Sample]:
    return [
        Sample(
            source=source,
            sample_id=start_id + i,
            text=f"{source} example {i}",
        )
        for i in range(size)
    ]


def allocate_counts(batch_size: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {total_ratio}")

    raw = {name: batch_size * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}

    remainder = batch_size - sum(counts.values())
    if remainder > 0:
        # Use largest remainder to keep total exactly equal to batch_size.
        order = sorted(
            ratios.keys(),
            key=lambda name: raw[name] - counts[name],
            reverse=True,
        )
        for i in range(remainder):
            counts[order[i]] += 1

    return counts


def sample_batch(
    pools: Dict[str, Sequence[Sample]],
    batch_size: int,
    ratios: Dict[str, float],
    rng: random.Random,
) -> List[Sample]:
    counts = allocate_counts(batch_size, ratios)
    batch: List[Sample] = []

    for source, need in counts.items():
        pool = pools[source]
        if need > len(pool):
            raise ValueError(
                f"source={source} needs {need} samples, but pool only has {len(pool)}"
            )
        batch.extend(rng.sample(list(pool), need))

    rng.shuffle(batch)
    return batch


def simulate_epoch(
    pools: Dict[str, Sequence[Sample]],
    ratios: Dict[str, float],
    batch_size: int,
    steps: int,
    seed: int = 42,
) -> Counter:
    rng = random.Random(seed)
    counter: Counter = Counter()

    for _ in range(steps):
        batch = sample_batch(pools, batch_size, ratios, rng)
        counter.update(sample.source for sample in batch)

    return counter


def main() -> None:
    # Coarse-grained LLaVA-v1.5 style split:
    # academic = 450K, general = 158K + 40K = 198K
    academic_size = 450_000
    general_size = 198_000
    total_size = academic_size + general_size

    ratios = {
        "academic": academic_size / total_size,
        "general": general_size / total_size,
    }

    pools = {
        "academic": build_samples("academic", size=10_000, start_id=0),
        "general": build_samples("general", size=10_000, start_id=100_000),
    }

    batch_size = 128
    steps = 200

    counts = simulate_epoch(
        pools=pools,
        ratios=ratios,
        batch_size=batch_size,
        steps=steps,
        seed=123,
    )

    total_seen = sum(counts.values())
    observed = {
        name: counts[name] / total_seen
        for name in ratios
    }

    print("target ratios:")
    for name, ratio in ratios.items():
        print(f"  {name:8s} = {ratio:.4f}")

    print("\nobserved ratios:")
    for name, ratio in observed.items():
        print(f"  {name:8s} = {ratio:.4f}")

    print("\ncounts:")
    for name in ratios:
        print(f"  {name:8s} = {counts[name]}")


if __name__ == "__main__":
    main()
```

这段代码的关键点有三个。

第一，`allocate_counts()` 先根据目标比例把一个 batch 应该包含多少类样本算出来，再用“最大余数法”补齐四舍五入带来的误差。这样可以保证每个 batch 的总数严格等于 `batch_size`。

第二，`sample_batch()` 是按来源池分别采样，而不是把全部样本揉成一个大池子再赌随机数能否自然还原目标分布。工程里如果你关心稳定复现，这一步通常不能省。

第三，`simulate_epoch()` 让你在 epoch 级别统计观测比例。真实训练时，也应当把这种统计打到日志里，而不是凭感觉判断“配比应该差不多”。

如果用上面的粗粒度口径，目标比例大致是：

$$
\text{general}=\frac{198}{648}\approx 0.3056,\quad
\text{academic}=\frac{450}{648}\approx 0.6944
$$

对应到一个 `batch_size = 128` 的 batch，通常会分到：

$$
128 \times 0.3056 \approx 39
$$

也就是大约 `39` 条通用样本和 `89` 条学术样本。实际代码里允许在相邻 batch 上有微小波动，但长期均值应当收敛到目标配比。

如果你想进一步贴近真实训练流程，可以把来源从两类扩展到多类。例如：

| 来源标签 | 含义 |
|---|---|
| `llava_158k` | GPT 生成的多模态指令数据 |
| `sharegpt_40k` | 多轮对话风格数据 |
| `vqav2` | 标准视觉问答 |
| `gqa` | 场景图推理问答 |
| `ocrvqa` | OCR 相关问答 |
| `docvqa` | 文档问答 |
| `refcoco` | 细粒度定位与指代解析 |

这时最实用的工程做法，是把“总配比”拆成两层：

1. 先决定大类配比  
例如 academic vs general。

2. 再决定类内子来源配比  
例如 academic 内部 VQAv2、GQA、OCRVQA、DocVQA 各占多少。

这样做的好处是，调参时可以独立控制“任务主体”和“任务内部结构”，不会每次改一个子集就把全局分布一起打乱。

---

## 工程权衡与常见坑

在实际复现里，最常见的问题通常不是模型结构写错，而是**数据口径不清、采样逻辑不透明、统计指标不一致**。这三类问题会导致你以为自己在复现 LLaVA-v1.5，实际训练出来的却是另一种行为分布。

先看 Pretrain。LLaVA 用的不是 LAION/CC/SBU 原始全量，而是过滤后的 `LCS-558K`。这件事很关键，因为 Pretrain 阶段追求的是“覆盖面广但语义相对干净”的对齐数据。如果直接使用未过滤的大规模互联网图文池，模型学到的会是更强的网络分布偏差，例如高频场景、强视觉模板、热门概念重复出现。这样做短期内可能也能训练，但对齐质量往往更不稳，后续微调更容易出现幻觉、概念误绑、细粒度识别偏科。[Hugging Face 数据卡](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)

再看 v1.5 指令微调。很多人为了省事，会把所有样本合成一个统一 JSON，然后让 DataLoader 完全随机采样。这在做 demo 时没问题，但在正式训练里风险很大，因为你失去了两个关键能力：

1. 无法追踪每类数据实际被抽到了多少。
2. 无法区分“样本占比正确”和“token 占比正确”是不是同一回事。

这一点对新手尤其重要。因为训练时模型真正处理的是 token，不是“条数”。假设学术样本平均只有 `20` 个 token，而通用对话平均有 `120` 个 token，那么即便通用对话只占三成样本，它在 token 维度的影响也可能远高于三成。结果就是：你以为自己保留了 benchmark 能力，实际却在悄悄把模型推向长回答和聊天风格。

可以用一个简单公式说明：

$$
\text{effective\_weight}_i \propto \text{sample\_count}_i \times \text{avg\_tokens}_i
$$

也就是说，如果两个来源的样本条数相同，但平均长度差很多，它们对训练的“真实作用力”并不相同。

一个典型反例是：把 `198K` 的通用对话数据人为重复 `3` 倍，再和学术数据混合。表面上看，你只是想“增强对话能力”；实际上模型更频繁见到的是长回答、解释型回答、聊天式回答。直接后果通常是：

- VQA 类验证集掉点
- 答案长度变长
- 对需要短答案的任务更容易答过头
- 定位和 OCR 类任务变得不够利落

还有一种常见坑是**把统计口径写混**。例如正文里说“academic mixture 为 450K”，代码里却按照细粒度子集实际加到了 `467K`，但日志和实验表没有说明口径变化。最后别人复现时，看你的总数、比例、代码都对不上。这个问题不大，但非常伤可复现性。

下面这张检查表可以作为最低限度自检。

| 检查项 | 为什么要查 |
|---|---|
| Pretrain 是否使用过滤后的 558K 子集 | 避免对齐阶段概念分布过斜 |
| 是否保留 `source` 字段 | 方便定位哪类数据导致行为变化 |
| batch 是否按目标比例显式组装 | 避免完全随机带来的分布漂移 |
| 记录的是样本占比还是 token 占比 | 两者经常不一致 |
| GPT 指令与 ShareGPT 是否分开统计 | 两者都属于“通用”，但作用不同 |
| 多卡训练是否记录全局采样分布 | 防止单卡局部分布与全局不一致 |
| 日志里是否区分粗粒度与细粒度数据口径 | 避免总数对不上 |
| 图片路径和元数据是否统一 | 避免某类数据因 I/O 问题被实际欠采样 |

这里再补一个实际工程里常被低估的问题：**I/O 会反向影响配比**。如果 academic mixture 由多个子集组成，而且图像目录结构、文件大小、压缩格式都不一致，那么某些来源的读取延迟会更高。理论上你给这些来源设置了固定采样权重，但实际训练中如果某个数据源频繁加载失败、超时或被缓存策略排斥，它的有效占比会下降。你在日志里看到的是“训练吞吐变慢”，背后真正发生的可能是“采样配方被硬件和文件系统偷偷改写”。

因此，正式训练时至少要同时监控三类量：

| 指标 | 作用 |
|---|---|
| 样本级来源分布 | 检查采样器是否按设计工作 |
| token 级来源分布 | 检查长样本是否隐性过权重 |
| I/O 与加载失败率 | 检查数据管线是否在改变真实配比 |

如果这三类统计缺失，你很难判断模型行为变化究竟是配比设计问题，还是数据管线问题。

---

## 替代方案与适用边界

如果你的训练预算小，不必机械照搬 v1.5 的原始数字。真正该守住的不是“必须复现到最后一千条样本”，而是两件事：

1. 阶段分工不能乱  
先对齐，再任务化。

2. 目标比例必须可解释  
你要知道自己为什么偏向学术任务，或为什么偏向对话能力。

如果你的目标是做偏对话型的视觉助手，例如客服、内容创作助手、旅游讲解、商品图文生成，那么可以保留 Pretrain 的对齐思路，同时在指令微调阶段提高通用对话数据占比。这样做的收益是输出更自然、解释更完整、多轮承接更好；代价是标准化 VQA 和短答案任务可能下降。因为模型会更倾向于“把话说完整”，而不是“把答案说短且准”。

如果你的目标是做偏学术任务的系统，例如文档问答、报表理解、图表 QA、OCR 密集场景，那么可以把通用占比 $r$ 从约 `30%` 再往下压，例如压到 `25%` 左右，同时补更任务化的数据。这样做的收益是短答案能力和 benchmark 表现更稳；代价是开放式问答的自然度、对话感和解释性会弱一些。[MouSi](https://xuanjing-huang.github.io/files/mousi.pdf)

可以把三条典型路径并排看：

| 路径 | 适用场景 | 建议比例倾向 | 需要补的数据 |
|---|---|---|---|
| 偏重对话 | 通用视觉助手、客服、多轮交互 | 提高通用对话占比 | ShareGPT4V、ShareGPT4V-mix 一类数据 |
| 偏重学术 | VQA、图表理解、文档问答 | 保持学术任务为主体，适当压低 $r$ | 更任务化的视觉问答与文档理解数据 |
| 中间路线 | 既要 benchmark 也要可聊天 | 维持约 30% 通用对话 | 延续 v1.5 mix 的基本思路 |

这里最容易犯的错误，是把“偏重对话”理解成“多加一点聊天数据就行”。实际上更准确的说法是：你在重新定义模型的行为优先级。比例一旦改变，模型默认学会的回答风格、答案长度、任务偏好都会一起变。

为了让这个边界更清楚，可以用一个“能力三角”来理解：

| 能力 | 更依赖哪类数据 |
|---|---|
| 短答案准确率 | 学术任务型数据 |
| 开放式描述自然度 | 通用对话型数据 |
| 多轮上下文承接 | ShareGPT 类对话数据 |

你很难在固定算力和固定样本预算下，把三者同时推到最优。数据配比的作用，就是明确你优先保什么、允许牺牲什么。

因此，`LLaVA-Pretrain` 和 `LLaVA-v1.5` 的差别，不能简单理解为“一个 558K，一个 665K”。更准确的理解是：

- 前者在做**空间对齐**
- 后者在做**能力配方**

前者解决“视觉信息进不进得去语言模型”；后者解决“进去之后模型会表现成什么样”。前者像把电路接通，后者像决定电流最终驱动什么设备。前者没做好，后面的问答和对话都会飘；后者配比失衡，模型就会明显偏科。

---

## 参考资料

1. LLaVA Pretrain 数据卡：<https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain>  
2. LLaVA-v1.5 模型卡（PaddleNLP 转载 Hugging Face 信息）：<https://paddlenlp.readthedocs.io/zh/latest/website/liuhaotian/llava-v1.5-7b/index.html>  
3. MouSi 论文附录，表 10 给出 LLaVA-v1.5 默认 finetune 数据的细粒度拆分：<https://xuanjing-huang.github.io/files/mousi.pdf>  
4. LLaVA-1.5 665K 指令数据集镜像说明：<https://huggingface.co/datasets/kaiyuyue/llava-1.5-665k-instructions>  
5. LLaVA 项目仓库（用于理解阶段划分与训练脚本组织）：<https://github.com/haotian-liu/LLaVA>
