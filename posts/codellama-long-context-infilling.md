## 核心结论

CodeLlama 的长上下文填充训练可以概括为两件事同时成立：一是基于 Llama 2 再做约 500B token 的代码续训，让模型先学会“代码分布”；二是在长上下文微调（LCFT，Long Context Fine-Tuning，白话说就是先拿更长序列把模型再训一遍）阶段，把训练长度拉到 16K，并把 RoPE 的基频 $\theta$ 从 10000 调到 1000000，用更长的位置周期支撑推理时的 100K 级上下文外推。

这不是“模型突然记忆力变强”这么简单，而是位置编码和训练分布被一起改了。前者解决“位置表示在超长距离下转得太快”的问题，后者解决“模型从来没见过长序列所以不会用”的问题。两者缺一不可。

Fill-in-the-Middle，简称 FIM，白话说就是“不是只接着末尾续写，而是在前缀和后缀之间补中间一段”。CodeLlama 相关工程实践里，PSM 和 SPM 是两种常见拼接格式：PSM 是 `Prefix-Suffix-Middle`，SPM 是 `Suffix-Prefix-Middle`。训练时混合这些格式，可以让模型既支持 IDE 式中间补全，也尽量保留推理阶段的 KV cache 复用能力。

“新手版”理解可以直接用一个例子：把整个仓库里相关文件的 `import`、注释、接口定义和 docstring 按 PSM 结构输入模型，在中间留一个空洞让它补函数体。如果上下文足够长，模型就能像 IDE 一样综合多个文件的信息去补全，而不是只看当前文件的前几十行。

| 能力维度 | 原始 Llama 2 基线 | CodeLlama 的 LCFT 阶段 | 目标推理能力 |
|---|---|---|---|
| 主要语料 | 通用语料 | 代码与代码相关语料续训 | 面向代码生成与理解 |
| 长度适配 | 原始上下文窗口 | 16K 长序列继续微调 | 外推到 100K 级 |
| RoPE 基频 $\theta$ | 10000 | 调大到 1000000 | 降低远距离位置退化 |
| FIM 支持 | 非重点 | 训练中显式加入 | 支持 infill 与跨文件补全 |

---

## 问题定义与边界

问题的核心不是“参数量不够”，而是“上下文覆盖不到”。传统代码模型哪怕能写出局部正确的函数，也常常不知道另一个文件里已经定义了什么接口，不知道几十 KB 之前的注释还在约束当前实现，更不知道后文已经写好的调用方式。

这类问题有两个边界。

第一是长度边界。很多模型实际有效上下文集中在 16K 到 32K。仓库级输入很容易超过这个范围，尤其是把多个文件、依赖说明、API 注释、测试样例一起塞进来时。

第二是注意力退化边界。注意力可以理解为“模型在生成当前 token 时回头看前文的能力”。即使理论窗口没超限，距离过远的位置也可能因为位置编码周期过短而表现变差，出现“越远越弱”的现象。

“新手版”例子：你把一个函数的 docstring 放在文件开头，50KB 之后才出现参数合法范围的说明。短上下文模型通常有两种失败方式：一种是直接截断，后面的说明根本没进模型；另一种是虽然都进了模型，但生成函数体时几乎不再参考最前面的约束，最后写出和文档不一致的实现。

| 问题 | 传统模型常见表现 | 目标能力 |
|---|---|---|
| 单文件很长 | 前部注释被遗忘 | 长距离依赖仍可引用 |
| 多文件联合推理 | 只能看局部片段 | 覆盖整仓库关键上下文 |
| 中间补全 | 只擅长从末尾续写 | 能利用前缀和后缀补中间 |
| 远距离一致性 | 接口名、类型约束漂移 | 保持跨文件约束一致 |

这里还要划清边界。100K 长上下文不等于“无限理解整个仓库”。如果仓库本身有几百万 token，仍然需要筛选输入。长上下文只是把“可原生关注的信息量”从几十页代码提升到一个更接近实际工程的规模，而不是替代检索、摘要和依赖分析。

---

## 核心机制与推导

RoPE，Rotary Positional Embedding，白话说就是“把位置信息编码成不同频率的旋转角度”，让注意力知道第几个 token 在哪里。它的关键频率项常写作：

$$
f_i=\theta^{-2i/d}
$$

其中：

- $d$ 是嵌入维度；
- $i$ 是第 $i$ 个频率通道；
- $\theta$ 是基频底数；
- $f_i$ 越大，旋转越快，周期越短；
- 周期越短，远距离 token 更容易出现角度快速绕圈，导致位置区分在超长范围内变得不稳定。

原始 Llama 2 常用 $\theta=10000$。CodeLlama 的长上下文思路是在 LCFT 之后把它提高到 $\theta=1000000$。因为：

$$
f_i=\theta^{-2i/d}
$$

当 $\theta$ 变大时，除极低频通道外，大多数通道的频率都会下降，也就是旋转更慢、周期更长。直观上，相当于把位置编码里的“波长”拉长，让模型在更远距离上仍能保持可分辨的相位关系。

“新手版”可以把它想成几根不同周期的波。原来这些波在 16K 以内变化还算合适，但到了 100K 时转得太快，远处的位置模式已经挤在一起。把基频调大，就是把这些波的波长拉长，让 100K 范围内的位置信号不至于过密重叠。

下面给一个玩具例子。假设我们只看 $d=8$ 时的几个通道：

| $d$ | $i$ | $\theta=10000$ 时 $f_i$ | $\theta=1000000$ 时 $f_i$ | 影响 |
|---|---:|---:|---:|---|
| 8 | 0 | 1 | 1 | 最低通道不变 |
| 8 | 1 | $10^{-1}$ | $10^{-1.5}$ | 频率下降，周期变长 |
| 8 | 2 | $10^{-2}$ | $10^{-3}$ | 更适合长距离 |
| 8 | 3 | $10^{-3}$ | $10^{-4.5}$ | 高频通道拉得更长 |

这里“线性外推”不是说所有内部动力学都严格线性，而是指位置可表示范围随着基频扩大而显著延长，工程上表现为训练在 16K，推理能稳定外推到远高于 16K 的长度。

还要注意，单改 RoPE 不够。因为模型如果从未见过长输入，即使位置编码能表示更远位置，也不代表它会在这些位置上正确地分配注意力。所以 LCFT 先用 16K 长序列继续训练，目的是让模型适应“长输入下该怎么读代码”。然后再配合更大的 $\theta$，把可推理长度继续往 100K 推。

一个更贴近工程的真实例子是跨文件补全。你在 `router.py` 里写：

- 前缀给出 HTTP 路由定义、参数校验器和 service 接口；
- 后缀给出测试文件里对返回 JSON 结构的断言；
- 中间空着控制器函数体。

如果模型只能看 8K 或 16K，它可能根本装不下这些辅助信息。若上下文足够长，再加上 FIM，它就能同时参考接口声明和测试约束，在中间补出更一致的函数体。

---

## 代码实现

工程实现可以拆成两段：训练侧做 LCFT 和 RoPE 参数调整，推理侧按 FIM 格式组装输入。

训练侧的核心不是“重新训练一个模型”，而是保留现有 tokenizer 和主体权重，只修改上下文长度配置、位置编码基频和数据打包方式。伪代码如下：

```python
import math

def rope_freq(theta: float, i: int, d: int) -> float:
    return theta ** (-2 * i / d)

# 玩具例子：更大的 theta 会让大多数通道频率更低
f_old = rope_freq(10_000, 2, 8)
f_new = rope_freq(1_000_000, 2, 8)

assert f_new < f_old
assert abs(rope_freq(10_000, 0, 8) - 1.0) < 1e-12
assert abs(rope_freq(1_000_000, 0, 8) - 1.0) < 1e-12

def build_training_config():
    return {
        "base_model": "llama2",
        "tokenizer": "reuse",
        "context_length": 16_384,
        "rope_theta": 1_000_000,
        "continue_pretrain_tokens": 500_000_000_000,
        "fim_rate": 0.5,
        "fim_formats": ["PSM", "SPM"],
    }

cfg = build_training_config()
assert cfg["context_length"] == 16_384
assert cfg["rope_theta"] == 1_000_000
assert 0.0 <= cfg["fim_rate"] <= 1.0
```

如果把它翻成更接近训练脚本的参数片段，大致是这样：

```python
train_args = {
    "model_name": "codellama-base",
    "context_window": 16384,
    "rope_theta": 1_000_000,
    "tokenizer_mode": "reuse_llama2",
    "objective": "causal_lm_plus_fim",
    "fim_rate": 0.5,
    "fim_mix": {"PSM": 0.5, "SPM": 0.5},
}
```

推理侧要解决两个目标：一是让模型知道哪里是前缀、哪里是后缀、哪里需要补；二是尽可能保留缓存复用。下面给一个简单的 FIM/PSM 数据构造示例：

```python
FIM_PREFIX = "<PRE>"
FIM_SUFFIX = "<SUF>"
FIM_MIDDLE = "<MID>"

def build_psm(prefix: str, suffix: str) -> str:
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

def build_spm(prefix: str, suffix: str) -> str:
    return f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"

repo_imports = "from service.user import UserService\nfrom schemas import UserOut\n"
docstring = '"""Return a user profile by id."""\n'
prefix = repo_imports + docstring + "def get_user_profile(user_id: int):\n    "
suffix = "\n\n# test expects {'id': int, 'name': str}\n"

psm_prompt = build_psm(prefix, suffix)
spm_prompt = build_spm(prefix, suffix)

assert psm_prompt.startswith("<PRE>")
assert "<MID>" in psm_prompt
assert spm_prompt.startswith("<SUF>")
```

“新手版”理解这个输入就够了：前缀里放你已经写好的内容，后缀里放后面不允许被破坏的内容，`<MID>` 代表让模型在这里补。这样模型不是盲写，而是在“前后夹住”的条件下生成中间代码。

一个真实工程例子可以是 IDE 插件：

1. 收集当前文件前 200 行代码；
2. 收集目标符号依赖的其他文件 `import`、类型定义和 docstring；
3. 收集测试文件里与当前函数相关的断言；
4. 用 PSM 或 SPM 结构拼成一个 32K 到 100K 的 prompt；
5. 请求模型在 `<MID>` 后生成函数体。

关键超参数可以这样看：

| 超参数 | 典型值 | 作用 |
|---|---:|---|
| `context_length` | 16384 | LCFT 训练长度 |
| `rope_theta` | 1000000 | 拉长 RoPE 周期 |
| `fim_rate` | 0.5 左右 | 控制多少样本做中间补全 |
| `PSM/SPM` 比例 | 50/50 常见 | 减少格式偏置 |
| 推理上下文 | 可到 100K 级 | 覆盖更长代码上下文 |

流程可以概括为：

| 阶段 | 输入 | 动作 | 输出 |
|---|---|---|---|
| 续训 | 代码与相关语料 | 学代码分布 | 代码能力增强 |
| LCFT | 16K 长序列 | 适应长上下文 | 长输入稳定性增强 |
| RoPE 调整 | $\theta=10^6$ | 延长位置周期 | 100K 级外推能力 |
| FIM 推理 | prefix + suffix | 中间补全 | IDE 式 infill |

---

## 工程权衡与常见坑

第一个常见坑是把“支持 FIM”理解成“训练时只喂 FIM 就行”。这通常不成立。只用一种格式会产生格式偏置。只用 PSM，模型会更习惯前缀在前、后缀在中间的排序；只用 SPM，缓存复用和常规续写路径可能受影响；只用纯 FIM 而忽略常规左到右续写，又会让普通生成能力下降。

“新手版”例子：如果你只用 PSM，某些场景里模型会更依赖前半页内容，后缀约束虽然存在，但利用得不稳定，补出来的中间代码可能和后面的测试或调用方式冲突。它不是“看不到后缀”，而是“对后缀的使用不够稳”。

第二个坑是高估 100K 的可用性。理论能放进 100K token，不代表每个 token 都同等有价值。仓库里大量无关代码会稀释注意力，导致真正关键的接口说明和测试断言被埋没。长上下文要配合输入选择，而不是把整个项目机械塞满。

第三个坑是显存和延迟。KV cache 可以理解为“模型为了后续生成保存的历史键值状态”。上下文越长，KV cache 越大。100K 级上下文会显著增加显存占用和首 token 延迟。很多 IDE 实时补全场景其实更在意 100 毫秒到 1 秒级响应，而不是最大可读长度。

第四个坑是注意力稳定性监控不足。做长上下文推理时，不要只看最终代码是否可运行，还要看：

| 方案 | infill 质量 | KV cache 复用 | 风险 |
|---|---|---|---|
| 仅 PSM | 中等 | 较好 | 后缀利用不稳定 |
| PSM+SPM+FIM 混合 | 较好 | 较好 | 训练实现更复杂 |
| 仅 FIM | 某些任务好 | 一般 | 常规续写可能退化 |

建议监控指标包括：

- 注意力权重是否过度集中在最近 token；
- cache hit rate 是否因为格式切换而下降；
- 长距离约束是否在输出中被遵守；
- 100K 输入下显存和首 token 延迟是否可接受。

如果你在真实服务里部署，一个很实用的检查方式是准备“带后缀约束的回归样本”。例如后缀明确要求函数返回 `UserOut`，而模型补出的中间逻辑返回了字典或元组，这就说明格式训练或长上下文利用有问题。

---

## 替代方案与适用边界

长上下文原生注意力不是唯一方案。很多任务里，更便宜的办法是检索增强或滑动窗口。

检索增强，RAG，白话说就是“先查资料，再把查到的片段喂给模型”。它不要求模型原生看到 100K token，而是靠外部检索把关键片段挑出来。优点是成本低，缺点是跨片段关系和全局顺序信息弱一些。

滑动窗口，白话说就是“把长输入切成多个重叠小窗，分批处理”。它适合摘要、扫描和粗筛，不适合要求强全局一致性的精确补全。

“新手版”例子：如果你的需求是 IDE 里实时提示，不一定非要上 100K 原生上下文。常见可行方案是 `32K + RAG + chunking`。本质上就是先把长文章分段，再按需查最相关的几段，而不是每次都全文通读。

| 方案 | 适用边界 | trade-off |
|---|---|---|
| 原生 100K 长上下文 | 跨文件补全、整段代码重写、全局一致性要求高 | 显存和延迟高 |
| 32K + RAG | 实时问答、局部补全、知识定位 | 检索错了就答偏 |
| Sliding Window | 长文扫描、摘要、规则检查 | 全局依赖容易断裂 |
| 静态分析 + 小上下文模型 | 类型约束明确、结构化代码库 | 召回强，但生成自由度低 |

下面给一个简化版 sliding window + prompt assembly 伪代码：

```python
def assemble_prompt(chunks, query, topk=3):
    ranked = sorted(chunks, key=lambda x: overlap_score(x, query), reverse=True)
    selected = ranked[:topk]
    return "\n\n".join(selected) + "\n\nTask:\n" + query

def overlap_score(chunk, query):
    chunk_words = set(chunk.lower().split())
    query_words = set(query.lower().split())
    return len(chunk_words & query_words)

chunks = [
    "user service defines get_user_profile and return schema",
    "payment module unrelated",
    "tests expect id and name in response",
]
query = "implement get_user_profile return schema id name"

prompt = assemble_prompt(chunks, query)
assert "payment module unrelated" not in prompt
assert "get_user_profile" in prompt
```

它适合“找相关信息再生成”，但不等价于原生 100K 注意力。后者的价值在于，模型可以在一次前向中直接建模更长距离的交互关系，而不只是处理被检索器筛出来的局部片段。

所以适用边界很明确：

- 你需要整仓库级、前后文强耦合的补全时，长上下文更合适。
- 你需要低延迟交互式提示时，RAG 或较短窗口通常更实用。
- 你需要极高吞吐、成本敏感的批处理时，滑窗和分层摘要往往更划算。

---

## 参考资料

| 来源 | 核心内容 | 链接说明 |
|---|---|---|
| Meta / InfoQ 报道 | CodeLlama 基于 Llama 2 做约 500B token 代码续训，并强调长上下文与代码任务能力 | InfoQ: Meta 发布 Code Llama，介绍训练规模与 100K 级上下文能力 |
| CSDN RoPE 解析 | 解释 RoPE 频率公式 $f_i=\theta^{-2i/d}$，以及将基频从 10000 调到 1000000 的外推思路 | CSDN: CodeLlama 长上下文与 RoPE base 调整解析 |
| FIM 实践资料 | 讨论 PSM/SPM/FIM 训练格式混合，对中间补全效果和工程推理格式的影响 | Hugging Face / HIT-SCIR Abacus-FIM 页面，以及相关 FIM 工程总结 |
| 其他中文工程解读 | 补充 PSM/SPM 混合、KV cache 复用与实际推理格式的经验 | linsight 等文章，对 FIM 格式与使用方式有更工程化说明 |

1. InfoQ 对 Meta CodeLlama 的报道，适合了解 500B token 续训、代码任务定位和长上下文能力。
2. CSDN 的 RoPE 公式解析，适合理解为什么 $\theta$ 从 10000 提到 1000000 会改善长距离位置表示。
3. Hugging Face 上与 FIM 相关的资料，适合理解 PSM、SPM 和中间补全格式的训练差异。
4. 如果要进一步核对实现细节，应以 Meta 官方论文和模型说明为准，中文文章更适合做机制入门与工程直觉建立。
