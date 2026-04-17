## 核心结论

DeepSeek Coder 的代码预训练策略，核心不是“把更多代码喂给模型”这么简单，而是把**训练样本的组织方式**改成更接近真实仓库。仓库级，意思是模型看到的不再只是一个孤立文件，而是一组带依赖关系的文件；FIM，意思是模型不只学习“从左往右续写”，还学习“根据前后文把中间缺口补上”。

这套设计可以概括成两件事：

1. 用 **2T tokens** 从头训练基础模型，其中约 **87% 是代码，13% 是自然语言**。自然语言这里主要是文档、注释、讨论文本，它的作用是补充接口说明、约束和任务意图。
2. 用 **repo-level 组织 + FIM 训练**，把模型能力从“当前文件补全”推进到“跨文件理解、插入和补全”。

直观看，DeepSeek Coder 不是只学“下一行最像什么”，而是在学“一个项目里，哪个文件先出现、哪些定义先被声明、一个缺失片段应该和前后代码如何对齐”。这也是它比传统单文件 next-token 代码模型更像工程工具的原因。

先看数据分布与窗口配置：

| 项目 | 数值 | 含义 |
| --- | --- | --- |
| 总训练量 | 2T tokens | 从头预训练的总语料规模 |
| 代码占比 | 87% | 约 1.74T tokens，主体是多语言代码 |
| 自然语言占比 | 13% | 约 260B tokens，包含文档和讨论文本 |
| 可靠上下文窗口 | 16K | 训练和使用时最稳定的上下文长度 |

一个最小玩具例子是：仓库里有 `core.py -> helper.py -> main.py` 的依赖链。训练时不是随机打乱，而是按依赖顺序先放 `core.py`，再放 `helper.py`，最后放 `main.py`。这样模型在生成 `main.py` 时，前面已经看过它依赖的定义。

---

## 问题定义与边界

代码模型的基本任务，看起来像语言模型：给定前缀，预测下一个 token。但真正的工程代码有两个额外难点。

第一，**代码不是单段文本，而是分布在多个文件里的约束系统**。约束系统的意思是，一个函数名、类名、配置项、测试约定，通常由多个文件共同决定。只看当前文件，信息经常不够。

第二，很多真实编辑操作不是“往后写”，而是“往中间插”。插入，白话讲，就是前后两段代码已经固定，中间缺一块逻辑，需要补上。比如函数签名和返回语句已经有了，中间缺校验逻辑；测试的 `setup` 和 `teardown` 已经写好，中间缺测试主体。

因此，这篇文章讨论的问题边界是：

| 问题 | DeepSeek Coder 的做法 | 不解决什么 |
| --- | --- | --- |
| 跨文件依赖理解 | 用 repo-level 拼接训练样本 | 不等于完整构建或执行项目 |
| 中间插入代码 | 用 FIM 训练 | 不保证所有插入都语义正确 |
| 长上下文工程代码 | 重点优化到 16K | 不意味着越长越稳 |
| 多语言代码生成 | 大规模多语言代码语料 | 不等于所有语言同等强 |

形式化地说，若仓库依赖图记为 $G=(V,E)$，其中边 $u \rightarrow v$ 表示“`v` 依赖 `u`”，那么训练样本中的文件顺序应满足：

$$
order = topo(G)
$$

也就是拓扑排序。拓扑排序，白话讲，就是任何被依赖的文件都要排在依赖它的文件前面。于是有：

$$
\forall (u \rightarrow v)\in E,\quad pos(u) < pos(v)
$$

这不是学术上的小修饰，而是样本是否合理的硬约束。如果 `main.py` 在前、`utils.py` 在后，模型在训练时会被迫学习一种错误世界观：先使用，再定义。

边界也要说清楚。DeepSeek Coder 的 repo-level 组织主要服务于**统计学习**，不是在训练阶段真的运行整个仓库。所以它学到的是“在大量工程中，依赖和实现通常如何共同出现”，而不是“这个仓库一定能跑通”。

---

## 核心机制与推导

核心机制分成两层：**仓库级组织**和 **FIM 训练**。

### 1. 仓库级组织：把“项目结构”编码进样本

传统做法常把代码文件当独立文本。这样简单，但会丢掉跨文件关系。DeepSeek Coder 更接近真实项目流：先解析仓库内依赖，再按依赖顺序拼接多个文件，构成一个更长的训练样本。

玩具例子：

- `utils.py` 定义 `normalize(text)`
- `service.py` 调用 `normalize(text)`
- `main.py` 调用 `service.run()`

合理顺序是 `utils.py -> service.py -> main.py`。如果反过来，模型先看到调用，再看到定义，跨文件统计关系会被稀释。

它背后的直觉不复杂：模型虽然没有显式“看懂”依赖图，但如果训练样本总是把定义放在调用前，它就会更容易形成稳定的表示，知道某些名称、接口和注释通常先出现在哪里。

### 2. FIM：从“续写”变成“填空”

FIM 是 Fill-in-the-Middle，白话讲是“中间填空”。传统 next-token 训练输入是前缀，目标是后续 token。FIM 则把一个连续片段拆成前缀 `P`、中段 `M`、后缀 `S`，训练模型根据 `P` 和 `S` 恢复 `M`。

DeepSeek Coder 重点使用的是 **PSM** 组织方式，即 Prefix-Suffix-Middle。形式化写作：

$$
input = \langle fim\_begin \rangle P \langle fim\_hole \rangle S \langle fim\_end \rangle M
$$

模型真正要学的是生成 `M`。这里的特殊 token，本质上是“告诉模型哪里开始填空、哪里是空洞、哪里结束提示”的标记。

如果只做从左到右训练，模型会更擅长这种任务：

```python
def parse(x):
    # 继续往后写
```

但真实 IDE 场景更像这样：

```python
def parse(x):
    # 这里缺一段
    return result
```

此时前后文都已知，模型需要补中间。这就是 FIM 的价值。

再看一个新手可理解的玩具例子。假设原代码是：

```python
def area(r):
    pi = 3.14
    return pi * r * r
```

取前缀 `P = "def area(r):\n"`，后缀 `S = "\n    return pi * r * r"`，中段 `M = "    pi = 3.14"`。FIM 样本就是把 `P` 和 `S` 放前面，把 `M` 当目标。模型学的不是“接着写什么”，而是“中间缺的那段应该是什么”。

### 3. 为什么 FIM 比例不是越高越好

如果所有样本都改成 FIM，模型的普通左到右续写能力会受损。因为很多实际任务仍是顺序生成。因此工程上会在 FIM 和标准 next-token 之间做平衡。论文和公开材料提到，DeepSeek Coder 使用了显著比例的 FIM 样本，其中常见设定是 **0.5 左右**。

可以把训练目标写成混合目标：

$$
\mathcal{L} = (1-\lambda)\mathcal{L}_{L2R} + \lambda\mathcal{L}_{FIM}
$$

其中 $\lambda \approx 0.5$。这不是公式推导出的唯一答案，而是经验上的折中：既保留顺序续写能力，又强化插入能力。

### 4. 16K 窗口为什么重要

窗口，白话讲，就是模型一次能稳定利用的上下文长度。repo-level 拼接如果太短，装不下多个相关文件；太长，又会带来质量衰减和训练代价上升。DeepSeek Coder 公开材料把 **16K** 视为更可靠的区间。

真实工程例子是一个 Python Web 服务：

- `schemas.py` 定义请求结构
- `service.py` 实现业务逻辑
- `routes.py` 暴露 HTTP 接口
- `tests/test_routes.py` 编写接口测试

如果这些核心文件能一起进入一个 16K 窗口，模型在补全 `routes.py` 时，就更可能同时参考数据结构、业务函数和测试风格。这样生成结果更容易保持命名一致、参数一致、异常处理一致。

---

## 代码实现

下面用一个可运行的简化 Python 例子，演示“依赖拓扑排序 + repo 拼接 + FIM 样本构造”的基本思路。它不是 DeepSeek Coder 的官方实现，但逻辑与公开描述一致。

```python
from collections import defaultdict, deque

def topo_sort(nodes, edges):
    graph = defaultdict(list)
    indegree = {n: 0 for n in nodes}

    for dep, user in edges:
        graph[dep].append(user)
        indegree[user] += 1

    q = deque([n for n in nodes if indegree[n] == 0])
    order = []

    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in graph[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    if len(order) != len(nodes):
        raise ValueError("dependency cycle detected")
    return order

def pack_repo(files, order):
    parts = []
    for name in order:
        parts.append(f"# file: {name}\n{files[name]}\n")
    return "\n".join(parts)

def make_fim_sample(text, start, end):
    prefix = text[:start]
    middle = text[start:end]
    suffix = text[end:]
    sample = (
        "<fim_begin>"
        + prefix
        + "<fim_hole>"
        + suffix
        + "<fim_end>"
        + middle
    )
    return sample, middle

files = {
    "core.py": "def normalize(x):\n    return x.strip().lower()\n",
    "helper.py": "from core import normalize\n\ndef clean(x):\n    return normalize(x)\n",
    "main.py": "from helper import clean\n\nprint(clean('  Hi '))\n",
}

nodes = list(files.keys())
edges = [
    ("core.py", "helper.py"),
    ("helper.py", "main.py"),
]

order = topo_sort(nodes, edges)
assert order == ["core.py", "helper.py", "main.py"]

packed = pack_repo(files, order)
assert packed.index("core.py") < packed.index("helper.py") < packed.index("main.py")

sample, target = make_fim_sample(packed, 20, 55)
assert "<fim_begin>" in sample
assert "<fim_hole>" in sample
assert "<fim_end>" in sample
assert target == packed[20:55]
```

这个例子对应三步：

1. `topo_sort` 根据依赖边确定文件顺序。
2. `pack_repo` 把多个文件串成一个更长样本，并显式标注路径。
3. `make_fim_sample` 从拼接后的文本中挖掉中段，构造 FIM 训练样本。

真实工程里，数据管线会复杂得多，通常至少包含下面这些环节：

| 阶段 | 作用 | 典型风险 |
| --- | --- | --- |
| 仓库抓取与过滤 | 去掉低质量、重复、不可解析仓库 | 噪声代码过多 |
| 依赖解析 | 推断文件间导入关系 | 动态导入难以静态分析 |
| 文件排序与拼接 | 构造 repo-level 样本 | 顺序错误会污染统计模式 |
| 去重 | 降低重复训练样本比例 | 近似重复难完全消除 |
| FIM 变换 | 生成中间填空样本 | 切分位置不合理会破坏语义 |

真实工程例子可以想成一个中型后端服务。训练样本可能不是单独的 `service.py`，而是：

1. `models.py`
2. `repository.py`
3. `service.py`
4. `api.py`
5. `tests/test_api.py`

如果在 `service.py` 中间挖掉“参数校验 + 异常处理”这段，模型要根据前缀和后缀补齐逻辑，就必须同时理解上游数据模型和下游 API 返回格式。这正是 repo-level + FIM 联合作用的地方。

---

## 工程权衡与常见坑

这套策略效果强，但工程成本也高，尤其高在“数据组织”而不是“模型结构”。

第一类权衡是**窗口长度**。16K 可以装下更多文件上下文，但样本构造、训练显存、推理时延都会上升。窗口更长不自动代表更好，因为有效信息密度会下降，模型也可能把远处无关代码混进当前决策。

第二类权衡是**依赖排序的准确性**。拓扑排序只有在依赖图足够可信时才有意义。现实仓库会有：

- 动态导入
- 条件导入
- 生成代码
- 多语言混合调用

这些情况会让静态解析不完整。于是工程上通常只能做到“多数情况下合理”，而不是“完全精确”。

第三类权衡是 **FIM 比例**。比例太低，模型插入能力弱；比例太高，普通续写能力下降。0.5 左右的设定，本质是经验上的平衡点。

常见坑可以直接列成表：

| 坑 | 具体表现 | 规避方法 |
| --- | --- | --- |
| 文件顺序错乱 | 模型先见调用后见定义，跨文件补全漂移 | 使用依赖图拓扑排序 |
| 样本超过可靠窗口 | 远距离信息利用差，生成开始跑偏 | 训练和推理都控制在稳定窗口内 |
| FIM 切分位置不自然 | 挖掉半个标识符或半个语句，样本失真 | 尽量按语法边界切分 |
| 路径信息缺失 | 模型不知道代码来自哪个文件 | 拼接时加入文件路径注释 |
| 仓库重复太多 | 模型记忆重复模板，泛化下降 | 做 repo-level 去重 |

一个很典型的坑是“只做拼接，不做路径标注”。如果把多个文件简单连起来，模型知道有多段代码，但未必知道边界和来源。路径标注的作用很直接：把“这段代码属于哪个文件”变成显式信息。对跨文件补全来说，这常常比多喂几行普通代码更有价值。

---

## 替代方案与适用边界

DeepSeek Coder 的策略不是唯一方案，只是对“工程代码生成”较有效的一组选择。

第一种替代方案是**纯 next-token 单文件训练**。它实现简单、吞吐高、数据构造成本低。对于单脚本补全、小规模函数生成、竞赛题代码，这类方案已经够用。缺点是跨文件依赖能力弱，中间插入能力也弱。

第二种替代方案是**RAG 式检索补全**。RAG，白话讲，就是不把整个仓库都塞进参数里，而是在推理时临时检索相关文件，再拼进提示词。它的优点是灵活，不必在预训练阶段就把所有依赖关系编码进去；缺点是检索质量直接决定上限，而且延迟更高。

第三种替代方案是**其他 FIM 排列方式**，例如 SPM。不同排列会改变模型看到前后文的顺序，对某些任务可能有细微影响，但总体思想一致：让模型学会“中间填空”。

可以用一个对照表总结：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 单文件 Next-Token | 简单、便宜、训练稳定 | 跨文件理解弱 | 小脚本、函数续写 |
| Repo-Level + FIM | 跨文件补全强，贴近 IDE 编辑 | 数据管线复杂，成本高 | 工程项目、仓库级补全 |
| 检索增强生成 | 不必全靠参数记忆，更新灵活 | 依赖检索质量和推理链路 | 大仓库在线辅助 |
| 其他 FIM 变体 | 可针对任务调整 | 调参复杂，收益未必稳定 | 特殊插入场景实验 |

适用边界也要明确。若你的目标只是“给 LeetCode 题续写函数”，repo-level 组织未必划算；若你的目标是“在已有后端仓库里补一个中间层函数，并保持与测试、模型、路由一致”，那么 DeepSeek Coder 这类策略就明显更合适。

一句话概括边界：**问题越像真实工程编辑，repo-level + FIM 的价值越大；问题越像孤立文本续写，简单 next-token 越经济。**

---

## 参考资料

- DeepSeek Coder 官网：https://deepseekcoder.github.io/
- DeepSeek-Coder GitHub README：https://github.com/deepseek-ai/deepseek-coder
- DeepSeek Coder Architecture（DeepWiki）：https://deepwiki.com/deepseek-ai/DeepSeek-Coder/2.1-architecture
- DeepSeek Coder Code Insertion（DeepWiki）：https://deepwiki.com/deepseek-ai/DeepSeek-Coder/3.2-code-insertion
- CMU CodeGen 2024 课程资料：https://cmu-codegen.github.io/s2024/static_files/codegen_s2024_8_data.pdf
- DeepSeek-Coder 论文整理页：https://ytx-readings.github.io/AI/papers/LLM/DeepSeek-Coder.pdf
- Berkeley EECS 2025 LLM for Code 报告：https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-50.pdf
- MarkTechPost 对 DeepSeek-Coder-V2/FIM 的整理：https://www.marktechpost.com/wp-content/uploads/2024/06/paper.pdf
