## 核心结论

DeepSeek-MoE 的核心不是“把模型做大”，而是“把 FFN 拆细，再稀疏组合”。MoE，Mixture of Experts，直译是“专家混合”，可以把它理解成：每个 token 不再调用整块前馈网络，而是只调用少数几个子网络。这里的 FFN，Feed-Forward Network，指 Transformer 每层里注意力之后的那块两层感知机，它通常占了模型里很大一部分参数量。

DeepSeek-MoE 在传统 MoE 基础上做了两件事，而且这两件事必须放在一起看：

1. 把原来 $N$ 个大专家继续切成 $mN$ 个小专家，也就是细粒度专家分割。
2. 固定保留 $K_s$ 个共享专家，所有 token 都会经过它们，其余专家再由路由器按需选择。

这让每个 token 同时拿到两类能力：

- 一类是所有任务都常用的“基础能力”，由共享专家稳定提供。
- 一类是按上下文临时组合的“专项能力”，由路由专家动态提供。

直观地说，原来是“8 个大书架里挑 2 个”；现在是“64 个小书架里挑 8 个，其中 2 个书架永远开放”。组合空间变大了，但每次真正参与计算的参数仍然有限，所以总参数可以上去，单次前向成本不必按比例上涨。

常见写法是：

$$
h_{\text{MoE}}(x)=\sum_{i=1}^{K_s}\mathrm{FFN}^{(s)}_i(x)+\sum_{j=1}^{E_r}\tilde g_j(x)\,\mathrm{FFN}^{(r)}_j(x)
$$

其中：

- $x$ 是当前 token 的隐状态向量
- $K_s$ 是共享专家数
- $E_r$ 是路由专家数
- $\tilde g_j(x)$ 是路由后保留下来的权重
- 上标 $(s)$ 表示 shared expert，上标 $(r)$ 表示 routed expert

如果把“打分”和“top-k 截断”写完整，通常是：

$$
s_j(x)=\sigma(w_j^\top x+b_j)
$$

$$
\mathcal{T}(x)=\operatorname{TopK}\left(\{s_j(x)\}_{j=1}^{E_r},\,k-K_s\right)
$$

$$
\tilde g_j(x)=
\begin{cases}
\dfrac{s_j(x)}{\sum\limits_{l\in\mathcal{T}(x)} s_l(x)}, & j\in\mathcal{T}(x) \\
0, & j\notin\mathcal{T}(x)
\end{cases}
$$

这里的门控，gating，意思就是“给每个专家打分，并决定谁上场”。DeepSeek-MoE 常用归一化 sigmoid，而不是传统 softmax。差别在于：

- `softmax` 是强竞争：一个专家分高，别的专家分数会被同步压低。
- `sigmoid + normalize` 是先各自打分，再对保留下来的专家做归一化，竞争更缓和。

对训练来说，这通常意味着两个结果：

| 门控方式 | 竞争关系 | 常见结果 |
|---|---|---|
| softmax | 强竞争 | 更容易把流量推向少数专家 |
| normalized sigmoid | 弱一些 | 专家更容易学出稳定分工 |

在公开资料中，DeepSeek-MoE 16B 的典型对比是：总参数约 16.4B、每 token 激活参数约 2.8B，而对比的 LLaMA2 7B 是 6.7B 全激活；4k token FLOPs 约为 74.4T 对 187.9T。意思很直接：总参数更大，但每次前向真正算的更少。它追求的不是“单个 token 一次算更多”，而是“让同样的计算预算调用更合适的参数子集”。

---

## 问题定义与边界

传统 MoE 的问题不是“专家不够多”，而是“专家太粗”。一个大专家往往混合很多能力，比如代码、数学、常识、语法、跨语言对齐全挤在同一个 FFN 里。这样做表面上也能稀疏激活，但专家分工不够清楚，组合价值会被削弱。

可以把问题拆成两类：

| 问题 | 现象 | 后果 |
|---|---|---|
| 知识混杂 | 单个专家承担多个领域 | 专家不够专一，token 选中它时拿到的是“混合包”而不是“专项能力” |
| 知识冗余 | 多个被激活专家都学到同样基础能力 | 参数利用率低，很多参数被重复存通用知识 |

DeepSeek-MoE 的边界也很明确：它解决的是“如何在相同或接近的激活预算下，让专家组合更细、更稳定”，而不是彻底消灭路由开销，也不是保证所有场景都比 dense 更优。

这个边界很重要，因为新手常把三件事混在一起：

| 命题 | DeepSeek-MoE 是否直接解决 |
|---|---|
| 扩大总参数量 | 部分解决，但不是重点 |
| 降低单 token 激活计算 | 是 |
| 消除系统复杂度 | 否，MoE 系统复杂度通常更高 |

一个面向新手的玩具例子：

原来有 8 个大专家，每次选 2 个。现在令 $m=4$，把每个专家切成 4 份，总数就变成 32 个小专家；如果保持激活参数量接近不变，就可以改成每次选 8 个更小的专家。这样一句关于“Python 多线程 I/O”的 token 序列，可以同时调到“编程语法”“操作系统”“英文标识符”“技术写作”几个更窄的子能力，而不必把这些能力绑定在两个大专家里。

这里可以顺手说明两个经常让新手困惑的点：

| 术语 | 含义 |
|---|---|
| token | 模型切分后的最小处理单位，不一定等于一个汉字或一个单词 |
| 激活参数 | 当前这次前向里真的参与乘加运算的参数，不等于模型总参数 |

真实工程边界则更清楚。DeepSeek-MoE 16B 的目标不是取代所有 7B dense 模型，而是在大规模训练和多任务场景下，用更低 FLOPs 获得接近甚至更好的效果。常被引用的一组对比是：

| 模型 | 总参数 | 激活参数 | FLOPs / 4k tokens |
|---|---:|---:|---:|
| LLaMA2 7B | 6.7B | 6.7B | 187.9T |
| DeepSeek-MoE 16B | 16.4B | 2.8B | 74.4T |

这里的“激活参数”可以理解为“这次前向真正参与计算的参数规模”。所以 DeepSeek-MoE 的核心收益来自稀疏激活，不是来自参数总量本身。总参数变大只是前提，真正产生效率收益的是“只激活其中一部分”。

---

## 核心机制与推导

先看结构。设传统 MoE 有 $N$ 个专家，每次激活 $K$ 个。DeepSeek-MoE 引入细粒度因子 $m$ 后：

- 专家总数变为 $E=mN$
- 每个专家尺寸缩小到原来的约 $1/m$
- 每次激活专家数变为 $k=mK$
- 其中固定有 $K_s$ 个共享专家始终激活
- 剩余 $k-K_s$ 个由路由器从 routed experts 中选出

这等于把“少量大块能力”改成“更多小块能力的拼装”。

如果把“计算量近似不变”写成公式，会更容易理解。设原始单个 FFN 的中间维度是 $d_{\text{ff}}$，模型维度是 $d$。Dense FFN 近似成本可以写成：

$$
\text{Cost}_{\text{dense}} \propto 2 d\, d_{\text{ff}}
$$

传统 top-$K$ MoE 若有 $N$ 个完整专家、每次激活 $K$ 个，则近似为：

$$
\text{Cost}_{\text{MoE}} \propto K \cdot 2 d\, d_{\text{ff}}
$$

DeepSeek-MoE 把每个专家切成原来的 $1/m$，每次激活 $mK$ 个，则：

$$
\text{Cost}_{\text{DeepSeek-MoE}} \propto mK \cdot 2 d\, \frac{d_{\text{ff}}}{m}
= K \cdot 2 d\, d_{\text{ff}}
$$

这就是它最关键的直觉：专家数量更多了，但单专家更小；激活专家数也更多了，但总乘加量可以保持在同一量级。也就是说，它不是免费午餐，而是把同一预算重新分配成了“更多、更细的模块”。

一个最常见的数值例子是：

- $m=4$
- $N=16$
- $K=2$
- $K_s=2$

那么：

- 总专家数 $E=mN=64$
- 总激活数 $k=mK=8$
- 共享专家固定 2 个
- 路由专家再选 $8-2=6$ 个
- 每个专家大小约是原始 FFN 的 $1/4$

这就是“64 个专家里每次用 8 个”的来源。它的关键不是 64 这个数本身，而是更高的组合自由度。组合数从“16 选 2”变成“62 个路由专家里选 6，再加 2 个共享”，理论上能表达的模式明显更多。

如果只看组合数量，二者差别非常大：

$$
\binom{16}{2}=120
$$

而细粒度版本在共享专家固定后，路由部分的组合数是：

$$
\binom{62}{6}=61,474,519
$$

这个数不代表模型真的会均匀用到所有组合，但它说明了一点：在相近的计算预算下，可供 token 选择的能力拼装方式大得多。

为什么共享专家有意义？因为很多 token 都需要通用语言能力，比如语法、标点、基础事实、常见句法模式、跨任务共通知识。如果这些内容也完全交给路由专家学习，就容易出现每个 routed expert 都学一点“底层常识”，最后专项能力被通用能力挤占。共享专家就是把这部分“底座能力”单独拿出来，强制所有 token 共用。

可以把一个 token 经过 MoE 层的流程拆成 4 步：

| 步骤 | 发生什么 |
|---|---|
| 1. 共享专家计算 | 所有 token 都过共享专家，拿到稳定底座表示 |
| 2. 路由器打分 | 对所有 routed experts 生成分数 |
| 3. top-k 选择 | 只保留少数 routed experts，维持稀疏性 |
| 4. 加权合并 | 把共享输出和路由输出相加，送往后续层 |

玩具例子可以这样看。假设一句话是“Python 里的 GIL 为什么影响 CPU 密集任务”。它至少包含三类信息：

- “Python”“GIL”需要编程运行时语境
- “CPU 密集”需要系统性能语境
- 中文问句结构又需要基础语言能力

共享专家稳定提供“基础语言能力”，路由专家再分别补“编程运行时”“并发模型”“性能分析”。这样比把全部内容压进 2 个大专家更容易形成清晰分工。

再看门控。softmax 门控的问题是强竞争。若 logits 为 $z_j$，则：

$$
p_j=\frac{e^{z_j}}{\sum_l e^{z_l}}
$$

任何一个 $z_j$ 升高，别的专家概率都会被连带压缩。对分类任务这很自然，但对专家分工未必理想，因为 MoE 不是在做“只能选一个答案”的硬竞争，而是在做“挑几个合适模块一起干活”。

归一化 sigmoid 则先独立打分，再统一归一化，形式上是：

$$
s_j=\sigma(w_j^\top x+b_j), \quad
\tilde g_j=\frac{s_j}{\sum_{l\in\mathcal{T}(x)} s_l}
$$

它的含义可以直接读成：

1. 每个专家先独立回答“我是否适合处理这个 token？”
2. 只保留前若干个最合适的专家
3. 在保留下来的专家内部再归一化权重

如果再做 top-k，只保留前 $k-K_s$ 个 routed experts，那么共享专家不参与竞争，基础知识不会被“挤掉”；路由专家之间的竞争也比 softmax 更缓和。对于训练稳定性，这通常比“全员强竞争”更容易维护专家利用率。

真实工程例子则接近 DeepSeek-MoE 16B 的配置：在多层 Transformer 中，把多数层的 FFN 换成 MoE 层，每层使用少量共享专家和更多细粒度专家，每 token 只激活少量专家，总激活参数约 2.8B。这样做的价值不是某一层变得神奇，而是所有层叠加后，专家利用率、训练稳定性和算力效率一起改善。

---

## 代码实现

下面给一个可运行的简化版 Python 实现。它不是完整训练代码，但会把四个关键点写全：

1. 共享专家始终参与计算
2. 路由专家先做 `sigmoid`
3. 再做 `top-k`
4. 对保留下来的权重重新归一化后合并输出

为了避免依赖第三方库，下面的实现只用 Python 标准库，输入是一个向量，专家是一个两层 MLP 的玩具版本。

```python
import math
from dataclasses import dataclass
from typing import List, Dict


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def silu(x: float) -> float:
    return x * sigmoid(x)


def dot(a: List[float], b: List[float]) -> float:
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
    return [dot(row, vec) for row in mat]


def vec_add(a: List[float], b: List[float]) -> List[float]:
    assert len(a) == len(b)
    return [x + y for x, y in zip(a, b)]


def vec_scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]


@dataclass
class ToyFFN:
    w1: List[List[float]]
    b1: List[float]
    w2: List[List[float]]
    b2: List[float]

    def forward(self, x: List[float]) -> List[float]:
        h = vec_add(matvec(self.w1, x), self.b1)
        h = [silu(v) for v in h]
        y = vec_add(matvec(self.w2, h), self.b2)
        return y


def router_logits(x: List[float], router_w: List[List[float]], router_b: List[float]) -> List[float]:
    scores = []
    for w, b in zip(router_w, router_b):
        scores.append(dot(w, x) + b)
    return scores


def topk_indices(values: List[float], k: int) -> List[int]:
    assert 0 < k <= len(values)
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]


def deepseek_moe_forward(
    x: List[float],
    shared_experts: List[ToyFFN],
    routed_experts: List[ToyFFN],
    router_w: List[List[float]],
    router_b: List[float],
    k_total: int,
) -> Dict[str, object]:
    k_shared = len(shared_experts)
    k_routed = k_total - k_shared

    assert k_routed > 0
    assert len(routed_experts) == len(router_w) == len(router_b)

    # 1. shared experts: 始终参与
    output_dim = len(shared_experts[0].b2)
    shared_out = [0.0] * output_dim
    for expert in shared_experts:
        shared_out = vec_add(shared_out, expert.forward(x))

    # 2. routed experts: 先各自打分
    logits = router_logits(x, router_w, router_b)
    raw_scores = [sigmoid(v) for v in logits]

    # 3. 只保留 top-k routed experts
    kept = topk_indices(raw_scores, k_routed)
    kept_scores = [raw_scores[i] for i in kept]
    score_sum = sum(kept_scores)
    routed_weights = {i: raw_scores[i] / score_sum for i in kept}

    # 4. 合并 routed experts 输出
    routed_out = [0.0] * output_dim
    for i, weight in routed_weights.items():
        expert_out = routed_experts[i].forward(x)
        routed_out = vec_add(routed_out, vec_scale(expert_out, weight))

    final_out = vec_add(shared_out, routed_out)

    return {
        "output": final_out,
        "logits": logits,
        "raw_scores": raw_scores,
        "selected_routed_experts": kept,
        "selected_weights": routed_weights,
        "shared_expert_count": k_shared,
        "routed_expert_count": len(routed_experts),
    }


def build_toy_ffn(scale: float) -> ToyFFN:
    return ToyFFN(
        w1=[
            [0.20 * scale, -0.10 * scale, 0.05 * scale],
            [0.00 * scale, 0.15 * scale, 0.10 * scale],
            [0.12 * scale, 0.08 * scale, -0.04 * scale],
            [-0.05 * scale, 0.07 * scale, 0.20 * scale],
        ],
        b1=[0.01 * scale, -0.02 * scale, 0.00, 0.03 * scale],
        w2=[
            [0.10 * scale, 0.05 * scale, -0.02 * scale, 0.04 * scale],
            [-0.03 * scale, 0.08 * scale, 0.06 * scale, 0.02 * scale],
            [0.07 * scale, -0.01 * scale, 0.03 * scale, 0.09 * scale],
        ],
        b2=[0.00, 0.01 * scale, -0.01 * scale],
    )


if __name__ == "__main__":
    x = [0.6, -1.2, 0.3]

    shared_experts = [
        build_toy_ffn(1.0),
        build_toy_ffn(0.7),
    ]

    routed_experts = [
        build_toy_ffn(0.5),
        build_toy_ffn(0.8),
        build_toy_ffn(1.1),
        build_toy_ffn(0.6),
        build_toy_ffn(1.3),
        build_toy_ffn(0.9),
    ]

    router_w = [
        [0.2, -0.1, 0.4],
        [0.7, 0.1, -0.2],
        [0.5, -0.4, 0.3],
        [-0.3, 0.8, 0.1],
        [0.1, -0.2, 0.9],
        [0.6, 0.0, 0.2],
    ]
    router_b = [0.1, -0.2, 0.0, -0.1, 0.05, 0.12]

    result = deepseek_moe_forward(
        x=x,
        shared_experts=shared_experts,
        routed_experts=routed_experts,
        router_w=router_w,
        router_b=router_b,
        k_total=4,  # 2 shared + 2 routed
    )

    print("selected routed experts:", result["selected_routed_experts"])
    print("selected weights:", result["selected_weights"])
    print("output:", [round(v, 6) for v in result["output"]])

    # 基本正确性检查
    assert len(result["selected_routed_experts"]) == 2
    assert abs(sum(result["selected_weights"].values()) - 1.0) < 1e-9
    assert len(result["output"]) == 3
```

这段代码体现了四个实现点：

| 实现点 | 作用 |
|---|---|
| `shared_experts` 单独存放 | 共享专家不参加路由竞争 |
| `sigmoid -> top-k -> renormalize` | 先平滑打分，再保持稀疏性，最后保证保留专家的权重和为 1 |
| `selected_routed_experts` | 明确每个 token 实际调用了哪些路由专家 |
| `shared_out + routed_out` | 通用能力与专项能力在同一层内合并 |

如果第一次接触 MoE，可以把这段代码按下面方式理解：

| 变量 | 在真实模型里对应什么 |
|---|---|
| `x` | 某个 token 进入当前层 FFN 前的隐藏状态 |
| `ToyFFN` | 一个专家子网络 |
| `router_w/router_b` | 路由器参数，通常是线性层 |
| `selected_routed_experts` | 这个 token 真正被分配到的专家编号 |

如果放到 PyTorch 里，通常会用 `nn.ModuleList` 存专家模块，router 由一个线性层产生 logits。真实系统还需要两个额外部件：

| 组件 | 含义 | 作用 |
|---|---|---|
| dispatcher | 分发器 | 把属于同一专家的 token 聚到一起，批量送进该专家计算 |
| combiner | 合并器 | 把各专家输出按原 token 顺序拼回去 |
| capacity control | 容量控制 | 防止单个专家瞬间收到过多 token，造成拥塞 |

也就是说，论文里的“只选几个专家”在公式上很短，但在工程上并不只是一个 `top-k`。真正难的是把稀疏路由转成高效的并行计算。

---

## 工程权衡与常见坑

DeepSeek-MoE 不是“专家越多越好”。工程上最容易踩的坑有三类：

| 坑 | 现象 | 后果 | 规避方式 |
|---|---|---|---|
| 共享专家过多 | routed 专家存在感下降 | 专项能力预算被压缩，模型更像半 dense 结构 | 共享专家保持少量，常见是 $K_s=2$ |
| 没有共享专家 | 通用知识漂移到 routed experts | routed experts 学出大量重复内容 | 至少保留少量 always-on experts |
| 门控竞争过强 | 少数专家长期胜出 | 出现负载失衡或路由塌缩 | 用 normalized sigmoid，并配合负载约束 |

第一个坑很常见。假设总激活数固定是 8，如果把共享专家从 2 提到 4，那么 routed experts 只剩 4 个名额。结果是每个 token 的“专项能力预算”被压缩，模型会更像“半 dense、半 sparse”的折中体，细粒度专家的优势被削弱。

第二个坑是把 $K_s$ 设成 0。这样短期看似更“纯粹稀疏”，但基础知识会被迫散落到 routed experts 里。训练中常见后果是多个专家都在学相似内容，专家利用率表面均衡，实则区分度不高。也就是说，你看到了“很多专家都在工作”，但没得到“很多专家做不同工作”。

第三个坑是门控不稳。softmax 门控在一些配置下容易让少数专家持续胜出，其他专家长期接不到 token，最后形成 routing collapse，也就是“路由塌缩”。白话说，本来有很多专家，最后只剩几个在真正工作。

工程里通常会配合以下机制：

| 机制 | 作用 |
|---|---|
| utilization regularizer | 约束专家利用率不要过于集中 |
| capacity factor | 限制单专家一次可接收的 token 上限 |
| auxiliary load balancing loss | 在训练中鼓励 token 分配更均衡 |
| token dropping / rerouting | 当专家超载时丢弃或改派一部分 token |

对初级工程师来说，一个很实用的判断标准是直接看训练日志：

| 观察指标 | 正常现象 | 异常信号 |
|---|---|---|
| 每个专家接收的 token 数 | 有波动，但大体分散 | 长期只有少数专家特别忙 |
| router entropy | 适中 | 极低时说明选择过于僵化 |
| overflow rate | 偶发 | 长期偏高说明容量设置不合理 |
| shared/routed 输出占比 | 相对稳定 | shared 长期压倒 routed，说明稀疏收益不足 |

这里还有一个容易被忽略的系统权衡：MoE 省的是算术 FLOPs，不一定直接省总时延。因为除了专家计算，还存在跨设备通信、token 重排、负载不均、空转等待。对单机小 batch 场景，这些开销可能把理论收益吃掉一部分。所以“论文里更省 FLOPs”不等于“你的线上服务一定更低延迟”。

可以把训练侧和推理侧的难点分开看：

| 阶段 | 主要难点 |
|---|---|
| 训练 | 负载均衡、稳定收敛、专家并行通信 |
| 推理 | 小 batch 利用率低、dispatch/combine 开销、长尾专家拥塞 |

所以 DeepSeek-MoE 的价值成立，但它成立的前提是：你有能力把路由、并行和容量控制一起做好。只把论文公式抄进模型，不会自动得到论文里的结果。

---

## 替代方案与适用边界

DeepSeek-MoE 不是唯一选择。实际可以把方案分成三类：

| 方案 | 总参数 | 激活参数 | 共享专家 | 适用场景 |
|---|---|---|---|---|
| Dense 7B | 中等 | 全激活 | 无 | 小批量、实现简单、推理路径固定 |
| 传统 MoE | 高 | 低 | 通常无或弱化 | 需要稀疏扩容，但专家较粗 |
| DeepSeek-MoE | 更高 | 更低 | 有，固定少量 | 多任务、多域、大规模训练与成本敏感推理 |

Dense 模型的优点是实现简单、延迟稳定、不会有 dispatch/combine 和负载均衡问题。如果你只有单机推理、小 batch、对系统复杂度比对 FLOPs 更敏感，dense 7B 仍然非常合理。

传统 GShard、Switch、Mixtral 风格 MoE 的优势是已经验证过稀疏扩容路线，但它们和 DeepSeek-MoE 的设计重点并不完全一样：

| 方案 | 典型特点 | 相比 DeepSeek-MoE 的差别 |
|---|---|---|
| GShard | top-2 路由，强调大规模分片训练 | 专家通常更粗，未显式分出共享专家 |
| Switch Transformer | top-1 路由，简化通信和训练 | 路由更简单，但组合自由度更低 |
| Mixtral | 每 token 常选 2 个专家 | 工程成熟，但未把“共享知识”单独隔离成 always-on experts |
| DeepSeek-MoE | 细粒度专家 + 共享专家 | 强调减少冗余、提升专家专一化 |

这几类方案没有绝对高下，更多是取舍问题：

- 如果你优先追求实现简单，dense 更稳。
- 如果你优先追求极致扩容，传统 sparse MoE 仍然有效。
- 如果你明确遇到“专家太粗、知识冗余高、相同计算预算下组合不够灵活”的问题，DeepSeek-MoE 更对症。

DeepSeek-MoE 的适用边界主要在两点：

1. 你确实能承担更复杂的训练和推理系统，包括专家并行、路由调度、负载平衡。
2. 你的任务足够多样，值得用“共享底座 + 细粒度专项组合”的方式提高参数利用率。

如果场景只是单领域问答、短上下文、小模型部署，DeepSeek-MoE 的系统收益可能不如 dense 直接。如果场景是多语言、多域知识、长周期训练和成本敏感部署，它的设计就更有意义。

可以用一个更务实的判断表：

| 场景 | 更合适的选择 |
|---|---|
| 单卡部署、低延迟优先 | Dense |
| 多机训练、任务多样、算力成本敏感 | DeepSeek-MoE |
| 想先验证稀疏路线、降低架构改动 | 传统 MoE / Switch / Mixtral 路线 |

所以结论不是“DeepSeek-MoE 一定更先进”，而是“它在特定问题定义下，把稀疏 MoE 的参数利用率往前推了一步”。

---

## 参考资料

- DeepSeek-AI, *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, ACL 2024: https://aclanthology.org/2024.acl-long.70/
- DeepSeek-AI 官方仓库，含 16B 模型说明与配置入口: https://github.com/deepseek-ai/DeepSeek-MoE
- Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*, arXiv 2020: https://arxiv.org/abs/2006.16668
- Fedus, Zoph, Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, arXiv 2021: https://arxiv.org/abs/2101.03961
- Jiang et al., *Mixtral of Experts*, arXiv 2024: https://arxiv.org/abs/2401.04088
