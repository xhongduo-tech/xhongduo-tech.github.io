## 核心结论

Best-of-N Sampling，简称 BoN，可以直译为“采样 N 次后选最优”。它不是训练算法，而是**推理阶段**的解码策略：同一个基础模型面对同一个 prompt，先生成 $N$ 个候选回答，再由奖励模型或 verifier 打分，最后返回分数最高的那一个。形式化地说：

$$
y^\star = \arg\max_{y_i \in \{y_1,\dots,y_N\}} r_\phi(x, y_i)
$$

这里的奖励模型（reward model）可以白话理解为“一个专门负责打分的裁判模型”，它不负责生成，只负责判断哪个候选更符合人类偏好。

BoN 的价值很直接：**不改模型参数，只增加推理算力，就常能明显提高对齐质量**。因此它常被当作 RLHF 流程里的强 baseline，用来和 PPO、DPO 这类训练级方法横向比较。[RLHF Book](https://rlhfbook.com/c/10-rejection-sampling.html)、[Lecture 1](https://rlhfbook.com/teach/course/lec1-chap1-3/)

一个新手能直接理解的版本是：用户问一句话，模型并行生成 3 条回复，奖励模型分别打分，系统只返回最满意的那条。其余两条完整生成路径都会被丢弃，但正是这一次“多次采样 + 取最大值”，让最终回答更像人类想要的样子。

| 维度 | 单次采样 | Best-of-N |
| --- | --- | --- |
| 基础模型参数 | 不变 | 不变 |
| 推理次数 | 1 次 | $N$ 次 |
| 是否需要奖励模型 | 不一定 | 通常需要 |
| 质量上限 | 受单次采样限制 | 可通过探索更多候选提升 |
| 主要代价 | 低 | 约线性增至 $N$ 倍 |

结论可以压缩成一句话：**BoN 是“用算力换质量”的最简单有效方案之一。**

---

## 问题定义与边界

BoN 解决的不是“如何训练更好的策略”，而是“**生成出来以后，怎么从多个候选里选一个最好**”。

给定输入 $x$，基础策略 $\pi_{\text{ref}}(y\mid x)$ 生成 $N$ 个候选 $\{y_1,\dots,y_N\}$，奖励模型输出对应分数 $r_1,\dots,r_N$。选择器定义为：

$$
S(R)=\arg\max_{j\in[1,N]} r_j
$$

其中 $R=[r_1,\dots,r_N]$ 是奖励向量。这个公式表达的意思很朴素：谁分最高，就返回谁。[RLHF Book](https://rlhfbook.com/c/10-rejection-sampling.html)

玩具例子：

| 候选 | 文本 | 奖励分数 |
| --- | --- | --- |
| $y_1$ | 简短但模糊 | 0.3 |
| $y_2$ | 准确且完整 | 0.6 |
| $y_3$ | 冗长但部分正确 | 0.5 |

此时：

$$
S(R)=\arg\max([0.3,0.6,0.5])=2
$$

所以系统返回第 2 条。其余两条被丢弃，但成本已经付了 3 次生成和 3 次评分。

边界也很明确。

第一，BoN **强依赖奖励模型质量**。如果裁判本身打分不准，$\arg\max$ 会把误差放大。原本只是“偶尔错判”，到了 BoN 里会变成“专挑打分漏洞最大的样本”。这就是 reward hacking，也就是“模型钻评分规则的空子”。[Regularized BoN](https://aclanthology.org/2025.naacl-long.472.pdf)

第二，$N$ 不能无限增大。理论上采样更多更容易碰到高分候选，但如果高分是奖励模型幻觉，不是真实质量提升，那么 $N$ 越大，偏差越严重。工程上通常把 BoN 当作“高质量候选筛选器”，而不是无脑把 $N$ 拉满。

---

## 核心机制与推导

BoN 的机制可以拆成三步：**generate → score → select**。

1. `generate`：从参考策略 $\pi_{\text{ref}}$ 采样 $N$ 个回答  
2. `score`：用奖励模型 $r_\phi(x,y)$ 给每个回答打分  
3. `select`：取分数最高的回答返回

这三步看起来简单，但它背后对应的是一个明确的分布重加权过程。BOND 论文给出了 Best-of-N 分布的解析式：被选中的概率不是原始模型概率本身，而是“原始概率乘上一个与奖励排序相关的放大因子”。论文进一步指出，BoN 分布可以写成一个与 KL 正则化 RLHF 对应的形式。[BOND](https://proceedings.iclr.cc/paper_files/paper/2025/file/947f37882a394140f7add476bb99d1d3-Paper-Conference.pdf)

标准 KL 正则化 RLHF 的最优策略形状是：

$$
\pi_{\text{RL}}(y)\propto \pi_{\text{ref}}(y)\exp\left(\frac{r(y)}{\beta}\right)
$$

而 BOND 给出的结论是，BoN 可视为某个特殊奖励下的最优解，其正则强度满足：

$$
\beta_{\text{BOND}}=\frac{1}{N-1}
$$

直观解释是：$N$ 越大，等价于允许策略离参考分布偏得更远。因为你给了系统更多试错机会，它就更容易跳到“高奖励但不典型”的区域。

在连续近似和忽略校正项时，可以把它理解成：

$$
\pi_{\text{BoN}}(y)\propto \pi_{\text{ref}}(y)\exp\left(\frac{\log p_{\le}(y)+\text{corr}(y)}{\beta_{\text{BOND}}}\right)
$$

其中 $p_{\le}(y)$ 可以白话理解为“这个回答在奖励排序里处于什么分位”，`corr` 是离散采样重复碰撞带来的校正项。[BOND](https://proceedings.iclr.cc/paper_files/paper/2025/file/947f37882a394140f7add476bb99d1d3-Paper-Conference.pdf)

这也是为什么 BOND 能做蒸馏：既然 BoN 对应一个可描述的目标分布，就可以离线用 BoN 采样高质量样本，再训练一个单次采样策略去逼近它。这样线上只采样 1 次，也能接近原来 $N$ 次采样的效果。

真实工程例子：在摘要生成或指令跟随系统里，线上直接跑 $N=8$ 的 BoN 很贵，但离线先用 BoN 产生高质量数据，再蒸馏成单次策略，部署时只保留一次生成，这就是“离线多采样，线上单采样”的典型做法。[BOND](https://huggingface.co/papers/2407.14622)

---

## 代码实现

最小实现只有一个核心循环：生成多个候选，批量打分，返回最高分。

```python
from dataclasses import dataclass

@dataclass
class BaseModel:
    samples: dict

    def sample(self, prompt, i):
        return self.samples[prompt][i]

@dataclass
class RewardModel:
    scores: dict

    def score(self, prompt, candidate):
        return self.scores[(prompt, candidate)]

def best_of_n(prompt, base_model, reward_model, n):
    candidates = [base_model.sample(prompt, i) for i in range(n)]
    rewards = [reward_model.score(prompt, c) for c in candidates)]
    best_idx = max(range(n), key=lambda i: rewards[i])
    return candidates[best_idx], rewards[best_idx]

base = BaseModel(samples={
    "解释 TCP 三次握手": [
        "TCP 会握手三次，过程略。",
        "TCP 三次握手用于同步序列号并确认双方收发能力。",
        "TCP 连接时会发很多包。"
    ]
})

reward = RewardModel(scores={
    ("解释 TCP 三次握手", "TCP 会握手三次，过程略。"): 0.30,
    ("解释 TCP 三次握手", "TCP 三次握手用于同步序列号并确认双方收发能力。"): 0.82,
    ("解释 TCP 三次握手", "TCP 连接时会发很多包。"): 0.41,
})

answer, score = best_of_n("解释 TCP 三次握手", base, reward, 3)
assert answer == "TCP 三次握手用于同步序列号并确认双方收发能力。"
assert score == 0.82
```

工程实现里有两个关键点。

第一，**批量打分**。候选文本可以一次性送进奖励模型，减少 padding 和 RPC 往返。

第二，**并行生成**。如果底层推理框架支持并发采样，BoN 的墙钟时间不一定严格是 $N$ 倍，但总 token 成本通常仍近似线性增长。

| candidates | rewards | best index |
| --- | --- | --- |
| 3 条候选 | `[0.30, 0.82, 0.41]` | 1 |

对新手来说，这段代码的本质只有一句：**先多写几版答案，再把评分最高的那版交出去。**

---

## 工程权衡与常见坑

BoN 的优点是简单、稳定、黑盒友好。黑盒友好的意思是“即使你拿不到模型参数，只要能多次调用 API 并有一个外部打分器，也能做”。这让它特别适合快速验证一个任务是否能从“更好的选择器”中获益。[Regularized BoN](https://aclanthology.org/2025.naacl-long.472.pdf)

但它的代价也同样明确。

| 方案 | 训练成本 | 推理成本 | 稳定性 | 风险点 |
| --- | --- | --- | --- | --- |
| BoN | 低 | 高，约 $N$ 倍 | 高 | reward hacking |
| PPO/DPO | 高 | 低，通常 1 次 | 依实现而定 | 训练复杂 |
| BOND | 中高 | 低，通常 1 次 | 较高 | 需要蒸馏流程 |

最常见的坑有三个。

第一，**把奖励分数当真实质量**。奖励模型只是代理目标，不是真实人类偏好的完整表达。分高不等于真的更好。

第二，**$N$ 增长后出现过优化**。当 $N$ 很大时，更容易采到“特别会骗奖励模型”的异常样本。Regularized BoN 提出把 MBR 作为邻近正则项，选择目标可写成“奖励 + 邻近性”：

$$
y^\star=\arg\max_{y\in Y}\left[R(x,y)+\beta \cdot \frac{1}{N}\sum_{y'\in Y} U(y,y')\right]
$$

其中 $U(y,y')$ 是样本间相似度。MBR 可以白话理解为“选一个既高分、又不离群的候选”，避免专挑极端漏洞样本。[Regularized BoN](https://aclanthology.org/2025.naacl-long.472.pdf)

第三，**线上预算不够**。如果服务要求低延迟、低成本，BoN 往往只能用于离线数据构造、灰度实验，或者高价值请求，而不是默认全量开启。

---

## 替代方案与适用边界

如果把方法放在一张表里，差异会很清楚：

| 方法 | 核心思想 | 优点 | 局限 | 适合场景 |
| --- | --- | --- | --- | --- |
| BoN | 多采样后硬选最大分 | 简单、强 baseline | 推理贵 | 离线筛选、高价值请求 |
| Soft BoN | 用 softmax 代替硬 argmax | 更平滑，较少过优化 | 实现更复杂 | 想保留多样性 |
| BOND | 蒸馏 BoN 分布 | 线上单次采样 | 需训练 | 部署预算严格 |
| PPO/DPO | 直接训练策略 | 线上便宜 | 训练复杂 | 长期产品化 |

适用边界可以这样判断：

如果你现在有一个基础模型、一个还算可靠的奖励模型，但没有预算做复杂训练，BoN 是最直接的起点。

如果你发现 BoN 的线上成本太高，就不要继续把 $N$ 堆上去，而应该考虑 BOND 这类蒸馏路线，即先离线多采样、再训练一个单次采样策略。[BOND](https://proceedings.iclr.cc/paper_files/paper/2025/file/947f37882a394140f7add476bb99d1d3-Paper-Conference.pdf)

如果奖励模型噪声较大，BoN 会越来越危险，这时更合适的是 Soft BoN、MBR-BoN，或者直接回到更强的训练式对齐方法。

一句话概括适用边界：**BoN 适合先证明“多候选筛选”有效，不适合无条件长期当作最终部署方案。**

---

## 参考资料

- Nathan Lambert, *RLHF Book*，Rejection Sampling / Best-of-N 章节与课程讲义：[https://rlhfbook.com/c/10-rejection-sampling.html](https://rlhfbook.com/c/10-rejection-sampling.html), [https://rlhfbook.com/teach/course/lec1-chap1-3/](https://rlhfbook.com/teach/course/lec1-chap1-3/)
- Pier Giuseppe Sessa et al., *BOND: Aligning LLMs with Best-of-N Distillation*，ICLR 2025 论文：[https://proceedings.iclr.cc/paper_files/paper/2025/file/947f37882a394140f7add476bb99d1d3-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2025/file/947f37882a394140f7add476bb99d1d3-Paper-Conference.pdf)
- Yuu Jinnai et al., *Regularized Best-of-N Sampling with Minimum Bayes Risk Objective for Language Model Alignment*，NAACL 2025 论文：[https://aclanthology.org/2025.naacl-long.472.pdf](https://aclanthology.org/2025.naacl-long.472.pdf)
