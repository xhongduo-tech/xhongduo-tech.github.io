## 核心结论

Kimi k1.5 的核心突破，不是“上下文更长”这一个点，而是把**长上下文强化学习**和**长短两种推理链协同**组合成了一套能落地的训练方法。长上下文，白话讲就是模型一次能读进更多材料；推理链，白话讲就是模型在内部逐步拆题、验算、修正的过程。论文报告中，k1.5 在 128K token 的强化学习设置下，不依赖 Monte Carlo Tree Search，简称 MCTS，即“把答案拆成很多搜索分支反复试”的昂贵方法，也不依赖显式 value function，简称价值函数，即“先估计这条推理值不值得继续”的额外评分器，仍然拿到了有竞争力的数学与代码结果，如 AIME 77.5、LiveCodeBench Short-CoT 47.3。

Kimi VL 则是在这条路线之上，把“只会读文本”扩展成“同时看图、看长文档、再做推理”。它的关键不是简单给语言模型前面接一个图像编码器，而是用 MoonViT 处理高分辨率视觉输入，再用 MoE 解码器接长上下文推理。MoE，白话讲就是“很多专家模块里只激活一小部分”，这样总参数大，但单次推理成本可控。

一个适合新手的玩具例子是：你把一整本合同、几页财务图表、几张截图一起交给模型。Long-CoT 负责“慢慢想”，先把全文里的条款关系、图表异常、前后矛盾找出来；Short-CoT 负责“快回答”，把前面的大量推导压缩成短结论。Kimi k1.5 做的是把这套从“慢想”到“短答”的迁移训练出来，Kimi VL 则把输入从纯文本扩展到图文混合。

| 维度 | Kimi k1.5 | Kimi VL |
|---|---|---|
| 主要输入 | 纯文本为主，论文也覆盖多模态 RL | 图像 + 文本 + 长文档 |
| 关键机制 | 128K Context RL、Long-CoT/Short-CoT、OMD 风格更新 | MoonViT + MoE 解码器 + 128K 长上下文 |
| 公开亮点 | AIME 77.5，Short-CoT LiveCodeBench 47.3 | 长文档、OCR、高分辨率图像、多图理解 |
| 工程目标 | 提升长链推理质量与训练效率 | 把长链推理扩展到视觉和文档场景 |

---

## 问题定义与边界

这类模型要解决的问题，可以写成一句话：**在一次会话里处理超长输入，并保持推理质量稳定**。这里的“超长输入”包括两种。

第一种是长文本，比如 30k 字合同、法规、技术规范、代码仓库摘要。第二种是多模态输入，比如扫描件、图表、截图、表单、PDF 页面。传统短上下文模型常见问题是：能回答局部问题，但一旦问题依赖跨段落、跨页面、跨图文对齐，性能就明显下降。

边界也很明确。Kimi k1.5 的论文重点落在 128K token 级别的强化学习训练，目标是在这么长的轨迹里仍然能稳定更新策略。策略，白话讲就是“模型下一步更倾向怎么生成”。Kimi VL 的边界则扩展到高分辨率图像、长视频/长文档和多图场景，但它不是无限扩展，仍然受上下文长度、显存、视觉 token 数量和推理时延约束。

真实工程例子更直观。假设法务系统要审核一份 30k 字合同，附带 6 张赔付结构图和若干 PDF 扫描页，问题是：“哪些条款会引入额外责任，理由是什么？”这不是关键词检索问题，因为责任条款可能分散在定义、例外、附录、图表注释里。它也不是普通 OCR 问题，因为模型不仅要读出字，还要跨页关联，再给出解释。这个场景正好落在 Kimi k1.5 与 Kimi VL 想解决的交集上。

可以把流程先抽象成下面这个输入输出表：

| 阶段 | 输入 | 模块 | 输出 |
|---|---|---|---|
| 1 | 合同文本、图表、扫描页 | 文本编码 + MoonViT | 对齐后的长上下文表示 |
| 2 | 长上下文表示 | Long-CoT 推理 | 中间推导、证据定位、条款关联 |
| 3 | 中间推导 | Short-CoT 压缩回答 | 责任条款列表 + 逻辑解释 |

---

## 核心机制与推导

k1.5 这条路线最值得抓住的点，是它把“更强推理”解释成了“更长轨迹上的可稳定强化学习”。轨迹，白话讲就是模型从读题到生成完整答案的整段过程。传统 RL 在语言模型上容易不稳定，因为输出很长、奖励稀疏、重新采样完整长轨迹又太贵。

论文给出的关键思路是带 KL 约束的在线镜像下降。KL 散度，白话讲就是“新策略和旧策略差了多少”。公式可以写成：

$$
\pi_{t+1}=\arg\max_{\pi}\Big(\mathbb{E}_{\tau\sim\pi}[R(\tau)]-\lambda D_{KL}(\pi\|\pi_t)\Big)
$$

这里 $R(\tau)$ 是轨迹奖励，$\lambda$ 控制“大胆探索”和“保守更新”的平衡。$\lambda$ 越大，策略越不容易突然跳变；$\lambda$ 越小，模型更敢尝试新推理路径。工程上这很重要，因为长链推理一旦更新过猛，模型会出现回答长度暴涨、逻辑漂移甚至训练崩溃。

第二个机制是 Partial Rollout。它的意思不是每次都从头生成完整 128K 轨迹，而是**复用旧轨迹的大段前缀，只对后面一小段继续采样**。白话讲，像你批改一篇长证明，不会每次都从第一行重写，而是保留前面 80K token 的已有思路，只把后面 8K token 续写一版再比较奖励。这样做有两个好处：一是节省生成成本，二是让训练样本在连续上下文里更稳定。

下面这个简表能帮助理解：

| 项目 | 示例值 | 作用 |
|---|---|---|
| 旧轨迹长度 | 80K token | 保留大部分既有推理上下文 |
| 新续写长度 | 8K token | 在局部继续探索更优推理 |
| KL 正则 | $\lambda \cdot D_{KL}$ | 防止策略跳变过大 |
| Replay Buffer | 多轮轨迹缓存 | 复用高质量 Long-CoT 样本 |

Replay Buffer，白话讲就是“把以前跑出来的好轨迹存起来反复学”。它的价值在于，Long-CoT 的代价很高，但一旦某些长链推理已经证明有效，就可以反复拿来训练 Short-CoT，让短回答也继承长推理的结构化能力。所谓 long2short，本质上就是“先学会慢慢想，再学会简洁地答”。

---

## 代码实现

如果把这套机制压缩成最小实现，可以拆成三个对象：`PartialRolloutBuffer` 负责保存长轨迹前缀，`ReplayBuffer` 负责缓存高质量完整轨迹，`scheduler` 负责决定当前问题走 Long-CoT 还是 Short-CoT。

先看一个能运行的玩具代码。它不训练真实模型，只演示“带 KL 正则的策略更新”与“长短推理调度”的骨架。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def kl_div(p, q):
    eps = 1e-12
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))

def mirror_descent_step(old_policy, rewards, lam=0.5):
    # 对离散动作做一个简化的 OMD 风格更新：
    # new_pi(a) ∝ old_pi(a) * exp(reward(a) / lam)
    logits = [math.log(p) + r / lam for p, r in zip(old_policy, rewards)]
    return softmax(logits)

def route_reasoning(input_tokens, has_image):
    # 很粗糙的推理调度器：长输入或带图时优先 Long-CoT
    return "long_cot" if input_tokens > 8000 or has_image else "short_cot"

old_pi = [0.6, 0.4]          # 两个动作：短答、长答
rewards = [0.2, 0.9]         # 长答奖励更高
new_pi = mirror_descent_step(old_pi, rewards, lam=0.5)

assert abs(sum(new_pi) - 1.0) < 1e-9
assert new_pi[1] > old_pi[1]          # 高奖励动作概率上升
assert kl_div(new_pi, old_pi) > 0
assert route_reasoning(12000, False) == "long_cot"
assert route_reasoning(2000, False) == "short_cot"
assert route_reasoning(3000, True) == "long_cot"

print("old:", old_pi)
print("new:", [round(x, 4) for x in new_pi])
```

对应到真实训练流程，结构通常更像下面这样：

```python
class PartialRolloutBuffer:
    def sample_prefix(self, min_prefix_tokens=80000):
        """取一段旧轨迹前缀，供模型继续续写。"""
        ...

class ReplayBuffer:
    def add(self, trajectory, reward):
        """缓存高质量 Long-CoT 轨迹，供 long2short 迁移。"""
        ...

def train_step(model, env, partial_buffer, replay_buffer, lam=0.1):
    prefix = partial_buffer.sample_prefix()
    continuation = model.rollout(prefix, max_new_tokens=8192)
    traj = prefix + continuation
    reward = env.score(traj)

    # OMD / KL-regularized policy update
    model.update_with_kl(traj, reward, lam=lam)

    replay_buffer.add(traj, reward)

def infer(model, prompt, images=None):
    long_trace = None
    if len(prompt) > 8000 or images:
        long_trace = model.long_cot(prompt, images=images)
    return model.short_cot(prompt, images=images, distilled_state=long_trace)
```

这里最关键的不是语法，而是模块位置。`PartialRolloutBuffer` 决定训练是否省钱，`update_with_kl` 决定训练是否稳定，`ReplayBuffer` 决定 Long-CoT 能不能迁移给 Short-CoT。推理阶段再把 Long-CoT 的中间状态提炼给 Short-CoT，形成“慢想一次，快答多次”的工作流。

---

## 工程权衡与常见坑

128K 上下文并不等于“把所有内容都塞进去就会更强”。长上下文的真实成本包括显存、KV cache、时延和噪声干扰。KV cache，白话讲就是模型为前文每个 token 保留的历史计算状态。输入越长，缓存越大；如果无关内容太多，模型不仅更慢，还更容易把注意力浪费在无用片段上。

回到合同审核例子，一个常见误区是把整份合同、全部附件、所有图表、所有版本差异一次硬喂，然后重复发问。正确做法通常是：先做段落裁剪、图表筛选和页面归并，把与责任条款相关的内容优先保留；第二轮再触发 Long-CoT 深推理。也就是说，长上下文能力应该用来保留**有关系的长链信息**，不是保留**一切信息**。

训练阶段的坑更多。没有 KL 正则或长度惩罚时，模型容易学出“越写越长但不一定更对”的坏策略；如果 Replay Buffer 只保留单一题型，还会发生过拟合现象，导致模型只会在见过的轨迹模式里“复读式推理”。

| 常见坑 | 现象 | 规避策略 |
|---|---|---|
| 全量上下文直接喂入 | 显存高、响应慢、注意力分散 | 先做段落裁剪与证据召回 |
| 无 KL 正则化 | 策略发散、答案风格突变 | 用 OMD 风格更新控制步长 |
| 只追求超长 CoT | 成本高，短答退化 | 用 long2short 做迁移蒸馏 |
| Partial Rollout 过短 | 复用不足，训练变贵 | 保留长前缀，只续写关键段 |
| Buffer 数据单一 | 泛化差、题型偏置 | 混合数学、代码、文档样本 |

可以把稳定训练流程压成一句话：**RL 更新产生新轨迹，KL penalty/length penalty 控制更新幅度，Replay Buffer 回流高质量轨迹继续训练。**

---

## 替代方案与适用边界

为什么不直接上 MCTS + value function？因为在超长上下文里，这套方法成本很容易失控。MCTS 适合在有限动作空间里做多分支搜索，但语言模型的轨迹长度和分支数都非常大，一旦上下文扩到 128K，搜索树会变得极其昂贵。value function 也不是免费午餐，它需要额外学习“这条推理值不值”，而这在长文档、多模态场景里同样困难。

相比之下，k1.5 的路线更像是：不显式建很大的搜索树，而是让模型在长上下文里通过更长的 CoT 自己完成规划、反思和修正，再用 KL 约束把更新收住。这不是说它一定在所有任务都更优，而是说在“超长轨迹 + 高推理成本”的条件下更划算。

当然，不是所有任务都需要这套机制。如果只是 2K 字新闻摘要、单轮 FAQ、普通客服问答，标准短上下文模型加常规 RLHF 仍然更合适。RLHF，白话讲就是“根据人类偏好微调回答风格与有用性”。只有当任务同时满足“输入很长”“需要跨段或跨模态推理”“错误成本高”这几个条件时，Kimi k1.5 或 Kimi VL 的优势才明显。

| 方案 | 输入长度 | 模态 | 主要 RL/搜索组件 | 成本 | 适用场景 |
|---|---|---|---|---|---|
| Kimi k1.5 | 长，最高到 128K 级 | 文本为主 | Partial Rollout + OMD + Long/Short CoT | 中高 | 数学、代码、长文档推理 |
| Kimi VL | 长，最高到 128K 级 | 图像 + 文本 | MoonViT + MoE + Long-CoT/RL | 高 | OCR、长文档、图文问答、代理任务 |
| MCTS + value | 长轨迹时代价高 | 可扩展但复杂 | 搜索树 + 价值估计 | 很高 | 小规模高精搜索 |
| 普通短上下文 RLHF | 短到中等 | 文本为主 | 偏好对齐 | 低 | 摘要、闲聊、常规助手 |

判断是否“值得上 Kimi 级机制”，可以看三个问题：输入是否超过普通窗口、答案是否依赖跨页跨图证据、错误是否需要可解释推理链。如果三个里只有一个成立，往往没必要上这么重的系统。

---

## 参考资料

| 类型 | 名称 | 链接 | 说明 |
|---|---|---|---|
| 论文 | Kimi k1.5: Scaling Reinforcement Learning with LLMs | https://arxiv.org/abs/2501.12599 | k1.5 的核心论文，含 128K RL、Long-CoT/Short-CoT 与基准结果 |
| 代码/报告 | MoonshotAI/Kimi-k1.5 | https://github.com/MoonshotAI/Kimi-k1.5 | 官方仓库，概述 Partial Rollout、长上下文 RL 与简化框架 |
| 论文 | Kimi-VL Technical Report | https://arxiv.org/abs/2504.07491 | Kimi VL 技术报告，含 MoonViT、MoE、128K 多模态能力 |
| 代码/模型页 | MoonshotAI/Kimi-VL | https://github.com/MoonshotAI/Kimi-VL | 官方仓库，含架构、长文档、多图、视频与 Thinking 版本说明 |
| 模型卡 | moonshotai/Kimi-VL-A3B-Thinking | https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking | 模型卡，含 2.8B 激活参数、128K 上下文与多项评测 |
| 辅助解读 | Hugging Face Papers: 2501.12599 | https://huggingface.co/papers/2501.12599 | 对 k1.5 论文的便捷入口与摘要索引 |
| 辅助解读 | ScienceStack: Kimi-VL Technical Report | https://www.sciencestack.ai/paper/2504.07491 | 对 VL 技术报告的结构化摘要，便于快速定位要点 |
