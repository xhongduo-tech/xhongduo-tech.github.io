## 核心结论

DeepSeek-Math 的训练价值，不在于“把数学题喂得更多”，而在于把“数学网页筛得更准”和“强化学习信号做得更细”这两件事同时做对了。它先在 DeepSeek-Coder-v1.5 7B 的基础上继续预训练，额外加入约 120B 数学相关 token；再用 GRPO 做强化训练。GRPO 可以理解为“把同一道题的一组答案放在一起比较”，不再单独训练 critic，也就是价值网络，因此更省显存、更适合大模型在线采样训练。

从公开结果看，论文中的 DeepSeekMath 7B 在 MATH 上单样本达到 51.7%，64 次自一致提升到 60.9%。官方仓库同时给出 Instruct 和 RL 版本，说明 RL 版本在有工具设置下接近 60% MATH，已经进入当时开源数学模型第一梯队。更重要的是，这条路线验证了一个工程判断：在数学推理这种高结构化任务上，数据质量往往比盲目扩大数据规模更重要。

先看一个阶段性对比。这里把不同公开材料中的结果按“能力跃迁”整理成趋势表，重点看方向，不把它误解成同一评测设置下的一张严格横向表。

| 阶段 | 代表能力变化 | MATH |
| --- | --- | --- |
| Base | 继续预训练后，先获得数学领域底座 | 36.2% 到 51.7% 之间，取决于是否按论文正文/摘要口径 |
| Instruct | 让模型更会按题目要求输出步骤 | 约 46.8% 到 56.2% |
| RL | 用 GRPO 强化“更对的推理轨迹” | 约 51.7% 到 59.7% |

这里要抓住的不是某个单点数字，而是顺序：`高质量数学语料 -> 指令化 -> 组相对强化学习`。这三步叠加后，7B 模型就能逼近当时部分闭源大模型的数学推理水平。

---

## 问题定义与边界

DeepSeek-Math 要解决的问题，不是“做一个什么都能聊的大模型”，而是“做一个对数学推理足够强、且开源可复现的专用模型”。数学推理的特点是答案可验证、过程高度结构化、错误往往不是语法错而是中间推导错，所以它和普通聊天模型的训练重点不同。

它的训练边界也很明确：

| 维度 | DeepSeek-Math 的选择 | 含义 |
| --- | --- | --- |
| 目标任务 | 数学推理为主 | 优先提升解题、证明、数值推导 |
| 数据来源 | OpenWebMath 种子 + Common Crawl 抽取 | 从开放网页里找高质量数学文本 |
| 训练策略 | 继续预训练 + SFT/RL | 先学知识分布，再学输出格式和推理偏好 |
| 工具依赖 | 核心结果强调无工具也强 | 说明不是靠外部计算器“作弊” |
| 语言覆盖 | 英文、中文等多语言数学内容 | 不是只对英语竞赛题有效 |
| 去污染 | 对 MATH、GSM8K、CMATH 等做过滤 | 避免把测试题提前见过 |

“fastText 分类器”首次出现时可以简单理解为：一个轻量文本分类模型，用来判断网页是不是数学内容。它不是为了生成答案，而是为了从海量网页里先把“像数学”的东西捞出来。

玩具例子可以这样理解。假设你要从一亿个网页里找数学内容，最笨的方法是全抓下来继续训练；更合理的方法是先拿 OpenWebMath 当正样本、拿随机网页当负样本，训练一个 fastText 分类器，再给 Common Crawl 里的页面打分，只保留高分页面。这样做的本质是把“训练预算”优先用在更像数学的文本上，而不是浪费在无关内容上。

官方仓库给出的数据管道基本是下面这条链：

| 步骤 | 做什么 | 为什么 |
| --- | --- | --- |
| Seed | 用 OpenWebMath 做初始种子 | 先拿到一批高精度数学网页 |
| 分类 | 训练 fastText 区分数学/非数学页面 | 扩大召回范围 |
| 排名 | 在去重后的 Common Crawl 中打分检索 | 让高置信度页面优先进入语料 |
| 域扩展 | 找数学相关域名并人工标注 URL 模式 | 提高召回率，避免只会抓“最像种子”的页面 |
| 迭代 | 重复 4 轮 | 每轮用新正样本改进分类器 |
| 去重/去污染 | 去掉重复页和 benchmark 泄漏 | 防止训练集“抄到”测试题 |
| 多语言覆盖 | 保留多语种数学文本 | 提高泛化能力 |

这条管道最终得到 35.5M 数学网页、约 120B token。这个量不是随手抓出来的，而是筛选、扩展、去污染之后留下来的结果。

---

## 核心机制与推导

GRPO 全称是 Group Relative Policy Optimization。白话解释是：不是单独判断“这个答案绝对好不好”，而是把同一题生成的一组答案放在一起，比较谁比组内平均水平更好。

它的关键 advantage 公式是：

$$
A_i=\frac{r_i-\bar r}{\mathrm{std}(r)+\epsilon}
$$

其中 $r_i$ 是第 $i$ 个候选答案的 reward，$\bar r$ 是这一组候选答案的平均 reward。这个标准化的含义很直接：

- 高于组均值的答案，$A_i>0$，训练时被强化
- 低于组均值的答案，$A_i<0$，训练时被压制
- 不需要单独训练 critic，因此更省内存

“无需 critic”这句话很重要。critic 就是专门估计状态价值的网络，PPO 通常要多维护一套。GRPO 直接用组内 reward 当基线，省掉了这一套。

玩具例子：题目是 `2 + 2 × 6`。正确答案是 14。假设模型一次采样 8 个候选，其中 4 个答对、4 个答错，reward 设为正确得 1，错误得 0。

| 候选类型 | reward | 组均值 | std | advantage |
| --- | --- | --- | --- | --- |
| 正确答案 | 1 | 0.5 | 约 0.53 | 约 0.94 |
| 错误答案 | 0 | 0.5 | 约 0.53 | 约 -0.94 |

这说明 GRPO 学到的不是“reward=1 就一律加分”，而是“比同组其他答案更好的输出要被放大”。这就是“Group Relative”这个名字的来源。

策略更新时，GRPO 沿用了 PPO 的稳定训练思想：一边根据 advantage 拉高好答案概率，一边用 clip 和 KL 约束，避免模型一步走太猛。可以把目标写成下面这种组合：

$$
J(\theta)\approx \mathbb{E}\big[\min(r_t(\theta)A_t,\ \mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t)\big]-\beta D_{KL}(\pi_\theta\|\pi_{ref})
$$

其中

$$
r_t(\theta)=\frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})}
$$

这套机制适合数学题，因为数学题通常能给出相对明确的 reward。reward 可以是结果奖励，也就是最终答案对不对；也可以是过程奖励，也就是步骤是否满足某种格式、某一中间推导是否合理。DeepSeek-Math 相关资料里也讨论了 outcome supervision 和 process supervision。对数学任务来说，过程监督的价值在于，它能减少模型只靠“蒙对最终答案”拿奖励的情况。

---

## 代码实现

下面给一个可以直接运行的简化版 GRPO 玩具实现。它不依赖深度学习框架，只演示三件事：组内标准化、importance ratio、clip。

```python
import math

def mean(xs):
    return sum(xs) / len(xs)

def std(xs):
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

def compute_advantages(rewards, eps=1e-8):
    mu = mean(rewards)
    sigma = std(rewards)
    return [(r - mu) / (sigma + eps) for r in rewards]

def grpo_policy_loss(old_logps, new_logps, advantages, clip_eps=0.2):
    losses = []
    for old_lp, new_lp, adv in zip(old_logps, new_logps, advantages):
        ratio = math.exp(new_lp - old_lp)
        clipped_ratio = min(max(ratio, 1 - clip_eps), 1 + clip_eps)
        unclipped = -adv * ratio
        clipped = -adv * clipped_ratio
        losses.append(max(unclipped, clipped))
    return sum(losses) / len(losses)

# 玩具例子：8 个候选，4 对 4 错
rewards = [1, 1, 1, 1, 0, 0, 0, 0]
advs = compute_advantages(rewards)

# 前四个 advantage 应为正，后四个应为负
assert all(a > 0 for a in advs[:4])
assert all(a < 0 for a in advs[4:])

old_logps = [-1.0] * 8
new_logps = [-0.9, -0.95, -1.0, -0.85, -1.1, -1.2, -1.05, -1.15]
loss = grpo_policy_loss(old_logps, new_logps, advs)

assert loss > 0
assert round(mean(advs), 6) == 0.0

print("advantages:", [round(x, 2) for x in advs])
print("loss:", round(loss, 4))
```

如果把它映射到真实训练流程，核心步骤就是：

1. 对每个 prompt 采样 $G$ 个 completion  
2. 用 reward function 或 reward model 给每个 completion 打分  
3. 组内标准化得到 advantage  
4. 计算新旧策略概率比 `ratio`  
5. 做 clip，必要时加 KL 惩罚  
6. 反向传播更新策略模型

真实工程里的伪代码大致就是：

```python
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
ratio = exp(new_logps - old_logps)
clipped_ratio = clamp(ratio, 1 - eps, 1 + eps)
pg_loss = max(-advantages * ratio, -advantages * clipped_ratio)
loss = pg_loss + beta * kl_div
```

这里的 `KL` 可以理解为“别让新模型离参考模型偏得太远”。`clip` 可以理解为“即使某次 advantage 很大，也别允许一步把策略推飞”。

真实工程例子：如果你要做一个数学辅导机器人，底座已经有一定数学能力，那么最常见的路线是先做 SFT，让模型学会“分步推理 + 最终答案格式”；如果线上数据证明它仍然会经常在中间步骤走偏，再上 GRPO，把“答案正确”“步骤格式正确”“关键中间量正确”几个信号一起做成 reward。这样比直接做全量 RL 更稳。

---

## 工程权衡与常见坑

GRPO 的优点是 memory-friendly，但它不是“便宜”。Hugging Face 的 GRPOTrainer 示例明确写到，分布在 8 张 GPU 上训练，示例任务大约需要 1 天。也就是说，它是“比 PPO 省”，不是“随便一张卡就轻松跑”。

工程上最容易踩的坑有三类：

| 问题 | 现象 | 本质原因 | 处理思路 |
| --- | --- | --- | --- |
| reward 太单一 | 输出格式变漂亮，但推理并没变强 | 模型学会了钻 reward 空子 | 同时加入正确性、格式、数值抽取等奖励 |
| 数据老化 | 新抓网页质量下降 | 网页分布在变，旧分类器失效 | 定期重训 fastText 与去重规则 |
| 能力不均衡 | 算术和代数不错，几何和证明偏弱 | 训练数据分布偏向可验证题 | 单独监控 proof/geometry 子集 |

这里的“reward hacking”首次出现时可以理解为：模型不是学会了真正的解题能力，而是学会了怎样更容易拿高分。比如只给 final-answer reward，模型可能倾向于输出一个形式上像答案的结果，却不认真优化中间推理。再比如只给 format reward，它会很快学会写整齐的步骤模板，但答案未必更准。

一个真实工程场景是这样的。你用 8 卡做数学问答 RL，如果 reward 只有“最终答案是否匹配标准答案”，模型可能强化两类不理想行为：一类是偶然蒙对；另一类是推理过程胡写，但最后 boxed answer 恰好对。加入 format/process reward 后，模型至少会被约束为输出可解析、步骤更一致的推理轨迹，这对后续人工检查、日志分析和错误归因都更有价值。

还要注意一个边界：DeepSeek-Math 的成功不代表“RL 一定让模型学到全新的基础能力”。一些后续分析指出，RL 更明显的收益常常是把“原本就能答对但概率不够高”的正确轨迹抬上来，也就是改善输出分布，而不一定等价于底层知识容量暴增。这点对资源规划很重要。

---

## 替代方案与适用边界

不是所有团队都应该直接上 GRPO。更务实的选择通常分三档：

| 方案 | 成本 | 效果上限 | 适合场景 |
| --- | --- | --- | --- |
| 仅继续预训练 + SFT | 低到中 | 中 | 需要一个可部署、能稳定按格式解题的数学助手 |
| 单 reward 的 GRPO | 中到高 | 高 | 有明确可验证答案，且确实追求更高准确率 |
| 多 reward 的 GRPO | 高 | 更高但更难调 | 既要高准确率，又要过程结构化、可审计 |

如果你只想做一个数学辅导机器人，首选往往不是 RL，而是先做 SFT。原因很简单：SFT 已经能把模型从“会一点数学”推到“能按步骤回答数学题”，而且训练链路简单、调试成本低、上线风险小。等你真的观察到两个问题再上 RL：

1. 最终答案精度还不够  
2. 中间步骤经常出现系统性错误  

只有在这两点都明显存在时，GRPO 的额外训练成本才合理。

另一条替代路线是“小规模 SFT + 推理时投票”。它的逻辑是：不把钱花在复杂 RL 上，而是在推理阶段多采样几次，让模型自己投票或自一致。这种方法在工程上更容易上线，但代价是线上时延更高、推理成本更大，而且不能像 RL 那样真正重塑输出分布。

简短决策准则可以写成：

- 预算紧、想先上线：优先 SFT
- 题目可验证、追求更高准确率：考虑 GRPO
- 既要高准确率，又要步骤格式可控：考虑多 reward GRPO
- 在线成本比训练成本更敏感：优先训练；否则可以先用投票

---

## 参考资料

- [DeepSeekMath 论文摘要页（含 120B 数学 token、MATH 51.7%、64 样本自一致 60.9%）](https://arxiv.gg/abs/2402.03300)
- [DeepSeek-Math 官方仓库 README（含数据收集 5 步流程、35.5M 页面、120B token）](https://github.com/deepseek-ai/DeepSeek-Math)
- [Hugging Face TRL 文档：GRPOTrainer（含 advantage、KL、loss、8 GPU 约 1 天示例）](https://huggingface.co/docs/trl/grpo_trainer)
- [Hugging Face LLM Course：GRPO 数学细节与 8 候选示例](https://huggingface.co/learn/llm-course/chapter12/3a)
- [Hugging Face Cookbook：多 reward GRPO 数学推理实践](https://huggingface.co/learn/cookbook/trl_grpo_reasoning_advanced_reward)
- [DeepSeekMath 论文解读页（含 36.2/46.8/51.7 阶段结果与 process/outcome supervision 讨论）](https://www.emergentmind.com/papers/2402.03300)
