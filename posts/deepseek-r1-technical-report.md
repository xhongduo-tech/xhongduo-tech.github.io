## 核心结论

DeepSeek-R1 的关键突破，不是“模型突然会思考了”，而是把**可验证任务上的强化学习**真正做成了一条可复用训练流水线。强化学习（RL，白话说就是“做对了加分、做错了扣分，再反复试错”）先在 DeepSeek-R1-Zero 上证明：即使**没有人类推理标注**，大模型也会在奖励驱动下涌现出自我验证、回溯和改写解法这类行为。  

但 R1-Zero 也暴露出两个直接问题：一是可读性差，二是容易中英混杂。因此正式版 DeepSeek-R1 没有停留在“纯 RL”上，而是采用更完整的多阶段流程：**冷启动 SFT -> 面向推理的 GRPO -> 拒绝采样生成高质量样本 -> 再次 SFT -> 面向通用帮助度的第二轮 GRPO**。SFT 是监督微调，白话说就是“先拿一批范例把模型拉到可用起点”。GRPO 是 Group Relative Policy Optimization，白话说就是“让同一题的一组候选答案彼此比较，而不是单独训练一个 critic 评价器”。  

这条流水线的工程意义很明确：R1-Zero 证明“纯 RL 能学出推理行为”，R1 证明“要想上线部署，必须把可读性、语言一致性和通用帮助度重新拉回来”。最后再把大模型生成的约 80 万条高质量样本蒸馏给 Qwen、Llama 级小模型，相当于把“大模型在 RL 中摸索出来的解题习惯”压缩进更小、可本地部署的模型里。  

如果用新手能理解的话概括，这套方法不是一次训练，而是三段式流水线：**先学少量好样本，再靠 RL 自己摸索，再把摸索出来的好解法沉淀成新教材继续训练**。它的目标不是只把最终答案做对，而是把“怎么一步步做对”也变成模型稳定输出的一部分。

---

## 问题定义与边界

DeepSeek-R1 解决的问题，不是一般聊天模型的“会不会说人话”，而是**在复杂任务中，能不能稳定地产生多步推理，并且这些推理既正确又可读**。这里的“推理链”本质上是模型在输出最终答案前生成的一串中间步骤，用来完成拆解、验证、回退和重算。  

这件事只在一类任务上容易成立：**结果可验证**。可验证，白话说就是“外部程序或规则能判断你答得对不对”。例如数学题可以对标准答案，代码题可以跑测试用例，逻辑题可以检查约束是否满足。反过来，开放写作、闲聊、创意文案通常没有稳定的唯一评分器，RL 就很难直接训练。

| 任务类型 | 是否容易做 RL 奖励 | 原因 | DeepSeek-R1 的适配度 |
|---|---:|---|---|
| 数学 | 高 | 答案可程序化校验 | 很高 |
| 编程竞赛 | 高 | 可运行测试集判分 | 很高 |
| 逻辑/科学题 | 中高 | 可用规则或判题器验证 | 高 |
| 开放问答 | 中低 | 对错边界不稳定 | 需要后置对齐 |
| 写作/翻译/角色扮演 | 低 | 偏好强、标准弱 | 主要靠后续 SFT/RM |

一个“玩具例子”可以说明边界。假设题目是“求 $2+3\times4$”。这类题目的奖励很简单：最终答案是不是 14。模型如果先算成 20，再回头发现乘法优先，就能在 RL 中因为“修正后得分更高”逐渐学会回溯。  

一个“真实工程例子”是代码竞赛。模型生成一段程序后，评测系统直接执行隐藏测试。若通过率高，奖励高；若超时、崩溃或答案错误，奖励低。这样的反馈非常适合 RL，因为它不依赖人类逐句打标签。  

因此，DeepSeek-R1 的边界也很清楚：它最擅长的是**有验证器的推理任务**。到了软件工程长链路、真实业务决策、多轮检索这类“慢评估”场景，奖励延迟会显著变长，训练成本和不确定性都会上升。

---

## 核心机制与推导

DeepSeek-R1-Zero 的核心是 GRPO。传统 PPO 常配一个 critic，critic 是“专门估计当前动作值多少钱”的辅助网络。GRPO 省掉 critic，改成让同一问题采样出的多个候选答案**组内相互比较**。  

设同一题 $q$ 采样出 $G$ 个候选输出，奖励分别为 $r_1,\dots,r_G$，那么第 $i$ 个候选的相对优势可以写成：

$$
A_i=\frac{r_i-\mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

白话解释：如果某个候选比同组平均水平更好，它就拿正优势；更差就拿负优势。这样不需要单独训练 critic，也能知道“这条轨迹值不值得鼓励”。  

对应的优化目标保留了 PPO 的裁剪思想，也就是限制新旧策略差异别太大，避免训练发散：

$$
\mathcal{J}_{GRPO}(\theta)=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}
\min\left(
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}A_i,
\operatorname{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)},1-\varepsilon,1+\varepsilon\right)A_i
\right)
-\beta D_{KL}(\pi_\theta\|\pi_{ref})
\right]
$$

其中 KL 惩罚可以理解成“别一下子偏离参考模型太远”。  

DeepSeek-R1 正式版又额外处理了 R1-Zero 的混语问题。论文给出的语言一致性奖励可写成：

$$
R_{\text{language}}=
\frac{\text{目标语言词数}}{\text{推理链总词数}}
$$

如果目标语言是中文，而推理链里混入大量英文，分数就会下降。这个奖励并不直接保证正确，但它能保证样本在后续沉淀为 SFT 数据时更可读、更稳定。  

一个“玩具例子”如下。某题采样 4 个候选，奖励分别是 $[0.2,0.6,0.9,0.3]$。组平均约为 0.5，于是 0.9 的候选明显高于平均，应被强化；0.2 的候选明显低于平均，应被压制。如果 0.9 那条还全程使用目标语言，它不仅因正确而得分高，还会因语言一致性再加分。  

真正把 R1 做成产品级模型的关键不是单轮 GRPO，而是**拒绝采样**。拒绝采样，白话说就是“先大量生成，再只保留高质量样本”。在 R1 中，这一步大致承担三个作用：  
1. 从 RL checkpoint 中抽取已经学会自检和回溯的优质推理链。  
2. 过滤掉混语、冗长段落、格式混乱、不可读的轨迹。  
3. 把这些高质量轨迹沉淀成新的 SFT 数据。  

可以把它理解成一个简化流程：

`生成多个候选 -> 判分 -> 过滤低分 -> 检查语言一致性与可读性 -> 写回训练集`

这一步非常关键，因为纯 RL 学出来的行为不一定天然适合“给人看”，但经过筛选后的轨迹却很适合“教给下一版模型”。

关于与 o1 风格的对比，能确定的一点是：DeepSeek-R1 把这条“先推理、再沉淀、再蒸馏”的开放链路明确公开了；而 o1 更多体现为产品行为层面的强推理能力，外界难以直接看到同等粒度的训练细节。就可观测输出而言，两者都强调长推理、自检和多步展开；差异主要在**开放程度与训练披露**，而不是表面上“会不会一步一步写”。

---

## 代码实现

下面用一个最小可运行例子，模拟 GRPO 的组内优势计算，以及拒绝采样如何把高质量轨迹写回数据集。

```python
from statistics import mean, pstdev

def grpo_advantages(rewards):
    mu = mean(rewards)
    sigma = pstdev(rewards)
    if sigma == 0:
        return [0.0 for _ in rewards]
    return [(r - mu) / sigma for r in rewards]

def language_reward(target_words, total_words):
    assert total_words > 0
    return target_words / total_words

def rejection_sampling(candidates, score_threshold=0.6, lang_threshold=0.9):
    kept = []
    for item in candidates:
        score_ok = item["score"] >= score_threshold
        lang_ok = language_reward(item["target_words"], item["total_words"]) >= lang_threshold
        if score_ok and lang_ok:
            kept.append(item)
    return kept

# 玩具例子：同一题的 4 个候选
rewards = [0.2, 0.6, 0.9, 0.3]
adv = grpo_advantages(rewards)

assert len(adv) == 4
assert adv[2] > 0   # 0.9 高于组平均，应被鼓励
assert adv[0] < 0   # 0.2 低于组平均，应被压制

candidates = [
    {"id": "a", "score": 0.55, "target_words": 95, "total_words": 100},
    {"id": "b", "score": 0.72, "target_words": 98, "total_words": 100},
    {"id": "c", "score": 0.88, "target_words": 80, "total_words": 100},
    {"id": "d", "score": 0.91, "target_words": 97, "total_words": 100},
]

kept = rejection_sampling(candidates)
assert [x["id"] for x in kept] == ["b", "d"]
print("kept:", kept)
```

如果把它映射回 DeepSeek-R1 的训练流程，伪代码可以写成：

```python
actor = cold_start_sft(seed_cot_data)

actor = grpo_train(actor, reasoning_prompts)

samples = generate_from_checkpoint(actor)
selected = rejection_sampling(samples, score_threshold=0.6, lang_threshold=0.9)

sft_data = selected + general_alignment_data
actor = sft_train(actor, sft_data)

actor = grpo_train(actor, mixed_reasoning_and_general_prompts)
```

一个真实工程例子是：你训练代码助手时，先让模型在题库上做 GRPO，奖励来自单元测试；再从高分轨迹里筛出“测试通过且解释清楚”的样本，反向喂给 7B 或 14B 小模型做 SFT。这样小模型虽然没跑过大规模 RL，但会直接继承“大模型试错后留下的好解法”。

| score / 语言一致性 | 动作 |
|---|---|
| `score < 0.6` | 丢弃 |
| `score >= 0.6` 且语言一致性不足 | 重生成或人工过滤 |
| `score >= 0.6` 且语言一致性达标 | 进入 SFT 数据集 |

---

## 工程权衡与常见坑

DeepSeek-R1 的经验说明，**纯 RL 能造出能力，但不一定能造出产品**。R1-Zero 已经能在数学和代码上显著提升，却仍然会出现混语、长篇重复、结构松散等问题。这些问题不是“模型不聪明”，而是“奖励函数没有直接约束可读性”。  

常见坑主要有三类。第一类是**奖励太窄**。如果只看正确率，模型可能会学会极长的冗余推理，因为更长不一定被惩罚。第二类是**样本沉淀不干净**。未经过拒绝采样的 RL 轨迹，常常带有不稳定格式，直接继续 SFT 会把这些坏习惯固化。第三类是**小模型蒸馏只学答案，不学过程**。如果蒸馏集没有保留自检和回溯痕迹，小模型最后学到的只是“像答案”，不是“像推理”。  

| 风险 | 具体表现 | 缓解策略 |
|---|---|---|
| 语言混用 | 一段中文推理夹英文术语甚至整句英文 | 加语言一致性奖励，过滤混语样本 |
| 冗长重复 | 同一步骤反复改写，长度膨胀 | 设置格式约束，拒绝采样过滤 |
| 奖励黑客化 | 钻评分器空子，不是真会推理 | 尽量使用规则验证器，减少纯模型打分 |
| 蒸馏失真 | 小模型只会背答案，不会回溯 | 蒸馏高质量全过程样本，而非最终答案 |
| 通用能力不足 | 推理强，但写作/问答差 | 在后续 SFT 和第二轮 RL 中混入通用数据 |

一个新手容易犯的误解是：“既然 R1-Zero 证明纯 RL 有效，那正式版继续纯 RL 不就行了？”不行。原因很直接：**研究可行性**和**工程可部署性**不是同一个目标。前者追求能力涌现，后者还要可读、稳健、通用。R1 的多阶段设计本质上是在能力和可用性之间做折中。

---

## 替代方案与适用边界

如果你已经有大量高质量人工 CoT，最直接的办法仍然是 **SFT 优先**。CoT 是 chain-of-thought，白话说就是“把中间思路也写出来的示范答案”。这种路线便宜、稳定、容易复现，但上限通常受示范数据质量限制。  

如果你面对的是数学、代码、逻辑证明这类强可验证任务，而又不想为 critic 额外付出成本，GRPO 是更合适的路线。它比 PPO 更省一个价值网络，也更适合“大量同题多候选比较”的场景。  

如果任务本身很难验证，例如开放式写作、复杂软件工程、真实业务代理，纯 RL 的边界会很快出现。因为你缺的不是采样，而是**可靠奖励**。这时更现实的做法通常是：SFT 打底，配合少量偏好优化、工具调用和离线筛数，而不是直接指望 RL 端到端解决。

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| Pure SFT | 稳定、便宜、实现简单 | 上限受数据示范约束 | 已有大量高质量 CoT |
| PPO | 方法成熟，控制力强 | 需要 critic，资源更高 | 奖励复杂、团队已有 RL 基建 |
| GRPO | 无 critic，组内相对比较自然 | 依赖同题多采样与稳定判分 | 可验证推理任务 |
| SFT + GRPO + 拒绝采样 | 效果与可部署性较平衡 | 流程更长，数据工程复杂 | 想同时要能力、可读性和蒸馏价值 |

因此，DeepSeek-R1 最值得借鉴的不是某一个公式，而是它的总策略：**把 RL 当作“发现高质量推理行为”的工具，再把这些行为沉淀成监督数据，最后用更便宜、更稳定的方式扩散到小模型**。这比“让每个小模型都从零跑一遍大规模 RL”现实得多。

---

## 参考资料

- DeepSeek-R1 论文页（alphaXiv 镜像）：https://www.alphaxiv.org/models/deepseek/deepseek-r1  
- Nature 论文《DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning》：https://www.nature.com/articles/s41586-025-09422-z  
- DeepSeek-R1 技术解读与多阶段流程整理：https://deepseek-r1.com/deepseek-r1-paper-interpretation-key-technical-points/  
- DeepSeek-R1 技术揭秘文章：https://deepseek-r1.com/deepseek-r1-technology-revealed-core-principles-of-the-paper-are-broken-down-and-the-key-to-breakthrough-model-performance-is-revealed/
