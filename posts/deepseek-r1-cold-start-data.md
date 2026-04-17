## 核心结论

DeepSeek-R1 的关键设计，不是“直接用强化学习把推理能力练出来”，而是先用一批高质量冷启动数据把输出格式、语言一致性和自检习惯固定下来，再把强化学习用于继续拉高推理深度。这里的“冷启动”可以理解为：先给模型一套清晰、可模仿的推理写法样板，避免它一上来就在奖励驱动下学出可读性差、语言混杂的表达。

R1 和 R1-Zero 的区别，核心不在“会不会推理”，而在“推理结果能不能稳定给人看”。R1-Zero 证明了纯 RL 也能涌现出推理能力，但它常出现中英混杂、结构不稳、反思句式不一致的问题。R1 的做法是先用数千条人工整理的长思维链做 SFT，再进入 GRPO 强化学习，再做拒绝采样构建大规模 SFT 数据，最后再做一轮 RL 对齐。这个顺序的本质是：先格式，再深度；先可读，再放大奖励驱动的能力。

可以把四阶段流程压缩成一句话：

$$
\text{Cold-start SFT} \rightarrow \text{RL} \rightarrow \text{Rejection SFT} \rightarrow \text{Final RL}
$$

其中第一步解决“怎么说”，后面几步主要解决“说得够不够对、够不够深”。

| 阶段 | 主要目标 | 数据规模 | 主要奖励/信号 |
|---|---|---:|---|
| Cold-start SFT | 学会清晰的 `<think>` 推理格式 | 数千条 | 监督学习损失 |
| 第一阶段 RL | 拉高推理深度与正确率 | 在线采样 | 准确率、格式、语言一致性 |
| Rejection SFT | 回收高质量推理与写作样本 | 约 800K，其中约 600K 推理、200K 写作 | 拒绝采样后的监督信号 |
| 最终 RL | 对齐可用性与整体回答质量 | 在线采样 | 任务奖励与对齐奖励 |

---

## 问题定义与边界

先定义问题。这里说的“冷启动数据”，不是普通指令微调数据，而是带有完整推理轨迹的数据。所谓“推理轨迹”，白话说，就是把模型从审题、分解、试算、反思到核对的过程完整写出来，而不是只给最终答案。DeepSeek-R1 的冷启动样本通常用 `<think>...</think>` 包裹内部思考，并显式包含“我需要检查一下”“前一步可能有问题”这类反思语句。

边界也要说清楚。冷启动阶段不是为了把所有能力都教会模型，它主要负责三件事：

1. 固定一种清晰的 reasoning 外观。
2. 降低语言混杂和结构漂移。
3. 让模型在进入 RL 前就具备“先分析再作答”的惯性。

它不直接替代 RL。因为仅靠 SFT，模型能学到格式和常见套路，但很难持续逼近高难题上的最优推理路径。RL 才是后面放大推理深度的主引擎。

一个玩具例子可以说明边界。

假设任务是求 $1+2+\cdots+100$。如果冷启动样本只写：

- 步骤 1：求和
- 步骤 2：得 5050

这种数据几乎只教会模型“给个简短步骤”。但如果样本写成：

```text
<think>
目标是求 1 到 100 的和。
直接逐项相加可以做，但不高效。
我尝试首尾配对：1+100=101，2+99=101。
一共 50 对，所以结果应为 50×101=5050。
我再检查一下，100 个数确实能配成 50 对，没有遗漏。
</think>
答案：5050
```

那模型学到的就不只是答案，而是“先选方法，再验证”的行为模式。

真实工程里，问题更明显。比如一个面向中文用户的问答模型，如果直接用 RL 优化数学和代码任务，常会得到这种输出：前半段中文分析，后半段突然插入英语术语和半截代码式短句。正确率也许不低，但用户会直接觉得“乱”“不稳”“不像可上线产品”。所以冷启动不是锦上添花，而是把“研究原型”变成“工程可交付模型”的必要步骤。

---

## 核心机制与推导

R1 的训练逻辑可以拆成两个层次：格式塑形和能力放大。

第一层是冷启动 SFT。它的目标很直接：让模型先学会一种稳定的内部推理写法。这里最重要的不是样本数量极大，而是样本质量高，尤其是三类内容必须出现：

| 样本成分 | 作用 | 缺失后的后果 |
|---|---|---|
| 详细推理步骤 | 教模型分步展开 | 容易跳步，只报答案 |
| 自我反思语句 | 教模型回看前一步 | 错误后继续硬推 |
| 验证与复核 | 教模型在结尾检查 | 正确率和稳定性下降 |

第二层是 RL，DeepSeek-R1 使用的是 GRPO。GRPO 可以理解为一种“成组比较候选答案”的策略优化方法。白话说，它不单独训练一个 critic 去估值，而是一次生成一组候选回答，在组内比较谁更好，再用归一化后的优势信号更新策略。

一个常见写法可以表示为：

$$
J(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(r_i(\theta)\hat{A}_i,\ \text{clip}(r_i(\theta),1-\epsilon,1+\epsilon)\hat{A}_i\right)-\beta D_{KL}\left(\pi_\theta\|\pi_{\text{ref}}\right)\right]
$$

其中：

- $G$ 是同一问题生成的候选数。
- $r_i(\theta)$ 是新旧策略概率比。
- $\hat{A}_i$ 是组内归一化后的优势，白话说，就是“这个候选比同组其他答案好多少”。
- $D_{KL}$ 是 KL 正则，作用是别让新策略偏离参考模型太猛。

对 R1 来说，奖励函数也不是只看答对没答对。至少可以抽象成：

$$
R_{\text{final}} = R_{\text{accuracy}} + \lambda_1 R_{\text{format}} + \lambda_2 R_{\text{lang}}
$$

其中语言一致性奖励常写成：

$$
R_{\text{lang}}=\frac{\text{target-language words}}{\text{total CoT words}}
$$

这条式子的意思很直白：如果目标是中文回答，那推理链里中文词占比越高，奖励越高。它不是理解语义的万能指标，但对“中英混杂”这种工程问题非常有效，因为它把“语言稳定”直接变成了可优化目标。

这里要强调一个推导上的结论：RL 并不会天然学出好格式。RL 只会朝奖励函数钻。如果奖励主要是正确率，模型就优先学“怎样更可能拿高分”，不一定在乎表达是否清晰。因此 R1 要在进入 RL 前先把 `<think>` 结构、反思句式、检查动作种进模型，再通过 RL 放大这些行为在困难任务上的收益。

---

## 代码实现

下面给一个最小化玩具实现，演示“冷启动筛样本”和“语言一致性打分”的基本思路。它不是完整训练器，但可以直接运行，帮助理解数据构建的检查逻辑。

```python
import re
from dataclasses import dataclass

@dataclass
class Sample:
    prompt: str
    think: str
    answer: str

def has_reflection(text: str) -> bool:
    keywords = ["检查", "验证", "反思", "确认", "重新看", "可能有误"]
    return any(k in text for k in keywords)

def has_think_tag(text: str) -> bool:
    return "<think>" in text and "</think>" in text

def lang_consistency_ratio(text: str) -> float:
    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", text)
    if not tokens:
        return 0.0
    zh = sum(1 for t in tokens if re.match(r"[\u4e00-\u9fff]", t))
    return zh / len(tokens)

def quality_score(sample: Sample) -> float:
    full = sample.think
    score = 0.0
    if has_think_tag(full):
        score += 0.4
    if has_reflection(full):
        score += 0.3
    if lang_consistency_ratio(full) >= 0.8:
        score += 0.3
    return round(score, 2)

toy = Sample(
    prompt="求 1 到 100 的和",
    think=(
        "<think>先做首尾配对。1+100=101，2+99=101。"
        "一共 50 对，所以结果是 5050。"
        "我再检查一次，100 个数确实能配成 50 对，没有遗漏。</think>"
    ),
    answer="5050"
)

bad = Sample(
    prompt="求 1 到 100 的和",
    think="<think>pair numbers quickly and get 5050. done</think>",
    answer="5050"
)

assert has_think_tag(toy.think)
assert has_reflection(toy.think)
assert lang_consistency_ratio(toy.think) > lang_consistency_ratio(bad.think)
assert quality_score(toy) > quality_score(bad)

print("toy score =", quality_score(toy))
print("bad score =", quality_score(bad))
```

如果把这个思路扩展到真实训练流水线，顺序通常是这样的：

```python
# 1. 冷启动 SFT
for epoch in range(2):
    for batch in cold_start_loader:
        loss = model.supervised_loss(batch["prompt"], batch["target_with_think"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 2. 第一阶段 RL
for step in range(rl_steps):
    prompts = sample_prompts()
    candidates = model.generate_group(prompts, group_size=8)

    rewards = []
    for group in candidates:
        group_rewards = []
        for c in group:
            r_acc = accuracy_reward(c)
            r_fmt = format_reward(c)      # 是否有清晰 think 结构
            r_lang = language_reward(c)   # 语言是否一致
            group_rewards.append(r_acc + 0.2 * r_fmt + 0.1 * r_lang)
        rewards.append(group_rewards)

    model.grpo_update(prompts, candidates, rewards)

# 3. 拒绝采样，回收高质量样本
accepted = []
for prompt in large_prompt_pool:
    group = model.generate_group([prompt], group_size=16)[0]
    best = max(group, key=lambda c: score_for_rejection_sampling(c))
    accepted.append(best)

# 4. 用 accepted 再做大规模 SFT，然后进入最终 RL
```

真实工程例子可以这样看：假设团队在做一个中文编程助手。你准备 3000 条冷启动样本，每条都包含题目理解、方案比较、边界检查、最终回答，统一使用 `<think>` 包裹内部思考。先跑 1 到 2 个 epoch 的 SFT，让模型稳定学会“先分析，再输出”。然后再把这个 checkpoint 丢进 GRPO 流程，用代码单测通过率、数学正确率、语言一致性作为奖励。RL 跑完后，再从大规模候选里做拒绝采样，挑出表现最好的 80 万条样本回灌 SFT。这样得到的模型，通常比纯 RL 版本更像一个能长期服务用户的产品，而不是一个偶尔灵光一现的研究系统。

---

## 工程权衡与常见坑

最大权衡在于：冷启动会增加人工成本，但不做这一步，后面 RL 往往要花更高代价修复表达问题。换句话说，冷启动是在前面花钱，纯 RL 是把问题留到后面更贵地解决。

常见坑可以直接列成排查表：

| 坑 | 影响 | 典型原因 | 缓解方法 |
|---|---|---|---|
| 语言混用 | 用户觉得回答不稳 | RL 只追正确率 | 加语言一致性奖励，外加人工过滤 |
| 推理链很短 | 难题正确率不稳定 | 冷启动样本太短 | 强制保留长链与复核步骤 |
| 格式漂移 | 有时有 `<think>`，有时没有 | SFT 样本格式不统一 | 统一模板，做离线校验 |
| 写作能力倒退 | 非推理任务变差 | 只做 RL，不做回灌 SFT | 加拒绝采样后的大规模 SFT |
| 奖励投机 | 模型学会“像是对的”而不是真对 | 奖励单一或可钻空子 | 组合准确率、格式、语言、规则检查 |

一个常见误区是以为“只要 RL 够强，格式自然会收敛”。这在工程上通常不成立。因为奖励函数能看见的东西有限，模型会优先优化可量化部分。若格式奖励太弱，模型就可能把 `<think>` 写成几句装饰性文字；若语言奖励太弱，它就可能在难点处切回英文术语，因为那样更容易维持高分。

另一个坑是冷启动样本虽然“长”，但没有真实反思。比如全篇只是线性展开，没有“这一步是否成立”“有没有遗漏边界”的复核动作。这样的数据会让模型学会把回答写长，却不一定学会自检。长不是目的，结构化反思才是目的。

---

## 替代方案与适用边界

如果场景只关心推理能力，不关心可读性，比如内部研究验证、自动打分 benchmark、无需直接面向终端用户的 pipeline，那么 R1-Zero 这类纯 RL 路线是可以成立的。它的优势是流程更短，少一段人工整理冷启动数据的成本。代价是输出不稳定，语言混用和结构漂移要自己承担。

如果场景面向真实用户，尤其是客服、教育、编程助手、知识问答这类高频交互系统，那么 R1 的路线更合理。原因不是它一定“更聪明”，而是它更容易变成一个长期可维护的产品接口。用户对错误有容忍度时，往往仍然无法接受“看不懂”或“风格乱”。

| 方案 | 输出质量 | 工程成本 | 适用场景 | 主要风险 |
|---|---|---|---|---|
| R1-Zero / 纯 RL | 推理可能很强，但可读性不稳 | 较低 | 内部研究、自动评测 | 混语言、结构漂移 |
| R1 / 冷启动 + 四阶段 | 推理与可读性更均衡 | 较高 | 面向用户的正式产品 | 数据构建和训练流程更复杂 |
| 轻量蒸馏模型 + 冷启动 | 成本较低，效果取决于教师模型 | 中等 | 资源有限的小团队 | 上限受基座模型限制 |

还有一个现实边界：不是每个团队都能一次性构建高质量数千条长思维链。如果资源不足，可以先做缩小版冷启动集，优先覆盖高频任务、固定输出格式，再逐步用拒绝采样补足分布。这样虽然未必复制 R1 的完整效果，但能先解决最痛的可读性问题。

结论很简单。纯 RL 适合追求“能不能涌现”，冷启动加多阶段回灌适合追求“能不能上线”。这两个目标不冲突，但优化顺序不同。

---

## 参考资料

- Nature 论文：《DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning》  
  https://www.nature.com/articles/s41586-025-09422-z
- DeepSeek-R1 模型介绍与多阶段 pipeline 说明  
  https://deepseek-usa.ai/models/deepseek-r1/
- Emergent Mind 对 DeepSeek-R 系列的机制拆解  
  https://www.emergentmind.com/topics/deepseek-r
- alphaXiv 上的 DeepSeek-R1 训练细节整理  
  https://www.alphaxiv.org/models/deepseek/deepseek-r1
