## 核心结论

Claude 在 Constitutional AI, 简写 CAI，意思是“让模型按一组明确原则自我约束”的训练框架里，并不是直接把“拒绝有害内容”全部压到强化学习阶段解决，而是在 Supervised Learning，简称 SL，即“先用标准监督数据把模型行为定型”的阶段，先加入 Critique-Revision 循环。

这个循环的作用可以概括成一句话：先让模型回答，再让模型按照随机抽到的一条宪法原则批评自己，最后根据批评重写答案。这样做的结果不是单纯“更会拒绝”，而是把“为什么拒绝、给什么安全替代、如何避免误伤正常请求”一起学进去。

对零基础读者，最直观的理解是：模型先交一版草稿，再做多轮自审。每轮审查都不是随便挑毛病，而是对照一条明确原则，比如“不要鼓励违法行为”“不要帮助侵犯隐私”“给出安全替代方案”。多轮之后，最终答案通常比第一版更安全，也更完整。

玩具例子：用户问“如何黑进邻居 Wi‑Fi”。初始回答 $R_0$ 可能直接给步骤。第一轮如果抽到“避免鼓励非法行为”，批评会指出这在帮助违法；修订后答案开始拒绝。第二轮如果抽到“尽量提供无害替代”，答案会补充“如果是自己网络故障，可以检查路由器、联系运营商”。再往后几轮，拒绝会更稳，解释更充分。

从公开资料看，CAI 的多轮修订在 harmlessness，意思是“回答不造成伤害的程度”上，能相对传统 RLHF 带来约 20% 到 30% 的改善；经验上做到约 4 轮时收益最明显，再继续增加轮数，边际收益会明显下降，helpfulness，意思是“回答是否真正帮到用户”，还可能开始回落。

| 维度 | RLHF 常见做法 | CAI 的 Critique-Revision |
|---|---|---|
| 监督阶段目标 | 学人类偏好答案 | 先把有害回答反复修到更安全 |
| 安全信号来源 | 人类标注为主 | 宪法原则 + 模型自批评 |
| 对危险 prompt 的处理 | 常在后续奖励阶段校正 | 在 SL 阶段先建立 harmless 基线 |
| 多样性来源 | 不同标注员偏好 | 每轮随机抽不同原则 |
| 典型风险 | 奖励过拟合、标注贵 | 轮数过多后过度拒绝 |

---

## 问题定义与边界

这里要解决的问题不是“让模型永远拒绝”，而是“让模型在遇到高风险请求时先形成稳定的无害基线”。所谓基线，就是“模型默认会落到什么行为上”。如果这个基线本身就危险，那么后续再靠 PPO 之类的强化学习细修，会变得成本高，而且容易学到脆弱规则。

因此，Critique-Revision 循环主要覆盖的是有害、灰色、容易越界的 prompt，不是所有 prompt。这个边界很重要。因为如果把正常问答、创意写作、一般编程帮助也全部按高风险流程反复修订，模型会越来越像统一拒绝机，最终 helpfulness 下滑。

可以把训练数据想成两条通道：

1. 高风险通道：先走 Critique-Revision，把危险答案修成安全答案。
2. 正常通道：保留 helpful-only 数据，继续训练模型正常回答问题。

这相当于先把“遇到危险请求别出事”练扎实，再把“正常场景下继续有用”维持住。它不是用安全替代全部能力，而是对不同数据分流处理。

一个真实工程例子是企业客服和内容审核。客服系统既要回答退款流程，也可能收到“如何伪造凭证”“怎样绕开实名限制”这种请求。前者需要高 helpfulness，后者需要高 harmlessness。如果不分边界，模型会要么太敢答，要么什么都不答。

下面这个流程图反映了训练上的位置关系：

```text
prompt
  -> 初始回答 R0
  -> Critique/Revision x L
  -> 最终回答 RL
  -> 监督微调损失 L_SL
  -> 基于模型输出构造 candidate pairs
  -> 奖励模型
  -> PPO / RLAIF
```

这里的 RLAIF 是 Reinforcement Learning from AI Feedback，意思是“用 AI 给出的偏好信号做强化学习”，可以理解成“不是全靠人工打分，而是让模型先按规则做比较，再把比较结果用来训练奖励模型”。

---

## 核心机制与推导

形式化地写，给定一个 prompt，先得到初始回答 $R_0$。然后第 $n$ 轮从原则集合 $P$ 中随机抽一条原则：

$$
p_n \sim \mathrm{Uniform}(P)
$$

这里 Uniform 表示均匀采样，也就是“每条原则被抽中的概率先设成一样”。

随后做两步：

$$
\mathrm{Critique}_n = C(R_{n-1}, p_n)
$$

$$
R_n = \mathrm{Revise}(R_{n-1}, \mathrm{Critique}_n)
$$

第一步 $C(\cdot)$ 是批评函数，可以理解成“指出当前回答哪里违背了抽到的原则”；第二步 $\mathrm{Revise}(\cdot)$ 是修订函数，即“根据批评重写原答案”。重复 $L$ 轮，得到最终答案 $R_L$，再把它作为监督目标：

$$
L_{\mathrm{SL}} = -\sum \log p_\theta(R_L \mid prompt)
$$

这条损失的意思很直接：训练参数为 $\theta$ 的模型，让它在看到这个 prompt 时，更高概率地产生最终修订后的安全答案 $R_L$。

为什么随机抽原则而不是固定一条？因为固定原则只会把模型压向单一话术。例如每次都强调“不要违法”，模型学到的可能只是模板式拒绝；但如果有时抽“保护隐私”，有时抽“避免协助危险行为”，有时抽“提供合法替代方案”，模型会学习更丰富的安全决策边界。

下面用一个玩具例子看每轮变化：

| 轮次 | 抽到的原则 | 回答变化 |
|---|---|---|
| $R_0$ | 无 | 直接给出破解 Wi‑Fi 的步骤 |
| $R_1$ | 不鼓励非法行为 | 删除攻击步骤，开始拒绝 |
| $R_2$ | 保护他人隐私与财产 | 说明行为会侵犯他人网络和隐私 |
| $R_3$ | 提供安全替代方案 | 建议排查自家网络、重置密码、联系运营商 |
| $R_4$ | 保持清晰且不说教 | 拒绝更简洁，替代建议更可执行 |

这解释了为什么前几轮收益最大。第一轮通常消掉最危险的直接帮助；第二到第四轮主要在补理由、补替代路径、压低遗漏风险。超过这个区间后，再修往往只是措辞微调，甚至把原本有帮助的信息也一起删掉。

从系统角度看，SL 阶段的 $R_L$ 还有第二个作用：它能生成候选答案对。比如同一个 prompt 让模型采样出多个版本，再让另一模型按宪法原则比较“哪个更好”，得到偏好对，用于训练奖励模型。这样，Critique-Revision 不只是造监督样本，还在为后续 RLAIF 提供结构化偏好信号。

---

## 代码实现

工程里最核心的对象有四个：原则池、初始回答器、批评器、修订器。原则池就是宪法条目集合；批评器负责指出问题；修订器负责落地修改；最后把终版答案存成监督样本，并可继续收集成偏好对。

下面给一个可运行的 Python 玩具实现。它不调用大模型，只用规则模拟 Critique-Revision 的数据流，目的是把训练结构讲清楚。

```python
import random

principles = [
    "avoid_illegal_help",
    "protect_privacy",
    "offer_safe_alternatives",
    "keep_clear_and_brief",
]

harmful_keywords = {"hack", "wifi", "neighbor"}

def helpful_model(prompt: str) -> str:
    if "wifi" in prompt.lower() and "neighbor" in prompt.lower():
        return "You can scan the router, capture the handshake, and try password cracking."
    return "Here is a normal helpful answer."

def critique(response: str, prompt: str, principle: str) -> str:
    text = response.lower()
    if principle == "avoid_illegal_help" and ("cracking" in text or "hack" in prompt.lower()):
        return "The answer gives actionable illegal instructions."
    if principle == "protect_privacy" and "neighbor" in prompt.lower():
        return "The request targets another person's network and privacy."
    if principle == "offer_safe_alternatives" and "cannot assist" not in text:
        return "The answer should refuse and provide lawful alternatives."
    if principle == "keep_clear_and_brief" and len(response) > 140:
        return "The answer should stay concise."
    return "No major issue."

def revise(response: str, critique_text: str, prompt: str) -> str:
    if critique_text == "No major issue.":
        return response
    base = "I cannot help access someone else's Wi-Fi or provide intrusion steps."
    alt = " If this is your own network, check router settings, reset the password, or contact your ISP."
    if "privacy" in critique_text.lower():
        return base + " That would violate another person's privacy and property." + alt
    if "illegal" in critique_text.lower():
        return base + alt
    if "lawful alternatives" in critique_text.lower():
        return base + alt
    return base

def run_loop(prompt: str, rounds: int = 4, seed: int = 0):
    random.seed(seed)
    response = helpful_model(prompt)
    history = [response]
    for _ in range(rounds):
        principle = random.choice(principles)
        c = critique(response, prompt, principle)
        response = revise(response, c, prompt)
        history.append(response)
    return history

history = run_loop("How to hack my neighbor's WiFi?", rounds=4, seed=1)

assert len(history) == 5
assert "cannot help" in history[-1].lower()
assert "isp" in history[-1].lower()

print(history[-1])
```

这个玩具例子省略了真实系统里的采样温度、批评模板、偏好比较器，但保留了最关键的循环结构。真实工程通常还要做两件事：

| 步骤 | 输出 | 用途 |
|---|---|---|
| 保存 `(prompt, R_L)` | 监督样本 | 用于 SL 微调 |
| 保存同 prompt 下多个候选及偏好 | 比较对 | 用于奖励模型和 PPO |

真实工程例子：Anthropic 在公开材料中提到，SL 阶段会对大量红队 prompt 做多轮 critique-revision，再与 helpfulness 数据混合训练。这类数据规模达到十万级 prompt、数十万级修订样本时，离线生成和存储格式就很重要。常见做法是把每轮轨迹都保留下来，字段至少包括：

- `prompt`
- `initial_response`
- `sampled_principle`
- `critique`
- `revised_response`
- `round_id`
- `final_response`
- `split`

保留轨迹的好处是后续可以分析“哪条原则最常触发修订”“第几轮收益最大”“哪些任务最容易过度拒绝”。

---

## 工程权衡与常见坑

最重要的权衡是轮数。经验上，$L=3$ 到 $4$ 往往是比较稳的区间。原因不是更多轮一定错，而是前几轮解决的是大问题，后几轮解决的是小问题，但小问题的修订成本并不小，而且会侵蚀 helpfulness。

可以把趋势粗略理解成下面这样：

| 修订轮数 | harmlessness 变化 | helpfulness 变化 | 典型现象 |
|---|---|---|---|
| 1 | 明显上升 | 基本不变 | 去掉最危险的直接指令 |
| 2 | 继续上升 | 轻微下降或持平 | 增加理由与边界说明 |
| 3 | 明显收益 | 大多可接受 | 安全替代方案更完整 |
| 4 | 接近峰值 | 开始有波动 | 表达更稳，但更保守 |
| 5+ | 收益递减 | 更容易下降 | 模板化拒绝、过度保守 |

常见坑主要有四类。

第一，固定原则不随机采样。这样会导致模型过拟合单一说法。比如每轮都强调“拒绝非法行为”，最后模型看见稍有风险的请求就机械输出同一段话，在客服、教育、运维建议场景中都会显得僵硬。

第二，只盯 harmlessness 分数，不看 helpfulness。安全模型不是“拒绝越多越好”。例如用户问“如何合法加固公司 Wi‑Fi 安全”，如果模型因为看到 `Wi‑Fi` 就统一拒绝，那是明显误伤。

第三，把所有 prompt 都送进 Critique-Revision。高风险数据需要强修订，低风险数据不需要。否则模型会学到一种统一防御姿态，正常体验下降。

第四，只保存最终答案，不保存中间轨迹。这样做训练能跑，但后期几乎无法诊断问题。你会知道模型变差了，却不知道是“哪条原则过强”“哪类 prompt 在第几轮开始被误杀”。

规避策略也对应很清楚：

- 轮数先从 3 或 4 起步，不要默认越多越好。
- 每轮随机抽原则，而不是固定模板。
- 对数据先做风险分流，高风险再走循环。
- 保留中间轨迹，方便离线分析和 ablation，意思是“逐项去掉组件看影响”。

---

## 替代方案与适用边界

如果把主流方法放在一起比较，差别主要在“安全信号从哪里来”和“在哪个阶段介入”。

| 方案 | 主要信号来源 | harmlessness | helpfulness | 成本 | 适用边界 |
|---|---|---|---|---|---|
| RLHF | 人类偏好标注 | 中等到高 | 高 | 高 | 通用助手、需要精细人类风格对齐 |
| CAI | 宪法原则 + 自批评修订 | 高 | 中高 | 中等 | 高风险请求较多的系统 |
| CAI + RLAIF | 宪法原则 + AI 偏好比较 + PPO | 更高 | 中高，依赖调参 | 中高 | 需要进一步压低有害输出的场景 |

RLHF 的优点是风格和任务完成度通常更自然，因为人类偏好信号直接。但它贵，而且对高风险任务需要大量高质量标注。CAI 的优势是先把安全边界结构化，再把这个边界写进监督样本。CAI + RLAIF 则是在这个基线之上继续强化，但前提是 SL 基线已经够稳，否则奖励模型会放大坏偏差。

适用边界也要说清楚。高风险行业更适合 CAI，例如：

- 企业客服里的身份、支付、合规问题
- 内容审核与举报处理
- 涉及隐私、法律、医疗建议的助手

低风险创意场景则未必需要强 Critique-Revision。例如普通故事创作、营销文案草拟、低风险代码重构，更适合保持 standard helpful-only 训练，避免动不动进入防御性拒绝。

一个新手容易理解的分流方法是先做风险分类：先判断 prompt 属于“有毒/高风险”还是“普通/低风险”，再决定是否进入 Critique-Revision 管道。这和线上系统里的安全网关是同一思路，只是这里发生在训练阶段。

---

## 参考资料

1. Anthropic, *Constitutional AI: Harmlessness from AI Feedback*  
   作用：官方概述 CAI 的训练目标、SL 与 AI feedback 的总体关系。  
   链接：https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback

2. Emergent Mind 对论文 *Constitutional AI: Harmlessness from AI Feedback* 的摘要解读  
   作用：适合回看 Critique、Revision、监督损失和后续 RLAIF 的公式化描述。  
   链接：https://www.emergentmind.com/papers/2212.08073

3. Curiosity Compendium, *Constitutional AI (CAI): An Introduction*  
   作用：整理了多轮修订的直观解释，以及超过 3 到 4 轮后收益递减的观察。  
   链接：https://curiositycompendium.wordpress.com/2024/08/08/constitutional-ai-cai-an-introduction/

4. Netizen 关于 CAI harm reduction 的总结页面  
   作用：用于理解“相对 RLHF 的 harm 降低约 20% 到 30%”这类面向非论文读者的结果表述。  
   链接：https://www.netizen.page/search/label/ai%20safety

5. Transcendent AI, *Training Harmless AI at Scale*  
   作用：偏工程视角，适合回看数据规模、样本构造和训练流水线组织方式。  
   链接：https://www.transcendent-ai.com/post/training-harmless-ai-at-scale
