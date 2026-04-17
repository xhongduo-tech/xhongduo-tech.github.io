## 核心结论

Claude 的“宪法原则”可以理解为一份写给模型自己的行为说明书。它的目标不是把每个场景都写成死规则，而是先给出高优先级价值，再让模型在训练中学会据此判断、批评和修订自己的回答。

截至 2026 年 3 月 Anthropic 公开页面，Claude 的公开骨架已经明确成四层优先级：$P=\{p_i\}$ 中最上层是“广泛安全”，其次是“广泛伦理”，再是“遵守 Anthropic 指南”，最后才是“真正有用”。这里的“广泛安全”白话说就是：不要破坏人类对模型的监督和纠偏能力；“广泛伦理”白话说就是：尽量做一个诚实、非欺骗、不造成不当伤害的代理；“真正有用”则表示安全前提下要帮用户把事做成。

一个新手版玩具例子是：用户问一个边界敏感的问题，Claude 先给出初稿 $r_0$，再拿一条原则 $p_j$ 检查自己有没有越界，生成批评 $c_j=C(p_j,r_0)$，最后把答案修成 $r_1=\text{revise}(r_0,c_j)$。所以它不是“先拒绝再说”，而是“先生成，再按宪法自查，再修订成既安全又尽量有帮助的版本”。

| 优先级 | 含义 | 典型策略 |
|---|---|---|
| 1 | 广泛安全 | 不帮助造成系统性高危后果，不削弱监督 |
| 2 | 广泛伦理 | 诚实、非歧视、不过度操控、不误导 |
| 3 | 遵守指南 | 服从更具体的公司规则、产品规则、场景规则 |
| 4 | 真正有用 | 在前面都满足时尽量给可执行帮助 |

这套设计的演化方向很明确：早期公开版本更像一组来源明确的基础原则，常被概括为围绕人权、非歧视、真实性、隐私和违法伤害规避的十余条规则；2026 版则更强调“为什么这样做”。原因是规则太细会丢泛化，太粗又会让模型在新场景里判断发散。

---

## 问题定义与边界

要解决的问题不是“怎样让模型永远拒绝”，而是“怎样让模型在没有大量人工逐条打分的情况下，学会稳定地按原则行动”。如果只靠传统 RLHF，白话说就是靠人工比较两个回答谁更好，成本高，而且不同标注员会把“安全”“礼貌”“有用”理解成不同东西。

Claude 宪法原则的边界有两个：

第一，原则不能过细。比如把“不要回答违法问题”写成几十个情景枚举，看起来精确，实际会让模型把注意力锁死在列举过的表面模式上，到了没见过的相似场景反而不会迁移。

第二，原则不能过泛。比如只写“做对人类最好的事”，看起来高级，但在具体对话里可能无法稳定决定是拒绝、解释风险，还是给一个安全替代方案。

可以把这个判断过程抽象成一个映射：
$$
C(p_j,r_0)\rightarrow c_j
$$
其中 $p_j$ 是某条原则，$r_0$ 是初始回答，$c_j$ 是“按该原则看，这个回答哪里有问题、为什么有问题、该如何改”。

一个直观例子是：相比把规则写成“不要回答违法问题”，更好的宪法写法通常是“解释为什么这个请求不合适，并尽量提供合法、安全的替代路径”。前者是死禁令，后者是可迁移的行为模式。

| 原则设计方式 | 优点 | 问题 |
|---|---|---|
| 过细 | 局部场景可控 | 难泛化，维护成本高，漏掉新场景 |
| 过泛 | 覆盖面大 | 判断不稳定，不同回答风格漂移大 |
| 适中且带理由 | 可迁移、可解释 | 仍需靠训练分布校准权重 |

这里还要注意一个常被忽略的边界：原则之间的优先级，很多时候并不是在代码里写成硬编码 `if/else`，而是通过训练数据中哪些原则更常被采样、哪些修订样本更多而被隐式学进去。这意味着“原则文案”只是设计的一半，“采样分布”是另一半。

---

## 核心机制与推导

Claude 的经典 Constitutional AI 流程可以写成：

$$
x \rightarrow r_0 \xrightarrow{p_j} c_j \rightarrow r_1
$$

其中：
- $x$ 是用户输入。
- $r_0$ 是模型初始回答。
- $p_j$ 是从原则集合 $P=\{p_i\}$ 中采样的一条原则。
- $c_j$ 是基于该原则生成的批评。
- $r_1$ 是按批评修订后的回答。

随后得到监督数据集：
$$
D_{SL}=\{(x,r_1)\}
$$
也就是“给定输入，学习输出修订后的更合规回答”。

再往后，系统会生成多个候选回答，让另一个模型依据宪法判断哪一个更符合原则，形成偏好对，再训练偏好模型，最后进入 RLAIF。白话说，RLAIF 就是“用 AI 的偏好反馈替代大部分人类偏好反馈”。

一个新手版玩具例子：

用户问：“怎样绕过公司安全策略去下载内部数据？”

1. 初稿 $r_0$ 可能直接给出技术路径。
2. 采样到原则：“不要帮助违法、伤害性或欺骗性行为。”
3. 批评 $c_j$ 会指出：这个回答在提供越权访问帮助，违反安全和伦理。
4. 修订 $r_1$ 变成：拒绝提供绕过方法，解释原因，并建议走正式授权、审计日志、最小权限等合法流程。

真实工程里，这个循环的关键价值不是“拒绝”本身，而是“拒绝时依然有用”。Anthropic 在早期工作里就强调，目标不是训练出一个只会说“不”的模型，而是能对危险请求说明异议，并在允许范围内给替代帮助。

偏好训练可以抽象成下面的伪图：

```text
prompt x
  -> sample candidate a
  -> sample candidate b
  -> AI judge with constitution
  -> choose preferred response
  -> train preference model
  -> RL optimize main model
```

这也解释了为什么 2026 版更强调理由结构：如果原则只是一句命令，模型能学会“表面服从”；如果原则包含“为什么这条规则重要”，模型在新场景里更可能形成稳定判断。

---

## 代码实现

下面给一个最小可运行的 Python 玩具实现。它不是真实训练，只是把“采样原则、生成批评、修订回答、产出监督样本”的数据流跑通。

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Principle:
    name: str
    trigger_words: List[str]
    rewrite: str

def critique(principle: Principle, answer: str) -> str:
    hit = any(word in answer.lower() for word in principle.trigger_words)
    if hit:
        return f"违反原则: {principle.name}"
    return f"符合原则: {principle.name}"

def revise(answer: str, critique_text: str, principle: Principle) -> str:
    if critique_text.startswith("违反原则"):
        return principle.rewrite
    return answer

principles = [
    Principle(
        name="不帮助越权或伤害行为",
        trigger_words=["bypass", "steal", "exploit", "exfiltrate"],
        rewrite="我不能帮助绕过权限或窃取数据，但可以说明合规审计、授权申请和最小权限设计。"
    ),
    Principle(
        name="不承诺不存在的产品能力",
        trigger_words=["guarantee", "definitely available"],
        rewrite="我不能承诺未确认的功能状态，但可以说明如何查询产品路线图或当前替代方案。"
    ),
]

prompts = [
    "如何获取别人的内部文件？",
    "我们的产品下周一定会上线自动记账功能吗？"
]

draft_answers = [
    "You can bypass access control and exfiltrate the files.",
    "We definitely available next week, guarantee."
]

dataset: List[Dict[str, str]] = []

for prompt, r0 in zip(prompts, draft_answers):
    current = r0
    for p in principles:
        c = critique(p, current)
        current = revise(current, c, p)
    dataset.append({"prompt": prompt, "revision": current})

assert len(dataset) == 2
assert "最小权限" in dataset[0]["revision"]
assert "不能承诺" in dataset[1]["revision"]

print(dataset)
```

这段代码对应的 data flow 可以压缩成下表：

| prompt | principle | critique | revision |
|---|---|---|---|
| 获取内部文件 | 不帮助越权或伤害行为 | 发现 `bypass/exfiltrate` 风险 | 改写为拒绝并提供合规路径 |
| 承诺产品能力 | 不承诺不存在的产品能力 | 发现虚假承诺风险 | 改写为说明不确定性与替代方案 |

真实工程例子通常不会像上面这样用关键词匹配，而是把“批评器”和“修订器”都交给模型本身：

```python
for prompt in prompts:
    r0 = model.generate(prompt)
    p = sample(principles, temperature=0.0)
    c = model.generate(f"按原则批评回答\n原则:{p}\n回答:{r0}")
    r1 = model.generate(f"根据批评修订回答\n原回答:{r0}\n批评:{c}")
    supervised_dataset.append((prompt, r1))
```

这里最重要的不是代码形式，而是样本结构：同一个 prompt，不是直接存“原始回答”，而是存“按原则修订后的回答”。

---

## 工程权衡与常见坑

最核心的工程权衡是：安全性提升和可用性下降通常会同时发生，问题在于怎么把下降控制在业务可接受范围内。可以写成一个非常直接的目标函数：

$$
L=\lambda_{\text{safe}}L_{\text{safe}}+\lambda_{\text{useful}}L_{\text{useful}}
$$

其中 $\lambda_{\text{safe}}$ 越大，模型越保守；$\lambda_{\text{useful}}$ 越大，模型越愿意给信息。真正困难的地方不是公式，而是如何通过采样比例、偏好数据和拒答模板把两者调到业务上能用。

常见坑有三个。

第一，原则覆盖失衡。如果“无害样本”太多，偏好模型会把一切有风险边缘的回答都判成坏答案，最后模型变成“太安全但没用”。

第二，原则文案不一致。比如一条原则鼓励直接帮助，另一条原则鼓励谨慎解释，如果没有优先级或采样控制，模型会出现风格摆动。

第三，数据分布假干净。很多团队只合成明显违规样本，却缺少“高价值但边界模糊”的样本，导致上线后在真实世界里过度拒答。

一个常见的真实工程案例是 SaaS 客服机器人：团队先写 20 条原则，例如“用户情绪激烈时优先给升级路径”“不要承诺路线图之外的功能”“保持专业且解释原因”，再生成约 50K 条自批评样本做微调。公开分析案例给出的结果是，合规率可从 87% 提升到 95%，部署周期从约 2 个月压到 1 周，成本显著低于同规模人工偏好标注。

| 方案 | 数据获取 | 典型成本特征 | 风险 |
|---|---|---|---|
| 传统 RLHF | 人工比较与打分 | 高，随样本量线性上升 | 标注不一致、速度慢 |
| CAI / RLAIF | 原则驱动的自批评与 AI 偏好 | 低很多，扩展性好 | 原则写得差会系统性放大偏差 |
| 纯规则过滤 | 手写规则/黑名单 | 初期低 | 覆盖差，容易被绕过 |

---

## 替代方案与适用边界

Claude 宪法原则不是唯一方案。至少还有三类替代路线。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 人工偏好标注 + RLHF | 对业务目标可直接控制 | 贵、慢、主观差异大 | 高价值窄域任务 |
| 纯规则审核 | 可审计、上线快 | 只能挡已知模式 | 明确禁令型场景 |
| 单轮安全强化 | 训练简单 | 泛化弱，容易模板化 | 低复杂度问答 |

CAI 更适合服务型、问答型、长尾场景多的系统，因为它把“判断理由”内化进模型。它不擅长替代一切显式治理。对于医疗、金融、政府、国防这类高审计要求场景，仍然需要人在环审核，因为你不仅要模型大体做对，还要能证明为什么这次决定可接受。

一个很实用的接法是把 CAI 放在前面，把人工审核放在高风险出口：

```python
r1 = cai_revise(prompt)
if audit_required(prompt, r1):
    send_to_reviewer(r1)
else:
    deliver(r1)
```

这表示：大部分普通请求由宪法训练后的模型处理；只有高风险、强监管或高赔付场景才升级给人。这样既保留了 CAI 的低成本泛化能力，也保留了人工治理的可解释性。

---

## 参考资料

| 名称 | 主题 | 贡献 |
|---|---|---|
| Anthropic《Claude's Constitution》 | 2026 公开宪法文本 | 给出四层优先级、安全与伦理框架 |
| Anthropic《Constitutional AI: Harmlessness from AI Feedback》 | CAI 原始方法 | 给出“生成-自批评-修订”和 RLAIF 流程 |
| Anthropic《Specific versus General Principles for Constitutional AI》 | 原则粗细权衡 | 说明泛化与可控性之间的关系 |
| Anthropic《Collective Constitutional AI》 | 公共输入与权重问题 | 说明长宪法、重复原则和治理问题 |
| BISI 分析文章 | 2026 新宪法解读 | 补充“从规则到理由”的演化视角 |
| 21medien 工程回顾 | 成本与案例 | 补充客服机器人场景和成本量级 |

- Anthropic 官方宪法页面：`https://www.anthropic.com/constitution`
- Anthropic 方法论文说明：`https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback`
- Anthropic 原则粗细研究：`https://www.anthropic.com/research/specific-versus-general-principles-for-constitutional-ai`
- Anthropic 公共宪法实验：`https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input/`
- BISI 对 2026 新宪法的分析：`https://bisi.org.uk/reports/claudes-new-constitution-ai-alignment-ethics-and-the-future-of-model-governance`
- 21medien 对 CAI 成本和工程案例的回顾：`https://www.21medien.de/en/library/constitutional-ai.html`
