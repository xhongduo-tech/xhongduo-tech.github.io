## 核心结论

指令微调的作用，不是给模型增加“语言能力”，而是把预训练阶段已经学到的语言知识，重新组织成“按自然语言要求做事”的行为模式。预训练模型本质上只会做一件事：根据前文预测下一个 token。指令微调则把输入改写成“系统约束 + 用户意图 + 上下文”的结构，让模型学会在这个结构下输出目标答案。

一个最小结论可以写成下面两点：

1. 纯预训练模型擅长续写，不等于擅长执行任务。
2. 指令微调通过大量多任务 `(instruction, output)` 对，让模型把“看到描述就切换策略”的能力学出来。

玩具例子最容易说明这个差异。

假设输入是：

`Translate this sentence to Chinese: hello`

如果模型只接受过大规模续写训练，它可能继续生成和英语句子相关的文本，也可能输出定义、例句，甚至继续写英文。经过指令微调后，模型更可能稳定输出：

`你好`

这里“指令”就是自然语言任务描述，白话说，就是人直接告诉模型“你现在该干什么”。

下表对比三种训练方式的能力焦点：

| 训练方式 | 输入形式 | 学到的核心能力 | 泛化表现 |
|---|---|---|---|
| 原始预训练 | 普通文本前缀 | 续写语言分布 | 能写，但不一定按要求做 |
| 单任务微调 | 固定任务输入 | 学会一个映射 | 对该任务强，跨任务弱 |
| 指令微调 | 自然语言指令 + 上下文 | 识别意图并切换输出策略 | 对未见任务更容易零样本泛化 |

从目标函数看，监督微调阶段的核心就是：

$$
\mathcal{L}_{\text{SFT}}=\log P_\theta(y|x)
$$

其中 $x$ 是指令输入，$y$ 是期望回答。白话说，就是“看到这类要求时，把正确答案的概率拉高”。

如果进一步走到 InstructGPT 一类方法，训练目标会扩展为“既要符合人类偏好，又不能偏离原本会说人话的分布太远”：

$$
\max_\theta \mathbb{E}_{(x,y)\sim\pi_\theta}\left[r_\phi(x,y)-\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{SFT}}(y|x)}\right]+\gamma \mathbb{E}_{x\sim D_\text{pretrain}}[\log \pi_\theta(x)]
$$

这里的奖励模型、KL 惩罚、预训练混合项，分别对应“偏好方向”“稳定约束”“语言流畅性保底”。

---

## 问题定义与边界

指令微调要解决的问题可以精确定义为：

给定输入 $x=(s,u,c)$，其中：

- $s$ 是 system prompt，系统提示，白话说就是全局规则，比如“你是一个客服助手，回答要简洁且不能编造政策”。
- $u$ 是 user instruction，用户指令，白话说就是这次任务本身，比如“把下面对话总结成三条要点”。
- $c$ 是 context，上下文，白话说就是要处理的原始材料，比如一段对话、表格、代码片段。

目标是让模型输出 $y$，满足指令意图、格式要求和系统约束。

这和传统分类或翻译任务有两个重要不同：

1. 任务边界由自然语言描述，而不是固定字段定义。
2. 同一个模型要在很多任务之间动态切换，而不是只会一个任务。

一个真实工程例子是客服助手。用户输入：

- system: `你是企业客服助手，只输出事实，不做赔付承诺。`
- user: `Summarize the following conversation in bullet points.`
- context: 一段用户和客服的聊天记录

模型需要理解“Summarize”表示摘要任务，“bullet points”表示输出格式，“只输出事实”表示合规约束。也就是说，模型不仅要“看懂内容”，还要“看懂要求”。

可以把训练流程压缩成一个步骤链：

1. 预训练：学语言分布，得到会续写的底座模型。
2. 构造指令-回答对：把翻译、摘要、问答、分类等任务统一成自然语言描述。
3. SFT：让模型学会“按指令作答”。
4. RLHF：再用偏好数据把回答调得更稳、更像人想要的样子。

边界也要说清楚。指令微调不是万能方法，它通常只在下面场景有效：

| 场景 | 是否适合指令微调 | 原因 |
|---|---|---|
| 可用自然语言描述的任务 | 适合 | 任务意图能显式写进 prompt |
| 需要跨任务泛化 | 适合 | 多任务指令训练正是为此设计 |
| 极强精确性约束的结构化系统 | 需配合规则 | 仅靠生成模型不够稳 |
| 完全无标注、无模板的数据环境 | 效果受限 | 没有高质量指令-输出对，难以学到行为 |

FLAN 系列的关键发现之一是：任务数量通常比单个任务内样本数量更重要。白话说，让模型见过“很多种任务怎么被描述”，比只把“一个任务做很多遍”更能提升泛化。

下面这个表更能体现边界差异：

| 数据组织方式 | 模型学到什么 | 对新指令表现 |
|---|---|---|
| 没有指令模板 | 只记输入到输出的模式 | 遇到新描述容易失效 |
| 单任务微调 | 一个任务的固定习惯 | 跨任务切换能力弱 |
| 多任务指令微调 | 从描述推断任务类型 | 零样本和少样本更强 |

---

## 核心机制与推导

### 1. SFT：先把“按要求回答”学出来

SFT 是 Supervised Fine-Tuning，监督微调。白话说，就是拿标准答案直接教模型。

训练样本一般写成：

- 输入：`SYSTEM + USER + CONTEXT`
- 输出：`ASSISTANT RESPONSE`

目标函数仍然是标准语言模型对数似然：

$$
\mathcal{L}_{\text{SFT}}=\log P_\theta(y|x)
$$

更严格一点，若输出序列是 $y=(y_1,\dots,y_T)$，则：

$$
\log P_\theta(y|x)=\sum_{t=1}^{T}\log P_\theta(y_t|x,y_{<t})
$$

这表示模型在每一步都要把“正确下一个 token”的概率推高。

玩具例子如下：

- system: `You are a translation assistant.`
- user: `Translate this sentence to Chinese: hello`
- assistant: `你好`

如果只做 SFT，模型就会逐渐把“翻译到中文”这个任务描述和“输出中文译文”建立稳定映射。

### 2. 为什么多任务比单任务更重要

如果训练集只有翻译，模型容易把“输入里有英文句子”误当成“总该翻译”。一旦用户改问：

`Summarize this sentence in one short phrase: hello world from demo`

模型可能仍沿用翻译策略。

如果训练中同时包含翻译、摘要、情感分类、问答，模型才会学会一个更高层的规则：先判断任务意图，再决定输出形式。这个“先识别指令，再调用策略”的过程，就是指令微调的核心价值。

### 3. RLHF：再把回答调到更符合偏好

InstructGPT 的三阶段通常写成：

1. SFT
2. 奖励模型训练
3. PPO 优化

奖励模型 $r_\phi(x,y)$ 的作用，是学习“哪个回答更好”。白话说，就是把人工偏好压缩成一个打分器。

然后用 PPO 这类强化学习算法优化策略：

$$
\max_\theta \mathbb{E}_{(x,y)\sim\pi_\theta}\left[r_\phi(x,y)-\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{SFT}}(y|x)}\right]+\gamma \mathbb{E}_{x\sim D_\text{pretrain}}[\log \pi_\theta(x)]
$$

这里有三个关键项：

| 项 | 作用 | 白话解释 |
|---|---|---|
| $r_\phi(x,y)$ | 奖励高质量回答 | 鼓励更有帮助、更安全 |
| $-\beta \log \frac{\pi_\theta}{\pi_{\text{SFT}}}$ | KL 惩罚 | 别偏离 SFT 太远，防止“跑偏” |
| $\gamma \mathbb{E}[\log \pi_\theta(x)]$ | 预训练混合项 | 保留语言流畅性与基础能力 |

SFT、RLHF、只做 PPO 的差异可以直接对比：

| 方法 | 优点 | 风险 | 适用情况 |
|---|---|---|
| 仅 SFT | 实现简单、稳定 | 回答不一定最符合人类偏好 | 已有高质量监督数据 |
| SFT + RLHF | 更符合偏好与安全要求 | 工程复杂、成本高 | 对话助手、开放式生成 |
| 仅 PPO | 可能快速对齐奖励 | 极不稳定，容易奖励黑客 | 几乎不单独使用 |

“奖励黑客”是指模型学会投机取巧地骗高分，而不是真的完成任务。白话说，就是它懂得怎么拿分，不一定懂得怎么做好事。

---

## 代码实现

先看一个最小可运行的 Python 玩具实现。它不训练真实大模型，只模拟“指令模板 + 多任务路由”的核心思想，目的是把机制讲清楚。

```python
from dataclasses import dataclass

@dataclass
class Sample:
    system: str
    instruction: str
    input_text: str
    reference: str

def build_prompt(sample: Sample) -> str:
    return (
        f"SYSTEM: {sample.system}\n"
        f"USER: {sample.instruction}\n"
        f"INPUT: {sample.input_text}\n"
        f"ASSISTANT:"
    )

def simple_model(prompt: str) -> str:
    # 一个玩具策略：根据指令关键词切换任务
    lower = prompt.lower()
    if "translate" in lower and "hello" in lower:
        return "你好"
    if "sentiment" in lower and "great" in lower:
        return "positive"
    if "summarize" in lower:
        return "这是摘要。"
    return "unknown"

def sft_accuracy(dataset):
    correct = 0
    for sample in dataset:
        prompt = build_prompt(sample)
        pred = simple_model(prompt)
        if pred == sample.reference:
            correct += 1
    return correct / len(dataset)

dataset = [
    Sample(
        system="You are a helpful assistant.",
        instruction="Translate this sentence to Chinese.",
        input_text="hello",
        reference="你好",
    ),
    Sample(
        system="You are a helpful assistant.",
        instruction="Classify the sentiment.",
        input_text="movie was great",
        reference="positive",
    ),
    Sample(
        system="You are a helpful assistant.",
        instruction="Summarize the following text in one sentence.",
        input_text="News: product launch was delayed by weather.",
        reference="这是摘要。",
    ),
]

acc = sft_accuracy(dataset)
assert acc == 1.0
print("toy accuracy =", acc)
```

这个例子里，“模型”其实只是规则函数，但它体现了真实指令微调的数据组织方式：每条样本都包含 system、instruction、input、reference。真实训练时，`simple_model` 会被 Transformer 替代，`assert` 体现的是“训练样本上的基本一致性检查”。

下面给出更接近真实训练框架的伪代码：

```python
# 1. 数据准备
sample = {
    "system": "You are a customer support assistant. Be concise and factual.",
    "instruction": "Summarize the following conversation in bullet points.",
    "input": "User: My package is late...\nAgent: I checked the status...",
    "reference": "- Package delayed\n- Status checked\n- Next update tomorrow"
}

prompt = (
    f"<system>{sample['system']}</system>\n"
    f"<user>{sample['instruction']}\n{sample['input']}</user>\n"
    f"<assistant>"
)

target_text = sample["reference"]

# 2. SFT 前向与损失
input_ids = tokenizer(prompt)
target_ids = tokenizer(target_text)

logits = model(input_ids, labels=target_ids)
loss_sft = cross_entropy(logits, target_ids)

# 3. backward / update
loss_sft.backward()
optimizer.step()
optimizer.zero_grad()

# 4. RLHF/PPO 阶段
generated_ids = model.generate(input_ids)
logp_policy = policy_logprob(model, input_ids, generated_ids)
logp_ref = policy_logprob(sft_model, input_ids, generated_ids)

reward = reward_model_score(input_ids, generated_ids)
kl = logp_policy - logp_ref
ppo_objective = reward - beta * kl

# 5. PPO update
loss_ppo = -ppo_surrogate(ppo_objective, generated_ids)
loss_ppo.backward()
ppo_optimizer.step()
ppo_optimizer.zero_grad()
```

真实工程例子可以看在线客服或企业 Copilot。

假设公司要做一个内部知识助手，支持三类任务：

- 总结工单
- 改写客服回复
- 解释报错日志

如果只用 prompt engineering，团队每个场景都要手写 prompt，并且不同版本模型的输出不稳定。做指令微调后，可以把大量历史案例整理成统一格式：

- system: 角色和合规规则
- user: 任务描述
- input: 工单、日志、对话等内容
- assistant: 标准答案

这样模型就会在统一接口下工作。前端和服务端不需要为每个任务设计完全不同的推理链路，只需要更换 instruction 和 context。

---

## 工程权衡与常见坑

最常见的误区，不是模型太小，而是数据组织太差。

### 1. 只做单任务，模型不会真正“懂指令”

如果训练集只有翻译，模型容易形成错误归纳：看到英文就翻译。此时用户给摘要任务，模型仍可能输出翻译结果。这不是“模型笨”，而是训练目标没有要求它学会区分任务。

### 2. 没有自然语言模板，泛化会明显下降

只喂结构化字段，比如 `task_id=3`、`input=...`，模型能学会这个任务，但很难对新描述泛化。自然语言模板的价值，是让“任务语义”直接进入模型输入。

### 3. 多任务比例失衡，会把模型拉偏

如果 80% 数据都是问答，10% 是摘要，10% 是分类，模型容易把所有输入都往问答风格上靠。工程上通常要做任务重采样或配额控制。

### 4. 奖励模型会带入偏差

奖励模型来源于人工偏好，但人工偏好可能不稳定。比如标注员更喜欢“语气礼貌”的回答，模型就可能学会说得更圆滑，却牺牲信息密度。

下面这个表汇总常见坑：

| 常见坑 | 典型表现 | 规避措施 |
|---|---|---|
| 缺少指令模板 | 模型只能记样本模式，不能读懂新要求 | 每类任务至少设计多个自然语言模板 |
| 只做单任务微调 | 一换任务就失效 | 增加任务种类，而不只增加单任务样本 |
| 数据比例失衡 | 常见任务压制稀有任务 | 分桶采样、监控各任务验证集 |
| 奖励模型偏差 | 输出变得讨好但不准确 | 偏好标注标准化，加入事实性检查 |
| system prompt 不稳定 | 同任务输出风格波动大 | 固化系统层规则，不频繁改写 |
| 评测只看总分 | 某些子任务退化却未发现 | 按任务类型拆分评测指标 |

一个实用的数据流程是：

1. 每类任务至少写 2 到 5 个指令模板。
2. 同一语义任务换不同说法，比如“总结下面内容”“提炼下面对话要点”。
3. 修改模板后单独回归测试，确认不是只对某个措辞有效。
4. 评测集按任务划分，分别看翻译、摘要、问答、分类的准确率或人工偏好。

这比盯着一个总损失值更有工程意义。

---

## 替代方案与适用边界

如果数据少、算力紧，第一选择通常不是立刻做全量指令微调，而是先用 prompt engineering。它的本质是把任务说明直接写进推理时输入，不改模型参数。

例如：

`请把下面句子翻译成中文：hello`

这在很多强基座模型上已经能得到不错结果。但它有两个问题：

1. 稳定性依赖模型版本和 prompt 写法。
2. 一旦任务变多，prompt 模板管理会迅速复杂化。

LoRA 和 Adapter 是另一类常见替代方案。它们不是替代“指令微调思路”，而是替代“全参数更新方式”。白话说，任务还是指令微调任务，只是参数更新更轻量。

| 方案 | 算力成本 | 数据需求 | 输出稳定性 | 适用边界 |
|---|---|---|---|---|
| Prompt only | 最低 | 低 | 一般 | 快速验证、低频场景 |
| 全参数指令微调 | 最高 | 中到高 | 高 | 核心产品、长期迭代 |
| LoRA/Adapter 指令微调 | 中低 | 中 | 较高 | 算力受限但要定制化 |
| SFT + RLHF | 很高 | 高 | 最高，但工程复杂 | 面向开放式对话产品 |

适用边界可以直接这样理解：

- 如果产品还在探索期，先 prompt，再收集失败样本。
- 如果任务已稳定、请求量高、输出格式要求严，转向指令微调更合适。
- 如果预算有限，但必须定制行为，用 LoRA 做指令微调通常更现实。
- 如果面对开放式助手、安全要求高、用户偏好重要，再考虑 RLHF。

低延迟产品也常采用“先 prompt，后微调”的路线。原因很简单：前期先验证需求，后期再把高频 prompt 固化到模型参数里，减少在线提示长度和不稳定性。

---

## 参考资料

| 资料 | 主要贡献 | 工程启示 |
|---|---|---|
| Wei et al., 2022, FLAN | 证明多任务指令微调能显著提升零样本能力 | 任务种类比单任务堆量更关键 |
| Ouyang et al., 2022, InstructGPT | 提出 SFT → RM → PPO 的对齐流程 | 仅 SFT 不足以完全对齐偏好 |
| IBM Instruction Tuning 综述 | 解释自然语言模板的重要性 | 模板设计本身就是训练资产 |

1. Wei, J. et al. *Finetuned Language Models Are Zero-Shot Learners*. ICLR 2022.  
   链接：https://research.google/pubs/finetuned-language-models-are-zero-shot-learners/  
   注释：指令微调的代表性工作，核心结论是多任务自然语言指令能显著提升零样本泛化。

2. Ouyang, L. et al. *Training language models to follow instructions with human feedback*.  
   链接：https://arxiv.org/abs/2203.02155  
   注释：InstructGPT 论文，明确了 SFT、奖励模型、PPO 三阶段关系。

3. IBM. *What Is Instruction Tuning?*  
   链接：https://www.ibm.com/think/topics/instruction-tuning  
   注释：适合工程视角快速理解“为什么模板重要、为什么它不同于普通微调”。

4. OpenAI Spinning Up: PPO  
   链接：https://spinningup.openai.com/en/latest/algorithms/ppo.html  
   注释：如果要理解 PPO 本身，而不是只看 RLHF 流程，这份资料很直接。
