## 核心结论

指令微调里的“灾难遗忘”，本质不是模型突然坏了，而是训练分布变了。预训练阶段，模型在广泛语料上学习，近似拟合的是 $p_{\text{pretrain}}$；进入 SFT（Supervised Fine-Tuning，监督微调，意思是拿“指令-回答”配对数据继续训练）后，优化目标会把参数继续推向更窄的 $p_{\text{sft}}$。如果这个新分布太窄，模型原本的通用续写、冷门知识、语言建模能力就会被覆盖。

更直接地说：模型不是“学会了指令”就不付代价。它往往是用一部分旧能力，换来更强的“按要求回答”能力。只看指令集分数，模型会显得更听话；但如果同步看 MMLU、PopQA、预训练语料上的 perplexity（困惑度，意思是语言模型对文本是否“熟悉”）或 loss，常能看到通用能力下滑。

工程上最稳的缓解办法不是玄学调参，而是把预训练目标带回训练过程。最常见形式是：

$$
L_{\text{total}}(\lambda)=\lambda L_{\text{sft}}+(1-\lambda)L_{\text{pt}}
$$

其中 $L_{\text{sft}}$ 是指令数据损失，$L_{\text{pt}}$ 是预训练风格文本的 next-token loss，$\lambda$ 控制两者权重。Béthune 等在 ICML 2025 的结果给出一个很实用的信号：哪怕只在微调混入约 1% 的预训练数据，也能显著抑制对原预训练分布的遗忘。对很多真实系统，5% 到 20% 的 replay（回放，意思是把旧数据重新喂给模型）通常更稳。

一个新手能立刻理解的玩具例子是：你把模型只在大量客服话术上再训练，它回答“退款流程”“工单升级”会更像人工客服，但再问“谁发明了电灯”，答案可能开始犹豫、绕圈，甚至错误。原因不是知识库被删除，而是参数更新后，模型更习惯走“客服回答模板”那条路。若训练时插入少量通用文本，这种偏移往往会明显减轻。

---

## 问题定义与边界

灾难遗忘在这里的定义，不是“模型所有能力全面崩溃”，而是：在 SFT 过程中，模型对原预训练分布的拟合能力下降，进而表现为通用语言建模、长尾知识访问、跨域问答或泛化能力变差。

可以把它写成一个分布偏移问题。预训练阶段优化的是：

$$
L_{\text{pt}} = \mathbb{E}_{x \sim p_{\text{pretrain}}}[-\log p_\theta(x)]
$$

SFT 阶段如果只看指令集，则优化的是：

$$
L_{\text{sft}} = \mathbb{E}_{(q,a) \sim p_{\text{sft}}}[-\log p_\theta(a \mid q)]
$$

当 $p_{\text{sft}}$ 比 $p_{\text{pretrain}}$ 窄很多时，只优化 $L_{\text{sft}}$，就等于默认接受“对旧分布表现下降”这个副作用。

边界也要讲清楚。

| 问题 | 是否容易触发遗忘 | 原因 |
|---|---:|---|
| 指令集非常窄，只覆盖单一业务话术 | 高 | 梯度长期朝同一类表达和知识区域收缩 |
| 微调步数多、学习率高 | 高 | 参数偏移大，旧能力更容易被覆盖 |
| 每个 batch 混入少量通用文本 | 低到中 | 旧分布梯度持续存在，能提供“回拉力” |
| 用 LoRA/Adapter 只训练少量参数 | 中 | 参数改动更小，但不代表不会忘 |
| 同时监控 MMLU/PPL/长尾 QA | 更容易及时发现 | 不会被单一指令分数误导 |

新手版类比可以这样理解：如果一个人连续一个月只读莎士比亚台词，他的表达会越来越像戏剧对白，但这不代表他数学、百科、说明文写作都保持不变。若每天再读一点百科和教材，他就不容易把基础能力丢掉。这个类比不等于模型内部机制完全一样，但有助于理解“分布越窄，遗忘风险越高”。

所以，灾难遗忘的边界不是“只有连续多任务学习才会发生”。哪怕只有一次指令微调，只要训练目标足够偏、数据足够窄、步数足够多，它就会发生。

---

## 核心机制与推导

核心机制可以压缩成一句话：只用指令样本训练时，梯度只代表指令分布的偏好，于是参数会沿着“更像指令回答”的方向持续移动，而不是沿着“同时保留通用建模能力”的方向移动。

把总目标写出来：

$$
L_{\text{total}}(\lambda)=\lambda L_{\text{sft}}+(1-\lambda)L_{\text{pt}}
$$

它的梯度就是：

$$
\nabla_\theta L_{\text{total}}
=
\lambda \nabla_\theta L_{\text{sft}}
+
(1-\lambda)\nabla_\theta L_{\text{pt}}
$$

这条式子的意义非常直接：

1. 如果 $\lambda=1$，你就在做纯 SFT，所有更新都由指令数据决定。
2. 如果 $0<\lambda<1$，每一步更新都会被预训练分布“拉回一点”。
3. $\lambda$ 越大，模型越专注新任务；$\lambda$ 越小，模型越保守，更愿意保留旧能力。

这和“按比例往 mini-batch 里插入预训练样本”本质上是等价的。比如一个 batch 里 8 份是指令数据，2 份是预训练文本，那么从期望上看，你就在近似优化 $\lambda=0.8$ 的混合目标。

再看一个玩具例子。假设模型原来知道：

- “谁发明了电灯”常见答案是“托马斯·爱迪生”
- “退款流程”通常不是预训练语料里的高频主题

如果你连续很多步只训练客服指令，模型会把大量参数容量拿去优化“礼貌回复”“流程模板”“工单措辞”“拒绝与转接规则”。这些更新本身没错，但它们会和原有知识访问路径发生竞争。于是出现一种常见现象：模型回答客服问题更整齐了，但通用事实问答更不稳定了。

真实论文也支持这种机制理解。Wu 等在 NAACL 2024 的分析指出，指令微调会增强模型对指令部分的识别，并推动注意力头和前馈层朝“面向用户任务”的方向重组。MDPI 2026 那篇基于 OLMo-2 的工作则进一步观察到：SFT 会削弱对预训练语料的 verbatim memorization（逐字记忆，意思是对训练文本原句的精确再现能力），同时改善礼貌性、结构化等对齐属性，却让知识密集任务变差。两者拼在一起，刚好解释了为什么“更会按格式说话”和“更会保留原知识”不是同一件事。

可以把三种训练状态粗略画成这样：

$$
\text{SFT-only: } \nabla \approx \nabla L_{\text{sft}}
\quad\Longrightarrow\quad
\theta \text{ 快速偏向窄域}
$$

$$
\text{Mixed: } \nabla \approx \lambda \nabla L_{\text{sft}} + (1-\lambda)\nabla L_{\text{pt}}
\quad\Longrightarrow\quad
\theta \text{ 偏移但不失控}
$$

$$
\text{Replay + Low LR + PEFT: }
\quad \Delta\theta \text{ 更小，旧能力更容易保住}
$$

这里的 PEFT（参数高效微调，意思是只训练少量新增或低秩参数）不是万能药，但它减少了全模型大幅漂移的机会，所以常和 replay 一起用。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不训练真正的大模型，但把“混合采样”和“混合损失”的工程结构表达清楚了。

```python
from itertools import cycle, islice

def interleave_batches(instr_data, pt_data, pt_ratio=0.2, total_steps=10):
    assert 0.0 <= pt_ratio <= 1.0
    assert len(instr_data) > 0 and len(pt_data) > 0

    pt_steps = round(total_steps * pt_ratio)
    instr_steps = total_steps - pt_steps

    schedule = ["sft"] * instr_steps + ["pt"] * pt_steps
    # 一个简单的均匀打散策略：每隔固定位置插入 pt
    schedule.sort(key=lambda x: 0 if x == "sft" else 1)

    instr_iter = cycle(instr_data)
    pt_iter = cycle(pt_data)

    mixed = []
    for tag in schedule:
        if tag == "sft":
            mixed.append((tag, next(instr_iter)))
        else:
            mixed.append((tag, next(pt_iter)))
    return mixed

def total_loss(loss_sft, loss_pt, lambda_=0.85):
    assert 0.0 <= lambda_ <= 1.0
    return lambda_ * loss_sft + (1 - lambda_) * loss_pt

# 玩具数据
instr_samples = ["退款流程说明", "如何升级工单", "请用礼貌语气回复用户"]
pt_samples = ["Thomas Edison invented the light bulb.", "The capital of France is Paris."]

schedule = interleave_batches(instr_samples, pt_samples, pt_ratio=0.2, total_steps=10)

num_sft = sum(1 for tag, _ in schedule if tag == "sft")
num_pt = sum(1 for tag, _ in schedule if tag == "pt")

assert num_sft == 8
assert num_pt == 2

# 假设一个 step 上观测到的两类损失
loss = total_loss(loss_sft=1.2, loss_pt=1.8, lambda_=0.85)
assert abs(loss - 1.29) < 1e-9

print("schedule:", schedule)
print("mixed loss:", loss)
```

在真实训练里，思路通常是下面这样：

```python
lambda_ = 0.85

for instr_batch, pt_batch in zip(instr_loader, cycle(pt_loader)):
    loss_sft = model.compute_loss(instr_batch)
    loss_pt = model.compute_loss(pt_batch)
    loss = lambda_ * loss_sft + (1 - lambda_) * loss_pt

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

几个实用建议：

- `lambda=0.8` 到 `0.95` 常被当作起点，不是定律。
- 如果预训练回放样本质量高，哪怕只占 5%，也可能很有效。
- 若显存紧张，不一定非要同一步同时算两种 loss，也可以按 batch ratio 做交替采样。
- 如果你已经用了 LoRA，仍建议保留一小部分通用 replay，因为 LoRA 只能减少参数改动，不能自动保证知识不丢。

一个新手场景的完整闭环是：先把模型在客服数据上 SFT，发现“谁发明了电灯”开始答错；然后把 5% 的通用文本混回训练，重新微调，再测这个问题恢复正常，同时客服任务分数基本不掉。这个现象就足以说明，问题不是“模型不会学指令”，而是“只学指令时会把别的能力压掉”。

---

## 工程权衡与常见坑

真实工程里，最危险的坑不是不会缓解，而是没有正确观测。

很多团队只看两类指标：

- 指令集上的 loss
- 业务任务的通过率

这不够。因为灾难遗忘恰恰经常表现为“业务任务更好了，但通用能力更差了”。所以必须做双轨监控。

| 观察指标 | 看什么 | 常见坏信号 | 推荐动作 |
|---|---|---|---|
| 指令任务 accuracy / win rate | 是否更会按要求回答 | 上升 | 不能单独解释模型变好 |
| MMLU / PopQA | 通用知识与长尾事实 | 下滑 | 提高 replay 比例，降低 LR |
| 预训练文本 PPL 或 loss | 语言建模保真度 | 持续变差 | 混入通用文本或恢复 LM 目标 |
| OOD 问答集 | 跨域泛化 | 答案更模板化、更空泛 | 缩短训练步数，提早停止 |
| 回归比率 regression ratio | 新旧指标同时比较 | 新任务涨、旧任务明显跌 | 重新平衡 $\lambda$ |

一个真实工程例子：做医疗助手时，团队常会把模型在医学问答、病历摘要、合规回复上连续训练几天。短期看，专业术语更像了，回答也更保守；但如果同时跑通用 benchmark，常会看到 MMLU 下降、开放事实问答更不稳定。一个常见修复方式不是“再堆更多医疗数据”，而是每轮混入一小部分通用英文维基或原预训练风格文本，再配合更低学习率。这样做的目标不是让模型重新变回基础模型，而是给它保留一条访问通用知识的路径。

常见坑主要有五个：

1. 只看 SFT 验证集，不看通用基准。
2. replay 比例太低，低到梯度里几乎没有旧分布信号。
3. 学习率沿用预训练或大规模微调习惯，导致参数漂移过大。
4. 把 LoRA 当成免疫遗忘的保证，这在实践中并不成立。
5. 用非常窄的单业务模板数据训练很多 epoch，把模型训成“只会这一种说话方式”。

如果你的场景是金融、医疗、法务这种高风险问答，建议默认把“保留旧能力”视为一等公民目标，而不是事后补救项。

---

## 替代方案与适用边界

数据混合不是唯一方案，但通常是第一选择，因为最直接、最可解释。

| 方案 | 实现复杂度 | 通用能力损失风险 | 训练开销 | 适用场景 |
|---|---:|---:|---:|---|
| 直接 SFT | 低 | 高 | 低 | 低风险、短生命周期任务 |
| SFT + 预训练数据混合 | 中 | 低到中 | 中 | 大多数生产微调 |
| SFT + replay buffer | 中 | 低到中 | 中 | 有可保存旧数据样本时 |
| LoRA / Adapter + replay | 中 | 中到低 | 低到中 | 显存有限，希望少改主干 |
| EWC 类正则 | 高 | 中 | 中 | 无法方便保存旧数据时 |
| 蒸馏到新模型 | 高 | 低到中 | 高 | 对稳定性要求极高 |

这里的 replay buffer 就是保存一批代表性旧样本，训练时反复回放。EWC（Elastic Weight Consolidation，弹性权重固化）则是给“重要旧参数”更强约束，不让它们轻易被新任务改掉。它适合“不能直接保留旧数据”的环境，但在大语言模型里实现与调参通常比数据混合更麻烦。

对零基础读者，一个最实用的判断方法是：

- 如果你允许模型明显变成“专项助手”，可以接受一些旧能力损失，直接 SFT 也许够用。
- 如果你要求它既懂专业任务，又别丢基础常识，优先做 `SFT + replay`。
- 如果资源紧张，先用 `LoRA + 5%~10% replay + 低学习率`，这是很常见的折中。
- 如果是高风险系统，不要只做单一路线，至少要有混合训练和双轨评测。

所以，“替代方案”的真正边界不在算法名字，而在你是否接受模型偏离原始分布。只要答案是不接受，就应该把保留旧分布当成目标显式写进训练设计里。

---

## 参考资料

- Louis Béthune, David Grangier, Dan Busbridge, Eleonora Gualdoni, Marco Cuturi, Pierre Ablin. *Scaling Laws for Forgetting during Finetuning with Pretraining Data Injection*. ICML 2025. 关键贡献：量化了微调遗忘与预训练数据注入的关系，并给出“约 1% 注入即可显著防忘”的实证信号。https://proceedings.mlr.press/v267/bethune25a.html
- Xuansheng Wu, Wenlin Yao, Jianshu Chen, Xiaoman Pan, Xiaoyang Wang, Ninghao Liu, Dong Yu. *From Language Modeling to Instruction Following: Understanding the Behavior Shift in LLMs after Instruction Tuning*. NAACL 2024. 关键贡献：从注意力和前馈层解释了指令微调如何改变模型内部行为。https://aclanthology.org/2024.naacl-long.130/
- Jie Zhang, Chi-Ho Lin, Suan Lee. *Instruction Fine-Tuning Through the Lens of Verbatim Memorization*. *Electronics*, 2026. 关键贡献：基于 OLMo-2 观察到 SFT 会削弱对预训练语料的逐字记忆，同时改善对齐风格，却让知识密集任务退化。https://www.mdpi.com/2079-9292/15/2/377
- Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, Yue Zhang. *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning*. 2023/2025. 关键贡献：系统验证了 LLM 在持续微调中确实存在灾难遗忘，并讨论了模型规模和架构差异。https://arxiv.org/abs/2308.08747
- James Kirkpatrick et al. *Overcoming Catastrophic Forgetting in Neural Networks*. PNAS 2017. 关键贡献：提出 EWC，为“重要参数少动一点”这类正则方法提供经典基线。https://pmc.ncbi.nlm.nih.gov/articles/PMC5380101/
