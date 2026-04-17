## 核心结论

合成数据在 SFT（Supervised Fine-Tuning，监督微调，意思是用“输入-标准输出”样本直接教模型按要求回答）里的核心价值，不是替代人工标注，而是降低冷启动成本、扩大任务覆盖、把有限人工样本的作用放大。

更准确地说，它通常承担三个任务：

1. 让基础模型先学会任务格式，比如“看到指令就按格式答”“多轮对话要延续上下文”“领域问答要输出指定模板”。
2. 用低成本覆盖高频场景和边界场景，比如同一意图的不同问法、长尾错误输入、格式扰动。
3. 在真实数据很少时，先把模型拉到“可用区间”，再用人工数据做最后校准。

对初学者最重要的结论是：**合成数据最有效的用法往往是“先多后少”**。前期可以用大量合成数据把模型“预热”，后期再用少量高质量人工数据把模型拉回真实分布。一个常见起步方案是：先生成 1000 条指令-回答对，再人工审校其中 200 条，先训练合成部分，再用审校后的真实样本精调。这样做的目标不是一步到位追求最优性能，而是在几天内让模型先具备基本的 instruction following 能力。

下面这个表可以把“合成阶段”和“人工阶段”的分工看清楚：

| 阶段 | 主要目标 | 典型规模 | 数据来源 | 验证机制 |
| --- | --- | --- | --- | --- |
| 合成阶段 | 快速覆盖任务模式、格式和高频场景 | 大，常见是人工的 3-10 倍 | 教师模型生成、规则扩写、模板填充 | 规则过滤、去重、模型判别器、人抽样检查 |
| 人工阶段 | 校准真实语气、真实错误分布和安全边界 | 小，但质量要求高 | 人工标注、专家修订、真实业务日志清洗 | 全量审校、抽检一致性、离线评测 |

合成数据确实可能带来性能提升，但收益不是线性增长。数据量从 0 增加到“足够覆盖主要模式”时，收益通常很大；继续堆更多相似样本，边际收益会快速下降，甚至因为分布偏移而伤害泛化。这个现象可以简单写成“先快后慢”的收益曲线：当覆盖率从 $c_1$ 增长到 $c_2$ 时，模型收益增加，但当新增样本主要重复已有模式时，$\Delta \mathrm{Gain}$ 会明显变小。

---

## 问题定义与边界

SFT 的目标不是让模型“懂世界”，而是让模型**稳定地按指令输出目标形式**。如果基础模型已经有通用知识，SFT 更像是在教它“怎么做题、怎么说话、怎么遵守格式”。

因此，讨论“合成数据在 SFT 中的应用”，先要明确边界：你到底在合成什么。

常见的三类场景是：

| 场景 | 合成什么 | 目标能力 |
| --- | --- | --- |
| 指令遵循 | 指令-响应对、格式约束样本、拒答样本 | 看懂任务并按要求输出 |
| 对话能力 | 多轮上下文、角色设定、澄清追问、历史引用 | 连续对话和状态保持 |
| 领域知识 | 垂直领域问答、结构化查询、规范写作模板 | 专业任务执行 |

这里要避免一个常见误解：**合成数据不是“随便生成点问答”**。它必须满足三个约束。

第一，语言质量要稳定。句子不一定要华丽，但必须清晰、一致、可学。  
第二，任务类型要收敛。不能医疗问答、闲聊、代码修复、诗歌生成全混在一起。  
第三，领域边界要明确。模型以后要做什么，合成数据就围绕什么生成。

玩具例子：如果你在做一个医疗问答 SFT，小模型只允许处理“问病症-给就医建议”的规范回复，那你就应该只生成这类指令对，例如“发烧三天是否需要就医”“咳嗽伴胸闷怎么办”，并限制输出为“风险判断 + 建议行动 + 免责声明”。如果混入“讲一个笑话”或“写首诗鼓励病人”，模型就会学到无关行为，真实部署时容易跑偏。

工程上，常用一个混合损失来表达边界控制：

$$
L_{\mathrm{total}}=\lambda_{\mathrm{real}}L_{\mathrm{real}}+(1-\lambda_{\mathrm{real}})L_{\mathrm{synthetic}}
$$

这里的 $\lambda_{\mathrm{real}}$ 可以理解为“真实数据在训练目标中的拉力”。  
白话解释：这个系数越大，模型越听真实样本的话；这个系数越小，模型越更多受合成样本影响。

这不是纯数学装饰，而是工程约束。因为真实样本通常更接近线上分布，合成样本通常更容易做大规模覆盖。两者混合，才有可能同时解决“数据太少”和“分布不真”这两个问题。

---

## 核心机制与推导

为什么合成数据能在 SFT 里起作用？核心原因有两个。

第一，它能提供**模式密度**。  
模式密度可以理解为“同一种任务规则被反复、清晰地展示给模型的程度”。比如摘要任务里，输入很长，输出必须短、准确、不能扩写。合成数据可以快速生成大量同结构样本，让模型先稳定掌握这个映射关系。

第二，它能提供**覆盖率**。  
覆盖率就是“任务空间里有多少情况被训练集碰到过”。真实数据少的时候，模型容易记住少数写法；合成数据可以补足不同问法、不同长度、不同格式噪声，从而降低冷启动时的过拟合风险。

一个最小数值例子如下。假设你只有 100 条真实数据，但可以生成 900 条合成数据。你不希望模型完全学成“教师模型的说话方式”，于是设定 $\lambda_{\mathrm{real}}=0.2$，并让每个 batch 大约 20% 来自真实数据，80% 来自合成数据。这样做的含义是：

- 合成数据负责把模型拉进任务空间。
- 真实数据负责把模型钉在真实分布附近。

训练重点会随 $\lambda_{\mathrm{real}}$ 变化：

| $\lambda_{\mathrm{real}}$ | 训练重点 | 优点 | 风险 |
| --- | --- | --- | --- |
| 0.1 | 更依赖合成覆盖 | 冷启动快，成本低 | 容易学到教师口癖，真实场景偏移大 |
| 0.2-0.3 | 覆盖与校准平衡 | 常见工程甜点区 | 仍需做好合成数据过滤 |
| 0.5 | 更接近真实分布 | 输出更稳，风格更真 | 真实数据太少时，覆盖不足 |
| 0.8 以上 | 基本以人工数据为主 | 风险低，适合高安全场景 | 成本高，长尾覆盖弱 |

如果把模型效果做一个粗略期望，可以写成：

$$
\mathrm{Accuracy}_{\mathrm{hybrid}} \approx \lambda_{\mathrm{real}}A_{\mathrm{real}}+(1-\lambda_{\mathrm{real}})A_{\mathrm{synthetic}}
$$

这个式子不是严格定律，而是工程直觉：混合训练后的表现，大致受两种数据质量和配比共同决定。  
如果真实数据质量明显高于合成数据，那么随着 $\lambda_{\mathrm{real}}$ 增大，线上质量通常会更稳；但如果真实数据太少，模型可能连任务模板都没学扎实，反而不如先用合成数据预热。

真实工程例子比玩具例子更能说明问题。SyntheT2C 这类工作面向 Text2Cypher，Cypher 是图数据库查询语言，意思是“用自然语言生成图查询语句”。这类任务人工标注成本很高，因为要同时懂业务知识图谱和查询语法。论文做法是先合成 Query-Cypher 对，再拿这些数据做 SFT，结果能提升模型在医疗知识图谱查询上的正确率。它说明一件事：**当人工标注极贵、任务格式明确、可验证性强时，合成数据的价值会被放大。**

但收益会递减。原因也很简单：

1. 前 1000 条样本可能在教模型“什么叫这个任务”。
2. 后 1000 条样本可能只是“换个说法重复同一件事”。
3. 再往后，如果还是单一教师、单一模板、单一温度生成，新增样本甚至会让分布更窄。

所以规模效应的正确理解不是“越多越好”，而是“**先补覆盖，再补质量，最后控制重复**”。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖 PyTorch，但逻辑和真实训练一致：准备真实集与合成集，按比例采样，分别计算损失，再做加权。

```python
import random
from dataclasses import dataclass

@dataclass
class Sample:
    prompt: str
    target: str
    source: str  # "real" or "synthetic"

real_dataset = [
    Sample("用户：请总结这段文本", "输出：三点摘要", "real"),
    Sample("用户：把这句话改写成正式语气", "输出：正式改写结果", "real"),
]

synthetic_dataset = [
    Sample(f"合成指令{i}", f"合成回答{i}", "synthetic")
    for i in range(8)
]

def batch_sampler(real_data, synthetic_data, batch_size=5, lambda_real=0.2, seed=42):
    random.seed(seed)
    real_count = max(1, round(batch_size * lambda_real))
    syn_count = batch_size - real_count
    batch = random.sample(real_data, min(real_count, len(real_data)))
    batch += random.sample(synthetic_data, min(syn_count, len(synthetic_data)))
    random.shuffle(batch)
    return batch

def fake_loss(samples):
    """
    用一个玩具损失代替真实交叉熵。
    真实项目里这里会是 token-level cross entropy。
    """
    if not samples:
        return 0.0
    avg_prompt_len = sum(len(x.prompt) for x in samples) / len(samples)
    avg_target_len = sum(len(x.target) for x in samples) / len(samples)
    return round((avg_prompt_len + avg_target_len) / 100.0, 4)

def split_by_source(batch):
    real = [x for x in batch if x.source == "real"]
    synthetic = [x for x in batch if x.source == "synthetic"]
    return real, synthetic

lambda_real = 0.2
batch = batch_sampler(real_dataset, synthetic_dataset, batch_size=5, lambda_real=lambda_real)

real_batch, synthetic_batch = split_by_source(batch)
real_loss = fake_loss(real_batch)
synthetic_loss = fake_loss(synthetic_batch)

total_loss = lambda_real * real_loss + (1 - lambda_real) * synthetic_loss

assert len(batch) == 5
assert len(real_batch) >= 1
assert abs(total_loss - (lambda_real * real_loss + (1 - lambda_real) * synthetic_loss)) < 1e-9

print("batch:", batch)
print("real_loss:", real_loss)
print("synthetic_loss:", synthetic_loss)
print("total_loss:", round(total_loss, 4))
```

如果换成 PyTorch/Transformers，核心结构通常是下面这样：

```python
# 伪代码，展示工程结构
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row = self.rows[idx]
        # 把 instruction + answer 编码成模型训练需要的 token
        return self.tokenizer(row["prompt"], text_target=row["answer"], truncation=True)

    def __len__(self):
        return len(self.rows)

class RealDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row = self.rows[idx]
        return self.tokenizer(row["prompt"], text_target=row["answer"], truncation=True)

    def __len__(self):
        return len(self.rows)

def mixed_step(model, real_batch, synthetic_batch, lambda_real):
    # 分别前向，避免不同来源的数据统计完全混在一起
    real_loss = model(**real_batch).loss
    synthetic_loss = model(**synthetic_batch).loss
    loss = lambda_real * real_loss + (1 - lambda_real) * synthetic_loss
    return loss
```

这里有三个工程点值得新手注意。

第一，`SyntheticDataset` 和 `RealDataset` 最好分开管理。这样你可以分别做去重、质量评分、错误分析。  
第二，不要只在总数据级别控制比例，最好在 batch 级别也控制比例。否则一个 epoch 里前面全是合成数据、后面全是真实数据，训练会抖动。  
第三，`lambda_real` 可以做调度。前期小一些，让模型先吸收覆盖；后期逐步增大，让真实数据接管校准。例如从 0.2 线性升到 0.5，就是很常见的思路。

---

## 工程权衡与常见坑

合成数据在工程里最大的问题不是“能不能生成”，而是“生成出来的东西值不值得学”。

最常见的坑如下：

| 常见坑 | 现象 | 本质问题 | 规避方式 |
| --- | --- | --- | --- |
| 分布塌陷 | 回答越来越像同一个模板 | 单一教师、单一提示词、单一温度 | 多模型生成，多提示模板，多温度采样 |
| 重复错误 | 模型反复犯同一类错误 | 教师模型系统性偏差被复制 | 加规则校验、判别器、人工抽检 |
| 监督信号偏差 | 离线分数高，线上不好用 | 合成分布和真实分布不一致 | 保留 20%-30% 真实样本做校准 |
| 虚假多样性 | 看起来问题不同，本质答案套路一致 | 表层改写太多，语义覆盖太少 | 按任务属性聚类，而不是只做文本去重 |
| 评测污染 | 训练集与测试集模式重叠 | 生成时偷看了目标分布 | 严格拆分种子集、评测集、生成模板 |

一个典型失败案例是：某团队只用单一大模型生成客服问答，提示词也几乎不变。结果训练出来的学生模型句式极度统一，遇到真实用户的错别字、打断、反问、混合意图时表现很差。后来他们改成“多模型 + 多温度 + 判别器筛选”，比如让不同模型分别负责保守回答、结构化回答、自然对话回答，再做一致性和去重过滤，输出多样性才明显提升。

对新手来说，最实用的经验不是追求复杂算法，而是守住两条底线：

1. **永远不要把未审校的合成数据当成天然高质量数据。**
2. **永远不要让真实数据在最后训练阶段完全消失。**

很多实践建议会提到保留 20%-30% 的真实样本，这不是精确到小数点的定律，但它反映了一个稳妥原则：真实数据要在最终模型里持续提供“纠偏力”。

---

## 替代方案与适用边界

不是所有 SFT 都适合把合成数据当主角。是否采用、采用多少，主要看三件事：真实数据成本、任务可验证性、安全风险。

三种方案可以直接对比：

| 方案 | 成本 | 数据质量上限 | 启动速度 | 适用任务 |
| --- | --- | --- | --- | --- |
| 纯人工 | 高 | 最高 | 慢 | 医疗、法律、金融合规等高风险任务 |
| 纯合成 | 低 | 中等，强依赖过滤 | 快 | 格式明确、可验证、冷启动阶段 |
| 混合 | 中等 | 通常最好 | 较快 | 大多数实际 SFT 项目 |

纯人工的优点是分布真、风险低，但成本高，长尾覆盖也未必好，因为人工很难快速补齐大量变体。  
纯合成的优点是便宜、快、可扩展，但如果任务高度依赖真实语境或隐含规则，它很容易偏。  
混合方案通常是最稳的，因为它让两种数据各做自己擅长的事。

适用边界可以再细化一下：

- 如果任务是开放域指令遵循、客服模板、结构化抽取、文本重写，合成数据通常很有用。
- 如果任务是高安全场景，比如法律建议、医疗决策、金融合规审查，人工标注应是主体，合成只能做补充覆盖。
- 如果任务可自动验证，比如 SQL、Cypher、代码、正则表达式、分类标签，合成数据价值更高，因为可以程序化筛掉错误样本。
- 如果任务强依赖真实用户习惯，比如真实闲聊、多轮客服投诉、复杂口语表达，只做合成通常不够。

真实工程例子可以放在这条边界上看。医疗知识图谱里的 Text2Cypher 任务，输入输出结构清晰、查询语法可校验，所以“全合成生成 + 专家复核”是可行路径。相反，金融合规问答经常涉及最新规则、灰色边界、措辞风险和责任归属，这类任务更适合“真实数据为主，合成数据补低频边界”。

Self-Instruct 证明了一个方向：在公开基础任务上，经过过滤的合成指令数据，确实可以把基础模型的 instruction following 能力显著拉起来。但这不等于垂直业务也能照搬。越接近真实业务、越接近安全红线，人工校准的重要性越高。

---

## 参考资料

| 参考 | 年份 | 核心贡献 | 适用建议 |
| --- | --- | --- | --- |
| [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://aclanthology.org/2023.acl-long.754/) | 2023 | 证明用模型自生成并过滤的指令数据，可以显著提升 instruction following；论文报告对 vanilla GPT-3 在 Super-NaturalInstructions 上有 33% 绝对提升，并接近 InstructGPT-001 | 适合做通用指令数据冷启动，不适合直接替代垂直场景人工校准 |
| [Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls](https://aclanthology.org/2025.emnlp-main.544/) | 2025 | 系统分析合成数据比例、规模效应和收益边界；指出合适混合比依赖模型规模与数据预算，部分设置下约 30% 合成比例是有效区间 | 适合理解“规模不是越大越好”和“混合比例需要调”这两个工程事实 |
| [SyntheT2C: Generating Synthetic Data for Fine-Tuning Large Language Models on the Text2Cypher Task](https://aclanthology.org/2025.coling-main.46/) | 2025 | 在医疗知识图谱的 Text2Cypher 任务上，用合成 Query-Cypher 数据集提升 SFT 效果 | 适合结构清晰、可验证、人工标注昂贵的垂类任务 |
| [How to Generate Synthetic Training Data for LLM Fine-Tuning (2026 Guide)](https://blog.premai.io/how-to-generate-synthetic-training-data-for-llm-fine-tuning-2026-guide/) | 2026 | 从工程视角总结生成策略、过滤流程、模型塌陷风险和“保留真实样本底座”的经验 | 适合作为实操清单，尤其是多模型生成、过滤、混合训练流程设计 |
| [BARE: Combining Base and Instruction-Tuned Language Models for Better Synthetic Data Generation](https://huggingface.co/papers?q=synthetic+instruction+fine-tuning) | 2025 | 通过结合 base model 的多样性与 instruct model 的质量，强调“高质量合成数据不只看强教师，还看多样性”；摘要页提到 1000 条样本即可在 LiveCodeBench 上取得很强结果 | 适合理解“小规模高质量合成数据也能起启动作用”，但落地时仍需看原论文与复现 |
