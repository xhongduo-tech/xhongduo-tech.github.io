## 核心结论

KG-BERT 的核心思想很直接：把知识图谱中的三元组 $(h,r,t)$ 改写成一段文本，再交给 BERT 判断“这句话像不像真事实”。这里的“三元组”就是“头实体、关系、尾实体”三部分组成的一条知识，例如 `(Steve Jobs, founder_of, Apple)`。

更准确地说，KG-BERT 不是传统知识图谱嵌入模型那种“给每个实体学一个向量，再做几何运算”，也不是“双塔检索模型”。它更像一个 **cross-encoder**，也就是“把头、关系、尾拼到同一条输入序列里，让模型一起看完再做判断”。常见输入形式是：

`[CLS] 头实体描述 [SEP] 关系描述 [SEP] 尾实体描述 [SEP]`

然后取 `[CLS]` 位置的输出，接一个线性层和 `sigmoid`，得到这个三元组为真的概率。

它的优点是能直接利用自然语言语义，特别适合“结构不够密、但实体描述文本比较丰富”的知识图谱。它的缺点也很明确：推理成本高，因为候选尾实体很多时，要把每个候选都和头实体、关系重新拼接后跑一遍 BERT；长尾实体也容易出问题，因为“长尾实体”就是训练中出现很少、描述很短或很脏的实体，它们的文本信号不稳定。

至于题目里的 PeterParser，截至 **2026-04-14**，公开可验证资料并不能证明它是一个已发表、可复现的“生成式知识图谱补全模型”。目前能查到的公开信息更像企业文档处理流水线：OCR、版面解析、agentic chunking、metadata 优化。更稳妥的说法是：**它目前更像 KG 数据构建前的文档预处理工具，而不是已被公开论文确认的 KGC 模型本体。**

---

## 问题定义与边界

知识图谱补全，英文常写作 Knowledge Graph Completion，目标是根据已有事实推断缺失事实。形式化地说，就是学习一个打分函数：

$$
s(h,r,t)\in[0,1]
$$

当 $s(h,r,t)$ 越接近 1，表示模型越相信这个三元组为真；越接近 0，表示越像伪造样本。

如果把它建模成二分类，训练目标通常是交叉熵：

$$
L = -\left[y\log s + (1-y)\log(1-s)\right]
$$

其中 $y=1$ 表示真三元组，$y=0$ 表示负样本。

这里要先划清边界。KG-BERT 解决的是 **closed-world 候选判断** 问题：实体和关系通常默认已经在图里，模型做的是“这个候选链接成不成立”。它不是从零发明新实体，也不是直接从一大段原始文档里自动构图。

玩具例子可以先看这个：

- 正样本：`(Steve Jobs, founder_of, Apple)`
- 负样本：`(Steve Jobs, founded_in, Apple)` 或 `(Steve Jobs, founder_of, iPhone)`

KG-BERT 会把它们分别写成文本序列，再学习“哪种组合更像真实世界的表达”。

真实工程例子更接近这样：一个企业知识库里已经有“产品、团队、系统、负责人、依赖关系”这些实体和关系，但图谱不完整。你想补全 `(支付系统, owner_is, 某团队)` 或 `(报表服务, depends_on, Kafka)`。如果每个实体都有 wiki、说明文档、服务描述，那么 KG-BERT 这种“吃文本”的方法就有发挥空间。

PeterParser 在这个链条里的位置，如果按目前公开信息理解，更可能是前一步：把 HWP、DOCX、PPTX、Markdown 等文件解析成适合 LLM 或下游抽取模型消费的 chunk。它解决的是“文档怎么切、怎么清洗、怎么保留版面与元数据”，而不是直接学习 $s(h,r,t)$。

---

## 核心机制与推导

KG-BERT 的机制可以压缩成三步：

| 步骤 | 做什么 | 白话解释 |
| --- | --- | --- |
| 1 | 三元组转文本 | 把图上的边改写成一句模型能读的输入 |
| 2 | BERT 编码 | 让 Transformer 同时看头、关系、尾三部分上下文 |
| 3 | `[CLS]` 判真 | 用整句摘要向量判断“像不像真事实” |

输入可以写成：

`[CLS] head_text [SEP] relation_text [SEP] tail_text [SEP]`

这里的 token 表示“子词切分后的最小输入单元”；segment embedding 表示“这段 token 属于哪一段文本”；position embedding 表示“它在序列中的位置”。KG-BERT 沿用原始 BERT 的输入表示，三者相加后送入多层 Transformer。

流程可以记成：

`token 合并 -> Transformer 编码 -> CLS 向量 -> 线性层 -> sigmoid -> score`

如果记 `[CLS]` 的最终隐藏状态为 $\mathbf{h}_{cls}$，那么打分可以写成：

$$
s(h,r,t)=\sigma(\mathbf{w}^\top \mathbf{h}_{cls}+b)
$$

其中 $\sigma$ 是 sigmoid。

为什么这种方法有效？因为它把“结构匹配”转成了“语义一致性判断”。比如：

- `Steve Jobs [SEP] founder_of [SEP] Apple`
- `Steve Jobs [SEP] capital_of [SEP] Apple`

两条输入只有关系不同，但第二条在语义上明显不自然。BERT 预训练时已经学过大量语言共现模式，所以微调时更容易把这种“不自然”映射成低分。

但它也因此带来两个工程后果。

第一，**它很贵**。链接预测时，如果给定 $(h,r,?)$，你往往要枚举很多候选尾实体，每个候选都要重新跑一次编码器。结构嵌入模型常见做法是一次矩阵运算批量比较，而 KG-BERT 更像“把每个候选都当成一个独立样本做分类”。

第二，**它高度依赖文本质量**。如果尾实体只有一个短名字，几乎没有描述，例如“QX-17 模块”，模型就很难从文本里得到稳定语义。后续工作如 CAKGC 的思路，就是给 KG-BERT 这类文本模型补额外信号，例如实体类别信息、残差注意力、对比学习损失，用来缓解长尾实体表示不稳的问题。

顺带说结果边界。公开横向比较里，KG-BERT 在 FB15k-237 这类链接预测任务上并不属于今天的最强方案。后续论文复现实验里，KG-BERT 的 MRR 常见在 **0.24 左右到 0.27 左右**，而不是“只要上 BERT 就大幅领先”。这恰好说明：文本语义有帮助，但算力和候选枚举成本也是真问题。

---

## 代码实现

如果你想从 0 理解 KG-BERT，不必先跑完整训练。先实现“序列构造 + 标签 + 交叉熵”这个最小闭环就够了。

下面是一个可运行的玩具版本，用纯 Python 模拟 KG-BERT 的输入组织和二分类损失：

```python
import math

def build_sequence(head_text, relation_text, tail_text):
    return f"[CLS] {head_text} [SEP] {relation_text} [SEP] {tail_text} [SEP]"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def bce_loss(score, label):
    eps = 1e-12
    score = min(max(score, eps), 1 - eps)
    return -(label * math.log(score) + (1 - label) * math.log(1 - score))

pos = build_sequence(
    "Steve Jobs was the co-founder of Apple.",
    "founder_of",
    "Apple is an American technology company."
)
neg = build_sequence(
    "Steve Jobs was the co-founder of Apple.",
    "capital_of",
    "Apple is an American technology company."
)

# 假设模型对正样本打高分、对负样本打低分
pos_score = sigmoid(3.0)
neg_score = sigmoid(-3.0)

assert "[CLS]" in pos and "[SEP]" in pos
assert pos_score > 0.9
assert neg_score < 0.1
assert bce_loss(pos_score, 1) < bce_loss(neg_score, 1)
assert bce_loss(neg_score, 0) < bce_loss(pos_score, 0)

print("positive:", pos)
print("negative:", neg)
print("pos_loss:", round(bce_loss(pos_score, 1), 4))
print("neg_loss:", round(bce_loss(neg_score, 0), 4))
```

真正复现时，可以直接参考官方仓库。最实用的入门路径是：

| 步骤 | 操作 | 目的 |
| --- | --- | --- |
| 1 | 克隆 `yao8839836/kg-bert` | 拿到官方脚本 |
| 2 | 安装依赖 | 跑 BERT 微调 |
| 3 | 先跑 `run_bert_triple_classifier.py` | 先理解判真任务 |
| 4 | 再跑 `run_bert_link_prediction.py` | 进入真正 KGC 场景 |
| 5 | 检查 `entity2text.txt` / `relation2text.txt` | 确认文本描述质量 |

典型命令形态如下：

```bash
python run_bert_triple_classifier.py \
  --task_name kg \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir ./data/WN11 \
  --bert_model bert-base-uncased \
  --max_seq_length 20 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --output_dir ./output_WN11/ \
  --eval_batch_size 512
```

如果你做真实工程迁移，要改的重点通常不是模型结构，而是数据接口：

- 把实体 ID 映射到可读描述
- 把关系 ID 映射到稳定、明确的关系短语
- 生成负样本
- 统一训练集与预测时的文本模板

---

## 工程权衡与常见坑

最常见的误判，是把 KG-BERT 当成“有文本就一定比图嵌入强”。这不成立。它只是把一部分图问题转化成语言理解问题，前提是文本真的有信息量。

几个关键权衡如下：

| 维度 | KG-BERT | 工程含义 |
| --- | --- | --- |
| 文本利用 | 强 | 有实体描述时收益明显 |
| 结构利用 | 弱于专门图模型 | 对复杂拓扑模式未必占优 |
| 训练成本 | 高 | 微调 BERT 比传统 KGE 贵很多 |
| 推理成本 | 很高 | 候选枚举时要反复编码 |
| 可解释性 | 中等 | 能回看输入文本，但内部注意力不等于因果解释 |
| 长尾鲁棒性 | 一般 | 描述稀疏时分数抖动明显 |

真实工程里最容易踩的坑有四个。

第一，**关系文本写得太随意**。如果关系同时出现 `works_for`、`work for`、`employee_of` 三种写法，模型会把同一语义拆成多个模式。

第二，**实体描述过长或过脏**。BERT 有最大长度限制，超长后要截断。截断一旦把关键限定词切掉，例如“不是创始人而是 CEO”，结果会直接变形。

第三，**负采样太简单**。如果你只做随机替换尾实体，负样本往往过于容易，模型学到的是表面差异，不是真正关系边界。更好的做法是加入“难负样本”，也就是类型接近但实际上不成立的候选。

第四，**长尾实体文本太短**。例如一个内部服务只有名字，没有描述。此时模型学不到足够语义，分数会很不稳定。CAKGC 这类后续方法的启发是：给实体增加类别、层次、上下文属性，再用更强的注意力或对比学习去拉开表示距离。

一个很实际的落地建议是：**先做 triple classification，再做 link prediction**。前者更像普通二分类，链路短，便于快速验证“文本模板是否合理”；后者才是候选枚举的大头，成本高得多。

---

## 替代方案与适用边界

如果你的数据特点是“文本多、图稀疏、实体描述质量高”，KG-BERT 仍然值得做基线。但如果你面临下面这些情况，就要考虑替代方案：

- 候选实体很多，线上延迟敏感
- 图结构规律比文本更重要
- 实体描述很短，甚至没有自然语言说明
- 算力预算有限，不想全量微调 BERT

可以把替代路线分成三类：

| 方案 | 适合什么情况 | 代价 |
| --- | --- | --- |
| KG-BERT 判真 | 文本丰富、先做高质量基线 | 推理贵 |
| CAKGC 类增强 | 长尾严重，希望在文本路线上继续挖潜 | 结构更复杂 |
| 文档解析 + 生成式推理 | 原始资料主要是文档，还没形成高质量 KG | 前处理与提示设计成本高 |

A 路线是继续走 KG-BERT 思路，但加入 CAKGC 这类增强。它的重点不是推翻序列判别，而是往输入表示里补“类别信息”和更强的上下文建模，对长尾实体更友好。

B 路线则更接近你题目里提到的“PeterParser 视角”：先把文档做 OCR、版面解析、chunk 切分、metadata 清洗，再把输出交给一个明确的抽取或生成模块。这个模块可以是 LLM，根据文档片段去生成候选尾实体，再交由规则或判别器过滤。这里的输入输出接口会变成：

- 输入：文档 chunk + 实体上下文 + 待补全关系
- 输出：候选实体文本或候选三元组

但要强调边界：**这已经不是 KG-BERT 这类判别式 KGC 了，而是“文档预处理 + 候选生成 + 事实校验”的组合系统。** 如果没有公开论文、代码和评测，你不能把 PeterParser 直接写成“生成式 KGC 模型”而不加限定。

---

## 参考资料

1. KG-BERT 原始论文，arXiv，2019：<https://arxiv.org/abs/1909.03193>  
2. KG-BERT 官方代码仓库：<https://github.com/yao8839836/kg-bert>  
3. CAKGC，`Knowledge Graph Completion Using a Pre-Trained Language Model Based on Categorical Information and Multi-Layer Residual Attention`，MDPI Applied Sciences 2024：<https://www.mdpi.com/2076-3417/14/11/4453>  
4. PyPI 上的 `peterparser`，公开描述为区块链/DeFi 配置解析器，而非 KGC 模型：<https://pypi.org/project/peterparser/>  
5. BrainCrew 公司资料，公开描述 `PeterParser` 包含 OCR、Layout parsing、agentic chunking、metadata 优化等文档预处理能力：<https://www.saramin.co.kr/zf_user/company-info/view/csn/VGdvZDRMVW54U1hmQ1BWNmtzTzN1Zz09/company_nm/%EB%B8%8C%EB%A0%88%EC%9D%B8%ED%81%AC%EB%A3%A8%28%EC%A3%BC%29?nomo=1>  
6. Multi-Task Learning for Knowledge Graph Completion with Pre-trained Language Models，文中可见后续复现实验对 KG-BERT 的横向比较：<https://www.researchgate.net/publication/348345008_Multi-Task_Learning_for_Knowledge_Graph_Completion_with_Pre-trained_Language_Models>
