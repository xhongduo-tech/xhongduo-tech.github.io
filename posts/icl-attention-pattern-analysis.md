## 核心结论

ICL，in-context learning，中文常译为“上下文学习”，指模型**不更新参数，只靠提示词里的示例临时完成任务**。它不是单一步骤，而更像一条分层流水线：

1. 浅层先做**词面和格式匹配**。白话说，就是先看清楚“示例长什么样”“输入和标签分别放哪”。
2. 中层再做**语义对齐和输入-标签映射**。白话说，就是把“这个新问题”对上“前面哪类示例”和“应该对应哪个标签”。
3. 深层最后做**任务特定输出**。白话说，就是把已经对齐好的内部表示，真正翻译成要生成的答案 token。

这个分工不是拍脑袋的经验，而是多个可解释性实验的共同结果。Bansal 等人在 OPT-66B 上发现，**重要注意力头主要集中在中间层，重要 FFN 主要集中在后层**；Sun 等人在 Mistral-7B 上进一步发现，**真正直接影响 ICL 预测的 key heads 主要出现在中层**；Sia 等人则从“任务识别点”角度说明，模型通常会在某个中层附近完成“这是什么任务”的编码，之后继续看完整上下文的必要性明显下降。

一个适合初学者的玩具理解是“三段式流水线”：

- 浅层：先把模板记下来
- 中层：把新问题和模板对上号
- 深层：把对上的结果翻译成答案

| 层级 | 主要职责 | 典型现象 | 证据来源 |
|---|---|---|---|
| 浅层 | 词汇、位置、格式聚合 | 对提示模板敏感，但单独删掉通常不致命 | Sun et al. 2025；Sia et al. 2024 |
| 中层 | 语义对齐、输入-标签映射、关键头检索 | key heads 集中；错误交换会导致性能骤降 | Bansal et al. 2023；Sun et al. 2025 |
| 深层 | 将映射结果解码为答案；任务特定推理 | FFN 更敏感，关系到最终输出质量 | Bansal et al. 2023 |

---

## 问题定义与边界

先把问题说清楚。

> ICL 研究的问题不是“模型有没有学到世界知识”，而是“模型在一次前向传播里，怎样利用提示中的示例完成当前任务”。

这里有三个边界。

第一，ICL讨论的是**不做梯度更新**的任务适配。也就是说，模型参数不变，变化的是上下文里的示例。

第二，本文讨论的是**Transformer 内部不同层的分工**，不是“哪种提示词写法最好”。重点是注意力层和 FFN 层分别在干什么。

第三，本文说的“浅层 1-8、中层 9-20、深层 21+”不是所有模型的硬边界，而是一个**帮助理解的近似分段**。不同模型层数不同，例如 32 层模型和 64 层模型，真正的分界会平移。但“浅层偏格式、中层偏对齐、深层偏输出”这个趋势，在多篇论文里是一致的。

可以把问题抽象成下面这个文本图：

```text
提示词 = [示例1输入 -> 示例1标签] [示例2输入 -> 示例2标签] ... [新问题输入 -> ?]

浅层：识别每段的边界、格式、标签位置
中层：找到“新问题”和哪些示例最像，并提取输入-标签关系
深层：把提取出的关系变成最终答案 token
```

玩具例子可以用情感分类理解：

```text
评论: "This movie is great." 标签: positive
评论: "This movie is terrible." 标签: negative
评论: "The acting is wonderful." 标签: ?
```

模型如果只停在浅层，它只能看到三个句子格式很像；真正让它输出 `positive` 的，是中层把 “wonderful” 和前面的正向标签对齐，再由深层把这个对齐结果解码成标签词。

真实工程里，翻译任务更明显：

```text
English: I love apples. French: J'aime les pommes.
English: He is a teacher. French: Il est professeur.
English: We are ready. French: ?
```

这里浅层先识别出双语对格式，中层把 `English -> French` 这个映射关系编码出来，深层再继续生成法语序列。Sia 等人把这个中间时刻叫作**任务识别点**，白话说，就是模型“已经知道现在要做翻译了”的那一层。

---

## 核心机制与推导

Transformer 每层都不是“推翻上一层重来”，而是在残差连接上逐步叠加。Bansal 等人给出的层更新形式是：

$$
t^{(\ell+1)} = z^{(\ell)} + \mathrm{MHA}_\ell(\mathrm{LN}(z^{(\ell)}))
$$

$$
z^{(\ell+1)} = t^{(\ell+1)} + \mathrm{FFN}_\ell(t^{(\ell+1)})
$$

含义很直接：

- $z^{(\ell)}$ 是当前层输入表示
- MHA，多头注意力，负责“从上下文里看谁”
- FFN，前馈网络，负责“把已经聚合的信息做非线性变换”
- 残差连接保证旧信息不会被轻易抹掉

为什么这套结构天然会形成“浅层到深层”的分工？因为每层都在前一层基础上继续累积。于是信息流通常呈现下面的路径：

```text
词面模式/位置线索
  -> 示例结构抽象
  -> 输入-标签关系提取
  -> 当前问题与既有关系匹配
  -> 将匹配结果转成最终输出
```

Sun 等人分析 ICL 时，把关键机制概括为对**input-label mapping**的利用。这个词的白话解释是：模型在示例中学到“这种输入通常配这个标签”的内部关系。为了量化注意力头是否真的在做这种事，Bansal 等人采用了 prefix matching 和 copying 的分析框架。其核心注意力形式可以写成：

$$
s^h(M)=\mathrm{softmax}\left(\frac{MW_q^h (W_k^h)^T M^T}{\sqrt{d_h}}\right)
$$

这里的 $s^h(M)$ 是第 $h$ 个注意力头的注意力分数矩阵。白话说，它表示“当前 token 到底在看上下文中的哪些 token”。

如果一个头在新问题位置上，稳定指向前面示例中的标签 token，或者指向和当前前缀最匹配的示例片段，那么它就可能是 ICL 的关键头。

这时可以用一个非常小的玩具例子看“浅层匹配”和“中层映射”的区别：

| 提示内容 | 浅层能看到什么 | 中层能做什么 |
|---|---|---|
| `x=2 -> even` | `->` 是标签分隔符，`even` 出现在标签位 | 识别“偶数对应 even” |
| `x=3 -> odd` | 继续确认格式一致 | 识别“奇数对应 odd” |
| `x=8 -> ?` | 看到也是同样模板 | 把 `8` 映射到 `even` |

只靠浅层，模型知道“这里该输出一个标签”；只有进入中层，它才更可能知道“这个标签应该是 even 而不是 odd”。

再往后，深层 FFN 的作用开始变强。Bansal 等人发现，相比大量注意力头可以被移除，FFN 更难删，尤其后层 FFN 更敏感。这说明**注意力负责把相关信息找出来，深层 FFN 更像把这些信息压缩、组合，并投射到最终输出空间**。对生成式任务尤其如此，因为最后不是二选一标签，而是一串连续 token。

可以把整条机制画成简易流程图：

```text
浅层（看模板）
  -> 中层关键头（找对应示例与标签）
  -> 深层 FFN（把内部关系变成可输出答案）
```

这也是为什么很多论文里会看到一个共同结论：ICL 不是“模型临时学会了一个新算法”，更接近于**模型在上下文里定位已有能力，再把它调用出来**。

---

## 代码实现

工程上最常见的验证方式是做 ablation，中文常叫“消融实验”，也就是**有控制地关掉某些层、某些头，看性能怎么变**。

下面给一个可运行的 Python 玩具实现。它不依赖真实大模型，而是模拟“浅层格式分数 + 中层映射分数 + 深层输出分数”的三段式，并演示中层被破坏时性能更容易崩。

```python
from dataclasses import dataclass

@dataclass
class ToyICLState:
    shallow_score: float
    middle_score: float
    deep_score: float

def classify_with_icl(state: ToyICLState, threshold: float = 1.5) -> str:
    # 浅层负责确认模板，单独不足以给出标签
    # 中层负责把新输入和示例标签关系对齐
    # 深层负责把对齐后的结果稳定输出
    total = state.shallow_score * 0.2 + state.middle_score * 0.5 + state.deep_score * 0.3
    return "positive" if total >= threshold else "negative"

# 正常情况：三段都工作
normal = ToyICLState(shallow_score=1.0, middle_score=2.0, deep_score=2.0)
assert classify_with_icl(normal) == "positive"

# 只保留浅层，通常不够
shallow_only = ToyICLState(shallow_score=1.0, middle_score=0.0, deep_score=0.0)
assert classify_with_icl(shallow_only) == "negative"

# 中层被破坏，深层也很难“凭空”补回来
middle_broken = ToyICLState(shallow_score=1.0, middle_score=-1.0, deep_score=2.0)
assert classify_with_icl(middle_broken) == "negative"

# 中层正常但深层较弱，仍有机会完成简单分类
deep_weaker = ToyICLState(shallow_score=1.0, middle_score=2.0, deep_score=0.5)
assert classify_with_icl(deep_weaker) == "positive"
```

真实模型里的实现当然更复杂，但干预点很明确：要么改某层是否允许看上下文，要么改某些头的注意力分数，要么只在中深层挂 LoRA。

下面是接近工程代码的伪代码：

```python
def forward(hidden, layers, cutoff_layer=None, blocked_heads=None):
    for layer_id, layer in enumerate(layers):
        attn_scores = layer.compute_attn_scores(hidden)

        # 1. 在任务识别点之后，禁止继续看历史示例
        if cutoff_layer is not None and layer_id >= cutoff_layer:
            attn_scores = mask_context_attention(attn_scores)

        # 2. 屏蔽指定关键头，做 head ablation
        if blocked_heads and layer_id in blocked_heads:
            for head_id in blocked_heads[layer_id]:
                attn_scores[head_id] = zero_head(attn_scores[head_id])

        hidden = layer.apply_attention(hidden, attn_scores)
        hidden = layer.apply_ffn(hidden)

    return hidden
```

如果目的是低成本增强 ICL，而不是纯研究，也可以只在中深层加 LoRA：

```python
for layer_id, layer in enumerate(model.layers):
    if 12 <= layer_id <= 24:
        attach_lora(layer.attention)
        attach_lora(layer.ffn)
```

这里的经验不是“浅层永远不用动”，而是**优先改最可能影响任务识别和映射调用的中深层**。Sun 等人的 PC patching 进一步说明，只要定位到少量 key heads，就能对 ICL 行为做高杠杆干预。

真实工程例子可以这样理解：如果你在做一个翻译 API，希望少花显存和时延，最合理的办法不是盲目量化所有层，而是先测“任务识别点”大概在哪，再决定从那一层之后减少上下文依赖，或者把 LoRA 预算优先放在中深层。

---

## 工程权衡与常见坑

第一类坑是把 ICL 误解成“浅层模板匹配”。这会导致两个错误决策：一是觉得 prompt 只要格式统一就够了，二是觉得前几层最重要。现有结果恰好相反。Sun 等人在 Mistral-7B 上发现，**交换少量中层关键头的标签注意力后，准确率可从 100% 掉到 11%**。这说明真正脆弱的地方不是“格式是否看见”，而是“映射是否对齐”。

第二类坑是忽视深层 FFN。Bansal 等人表明，注意力头可以大规模裁剪，但 FFN 尤其后层更敏感。原因很简单：找到相关信息和把相关信息转成正确输出，是两件事。前者偏注意力，后者常由后层 FFN 参与完成。对于分类任务，这会影响标签 logit；对于翻译、代码生成、推理任务，这会直接影响整段输出。

第三类坑是过早做上下文剪枝。Sia 等人的结论不是“中途都不用看上下文”，而是“**过了任务识别点以后**，继续看上下文的收益下降”。如果你在识别点之前就切断示例，模型连当前是什么任务都还没稳定编码，性能自然会塌。

| 常见坑 | 后果 | 规避策略 |
|---|---|---|
| 以为浅层匹配足够 | 中层一删性能断崖式下滑 | 先定位 key heads，再谈裁剪 |
| 把 FFN 当作“只是附属模块” | 深层输出质量下降，推理任务明显变差 | 深层 FFN 谨慎处理，先做分层评估 |
| 统一裁剪所有层的上下文注意力 | 在任务识别前丢失关键信息 | 先找 recognition layer，再做截断 |
| LoRA 平铺到全模型 | 训练贵，收益不集中 | 优先中层注意力和深层输出模块 |
| 用单一任务得出普适结论 | 跨任务迁移失败 | 分类、翻译、生成分开评估 |

一个很典型的真实工程判断是：

- 如果任务是短标签分类，中层 key heads 往往是最值得保的。
- 如果任务是长文本生成或推理，深层 FFN 不能随便砍。
- 如果任务是高吞吐翻译，任务识别点后的上下文裁剪值得重点测试。

Sia 等人在 Llama3.1-8B 的英译法实验中报告：当 5-shot 提示在第 14 层左右已接近“识别完任务”，之后停止依赖示例上下文，可带来约 **45% 计算节省**。这类收益成立的前提，不是“上下文不重要”，而是“重要的信息已经被编码进当前表示”。

---

## 替代方案与适用边界

如果目标不是做机制研究，而是做更省资源的系统，常见替代方案有三类。

| 策略 | 适用场景 | 优点 | 风险 |
|---|---|---|---|
| 只在中深层挂 LoRA | 分类、翻译、轻量任务适配 | 参数少，训练集中 | 任务差异大时可能不够 |
| 识别点后剪枝上下文注意力 | 多示例提示、长上下文推理成本高 | 明显省算力和显存 | 识别点估错会掉点 |
| 只保留关键头并做 head-aware 干预 | 需要可解释控制 ICL | 干预精确 | 需要额外分析成本 |

可以按下面顺序实施：

1. 先做基线评测，确认任务是分类、翻译还是生成。
2. 用层级 masking 或 head ablation 粗定位中层关键区域。
3. 如果是高吞吐推理，继续找任务识别点。
4. 如果是低成本适配，先把 LoRA 预算放到中层注意力和深层输出模块。
5. 最后做跨任务回归测试，避免某个单任务上的“伪优化”。

它的适用边界也要说清楚。

对于**语言对齐类任务**，例如翻译、情感分类、自然语言推断，这套“浅层匹配-中层对齐-深层输出”的框架通常很稳，因为示例中的输入-标签关系比较清楚。

对于**复杂推理或长程生成任务**，例如数学推导、代码生成、多步规划，中层当然仍然重要，但深层 FFN 和后层解码过程的重要性会更高。这时如果你只盯着中层 key heads，而忽略深层生成链条，优化很容易失真。

一句话总结适用边界：**越像“从示例里找标签映射”的任务，中层越关键；越像“要把内部关系展开成复杂输出”的任务，深层越不能省。**

---

## 参考资料

- Hritik Bansal, Karthik Gopalakrishnan, Saket Dingliwal, Sravan Bodapati, Katrin Kirchhoff, Dan Roth. *Rethinking the Role of Scale for In-Context Learning: An Interpretability-based Case Study at 66 Billion Scale*. ACL 2023. 贡献：在 OPT-66B 上量化 attention heads 与 FFN 的重要性，发现重要头集中在中间层，重要 FFN 更偏后层，并证明大量头可裁剪但 FFN 更敏感。来源：ACL Anthology。
- Suzy Ahyah Sia, David Mueller, Kevin Duh. *Where does In-context Learning Happen in Large Language Models?* NeurIPS 2024. 贡献：提出任务识别点，说明模型在某一中层后对示例上下文的依赖显著下降，并给出在翻译任务上约 45% 推理节省的案例。来源：NeurIPS Proceedings。
- Chengpeng Sun 等. *Interpret and Improve In-Context Learning via the Lens of Input-Label Mappings*. ACL 2025. 贡献：提出 PC patching，从输入-标签映射角度定位 key heads，显示关键头主要位于中层，并验证少量关键头交换即可显著破坏 ICL。来源：ACL Anthology。
