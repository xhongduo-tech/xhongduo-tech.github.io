## 核心结论

In-Context Learning，简称 ICL，白话说就是“模型不改参数，只靠你放进提示词里的示例，临时学会当前任务”。它之所以有效，可以用信息论来描述：上下文 $C$ 中的示例为输入 $X$ 到输出 $Y$ 的映射提供信息，从而降低模型对 $Y$ 的不确定性。

一个直接的写法是：

$$
I(Y;X\mid C)=H(Y\mid C)-H(Y\mid X,C)
$$

这里的条件互信息，白话说就是“在已经看到上下文示例之后，输入 $X$ 还能额外告诉我们多少关于输出 $Y$ 的信息”。如果示例足够一致，模型就能更准确地判断“当前到底在做什么任务”，于是 $H(Y\mid X,C)$ 会下降，预测变得更稳定。

进一步地，Xie 等人在 2022 年给出一种统一解释：Transformer 在做 ICL 时，可以看成在隐式推断一个潜变量 $\theta$。这个 $\theta$ 表示“当前任务概念”，例如“这是情感分类”“这是英法翻译”“这是奇偶判断”。模型的预测可以写成：

$$
p(y\mid x,C)=\int p(y\mid x,\theta)\,p(\theta\mid C)\,d\theta
$$

这句话的意思是：模型先根据示例 $C$ 推断“当前任务最可能是什么”，再在这个任务假设下对新输入 $x$ 生成输出 $y$。所以 ICL 不是权重层面的学习，而是前向计算里的条件推断。

---

## 问题定义与边界

先把问题限定清楚。

我们关心的是：给定一组上下文示例

$$
C=\{(x_1,y_1),(x_2,y_2),\dots,(x_k,y_k)\}
$$

模型如何利用这些示例，在不更新参数的情况下，对新输入 $x$ 预测输出 $y$。

这里有三个边界。

第一，ICL 不等于微调。微调是改模型参数；ICL 是参数不动，只改输入文本。白话说，微调是“把知识写进模型”，ICL 是“把任务说明临时塞进上下文”。

第二，ICL 讨论的是任务适应，而不是知识记忆。比如“把下面句子分成正面或负面”是任务；“巴黎是法国首都”是事实知识。两者在大模型里都会影响输出，但机制不完全相同。本文讨论的是“示例如何定义任务”。

第三，信息论里的量要在随机变量层面定义，而不是只看某一个具体样本。很多初学者会把一个具体 prompt 代入公式，直接说“这里互信息是 0 或 1”，这通常不严谨。互信息衡量的是一个分布上的平均信息量，不是单条样本的直觉评分。

下面给一个玩具例子。

假设输入 $X\in\{0,1\}$，输出 $Y\in\{0,1\}$。但在你看到示例前，不知道当前任务到底是：

- 任务 A：$Y=X$
- 任务 B：$Y=1-X$

并且先验上两种任务各占一半。

这时如果没有上下文 $C$，即使你知道 $X=1$，也无法确定 $Y$ 是 1 还是 0，所以：

$$
H(Y\mid X)=1\text{ bit}
$$

现在给你两个示例：

$$
C=\{(0,0),(1,1)\}
$$

这组示例几乎把任务锁定为 A，也就是“输出等于输入”。于是对于新样本，给定 $X$ 和 $C$ 后，$Y$ 基本确定：

$$
H(Y\mid X,C)\approx 0
$$

因此条件互信息变大。直观上看，不是 $X$ 自己突然变得更有信息，而是示例 $C$ 让“该如何解释 $X$”这件事变清楚了。

可以用表格看得更直观：

| 场景 | 是否知道任务概念 | $H(Y\mid C)$ | $H(Y\mid X,C)$ | 直观含义 |
|---|---:|---:|---:|---|
| 无上下文 | 否 | 高 | 高 | 模型不知道要做什么 |
| 有少量但一致的示例 | 部分知道 | 中 | 低 | 模型开始理解映射规则 |
| 有足够且一致的示例 | 是 | 中或低 | 很低 | 模型可稳定按规则输出 |
| 示例冲突严重 | 否 | 高 | 高 | 模型无法确定任务概念 |

注意一个常见误区：有些文字会写成“加入示例后 $H(Y\mid C)=0$”。这只有在上下文本身已经直接告诉你待预测样本的标签时才成立。一般 ICL 场景里，$C$ 只帮助你识别任务，不会直接泄露新样本的答案，所以更关键的是 $H(Y\mid X,C)$ 的下降，而不是把 $H(Y\mid C)$ 简单看成 0。

---

## 核心机制与推导

信息论视角和贝叶斯视角，实际上在描述同一件事。

### 1. 信息论视角：示例降低不确定性

公式

$$
I(Y;X\mid C)=H(Y\mid C)-H(Y\mid X,C)
$$

可以拆成两部分理解。

- $H(Y\mid C)$：只看上下文时，对输出还有多不确定。
- $H(Y\mid X,C)$：既看上下文又看当前输入时，对输出还有多不确定。

如果上下文把任务定义得很清楚，那么输入 $X$ 的作用就会被“正确解释”。例如在情感分类中，示例告诉模型输出空间是 `Positive/Negative`，也告诉它判断依据大致是情绪倾向；于是看到新评论后，$X$ 对 $Y$ 的解释力显著增强。

### 2. 贝叶斯视角：模型先猜任务，再做预测

设 $\theta$ 是隐含任务概念。白话说，$\theta$ 就是“这组示例背后真正的规则”。

则模型预测可以写成：

$$
p(y\mid x,C)=\int p(y\mid x,\theta)p(\theta\mid C)d\theta
$$

这个式子分两步。

第一步，用示例更新任务后验：

$$
p(\theta\mid C)\propto p(C\mid \theta)p(\theta)
$$

第二步，在每个候选任务下预测，再加权求和。

这就是“隐式贝叶斯推断”。模型没有显式写出 $\theta$，也没有真的跑一个贝叶斯程序，但它在前向传播里表现出类似效果。

### 3. 为什么示例越一致，ICL 往往越强

因为一致示例会让 $p(\theta\mid C)$ 更尖锐。白话说，模型对“当前任务究竟是什么”更有把握。

如果示例是：

- `今天真不错 -> Positive`
- `这电影很无聊 -> Negative`
- `服务很好，下次还来 -> Positive`

那么模型大概率会把 $\theta$ 推向“二分类情感任务”。

但如果示例混成：

- `今天真不错 -> Positive`
- `hello -> 你好`
- `2+2 -> 4`

那么同一个上下文里同时出现情感分类、翻译、算术，$p(\theta\mid C)$ 就会发散。模型不是不会算，而是不知道你此刻要它做哪一种映射。

### 4. 玩具例子：二选一规则识别

继续用刚才的二元映射。

候选任务只有两个：

- $\theta_1: Y=X$
- $\theta_2: Y=1-X$

先验上：

$$
p(\theta_1)=p(\theta_2)=0.5
$$

若看到示例 $C=\{(0,0)\}$，则：

- 在 $\theta_1$ 下概率高
- 在 $\theta_2$ 下概率低

于是 $p(\theta_1\mid C)$ 上升。再看到第二个示例 $(1,1)$，后验进一步集中到 $\theta_1$。这时新输入 $x=0$ 的输出基本被锁定为 $y=0$。

这个过程不是“学会了新参数”，而是“在有限候选规则中快速识别了当前规则”。

### 5. 真实工程例子：few-shot 文本分类

假设你要做客服工单路由，把文本分到 `billing`、`technical`、`account` 三类。你不给模型训练数据，也不微调，只在 prompt 中放几个样例：

```text
Ticket: I was charged twice this month.
Label: billing

Ticket: I cannot reset my password.
Label: account

Ticket: The API returns 500 on upload.
Label: technical

Ticket: My invoice does not match my plan.
Label:
```

这里的上下文 $C$ 不只是“展示了答案格式”，更重要的是让模型推断出当前任务概念 $\theta$：

- 输入是一段工单文本
- 输出是固定标签集合
- 标签依据是工单语义，不是摘要、翻译或情感

如果样例稳定、标签边界清楚，模型就会把 $p(\theta\mid C)$ 压到正确区域，从而提高对最后一个工单标签的预测质量。

---

## 代码实现

下面用一个最小可运行的 Python 例子，把“上下文帮助识别任务概念”写成显式的贝叶斯更新。这个代码不是 Transformer 实现，而是一个可验证的玩具模型，用来对应上面的公式。

```python
import math

def normalize(dist):
    s = sum(dist.values())
    return {k: v / s for k, v in dist.items()}

def entropy_binary(p):
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

# 两个候选任务：
# theta_eq: y = x
# theta_flip: y = 1 - x
def predict_y(theta, x):
    if theta == "theta_eq":
        return x
    if theta == "theta_flip":
        return 1 - x
    raise ValueError(theta)

def posterior_theta(context):
    # 均匀先验
    post = {"theta_eq": 0.5, "theta_flip": 0.5}
    for x, y in context:
        likelihood = {}
        for theta in post:
            likelihood[theta] = 1.0 if predict_y(theta, x) == y else 0.0
        post = {theta: post[theta] * likelihood[theta] for theta in post}
        post = normalize(post)
    return post

def p_y_given_x_and_c(x, context):
    post = posterior_theta(context)
    p_y1 = 0.0
    for theta, p_theta in post.items():
        y = predict_y(theta, x)
        if y == 1:
            p_y1 += p_theta
    return {0: 1 - p_y1, 1: p_y1}

# 无上下文时，x=1，对 y 仍完全不确定
no_context = []
p0 = p_y_given_x_and_c(1, no_context)
assert abs(p0[0] - 0.5) < 1e-9
assert abs(p0[1] - 0.5) < 1e-9
assert abs(entropy_binary(p0[1]) - 1.0) < 1e-9

# 有一致示例后，任务被识别为 y=x
context = [(0, 0), (1, 1)]
p1 = p_y_given_x_and_c(1, context)
assert abs(p1[1] - 1.0) < 1e-9
assert abs(entropy_binary(p1[1]) - 0.0) < 1e-9

print("posterior:", posterior_theta(context))
print("p(y|x=1,C):", p1)
```

这段代码说明三件事。

第一，示例 $C$ 的作用是更新 $p(\theta\mid C)$。  
第二，预测不是直接记忆某条样例，而是在“候选任务空间”里重新加权。  
第三，随着上下文变一致，输出熵下降，也就是不确定性下降。

如果把它映射到真实大模型推理流程，组织方式通常是这样：

```python
def build_prompt(examples, query):
    parts = []
    for x, y in examples:
        parts.append(f"Input: {x}\nLabel: {y}\n")
    parts.append(f"Input: {query}\nLabel:")
    return "\n".join(parts)

def infer_label(model, tokenizer, examples, query, label_tokens):
    prompt = build_prompt(examples, query)
    tokens = tokenizer(prompt, return_tensors="pt")
    logits = model(**tokens).logits[0, -1]

    # 只比较候选标签首 token，对分类任务是常见近似
    scores = {label: logits[tokenizer.encode(label, add_special_tokens=False)[0]].item()
              for label in label_tokens}

    # softmax 得到近似的 p(y|x,C)
    import math
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    z = sum(exps.values())
    probs = {k: v / z for k, v in exps.items()}

    return max(probs, key=probs.get), probs
```

这个伪代码强调一点：工程上虽然没有显式维护 $\theta$，但模型输出分布 `probs` 可以看成经过上下文调制后的 $p(y\mid x,C)$。few-shot 示例越一致，这个分布通常越尖锐；示例越冲突，分布通常越平。

---

## 工程权衡与常见坑

ICL 在工程里并不是“多放几个例子就行”。真正影响效果的，通常是示例质量，而不是示例数量本身。

先看一个故障表。

| 问题 | 根因 | 直接影响 |
|---|---|---|
| 示例标签前后不一致 | $p(\theta\mid C)$ 无法集中 | 输出摇摆，格式和标签混乱 |
| 示例顺序随意 | 注意力对前后位置敏感 | 后部样例覆盖前部规则，结果不稳定 |
| 标签命名不稳定 | 输出空间定义模糊 | 模型可能生成同义标签或自由文本 |
| 示例风格与真实输入差太大 | 分布偏移过强 | 任务识别失败，泛化变差 |
| 示例过少 | 后验信号弱 | 模型依赖预训练先验而不是当前任务 |
| 示例过多且冗余 | 占用上下文窗口 | 关键信号被稀释，延迟升高 |
| 多任务样例混在一起 | 概念空间冲突 | 模型不知道当前要执行哪种映射 |

### 1. 顺序敏感

Transformer 的注意力机制，白话说就是“每个 token 都会参考上下文里别的 token，但参考强度不均匀”。因此样例顺序会影响任务识别。

比如在情感分类里，把一组清晰正负例放在前面，再把一条异常格式样例放在最后，最后这条往往会产生过强干扰。工程上常见做法是：把最标准、最能代表边界的示例放在靠后位置，减少尾部噪声。

### 2. 一致性比多样性更优先

对零基础读者来说，可以把一致性理解成“示例在同一种题型里说同一种话”。例如：

```text
Review: ...
Sentiment: Positive
```

如果前三个样例都用这个模板，模型更容易识别任务。但如果你突然插入：

```text
Text = ...
Class -> good
```

虽然语义上接近，形式上却引入了额外自由度。对于人类这不难理解，但对于 ICL，这会削弱“当前任务概念”的集中度。

### 3. 真正需要的是边界清楚的样例

真实工程里，最有价值的样例不是“最典型”的，而是“最能区分类别边界”的。

还是以工单路由为例：

- `I was charged twice` 明显是 `billing`
- `Password reset link expired` 明显是 `account`

这两条有用，但还不够。更关键的可能是边界样例：

- `My plan upgrade did not reflect in the invoice`

它同时带有账号变更和账单语义。如果这类边界样例标得清楚，模型对类别划分的后验会更稳定。

### 4. 真实工程例子：日志告警归因

假设你在做 SRE 场景下的告警分类，希望把告警归因到 `network`、`storage`、`application`。如果 few-shot 示例全部来自应用层异常，而线上查询里突然出现大量磁盘错误，模型容易沿用已有先验，把新样本误判到最常见标签。

这里的问题不是模型“不懂英文日志”，而是上下文 $C$ 对当前任务边界的覆盖不足。你需要补充：

- 典型样例
- 边界样例
- 容易混淆的负例

这样才能让 $p(\theta\mid C)$ 不只是识别“这是分类任务”，而是识别“这是哪一种分类边界”。

### 5. 实用 checklist

| 检查项 | 推荐做法 |
|---|---|
| 输出标签 | 固定拼写，固定大小写，固定数量 |
| 示例模板 | 全部统一，避免混用多种提示格式 |
| 示例顺序 | 把最关键、最标准的样例放后部 |
| 示例覆盖 | 包含典型样例和边界样例 |
| 示例长度 | 尽量接近真实输入长度 |
| 任务隔离 | 一个 prompt 只做一种映射任务 |
| 结果验证 | 对标签分布、格式漂移做自动检查 |

---

## 替代方案与适用边界

ICL 很强，但不是默认最优方案。要把它放到更大的工程决策里看。

| 方案 | 是否改参数 | 标注成本 | 在线延迟 | 适合场景 | 主要限制 |
|---|---:|---:|---:|---|---|
| ICL | 否 | 低到中 | 中到高 | 快速试验、低频任务、多任务切换 | 受上下文窗口和示例质量限制 |
| Fine-tuning | 是 | 中到高 | 低到中 | 稳定高频任务、固定标签空间 | 训练成本高，迭代慢 |
| Retrieval + ICL | 否 | 中 | 高 | 知识更新快、样例需动态匹配 | 检索质量决定上限 |
| Prefix/Adapter 等轻量调参 | 是 | 中 | 低到中 | 有一定训练条件但不想全量微调 | 仍需训练和部署管理 |

### 1. 什么时候优先用 ICL

- 任务经常变化，没法为每个任务都训练一个模型
- 你手里只有少量高质量示例
- 系统要求快速试验，能接受一定的推理成本
- 你需要同一个模型在一次会话中切换多个任务

### 2. 什么时候 ICL 不够

如果任务要求高可靠性、长尾边界复杂、标签空间固定且调用量大，单纯依赖 ICL 往往不稳。因为它把任务定义压在 prompt 上，而 prompt 容量有限，且对格式敏感。

例如一个大规模线上风控分类器，每天处理百万级请求，此时更常见的方案是：

- 先用监督数据做微调，让模型固化基础边界
- 再在推理时少量加入 ICL 示例，做最后的场景修正

### 3. Retrieval + ICL 什么时候更合适

检索增强，白话说就是“先从库里找最相关的示例或文档，再把它们塞进上下文”。它适合下面这种情况：

- 任务规则会随时间变化
- 不同子领域差异大
- 你无法把所有边界样例都固定写进 prompt

比如法律问答、客服路由、代码修复建议，这些任务都可能强依赖领域上下文。先检索相似案例，再做 ICL，等于让 $p(\theta\mid C)$ 建立在“更相关的 $C$”上，因此比固定 few-shot 更稳定。

### 4. 一个清晰边界

ICL 擅长的是“从少量示例中识别任务规则”。  
它不擅长的是“在示例极弱、规则极复杂、输出必须高度一致”的环境下单独承担生产级决策。

所以最稳妥的判断标准不是“ICL 能不能做”，而是：

- 当前任务规则能否被少量示例表达清楚
- 上下文窗口是否足够容纳关键样例
- 输出错误的代价是否允许 prompt 级不稳定性

如果这三个条件都不满足，就应该考虑微调、检索增强，或者两者与 ICL 的混合方案。

---

## 参考资料

- Xie, Sang Michael, et al. “An Explanation of In-context Learning as Implicit Bayesian Inference.” ICLR 2022. https://iclr.cc/virtual/2022/poster/6893
- 论文 PDF 镜像：Heidelberg CL course page. https://www.cl.uni-heidelberg.de/courses/ss25/the_mystery_of_in-context_learning_of_llms/papers/XieICLR2022.pdf
- IBM Think. “What is In-Context Learning?” https://www.ibm.com/think/topics/in-context-learning
- Wikipedia. “Conditional mutual information.” https://en.wikipedia.org/wiki/Conditional_mutual_information
- Emergent Mind. “In-Context Few-Shot Learning.” https://www.emergentmind.com/topics/in-context-few-shot-learning
