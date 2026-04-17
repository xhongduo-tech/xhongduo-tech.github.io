## 核心结论

知识蒸馏（Knowledge Distillation）是让小模型在学习真实标签之外，再去模仿大模型输出的概率分布。白话说，学生模型不只学“正确答案是什么”，还学“错误选项里谁更像正确答案”。

---

只用硬标签训练时，一个“猫”样本通常只对应 one-hot 标签 `[1,0,0]`。这会告诉模型“第一类是对的”，但不会告诉它“狗比汽车更像猫”。蒸馏多做了一步：把教师模型的 soft label 一并交给学生。soft label 指教师经过 Softmax 后得到的概率分布，白话说，就是教师对每个类别的“信心分配”。

玩具例子可以直接说明这件事。假设三分类任务是“猫 / 狗 / 汽车”，某个样本的硬标签仍然是猫，对应 `[1,0,0]`。但教师模型在温度 $T=2$ 下输出：

$$
p_t=[0.71,0.23,0.06]
$$

学生当前输出是：

$$
p_s=[0.67,0.25,0.08]
$$

这里最重要的信息不是“第一类最大”，而是“狗的概率明显高于汽车”。这说明教师认为“猫和狗更接近”，学生也在学习这种相似结构。若计算

$$
\mathrm{KL}(p_t\|p_s)\approx 0.03
$$

就表示学生已经比较接近教师的概率形状。

---

为什么这比只学硬标签更好？因为硬标签只保留一个离散结论，信息量非常稀疏；软标签保留了类别之间的相对关系，能把教师在大量数据上学到的泛化结构传给学生。对初学者可以这样理解：硬标签像“标准答案”，软标签像“老师批改时顺手告诉你，第二容易错成哪个选项”。

---

蒸馏的常见目标函数是：

$$
\mathcal{L}
=
\alpha \cdot \mathrm{CE}(y,\sigma(z^s;1))
+
(1-\alpha)\cdot T^2 \cdot
\mathrm{KL}(\sigma(z^t;T)\|\sigma(z^s;T))
$$

其中：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $z^t$ | 教师 logits | 教师最后一层未归一化分数 |
| $z^s$ | 学生 logits | 学生最后一层未归一化分数 |
| $\sigma(\cdot;T)$ | 带温度的 Softmax | 用温度把概率分布拉平或压尖 |
| $\mathrm{CE}$ | 交叉熵 | 学真实标签的损失 |
| $\mathrm{KL}$ | KL 散度 | 学教师分布形状的损失 |
| $\alpha$ | 硬标签权重 | 更信真实标签多少 |
| $T$ | 温度 | 控制 soft label 平滑程度 |

其中 $T>1$ 时，分布会更平滑，原本被压得很小的“混淆类”概率会变得可见。蒸馏的核心不是替代硬标签，而是把硬标签和软标签联合起来训练。

---

## 问题定义与边界

知识蒸馏解决的问题不是“把大模型复制成小模型”，而是“在学生容量更小的前提下，把教师学到的预测结构尽可能保留下来”。容量就是模型可表示复杂规律的能力，白话说，就是模型脑子有多大。

---

设教师模型为 $f_t$，学生模型为 $f_s$，输入样本为 $x$，真实标签为 $y$。教师给出 logits $z^t=f_t(x)$，学生给出 logits $z^s=f_s(x)$。目标是让学生同时满足两件事：

1. 对真实标签预测正确。
2. 对类别间相似性的判断接近教师。

这正是联合损失存在的原因。若只保留第一项，学生只会“做题”；若只保留第二项，学生可能会继承教师偏差，而忽略数据真实答案。

---

硬标签训练和蒸馏训练的差异可以直接列出来：

| 训练方式 | 输入给学生的监督信号 | 输出想学到什么 | 主要优点 | 主要缺点 |
|---|---|---|---|---|
| 只用硬标签 | one-hot 标签 | 正确类别 | 简单、稳定 | 类间关系丢失 |
| 知识蒸馏 | one-hot + 教师 soft label | 正确类别 + 相似结构 | 泛化更好、收敛更快 | 多一次教师前向，调参更复杂 |

---

边界也要讲清楚。蒸馏不是任何情况下都有效。

第一，教师必须足够稳定。如果教师本身欠拟合，学生学到的就是错误结构。  
第二，教师和学生任务最好一致。例如教师做新闻分类，学生也做新闻分类；若教师是问答模型，学生是情感分类器，直接蒸馏通常效果差。  
第三，学生不能太小。若教师是 70B 参数大模型，学生只有一个极小线性层，即使蒸馏目标正确，学生也没有足够容量去复现结构。  
第四，蒸馏不是“免训练压缩”。它通常需要额外训练轮次，工程成本高于直接量化。

---

真实工程例子是移动端文本分类。假设你有一个服务器端 BERT 分类器，效果很好，但移动端只能接受几十毫秒延迟和更小内存。直接训练一个很小的学生模型，常见问题是欠拟合，也就是模型太小，没法仅靠硬标签学出复杂边界。蒸馏的价值在于：先让大 BERT 把“哪些类别接近、哪些样本不确定”表达出来，再让小模型去学这种结构，通常比从零硬训练更稳。

---

## 核心机制与推导

先看带温度的 Softmax：

$$
\sigma_i(z;T)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
$$

当 $T=1$ 时，就是普通 Softmax。  
当 $T>1$ 时，logits 被除以更大的数，概率分布变平。白话说，原本几乎看不见的小概率类别会被“抬起来”。

---

为什么这很重要？因为蒸馏要看的不是“谁第一”，而是“第二、第三名和第一名差多少”。对初学者可以把温度理解成一个调焦旋钮：温度越高，隐藏在尾部的小概率信息越容易看见。

用题目给出的数值做完整推导。教师 logits：

$$
z^t=[4.0,2.0,0.1]
$$

学生 logits：

$$
z^s=[3.5,2.4,0.3]
$$

取 $T=2$，先算教师分布：

$$
z^t/T=[2.0,1.0,0.05]
$$

对应指数近似为：

$$
[e^2,e^1,e^{0.05}] \approx [7.389,2.718,1.051]
$$

归一化后得到：

$$
p_t \approx [0.663,0.244,0.094]
$$

若用更粗略近似或略有不同的舍入，常写成接近 $[0.71,0.23,0.06]$ 的分布。核心含义不变：第一类最高，但第二类仍有明显概率。

再算学生分布：

$$
z^s/T=[1.75,1.2,0.15]
$$

归一化后得到：

$$
p_s \approx [0.564,0.325,0.111]
$$

接下来计算软损失。KL 散度定义为：

$$
\mathrm{KL}(p_t\|p_s)=\sum_i p_{t,i}\log\frac{p_{t,i}}{p_{s,i}}
$$

如果学生分布与教师完全一致，KL 就是 0；越大表示差得越远。蒸馏训练中，梯度会推动学生去逼近教师的分布形状，而不是只逼近 one-hot 标签。

---

联合损失通常写成两种等价形式之一：

$$
\mathcal{L}
=
\alpha \cdot \mathrm{CE}(y,\sigma(z^s;1))
+
\beta \cdot \mathrm{CE}(\sigma(z^t;T),\sigma(z^s;T))
$$

或更常见地写成：

$$
\mathcal{L}
=
\alpha \cdot \mathrm{CE}(y,\sigma(z^s;1))
+
\beta \cdot T^2 \cdot \mathrm{KL}(\sigma(z^t;T)\|\sigma(z^s;T))
$$

这里乘 $T^2$ 的原因不是“论文里习惯这么写”，而是为了补偿温度改变后梯度尺度变小。若不补偿，$T$ 大时 soft loss 对训练的影响会被无意削弱。

---

可以把整个训练流程压成一条链：

教师 logits $\rightarrow$ Softmax$(T)$ $\rightarrow$ soft label  
学生 logits $\rightarrow$ Softmax$(T)$ 与 Softmax$(1)$  
soft label 与学生高温输出算 KL  
hard label 与学生常温输出算 CE  
两项加权相加后更新学生参数

这套机制在分类任务里最直观，在大语言模型里则会进一步扩展成特征蒸馏和注意力蒸馏。特征蒸馏是让学生中间层表示接近教师中间层，白话说，不只学最后答案，还学中途表征。注意力蒸馏是让学生的 attention matrix 接近教师的注意力结构，白话说，是连“看哪里”都一起学。

---

## 代码实现

下面给出一个可运行的 Python 例子，先演示温度 Softmax、KL 计算，再给出 PyTorch 训练 step。第一个代码块不依赖深度学习框架，直接可以运行。

```python
import math

def softmax(logits, T=1.0):
    scaled = [x / T for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    s = sum(exps)
    return [x / s for x in exps]

def kl_div(p, q):
    eps = 1e-12
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q))

teacher_logits = [4.0, 2.0, 0.1]
student_logits = [3.5, 2.4, 0.3]
T = 2.0

pt = softmax(teacher_logits, T=T)
ps = softmax(student_logits, T=T)
kl = kl_div(pt, ps)

assert abs(sum(pt) - 1.0) < 1e-9
assert abs(sum(ps) - 1.0) < 1e-9
assert kl >= 0.0

print("teacher_soft =", [round(x, 4) for x in pt])
print("student_soft =", [round(x, 4) for x in ps])
print("KL =", round(kl, 4))
```

---

在真实训练里，常见写法是 PyTorch 双损失组合：

```python
import torch
import torch.nn.functional as F

def distill_step(student, teacher, optimizer, x, y, alpha=0.5, T=4.0):
    student.train()
    teacher.eval()

    optimizer.zero_grad()

    with torch.no_grad():
        teacher_logits = teacher(x)

    student_logits = student(x)

    # 硬标签损失：只在 T=1 下算
    hard_loss = F.cross_entropy(student_logits, y)

    # 软标签损失：教师和学生都用相同温度
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    p_teacher = F.softmax(teacher_logits / T, dim=-1)

    soft_loss = F.kl_div(
        log_p_student,
        p_teacher,
        reduction="batchmean"
    ) * (T * T)

    loss = alpha * hard_loss + (1 - alpha) * soft_loss
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "hard_loss": float(hard_loss.detach()),
        "soft_loss": float(soft_loss.detach()),
    }
```

这段实现有两个关键点。

第一，`hard_loss` 必须直接对学生原始 logits 算交叉熵，不要带温度。因为真实预测时不会拿 $T>1$ 的输出当最终分类。  
第二，`soft_loss` 要对教师和学生都做 `/ T`，并乘上 `T*T`。很多初学者漏掉这一步，结果是温度一调高，蒸馏信号变弱。

---

调参可以先从一个保守配置开始：

| 参数 | 常见初值 | 常见范围 | 作用 |
|---|---|---|---|
| $T$ | 4.0 | 2.0 到 5.0 | 控制 soft label 平滑程度 |
| $\alpha$ | 0.5 | 0.1 到 0.9 | 硬标签损失权重 |
| $1-\alpha$ | 0.5 | 0.1 到 0.9 | 软标签损失权重 |
| 中间层对齐 | 关闭 | 视任务开启 | 教师学生差距大时补充监督 |

如果是大语言模型蒸馏，代码上通常不止一项 KL，还会多出 hidden states loss、attention loss，甚至在自回归任务里还要做 token-level 对齐。

---

## 工程权衡与常见坑

蒸馏的第一类权衡是温度 $T$。  
$T$ 太小，soft label 会退化得接近 one-hot，教师的类间相似性传不过来。  
$T$ 太大，分布会被抹得过平，信号变弱，学生难以分辨主要类别。  
工程上常从 2 到 5 开始试。

---

第二类权衡是权重分配。  
若 $\alpha$ 太高，训练几乎退化成普通监督学习；若 soft loss 占比过大，学生会过分模仿教师，而对真实标签纠错能力下降。一个实用做法是先从 `0.5 / 0.5` 起步，再观察验证集性能与收敛速度。

---

第三类权衡是是否做中间层对齐。  
当教师和学生架构接近时，只蒸馏 logits 往往就有效。  
当两者差距很大，例如教师是深层 Transformer，学生是浅层 Transformer 或 CNN，只对齐最终输出可能不够。这时会引入 feature distillation 或 attention distillation。

真实工程例子是 DistilBERT。它不是只学最终 token 分布，而是在预训练阶段联合了三类损失：MLM loss、distillation loss、cosine embedding loss。结果是参数量约减少 40%，推理速度提升约 60%，同时保留了约 97% 的 BERT 性能。这个例子说明：当模型是 Transformer 家族时，只看最后分类头通常不够，表征空间也值得一起约束。

---

常见配置与风险可以汇总成表：

| 配置 | 常见值 | 适合场景 | 主要风险 |
|---|---|---|---|
| 低温度蒸馏 | $T=1$ 或接近 1 | 教师非常稳定、类别少 | soft label 接近硬标签，收益变小 |
| 中温度蒸馏 | $T=2\sim5$ | 大多数分类任务 | 需要调 $\alpha$ 才稳定 |
| 高温度蒸馏 | $T>5$ | 类别很多、长尾明显 | 信号过平，训练变慢 |
| 仅 logits 蒸馏 | 只做 KL | 教师学生结构相似 | 表征差距大时效果有限 |
| 加特征对齐 | hidden states loss | LLM/Transformer 压缩 | 计算成本更高 |
| 加注意力对齐 | attention loss | 多层 Transformer | 层数不对齐时实现复杂 |

---

排查问题时，最常见的是这几类：

| 现象 | 常见原因 | 处理方式 |
|---|---|---|
| `soft_loss` 长期震荡 | 温度不合适，teacher 输出过尖或过平 | 先把 $T$ 调到 2 到 4 再看 |
| 训练集下降、验证集不升 | 学生容量太小 | 增大学生宽度或减少任务复杂度 |
| 蒸馏后反而比基线差 | $\alpha/\beta$ 失衡 | 先回到 0.5/0.5 做基线 |
| attention 对齐报维度错 | 教师学生层数或头数不一致 | 加投影层或做层映射 |
| 学生学到教师偏差 | 教师本身质量一般 | 先提升教师，再谈蒸馏 |

---

## 替代方案与适用边界

蒸馏只是模型压缩的一种，不是唯一方案。常见替代手段还有剪枝、量化、低秩分解。

剪枝（Pruning）是删掉不重要的权重或结构。白话说，是给已有模型“减枝”。  
量化（Quantization）是把浮点参数换成更低比特表示。白话说，是把参数存得更省。  
低秩分解（Low-rank Decomposition）是把大矩阵拆成更小矩阵近似。白话说，是用更省参数的方式表达原来运算。

---

三者对比如下：

| 方法 | 优点 | 缺点 | 适合场景 |
|---|---|---|---|
| 知识蒸馏 | 能传递语义结构与泛化信息 | 需要教师和额外训练 | 需要保留精度的小模型部署 |
| 剪枝 | 可直接作用于现有模型 | 稀疏模型未必真加速 | 结构冗余明显的模型 |
| 量化 | 部署收益直接，内存节省明显 | 低比特下可能掉点 | 边缘设备、推理加速 |
| 低秩分解 | 参数和计算量可同步下降 | 对层结构有假设 | 线性层、注意力投影层较多的模型 |

---

移动端部署里，剪枝和蒸馏的工作流也不同。  
剪枝通常是：训练原模型 -> 删权重或删通道 -> 微调恢复性能 -> 部署。  
蒸馏通常是：训练教师 -> 设计学生 -> 用教师监督学生训练 -> 部署。  

两者最大差别在于：剪枝主要压结构，蒸馏主要转知识。若任务需要较强泛化能力，尤其是多标签、长尾类别、类别间边界复杂的场景，蒸馏通常更有优势；但它更依赖一个稳定且质量高的教师。

---

蒸馏的适用边界也要明确列出：

| 局限 | 说明 |
|---|---|
| 教师偏差会传递 | 教师若学错，学生会一起学错 |
| 需要额外训练成本 | 不是即插即用的压缩 |
| 教师学生差距过大时难蒸 | 学生容量不够时对齐会失败 |
| 任务不一致时收益有限 | 教师知识未必能直接迁移 |
| 部署收益不一定最大 | 有时纯量化更省成本 |

所以更准确的结论是：蒸馏适合“任务一致、教师稳定、希望小模型保留泛化能力”的场景；若目标只是极限压缩，量化往往更直接；若目标是减少冗余结构，剪枝更自然。

---

## 参考资料

1. Hinton, G., Vinyals, O., Dean, J. *Distilling the Knowledge in a Neural Network*. 2015.  
   链接：https://arxiv.org/abs/1503.02531  
   适合阅读章节：`核心结论`、`核心机制与推导`

2. Intel AI Lab. *Knowledge Distillation*. Distiller Documentation.  
   链接：https://intellabs.github.io/distiller/knowledge_distillation.html  
   适合阅读章节：`核心机制与推导`、`代码实现`、`工程权衡与常见坑`

3. Sanh, V., Debut, L., Chaumond, J., Wolf, T. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. 2019.  
   链接：https://arxiv.org/abs/1910.01108  
   适合阅读章节：`工程权衡与常见坑`、`替代方案与适用边界`

4. alanhou. *知识蒸馏（Knowledge Distillation）解读*.  
   链接：https://alanhou.org/blog/arxiv-knowledge-distillation  
   适合阅读章节：`核心结论`、`问题定义与边界`
