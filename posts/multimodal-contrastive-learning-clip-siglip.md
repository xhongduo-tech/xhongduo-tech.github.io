## 核心结论

SigLIP 可以看成 CLIP 的一次目标函数替换：把“整批样本一起做 softmax 排名”改成“每一对图文单独做 sigmoid 二分类”。二分类的意思是，模型只判断这一对是否匹配，而不是先和整批其它样本一起竞争概率。

它的核心收益不是“公式更短”，而是切断了 softmax 对全局分母的依赖。CLIP 的损失要知道“这张图和 batch 里所有文本相比谁更像”，因此分布式训练时通常需要跨设备收集全部特征；SigLIP 的损失写成

$$
L=-\sum_{i,j}\log \sigma\!\big(y_{ij}\, t\, sim_{ij}\big)
$$

其中 $sim_{ij}$ 是图像 $i$ 与文本 $j$ 的相似度，通常用余弦相似度；余弦相似度就是把两个向量夹角转成 $[-1,1]$ 的分数，越大表示方向越一致。$y_{ij}\in\{+1,-1\}$ 表示是否匹配，$t$ 是温度参数；温度参数就是一个可学习缩放因子，用来控制 logits 的“陡峭程度”。

这个写法的关键在于：每一项损失只依赖当前这一对的 $sim_{ij}$ 和标签 $y_{ij}$。因此它不需要 softmax 的全局归一化，也就不强依赖全局 batch 通信。对工程实现来说，这一点往往比“理论上是否更优”更重要，因为多机训练里最贵的部分常常不是乘加，而是同步。

玩具例子可以这样理解。假设你手里有一张猫的图片和两段文本：“一只橘猫趴在窗台上”“一辆红色卡车停在路边”。CLIP 会让这张图同时和这两段文本、以及 batch 里更多文本一起竞争；SigLIP 则把“猫图-猫文”记成正例，把“猫图-卡车文”记成负例，然后分别算两个 sigmoid 损失，最后直接相加。新加一个负样本，只是多加一项，不会改写已有正样本的 softmax 分母。

真实工程例子是视觉塔 Stage0 预训练。视觉塔就是多模态模型里负责把图像编码成向量的模块，Stage0 则是先把视觉编码器单独训好、再接后续语言模块的阶段。PaLI-3、PaliGemma、Gemini 系列相关工作中，SigLIP 之所以被反复采用，不是因为它替换后一定在所有设定下都压倒 CLIP，而是因为它更适合大规模图文预训练的分布式约束：通信更轻，显存更容易控，作为视觉编码器预训练目标更稳定。

---

## 问题定义与边界

问题可以表述为：给定一批图像和一批文本，如何训练两个编码器，使匹配图文的向量更接近，不匹配图文的向量更远离。这里的“编码器”就是把原始输入变成向量的模型，例如 ViT 处理图片、Transformer 处理文本。

CLIP 的做法本质上是 InfoNCE 风格对比学习。对比学习就是“拉近正样本，推远负样本”的训练方式。它会构造一个 $N\times N$ 的相似度矩阵，对每张图像把对应文本当正例，其余文本当负例，然后做行方向 softmax；通常还会反向再做一次列方向 softmax。问题在于 softmax 的分母要包含同一批里所有候选项，因此如果图像特征和文本特征分散在多张卡上，就要把它们聚到一起再算。

SigLIP 的边界则更明确：它不是在解决“如何获得更多负样本”这个根本问题，而是在解决“损失函数是否必须依赖全局归一化”这个工程瓶颈。它把每一对图文独立看成二分类任务，避免了对全局分母的依赖，但并没有神奇地消灭负样本设计问题。负样本仍然重要，只是使用方式从“参与同一个 softmax 竞争”变成了“作为独立二分类项累加”。

下面的表格可以直接看出两类方法的差别：

| 维度 | CLIP softmax | SigLIP sigmoid |
|---|---|---|
| 基本目标 | 在整批候选中把正确配对排到最高 | 对每对图文独立判断是否匹配 |
| 梯度依赖范围 | 依赖同一 softmax 分母中的全部样本 | 只依赖当前这一对的 logit 与标签 |
| 通信需求 | 常需要跨设备收集整批特征 | 可在更局部的 batch 内完成 |
| 对 batch 的依赖 | 较强，batch 越大负样本越丰富 | 中等，更依赖负样本采样质量 |
| 新增负样本的影响 | 会改动 softmax 分母与相对概率 | 直接多加一项损失 |
| 适合场景 | 有能力维护大规模全局负样本 | 通信和显存更紧的分布式训练 |

新手版本可以用“配对题”理解。CLIP 像是老师要求你把一张图片和全班所有句子一起比，最后必须在整班里排第一；SigLIP 像是老师把若干卡片发到你手里，你只要逐张判断“这对是不是一组”。后者的信息范围更局部，所以更容易在资源有限时运行。

但它也有边界。论文和后续经验都表明，当 batch 极大时，softmax 的全局负样本优势会重新显现，SigLIP 的收益不再明显。经验上，盲目把 batch 推到极端大，并不能持续放大 SigLIP 的优势；更常见的有效策略是维持中等 batch，同时把负样本多样性和数据质量做好。

---

## 核心机制与推导

先定义相似度：

$$
sim_{ij}=\frac{\langle v_i, u_j\rangle}{\|v_i\|\,\|u_j\|}
$$

其中 $v_i$ 是图像向量，$u_j$ 是文本向量。这个式子就是余弦相似度，表示两个向量方向有多一致。

SigLIP 的监督信号是成对标签：

$$
y_{ij}=
\begin{cases}
+1, & \text{图像 } i \text{ 与文本 } j \text{ 匹配}\\
-1, & \text{不匹配}
\end{cases}
$$

然后用温度参数 $t$ 把相似度缩放成 logit：

$$
z_{ij}=t\,sim_{ij}
$$

于是单对损失是：

$$
\ell_{ij}=-\log \sigma(y_{ij} z_{ij})
$$

总损失就是所有采样对的求和：

$$
L=\sum_{i,j}\ell_{ij}
= -\sum_{i,j}\log \sigma\!\big(y_{ij}t\,sim_{ij}\big)
$$

如果再引入偏置 $b$，也可以写成：

$$
\ell_{ij}=-\log \sigma\!\big(y_{ij}(t\,sim_{ij}+b)\big)
$$

偏置 $b$ 的作用是平移判别边界。判别边界就是模型把“正”和“负”分开的那条线。负样本很多时，适当调 $b$ 可以减轻模型过早把大多数样本都压成负例的倾向。

它和 softmax 的关键不同在梯度。因为

$$
\frac{d}{dx}\big[-\log \sigma(x)\big] = \sigma(x)-1
$$

所以对单对 logit $z_{ij}$ 有：

$$
\frac{\partial \ell_{ij}}{\partial z_{ij}}
= y_{ij}\big(\sigma(y_{ij} z_{ij})-1\big)
$$

再乘上 $z_{ij}=t\,sim_{ij}$ 的链式法则，就得到：

$$
\frac{\partial \ell_{ij}}{\partial sim_{ij}}
= t\,y_{ij}\big(\sigma(y_{ij} z_{ij})-1\big)
$$

这里最重要的观察是：这个梯度只用到了当前这一对的 $sim_{ij}$、$y_{ij}$ 和 $t$。它不需要“其它所有负样本的相似度先求和再归一化”。这就是它为什么更适合减少跨设备同步。

看一个题目要求中的数值玩具例子。设正对相似度 $sim_{pos}=0.8$，负对相似度 $sim_{neg}=-0.2$，温度 $t=1.5$。

正对：

$$
z_{pos}=1.5\times 0.8=1.2,\quad
\ell_{pos}=-\log \sigma(1.2)\approx 0.26
$$

负对：

$$
z_{neg}=1.5\times (-0.2)=-0.3,\quad
\ell_{neg}=-\log \sigma(0.3)\approx 0.55
$$

如果按“负例标签为 $-1$，代入 $y_{ij}z_{ij}$”理解，本质上就是把负样本的错误置信度也映射到 log-sigmoid 上。你新增一个负样本，例如 $sim=-0.1$，只需再加一项对应损失。旧的正例梯度不会因为新负样本进入 softmax 分母而被重新归一化。

真实工程里的机制更有代表性。假设你在做一个中英双语电商搜索模型，视觉塔需要先在“商品图-标题文本”上做预训练。每轮训练中，不同 GPU 上拿到的是不同商品数据。CLIP 风格做法为了获得更大负样本池，通常倾向于 all-gather 全部图文向量；SigLIP 则可以只在当前设备或局部组内构造配对并累加损失。这样做并不意味着负样本更强，而是意味着训练吞吐更容易受算力限制，而不是先被通信拖死。

---

## 代码实现

实现上最直接的方式，是先得到归一化后的图像向量和文本向量，再计算相似度，然后把标签映射到 $\{0,1\}$，交给 `binary_cross_entropy_with_logits`。这个函数的意思是“输入原始 logit，内部自动做 sigmoid 再算二分类交叉熵”。

下面是一个可运行的 Python 玩具实现，用纯 `math` 模拟单对损失，并验证正样本更像时损失更小：

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def siglip_pair_loss(sim: float, y: int, t: float = 1.5, b: float = 0.0) -> float:
    assert y in (+1, -1)
    logit = y * (t * sim + b)
    return -math.log(sigmoid(logit))

def total_loss(pairs, t: float = 1.5, b: float = 0.0) -> float:
    return sum(siglip_pair_loss(sim, y, t=t, b=b) for sim, y in pairs)

# 玩具例子：一个正对，一个负对
pairs = [
    (0.8, +1),   # 图文匹配，相似度较高
    (-0.2, -1),  # 图文不匹配，相似度偏低
]

loss = total_loss(pairs, t=1.5)
assert loss > 0

# 如果正样本更像、负样本更不像，总损失应下降
better_pairs = [
    (0.9, +1),
    (-0.5, -1),
]
assert total_loss(better_pairs, t=1.5) < loss

# 如果把负样本错误地变得更像，损失会上升
worse_pairs = [
    (0.8, +1),
    (0.3, -1),
]
assert total_loss(worse_pairs, t=1.5) > loss

print(round(loss, 4))
```

如果换成 PyTorch，推荐直接用 logits 版本 BCE，而不是手写 `sigmoid` 后再取对数，原因是数值稳定性更好。数值稳定性就是大正数或大负数时不容易出现上溢、下溢或梯度异常。

```python
import torch
import torch.nn.functional as F

def siglip_loss(image_emb, text_emb, temperature, bias=0.0):
    # image_emb: [B, D]
    # text_emb:  [B, D]
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)

    sim = image_emb @ text_emb.T              # [B, B]
    logits = temperature * sim + bias

    targets = torch.eye(sim.size(0), device=sim.device)
    # 对角线是正样本 1，非对角线是负样本 0
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="sum",
    )
    return loss / sim.size(0)
```

这里有两个实现层面的点。

第一，虽然公式写成对所有 $(i,j)$ 求和，但工程上不一定必须保留完整的 $B\times B$ 矩阵。若 batch 很大，可以只采样部分负对，或者按块计算相似度矩阵再累加损失。SigLIP 的优势不在于“矩阵绝对不用构造”，而在于“损失不要求你把所有设备上的候选放进同一个 softmax 归一化”。

第二，标签表示可以有两种写法。数学公式常写 $y\in\{+1,-1\}$；`BCEWithLogits` 常用 target $\in\{0,1\}$。两者是等价重参数化，不要在实现里混用后忘记调整公式。

真实工程例子是多机视觉塔预训练。假设 64 张卡训练一个 ViT 视觉编码器和文本编码器，CLIP 风格常见做法是先 all-gather 特征，再算全局对比损失；SigLIP 可以只在每卡本地 batch 或节点内小组 batch 上采样负对，然后用 BCE 累加。这样做会让通信图更简单，也更容易把瓶颈留给矩阵乘法而不是网络带宽。

---

## 工程权衡与常见坑

SigLIP 的优点很明确，但工程上最常见的误解是：既然它不依赖全局 softmax，那就无限加负样本即可。实际并非如此。负样本数量过大时，sigmoid 二分类容易出现梯度饱和；梯度饱和就是 logit 太偏向一侧后，sigmoid 导数接近 0，参数虽然还在更新，但有效学习信号很弱。

常见坑和规避方式可以压缩成下表：

| 常见坑 | 现象 | 原因 | 规避策略 |
|---|---|---|---|
| 负样本过多 | 模型很快把大部分对都判负 | 正负比例失衡，负项主导损失 | 控制负样本采样量，做难负样本采样 |
| 温度 $t$ 过大 | 训练初期不稳定，logit 过陡 | sigmoid 快速进入饱和区 | 让 $t$ 可学习或用较小初值 |
| 温度 $t$ 过小 | 区分度不足，收敛慢 | logits 被压平 | 联合调大学习率或增大有效负样本难度 |
| 缺少偏置 $b$ | 正负阈值不合适 | 决策边界固定在 0 附近 | 在负样本很多时引入可学习 $b$ |
| batch 盲目做大 | 收益变小甚至不经济 | SigLIP 优势主要在通信，不是无限放大 batch | 用中等 batch，提升数据质量和采样策略 |
| 直接手写 `log(sigmoid(x))` | 数值不稳定 | 大负值时容易下溢 | 用 `binary_cross_entropy_with_logits` |

新手版本可以这么理解：如果你一轮给模型 10 倍负样本，但不调温度，也不控制采样，模型很可能学会“几乎全部判负”。这并不是它真的理解了图文，而是它找到了一个短期降低损失的偏置解。要解决这个问题，通常不是继续硬堆更多负样本，而是调 $t$、调 $b$、采更有信息量的负样本，或者限制负样本规模。

还有一个容易忽略的点是“没有全局归一化”不等于“完全不要全局信息”。如果你的业务非常依赖海量候选中的细粒度排序，例如超大规模跨模态检索，softmax 或带 memory bank 的方法仍然有价值。SigLIP 更像是把训练目标从“全局竞争”改成“局部可扩展的判别学习”，因此要接受一个现实：你减少了通信依赖，也可能减少了全局负样本对排序边界的直接塑形。

---

## 替代方案与适用边界

如果把主流方案放在同一坐标系里，可以把问题分成两维：一是你有没有能力维护大规模全局负样本，二是你的系统瓶颈是通信还是计算。

| 方法 | batch 要求 | 通信需求 | 负样本使用方式 | 更适合的场景 |
|---|---|---|---|---|
| SigLIP | 中等 batch 即可工作 | 较低 | 每对独立二分类，可局部采样 | 通信受限的分布式图文预训练 |
| CLIP softmax | 往往更吃大 batch | 较高 | 同批负样本进入同一 softmax | 追求全局排序、可承受 all-gather |
| NT-Xent / InfoNCE | 与 CLIP 接近 | 中到高 | 通过归一化概率对比正负样本 | 通用对比学习基线 |
| Triplet loss | 可较小 batch | 较低 | 依赖锚点、正例、负例三元组 | 明确采样规则的小规模任务 |
| Memory bank / MoCo 类 | 可小 batch | 中等 | 用队列维护历史负样本 | 想补足 batch 不够大的负样本池 |

适用边界可以直接说清楚。

如果你只有有限显存，训练设备也不适合每步都做重型 all-gather，SigLIP 通常比 CLIP 更现实。比如单机 8 卡或多机但带宽一般的环境，先把视觉塔训稳，比理论上把负样本池做得无限大更重要。

如果你有非常大的 batch、成熟的通信优化、并且任务高度依赖全局排序，softmax 方案仍然值得保留。因为 softmax 的优势就在“让正例必须战胜整批负例”，这对检索排名任务天然更贴近目标。

如果你需要的是结构简单、容易解释的局部判别目标，SigLIP 是很好的折中。它尤其适合 Stage0 视觉编码器预训练，因为这一步的目标通常是先学到强对齐表示，再交给后续生成式模块继续联合训练。

新手版选择规则可以写成一句话：设备只能稳定处理几百到几千规模的局部 batch，优先考虑 SigLIP；如果你真能稳定管理超大 batch 和全局负样本池，CLIP softmax 仍然有竞争力。

---

## 参考资料

1. 《Sigmoid Loss for Language Image Pre-Training》，ICCV 2023  
   信息点：提出 SigLIP，用成对 sigmoid 损失替代 CLIP 的 softmax 对比损失，核心贡献是去除全局归一化依赖。

2. 《Vision Language Models: Smaller, Faster, Stronger》，PaLI-3，2023  
   信息点：展示 SigLIP 视觉编码器在更大视觉语言系统中的落地方式，说明其在 Stage0 预训练中的工程价值。

3. PaliGemma 技术文档与模型说明，2024  
   信息点：说明 PaliGemma 采用 SigLIP 系列视觉塔与 Gemma 解码器组合，体现该路线已进入实际模型栈。

4. Gemini 系列公开技术说明，2023-2024  
   信息点：展示 Google 多模态体系中，SigLIP 风格视觉编码器已成为重要基础组件之一。

5. CLIP: 《Learning Transferable Visual Models From Natural Language Supervision》，2021  
   信息点：作为对照基线，理解 softmax 对比学习、全局 batch 依赖和大规模图文对齐训练范式。
