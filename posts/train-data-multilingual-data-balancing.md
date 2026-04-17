## 核心结论

多语言数据平衡的目标，不是让每种语言“看起来一样多”，而是让单个模型在多种语言上的验证集表现尽量接近。这里的“平衡”本质上是一个训练分布控制问题：训练时每个语言被抽到的频率，会决定它对参数更新的影响强弱。

直观地说，高资源语言就是“样本更多、梯度更响”的语言，梯度可以理解为训练时推动模型参数变化的信号。如果直接按原始数据量采样，高资源语言会长期主导更新方向，低资源语言即使存在，也容易被冲掉。于是训练目标虽然表面上是“多语言”，实际却更像“高资源语言优先，其他语言陪跑”。

常见的第一步做法是温度采样，也就是把原始语料比例 $q_i$ 做一次幂次变换：

$$
P_D(i)=\frac{q_i^{1/\tau}}{\sum_k q_k^{1/\tau}}
$$

其中 $\tau$ 控制平衡强度。这个办法简单、便宜、常常有效，但它的问题也很直接：$\tau$ 是全局固定超参数，不知道当前哪个语言真正更缺训练，也不知道哪个语言继续加样本已经收益很低。

因此，更稳的路线是把语言采样分布写成可学习形式：

$$
P_D(i;\psi)=\frac{e^{\psi_i}}{\sum_k e^{\psi_k}}
$$

这里的 $\psi_i$ 是语言级 scorer，可以理解为“给语言 $i$ 的可学习打分”。这样采样不再由静态语料规模决定，而是由验证集损失反向驱动。再往前一步，像 CLIMB 这类方法会显式建模“在给定 token 预算下，每个语言再多训练一点到底值不值”，本质是在估计边际收益。

一个新手版玩具例子：把多语言训练想成一场团队投票。英语有 1000 票，印地语有 50 票，直接多数决时，结果几乎永远偏向英语。数据平衡做的事，就是要么提高少数语言每一票的权重，要么根据最近的验证结果，自动决定谁下一轮该多投几票。

固定温度和可学习 scorer 的差异可以先看成下面这张表：

| 方法 | 核心思想 | 优点 | 局限 |
|---|---|---|---|
| 固定温度采样 | 用 $q_i^{1/\tau}$ 软化原始语料占比 | 实现简单，训练开销低 | 不能根据 dev 表现自适应调整 |
| 可学习 scorer | 用 $\psi_i$ 直接学习语言采样概率 | 能针对失衡语言动态补偿 | 需要额外验证与双层优化 |
| 边际收益建模 | 估计单位 token 对 dev loss 的收益 | 更接近预算最优分配 | 系统复杂，估计误差会放大 |

---

## 问题定义与边界

多语言数据平衡的定义可以写得很直接：在共享参数的多语言模型里，通过调整不同语言训练样本的权重或采样概率，使模型在多语言开发集上的总体目标最小。这里的开发集，简称 dev 集，就是训练过程中用来评估效果、但不参与参数拟合的小型验证数据。

它通常作用在两类系统里：

| 场景 | 是否常见 | 说明 |
|---|---|---|
| 多语言 NMT | 是 | NMT 是神经机器翻译，指端到端翻译模型 |
| 多语言 LLM 预训练/继续训练 | 是 | 多语言文本比例直接影响跨语言泛化 |
| 单语言模型 | 否 | 不存在语言间采样竞争 |
| 语言规模本来就接近的混合训练 | 有时不需要 | 简单采样可能已足够 |

什么时候必须关心它？当语言间数据量差异非常大，或者不同语言 dev loss 差距已经明显拉开时，就必须处理。什么时候可以先不做复杂平衡？当语言规模接近、任务相似、验证表现差异小，先用简单采样往往更划算。

一个最小数值例子能直接看出问题。假设语言 A 有 100 万句，语言 B 有 10 万句，则原始占比大约是：

$$
q_A=0.91,\quad q_B=0.09
$$

如果取 $\tau=0.5$，则

$$
q_A^{1/\tau}=0.91^2\approx0.83,\quad q_B^{1/\tau}=0.09^2\approx0.008
$$

归一化以后：

$$
P_D(A)\approx \frac{0.83}{0.838}\approx0.99,\quad
P_D(B)\approx \frac{0.008}{0.838}\approx0.01
$$

这说明一个容易被忽略的事实：温度采样并不天然“照顾低资源”。如果参数取值不合适，它甚至会让低资源语言更难被采到。

新手版理解可以更直白一些：A 语料 100 万句，B 语料 10 万句，训练 loader 每次抽 batch 时，大多数时候都能抽到 A。只要采样策略不改，B 在整个训练过程中就像总被插队。

是否需要数据平衡，可以先用下面的判断表：

| 情况 | 是否建议做数据平衡 | 原因 |
|---|---|---|
| 高资源/低资源相差 10 倍以上 | 建议 | 高资源会主导梯度 |
| 多语言 dev loss 差距持续扩大 | 建议 | 说明训练分布与目标不一致 |
| 各语言数据规模接近 | 可选 | 固定采样可能已足够 |
| 只关心高资源语言表现 | 不一定 | 平衡可能反而牺牲主语言峰值 |
| 训练预算极紧 | 先从简单温度开始 | scorer 方法额外开销更高 |

问题边界也要说清楚。数据平衡不是万能补药。它解决的是“训练看到什么、看到多少”的问题，不直接解决模型容量不足、tokenizer 质量差、语种脚本差异过大、标注噪声不一致等问题。如果这些基础条件很差，单独改采样分布通常救不回来。

---

## 核心机制与推导

多语言数据平衡通常分三层，从简单到复杂依次升级。

第一层是温度采样。它的目的，是把极端倾斜的语料分布做软化。若 $q_i$ 是语言 $i$ 的原始语料占比，那么：

$$
P_D(i)=\frac{q_i^{1/\tau}}{\sum_k q_k^{1/\tau}}
$$

当 $\tau>1$ 时，分布会更平，低资源语言更容易被采到；当 $\tau<1$ 时，分布更尖，高资源语言更占优。它的优点是没有额外学习参数，缺点是所有语言共享同一个调节旋钮。

第二层是可学习 scorer。核心思想是：不要猜哪种语言该多采，直接让验证集来决定。把每个语言对应一个参数 $\psi_i$，用 softmax 得到采样分布：

$$
P_D(i;\psi)=\frac{e^{\psi_i}}{\sum_k e^{\psi_k}}
$$

softmax 可以理解为“把任意实数打分变成总和为 1 的概率分布”。如果某个语言的 $\psi_i$ 上升，它被抽中的概率就会上升。

MultiDDS 一类方法的关键是双层优化。双层优化就是“内层先训练模型，外层再根据 dev 表现更新采样策略”。可写成：

$$
\theta^*(\psi)=\arg\min_{\theta}\mathcal{L}_{train}(\theta,\psi)
$$

$$
\min_{\psi}\ \mathcal{L}_{dev}(\theta^*(\psi))
$$

其中 $\theta$ 是模型参数，$\psi$ 是采样分布参数。内层优化回答“给定当前采样策略，模型该怎么训”；外层优化回答“什么采样策略能让 dev loss 更低”。

如果只看一小步更新，可写成：

$$
\theta'=\theta-\eta_\theta \nabla_\theta \mathcal{L}_{train}(\theta,\psi)
$$

然后用 dev loss 更新 $\psi$：

$$
\psi'=\psi-\eta_\psi \nabla_\psi \mathcal{L}_{dev}(\theta')
$$

这里的关键不在公式形式，而在依赖关系：$\psi$ 不是直接拟合训练损失，而是通过“影响训练样本分布”间接影响 dev loss。这就是它比固定温度更强的原因。

一个玩具例子：只有英语和西班牙语两种语言。当前 dev 上英语 loss 已经很低，西语 loss 明显偏高。固定温度方法仍按静态比例采样；scorer 方法则会逐步提高西语的 $\psi_{es}$，让接下来更多 batch 来自西语，直到 dev 上两者差距收敛。

真实工程例子：训练一个 1.2B 多语言 LLM，包含英语、中文、阿拉伯语、印地语、印尼语等语种。若直接按爬取数据量混合，英语与高资源语言会占据绝大多数 token。CLIMB 这类方法进一步考虑：在总 token 预算固定时，某语言再增加 1 个单位训练量，对整体 dev loss 的下降是否仍然显著。如果某高资源语言已经接近饱和，那么继续加它的样本，收益就低于补充低资源语言。

三类机制的流程对比如下：

| 方法 | 依据什么决定采样 | 更新频率 | 适合什么阶段 |
|---|---|---|---|
| 温度采样 | 原始语料比例 $q_i$ | 训练前固定 | 基线系统、预算紧张 |
| scorer 方法 | 语言参数 $\psi_i$ + dev loss | 周期性更新 | 需要更稳的跨语言均衡 |
| CLIMB 类方法 | 边际收益与 token 预算 | 分阶段估计 | 大规模预训练预算优化 |

---

## 代码实现

工程上最小可用结构通常分成三个模块：

| 模块 | 职责 | 关键输入 |
|---|---|---|
| Sampler | 按当前概率采语言/采样本 | $P_D(i)$ 或 $P_D(i;\psi)$ |
| Scorer | 维护 $\psi$ 并 softmax 成概率 | dev loss、学习率 |
| DevEval | 周期评估各语言 dev loss | 当前模型参数 |

下面先给一个可运行的 Python 玩具实现，只演示采样分布与简单更新逻辑：

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def update_psi(psi, dev_losses, lr=0.1):
    # loss 高的语言提高权重；这里是玩具规则，不是严格论文梯度
    mean_loss = sum(dev_losses) / len(dev_losses)
    new_psi = []
    for p, loss in zip(psi, dev_losses):
        new_psi.append(p + lr * (loss - mean_loss))
    return new_psi

psi = [0.0, 0.0]                 # 英语, 低资源语
probs = softmax(psi)
assert abs(sum(probs) - 1.0) < 1e-9
assert probs[0] == probs[1] == 0.5

dev_losses = [1.2, 2.0]          # 低资源语更差
psi = update_psi(psi, dev_losses, lr=0.5)
probs = softmax(psi)

assert probs[1] > probs[0]       # 后续应更多采低资源语
print(probs)
```

上面这个例子只是帮助理解：如果某语言 dev loss 更高，就提高它后续被采样的概率。真实系统里，$\psi$ 的更新不是手写规则，而是通过自动求导近似双层优化。

下面是更接近训练循环的伪代码：

```python
# pseudo code
psi = init_language_scores(num_langs)   # learnable
theta = init_model()

for step in range(train_steps):
    probs = softmax(psi)

    lang_id = sample_language(probs)
    batch = next_batch(lang_id)
    train_loss = model_forward(theta, batch)
    theta = theta - lr_theta * grad(train_loss, theta)

    if step % eval_interval == 0:
        dev_losses = []
        for lid in range(num_langs):
            dev_losses.append(eval_dev_loss(theta, lid))

        # bilevel approximation:
        # psi 梯度来自 dev loss 对采样分布的敏感性
        psi_grad = approx_meta_grad(theta, psi, dev_losses)
        psi = psi - lr_psi * psi_grad
```

这里有三个实现要点。

第一，采样器必须支持热更新。热更新就是训练过程中不重启 dataloader，也能把新概率立即生效。否则 dev 已经发现某语言表现差，但训练还在沿用旧分布。

第二，dev 评估不能只看整体平均值。平均值会掩盖语言尾部风险。工程里至少要保留 per-language dev loss，否则 scorer 没法知道该补哪一侧。

第三，更新频率不能太高。每个 step 都做全量 dev 评估通常成本过大。实际里常按固定 interval、滑动窗口，或只抽一部分代表性 dev 样本。

如果把 $\psi$ 更新写成更抽象的形式，可以记为：

$$
\psi \leftarrow \psi - \eta_\psi \frac{\partial \mathcal{L}_{dev}}{\partial \psi}
$$

而由于 $\mathcal{L}_{dev}$ 并不直接依赖 $\psi$，实际链式关系是：

$$
\frac{\partial \mathcal{L}_{dev}}{\partial \psi}
=
\frac{\partial \mathcal{L}_{dev}}{\partial \theta'}
\cdot
\frac{\partial \theta'}{\partial \psi}
$$

这正是双层优化比普通 loss weighting 更复杂的地方。

---

## 工程权衡与常见坑

第一个常见坑，是把固定温度当成长期方案。它适合做基线，不适合自动应对所有语言状态变化。某语言在训练早期缺数据，后期可能已经饱和；另一语言可能因 tokenizer 不友好，始终学习更慢。固定 $\tau$ 无法反映这些动态。

第二个常见坑，是 tokenizer 偏置。tokenizer 可以理解为把文本切成模型可处理 token 的规则系统。如果 tokenizer 训练只看高资源语言，低资源或非 Latin 脚本语言会被切得更碎。切得更碎意味着序列更长、训练更慢、同样 token 预算下有效信息更少。

假设某语言原本平均一句 100 个字符，平衡 tokenizer 时变成 40 个 token；若 tokenizer 严重偏向高资源语言，可能变成 67 个 token，则长度增幅为：

$$
\frac{67-40}{40}=67.5\%
$$

这就是“子词繁殖率”上升的直接成本。它不只影响训练效率，也会影响公平性，因为同一句内容在不同语言上占用的训练预算完全不同。

第三个坑，是只平衡采样，不平衡验证。若 dev 集本身严重偏向高资源语言，那么即便用了 scorer，优化方向仍可能偏向头部语言。训练分布和评估分布都要检查。

第四个坑，是高频更新 scorer 导致训练震荡。若每次 dev 波动都立即大幅调高某语言采样率，训练会在不同语言之间来回摆动，最后谁都学不稳。

常见问题与规避策略如下：

| 常见坑 | 表现 | 规避策略 |
|---|---|---|
| 固定 $\tau$ 长期不变 | 低资源语言长期不收敛 | 用 dev 驱动周期调参或 scorer |
| tokenizer 只看高资源语料 | 低资源语言 token 长度暴涨 | tokenizer 训练前先做平衡采样 |
| 只看整体 dev loss | 尾部语言退化被掩盖 | 保留 per-language 指标 |
| scorer 更新过快 | 采样分布来回震荡 | 降低更新频率、做平滑 |
| 低资源语料质量差 | 加权后反而拖累整体 | 先做去重、清洗、质量过滤 |

下面给一个 tokenizer 训练前的平衡采样示意代码：

```python
import random

def rebalance_for_tokenizer(lang_to_sentences, target_per_lang):
    balanced = []
    for lang, sents in lang_to_sentences.items():
        if len(sents) >= target_per_lang:
            balanced.extend(random.sample(sents, target_per_lang))
        else:
            # 数据少就重复采样，保证 tokenizer 不被头部语言垄断
            picks = [random.choice(sents) for _ in range(target_per_lang)]
            balanced.extend(picks)
    return balanced

toy = {
    "en": ["hello world"] * 100,
    "hi": ["नमस्ते दुनिया"] * 10,
}
balanced = rebalance_for_tokenizer(toy, target_per_lang=20)
assert len(balanced) == 40
assert sum(1 for x in balanced if x == "hello world") == 20
```

真实工程里，这一步通常发生在 tokenizer 预处理阶段，而不是主训练阶段。原因很简单：如果切词器一开始就带偏，后面再怎么平衡训练采样，也是在坏 token 边界上补课。

---

## 替代方案与适用边界

除了可学习 scorer，还有几条常见替代路线。

第一类是 dev-driven 的 $\tau$ 搜索。做法是先在小规模实验上扫几组 $\tau$，选择 dev 最优值。这比拍脑袋设温度强，但本质仍是静态超参数搜索。

第二类是 loss weighting。它不是改变采样概率，而是在 loss 聚合时给不同语言不同系数，例如：

$$
w_i = \frac{1}{\sqrt{N_i}}
$$

其中 $N_i$ 是语言 $i$ 的样本量。这样做的含义是：batch 还是照常采，但低资源语言算损失时权重更大。它和采样加权的区别是：

$$
\text{sampling weighting: } P_D(i)\uparrow
$$

$$
\text{loss weighting: } \mathcal{L}=\sum_i w_i \mathcal{L}_i
$$

前者改变“看见谁的频率”，后者改变“看见后算多重”。

第三类是 language-specific adapter。adapter 可以理解为插在主模型里的小模块，让不同语言有局部专用参数。它减轻了共享参数竞争，但增加了系统复杂度与部署成本。

第四类是后处理重加权，例如继续训练某些弱势语言，或者做语言特定微调。这适合预算有限但有明确重点语言的团队。

替代方法对比如下：

| 方法 | 推荐场景 | 主要缺点 |
|---|---|---|
| 固定温度采样 | 快速基线、训练预算紧 | 对动态失衡反应差 |
| dev-driven $\tau$ 搜索 | 语言数不多、实验可控 | 仍需反复试参 |
| 可学习 scorer | 追求跨语言均衡最优 | 实现复杂、验证成本高 |
| loss weighting | dataloader 不方便改 | 不一定解决采样覆盖问题 |
| language-specific adapter | 语言差异大、容量足 | 参数与部署更复杂 |
| 后处理重加权/继续训练 | 有重点语言、增量迭代 | 全局最优性较弱 |

适用边界也要明确。若语言数据本来接近、dev loss 也接近，可学习 scorer 带来的额外复杂度未必值得。若系统对训练吞吐和实现简单性要求极高，往往先上“温度采样 + per-language dev 监控”更务实。若是超大模型预训练，token 预算动辄数万亿，边际收益建模才更可能带来显著收益。

对小团队来说，一个现实决策流程通常是：

1. 先看各语言数据规模与 dev loss。
2. 如果失衡轻，先用固定 $\tau$。
3. 如果低资源持续掉队，再引入 scorer 或 loss weighting。
4. 如果 tokenizer 已经明显偏置，优先修 tokenizer 采样，而不是只调主训练。

---

## 参考资料

| 资料 | 关键贡献 | 建议查看 |
|---|---|---|
| Wang et al. 2020, *Balancing Training for Multilingual Neural Machine Translation* | 提出基于 dev loss 的 MultiDDS 思路，将语言采样写成可学习 scorer 并用双层优化更新 | 采样分布定义、bi-level 优化部分 |
| Guo et al. 2025, *CLIMB* | 建模 token 预算下各语言对验证损失的边际收益，用于大规模多语言 LLM 数据配比 | 数据分布推导、预算约束实验 |
| Selvamurugan et al. 2025, *From Bias to Balance* | 从偏置与公平性角度系统讨论多语言数据分布问题 | 现象总结、失败模式分析 |

- Wang et al. 2020：核心价值是把“拍超参数”改成“让 dev loss 反向决定采样权重”。
- Guo et al. 2025：核心价值是把“多采谁”升级成“预算固定时谁的边际收益更高”。
- Selvamurugan et al. 2025：核心价值是从更系统的角度解释多语言训练为什么会天然偏向高资源语言。

- ACL Anthology: https://aclanthology.org/2020.acl-main.754.pdf
- OpenReview: https://openreview.net/pdf/e9de76524170f56669841fa05c57f5a383c03730.pdf
