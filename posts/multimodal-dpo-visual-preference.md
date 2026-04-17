## 核心结论

多模态 DPO 的核心价值，不是“再训一次模型”，而是把“同一张图下，哪句描述更贴图”直接变成训练信号。这里的“贴图”也叫 `grounded`，白话就是“话里的对象、属性、关系，真的能在图里找到”。当偏好对专门围绕幻觉构造时，模型会学到一件很具体的事：在相同图像条件下，提高正确描述的相对概率，压低错误描述的相对概率。

V-DPO 说明，视觉偏好数据里最有效的一类，往往不是泛化的“回答更自然”，而是“同图像下正确描述 vs 幻觉描述”的对比。RLHF-V 进一步说明，哪怕只做人类对 hallucination 片段的细粒度修正，也能用 Dense DPO 明显降低幻觉率。OPA-DPO 则补上了工程上的关键点：如果偏好数据离参考策略太远，训练会变钝，必须先把数据分布拉回到 on-policy，再做 DPO 更稳定。

一个玩具例子足够说明问题。图里是“男孩在吃比萨”，正样本写“男孩正吃一片比萨”，负样本写“他在吃一盘虫子”。DPO 不需要显式写一个“虫子错误”的规则，它只需要在同一图像条件下反复看到这类 chosen/rejected 对，就会把“比萨”方向的概率往上推，把“虫子”方向的概率往下推。

下表只汇总公开摘要中能稳定确认的收益，不混入无法直接核实的细项指标：

| 方法 | 数据规模 | 可稳定确认的收益 | 说明 |
|---|---:|---|---|
| Base MLLM | 0 | 作为对照 | 通常更依赖语言先验 |
| RLHF-V | 1.4k 人工细粒度修正 | 相对 base 幻觉率下降 34.8% | 重点是段级纠错而不是整句重写 |
| OPA-DPO | 4.8k on-policy 样本 | 相对上一代 SOTA，AMBER 再降 13.26%，Object-Hal 再降 5.39% | 重点是先做 on-policy alignment |
| TPR | 依任务而定 | 多基准平均优于先前方法约 20%，ObjectHal 最多降 93% | 重点是系统化拉大奖励间隔 |

---

## 问题定义与边界

这里说的 `hallucination`，白话就是“模型说了图里没有的东西，或者把图里已有的东西说错了”。它不是单一错误，而是至少有四类：

| 维度 | 典型错误 | 标注时看什么 | 推荐偏好对构造 |
|---|---|---|---|
| 对象 | 把猫说成狗 | 图中是否存在该实体 | 正确对象 vs 错误对象 |
| 属性 | 红车说成蓝车 | 颜色、数量、材质是否一致 | 正确属性 vs 错误属性 |
| 关系 | 人在骑车说成人推车 | 主体间动作或空间关系 | 正确关系 vs 错误关系 |
| 上下文 | 室内说成街道 | 场景、用途、时间线 | 正确场景 vs 错误场景 |

边界也要说清。多模态 DPO 主要解决“输出偏好”问题，不直接解决视觉编码器看不清的问题。图像本身模糊、OCR 失败、医学图像超出基座模型分布、标注者本身判断失误，这些都不是 DPO 单独能修好的。它做的是：在模型“看得到但容易乱说”的区域里，减少语言先验压过视觉证据的情况。

再看一个基础例子。给模型一张猫的照片，它输出“画面里有一只狗”。这就是对象级幻觉。把“画面里有一只猫”设为 chosen，把“画面里有一只狗”设为 rejected，本质上是在同一个输入 $x$ 上学习一个更强的相对约束，而不是学习两个孤立句子。

真实工程里，这个边界更重要。比如医学影像摘要、保险理赔核验、法务证据图像说明，容错率都很低。此时团队不该优先收集“更流畅”的答案，而应优先收集“更 grounded”的答案，因为真正的风险来自虚构对象、虚构属性和错误关系。

---

## 核心机制与推导

DPO 的目标可以写成：

$$
\mathcal{L}_{\rm DPO}
=
-\mathbb{E}_{(x,y^+,y^-)}
\log \sigma \Big(
\beta[
\log \frac{\pi_\theta(y^+|x)}{\pi_{\rm ref}(y^+|x)}
-
\log \frac{\pi_\theta(y^-|x)}{\pi_{\rm ref}(y^-|x)}
]
\Big)
$$

这里的 `参考策略` $\pi_{\rm ref}$，白话就是“训练前的旧模型，用来限制新模型不要偏航太远”。$\beta$ 控制约束强度。$\beta$ 太小，学习信号弱；$\beta$ 太大，模型容易过激更新。

把“男孩吃比萨”例子代进去。设 $\beta=0.5$，参考策略对正确描述和错误描述的概率分别是 $0.4$ 和 $0.1$，当前模型是 $0.35$ 和 $0.25$，则内部间隔为：

$$
0.5\Big[\log(0.35/0.4)-\log(0.25/0.1)\Big]
\approx -0.525
$$

因为这个值是负的，$\sigma(\cdot)$ 会偏小，损失变大，梯度就会推动模型提高 $y^+$ 的相对概率，同时压低 $y^-$。直观上，DPO 学的不是“哪句绝对正确”，而是“在同一张图下，哪句应该比哪句更可能”。

真正决定效果的，往往不是公式，而是 `reward gap`，白话就是“正负样本之间到底差多大”。如果正样本只是稍微比负样本好一点，或者两句都半对半错，sigmoid 大量停在 0.5 附近，梯度就弱。TPR 的价值正是在数据构造阶段系统化拉开这个间隔。

还要注意 `off-policy`。白话就是“你拿来训练的好答案，和参考模型平时会说的话差太远”。这时参考模型对 chosen 句子的概率可能接近 0，KL 约束会让有效梯度变小。OPA-DPO 先做 on-policy alignment，就是先把“专家修正后的答案”往模型原本分布附近拉一拉，再做 DPO。

可以把迁移过程理解成一条简单链路：

`同图像正负对` -> `计算相对 log-odds` -> `提升 grounded 描述` -> `压低幻觉描述` -> `输出从语言主导转向图像对齐`

---

## 代码实现

下面是一个可运行的最小实现，只保留 DPO 的核心数学。它不是训练完整 VLM，而是演示“正样本概率上升、负样本概率下降”时，损失如何变化。

```python
import math

def dpo_loss(pi_plus, pi_minus, ref_plus, ref_minus, beta=0.5):
    margin = beta * (
        math.log(pi_plus / ref_plus) - math.log(pi_minus / ref_minus)
    )
    return -math.log(1 / (1 + math.exp(-margin)))

# 玩具例子：男孩吃比萨 vs 男孩吃虫子
loss_before = dpo_loss(
    pi_plus=0.35, pi_minus=0.25,
    ref_plus=0.40, ref_minus=0.10,
    beta=0.5
)

# 假设训练后，模型更偏向 grounded 描述
loss_after = dpo_loss(
    pi_plus=0.50, pi_minus=0.12,
    ref_plus=0.40, ref_minus=0.10,
    beta=0.5
)

assert loss_after < loss_before
print(round(loss_before, 4), round(loss_after, 4))
```

在真实训练里，数据结构通常是 `(image, prompt, chosen, rejected)`。如果做 RLHF-V 风格的 Dense DPO，则会再细一层，变成“整段回答中的多个纠错片段”，每个片段各自产生偏好信号。一个简化训练循环如下：

```python
for image, prompt, y_plus, y_minus in pref_data:
    logp_plus = model.log_prob(image, prompt, y_plus)
    logp_minus = model.log_prob(image, prompt, y_minus)
    ref_plus = ref_model.log_prob(image, prompt, y_plus)
    ref_minus = ref_model.log_prob(image, prompt, y_minus)

    margin = beta * ((logp_plus - ref_plus) - (logp_minus - ref_minus))
    loss = -log_sigmoid(margin).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

真实工程例子可以这样落地。做医学影像报告摘要时，先让基座模型生成初稿，再让标注者只修“虚构病灶、错误侧别、错误数量”这些片段，不改无关措辞。然后把“修正后片段”作为 chosen，“原始片段”作为 rejected。这样数据贵，但非常高效，因为每一条都直击风险点。

---

## 工程权衡与常见坑

最常见的坑不是模型太小，而是偏好数据不对。

| 常见坑 | 现象 | 根因 | 缓解策略 |
|---|---|---|---|
| Off-policy 过重 | 训练几乎不降，或效果不稳定 | 专家答案离参考策略太远 | 先做 OPA，对齐到 on-policy |
| Reward gap 太小 | loss 有变化，指标不明显 | 正负样本差异不够尖锐 | 构造更干净的对比对，剔除模糊样本 |
| 自动评审偏长文本 | chosen 常常更长但不一定更准 | Judge 有长度或宽容偏差 | 多轮采样投票，人工抽检 |
| 参考策略选错 | 训练后输出变保守或跑偏 | ref 过旧或与数据不匹配 | 用冻结的近邻版本做 ref |
| 只追求低幻觉率 | 模型变得过于保守 | 学会“少说”而不是“说对” | 同时监控覆盖率与任务完成率 |

GPT-4 类评审器的偏差要特别小心。现有研究能稳定支持的一点是：在“打分而非排序”任务上，GPT-4 对较长回答存在偏好，且可能更宽容。这放到视觉偏好数据里，就可能把“更长但掺了猜测”的文本误选为 chosen。对新手团队，最稳妥的方法不是完全禁用自动评审，而是用它做第一轮筛选，再对高风险样本做人审复核。

还有一个常见误解：以为只要把 rejected 写得越离谱越好。其实不对。若负样本总是“虫子”“外星人”这种极端错误，模型学到的只是避免极端胡说；但真实线上错误往往更细，比如“左肺下叶”说成“右肺下叶”，“两个杯子”说成“三个杯子”。高价值数据通常来自“近错样本”，也就是与真实答案很像、但关键视觉细节错了的负例。

---

## 替代方案与适用边界

DPO 不是唯一方案，但它在“有偏好对、想低成本稳定训练”时非常合适。若团队的数据和目标不同，可以考虑下面三类路线：

| 方案 | 数据需求 | 优点 | 风险边界 | 更适合什么场景 |
|---|---|---|---|---|
| 人工偏好 + DPO | 中等，人工最贵 | 标签最准，风险可控 | 扩展慢 | 医疗、法务、金融合规 |
| TPR / reward gap rewriting | 中等，重写质量关键 | 能系统化拉大奖励间隔 | 重写器错了会放大噪声 | 需要精细控制对象/属性/关系 |
| RLAIF-V / AI feedback | 低到中等 | 成本低，扩展快 | 评审器偏差会传染 | 先做大规模原型与数据冷启动 |

TPR 的思路很适合解释给初学者。假设原句里有“桌上放着一个杯子”，你可以把某个 `topic`，也就是“一个局部语义块”，替换成“杯中冒着热气的咖啡”作为正样本，再替换成“杯中装着果汁”作为负样本。这样 reward gap 比只改一个模糊形容词更清楚。

RLAIF-V 的适用边界也很明确。它适合没有足够人工预算，但又希望快速构建视觉偏好对的团队。问题在于，AI 反馈器本身如果视觉 grounding 不稳，错误会被批量写进数据集。所以它更像“扩大数据量的引擎”，不是“免审的真理机器”。

如果只用一句话概括选择标准：高风险领域优先人工细粒度修正，追求数据效率时优先 on-policy DPO，追求更强控制力时考虑 TPR，预算有限且需要快速扩张时考虑 RLAIF-V。

---

## 参考资料

- Yuxi Xie, Guanzhen Li, Xiao Xu, Min-Yen Kan. 2024. [V-DPO: Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization](https://aclanthology.org/2024.findings-emnlp.775/).
- Tianyu Yu et al. 2023/2024. [RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback](https://huggingface.co/papers/2312.00849).
- Zhihe Yang et al. 2025. [Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key](https://www.microsoft.com/en-us/research/publication/mitigating-hallucinations-in-large-vision-language-models-via-dpo-on-policy-data-hold-the-key/).
- Microsoft Research Blog. 2025-10-26. [OPA-DPO: Efficiently minimizing hallucinations in large vision-language models](https://www.microsoft.com/en-us/research/articles/opa-dpo-efficiently-minimizing-hallucinations-in-large-vision-language-models/).
- Lehan He et al. 2025. [Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](https://openreview.net/forum?id=fJRuMulPkc).
- Project page. 2025. [TPR: Systematic Reward Gap Optimization](https://tpr-dpo.github.io/).
- Tianyu Yu et al. 2024. [RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness](https://huggingface.co/papers/2405.17220).
- Abdullah Al Zubaer et al. 2025-10-08. [GPT-4 shows comparable performance to human examiners in ranking open-text answers](https://www.nature.com/articles/s41598-025-21572-8).
