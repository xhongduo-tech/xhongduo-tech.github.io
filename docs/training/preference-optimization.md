---
title: 偏好数据与直接偏好优化
description: 从成对回答、序列概率和参考策略，理解DPO目标及质量边界。
kind: topic
status: planned
---

# 偏好数据与直接偏好优化

::: info 待研究专题
先验证成对数据与CPU目标，再按条件开展真实DPO。当前没有偏好标注、训练或胜率成绩。
:::

## 问题与前置依赖

chosen为什么优于rejected，依据是否可靠？目标如何利用策略与参考模型的回答概率？偏好判别更好，是否代表生成更正确？

学习依赖是[SFT](/training/sft)、response mask、序列logprob、sigmoid和固定协议评测。本路线选择可复现SFT检查点与独立测试作为起点；这不是DPO目标对所有策略初始化的必要限制。

本页选择离线DPO：从固定偏好对优化策略，不要求训练时在线rollout或单独训练奖励模型。DPO有其他变体和数据流程，不能将本页的离线设定推广为所有偏好学习。[原论文第4节](https://arxiv.org/html/2305.18290v3)给出这里采用的基础目标。

## 核心机制

只对回答位置求和：

$$
\ell_\pi(y\mid x)=\sum_t m_t\log\pi(y_t\mid x,y_{<t}).
$$

对同一prompt的优选回答`y+`与较差回答`y-`，标准sigmoid DPO为：

$$
\mathcal L_{\rm DPO}=-\mathbb E\log\sigma\!\left(\beta\left[(\ell_\theta^+-\ell_{\rm ref}^+)-(\ell_\theta^--\ell_{\rm ref}^-)\right]\right).
$$

`πref`为冻结参考策略，常以SFT模型初始化策略和参考；具体参考设置需记录。`β`是目标尺度及参考约束解释中的参数，不能只由loss数值判断其质量作用。这里使用回答token的logprob之和；换成长度平均会改变目标，必须显式说明。

prompt不直接计入completion目标，但仍影响条件概率。仅比较chosen/rejected的原始概率，或忘掉参考项，都不是这个DPO目标。[TRL文档](https://huggingface.co/docs/trl/en/dpo_trainer)同时提供其他损失形式，实验需要明确选用哪一种。

## reference怎样进入目标

基础DPO的推导先考虑固定prompt上的KL正则奖励最大化：

$$
\max_\pi\ \mathbb E_{y\sim\pi(\cdot|x)}r(x,y)
-\beta D_{\rm KL}(\pi(\cdot|x)\Vert\pi_{\rm ref}(\cdot|x)).
$$

在 $\beta>0$、参考策略支持集内、奖励固定且能表示最优分布的理想条件下，最优策略满足：

$$
\pi^*(y|x)=\frac{\pi_{\rm ref}(y|x)\exp(r(x,y)/\beta)}{Z(x)},
\quad
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\rm ref}(y|x)}+\beta\log Z(x).
$$

再采用Bradley–Terry形式`P(y^+\succ y^-|x)=σ(r(x,y^+)-r(x,y^-))`。同一prompt的`β log Z(x)`相减消失，便得到前面的DPO目标。它解释了reference项的来源，也说明为什么必须是同一个prompt下的回答对：换成两个不同prompt，不能用同一项抵消归一化常数。

用参数化`πθ`代入并拟合有限偏好数据，并不保证已经获得上述理想最优策略，也不能证明标注偏好是真实任务奖励。reference是冻结的比较基准，不是PPO中的行为策略；离线DPO也不靠它生成每一批在线rollout。

推导对应[原论文附录A.1–A.2](https://arxiv.org/html/2305.18290v3#A1)。这里还要求归一化项 $Z(x)$ 有限；理想分布表达与有限神经网络的实际拟合效果分别讨论。

### 一组可手算的logratio

设两个回答的**已正确mask和shift的序列logprob**为：

| 回答 | 当前策略ℓθ | reference ℓref | logratio ℓθ−ℓref |
| --- | --- | --- | --- |
| chosen | -3.0 | -2.0 | -1.0 |
| rejected | -4.0 | -2.5 | -1.5 |

取`β=2`，DPO logit为`z=2[-1-(-1.5)]=1`，损失为`softplus(-1)≈0.313262`。如果漏掉reference，就会错误地得到`z=2`与约`0.126928`的损失。两个回答的绝对概率都比reference低，仍可能具有正的DPO margin；不能把正margin解释成“chosen概率一定比基线升高”。

$$
\frac{\partial\mathcal L}{\partial\ell_\theta^+}
=-\beta\sigma(-z),\qquad
\frac{\partial\mathcal L}{\partial\ell_\theta^-}
=+\beta\sigma(-z).
$$

此例两项约为`-0.537883`与`+0.537883`。这是对序列logprob中间量的导数；实际模型的归一化概率及共享参数彼此耦合，不能把两项当成可独立调整的最终参数。

以下参考只验证目标层，不冒充真实模型训练：

```python
import torch
import torch.nn.functional as F

policy = torch.tensor([-3., -4.], dtype=torch.float64, requires_grad=True)
reference = torch.tensor([-2., -2.5], dtype=torch.float64)
logratio = policy - reference
z = 2.0 * (logratio[0] - logratio[1])
loss = F.softplus(-z)
loss.backward()
print(z, loss, policy.grad)
```

交换chosen/rejected后，`z=-1`、损失约`1.313262`。策略与reference完全一致时，`z=0`且损失为`log 2`，但策略的目标梯度通常并不为零。实际模型验证这条初始预期时还需关闭随机dropout、匹配模板与精度；允许声明范围内的数值误差，不能仅凭loss接近`log 2`判定整个数据链正确。

### 长度、EOS与截断为什么属于目标定义

将序列求和换成长度平均，不只是稳定数值。构造如下局部条件logprob：

| 回答 | 有效计分token数 | 每token当前/reference logprob | 求和后的logratio |
| --- | --- | --- | --- |
| chosen | 3 | -2.0 / -2.2 | 0.6 |
| rejected | 1 | -1.0 / -1.4 | 0.4 |

求和给出margin`0.2`；各自除以长度后变成`0.2-0.4=-0.2`，连方向都反转。这里仅构造计分token的概率，不对应实际文本。reference比较消除了部分基线差异，却不保证消除所有长度偏好；应按长度分组检查训练与独立生成结果，不能暗中改平均来“修复”原目标。

对真实数据逐项核对：prompt与回答的token边界是否一致；EOS是否计分；截断是否删除回答差异或结束标记；padding是否排除；策略和reference是否使用同一序列与mask。若截断后chosen/rejected只剩相同token，两者不再保留原来的偏好证据，应标记、修复或排除，而不是继续当作有效偏好。

拼接prompt和回答后分词，不一定等于分别分词再拼接。模板边界和tokenizer规则必须经过实际ID核对；缓存reference logprob时一并绑定input IDs、completion mask、截断规则和reference revision。更改其中任何一项后重新计算，不能只靠原始文本哈希判断缓存有效。

## 偏好对需要保留什么证据

以下为合成教学对，不是实际标注成果：

```json
{"prompt":[{"role":"user","content":"2+2等于多少？"}],"chosen":[{"role":"assistant","content":"4。"}],"rejected":[{"role":"assistant","content":"5。"}]}
```

另记录样本ID、来源/生成模型、偏好维度、标注者或验证规则、置信与分歧、长度、模板版本和split。按prompt族/来源分组切分，避免同一问题轻微改写跨训练测试。

偏好可以来自人、规则或模型判断，可靠性与适用范围不同。不要让“更长、格式漂亮、迎合”在未声明的情况下替代正确性；保留难判断、平局和分歧，不强行制造明确偏好。

## 实现与源码阅读切入

固定TRL提交`f27a906d39cc60aeea576e616f9c829dc4a6ce44`：从[DPO loss与completion logprob](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/dpo_trainer.py#L1377)追踪模板、padding、causal shift、mask、策略/参考前向和损失分支。

配置中明确基础sigmoid目标、参考模型或reference adapter、β、截断与是否预计算参考logprob。缓存必须绑定数据、模板和参考revision；LoRA只是参数化方式，不替代这些语义检查。

| 本页中间量 | 固定源码位置 | 重点核对 |
| --- | --- | --- |
| 回答位置与序列概率和 | [labels、shift mask与logps](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/dpo_trainer.py#L1377) | completion_mask怎样形成labels；被忽略位置怎样从模型返回的logprob排除 |
| 冻结reference | [reference前向或adapter切换](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/dpo_trainer.py#L1399) | 预计算缓存、独立模型、禁用adapter或固定ref adapter不是同一配置 |
| 两个logratio | [标准reverse-KL分支](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/dpo_trainer.py#L1430) | chosen/rejected拆分顺序、reference相减与delta_score |
| 最终目标 | [sigmoid loss分支](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/dpo_trainer.py#L1464) | β、logsigmoid、各样本聚合和额外目标 |

本文基础例子对应单一sigmoid、`reverse_kl`与`ld_alpha=None`的序列求和目标；其他散度、长度处理和混合损失不能直接用同一组期望值验收。表中入口也未覆盖分词器和模型内部的完整执行路径，实际运行仍需继续追踪。

## 分级实验

| 层级 | 拟执行实验 | 对照 |
| --- | --- | --- |
| CPU概率 | tiny logits计算回答序列logprob，加入padding和被mask的prompt位置 | 检查mask、shift及padding不变性，直接比较逐token与求和结果 |
| CPU目标 | 令策略初始等于参考，计算loss，再交换chosen/rejected并做一次小更新 | 标准目标初始loss应为log 2；不据此声称训练了LLM或提升偏好 |
| CPU数据审计 | 故意构造长度偏置、重复prompt和错误chosen | 检查标注规则及能否发现反例 |
| 可选GPU | 同一SFT起点、固定偏好数据与有界预算，完成一次DPO | 与未优化SFT在同一生成/评测协议下比较；保存参考配置、成本与退化 |

## 指标与验收产物

交付数据card、偏好规则和分歧记录、CPU目标校验、模型/参考manifest、原始生成、评测与检查点。除reward margin或偏好判别正确率外，还检查独立任务正确性、格式、长度和能力退化。

人工或模型成对评测应盲化身份并交换位置；记录样本量与不确定性。判别数据上的高accuracy不能代替真实生成测试，也不能把标注模型的偏好直接当成人类总体偏好。

深入验收按因果链核对：从模板还原有效回答token→复算当前与reference的逐token概率→得到序列和与margin→检验损失及梯度符号→做一次受控更新→用独立生成检查收益与退化。至少故意制造一次错reference、错mask或截断丢失偏好的故障，并解释为何某些训练指标仍可能看起来正常。

## 常见陷阱

chosen/rejected反置；把prompt复制进completion或截断后仅剩相同前缀；序列概率偷偷改成平均值；参考缓存未随模板更新；用同一批偏好反复调参并宣称泛化；只看训练loss或长度增长。完成DPO不等于完成PPO/GRPO，也不等于拥有在线RL流水线。

## 第一方资料

核验日期：2026-11-16。

本轮目标推导与固定TRL分支复核：2026-11-17。所有数值算例均为解析预期，没有执行真实偏好训练。

- [DPO原论文 v3](https://arxiv.org/html/2305.18290v3)：目标、参考策略及其推导。
- [TRL DPO Trainer](https://huggingface.co/docs/trl/en/dpo_trainer)：数据格式、损失配置与实现边界。
- [固定DPO源码](https://github.com/huggingface/trl/blob/f27a906d39cc60aeea576e616f9c829dc4a6ce44/trl/trainer/dpo_trainer.py#L1468)：对应基础sigmoid分支；实际运行另锁依赖。

关联：[训练地图](/training/) · [后训练项目](/projects/post-training-lab) · [RL训练](/training/rl-training) · [评测](/evaluation/)。
