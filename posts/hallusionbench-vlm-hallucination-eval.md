## 1) 一句话核心定义

HallusionBench 是一个专门测视觉语言模型幻觉的 yes/no 基准，核心不是测“会不会描述图”，而是测答案是否真的被图像上下文约束住。

公开版我核到的是 346 张图、1129 道题，不是 4K+；如果你看到 4K+，更像是后续派生整理版，不是官方原始公开版。

---

## 2) 面向新手的直观解释

它把同一个问题放到不同视觉上下文里，逼模型暴露自己到底是“看图答题”，还是“靠语言先验猜题”。

官方仓库把题分成 `VD` 和 `VS` 两类，再配 `Easy / Hard` 控制组。`VD` 要靠图像本身才能答，`VS` 则是文字可答、图像只是补充。GPT-4V 在官方榜单上的 `question-pair accuracy` 只有 31.42%，其余模型大多低于 16%，说明这个基准确实能打出模型短板。

---

## 3) 关键公式/机制

统一记号如下：

`q` = 问题，`I` = 图像，`y ∈ {0,1}` = yes/no 标注，`ŷ` = 模型预测。

简化后的模型可写成：

`ŷ = f(V_enc(I), L_enc(q))`

其中 `V_enc` 是视觉编码，`L_enc` 是语言编码。若 `L_enc` 过强，模型容易走语言先验，出现语言幻觉；若 `V_enc` 把视觉线索读歪，模型会被错误视觉证据带偏，表现为视觉幻觉。

HallusionBench 常用的三个指标可以写成：

`Acc_q = (1/|Q|) * Σ 1[ŷ_i = y_i]`

`Acc_fig = (1/|F|) * Σ 1[图 f 上的所有题都答对]`

`Acc_pair = (1/|P|) * Σ 1[ŷ(q,I_easy)=y_easy ∧ ŷ(q,I_hard)=y_hard]`

`Acc_pair` 最严格，因为它检查同一题在不同上下文下是否保持一致。

---

## 4) 一个最小数值例子

设有 2 个 question pair：

| Pair | Easy 真值 | Hard 真值 | 模型预测 |
| --- | --- | --- | --- |
| A | yes | no | yes / no |
| B | no | yes | no / no |

则：

`Acc_q = 3/4 = 75%`

`Acc_pair = 1/2 = 50%`

这说明只看单题正确率会高估模型。pair 指标专门抓“同题不同图是否自洽”。

---

## 5) 一个真实工程场景

做多模态客服或文档助手时，模型常要判断截图里有没有报错、表格是否被改过、图表结论是否成立。

这类场景最怕两种错：一是没看图就按文本模板回答，二是看图后被局部噪声带偏。HallusionBench 的控制组思路可以直接移植到工程评测里，把“原图 vs 编辑图”“有图 vs 无图”做成固定回归集。

---

## 6) 常见坑与规避

| 坑 | 现象 | 规避 |
| --- | --- | --- |
| 只看总准确率 | 看不出是否自相矛盾 | 同时看 `Acc_pair` 和 `Acc_fig` |
| 把开放式回答直接判分 | 格式噪声掩盖模型真实能力 | 统一抽取 yes/no 或选项字母 |
| 只调公开样本 | 容易过拟合题型模板 | 留出独立验证切片 |
| 混淆两类错误 | 纠偏方向搞反 | 先判定是语言幻觉还是视觉幻觉，再选对策 |
| 只做提示词工程 | 不能修复根因 | 结合负样本、编辑图、重加权训练 |

实用规则：语言幻觉优先补“图像约束”和“负样本”，视觉幻觉优先补“视觉编码质量”和“局部证据验证”。

---

## 7) 参考来源

1. [官方 GitHub 仓库](https://github.com/FuxiaoLiu/HallusionBench) - 数据说明、leaderboard、评测脚本。
2. [CVPR 2024 页面](https://cvpr.thecvf.com/virtual/2024/poster/29422) - 官方摘要、1129 题/346 图、31.42% question-pair accuracy。
3. [Hugging Face 数据集页](https://huggingface.co/datasets/lmms-lab/HallusionBench) - 公开数据集镜像与行数信息。
4. [HallusionBench 作者主页](https://tianruiguan.phd/publication/2023-10-24-hallusion) - 论文简介与公开版本说明。
5. [Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning](https://www.microsoft.com/en-us/research/publication/mitigating-hallucination-in-large-multi-modal-models-via-robust-instruction-tuning/) - 负样本/鲁棒指令微调，适合作为缓解策略参考。
