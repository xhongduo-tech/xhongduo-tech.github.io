## Activation Patching：激活修补定位关键组件

## 核心结论
- 关键点
  - Activation patching 通过在“干净”与“扰动”前向间直接插换激活，用可观测的输出恢复判断各层/子空间的因果角色，是电路解释的核心工具。citeturn1view0
  - 用归一化的恢复度量（差值/概率）衡量补丁效果，可把部分信息贡献展开成 0~1 区间，便于在多层面比较。citeturn1view0turn4search4
- 必须示例
  - 简单情境：clean run 正确回答“谁赢了比赛”，corrupted run 改写了 prompt 导致错误。如果把第 8 层 residual stream 的激活从 clean 转入 corrupt 后，模型重新输出正确答案；这种“行为恢复”意味着第 8 层携带了“胜者信息”。citeturn4search4
- 必要元素
  - 公式：Effect(l*, p*) = (P_patched − P_orig) / (P_clean − P_orig)，1 表示某个激活单独可复现 clean 行为，0 表示无效。citeturn4search4

## 问题定义与边界
- 关键点
  - 执行前必须明确 clean 与 corrupted 运行，用同一 prompt 的正确/错误版本分别记录激活，以便 patch 时有可替换的“干净信号”。citeturn1view0
  - 结果依赖 corrupt 输入、位置/层选择和度量，重复测不同 corrupt 版本与 metrics 能界定解释边界。citeturn3view0
- 必须示例
  - 举例：clean prompt 要求“列出三位总统”，corrupted prompt 改成“列出三位台风”，两者差异明确，便于确定是哪个 token/通道把问题“翻译”成错误语义。citeturn3view0
- 必要元素
  - 表格：列出常用度量及特点（logit 差、概率、KL、top-k token 评分），帮助工程对齐不同问题/吞吐限制，并提醒 logit 差可以早于概率显著。citeturn4search2

| 度量 | 描述 | 工程信号 |
| --- | --- | --- |
| Logit 差值 | 正确 token 与错误 token logit 之差，适合捕捉 logit 池中微妙方向 | 便于看“倾向”方向，适合快照式 sweep |
| 概率恢复 | 对 clean token 概率归一化，范围 0~1，对用户感知更直观 | 适合反馈系统，但受 temperature、softmax 饱和影响 |
| KL / 概率分布 | 将 patch 后 token 分布与 clean 比，适合衡量多 token 还原 | 计算略贵，用于追踪复杂行为 |

## 核心机制与推导
- 关键点
  - 三阶段机制：先缓存 clean/corrupt 激活，再将某个 hookpoint 的激活替换进入 corrupt run，最后以指定度量观察输出是否恢复。citeturn1view0turn3view0
  - 归一化指标（Effect 或 Recovery）等价于把 patch run 在 clean 与 corrupt 之间插值，便于用 0~1 比较不同层贡献。citeturn4search5
- 必须示例
  - 若第 6 层某 token 的 patched probability 达到 0.8（clean 1.0，corrupt 0.1），Effect ≈ (0.8−0.1)/(1.0−0.1)=0.78，说明该 token 的中间激活已贡献约 78% 的正确概率。citeturn4search4
- 必要元素
  - 公式/推导：Effect(l*, p*) 及其扩展到 logit 差与多个 token，阐明 numerator 表示 patch 增量，denominator 表示可恢复空间。citeturn4search5

## 代码实现
- 关键点
  - 需要在框架（如 TransformerLens）中注册 hookpoint，缓存 clean/corrupt activations，patch run 时将 target 激活硬切换。citeturn3view0
  - 工程实践里通常对层/位置做批量 sweep，把每次 patch 的 Effect 存成表格，供后续排序、可视化与可解释编辑。citeturn3view0
- 必须示例
  - 伪代码示例：在 forward hook 里调用 `cache_activation(layer, token, clean)`，patch 时用 `set_activation(layer, token, cached_clean)`，最后用 `logit_diff(patched) - logit_diff(corrupt)` 计算恢复幅度。citeturn4search4
- 必要元素
  - 代码片段：
    ```python
    def logit_diff(model, tokens):
        logits = model(tokens)
        return logits[correct_token] - logits[wrong_token]

    patched_effect = (logit_diff(patched_run, tokens) - logit_diff(corrupt_run, tokens)) /
                     (logit_diff(clean_run, tokens) - logit_diff(corrupt_run, tokens))
    ```
    该片段体现对比式恢复度量。citeturn4search4

## 工程权衡与常见坑
- 关键点
  - 单看一个 metric 或单个 token 片段容易高估因果贡献，需同步报告概率、KL、token_set，并配合显著性或运行次数。citeturn4search2
  - patching 成本高，逐层/逐 token 的插换可爆炸，需限制搜索范围或与 attribution patching 等近似方法组合。citeturn0search7
- 必须示例
  - 真实工程：一次 patch sweep 让 logit 差在 10 层里恢复 70% 以上，若只看第 5 层概率恢复 90%，但忽略其他层，可能错判“单层控制”而忽略信息在残差流中的分布。citeturn4search2
- 必要元素
  - 表格：对比“仅 logit”与“combination metrics + repeat runs”的稳健性，引导团队在追踪时候采集多维度数据。citeturn4search2

| 方案 | 优点 | 风险 |
| --- | --- | --- |
| 单一 logit 差 | 计算快、便于排序 | 容易被 temperature 影响，可能把统计偏差当因果 |
| 多指标 + 重复 runs | 把噪声劣化为可控误差 | 运行慢，需要 pipeline 自动化与缓存 |

## 替代方案与适用边界
- 关键点
  - Path patching 进一步把 patch 操作限制到特定计算路径，能回答“哪个 attention 头 → 哪个 FFN”而不是单层。citeturn0search2
  - Attribution patching 等梯度近似方法在大模型上更高效，可用于初步筛选，再在小模型/子集上做彻底 patch。citeturn0search9
- 必须示例
  - 例如 path patching 只替换 residual stream 中经过某个 attention head 的路径，若恢复率高，说明该 head 是“突出路径”；如果 path patching 仍低，说明系统需要多个路径协同。citeturn0search2
- 必要元素
  - 表格：比较 activation patching、path patching、attribution patching 的粒度与计算负担，便于根据模型大小/可用 compute 选择策略。citeturn0search2turn0search9

| 技术 | 粒度 | 计算负担 |
| --- | --- | --- |
| Activation patching | 单层/子空间 | 高，需要先缓存 activations |
| Path patching | Path（head/FFN 组合） | 中等，需追踪依赖图 |
| Attribution patching | 梯度/可加性近似 | 低，可做大模型初筛 |

## 参考资料
1. Agentica 解释 activation patching、clean/corrupt/patch 三步与 circuits 背景。citeturn1view0
2. SciX（Heimersheim & Nanda 2024）总结 activation patching 的实际用法、度量与验证建议。citeturn3view0
3. EmergentMind metrics 页列出 logit/prob/KL 等差异与工程启示。citeturn4search2
4. EmergentMind 的 patching 实验解读了 patch 成本、缓存与 logging 流程。citeturn4search3
5. MBrenndoerfer 易懂指南给出 normalized effect 公式与恢复解释。citeturn4search5
