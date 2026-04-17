## 核心结论

SWE-bench Verified 的评测意义，不在于“题更少”，而在于它把原始 SWE-bench 里大量“题意不清、测试不稳、正确解不唯一”的噪声样本先删掉，再比较模型修 bug 的真实能力。[SWE-bench 原始介绍](https://www.swebench.com/original.html)、[OpenAI 对 Verified 的说明](https://openai.com/index/introducing-swe-bench-verified/)

这里的“benchmark”就是基准测试，白话说是一套固定题库，用来比较不同模型谁更强。Verified 从原始 2,294 题里，不是机械抽样，而是由 93 名有 Python 经验的工程师人工审查 1,699 个候选样本，最终留下 500 题；每题由 3 名专家独立评审，目标是保证题目描述清楚、测试能正确判题、任务在给定信息下可解。[OpenAI 官方说明](https://openai.com/index/introducing-swe-bench-verified/)、[Verified 页面](https://www.swebench.com/verified.html)

这会直接改变排行榜的含义。原始 Full 排行榜更容易混入“模型刚好撞上测试细节”的分数；Verified 则更接近“模型是否真的理解仓库、定位问题并写出有效补丁”。一个常被引用的例子是 Claude 3.5 Sonnet：在公开报道中，它在 SWE-bench Full 为 33.4%，在 SWE-bench Verified 为 49.0%。如果把 49% 乘以 500，可得 245 题，这表示在 500 道人工确认边界清晰的题里，它能稳定解出约 245 题。[The Verge，2024-10-22](https://www.theverge.com/2024/10/22/24276822/anthopic-claude-3-5-sonnet-computer-use-ai)

因此，Verified 的核心价值是把“刷模糊题”的空间压缩到更小，让评测更像能力测量，而不是 prompt 取巧竞赛。

---

## 问题定义与边界

SWE-bench 的任务来自 12 个真实 Python 开源仓库，共 2,294 个 issue-PR 对。模型看不到测试，只能拿到 issue 描述和修复前代码库，然后生成一个 patch；如果补丁应用后通过 Fail-to-Pass 测试和回归测试，才算解决问题。[SWE-bench 原始页面](https://www.swebench.com/original.html)

问题在于，原始数据集并不天然等于“公平题库”。OpenAI 在制作 Verified 时明确指出，原始 SWE-bench 存在三类问题：任务描述可能欠明确，测试可能过于依赖某种具体实现，环境差异还可能带来偶发失败。[OpenAI，2024-08-13](https://openai.com/index/introducing-swe-bench-verified/)

“人工验证子集”这个词，白话说就是先让人把有争议的题筛掉，再拿剩下的题做比赛。它的边界不是“最全面”，而是“最适合做稳定比较”。所以 Verified 不是 Full 的替代品，而是一个更可信的子视角。

| 维度 | SWE-bench Full | SWE-bench Verified |
|---|---|---|
| 题目数 | 2,294 | 500 |
| 是否人审 | 否，原始全量集合 | 是，人工筛选 |
| 描述清晰度 | 不保证一致 | 明确要求可理解、可执行 |
| 测试稳定性 | 可能含歧义或错杀 | 目标是测试与题意一致 |
| 评测目标 | 覆盖更广 | 比较更稳、更公平 |

一个玩具例子可以直接说明边界差异。假设有 3 道题：

| 题号 | 题意是否清晰 | 测试是否可靠 | 是否纳入 Verified |
|---|---|---|---|
| A | 是 | 是 | 是 |
| B | 是 | 是 | 是 |
| C | 否 | 否 | 否 |

如果某模型在 C 题上“侥幸”过测，这个结果在 Verified 里不会被计入，因为 C 本身就不该进入可靠评测集。也就是说，Verified 先清洗题库，再统计得分。

---

## 核心机制与推导

SWE-bench 的公开榜单使用 `% Resolved`，白话说就是“解出题目的比例”。按榜单定义可写成：

$$
\text{Resolved \%}=\frac{\text{通过评测的实例数}}{\text{总实例数}}\times 100\%
$$

这里的关键不是公式本身，而是分母代表什么。[SWE-bench Leaderboard](https://www.swebench.com/)

在 Full 中，分母是 2,294；在 Verified 中，分母固定为 500。由于 Verified 先做了人工过滤，所以这 500 不再是“所有题”，而是“被认为足以公平判分的题”。这意味着排名变化不只是样本数变了，而是样本质量变了。

继续看前面的玩具例子。若 Verified 只保留 A、B 两题，而模型解出了 2 题，那么：

$$
\text{Resolved \%}=\frac{2}{2}\times 100\%=100\%
$$

即使它在被剔除的 C 题上也过了测试，分数仍然不会增加，因为 C 不属于可信分母。这个机制的实质是：把模糊题的权重降到 0。

下面用一个最小 Python 例子演示这个计算：

```python
def resolved_percent(resolved: int, total: int) -> float:
    assert total > 0
    assert 0 <= resolved <= total
    return resolved / total * 100

toy = resolved_percent(2, 2)
verified_claude35 = resolved_percent(245, 500)

assert toy == 100.0
assert verified_claude35 == 49.0

print(toy, verified_claude35)
```

真实工程里，这个机制的重要性更高。假设团队在做代码修复 agent，比较两个版本 `agent-v1` 和 `agent-v2`。如果 `v2` 在 Full 上提升 4 个点，但在 Verified 上完全不变，那么更合理的解释往往不是“模型修 bug 变强了”，而是它更擅长利用原始数据里的歧义、测试漏洞或偶然模式。相反，如果 Verified 也同步上升，才更像真实修复能力提升。

Claude 3.5 Sonnet 从 33.4% 到 49.0% 的跳升，常被拿来说明这一点：在更清晰的题集上，它能展现出更高的稳定解题率。这个数字不自动证明“模型已经可靠”，但它说明原始 Full 的噪声，确实会压低或扭曲模型表现。[The Verge，2024-10-22](https://www.theverge.com/2024/10/22/24276822/anthopic-claude-3-5-sonnet-computer-use-ai)

---

## 代码实现

如果你想把 Verified 接进工程流程，官方现在推荐用 `sb-cli`。它是一个命令行工具，白话说就是把“提交预测、等待评测、下载报告”这套动作标准化。[sb-cli Quick Start](https://www.swebench.com/sb-cli/quick-start/)、[sb-cli Submit](https://www.swebench.com/sb-cli/user-guide/submit/)

最小流程如下：

```bash
pip install sb-cli
export SWEBENCH_API_KEY=your_api_key

sb-cli submit swe-bench_verified dev \
  --predictions_path preds.json \
  --run_id my-ci-run

sb-cli get-report swe-bench_verified dev my-ci-run -o reports/

sb-cli list-runs swe-bench_verified dev
```

`preds.json` 的核心格式也很直接：`instance_id` 标识题目，`model_patch` 是你的补丁文本，`model_name_or_path` 是模型名或系统名。[sb-cli Quick Start](https://www.swebench.com/sb-cli/quick-start/)

```json
{
  "django__django-14915": {
    "model_patch": "diff --git a/file.py b/file.py\n...",
    "model_name_or_path": "my-agent-v2"
  }
}
```

一个真实工程例子是，把它接进 CI。流程通常是：从固定提交版本生成预测文件，调用 `sb-cli submit swe-bench_verified dev`，保存 `run_id`，再拉取报告并解析 resolved count。这样做的价值不是“上线前一定跑 500 题”，而是当你改了检索、工具调用、补丁后处理或测试重试策略时，可以用同一套可靠题库看改动是否真的有收益。

---

## 工程权衡与常见坑

Verified 更公平，但不是没有代价。第一，500 题比 2,294 题更稳，却也更窄；它适合比较，不一定足够覆盖所有失败模式。第二，人工筛选提高了题目质量，但不代表测试设计从此完美。[OpenAI，2026-02-23](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)

常见坑可以压缩成一张表：

| 风险 | 结果 | 缓解方式 |
|---|---|---|
| 只看 Full 排行 | 可能把“过模糊题”误判成真实进步 | 同步跑 Verified |
| 不固定运行配置 | 结果波动，难复现 | 固定模型版本、prompt、seed、实例集 |
| 把单次高分当能力 | 容易追逐偶然收益 | 连续多次评测看稳定性 |
| 只看总分不看报告 | 不知道进步来自哪类题 | 按仓库、错误类型拆解报告 |

对小团队最实际的坑是“优化方向错了”。比如你调了一个更激进的搜索策略，Full 分数涨了，但 Verified 没涨，说明这个策略更可能是在模糊题里碰运气，而不是更好地理解代码。此时继续堆这条路线，研发资源就会被错误信号带偏。

---

## 替代方案与适用边界

截至 **2026 年 2 月 23 日**，OpenAI 已明确表示不再把 SWE-bench Verified 作为 frontier 编码能力的主要度量，并建议改报 SWE-bench Pro。原因有两类：一类是测试仍可能错杀正确解；另一类是训练污染，也就是模型可能在训练中见过题目或答案，导致分数不再纯粹反映实时推理和修复能力。[OpenAI，2026-02-23](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)

“污染”这个词，白话说就是考试题和答案提前混进了训练材料，导致高分不完全代表现场能力。

所以今天谈 Verified，结论应该分层：

| 场景 | Verified 是否合适 | 原因 |
|---|---|---|
| 小团队做 CI 回归 | 合适 | 题目清晰，反馈稳定，便于对比版本 |
| 中等能力模型做方法研究 | 合适 | 能降低噪声，便于看真实趋势 |
| Frontier 模型公开发布 | 不再充分 | 污染和测试缺陷会扭曲高分段比较 |
| 80%+ 高分模型细粒度区分 | 不合适 | 天花板效应和错杀问题更明显 |

这就是 Verified 的真正适用边界：它不是“永远正确的终极榜单”，而是一个在特定阶段非常有用的过渡基准。对零基础读者来说，可以把它理解成一套“先把烂题删掉”的工程考试；对工程团队来说，它最有价值的地方，是帮助你判断提升来自真实修复能力，还是来自对噪声数据的适配。等模型接近高分段，就需要迁移到更新、更少污染的评测，比如 SWE-bench Pro，或者自建私有、持续刷新的任务集。[OpenAI，2026-02-23](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)

---

## 参考资料

- [SWE-bench 原始页面](https://www.swebench.com/original.html)
- [OpenAI：Introducing SWE-bench Verified（2024-08-13）](https://openai.com/index/introducing-swe-bench-verified/)
- [SWE-bench Verified 官方页面](https://www.swebench.com/verified.html)
- [SWE-bench Leaderboard](https://www.swebench.com/)
- [sb-cli Quick Start](https://www.swebench.com/sb-cli/quick-start/)
- [sb-cli Submit 文档](https://www.swebench.com/sb-cli/user-guide/submit/)
- [OpenAI：Why SWE-bench Verified no longer measures frontier coding capabilities（2026-02-23）](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- [The Verge：Anthropic’s latest AI update can use a computer on its own（2024-10-22）](https://www.theverge.com/2024/10/22/24276822/anthopic-claude-3-5-sonnet-computer-use-ai)
