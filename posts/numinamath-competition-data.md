## 核心结论

NuminaMath 的核心资产不是单个模型，而是一套面向竞赛数学的高质量数据构建与训练流程。它先用大规模 CoT（Chain of Thought，白话说就是“把解题思路一步步写出来”）数据教模型形成文字推理能力，再用 TIR（Tool-Integrated Reasoning，白话说就是“边推理边写代码并执行”）数据教模型调用 Python 工具处理复杂计算，最终在 AIMO Progress Prize 的受限推理环境中取得优势。

简化地看，它的训练链条可以写成：

$$
M_{\text{CoT}}=\text{SFT}(M_0,D_{\text{CoT}})
$$

$$
M_{\text{TIR}}=\text{SFT}(M_{\text{CoT}},D_{\text{TIR}})
=\text{SFT}(\text{SFT}(M_0,D_{\text{CoT}}),D_{\text{TIR}})
$$

其中 SFT（Supervised Fine-Tuning，白话说就是“拿标准答案监督模型学习”）不是一次完成全部能力，而是按课程顺序分阶段完成。

先看全貌：

| 阶段 | 数据规模 | 数据形式 | 训练后模型 | 作用 |
|---|---:|---|---|---|
| CoT 阶段 | 约 860k 题 | 题目 + 文字推理 + 最终答案 | NuminaMath-7B-CoT | 先学“怎么想” |
| TIR 阶段 | 约 70k 题 | 题目 + 推理片段 + Python 代码 + 执行输出 | NuminaMath-7B-TIR | 再学“怎么算、怎么调错” |

玩具例子可以这样理解：先让模型看 5 道 AMC 题的详细文字解法，它会模仿“列条件、化简、代入、得结论”的写法；再给它 2 道需要枚举或验证的题，让它生成 Python 代码并运行。前者训练“思路表达”，后者训练“工具协同”。这比一开始就要求模型直接写代码更稳。

真实工程例子是 AIMO Kaggle 提交环境：每次提交只有 9 小时，而且硬件是 1 张 P100 或 2 张 T4。这里不是“理论上最强”获胜，而是“在时间、显存、随机性都受限时，哪套数据和推理流程最稳”获胜。NuminaMath 的答案是：高质量竞赛数据 + CoT→TIR 两阶段训练 + SC-TIR 多候选执行投票。

---

## 问题定义与边界

这类数据集要解决的问题，不是“让模型会做一般数学题”，而是“让模型稳定解决 AMC/AIME/IMO 及相近难度的竞赛题，并输出可验证的整数答案”。这是一个更窄但更硬的目标，因为竞赛题通常要求多步推导、分类讨论、构造、枚举或数论技巧。

边界主要有四层：

| 边界 | 具体做法 | 原因 |
|---|---|---|
| 题目来源边界 | 主要来自 AoPS、竞赛题库、中文高中与奥数资料、在线试卷 PDF | 保证题目足够难，且题型贴近目标赛题 |
| 输出形式边界 | 不只保留题目和最终答案，还要保留 CoT 或 TIR 解法路径 | 模型要学“过程”，不是只记答案 |
| 评测边界 | 优先选择整数输出题 | 便于自动判题和多数投票 |
| 难度边界 | 覆盖高中数学到竞赛级别，但重点贴近 AMC12 到 AIME 区间 | 与 AIMO 赛题难度更一致 |

NuminaMath 的 CoT 数据卡给出的训练集规模是 859,494 个样本，测试集 100 个样本；TIR 数据卡给出的训练集规模是 72,441 个样本，测试集 99 个样本。工程上通常会把它们概括为“约 860k CoT + 约 70k TIR”。

为了避免只盯着公开榜调参，团队建立了四个内部验证集：

| 验证集 | 数量 | 难度定位 | 用途 |
|---|---:|---|---|
| AMC | 83 | 接近公开竞赛题中的较易部分 | 评估是否已覆盖主流中等难题 |
| AIME | 90 | 比 AMC 更难 | 评估复杂多步题能力 |
| MATH level 4 | 754 | 中高难度 | 扩大样本量，减少小验证集噪声 |
| MATH level 5 | 721 | 高难度 | 观察顶难题的真实上限 |

一个新手友好的理解方式是：不要只拿“最后那 50 道隐藏题”当方向盘。因为样本太少，调参很容易被偶然性带偏。更稳的做法是，先拿 83 道 AMC12 整数题看基础盘，再拿 90 道 AIME 看高难盘，最后用 MATH 的大样本看整体趋势。

---

## 核心机制与推导

NuminaMath 的关键不是“堆更多题”，而是把题按能力链拆成两段。

第一段是 CoT 训练。模型先在 $D_{\text{CoT}}$ 上学习如何把数学问题写成连贯推导。它学到的不是代码执行，而是“定义变量、列式、分步推导、归纳中间结论、给出最终答案”的语言结构。形式上：

$$
M_{\text{CoT}}=\text{SFT}(M_0,D_{\text{CoT}})
$$

第二段是 TIR 训练。这里模型不再只输出自然语言，而是输出“文字解释 + Python 代码 + 代码输出 + 后续修正”。它学到的是：当纯脑算不稳时，应该把一部分计算外包给解释器，并根据 traceback（程序报错堆栈，白话说就是“程序在哪一行炸了”）修正下一轮生成。形式上：

$$
M_{\text{TIR}}=\text{SFT}(M_{\text{CoT}},D_{\text{TIR}})
$$

如果把这套机制展开，可以写成下面这个流程：

```text
竞赛题
  ↓
CoT 阶段：学会写解题思路
  ↓
TIR 阶段：学会插入 Python 代码并读取执行结果
  ↓
SC-TIR 解码：生成多个候选 → 执行代码 → 读 traceback → 自修正
  ↓
多数投票
  ↓
最终整数答案
```

SC-TIR 里的 SC 是 Self-Consistency，白话说就是“别只信一次采样，生成多份答案再投票”。NuminaMath 的做法不是简单多数投票，而是“带执行反馈的多数投票”：

1. 对同一道题生成 $N$ 个候选。
2. 每个候选先生成一段代码。
3. 执行代码，拿到输出或 traceback。
4. 再继续生成下一段，最多走 $M$ 层。
5. 剪掉明显坏掉的分支。
6. 对剩下的最终答案做投票。

其获胜配置是 $N=48, M=4$。这意味着每题不是“问一次”，而是在受限时间内做了 48 路候选、最多 4 轮自修正的搜索。这样做的本质，是把单次生成的随机误差压低。

玩具例子：假设一道题要求求满足条件的整数个数。纯 CoT 方式可能会写出“分类讨论后有 12 个”，但中间某一步漏掉边界。TIR 方式则会写出枚举代码验证所有候选，并把输出与推理对齐。如果第一次代码变量名写错，traceback 会提示 `NameError`，下一轮修正后还能继续。

真实工程例子：AIMO 题目按随机顺序输入，前几题如果刚好偏难，模型在单路长思考里容易耗尽时间。SC-TIR 通过固定候选数和深度，把时间预算分配得更可控，同时用投票降低单条路径失误带来的波动。

---

## 代码实现

工程上，CoT 和 TIR 数据都可以抽象成统一样本结构：`problem`、`solution`、`messages`。前者方便直接训练，后者方便对接聊天格式微调。

下面给一个可运行的简化版玩具实现，用 Python 模拟 SC-TIR 的“执行、过滤、投票”主流程。它不是原始仓库代码，但能准确表达机制。

```python
from collections import Counter

def run_candidate(candidate):
    env = {}
    try:
        exec(candidate["code"], {}, env)
        value = env["solve"]()
        return {"ok": True, "answer": value, "traceback": None}
    except Exception as e:
        return {"ok": False, "answer": None, "traceback": type(e).__name__}

def sc_tir_vote(candidates):
    results = []
    for cand in candidates:
        result = run_candidate(cand)
        if result["ok"]:
            results.append(result["answer"])

    if not results:
        return None

    return Counter(results).most_common(1)[0][0]

toy_candidates = [
    {"code": "def solve():\n    return sum(i for i in range(1, 5))"},   # 10
    {"code": "def solve():\n    x = 1/0\n    return x"},                 # ZeroDivisionError
    {"code": "def solve():\n    return 10"},                             # 10
    {"code": "def solve():\n    return 9"}                               # 9
]

answer = sc_tir_vote(toy_candidates)
assert answer == 10
print(answer)
```

这段代码表达了三件事：

1. 候选不是只看文本，而是执行代码后再判断。
2. 失败分支不会直接污染最终投票。
3. 多个正确候选会把结果集中到同一个答案上。

如果把它写成更接近训练/推理系统的伪代码，大致是：

```python
def solve_with_sc_tir(problem, N=48, M=4):
    states = [init_prompt(problem) for _ in range(N)]

    for _ in range(M):
        new_states = []
        for state in states:
            completion = sample_until_code_block(state)
            exec_result = execute_python(completion.code)
            state = append_feedback(state, completion, exec_result)
            if is_sensible(state):
                new_states.append(state)
        states = new_states

    answers = [extract_final_answer(s) for s in states if has_answer(s)]
    return majority_vote(answers)
```

关键参数可以整理成表：

| 参数 | 值 | 作用 |
|---|---:|---|
| 候选数 `N` | 48 | 提高覆盖率，降低单次采样失误 |
| 深度 `M` | 4 | 允许根据 traceback 自修正多轮 |
| 量化精度 | 8-bit | 压缩显存占用，适配 T4 |
| 单次提交预算 | 9 小时 | 决定推理搜索不能无限扩张 |

真实工程里，8-bit quantization（量化，白话说就是“用更少比特存模型参数”）不是为了学术指标更高，而是为了让模型在 Kaggle 的 T4 上装得下、跑得动。原团队使用 AutoGPTQ 做后训练量化，并用训练数据做校准。

---

## 工程权衡与常见坑

第一类坑是过拟合公开榜。隐藏测试集只有 50 道题，如果你看到一次提交从 72% 涨到 80%，很可能不是模型真的更强，而是当天抽样和题序更友好。NuminaMath 的做法是用 4 个内部验证集对冲这个问题，并且每次评估跑 5 到 10 个 seed。seed 是随机种子，白话说就是“控制采样随机性的开关”。他们观察到 SC-TIR 下不同 seed 的波动通常在 1% 到 3%。

第二类坑是把“会写代码”误当成“会用工具”。单轮生成一段 Python 并不等于真正的 TIR。很多难题需要“生成代码 -> 执行 -> 读输出 -> 再修正”。这也是他们放弃单步 Python 解法、转向多轮 SC-TIR 的原因之一。

第三类坑是硬件约束带来的算法反转。在 H100 上成立的技巧，放到 T4 上可能不成立。比如：
- T4 不支持 `bfloat16`。
- 16-bit 权重本身就接近 32GB VRAM 量级。
- 如果不量化，KV cache 和并发会被严重限制。

因此，工程最优解未必是“更深搜索”，而可能是“更轻模型 + 更稳投票”。

| 风险点 | 现象 | 对策 |
|---|---|---|
| 公开榜过拟合 | leaderboard 提升但复现实验不稳 | 建立 AMC/AIME/MATH 四套验证集 |
| 随机性过大 | 同模型不同次结果差异明显 | 每次评估跑 5-10 个 seed |
| 工具链不稳定 | 代码生成后执行报错多 | 引入 traceback 反馈与剪枝 |
| 显存不足 | T4 上加载或推理失败 | 8-bit 量化 |
| 搜索过深 | 时间耗尽，后面题目没算完 | 固定 N=48、M=4，保守控时 |

一个典型的真实工程场景是：某次配置在公开 leaderboard 上看起来有 80% 精度，但换到 AMC level 5 风格的内部验证时，多个 seed 下均值只有 62%，而且波动很大。这说明它可能是在“撞对答案”，不是在“稳定求解”。多验证集和多 seed 的价值就在这里。

---

## 替代方案与适用边界

可以把 NuminaMath 路线和几种替代方案做一个直接对比：

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| CoT-only | 数据便宜、训练简单 | 不会稳定调用代码，复杂计算易出错 | 中低难度、纯文字推理为主 |
| CoT+TIR | 推理与工具协同，适合复杂竞赛题 | 数据构建和推理系统更复杂 | 需要代码执行反馈的高难场景 |
| TIR-only | 直接强调工具能力 | 缺少自然语言推理底座，易写出机械代码 | 题目高度结构化、算术与枚举占主导 |
| 单步 Python 解法 | 实现简单 | 缺少多轮自修正，难题鲁棒性差 | 简单可编程题 |
| 更深搜索不量化 | 理论上精度可能更高 | 受显存和时限限制，实际可能跑不完 | 高资源离线评测 |

为什么不直接上 TIR-only？因为工具调用不是凭空出现的。模型首先要理解题目结构、知道该验证什么、知道代码结果该如何映射回数学结论。CoT 阶段提供的是“问题表征能力”，TIR 阶段提供的是“计算执行能力”。少了前者，后者往往只会写杂乱代码。

低资源环境下，优先级通常是：
1. 保留 CoT→TIR 两阶段顺序。
2. 控制 TIR 数据质量而不是盲目扩数量。
3. 用 8-bit 量化换可运行性。
4. 用多验证集和多 seed 保证调参可信。

高资源环境下，可以尝试：
1. 提高 TIR 样本量和覆盖题型。
2. 放宽量化，改用 16-bit 获得更高单模型精度。
3. 增加 SC-TIR 深度或候选数。
4. 引入更强的执行环境与缓存策略。

但边界也很明确：如果题目本身不需要枚举、数值验证或程序辅助，TIR 的收益会下降；如果硬件极度受限，过深的 SC-TIR 搜索也会得不偿失。

---

## 参考资料

1. Hugging Face, `AI-MO/NuminaMath-CoT` 数据卡，2024，说明 CoT 数据集规模、来源与处理流程。  
   https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/blob/main/README.md

2. Hugging Face, `AI-MO/NuminaMath-TIR` 数据卡，2024，说明约 70k TIR 样本、GPT-4 生成流程与三次过滤校验。  
   https://huggingface.co/datasets/AI-MO/NuminaMath-TIR/blob/main/README.md

3. Project Numina GitHub 仓库 `aimo-progress-prize` README，2024，说明两阶段训练、8-bit 量化、项目结构与训练命令。  
   https://github.com/project-numina/aimo-progress-prize

4. BARD AI 转载博文《How NuminaMath Won the first AIMO Progress Prize》，2025-12-30，汇总 AIMO 约束、SC-TIR 参数、验证集与工程取舍。  
   https://bardai.ai/2025/12/30/how-numinamath-won-the-first-aimo-progress-prize/

5. MuMath-Code 论文，Numina 训练配方的直接方法来源，用于支持 CoT→TIR 两阶段 SFT 的设计。  
   https://arxiv.org/abs/2405.12544

6. ToRA 论文，提供工具集成推理的数据格式与执行反馈思路，是 TIR 数据构建的重要参考。  
   https://arxiv.org/abs/2309.17452
