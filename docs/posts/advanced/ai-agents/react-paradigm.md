---
title: ReAct 推理行动范式
date: 2026-08-07
---

# ReAct 推理行动范式

<div class="epigraph">
<p>推理让行动有依据，行动让推理有材料——二者交替，才能解决需要多步思考的复杂任务。</p>
<footer>—— 姚顺雨 等（Shunyu Yao, *ReAct: Synergizing Reasoning and Acting in Language Models\*, 2022）</footer>
</div>

<div class="article-byline">
<p>第四级 · AI 智能体 ｜ ReAct（Yao et al., 2022） ｜ 2026-08-07</p>
</div>

## 为什么从 ReAct 讲起

前五篇我们把智能体的零件各讲了一遍：循环、工具、记忆、反思。
现在到了**组装**的时刻——ReAct 是把这些零件拧成一个成名范式的第一颗螺丝钉，
也是今天几乎所有智能体框架的共同祖先。
<span class="marginnote">回望本专题结构：第一篇「Agent 基础」讲零件，
第二篇「Agent 范式」讲组装。
ReAct 是第二篇的第一篇，
因为它奠定了「推理与行动交替进行」的模板——后面所有范式都是对它的扩展或修改。
你在第四级《大模型原理》里学过的思维链（CoT），
正是 ReAct 的推理组件。
</span>

ReAct 的名字来自 Reasoning + Acting 的合成。
它要回答的问题是：**大模型光会「想」（CoT）或光会「做」（只调工具），
都不够；
能不能让「想」和「做」交替出现，
互相成就？**

## 1 三种基线，一种直觉

理解 ReAct 的最好方式，
是看它对比的三条路线：

**只推理（Reasoning-only / CoT）**：模型连续输出推理步骤，
  最终给出答案。
  优点是不依赖外部环境，
  缺点是**推理脱离现实**——它只能凭训练时的知识「想」，
  遇到「需要实时信息」或「需要验证」的任务就抓瞎。
  比如「现在北京天气如何」这类问题，
  纯推理只能编。
**只行动（Acting-only）**：模型只调用工具、不做显式推理，
  把工具结果直接用于回答。
  优点是接入了现实，
  缺点是**缺乏方向**——不知道先查什么、查完怎么用，
  容易像无头苍蝇。
**ReAct（推理 + 行动交替）**：模型在**思维（Thought）**与**动作（Action）**之间交替：先想「我需要查什么」，
  再执行，
  再观察结果，
  再想下一步。

**ReAct 的直觉：推理提供方向，
行动提供材料。** 这就像解数学题：既有「我要设 $x$ 为……然后……」的思考，
又有「代入公式计算」的具体操作，
思考与操作交替推进。
<span class="marginnote">用第 2 篇的循环语言：ReAct 把「决策」拆成了可读的 Thought，
把「行动」写成了结构化的 Action，
把「观察」接上了真实工具的结果。
它最大的贡献是把原本隐藏在权重里的决策过程<strong>显式地展开在 token 流里</strong>——这既让决策可解释，
也给了模型「边想边查」的自由。
</span>

## 2 ReAct 的交互格式

ReAct 的核心是一个**交替的 token 格式**，
每一轮循环都输出三行之一：

```text
Thought: 我需要查一下「北京今天的天气」。
Action: Search[北京天气]
Observation: 北京今天晴，25°C，微风。
```

**Thought（思维）**：自然语言推理，
  回答「我为什么这么做」。
**Action（动作）**：结构化的工具调用，
  如 `Search[...]`、`Lookup[...]`、`Finish[...]`。
  格式因框架而异，
  但思想一致：**动作必须可被系统解析并执行**。
**Observation（观察）**：系统执行动作后回填的真实结果。
  它不经过模型生成，
  而是来自工具的真实返回。
  <span class="marginnote">注意 Observation 的「真实性」：它是环境写回上下文的，
  不是模型编的。
  这个「外部事实注入」是 ReAct 与纯 CoT 的本质差异——纯 CoT 的所有中间步骤都是模型自产自销，
  ReAct 则不断用真实观察校准推理。
  </span>

**重点：Thought-Action-Observation 三行的交替，
让「推理」与「行动」在同一个 token 流里共享上下文。** 模型既能看到自己之前的思考（防止跑偏），
又能看到工具的真实反馈（防止瞎想）。
这就是 ReAct 名字里「协同」（synergizing）的含义。

## 3 公式解析：ReAct 的状态转移

把 ReAct 的循环写成状态转移，
能看清它与第 2 篇一般循环的异同。
设第 $t$ 轮的上下文为 $h_t$，
则一轮转移为：

$$
h_{t+1} = h_t + \underbrace{\text{Thought}_t + \text{Action}_t}_{\text{模型生成}} + \underbrace{\text{Observation}_t}_{\text{环境回填}}
$$

更细地看，
Action 的选择受当前 Thought 支配，
而 Thought 又依赖之前的观察：

$$
\begin{aligned}
\text{Thought}_t &\sim P_{\theta}\big(\cdot \mid h_t\big) \\
\text{Action}_t &\sim P_{\theta}\big(\cdot \mid h_t, \text{Thought}_t\big) \\
\text{Observation}_t &= \text{Env}\big(\text{Action}_t\big) \\
h_{t+1} &= h_t \oplus \text{Thought}_t \oplus \text{Action}_t \oplus \text{Observation}_t
\end{aligned}
$$

逐项拆解：

- **$P_\theta$ 是同一个语言模型**：Thought 和 Action 都由同一个模型 $P_\theta$ 生成，
  只是**格式不同**——Thought 用自然语言，
  Action 用结构化语法。
  这避免了「推理模型」与「行动模型」分家的复杂工程。
- **$\text{Observation}_t = \text{Env}(\text{Action}_t)$**：观察不经过模型，
  直接从环境返回。
  这一行是 ReAct 的「真实锚点」——它把外部世界强行写进了模型自产自销的推理流。
- **$h_{t+1} = h_t \oplus \cdots$**：上下文不断**累加**（$\oplus$ 表示拼接）。
  这意味着 ReAct 的循环是**有记忆的**：第 $t+1$ 轮的模型能看到第 $t$ 轮全部思考与观察。
  <span class="marginnote">累加是 ReAct 优雅也脆弱的地方：所有历史都堆在上下文里，
  token 消耗随轮次线性增长。
  长任务会撞上上下文窗口上限——第 7 篇《思维链与规划》的 Plan-and-Execute 与第 17 篇《Agent 的长期记忆与状态》都是对「上下文无界增长」问题的回应。
  </span>
- **终止条件**：模型输出 `Finish` 动作或到达最大轮数即停止。

**为什么这套式子重要？** 因为它揭示了 ReAct 的两个设计精髓：**单一模型承担推理与行动**（降低了系统复杂度），
**环境观察持续注入**（保证了推理与现实对齐）。

## 4 ReAct 的优势与局限

ReAct 的实验结果（HotpotQA、ALFWorld 等基准）显示了它的长处，
也暴露了它的边界：

**优势：**
- **可解释性**：Thought 把决策过程摊开，
  用户能看见「模型为什么这么做」，
  便于调试与信任。
- **灵活纠错**：观察会推翻错误的假设，
  模型能即时改道，
  不需要重头再来。
- **少样本成本低**：ReAct 通过提示词即可实现，
  不需要微调模型。

**局限：**
- **上下文膨胀**：长任务中 Thought-Action-Observation 层层累加，
  窗口很快耗尽。
- **推理质量是天花板**：ReAct 的上限受限于基础模型的推理能力——模型不会想，
  交替也没用。
- **无学习能力**：标准 ReAct 不更新参数，
  失败教训不跨任务复用（除非接上第 5 篇的反思）。

**辨析｜易错点：** 有人把「调了工具的对话」就叫 ReAct，
其实 ReAct 的关键在**Thought 的显式存在**。
如果模型只是「调用工具 → 看到结果 → 直接回答」，
没有中间推理步骤，
那只是「带工具的对话」，
不是 ReAct。
**判断标准：循环里有没有独立的 Thought 步骤。**

## 5 小结

- ReAct 在**推理（CoT）与行动（工具调用）**之间交替，
  让思考提供方向、行动提供材料。
- 交互格式三要素：**Thought（思维）→ Action（动作）→ Observation（观察）**，
  观察由环境真实回填。
- 状态转移的数学本质：Thought 与 Action 由单一模型生成，
  Observation 由环境注入，
  上下文逐步累加。
- ReAct 的优势是**可解释、可纠错、低成本**；
  局限是**上下文膨胀、受限于基础模型推理能力、无参数学习**。

在下一节，
ReAct 的「每步都交替」在长任务里太碎，
我们来看另一种范式——先整体规划再执行的**思维链与规划（Plan-and-Execute）**。
