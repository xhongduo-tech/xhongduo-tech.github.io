---
title: 智能体通信语言（KQML/FIPA-ACL）
date: 2026-08-07
---

# 智能体通信语言（KQML/FIPA-ACL）

<div class="epigraph">
<p>说一句话就是在实施一种行为——承诺、请求、断言、提问，语言本身即行动。</p>
<footer>—— 约翰 · 塞尔（John Searle, *Speech Acts\*）</footer>
</div>

<div class="article-byline">
<p>第四级 · 多智能体系统（经典 MAS） ｜ Michael Wooldridge《An Introduction to MultiAgent Systems》第8章 ｜ 2026-08-07</p>
</div>

## 为什么从通信开始

上一篇讲到，多智能体交互的难点之一是**他人意图不可见**——你不知道队友的信念、愿望、计划。破解这一点的最直接手段就是**通信**：把自己的心智状态**说**出来。但「说话」在两个自主智能体之间远不是发个字符串那么简单：我得确保你能听懂、你得知道我的话代表什么承诺、我们还得分清「我告诉你」「我问你」「我请求你」之间的差别。这一篇讲智能体通信语言的两次标准化尝试——**KQML** 与 **FIPA-ACL**，以及它们背后的**言语行为理论**。它也是后面协商、拍卖协议的文字载体。

## 1 智能体为什么要说话

通信不是装饰，它直接为多智能体协作服务，至少承担四类功能：<span class="marginnote">Wooldridge 教材把通信称为「交互的语言层」：谈判协议（第7章）规定「何时说什么」，通信语言规定「一句话是什么」——两者一层一层叠起来，才构成完整的会话。</span>

**信息传递**：把信念告诉别人，消除对方的信息不对称。例如「报告：仓库已清空」。
**意图表达**：把自己的承诺说清楚，让对方敢于依赖。例如「承诺：明天 10 点前交货」。
**协调同步**：对齐行动时机与内容，避免选到不一致组合。
**协商推进**：提议、还价、接受、拒绝——谈判的每一步都是一次言语行为。

一句话：**通信的价值 = 让其他智能体的世界模型变得更可预测**，从而让协作（依赖他人）成为可能。

## 2 言语行为理论：通信的哲学根基

智能体通信语言不是从网络协议直接长出来的，它的哲学根基是**言语行为理论（speech act theory）**，由奥斯汀（Austin）提出、塞尔（Searle）系统化。

核心洞见：**说出一个句子本身就是一种行为**——「我承诺明天交货」这句话不是说它在描述世界，而是**用它完成了一件事**（做出承诺）。塞尔把这称为**言外行为（illocutionary act）**，并给出一套类型：断言（assert）、提问（ask）、请求（request）、命令（command）、承诺（promise）、拒绝（refuse）……<span class="marginnote">「一句话在做事」——这个思想直接映射到智能体通信语言：KQML/FIPA-ACL 里的每一个 <strong>performative/通信行为</strong>，几乎就是塞尔言外行为类型表的计算机实现。</span>

言语行为理论给智能体通信语言划了两条红线：第一，**通信是动作**，所以要有语义（它改变什么状态）；第二，**通信有类型**，不同动作（问 vs 承诺）语义完全不同，不能混为一谈。

## 3 通信语言的层次结构

一个完整的智能体通信栈通常分三层，缺一不可：<span class="marginnote">这个分层与网络七层模型异曲同工，但各层回答的问题完全不同：内容层答「说什么」，消息层答「怎么说」，协议层答「何时说」。</span>

**内容语言（content language）**：描述「话题」的语言，如 KIF（Knowledge Interchange Format）、FIPA-SL、乃至普通 JSON。它负责表达「明天 10 点交货」这件事本身。

**消息语言（message language）**：也就是 **ACL（Agent Communication Language）**，负责给每条消息包上「这是什么动作」的外壳——`inform`、`request`、`promise`。KQML 与 FIPA-ACL 都是消息语言。

**交互协议（interaction protocol）**：规定消息之间的**时序与合法性**——谁先发、收到 `request` 后合法响应有哪些。这是下一篇《协商与谈判协议》的主角。

**辨析｜易错点：** 新手常把「内容语言」和「消息语言」混为一谈。判据很简单：剥掉 `inform(agent1, agent2, "φ")` 里的 `inform` 动作外壳，剩下的 `φ` 就是内容；而这层外壳——动作类型本身——才是 ACL 的职责。协议层则完全不管单条消息说什么，只管消息序列。

## 4 KQML：实践先行

**KQML（Knowledge Query and Manipulation Language）** 是 1990 年代知识共享计划（Knowledge Sharing Effort）产出的第一种广泛使用的智能体通信语言。它的核心概念是**performative（表演语）**——即通信动作，如 `ask-if`、`tell`、`request`、`achieve`、`reply`。

一条 KQML 消息长这样：

```
(request
  :sender    agent-1
  :receiver  agent-2
  :content   (deliver-parcel "2026-08-07")
  :language  KIF
  :ontology  parcel-domain)
```

KQML 的功劳是把「动作外壳 + 内容 + 元信息（发送方、接收方、本体）」的结构固化下来，并且**让 performative 成为头等公民**——它把智能体之间的对话看成 performative 的交换，而非裸数据的搬运。<span class="marginnote">注意 `:ontology` 字段——它声明内容里的词汇表来自哪个本体。没有共同本体，两个智能体即使说同一种语法，也谈不上理解。本体的概念在知识表示/知识图谱专题中展开。</span>

但 KQML 有一个致命短板：**它没有正式语义**。`ask-if` 到底意味着什么、`tell` 成功了会引起什么状态变化，标准文本没有严格定义，于是不同实现之间行为不一致——「都能解析语法，却达不成共同理解」。

## 5 FIPA-ACL：语义补课

为了补 KQML 的语义短板，FIPA（Foundation for Intelligent Physical Agents）制定了 **FIPA-ACL**。它保留了 performative 风格，但给每个**通信行为（communicative act）** 配上了**形式语义**——用模态逻辑严格说明「在什么前提下合法发出，成功后会带来什么理性结果」。<span class="marginnote">FIPA-ACL 的语义受语言学家 Cohen & Levesque 的「理性行为语义」启发：通信行为不是任意的，它必须与智能体的信念、意图等心智状态挂钩，才能被理性地发出与理解。</span>

FIPA-ACL 的常用通信行为包括 `inform`（告知）、`request`（请求）、`query-if`（询问）、`propose`（提议）、`accept-proposal`（接受提议）、`reject-proposal`（拒绝提议）、`confirm`（确认）等。注意到它与协商协议强相关——`propose`/`accept-proposal` 正是为下一篇的谈判会话预留的通信行为。

相比 KQML，FIPA-ACL 的核心改进是：**每条通信行为都有（1）可行性前提 FP（Feasibility Precondition）与（2）理性效果 RE（Rational Effect）**，且语义用公共形式语言（SL）定义，理论上一套语义、到处实现。

## 6 公式解析：`inform` 通信行为的形式语义

拿最基础的 `inform` 做解剖，看 FIPA-ACL 的语义怎么「把一句话翻译成心智状态变化」。记 $B(i, \varphi)$ 为「智能体 $i$ 相信 $\varphi$」，$U(i, \varphi)$ 为「$i$ 不确定 $\varphi$」，则：

$$
\text{FP}(inform(i,j,\varphi)) = B(i,\varphi) \land \neg B(i, B(j,\varphi))
$$

$$
\text{RE}(inform(i,j,\varphi)) = B(j,\varphi)
$$

逐步拆解：

- **第一步，可行性前提 FP**：智能体 $i$ 发出 `inform` 的资格是——$i$ **自己相信** $\varphi$（$B(i,\varphi)$），并且 $i$ **不认为 $j$ 已经知道** $\varphi$（$\neg B(i, B(j,\varphi))$）。否则这个 `inform` 是没意义的（告诉别人他已经知道的事）。
- **第二步，理性效果 RE**：这条消息的**目的**是让 $j$ 相信 $\varphi$，即 $B(j,\varphi)$。注意 RE 是「理性效果」而非「必然效果」——$j$