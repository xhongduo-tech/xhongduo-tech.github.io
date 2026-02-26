## 引言：一个副业项目，震动了整个 AI 行业

2025 年 11 月，奥地利独立开发者 Peter Steinberger 发布了一个叫 **Clawdbot** 的小项目。他自称这是一个"playground project"——一个让他探索 AI agent 边界的游乐场。

两个月后，这个项目以 **OpenClaw** 的名字在 GitHub 上拥有了 **14 万 Star、2 万 Fork**，并吸引了 OpenAI CEO Sam Altman 亲自发推宣布：Steinberger 加入 OpenAI，专注于下一代个人智能体的研发。

为什么一个副业项目能做到这一点？

不是因为它用了什么别人没有的技术，而是因为它第一次把几个早已存在、却一直相互孤立的能力——**工具调用、持久记忆、可扩展技能、消息平台集成、主动触发**——整合进一个足够简单、足够开放的框架，让每个开发者都能在自己的机器上跑起一个真正能"自主干活"的 AI 代理。

这篇文章拆解 OpenClaw 的全貌：它的历史、架构、技术细节，以及它对整个 AI agent 行业发出的信号。

---

## 先厘清一件事：什么是"真正的 AI 代理"

在理解 OpenClaw 之前，需要先建立一个概念上的区分。

**普通 AI 助手**（ChatGPT 网页、Claude 网页等）的工作模式：

```
用户发消息 → AI 生成回复 → 对话结束
```

每次对话是**无状态**的。AI 不记得上次说过什么（除非你在同一会话里），不会主动联系你，不能真正操作你的文件、发邮件、或者执行代码然后把结果告诉你。

**AI 代理（Agent）**的工作模式：

```
触发（用户消息 / 定时任务 / 外部事件）
        ↓
  加载记忆与上下文
        ↓
  LLM 思考 → 选择工具 → 执行工具 → 观察结果
        ↓ (循环，直到任务完成)
  生成回复 → 保存记忆
        ↓
       等待下一次触发
```

Agent 是**有状态、可主动、能行动**的。它记得历史，能调用工具（运行代码、查数据库、发消息、调 API），能在你不在的时候自己做事。

OpenClaw 是后者的一个具体实现。

---

## 时间线：从 Clawdbot 到 OpenAI

| 时间 | 事件 |
|------|------|
| 2025 年 11 月 | Peter Steinberger 发布 **Clawdbot** |
| 2026 年 1 月 27 日 | Anthropic 商标投诉 → 改名 **Moltbot** |
| 2026 年 1 月 30 日 | "Moltbot 念起来不顺口" → 改名 **OpenClaw** |
| 2026 年 2 月 2 日 | GitHub 140,000 ★，20,000 Fork |
| 2026 年 2 月 14 日 | Steinberger 宣布加入 OpenAI |
| 2026 年 2 月 15 日 | Sam Altman 公开确认，OpenClaw 移交独立基金会，OpenAI 赞助 |

三次改名，两个月时间，一个副业项目完成了大多数初创公司用数年才能走完的路径。

---

## 核心架构：六个组件，一个进程

OpenClaw 的工程哲学极度简单：**没有数据库，没有微服务，没有供应商锁定**。

整个系统是一个运行在你本地机器（或 VPS）上的 **Node.js 单进程**，默认监听 `127.0.0.1:18789`。

```
┌─────────────────────────────────────────────────────┐
│                    OpenClaw 进程                      │
│                                                     │
│  ┌──────────┐    ┌──────────────┐   ┌────────────┐  │
│  │ Gateway  │───▶│  Agent Loop  │──▶│    LLM     │  │
│  │（消息路由）│    │（核心处理循环）│   │（Claude/   │  │
│  └──────────┘    └──────┬───────┘   │ GPT/DeepSeek)│
│       ▲                 │           └────────────┘  │
│       │           ┌─────▼─────┐                     │
│  WhatsApp        │   Tools   │    ┌─────────────┐  │
│  Telegram  ◀──── │（工具执行）│───▶│   Memory    │  │
│  Discord         └─────▲─────┘    │（~/clawd/）  │  │
│  Signal                │          └─────────────┘  │
│                  ┌─────┴─────┐                     │
│                  │  Skills   │    ┌─────────────┐  │
│                  │（技能插件）│    │  Heartbeat  │  │
│                  └───────────┘    │（定时心跳）  │  │
│                                   └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

下面逐一拆解这六个组件。

---

### 一、Gateway：统一消息入口

Gateway 通过 WebSocket 协议同时管理所有消息平台的连接：WhatsApp、Telegram、Discord、Slack、Signal……

它负责三件事：
1. **身份验证**：确认消息来自授权设备
2. **消息路由**：把消息分发给对应的 Agent 实例
3. **安全执行**：通过设备令牌（Device Token）限制每个接入设备的权限范围

这个设计的核心价值是**平台无关性**。你可以在 Telegram 里发出一条指令，然后在 WhatsApp 里收到结果。底层 Agent 逻辑完全相同，变的只是消息的载体。

---

### 二、Agent Loop：每次对话背后的循环

这是整个系统的核心引擎。每当一条消息进入，Agent Loop 执行以下步骤：

```
1. 认证     → 确认来源合法
2. 路由     → 找到对应的 Agent
3. 加载会话  → 从文件系统读取历史记忆
4. 上下文组装 → 构建完整的系统提示词
5. LLM 调用  → 发送给模型（Claude / GPT / DeepSeek）
6. 工具执行  → 模型决定调用工具 → 执行 → 观察结果
7. 循环检查  → 任务未完成？回到步骤 5
8. 生成回复  → 发回消息平台
9. 保存记忆  → 写入文件系统
```

步骤 5-7 是**工具调用循环（Tool Use Loop）**，也是让 Agent 与普通聊天机器人本质不同的地方。模型不只是生成文字，它会：

- 决定"我需要执行一段 Python 代码"
- 执行，获得输出
- 把输出纳入上下文，继续推理
- 决定下一步动作

这个循环一直持续，直到模型认为任务完成，或达到最大步数限制。

---

### 三、持久化记忆：文件系统即数据库

OpenClaw 的记忆系统不使用任何外部数据库。所有状态存储为 **Markdown 文件**，位于 `~/clawd/` 目录：

```
~/clawd/
├── AGENTS.md      # 定义 Agent 的行为和人格
├── SOUL.md        # Agent 的核心价值观和约束
├── TOOLS.md       # 可用工具列表和说明
├── memory/
│   ├── 2026-02-10_14-30.md   # 按时间戳归档的记忆日志
│   ├── 2026-02-11_09-15.md
│   └── ...
└── skills/
    └── [各种技能目录]
```

这个设计极度简单，却有几个关键优势：

- **可版本控制**：整个记忆目录可以 `git init`，你能看到 Agent 的"成长轨迹"，能回滚任何时间点的状态
- **人类可读**：打开文件就能看到发生了什么，不需要任何工具
- **可移植**：整个目录复制到另一台机器，Agent 记忆完整迁移

---

### 四、Skills 系统：插件化的能力扩展

Skills 是 OpenClaw 最具工程美感的设计之一。

**一个 Skill = 一个目录 + 一个 `SKILL.md` 文件**（可选附带脚本和资源文件）。

`SKILL.md` 的结构：

```markdown
---
name: github-pr-review
description: Automatically review GitHub pull requests and post comments
version: 1.0.0
author: openclaw-community
permissions:
  - read:github
  - write:github_comments
---

# GitHub PR Review Skill

You are reviewing a GitHub pull request. When triggered:

1. Fetch the PR diff using the GitHub API
2. Analyze for: security issues, logic errors, code style
3. Post a structured review comment

## Examples

User: "Review PR #42 in myrepo/backend"
Action: Fetch PR, analyze, post review

## Tools Required

- `github_api`: For fetching PR content and posting comments
```

这个文件就是 Skill 的全部——纯自然语言，告诉模型在这个场景下该怎么做、能调用什么工具。

ClawHub（OpenClaw 官方技能市场）目前已收录超过 **2,857 个社区技能**，涵盖：
- 代码审查、GitHub 操作、CI/CD
- 邮件处理、日历管理
- 数据库查询、API 集成
- DevOps、监控告警
- 个人助理、日程规划

**关键技术：选择性注入（Selective Injection）**

OpenClaw 不会把所有 Skill 的全文一股脑塞进系统提示词——那会让 prompt 膨胀到数万 token，严重降低模型性能。

实际的注入策略分两步：

```
第一步：把所有可用 Skill 的「名称 + 描述 + 路径」列表注入 prompt
         （紧凑格式，通常只有几百 token）

第二步：模型判断当前任务需要哪个 Skill 时，
         自行读取对应 SKILL.md 的完整内容
```

这是一种**按需加载**的设计——模型先看"目录"，有需要再"翻书"。

---

### 五、Heartbeat：从被动响应到主动行动

Heartbeat 是 OpenClaw 和普通聊天机器人最直观的差距所在。

它是一个 **cron 定时任务**（默认每 30 分钟触发一次），让 Agent 在没有人类消息输入的情况下主动执行操作：

- 检查你的邮件，有重要邮件主动推送摘要
- 监控服务状态，异常时发 Telegram 告警
- 定期整理记忆，归档旧日志
- 执行预定的定期任务

Heartbeat 采用**两级触发**策略来控制成本：

```
cron 触发
   ↓
确定性脚本检查（不调用 LLM，极低成本）
 ├── 有邮件？ → 有 → 触发 LLM 分析 → 生成摘要推送
 │              否 → 什么都不做，继续等待
 ├── 服务异常？ → 是 → 触发 LLM 诊断 → 发送告警
 └── 无变化 → 退出
```

只有当确定性检查发现有"值得处理的变化"时，才调用 LLM。大多数 Heartbeat 周期里，LLM 根本不会被调用，成本接近于零。

---

### 六、安全模型：假设 LLM 会被攻击

OpenClaw 的安全设计建立在一个悲观假设上：**LLM 本身是不可完全信任的**——无论是越狱攻击（Prompt Injection）还是模型幻觉，都可能让模型做出危险操作。

因此安全机制是多层的：

| 层级 | 机制 |
|------|------|
| 工具审批 | 高危操作（写文件、发消息、调用外部 API）需要人工确认 |
| 权限分离 | 读权限和写权限独立配置，只给必要权限 |
| 设备令牌 | 每个接入设备有独立令牌，限制该设备可触发的操作范围 |
| Skill 审查 | 社区 Skill 推荐人工审查，官方 Skill 经过签名验证 |

---

## 技术核心：上下文组装（Context Assembly）

如果说 Agent Loop 是 OpenClaw 的引擎，那**上下文组装**就是燃料配方。

每次 LLM 调用前，系统会构建一个完整的上下文，结构如下：

```
┌──────────────────────────────────────────────────────┐
│                 最终系统提示词                          │
├──────────────────────────────────────────────────────┤
│ 1. 核心指令（Core Instructions）                        │
│    来自 SOUL.md：Agent 的基本行为准则                   │
├──────────────────────────────────────────────────────┤
│ 2. 工具列表（Tools Prompt）                             │
│    来自 TOOLS.md：当前可用工具的描述                     │
├──────────────────────────────────────────────────────┤
│ 3. Skills 目录（Skills Prompt）                         │
│    所有可用 Skill 的紧凑列表（名称+描述）                 │
├──────────────────────────────────────────────────────┤
│ 4. 引导上下文（Bootstrap Context）                      │
│    环境级别信息：时区、用户偏好、当前项目等               │
├──────────────────────────────────────────────────────┤
│ 5. 运行时覆盖（Per-run Overrides）                      │
│    本次任务的特定覆盖配置（如果有）                       │
├──────────────────────────────────────────────────────┤
│ 6. 历史记忆（Memory）                                   │
│    从 ~/clawd/memory/ 读取的相关历史记录                  │
└──────────────────────────────────────────────────────┘
```

上下文组装是 agentic 系统里最关键的工程决策——**模型知道什么、相信什么、能做什么，全部通过这个阶段决定**。

---

## 与传统 AI 助手的本质差距

|  | 传统 AI 助手 | OpenClaw |
|--|-------------|----------|
| **状态** | 无状态（会话内有效） | 有状态（持久记忆） |
| **触发方式** | 被动（等待用户输入） | 主动 + 被动（Heartbeat + 消息） |
| **工具能力** | 有限（沙盒内） | 真实工具（本地文件、API、代码执行） |
| **扩展方式** | 无法扩展 | Skills 系统（社区 2857+ 技能） |
| **运行位置** | 供应商服务器 | 你的机器 / VPS |
| **数据主权** | 供应商持有 | 完全在你本地 |
| **可定制性** | 接近零 | 完全可配置（SOUL.md、AGENTS.md） |

---

## OpenAI 的收购：行业读什么信号

2026 年 2 月 15 日，Sam Altman 在 X 上写道：

> "Peter Steinberger is joining OpenAI to drive the next generation of personal agents. He is a genius with a lot of amazing ideas about the future of very smart agents interacting with each other to do very useful things for people. We expect this will quickly become core to our [work]."

这不是一次普通的人才招募。

Altman 的这段话透露了 OpenAI 对未来的判断：**个人 AI 代理（Personal Agents）将成为 AI 应用的核心形态**，而 ChatGPT 当前的对话式交互范式，可能只是过渡阶段。

几个值得注意的信号：

1. **开源承诺**：OpenAI 选择不关闭 OpenClaw，而是将其移交独立基金会并继续赞助。这说明他们在意社区生态，不想被视为"收购即扼杀"的公司。

2. **技术方向**：一个能在本地运行、跨消息平台、有持久记忆的 agent 框架，和 OpenAI 正在推进的 GPT Actions、Custom GPTs、以及更长期的 Operator 产品线高度契合。

3. **竞争格局**：Google 有 Project Astra，Anthropic 有 Claude Computer Use，Meta 有 Llama agent 研究，Microsoft 有 Copilot agent。OpenAI 招揽最热门的开源 agent 框架的核心作者，是在抢占个人 agent 赛道的话语权。

VentureBeat 的评论标题直接：*"OpenAI's acquisition of OpenClaw signals the beginning of the end of the ChatGPT era"*。

这个标题也许激进，但逻辑是清晰的：**从"问答助手"到"自主代理"，这是 AI 应用形态的下一次范式转移**。

---

## 如何开始：本地跑起 OpenClaw

```bash
# 克隆仓库
git clone https://github.com/openclaw/openclaw
cd openclaw

# 安装依赖
npm install

# 配置你的 LLM API Key（支持 Claude / GPT / DeepSeek）
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 启动
npm start
```

启动后，访问 `http://127.0.0.1:18789`，按提示连接你的消息平台（Telegram 是最推荐的入门选项，配置最简单）。

**AGENTS.md** 是你定制 Agent 人格的地方：

```markdown
# My Agent

You are a focused productivity assistant. You:
- Prioritize brevity: answers are concise unless depth is requested
- Manage my TODO list in ~/clawd/todos.md
- Alert me to emails from [specific senders] immediately
- Summarize all other emails in the daily digest at 08:00
```

**从哪个 Skill 开始？**

推荐三个入门 Skill（从 ClawHub 安装）：
- `daily-digest`：每天定时生成邮件/日历摘要
- `github-monitor`：监控 PR 和 Issue 并推送
- `smart-reminders`：基于上下文的智能提醒

---

## 总结：简单是最深刻的工程洞见

OpenClaw 的成功，技术层面的答案很简单：

- 没有数据库 → 文件系统
- 没有插件框架 → 一个 Markdown 文件
- 没有复杂调度 → 一个 cron job
- 没有微服务 → 一个 Node.js 进程

每个选择都在最大化**可理解性**和**可修改性**，而不是工程上的"最优解"。

这让每一个普通开发者都能读懂、改动、并在此之上构建。这才是它获得 14 万 Star 的真正原因。

OpenAI 收购的，不只是一个代码仓库，而是这个关于 AI agent 应该怎么做的**第一性原理答案**。
