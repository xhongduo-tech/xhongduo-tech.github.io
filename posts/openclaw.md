## 引言：一个副业项目，震动了整个 AI 行业

2025 年 11 月，奥地利独立开发者 Peter Steinberger 发布了一个叫 **Clawdbot** 的小项目，自称"playground project"。

两个月后，它以 **OpenClaw** 的名字在 GitHub 上拥有 **14 万 Star、2 万 Fork**，并且 OpenAI CEO Sam Altman 亲自发推宣布：Steinberger 加入 OpenAI，专注下一代个人智能体研发。

| 时间 | 事件 |
|------|------|
| 2025 年 11 月 | Steinberger 发布 **Clawdbot** |
| 2026 年 1 月 27 日 | Anthropic 商标投诉 → 改名 **Moltbot** |
| 2026 年 1 月 30 日 | "Moltbot 念起来不顺口" → 改名 **OpenClaw** |
| 2026 年 2 月 2 日 | GitHub 140,000 ★，20,000 Fork |
| 2026 年 2 月 14 日 | Steinberger 宣布加入 OpenAI |
| 2026 年 2 月 15 日 | Sam Altman 公开确认，OpenClaw 移交独立基金会，OpenAI 赞助 |

为什么能做到这一点？不是因为用了什么别人没有的技术，而是因为它第一次把几个早已存在却相互孤立的能力——**工具调用、持久记忆、可扩展技能、消息平台集成、主动触发**——整合进一个足够简单、足够开放的框架。

这篇文章从架构、Skills 系统、权限配置、部署环境到实战指南，全面拆解 OpenClaw。

---

## 先厘清：什么是"真正的 AI 代理"

普通 AI 助手（ChatGPT 网页、Claude 网页）的工作模式：

```
用户发消息 → AI 生成回复 → 对话结束
```

每次对话**无状态**。AI 不记得上次说过什么，不会主动联系你，不能操作你的文件或执行代码。

AI 代理（Agent）的工作模式：

```
触发（用户消息 / 定时任务 / 外部事件）
        ↓
  加载记忆与上下文
        ↓
  LLM 思考 → 选择工具 → 执行工具 → 观察结果
        ↓（循环，直到任务完成）
  生成回复 → 保存记忆
        ↓
       等待下一次触发
```

Agent 是**有状态、可主动、能行动**的。OpenClaw 是后者的一个具体实现，核心差距如下：

|  | 传统 AI 助手 | OpenClaw |
|--|-------------|----------|
| **状态** | 无状态（会话内有效） | 有状态（持久记忆） |
| **触发** | 被动等待 | 主动 + 被动（Heartbeat + 消息） |
| **工具** | 有限沙盒 | 真实工具（文件、API、代码执行） |
| **扩展** | 无法扩展 | Skills 系统（社区 2,857+ 技能） |
| **运行位置** | 供应商服务器 | 你的机器 / VPS |
| **数据主权** | 供应商持有 | 完全本地 |

---

## 核心架构：六个组件，一个进程

OpenClaw 的工程哲学极度简单：**没有数据库，没有微服务，没有供应商锁定**。

整个系统是运行在你本地或 VPS 上的 **Node.js 单进程**，默认监听 `127.0.0.1:18789`。

```
┌─────────────────────────────────────────────────────────┐
│                      OpenClaw 进程                        │
│                                                         │
│  ┌───────────┐    ┌───────────────┐   ┌──────────────┐  │
│  │  Gateway  │──▶│  Agent Loop   │──▶│     LLM      │  │
│  │（消息路由）│    │（核心处理循环）│   │ Claude / GPT │  │
│  └───────────┘    └──────┬────────┘   │  / DeepSeek  │  │
│        ▲                 │            └──────────────┘  │
│        │           ┌─────▼──────┐                       │
│   WhatsApp         │   Tools    │    ┌──────────────┐   │
│   Telegram  ◀───── │（工具执行） │───▶│    Memory    │   │
│   Discord          └─────▲──────┘    │  ~/clawd/    │   │
│   Signal                 │           └──────────────┘   │
│                    ┌─────┴──────┐                       │
│                    │   Skills   │    ┌──────────────┐   │
│                    │（技能插件） │    │  Heartbeat   │   │
│                    └────────────┘    │（定时心跳）   │   │
│                                      └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Gateway** 通过 WebSocket 协议同时管理所有消息平台连接，负责身份验证、消息路由和安全执行。核心价值是**平台无关性**：你可以在 Telegram 发出指令，在 WhatsApp 收到结果。

**Agent Loop** 是引擎：消息进入后，认证 → 加载记忆 → 组装上下文 → LLM 调用 → 工具执行 → 循环直到完成 → 保存记忆。工具调用循环让模型不只生成文字，而是真正执行并观察结果。

**持久化记忆** 以 Markdown 文件存储在 `~/clawd/`，整个目录可以 `git init` 做版本控制，可随时回滚任意时间点的 Agent 状态。

**Heartbeat** 是 cron 定时任务（默认每 30 分钟），让 Agent 无需用户消息即可主动执行——先用确定性脚本判断是否有值得处理的变化，有才调用 LLM，成本近乎为零。

---

## Skills 系统：深度拆解

Skills 是 OpenClaw 最具工程美感的设计，也是它能快速积累社区生态的核心原因。

### Skills 是什么

**一个 Skill = 一个目录 + 一个 `SKILL.md` 文件**（可选附带 `scripts/` 和 `references/` 子目录）。

```
~/clawd/skills/
├── github-pr-review/
│   ├── SKILL.md          ← 必须
│   ├── scripts/
│   │   └── fetch-pr.sh   ← 可选
│   └── references/
│       └── style-guide.md ← 可选
├── daily-digest/
│   └── SKILL.md
└── smart-alerts/
    └── SKILL.md
```

`SKILL.md` 的完整结构：

```markdown
---
name: github-pr-review
description: Review GitHub pull requests and post structured comments with security, logic, and style analysis
version: 1.0.0
author: openclaw-community
requiredPermissions:
  - read:github
  - write:github_comments
---

# GitHub PR Review

When the user asks to review a PR, do the following:

1. Extract the repo and PR number from the message
2. Call `github_api` to fetch the diff
3. Analyze for: security vulnerabilities, logic errors, code style issues
4. Post a comment with three sections: 🔴 Critical / 🟡 Suggestions / ✅ Looks Good

## When to activate this skill

- "Review PR #42"
- "Check the latest pull request in myrepo/backend"
- Any message mentioning "pull request", "PR", or "code review"

## Tools used

- `github_api` — fetch PR content and post comments
- `bash` — optional local diff processing

## Example output

> **PR #42 Review**
> 🔴 Critical: SQL query on line 47 is vulnerable to injection
> 🟡 Suggestion: Extract magic number 3600 to a named constant
> ✅ Looks Good: Error handling and test coverage are solid
```

这就是 Skill 的全部——纯自然语言，无需写一行代码即可扩展 Agent 能力。

### 选择性注入：为什么 Skill 不会撑爆 prompt

这是 Skills 系统最关键的工程细节。OpenClaw 不会把所有 Skill 的完整内容塞进每次的系统提示词——如果你安装了 100 个 Skill，那样会消耗数万 token，严重拖慢速度和质量。

实际的两步策略：

```
第一步（每次请求都执行）：
  把所有可用 Skill 的「名称 + description 字段」列表注入 prompt
  格式紧凑，通常 200–400 token

       ↓

第二步（按需触发）：
  模型读取列表，判断当前任务与哪个 Skill 相关
  主动 read() 该 Skill 的 SKILL.md 完整内容
  加载后继续推理并执行
```

这是**按需加载**：模型先看目录，有需要再翻书。**`description` 字段因此极为关键**——它是模型判断"要不要读这个 Skill"的唯一依据，写得不清晰，Skill 就永远不会被激活。

### 自己写一个 Skill：15 分钟入门

以写一个"每天早 8 点总结今日日历并推送到 Telegram"为例：

```bash
# 1. 创建目录
mkdir -p ~/clawd/skills/morning-brief
cd ~/clawd/skills/morning-brief
```

新建 `SKILL.md`：

```markdown
---
name: morning-brief
description: Generate a morning briefing with today's calendar events and top priorities, sent automatically at 08:00
version: 1.0.0
requiredPermissions:
  - read:calendar
  - send:telegram
---

# Morning Brief

Every day at 08:00 (triggered by Heartbeat), generate a morning briefing:

1. Fetch today's calendar events via `google_calendar` tool
2. Identify the top 3 priorities based on event urgency and my notes in ~/clawd/priorities.md
3. Format as a clean digest and send via `telegram_send`

## Format

> 📅 **Morning Brief — {date}**
>
> **Today's events:**
> - 10:00 Team standup (30 min)
> - 14:00 Product review (1 hr)
>
> **Top priorities:**
> 1. Finish the deployment before the 14:00 review
> 2. Reply to the pending PR reviews
> 3. Update project roadmap doc

## Heartbeat trigger

Run this skill during every 08:00 Heartbeat check.
```

```bash
# 2. 重启 Gateway（Skill 在启动时快照）
npm restart

# 3. 验证 Skill 已被识别
# 在 Telegram 向 Agent 发送：
# "What skills do you have?"
# 应该能看到 morning-brief 出现在列表中
```

### 安装社区 Skill

```bash
# 从 ClawHub 安装（官方 CLI）
clawhub install github-pr-review
clawhub install daily-digest
clawhub install smart-reminders

# 或者手动克隆
git clone https://github.com/VoltAgent/awesome-openclaw-skills ~/clawd/skills/community
```

ClawHub 目前收录 **2,857+ 社区技能**，涵盖代码审查、邮件处理、日历管理、DevOps 监控、数据库查询等。

---

## 权限系统：如何精确控制 Agent 能做什么

OpenClaw 的安全设计建立在一个假设上：**LLM 是不可完全信任的**——Prompt Injection、模型幻觉，都可能让 Agent 做出危险操作。因此权限是多层叠加的。

### 七层权限优先级

从低到高（后面的配置会覆盖前面的）：

```
Tool Profile          ← Skill 声明的默认权限
Provider Profile      ← LLM 提供商层面的限制
Global Policy         ← ~/.clawd/policies/global.json
Provider Policy       ← 针对特定 LLM 的策略
Agent Policy          ← 针对特定 Agent 的策略
Group Policy          ← 多用户场景下的群组策略
Sandbox Policy        ← 最高优先级，沙盒强制限制
```

大多数个人用户只需关注三层：Skill 声明权限 → Global Policy → Agent Policy。

### 三类核心权限

**文件权限**：控制 Agent 能读写哪些路径。

```json
// ~/.clawd/policies/global.json
{
  "file": {
    "read": ["~/clawd/**", "~/Documents/work/**"],
    "write": ["~/clawd/**"],
    "deny": ["~/.ssh/**", "~/.aws/**", "/etc/**"]
  }
}
```

**Shell 命令权限**：控制能执行哪些命令，三种模式：

```json
{
  "exec": {
    "mode": "ask",          // "allow" | "ask" | "deny"
    "allowlist": [
      "git *",
      "npm *",
      "python3 ~/clawd/scripts/**"
    ],
    "denylist": [
      "rm -rf *",
      "curl * | bash",
      "sudo *"
    ]
  }
}
```

- `allow`：全部放行（危险，不推荐）
- `ask`：每次执行前推送确认消息给你（推荐生产环境）
- `deny`：全部拒绝

**网络权限**：控制能访问哪些外部 API。

```json
{
  "network": {
    "allowedHosts": [
      "api.github.com",
      "api.anthropic.com",
      "calendar.google.com"
    ],
    "blockHosts": [
      "*.local",
      "169.254.*"    // 阻止访问 AWS metadata 服务
    ]
  }
}
```

### 设备令牌：不同设备不同权限

每个接入设备（你的手机、工作电脑、家里的 iPad）都有独立的设备令牌，可以设置不同的权限范围：

```json
// ~/.clawd/devices.json
{
  "devices": {
    "phone-personal": {
      "scopes": ["read:calendar", "send:telegram", "read:files"],
      "deny": ["exec:shell", "write:files"]
    },
    "mac-work": {
      "scopes": ["*"],   // 工作机全权限
      "deny": ["delete:files"]
    }
  }
}
```

这个设计的价值：你的手机只能读日历和收消息，即使有人拿到你的手机发出指令，也无法让 Agent 执行危险命令。

### 常见权限报错及修复

**`EACCES: permission denied`** — 文件系统层面，检查路径是否在 `file.read/write` 白名单内。

```bash
# 快速诊断：用 dry-run 模式列出 Skill 需要的权限
clawd skill run github-pr-review --dry-run
```

**`missing scope: operator.read`** — Skill 声明了某个权限但当前 policy 没有授权，按最小原则补充：

```json
// ~/.clawd/policies/agent.json
{
  "additionalScopes": ["operator.read"]
}
```

**`EPERM: operation not permitted`** — Policy 层面限制，不是文件权限问题。检查 `exec.mode` 是否为 `deny`，或命令是否在 `denylist` 里。

---

## 部署环境：四种选择的完整对比

这是最常被忽视却最重要的决策之一。不同环境的能力边界差距极大。

### 方案一：本地 Mac（开发调试首选）

**优势：**
- 零额外成本，立刻上手
- 支持 **iMessage**（仅 macOS 原生环境可用）
- 本地文件访问最便捷，无需配置隧道
- 支持本地 LLM 推理（Ollama + LLaMA）

**劣势：**
- 电脑休眠时 WebSocket 断线，WhatsApp/Telegram 连接中断
- 需要在系统设置里**关闭自动睡眠**（System Settings → Battery → Prevent sleep）
- 家庭网络不稳定会影响可靠性
- 不适合生产使用

**适合谁：** 刚开始探索、只需 iMessage、不在乎 24/7 在线。

### 方案二：Mac Mini（全能本地服务器）

**优势：**
- 支持 iMessage + 本地推理的唯一"完美方案"
- 功耗低（约 10W），7×24 常开成本极低（约 ¥10/月电费）
- 本地网络内全速访问文件系统
- 一次性投入约 ¥4,000–8,000，无月租

**劣势：**
- 依赖家庭网络稳定性
- 远程访问需要配置端口转发或 Tailscale
- 硬件故障需要自行处理

**适合谁：** 重度个人使用者，想要 iMessage 集成，接受一次性硬件投入。

```bash
# Mac Mini 防睡眠设置
sudo pmset -a sleep 0
sudo pmset -a disablesleep 1

# 用 Tailscale 暴露给外部访问（无需公网 IP）
brew install tailscale
sudo tailscale up
```

### 方案三：VPS 云服务器（生产推荐）

**优势：**
- 数据中心级稳定性，真正的 24/7 在线
- Agent 与你的个人桌面隔离，安全边界清晰
- 可选就近节点降低延迟
- 起步价约 ¥25–150/月（DigitalOcean、Vultr、Linode 等）

**劣势：**
- **不支持 iMessage**（需要 macOS 环境）
- 文件访问需要提前同步或挂载
- 有持续月租成本

**最低配置推荐：** 1 核 1GB RAM 足够跑单用户 OpenClaw，但如果要跑本地模型需要更多资源。

```bash
# Ubuntu 22.04 VPS 快速部署
curl -fsSL https://get.docker.com | bash
git clone https://github.com/openclaw/openclaw && cd openclaw
cp .env.example .env && vim .env   # 填入 API Key
docker compose up -d

# 查看日志
docker compose logs -f
```

**适合谁：** 想要稳定 24/7 服务、不需要 iMessage、愿意接受月租的用户。

### 方案四：混合架构（最优解）

OpenClaw 官方架构图里有一种混合方案，综合了以上优点：

```
VPS（Gateway 层）
  ├── 运行公网 Telegram/WhatsApp Bot
  ├── 处理认证和路由
  └── 通过 Tailscale 隧道连接 ↓

Mac Mini（Worker 层）
  ├── iMessage 集成
  ├── 本地文件访问
  └── 本地 LLM 推理（可选）
```

VPS 暴露公网接口，Mac Mini 处理本地特权操作，两者通过加密隧道通信。这是对稳定性和能力都有要求的用户的最终形态。

### 四种方案一览

| | 本地 Mac | Mac Mini | VPS | 混合 |
|--|---------|---------|-----|------|
| **iMessage** | ✅ | ✅ | ❌ | ✅ |
| **24/7 在线** | ❌ | ✅ | ✅ | ✅ |
| **月租成本** | ¥0 | ~¥10 电费 | ¥25–150 | ¥25–150 |
| **一次性成本** | ¥0 | ¥4,000–8,000 | ¥0 | ¥4,000–8,000 |
| **安全隔离** | 低 | 中 | 高 | 高 |
| **适合阶段** | 探索 | 个人深度用 | 生产 | 终态 |

官方推荐路径：**第 1 月本地跑 → 第 2 月 Docker 化 → 第 3 月迁移 VPS**。

---

## 消息平台：选哪个连接

不同消息平台的技术实现差异巨大，直接影响可靠性和功能完整性。

### Telegram（入门首选）

使用官方 Bot API + 长轮询，**无需公网 IP、域名或 SSL 证书**，家庭宽带直接可用。功能最完整，社区支持最好。

```bash
# 1. 在 Telegram 找 @BotFather，新建 Bot，获取 token
# 2. 在 .env 里配置
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

几乎所有文档和社区 Skill 都优先支持 Telegram，**推荐所有人从这里开始**。

### WhatsApp（手机用户首选）

使用 Baileys 库逆向 WhatsApp Web 协议，扫码连接：

```bash
# 在 .env 里启用 WhatsApp
WHATSAPP_ENABLED=true

# 启动后访问 http://localhost:18789/connect/whatsapp
# 用手机扫码（设置 → 已关联的设备 → 关联新设备）
```

**注意**：Baileys 是非官方实现，WhatsApp 协议更新时可能短暂失效。**使用专用号码而非主号**，避免被封号风险。

### Signal（隐私优先）

端对端加密，元数据收集最少，但配置最复杂，需要命令行工具和加密密钥管理。除非有明确隐私需求，不推荐作为入门选择。

### Discord（团队 / 社区场景）

适合多人共享同一个 Agent，有基于 Guild 的权限管理，支持 Webhook。如果你想部署一个给团队用的 AI 助手，Discord 是最合适的平台。

| | Telegram | WhatsApp | Signal | Discord |
|--|---------|---------|--------|---------|
| **配置难度** | 最简单 | 中等 | 最复杂 | 简单 |
| **稳定性** | 高 | 中（非官方库） | 高 | 高 |
| **隐私** | 中 | 低 | 最高 | 中 |
| **多人支持** | 有限 | 有限 | 有限 | 原生 |
| **推荐场景** | 所有人入门 | 手机优先 | 隐私需求 | 团队共用 |

---

## 实战：从零到第一个有用的 Agent

### 第一步：安装与启动

```bash
git clone https://github.com/openclaw/openclaw
cd openclaw
npm install

cp .env.example .env
```

编辑 `.env`，最少只需要填两个字段：

```bash
# 选择你的 LLM（三选一）
ANTHROPIC_API_KEY=sk-ant-...     # Claude（推荐）
OPENAI_API_KEY=sk-...            # GPT
DEEPSEEK_API_KEY=sk-...          # DeepSeek（最便宜）

TELEGRAM_BOT_TOKEN=...           # 从 @BotFather 获取
```

```bash
npm start
# 看到 "Gateway listening on 127.0.0.1:18789" 即启动成功
```

### 第二步：配置 Agent 人格（AGENTS.md）

```markdown
# My Assistant

You are my personal productivity assistant. Core rules:

1. **Brevity**: Keep answers short unless I ask for detail
2. **Memory**: Log important info and decisions to ~/clawd/memory/
3. **Proactive**: During heartbeat, check for urgent emails and alert me
4. **Language**: Reply in Chinese unless I write in English

## What I care about

- Software engineering projects (TypeScript, Python)
- Stay informed on LLM research papers
- Daily schedule and meeting prep
```

### 第三步：安装三个入门 Skill

```bash
# 每日简报
clawhub install daily-digest

# GitHub 监控（需要配置 GITHUB_TOKEN）
clawhub install github-monitor

# 智能提醒
clawhub install smart-reminders

# 重启使 Skill 生效
npm restart
```

### 第四步：测试几个真实对话

在 Telegram 发送：

```text
你好，帮我列一下今天还没完成的任务
```

```text
帮我 review 一下 github.com/myorg/backend 最新的 PR
```

```text
我现在开始一个新项目，帮我在 ~/clawd/projects/ 下建一个叫
api-gateway 的项目文件夹，记录下项目目标和技术栈
```

第三条指令会触发文件写入权限确认（如果 `exec.mode = "ask"`），你会在 Telegram 收到一条确认消息，回复"确认"即执行。

### 第五步：设置 Heartbeat 定时任务

```bash
# 编辑 crontab
crontab -e

# 每天 8:00 触发 morning-brief skill
0 8 * * * curl -s http://127.0.0.1:18789/heartbeat

# 每 30 分钟常规心跳（邮件检查、服务监控等）
*/30 * * * * curl -s http://127.0.0.1:18789/heartbeat
```

---

## OpenAI 收购：行业读什么信号

2026 年 2 月 15 日，Sam Altman 在 X 上写道：

> "Peter Steinberger is joining OpenAI to drive the next generation of personal agents. He is a genius with a lot of amazing ideas about the future of very smart agents interacting with each other to do very useful things for people."

几个值得解读的信号：

**开源承诺而非收购关闭**：OpenAI 选择把 OpenClaw 移交独立基金会并赞助，表明他们理解社区生态的价值，不想重蹈"收购即扼杀"的覆辙。

**技术方向高度吻合**：本地运行、跨平台、持久记忆的 agent 框架，与 OpenAI 正在推进的 GPT Actions、Custom GPTs 和更长期的 Operator 产品线直接相关。

**竞争格局的信号**：Google 有 Project Astra，Anthropic 有 Claude Computer Use，Microsoft 有 Copilot agent。OpenAI 招揽最热门开源 agent 框架的核心作者，是在抢占**个人 agent 赛道**的定义权。

VentureBeat 的评论标题直接：*"OpenAI's acquisition of OpenClaw signals the beginning of the end of the ChatGPT era"*。

逻辑是清晰的：**从"问答助手"到"自主代理"，是 AI 应用形态的下一次范式转移**。OpenClaw 证明了这个转移可以用极度简单的工程实现，让每个开发者都能参与。

---

## 总结：简单是最深刻的工程洞见

OpenClaw 的成功，技术层面的答案很简单：

- 没有数据库 → 文件系统
- 没有插件框架 → 一个 Markdown 文件
- 没有复杂调度 → 一个 cron job
- 没有微服务 → 一个 Node.js 进程

每个选择都在最大化**可理解性**和**可修改性**，而不是工程上的最优解。这让每一个普通开发者都能读懂、改动、并在此之上构建。这才是 14 万 Star 的真正原因。

OpenAI 收购的，不只是一个代码仓库，而是这个关于 AI agent 应该怎么做的**第一性原理答案**。
