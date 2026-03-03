## OpenClaw 是什么

**OpenClaw** 是一个开源的 AI 代理（Agent）框架，基于 Node.js 单进程运行，集成工具调用、持久记忆、可扩展技能（Skills）、消息平台接入和定时主动触发能力。开发者可以在本地或服务器上部署一个能自主执行任务的 AI 代理。

项目原名 Clawdbot，由奥地利开发者 Peter Steinberger 于 2025 年 11 月发布。两个月内获得 14 万 GitHub Star，Steinberger 随后加入 OpenAI。

| 时间 | 事件 |
|------|------|
| 2025 年 11 月 | Steinberger 发布 **Clawdbot** |
| 2026 年 1 月 27 日 | Anthropic 商标投诉 → 改名 **Moltbot** |
| 2026 年 1 月 30 日 | 再次改名 **OpenClaw** |
| 2026 年 2 月 2 日 | 140,000 Star，20,000 Fork |
| 2026 年 2 月 15 日 | Sam Altman 宣布 Steinberger 加入 OpenAI，项目移交独立基金会 |

---

## AI 代理与 AI 助手的区别

普通 AI 助手（ChatGPT、Claude 网页版）是无状态的：用户发消息，模型生成文本，交互结束。不保留上下文，不能主动触发，无法执行外部操作。

**AI 代理（Agent）** 是有状态的自主执行系统：

```
触发（消息 / 定时 / 外部事件）
      ↓
加载历史记忆 + 组装上下文
      ↓
LLM 思考 → 选择工具 → 执行工具 → 观察结果
      ↓（循环，直到完成）
回复 → 保存记忆 → 等待下次触发
```

核心差异在于：代理能执行工具（写文件、调 API、跑代码），能保持跨会话记忆，能被定时任务或外部事件主动触发。OpenClaw 是这一模式的具体实现。

---

## 安装前的三个决策

### 决策一：选择 LLM

OpenClaw 支持多个模型提供商，选择取决于能力与成本的权衡。

| 提供商 | 推荐模型 | 输入价格 | 输出价格 | 适合场景 |
|--------|---------|---------|---------|---------|
| **Anthropic** | Claude Sonnet 4.6 | $3/M | $15/M | 能力与成本最均衡 |
| Anthropic | Claude Haiku 4.5 | $0.80/M | $4/M | 高频简单任务 |
| Anthropic | Claude Opus 4.6 | $30/M | $30/M | 复杂分析，成本高 |
| **OpenAI** | GPT-4o | $15/M | $60/M | 响应速度快（1-2s） |
| **DeepSeek** | V3 | $0.27/M | $1.10/M | 最低成本，复杂推理较弱 |
| Google | Gemini Flash-Lite | $0.05/M | $0.20/M | 极低成本，速度最快 |

选择建议：

- 控制成本优先 → **DeepSeek V3**（价格约为 Claude Sonnet 的 1/10）
- 稳定质量优先 → **Claude Sonnet 4.6**
- 高频简单交互 → **Claude Haiku 4.5** 或 **Gemini Flash-Lite**
- 响应速度优先 → **GPT-4o**

`openclaw.json` 支持多模型路由策略，按任务复杂度自动切换模型：

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-sonnet-4-6"
    },
    "routing": {
      "simple": "anthropic/claude-haiku-4-5",
      "complex": "anthropic/claude-opus-4-6"
    }
  }
}
```

API Key 获取地址：
- Claude：[console.anthropic.com](https://console.anthropic.com)
- OpenAI：[platform.openai.com](https://platform.openai.com)
- DeepSeek：[platform.deepseek.com](https://platform.deepseek.com)

---

### 决策二：选择消息平台

OpenClaw 通过消息平台与用户交互，四个选项特性差异明显。

**Telegram（推荐首选）**：使用官方 Bot API + 长轮询，无需公网 IP、域名或 SSL 证书，家庭宽带直接可用。功能最完整，社区 Skill 优先支持。

**WhatsApp**：使用 Baileys 库逆向 WhatsApp Web 协议，扫码连接。Baileys 是非官方实现，协议更新时可能短暂失效。建议使用专用号码，避免主号被封。

**Signal**：端对端加密，元数据最少。配置复杂，需要命令行和加密密钥管理，适合有明确隐私需求的用户。

**Discord**：适合多人共享同一 Agent，有基于 Guild 的权限管理，适合团队场景。

| | Telegram | WhatsApp | Signal | Discord |
|--|---------|---------|--------|---------|
| **配置难度** | 低 | 中 | 高 | 中 |
| **稳定性** | 高 | 中 | 高 | 高 |
| **隐私** | 中 | 低 | 最高 | 中 |
| **多人支持** | 有限 | 有限 | 有限 | 原生 |
| **推荐场景** | 首选 | 手机用户 | 隐私需求 | 团队 |

---

### 决策三：选择部署环境

OpenClaw 需要持续运行——维持消息平台的 WebSocket 长连接和定时心跳任务。机器休眠或关机会导致连接中断。

**选项 A：本地 Mac（探索阶段）**

零成本，立即可用。Mac 休眠会断连，需关闭自动睡眠（System Settings → Battery → Prevent sleep）。唯一支持 iMessage 接入的选项。

**选项 B：Mac Mini 长开服务器（个人深度使用）**

功耗约 10W，7x24 常开电费约 10 元/月，一次性硬件投入约 4,000-8,000 元。支持 iMessage + 本地文件访问 + 本地 LLM（Ollama）。

**选项 C：VPS 云服务器（生产推荐）**

数据中心级稳定性，24/7 在线，Agent 与个人桌面隔离。不支持 iMessage。最低配置 1 核 1GB RAM，建议 2 核 2GB，月费约 25-150 元。

**选项 D：混合架构**

VPS 运行 Gateway（公网接口、消息 Bot），Mac Mini 运行 Worker（iMessage、本地文件），两者通过 Tailscale 加密隧道通信。

| | 本地 Mac | Mac Mini | VPS | 混合 |
|--|---------|---------|-----|------|
| **iMessage** | 支持 | 支持 | 不支持 | 支持 |
| **24/7 在线** | 否 | 是 | 是 | 是 |
| **月租** | 0 | ~10 元电费 | 25-150 元 | 25-150 元 |
| **一次性投入** | 0 | 4,000-8,000 元 | 0 | 4,000-8,000 元 |
| **安全隔离** | 低 | 中 | 高 | 高 |
| **适合阶段** | 探索 | 个人深度 | 生产 | 终态 |

官方推荐路径：第 1 月本地运行 → 第 2 月 Docker 化 → 第 3 月迁移 VPS。

---

## 安装流程

### 第零步：检查 Node.js 版本

OpenClaw 要求 Node.js 22 或以上，18 和 20 会报语法错误。

```bash
node --version
# 输出示例：v22.13.0
```

版本不符时通过 nvm 安装：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc   # 或 source ~/.zshrc

nvm install 22
nvm use 22
nvm alias default 22

node --version   # 确认：v22.x.x
```

---

### 第一步：安装 OpenClaw

```bash
npm install -g openclaw@latest
```

验证安装：

```bash
openclaw --version
```

如果报 `openclaw: command not found`，将 npm 全局 bin 目录加入 PATH：

```bash
npm config get prefix
# 在 ~/.zshrc 或 ~/.bashrc 末尾添加：
export PATH="$PATH:$(npm config get prefix)/bin"
source ~/.zshrc

openclaw --version
```

如果 `npm install -g` 报权限错误（不要用 sudo）：

```bash
# 修复 npm 目录所有权
sudo chown -R $USER:$(id -gn $USER) ~/.npm
sudo chown -R $USER:$(id -gn $USER) /usr/local/lib/node_modules

# 重新安装
npm install -g openclaw@latest
```

---

### 第二步：运行初始化向导

```bash
openclaw onboard --install-daemon
```

向导引导完成以下配置：
1. 选择 LLM 提供商并填入 API Key
2. 选择消息平台
3. 配置权限策略
4. 安装系统服务（daemon，开机自启）

验证配置：

```bash
openclaw doctor        # 全绿表示配置正确
openclaw doctor --fix  # 自动修复检测到的问题
```

---

### 第三步：配置 .env 和 openclaw.json

向导生成的配置位于 `~/.openclaw/openclaw.json`，关键字段：

```json
{
  "gateway": {
    "port": 18789,
    "host": "127.0.0.1",
    "mode": "local"
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-sonnet-4-6"
    }
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "botToken": "${TELEGRAM_BOT_TOKEN}",
      "dmPolicy": "pairing"
    }
  }
}
```

敏感信息（API Key、Bot Token）存放在 `~/.openclaw/.env`，不写入 JSON：

```bash
# ~/.openclaw/.env
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
OPENCLAW_GATEWAY_TOKEN=随机长字符串

# 限制文件权限
chmod 600 ~/.openclaw/.env
```

生成安全的 Gateway Token：

```bash
openssl rand -hex 32
```

---

### 第四步：连接消息平台

#### Telegram 配置（推荐）

**创建 Bot：**

1. 在 Telegram 搜索 `@BotFather`
2. 发送 `/newbot`
3. 输入显示名称（如 "My AI"）
4. 输入用户名（必须以 `bot` 结尾，如 `my_ai_helper_bot`）
5. BotFather 返回 Token，格式类似：`123456789:AAF-xxxxxxxxxxxxxxxxxxx`

**配置群组隐私模式（仅群组使用时需要）：**

在 BotFather 中发送 `/setprivacy`，选择目标 Bot，设为 DISABLED。默认 Enabled 模式下 Bot 仅接收 @ 消息，DISABLED 后可接收群组内所有消息。私聊不受影响。

**填入配置：**

```bash
# 方式一：环境变量（推荐）
echo "TELEGRAM_BOT_TOKEN=123456789:AAF-xxx" >> ~/.openclaw/.env

# 方式二：写入 openclaw.json
openclaw config set channels.telegram.botToken "123456789:AAF-xxx"
```

**启动并完成配对：**

```bash
openclaw gateway
```

在 Telegram 向 Bot 发送 `/start`，Bot 返回配对码。在终端确认：

```bash
openclaw pairing approve telegram <配对码>
```

配对完成后设备绑定到 Gateway，后续直接对话即可。

---

#### WhatsApp 配置

```bash
openclaw config set channels.whatsapp.enabled true
```

```json
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "dmPolicy": "pairing",
      "allowFrom": ["+手机号"]
    }
  }
}
```

扫码连接：

```bash
openclaw channels login --channel whatsapp
```

终端显示 QR 码后，在手机上操作：WhatsApp → 设置 → 已关联的设备 → 关联新设备 → 扫码。看到 `device linked / session saved` 即成功。

> 建议使用备用手机或 eSIM 的专用号码注册新 WhatsApp 账号。Baileys 为非官方实现，存在被封号风险，专用号码可避免影响主账号。

常见问题：
- QR 码过期 → 重新运行命令后立即扫码
- "Can't link new devices" → WhatsApp 限流，等待 24-48 小时
- 会话频繁掉线 → 确认 Gateway 持续运行

---

### 第五步：配置 AGENTS.md（定义 Agent 人格）

```bash
nano ~/.openclaw/AGENTS.md
```

```markdown
# My Assistant

You are my personal productivity assistant. Core rules:

1. **简洁**：回答简短，除非我要求详细
2. **记忆**：重要决策和信息存到 ~/.openclaw/memory/
3. **语言**：默认中文，我用英文时英文回复
4. **主动**：Heartbeat 时检查重要邮件，有紧急情况主动通知

## 我的关注点

- 软件工程项目（TypeScript、Python）
- LLM 领域最新进展
- 日程和会议准备

## 禁止事项

- 不要在未经确认的情况下删除文件
- 不要向第三方分享我的私人信息
```

---

### 第六步：设置 Heartbeat（定时触发）

```bash
crontab -e
```

添加定时任务：

```bash
# 每天早 8 点：触发晨报 Skill
0 8 * * * curl -s http://127.0.0.1:18789/heartbeat

# 每 30 分钟：常规心跳（邮件检查、服务监控）
*/30 * * * * curl -s http://127.0.0.1:18789/heartbeat
```

---

### 第七步：安装 Skills

```bash
# 搜索可用 Skill
clawhub search daily-digest

# 安装推荐的入门 Skill
clawhub install daily-digest      # 每日简报
clawhub install github-monitor    # GitHub PR/Issue 监控
clawhub install smart-reminders   # 智能提醒

# 重启 Gateway（Skill 在启动时加载）
openclaw gateway restart
```

在 Telegram 向 Bot 发送 `你现在有哪些 Skills？` 验证 Skill 已被识别。

---

### VPS 部署（生产环境）

**基础安全配置：**

```bash
# 1. 创建专用非 root 用户
sudo useradd -m -s /bin/bash openclaw
sudo usermod -aG sudo openclaw
sudo -u openclaw ssh-keygen -t ed25519

# 2. 禁止 root 登录和密码认证
sudo nano /etc/ssh/sshd_config
# PermitRootLogin no
# PasswordAuthentication no
sudo systemctl restart sshd

# 3. 防火墙配置
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
# 注意：18789 不开放给公网，通过 SSH 隧道访问

# 4. 确认 Gateway 只绑定本地
# openclaw.json 里 "host": "127.0.0.1"
# "host": "0.0.0.0" 会暴露公网，不要使用
```

**Docker 部署（推荐）：**

```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw

export ANTHROPIC_API_KEY="sk-ant-..."
export TELEGRAM_BOT_TOKEN="123456:..."
export OPENCLAW_GATEWAY_TOKEN=$(openssl rand -hex 32)

./docker-setup.sh
```

生成的 `docker-compose.yml` 核心结构：

```yaml
services:
  openclaw-gateway:
    image: openclaw:local
    ports:
      - "127.0.0.1:18789:18789"   # 只绑本地
    restart: unless-stopped
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - OPENCLAW_GATEWAY_TOKEN=${OPENCLAW_GATEWAY_TOKEN}
    volumes:
      - ~/.openclaw:/home/node/.openclaw   # 持久化配置和记忆
    user: "node:node"   # 非 root 运行
```

```bash
docker compose up -d       # 启动
docker compose logs -f     # 查看日志
docker compose restart     # 重启
```

通过 SSH 隧道从本地访问 VPS 上的 Dashboard，无需暴露端口：

```bash
ssh -L 18789:127.0.0.1:18789 user@vps-ip
# 本地浏览器打开 http://localhost:18789
```

---

### 验证安装

```bash
openclaw --version      # 有版本号输出
openclaw doctor         # 全绿
openclaw status         # 显示 Gateway: running
```

打开 Dashboard `http://localhost:18789`，确认：
- 模型下拉菜单包含已配置的模型
- 消息平台状态显示 connected
- 在 Telegram/WhatsApp 发送测试消息，5 秒内收到回复

---

## Skills 系统

### Skill 的结构

一个 Skill 由一个目录和一个 `SKILL.md` 文件组成，可选附带 `scripts/` 和 `references/`。

```
~/.openclaw/skills/
├── github-pr-review/
│   ├── SKILL.md          ← 必须
│   ├── scripts/
│   │   └── fetch-pr.sh
│   └── references/
│       └── style-guide.md
├── morning-brief/
│   └── SKILL.md
└── smart-alerts/
    └── SKILL.md
```

`SKILL.md` 完整结构：

```markdown
---
name: github-pr-review
description: Review GitHub pull requests and post structured comments
           with security, logic, and style analysis
version: 1.0.0
author: you
requiredPermissions:
  - read:github
  - write:github_comments
---

# GitHub PR Review

When the user asks to review a PR:

1. Extract repo and PR number from the message
2. Use `github_api` tool to fetch the diff
3. Analyze for: security issues, logic errors, code style
4. Post a review comment with three sections:
   🔴 Critical | 🟡 Suggestions | ✅ Looks Good

## When to activate this skill

- "Review PR #42"
- "Check the latest pull request in myrepo/backend"
- Any message mentioning "pull request", "PR", "code review"

## Tools used

- `github_api` — fetch PR content, post comments
- `bash` — optional local diff processing
```

### 选择性注入机制

OpenClaw 不会将所有 Skill 的完整内容注入每次请求的系统提示词。实际策略分两步：

```
每次请求：
  注入所有 Skill 的「name + description」列表（约 300 token）
        ↓
  模型判断哪个 Skill 与当前任务相关
        ↓
  主动读取该 SKILL.md 完整内容，加载后继续推理
```

**`description` 字段决定了模型是否会激活该 Skill。** `description` 需要明确包含：Skill 的功能、触发场景和关键词。描述不清晰的 Skill 不会被激活。

### 自定义 Skill 示例

以"每天早 8 点总结日历并推送到 Telegram"为例：

```bash
mkdir -p ~/.openclaw/skills/morning-brief
```

创建 `~/.openclaw/skills/morning-brief/SKILL.md`：

```markdown
---
name: morning-brief
description: Every morning at 08:00, generate a briefing with today's
           calendar events and top 3 priorities, then send to Telegram.
           Triggered automatically by Heartbeat.
version: 1.0.0
requiredPermissions:
  - read:calendar
  - send:telegram
---

# Morning Brief

**Triggered by**: Heartbeat at 08:00

**Steps**:
1. Fetch today's calendar events using `google_calendar` tool
2. Check ~/my-notes/priorities.md for standing priorities
3. Format a clean digest
4. Send via `telegram_send`

## Output format

> 📅 **Morning Brief — {date}**
>
> **Today**
> - 10:00 Team standup (30 min)
> - 14:00 Product review (1 hr)
>
> **Top priorities**
> 1. Deploy before 14:00 review
> 2. Reply to pending PRs

## Heartbeat config

Run during every 08:00 Heartbeat trigger.
```

```bash
# 重启 Gateway 使 Skill 生效
openclaw gateway restart
```

### clawhub 命令参考

```bash
clawhub search <关键词>       # 搜索可用 Skill
clawhub install <skill-name> # 安装
clawhub list                 # 查看已安装 Skill
clawhub update <skill-name>  # 更新
clawhub uninstall <skill-name> # 卸载
clawhub info <skill-name>    # 查看详情
clawhub sync                 # 重新扫描目录同步
```

---

## 权限系统

### 三类核心权限

**文件权限**（配置在 `~/.openclaw/policies/global.json`）：

```json
{
  "file": {
    "read":  ["~/.openclaw/**", "~/Documents/work/**"],
    "write": ["~/.openclaw/**", "~/Documents/work/**"],
    "deny":  ["~/.ssh/**", "~/.aws/**", "/etc/**", "~/.config/**"]
  }
}
```

**Shell 执行权限**（三种模式：`allow` / `ask` / `deny`）：

```json
{
  "exec": {
    "mode": "ask",
    "allowlist": [
      "git *",
      "npm *",
      "python3 ~/.openclaw/scripts/**"
    ],
    "denylist": [
      "rm -rf *",
      "curl * | bash",
      "sudo *",
      "chmod 777 *"
    ]
  }
}
```

- `allow`：全部放行，仅限完全信任的环境
- `ask`：每次执行前通过消息平台发送确认请求（生产推荐）
- `deny`：禁止所有 Shell 操作

**网络权限**：

```json
{
  "network": {
    "allowedHosts": [
      "api.github.com",
      "api.anthropic.com",
      "calendar.google.com"
    ],
    "blockHosts": [
      "169.254.*",
      "*.local"
    ]
  }
}
```

### 设备令牌

不同设备可配置不同权限范围：

```json
{
  "devices": {
    "phone-personal": {
      "scopes": ["read:calendar", "send:telegram", "read:files"],
      "deny":   ["exec:shell", "write:files"]
    },
    "mac-work": {
      "scopes": ["*"],
      "deny":   ["delete:files"]
    },
    "ipad-readonly": {
      "scopes": ["read:*"],
      "deny":   ["write:*", "exec:*"]
    }
  }
}
```

手机设备限制为只读和发消息，即使设备被他人获取，也无法通过 Agent 执行危险操作。

### 权限优先级（七层）

后层配置覆盖前层：

```
Skill 声明的默认权限（最低）
     ↓
LLM 提供商层面限制
     ↓
Global Policy（~/.openclaw/policies/global.json）
     ↓
Provider Policy（针对特定 LLM）
     ↓
Agent Policy（针对特定 Agent）
     ↓
Group Policy（多用户场景）
     ↓
Sandbox Policy（最高，强制限制）
```

个人用户通常只需关注 Skill 声明 → Global Policy → Agent Policy 三层。

### 常见权限错误

**`EACCES: permission denied`**（文件系统层面）：

```bash
# 查看 Skill 需要的权限
clawd skill run github-pr-review --dry-run

# 将缺少的路径加入 file.read/write 白名单
```

**`missing scope: operator.read`**（Policy 层面）：

```json
// ~/.openclaw/policies/agent.json
{
  "additionalScopes": ["operator.read"]
}
```

**`EPERM: operation not permitted`**（Policy 层面）：检查 `exec.mode` 是否为 `deny`，或操作命令是否在 `denylist` 中。

---

## 架构概览

OpenClaw 由六个核心组件构成：

```
┌─────────────────────────────────────────────────────────┐
│                      OpenClaw 进程                        │
│                                                         │
│  ┌───────────┐    ┌───────────────┐   ┌──────────────┐  │
│  │  Gateway  │──▶│  Agent Loop   │──▶│     LLM      │  │
│  │（消息路由）│    │（核心处理循环）│   │ Claude/GPT/  │  │
│  └───────────┘    └──────┬────────┘   │  DeepSeek    │  │
│        ▲                 │            └──────────────┘  │
│        │           ┌─────▼──────┐                       │
│   WhatsApp         │   Tools    │    ┌──────────────┐   │
│   Telegram  ◀───── │（工具执行） │───▶│    Memory    │   │
│   Discord          └─────▲──────┘    │ ~/.openclaw/ │   │
│   Signal                 │           └──────────────┘   │
│                    ┌─────┴──────┐                       │
│                    │   Skills   │    ┌──────────────┐   │
│                    │（技能插件） │    │  Heartbeat   │   │
│                    └────────────┘    │（定时心跳）   │   │
│                                      └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Agent Loop** 是核心引擎：消息进入后，执行认证 → 加载记忆 → 组装上下文 → LLM 调用 → 工具执行 → 循环直到任务完成 → 保存记忆。工具调用循环使模型能执行操作并观察结果，而非仅生成文本。

**持久化记忆** 以 Markdown 文件存储在 `~/.openclaw/`，支持 `git init` 做版本控制，可回滚到任意时间点的状态。

**Heartbeat** 由 cron 定时任务驱动：先用确定性脚本判断是否有需要处理的变化，有变化才调用 LLM。大多数心跳周期不消耗 token。

---

## OpenAI 收购的信号

Sam Altman 在公告中表示：

> "Peter Steinberger is joining OpenAI to drive the next generation of personal agents. He is a genius with a lot of amazing ideas about the future of very smart agents interacting with each other to do very useful things for people."

OpenAI 没有关闭 OpenClaw，而是移交独立基金会并继续赞助。技术方向上，本地运行、跨平台、持久记忆的代理框架与 OpenAI 正在推进的 GPT Actions 和 Operator 产品线高度契合。

OpenClaw 用最简单的工程实现验证了"从问答助手到自主代理"这一应用形态转变的可行性：没有数据库，没有微服务，一个 Node.js 进程加一个 Markdown 文件作为插件系统。这种低门槛的设计让每个开发者都能参与，也是 14 万 Star 的核心原因。
