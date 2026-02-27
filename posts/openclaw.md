## 引言

2025 年 11 月，奥地利开发者 Peter Steinberger 发布了 **Clawdbot**，自称"副业玩具"。两个月后它以 **OpenClaw** 的名字拿到 14 万 GitHub Star，OpenAI CEO Sam Altman 亲自宣布 Steinberger 加入 OpenAI。

| 时间 | 事件 |
|------|------|
| 2025 年 11 月 | Steinberger 发布 **Clawdbot** |
| 2026 年 1 月 27 日 | Anthropic 商标投诉 → 改名 **Moltbot** |
| 2026 年 1 月 30 日 | "念起来不顺口" → 改名 **OpenClaw** |
| 2026 年 2 月 2 日 | 140,000 ★，20,000 Fork |
| 2026 年 2 月 15 日 | Sam Altman 宣布 Steinberger 加入 OpenAI，项目移交独立基金会 |

OpenClaw 做的事情不复杂：把**工具调用、持久记忆、可扩展技能、消息平台集成、定时主动触发**整合进一个 Node.js 单进程，让任何开发者都能在自己的机器上跑起一个真正能自主干活的 AI 代理。

这篇文章从"装好并跑起来"出发，覆盖安装流程中的每一个选择、Skills 系统的工作原理、权限配置，以及各种部署环境的取舍。

---

## 先搞清楚：AI 代理和 AI 助手的区别

普通 AI 助手（ChatGPT、Claude 网页）：

```
用户发消息 → AI 生成文字 → 结束
```

每次无状态，不记得上次，不能主动找你，不能真正执行任何操作。

**AI 代理（Agent）**：

```
触发（消息 / 定时 / 外部事件）
      ↓
加载历史记忆 + 组装上下文
      ↓
LLM 思考 → 选择工具 → 执行工具 → 观察结果
      ↓（循环，直到完成）
回复 → 保存记忆 → 等待下次触发
```

有状态、能主动行动、能真正执行工具（写文件、调 API、跑代码）。OpenClaw 是这个模式的具体实现。

---

## 安装前：三个先决定的问题

安装 OpenClaw 之前，需要先决定三件事。选错了之后会很麻烦。

### 决策一：用哪个 LLM？

OpenClaw 支持多个模型提供商，选择主要看**能力 vs 成本**的权衡：

| 提供商 | 推荐模型 | 输入价格 | 输出价格 | 适合场景 |
|--------|---------|---------|---------|---------|
| **Anthropic** | Claude Sonnet 4.5 | $3/M | $15/M | **推荐：能力与成本最均衡** |
| Anthropic | Claude Haiku 4.5 | $0.80/M | $4/M | 高频简单任务（提醒、快速回答） |
| Anthropic | Claude Opus 4.6 | $30/M | $30/M | 复杂分析、研究，成本高 |
| **OpenAI** | GPT-4o | $15/M | $60/M | 响应速度快（1-2s），工具调用稳定 |
| **DeepSeek** | V3 | $0.27/M | $1.10/M | **最便宜**，但复杂推理较弱 |
| Google | Gemini Flash-Lite | $0.05/M | $0.20/M | 极低成本，速度最快 |

**怎么选：**

- 刚开始探索，想控制成本 → **DeepSeek V3**（价格是 Claude Sonnet 的 1/10）
- 日常使用，想要稳定质量 → **Claude Sonnet 4.5**（大多数人的最终选择）
- 高频简单交互为主 → **Claude Haiku 4.5** 或 **Gemini Flash-Lite**
- 对响应速度要求高 → **GPT-4o**

进阶用法：在 `openclaw.json` 里配置**多模型策略**，简单任务用便宜模型，复杂任务自动切换到 Opus：

```json
{
  "agents": {
    "defaults": {
      "model": "anthropic/claude-sonnet-4-5"
    },
    "routing": {
      "simple": "anthropic/claude-haiku-4-5",
      "complex": "anthropic/claude-opus-4-6"
    }
  }
}
```

拿到 API Key 的方式：
- Claude：[console.anthropic.com](https://console.anthropic.com)
- OpenAI：[platform.openai.com](https://platform.openai.com)
- DeepSeek：[platform.deepseek.com](https://platform.deepseek.com)

---

### 决策二：用哪个消息平台？

OpenClaw 通过消息平台和你交互。四个选项差异很大：

**Telegram（推荐所有人首选）**

使用官方 Bot API + 长轮询。**无需公网 IP、域名、SSL 证书**，家庭宽带直接可用。功能最完整，社区 Skill 优先支持，新手体验最好。

**WhatsApp（手机用户）**

使用 Baileys 库逆向 WhatsApp Web 协议，扫码连接。如果你日常就在 WhatsApp，这是最自然的选择。但有两个注意点：
1. Baileys 是**非官方实现**，WhatsApp 协议更新时可能短暂失效
2. 强烈建议用**专用号码**（备用机/eSIM），不要用主号——一旦被封不影响个人账号

**Signal（隐私优先）**

端对端加密，元数据最少。配置复杂，需要命令行和加密密钥管理。除非有明确隐私需求，不推荐新手起步时选。

**Discord（团队 / 社区）**

适合多人共享同一个 Agent，有基于 Guild 的权限管理。适合部署给团队用的共享助手。

| | Telegram | WhatsApp | Signal | Discord |
|--|---------|---------|--------|---------|
| **配置难度** | ⭐（最简单） | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **稳定性** | 高 | 中 | 高 | 高 |
| **隐私** | 中 | 低 | 最高 | 中 |
| **多人支持** | 有限 | 有限 | 有限 | 原生 |
| **推荐程度** | 首选 | 手机用户 | 特殊需求 | 团队场景 |

---

### 决策三：部署在哪里？

部署环境决定了 OpenClaw 的能力边界，这是最关键的决策。

OpenClaw 需要**持续运行**——它维持着和消息平台的 WebSocket 长连接，以及定时心跳任务。机器睡眠或关机，连接就断。

**选项 A：本地 Mac（探索阶段）**

零成本，立刻可用。最大问题是 Mac 休眠会断连。需要在系统设置里关掉自动睡眠（System Settings → Battery → Prevent sleep）。

唯一能原生支持 **iMessage** 的选项——如果你需要用 iMessage 接入 Agent，必须选这条路。

**适合：** 刚开始学，或者确实需要 iMessage。

**选项 B：Mac Mini 长开服务器（个人深度用）**

功耗约 10W，7×24 常开电费约 ¥10/月，一次性硬件投入约 ¥4,000–8,000。支持 iMessage + 本地文件访问 + 可选本地 LLM（Ollama）。

**适合：** 重度个人用户，需要 iMessage 且想要稳定在线。

**选项 C：VPS 云服务器（生产推荐）**

数据中心级稳定性，真正 24/7 在线。Agent 与个人桌面隔离，安全边界清晰。**不支持 iMessage**（需要 macOS 环境）。

最低配置：1 核 1GB RAM 可以跑，但建议 2 核 2GB。起步价约 ¥25–150/月（DigitalOcean、Vultr、搬瓦工等）。

**适合：** 需要稳定可靠，不在乎 iMessage，可以接受月租。

**选项 D：混合架构（最终形态）**

VPS 跑 Gateway（公网接口、Telegram/WhatsApp Bot），Mac Mini 跑 Worker（iMessage、本地文件），两者通过 Tailscale 加密隧道通信。兼顾稳定性和本地能力。

**适合：** 对稳定性和能力都有要求的用户。

| | 本地 Mac | Mac Mini | VPS | 混合 |
|--|---------|---------|-----|------|
| **iMessage** | ✅ | ✅ | ❌ | ✅ |
| **24/7 在线** | ❌ | ✅ | ✅ | ✅ |
| **月租** | ¥0 | ~¥10 电费 | ¥25–150 | ¥25–150 |
| **一次性** | ¥0 | ¥4,000–8,000 | ¥0 | ¥4,000–8,000 |
| **安全隔离** | 低 | 中 | 高 | 高 |
| **推荐阶段** | 探索 | 个人深度 | 生产 | 终态 |

官方推荐路径：**第 1 月本地跑 → 第 2 月 Docker 化 → 第 3 月迁移 VPS**。

---

## 安装流程

### 第零步：检查 Node.js 版本

OpenClaw **需要 Node.js 22 或以上**，18 和 20 会报语法错误。

```bash
node --version
# 输出示例：v22.13.0
```

如果版本不对：

```bash
# 使用 nvm 安装（推荐）
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc   # 或 source ~/.zshrc

nvm install 22
nvm use 22
nvm alias default 22   # 设为默认版本

node --version   # 确认：v22.x.x
```

---

### 第一步：安装 OpenClaw

```bash
npm install -g openclaw@latest
```

安装完成后验证：

```bash
openclaw --version
```

如果报 `openclaw: command not found`：

```bash
# 查找 npm 全局安装路径
npm config get prefix

# 把该路径下的 bin 目录加入 PATH
# 在 ~/.zshrc 或 ~/.bashrc 末尾加上：
export PATH="$PATH:$(npm config get prefix)/bin"

# 重载
source ~/.zshrc

# 再试
openclaw --version
```

如果 `npm install -g` 报权限错误（不要用 sudo，治标不治本）：

```bash
# 修复 npm 目录所有权
sudo chown -R $USER:$(id -gn $USER) ~/.npm
sudo chown -R $USER:$(id -gn $USER) /usr/local/lib/node_modules

# 然后重新安装（不需要 sudo）
npm install -g openclaw@latest
```

---

### 第二步：运行初始化向导

```bash
openclaw onboard --install-daemon
```

向导会引导你完成：
1. 选择 LLM 提供商和填入 API Key
2. 选择消息平台
3. 配置基本权限策略
4. 安装系统服务（daemon，让 OpenClaw 开机自启）

完成后验证配置是否正确：

```bash
openclaw doctor
# 全绿说明配置没问题

openclaw doctor --fix
# 有问题会尝试自动修复
```

---

### 第三步：配置 .env 和 openclaw.json

向导生成的配置在 `~/.openclaw/openclaw.json`。关键字段说明：

```json
{
  "gateway": {
    "port": 18789,
    "host": "127.0.0.1",   // 只绑定本地，不暴露公网
    "mode": "local"
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-sonnet-4-5"   // 默认模型
    }
  },
  "channels": {
    "telegram": {
      "enabled": true,
      "botToken": "${TELEGRAM_BOT_TOKEN}",   // 引用环境变量
      "dmPolicy": "pairing"   // 陌生人需要审批才能使用
    }
  }
}
```

敏感信息（API Key、Bot Token）放在 `~/.openclaw/.env` 而不是 JSON 文件里：

```bash
# ~/.openclaw/.env
ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
OPENCLAW_GATEWAY_TOKEN=你生成的随机长字符串

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

**第 1 步：创建 Bot**

1. 打开 Telegram，搜索 `@BotFather`
2. 发送 `/newbot`
3. 输入显示名称（如 "My AI"）
4. 输入用户名（必须以 `bot` 结尾，如 `my_ai_helper_bot`）
5. BotFather 返回 Token，格式类似：`123456789:AAF-xxxxxxxxxxxxxxxxxxx`

**第 2 步：配置隐私模式（如果要在群组用）**

在 BotFather 发送：
```
/setprivacy
选择你的 Bot
选择 DISABLED
```

默认的 "Enabled" 模式下 Bot 只能看到 @ 它的消息，Disabled 后能看到所有消息。**个人私聊不受影响，只有群组才需要这一步。**

**第 3 步：填入配置**

```bash
# 方式一：环境变量（推荐）
echo "TELEGRAM_BOT_TOKEN=123456789:AAF-xxx" >> ~/.openclaw/.env

# 方式二：直接写进 openclaw.json
openclaw config set channels.telegram.botToken "123456789:AAF-xxx"
```

**第 4 步：启动并完成配对**

```bash
openclaw gateway
```

在 Telegram 向你的 Bot 发送 `/start`，Bot 会返回一个配对码。在终端确认：

```bash
openclaw pairing approve telegram <配对码>
```

配对完成后，这台设备就绑定到了你的 Gateway，后续直接对话即可。

---

#### WhatsApp 配置

```bash
# 在配置里启用 WhatsApp
openclaw config set channels.whatsapp.enabled true
```

```json
// openclaw.json 里的 whatsapp 配置
{
  "channels": {
    "whatsapp": {
      "enabled": true,
      "dmPolicy": "pairing",
      "allowFrom": ["+你自己的手机号"]
    }
  }
}
```

扫码连接：

```bash
openclaw channels login --channel whatsapp
```

终端显示 QR 码后，立即用手机扫描：

> 打开 WhatsApp → 设置 → 已关联的设备 → 关联新设备 → 扫码

看到 `device linked / session saved` 即成功。

**关于号码选择：**

- **首选**：用备用手机或 eSIM 的专用号码注册一个新 WhatsApp 账号——即使被封也不影响主号
- **备选**：用主号，但 Bot 的消息会出现在你"给自己发消息"的对话里，体验有点奇怪

**常见问题：**

- QR 码过期 → 太慢了，重跑命令立刻扫
- "Can't link new devices" → WhatsApp 在限流，等 24–48 小时
- 会话频繁掉线 → 确认 Gateway 真正在持续运行，不是跑完就退了

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

### 第六步：设置 Heartbeat（主动触发）

```bash
crontab -e
```

添加以下两条：

```bash
# 每天早 8 点：触发晨报 Skill（日历摘要、优先任务）
0 8 * * * curl -s http://127.0.0.1:18789/heartbeat

# 每 30 分钟：常规心跳（邮件检查、服务监控）
*/30 * * * * curl -s http://127.0.0.1:18789/heartbeat
```

---

### 第七步：安装 Skills

```bash
# 查看可安装的 Skill
clawhub search daily-digest

# 安装推荐的三个入门 Skill
clawhub install daily-digest      # 每日简报
clawhub install github-monitor    # GitHub PR/Issue 监控
clawhub install smart-reminders   # 智能提醒

# 安装完需要重启 Gateway（Skill 在启动时快照）
openclaw gateway restart
```

验证 Skill 已被识别——在 Telegram 发给 Bot：

```text
你现在有哪些 Skills？
```

---

### VPS 部署（生产环境）

如果选择 VPS，还需要额外的安全加固步骤。

**基础安全配置：**

```bash
# 1. 创建专用非 root 用户
sudo useradd -m -s /bin/bash openclaw
sudo usermod -aG sudo openclaw
sudo -u openclaw ssh-keygen -t ed25519

# 2. 禁止 root 登录
sudo nano /etc/ssh/sshd_config
# 修改：PermitRootLogin no
# 修改：PasswordAuthentication no
sudo systemctl restart sshd

# 3. 防火墙配置
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp     # SSH（或改成自定义端口）
# 注意：18789 不要开放给公网，通过 SSH 隧道访问

# 4. 确认 Gateway 只绑定本地
# openclaw.json 里：
# "host": "127.0.0.1"  ← 正确
# "host": "0.0.0.0"   ← 危险，不要这样
```

**Docker 部署（推荐用于 VPS）：**

```bash
git clone https://github.com/openclaw/openclaw.git
cd openclaw

# 设置必要的环境变量
export ANTHROPIC_API_KEY="sk-ant-..."
export TELEGRAM_BOT_TOKEN="123456:..."
export OPENCLAW_GATEWAY_TOKEN=$(openssl rand -hex 32)

# 生成 docker-compose.yml
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
# 启动
docker compose up -d

# 查看日志
docker compose logs -f

# 重启
docker compose restart
```

**从本地远程访问 VPS 上的 Dashboard（不暴露端口）：**

```bash
ssh -L 18789:127.0.0.1:18789 user@你的vps-ip
# 然后在本地浏览器打开 http://localhost:18789
```

---

### 验证安装是否成功

```bash
openclaw --version      # 有版本号输出 ✓
openclaw doctor         # 全绿 ✓
openclaw status         # 显示 Gateway: running ✓
```

打开 Dashboard：`http://localhost:18789`

- 模型下拉菜单能选到你配置的模型 ✓
- 消息平台状态显示绿色（connected） ✓
- 在 Telegram/WhatsApp 发一条测试消息，5 秒内收到回复 ✓

---

## Skills 深度拆解

### Skills 到底是什么

**一个 Skill = 一个目录 + 一个 `SKILL.md` 文件**，可选附带 `scripts/` 和 `references/`。

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

### 选择性注入：description 是激活的关键

OpenClaw 不会把所有 Skill 的完整内容塞进每次的系统提示词——100 个 Skill 的完整内容会消耗数万 token。

实际策略分两步：

```
每次请求：
  把所有 Skill 的「name + description」列表注入 prompt（~300 token）
        ↓
模型读列表，判断哪个 Skill 和当前任务相关
        ↓
主动 read() 该 SKILL.md 的完整内容，加载后继续推理
```

**`description` 字段是模型决定"要不要读这个 Skill"的唯一依据。**写得不清晰，Skill 永远不会被激活。好的 description 要包含：这个 Skill 做什么、什么场景触发、有没有关键词。

### 自己写一个 Skill：完整示例

以"每天早 8 点总结日历并推送到 Telegram"为例：

```bash
mkdir -p ~/.openclaw/skills/morning-brief
```

新建 `~/.openclaw/skills/morning-brief/SKILL.md`：

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
# 重启 Gateway 让 Skill 生效
openclaw gateway restart

# 验证已被识别
# 在 Telegram 问：你现在有哪些 Skills？
```

### clawhub 常用命令

```bash
clawhub search <关键词>       # 搜索可用 Skill
clawhub install <skill-name> # 安装
clawhub list                 # 查看已安装的 Skill
clawhub update <skill-name>  # 更新某个 Skill
clawhub uninstall <skill-name> # 卸载
clawhub info <skill-name>    # 查看详情
clawhub sync                 # 扫描目录重新同步
```

---

## 权限系统：精确控制 Agent 能做什么

### 三类核心权限

**文件权限**：配置在 `~/.openclaw/policies/global.json`

```json
{
  "file": {
    "read":  ["~/.openclaw/**", "~/Documents/work/**"],
    "write": ["~/.openclaw/**", "~/Documents/work/**"],
    "deny":  ["~/.ssh/**", "~/.aws/**", "/etc/**", "~/.config/**"]
  }
}
```

**Shell 执行权限**：三种模式

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

- `allow`：全部放行（只在完全信任环境用）
- `ask`：每次执行前通过 Telegram/WhatsApp 推一条确认消息给你（**生产推荐**）
- `deny`：全部拒绝 Shell 操作

**网络权限**：限制能访问哪些外部服务

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

### 设备令牌：不同设备不同权限

```json
// ~/.openclaw/devices.json
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

手机只能读和发消息，即使有人拿到你的手机发指令，也无法让 Agent 执行危险操作。

### 七层权限优先级

后面的配置覆盖前面的：

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

大多数个人用户只需要关注：Skill 声明 → Global Policy → Agent Policy 三层。

### 常见权限错误及修复

**`EACCES: permission denied`**（文件系统层面）

```bash
# 用 dry-run 看 Skill 需要什么权限
clawd skill run github-pr-review --dry-run

# 把缺少的路径加入 file.read/write 白名单
```

**`missing scope: operator.read`**（Policy 层面）

```json
// ~/.openclaw/policies/agent.json 里补上缺少的 scope
{
  "additionalScopes": ["operator.read"]
}
```

**`EPERM: operation not permitted`**（Policy 层面，不是文件权限）

检查 `exec.mode` 是否为 `deny`，或操作命令是否在 `denylist` 里。

---

## 架构简览：六个组件

前面已经完整走完了安装流程，这里给出完整的架构图作为参考：

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
│   Discord          └─────▲──────┘    │  ~/clawd/    │   │
│   Signal                 │           └──────────────┘   │
│                    ┌─────┴──────┐                       │
│                    │   Skills   │    ┌──────────────┐   │
│                    │（技能插件） │    │  Heartbeat   │   │
│                    └────────────┘    │（定时心跳）   │   │
│                                      └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Agent Loop** 是核心引擎：消息进来后，认证 → 加载记忆 → 组装上下文 → LLM 调用 → 工具执行 → 循环直到完成 → 保存记忆。工具调用循环让模型能真正执行并观察结果，而不只是生成文字。

**持久化记忆** 以 Markdown 文件存储在 `~/.openclaw/`，可以 `git init` 做版本控制，回滚任意时间点的状态。

**Heartbeat** 是 cron 定时任务：先用确定性脚本判断是否有值得处理的变化，有才调用 LLM——大多数心跳周期不消耗任何 token。

---

## OpenAI 收购：读什么信号

Sam Altman 在公告里说：

> "Peter Steinberger is joining OpenAI to drive the next generation of personal agents. He is a genius with a lot of amazing ideas about the future of very smart agents interacting with each other to do very useful things for people."

OpenAI 没有关闭 OpenClaw，而是移交独立基金会并继续赞助。技术方向上，本地运行、跨平台、持久记忆的 agent 框架和 OpenAI 正在推进的 GPT Actions、Operator 产品线高度吻合。

更深的信号：**从"问答助手"到"自主代理"，是 AI 应用形态的下一次范式转移**。OpenClaw 用最简单的工程实现证明了这个转移的可行性——没有数据库，没有微服务，一个 Node.js 进程，一个 Markdown 文件作为插件系统。

这让每一个普通开发者都能参与，也正是 14 万 Star 的真正原因。
