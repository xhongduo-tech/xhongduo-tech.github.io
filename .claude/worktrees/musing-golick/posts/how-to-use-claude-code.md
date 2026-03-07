## 什么是 Claude Code？

网页版大模型（如 ChatGPT 或 Claude）本质上是受限的对话容器，**缺乏本地项目级上下文，且没有系统执行权限。**

相比之下，Claude Code 是直接部署在本地终端的自动化编程代理（Agent）。它突破了"纯文本交互"的局限，具备全局代码库检索和本地读写/执行权限。当你输入指令（如"修复某函数 bug"），它能自主完成"分析源码 -> 定位缺陷 -> 修改文件 -> 运行测试"的工程闭环。凭借查阅文档和并发调度多个子代理（Sub-agents）的能力，它将 AI 的角色从单纯的"代码片段生成器"升级为了可以直接在本地环境交付结果的自动化执行引擎。

---

## 安装与配置

### 安装 Claude Code

macOS / Linux / WSL：

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Windows PowerShell：

```bash
irm https://claude.ai/install.ps1 | iex
```

安装完成后验证：

```bash
claude --version
claude doctor
```

`claude doctor` 检查安装状态，发现问题时会给出修复建议。

### 登录账号

需要 Anthropic 账号（Claude Pro、Max、Teams 或 Enterprise 均可）。运行 `claude`，首次启动会引导浏览器完成登录授权。

### 接入第三方模型平台

以火山引擎 ARK 平台为例，修改两个配置文件即可接入，无需搭建代理。

编辑 `~/.claude/settings.json`，填入 API Key 和模型名称：

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "<ARK_API_KEY>",
    "ANTHROPIC_BASE_URL": "https://ark.cn-beijing.volces.com/api/coding",
    "ANTHROPIC_MODEL": "<Model_Name>"
  }
}
```

编辑 `~/.claude.json`，跳过首次登录引导：

```json
{
  "hasCompletedOnboarding": true
}
```

保存后重新运行 `claude` 即可。

> 不同平台对 Claude Code 工具调用格式的支持程度不一，实际体验可能与官方模型有差距。

---

## 基本使用

进入项目目录，运行 `claude`，出现 `>` 提示符后直接输入指令：

```bash
cd your-project
claude
```

```text
> 帮我看看 src/auth.js 这个文件，有没有什么安全隐患？
```

Claude 直接读取本地文件并给出分析，不需要手动粘贴代码。

Claude Code 支持的任务类型：

| 任务类型 | 示例 |
|---------|------|
| 修复 Bug | "登录时偶尔会报 undefined，帮我找找原因" |
| 开发新功能 | "给用户设置页加一个头像上传功能" |
| 重构代码 | "这个文件太长了，帮我拆成几个模块，逻辑不变" |
| 理解代码库 | "这个项目的认证流程是怎么运作的" |
| 写测试 | "给 utils/format.js 里的函数写单元测试，用 Jest" |
| 代码审查 | "帮我 review 一下最近这次提交的改动" |
| Git 操作 | "帮我写一个语义化的 commit message 然后提交" |
| 框架迁移 | "把状态管理从 Redux 迁移到 Zustand，保持行为一致" |
| 安全审计 | "扫一下整个项目，找出所有可能的 SQL 注入和 XSS 风险" |

---

## 核心交互技巧

### 约束比目标更重要

指令中明确约束条件，可以防止 Claude 改动超出预期范围。

```text
# 模糊
> 优化一下数据库查询

# 清晰
> 优化 src/db/users.js 里的 getUsersByAge 函数的查询性能，
  不要改变返回值的数据结构，改完跑一下现有的测试确保没有破坏
```

### 先确认计划，再执行

对于涉及多个文件的改动，先让 Claude 输出改动计划：

```text
> 我想把整个项目的用户认证从 JWT 改成 Session，
  先告诉我你打算怎么改，改哪些文件，我确认了再动手
```

确认无误后再执行，这是避免意外改动的最有效方式。

### 用 `@` 引用文件

```text
> @src/components/Header.tsx 和 @src/styles/global.css 看一下，
  现在 header 的样式为什么在移动端会溢出？
```

`@文件路径` 让 Claude 直接定位到指定文件。

### 用 `!` 执行 shell 命令

在 `>` 后加 `!` 直接执行 shell 命令，不经过 Claude 处理：

```bash
!npm test
!git log --oneline -10
!ls src/
```

---

## 命令速查

### 斜杠命令

在 `>` 提示符后输入 `/` 开头的指令：

| 命令 | 作用 | 使用场景 |
|-----|------|---------|
| `/help` | 查看所有可用命令 | 忘记命令名称时 |
| `/clear` | 清除对话历史 | 切换到不相关的新任务前 |
| `/compact` | 压缩上下文，节省 token | 对话过长导致遗忘时 |
| `/model` | 切换模型（Sonnet / Opus / Haiku）| 调整推理能力或控制成本 |
| `/init` | 生成 CLAUDE.md 记忆文件 | 首次在新项目中使用时 |
| `/memory` | 编辑记忆文件 | 追加或修正项目约定 |
| `/permissions` | 管理工具权限 | Claude 因权限不足无法执行操作时 |
| `/mcp` | 管理 MCP 外部服务连接 | 接入 GitHub、数据库等外部工具 |
| `/status` | 查看版本、模型、账号信息 | 确认当前状态 |
| `/cost` | 查看 token 消耗和费用 | 监控会话成本 |
| `/context` | 可视化上下文使用量 | 判断是否需要 `/compact` |
| `/rewind` | 撤销最近的代码改动 | Claude 改了不该改的东西时 |
| `/vim` | 开启 vim 键位模式 | vim 用户 |

### 键盘快捷键

| 快捷键 | 作用 |
|-------|------|
| `Ctrl+C` | 中断当前生成 |
| `Ctrl+L` | 清屏（保留对话历史）|
| `Ctrl+O` | 切换详细模式（查看工具调用过程）|
| `Ctrl+R` | 搜索历史命令 |
| `Shift+Tab` | 循环切换权限模式（计划 / 自动执行 / 普通）|
| `Option+T`（macOS）| 切换深度思考模式 |
| `Esc+Esc` | 撤销并重来 |

---

## 深度思考模式

开启后，模型在回答前进行扩展的内部推理，输出更深入的分析。

### 开启方式

- **快捷键**：`Option+T`（macOS）或 `Alt+T`（Windows/Linux）
- **切换模型**：`/model` 选择 `claude-opus-4-6`，设置思考强度（low / medium / high）。Opus + high 是当前最强推理组合。
- **查看思考过程**：`Ctrl+O` 开启详细模式，内部推理以灰色斜体显示。

### 适用场景

深度思考适合以下场景：

- 架构决策：分析系统瓶颈和扩展方案
- 复杂 Bug 排查：表面现象和根本原因之间推理链较长
- 技术方案对比：需要权衡多个方案的长期影响
- 数学或算法问题：需要严格推导

简单任务（改变量名、解释单行代码）不需要深度思考，普通模式更快。

---

## 多智能体

Claude Code 支持将任务拆分为并行子任务，由多个独立 AI 实例同时处理，主实例负责协调和汇总。

### 使用方式

在指令中表达并行需求，Claude 会自动分配子代理：

```text
> 这个应用有性能问题，页面加载很慢。
  请从两个方向同时排查：
  1. 前端：检查不必要的重渲染和大体积资源
  2. 后端：检查 API 响应时间和数据库慢查询
  两边同时查，最后汇总报告
```

```text
> 给这个项目写完整的测试覆盖。
  三个方向并行：
  - utils 函数的单元测试
  - API 路由的集成测试
  - React 组件的 snapshot 测试
```

### 实验性 Agent Teams

上述子代理是内置行为。实验性多智能体团队（每个智能体有独立上下文窗口、可互相通信）需要额外配置。

> 这是实验性功能，API 可能随版本变化。以下配置适用于撰文时的版本。

在 `~/.claude/settings.json` 中添加：

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

启动时使用：

```bash
claude --teammate-mode in-process
```

---

## 深度研究模式

Claude Code 内置 **Explore 代理**，专用于全面探索代码库或深入调查复杂问题。

### 代码库探索

```text
> 我刚接手这个项目，完全不了解它的结构。
  请全面探索：数据流、核心模块、模块间依赖关系，
  写一份架构摘要
```

### 结合网络搜索做技术调研

```text
> 我们需要给 API 加限流功能。
  请研究主流限流算法（令牌桶、漏桶、滑动窗口等），
  结合当前的 Redis 使用情况，给出具体实现方案
```

Claude 会结合代码分析和网络搜索，给出针对当前项目的具体建议。

### 推荐工作流：研究-计划-执行

对于重要改动，分三步进行：

```bash
# 第一步：只读研究
> 研究一下把状态管理从 Redux 迁移到 Zustand 的可行性，
  分析改动范围，不要做任何修改

# 第二步：制定计划
> 根据分析结果，制定分阶段的迁移计划

# 第三步：执行
> 开始执行第一阶段
```

---

## CLAUDE.md：项目级记忆

每次启动新会话，Claude Code 自动读取项目根目录下的 **CLAUDE.md** 文件。该文件定义项目约定，Claude 在每次对话中遵守。

### 生成初始文件

```bash
> /init
```

Claude 扫描项目结构（package.json、配置文件、代码组织），生成 CLAUDE.md 草稿。

### 推荐内容

```markdown
# 项目说明

## 技术栈
- 前端：React 19 + TypeScript，Vite 构建
- 后端：Node.js + Fastify
- 数据库：PostgreSQL，ORM 用 Drizzle
- 测试：Vitest + Testing Library

## 常用命令
- 开发：`npm run dev`
- 测试：`npm test`
- 类型检查：`npm run typecheck`
- 构建：`npm run build`

## 代码规范
- 使用 ES Modules（import/export），不用 CommonJS
- 所有新组件用函数组件 + Hooks，不用 Class 组件
- 变量命名用 camelCase，组件用 PascalCase
- 每个功能改动后必须跑一遍测试

## 注意事项
- 不要修改 src/legacy/ 目录下的文件（历史遗留代码）
- API 接口的返回值结构不要改（有下游依赖）
- 数据库迁移文件一旦提交不能修改，要回滚请新建迁移
```

应写与不应写：

| 应该写 | 不应该写 |
|--------|---------|
| Claude 猜不到的特殊命令 | 通用约定（如"用英文注释"）|
| 项目特有的规范 | 详细 API 文档（链接即可）|
| 禁止改动的区域 | 变化频繁的内容 |
| 架构决策的原因 | 显而易见的信息 |

CLAUDE.md 过长会被忽略，保持简洁，定期清理过时内容。

### 多层级记忆

| 文件位置 | 生效范围 | 是否提交 Git |
|---------|---------|------------|
| `./CLAUDE.md` | 当前项目（团队共享）| 提交 |
| `./CLAUDE.local.md` | 当前项目（个人）| 不提交（自动忽略）|
| `~/.claude/CLAUDE.md` | 所有项目（全局）| — |

个人偏好（如"不要自动提交代码"）放 `CLAUDE.local.md` 或全局文件，不要污染团队共享的 `CLAUDE.md`。

---

## 实战工作流

### 接手陌生项目

```bash
cd unknown-project
claude

> /init
> 帮我读懂这个项目：核心流程、主要模块、模块间依赖、
  推荐的代码阅读顺序
> 项目怎么跑起来？有什么需要注意的配置？
```

### 从报错到修复

```bash
> 运行 npm test 时报错：
  [粘贴完整报错信息]
  找出根本原因，不是只修复表面症状

> 找到原因后先告诉我改动计划

> 开始修复。改完跑一遍测试确认没有引入新问题
```

### 功能开发

```bash
> 需求：给用户列表页加按注册日期范围筛选的功能。
  先分析现有筛选逻辑，给出改动计划

（确认计划）

> 开始实现：
  - 筛选逻辑放在 src/hooks/useUserFilter.ts
  - 日期选择器用已有的 DateRangePicker 组件
  - 写对应的单元测试

（实现完成）

> 写一个 git commit message，描述所有改动
```

### 代码审查与重构

```bash
> review 最近这次提交，重点关注：
  1）SQL 注入风险
  2）未处理的边界情况
  3）可以简化的逻辑
  给出具体改进建议

> 你提到的第 2 点，帮我修复，其他我自己来
```

---

## 进阶技巧

### `-p` 模式：一次性任务

不需要交互式会话时，用 `-p`（`--print`）模式：

```bash
# 快速解释一个文件
claude -p "解释一下 src/auth.js 的实现逻辑"

# 结合管道
cat error.log | claude -p "分析这个错误日志，找出最高频的问题"

# 集成到 CI 脚本
git diff main | claude -p "生成一份这次改动的中文变更说明"
```

### `--verbose`：查看工具调用

```bash
claude --verbose
```

详细模式下可以看到 Claude 每一步的工具调用（打开哪个文件、执行什么命令、搜索什么内容），用于调试决策逻辑。

### 上下文管理

```bash
# 使用 worktree 在独立 git 工作树里处理新分支
claude -w feature-auth

# 不同任务之间清理上下文
> /clear

# 长会话里定期压缩
> /compact 专注于认证模块的问题，其他部分不重要
```

### 模型切换控制成本

```text
> /model haiku   # 简单任务，最快最省
> /model sonnet  # 日常编码，平衡性能与成本（默认）
> /model opus    # 复杂架构决策，最强但最贵
```

根据任务复杂度切换模型。

### 权限控制

```bash
# 计划模式：只分析不执行
claude --permission-mode plan

# 或在对话中按 Shift+Tab 切换
```

计划模式下 Claude 只输出分析和建议，不修改任何文件，适合探索阶段。

---

## 常见问题

**Q：Claude 改了不想改的东西**

用 `/rewind` 撤销最近的改动。或按 `Esc` 中断正在进行的操作。已保存的改动用 `git checkout` 回滚。

**Q：对话太长，Claude 开始遗忘上下文**

运行 `/compact` 压缩历史，或 `/clear` 开启新会话。每个不相关的任务之间用 `/clear` 分隔。

**Q：如何更安全地使用**

1. 重要改动前用 `--permission-mode plan` 先看计划
2. 在独立 git 分支上工作（用 `-w` 创建 worktree）
3. 改完查看 `git diff` 确认后再提交

**Q：如何查看 token 消耗**

```text
> /cost
```

---

## 总结

Claude Code 的使用路径：

```text
第一周   → 读代码、解释报错、修小 bug
第二周   → 明确约束条件，先计划再执行
第一个月 → 配置 CLAUDE.md，用深度思考处理复杂问题
持续使用 → 多智能体并行、自动化流水线、构建个人工作流
```

核心原则：将 Claude Code 作为协作对象而非单向工具。明确背景、约束和期望，在关键节点确认方向，及时纠正偏差。

---

*相关资源：*
- *官方文档：[code.claude.com](https://code.claude.com)*
- *更新日志：[anthropic.com/changelog](https://www.anthropic.com/changelog)*
