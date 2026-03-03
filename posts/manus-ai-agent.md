## Manus 是什么

Manus 是由 Monica（蝴蝶效应）于 2025 年 3 月 6 日发布的通用 AI Agent，运行在云端沙盒环境中，可以自主完成从市场调研、数据分析到软件开发的端到端任务。

它与 ChatGPT、Claude 等对话型 AI 的本质区别：**对话型 AI 返回文本，Manus 返回结果**。用户描述目标，Manus 自主规划路径、调用工具、执行操作，直到任务完成。

---

## 执行环境：沙盒 Ubuntu

Manus 的每次任务运行在独立的云端 Ubuntu 沙盒中，内置完整的工具链：

- 浏览器（Playwright 驱动的无头 Chrome）
- Python 运行时
- Shell 终端
- 文件系统读写
- 外部 API 调用

沙盒的隔离性保证了任务执行的安全边界，同时提供了远超"浏览器内操作"的能力范围。这是 Manus 与 OpenAI Operator 等仅限浏览器操作的 Agent 的关键差异。

---

## CodeAct：用 Python 代码作为行动格式

Manus 的行动机制基于 **CodeAct 范式**（来自 2024 年的同名研究论文）：Agent 的每个动作不是调用预定义的 JSON 工具接口，而是生成并执行一段 Python 代码。

对比两种行动格式：

```json
// 传统工具调用格式
{
  "tool": "browser_click",
  "params": { "selector": "#submit-btn" }
}
```

```python
# CodeAct 格式：生成可执行代码
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    page = p.chromium.launch().new_page()
    page.goto("https://example.com")
    page.click("#submit-btn")
```

CodeAct 的优势：代码可以自由组合控制流、处理异常、复用函数，表达能力远超固定工具调用；同时代码本身即文档，执行轨迹可审查。

---

## 多 Agent 架构

Manus 的内部由三类 Agent 协同完成任务：

| Agent | 职责 |
|-------|------|
| **Planner** | 将用户目标拆解为有序子任务，写入 `todo.md`，并在执行中动态调整计划 |
| **Executor** | 执行具体操作：生成 Python 代码、调用工具、与环境交互 |
| **Verifier** | 对 Executor 的输出做正确性验证，失败时触发重新规划 |

底层模型：Planner 和 Executor 使用 Anthropic Claude 3.5/3.7 Sonnet，部分子任务使用阿里 Qwen 的微调版本。不同子任务动态选择最适合的模型。

`todo.md` 是 Manus 的"工作记忆"：任务开始时由 Planner 生成，Executor 执行完每个步骤后更新状态，Verifier 验证后标记完成。这个文件同时作为上下文传递给后续步骤，解决了长任务中的上下文衔接问题。

---

## Agent Loop

Manus 的执行遵循单步一动的循环：

```
while 任务未完成:
    1. 读取当前状态（环境 + todo.md + 事件流）
    2. 规划下一步动作
    3. 执行一个动作（one action per iteration）
    4. 观察执行结果
    5. 更新 todo.md
```

"one action per iteration" 是刻意的设计约束：每次循环只执行一个原子操作，强制 Agent 在每步之后观察环境反馈。这比"一次性生成所有步骤"的方案更鲁棒——环境状态在执行中会变化，早期规划的后续步骤往往需要根据实际结果调整。

---

## 记忆系统

Manus 的记忆分四层：

| 层次 | 实现 | 作用域 |
|------|------|--------|
| 工作记忆 | 事件流（event stream） | 当前会话 |
| 任务状态 | `todo.md` 文件 | 当前任务 |
| 持久存储 | 文件系统中间结果 | 跨步骤 |
| 知识检索 | 向量数据库 + RAG | 跨任务 |

长任务（超出上下文窗口）依赖文件系统作为外部记忆：Executor 将中间结果写入文件，后续步骤从文件读取，而不是依赖上下文中的历史信息。

---

## 基准测试：GAIA

GAIA（General AI Assistants benchmark）评测通用 AI Agent 在真实任务上的表现，分 Level 1-3 三个难度级别。Manus 在 2025 年 3 月的公开榜单上：

| 系统 | Level 1 | Level 2 |
|------|---------|---------|
| Manus | **86.5%** | **70.1%** |
| OpenAI Deep Research | 74.3% | — |
| 此前 SOTA | 67.9% | — |

GAIA 的 Level 1 任务示例：查找特定信息、执行多步网页操作、整合多来源数据输出报告。Level 2 开始涉及需要较长规划链的复合任务。

---

## 与同类系统的定位差异

| 系统 | 执行环境 | 行动粒度 | 典型用途 |
|------|---------|---------|---------|
| Manus | 云端 Ubuntu 沙盒 | Python 代码 | 端到端复杂任务 |
| OpenAI Operator | 浏览器 | 点击/输入 | 网页操作自动化 |
| Claude Code | 本地终端 | Shell + 文件编辑 | 代码库开发 |
| Devin | 云端 VM | Shell + IDE | 软件工程任务 |

Manus 的定位是"通用"——不局限于编码或浏览器，覆盖需要跨多类工具协作的任务。代价是延迟较高（复杂任务需要多轮循环），不适合需要即时响应的场景。

---

## OpenManus：开源复现

Manus 发布后数日，MetaGPT 团队发布了 **OpenManus**，在 GitHub 上获得 4 万 Star。OpenManus 复现了核心架构：

- 基础模型：CodeActAgent（Mistral 7B 微调）
- 工具链：LangChain + Playwright
- 沙盒：Docker 容器
- 规划：`todo.md` 模式

OpenManus 验证了 Manus 架构的可复现性，也说明其核心创新不在模型本身，而在工具集成和 Agent Loop 的工程实现上。

---

## 参考资料

- [Manus 官方博客](https://manus.im/blog)
- [From Mind to Machine: The Rise of Manus AI as a Fully Autonomous Digital Agent](https://arxiv.org/abs/2505.02024)
- [CodeAct: Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030)
- [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)
- [OpenManus GitHub](https://github.com/mannaandpoem/OpenManus)
