## 核心结论

构建代码生成 Agent，重点不是“让模型一次写对代码”，而是把“需求理解、上下文收集、计划生成、代码产出、执行验证、失败修复”做成闭环。对零基础读者，可以把 Agent 理解成一个会反复自查的自动开发流程；但对工程实践来说，它本质上是一个受约束的执行系统，不只是聊天模型。

一个可落地的代码生成 Agent，通常至少包含六步：

| 传统手工开发 | Agent 流程 |
| --- | --- |
| 读需求 | U: Understand，理解需求 |
| 找文件 | C: Context，收集上下文 |
| 想方案 | P: Plan，制定修改计划 |
| 手写代码 | G: Generate，生成代码与测试 |
| 手动运行 | V: Verify，做 AST 检查、沙箱执行、测试验证 |
| 出错再改 | I: Iterate，带着错误反馈继续修复 |

核心判断标准不是“模型回答得像不像工程师”，而是“验证是否通过”。只有当沙箱限制、安全规则、语法合法性和测试结果都通过时，当前回合才可以停止。否则就继续修复，或者在达到上限后交给人工。

玩具例子：收到“给登录表单加邮箱格式校验”的需求，Agent 不应该直接写代码，而应先拆解成待办，例如“定位表单组件”“确认校验库是否已存在”“补充测试”“修改提交逻辑”。然后读取项目文件，生成补丁，运行测试。如果 `pytest` 或前端测试失败，就把报错作为下一轮输入继续修复。

真实工程例子：在一个 Node.js 单体仓库里，Agent 收到“修改用户注册 API，密码长度改为至少 12 位，并补充回归测试”的任务。它会先读取 `package.json`、目标路由、服务层、测试目录，再生成修改方案，写入候选补丁，在隔离子进程中执行测试。若失败，它记录“哪个文件、哪条测试、哪种异常”后继续迭代，直到通过或超过最大轮次。

---

## 问题定义与边界

代码生成 Agent 不是“什么都能自动做”的通用程序员。先定义边界，系统才可控。这里讨论的是一个面向仓库内代码修改的 Agent，典型输入是自然语言需求，典型输出是补丁、测试结果、日志和失败原因。

它适合的任务边界通常包括：

| 维度 | 建议约束 |
| --- | --- |
| 项目类型 | 单语言或少量语言的仓库，例如 Node.js、Python Web 服务 |
| 可修改范围 | 只允许改 `src/`、`tests/`、`docs/` 等白名单目录 |
| 依赖策略 | 默认不能新增依赖，或只允许从白名单中新增 |
| 迭代上限 | 例如最多 5 次 |
| 验证方式 | 语法检查 + 单元测试 + 静态规则 |
| 失败出口 | 超过上限、越权访问、重复振荡后交由人工接管 |

“边界条件”这四个字，白话讲就是系统允许做到哪一步、不能碰什么东西、失败后怎么停。没有边界，Agent 很容易从“自动化助手”变成“自动制造风险”。

一个简单的终止条件可以写成：

$$
\text{stop} =
(\text{iteration} > \text{max\_iterations})
\lor
(\text{tests\_passed} = true)
\lor
(\text{security\_violation} = true)
$$

也可以写成更接近程序判断的形式：

$$
\text{if } i > N \text{ or passed = true then stop else continue}
$$

玩具例子：Agent 处理“把 `/api/profile` 返回字段 `nickName` 改成 `displayName`”。边界是只能修改 `src/` 下文件，最多 5 次迭代，不能访问仓库外路径。如果第 6 次还没通过测试，就通知工程师接管，而不是无限重试。

真实工程里，边界还应包括运行资源限制，例如 CPU 时间、内存、网络权限、环境变量暴露范围。因为生成代码不仅会“写错”，还可能“乱跑”。一个能写代码的系统，如果没有执行边界，本质上就是一个受限不足的自动脚本执行器。

---

## 核心机制与推导

代码生成 Agent 的核心流程可以抽象为：

$$
U \rightarrow C \rightarrow P \rightarrow G \rightarrow V \rightarrow I
$$

其中：

- `U` 是需求理解，白话讲就是把人说的话变成可执行任务。
- `C` 是上下文收集，白话讲就是找出和当前需求真正相关的代码与配置。
- `P` 是计划，白话讲就是先决定改哪里、先测什么、风险在哪。
- `G` 是生成，白话讲就是产出代码、测试和补丁。
- `V` 是验证，白话讲就是不相信模型，必须靠工具检查。
- `I` 是迭代，白话讲就是把失败结果喂回去重新修。

如果验证失败，流程不是回到开头全部重来，而是带着失败反馈进入下一轮生成：

$$
V_{\text{fail}} \rightarrow \text{feedback} \rightarrow G
$$

这套机制的关键不在“生成”，而在“反馈闭环”。因为大部分真实仓库问题不是语法层错误，而是上下文理解不完整、接口假设错误、测试覆盖不足。

### 1. U: 理解需求

Agent 先把自然语言改写成结构化任务，例如：

- 目标文件可能在哪
- 预期行为是什么
- 是否需要修改测试
- 风险是接口兼容还是数据迁移

比如“添加登录表单验证”，不能只抽成“加校验”，而应拆成“邮箱格式检查”“密码非空”“错误提示 UI”“提交流程阻断”“对应单测”。

### 2. C: 收集上下文

上下文不是“把整个仓库塞给模型”。那样成本高，而且噪声大。正确做法是按任务收集最小相关集，例如读取：

- `package.json` 或 `pyproject.toml`
- 入口文件和路由
- 被修改模块
- 已有测试文件
- 风格配置，如 ESLint、Prettier、pytest.ini

这里可以用 AST，抽象语法树，白话讲就是“把代码解析成结构化树，而不是按文本乱猜”。AST 让 Agent 能知道一个符号是函数定义、导入语句，还是对象属性，从而减少误改。

### 3. P: 制定计划

计划不是给用户看的废话，而是系统内部的约束。一个靠谱计划通常至少要回答：

- 改哪些文件
- 为什么改这些文件
- 哪些测试要新增或更新
- 成功判据是什么
- 失败后先修哪一类错误

### 4. G: 生成代码与测试

代码生成阶段最好同时生成测试，原因很简单：如果只写功能，不写验证，下一轮就没有高质量反馈。测试驱动的代码生成，本质是先把“需求”翻译成“可执行判据”。

### 5. V: 验证

验证一般分三层：

| 验证层 | 作用 |
| --- | --- |
| AST/语法检查 | 拦住最基础的语法错误 |
| 沙箱执行 | 防止越权访问、危险命令、失控运行 |
| 测试运行 | 判断功能是否满足需求 |

新手例子：Agent 修改 Node.js 项目时，先读取 `package.json` 和目标模块，再用 `acorn` 做 AST 解析，确认 JavaScript 语法合法，然后在隔离 `child_process` 里跑 `npm test`。如果报 `Cannot find module`，那不是“模型不聪明”，而是依赖假设和仓库事实不一致。

真实工程里，验证日志应结构化保存，例如：

- 第几轮生成
- 改了哪些文件
- AST 是否通过
- 测试命令和退出码
- 首个失败用例
- 是否出现越权路径
- 是否与上一轮修改重复

这样下一轮修复才有依据，而不是让模型再次盲猜。

---

## 代码实现

下面先给一个最小可运行的 Python 玩具例子，用来说明“有上限的迭代闭环”是什么。它不负责写真实代码，只演示停止条件和反馈修复逻辑。

```python
def agent_loop(max_iterations, outcomes):
    """
    outcomes: 每轮是否通过验证的列表
    """
    logs = []
    for i, passed in enumerate(outcomes, start=1):
        logs.append({"iteration": i, "passed": passed})
        if passed:
            return {"status": "success", "iterations": i, "logs": logs}
        if i >= max_iterations:
            return {"status": "handoff", "iterations": i, "logs": logs}
    return {"status": "handoff", "iterations": len(logs), "logs": logs}

result1 = agent_loop(5, [False, False, True])
assert result1["status"] == "success"
assert result1["iterations"] == 3

result2 = agent_loop(5, [False, False, False, False, False, False])
assert result2["status"] == "handoff"
assert result2["iterations"] == 5
```

这个例子说明两件事：

1. Agent 不是无限修复器，必须有上限。
2. 是否停止，取决于验证结果，而不是模型“觉得自己写对了”。

下面是一个更接近工程实践的 Node.js 片段，展示如何读取文件、做 AST 检查、启动受限测试进程并记录日志。这里的代码是简化版，但结构是真实可落地的。

```javascript
const fs = require('fs/promises');
const path = require('path');
const acorn = require('acorn');
const { spawn } = require('child_process');

async function readContext(projectRoot, files) {
  const context = {};
  for (const file of files) {
    const abs = path.join(projectRoot, file);
    context[file] = await fs.readFile(abs, 'utf8');
  }
  return context;
}

function validateJsSyntax(source, filePath) {
  try {
    acorn.parse(source, {
      ecmaVersion: 'latest',
      sourceType: 'module'
    });
    return { ok: true, filePath };
  } catch (error) {
    return {
      ok: false,
      filePath,
      message: error.message
    };
  }
}

function runTests(projectRoot) {
  return new Promise((resolve) => {
    const child = spawn('npm', ['test', '--', '--runInBand'], {
      cwd: projectRoot,
      env: {
        PATH: process.env.PATH,
        NODE_ENV: 'test'
      },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (buf) => {
      stdout += buf.toString();
    });

    child.stderr.on('data', (buf) => {
      stderr += buf.toString();
    });

    child.on('close', (code) => {
      resolve({
        ok: code === 0,
        code,
        stdout,
        stderr
      });
    });
  });
}

async function runIteration(projectRoot, candidateFile, logger) {
  const source = await fs.readFile(candidateFile, 'utf8');
  const astResult = validateJsSyntax(source, candidateFile);

  logger.push({
    stage: 'ast',
    result: astResult
  });

  if (!astResult.ok) {
    return { ok: false, reason: 'ast_failed', detail: astResult };
  }

  const testResult = await runTests(projectRoot);

  logger.push({
    stage: 'test',
    result: {
      ok: testResult.ok,
      code: testResult.code
    }
  });

  if (!testResult.ok) {
    return { ok: false, reason: 'test_failed', detail: testResult };
  }

  return { ok: true };
}
```

这段代码表达了三个工程原则。

第一，先做静态合法性检查，再运行测试。因为语法都不过，直接跑测试只会浪费时间。

第二，测试必须放进受限子进程。`child_process.spawn` 本身不是沙箱，但它是构建沙箱的执行入口。真实系统里还会叠加容器、权限裁剪、网络关闭、目录白名单和超时终止。

第三，日志必须结构化。prompt、response、AST 结果、测试退出码、错误摘要都应该保存。因为最终交付物通常不只是代码，还包括“为什么这次 patch 是可信的”。

真实工程例子：如果 Agent 在一个 API 服务仓库中修复“订单创建时缺少库存检查”的 bug，推荐流程是：

- 读取 `package.json`、库存服务、订单服务和现有测试。
- 用 AST 检查候选修改文件。
- 在隔离测试环境里执行 `npm test` 或目标测试子集。
- 若通过，创建 feature 分支并输出补丁。
- 若失败，提取首个稳定错误，例如断言失败或模块导入错误，再进入下一轮修复。

---

## 工程权衡与常见坑

代码生成 Agent 最大的误区，是把问题理解成“提示词写得不够好”。真实问题往往在工程约束上。

| 常见坑 | 典型表现 | 规避手段 |
| --- | --- | --- |
| 依赖未安装 | 生成代码引用不存在的包 | 前置依赖检查，禁止随意新增依赖 |
| 路径越权 | 访问绝对路径或仓库外目录 | 路径白名单，只允许工作区内读写 |
| 修复振荡 | 一轮修这个，下一轮又改回去 | 记录失败历史，做补丁去重和相似度检查 |
| 上下文过多 | 把整个仓库塞给模型，结果改错文件 | 基于导入链、调用链做最小检索 |
| 测试伪通过 | 测试覆盖不足，功能其实没完成 | 为关键需求补充行为测试 |
| 沙箱过弱 | 生成代码执行危险命令 | 禁网、限时、限目录、限环境变量 |

“振荡”是非常常见的问题。白话讲，就是 Agent 在两个错误解之间来回跳。比如第一轮把表单校验写在前端，测试失败；第二轮删掉前端校验，改后端；第三轮又因为 UI 测试失败，再把前端逻辑补回但把后端改坏。没有历史去重时，它可能一直重复。

玩具例子：Agent 生成了 `fs.readFile('/etc/passwd')` 这样的代码来“读取配置”。这不是普通 bug，而是越权访问。正确策略不是“提醒模型注意安全”，而是让路径检查器直接拒绝，并把错误写入日志。如果相同类型越权连续出现超过阈值，例如 2 次，就停止局部 patch，改为重新生成整体方案，或者交人工审批。

真实工程里，成本也必须算清楚。每轮 AST 检查、沙箱启动、测试运行都会增加耗时。若仓库测试需要 12 分钟，5 轮就是 1 小时级别。此时更合理的做法通常是：

- 先跑受影响测试子集
- 最后一轮再跑全量测试
- 对高风险模块要求人工审批
- 把最大迭代次数降到 3 而不是盲目设 10

所以，Agent 不是“自动化越多越好”，而是“在验证成本和错误风险之间找平衡”。

---

## 替代方案与适用边界

单 Agent 不是唯一方案。不同团队、不同任务复杂度，适合的自动化形态不一样。

| 方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 单 Agent | 中小型仓库、明确需求、低到中等风险修改 | 架构简单，落地快 | 容易把规划、编码、验证耦合在一起 |
| 多 Agent | 复杂任务、需要更强分工 | 可拆成规划、编码、测试、审查多个角色 | 协调成本更高，日志更复杂 |
| 纯人类 TDD | 高风险业务、核心链路、强合规环境 | 可解释性强，风险最低 | 开发速度慢，自动化程度低 |
| Code Runner 型工具 | 只关心执行，不关心改代码策略 | 简单直接 | 缺乏计划和修复能力 |

多 Agent 是一种常见替代方案。比如把系统拆成：

- `Planner Agent` 负责任务分解
- `Coder Agent` 负责生成实现
- `Test Agent` 负责补测试和解释失败
- `Reviewer Agent` 负责检查风险

这类设计在研究中很常见，例如 AgentCoder 就把代码生成、测试设计和测试执行拆成不同角色。它的核心价值不是“更多模型一起工作”，而是“把反馈职责独立出来”，避免写代码的角色同时给自己判卷。

玩具例子：一个 Agent 写登录接口，另一个专门写测试。测试 Agent 先给出“错误密码应返回 401”“空邮箱应返回 400”。编码 Agent 再根据这些测试生成实现。这样比单 Agent 自己写实现、自己猜验证条件更稳。

真实工程边界上，以下场景不适合全自动放行：

- 大型遗留系统，模块耦合重，上下文收集代价高
- 金融、医疗、权限控制等安全敏感代码
- 涉及数据库迁移、基础设施变更、跨服务协议升级
- 无测试或测试高度不可信的仓库

在这些场景里，更合理的模式通常是“Agent 生成候选补丁 + 人工审批 + 受控执行”。也就是说，Agent 适合当加速器，不适合在高风险区域当最终裁决者。

---

## 参考资料

| 资料 | 用途 | 重点结论 |
| --- | --- | --- |
| [Code Generation Agents: Architecture and Implementation](https://www.grizzlypeaksoftware.com/library/code-generation-agents-architecture-and-implementation-75u4ubg7) | 工程化架构参考 | 给出需求理解、上下文收集、生成、沙箱验证、迭代修复的完整 Node.js 实现思路 |
| [AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation](https://arxiv.org/abs/2312.13010) | 多 Agent 分工参考 | 将编程、测试设计、测试执行拆分，证明测试反馈可显著提升代码生成效果 |
| [Test-Driven Development and LLM-based Code Generation](https://conf.researchr.org/details/ase-2024/ase-2024-research/127/Test-Driven-Development-and-LLM-based-Code-Generation) | TDD 与代码生成结合参考 | 在问题描述之外加入测试，有助于提升生成代码的功能正确性 |
| [Blueprint2Code: a multi-agent pipeline for reliable code generation via blueprint planning and repair](https://www.frontiersin.org/articles/10.3389/frai.2025.1660912/full) | 规划先行的多 Agent 架构参考 | 先做 blueprint，再编码和修复，能降低直接生成带来的结构性错误 |
| [CODESIM: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging](https://paperswithcode.com/paper/codesim-multi-agent-code-generation-and-1) | 规划验证与调试参考 | 在生成前做计划模拟，说明“先验证思路再写代码”对复杂任务更重要 |

如果要把现有单 Agent 系统继续扩展，最值得借鉴的是这些资料中的 feedback loop 思路：不要只把测试当最终验收，而要把失败信息结构化成下一轮输入的一部分，例如“失败测试名、断言差异、调用栈摘要、变更文件列表”。这样迭代不是重复生成，而是基于证据修复。
