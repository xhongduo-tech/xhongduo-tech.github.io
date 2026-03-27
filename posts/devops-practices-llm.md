## 核心结论

DevOps 在大模型服务中的核心作用，不是“把部署做快一点”，而是把代码、模型、基础设施、配置、测试、监控都变成可版本化、可审计、可重复执行的工件。可版本化，意思是每一次变更都能在 Git 或制品库里找到具体版本；可审计，意思是你能追溯“是谁、在什么时候、改了什么”；可重复执行，意思是同一套定义在不同环境里能得到尽量一致的结果。

如果把大模型服务当作普通软件来发布，一个最小可行流程通常就是 `build -> test -> deploy`。构建阶段产出镜像、模型包或部署清单；测试阶段验证代码逻辑、接口联通、推理输出是否满足约束；部署阶段把通过验证的版本推到测试环境、预发环境或生产环境。这个流程的总时长可以写成：

$$
T_{\text{部署}} = T_{\text{构建}} + T_{\text{测试}} + T_{\text{交付}}
$$

结论很直接：只要这三段是自动化的，发布就能稳定缩短；只要其中任何一段依赖人工记忆，系统就会重新退化成“手工上线”。

一个适合新手理解的例子是：团队把源码、配置文件、Terraform 脚本、测试脚本都放进同一个 Git 仓库。每次 `git push` 后，GitHub Actions 自动构建镜像、跑 `pytest`、执行部署脚本，任何一步失败都立即停止并通知开发者。这里的 CI/CD 流水线，就是“一套自动执行的软件装配线”。

| 阶段 | 主要输入 | 主要输出 | 常见触发器 |
| --- | --- | --- | --- |
| Build | 源码、依赖、Dockerfile、模型文件引用 | 镜像、制品、SBOM、版本号 | `push`、`merge`、打 tag |
| Test | 构建产物、测试脚本、测试数据 | 测试报告、覆盖率、质量门禁结果 | Build 成功后自动触发 |
| Deploy | 镜像、配置、IaC 脚本、密钥引用 | 已部署服务、发布记录、回滚点 | Test 通过、手工审批、定时任务 |

---

## 问题定义与边界

这篇文章讨论的“自动化部署与持续集成”，边界是大模型在线服务的工程发布，不讨论模型训练算法本身。重点对象包括四类工件：

| 对象 | 具体内容 | 为什么必须版本化 |
| --- | --- | --- |
| 代码 | API 服务、推理逻辑、预处理代码 | 便于回滚与追踪功能变更 |
| 模型 | 权重版本、量化版本、推理适配层 | 避免“模型文件是谁替换的”这种黑箱 |
| 基础设施 | Terraform、Ansible、Kubernetes 清单 | 保证环境一致，减少手工改配置 |
| 监控与告警 | Prometheus 规则、告警阈值、Dashboard 定义 | 监控本身也会影响运维结果 |

问题的根源通常不是“不会部署”，而是部署过程不可重复。常见现象包括：

| 当前做法 | 直接后果 | DevOps 理想状态 |
| --- | --- | --- |
| 手工登录服务器改配置 | 改动不可追踪 | 配置文件进入版本库 |
| 测试环境和生产环境手工搭建 | 环境漂移，意思是两个环境逐渐变得不一样 | 用 IaC 统一创建 |
| 部署靠个人经验执行命令 | 人一换就失效 | 流水线自动执行 |
| 发布成功与否靠群里口头确认 | 反馈慢且模糊 | 测试、日志、告警自动回传 |

玩具例子可以这样看。一个小团队维护一个问答机器人 API，初期做法是：开发者把新镜像打好后手工 `scp` 到服务器，再 SSH 上去改 `.env`，最后重启容器。上线失败后，没人记得是镜像版本错了、环境变量漏了，还是端口映射没改。改成 DevOps 后，镜像标签、环境变量模板、部署脚本都进入仓库，流水线自动完成构建和发布，问题立刻从“靠猜”变成“看日志定位”。

这里要明确一个边界：DevOps 不能替代需求评审，也不能保证模型回答质量天然正确。它解决的是交付可靠性，不是业务正确性本身。换句话说，DevOps 保证“按定义上线”，但不替你决定“定义本身是否合理”。

---

## 核心机制与推导

持续集成，英文是 Continuous Integration，白话讲就是“每次代码改动都尽早并入主线并自动验证”；持续交付或持续部署，白话讲就是“验证通过后，自动把系统送到目标环境”。这两个概念合起来，才构成常说的 CI/CD。

它的机制可以拆成三层。

第一层是 pipeline as code，也就是“把流水线本身写成代码”。GitHub Actions 用 YAML，GitLab CI 用 `.gitlab-ci.yml`，Jenkins 常用 `Jenkinsfile`。这样做的价值不是格式统一，而是让“发布流程”也接受版本控制、代码评审和回滚。

第二层是工件传递。Build 阶段生成的镜像、模型包、配置模板，不应该在后续阶段重新猜测或重新生成，而应作为明确产物传给 Test 和 Deploy。否则你在测试里验证的版本，未必就是最终部署的版本。

第三层是可观察性与失败控制。可观察性，意思是系统运行状态可被度量和查看；在流水线里，它体现为日志、测试报告、耗时、告警、回滚记录。失败控制包括重试、超时、门禁和人工审批。比如生产部署前要求一名负责人审批，就是把风险显式化。

部署时间公式并不复杂：

$$
T_{\text{部署}} = T_{\text{构建}} + T_{\text{测试}} + T_{\text{交付}}
$$

但它带来一个工程判断：如果你只盯着“部署命令执行得快不快”，通常抓不到主要矛盾。真正拖慢迭代的，往往是测试设计不合理，或者构建与部署中重复做了本可缓存的工作。

例如一个玩具流水线里：

- 构建镜像 8 分钟
- 运行单元测试和集成测试 5 分钟
- 部署到测试环境并做健康检查 3 分钟

那么总时长就是 $8 + 5 + 3 = 16$ 分钟。4 小时是 240 分钟，理论上最多可以完成 15 次这样的完整迭代。这说明自动化的价值不仅是省人力，还直接决定单位时间内能完成多少有效试错。

一个更接近真实工程的例子是大模型 API 服务。Build 阶段负责构建推理服务镜像、下载固定版本模型权重、生成镜像标签；Test 阶段先跑单元测试，再跑集成测试验证“API -> 推理服务 -> 向量库或缓存”的链路，最后抽样跑端到端测试检查关键问答质量；Deploy 阶段用 Terraform 创建或更新云资源，再用 Ansible 或 Kubernetes 清单完成应用部署，最后通过 Prometheus 指标确认错误率、延迟和 GPU 利用率是否异常。

下面是一个极简的流水线结构示意：

```yaml
name: llm-service-ci

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t my-llm:${{ github.sha }} .

  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Apply infra
        run: terraform apply -auto-approve
```

这个结构的重点不是工具，而是依赖关系：`test` 依赖 `build`，`deploy` 依赖 `test`。前一阶段失败，后一阶段不能继续。这就是质量门槛，意思是“没有通过门禁，就不允许往前走”。

---

## 代码实现

在实际工程里，最常见的做法是把“快速反馈”和“复杂交付”拆开。

GitHub Actions 适合做仓库内触发的快速 CI，比如代码检查、镜像构建、单元测试。GitLab CI 在多项目、多环境编排方面更强，适合把基础设施变更、环境部署、审批流程编进流水线。Jenkins 则常用于存量系统，或者需要长时间运行的回归任务，比如 2 小时的批量评测。

先看一个适合新手的 GitHub Actions 示例：

```yaml
name: deploy-llm-service

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t registry.example.com/llm:${{ github.sha }} .

  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Unit and integration tests
        run: pytest -q

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Terraform deploy
        run: terraform apply -auto-approve
```

再看一个更接近多环境管理的 GitLab CI 结构：

```yaml
stages:
  - build
  - test
  - deploy

build_image:
  stage: build
  script:
    - docker build -t registry.example.com/llm:$CI_COMMIT_SHA .
  artifacts:
    paths:
      - image_tag.txt

run_tests:
  stage: test
  needs: ["build_image"]
  script:
    - pytest -q
    - python smoke_test.py

deploy_prod:
  stage: deploy
  needs: ["run_tests"]
  script:
    - terraform init
    - terraform apply -auto-approve
```

如果要让新手真正理解“阶段顺序为什么可靠”，最简单的方法是写一个小程序模拟流水线调度。下面这个 Python 代码块可以直接运行：

```python
from dataclasses import dataclass

@dataclass
class Stage:
    name: str
    duration: int
    success: bool = True

def run_pipeline(stages):
    total = 0
    executed = []
    for stage in stages:
        if not stage.success:
            executed.append((stage.name, "failed"))
            return total, executed
        total += stage.duration
        executed.append((stage.name, "passed"))
    return total, executed

stages = [
    Stage("build", 8),
    Stage("test", 5),
    Stage("deploy", 3),
]

total, executed = run_pipeline(stages)
assert total == 16
assert executed[-1] == ("deploy", "passed")

broken = [
    Stage("build", 8),
    Stage("test", 5, success=False),
    Stage("deploy", 3),
]
total2, executed2 = run_pipeline(broken)
assert total2 == 8
assert executed2[-1] == ("test", "failed")
assert ("deploy", "passed") not in executed2

print(total, executed)
print(total2, executed2)
```

这个例子的意义很直接：测试失败时，总耗时只累计到失败点，部署不会执行。真实 CI/CD 平台就是把这类依赖关系工程化了。

不同工具的适用场景可以先按这个表理解：

| 方案 | 优势 | 短板 | 适用场景 |
| --- | --- | --- | --- |
| GitHub Actions | 上手快，和 GitHub 仓库集成紧密 | 多团队大规模编排能力一般 | 单仓库、初创团队、快速验证 |
| GitLab CI | 仓库、制品、环境、权限一体化较强 | 学习与治理成本略高 | 多环境、多项目、平台化团队 |
| Jenkins | 插件多、历史包袱系统兼容性强 | 维护成本高，配置不规范易失控 | 存量企业系统、长时间回归任务 |

真实工程例子通常不是“三选一”，而是组合。比如：GitHub Actions 负责开发者每次提交后的快速检查；GitLab CI 负责跨仓库环境部署与 IaC 编排；Jenkins 负责夜间长时回归和性能基线测试。这样拆分的原则是：谁更适合承担哪种时延和复杂度，就让谁承担，不必强求统一到单一工具。

---

## 工程权衡与常见坑

最常见的问题不是工具选错，而是把自动化做成“半自动”。半自动的意思是：看起来有流水线，但关键步骤仍靠人补。这样一来，风险并没有消失，只是被隐藏了。

常见坑与缓解方式如下：

| 问题 | 具体表现 | 缓解措施 |
| --- | --- | --- |
| 环境漂移 | 测试环境可用，生产环境报缺依赖或配置错误 | 使用 IaC，统一环境定义 |
| 只跑端到端测试 | 反馈慢，一次提交等几十分钟甚至几小时 | 建立测试金字塔，更多依赖单元与集成测试 |
| Jenkins 任务散乱 | job 名称混乱、脚本分散、依赖不清 | 用共享库和多分支流水线治理 |
| 配置不入库 | 密钥之外的大量配置靠口头同步 | 配置模板、变量文件进入版本库 |
| 失败日志不可读 | 出错后只有“Job failed” | 输出结构化日志和明确阶段报告 |

测试金字塔，白话讲就是“底层便宜的测试多一些，顶层昂贵的测试少一些”。如果所有验证都押在 E2E，也就是端到端测试上，流水线就会越来越慢。一个更合理的比例通常是：大量单元测试负责快速发现局部逻辑错误，少量集成测试验证系统模块之间的连接，再用更少的端到端测试覆盖最关键的业务路径。

一个典型缺陷案例是：测试环境由开发手工搭，生产环境由运维手工搭。模型服务在测试环境里请求正常，但发布到生产后因为缺少 GPU 驱动版本匹配或环境变量命名不同而启动失败。改进方式不是“要求大家更细心”，而是把机器、网络、密钥引用、配置模板都写进 Terraform 和部署流水线，做到同一套定义生成多个环境。

Terraform 模块复用的简化示意如下：

```hcl
module "llm_service_prod" {
  source        = "./modules/llm_service"
  env           = "prod"
  replica_count = 3
  image_tag     = var.image_tag
}
```

这段配置的重点不在 HCL 语法，而在模块化。模块化，白话讲就是“把重复建设收进一个可复用单元”。如果测试、预发、生产三套环境都从同一个模块生成，你出错的概率会显著下降。

还有一个容易被低估的问题是“回滚策略”。很多团队只自动化了发布，没有自动化回滚。结果一旦生产指标恶化，大家又回到手工救火。更稳妥的做法是：每次部署保留明确的上一版本镜像标签、上一版配置快照和数据库迁移策略，并把回滚也纳入流水线或至少形成标准脚本。

---

## 替代方案与适用边界

对小团队或原型阶段，不需要一开始就把链路做得很重。一个单仓库配合 GitHub Actions，已经足够覆盖大部分“代码提交 -> 自动测试 -> 自动部署到测试环境”的需求。此时最重要的不是工具数量，而是先把版本控制、自动测试、可重复部署建立起来。

当团队进入多模型、多环境、多集群阶段，单工具往往开始吃力。此时更适合采用分层方案：GitHub Actions 负责轻量开发反馈，GitLab CI 负责环境与发布编排，Terraform 管基础设施，Jenkins 保留给长时回归、兼容性验证或遗留系统集成。分层的意义是降低单点复杂度，而不是堆工具。

| 方案 | 团队规模 | 系统复杂度 | 适用边界 |
| --- | --- | --- | --- |
| 单工具方案 | 1 到 5 人 | 单仓库、单集群、单环境为主 | 初创项目、原型阶段 |
| GitHub Actions + IaC | 3 到 10 人 | 有测试、预发、生产环境 | 常规互联网服务 |
| GitLab CI + Terraform + Jenkins | 10 人以上或多团队 | 多仓库、多集群、审批与合规要求高 | 平台化、企业级交付 |

一个适用边界的例子是：某模型服务每次迭代流水线耗时 16 分钟，其中构建 8 分钟、测试 5 分钟、部署 3 分钟。4 小时可完成约 15 次迭代，这很适合频繁调整 prompt、路由策略、服务配置的在线产品。但如果你需要每晚跑一次 2 小时的大规模回归评测，比如验证 2 万条样本集上的回答稳定性，那么这类任务不适合阻塞每次提交的主流水线，通常更适合作为 Jenkins 定时作业或独立质量门禁。

人工干预并不是坏事，关键是只把它放在真正高风险的位置。例如：

- 生产环境发布前需要负责人审批
- 涉及数据库不可逆迁移时需要人工确认
- 大规模模型切换时需要人工观察核心指标 10 到 30 分钟

这类人工步骤应当是“显式的门禁”，而不是“靠人临场记得做什么”。显式门禁仍然属于自动化体系的一部分，因为它被流水线定义、记录和追踪。

---

## 参考资料

| 来源 | 重点摘要 | URL |
| --- | --- | --- |
| TechRadar: Breaking silos, unifying DevOps and MLOps into a unified software supply chain | 强调代码、模型、基础设施和监控应统一纳入软件供应链治理 | https://www.techradar.com/pro/breaking-silos-unifying-devops-and-mlops-into-a-unified-software-supply-chain |
| GitHub Docs: GitHub Actions | 说明如何用 workflow 把构建、测试、部署写成代码并由事件触发 | https://docs.github.com/actions |
| GitLab Docs: GitLab CI/CD | 介绍 `stages`、`needs`、制品传递和环境部署等核心能力 | https://docs.gitlab.com/ci/ |
| GitHub Blog: Getting started with DevOps automation | 从工程实践角度说明自动化构建、发布和反馈闭环的价值 | https://github.blog/enterprise-software/devops/getting-started-with-devops-automation/ |
| Harness: IaC in CI/CD Pipelines Best Practices | 讨论基础设施即代码如何与流水线集成，以及模块化复用与治理建议 | https://www.harness.io/harness-devops-academy/iac-in-ci-cd-pipelines-best-practices |
| CloudOps Innovation: CI/CD Pipeline Best Practices | 汇总常见流水线问题，如环境漂移、测试过重、日志不可追踪等 | https://cloudopsinnovation.com/blog/ci-cd-pipeline-best-practices |
