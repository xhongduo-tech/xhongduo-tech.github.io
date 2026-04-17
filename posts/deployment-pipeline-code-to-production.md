## 核心结论

部署流水线，是把“代码改动”稳定地变成“线上可用版本”的自动化路径。它最核心的价值不是省几次点击，而是把发布过程从“靠人记步骤”改成“靠系统执行规则”。对初级工程师来说，可以先记住一个最小模型：代码提交、构建、测试、部署、验证，五步缺一不可。

用集合表示，可以写成：

$$
P = \{C, B, T, D, V\}
$$

其中，$C$ 是 Code Commit，指代码进入版本库；$B$ 是 Build，指把源码变成可运行产物；$T$ 是 Test，指自动化检查；$D$ 是 Deploy，指把新版本放到目标环境；$V$ 是 Verify，指确认新版本真的正常工作。

部署流水线的关键不是“自动”，而是“可重复、可追踪、可回滚”。如果同一份提交每次都能产出同一份构建物，测试结果可查，部署步骤固定，验证规则明确，那么线上事故会明显减少。反过来，只要其中一个环节依赖人工口头传递，系统就会退化成“半自动碰运气”。

一个新手容易理解的玩具例子是：开发者 `git push` 后，GitHub Actions 自动执行 4 个阶段，构建 3 分钟，测试 5 分钟，部署 1 分钟，验证 2 分钟，总耗时 11 分钟。测试阶段占比为：

$$
\frac{5 + 2}{11} \approx 64\%
$$

这里把验证也算进“质量检查时间”，因为验证本质上是在生产前后继续确认质量。这个比例说明一个事实：流水线大部分时间不是在“发布”，而是在“建立发布可信度”。

| 阶段 | 输入产物 | 输出产物 | 常用工具 |
| --- | --- | --- | --- |
| Code | 提交记录、分支、PR | 可追踪版本号、变更集 | GitHub、GitLab |
| Build | 源码、依赖、配置 | artifact，白话说就是可复用的构建产物 | GitHub Actions、Jenkins |
| Test | artifact、测试脚本、测试数据 | 测试报告、覆盖率、质量门禁结果 | GitHub Actions、GitLab CI、Jenkins |
| Deploy | 镜像、压缩包、部署脚本 | 新环境中的运行实例 | CircleCI、Argo CD、kubectl |
| Verify | 服务地址、监控指标、探针脚本 | 健康结论、告警、回滚信号 | Prometheus、Grafana、curl |

---

## 问题定义与边界

部署流水线要解决的问题很具体：手动发布不可重复，也不透明。一个人本地打包、另一个人拷文件、第三个人改配置，这种流程短期能跑，长期一定会出问题，因为没人能稳定回答三个问题：线上到底跑的是哪份代码、它是怎么上去的、出错时该回滚到哪里。

所谓“不可重复”，就是同一个版本在不同人电脑上、不同时间、不同命令顺序下，结果可能不一样。所谓“不透明”，就是没有统一日志、没有统一状态、没有统一产物。流水线的作用，就是给每一步加上明确输入、明确输出和明确状态。

形式化一点，可以把它写成有依赖的阶段序列：

$$
C \rightarrow B \rightarrow T \rightarrow D \rightarrow V
$$

并且每一步都满足：

$$
output_i = f(input_i, rules_i)
$$

意思是，每个阶段的输出都由输入和规则决定，而不是靠操作者临场决定。这也是为什么成熟团队强调 artifact。artifact 可以理解成“阶段间传递的标准包”，比如 Docker 镜像、编译后的二进制、压缩后的前端静态文件。它的意义是：测试阶段测的，就是部署阶段用的，不允许“测试一个包，部署另一个包”。

边界也要讲清楚。本文讨论的是“从代码提交到生产验证”的交付路径，不讨论需求设计、架构评审、用户反馈闭环，也不展开模型训练流水线。它关注的是上线过程，而不是产品研发全过程。

一个对比很直观：

| 方式 | 是否有统一日志 | 是否有固定产物 | 是否容易回滚 | 风险 |
| --- | --- | --- | --- | --- |
| 手动发布 | 弱 | 弱 | 差 | 步骤漏执行、版本混乱 |
| 半自动脚本 | 中 | 中 | 中 | 依赖执行人习惯 |
| 完整流水线 | 强 | 强 | 强 | 前期搭建成本更高 |

“手动发布无法追踪日志，流水线则在每阶段记录 artifact 和状态标签”，这句话可以落到具体对象上。比如一次发布会生成：

```json
{
  "commit": "a1b2c3d",
  "artifact": "web-2026.03.27.1.tar.gz",
  "test_status": "passed",
  "deploy_status": "staging_ok",
  "verify_status": "healthy"
}
```

这个 JSON 不重要，重要的是它表达的思想：流水线不是一串脚本，而是一条状态链路。

---

## 核心机制与推导

流水线之所以可靠，靠的是两类东西同时存在：共享制品和状态门禁。

共享制品前面提过，就是 artifact。它解决“上游和下游看到的是不是同一个版本”的问题。状态门禁，是指某一步没有达到规则，就不允许进入下一步。门禁这个词可以直白理解成“通行条件”。例如单元测试没过，部署 job 不启动；部署后健康检查失败，流量不切换。

因此，一个简化的推导可以写成：

$$
D = g(B, T, strategy)
$$

也就是部署阶段的行为，不只依赖构建结果 $B$，也依赖测试结果 $T$ 和部署策略 `strategy`。再进一步，最终是否放量，要由验证结果决定：

$$
release = 
\begin{cases}
continue, & \text{if } V = healthy \\
rollback, & \text{if } V = unhealthy
\end{cases}
$$

这里的部署策略很关键。它决定“新版本以什么方式接触真实流量”。常见策略如下：

| 策略 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 滚动更新 | 多实例服务，允许逐台替换 | 资源开销小 | 版本混跑，排障复杂 |
| 蓝绿部署 | 对稳定性要求高，环境可双份维护 | 切换快，回滚直接 | 成本高，需要两套环境 |
| 金丝雀发布 | 需要小流量试错 | 风险控制细，适合线上观察 | 指标设计复杂 |
| A/B 测试 | 需要比较用户行为差异 | 可验证业务效果 | 不只是发布，还涉及实验设计 |

玩具例子可以这样理解。假设一个服务有 10 台机器，滚动更新一次替换 2 台。如果前 2 台的错误率没有上升，再替换后续机器。这本质上是在控制每次暴露给用户的风险面。若用蓝绿部署，则是先准备一整套新环境，验证通过后把流量从旧环境切到新环境。切流量可以理解成“让用户请求改走另一条路”。

真实工程例子更典型。一个面向全球用户的后端服务，先在新加坡区部署绿色环境，再只给 5% 流量观察 15 分钟。如果 Prometheus 中的错误率、P95 延迟、核心业务转化率没有恶化，再扩大到 25%、50%、100%。这里常见做法其实是“蓝绿部署承载环境切换，金丝雀发布控制流量比例”，两者并不冲突。

下面这个伪代码展示了 artifact 传播和状态更新：

```json
{
  "pipeline_id": "p-20260327-001",
  "commit": "a1b2c3d",
  "artifact": {
    "name": "api-image",
    "tag": "a1b2c3d"
  },
  "stages": [
    {"name": "build", "status": "passed"},
    {"name": "test", "status": "passed"},
    {"name": "deploy", "strategy": "blue-green+canary", "status": "running"},
    {"name": "verify", "status": "pending"}
  ]
}
```

这里最重要的机制有三条。

第一，阶段必须消费同一份产物，而不是各自重新生成。否则测试阶段通过，不代表部署的是同一版本。

第二，阶段状态必须可机读。机读，白话说就是系统能自动判断，不靠人读截图。只有这样，失败时才能自动阻断、自动通知、自动回滚。

第三，验证不能只看“进程活着”。真正有效的验证至少包括健康探针、关键接口、核心指标三个层次。进程活着不等于服务可用，服务可用也不等于业务正常。

---

## 代码实现

下面给一个最小可运行的思路。先用 Python 模拟流水线状态转移，再给一个 GitHub Actions 配置片段。Python 代码的目的不是替代 CI 工具，而是把“只有前一步成功，后一步才执行”的规则写清楚。

```python
from dataclasses import dataclass

@dataclass
class Stage:
    name: str
    status: str = "pending"

def run_pipeline(build_ok: bool, test_ok: bool, deploy_ok: bool, verify_ok: bool):
    stages = {
        "build": Stage("build"),
        "test": Stage("test"),
        "deploy": Stage("deploy"),
        "verify": Stage("verify"),
    }

    stages["build"].status = "passed" if build_ok else "failed"
    if stages["build"].status != "passed":
        return stages

    stages["test"].status = "passed" if test_ok else "failed"
    if stages["test"].status != "passed":
        return stages

    stages["deploy"].status = "passed" if deploy_ok else "failed"
    if stages["deploy"].status != "passed":
        return stages

    stages["verify"].status = "passed" if verify_ok else "failed"
    return stages

ok = run_pipeline(True, True, True, True)
assert ok["build"].status == "passed"
assert ok["verify"].status == "passed"

bad = run_pipeline(True, False, True, True)
assert bad["test"].status == "failed"
assert bad["deploy"].status == "pending"
```

这段代码表达的是：

$$
build\_outcome \rightarrow test\_input \rightarrow deploy\_input \rightarrow verify\_input
$$

也就是前一个 job 的输出状态，决定后一个 job 是否有资格开始。

对应到 GitHub Actions，可以写成这样：

```yaml
name: deploy-pipeline

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build artifact
        run: |
          mkdir -p dist
          echo "release package" > dist/app.txt
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: build-output
          path: dist/

  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build-output
          path: dist/
      - name: Run tests
        run: test -f dist/app.txt

  deploy:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - name: Deploy to staging
        run: ./scripts/deploy.sh staging

  verify:
    runs-on: ubuntu-latest
    needs: deploy
    steps:
      - name: Verify health
        run: curl -f https://staging.example.com/health
```

这个配置的重点不是语法，而是依赖关系：`test needs build`，`deploy needs test`，`verify needs deploy`。这就是最小流水线骨架。

一个真实工程例子可以再具体一点。假设你维护的是一个推荐服务：

1. 开发者提交代码到 `main`。
2. `build` job 构建 Docker 镜像并打上 commit tag。
3. `test` job 执行单元测试、接口测试、静态扫描。
4. `deploy` job 把镜像先发到 `staging`，再通过 Helm 或 `kubectl` 做灰度。
5. `verify` job 用 `curl` 检查 `/health`，再查 Prometheus 指标是否越过阈值。
6. 若错误率超过门限，自动触发回滚脚本。

阶段和命令可以整理成速查表：

| 阶段 | 常见脚本/命令 | 输出 |
| --- | --- | --- |
| Build | `docker build`、打包脚本 | 镜像、压缩包 |
| Test | `pytest`、接口测试脚本、扫描工具 | 测试报告、扫描报告 |
| Deploy | `kubectl apply`、Helm、发布脚本 | 新版本实例 |
| Verify | `curl`、Prometheus 查询、日志检查脚本 | 健康结果、回滚信号 |

---

## 工程权衡与常见坑

流水线不是阶段越多越好，而是“能否支撑可信交付”最重要。增加一个阶段会增加时间成本，但如果这个阶段负责发现高代价故障，它通常值得保留。最常见的错误，不是工具选错，而是把关键控制点省掉了。

第一类坑是跳过 Verify。很多团队以为“部署成功”就等于“发布成功”，这是错误的。部署成功只说明新版本被放上去了，不说明接口返回对、指标正常、用户体验没坏。一个典型失败案例是：新版本容器启动正常，但探针没打到真正依赖的数据库查询，结果健康检查绿了，核心查询接口却在生产报错。改进办法是 Verify 同时检查 HTTP 状态、核心业务接口和关键指标。

第二类坑是没有监控和审计。审计，白话说就是“事后能查清谁在什么时候发布了什么”。如果没有审计日志，线上回滚时往往先花半小时查版本，真正修问题的时间反而更少。

第三类坑是 Secrets 管理混乱。Secrets 指密码、令牌、密钥等敏感信息。把它们写进脚本、仓库或镜像，短期方便，长期一定泄漏。正确做法是使用平台密钥管理能力，比如 GitHub Actions Secrets、云厂商 KMS、Vault。

第四类坑是质量门槛失真。比如强行要求覆盖率 90%，结果团队只写“凑数测试”；或者完全没有门槛，任何变更都能直接上线。门槛必须服务风险，而不是服务报表。

| 问题 | 后果 | 规避措施 |
| --- | --- | --- |
| 漏掉验证阶段 | 线上异常晚发现 | 把健康检查和关键接口校验写入流水线 |
| 缺少监控与告警 | 无法判断是否回滚 | 接入 Prometheus/Grafana，并设阈值 |
| Secrets 裸露 | 凭据泄漏 | 使用密钥管理器，流水线只读取引用 |
| 只测不扫 | 漏掉依赖漏洞和镜像风险 | 增加安全扫描阶段 |
| 环境不一致 | staging 过、production 挂 | 固定镜像、固定配置模板、基础设施代码化 |

下面是一个简单的验证脚本模板：

```bash
#!/usr/bin/env bash
set -e

BASE_URL="$1"

curl -fsS "${BASE_URL}/health" >/dev/null
curl -fsS "${BASE_URL}/api/recommend?id=123" >/dev/null

echo "verify passed"
```

这个脚本很短，但思路是对的：验证不是“看页面能不能打开”，而是调用真正关键的接口。工程上还应继续补两类检查：指标检查和日志检查。比如最近 5 分钟错误率是否升高，是否出现新版本独有异常。

---

## 替代方案与适用边界

不是所有团队一开始都需要完整 CI/CD。对小团队、原型项目、短期实验，更合理的做法通常是“先标准化，再自动化”。标准化，白话说就是先把步骤固定成脚本；自动化，则是在这个基础上交给平台执行。

一个常见替代方案是单脚本流程。比如团队先维护一个 `build-and-deploy.sh`，由负责人手动触发，但脚本内部仍然包含 `build -> test -> deploy -> verify` 四步。这种方式的优点是搭建快，适合项目早期；缺点是仍然依赖执行人，审计和并发控制较弱。

再进一步，可以做成半自动流程：构建和测试由 CI 自动完成，部署需要人工点击确认。这种方式适合对生产权限控制严格、发布频率不高的团队。它牺牲了一部分速度，换来更清晰的审批边界。

| 方案 | 适用条件 | 优点 | 风险 |
| --- | --- | --- | --- |
| 手动脚本 | 个人项目、早期原型 | 上手快，成本低 | 易漏步骤，追踪差 |
| 单阶段流水线 | 小团队、低频发布 | 有基础日志和状态 | 控制粒度不足 |
| 半自动流程 | 需要审批、发布频率中等 | 风险更可控 | 人工等待增加 |
| 完整 CI/CD | 多人协作、高频发布、线上影响大 | 可重复、可审计、可回滚 | 初期建设成本更高 |

一个初创团队的真实边界通常是这样的：用户量还小，服务只有 1 到 2 个，业务变动快。这时强行上复杂蓝绿架构可能得不偿失。更务实的方案是：

1. 用脚本固定构建与部署命令。
2. 保留最基本的自动测试。
3. 无论是否自动部署，都保留 Verify。
4. 明确回滚手册，例如“保留上一个镜像 tag，异常时 5 分钟内切回”。

也就是说，完整流水线不是唯一答案，但“标准化产物、自动化验证、可回滚”这三件事，几乎没有商量空间。无论你用 GitHub Actions、GitLab CI、Jenkins 还是 shell 脚本，本质都绕不开这三个要求。

---

## 参考资料

- TechTarget, *CI/CD Pipelines Explained*. 介绍 CI/CD 流水线的阶段划分、自动化价值与验证环节的作用，适合理解整体交付链路。
- Compile N Run, *CI/CD Components*. 对构建、测试、部署、验证的组件化拆分较清晰，适合建立初学者的阶段模型。
- TechBuzz Online, *Automated Deployment Strategies*. 补充蓝绿部署、金丝雀发布等策略的风险控制思路。
- OpsMoon, *CI/CD Pipeline Best Practices*. 侧重监控、审计、可观测性和流水线最佳实践，适合理解生产环境治理。
- GitHub Actions Documentation. 用于查阅 `needs`、artifact 上传下载、环境变量和 secrets 等具体配置。
- Jenkins Documentation. 适合理解传统 CI/CD 平台如何组织 pipeline、stage、agent 和插件生态。
