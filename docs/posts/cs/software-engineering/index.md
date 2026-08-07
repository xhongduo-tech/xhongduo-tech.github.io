---
pageClass: plain-doc
---

# 软件工程

学完软件工程 = 写完这一页。对标《软件工程：实践者的研究方法》（Pressman）与《构建之法》（邹欣）的知识体系，从过程、需求、设计、测试到项目管理与质量保障，逐节写成博文。

## 主题规划

<ProgressGrid cat="cs/software-engineering" />


### 第 1 章 软件工程概述

- [x] [软件的本质与软件危机](./nature-of-software)
- [x] [软件工程的定义与层次](./definition-and-layers)
- [x] [软件过程框架与普适性活动](./process-framework)
- [x] [软件神话与常见误区](./software-myths)
- [x] [个人软件过程（PSP）与结对编程](./psp-and-pair-programming)

### 第 2 章 软件过程模型

- [x] [瀑布模型及其适用场景](./waterfall-model)
- [x] [增量模型与迭代模型](./incremental-and-iterative-models)
- [x] [原型模型（快速原型与演化式原型）](./prototype-model)
- [x] [螺旋模型与风险驱动开发](./spiral-model)
- [x] [敏捷开发与敏捷宣言](./agile-development-and-manifesto)
- [x] [Scrum：角色、事件与工件](./scrum-roles-events-artifacts)
- [x] [极限编程（XP）：实践与价值观](./extreme-programming-practices-and-values)
- [x] [MSF 与典型团队开发流程（《构建之法》视角）](./msf-and-team-development-process)

### 第 3 章 需求工程

- [x] [需求工程的任务：起始、获取、细化、协商、规格说明、确认与管理](./requirements-engineering-tasks)
- [x] [需求获取技术：访谈、问卷、观察与联合需求计划（JRP）](./requirements-elicitation-techniques)
- [x] [用户故事与用例建模](./user-stories-and-use-case-modeling)
- [x] [基于场景的分析模型](./scenario-based-analysis-model)
- [x] [基于类的分析建模（CRC 与类图）](./class-based-analysis-modeling-crc)
- [x] [行为建模：状态图与顺序图](./behavioral-modeling-state-sequence)
- [x] [数据建模与数据流图（DFD）](./data-modeling-dfd)
- [x] [软件需求规格说明（SRS）的编写](./software-requirements-specification)
- [x] [需求确认与需求跟踪](./requirements-validation-and-traceability)

### 第 4 章 软件设计概念

- [x] [设计的基本概念：抽象、体系结构、模式、关注点分离](./design-basic-concepts-abstraction)
- [x] [模块化与信息隐藏](./modularity-and-information-hiding)
- [x] [功能独立：内聚与耦合](./cohesion-and-coupling)
- [x] [SOLID 设计原则：单一职责与开闭原则](./solid-srp-ocp)
- [x] [SOLID 设计原则：里氏替换、接口隔离与依赖倒置](./solid-lsp-isp-dip)
- [x] [面向对象设计：类设计与构件级设计](./class-design-and-component-level-design)

### 第 5 章 体系结构设计

- [x] [软件体系结构的作用与描述视图（4+1 视图）](./software-architecture-4-plus-1-views)
- [x] [体系结构风格：以数据为中心与数据流风格](./architectural-styles-data-centric-data-flow)
- [x] [体系结构风格：调用-返回与面向对象风格](./architectural-styles-call-return-oo)
- [x] [体系结构风格：层次结构与事件驱动](./architectural-styles-layered-event-driven)
- [x] [微服务体系结构：特征、拆分与通信](./microservices-architecture)
- [x] [体系结构评审与评估方法（ATAM）](./architectural-evaluation-atam)
- [x] [体系结构决策记录与技术选型](./architecture-decision-records)

### 第 6 章 设计模式

- [x] [设计模式概述：GoF 分类与模式要素](./design-patterns-overview-gof)
- [x] [创建型模式：单例与工厂方法](./creational-patterns-singleton-factory-method)
- [x] [创建型模式：抽象工厂、建造者与原型](./creational-patterns-abstract-factory-builder-prototype)
- [x] [结构型模式：适配器与桥接](./structural-patterns-adapter-bridge)
- [x] [结构型模式：组合、装饰与外观](./structural-patterns-composite-decorator-facade)
- [x] [结构型模式：享元与代理](./structural-patterns-flyweight-proxy)
- [x] [行为型模式：责任链与命令](./behavioral-patterns-chain-of-responsibility-command)
- [x] [行为型模式：解释器与迭代器](./behavioral-patterns-interpreter-iterator)
- [x] [行为型模式：中介者与备忘录](./behavioral-patterns-mediator-memento)
- [x] [行为型模式：观察者与状态](./behavioral-patterns-observer-state)
- [x] [行为型模式：策略、模板方法与访问者](./behavioral-patterns-strategy-template-method-visitor)
- [x] [反模式与设计模式的误用](./antipatterns-and-pattern-misuse)

### 第 7 章 用户界面设计

- [x] [用户界面设计的黄金规则](./ui-design-golden-rules)
- [x] [界面分析：用户、任务与环境分析](./interface-analysis-user-task-environment)
- [x] [界面设计步骤与原型评估](./interface-design-steps-and-evaluation)
- [x] [可用性工程与用户体验度量](./usability-engineering-and-ux-metrics)

### 第 8 章 软件测试

- [x] [软件测试的目标、原则与测试心理学](./software-testing-goals-principles)
- [x] [测试级别：单元测试、集成测试、确认测试与系统测试](./test-levels)
- [x] [白盒测试：基本路径测试与控制结构测试](./white-box-testing-basis-path)
- [x] [黑盒测试：等价类划分与边界值分析](./black-box-testing-equivalence-class-boundary)
- [x] [黑盒测试：决策表与因果图](./black-box-testing-decision-table-cause-effect)
- [x] [集成测试策略：自顶向下、自底向上与持续集成](./integration-testing-strategies)
- [x] [回归测试与冒烟测试](./regression-and-smoke-testing)
- [x] [测试覆盖率：语句、分支、路径与条件覆盖](./test-coverage-criteria)
- [x] [测试用例设计、评审与管理（《构建之法》视角）](./test-case-design-review-management)
- [x] [自动化测试与测试驱动开发（TDD）](./automated-testing-and-tdd)
- [x] [测试文档：测试计划与测试报告](./test-documentation)

### 第 9 章 软件维护与演化

- [x] [软件维护的类型：改正性、适应性、完善性与预防性](./software-maintenance-types)
- [x] [软件可维护性与维护成本](./maintainability-and-maintenance-cost)
- [x] [遗留系统的演化与再工程](./legacy-system-evolution-reengineering)
- [x] [逆向工程与正向工程](./reverse-engineering-and-forward-engineering)
- [x] [软件演化法则（Lehman 定律）](./lehman-laws-of-software-evolution)

### 第 10 章 软件项目管理

- [x] [软件项目管理的人员、产品、过程与项目（4P）](./software-project-management-4p)
- [x] [软件度量与项目估算基础](./software-measurement-and-estimation-basics)
- [x] [规模估算：代码行（LOC）与功能点（FP）](./size-estimation-loc-function-point)
- [x] [工作量估算模型：COCOMO 与用例点](./effort-estimation-cocomo-use-case-points)
- [x] [敏捷估算：故事点与计划扑克](./agile-estimation-story-points-planning-poker)
- [x] [项目进度计划：WBS、甘特图与关键路径](./project-scheduling-wbs-gantt-critical-path)
- [x] [风险管理：识别、预测、求精与缓解（RMMM）](./risk-management-rmmm)
- [x] [团队组织与沟通管理](./team-organization-and-communication)

### 第 11 章 软件质量保证

- [x] [软件质量的概念：Garvin 与 McCall 质量模型](./software-quality-concepts-garvin-mccall)
- [x] [软件质量保证（SQA）活动与正式技术评审](./sqa-activities-and-formal-review)
- [x] [软件可靠性、可用性与安全性](./software-reliability-availability-safety)
- [x] [六西格玛与过程改进（CMMI）](./six-sigma-and-process-improvement-cmmi)
- [x] [质量成本与缺陷预防](./cost-of-quality-and-defect-prevention)

### 第 12 章 配置管理与版本控制

- [x] [软件配置管理（SCM）的任务与基线](./software-configuration-management-baseline)
- [x] [版本控制模型：集中式与分布式](./version-control-models)
- [x] [Git 工作流：分支策略与合并（Git Flow 与 GitHub Flow）](./git-workflows-branching)
- [x] [构建管理与发布管理](./build-and-release-management)

### 第 13 章 DevOps 与 CI/CD

- [x] [DevOps 文化与 CALMS 框架](./devops-culture-calms)
- [x] [持续集成（CI）：流水线与实践](./continuous-integration)
- [x] [持续交付与持续部署（CD）](./continuous-delivery-and-deployment)
- [x] [基础设施即代码（IaC）与配置管理工具](./infrastructure-as-code)
- [x] [监控、日志与可观测性](./monitoring-logging-observability)

### 第 14 章 代码评审与重构

- [x] [代码评审的价值与正式评审流程](./code-review-value-and-process)
- [x] [结对编程与轻量级代码审查实践](./pair-programming-and-lightweight-review)
- [x] [代码坏味道（Bad Smells）识别](./code-smells)
- [x] [重构手法：提炼、搬移与重组数据](./refactoring-extract-move-reorganize)
- [x] [重构手法：简化条件表达式与简化方法调用](./refactoring-simplify-conditionals-calls)
- [x] [大型重构与重构到模式](./large-refactoring-and-refactoring-to-patterns)
- [x] [静态代码分析与代码质量工具](./static-analysis-and-quality-tools)

### 第 15 章 软件度量

- [x] [软件度量的分类：过程、项目与产品度量](./software-measurement-classification)
- [x] [面向规模的度量与功能点度量](./size-oriented-and-function-point-metrics)
- [x] [代码质量度量：圈复杂度与 Halstead 度量](./code-quality-metrics-cyclomatic-halstead)
- [x] [面向对象度量（CK 度量套件）](./object-oriented-metrics-ck-suite)

> 写作完成后：在本目录新建 `xxx.md`，然后把上面对应条目改为 `- [x] [标题](./xxx)`。
