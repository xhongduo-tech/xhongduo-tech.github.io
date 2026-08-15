---
title: 互操作框架与 FHIR 标准
date: 2026-08-07
---

# 互操作框架与 FHIR 标准

<div class="epigraph">
<p>FHIR 把医疗互操作从「请厂商定制接口」变成了「用 Web 的方式想问题」。</p>
<footer>—— 编者按（本文）</footer>
</div>

<div class="article-byline">
<p>第五级 · 生物医学信息学 ｜ Hoyt《Health Informatics》第3章 ｜ 2026-08-07</p>
</div>

## 为什么从 FHIR 讲起

上一章以「v3 的失败教训」收尾，而这场失败直接催生了本世纪医疗互操作最重要的标准——**FHIR（Fast Healthcare Interoperability Resources，快速医疗互操作资源）**。FHIR 的设计哲学与 v3 针锋相对：不追求「一次建模统一宇宙」，而是提供**一组模块化的资源（resource）**，用**现代 Web 技术（REST、JSON、HTTP）** 实现「开箱即用」的互操作。如果说 v2 是「务实但松散」、v3 是「严谨但笨重」，FHIR 想兼得两者的长处。这一章解剖 FHIR 的资源模型、REST 接口、profile 机制与生态。<span class="marginnote">对「从极限到大模型」主线：FHIR 把医疗数据变成「可编程的资源 + REST API」，本质是给医疗数据装上了「开发者友好的接口层」——今天所有医疗 AI 应用取数据，几乎都从 FHIR 开始。</span>

## 1 从 v3 的废墟上长出的新哲学

FHIR 的诞生（2011 年起 HL7 内部孵化，2014 年发布标准版，2019 年成为国际标准）建立在对前两代的反思上：

**模块化而非整体建模**：v3 试图用一个 RIM 覆盖一切，FHIR 则拆成**一百多种资源（resource）**——`Patient`、`Observation`、`Medication`、`Encounter`、`Condition`……每种资源是独立、可复用的语义单元。需要的资源取走即用，不必理解全模型。<span class="marginnote">资源是 FHIR 的原子单位，类似面向对象的「类」：每个资源有 id、元数据、结构化字段与标准 JSON/XML 序列化。它的粒度设计很讲究——`Observation` 一条管一个观测（如一次血糖），天然适配「可组合」的数据交换。</span>

**Web 原生**：资源通过 **RESTful API** 暴露——`GET /Patient/123` 取患者、`POST /Observation` 提交观测。HTTP 动词、JSON 序列化、状态码、分页——开发者零门槛上手。

**80/20 原则**：标准覆盖「多数场景能直接用」的 80%，剩下 20% 靠 **profile（描述文件）与扩展（extension）** 本地化，而不是塞进核心标准。<span class="marginnote">80/20 原则是 FHIR 对 v3 教训的直接回应：核心标准保持精简，把「地区差异、机构差异」推向扩展层——这正是 TCP/IP「内核最小、能力外置」的经典思路。</span>

## 2 资源模型：FHIR 的积木

FHIR 资源是一组结构化的信息单元，理解它们要抓住三个要点：

**常用资源**：`Patient`（患者人口学）、`Observation`（观测结果：化验、体征）、`Condition`（诊断/问题）、`MedicationRequest`（用药医嘱）、`Encounter`（就诊事件）、`Procedure`（操作）、`DocumentReference`（文档引用）、`DiagnosticReport`（检查报告）。日常互操作的 80% 由约 30 种资源覆盖。

**资源引用**：资源之间通过**引用（reference）** 连接——`Observation.subject` 指向 `Patient`，`DiagnosticReport.result` 指向多个 `Observation`。这种引用网络让「患者的完整视图」可以由多个资源拼装出来。<span class="marginnote">资源引用是 REST 的世界观：不用「一个巨型对象包罗万象」，而是「小资源互相指来指去」，客户端按需跟随引用取数据——对应「从极限到大模型」里 API 设计的「小服务、明契约」哲学。</span>

**数据类型**：资源字段使用标准数据类型（string、dateTime、Quantity 带单位、CodeableConcept 带编码）——**Quantity 强制带单位**（如 `{"value": 138, "unit": "mmHg"}`）直接消除「单位歧义」这类经典医疗数据坑。

## 3 REST 接口：互操作变成「调 API」

FHIR 的交换核心是 RESTful API。核心操作：

**读取（read）**：`GET [base]/Observation/123` 按 id 取单条资源。
**检索（search）**：`GET [base]/Observation?subject=Patient/123&code=LOINC|2339-0` 按条件查——如「查这位患者所有血糖观测」。
**创建/更新（create/update）**：`POST` / `PUT` 提交或更新资源。
**条件操作**：`conditional create` 等支持「有则更新、无则创建」的幂等语义。

配套机制让 API 可用在真实医疗场景：

- **分页（pagination）**：大批结果分页返回，避免一次拉爆。
- **版本（version）**：每次更新产生新版本，支持历史追溯与审计。
- **操作（operations）**：如 `$everything` 一键拉取患者全部数据。
- **批量与事务（batch/transaction）**：多资源一次性提交，事务保证「要么全成、要么全不成」。<span class="marginnote"><strong>交易型互操作（transaction-based）</strong>是医疗系统最看重的能力：把一次就诊的多条数据（诊断、医嘱、化验）打包成一个事务提交，语义一致、可回滚，避免「半套数据」污染接收方。</span>

## 4 Profile：让标准长在本地

FHIR 的「开箱即用」只是第一步，真实世界必须**裁剪与扩展**，这就是 profile 与扩展的用武之地：

**扩展（extension）**：FHIR 允许在资源上挂自定义字段——用「扩展」机制表达核心标准没有的本地语义（如「患者的方言偏好」）。扩展有严格的结构约束（URL 标识、数据类型），不会污染标准字段。<span class="marginnote">扩展是 FHIR 给「标准之外」留的口子：任何机构都可以定义自己的扩展并公开其结构，但核心字段的语义保持全球一致——「有边界地自由」是它区别于 v2 Z 段自由文本的关键。</span>

**Profile（描述文件）**：一组约束，规定某个用例中资源字段的「必填、可选、值域、Cardinality」。比如「中国住院患者档案」profile 可规定民族字段必填、身份证号格式校验。<span class="marginnote">可以把 profile 理解为「接口实现规范」：核心标准是「接口」，profile 是「本机构/本国家的具体实现要求」。著名的例子有 <strong>US Core</strong>（美国基线 profile）、德国的 MIO（Musterimplementierungsleitfaden）、中国的卫生信息 profile 工作。</span>

**CapabilityStatement（能力声明）**：每个 FHIR 服务器公开自己支持哪些资源、操作、profile——客户端先读能力声明，再决定怎么调用。这是「可发现性」的关键。

**重点：** profile 机制让「全球标准」与「本地现实」达成和解——标准不必为每个国家的特殊需求改版，本地也不必抛弃国际标准另起炉灶。这与「从极限到大模型」里「开放标准 + 企业定制」的分层思路一脉相承。

## 5 核心对比表：三代消息标准

纯概念主题用核心对比表替代公式解析——把 v2、v3、FHIR 摆在一起看最清楚：

| 维度 | HL7 v2 | HL7 v3 / RIM | FHIR |
| --- | --- | --- | --- |
| 哲学 | 务实、分段消息 | 严谨、全域建模 | 模块化、Web 原生 |
| 载体 | 竖线分隔文本（\|） | XML | JSON / XML / RDF |
| 建模 | 无强模型，字段松散 | RIM 严格参考模型 | 独立资源 + 引用 |
| 语义 | 弱（Z 段自由扩展） | 强但难落地 | 中等，靠 profile 强化 |
| 上手 | 快，兼容老系统 | 极慢 | 快，开发者友好 |
| 扩展 | Z 段（易混乱） | 规则繁琐 | 扩展 + profile 规范 |
| 现实地位 | 仍大量服役 | 大量失败、少数部署 | 全球推进、主流方向 |

这张表回答了「为什么 FHIR 能赢」：它把 v2 的「快」与 v3 的「稳」用现代软件工程重新组织了。<span class="marginnote">现实中三者在相当长时间内会共存：v2 管存量接口、FHIR 管新建系统、v3 的 RIM 思想沉淀进了各版本互操作的语义约束。读懂这张表，就理解了 HL7 家族三十年的演进逻辑。</span>

## 6 FHIR 生态：SMART、Bulk Data 与 CDS Hooks

FHIR 真正的力量来自围绕它长出的生态工具，它们把「有 API」升级为「能协作」：

**SMART on FHIR**：在 FHIR 之上加了一层 OAuth 2.0 授权——第三方应用获得患者/医生授权后，即可通过 FHIR API 读取数据、在 EHR 内嵌运行（如一个「房颤风险评分」App 直接跑在医院门户里）。它是「医疗应用商店」的技术底座。<span class="marginnote"><strong>SMART</strong>（Substitutable Medical Applications, Reusable Technologies）是美国 2009 年后 ARRA 法案背景下发展出的「可替换、可复用」应用框架：让医疗 App 像手机 App 一样即装即用，跨 EHR 厂商移植。</span>

**Bulk Data（批量数据）**：`$export` 操作允许一次导出整个患者群体的 FHIR 数据——这是**医学大数据与联邦学习**从 EHR 取数的主要通道，科研与公共卫生监测的命脉。

**CDS Hooks**：临床决策支持以「钩子（hook）」方式嵌入工作流——医生开医嘱时，EHR 触发钩子，调用外部 CDS 服务返回建议卡片。它与第4篇《临床决策支持系统》直接衔接。<span class="marginnote">CDS Hooks 的「事件钩子 + 外部服务 + 卡片回显」模式，让 CDS 与 EHR 解耦：决策逻辑可以升级、替换，而不必改 EHR 本体——呼应「从极限到大模型」的微服务与插件化架构。</span>

## 7 局限与挑战：FHIR 不是银弹

FHIR 有清晰的边界，正视它才能用对它：

**语义深度不足**：资源的字段语义比 RIM 松——同一数据在不同系统里仍可能「长得一样、含义不同」，语义互操作必须靠 profile + 术语绑定（SNOMED、LOINC）补足，这与第3篇受控术语章节直接相关。

**版本漂移**：FHIR 有 R4、R4B、R5 等版本，资源定义在演进——存量接口的「版本锁定」与「升级迁移」是现实痛点。<span class="marginnote">截至 2026 年，<strong>R4</strong> 仍是部署最广的基线（2019 年发布），R5 已发布但生态迁移缓慢——「向后兼容」是 FHIR 版本治理的头等议题。</span>

**治理碎片化**：各国家/地区 profile 百花齐放，若缺乏协调，「本地 profile」本身又会长成新的孤岛——互操作需要「元治理」。

**重点：** FHIR 解决的是「数据怎么流通」的传输层问题，它不替你做「数据怎么定义语义」「数据怎么保障安全」——这些是第3篇术语、隐私与治理章节的职责。把 FHIR 放进互操作的完整拼图里，才不会「技术兴奋、治理翻车」。

### 常见疑问

**问：FHIR 与「HL7 这个名字」什么关系？**
答：FHIR 由 HL7 组织制定，是 HL7 家族的第四个里程碑标准（v2、v3、CDA、FHIR）。它不推翻前辈，而是并存：新系统优先 FHIR，存量系统继续 v2，两者通过**接口适配**桥接——很多「FHIR 网关」就是把老系统的 v2 消息实时翻译成 FHIR 资源。

**问：为什么 2014 年才有人觉得该这么做？**
答：因为条件在 2010 年前后成熟——**REST、JSON、OAuth 已成为 Web 世界的事实标准**，医生与患者对「手机上看化验单」有了普遍期待，而移动互联网让「App 读取医院数据」成为可见的场景。FHIR 是「把 Web 的答案搬到医疗」，时机与技术都到位了。

**关联阅读**：FHIR 的传输层之上站着语义层——见第3篇《医学受控术语》；授权与安全细节见《健康信息安全、隐私与 HIPAA》；FHIR 数据流向决策与模型见第4篇《医疗数据分析与机器学习》。

## 8 小结

- FHIR 以**模块化资源 + REST/JSON + 80/20 原则**回应 v2/v3 的两难，2019 年成为国际标准。
- 资源模型三要素：**常用资源、资源引用、带语义的数据类型**（Quantity 强制单位）。
- REST 接口提供 read/search/create/update、分页、版本、事务；`$everything`、`$export` 等操作扩展能力。
- **profile 与扩展**把全球标准裁剪到本地现实，CapabilityStatement 让能力可发现。
- 生态工具 **SMART（授权）、Bulk Data（批量）、CDS Hooks（决策支持）** 把「有 API」升级为「能协作」。
- 局限清醒认识：**语义深度、版本漂移、治理碎片化**——FHIR 是互操作的重要一环而非全部。

在下一节，我们将离开「消息与 API」的世界，转向医疗影像的另一套标准语言——**DICOM**，看图像数据如何被标准化、存储与交换。
