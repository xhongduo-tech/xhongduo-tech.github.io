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
- **检索（search）**：`GET [base]/Observation?subject=Patient/123&code=LOINC|2339-0` 按条件查——如「查这位患者所有血糖观测」。
- **创建/更新（create/update）**：`POST` / `PUT` 提交或更新资源。
- **条件操作**：`conditional create` 等支持「有则更新、无则创建」的幂等语义。

配套机制让 API 可用在真实医疗场景：

- **分页（pagination）**：大批结果分页返回。
- **版本（version）**：每次更新产生新版本，支持历史追溯。
- **操作（operations）**：如 `$everything` 一键拉取患者全部数据。
- **批量与事务（batch/transaction）**：多资源一次性提交。<span class="marginnote">`$</span>