---
title: 静态与污点分析
date: 2026-09-08
section: cs
---

# 静态与污点分析

<div class="epigraph">
<p>污点把「不可信来源」标到字节上，看它是否流进「危险汇点」。静态分析不执行程序，因此必须抽象；假阳性是税，假阴性是洞。</p>
<footer>—— 据 Denning 的信息流；Schwartz, Avgerinos and Brumley 对污点的 SoK；Soot/CodeQL 一类工程</footer>
</div>

上一课[KLEE](/cs/symbolic-execution)沿路径跑。缺口是**不执行也能问流**：源→汇。本课收污点与静态告警的合同，不把 CodeQL 查询写成对外部目标的作战。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

命令注入、SQL、XSS 在 Web 课会再出现：共同形状是不可信数据进解释器。污点：源（输入）、汇（exec/SQL）、净化器。静态：控制流图上的近似。缺口是抽象与净化器是否真净化。

### 净化器撒谎

黑名单替换不是净化。污点会信你标注的 sanitizer。

<span class="marginnote">Denning；BitBlaze/TaintCheck 一类动态污点点名。本课禁止扫描他人站点。</span>

## 方法

画源–汇。对照动态污点（运行时，开销）与静态（编译期，近似）。下一课 sanitizer：把未定义行为变成可 fuzz 的崩溃。

```mermaid
flowchart TD
  SRC["不可信源"] --> FLOW["数据流"]
  FLOW --> SNK["危险汇点"]
  SAN["真净化"] --> CUT["切断污点"]
```

## 机制

发现要可编码成规则。下一课 ASan/UBSan/MSan 把一类空间/未定义错误变成确定失败。

污点沿着赋值、传参、拼接一路「染色」，静态分析要决定：哪些传递追、哪些净化点停、哪些路径只做近似。

```mermaid
flowchart TD
  SRC["污点源: request 参数"] --> P1["赋值/拼接: 污点跟着走"]
  P1 --> Q{"经过净化器?"}
  Q -- "是" --> CLN["净化: 污点清除"]
  CLN --> SINK2["到达汇点: 无告警"]
  Q -- "否" --> P2["继续传递"]
  P2 --> SINK["到达汇点: exec/SQL"]
  SINK --> ALARM["报一条污点流告警"]
```

## 边界

本课不给绕过 WAF 的污点例子。Sanitizers 下一课。

<span class="marginnote">术语翻译：污点分析就是把「这份数据是用户给的」这个事实，像染料一样染在变量上，然后问「这滴染料最后流进了哪些危险函数」——全程不运行程序，只在控制流图上做可达性推理。</span>

<span class="marginnote">数字实例：一条链上如果有 $3$ 处可能的净化点，静态分析对每处都只能选「信」或「不信」：全信可能漏报，全不信就是 $2^3=8$ 种组合要逐条排查——这就是为什么污点工具的告警总带着「可能」二字。</span>

## 小结

- 符号执行跑路径；污点问源是否到汇。
- 静态要抽象；净化器标注会撒谎。
- 与 Web 解释器洞同形。
- 下一课 Sanitizers。
- 出处：Denning；Schwartz, Avgerinos and Brumley；对照 CodeQL 文档。
