---
title: 内存安全语言迁移
date: 2026-09-08
section: cs
---

# 内存安全语言迁移

<div class="epigraph">
<p>缓解堆叠提高利用代价；内存安全语言把越界、UAF、数据竞争从语义里删掉。迁移是 TCB 决策：新代码默认安全语言，遗留 C 用边界与沙箱围住。</p>
<footer>—— 据 Miller 对内存安全的论述；Rust 所有权；seL4 对 C 子集验证的对照；Anderson 对 TCB</footer>
</div>

上一课[PAC](/cs/pointer-authentication)仍假设 C 能写出任意写。缺口是**换语义**：Rust/Go/Java/.NET 等在各自模型里取消空间安全洞（不安全块与运行时漏洞除外）。本课讲迁移策略，不写语言教程。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

C 的对象模型允许本单元全部洞。安全语言：界检查、GC 或借用、类型。缺口是 FFI 与 `unsafe`：洞从边界再进来。策略：新组件默认安全语言，C 缩进 TCB。

<span class="marginnote">术语翻译：TCB（可信计算基）就是你不得不「信它没漏洞」的那批代码——它出洞，一切外围防护作废。语言迁移的本质是缩小 TCB：把业务代码移出这份名单，名单上只留运行时和 FFI 这些绕不开的部分。</span>

### 运行时仍要补丁

GC 实现、JIT、标准库仍是 C/C++ 时，沙箱与更新策略不能停。

<span class="marginnote">Chrome/Android 等公开的语言迁移叙述。seL4 用验证过的 C 子集，是另一条路，本课程最后一课再收。</span>

## 方法

分绿场与棕场：新服务用安全语言；解析器等攻击面优先重写；FFI 最小化。对照硬件缓解：迁移期仍开。下一课序改「如何发现」：fuzz。

<span class="marginnote">直觉类比：绿场像在空地上盖新楼，直接按新规范施工；棕场像改造还在住人的老楼——不能推倒重来，只能先把最危险的承重墙（解析器这类直接摸不可信输入的攻击面）换掉，其余先加上支撑（沙箱、硬件缓解）继续用。</span>

```mermaid
flowchart TD
  NEW["新代码"] --> SAFE["内存安全语言"]
  OLD["遗留 C"] --> BOX["沙箱与缓解"]
  FFI["边界"] --> REV["审查与测试"]
```

## 机制

利用课序封口：从栈到硬件到语言。发现与工程课序从覆盖引导 fuzz 起，承认遗留 C 仍在。

<span class="marginnote">常见误区：以为「改用 Rust/Go 就安全了」。实际上 GC、JIT、标准库内部可能仍是 C/C++，FFI 一调到 C，越界和 UAF 就从边界回来——所以运行时要继续打补丁、遗留部分要继续沙箱化，迁移不等于卸责。</span>

```mermaid
flowchart TD
  APP["安全语言写的业务代码"] --> RT["语言运行时: GC/JIT"]
  APP --> LIB["标准库内部实现"]
  APP --> FFI["FFI 胶水与遗留 C"]
  RT --> TCB["仍是 TCB: 沙箱与补丁不能停"]
  LIB --> TCB
  FFI --> TCB
```

## 边界

本课不比较全部语言性能神话。下一课覆盖引导 fuzzing。

## 小结

- PAC/CET 是缓解；安全语言改语义。
- FFI 与运行时仍是 TCB。
- 迁移是攻击面排序，不是一天重写内核。
- 下一课序：覆盖引导 fuzz。
- 出处：Miller 等对内存安全；Rustonomicon 的 unsafe 边界；Anderson；Klein et al. seL4（对照）。
