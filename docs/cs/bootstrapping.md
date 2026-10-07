---
title: 自举
date: 2026-09-08
section: cs
---

# 自举

<div class="epigraph">
<p>用已有编译器编译新编译器的源，再用新编译器编译自己，直到二进制不动点。信任问题：第一份二进制从哪来。</p>
<footer>—— 据 Thompson, Reflections on Trusting Trust, 1984；Wirth 自举传统；对照 CompCert 抽取整理</footer>
</div>

上一课[Csmith](/cs/compiler-fuzzing) 测的是已得到的编译器。本课收束「类型、优化与运行时」：缺口是**自举链**——Rust/GHC/Go 都用上一代编下一代。首课 [lex/flex](/cs/lex-flex) 在异常表之后从生成器切入；现在回到信任基。下一课程操作系统从 [内核与用户态](/cs/kernel-user) 起，不在本课打开 VFS。

## 问题

新语言没有编译器。阶段：用 C 写一个子集编译器，再扩展，直到能编自己。不动点：`cc_new(src) == cc_old(src)` 的二进制（或 ABI 允许的等价）。缺口是**阶段与信任**，不是差分测试。

Thompson：编译器可插入后门且自复制，源码审计不够。对策：多样编译器比较、从更小信任基重建（Guix 引导点名）。

### 自举不是「递归函数」

是工程过程，不是 $Y$ 组合子。不要和[递归定理](/cs/recursion-theorem) 混为一谈，尽管都有「自己谈自己」。

<span class="marginnote">Thompson 1984 Turing 演讲。GHC stage。Rust bootstrap。CompCert 用 Coq 抽取，信任核不同。本课不写完整可引导发行版。</span>

<span class="marginnote">直觉类比：自举像「拉着自己的鞋带把自己提起来」——先有别人给的旧鞋带（stage0 二进制），编出一版新鞋带，再用新鞋带编自己，两版完全一样就说明站稳了。</span>

## 方法

保留 stage0 二进制。stage1：stage0 编源。stage2：stage1 再编，应与 stage1 输出一致（或可解释的差异：路径嵌入）。交叉自举：用三元组在主机上为 target 产出。

```mermaid
flowchart TD
  S0["stage0 二进制"] --> S1["编译源 → stage1"]
  S1 --> S2["再编译 → stage2"]
  S2 --> FIX["比较 / 不动点"]
```

与 LTO/PGO：自举配置要钉死，否则二进制永远不等。

## 机制

时间戳、随机、`__DATE__` 破坏可复现。可复现构建是自举验证的朋友。不要在自举时混主机头文件——[交叉](/cs/cross-compile-triple)。

多样复现：两份无关编译器编同一源，比 Thompson 后门。

stage1 与 stage2 输出若不一致，问题出在哪？三类典型原因各有对策：

```mermaid
flowchart TD
  D["stage1 ≠ stage2"] --> T["时间戳 / __DATE__ 嵌入"]
  D --> P["构建路径或随机顺序不同"]
  D --> O["优化与 LTO 配置不一致"]
  T --> F["用 SOURCE_DATE_EPOCH 等钉死"]
  P --> F
  O --> F
  F --> FIX["可复现：比较通过"]
```

<span class="marginnote">为什么重要：比较不过就不敢说「到了不动点」，Thompson 式后门恰恰藏在「二进制和源码对不上」的缝隙里——可复现构建让每一步都能第三方重放验证。</span>

## 边界

本课不写内核引导。后课默认：语言实现靠自举链；信任可被 Thompson 攻击，需多样重建。「类型、优化与运行时」到此结束；下一课 [内核与用户态](/cs/kernel-user)。

也不把自举当生物课。

<span class="marginnote">常见误区：初学者容易以为「自举=递归调用自己」。递归是一次运行内的调用栈；自举是一个跨多轮构建的**工程流程**，每轮产物是独立的二进制，轮与轮之间靠文件与比较衔接。</span>

## 小结

- 自举：用编译器编译自己，直到二进制稳定。
- Thompson：恶意编译器可存活在二进制里。
- 与 CompCert/Csmith：证明、测试、信任链三条腿。
- 出处：Thompson, 1984；对照 Leroy、Yang et al.；GNU/LLVM 自举实践。
