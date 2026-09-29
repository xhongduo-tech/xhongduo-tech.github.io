---
title: 微操作缓存
date: 2026-09-08
section: cs
---

# 微操作缓存

<div class="epigraph">
<p>变长宏指令译成定长微操作的代价很高；把已经译好的 uop 按热路径存起来，命中时取指不再走长度扫描。</p>
<footer>—— 据 Rotenberg, Bennett, and Smith, Trace Cache, MICRO 1996；Intel 对 uop cache 的公开描述 整理</footer>
</div>

[上一课](/cs/fetch-decode-width)把前端带宽钉成供给上限，并点名变长译码是热点。本课不重做边界扫描。缺口是：**用一块小的、按 uop 组织的 cache 记住最近译过的指令，命中则绕过传统译码器。**

## 问题

x86 核每拍要给重命名输送 4–6 个 uop，长度译码与微码协助在频率上很痛。热循环可能只有几十条宏指令，却被反复译码。缺口不是再加一条预译码流水，而是**把「PC → uop 序列」缓存下来**，用 SRAM 命中代替组合译码。

<span class="marginnote">Trace cache 存的是动态执行痕迹，可跨越宏指令与分支。Intel Sandy Bridge 起公开的 DSB（decoded stream buffer / uop cache）更接近「按 IP 对齐的已译码 uop」，命中则 MITE 译码器睡觉。</span>

## 方法

uop cache：索引取自指令指针（可虚拟），行内存若干 uop、下一块指针、是否以分支结束。命中：每拍从该结构弹出最多 $D_{\text{uop}}$ 个已译码 uop 进重命名，I-cache 与长度译码不参与。缺失：走普通取指/译码，填入 uop cache。自修改代码或 icache 作废时同步清空。

```mermaid
flowchart TD
  IP["指令指针"] --> UC["uop cache"]
  UC -->|"命中"| REN["直接重命名"]
  UC -->|"缺失"| MITE["长度译码 + 微码"]
  MITE --> FILL["填 uop cache"]
  MITE --> REN
```

与 I-cache 的包含关系：uop cache 是译码后的旁路，不替代指令页的一致性；SMC 必须两边都看到。

## 机制

前端每拍实际能送多少 uop，取决于行边界与分支预测共同决定的一连串检查：

```mermaid
flowchart TD
  BR["分支预测给出下一块 IP"] --> LOOK{"uop cache 命中?"}
  LOOK -->|缺失| MIT["MITE 慢路径译码"]
  MIT --> RES["本拍供给大幅下降"]
  LOOK -->|命中| EDGE{"行内下一个边界＜br/＞是分支?"}
  EDGE -->|否| POPS["整行连续弹出至端口宽度"]
  POPS --> RN["送重命名"]
  EDGE -->|是| PR{"预测对了吗?"}
  PR -->|对| JUMP["无缝跳到后继行, 继续弹"]
  JUMP --> RN
  PR -->|错| FLUSH["冲刷前端, 重新取指"]
```

前端 CPI：热路径上译码器功耗与延迟下降，有效 $D$ 接近 uop 端口宽度。冷路径、巨函数、以及 uop 行装不下的超长指令序列仍走慢路径。容量以 uop 计，不是字节：一条复杂宏指令可能占多个槽，反而降低密度。

与[分支预测](/cs/gshare-predictor)：uop 行往往在预测分支处截断，预测准才能连续弹出。误预测仍冲刷，只是重填可能再次命中 uop cache。

<span class="marginnote">容量按 uop 槽计是个容易忽视的坑：典型实现约几千个 uop，一条 `mov` 大约占 1 槽，而一条复杂的读改写宏指令可能拆成 3–4 槽。同样是 100 条宏指令的热循环，全是简单指令约占 100 槽，全是复杂指令可能吃掉 400 槽——「代码短」不等于「uop 少」。</span>

<span class="marginnote">直觉类比：uop cache 就是「备好的翻译稿」。原文（x86 变长指令）第一次出现时翻译员逐句现场译（MITE 慢路径），并把译文抄在本子上；下次再读到同一段，直接念译稿，翻译员去睡觉。一旦有人偷偷改了原文（自修改代码），本子必须整本作废重译。</span>

<span class="marginnote">常见误区：以为 uop cache 命中就能「无限连续」供给。行在预测分支处截断，若预测器摇头（预测错误），前端照样冲刷重来——uop cache 只是省掉「重新翻译」，省不掉「重新排版」。所以分支预测准是它收益的前提，两者是乘法关系不是替代关系。</span>

## 边界

本课不把循环流缓冲当成 uop cache 的子集讲完——LSB 是「锁定一个循环的 uop 流」，下一课。也不把 GPU 的指令 cache 写成 uop cache。微码 ROM 仍服务极少见的复杂指令，不进热 cache。

后课默认：热路径可以不经宏指令译码器。更小、锁定的循环还可以连 cache 标签比较都省掉。

## 小结

- uop cache 缓存已译码微操作，绕过变长译码。
- 命中率取决于热路径大小与分支截断。
- 把循环锁定在更浅的缓冲里，是下一课 LSB。
- 出处：Rotenberg et al., *MICRO*, 1996；Intel 微结构公开文档中的 DSB。
