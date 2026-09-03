---
title: 2×2 patch merge 控制视觉 token 数
date: 2026-09-03
section: llm
---

# 2×2 patch merge 控制视觉 token 数

<div class="epigraph">
<p>原生分辨率会把视觉序列拉得很长；先把相邻四个 patch 合成一个标记，再交给语言模型，长度才压得住。</p>
<footer>—— Wang 等，Qwen2-VL，2024；Qwen2.5-VL / Qwen3-VL 技术报告</footer>
</div>

动态分辨率让 ViT 按原图像素切块，不再把图缩到固定边长。代价立刻出现：一张高清扫描件可以生成数千个 patch，若原样送进 Decoder，prefill 的二次注意力与 KV 缓存都会被视觉侧撑爆。Qwen2-VL 起在 ViT 之后加一层简单的空间合并：相邻 $2\times 2$ 个 patch 拼成一个视觉 token。Qwen2.5-VL 与 Qwen3-VL 沿用同一压缩比，只是把投影写成两层 MLP，并与语言模型隐维对齐。本篇只写这一步合并如何控制序列长度，不把窗口注意力或 DeepStack 再展开成并列架构。

## 问题

视觉编码器以固定 patch 边长 $p$ 切图。Qwen2-VL / Qwen2.5-VL 取 $p=14$。高 $H$、宽 $W$ 的图得到

$$
N_{\mathrm{patch}}=\frac{H}{p}\cdot\frac{W}{p}
$$

个 ViT 标记。原生分辨率意味着 $H,W$ 随输入变，文档、截屏、海报都会把 $N_{\mathrm{patch}}$ 推到数千。语言模型的上下文既要装视觉序列，又要装提问与历史，视觉侧若按 patch 一对一进 Decoder，多图与长视频会先把预算吃完。

LLaVA-NeXT 一类方案用 AnyRes：把大图切成若干固定分辨率的子图，每块各自编码再拼接。它控制的是「块数 × 固定 token」，不是「随像素线性增长后再整体压缩」。Qwen 系列走另一条路：ViT 仍看细网格，送进 LLM 之前再做一次空间下采样。问题因此是：下采样必须可微、必须保持二维邻接，并且压缩比要足够把 $224\times 224$ 量级的图压到几十个标记，同时不把文字笔画糊成一团。

<span class="marginnote">合并发生在 ViT 之后、语言模型之前。ViT 内部仍按未合并的 patch 做注意力，细空间结构先在编码器里混合，再被压成较短的视觉前缀。</span>

### 长度为啥不能只靠缩小输入图

把图先缩到 $224\times 224$ 再编码，token 数固定，但小字、表格线、角标会先被插值抹掉。文档与 OCR 恰恰依赖这些高频细节。动态分辨率把细节留给 ViT；若不同时压缩送进 LLM 的序列，细节预算会变成上下文爆炸。$2\times 2$ 合并是这两极之间的硬约定：编码器按 $14$ 像素看，语言模型按 $28$ 像素当一个视觉词。

## 方法

### 相邻四 patch 拼成一词

记 ViT 末层（或 merger 输入）在网格 $(i,j)$ 上的隐向量为 $\mathbf{h}_{i,j}\in\mathbb{R}^{d_v}$。空间合并大小 $s=2$，把

$$
\mathbf{u}_{i,j}=\big[\mathbf{h}_{2i,2j};\,\mathbf{h}_{2i,2j+1};\,\mathbf{h}_{2i+1,2j};\,\mathbf{h}_{2i+1,2j+1}\big]\in\mathbb{R}^{4d_v}
$$

送入 LayerNorm 与两层 MLP，映到语言模型隐维 $d$：

$$
\mathbf{z}_{i,j}=\mathrm{MLP}\big(\mathrm{LN}(\mathbf{u}_{i,j})\big)\in\mathbb{R}^{d}.
$$

实现里常把整段序列 `view` 成每行 $4d_v$，再走 GELU 夹心的线性层。Qwen2-VL 技术报告把这一层称作 ViT 之后的简单 MLP；开源实现里模块名是 PatchMerger，默认 `spatial_merge_size=2`。Qwen3-VL 仍写「两层 MLP 把 $2\times 2$ 视觉特征压成一个视觉 token」，压缩比没有改成 $3\times 3$ 或可学习池化。

压缩后的网格边长各除以 $2$，标记数为

$$
N_{\mathrm{tok}}=\frac{H}{2p}\cdot\frac{W}{2p}=\frac{N_{\mathrm{patch}}}{4}.
$$

Qwen2-VL 给出对照：分辨率 $224\times 224$、$p=14$ 时，$N_{\mathrm{patch}}=16\times 16=256$，合并后 $8\times 8=64$，再在两端加上视觉起止特殊标记，进入 LLM 的是 $66$ 个视觉相关 token。边长必须能被 $2p$ 整除。Qwen2.5-VL 因此在送入 ViT 前把高宽调整为 $28$ 的倍数——$28=2\times 14$，正好对齐「一 token 对应 $28\times 28$ 像素」。

```mermaid
flowchart LR
  Img["原生分辨率图像"] --> Patch["p 等于 14 切块"]
  Patch --> ViT["ViT 按 patch 编码"]
  ViT --> Merge["2 乘 2 拼接再 MLP"]
  Merge --> LLM["进入语言模型的视觉前缀"]
```

### 与投影器、与 AnyRes 的分工

LLaVA 的投影器把**已经固定长度**的视觉序列从 ViT 维映到 LLM 维，不负责把 $N$ 除以四。Qwen 的 merger 同时做两件事：空间下采样与维数对齐。AnyRes 用多块固定分辨率拼接来逼近大图，块与块之间没有「四合一」的共享 MLP；Qwen 用单幅动态网格加一次规则合并。两者都在控制视觉 token 数，轴不同：一块是「先切图再编码」，一块是「先细编码再合并」。不要把 $2\times 2$ merge 写成任意分辨率池化，也不要写成可学习的 token 剪枝——报告里是规则邻域拼接，不是 Top-$k$ 丢弃。

<span class="marginnote">视频侧 Qwen2.5-VL 还把连续两帧在时间上打成一个 3D patch，那是时间维的另一次压缩，与空间 $2\times 2$ 正交。本篇的 $N_{\mathrm{tok}}$ 公式只写空间合并。</span>

## 机制

四合一能工作，是因为相邻 patch 在自然图像与印刷品上都高度相关：同一字形、同一表格线、同一色块会跨过 $14$ 像素边界。MLP 看到的是已经过 ViT 混合的四元组，不是生像素，所以合并更接近「读完邻域再写一个词」，而不是「四个互不相关的向量硬平均」。压缩比固定为 $4$，梯度路径短，训练稳定，也让动态分辨率的长度预测成为算术：像素定了，token 数就定了。

对 OCR 而言，一个合并 token 覆盖 $28\times 28$ 像素。中文印刷体在常见扫描 dpi 下，这一窗口往往仍能包住一个字或半个词；英文小字、密集脚注则可能多个字符挤进同一 token。这是后续 [文字定位](/llm/qwen-ocr-text-grounding) 仍要靠更高输入分辨率补救的原因：合并比不改，只把图开得更大，让每个 $28\times 28$ 窗口里的字更少。

<span class="marginnote">特殊标记 `&lt;|vision_start|&gt;` 与 `&lt;|vision_end|&gt;` 包住压缩后的序列，让语言模型知道视觉前缀的边界。它们不参与 $2\times 2$ 合并，却计入上下文长度。</span>

## 边界与工程取舍

$s=2$ 是系列默认，不是 sweeper 搜出来的最优整数。$s=1$ 等于不压缩，动态分辨率的长度不可接受；$s=4$ 则一 token 覆盖 $56\times 56$ 像素，表格细线与角标更容易糊。报告没有把 merge size 写成任务相关的可学习门，复现时应把它当成与 $p=14$ 绑在一起的架构常数。

合并是规则网格，不能处理「这四个 patch 语义无关、不该合成一词」的情况：跨栏缝、图文交界、被裁切的半个字，都会被强制拼进同一 $\mathbf{u}_{i,j}$。窗口注意力可以在 ViT 内部先做局部混合，缓解但不取消这一硬切。另一边界是多图拼接：每张图各自合并，再与文本交错；merger 不在图与图之间共享邻域。

工程上，merger 的输出维必须等于 LLM 隐维（Qwen2.5-VL 的 3B/7B/72B 分别把 merger 输出接到 $2048$、$3584$、$8192$）。换底座语言模型却忘了改 merger 最后一层，是静默的形状错误。推理时 `min_pixels` / `max_pixels` 通过限制 $H\times W$ 间接限制 $N_{\mathrm{tok}}$；它们与 $2\times 2$ 合并相乘，而不是替代合并。把像素上限开到很大却以为 token 仍接近 $64$，会把文档页的上下文直接打满。

## 小结

- 动态分辨率按 $p=14$ 产生可变长 patch 序列；$2\times 2$ patch merge 把相邻四向量拼接，经 MLP 压成一个与 LLM 对齐的视觉 token，长度除以四。
- $224\times 224$ 时合并后约 $64$ 个视觉标记，加起止符后为 $66$；高宽需对齐 $28$ 的倍数。
- 合并在 ViT 之后，细网格注意力仍在编码器内发生；它不是 AnyRes 切块，也不是可学习剪枝。
- 固定压缩比让长度可事先算清，也把过密文字挤进同一 token，OCR 要靠提高输入分辨率而不是改 $s$。
- 出处：Wang 等，*Qwen2-VL*，2024；Qwen2.5-VL 技术报告；Qwen3-VL 技术报告（仍用 $2\times 2$ MLP merger）；对照 LLaVA-NeXT 的 AnyRes 切块。
