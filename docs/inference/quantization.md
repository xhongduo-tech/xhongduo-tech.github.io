---
title: 推理量化的误差与执行成本
description: 分清权重、激活与KV量化，用误差、真实内存和内核支持共同判断收益。
kind: topic
status: planned
---

# 推理量化的误差与执行成本

::: info 专题计划 · 待研究
计划先做线性层量化的 CPU 教学实验，再测试一个有明确内核支持的模型格式。尚无本人的压缩、质量或提速结果。
:::

## 要解决的问题与前置依赖

量化使用更低精度表示部分数值，有机会减少存储与数据移动，但还需要考虑误差、打包布局、反量化和实际硬件内核。文件更小不必然代表峰值显存更小或请求更快。

先掌握矩阵乘法、浮点数和误差范数，能解释线性层 $Y=XW$ 的 shape；随后了解 [KV Cache](/inference/kv-cache)和[服务指标](/inference/serving-systems)。不必先实现完整低精度内核。

## 先说明量化的对象

| 维度 | 必须写清的内容 |
| --- | --- |
| 权重／激活／KV | 哪些张量降精度，哪些仍保留高精度 |
| W4A16 等名称 | 描述权重与激活位宽；没有自动说明 KV、累加精度与实际编码 |
| 粒度 | per-tensor、per-channel 或 per-group；group size 和对应 axis |
| 数值格式 | 对称／非对称整数，FP8、FP4 等浮点格式不能混为一谈 |
| 参数与布局 | scale、zero point、clip、packing、累加和输出 dtype |
| 过程 | PTQ 在训练后转换；QAT 在训练中模拟或适配量化误差 |

这是记录方案的坐标，不是建议把所有格式同时实践。

## 最小公式：对称整数教学模型

对 $b\ge2$，选择对称范围 $q_{\max}=2^{b-1}-1$：

$$
s=\frac{\max |x|}{q_{\max}},\qquad
q=\operatorname{clip}
\left(\operatorname{round}(x/s),-q_{\max},q_{\max}\right),
\qquad \hat{x}=sq.
$$

全零张量需单独处理，例如令 $s=1,q=0$；round 的规则也应固定。这个式子刻意使用对称整数范围，不覆盖所有 INT4 编码、非对称量化或低精度浮点格式。

单看 $\lVert W-\hat W\rVert$ 不够。对线性层还应检查：

$$
E_Y=\frac{\lVert XW-X\hat W\rVert_F}
{\lVert XW\rVert_F+\varepsilon}.
$$

输入分布会影响输出误差，因此校准数据应代表计划部署的任务；小线性层误差更低也不直接保证整模型能力更好。


## 为什么相同权重误差会产生不同输出误差

令 $\Delta W=W-\hat W$。线性层误差满足：

$$
\|X\Delta W\|_F^2
=\operatorname{tr}\!\left(
\Delta W^{\mathsf T}X^{\mathsf T}X\Delta W
\right).
$$

因此误差落在哪个输入方向，与误差本身的大小同样重要。$X^{\mathsf T}X$ 把校准输入的统计带入了目标；这为理解 GPTQ 的近似二阶信息提供了入口，但不等于已经推导完整 GPTQ 更新算法。

用两个给定近似候选构造反例：

$$
W=I_2,\qquad
\hat W_A=\operatorname{diag}(0.9,1),\qquad
\hat W_B=\operatorname{diag}(1,0.9).
$$

两者的权重 Frobenius 误差都是 0.1，但输出不同：

| 输入 $X$ | A 的输出绝对误差 | B 的输出绝对误差 | A／B 的相对误差 |
| --- | ---: | ---: | ---: |
| $\operatorname{diag}(4,1)$ | 0.4 | 0.1 | 约 9.701%／2.425% |
| $\operatorname{diag}(1,4)$ | 0.1 | 0.4 | 约 2.425%／9.701% |

分母都是 $\|XW\|_F=\sqrt{17}$，暂不计数值稳定项。输入分布变了，候选排序便反转。**这些候选是数学上给定的近似，不声称来自同一个 min-max 量化器，也不是 GPTQ／AWQ 的运行结果。** 整模型还有非线性与后续层，不能把此处的排序直接当作任务能力排序。

### 一个真实可算的量化与元数据对照

对一维教学权重 $[0,0.1,0.2,2]$ 使用前述 4 bit 对称公式，$q_{\max}=7$：

- 整个张量一个 scale：$s=2/7$，整数为 $[0,0,1,7]$，反量化为 $[0,0,2/7,2]$。
- 每两个元素一组：第一组的 scale 为 $0.1/7$，能恢复 $[0,0.1]$；第二组仍恢复为 $[2/7,2]$。

第一种 Frobenius 误差约为 0.131708，第二种约为 0.085714；分组改善了这个例子的误差，但增加 scale 数。这里没有 zero point，且使用指定的对称范围，与部分实际 INT4 格式不同。

如果整数真正紧密打包，scale 使用 FP16，则四个 4 bit 权重占 2 bytes；一个 scale 时共 4 bytes，两个 scale 时共 6 bytes。原始四个 FP16 权重为 8 bytes，因此不能把实际压缩比一律写成 $16/4=4$。

一般可先按下式记账，再加实际 padding、对齐与布局开销：

$$
M_{\mathrm{weight}}
\approx N\frac{b}{8}
+G(s_{\mathrm{scale}}+s_{\mathrm{zero}}).
$$

$G$ 是实际 group 数，scale／zero 的存储量按 bytes 计。逐行分组的尾部不能随意跨行合并；缓存、激活与 workspace 仍须另算。在浮点数组里保存整数值，则没有获得上述打包占用。

## 三条论文路线分别做什么

- **GPTQ：** 利用近似二阶信息进行训练后权重量化，关注逐层误差补偿。[GPTQ v2](https://arxiv.org/abs/2210.17323v2)
- **AWQ：** 利用激活统计识别重要权重通道并选择缩放，研究低位宽 weight-only 量化；不是仅按权重绝对值逐项保留。[AWQ v6](https://arxiv.org/abs/2306.00978v6)
- **SmoothQuant：** 通过等价缩放将 activation outlier 带来的量化困难迁移到权重侧，研究权重与激活低精度执行。[SmoothQuant v7](https://arxiv.org/abs/2211.10438v7)

等价缩放可先用可逆对角矩阵 $D$ 检查 $XW=(XD^{-1})(DW)$，再分别观察低精度转换引入的误差。该恒等式是理解入口，不能替代三篇论文各自的优化目标和实现。


### 等价缩放保持什么，又改变了什么

取 $X=[4,1]$、$W=I_2$、$D=\operatorname{diag}(4,1)$。变换后：

$$
XD^{-1}=[1,1],\qquad
DW=\operatorname{diag}(4,1),\qquad
(XD^{-1})(DW)=[4,1].
$$

未量化时结果不变；激活最大值从 4 降到 1，权重最大值却从 1 升到 4。难点在两边重新分配，而非凭恒等式消失。使用同样粗糙的 per-tensor、2 bit 对称教学量化时，变换前后都可能只得到 $[4,0]$：一边被精确表示，不代表两边乘积就更准。

真实模型中的缩放还必须覆盖对应接口的所有相关消费者，避免只补偿一条分支。SmoothQuant 的[归一化与线性层联合调整](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py#L19)是源码阅读入口；须解释每个被修改参数的补偿关系，不能只看函数名。

## 从算法目标追到实际执行

| 层次 | 固定源码入口 | 要核对的因果关系 |
| --- | --- | --- |
| 数据如何进入误差目标 | [GPTQ.add_batch](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L32) | 输入统计怎样累积，代码中的权重转置约定如何对应 $Y=XW$ |
| 如何补偿量化误差 | [GPTQ.fasterquant](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py#L60) | 阻尼、分块、逆信息与后续权重更新怎样衔接；矩阵近似的边界是什么 |
| 数值模拟是否改变物理表示 | [AWQ pseudo_quantize_tensor](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py#L61)与[real_quantize_model_weight](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py#L126) | 反量化浮点模拟与替换为实际低位宽线性层是不同交付物 |
| 缩放后模型是否仍等价 | [SmoothQuant 的缩放实现](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py#L19) | 先在不量化时核对补偿恒等，再评价量化后的误差 |

这些入口已核验存在，逐算法源码分析仍待完成。上述 AWQ 固定快照的真实替换路径要求 zero point 配置，不能把本页对称整数 toy 默认为它的逐项 reference。尤其不能用一份朴素四舍五入代码，同时声称完成 GPTQ、AWQ 和 SmoothQuant 的实现验收。

## 数值误差与运行瓶颈需要两条证据链

量化可能降低权重或 KV 的存储与访存压力；实际收益还取决于是否使用匹配的打包内核、反量化／累加成本、batch 大小和瓶颈位置。若时间主要花在排队、通信或另一个未量化张量上，权重更小可能没有明显延迟收益。

可用如下近似问题检查思路，但不能把它当作实测：数据移动下界随字节数减少，计算下界由算力与算术量决定；总时间还受同步、布局转换和不可重叠部分限制。因此“模型文件缩小”只支持存储主张，不能独自支持速度主张。

| 对照或否证 | 固定什么 | 哪种失败需要重新解释 |
| --- | --- | --- |
| 量化／反量化 reference | 相同 rounding、group、dtype 和输入 | 数值不能由已记录参数复算 |
| 未量化的缩放补偿 | 关闭量化，仅改变等价变换 | 输出超出容差，说明变换或消费者覆盖不完整 |
| 校准域与外部输入 | 冻结转换方案，分开评价两个输入集 | 排序反转或误差增加，说明分布依赖不可忽略 |
| 模拟／真实 packed 执行 | 同格式与权重，核对实际内核和布局 | 名称相同却使用回退路径，不能解释为目标内核加速 |
| 容量／服务表现 | 同任务质量、负载和测量边界 | 显存降低但延迟不降，应保留容量收益并收窄速度结论 |

## CPU 小实验：观察误差从哪里来

计划固定一组小矩阵和随机种子：

1. 实现 FP32 reference、朴素 per-tensor 和分组对称整数转换；输出 scale、量化范围及反量化后的数值。
2. 分别用正常输入、少量激活 outlier、分布变化的输入，比较权重误差与输出相对误差。
3. 检查全零、常量、边界值和 group 尾部，固定 rounding 与 clipping 规则。
4. 分开计算理论位宽占用、scale／zero point 开销和实际打包后的字节数；在浮点数组里存整数并没有得到真正的低位宽存储。

这不是 GPTQ／AWQ／SmoothQuant 完整复现，也不能用 NumPy 的运行时间证明 GPU 量化内核加速。

## 后续 GPU 实测

只选一个受支持格式和一个小模型，确认 checkpoint、group size、packing、引擎及 GPU 架构匹配。用同一模型的 BF16／FP16 参考版本对比，冻结 tokenizer、输入集、上下文与并发。

记录任务质量、峰值显存、加载／编译开销、稳态 TTFT／TPOT、吞吐，以及实际运行内核和是否回退。权重量化与 KV 量化单独改变；不把“更低位宽”当作无条件最优，也不以单一困惑度指标代表全部任务能力。

## 验收产物

- 一份量化配置表，能确定张量、位宽、粒度、scale、packing 与累加精度。
- CPU 量化／反量化脚本、误差图、边界测试和理论／实际字节对照。
- 硬件兼容性与质量评价记录；没有运行的项目保留待实测状态。

## 常见错误

- 只看模型文件大小，忽略 KV、临时区和内核工作空间。
- 把 W4A16 解释为所有计算和缓存都是 4 bit。
- 在不同数据、不同采样或不同模型版本间直接比较质量与速度。
- 使用未经核实的转换工具名称代替格式说明，或把内核回退后的结果误称为低位宽执行。

## 一手资料与版本

资料与具体源码入口核验日期：**2026-11-17**。新增矩阵、误差和字节数均为解析教学对照，尚未运行个人模型实验。

- [GPTQ，arXiv:2210.17323v2](https://arxiv.org/abs/2210.17323v2)，2023-03-22；[作者源码固定快照](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf)。
- [AWQ，arXiv:2306.00978v6](https://arxiv.org/abs/2306.00978v6)，2026-04-25；[作者源码固定快照](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61)。
- [SmoothQuant，arXiv:2211.10438v7](https://arxiv.org/abs/2211.10438v7)，2024-03-29；[作者源码固定快照](https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5)。

上述源码快照是后续阅读定位，不表示已跑通它们的当前工具链；真正实验需另核对安装依赖与支持矩阵。
