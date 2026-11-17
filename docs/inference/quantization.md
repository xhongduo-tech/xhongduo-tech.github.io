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

## 三条论文路线分别做什么

- **GPTQ：** 利用近似二阶信息进行训练后权重量化，关注逐层误差补偿。[GPTQ v2](https://arxiv.org/abs/2210.17323v2)
- **AWQ：** 利用激活统计识别重要权重通道并选择缩放，研究低位宽 weight-only 量化；不是仅按权重绝对值逐项保留。[AWQ v6](https://arxiv.org/abs/2306.00978v6)
- **SmoothQuant：** 通过等价缩放将 activation outlier 带来的量化困难迁移到权重侧，研究权重与激活低精度执行。[SmoothQuant v7](https://arxiv.org/abs/2211.10438v7)

等价缩放可先用可逆对角矩阵 $D$ 检查 $XW=(XD^{-1})(DW)$，再分别观察低精度转换引入的误差。该恒等式是理解入口，不能替代三篇论文各自的优化目标和实现。

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

核验日期：**2026-11-16**。

- [GPTQ，arXiv:2210.17323v2](https://arxiv.org/abs/2210.17323v2)，2023-03-22；[作者源码固定快照](https://github.com/IST-DASLab/gptq/tree/2d65066eeb06a5c9ff5184d8cebdf33662c67faf)。
- [AWQ，arXiv:2306.00978v6](https://arxiv.org/abs/2306.00978v6)，2026-04-25；[作者源码固定快照](https://github.com/mit-han-lab/llm-awq/tree/d6e797a42b9ef7778de8ee2352116e0f48a78d61)。
- [SmoothQuant，arXiv:2211.10438v7](https://arxiv.org/abs/2211.10438v7)，2024-03-29；[作者源码固定快照](https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5)。

上述源码快照是后续阅读定位，不表示已跑通它们的当前工具链；真正实验需另核对安装依赖与支持矩阵。
