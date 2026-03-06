## LayerNorm 成为 Transformer 标配，是因为它不依赖 batch 统计

BatchNorm 在 CNN 时代极其成功，但到了 Transformer 和序列模型里，它对 batch 统计的依赖反而成为负担。LayerNorm 的关键优势在于：**对每个样本单独做归一化，不需要跨样本共享统计量。**

---

## 公式与含义

给定一个 token 的隐藏向量 `x`，LayerNorm 计算：

```text
μ = mean(x)
σ^2 = mean((x-μ)^2)
LN(x) = γ (x-μ) / sqrt(σ^2 + ε) + β
```

这里均值和方差是在特征维上计算，而不是在 batch 维上。  
这意味着即便 batch size 为 1，LayerNorm 也完全可用。

---

## 为什么这对 Transformer 很重要

LLM 训练和推理常有以下特点：

- batch size 可能受显存限制较小
- 序列长度可变
- 服务场景中经常是单条样本推理

若还依赖 BatchNorm，就会面临训练与推理统计不一致、小 batch 噪声大等问题。LayerNorm 则天然避免了这些麻烦。

---

## Pre-LN 与 Post-LN

LayerNorm 在 Transformer 中的放置位置非常关键。

- Post-LN：子层后归一化，原始 Transformer / BERT 常见
- Pre-LN：子层前归一化，现代大模型更常见

Pre-LN 的优势主要在于梯度更稳定，更适合深层堆叠。因此许多大规模预训练模型都转向了 Pre-LN 或其变体。

---

## RMSNorm 为什么又进一步简化

RMSNorm 去掉了减均值步骤，只保留基于均方根的缩放。  
这意味着它假设“控制尺度”比“强制零均值”更关键。实践上，在许多 LLM 中它效果与 LayerNorm 接近甚至更好，同时实现更简单、数值更稳。

这说明归一化真正关键的部分，常常是控制激活尺度，而不一定是完整标准化。

---

## 小结

LayerNorm 之所以成为 Transformer 标配，并不是因为它比 BatchNorm 更“先进”，而是因为它更匹配序列建模的现实约束：可变长度、小 batch、单样本推理。它通过对每个样本独立归一化，让训练与部署都更稳定。

Pre-LN 与 RMSNorm 的进一步流行，也说明现代大模型在归一化问题上的目标越来越明确：优先保障深层训练稳定，而不是机械追求最标准的统计形式。
