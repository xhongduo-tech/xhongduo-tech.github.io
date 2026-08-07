---
title: loss mask：只对回答部分计算损失的实现细节与常见错误
date: 2026-08-07
---

# loss mask：只对回答部分计算损失的实现细节与常见错误

<div class="epigraph">
<p>该算的账，一分不能少；不该算的账，一分不能多。</p>
<footer>—— 引意自会计学常识</footer>
</div>

<div class="article-byline">
<p>第四级 · 大模型微调 ｜ 大模型微调知识树 第二章 ｜ 2026-08-07</p>
</div>

## 为什么从 loss mask 开始

指令微调的损失函数，第一行文档就写着「只对回答部分计算损失」。但这句话从纸面到代码，藏着整整一层**对齐（alignment）**问题：logits 在哪个位置、labels 指向哪个 token、哪些位置该被忽略——错一位，损失就算错了对象；错一批，模型就学会了「预测用户的话」。

loss mask 是微调代码里**最容易写错、又最不容易被发现**的部分：训练不报错，loss 会下降，模型却悄悄「变笨」——因为它学的是「如何续写整段对话」而不是「如何作答」。本节把 loss mask 的数学、代码与错误模式一次讲透。<span class="marginnote">为什么错误难发现？因为「只算回答损失」与「全部算损失」两种写法，训练曲线都很平滑，loss 都在降。区别要到评测对话质量时才暴露：全算损失训练的模型，回答开头喜欢复述用户问题、甚至反串用户口吻。所以这是一个「不报错但有毒」的坑。</span>

## 1 目标对齐：每个位置该预测哪个 token

因果语言模型在每个位置只做一件事：**用当前位置的 logits 预测下一个 token**。设输入序列为 $x_1, x_2, \dots, x_T$，则位置 $t$ 的模型输出 $\mathrm{logits}_t$ 应该去拟合目标 $y_{t+1}$（即 $x_{t+1}$）。

于是**labels 就是 input_ids 左移一位**：第 $t$ 个 label 是第 $t+1$ 个 token。这是全篇理解 loss mask 的锚点——**「mask 位置 $t$」意味着「放弃让模型预测 $x_{t+1}$」**，而不是「忽略 $x_t$ 本身」。

把一条「指令 + 回答」摊开看：

| 位置 t | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| input_ids | `<bos>` | 请 | 翻译 | 这段 | 话 | Sure | , | 我来 |
| labels[t]（= x[t+1]） | 请 | 翻译 | 这段 | 话 | Sure | , | 我来 | … |
| mask[t] | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |

关键在**过渡位置**：第 4 行 `话` 的 target 是第 5 个 token `Sure`——这是回答的第一个词。**它必须被计入损失**：教模型「读完整条指令后，接出第一个回答词」，正是指令微调的核心目标。若有人把 mask 做成「指令部分全部遮住」时多遮一位，第一个回答词就被误杀了（见第 3 节错误二）。

## 2 公式解析：带掩码的交叉熵

把「只算回答」写进公式，就是在标准交叉熵里乘一个掩码 $m_t$：

$$
L = -\frac{1}{\sum_{t} m_t} \sum_{t=1}^{T-1} m_t \, \log P_\theta\big(x_{t+1} \mid x_{1:t}\big), \qquad m_t = \begin{cases} 1, & x_{t+1} \text{ 是回答 token} \\ 0, & \text{否则} \end{cases}
$$

逐项拆解：

- $P_\theta(x_{t+1} \mid x_{1:t})$：位置 $t$ 处模型对下一个 token 的预测概率，这是所有 LM 损失的基础项；
- $m_t$：**掩码开关**。当目标 $x_{t+1}$ 属于回答（assistant 内容）时 $m_t=1$，属于指令/系统提示词/角色标记/填充时 $m_t=0$；
- $m_t \log(\cdot)$：$m_t=0$ 时这一项为 0，即「这个位置不贡献任何梯度」；
- $\sum_t m_t$：**分母用「被计数的位置数」而非总长度**——这样不同长短的样本、不同遮罩比例的样本，损失尺度可比；
- 求和到 $T-1$：最后一个位置没有「下一个 token」，不计。

**直觉**：掩码相当于给损失函数发了一张「豁免名单」——名单外的位置照常学习，名单内的位置（指令、角色标记、填充）直接跳过。模型只从「回答该怎么写」里学梯度，永远不学「用户下一句会说什么」。<span class="marginnote">为什么「预测用户话」有害？因为模型一旦擅长预测用户，生成时就容易「替用户把话说完」，或出现「我猜你想问……」的越权行为。把用户位置 mask 掉，等于从训练上掐断这条坏习惯的来源。</span>

## 3 实现细节与常见错误

掩码的原理一句话，代码里的坑却有七八个。逐个过。

**正确的实现骨架**（PyTorch 风格）：

```python
labels = input_ids.clone()                     # 先复制，再定点替换
labels[mask == 0] = -100                        # mask=0 的位置设为 -100
loss = F.cross_entropy(logits.view(-1, V),
                       labels.view(-1), ignore_index=-100)
```

HF 的 CausalLM 在内部已做 `shift_logits = logits[:, :-1]`、`shift_labels = labels[:, 1:]` 的位移，所以我们传入的 `labels` 就是「第 t 个 label = 第 t+1 个 token」的天然对齐形态。`-100` 是 PyTorch 交叉熵约定俗成的**忽略索引**——等于告诉损失函数「这些位置不要算」。

**错误一：把 mask 当「回答掩码」而不是「目标掩码」。** 有人想「只算回答」，就把回答位置标成 1、其余为 0，然后在 `logits` 上乘 mask——这是把 mask 用错了对象。掩码必须作用在**目标（labels）**上，而不是 logits 上：logits 每个位置都要算，只是「目标不是回答的」位置被忽略。

**错误二：过渡位置被误杀。** 如第 1 节的表格，mask 从「第一个回答词」开始。若 `labels[:, :inst_len + 1] = -100`（多遮一位），第一个回答词就消失了。判断方法很简单：**检查最后一个指令位置的目标是不是回答的第一个 token**。

**错误三：用 0 代替 -100。** 若把不该算的位置设成 0 而不是 -100，模型会被训练去「预测 pad token」——padding 区域全是 0，模型学出「看到一切就输出 pad」的废习惯。`-100` 的「忽略」语义与 `0` 的「目标」语义完全不同，不可互换。<span class="marginnote">`ignore_index` 不只限于 -100，任何负数都行；但 HF 生态默认 -100，社区代码、可视化工具都按它约定，擅自改别的值会与下游工具不兼容。</span>

**错误四：多轮对话只 mask 了第一轮。** 多轮样本的掩码要覆盖**所有非 assistant 内容**：system、每一轮 user、角色标记、轮次终止 token 全部 mask；每一轮 assistant 回答都保留（上一节《多轮对话数据》已强调）。

**错误五：padding 没 mask。** batch 里长度不足的样本会被 pad，pad 位置的目标也必须 -100，否则损失里混入「预测 pad」的噪音项。这要求数据整理（collate）阶段**同步生成 attention_mask 与 label mask**，两者边界必须一致。

**错误六：归一化分母选错。** 有人用 `loss.sum() / total_tokens` 而不是 `loss.sum() / masked_tokens`。当样本中指令占比高时，分母被虚增，loss 数值被压低——虽然相对大小仍可比，但与 `masked_tokens` 分母的 loss 不具可比性，模型选择（早停阈值、不同配比对比）会失真。

## 4 自查清单：写对 loss mask 的四问

写完掩码逻辑，用这四问自查一遍，基本可以排除绝大多数错误：

1. **过渡位置对吗？** 最后一个指令位置的目标，是不是回答的第一个 token？
2. **角色标记 mask 了吗？** `<|im_start|>assistant` 这类标记自身该被 mask（它只是语法，不是回答内容）。
3. **padding mask 了吗？** labels 的 pad 位置是否也是 -100，与 attention_mask 一致？
4. **分母用对了吗？** 归一化是否只对「被计数的位置」求和？

再补一个可视化排查技巧：训练时把 `logits.argmax(-1)` 解码出来，与原文逐词对比——如果模型在指令位置就能「预测出用户的话」，说明 mask 没生效或作用错了位置。一眼就能看出问题在哪。<span class="marginnote">很多开源微调框架（LLaMA-Factory、Axolotl）已经把 loss mask 封装成「传一条 instruction_mask 就行」的接口，但这不等于可以不懂原理——它们同样会把「多轮」「padding」「特殊 token」这类边缘情况留给使用者处理，第 5 节的排查技能始终有用。</span>

### 一个完整的 batch 组装示例

把「attention_mask + label mask」同步组装好，是错误五的根治方案。下面这个数据整理函数值得逐行读：

```python
def collate(batch, tokenizer, pad_token_id=0):
    input_ids, attn_masks, label_masks = [], [], []
    for item in batch:
        ids = item["input_ids"]                 # 已按模板序列化
        # 1. attention_mask：真 token 为 1，padding 为 0
        attn = [1] * len(ids)
        # 2. label_mask：assistant 内容为 1，其余为 0（含 padding 位置）
        lm = item["response_mask"]              # 长度与 ids 一致
        input_ids.append(ids); attn_masks.append(attn); label_masks.append(lm)

    # 统一 padding 到 batch 最大长度
    max_len = max(len(x) for x in input_ids)
    labels = []
    for ids, lm in zip(input_ids, label_masks):
        pad_n = max_len - len(ids)
        labels.append(ids + [pad_token_id] * pad_n)          # 先补成整行
        lm = lm + [0] * pad_n                                 # padding 处 mask 为 0
        labels[-1] = [t if m else -100 for t, m in zip(labels[-1], lm)]
    return torch.tensor(input_ids), torch.tensor(attn_masks), torch.tensor(labels)
```

注意最后一行的双层替换：**先按 label_mask 把非回答位置替换成 -100，再对 padding 位置同样替换**——两条路径最终殊途同归，保证「非回答」与「padding」在 labels 里都是 `-100`，而 `attention_mask` 只关心「是不是真 token」。两个 mask 各管一件事，边界必须对齐。

## 5 小结

- **loss mask 的作用对象是「目标」而非 logits**：位置 $t$ 的 target 是 $x_{t+1}$，mask 掉位置 $t$ = 放弃让模型预测 $x_{t+1}$。
- **过渡位置必须保留**：最后一个指令位置的目标是回答第一个词，这是指令微调最重要的一个梯度来源。
- **`-100` 是忽略索引，不是 0**：0 会让模型去预测 pad，必须用 `ignore_index=-100`。
- 多轮样本要 mask **所有非 assistant 内容**；padding 位置也要 mask，且与 attention_mask 对齐。
- 归一化分母应取**被计数的位置数**，否则 loss 数值失真、跨配比不可比。
- 自查四问 + 解码可视化，能在训练早期就发现掩码错误，而不是等模型「变笨」了才回头查。

在下一节，我们离开「单条样本怎么算损失」，进入「多条样本怎么拼」：**packing——样本拼接、跨样本注意力隔离与位置编码处理**。
