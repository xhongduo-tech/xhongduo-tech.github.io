## 核心结论

Logit Lens、Activation Patching、Sparse Autoencoder，分别解决三个不同问题：模型“中间层在想什么”“哪个组件真的起作用”“这些高维激活能不能拆成可命名的方向”。三者连起来，就是一个完整闭环：

观察 $\rightarrow$ 干预 $\rightarrow$ 方向分解

可以写成一条最短工作流：

| 阶段 | 工具 | 直接回答的问题 | 输出形式 |
|---|---|---|---|
| 观察 | Logit Lens | 某层是否已经表达出目标信息 | token 概率、logit 排名 |
| 干预 | Activation Patching | 哪个头、哪层、哪段残差流对结果有因果作用 | 恢复率、$\Delta$logit |
| 分解 | SAE / SNMF | 起作用的激活能否拆成稳定、可解释的特征方向 | 稀疏特征、方向强度 |

这里的“残差流”是 Transformer 主干里持续累加信息的向量通道，可以把它理解成“每层都往同一条总线上写信息”。“因果作用”指某个组件不是只和答案相关，而是替换它之后，答案真的跟着变化。

对初学者最重要的结论是：  
只看 Logit Lens，不足以证明某层“负责”某个能力；它只能说明信息已经存在。要证明“谁在用这份信息”，必须做 Activation Patching。再往下，如果想知道“这份信息具体是怎样编码的”，才轮到 SAE。

一个玩具例子是句子：

> The Eiffel Tower is located in the city of ___

如果在 GPT-2 的第 8 层做 Logit Lens，发现 `Paris` 的概率已经升到 93%，说明模型在这一层的表示里，答案几乎已经显性存在。它未必已经输出 `Paris`，但“会说 Paris”这件事已经写进中间状态。

一个真实工程例子是微调审计。假设你拿到一个来源不明的窄域微调模型，不知道它被喂过什么数据。对比 base model 和 finetuned model 在同一批普通文本上的中层激活差，再用 Logit Lens 投影，就可能直接看到 `culinary`、`recipe`、`cat` 这类词持续冒出来。再结合 patching 和 steering，就能定位“微调痕迹”到底写在哪些层、哪些头、哪些方向上。

---

## 问题定义与边界

这类解释性工具，不是在回答“模型总体为什么聪明”，而是在回答更窄、更可验证的问题：

某个具体能力，是如何被表示、被调用、被修改的？

这里的“能力”可以是：

| 能力 | 观察对象 | 验证手段 | 限用场景 |
|---|---|---|---|
| 地理事实回忆 | 中间层 token 分布 | patch 某层残差流或注意力头 | 自回归语言模型 |
| 指代消解 | IOI 任务中的名称相关 heads | patch 特定 head 输出 | Transformer |
| 微调域偏移 | base/ft 激活差 | patch + steering | 有可比对的 base 模型 |
| 风格偏置 | MLP 激活方向 | SAE/SNMF 特征解释 | 激活可稳定采样 |

本文的边界也要说清楚。

第一，只讨论 deterministic 的 Transformer 语言模型。“deterministic”指同一个输入、同一套参数下，前向传播结果固定，不涉及采样噪声对内部分析的干扰。  
第二，只讨论 activation-level 分析，也就是直接看和改中间激活，不讨论 RL 策略网络、多模态交叉注意力、外部工具调用链。  
第三，只讨论“局部机制解释”，不宣称这些方法能给出完整的人类级语义解释。

一个经典问题是 IOI，Indirect Object Identification，白话说就是“句子里两个名字容易混淆时，模型最后到底该指向谁”。例如：

> When Mary and John went to the store, John gave a drink to ___

正确补全应偏向 `Mary`。如果模型把 `Mary` 和 `John` 搞反，问题就变成：到底是哪几个注意力头在把“正确受词”搬运到最后预测位置？这就是著名的 Name Mover 机制分析问题。

因此，本文不是做黑盒评测，而是做三段式定位：

1. 信息是否已经出现。
2. 哪个组件真正使用了它。
3. 它在高维空间里沿哪些方向编码。

---

## 核心机制与推导

### 1. Logit Lens：把中间层直接投到词表

“unembed 矩阵”是把最终隐藏状态变成词表分数的线性映射。Logit Lens 的思路很直接：既然最后一层能这样投，那中间层也可以先做归一化，再提前投一次。

$$
\text{LogitLens}(\mathbf{h}_\ell)=\text{LN}(\mathbf{h}_\ell)W_U
$$

其中：

- $\mathbf{h}_\ell$ 是第 $\ell$ 层残差流状态；
- $\text{LN}$ 是 LayerNorm，白话说是把数值尺度整理到模型习惯的范围；
- $W_U$ 是输出到词表的 unembedding 矩阵。

如果某层投影后，`Paris` 的 logit 已经远高于其他词，就说明“Paris 相关信息”已经出现在这一层的表示里。  
例如一个简化数字：

$$
P(\text{Paris}\mid \mathbf{h}_8)=0.93
$$

这不等于第 8 层“决定输出 Paris”，只等于“第 8 层的状态，拿去直接读，已经非常像 Paris”。

### 2. Activation Patching：把干净激活换回去，测因果恢复

Activation Patching 的基本动作是：  
先准备一对输入，一个 `clean` 会得到正确答案，一个 `corrupt` 会得到错误答案。然后把 `clean` 输入在某个组件上的激活，替换到 `corrupt` 输入对应位置上，看结果是否恢复。

$$
\hat{y}=M(x^{\text{corrupt}};\; a_i \leftarrow a_i^{\text{clean}})
$$

常见度量是目标 logit 差恢复量，例如在 IOI 中：

$$
\Delta L = \text{logit}(\text{Mary})-\text{logit}(\text{John})
$$

再定义恢复率：

$$
\text{Recovery}=\frac{\Delta L_{\text{patched}}-\Delta L_{\text{corrupt}}}{\Delta L_{\text{clean}}-\Delta L_{\text{corrupt}}}
$$

如果第 9 到 11 层某些 heads 的 patch recovery 超过 90%，说明这些头对 Name Mover 机制有强因果贡献。这里的“注意力头”可以白话理解为“同一层里负责不同信息路由的小子模块”。

### 3. SAE：把 MLP 激活拆成稀疏特征

MLP 激活往往是高维连续向量，直接看很难解释。Sparse Autoencoder，简称 SAE，是一种带稀疏约束的自编码器。白话说，它强迫模型用“尽量少的特征开关”去重建原始激活，于是容易长出更接近“概念方向”的表示。

设输入激活为 $\mathbf{x}$，编码为 $\mathbf{f}$：

$$
\mathbf{f}=\phi(W_e\mathbf{x}+b_e), \quad \hat{\mathbf{x}}=W_d\mathbf{f}+b_d
$$

目标函数常写成：

$$
\mathcal{L} = \|\mathbf{x}-\hat{\mathbf{x}}\|_2^2 + \lambda \|\mathbf{f}\|_1
$$

其中 $\|\mathbf{f}\|_1$ 是稀疏正则，意思是希望只有少数特征被点亮。  
如果某个特征只在“法餐、烘焙、食谱”文本上稳定激活，那它就可能对应一个“烹饪域方向”。

### 4. 三者如何形成闭环

可以把流程压缩成下面这个示意：

| 步骤 | 输入 | 操作 | 结果 |
|---|---|---|---|
| 观察 | 中间层残差流 | Logit Lens 投影 | 看见候选 token 提前出现 |
| 干预 | clean/corrupt 激活 | patch 某层/某头 | 验证哪个组件导致恢复 |
| 分解 | 高影响层的 MLP 激活 | SAE/SNMF 分解 | 找到可命名特征方向 |

所以它不是三种彼此竞争的方法，而是三种粒度不同的方法。  
Logit Lens 先告诉你“哪里可能有东西”；Activation Patching 再告诉你“哪里真的有用”；SAE 最后告诉你“这个有用的东西长什么样”。

---

## 代码实现

下面先给一个最小可运行玩具例子，用纯 Python 模拟 “logit lens 观察 + patching 恢复”。它不是 Transformer 真实现，但逻辑结构与真实分析一致。

```python
import math

VOCAB = ["London", "Paris", "Berlin"]

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def logit_lens(hidden, unembed):
    # hidden: [d], unembed: [d][vocab]
    logits = []
    for j in range(len(unembed[0])):
        logits.append(sum(hidden[i] * unembed[i][j] for i in range(len(hidden))))
    return logits

def patch_activation(corrupt_hidden, clean_hidden, idxs):
    patched = corrupt_hidden[:]
    for i in idxs:
        patched[i] = clean_hidden[i]
    return patched

# 2维隐藏状态，设计成第二维越大越偏向 Paris
unembed = [
    [0.2, 0.1, 0.0],   # dim 0
    [0.1, 2.0, 0.3],   # dim 1 strongly supports Paris
]

clean_h = [1.0, 2.0]      # clean prompt
corrupt_h = [1.0, 0.1]    # corrupt prompt loses the Paris feature

clean_logits = logit_lens(clean_h, unembed)
corrupt_logits = logit_lens(corrupt_h, unembed)

clean_probs = softmax(clean_logits)
corrupt_probs = softmax(corrupt_logits)

paris_id = VOCAB.index("Paris")

assert clean_probs[paris_id] > corrupt_probs[paris_id]

# 只把第2维从 clean 换回 corrupt，模拟 patch 单个关键方向
patched_h = patch_activation(corrupt_h, clean_h, idxs=[1])
patched_logits = logit_lens(patched_h, unembed)
patched_probs = softmax(patched_logits)

assert patched_probs[paris_id] == clean_probs[paris_id]
assert patched_probs[paris_id] > 0.7

print("clean Paris prob =", round(clean_probs[paris_id], 4))
print("corrupt Paris prob =", round(corrupt_probs[paris_id], 4))
print("patched Paris prob =", round(patched_probs[paris_id], 4))
```

这个例子里，“把错误激活换成正确激活”后，`Paris` 概率恢复，说明被替换的那一维对结果有因果贡献。

如果映射到真实 Transformer，典型 pipeline 是：

```python
def pipeline(model, clean_tokens, corrupt_tokens, target_token_id):
    clean_cache = run_and_cache(model, clean_tokens)
    corrupt_cache = run_and_cache(model, corrupt_tokens)

    lens_scores = {}
    for layer in range(model.n_layers):
        resid = clean_cache["resid_post", layer]
        lens_scores[layer] = logit_lens_project(model, resid)

    patch_scores = {}
    for layer in range(model.n_layers):
        for head in range(model.n_heads):
            patched_logits = run_with_patch(
                model=model,
                base_tokens=corrupt_tokens,
                clean_cache=clean_cache,
                patch_site=("attn_head_out", layer, head),
            )
            patch_scores[(layer, head)] = delta_logit(
                patched_logits, target_token_id
            )

    sae = fit_sae(clean_cache["mlp_post", important_layer])
    features = sae.encode(clean_cache["mlp_post", important_layer])

    return lens_scores, patch_scores, features
```

实现时有三个关键点。

第一，缓存中间激活。否则每 patch 一个头都重新完整前向，成本很高。  
第二，patch scope 要尽量小。不要一上来全层全位置替换，先只看最后一个位置、只看特定 head 输出、只看候选层。  
第三，SAE 不要只追求重建误差低，还要检查特征是否可复现、可命名、可 patch 验证。

一个真实工程例子是微调追踪。你有 base model 和 finetuned model：

1. 用相同普通文本跑两边，记录每层残差流差值 $\bar{\delta}_\ell$。  
2. 对 $\bar{\delta}_\ell$ 做 Logit Lens，看看哪些域词汇被稳定抬高。  
3. 对高差异层做 patching，验证这些偏移是否真的改变目标输出。  
4. 对该层 MLP 激活训 SAE，拆出“烘焙方向”“宠物方向”“广告语气方向”等可解释特征。

这样你拿到的不再只是“这个模型像是被微调过”，而是“第 10 层若干头和第 12 层 MLP 特征在持续写入烹饪域偏置”。

---

## 工程权衡与常见坑

最常见的误判，是把“可读出”误认为“有因果作用”。

例如 Logit Lens 显示第 8 层就已经出现 `Paris`，但你 patch 第 8 层相关头时，目标 logit 几乎不恢复。这意味着什么？  
意味着第 8 层确实携带了 `Paris` 线索，但最后真正决定输出的，可能是更后面的组件。前者是 correlation，后者才接近 causal role。

下面是一个常见风险表：

| 风险 | 典型表现 | 后果 | 缓解措施 |
|---|---|---|---|
| 只看 Logit Lens | 中间层早早出现正确词 | 误判为“该层负责答案” | 必做 patching 验证 |
| patch 范围过大 | 一次替换整层残差流 | 很难定位真实责任组件 | 先从 head、position 级别细化 |
| clean/corrupt 对不合理 | 两个输入差太多 | patch 信号混入语义分布偏差 | 构造最小对照对 |
| SAE 过稀疏 | 特征很好看但不稳定 | 学到伪方向 | 调 $\lambda$，加跨批次复现检验 |
| SAE 只看重建 | 特征不可命名 | 难用于工程决策 | 加人工标签或 patch 验证 |

Activation Patching 的另一个现实问题是贵。粗略估计如下：

| patch 粒度 | 前向次数 | 适用阶段 |
|---|---|---|
| 整层残差流 | 低 | 粗筛候选层 |
| 单层单位置 | 中 | 定位时间步 |
| 单层单 head | 高 | 定位路由组件 |
| 单神经元/单 SAE 特征 | 很高 | 精细机制分析 |

所以实操顺序一般不是从最细开始，而是：

1. 先做 Logit Lens，看哪些层开始出现目标 token。  
2. 再 patch 整层或整类模块，粗筛高影响区。  
3. 最后只对候选层做 head-level patching。  

这比全模型暴力枚举更现实。

SAE 的坑更隐蔽。稀疏正则太强，特征会看起来很“干净”，但其实是不稳定分桶；太弱，又退化成普通压缩器，没有解释性。工程上通常要同时看三件事：

| 指标 | 含义 |
|---|---|
| 重建误差 | 原激活保留了多少 |
| 稀疏度 | 每次点亮多少特征 |
| 因果可验证性 | patch 或 steering 后输出是否按预期变化 |

如果一个 SAE 特征只能在可视化上讲故事，不能在 patching 或 steering 中复现作用，它就不该被当成可靠机制结论。

---

## 替代方案与适用边界

如果任务目标不是做完整机制闭环，也有更轻量的替代方案。

| 方法 | 相比 Logit Lens / Patching / SAE 的位置 | 优点 | 局限 |
|---|---|---|---|
| Tuned Lens | Logit Lens 的替代 | 对中间层投影更稳 | 需额外训练投影器 |
| Patchscope | Patching 的轻量前筛 | 先找高影响区域，省计算 | 解释粒度较粗 |
| Head swapping | 局部替代 patching | 适合快速看 head 功能 | 不如严格 patch 精确 |
| SNMF | SAE 的替代 | 更强调非负、部件化分解 | 灵活性较弱 |
| Steering vectors | patch 后的干预延伸 | 可直接做控制实验 | 不能单独证明机制 |

“Tuned Lens”可以白话理解为“专门为每一层训练一个读出头”，它通常比直接复用最终 unembed 更稳，因为中间层分布与最后一层并不完全同构。如果你只想看趋势，不追求纯参数共享的解释，Tuned Lens 往往更实用。

“Patchscope”适合大模型粗定位。新手可以把它理解成：  
先用便宜方法找到“最爱干扰答案的位置”，再把精细 patching 火力集中在那里，而不是全图扫描。

“SNMF”是 Sparse Non-negative Matrix Factorization，白话说是“用非负部件拼出原向量”。它不一定比 SAE 更强，但在一些需要更稳定、可组合语义部件的场景里更容易解释。

这些方法的适用边界也很清楚：

1. 如果没有 clean/corrupt 成对样本，Activation Patching 很难严格做。  
2. 如果模型是多模态或强工具调用系统，单看残差流可能漏掉关键信号。  
3. 如果你只关心“模型有没有某种域偏置”，Logit Lens 或 Tuned Lens 可能已经够用，不必上 SAE。  
4. 如果你要做安全审计、微调追踪、能力定位，最好还是走完整闭环，否则容易把相关性当机制。

---

## 参考资料

1. Learn Mechanistic Interpretability, “Logit Lens and Tuned Lens”. 用于理解中间层投影、层间语义演化与 Tuned Lens 的动机。  
2. Learn Mechanistic Interpretability, “Activation Patching”. 用于理解 clean/corrupt 对照、组件替换和因果恢复度量。  
3. Learn Mechanistic Interpretability, “Finetuning Traces in Activations”. 用于理解微调前后激活差、域痕迹识别与 patching 在微调审计中的应用。  
4. Anthropic 等关于 Transformer circuits 与 IOI/Name Mover 的机制解释工作。用于理解注意力头级别因果机制分析。  
5. Sparse Autoencoder 相关论文与实现资料。用于理解高维激活的稀疏方向分解、特征可解释性与重建权衡。  
6. SNMF 与非负分解相关资料。用于理解当 SAE 不稳定时的替代方向分解方法。  
7. 实际开源工具链如 TransformerLens。用于缓存激活、做 logit lens、head patching 与残差流分析的工程实现。
