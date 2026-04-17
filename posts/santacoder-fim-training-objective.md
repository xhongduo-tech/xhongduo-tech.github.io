## 核心结论

SantaCoder 的 FIM（Fill-in-the-Middle，中间填空，意思是“给定前后文，补中间缺失内容”）训练目标，本质上不是换了一种模型结构，而是把同一段代码重排成更适合“补洞”的训练样本。它仍然是解码器式自回归模型，只是训练时不再只学“从左往右续写”，还学“看到前后文后重建中间”。

对 SantaCoder 这类代码模型，关键设计有三点：

| 设计点 | 含义 | 结论 |
|---|---|---|
| FIM token | 用 `<fim-prefix>`、`<fim-suffix>`、`<fim-middle>` 标记三段代码 | 让普通自回归模型能表达“中间补全”任务 |
| PSM / SPM | 两种训练排布：Prefix-Suffix-Middle 或 Suffix-Prefix-Middle | SPM 在 single-line infilling 上通常更优 |
| FIM rate | 训练样本中有多少比例被改写成 FIM 格式 | 50% 左右最稳，100% 会伤害纯左到右能力 |

SantaCoder 1.1B 参数、2048 上下文窗口、以 Python/Java/JavaScript 为主训练，采用“普通自回归 + FIM 混合训练”的思路。这样做的目标很直接：不牺牲编辑器里的普通补全能力，同时让“在光标处补几行代码”这个能力几乎零额外成本地出现。

一个最重要的经验结论是：FIM 不能全开。FIM rate 从 0% 提高到 50% 时，infilling 能力明显增强；继续提高到 75% 或 90%，通常还能维持；但到 100% 时，模型的纯左到右建模能力开始下降。这说明 FIM 不是替代 AR（Autoregressive，自回归，意思是“按顺序一个 token 一个 token 预测”），而是和 AR 配比使用。

---

## 问题定义与边界

先定义问题。代码补全有两种常见形态：

1. 左到右续写：给定开头，预测后面内容。
2. 中间填空：给定前文和后文，补中间缺失内容。

传统自回归模型天然擅长第一种，不天然擅长第二种。因为它在训练时只看左边，不看右边。但真实编辑器场景里，用户经常在一个已有函数中间插入几行逻辑，这时右边上下文其实非常关键。

FIM 的做法是把一段原始序列拆成三部分：

| 术语 | 含义 | 白话解释 | 典型标记 |
|---|---|---|---|
| prefix | 缺口前的内容 | 光标前已经写好的部分 | `<fim-prefix>` |
| middle | 要预测的缺口 | 现在要补上的那段 | `<fim-middle>` |
| suffix | 缺口后的内容 | 光标后已经存在的部分 | `<fim-suffix>` |

例如原始代码是：

```python
def add_one(x):
    y = x + 1
    return y
```

如果把 `y = x + 1` 当作缺口，那么：

- prefix: `def add_one(x):\n`
- middle: `    y = x + 1\n`
- suffix: `    return y\n`

FIM 训练不会直接把整段按原顺序喂给模型，而是改写成某种特殊顺序，例如：

- PSM: `<fim-prefix> prefix <fim-suffix> suffix <fim-middle> middle`
- SPM: `<fim-suffix> suffix <fim-prefix> prefix <fim-middle> middle`

这样模型在生成 `middle` 时，已经“见过” prefix 和 suffix。

这里有两个边界必须说清：

| 边界问题 | 推荐做法 | 原因 |
|---|---|---|
| 切分粒度 | 先在原始文本上切 prefix/middle/suffix，再 tokenization | 避免切分后破坏上下文边界 |
| 打包时机 | 在 pack 前做 FIM，即 context-level FIM | 否则 prefix/suffix 可能被截断 |
| 作用范围 | 通常在单文件、单函数或局部上下文内使用 | 跨文件 FIM 依赖检索，不是 FIM 本身解决的 |
| 是否替代 AR | 不替代 | 真实系统仍需要普通续写能力 |

所谓 context-level FIM，就是先决定这一条训练样本的前中后三段，再送去分词和长度裁剪；而不是先把数据拼满训练长度后，再在 token 序列里随便切一刀。后者很容易把 prefix 或 suffix 截断，结果训练到的不是“看全前后文补中间”，而是“看一半前后文猜中间”，这会直接损伤效果。

---

## 核心机制与推导

FIM 的核心不是新损失函数，而是条件变了。模型预测目标仍是交叉熵，只是条件从“只看左边”变成了“看前后文，再按顺序生成中段”。

设原始序列为 $x_1, x_2, ..., x_n$，其中中间缺口是 $x_{p+1:p+m}$。那么 FIM 目标可以写成：

$$
\mathcal{L}_{FIM}
=
-\sum_{t=1}^{m}
\log P_\theta \left(
x_{p+t}\mid \text{prefix},\ \text{suffix},\ x_{p+1:p+t-1}
\right)
$$

这句话的意思是：损失只对 middle 这段求和。prefix 和 suffix 主要作为条件输入，不是本次要被重建的目标。

玩具例子最容易看懂。

原始代码：

```python
if score >= 60:
    print("pass")
else:
    print("fail")
```

假设缺口是 `print("pass")`。则：

- prefix: `if score >= 60:\n    `
- middle: `print("pass")\n`
- suffix: `else:\n    print("fail")\n`

如果用 PSM，训练输入可近似写成：

```text
<fim-prefix>if score >= 60:
    <fim-suffix>else:
    print("fail")
<fim-middle>print("pass")
```

如果用 SPM，则顺序交换：

```text
<fim-suffix>else:
    print("fail")
<fim-prefix>if score >= 60:
    <fim-middle>print("pass")
```

两者最终监督目标都一样，都是让模型生成 `print("pass")`。差别在于：生成时谁离 `<fim-middle>` 更近。SPM 往往把 prefix 放得更靠近待预测内容，因此在单行 infill 上更占优，尤其当中间缺失内容更强依赖左边局部语法时更明显。

可以把整个过程理解成一个三步流：

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 从原始代码抽样一个 middle span | 得到 prefix / middle / suffix |
| 2 | 按 PSM 或 SPM 重排并插入特殊 token | 构造训练输入 |
| 3 | 只对 middle 位置计算 loss | 学到中间补全能力 |

这里还有一个经常被忽略的推导点：为什么 100% FIM 不好？因为训练分布和推理分布变了。如果所有样本都改写成 FIM，模型就很少看到“普通原始顺序”的代码序列，于是纯续写时的困惑度会上升。换句话说，模型被过度训练成“先读特殊控制 token，再处理重排结构”，自然会削弱最常见的左到右生成分布。

因此混合目标更合理。设 FIM rate 为 $r$，则总目标可写成：

$$
\mathcal{L}
=
(1-r)\mathcal{L}_{AR}
+
r\mathcal{L}_{FIM}
$$

它不是严格按 batch 加权的解析式，而是对训练数据采样策略的近似描述：有一部分样本保持原始 AR 形式，另一部分样本改写成 FIM 形式。$r=0$ 就是纯 AR，$r=1$ 就是纯 FIM。

从实验结论看，50% 左右通常是拐点：FIM 已经学得足够强，但 AR 分布还没有被严重破坏。

---

## 代码实现

先看一个最小可运行的玩具实现。它不依赖大模型，只演示 FIM 样本如何构造，并验证 PSM / SPM 转换是否保留了原始三段信息。

```python
def make_fim_sample(text: str, start: int, end: int, mode: str = "PSM"):
    assert 0 <= start <= end <= len(text)
    prefix = text[:start]
    middle = text[start:end]
    suffix = text[end:]

    if mode == "PSM":
        prompt = f"<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>{middle}"
    elif mode == "SPM":
        prompt = f"<fim-suffix>{suffix}<fim-prefix>{prefix}<fim-middle>{middle}"
    else:
        raise ValueError("mode must be PSM or SPM")

    return {
        "prefix": prefix,
        "middle": middle,
        "suffix": suffix,
        "prompt": prompt,
    }


code = "def add_one(x):\n    y = x + 1\n    return y\n"
target = "    y = x + 1\n"
start = code.index(target)
end = start + len(target)

psm = make_fim_sample(code, start, end, "PSM")
spm = make_fim_sample(code, start, end, "SPM")

assert psm["middle"] == target
assert spm["middle"] == target
assert "<fim-prefix>" in psm["prompt"]
assert "<fim-suffix>" in psm["prompt"]
assert "<fim-middle>" in psm["prompt"]
assert psm["prefix"] + psm["middle"] + psm["suffix"] == code
assert spm["prefix"] + spm["middle"] + spm["suffix"] == code

print(psm["prompt"])
print("---")
print(spm["prompt"])
```

真实工程里，流程不是“把 middle 直接附在 prompt 后面让模型看见”，而是：

1. 训练时：middle 作为目标序列参与 teacher forcing。
2. 推理时：只给 `<fim-prefix> prefix <fim-suffix> suffix <fim-middle>`，让模型生成后续 token，直到停止。

如果直接使用 SantaCoder，推理代码大致是这样：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "bigcode/santacoder"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True
)

prefix = "def greet(name):\n    "
suffix = "\n    return message\n"

prompt = f"<fim-prefix>{prefix}<fim-suffix>{suffix}<fim-middle>"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    return_token_type_ids=False
)

outputs = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=False
)

text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(text)

assert "<fim-prefix>" in prompt
assert "<fim-suffix>" in prompt
assert "<fim-middle>" in prompt
```

这里的真实工程例子是编辑器插件。用户在函数中间留了一个缺口：

```python
def greet(name):
    # 光标在这里
    return message
```

插件把光标前的内容作为 prefix，把光标后的内容作为 suffix，构造 FIM prompt 后发给模型。模型生成：

```python
message = f"Hello, {name}"
```

这和普通补全接口几乎一样，差别只是 prompt 结构变了。也正因此，SantaCoder 可以在不换模型架构的前提下支持 infill。

FIM rate 的经验趋势也可以直接整理成表：

| FIM rate | AR perplexity 趋势 | infill pass rate 趋势 | 结论 |
|---|---|---|---|
| 0% | 最稳 | 几乎没有专门能力 | 只能做普通续写 |
| 25% | 基本稳定 | 明显提升 | 已开始具备实用 infill |
| 50% | 仍稳定 | 通常最佳 | 最常见推荐点 |
| 75% | 小幅波动 | 仍较强 | 可用于偏编辑器场景 |
| 90% | 多数设置仍可接受 | 接近峰值 | 需要关注 AR 回退 |
| 100% | 明显变差 | 不一定继续升 | 过度偏向 FIM |

---

## 工程权衡与常见坑

第一个权衡是训练目标配比。FIM 是能力增强，不是主任务替换。对代码模型来说，真实流量里仍然大量存在普通续写，因此保留 AR 样本是必要的。只要 FIM rate 过高，模型就可能学到一种“只擅长在特殊 token 控制下补中间”的分布，普通补全反而退化。

第二个权衡是 PSM 和 SPM。两者都能工作，但偏好不同。

| 方案 | 优点 | 风险 | 适合场景 |
|---|---|---|---|
| PSM | 顺序更直觉，阅读友好 | middle 离 prefix 可能较远 | 较长代码块、结构化上下文 |
| SPM | prefix 更靠近待生成段，单行表现常更好 | 对某些长 suffix 结构不够直观 | single-line infill、局部编辑 |

第三个权衡是切分位置。middle 太短，训练信号有限；middle 太长，又可能把任务变成“重写整段函数”。工程上通常会按字符或 token 长度随机采样一个 span，而不是固定只挖单行。

常见坑主要有四个。

| 坑 | 现象 | 原因 | 处理方式 |
|---|---|---|---|
| pack 后再做 FIM | 生成缺失内容质量很差 | prefix/suffix 被截断 | 必须在 context-level 做 FIM |
| 100% 训练成 FIM | 普通补全变差 | AR 分布被破坏 | 保留 50% 左右 AR 样本 |
| 只训练 PSM | 单行补全不够强 | middle 与左局部连接不紧 | 混合 PSM 和 SPM |
| 特殊 token 管理错误 | 模型把标记当普通文本输出 | tokenizer / special tokens 配置不一致 | 明确注册 `<fim-*>` token |

真实工程里最典型的错误，是编辑器插件先把整文件截成 2048 token，再从截断结果里硬切 prefix 和 suffix。这样做看起来“也给了前后文”，但实际上光标附近的重要语义很可能已经丢失，尤其是函数声明、缩进层级、返回变量名等。正确做法是先围绕光标裁上下文，再构造 FIM，再分词，再做长度控制。

---

## 替代方案与适用边界

如果需求只是“根据已有开头继续写下去”，那就不一定需要 FIM。纯 AR checkpoint 更简单，也更贴近传统代码补全服务。SantaCoder 提供 `no-fim` 这类变体，本质上就是为这种场景保留的回退选项。

可以用下面这张表快速判断：

| 需求 | 推荐方案 | 原因 |
|---|---|---|
| 普通左到右续写 | `no-fim` 或低 FIM rate 模型 | 分布最匹配 |
| 单行中间补全 | SPM 或混合 PSM/SPM | 左局部连续性更强 |
| 多行局部重构 | 混合 FIM，优先 50% 左右 rate | 兼顾 AR 与 infill |
| 长文档生成 | AR 为主，FIM 为辅 | 长程顺序建模仍更重要 |
| 跨文件补全 | 检索 + AR/FIM | FIM 本身不解决跨文件召回 |

还要明确一个适用边界：FIM 解决的是“条件生成顺序”问题，不解决“知识召回”问题。比如模型要补的中间代码依赖另一个文件中的类定义，那真正需要的是检索、索引或仓库级上下文拼接，而不是单纯提高 FIM rate。

另一个边界是：SPM 在 single-line infilling 上更优，不代表永远优于 PSM。对于长段文档、注释块或右侧结构很复杂的场景，PSM 仍然可能更稳定。工程上最稳妥的办法不是押注一种排布，而是把两者都纳入训练分布，让模型学会适应不同缺口形态。

---

## 参考资料

- SantaCoder 官方 README: https://huggingface.co/bigcode/santacoder/blob/main/README.md
- SantaCoder README 中的 FIM 使用示例: https://huggingface.co/bigcode/santacoder/blame/main/README.md
- Efficient Training of Language Models to Fill in the Middle 论文综述: https://www.emergentmind.com/articles/2207.14255
- Fill-in-the-Middle 方法综述: https://www.emergentmind.com/topics/fill-in-the-middle-fim-993a87a4-2524-4748-943d-0b0ae4448bd1
- FIM 论文总结与 context-level / PSM / SPM 说明: https://imaddabbura.github.io/papers-summaries/fim.html
- FIM rate 消融结果的工程化总结: https://www.ownyourai.com/efficient-training-of-language-models-to-fill-in-the-middle/
- 论文镜像页面: https://www.researchgate.net/publication/362324241_Efficient_Training_of_Language_Models_to_Fill_in_the_Middle
