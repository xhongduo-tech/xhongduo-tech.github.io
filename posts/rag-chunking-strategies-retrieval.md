## 核心结论

`Chunking` 是把文档切成可检索单元的策略。白话说，它决定检索系统一次“看哪一小块”。在 RAG 里，检索器搜索的不是整篇文档，而是 chunk，所以 chunk 的大小、边界、重叠方式，直接决定三件事：能不能召回证据、召回时噪声有多大、答案能不能带着完整引用返回。

最实用的判断标准不是“切得细不细”，而是“一个 chunk 是否尽量覆盖一个最小可回答单元”。最小可回答单元，指回答一个局部问题所需的最小完整证据，比如一个制度条款、一段 API 调用步骤、一个报错的处理规则。

下面这张表可以先建立直觉：

| 维度 | chunk 太小 | chunk 太大 | 合理 chunk |
|---|---|---|---|
| 召回 | 证据被切碎 | 相关内容被稀释 | 更容易命中 |
| 噪声 | 较低 | 较高 | 可控 |
| 引用 | 不完整 | 冗长 | 可直接引用 |

玩具例子很直接。一篇 2000 字的重置密码说明文档，如果整篇作为一个 chunk，检索向量会混入大量“账户安全”“短信验证”“管理员审批”之类无关内容，相关信号被稀释。如果切成过短的小块，“前置条件”和“重置步骤”又可能分到不同 chunk，模型只拿到半段答案。合理的 chunking 不是追求最短，而是让一个 chunk 尽量自包含一个局部问题的完整证据。

---

## 问题定义与边界

本文讨论的是“如何切文档”，不是 embedding 模型怎么选，也不是 rerank 怎么做。`Embedding` 可以理解为把文本映射成向量，方便按语义相似度检索；`rerank` 可以理解为对初召回结果再做一次更精细排序。它们都重要，但都发生在 chunk 已经被定义之后。

问题边界如下：

| 讨论内容 | 本文是否覆盖 |
|---|---|
| 文档切分策略 | 是 |
| 召回指标 | 是 |
| embedding 选型 | 否，提及但不展开 |
| rerank / 重排 | 否，提及但不展开 |
| 生成模型提示词 | 否，仅作为下游结果 |

为什么边界要说清楚？因为很多 RAG 调优失败，并不是模型不够强，而是检索单元一开始就设计错了。用户问“怎么重置密码”，系统真正做的是：从很多 chunk 里找最像“重置密码步骤”的几段，再把它们送给模型。如果“步骤”在一个 chunk，“异常情况”在另一个 chunk，而 top-k 只命中前者，那么模型就会给出看似正确、实际不完整的答案。

所以，RAG 的第一层对象不是文档，而是 chunk。检索质量先受 chunk 设计约束，再受检索器和生成模型影响。

---

## 核心机制与推导

设文档集合经过切分后得到 $C=\{c_1,c_2,\dots,c_n\}$，查询为 $q$，检索器给每个 chunk 一个打分 $s(q,c_i)$，最终返回前 $K$ 个结果：

$$
TopK(q)=\operatorname{arg\,topK}_{c_i \in C} s(q,c_i)
$$

常见召回指标是：

$$
Recall@K = \frac{|Rel(q) \cap TopK(q)|}{|Rel(q)|}
$$

其中 $Rel(q)$ 表示与问题 $q$ 真实相关的证据集合。白话说，`Recall@K` 衡量“该找到的证据，有多少真的进了前 K 个结果”。

chunking 为什么会影响召回？因为它改变了 $c_i$ 的粒度和边界，从而改变了打分对象本身。原本一段完整证据，如果被切成两半，每一半与 query 的相似度都可能下降；原本一句关键规则，如果被埋在 800 token 的大 chunk 中，相关信号又可能被无关内容稀释。

文档流可以写成：

`文档 D -> 切成 chunks C -> 检索打分 s(q, c_i) -> TopK -> 送入模型 -> 生成答案`

看一个数值化玩具例子。假设文档共 260 token，问题的核心证据跨越第 95 到 155 token。

- `chunk_size=100, overlap=0`  
  切分结果：`[1-100] [101-200] [201-260]`  
  证据被切断，前 100 里只有上半段，101-200 里只有下半段。
- `chunk_size=100, overlap=20`  
  切分结果：`[1-100] [81-180] [161-260]`  
  第二个 chunk 完整覆盖 95-155，证据第一次以“可回答单元”的形式出现。

这里 overlap 的作用不是简单重复文本，而是减少“边界切断”。边界切断，指一个完整证据刚好跨过切分边界，导致没有任何单个 chunk 能独立支持答案。

但 `Recall@K` 也不是唯一指标。一个系统可能 Recall 很高，却把很多重复 chunk 一起送进上下文，造成噪声上升、引用混乱、答案变长却不更准。所以还要一起看 precision、groundedness 和答案正确率。`Groundedness` 可以理解为答案是否真正站得住脚，是否能被检索证据支持。

真实工程例子比玩具例子更常见：企业 API 文档里，一个调用流程常分散在“前置权限”“请求参数”“错误码处理”三个小节。如果你只按固定 token 切，常会把参数定义和异常处理拆开。结果是检索命中了“调用步骤”，却漏掉“调用前必须开通权限”这条硬约束。模型据此生成的答案语法正确，但业务上错误。这不是生成模型的问题，而是 chunk 把证据结构打碎了。

---

## 代码实现

工程上不要按字符数硬切，而要按 tokenizer 的 token 数切。因为模型上下文窗口按 token 计费和限制，不按字符计数。中文里“字符数接近”并不等于“token 数接近”。

下面给一个可运行的简化实现。它先按句子组织，再按 token 预算聚合，并保留 metadata。这里的 `metadata` 就是来源信息，比如标题、文件名、chunk 编号，后续做引用和回溯都要靠它。

```python
import re
from typing import List, Dict

def count_tokens(text: str) -> int:
    # 玩具实现：用词和标点近似 token，真实工程应替换为目标模型 tokenizer
    return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))

def split_document(sentences: List[str], source_file: str, section_title: str,
                   max_tokens: int = 20, overlap_sentences: int = 1) -> List[Dict]:
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current and current_tokens + sent_tokens > max_tokens:
            text = " ".join(current)
            chunks.append({
                "text": text,
                "metadata": {
                    "source_file": source_file,
                    "section_title": section_title,
                    "chunk_id": len(chunks),
                }
            })
            tail = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = tail + [sent]
            current_tokens = count_tokens(" ".join(current))
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        chunks.append({
            "text": " ".join(current),
            "metadata": {
                "source_file": source_file,
                "section_title": section_title,
                "chunk_id": len(chunks),
            }
        })
    return chunks

sentences = [
    "重置密码前，账户必须完成邮箱验证。",
    "进入设置页后点击安全中心。",
    "选择重置密码并输入短信验证码。",
    "如果三次失败，账户会被临时锁定。"
]

chunks = split_document(sentences, "help.md", "密码管理", max_tokens=16, overlap_sentences=1)

assert len(chunks) >= 2
assert chunks[0]["metadata"]["source_file"] == "help.md"
assert any("账户必须完成邮箱验证" in c["text"] for c in chunks)
assert any("输入短信验证码" in c["text"] for c in chunks)
```

这个实现有三个关键点。

第一，切分预算是 token，不是字符。  
第二，overlap 以句子为单位，而不是机械截尾字符串。  
第三，metadata 和文本一起存，避免“检索到了，但不知道原文在哪”。

常用参数可以先这样起步：

| 参数 | 作用 | 常见起点 |
|---|---|---|
| `chunk_size` | 控制上下文长度 | 200-500 tokens |
| `overlap` | 减少边界切断 | 10%-20% |
| `top_k` | 控制进入上下文的 chunk 数 | 3-10 |

评测流程也要同步建立，否则调 chunking 只能靠感觉：

| 输入 | 输出 |
|---|---|
| query、chunk 列表、标准证据 | `Recall@K` |
| query、top-k 结果、相关标签 | precision |
| query、最终答案、标准答案 | 正确率 |
| 答案、引用 chunk | groundedness |

最小工程闭环是：先做一个固定切分基线，再离线跑一组 query-evidence 样本，最后只改 chunking 参数观察指标变化。这样你能知道效果变化到底来自切分，还是来自别的模块。

---

## 工程权衡与常见坑

chunk 太小和 chunk 太大都不是“错误”，它们只是对应不同失败模式。太小会丢完整性，太大会丢区分度。实际任务要在“证据完整”和“检索可分辨”之间取平衡。

下面这张表是工程里最常见的问题：

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 按字符数切 | token 超限或切得不均 | 按 tokenizer 计数 |
| chunk 过小 | 证据不完整 | 以“最小可回答单元”为目标 |
| chunk 过大 | 噪声多 | 先章节、段落感知切分 |
| overlap 过高 | 重复内容挤占 top-k | 控制重叠并去重 |
| 只看 Recall@K | 忽略最终可用性 | 加 precision / groundedness / 正确率 |
| 盲目 semantic chunking | 算力高但收益不稳 | 先做离线评测 |

真实工程里还有一个高频坑：只保留子 chunk，不保留父结构。比如命中了一个参数说明片段，但你已经丢掉它属于哪个章节、哪个接口、哪份文档。结果模型能回答，却很难引用，前端也没法给用户展示“答案来自哪里”。这时更稳的方案通常是“父子双层结构”：索引子 chunk，引用父 section。

另一个常被忽略的问题是 top-k 被重复 chunk 占满。高 overlap 会让多个近似 chunk 同时进入结果集，看起来召回很高，实际送进模型的是同一段话的不同副本。这样不仅浪费上下文，还会让模型误以为某条证据被多次独立支持。去重、MMR 或按父节点聚合，通常要和 overlap 一起设计。

---

## 替代方案与适用边界

固定长度切分只是基线，不是标准答案。不同文档类型适合不同策略。

| 方案 | 优点 | 缺点 | 适合场景 |
|---|---|---|---|
| 固定长度切分 | 简单稳定 | 易切断语义 | 基线、通用文档 |
| 标题/段落切分 | 结构清晰 | 粒度不均 | 文档、手册、制度 |
| 句子级 overlap | 降低边界丢失 | 重复内容多 | 证据跨句场景 |
| semantic chunking | 更贴近语义 | 成本高、收益不稳 | 有足够评测资源时 |
| late chunking / contextual retrieval | 上下文保留更强 | 实现复杂 | 高价值、高精度场景 |

对零基础读者，一个实用判断可以记成三句话。

短新闻、短公告、FAQ 这类文本短且结构简单，固定长度 chunk 往往已经够用。  
制度、SOP、技术手册、API 文档这类文本有明显章节结构，优先保留标题和段落边界。  
证据经常跨句、跨小节时，再考虑 overlap、父子结构，或者更复杂的 late chunking。

`Late chunking` 可以白话理解为：先让模型或编码器看到更大上下文，再在后面阶段决定如何切或如何取证据。它的优点是上下文保留更强，缺点是实现复杂、成本更高。`Semantic chunking` 则是试图按语义自然断点切，而不是按固定长度切。它听起来更高级，但研究和工程实践都表明，收益并不稳定，不能默认优于简单基线。

所以正确顺序通常是：先做固定长度或标题感知切分，建立离线评测基线；如果指标和业务表现仍然不够，再升级更复杂方案。不要反过来。

---

## 参考资料

1. [LangChain Retrieval Docs](https://docs.langchain.com/oss/python/langchain/retrieval)
2. [LlamaIndex Node Parser Usage Pattern](https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/)
3. [Rethinking Chunk Size For Long-Document Retrieval: A Multi-Dataset Analysis](https://arxiv.org/abs/2505.21700)
4. [Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation](https://arxiv.org/abs/2504.19754)
5. [Is Semantic Chunking Worth the Computational Cost?](https://arxiv.org/abs/2410.13070)
