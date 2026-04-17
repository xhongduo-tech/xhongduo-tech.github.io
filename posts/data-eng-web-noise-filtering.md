## 核心结论

网页噪声过滤的目标，不是“把 HTML 变短”，而是把真正可读、可分析、可训练的正文块从页面模板里分离出来。这里的“正文块”指承载主要语义的信息区域，例如新闻正文、博客正文、商品详情说明；“boilerplate”指反复出现但不承载主内容的模板噪声，例如导航栏、广告、评论区、相关推荐、版权声明。

工程上最稳妥的做法，不是依赖单一算法，而是做成级联过滤。一个可落地的顺序是：先用规则做粗筛，再用统计密度做细筛，最后用 DOM 层级关系补正文、修边界。对应到常见方法，就是把 Readability、正文密度法、DRCT 组合成“粗筛→细筛→结构润色”的流水线。原因很直接：网页噪声类型多，单轮规则容易漏删，单一分数又容易误删，只有多轮累计，才能把导航、广告、评论、推荐等不同噪声逐步压下去。

给零基础读者一个最直观的理解：先把网页拆成一棵树，再删掉名字像 `menu`、`nav`、`ad`、`comment` 的枝干；接着在剩余节点里找“文字多、链接少”的段落；最后检查这些段落的父节点和兄弟节点，看看有没有被切断的正文补回来。这就是现代网页正文提取系统最核心的思路。

| 过滤阶段 | 主要方法 | 看什么特征 | 主要目标 |
| --- | --- | --- | --- |
| 粗筛 | 黑白名单规则、标签规则 | `class`/`id`、标签名、位置、可见性 | 快速删掉明显噪声 |
| 细筛 | Readability、TBD、浅层文本特征 | 链接密度、文本密度、文本长度、标点比例 | 找出最像正文的块 |
| 结构润色 | DRCT、父子兄弟聚合 | DOM 深度、路径、邻近关系、父块一致性 | 补全断裂正文，抑制孤立噪声 |
| 复核 | 规则或分类器二次清洗 | 评论特征、推荐特征、视觉特征、重复模板特征 | 去掉残留低价值内容 |

可以把这套流程记成一句话：**先删明显噪声，再给候选块打分，最后按 DOM 关系把正文拼完整。**

---

## 问题定义与边界

输入通常是原始 HTML，或者浏览器渲染后的 DOM。DOM 是“文档对象模型”，白话说就是浏览器把网页解析成的一棵树，每个标签都是树上的一个节点。输出不是完整页面，而是一组被判定为正文的节点，或者这些节点拼接后的纯文本。

这个问题的边界要先说清楚。正文提取不是通用的页面理解系统，它不负责回答“用户最关心哪个区域”，也不保证保留页面所有有用内容。它只解决一个更窄的问题：从网页中尽可能准确地保留主文本，去掉模板噪声，供摘要、搜索索引、向量化、训练语料清洗、知识库构建等下游任务使用。

常见噪声类型和特征如下：

| 噪声类型 | 常见位置 | 典型标签/结构 | 典型特征 |
| --- | --- | --- | --- |
| 导航栏 | 页头、侧边栏 | `nav`、`ul/li/a` | 链接多、文本短、重复词多 |
| 广告 | 页头、正文中插、侧边栏 | `iframe`、`aside`、推广容器 | 链接多、图片多、语义不连续 |
| 评论区 | 页尾 | 列表容器、多条重复结构 | 用户名/时间戳密集、层级重复 |
| 推荐阅读 | 正文后 | 卡片列表 | 标题短、链接密集、模板相似 |
| 版权信息 | 页尾 | `footer` | 文本固定、跨页重复 |
| 社交分享 | 标题周围、侧边浮层 | 图标组、按钮组 | 文本少、链接比极高 |

很多新手会把“噪声过滤”和“删除所有无用标签”混为一谈，这两个目标并不一样。正文提取关注的是**语义主块**，不是 HTML 清理本身。举个简单例子：

| 页面区域 | HTML 复杂度 | 对摘要/检索是否重要 |
| --- | --- | --- |
| 正文段落 | 中等 | 高 |
| 相关文章卡片 | 中等 | 低 |
| 版权声明 | 低 | 低 |
| 参数表 | 高 | 视业务而定 |

所以，“标签复杂”不等于“噪声”，“文本长”也不一定等于“正文”。评论区可能很长，但对新闻摘要通常没有价值；商品参数表可能不是自然语言段落，但对电商搜索很有价值。

玩具例子可以这样想：一个新闻页像一个装满杂物的箱子，里面既有正文书籍，也有菜单、广告单、标签纸、说明卡。正文提取不是重新整理整个箱子，而是尽量只把“书”搬出来。这个比喻只帮助理解边界，不代替技术定义。

真实工程里边界会更复杂。比如商品详情页往往既有主描述，也有参数表、问答、推荐搭配。此时“正文”不是单一自然语言段落，而可能是主描述加关键规格。如果你的下游任务是做搜索召回，保留参数表可能有价值；如果是做新闻摘要，保留评论区通常没有价值。所以正文提取始终和业务目标绑定。

输入输出可以抽象成下面这种格式：

```python
input_page = {
    "slug": "sample-news",
    "html": "<html>...</html>",
    "rendered": False
}

output_blocks = [
    {"xpath": "/html/body/main/article/p[1]", "label": "content"},
    {"xpath": "/html/body/main/article/p[2]", "label": "content"},
    {"xpath": "/html/body/footer", "label": "boilerplate"}
]
```

如果再往前走一步，可以把任务形式化为一个节点分类问题。设页面 DOM 树为 $G=(V,E)$，其中 $V$ 是节点集合，$E$ 是父子边集合。目标是学习或构造一个函数：

$$
f: V \rightarrow \{\text{content}, \text{boilerplate}\}
$$

有些系统只输出单个主块，有些系统输出多个正文块。后者更适合处理“标题 + 导语 + 正文 + 关键表格”这类非单块页面。

---

## 核心机制与推导

最常见的第一类信号是链接密度。链接密度可以理解为“一个块里有多少文字是包在链接里的”。导航栏、推荐区、标签区通常链接很多，正文段落则大多是普通文本。定义为：

$$
LinkDensity(v)=\frac{L_v}{T_v+\epsilon}
$$

其中，$L_v$ 是节点 $v$ 内链接文本字符数，$T_v$ 是总文本字符数，$\epsilon$ 是防止分母为 0 的微小值，工程里常取 $10^{-6}$ 或直接在空文本时返回 1。

例如一个段落共有 80 个字符，其中 10 个字符来自 `<a>` 标签，那么：

$$
LinkDensity=\frac{10}{80}=0.125
$$

如果阈值设为 0.3，这个块大概率保留。相反，一个导航块只有 40 个字符，但 32 个字符都在链接里，那么链接密度是 0.8，基本可以判成噪声。

第二类信号是正文密度，常写成 TBD，Text Block Density。它衡量“非链接文字相对结构标签有多密”。白话说，如果一个块有很多连续自然语言，而不是由大量标签切碎，它更像正文。一个常见形式是：

$$
TBD(v)=\frac{C_v+1}{N_v+1}
$$

其中，$C_v$ 是非链接字符数，$N_v$ 是非链接标签数。加 1 是平滑项，避免分母为 0。比如某个候选块有 500 个非链接字符、20 个非链接标签：

$$
TBD=\frac{500+1}{20+1}\approx 23.86
$$

这类值通常明显高于导航或推荐卡片，因为后者标签多、文本碎、句子短。

仅靠 TBD 还不够，因为有些评论区文本很多，TBD 也可能不低。于是工程上往往再引入标点、句长、停用词比例等“自然语言连贯性”特征。比如定义一个简化的文本得分：

$$
TextScore(v)=w_1\cdot \log(1+T_v)+w_2\cdot P_v+w_3\cdot S_v
$$

其中：
- $T_v$：总文本长度。
- $P_v$：标点密度或句子完整度分数。
- $S_v$：停用词比例或自然语言样式分数。

为什么这些特征有效？因为真正的正文通常包含完整句子，会出现句号、逗号、引号、数字和较稳定的语法模式；导航、推荐卡片、标签云则更像短碎片。

第三类信号来自 DOM 层级。DRCT 这类方法不只看单个节点，而是看节点与父节点、兄弟节点的关系。层级关系可以理解为“这个节点在树里和谁挨着、归谁管”。正文通常不是孤立单点，而是一串结构相近、深度相近、相邻分布的段落；广告和分享按钮常是孤立块，或者路径模式和正文明显不同。

所以一个更实用的综合分数会长成这样：

$$
Score(v)=\alpha\cdot TextScore(v)-\beta\cdot LinkDensity(v)+\gamma\cdot StructureScore(v)
$$

其中：
- $TextScore(v)$ 可以来自文本长度、标点数量、TBD。
- $LinkDensity(v)$ 用来惩罚链接密集块。
- $StructureScore(v)$ 反映父子兄弟聚合后的稳定性。

如果把结构分进一步拆开，可以写成：

$$
StructureScore(v)=\lambda_1\cdot ParentConsistency(v)+\lambda_2\cdot SiblingSupport(v)+\lambda_3\cdot DepthPrior(v)
$$

这里三个量分别表示：
- `ParentConsistency`：这个节点和父节点下其他候选块是否属于同一正文区域。
- `SiblingSupport`：左右兄弟节点是否也像正文。
- `DepthPrior`：该深度在目标站点上是否常出现正文。

三类分数的区别如下：

| 分数/特征 | 它回答的问题 | 对正文有利的模式 | 对噪声有利的模式 |
| --- | --- | --- | --- |
| LinkDensity | 这个块是不是“链接堆” | 链接少、自然句子多 | 导航、推荐、标签云 |
| TBD | 这个块是不是“文字密集区” | 连续文本长、标签切分少 | 卡片流、组件流、评论元数据 |
| DOM 层级得分 | 这个块能否和上下文组成正文区域 | 多个相邻段落、层级一致 | 孤立广告、浮层、杂散模块 |

玩具例子可以用两块节点对比：

- 块 A：`<div><p>这是一段 80 字的正文，其中只有 10 字是超链接。</p></div>`
- 块 B：`<ul><li><a>首页</a></li><li><a>科技</a></li>...</ul>`

块 A 的 `LinkDensity=0.125`，TBD 也高，因为有连续文本；块 B 的链接密度高、标签也碎，基本不是正文。

再看一个更接近真实站点的例子：

| 块 | 文本长度 | 链接文本长度 | 非链接标签数 | LinkDensity | TBD | 直觉判断 |
| --- | --- | --- | --- | --- | --- | --- |
| 正文段 1 | 220 | 12 | 8 | 0.055 | 26.1 | 正文 |
| 正文段 2 | 180 | 0 | 6 | 0.000 | 25.9 | 正文 |
| 推荐区 | 90 | 60 | 12 | 0.667 | 2.4 | 噪声 |
| 评论项 | 140 | 8 | 25 | 0.057 | 5.1 | 容易误判，需要结构特征 |
| 页脚版权 | 48 | 6 | 10 | 0.125 | 3.9 | 噪声 |

这里评论项是典型边界样本。它的链接密度不高，文本也不少，但结构上往往表现为“多条重复、时间戳密集、用户名密集、列表模板明显”，所以要靠结构和后处理把它压下去。

真实工程例子是新闻聚合器。它抓取一个新闻站点后，先删除 `nav/header/footer/aside` 和 `comment/related/recommend` 命名块，再对剩余候选段落计算链接密度和文本密度，选出高分主块，最后把同一父节点下的相邻正文段补回去。这样输出给摘要模型的文本会更干净，关键词抽取和向量检索质量通常都会提升。

---

## 代码实现

下面给一个**可直接运行**的简化版实现。它不是完整工业级解析器，但完整展示了“粗筛→细筛→结构聚合”的核心逻辑。这里我们不真正解析 HTML，而是直接操作候选节点对象，目的是把机制讲清楚，并保证代码开箱可运行。

```python
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Node:
    node_id: str
    tag: str
    text_len: int
    link_text_len: int
    non_link_tags: int
    depth: int
    class_id: str
    parent_id: str | None = None
    punctuation_count: int = 0
    visible: bool = True


BLACKLIST_HINTS = {
    "nav", "menu", "ads", "ad-", "advert", "comment",
    "footer", "related", "recommend", "share", "sidebar"
}


def is_blacklisted(node: Node) -> bool:
    text = f"{node.tag} {node.class_id}".lower()
    return any(hint in text for hint in BLACKLIST_HINTS)


def link_density(node: Node) -> float:
    if node.text_len <= 0:
        return 1.0
    return node.link_text_len / node.text_len


def text_block_density(node: Node) -> float:
    non_link_chars = max(node.text_len - node.link_text_len, 0)
    return (non_link_chars + 1) / (node.non_link_tags + 1)


def punctuation_density(node: Node) -> float:
    if node.text_len <= 0:
        return 0.0
    return node.punctuation_count / node.text_len


def text_score(node: Node) -> float:
    # 文本越长、标点越自然，越像正文
    return (
        0.06 * node.text_len
        + 20.0 * punctuation_density(node)
        + 3.0 * text_block_density(node)
    )


def structure_score(node: Node, siblings: List[Node]) -> float:
    # 正文通常出现在中等深度，且有相似兄弟节点支持
    depth_bonus = 1.0 if 3 <= node.depth <= 8 else 0.2

    good_siblings = 0
    for sib in siblings:
        if sib.node_id == node.node_id:
            continue
        if abs(sib.depth - node.depth) <= 1 and sib.text_len >= 80 and link_density(sib) < 0.35:
            good_siblings += 1

    sibling_bonus = min(good_siblings * 0.5, 1.5)
    return depth_bonus + sibling_bonus


def content_score(node: Node, siblings: List[Node]) -> float:
    if not node.visible:
        return -999.0
    if is_blacklisted(node):
        return -999.0

    ld = link_density(node)
    tscore = text_score(node)
    sscore = structure_score(node, siblings)

    return tscore - 120.0 * ld + 15.0 * sscore


def group_by_parent(nodes: List[Node]) -> Dict[str | None, List[Node]]:
    groups: Dict[str | None, List[Node]] = {}
    for node in nodes:
        groups.setdefault(node.parent_id, []).append(node)
    return groups


def extract_main(nodes: List[Node]) -> List[Node]:
    parent_groups = group_by_parent(nodes)

    # 粗筛：去掉黑名单、不可见节点和极短文本
    stage1 = [
        n for n in nodes
        if n.visible and not is_blacklisted(n) and n.text_len >= 30
    ]

    # 细筛：用链接密度和正文密度做第一轮过滤
    stage2 = [
        n for n in stage1
        if link_density(n) < 0.45 and text_block_density(n) > 3.0
    ]

    if not stage2:
        return []

    # 综合打分
    ranked = sorted(
        stage2,
        key=lambda n: content_score(n, parent_groups.get(n.parent_id, [])),
        reverse=True
    )

    main = ranked[0]
    result = []

    # 结构聚合：保留与主块结构接近、得分为正的节点
    for node in ranked:
        siblings = parent_groups.get(node.parent_id, [])
        score = content_score(node, siblings)

        same_parent = node.parent_id == main.parent_id
        close_depth = abs(node.depth - main.depth) <= 1
        strong_score = score > 0

        if node.node_id == main.node_id or (same_parent and close_depth and strong_score):
            result.append(node)

    # 按原始顺序输出
    selected_ids = {n.node_id for n in result}
    return [n for n in nodes if n.node_id in selected_ids]


def demo() -> None:
    nodes = [
        Node(
            node_id="top-nav",
            tag="nav",
            text_len=60,
            link_text_len=50,
            non_link_tags=12,
            depth=2,
            class_id="top-nav",
            parent_id="header",
            punctuation_count=1,
        ),
        Node(
            node_id="article-p1",
            tag="p",
            text_len=180,
            link_text_len=8,
            non_link_tags=6,
            depth=5,
            class_id="article-body",
            parent_id="article",
            punctuation_count=12,
        ),
        Node(
            node_id="article-p2",
            tag="p",
            text_len=220,
            link_text_len=0,
            non_link_tags=7,
            depth=5,
            class_id="article-body",
            parent_id="article",
            punctuation_count=15,
        ),
        Node(
            node_id="article-p3",
            tag="p",
            text_len=160,
            link_text_len=5,
            non_link_tags=5,
            depth=5,
            class_id="article-body",
            parent_id="article",
            punctuation_count=10,
        ),
        Node(
            node_id="related-box",
            tag="div",
            text_len=90,
            link_text_len=60,
            non_link_tags=10,
            depth=4,
            class_id="related-posts",
            parent_id="article",
            punctuation_count=2,
        ),
        Node(
            node_id="comment-1",
            tag="div",
            text_len=130,
            link_text_len=4,
            non_link_tags=28,
            depth=6,
            class_id="comment-item",
            parent_id="comments",
            punctuation_count=3,
        ),
        Node(
            node_id="site-footer",
            tag="footer",
            text_len=40,
            link_text_len=10,
            non_link_tags=8,
            depth=2,
            class_id="site-footer",
            parent_id="root",
            punctuation_count=1,
        ),
    ]

    main_blocks = extract_main(nodes)
    main_ids = [n.node_id for n in main_blocks]

    assert "article-p1" in main_ids
    assert "article-p2" in main_ids
    assert "article-p3" in main_ids
    assert "top-nav" not in main_ids
    assert "related-box" not in main_ids
    assert "comment-1" not in main_ids
    assert "site-footer" not in main_ids

    print("Extracted content blocks:", main_ids)


if __name__ == "__main__":
    demo()
```

这段代码直接运行会输出：

```text
Extracted content blocks: ['article-p1', 'article-p2', 'article-p3']
```

这段代码体现了几个关键点：

| 函数 | 输入 | 输出 | 作用 |
| --- | --- | --- | --- |
| `is_blacklisted` | 节点 | 布尔值 | 根据标签名与 `class/id` 做粗筛 |
| `link_density` | 节点 | 浮点数 | 判断是否链接过密 |
| `text_block_density` | 节点 | 浮点数 | 判断是否文字密集 |
| `punctuation_density` | 节点 | 浮点数 | 判断文本是否像连续自然语言 |
| `structure_score` | 节点和兄弟节点 | 浮点数 | 用深度和邻近节点补结构分 |
| `content_score` | 节点和兄弟节点 | 浮点数 | 综合排序候选节点 |
| `extract_main` | 节点列表 | 主正文节点列表 | 完成整轮正文提取 |

如果把这段代码映射到真实 DOM，可以这样理解：

1. `Node` 对应 DOM 树里的一个候选块，例如 `<p>`、`<div>`、`<article>`。
2. `text_len` 对应节点文本长度。
3. `link_text_len` 对应该节点内部所有 `<a>` 文本长度之和。
4. `parent_id` 和 `depth` 对应节点在 DOM 树中的位置。
5. `extract_main` 对应整个正文提取流程。

如果换成真实 DOM，实现流程通常是：

1. 解析 HTML，得到 DOM 树。
2. 删除明显噪声标签和命名块。
3. 遍历候选节点，计算文本长度、链接文本长度、非链接标签数、深度、标点数。
4. 根据阈值和综合分数选主块。
5. 用父节点和兄弟节点把断开的正文补齐。
6. 再跑一轮去评论、去推荐、去版权的后处理。

真实工程例子：抓新闻页面时，正文中常常插入“相关阅读”卡片。它可能嵌在正文容器里，因此第一轮不会被删干净。正确做法是先识别主块，再对主块内部的子节点做第二轮低价值内容过滤，而不是只在全局做一次删除。

如果你要真正处理 HTML，通常还会加两层工程逻辑：

| 工程层 | 作用 | 常见实现 |
| --- | --- | --- |
| DOM 获取层 | 拿到原始或渲染后的页面结构 | `requests`、Playwright、Puppeteer |
| 特征计算层 | 计算长度、密度、路径、可见性 | DOM 遍历、XPath、CSS Selector |
| 提取决策层 | 打分、阈值过滤、结构聚合 | Readability、规则系统、分类器 |
| 后处理层 | 去残留推荐、去评论、文本清洗 | 正则、模式规则、二分类器 |

---

## 工程权衡与常见坑

最大的问题是误删和漏删。单轮规则删除很快，但容易把正文里夹带的广告说明、引用块、作者注释一并删掉；只看统计分数，又可能把长评论、参数表、论坛回复误判为正文。因此工程上通常采用“先保守筛、后局部修”。

常见坑和规避方式如下：

| 坑 | 原因 | 后果 | 规避措施 |
| --- | --- | --- | --- |
| 单轮规则误删正文 | 黑名单命中过宽 | 正文断裂 | 用 DRCT 或父兄弟补回相邻正文 |
| 评论区被当正文 | 评论字数很多、链接密度不高 | 摘要污染、向量偏移 | 加入时间戳、用户名、重复结构特征 |
| 推荐阅读残留 | 与正文共父节点 | 关键词噪声高 | 在主块内部再按链接密度清洗子块 |
| 动态页面漏正文 | 只抓初始 HTML | 正文为空或不完整 | 先跑 headless 渲染，再做提取 |
| 跨站点泛化差 | 规则过拟合某站模板 | 新站效果骤降 | 引入 DOM 特征分类器做二次判别 |
| 多语言页面效果不稳 | 标点和句长分布不同 | 阈值失效 | 按语言分桶设阈值 |
| 列表型正文被误删 | 正文本身就有很多链接 | 文档目录、FAQ 丢失 | 对文档站单独建规则或站点模板 |

多轮清洗的累积效果很重要。第一轮通常解决 60% 到 80% 的明显噪声，第二轮才能处理“长得像正文但其实不是正文”的区域，例如评论区、推荐流、问答区。对于下游是向量检索或训练语料的系统，这种增量收益很大，因为模型对系统性噪声非常敏感。

可以把误删和漏删理解成一个阈值问题。设保留阈值为 $\tau$，当 $Score(v) \ge \tau$ 时保留节点。则：

- $\tau$ 太低：漏删少，但噪声残留多。
- $\tau$ 太高：页面很干净，但正文容易断裂。

这就是为什么生产系统很少只设一个全局阈值，而是会按页面类型分桶，例如新闻页、博客页、文档页、电商页分别调参。

再看一个真实工程例子。假设你在做一个知识库构建管线，数据源包括新闻站、博客、文档站。新闻页用规则效果很好，文档站有很多目录和锚点链接，博客页底有长评论。如果你只做一次通用规则清洗，输出会混入目录、评论和推荐项。更好的做法是：

1. 原始 HTML 先做规则粗筛。
2. 渲染后的 DOM 再跑一次候选打分。
3. 主块内部再做子块级去噪。
4. 最后用一个轻量分类器复核。

动态渲染页面需要特别注意。很多站点正文是异步插入 DOM 的，只抓原始 HTML 会直接漏掉主体内容。下面这个 stub 表示工程上常见的“渲染 hook”：

```python
def get_dom(html: str, need_render: bool = False):
    if need_render:
        # 实际工程里这里接 Playwright / Puppeteer
        # 返回 JS 执行后的 DOM
        return "rendered_dom"
    return "raw_dom"
```

还有一个常被忽略的坑是评测方式。只看“提取后文本长度变短了”没有意义，真正该看的通常是：

| 指标 | 它衡量什么 | 适用场景 |
| --- | --- | --- |
| Precision | 提取出的内容里有多少真的是正文 | 噪声污染敏感任务 |
| Recall | 真正文里有多少被保留下来 | 摘要、知识抽取 |
| F1 | Precision 和 Recall 的平衡 | 通用离线评测 |
| 下游指标 | 检索 NDCG、摘要 ROUGE、分类准确率 | 生产效果验证 |

如果正文提取服务于搜索、摘要、训练数据清洗，最后一定要看下游任务是否改善，而不是只看提取器本身的分数。

---

## 替代方案与适用边界

如果站点结构稳定，比如一批新闻站或公司自有博客，规则加统计方法通常已经足够。这里“规则”就是人工写的判断条件，“统计方法”就是链接密度、文本密度、长度阈值这些数值特征。它们可解释、便于调参、成本低。

如果面对跨站点、多模板、样式变化大的页面，单纯规则就容易失效。这时可以加 DOM 特征分类器。分类器就是一个学“这个节点像正文还是像噪声”的模型。常见输入特征包括文本长度、链接密度、文本密度、DOM 深度、标签路径、视觉面积、是否可见等。它的核心不是替代规则，而是补规则的泛化能力。

一个简化的特征表示可以写成：

$$
x_v=[\text{text\_len},\text{link\_density},\text{text\_density},\text{depth},\text{visual\_area},\text{tag\_type},\text{visible}]
$$

然后训练二分类模型：

$$
P(y_v=\text{content}\mid x_v)
$$

当这个概率高于阈值时，节点保留；否则丢弃。这里的“概率”可以理解为模型对“这是正文”的信心分数。

如果再进一步，可以把结构信息建模成图或序列。比如把 DOM 路径编码进特征，或者对节点序列做分类。但工程上要注意：模型越复杂，越依赖标注数据、特征稳定性和部署成本。正文提取通常不是业务主模型，复杂度需要和收益匹配。

不同方案的适用场景如下：

| 方案 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| 纯规则 | 快、可解释、易维护 | 跨站点泛化弱 | 单站或模板稳定站群 |
| 规则 + 统计 | 成本低、效果稳 | 边界样本仍会误判 | 新闻、博客、资讯聚合 |
| 规则 + ML | 泛化更强 | 需要标注数据和训练 | 多站点采集、复杂模板 |
| 先渲染再提取 | 适配 JS 页面 | 成本高、吞吐低 | SPA、异步加载重页面 |
| 视觉分块 + DOM | 能处理布局噪声 | 实现复杂、代价高 | 高视觉差异页面 |

新手可以这样理解：

- 如果所有页面都长得差不多，规则足够。
- 如果每个站点 DOM 都不一样，就让分类器学“正文长什么样”。
- 如果页面根本不在原始 HTML 里，而是 JS 运行后才出现，那就必须先渲染，再做同样的过滤。

还有一个边界是多模态页面。比如视频页、商品页、互动页，它们的主价值可能不在大段文本，而在视频、参数表、图片说明、用户问答。此时“正文提取”只能覆盖其中一部分，不能等价于“内容理解”。

可以把不同业务下的“正文”定义差异总结为：

| 页面类型 | 常见主内容 | 不能机械删除的区域 |
| --- | --- | --- |
| 新闻页 | 标题、导语、正文 | 图注、引用块 |
| 博客页 | 正文、代码块、图片说明 | 目录、脚注 |
| 文档页 | 正文、参数表、API 示例 | 目录、锚点导航有时要保留 |
| 电商页 | 描述、规格、卖点 | 参数表、配送说明 |
| 视频页 | 标题、简介、字幕 | 时间轴、章节信息 |

所以正文提取没有放之四海而皆准的固定答案，只有**围绕下游任务定义“什么该保留”**这一条硬约束。

---

## 参考资料

| 资料 | 主要贡献 |
| --- | --- |
| Kohlschütter, Fankhauser, Nejdl, *Boilerplate Detection using Shallow Text Features*（WSDM 2010）<br>链接：https://www.wsdm-conference.org/2010/proceedings/docs/p441.pdf | 代表性浅层文本特征方法，说明仅靠少量文本特征也能有效区分正文与模板噪声 |
| Mozilla Readability 项目说明与实现<br>链接：https://github.com/mozilla/readability | 工程上最常见的正文提取实现之一，展示了基于候选打分、链接密度惩罚和逐步清洗的思路 |
| López, Silva, Insa, *Content Extraction based on Hierarchical Relations in DOM Structures*（Polibits 2012）<br>链接：https://www.scielo.org.mx/scielo.php?lng=en&nrm=iso&pid=S1870-90442012000100002&script=sci_abstract&tlng=en | 强调 DOM 父子兄弟关系，不只看单节点分数，而是利用层级关系提升正文块完整性 |
| Uțiu, Ionescu, *Learning Web Content Extraction with DOM Features*（ICCP 2018）<br>链接：https://www.researchgate.net/publication/329061153_Learning_Web_Content_Extraction_with_DOM_Features | 把文本、结构、DOM 特征组合成分类问题，展示规则系统向学习式系统演进的路线 |

可以把这四类资料理解成一条技术演进链：先有规则和浅层文本特征，再有链接密度、文本密度等统计特征，接着进入 DOM 层级聚合，最后发展到基于 DOM 特征的机器学习分类器。成熟工程系统通常不是四选一，而是把它们组合起来。

如果你只想先抓住最重要的学习顺序，可以按下面这条线走：

1. 先理解 DOM 树是什么，知道节点、父节点、兄弟节点这些概念。
2. 再理解 `LinkDensity` 和 `TBD` 为什么能区分正文和导航。
3. 然后理解为什么正文不能只按单节点打分，必须看结构关系。
4. 最后再看分类器方案，把规则和特征统一到一个学习框架里。
