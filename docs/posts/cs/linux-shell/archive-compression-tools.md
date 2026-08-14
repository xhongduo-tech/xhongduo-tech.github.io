---
title: 压缩、打包与归档：tar/gzip/bzip2/xz
date: 2026-08-07
---

# 压缩、打包与归档：tar/gzip/bzip2/xz

<div class="epigraph">
<p>打包是把一堆文件变成一个，压缩是让一个变得更小——Linux 把这拆成两步。</p>
<footer>—— Unix 工具哲学的又一次实践（组合而非合并）</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ 鸟哥《Linux私房菜》 第8章 ｜ 2026-08-07</p>
</div>

## 为什么从压缩与归档开始

备份、传输、分发——几乎每个运维日常都要回答同一个问题：**怎么把一堆文件装进一个体积合理的包里？** Windows 用户习惯的「右键 → 压缩为 zip」在 Linux 里被拆成了两件事：**打包（archiving）** 把多个文件并成一个归档文件，**压缩（compression）** 把单个文件变小。拆成两步看似麻烦，实际是刻意设计——它让「归档」与「压缩」各自独立演进，也让 `tar` + `gzip`/`bzip2`/`xz` 的组合成为整个软件分发世界的事实标准。从官网下载的每个 `.tar.gz` 源码包，都是这套哲学的产物。

## 1 打包与压缩：两个概念、一条命令

**打包（archive）** 用一个文件装下多个文件与目录，保留目录结构；**压缩（compress）** 用算法把文件的重复模式消除，让体积变小。Linux 里打包的主力是 **`tar`**（tape archive），压缩的主力是 `gzip`、`bzip2`、`xz`。

但日常你几乎总是写 `tar -czvf`——因为 tar 支持在打包的同时**调用压缩器**完成压缩，一步到位：

```bash
tar -czvf backup.tar.gz /home/data      # 打包 + gzip 压缩
tar -xzvf backup.tar.gz -C /restore     # 解压 + 解包到指定目录
tar -tzvf backup.tar.gz                 # 只看包里有什么（不解包）
```

选项记忆法（**`v` 不看、`f` 必在最后、`C` 是目的地**）：

| 选项 | 含义 |
| --- | --- |
| `c` | create，创建归档 |
| `x` | extract，解出归档 |
| `t` | list，列出内容 |
| `z` | 用 gzip 压缩/解压 |
| `j` | 用 bzip2 |
| `J` | 用 xz |
| `v` | verbose，显示过程 |
| `f` | 指定文件名（**必须最后**，后面紧跟文件名） |
| `C` | 解压到指定目录 |

<span class="marginnote">`f` 必须紧跟在文件名前且放在选项串末尾，因为 tar 把 `f` 后面的第一个参数当作文件名。写 `tar -fczv` 会出问题——`z` 和 `v` 会被当成文件名的一部分。</span>

**易错点**：`tar -czf` 三个字母各有分工，很多人把 `z` 当成「zip」记，其实 `z` 只指 gzip。要解压 `.tar.bz2` 必须用 `j`，`.tar.xz` 用 `J`。记不住时，用 `tar -xaf`（`-a` 让 tar 从后缀自动推断压缩器），一劳永逸。

## 2 压缩器的横向比较：速度与体积的权衡

`gzip`、`bzip2`、`xz` 三者的本质区别是**压缩算法**不同，带来的权衡是「压得小 vs 压得快」：

**核心对比表：三大压缩器**

| 维度 | gzip | bzip2 | xz |
| --- | --- | --- | --- |
| 算法 | DEFLATE | Burrows-Wheeler | LZMA2 |
| 压缩比 | 最低 | 中等 | 最高 |
| 速度 | 最快 | 中等 | 最慢 |
| 常见后缀 | `.gz` | `.bz2` | `.xz` |
| 典型场景 | 日常备份、日志轮替 | 较保守的分发 | 软件源码、追求最小体积 |

单独使用时它们都只压缩**一个文件**，所以看源码包总是 `.tar.gz` 而非直接 `.gz`——先 tar 打包，再压缩器压。

```bash
gzip file.txt             # file.txt → file.txt.gz（原文件消失）
gzip -d file.txt.gz       # 解压（等价 gunzip）
gzip -k file.txt          # -k 保留原文件
xz -k file.txt            # 同样用法，-k 保留原件
```

**易错点**：`gzip` 默认压缩后**删除原文件**——`gzip file.txt` 之后 file.txt 不见了，只剩 `file.txt.gz`。不想要这个行为就加 `-k`。这是压缩工具与打包工具最常被混淆的行为差异。

## 3 公式解析：为什么 `-9` 能压得更小

压缩器普遍支持 `-1` 到 `-9` 的等级参数：等级越高，压缩越慢但体积越小。背后的直觉可以用一个「查找重复」的模型理解：

$$
\text{压缩率} \approx \frac{\text{输入大小} - \text{可消除的冗余}}{\text{输入大小}}, \qquad
\text{工作量} \propto \text{搜索重复模式的开销}
$$