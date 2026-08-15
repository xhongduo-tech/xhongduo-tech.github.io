---
title: 查找与过滤：find、grep、sort、uniq 与管道
date: 2026-08-07
---

# 查找与过滤：find、grep、sort、uniq 与管道

<div class="epigraph">
<p>这是 Unix 哲学：让每个程序做好一件事，并用管道把它们接起来。</p>
<footer>—— 道格拉斯 · 麦克罗伊（Douglas McIlroy），管道概念的提出者</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ 鸟哥《Linux私房菜》 第10章 ｜ 2026-08-07</p>
</div>

## 为什么从查找与过滤开始

前两节我们学会了「在已知位置操作已知文件」，但真实工作里你常常只记得几个碎片：某个日志里出现过 `error 503`、某份配置改过端口、哪个目录占用了磁盘。这时候需要的是**在未知里找已知**的能力。本章把四个命令（`find`、`grep`、`sort`、`uniq`）和一个思想（**管道**）放一起讲——它们是 Unix「小工具、大组合」哲学的第一次完整展示。管道思想会贯穿后面所有 Shell 脚本，也是理解现代数据处理（从 `awk` 到大数据流水线）的同一个内核。<span class="marginnote">管道由道格拉斯·麦克罗伊在 1972 年的贝尔实验室提出，他用一句话点破本质：<strong>让每个程序做好一件事，并用管道把它们接起来</strong>。今天 Big Data 里的 MapReduce、流式处理，骨子里仍是「小步骤串成流水线」。</span>

## 1 管道：把命令接成流水线

**管道（pipe）** 用竖线 `|` 把前一条命令的**标准输出**接到后一条命令的**标准输入**：

```bash
ls -l | grep "\.sh$"
```

这条命令的意思：`ls -l` 的输出不再打印到屏幕，而是直接流进 `grep`，由 grep 只挑出以 `.sh` 结尾的行。<span class="marginnote">管道与重定向的差别：`>` 是把输出写进文件，`|` 是把输出递给另一个程序。管道是<strong>内存中的数据流</strong>，两边程序同时运行，前边生产、后边消费——这才是「流式处理」的雏形。</span>

**公式解析：管道的三段式**

$$
\text{源命令} \xrightarrow{\text{标准输出}} \text{管道} \xrightarrow{\text{标准输入}} \text{过滤命令} \xrightarrow{\text{标准输出}} \text{终端}
$$

拆解这个公式：

- **第一步**：左侧命令只负责「生产」，它把结果写到标准输出，不关心谁在读。
- **第二步**：`|` 在内存里建立一条字节流，把左侧输出接进右侧输入。
- **第三步**：右侧命令只负责「消费与加工」，它的输出继续流向下游或屏幕。

关键洞察：**管道里的每一步都只做一件事，但组合起来能做任何事。** 例如统计日志里 503 出现次数：

```bash
grep "503" access.log | wc -l
```

grep 挑出所有含 503 的行，wc -l 数行数。两个命令各自简单，拼起来完成一个「计数」任务。

## 2 查找文件：find

**`find`** 在目录树里按条件搜索文件，是「文件在哪」的标准答案。它比 `ls -R` 灵活得多，核心是「路径 + 条件 + 动作」三段式：

```bash
find /var/log -name "*.log"          # 按文件名匹配（-iname 忽略大小写）
find / -type d -name "node_modules"  # 找类型为目录的
find ~ -size +100M                   # 找大于 100MB 的文件
find /tmp -mtime -1                  # 24 小时内修改过的文件
find . -perm 777                     # 找权限为 777 的文件（安全隐患排查）
```

**易错点**：`find` 的 `-name` 是**全名匹配**，`*.log` 里的 `*` 是 find 自己的通配符，不是 shell 的通配符——所以一定要给模式加引号，否则 shell 会在 find 拿到之前就展开它。<span class="marginnote">`find` 默认不加引号时，若当前目录恰好有匹配的文件，shell 会把文件名直接传给 find，结果与你预期完全不符。`-name "*.log"` 加引号是职业习惯。</span>

`find` 还可以对查到的每个结果执行命令：

```bash
find . -name "*.tmp" -exec rm {} \;
```

`{}` 是「当前文件名」的占位符，`\;` 是 `-exec` 的结束符。<span class="marginnote">`-exec` 的结尾必须是 `\;` 或 `+`——`\;` 逐个执行、`+` 把所有结果拼成一条命令一次性执行（后者更快）。注意 `\;` 里分号前的反斜杠是为了不让 shell 吃掉分号。</span>

## 3 过滤文本：grep

**`grep`**（global regular expression print）在文本里按正则表达式找行。它是命令行使用频率最高的一把刀：

```bash
grep "error" app.log                # 基本搜索
grep -i "error" app.log             # 忽略大小写
grep -v "debug" app.log             # 反向：输出不含 debug 的行
grep -n "error" app.log             # 显示行号
grep -r "TODO" src/                 # 递归搜索整个目录
grep -c "error" app.log             # 只数匹配行数
grep -E "error|fatal" app.log       # 扩展正则：多模式
```

**核心对比表：grep 家族常用选项**

| 选项 | 含义 | 典型场景 |
| --- | --- | --- |
| `-i` | 忽略大小写 | 搜 `error` 也想命中 `Error` |
| `-v` | 反向选择 | 过滤掉日志噪音行 |
| `-n` | 显示行号 | 定位到具体代码行 |
| `-r` / `-R` | 递归目录 | 在大仓库里找关键字 |
| `-c` | 只输出计数 | 快速统计出现次数 |
| `-E` | 扩展正则 | 用 `a\|b` 之类多分支 |
| `-w` | 整词匹配 | 避免 `cat` 命中 `concatenate` |

**易错点**：`grep` 默认是「子串匹配」，`grep "cat"` 会同时命中 `cat`、`catalog`、`concatenate`。想要整词，加 `-w`；想要精确一行，用 `^...$` 锚定行首行尾——`grep "^error"` 只匹配行首的 error。正则的细节我们会在《正则表达式与文件格式化处理》里专门展开，这里先记住 `^` 行首、`$` 行尾、`.` 任意字符三个记号。

## 4 排序与去重：sort 与 uniq

**`sort`** 按行排序，**`uniq`** 消除相邻的重复行——它俩几乎总是配对使用：

```bash
sort file.txt                        # 字典序排序
sort -n file.txt                     # 按数值排序（10 会排在 9 后面）
sort -rn file.txt                    # 数值 + 倒序
sort -t: -k3 -n /etc/passwd          # 以 : 分隔，按第 3 字段数值排序
sort file.txt | uniq                 # 去重（uniq 只去相邻重复）
sort file.txt | uniq -c              # 去重并统计每行出现次数
sort file.txt | uniq -c | sort -rn   # 统计并倒序 = 频率排行榜
```

**核心对比表：sort 的常用选项**

| 选项 | 含义 |
| --- | --- |
| `-n` | 按数值排序（默认是字典序） |
| `-r` | 倒序（reverse） |
| `-t` | 指定字段分隔符 |
| `-k` | 按第几个字段排序 |
| `-u` | 去重（等价 sort + uniq） |

**易错点**：`uniq` 只去**相邻**的重复行——`a b a` 里两个 `a` 不相邻，`uniq` 一个都不去，必须先用 `sort` 让相同行聚到一起。「统计出现次数」的流水线 `sort | uniq -c | sort -rn` 是日志分析里出场率最高的三连，几乎可以当口诀背下来。<span class="marginnote">频率统计的完整姿势：`grep -o "..." file | sort | uniq -c | sort -rn`——`grep -o` 只输出匹配的部分，配合 sort/uniq 就能数出「哪个值出现最多」。这四段式正是 count-by-key 的原型，awk 与大数据引擎都在做同一件事。</span>

**公式解析：频率榜三连的流向**

$$
\text{原始流} \xrightarrow{\text{sort 聚拢相同行}} \xrightarrow{\text{uniq -c 计数}} \xrightarrow{\text{sort -rn 倒序}} \text{频率榜}
$$

拆解这条流水线：

- **第一步**：`sort` 把相同内容的行聚到相邻位置——这是 uniq 能去重的前提。
- **第二步**：`uniq -c` 把相邻相同行合并成一行「次数 + 内容」，输出形如 `42 app.log`。
- **第三步**：`sort -rn` 按次数数值倒序，频率最高的排第一——得到 Top N。

每一段的输出就是下一段的输入，命令各自只会一招，组合起来完成「统计 Top N」——这正是管道哲学的全部。

## 5 小结

- **管道 `|`**：前一命令的标准输出接后一命令的标准输入，内存中的字节流，两边同时运行。
- **find**：路径 + 条件 + 动作；`-name` 要加引号、`-type d` 找目录、`-size` 找大小、`-exec {} \;` 对结果执行命令。
- **grep**：默认子串匹配；`-i` 忽略大小写、`-v` 反向、`-E` 扩展正则、`-c` 计数、`-w` 整词。
- **sort 与 uniq**：uniq 只去相邻重复，必须配 sort；`-n` 数值、`-r` 倒序、`-t/-k` 字段。
- **频率排行榜**：`sort | uniq -c | sort -rn` 三连，日志分析的基本功。

在下一节，我们把「编辑文件」这件天天要做的事讲透：模式切换、插入、删除、复制粘贴与查找替换——这就是**vim 程序编辑器与文本处理**。
