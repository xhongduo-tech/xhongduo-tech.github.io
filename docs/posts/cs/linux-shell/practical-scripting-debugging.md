---
title: 编写实用脚本：脚本排错与日常运维脚本实战
date: 2026-08-07
---

# 编写实用脚本：脚本排错与日常运维脚本实战

<div class="epigraph">
<p>脚本写出来那天不是结束，是排错的开始——而排错有方法，不是靠瞪眼。</p>
<footer>—— 调试的工程化共识（Debugging discipline）</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ Blum《Shell 脚本编程大全》 第24章 ｜ 2026-08-07</p>
</div>

## 为什么从实用脚本与排错开始

到此我们已经走完了命令行的全程：从文件、权限、管道，到脚本的变量、分支、循环、函数，再到 sed/awk 的文本处理。本章把知识收拢成**两条最后的技能**：**排错**——脚本坏了怎么系统化地找原因；**实战**——把完整工具串成一个能用的运维脚本。这一章也是整条学习路径的验收：写一个「带检查、带日志、带退出状态、可复用」的脚本，就是本章的毕业设计。

## 1 排错的第一原则：让 shell 告诉你它干了什么

排错不是猜，而是**观察**。bash 提供一组调试开关，让脚本把「内心活动」打印出来：

```bash
bash -x script.sh        # 执行前打印每条命令（展开后）
bash -v script.sh        # 打印原始输入行
bash -n script.sh        # 只做语法检查，不执行
```

**`-x`（xtrace）** 是最常用的：每执行一条命令，先打印展开后的命令，`+` 前缀开头。变量替换后的真实值一目了然——脚本「以为自己在做什么」与「实际做了什么」的落差，全在 `+` 行里现形。

**公式解析：set 开关——在脚本内部开启排错**

$$
\text{set} \; \underbrace{-x}_{\text{执行跟踪}} \quad \underbrace{-e}_{\text{出错即退}} \quad \underbrace{-u}_{\text{未定义变量报错}} \quad \underbrace{-o\;\text{pipefail}}_{\text{管道任一失败即失败}}
$$

拆解这四个开关：

- **`set -x`**：等价 `bash -x`，脚本内部从该行起打印每条命令的展开。
- **`set -e`**：**任何命令失败（非 0）立即退出**——防止「错误发生后继续执行造成二次伤害」。
- **`set -u`**：使用**未定义变量**时报错退出——把 `$typo` 这类拼写错误当场暴露。
- **`set -o pipefail`**：管道里**任一命令失败**整体算失败（默认只看最后一条）。

**易错点**：`set -e` 会「误伤」合法场景——`if` 的条件命令、`grep` 找不到匹配（返回 1）都会触发退出。配合 `set -e` 时，预期可能失败的命令要写成 `grep ... || true` 或放进 `if` 条件里。<span class="marginnote">`set -ex` 组合（`-e` 与 `-x` 一起）是「失败即停 + 全程跟踪」的强排错模式，脚本出问题时临时加上、定位后移除。`trap 'echo 行号 $LINENO' ERR` 还能在出错时打印行号。</span>

## 2 现代排错：shellcheck 与防御式写法

老练的 bash 排错不再只靠人眼——**`shellcheck`** 是 bash 的静态检查器，像 lint 一样揪出隐患：

```bash
shellcheck script.sh
```

它会指出「变量没加引号」「`cd` 后忘了检查」「`$?` 被中间的 echo 覆盖」这类防不胜防的坑。安装：`apt install shellcheck` 或 `brew install shellcheck`。

**防御式写法的五条纪律**，让脚本从源头少出错：

1. **变量永远加引号**：`"$var"`——防空格、防通配符展开。
2. **`cd` 后检查结果**：`cd "$dir" || exit 1`，cd 失败就停，别在错误目录里继续。
3. **`$?` 立即消费**：上一条命令的状态码别等到 `echo` 之后才用。
4. **关键命令校验**：`command -v tool` 确认工具存在再调用。
5. **函数内变量 `local`**：不污染全局，见函数一章。

**易错点**：`cd "$dir" && 后续命令` 的链式写法虽然安全，但只保护这一行——后面的命令仍在错误目录跑。要「整个脚本都基于正确目录」，在脚本开头 `cd "$(dirname "$0")"` 或 `cd /指定的绝对路径`，并 `|| exit 1`。

## 3 实战：一个完整的备份脚本

把所有知识组装成一个「带校验、带日志、可排错」的备份脚本：

```bash
#!/bin/bash
# 备份 /data 目录到 /backup，保留 7 天，记录日志
set -euo pipefail

src="/data"
dst="/backup"
stamp="$(date +%Y%m%d_%H%M%S)"
log="/var/log/backup.log"

log_msg() {
    echo "[$(date '+%F %T')] $*" | tee -a "$log"
}

mkdir -p "$dst" || exit 1

log_msg "开始备份 $src → $dst/${src##*/}_$stamp.tar.gz"
tar -czf "$dst/${src##*/}_$stamp.tar.gz" -C "$(dirname "$src")" "$(basename "$src")"
log_msg "打包完成，大小: $(du -h "$dst/${src##*/}_$stamp.tar.gz" | cut -f1)"

# 清理 7 天前的旧备份
find "$dst" -name "*.tar.gz" -mtime +7 -delete
log_msg "已清理 7 天前的旧备份"

log_msg "备份完成"
```

逐行读这个脚本，能看到前面每一章的影子：

`set -euo pipefail`：本章的排错开关，一个不落。
`log_msg()`：函数 + `tee -a`（呈现数据一章的 FD 与 tee），每次操作留痕。
- `${src##*/}`：参数扩展取「路径最后一个分量」——不写死文件名，换源目录也能跑。
- `tar -czf ... -C ... `：压缩打包一章的选项；`du -h | cut -f1`：管道与文本处理。

**易错点**：`set -e` 下 `grep` 找不到会退出——清理命令 `find ... -delete` 删不到文件时返回非 0 也会退出。这类「找不到是正常情况」的命令，要么容忍退出码、要么 `|| true`。日志写进 `/var/log` 需要 root，普通用户跑时把 `log` 改到 `$HOME` 下。

## 4 排查脚本的三个层次

脚本出问题，按三层递进排查：

**第一层：语法与启动。** `bash -n script.sh` 验语法；`bash -x script.sh` 看全程展开。`command not found` 多半是 PATH 问题或脚本没 `chmod +x`。

**第二层：变量与逻辑。** 输出中间变量：`echo "DEBUG: src=$src dst=$dst"`（或用 `set -x`）。`$?` 在错误点立刻取——很多脚本「悄悄失败」是因为状态码被随后的命令覆盖。

**第三层：环境与权限。** cron 里跑不通、终端里能跑通，九成是环境差异——`cron` 的 PATH 极简、不读 `.bashrc`。脚本开头显式 `export PATH=/usr/local/bin:/usr/bin:/bin` 或 `source` 需要的环境文件。<span class="marginnote">排错口诀：<strong>「先看它在哪失败（`-x`），再看失败时环境（env），最后看权限（ls -l、sudo）。」</strong> 一次只看一层，别在不知道失败位置时乱改代码。</span>

## 5 小结

- **排错靠观察不靠猜**：`bash -x` 看展开、`-n` 验语法；`set -euo pipefail` 是脚本的「安全带」。
- **`set -e` 会误伤合法失败**：预期失败的命令用 `|| true` 或放进 `if` 条件。
- **shellcheck 是静态检查器**：把引号、`cd`、`$