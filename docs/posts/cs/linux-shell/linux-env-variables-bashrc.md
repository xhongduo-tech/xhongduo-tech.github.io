---
title: 使用 Linux 环境变量与 Bash 启动文件
date: 2026-08-07
---

# 使用 Linux 环境变量与 Bash 启动文件

<div class="epigraph">
<p>环境变量是 shell 传给每个程序的便条：我的家在哪、我在哪、我该去哪找可执行文件。</p>
<footer>—— 进程环境的本质</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ Blum《Shell 脚本编程大全》 第6章 ｜ 2026-08-07</p>
</div>

## 为什么从环境变量开始

你有没有好奇过：脚本里那个 `$PATH`、`$HOME` 是哪来的？为什么改完 `~/.bashrc` 要 `source` 一下才生效？答案都在**环境变量（environment variable）** 与 **bash 启动文件**里。环境变量是 shell 启动进程时传给它的「环境便条」，而启动文件决定了每次登录时这份便条怎么被写。本章先讲环境变量的读写与生命周期，再重点拆解 bash 的四类启动文件——这是「为什么我的 alias 在脚本里用不了」「为什么 cron 里 PATH 不对」等一批经典问题的总根源。

## 1 环境变量：全局 vs 局部

变量分两类，差别在**能否传给子进程**：

- **全局变量（环境变量）**：定义后 `export`，随进程派生传给子进程。`PATH`、`HOME`、`USER`、`LANG` 都是。
- **局部变量**：只在当前 shell 有效，子进程看不到。

```bash
MYVAR="hello"          # 局部变量
export MYVAR="hello"   # 全局变量：export 后子进程可见
echo $PATH             # 查看 PATH
env                    # 列出当前全部环境变量
printenv PATH          # 查看单个环境变量
```

`export` 的本质是给变量打上「可继承」标记——子进程（包括脚本里启动的任何程序）会复制一份当前环境。**子进程只能继承、不能回写**：子进程改环境变量，父进程无感。<span class="marginnote">`env` 是「看全部环境」的命令，`printenv` 按名查一个。`set` 会列出<strong>所有</strong>变量（含函数、局部变量）——比 `env` 多得多，平时看环境用 `env` 就够。</span>

**易错点**：`export MYVAR=hello` 之后改值只需 `MYVAR=world`——**变量已经 export 过，再赋值仍是全局的**，不必重复 export。另外 `env -i` 启动一个「干净环境」，`env VAR=x cmd` 为单条命令临时设置环境。

## 2 PATH：环境变量中最重要的一位

**`PATH`** 定义「敲一个命令时去哪找可执行文件」的目录列表，冒号分隔：

```bash
echo $PATH
# /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

当你在任何目录敲 `ls`，bash 按 PATH 顺序在每个目录里找 `ls`，找到就执行；全找不到才报 `command not found`。<span class="marginnote">把自定义脚本目录加进 PATH：`export PATH="$HOME/bin:$PATH"`（把新目录放<strong>最前</strong>优先命中）。这条命令写进 `~/.bashrc` 后，每次登录都能直接用你的脚本。不要在当前目录里留 `PATH=.`——安全风险，别人放个同名恶意程序就能劫持你的命令。</span>

**易错点**：`PATH` 被覆盖的后果是灾难性的——`PATH=` 清空后，`ls`、`cp` 全部 `command not found`。脚本里**不要重置 PATH**，只追加：`export PATH="$APP_DIR/bin:$PATH"`。cron 里 PATH 极简导致脚本命令找不到，也是同一问题的另一面。

## 3 公式解析：常见环境变量的分工

理解环境变量最快的方式是把「常用清单」分组看——它们大致回答五个问题：

$$
\underbrace{\text{HOME}}_{\text{家在哪}} \quad \underbrace{\text{PATH}}_{\text{去哪找命令}} \quad \underbrace{\text{LANG}}_{\text{说什么语言}} \quad \underbrace{\text{PS1}}_{\text{提示符长啥样}} \quad \underbrace{\text{USER / SHELL}}_{\text{我是谁 / 用什么壳}}
$$

逐一拆解：

- **`HOME`**：当前用户家目录，`cd` 无参时去哪、`~` 展开成什么，都由它定。
- **`LANG`**：本地化语言，影响排序与编码。`LANG=zh_CN.UTF-8` 下中文正常显示，`LANG=C` 是传统英文环境（也常用作「按字节排序」的稳妥选择）。
- **`PS1`**：命令行提示符格式。`PS1='[\u@\h \W]\$ '` 显示「用户@主机 目录」，`\u` 用户名、`\h` 主机名、`\W` 当前目录名。
- **`USER` / `SHELL`**：当前用户与登录 shell 路径，脚本里判断身份时用。

**易错点**：环境变量是**字符串**，没有类型。`LANG=C` 与 `LANG=C.UTF-8` 有细微差别（前者可能缺少部分编码支持）；`PS1` 里的转义只在**交互 shell** 生效，脚本里改 PS1 没意义。

### 环境变量传给谁：脚本、cron 与 systemd

环境变量的继承链值得专门看清：**父进程 export 的变量会流进所有后代进程**。

- **脚本里读环境变量**：脚本直接 `echo "$API_KEY"`——只要启动脚本的 shell 里有这个 export，脚本就能读。给脚本传参之外，用环境变量传递配置（数据库地址、密钥）是常见做法。
- **cron 里配置环境**：cron 环境精简，需要时在 crontab 顶部声明：`API_KEY=xxx` 占一行，之后的任务行都能用。也可以让任务脚本先 `source ~/.profile` 再干活。
- **systemd 服务里配环境**：unit 文件的 `[Service]` 段写 `Environment=API_KEY=xxx` 或 `EnvironmentFile=/etc/myapp.env`——这是现代服务最标准的环境注入方式。

这三处是「环境变量在自动化里真正发光」的场景：把配置与代码分离，改环境不改脚本。<span class="marginnote">给单条命令临时设环境：`API_KEY=secret ./deploy.sh` 前缀写法只在这次命令有效。而 `unset VAR` 删除变量、`set -a` 让后续赋值自动 export——少数脚本会用到，知道存在即可。</span>

## 4 核心对比表：bash 的四类启动文件

bash 启动时**按 shell 类型**决定读哪些文件，这是本章最值得记的对照：

| shell 类型 | 读取顺序 | 说明 |
| --- | --- | --- |
| 登录 shell | `/etc/profile` → `~/.bash_profile`（或 `.bash_login`/`.profile`） | ssh 登录、`su -` |
| 非登录交互 shell | `~/.bashrc` | 登录后新开终端 |
| 非交互 shell | `$BASH_ENV`（若设） | 脚本执行，默认不读任何文件 |
| 退出时 | `~/.bash_logout` | 登录 shell 退出 |

**核心对比表：三个用户级文件的职责**

| 文件 | 何时读 | 放什么 |
| --- | --- | --- |
| `~/.bash_profile` | 登录 shell | 环境变量、PATH、启动一次的逻辑 |
| `~/.bashrc` | 每个交互 shell | 别名、函数、提示符、快捷键 |
| `~/.bash_logout` | 登录 shell 退出 | 清理、记录 |

**重点：登录 shell 通常自己不读 `.bashrc`，而是由 `.bash_profile` 里的一行主动 source 它**——发行版默认在 `.bash_profile` 里写了：

```bash
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
```

这就是「改别名要改 `.bashrc`、改环境变量要看 `.bash_profile`」的分工来源。而**脚本（非交互）默认一个都不读**——所以脚本里 `PATH` 极简、别名全部失效。<span class="marginnote">要「每次登录都有效」的别名与环境变量写在 `.bashrc`（因为它总被 source）。`su -` 与 `su` 的区别之一：`su -` 是登录 shell、读 `.bash_profile`，`su` 是非登录、只读 `.bashrc`——很多「为什么 su 进去 PATH 不对」的谜底就在这。</span>

**易错点**：改完启动文件**不会自动生效**——要么重新登录，要么 `source ~/.bashrc` 手动重载。忘记 source 就以为「改了没用」，是新手最常陷入的循环。

## 5 小结

- **export 让变量可继承**：子进程复制环境但改不回父进程；`env`/`printenv` 查看。
- **PATH 是找命令的目录清单**：追加而非重置，`export PATH="$dir:$PATH"` 放最前优先命中。
- **环境变量都是字符串**：HOME/PATH/LANG/PS1/USER 各回答一个问题。
- **启动文件按 shell 类型分工**：`.bash_profile` 管登录环境、`.bashrc` 管每个交互 shell、`.bash_logout` 管退出。
- **脚本不读启动文件**：所以脚本 PATH 精简、别名失效；改完配置要 `source` 重载。
- **配置与代码分离**：cron 顶部声明、systemd `EnvironmentFile=` 注入，改环境不动脚本。

在下一节，我们让脚本学会「听指令、受控制」——**脚本控制**：信号捕捉、后台运行与运行控制。
